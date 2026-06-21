"""Dataset and data loading utilities for protein structure files.

Contains ProteinDataset and ClusteredProteinDataset for lazy-loading protein
structures from JSONL files, collate helpers for variable-length batching, and
a factory function that assembles bucketed train/val/test DataLoaders with
optional DDP support.
"""

import dataclasses
import hashlib
import io
import multiprocessing as mp
import queue
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import ClassVar, cast

import numpy as np
import numpy.typing as npt
import structlog
import torch
import torch.distributed as dist
import torch.utils.data
import webdataset as wds
from helpers.atom_utils import (
    Protein,
    center_positions,
    make_fixed_size,
    make_np_example,
    restype_order,
    truncate_to_length,
)
from helpers.context_managers import (
    FatalOnError,
    ShardWorkerNotInitializedError,
    ShardWorkerState,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
)
from structlog.typing import FilteringBoundLogger
from torch.utils.data.distributed import DistributedSampler
from train.train_config import TrainArgs, TrainConfig
from typing_extensions import override
from webdataset.writer import TarWriter


class ProteinEntry(BaseModel):
    """Minimal schema for protein JSONL entries.

    Validates the three fields required by ProteinDataset and
    ClusteredProteinDataset — name, seq, and coords.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    name: str
    seq: str
    coords: dict[str, list[list[float]]]


class ProteinNamesManifest(RootModel[list[str]]):
    """Pydantic root model holding a flat list of protein entry names."""

    root: list[str]


class DatasetSplitsManifest(BaseModel):
    """Train/validation/test split lists, plus optional CATH topology mapping.

    Attributes:
        model_config: Pydantic config; ``extra="ignore"`` silently drops
            unknown fields encountered when loading the manifest.
        train: Protein entry names in the training split.
        validation: Protein entry names in the validation split.
        test: Protein entry names in the test split.
        cath_nodes: Optional mapping from protein chain name to its CATH
            topology codes (e.g. ``"2fyz.A": ["1.20.5"]``). Empty when the
            manifest does not include CATH metadata.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")
    train: list[str]
    validation: list[str]
    test: list[str]
    cath_nodes: dict[str, list[str]] = Field(default_factory=dict)


class ProteinDataset(
    torch.utils.data.Dataset[Protein],
):
    """Lazy-loading Dataset backed by a JSONL file.

    Scans the file once at construction to build a name→byte-offset index
    (only offsets are kept in RAM, not the protein data).  Each __getitem__
    seeks to the relevant line and parses only that entry.

    Compatible with num_workers > 0: the open file handle is excluded from
    pickling and re-opened lazily inside each worker process.

    JSONL format expected per line:
        {"name": "1abc.A", "seq": "ACDEF...", "coords": {"N": [[x,y,z],...],
         "CA": [...], "C": [...], "O": [...]}, ...}

    Args:
        jsonl_path:     Path to the JSONL file.
        names:          List of entry names (e.g. "1abc.A") to include.
        max_seq_length: Sequences longer than this are truncated; shorter ones
                        are zero-padded to this length.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        names: list[str],
        max_seq_length: int = 256,
    ) -> None:
        self.jsonl_path: Path = Path(jsonl_path)
        self.max_seq_length: int = max_seq_length
        self._file: io.BufferedReader | None = None

        name_set = set(names)
        offsets: list[int] = []
        byte_pos = 0
        with self.jsonl_path.open("rb") as f:
            for raw_line in f:
                if ProteinEntry.model_validate_json(raw_line).name in name_set:
                    offsets.append(byte_pos)
                byte_pos += len(raw_line)

        self._offsets: list[int] = offsets

    # ------------------------------------------------------------------
    # File-handle lifecycle — excluded from pickle so multiprocessing works

    def _open(self) -> io.BufferedReader:
        if self._file is None:
            self._file = self.jsonl_path.open("rb")
        return self._file

    def __getstate__(self) -> dict[str, object]:
        """Return picklable state with the open file handle set to None.

        Replaces the open ``_file`` handle with ``None`` before pickling so
        that the object can be serialised and sent to DataLoader worker
        processes, which will re-open the file lazily via ``_open``.

        Returns:
            A copy of ``__dict__`` with ``_file`` set to ``None``.
        """
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def __del__(self) -> None:
        """Close the underlying file handle on deletion.

        Ensures the JSONL file descriptor is released when the dataset object
        is garbage-collected, even if ``__getstate__`` was never called.
        """
        if self._file is not None:
            self._file.close()

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of entries in the dataset.

        Returns:
            Count of protein entries whose names were present in the supplied
            ``names`` list and found in the JSONL file.
        """
        return len(self._offsets)

    @override
    def __getitem__(
        self,
        idx: int,
    ) -> Protein:
        """Return the parsed JSON entry at the given index.

        Seeks directly to the pre-computed byte offset for ``idx``, parses the
        JSONL line, centres and pads the coordinates to ``max_seq_length``.

        Args:
            idx: Integer index in ``[0, len(self))``.

        Returns:
            Protein with atom37 coordinates truncated to at most
            max_seq_length residues.
        """
        f = self._open()
        _ = f.seek(self._offsets[idx])
        entry = ProteinEntry.model_validate_json(f.readline())

        np_example = make_np_example(entry.coords)
        center_positions(np_example)
        make_fixed_size(np_example, self.max_seq_length)
        truncate_to_length(np_example, self.max_seq_length)

        n_res: int = np_example["atom_positions"].shape[0]

        raw = [
            restype_order.get(aa, restype_order["X"])
            for aa in entry.seq[:n_res]
        ]
        aatype = np.array(
            raw + [restype_order["X"]] * (n_res - len(raw)),
            dtype=np.intp,
        )

        return Protein(
            atom_positions=np_example["atom_positions"].astype(np.float64),
            aatype=aatype,
            atom_mask=np_example["atom_mask"].astype(np.float64),
            residue_index=np_example["residue_index"].astype(np.intp),
            chain_index=np.zeros(n_res, dtype=np.intp),
            b_factors=np.zeros((n_res, 37), dtype=np.float64),
        )


class ShardMetadata(BaseModel):
    """Persisted record of the parameters used to build a shard directory.

    Serialised as ``shard_metadata.json`` alongside the shard tars and read
    back at train time to verify that an existing shard directory matches
    the current configuration before re-use.

    Attributes:
        model_config: Pydantic config — frozen to prevent mutation after init.
        names_hash: SHA-256 hex digest of the sorted protein names list.
        token_budget: Maximum padded token cost per batch used during sharding.
        shard_size: Maximum number of proteins per shard.
        n_shards: Total number of shard tars written.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    names_hash: str
    token_budget: int
    shard_size: int
    n_shards: int


class FFDWorkerPlan(BaseModel):
    """One DataLoader worker's per-epoch plan for FFD batch streaming.

    Attributes:
        shard_ids: Global shard IDs this worker streams this epoch, in
            iteration order.
        permutations: For each shard, a list mapping streaming position
            (i.e. WebDataset arrival order) to the protein's sorted rank
            in the prepended-and-permuted local sequence.
        batch_ends: For each shard, a list of cumulative batch-end
            positions in the prepended-and-permuted sequence
            ``carry_in_sizes[k] + permuted shard k proteins``.
        carry_in_sizes: For each shard, the number of proteins prepended
            from the previous shard's carry-over. Always zero for length-
            disjoint shards (the typical case with globally sorted shards),
            but stored explicitly so workers can validate the carry buffer.
    """

    shard_ids: Sequence[int]
    permutations: Sequence[Sequence[int]]
    batch_ends: Sequence[Sequence[int]]
    carry_in_sizes: Sequence[int]


class FFDBatchPlan(BaseModel):
    """Full per-epoch FFD plan for one rank, partitioned by DataLoader worker.

    Attributes:
        worker_plans: One :class:`FFDWorkerPlan` per DataLoader worker; the
            ``i``th entry is consumed by the worker with ``worker_info.id ==
            i``.
    """

    worker_plans: list[FFDWorkerPlan]


@dataclasses.dataclass(frozen=True)
class ShardBudgetParameters:
    """All scalar inputs needed to compute one epoch's FFDBatchPlan.

    Attributes:
        shard_dir: Directory containing the shard tars and metadata.
        structlog_path: Path to the structlog output file for the worker.
        token_budget: Maximum padded token cost per batch.
        max_seq_len: Per-protein hard truncation ceiling.
        seed: RNG seed; an epoch offset is added per epoch for diversity.
        n_threads: Threads inside the worker subprocess for parallel packing.
        world_size: Number of DDP processes.
        rank: This process's DDP rank.
        n_proteins_in_shard: Expected number of proteins per shard tar.
        noise_magnitude: Half-width of the uniform noise added to each
            protein's length before sorting; controls cross-epoch batch
            diversity vs. within-batch length variance.
        num_workers: Number of DataLoader worker processes per rank. The
            plan is partitioned across workers so each worker's FFD batches
            stay within its strided shard assignment.
    """

    shard_dir: Path
    structlog_path: Path
    token_budget: int
    max_seq_len: int
    seed: int
    n_threads: int
    world_size: int
    rank: int
    n_proteins_in_shard: int
    noise_magnitude: int
    num_workers: int


class ProteinShardDataset(torch.utils.data.IterableDataset[list[Protein]]):
    """Plan-driven streaming dataset backed by WebDataset tar shards.

    Each epoch a ShardBatchPlan is injected via set_plan before the DataLoader
    iterates. __iter__ assigns shards to DataLoader workers round-robin
    (worker w takes shard_order[w::num_workers]), streams each assigned shard
    sequentially via WebDataset, and cuts proteins into pre-computed batches
    using cut_stream_into_batches.

    Shards are created from the source JSONL on first construction if the
    shard metadata file does not yet exist. At most one tar file is open per
    DataLoader worker at any moment.

    Args:
        budget_parameters: Scalar batching and shard configuration.
        names: Protein entry names to include in the training split.
        dataset_jsonl: Path to the source JSONL protein database.
    """

    def __init__(
        self,
        budget_parameters: ShardBudgetParameters,
        names: list[str],
        dataset_jsonl: Path,
    ) -> None:
        self.names: list[str] = names
        self.dataset_jsonl: Path = dataset_jsonl
        self.n_proteins_in_shard: int = budget_parameters.n_proteins_in_shard
        self.shard_dir: Path = budget_parameters.shard_dir
        sidecar_file: str = "shard_sidecar.npz"
        manifest_file: str = "shard_metadata.json"
        lengths_file: str = "all_protein_lengths.npy"
        self.shard_sidecar_path: Path = self.shard_dir / sidecar_file
        self.shard_metadata_path: Path = self.shard_dir / manifest_file
        self.lengths_path: Path = self.shard_dir / lengths_file

        self.structlog_path: Path = budget_parameters.structlog_path
        self._log: FilteringBoundLogger = cast(
            FilteringBoundLogger,
            structlog.get_logger(),
        )
        self.token_budget: int = budget_parameters.token_budget
        self.max_seq_length: int = budget_parameters.max_seq_len

        # Construct the shards if they do not already exist.
        if not self.shard_metadata_path.exists():
            self._log.info(
                "shards_do_not_exist",
            )
            n_shards, all_lengths, shard_sizes_list = self.build_sorted_shards()
            _ = self.write_shard_metadata_sidecar(
                n_shards,
                all_lengths,
                shard_sizes_list,
            )

        # prefill the batch
        self._plan: FFDBatchPlan | None = None

    def build_sorted_shards(self) -> tuple[int, list[int], list[int]]:
        """Globally sort proteins descending by length and write to shard tars.

        Single JSONL pass collects ``(name, seq_len, byte_offset)`` for each
        included protein. Entries are sorted globally descending by
        ``seq_len`` and sliced into sequential chunks of
        ``n_proteins_in_shard``. Each chunk is written to
        ``shard_{id:05d}.tar`` by seeking to the recorded byte offsets in the
        source JSONL.

        Returns:
            Tuple of ``(n_shards, all_lengths, shard_sizes_list)`` where
            ``all_lengths`` contains the ``seq_len`` of every protein in
            global shard order (descending) and ``shard_sizes_list`` contains
            the protein count per shard.
        """
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        name_set = set(self.names)

        entries: list[tuple[str, int, int]] = []
        with self.dataset_jsonl.open("rb") as f:
            byte_pos = 0
            for raw_line in f:
                entry = ProteinEntry.model_validate_json(raw_line)
                if entry.name in name_set:
                    entries.append(
                        (
                            entry.name,
                            len(entry.seq),
                            byte_pos,
                        ),
                    )
                byte_pos += len(raw_line)

        # Global descending sort by seq_len.
        entries.sort(key=lambda e: -e[1])

        all_lengths: list[int] = []
        shard_sizes_list: list[int] = []
        shard_id: int = 0

        with self.dataset_jsonl.open("rb") as src:
            for shard_start in range(
                0,
                max(len(entries), 1),
                self.n_proteins_in_shard,
            ):
                shard_entries = entries[
                    shard_start : shard_start + self.n_proteins_in_shard
                ]
                if not shard_entries:
                    continue
                shard_path: Path = self.shard_dir / f"shard_{shard_id:05d}.tar"
                with TarWriter(str(shard_path)) as sink:
                    for local_idx, ent in enumerate(shard_entries):
                        _ = src.seek(ent[2])  # byte_pos
                        raw_line = src.readline()
                        _ = sink.write(
                            {
                                "__key__": f"{local_idx:06d}",
                                "json": raw_line,
                            },
                        )
                        all_lengths.append(ent[1])  # seq_len
                shard_sizes_list.append(len(shard_entries))
                shard_id += 1

        n_shards = shard_id
        self._log.info(
            "sorted_shards_built",
            n_shards=n_shards,
        )
        return n_shards, all_lengths, shard_sizes_list

    def write_shard_metadata_sidecar(
        self,
        n_shards: int,
        all_lengths: list[int],
        shard_sizes_list: list[int],
    ) -> ShardMetadata:
        """Persist shard layout to disk and return a ShardMetadata summary.

        Writes three files to ``self.shard_dir``:

        - ``all_protein_lengths.npy``: int16 array of every protein's seq_len
          in global shard order.
        - ``shard_sidecar.npz``: int32 arrays ``shard_starts`` and
          ``shard_sizes`` giving the global protein offset and count per shard.
        - ``shard_metadata.json``: ShardMetadata JSON including a SHA-256
          digest of the sorted names list for cache-validity checks.

        Args:
            n_shards: Total number of shard tars written by compute_shards.
            all_lengths: Sequence lengths in global shard order.
            shard_sizes_list: Protein count per shard.

        Returns:
            ShardMetadata instance also serialised to shard_metadata.json.
        """
        lengths_arr: npt.NDArray[np.int16] = np.array(
            all_lengths,
            dtype=np.int16,
        )
        np.save(self.lengths_path, lengths_arr)

        sizes_arr: npt.NDArray[np.int32] = np.array(
            shard_sizes_list,
            dtype=np.int32,
        )
        starts_arr: npt.NDArray[np.int32] = np.zeros(n_shards, dtype=np.int32)
        if n_shards > 1:
            _ = np.cumsum(sizes_arr[:-1], out=starts_arr[1:])
        np.savez(
            self.shard_sidecar_path,
            shard_starts=starts_arr,
            shard_sizes=sizes_arr,
        )

        # get a stable SHA-256 hex digest of the sorted names list.
        names_hash = hashlib.sha256(
            ProteinNamesManifest(root=sorted(self.names))
            .model_dump_json()
            .encode(),
        ).hexdigest()

        shard_metadata_manifest = ShardMetadata(
            names_hash=names_hash,
            token_budget=self.token_budget,
            shard_size=self.n_proteins_in_shard,
            n_shards=n_shards,
        )

        # Serialize directly to bytes using Rust core.
        json_bytes = shard_metadata_manifest.model_dump_json(indent=2).encode(
            "utf-8",
        )
        _ = (self.shard_metadata_path).write_bytes(json_bytes)
        self._log.info(
            "shard_metadata_written",
            shard_sidecar=self.shard_sidecar_path,
            shard_metadata_manifest=self.shard_metadata_path,
        )
        return shard_metadata_manifest

    def set_plan(self, plan: FFDBatchPlan) -> None:
        """Inject the FFDBatchPlan for the next epoch.

        Called by ShardDataLoader.__iter__ before the underlying DataLoader
        starts. Because persistent_workers=False, workers restart each epoch
        and pick up the updated plan via pickling of ProteinShardDataset._plan.

        Args:
            plan: Pre-computed plan from ShardDataLoader.compute_ffd_plan.
        """
        self._plan = plan

    def parse_protein(self, sample: dict[str, object]) -> Protein:
        """Parse one WebDataset sample dict into a Protein.

        Args:
            sample: Dict with key "json" containing the decoded protein dict
                (name, seq, coords) — produced by wds.WebDataset.decode("json").

        Returns:
            Protein with atom37 coords truncated to max_seq_length residues.
        """
        raw = sample["json"]
        entry = (
            ProteinEntry.model_validate_json(raw)
            if isinstance(raw, bytes | str | bytearray)
            else ProteinEntry.model_validate(raw)
        )
        np_example = make_np_example(entry.coords)
        center_positions(np_example)
        truncate_to_length(np_example, self.max_seq_length)
        n_res: int = np_example["atom_positions"].shape[0]
        aatype = np.array(
            [
                restype_order.get(aa, restype_order["X"])
                for aa in entry.seq[:n_res]
            ],
            dtype=np.intp,
        )
        return Protein(
            atom_positions=np_example["atom_positions"].astype(np.float64),
            aatype=aatype,
            atom_mask=np_example["atom_mask"].astype(np.float64),
            residue_index=np_example["residue_index"].astype(np.intp),
            chain_index=np.zeros(n_res, dtype=np.intp),
            b_factors=np.zeros((n_res, 37), dtype=np.float64),
        )

    @override
    def __iter__(self) -> Iterator[list[Protein]]:
        """Yield FFD batches for this worker's assigned shards.

        For each shard in the worker's plan:
        1. Stream all proteins via WebDataset into a local list.
        2. Reorder them in memory using the per-shard permutation
           (``permutations[k][local_pos]`` is the sorted rank of the protein
           at streaming position ``local_pos`` within shard ``k``).
        3. Prepend the carry-over buffer from the previous shard.
        4. Cut at ``batch_ends[k]``; yield each completed batch.
        5. Stash any trailing proteins as the new carry-over buffer.

        If set_plan has not been called, yields nothing.
        """
        plan = self._plan
        if plan is None:
            return

        worker_info = torch.utils.data.get_worker_info()
        worker_id: int = worker_info.id if worker_info is not None else 0
        worker_plan = plan.worker_plans[worker_id]

        carry_over: list[Protein] = []
        for k in range(len(worker_plan.shard_ids)):
            sid = worker_plan.shard_ids[k]
            url = str(self.shard_dir / f"shard_{sid:05d}.tar")
            ds: Iterable[dict[str, object]] = cast(
                Iterable[dict[str, object]],
                wds.DataPipeline(  # pyright: ignore[reportUnknownMemberType,reportAttributeAccessIssue]
                    wds.SimpleShardList(  # pyright: ignore[reportUnknownMemberType,reportAttributeAccessIssue]
                        [url],
                    ),
                    wds.tarfile_to_samples(),  # pyright: ignore[reportUnknownMemberType,reportAttributeAccessIssue]
                ),
            )
            shard_proteins: list[Protein] = [self.parse_protein(s) for s in ds]

            permutation = worker_plan.permutations[k]
            sorted_proteins: list[Protein | None] = [None] * len(shard_proteins)
            for local_pos, sorted_rank in enumerate(permutation):
                sorted_proteins[sorted_rank] = shard_proteins[local_pos]
            sorted_proteins_checked: list[Protein] = [
                p for p in sorted_proteins if p is not None
            ]

            full: list[Protein] = carry_over + sorted_proteins_checked
            prev_cut = 0
            for cut in worker_plan.batch_ends[k]:
                yield full[prev_cut:cut]
                prev_cut = cut
            carry_over = full[prev_cut:]

        if carry_over:
            yield carry_over


# this will be replaced with featurize
def identity_collate(batch: list[Protein]) -> list[Protein]:
    """Pass pre-assembled protein batches through without default stacking."""
    return batch


def batch_count_in_ffd_plan(plan: FFDBatchPlan) -> int:
    """Total batch count across all workers for single epoch's FFDBatchPlan."""
    return sum(len(be) for wp in plan.worker_plans for be in wp.batch_ends)


def plan_cache(
    budget: ShardBudgetParameters,
    sidecar_path: Path,
    cache_dir: Path,
) -> Path:
    """Return the JSON cache file path for a given budget.

    Computes a hex digest uniquely identifying a plan's inputs. Hashes all
    budget fields (excluding structlog_path, which does not affect plan
    content) plus the sidecar file's size and mtime so the key changes if the
    shard data is regenerated.

    Args:
        budget: Epoch-level packing parameters.
        sidecar_path: Path to the shard sidecar metadata file.
        cache_dir: Directory in which plan cache files are stored.

    Returns:
        Path to the ``<key>.json`` cache file.
    """
    h = hashlib.sha256()
    for part in (
        str(budget.shard_dir),
        str(budget.token_budget),
        str(budget.max_seq_len),
        str(budget.noise_magnitude),
        str(budget.seed),
        str(budget.world_size),
        str(budget.rank),
        str(budget.num_workers),
        str(budget.n_proteins_in_shard),
    ):
        h.update(part.encode())
    stat = sidecar_path.stat()
    h.update(str(stat.st_size).encode())
    h.update(str(stat.st_mtime_ns).encode())
    return cache_dir / f"{h.hexdigest()[:24]}.json"


class ShardDataLoader(torch.utils.data.DataLoader[list[Protein]]):
    """Plan-driven DataLoader wrapper for WebDataset shard streaming.

    Encapsulates ProteinShardDataset, DataLoader, and a plan prefetch queue.
    Each call to __iter__ dequeues the next pre-computed ShardBatchPlan,
    injects it into the dataset, schedules the following epoch's plan
    asynchronously, and delegates iteration to the underlying DataLoader.
    Workers restart each epoch (persistent_workers=False) and pick up the
    updated plan via pickling of ProteinShardDataset._plan.

    Args:
        dataset: Pre-constructed ProteinShardDataset to stream from.
        budget: Scalar batching and shard configuration shared with the
            dataset.
        num_workers: Number of DataLoader worker processes.
        batch_prefetch_depth: DataLoader prefetch_factor per worker.
        prefetch_epochs: Number of epoch plans to precompute at startup.
    """

    def __init__(
        self,
        *,
        dataset: ProteinShardDataset,
        budget: ShardBudgetParameters,
        num_workers: int,
        batch_prefetch_depth: int,
        prefetch_epochs: int,
    ) -> None:
        self.shard_dataset: ProteinShardDataset = dataset
        super().__init__(  # pyright: ignore[reportUnknownMemberType]
            self.shard_dataset,
            batch_size=None,
            collate_fn=identity_collate,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=batch_prefetch_depth if num_workers > 0 else None,
        )
        self.budget: ShardBudgetParameters = budget
        self.world_size: int = budget.world_size
        self.prefetch_epochs: int = prefetch_epochs
        self.base_seed: int = budget.seed
        self.epoch: int = 0
        self._log: FilteringBoundLogger = cast(
            FilteringBoundLogger,
            structlog.get_logger(),
        )
        self.structlog_path: Path = budget.structlog_path
        self.protein_lengths_path: Path = self.shard_dataset.lengths_path
        self.shard_sidecar_path: Path = self.shard_dataset.shard_sidecar_path

        self.process_executor: ProcessPoolExecutor = ProcessPoolExecutor(
            max_workers=1,
            mp_context=mp.get_context("spawn"),
            initializer=ShardWorkerState.init_worker,
            initargs=(
                self.protein_lengths_path,
                self.shard_sidecar_path,
                self.structlog_path,
            ),
        )
        self.queue_watcher: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=1,
        )
        self.process_queue: queue.Queue[FFDBatchPlan] = queue.Queue(
            maxsize=self.prefetch_epochs,
        )
        self.plan_cache_dir: Path = budget.shard_dir / "plan_cache"
        self.plan_cache_dir.mkdir(parents=True, exist_ok=True)
        self._log.info(
            "process_queue_initialized",
            queue_depth=self.prefetch_epochs,
        )

        first_plan: FFDBatchPlan | None = None
        for offset in range(self.prefetch_epochs):
            epoch_budget = dataclasses.replace(
                budget,
                seed=self.base_seed + offset,
                num_workers=num_workers if num_workers > 0 else 1,
            )
            cache_path = plan_cache(
                epoch_budget,
                self.shard_sidecar_path,
                self.plan_cache_dir,
            )
            if cache_path.exists():
                plan = FFDBatchPlan.model_validate_json(cache_path.read_bytes())
                self._log.info(
                    "ffd_plan_cache_hit",
                )
            else:
                plan = self.process_executor.submit(
                    self.compute_ffd_plan,
                    epoch_budget,
                ).result()
                _ = cache_path.write_bytes(plan.model_dump_json().encode())
                self._log.info(
                    "ffd_plan_computed_and_cached",
                )
            self.process_queue.put(plan)
            if first_plan is None:
                first_plan = plan

        self._cached_len: int = (
            batch_count_in_ffd_plan(first_plan) if first_plan is not None else 0
        )
        self._log.info(
            "shard_prefetch_complete",
            prefetch_epochs=self.prefetch_epochs,
        )

    @staticmethod
    def ffd_pack(
        sorted_effective_lengths: list[int],
        token_budget: int,
    ) -> list[int]:
        """First-Fit-Decreasing pack with padded `(count + 1) * L_max²` budget.

        Walks a descending-sorted, max-seq-length-clamped length list once. For
        each protein, decides whether to extend the current open batch or
        close it and start a new one. L_max for the budget check is the first
        protein in the batch (never grows mid-batch because the list is sorted
        descending). Proteins whose own L² already exceeds the budget are
        emitted as solo batches, matching today's pack_shard policy.

        Args:
            sorted_effective_lengths: Protein lengths after applying
                ``min(L, max_seq_len)`` and sorting descending by the noisy
                sort key.
            token_budget: Maximum padded token cost per batch.

        Returns:
            List of cumulative batch-end positions (1-indexed element
            count at each cut). An empty input returns an empty list.
        """
        batch_ends: list[int] = []
        token_budget = token_budget * token_budget
        current_count = 0
        batch_max_sq = 0
        for i, length in enumerate(sorted_effective_lengths):
            cost_from_item = length * length
            if cost_from_item > token_budget:
                if current_count > 0:
                    batch_ends.append(i)
                    current_count = 0
                batch_ends.append(i + 1)
                batch_max_sq = 0
                continue
            if current_count == 0:
                batch_max_sq = cost_from_item
                current_count = 1
            elif (current_count + 1) * batch_max_sq > token_budget:
                batch_ends.append(i)
                batch_max_sq = cost_from_item
                current_count = 1
            else:
                current_count += 1
        if current_count > 0:
            batch_ends.append(len(sorted_effective_lengths))
        return batch_ends

    @staticmethod
    def compute_ffd_plan(
        budget: ShardBudgetParameters,
    ) -> FFDBatchPlan:
        """Compute one epoch's FFDBatchPlan inside the WorkerState subprocess.

        Shuffles all shard IDs with ``rng(budget.seed)``, takes this rank's
        strided slice, then partitions that slice across DataLoader workers.
        For each (worker, shard) pair: looks up the shard's lengths, adds
        uniform noise ``rng.uniform(-noise_magnitude, noise_magnitude)`` to
        each length, sorts descending by the noisy key, runs
        :meth:`ffd_pack` on the clamped sorted lengths. Per-shard FFD is
        equivalent to per-worker global FFD here because globally sorted
        shards have disjoint length ranges, so FFD batches never span
        shards (carry_in_sizes is always zero).

        Args:
            budget: Scalar parameters for this epoch's plan computation.

        Returns:
            FFDBatchPlan with one FFDWorkerPlan per DataLoader worker.

        Raises:
            ShardWorkerNotInitializedError: If the worker state has not been
                initialised (lengths, shard_starts, or shard_sizes is None).
        """
        with FatalOnError():
            log: FilteringBoundLogger = cast(
                FilteringBoundLogger,
                structlog.get_logger(),
            )
            ws = ShardWorkerState.get()
            if (
                ws.lengths is None
                or ws.shard_starts is None
                or ws.shard_sizes is None
            ):
                raise ShardWorkerNotInitializedError
            lengths = ws.lengths
            shard_starts = ws.shard_starts
            shard_sizes = ws.shard_sizes
            n_shards = len(shard_starts)

            rng = np.random.default_rng(budget.seed)
            all_ids: npt.NDArray[np.int32] = np.arange(n_shards, dtype=np.int32)
            rng.shuffle(all_ids)
            rank_ids: npt.NDArray[np.int32] = all_ids[
                budget.rank : len(all_ids) : budget.world_size
            ]

            log.info(
                "ffd_plan_start",
                seed=budget.seed,
                n_rank_shards=len(rank_ids),
                num_workers=budget.num_workers,
            )

            worker_plans: list[FFDWorkerPlan] = []
            for w in range(budget.num_workers):
                worker_shard_ids: list[int] = cast(
                    list[int],
                    rank_ids[w :: budget.num_workers].tolist(),
                )
                with ThreadPoolExecutor(
                    max_workers=budget.n_threads,
                ) as pool:
                    futures: list[Future[tuple[list[int], list[int]]]] = [
                        pool.submit(
                            ShardDataLoader.pack_one_shard,
                            sid,
                            lengths,
                            shard_starts,
                            shard_sizes,
                            budget,
                            int(rng.integers(0, np.iinfo(np.int64).max)),
                        )
                        for sid in worker_shard_ids
                    ]
                permutations, batch_ends = zip(
                    *(f.result() for f in futures),
                    strict=False,
                )
                worker_plans.append(
                    FFDWorkerPlan(
                        shard_ids=worker_shard_ids,
                        permutations=permutations,
                        batch_ends=batch_ends,
                        carry_in_sizes=[0] * len(worker_shard_ids),
                    ),
                )

            log.info(
                "ffd_plan_done",
                n_workers=len(worker_plans),
                n_batches=sum(
                    len(be) for wp in worker_plans for be in wp.batch_ends
                ),
            )
            return FFDBatchPlan(worker_plans=worker_plans)

    @staticmethod
    def pack_one_shard(
        shard_id: int,
        lengths: npt.NDArray[np.int16],
        shard_starts: npt.NDArray[np.int32],
        shard_sizes: npt.NDArray[np.int32],
        budget: ShardBudgetParameters,
        seed: int,
    ) -> tuple[list[int], list[int]]:
        """Noisy-sort then FFD-pack one shard into variable-size batches.

        Adds uniform random noise to each protein's length before sorting
        descending, which breaks the strict length ordering and prevents
        every epoch from producing identical batch groupings. The noisy
        sorted lengths are then passed to ``ffd_pack``, which uses a
        First-Fit Decreasing bin-packing algorithm to group proteins into
        batches whose total padded token cost stays within
        ``budget.token_budget``.

        The return value contains two arrays: an inverse permutation that
        maps each protein's original streaming position within the shard
        to its rank in the sorted order, and a ``batch_ends`` cut array
        that encodes where each packed batch ends in sorted order.

        Args:
            shard_id: Global shard index into shard_starts/shard_sizes.
            lengths: Global int16 array of seq_len per protein in shard
                order.
            shard_starts: Start index into lengths for each shard.
            shard_sizes: Protein count for each shard.
            budget: Epoch-level packing parameters; ``max_seq_len``,
                ``token_budget``, and ``noise_magnitude`` are read from
                this object.
            seed: Per-shard RNG seed so threads do not contend on a
                shared RNG.

        Returns:
            Tuple ``(inverse_permutation, batch_ends)`` where
            ``inverse_permutation[local_pos]`` is the protein's sorted
            rank within this shard, and ``batch_ends`` is the FFD
            cumulative cut array.
        """
        start = int(cast(np.intp, shard_starts[shard_id]))
        size = int(cast(np.intp, shard_sizes[shard_id]))
        shard_lengths = lengths[start : start + size].astype(np.int32)
        rng = np.random.default_rng(seed)
        noise = rng.uniform(
            -budget.noise_magnitude,
            budget.noise_magnitude,
            size=size,
        )
        sort_key = shard_lengths.astype(np.float64) + noise
        # argsort of -key gives descending order indices.
        sorted_order: npt.NDArray[np.int32] = np.argsort(
            -sort_key,
            kind="stable",
        ).astype(np.int32)
        # inverse_permutation[streaming_pos] = sorted rank.
        inverse_permutation: npt.NDArray[np.int32] = np.empty_like(sorted_order)
        inverse_permutation[sorted_order] = np.arange(size, dtype=np.int32)
        sorted_effective = np.minimum(
            shard_lengths[sorted_order],
            budget.max_seq_len,
        )
        batch_ends = ShardDataLoader.ffd_pack(
            cast(list[int], sorted_effective.tolist()),
            budget.token_budget,
        )
        return cast(list[int], inverse_permutation.tolist()), batch_ends

    @override
    def __iter__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
    ) -> Iterator[list[Protein]]:
        """Dequeue next plan, inject into dataset, delegate to DataLoader."""
        plan = self.process_queue.get()
        self._cached_len = batch_count_in_ffd_plan(plan)

        self.shard_dataset.set_plan(plan)

        seed = self.base_seed + self.epoch + self.prefetch_epochs
        epoch_budget = dataclasses.replace(
            self.budget,
            seed=seed,
            num_workers=self.num_workers if self.num_workers > 0 else 1,
        )
        cache_path = plan_cache(
            epoch_budget,
            self.shard_sidecar_path,
            self.plan_cache_dir,
        )

        def wait_and_enqueue() -> None:
            """Load from cache or compute, save, then enqueue the next plan."""
            with FatalOnError():
                if cache_path.exists():
                    cached = FFDBatchPlan.model_validate_json(
                        cache_path.read_bytes(),
                    )
                    self._log.info("ffd_plan_cache_hit", seed=seed)
                    self.process_queue.put(cached)
                    return
                future: Future[FFDBatchPlan] = self.process_executor.submit(
                    self.compute_ffd_plan,
                    epoch_budget,
                )
                computed = future.result()
                _ = cache_path.write_bytes(plan.model_dump_json().encode())
                self._log.info("ffd_plan_computed_and_cached", seed=seed)
                self.process_queue.put(computed)

        _ = self.queue_watcher.submit(wait_and_enqueue)
        self.epoch += 1
        return super().__iter__()

    @override
    def __len__(self) -> int:
        """Return exact number of batches this rank yields for current epoch.

        Returns:
            Sum of batches across all shards assigned to this rank, computed
            from the pre-built ShardBatchPlan and updated each epoch in
            __iter__.
        """
        return self._cached_len

    def __del__(self) -> None:
        """Shut down executor and watcher without blocking process exit.

        Uses ``hasattr`` guards because ``__del__`` is called even when
        ``__init__`` raised partway through. If ``super().__init__()`` or
        any line before the executor assignments throws, Python still
        garbage-collects the partially-constructed object and fires
        ``__del__``, but ``process_executor`` and ``queue_watcher`` were
        never set.
        """
        if hasattr(self, "process_executor"):
            self.process_executor.shutdown(wait=False)
        if hasattr(self, "queue_watcher"):
            self.queue_watcher.shutdown(wait=False)


def make_bucketed_data_loaders(
    *,
    cfg: TrainConfig,
    extra_train_args: TrainArgs,
) -> tuple[
    torch.utils.data.DataLoader[list[Protein]],
    torch.utils.data.DataLoader[list[Protein]],
    torch.utils.data.DataLoader[list[Protein]],
]:
    """Build the train shard loader and val/test loaders; auto-detects DDP.

    When ``dist.is_initialized()`` is True the val and test loaders are built
    with ``DistributedSampler``; otherwise they behave identically to the
    single-GPU case. The train loader is always shard-based and rank-aware
    via ``ShardBudgetParameters``.

    Args:
        cfg: Training configuration; controls token budget, sequence length,
            cluster count, and other loader parameters.
        extra_train_args: Paths to the dataset JSONL, splits JSON, shard
            directory, and structured log file; also carries the debug_run
            flag.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    is_ddp: bool = dist.is_initialized()
    rank: int = dist.get_rank() if is_ddp else 0
    world_size: int = dist.get_world_size() if is_ddp else 1

    splits = DatasetSplitsManifest.model_validate_json(
        extra_train_args.keys_for_splits_json.read_bytes(),
    )

    val_set = ProteinDataset(
        extra_train_args.dataset_jsonl,
        splits.validation,
        max_seq_length=cfg.test_loader.max_seq_length,
    )
    test_set = ProteinDataset(
        extra_train_args.dataset_jsonl,
        splits.test,
        max_seq_length=cfg.test_loader.max_seq_length,
    )
    train_names = (
        splits.train[:252] if extra_train_args.debug_run else splits.train
    )

    budget = ShardBudgetParameters(
        shard_dir=extra_train_args.shard_dir,
        structlog_path=extra_train_args.structlog_jsonl,
        token_budget=cfg.train_loader.token_budget,
        max_seq_len=cfg.train_loader.max_seq_length,
        seed=cfg.train_loader.seed,
        n_threads=cfg.train_loader.n_threads,
        n_proteins_in_shard=len(train_names) // cfg.train_loader.n_shards,
        world_size=world_size,
        rank=rank,
        noise_magnitude=cfg.train_loader.noise_magnitude,
        num_workers=cfg.train_loader.num_workers,
    )
    train_set = ProteinShardDataset(  # this should take the structlog_jsonl too
        budget_parameters=budget,
        names=train_names,
        dataset_jsonl=extra_train_args.dataset_jsonl,
    )

    train_loader = ShardDataLoader(
        dataset=train_set,
        budget=budget,
        num_workers=cfg.train_loader.num_workers,
        batch_prefetch_depth=cfg.train_loader.batch_prefetch_depth,
        prefetch_epochs=cfg.train_loader.epoch_prefetch_depth,
    )

    if is_ddp:
        val_sampler = DistributedSampler(
            val_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
        test_sampler = DistributedSampler(
            test_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=cfg.test_loader.batch_size,
            sampler=val_sampler,
            num_workers=cfg.test_loader.num_workers,
            pin_memory=True,
            collate_fn=identity_collate,
        )
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=cfg.test_loader.batch_size,
            sampler=test_sampler,
            num_workers=cfg.test_loader.num_workers,
            pin_memory=True,
            collate_fn=identity_collate,
        )
    else:
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=cfg.test_loader.batch_size,
            shuffle=False,
            num_workers=cfg.test_loader.num_workers,
            pin_memory=True,
            collate_fn=identity_collate,
        )
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=cfg.test_loader.batch_size,
            shuffle=False,
            num_workers=cfg.test_loader.num_workers,
            pin_memory=True,
            collate_fn=identity_collate,
        )

    return (
        train_loader,
        cast(torch.utils.data.DataLoader[list[Protein]], val_loader),
        cast(torch.utils.data.DataLoader[list[Protein]], test_loader),
    )
