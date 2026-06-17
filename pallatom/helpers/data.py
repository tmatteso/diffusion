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
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import ClassVar, cast

import numpy as np
import numpy.typing as npt
import structlog
import torch
import torch.distributed as dist
import torch.utils.data
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
from helpers.useful_objects import TrainArgs
from pydantic import BaseModel, ConfigDict, Field, RootModel
from structlog.typing import FilteringBoundLogger
from torch.utils.data.distributed import DistributedSampler
from train.train_config import TrainConfig
from typing_extensions import Self, override
from webdataset.compat import WebDataset
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


@dataclasses.dataclass(frozen=True)
class ClusterMetadataEntry:
    """Metadata collected in pass 1 for one protein.

    Attributes:
        name: Protein entry name.
        seq_len: Sequence length in residues.
        byte_offset: Byte position of this entry's line in the source JSONL.
    """

    name: str
    seq_len: int
    byte_offset: int

    def __lt__(self, other: Self) -> bool:
        """Compare entries by sequence length for sort order within a cluster.

        Args:
            other: Another ClusterMetadataEntry to compare against.

        Returns:
            True if this entry's seq_len is less than other's.
        """
        return self.seq_len < other.seq_len


class ShardMetadata(BaseModel):
    """Persisted record of the parameters used to build a shard directory.

    Serialised as ``shard_metadata.json`` alongside the shard tars and read
    back at train time to verify that an existing shard directory matches
    the current configuration before re-use.

    Attributes:
        model_config: Pydantic config — frozen to prevent mutation after init.
        names_hash: SHA-256 hex digest of the sorted protein names list.
        token_budget: Maximum padded token cost per batch used during sharding.
        n_clusters: Number of length-based clusters the dataset was split into.
        shard_size: Maximum number of proteins per shard.
        n_shards: Total number of shard tars written.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    names_hash: str
    token_budget: int
    n_clusters: int
    shard_size: int
    n_shards: int


@dataclasses.dataclass
class ShardBatchPlan:
    """Compact per-epoch batch plan for shard-based streaming.

    Attributes:
        shard_order: Global shard IDs this rank will stream this epoch.
        flat_batch_ends: Local cumulative batch-end positions (protein count
            within each shard at each cut), concatenated across all shards in
            shard_order order.
        batches_per_shard: Number of batches produced by each shard in
            shard_order.
    """

    shard_order: npt.NDArray[np.int32]
    flat_batch_ends: npt.NDArray[np.int32]
    batches_per_shard: npt.NDArray[np.int32]


@dataclasses.dataclass
class FFDWorkerPlan:
    """One DataLoader worker's per-epoch plan for FFD batch streaming.

    Attributes:
        shard_ids: Global shard IDs this worker streams this epoch, in
            iteration order.
        permutations: For each shard, an int32 array mapping streaming
            position (i.e. WebDataset arrival order) to the protein's index
            in the prepended-and-permuted local sequence.
        batch_ends: For each shard, an int32 array of cumulative batch-end
            positions in the prepended-and-permuted sequence
            ``carry_in_sizes[k] + permuted shard k proteins``.
        carry_in_sizes: For each shard, the number of proteins prepended
            from the previous shard's carry-over. Always zero for length-
            disjoint shards (the typical case with globally sorted shards),
            but stored explicitly so workers can validate the carry buffer.
    """

    shard_ids: npt.NDArray[np.int32]
    permutations: list[npt.NDArray[np.int32]]
    batch_ends: list[npt.NDArray[np.int32]]
    carry_in_sizes: npt.NDArray[np.int32]


@dataclasses.dataclass
class FFDBatchPlan:
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
        n_clusters: Number of length-based clusters to partition the dataset
            into before sharding.
    """

    def __init__(
        self,
        budget_parameters: ShardBudgetParameters,
        names: list[str],
        dataset_jsonl: Path,
        n_clusters: int,
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
            "FilteringBoundLogger",
            structlog.get_logger(),
        )
        self.n_clusters: int = n_clusters
        self.token_budget: int = budget_parameters.token_budget
        self.max_seq_length: int = budget_parameters.max_seq_len

        self.bin_width: int = self.token_budget // self.n_clusters
        # Construct the shards if they do not already exist.
        if not self.shard_metadata_path.exists():
            self._log.info(
                "shards_do_not_exist",
                shard_manifest_file=self.shard_metadata_path,
            )
            cluster_metadata: list[list[ClusterMetadataEntry]] = (
                self.compute_clusters()
            )
            n_shards: int
            all_lengths: list[int]
            shard_sizes_list: list[int]
            n_shards, all_lengths, shard_sizes_list = self.compute_shards(
                cluster_metadata,
            )
            _ = self.write_shard_metadata_sidecar(
                n_shards,
                all_lengths,
                shard_sizes_list,
            )

        # prefill the batch
        self._plan: ShardBatchPlan | None = None

    def compute_clusters(self) -> list[list[ClusterMetadataEntry]]:
        """Scan the source JSONL and assign each protein to a length cluster.

        Reads the dataset JSONL in a single sequential pass, filters to
        ``self.names``, and assigns each protein to one of ``self.n_clusters``
        equal-width length bins (plus an overflow bin for sequences longer than
        ``self.token_budget``). Each cluster is sorted ascending by seq_len.

        Returns:
            List of n_clusters + 1 clusters, each a sorted list of
            ClusterMetadataEntry objects.
        """
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        name_set = set(self.names)

        # ── Pass 1: collect metadata ──────────────────────────────────────────
        cluster_metadata: list[list[ClusterMetadataEntry]] = [
            [] for _ in range(self.n_clusters + 1)
        ]
        with self.dataset_jsonl.open("rb") as f:
            byte_pos = 0
            for raw_line in f:
                entry = ProteinEntry.model_validate_json(raw_line)
                if entry.name in name_set:
                    seq_len = len(entry.seq)
                    if seq_len > self.token_budget:
                        k = self.n_clusters
                    else:  # overflow bin
                        k = min(
                            (seq_len - 1) // self.bin_width,
                            self.n_clusters - 1,
                        )
                    cluster_metadata[k].append(
                        ClusterMetadataEntry(
                            name=entry.name,
                            seq_len=seq_len,
                            byte_offset=byte_pos,
                        ),
                    )
                byte_pos += len(raw_line)

        for cluster in cluster_metadata:
            cluster.sort()

        self._log.info(
            "clusters_computed",
            n_shards=self.n_clusters,
        )
        return cluster_metadata

    def compute_shards(
        self,
        cluster_metadata: list[list[ClusterMetadataEntry]],
    ) -> tuple[int, list[int], list[int]]:
        """Write WebDataset tar shards from sorted cluster metadata.

        Iterates over clusters in cluster_metadata, slicing each into
        fixed-size shards of ``self.n_proteins_in_shard`` proteins. Each shard
        is written as ``shard_{id:05d}.tar`` in ``self.shard_dir``. Proteins
        are retrieved from the source JSONL by stored byte offsets.

        Args:
            cluster_metadata: Per-cluster sorted lists of ClusterMetadataEntry
                objects, as returned by compute_clusters.

        Returns:
            Tuple of (n_shards, all_lengths, shard_sizes_list) where
            all_lengths contains the seq_len of every protein in global shard
            order and shard_sizes_list contains the protein count per shard.
        """
        all_lengths: list[int] = []
        shard_sizes_list: list[int] = []
        shard_id: int = 0

        with self.dataset_jsonl.open("rb") as src:
            for cluster in cluster_metadata:
                for shard_start in range(
                    0,
                    max(len(cluster), 1),
                    self.n_proteins_in_shard,
                ):
                    shard_entries: list[ClusterMetadataEntry] = cluster[
                        shard_start : shard_start + self.n_proteins_in_shard
                    ]
                    if not shard_entries:
                        continue
                    shard_path: Path = (
                        self.shard_dir / f"shard_{shard_id:05d}.tar"
                    )
                    with TarWriter(str(shard_path)) as sink:
                        for local_idx, cluster_metadata_entry in enumerate(
                            shard_entries,
                        ):
                            _ = src.seek(cluster_metadata_entry.byte_offset)
                            raw_line = src.readline()
                            _ = sink.write(
                                {
                                    "__key__": f"{local_idx:06d}",
                                    "json": raw_line,
                                },
                            )
                            all_lengths.append(cluster_metadata_entry.seq_len)
                    shard_sizes_list.append(len(shard_entries))
                    shard_id += 1

        n_shards = shard_id
        self._log.info(
            "shards_computed",
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
            n_clusters=self.n_clusters,
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

    def set_plan(self, plan: ShardBatchPlan) -> None:
        """Inject the ShardBatchPlan for the next epoch.

        Called by ShardBatchSampler.set_epoch before the DataLoader iterates.
        Because the DataLoader uses persistent_workers=False, workers restart
        each epoch and receive the updated plan via pickling.

        Args:
            plan: Pre-computed plan from ShardBatchSampler.
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

    @staticmethod
    def cut_stream_into_batches(
        protein_iter: Iterator[Protein],
        local_ends: npt.NDArray[np.int32],
    ) -> Iterator[list[Protein]]:
        """Cut a sequential protein stream into batches using pre-computed ends.

        Args:
            protein_iter: Sequential stream of Protein objects from one shard.
            local_ends: Local cumulative batch-end positions (1-indexed protein
                count within the shard at each cut). Produced by _pack_shard.

        Yields:
            list[Protein]: Non-empty lists of Protein objects, one per batch
                cut.
        """
        batch: list[Protein] = []
        cut_idx = 0
        n_cuts = len(local_ends)
        for i, protein in enumerate(protein_iter):
            batch.append(protein)
            if cut_idx < n_cuts and i + 1 == int(
                cast("np.int32", local_ends[cut_idx]),
            ):
                yield batch
                batch = []
                cut_idx += 1
        if batch:
            yield batch

    @override
    def __iter__(self) -> Iterator[list[Protein]]:
        """Yield complete protein batches for this worker's assigned shards.

        If set_plan has not been called, yields nothing. Worker w of
        num_workers takes shard_order[w::num_workers]. For each assigned shard,
        streams the tar via WebDataset and cuts proteins into batches using the
        pre-computed local_ends slice from flat_batch_ends.
        """
        plan = self._plan
        if plan is None:
            return

        worker_info = torch.utils.data.get_worker_info()
        worker_id: int = worker_info.id if worker_info is not None else 0
        num_workers: int = (
            worker_info.num_workers if worker_info is not None else 1
        )

        shard_batch_offsets: npt.NDArray[np.int32] = np.concatenate(
            [np.array([0], dtype=np.int32), np.cumsum(plan.batches_per_shard)],
        )

        for pos in range(worker_id, len(plan.shard_order), num_workers):
            sid = int(cast("np.int32", plan.shard_order[pos]))
            batch_start = int(cast("np.int32", shard_batch_offsets[pos]))
            batch_end = int(cast("np.int32", shard_batch_offsets[pos + 1]))
            local_ends = plan.flat_batch_ends[batch_start:batch_end]

            url = str(self.shard_dir / f"shard_{sid:05d}.tar")
            decoded: object = cast(
                "object",
                WebDataset(url),
            )
            ds: Iterable[dict[str, object]] = cast(
                "Iterable[dict[str, object]]",
                decoded,
            )
            protein_iter: Iterator[Protein] = (
                self.parse_protein(s) for s in ds
            )
            yield from self.cut_stream_into_batches(protein_iter, local_ends)


def identity_collate(batch: list[Protein]) -> list[Protein]:
    """Pass pre-assembled protein batches through without default stacking."""
    return batch


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
            "FilteringBoundLogger",
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
        self.process_queue: queue.Queue[ShardBatchPlan] = queue.Queue(
            maxsize=self.prefetch_epochs,
        )
        self._log.info(
            "process_queue_initialized",
            queue_depth=self.prefetch_epochs,
        )
        first_plan: ShardBatchPlan | None = None
        for offset in range(self.prefetch_epochs):
            epoch_budget = dataclasses.replace(
                budget,
                seed=self.base_seed + offset,
            )
            plan: ShardBatchPlan = self.process_executor.submit(
                self.compute_shard_plan,
                epoch_budget,
            ).result()
            self.process_queue.put(plan)
            if first_plan is None:
                first_plan = plan

        self._cached_len: int = (
            int(
                first_plan.batches_per_shard.sum(),  # pyright: ignore[reportAny]
            )
            if first_plan is not None
            else 0
        )
        self._log.info(
            "shard_prefetch_complete",
            prefetch_epochs=self.prefetch_epochs,
        )

    # these two must be static methods
    @staticmethod
    def pack_shard(
        shard_id: int,
        lengths: npt.NDArray[np.int16],
        shard_starts: npt.NDArray[np.int32],
        shard_sizes: npt.NDArray[np.int32],
        max_seq_len: int,
        token_budget: int,
    ) -> npt.NDArray[np.int32]:
        """Greedy-pack a pre-sorted shard; return local cumulative batch ends.

        Each shard is sorted ascending by seq_len, so the current protein's
        effective length is always the max in any in-progress batch. The batch
        cost check is (current_count + 1) * eff_len <= token_budget.

        Args:
            shard_id: Global shard index to index shard_starts/shard_sizes.
            lengths: Global int16 array of seq_len per protein in shard order.
            shard_starts: Start index into lengths for each shard.
            shard_sizes: Protein count for each shard.
            max_seq_len: Hard truncation ceiling applied to effective lengths.
            token_budget: Maximum padded token cost per batch.

        Returns:
            Int32 array of local cumulative batch-end positions (1-indexed
            protein count within the shard at each cut).
        """
        start_of_current_shard = int(cast("np.intp", shard_starts[shard_id]))
        size_of_current_shard = int(cast("np.intp", shard_sizes[shard_id]))

        effective_lengths = np.minimum(
            lengths[
                start_of_current_shard : start_of_current_shard
                + size_of_current_shard
            ].astype(np.int32),
            max_seq_len,
        )
        batches: list[int] = []
        current_item_count = 0
        for local_index, residue_count in enumerate(
            cast("list[int]", effective_lengths.tolist()),
        ):
            # eventually this would include the atom attention calculation
            cost_from_item = residue_count * residue_count

            if cost_from_item > token_budget:
                if current_item_count > 0:
                    # end the batch before adding the new item
                    batches.append(local_index)
                    current_item_count = 0
                # end the batch, the batch is only the oversize item
                batches.append(local_index + 1)
            elif (current_item_count + 1) * cost_from_item > token_budget:
                # end the batch before adding the new item
                batches.append(local_index)
                # new batch starts with the new item
                current_item_count = 1
            else:
                # continue filling the batch
                current_item_count += 1
        # fill another batch from trailing elements at end of loop.
        if current_item_count > 0:
            batches.append(size_of_current_shard)
        return np.array(batches, dtype=np.int32)

    @staticmethod
    def compute_shard_plan(budget: ShardBudgetParameters) -> ShardBatchPlan:
        """Compute one epoch's ShardBatchPlan inside the WorkerState subprocess.

        Shuffles all shard IDs with RNG(budget.seed), assigns this rank's slice
        (rank::world_size), then greedy-packs each assigned shard in parallel
        via a ThreadPoolExecutor. Wrapped in FatalOnError so subprocess
        failures are logged before propagating.

        Args:
            budget: Scalar parameters for this epoch's plan computation.

        Returns:
            ShardBatchPlan with shard_order, flat_batch_ends, batches_per_shard.

        Raises:
            ShardWorkerNotInitializedError: If the worker state has not been
                initialized (lengths, shard_starts, or shard_sizes is None).
        """
        with FatalOnError():
            log: FilteringBoundLogger = cast(
                "FilteringBoundLogger",
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
            # Compute separate shards for each GPU. rank strided.
            # Syntax is sequence[start:stop:step].
            rank_ids: npt.NDArray[np.int32] = all_ids[
                budget.rank : len(all_ids) : budget.world_size
            ]

            log.info(
                "shard_plan_start",
                seed=budget.seed,
                n_rank_shards=len(rank_ids),
            )

            with ThreadPoolExecutor(max_workers=budget.n_threads) as pool:
                futures: list[Future[npt.NDArray[np.int32]]] = [
                    pool.submit(
                        ShardDataLoader.pack_shard,
                        int(sid),
                        lengths,
                        shard_starts,
                        shard_sizes,
                        budget.max_seq_len,
                        budget.token_budget,
                    )
                    for sid in cast("Iterable[np.int32]", rank_ids)
                ]
                batches_across_all_shards: list[npt.NDArray[np.int32]] = [
                    f.result() for f in futures
                ]

            batches_per_shard: npt.NDArray[np.int32] = np.array(
                [
                    len(shard_batches)
                    for shard_batches in batches_across_all_shards
                ],
                dtype=np.int32,
            )
            flattened_batches_across_all_shards: npt.NDArray[np.int32] = (
                np.concatenate(batches_across_all_shards)
                if batches_across_all_shards
                else np.empty(0, dtype=np.int32)  # why is this else here?
            )
            log.info(
                "shard_plan_done",
                n_batches=int(
                    np.intp(
                        batches_per_shard.sum(),  # pyright: ignore[reportAny]
                    ),
                ),
                n_shards=len(rank_ids),
            )
            return ShardBatchPlan(
                shard_order=rank_ids,
                flat_batch_ends=flattened_batches_across_all_shards,
                batches_per_shard=batches_per_shard,
            )

    @override
    def __iter__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
    ) -> Iterator[list[Protein]]:
        """Dequeue next plan, inject into dataset, delegate to DataLoader.

        Returns:
            Iterator of list[Protein] batches from the underlying DataLoader.
        """
        plan = self.process_queue.get()
        self._cached_len = int(
            plan.batches_per_shard.sum(),  # pyright: ignore[reportAny]
        )

        self.shard_dataset.set_plan(plan)

        seed = self.base_seed + self.epoch + self.prefetch_epochs
        epoch_budget = dataclasses.replace(self.budget, seed=seed)
        future: Future[ShardBatchPlan] = self.process_executor.submit(
            self.compute_shard_plan,
            epoch_budget,
        )

        def wait_and_enqueue() -> None:
            """Block on subprocess future and enqueue resolved plan."""
            with FatalOnError():
                self.process_queue.put(future.result())
                self._log.info("shard_plan_enqueued", seed=seed)

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
        """Shut down executor and watcher without blocking process exit."""
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
    torch.utils.data.DataLoader[Protein],
    torch.utils.data.DataLoader[Protein],
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

    budget = ShardBudgetParameters(
        shard_dir=extra_train_args.shard_dir,
        structlog_path=extra_train_args.structlog_jsonl,
        token_budget=cfg.train_loader.token_budget,
        max_seq_len=cfg.train_loader.max_seq_length,
        seed=cfg.train_loader.seed,
        n_threads=cfg.train_loader.n_threads,
        n_proteins_in_shard=cfg.train_loader.n_proteins_in_shard,
        world_size=world_size,
        rank=rank,
        noise_magnitude=cfg.train_loader.noise_magnitude,
        num_workers=cfg.train_loader.num_workers,
    )

    train_names = (
        splits.train[:252] if extra_train_args.debug_run else splits.train
    )

    train_set = ProteinShardDataset(  # this should take the structlog_jsonl too
        budget_parameters=budget,
        names=train_names,
        dataset_jsonl=extra_train_args.dataset_jsonl,
        n_clusters=cfg.train_loader.n_clusters,
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
        )
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=cfg.test_loader.batch_size,
            sampler=test_sampler,
            num_workers=cfg.test_loader.num_workers,
            pin_memory=True,
        )
    else:
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=cfg.test_loader.batch_size,
            shuffle=False,
            num_workers=cfg.test_loader.num_workers,
            pin_memory=True,
        )
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=cfg.test_loader.batch_size,
            shuffle=False,
            num_workers=cfg.test_loader.num_workers,
            pin_memory=True,
        )

    return (train_loader, val_loader, test_loader)
