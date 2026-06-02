"""Dataset and data loading utilities for protein structure files."""

import io
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data
from helpers.atom_utils import (
    center_positions,
    make_fixed_size,
    make_np_example,
    truncate_to_length,
)
from helpers.batch_types import ProteinBatch
from helpers.bucketed_sampler import BucketedBatchSampler
from helpers.cluster_index import ClusterIndex
from jaxtyping import Float
from torch.utils.data.distributed import DistributedSampler
from train.train_config import TrainConfig


class ProteinDataset(torch.utils.data.Dataset[Mapping[str, torch.Tensor | str]]):
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
        self.jsonl_path = Path(jsonl_path)
        self.max_seq_length = max_seq_length
        self._file: io.BufferedReader | None = None

        name_set = set(names)
        offsets: list[int] = []
        byte_pos = 0
        with open(self.jsonl_path, "rb") as f:
            for raw_line in f:
                if json.loads(raw_line)["name"] in name_set:
                    offsets.append(byte_pos)
                byte_pos += len(raw_line)

        self._offsets = offsets

    # ------------------------------------------------------------------
    # File-handle lifecycle — excluded from pickle so multiprocessing works

    def _open(self) -> None:
        if self._file is None:
            self._file = open(self.jsonl_path, "rb")  # noqa: SIM115

    def __getstate__(self) -> dict[str, object]:
        """Return picklable state with the open file handle set to None."""
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def __del__(self) -> None:
        """Close the underlying file handle on deletion."""
        if self._file is not None:
            self._file.close()

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of entries in the dataset."""
        return len(self._offsets)

    def __getitem__(self, idx: int) -> Mapping[str, Float[torch.Tensor, "..."] | str]:
        """Return the parsed JSON entry at the given index."""
        self._open()
        self._file.seek(self._offsets[idx])  # type: ignore[union-attr]
        entry = json.loads(self._file.readline())  # type: ignore[union-attr]

        np_example = make_np_example(entry["coords"])
        center_positions(np_example)
        make_fixed_size(np_example, self.max_seq_length)  # truncation

        sample = {k: torch.tensor(v, dtype=torch.float32) for k, v in np_example.items()}
        sample["seq"] = entry["seq"][: self.max_seq_length]  # padding?
        return sample


class ClusteredProteinDataset(
    torch.utils.data.Dataset[Mapping[str, Float[torch.Tensor, "..."] | str]]
):
    """Lazy-loading Dataset backed by per-cluster JSONL files built by ClusterIndex.

    At construction, builds or loads a ClusterIndex (writing 65 cluster files the first
    time, reading cached files thereafter). __getitem__ seeks directly into the correct
    cluster file and returns the protein at its actual residue count — no padding. Items
    are variable-length; padding is deferred to the collate function.

    Compatible with num_workers > 0: all file handles are excluded from pickling and
    re-opened lazily inside each worker process.

    Args:
        jsonl_path:     Path to the source JSONL protein dataset.
        names:          List of entry names to include.
        max_seq_length: Per-protein hard truncation ceiling; proteins longer than this
                        are truncated at load time. Default 256.
        token_budget:   Packing budget passed to ClusterIndex; defines bin widths and
                        the overflow cluster boundary. Default 512.
        n_clusters:     Number of regular length clusters. Default 64.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        names: list[str],
        max_seq_length: int = 256,
        token_budget: int = 512,
        n_clusters: int = 64,
    ) -> None:
        self.max_seq_length = max_seq_length
        self.token_budget = token_budget
        self.cluster_index = ClusterIndex(jsonl_path, names, token_budget, n_clusters)
        self._files: list[io.BufferedReader | None] = [None] * (n_clusters + 1)

    def _open(self, k: int) -> None:
        """Open cluster file k lazily if not already open."""
        if self._files[k] is None:
            self._files[k] = open(self.cluster_index.cluster_file(k), "rb")  # noqa: SIM115

    def __getstate__(self) -> dict[str, object]:
        """Return picklable state with all open file handles set to None."""
        state = self.__dict__.copy()
        state["_files"] = [None] * len(self._files)
        return state

    def __del__(self) -> None:
        """Close all open cluster file handles on deletion."""
        for f in self._files:
            if f is not None:
                f.close()

    def __len__(self) -> int:
        """Return the total number of proteins across all clusters."""
        return len(self.cluster_index)

    def __getitem__(self, idx: int) -> Mapping[str, Float[torch.Tensor, "..."] | str]:
        """Return the protein at flat index idx at its actual (un-padded) length.

        Args:
            idx: Flat dataset index in [0, len(self)).

        Returns:
            Dict with atom_positions (N, 37, 3), atom_mask (N, 37),
            residue_index (N,), and seq (str), where N <= token_budget.
        """
        cluster_id = int(self.cluster_index.flat_to_cluster[idx])
        local_idx = int(self.cluster_index.flat_to_local[idx])
        offset = int(self.cluster_index.cluster_offsets[cluster_id][local_idx])

        self._open(cluster_id)
        self._files[cluster_id].seek(offset)  # type: ignore[union-attr]
        raw = self._files[cluster_id].readline()  # type: ignore[union-attr]
        entry: Mapping[str, object] = json.loads(raw)

        np_example = make_np_example(entry["coords"])  # type: ignore[arg-type]
        center_positions(np_example)
        truncate_to_length(np_example, self.max_seq_length)

        sample: dict[str, Float[torch.Tensor, "..."] | str] = {
            k: torch.tensor(v, dtype=torch.float32) for k, v in np_example.items()
        }
        seq = cast(str, entry["seq"])
        sample["seq"] = seq[: self.max_seq_length]
        return sample


def to_protein_batch_dynamic(
    samples: list[Mapping[str, Float[torch.Tensor, "..."] | str]],
) -> ProteinBatch:
    """Collate variable-length protein samples into a ProteinBatch, padding to batch max.

    Pads each sample's tensors along axis 0 to the longest sample in the batch.
    Within a length-bucketed batch, all samples are within one bucket width of each
    other, so padding is near-zero.

    Args:
        samples: List of per-protein dicts from ClusteredProteinDataset.__getitem__.

    Returns:
        ProteinBatch with tensors of shape (B, max_len, ...).
    """
    max_len = max(cast(torch.Tensor, s["atom_positions"]).shape[0] for s in samples)
    padded_positions: list[Float[torch.Tensor, "..."]] = []
    padded_mask: list[Float[torch.Tensor, "..."]] = []
    padded_residue_index: list[Float[torch.Tensor, "..."]] = []
    seqs: list[str] = []

    for s in samples:
        pos = cast(torch.Tensor, s["atom_positions"])  # (n, 37, 3)
        mask = cast(torch.Tensor, s["atom_mask"])  # (n, 37)
        ridx = cast(torch.Tensor, s["residue_index"])  # (n,)
        n = pos.shape[0]
        pad = max_len - n
        if pad > 0:
            pos = F.pad(pos, (0, 0, 0, 0, 0, pad))
            mask = F.pad(mask, (0, 0, 0, pad))
            ridx = F.pad(ridx, (0, pad))
        padded_positions.append(pos)
        padded_mask.append(mask)
        padded_residue_index.append(ridx)
        seqs.append(cast(str, s["seq"]))

    return ProteinBatch(
        atom_positions=torch.stack(padded_positions),
        atom_mask=torch.stack(padded_mask),
        residue_index=torch.stack(padded_residue_index),
        seq=seqs,
    )


def make_bucketed_data_loaders(
    *,
    cfg: TrainConfig,
    jsonl_path: str | Path,
    splits_path: str | Path,
    num_workers: int,
    debug_run: bool,
) -> tuple[
    torch.utils.data.DataLoader[ProteinBatch],
    torch.utils.data.DataLoader[ProteinBatch],
    torch.utils.data.DataLoader[ProteinBatch],
    BucketedBatchSampler,
]:
    """Build bucketed train loader and val/test loaders; auto-detects DDP.

    When called inside ``with DistProcessGroup():``, ``dist.is_initialized()`` is
    True and the loaders are built with DDP-aware samplers (BucketedBatchSampler
    sharded by rank, DistributedSampler for val/test).  Outside a process group the
    loaders behave identically to the single-GPU case.

    Args:
        cfg:         TrainConfig; cfg.train_loader.token_budget controls packing budget.
        jsonl_path:  Path to the JSONL protein dataset.
        splits_path: Path to a JSON file with keys "train", "validation", "test".
        num_workers: DataLoader worker processes per rank.
        debug_run:   If True, restrict training to the first 252 protein names.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, train_sampler).
    """
    is_ddp: bool = dist.is_initialized()
    rank: int = dist.get_rank() if is_ddp else 0
    world_size: int = dist.get_world_size() if is_ddp else 1

    with open(splits_path) as f:
        splits: dict[str, list[str]] = json.load(f)

    train_names = splits["train"][:252] if debug_run else splits["train"]

    train_set = ClusteredProteinDataset(
        jsonl_path,
        train_names,
        max_seq_length=cfg.train_loader.max_seq_length,
        token_budget=cfg.train_loader.token_budget,
    )
    val_set = ProteinDataset(
        jsonl_path,
        splits["validation"],
        max_seq_length=cfg.test_loader.max_seq_length,
    )
    test_set = ProteinDataset(
        jsonl_path,
        splits["test"],
        max_seq_length=cfg.test_loader.max_seq_length,
    )

    train_sampler = BucketedBatchSampler(
        train_set.cluster_index,
        token_budget=cfg.train_loader.token_budget,
        max_seq_len=cfg.train_loader.max_seq_length,
        world_size=world_size,
        rank=rank,
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=to_protein_batch_dynamic,
    )

    if is_ddp:
        val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False)
        test_sampler = DistributedSampler(
            test_set, num_replicas=world_size, rank=rank, shuffle=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=cfg.test_loader.batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=to_protein_batch_dynamic,
        )
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=cfg.test_loader.batch_size,
            sampler=test_sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=to_protein_batch_dynamic,
        )
    else:
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=cfg.test_loader.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=to_protein_batch_dynamic,
        )
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=cfg.test_loader.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=to_protein_batch_dynamic,
        )

    return cast(
        """tuple[
            torch.utils.data.DataLoader[ProteinBatch],
            torch.utils.data.DataLoader[ProteinBatch],
            torch.utils.data.DataLoader[ProteinBatch],
            BucketedBatchSampler
        ]""",
        (train_loader, val_loader, test_loader, train_sampler),
    )
