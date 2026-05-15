"""Dataset and data loading utilities for protein structure files."""

import io
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import torch
import torch.utils.data
from helpers.atom_utils import center_positions, make_fixed_size, make_np_example
from helpers.featurize import ProteinBatch
from torch.utils.data.distributed import DistributedSampler
from train.train_config import TrainConfig


def _to_protein_batch(samples: list[Mapping[str, torch.Tensor | str]]) -> ProteinBatch:
    """Collate a list of per-protein dicts into a ProteinBatch."""
    return ProteinBatch(
        atom_positions=torch.stack(
            cast(list[torch.Tensor], [s["atom_positions"] for s in samples])
        ),
        atom_mask=torch.stack(cast(list[torch.Tensor], [s["atom_mask"] for s in samples])),
        residue_index=torch.stack(cast(list[torch.Tensor], [s["residue_index"] for s in samples])),
        seq=cast(list[str], [s["seq"] for s in samples]),
    )


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

    def __getitem__(self, idx: int) -> Mapping[str, torch.Tensor | str]:
        """Return the parsed JSON entry at the given index."""
        self._open()
        self._file.seek(self._offsets[idx])  # type: ignore[union-attr]
        entry = json.loads(self._file.readline())  # type: ignore[union-attr]

        np_example = make_np_example(entry["coords"])
        center_positions(np_example)
        make_fixed_size(np_example, self.max_seq_length)

        sample = {k: torch.tensor(v, dtype=torch.float32) for k, v in np_example.items()}
        sample["seq"] = entry["seq"][: self.max_seq_length]
        return sample


def make_data_loaders(
    cfg: TrainConfig,
    jsonl_path: str | Path,
    splits_path: str | Path,
    num_workers: int = 0,
    debug_run: bool = True,
) -> tuple[
    torch.utils.data.DataLoader[ProteinBatch],
    torch.utils.data.DataLoader[ProteinBatch],
    torch.utils.data.DataLoader[ProteinBatch],
]:
    """Build train, validation, and test DataLoaders from a JSONL file and a splits JSON.

    Args:
        cfg:         TrainConfig — batch_size and max_seq_length are read from
                     cfg.loader.
        jsonl_path:  Path to the JSONL protein dataset.
        splits_path: Path to a JSON file with keys "train", "validation", and
                     "test", each a list of entry names.
        num_workers: DataLoader worker processes (0 = main process only).
        debug_run:   If True, restrict each split to 252 samples for fast iteration.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    with open(splits_path) as f:
        splits: dict[str, list[str]] = json.load(f)

    train_set = ProteinDataset(
        jsonl_path,
        splits["train"],
        max_seq_length=cfg.train_loader.max_seq_length,
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

    if debug_run:
        subset_sampler = torch.utils.data.SubsetRandomSampler(range(252))

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=cfg.train_loader.batch_size,
            sampler=subset_sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=_to_protein_batch,
        )
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=cfg.train_loader.batch_size,
            sampler=subset_sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=_to_protein_batch,
        )
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=cfg.train_loader.batch_size,
            sampler=subset_sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=_to_protein_batch,
        )

    else:
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=cfg.train_loader.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=_to_protein_batch,
        )
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=cfg.test_loader.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=_to_protein_batch,
        )
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=cfg.test_loader.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=_to_protein_batch,
        )

    return cast(
        tuple[
            torch.utils.data.DataLoader[ProteinBatch],
            torch.utils.data.DataLoader[ProteinBatch],
            torch.utils.data.DataLoader[ProteinBatch],
        ],
        (train_loader, val_loader, test_loader),
    )


def make_ddp_data_loaders(
    cfg: TrainConfig,
    jsonl_path: str | Path,
    splits_path: str | Path,
    rank: int,
    world_size: int,
    num_workers: int = 0,
) -> tuple[
    torch.utils.data.DataLoader[ProteinBatch],
    torch.utils.data.DataLoader[ProteinBatch],
    torch.utils.data.DataLoader[ProteinBatch],
]:
    """Build train/val/test DataLoaders backed by DistributedSampler for DDP training."""
    with open(splits_path) as f:
        splits: dict[str, list[str]] = json.load(f)

    train_set = ProteinDataset(
        jsonl_path, splits["train"], max_seq_length=cfg.train_loader.max_seq_length
    )
    val_set = ProteinDataset(
        jsonl_path, splits["validation"], max_seq_length=cfg.test_loader.max_seq_length
    )
    test_set = ProteinDataset(
        jsonl_path, splits["test"], max_seq_length=cfg.test_loader.max_seq_length
    )

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_set, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=cfg.train_loader.batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_to_protein_batch,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=cfg.test_loader.batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_to_protein_batch,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=cfg.test_loader.batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_to_protein_batch,
    )
    return cast(
        tuple[
            torch.utils.data.DataLoader[ProteinBatch],
            torch.utils.data.DataLoader[ProteinBatch],
            torch.utils.data.DataLoader[ProteinBatch],
        ],
        (train_loader, val_loader, test_loader),
    )


class _FileLogProcessor:
    """Structlog processor that appends JSON lines to a file, then passes the event dict through.

    Opens the log file with line-buffering (buffering=1) on construction so each line is flushed
    to disk immediately — critical for long runs where buffered writes could be lost on a crash.

    Context manager (__enter__ / __exit__): closes the file when the with-block ends. Use via
    contextlib.ExitStack so the file is closed even if the caller raises.

    Structlog processor (__call__): serializes the event dict to JSON and appends it, then
    returns the dict unchanged so downstream processors (e.g. ConsoleRenderer) still run.
    Insert before ConsoleRenderer so the raw dict reaches disk before it is colorized.
    """

    def __init__(self, path: str) -> None:
        self._f = open(path, "w", buffering=1)  # noqa: SIM115

    def __enter__(self) -> "_FileLogProcessor":
        return self

    def __exit__(self, *_: object) -> None:
        self._f.close()

    def __call__(
        self, _logger: object, _method: str | None, event_dict: dict[str, object]
    ) -> dict[str, object]:
        self._f.write(json.dumps(event_dict) + "\n")
        return event_dict
