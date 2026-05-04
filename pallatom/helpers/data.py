import io
import json
import torch
import torch.utils.data
from pathlib import Path
from typing import Optional

from helpers.atom_utils import make_fixed_size, make_np_example, center_positions
from train.train_config import TrainConfig


class ProteinDataset(torch.utils.data.Dataset):
    """
    Lazy-loading Dataset backed by a JSONL file.

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
        jsonl_path:     str | Path,
        names:          list[str],
        max_seq_length: int = 256,
    ) -> None:
        self.jsonl_path     = Path(jsonl_path)
        self.max_seq_length = max_seq_length
        self._file: Optional[io.BufferedReader] = None

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
            self._file = open(self.jsonl_path, "rb")

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def __del__(self) -> None:
        if self._file is not None:
            self._file.close()

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, idx: int) -> dict:
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
    cfg:         TrainConfig,
    jsonl_path:  str | Path,
    splits_path: str | Path,
    num_workers: int = 0,
    debug_run: bool = True
) -> tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    """
    Build train, validation, and test DataLoaders from a JSONL file and a
    splits JSON.

    Args:
        cfg:         TrainConfig — batch_size and max_seq_length are read from
                     cfg.loader.
        jsonl_path:  Path to the JSONL protein dataset.
        splits_path: Path to a JSON file with keys "train", "validation", and
                     "test", each a list of entry names.
        num_workers: DataLoader worker processes (0 = main process only).

    Returns:
        (train_loader, val_loader, test_loader)
    """
    with open(splits_path) as f:
        splits: dict[str, list[str]] = json.load(f)

    train_set = ProteinDataset(
        jsonl_path, splits["train"],
        max_seq_length=cfg.train_loader.max_seq_length,
    )
    val_set = ProteinDataset(
        jsonl_path, splits["validation"],
        max_seq_length=cfg.test_loader.max_seq_length,
    )
    test_set = ProteinDataset(
        jsonl_path, splits["test"],
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
        )
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=cfg.train_loader.batch_size,
            sampler=subset_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=cfg.train_loader.batch_size,
            sampler=subset_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )

    else:
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=cfg.train_loader.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=cfg.test_loader.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=cfg.test_loader.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader, test_loader
