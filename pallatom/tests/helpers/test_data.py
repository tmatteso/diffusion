"""Tests for dataset and data loading utilities."""

import json
import math
import os
import pathlib
import pickle
from collections.abc import Mapping

import numpy as np
import pytest
import torch
from helpers.data import ProteinDataset, _FileLogProcessor, make_data_loaders, make_ddp_data_loaders
from torch.utils.data.distributed import DistributedSampler
from train.train_config import TestLoaderConfig as EvalLoaderConfig
from train.train_config import TrainConfig, TrainLoaderConfig

_N_RES_DATA = 6  # residues per synthetic entry
_MAX_SEQ = 8  # padded / truncated to this length
_ENTRY_NAMES = ["1aa.A", "2bb.A", "3cc.A", "4dd.A", "5ee.A"]
_TRAIN_NAMES = ["1aa.A", "2bb.A", "3cc.A"]
_VAL_NAMES = ["4dd.A"]
_TEST_NAMES = ["5ee.A"]
_N_DEBUG = 252  # debug_run sampler uses SubsetRandomSampler(range(252))


def _make_coords(n: int) -> Mapping[str, list[list[float]]]:
    return {atom: np.random.randn(n, 3).tolist() for atom in ("N", "CA", "C", "O")}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def jsonl_path(tmp_path: pathlib.Path) -> str:
    """Write a temporary JSONL file with synthetic protein entries and return its path."""
    path = tmp_path / "proteins.jsonl"
    with open(path, "w") as f:
        for name in _ENTRY_NAMES:
            entry = {
                "name": name,
                "seq": "ACDEFG"[:_N_RES_DATA],
                "coords": _make_coords(_N_RES_DATA),
            }
            f.write(json.dumps(entry) + "\n")
    return str(path)


@pytest.fixture
def splits_path(tmp_path: pathlib.Path) -> str:
    """Write a temporary splits JSON with train/val/test name lists and return its path."""
    path = tmp_path / "splits.json"
    with open(path, "w") as f:
        json.dump({"train": _TRAIN_NAMES, "validation": _VAL_NAMES, "test": _TEST_NAMES}, f)
    return str(path)


@pytest.fixture
def train_dataset(jsonl_path: str) -> ProteinDataset:
    """Provide a ProteinDataset loaded from the synthetic JSONL with training names."""
    return ProteinDataset(jsonl_path, _TRAIN_NAMES, max_seq_length=_MAX_SEQ)


@pytest.fixture
def cfg() -> TrainConfig:
    """Provide a minimal TrainConfig with batch_size=2 and max_seq_length=_MAX_SEQ."""
    return TrainConfig(
        train_loader=TrainLoaderConfig(batch_size=2, max_seq_length=_MAX_SEQ),
        test_loader=EvalLoaderConfig(batch_size=2, max_seq_length=_MAX_SEQ),
    )


# ---------------------------------------------------------------------------
# ProteinDataset — length and indexing
# ---------------------------------------------------------------------------


def test_protein_dataset_len(train_dataset: ProteinDataset):
    """ProteinDataset length equals the number of listed training names."""
    assert len(train_dataset) == len(_TRAIN_NAMES)


def test_protein_dataset_excludes_unlisted_names(jsonl_path: str):
    """ProteinDataset excludes entries whose names are not in the provided list."""
    ds = ProteinDataset(jsonl_path, ["1aa.A"], max_seq_length=_MAX_SEQ)
    assert len(ds) == 1


def test_protein_dataset_empty_names(jsonl_path: str):
    """ProteinDataset with an empty names list has length 0."""
    ds = ProteinDataset(jsonl_path, [], max_seq_length=_MAX_SEQ)
    assert len(ds) == 0


# ---------------------------------------------------------------------------
# ProteinDataset — sample structure
# ---------------------------------------------------------------------------


def test_protein_dataset_sample_keys(train_dataset: ProteinDataset):
    """Each ProteinDataset sample contains atom_positions, atom_mask, residue_index, and seq."""
    sample = train_dataset[0]
    assert "atom_positions" in sample
    assert "atom_mask" in sample
    assert "residue_index" in sample
    assert "seq" in sample


def test_protein_dataset_atom_positions_shape(train_dataset: ProteinDataset):
    """ProteinDataset pads/truncates atom_positions to (_MAX_SEQ, 37, 3)."""
    val = train_dataset[0]["atom_positions"]
    assert isinstance(val, torch.Tensor)
    assert val.shape == (_MAX_SEQ, 37, 3)


def test_protein_dataset_atom_mask_shape(train_dataset: ProteinDataset):
    """ProteinDataset pads/truncates atom_mask to (_MAX_SEQ, 37)."""
    val = train_dataset[0]["atom_mask"]
    assert isinstance(val, torch.Tensor)
    assert val.shape == (_MAX_SEQ, 37)


def test_protein_dataset_residue_index_shape(train_dataset: ProteinDataset):
    """ProteinDataset pads/truncates residue_index to a 1-D tensor of length _MAX_SEQ."""
    val = train_dataset[0]["residue_index"]
    assert isinstance(val, torch.Tensor)
    assert val.shape == (_MAX_SEQ,)


def test_protein_dataset_tensor_fields_are_float32(train_dataset: ProteinDataset):
    """All tensor fields in a ProteinDataset sample have dtype float32."""
    sample = train_dataset[0]
    for key in ("atom_positions", "atom_mask", "residue_index"):
        val = sample[key]
        assert isinstance(val, torch.Tensor), f"{key} is not a tensor"
        assert val.dtype == torch.float32, f"{key} is not float32"


def test_protein_dataset_seq_is_string(train_dataset: ProteinDataset):
    """The seq field in a ProteinDataset sample is a plain Python str."""
    assert isinstance(train_dataset[0]["seq"], str)


def test_protein_dataset_seq_truncated_to_max_seq_length(tmp_path: pathlib.Path):
    """ProteinDataset truncates the sequence string to max_seq_length characters."""
    path = str(tmp_path / "long.jsonl")
    with open(path, "w") as f:
        entry = {"name": "long.A", "seq": "A" * 100, "coords": _make_coords(100)}
        f.write(json.dumps(entry) + "\n")
    ds = ProteinDataset(path, ["long.A"], max_seq_length=10)
    assert len(ds[0]["seq"]) == 10


def test_protein_dataset_all_items_accessible(train_dataset: ProteinDataset):
    """Every index in ProteinDataset is accessible and returns an atom_positions key."""
    for i in range(len(train_dataset)):
        assert "atom_positions" in train_dataset[i]


# ---------------------------------------------------------------------------
# ProteinDataset — pickle / multiprocessing compatibility
# ---------------------------------------------------------------------------


def test_protein_dataset_picklable_before_open(train_dataset: ProteinDataset):
    """ProteinDataset can be pickled and restored before the JSONL file handle is opened."""
    ds2 = pickle.loads(pickle.dumps(train_dataset))
    assert len(ds2) == len(train_dataset)


def test_protein_dataset_picklable_after_open(train_dataset: ProteinDataset):
    """ProteinDataset can be pickled and restored after accessing a sample (file handle open)."""
    _ = train_dataset[0]  # opens the file handle
    ds2 = pickle.loads(pickle.dumps(train_dataset))
    assert ds2[0]["atom_positions"].shape == (_MAX_SEQ, 37, 3)


# ---------------------------------------------------------------------------
# make_data_loaders — loader counts and batch shapes
# ---------------------------------------------------------------------------


def test_make_data_loaders_returns_three_loaders(
    cfg: TrainConfig, jsonl_path: str, splits_path: str
):
    """make_data_loaders returns a 3-tuple of (train, val, test) DataLoaders."""
    assert (
        len(
            make_data_loaders(
                cfg=cfg,
                jsonl_path=jsonl_path,
                splits_path=splits_path,
                num_workers=0,
                debug_run=True,
            )
        )
        == 3
    )


def test_make_data_loaders_train_len(cfg: TrainConfig, jsonl_path: str, splits_path: str):
    """make_data_loaders train loader length matches ceil(n_train / batch_size)."""
    train_loader, _, _ = make_data_loaders(
        cfg=cfg, jsonl_path=jsonl_path, splits_path=splits_path, num_workers=0, debug_run=False
    )
    expected = math.ceil(len(_TRAIN_NAMES) / cfg.train_loader.batch_size)
    assert len(train_loader) == expected


def test_make_data_loaders_batch_atom_positions_shape(
    cfg: TrainConfig, jsonl_path: str, splits_path: str
):
    """make_data_loaders produces batch with atom_positions shape (batch_size, _MAX_SEQ, 37, 3)."""
    train_loader, _, _ = make_data_loaders(
        cfg=cfg, jsonl_path=jsonl_path, splits_path=splits_path, num_workers=0, debug_run=False
    )
    batch = next(iter(train_loader))
    assert batch.atom_positions.shape == (cfg.train_loader.batch_size, _MAX_SEQ, 37, 3)


def test_make_data_loaders_batch_seq_is_list_of_strings(
    cfg: TrainConfig, jsonl_path: str, splits_path: str
):
    """make_data_loaders collates the seq field into a list of str, one per batch item."""
    train_loader, _, _ = make_data_loaders(
        cfg=cfg, jsonl_path=jsonl_path, splits_path=splits_path, num_workers=0, debug_run=False
    )
    batch = next(iter(train_loader))
    assert isinstance(batch.seq, list)
    assert all(isinstance(s, str) for s in batch.seq)


# ---------------------------------------------------------------------------
# make_data_loaders — debug_run=True
# ---------------------------------------------------------------------------

# The debug sampler is SubsetRandomSampler(range(8)), so iterating requires
# a dataset with ≥256 entries; len() only reads sampler size and works cheaply.


@pytest.fixture
def debug_jsonl_path(tmp_path: pathlib.Path) -> str:
    """Write a JSONL file with _N_DEBUG synthetic protein entries for debug-mode testing."""
    path = tmp_path / "debug_proteins.jsonl"
    names = [f"{i:04d}x.A" for i in range(_N_DEBUG)]
    with open(path, "w") as f:
        for name in names:
            entry = {
                "name": name,
                "seq": "ACDEFG"[:_N_RES_DATA],
                "coords": _make_coords(_N_RES_DATA),
            }
            f.write(json.dumps(entry) + "\n")
    return str(path)


@pytest.fixture
def debug_splits_path(tmp_path: pathlib.Path) -> str:
    """Write a splits JSON assigning all _N_DEBUG entries to train for debug-mode testing."""
    names = [f"{i:04d}x.A" for i in range(_N_DEBUG)]
    path = tmp_path / "debug_splits.json"
    with open(path, "w") as f:
        json.dump({"train": names, "validation": names[:1], "test": names[:1]}, f)
    return str(path)


def test_make_data_loaders_debug_train_len(
    cfg: TrainConfig, debug_jsonl_path: str, debug_splits_path: str
):
    """Debug-mode train loader length equals ceil(_N_DEBUG / batch_size)."""
    train_loader, _, _ = make_data_loaders(
        cfg=cfg,
        jsonl_path=debug_jsonl_path,
        splits_path=debug_splits_path,
        num_workers=0,
        debug_run=True,
    )
    expected = math.ceil(_N_DEBUG / cfg.train_loader.batch_size)
    assert len(train_loader) == expected


def test_make_data_loaders_debug_batch_atom_positions_shape(
    cfg: TrainConfig, debug_jsonl_path: str, debug_splits_path: str
):
    """Debug-mode batches have atom_positions shape (batch_size, _MAX_SEQ, 37, 3)."""
    train_loader, _, _ = make_data_loaders(
        cfg=cfg,
        jsonl_path=debug_jsonl_path,
        splits_path=debug_splits_path,
        num_workers=0,
        debug_run=True,
    )
    batch = next(iter(train_loader))
    assert batch.atom_positions.shape == (cfg.train_loader.batch_size, _MAX_SEQ, 37, 3)


def test_make_data_loaders_debug_batch_seq_is_list_of_strings(
    cfg: TrainConfig, debug_jsonl_path: str, debug_splits_path: str
):
    """Debug-mode batches collate seq into a list of str, one per batch item."""
    train_loader, _, _ = make_data_loaders(
        cfg=cfg,
        jsonl_path=debug_jsonl_path,
        splits_path=debug_splits_path,
        num_workers=0,
        debug_run=True,
    )
    batch = next(iter(train_loader))
    assert isinstance(batch.seq, list)
    assert all(isinstance(s, str) for s in batch.seq)


# ---------------------------------------------------------------------------
# make_ddp_data_loaders
# ---------------------------------------------------------------------------


def test_make_ddp_data_loaders_returns_three_loaders(
    cfg: TrainConfig, jsonl_path: str, splits_path: str
):
    """make_ddp_data_loaders returns non-None train, val, and test DataLoaders."""
    train_loader, val_loader, test_loader = make_ddp_data_loaders(
        cfg, jsonl_path, splits_path, rank=0, world_size=1
    )
    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None


def test_make_ddp_data_loaders_train_sampler_is_distributed(
    cfg: TrainConfig, jsonl_path: str, splits_path: str
):
    """make_ddp_data_loaders wraps the train dataset in a DistributedSampler."""
    train_loader, _, _ = make_ddp_data_loaders(cfg, jsonl_path, splits_path, rank=0, world_size=1)
    assert isinstance(train_loader.sampler, DistributedSampler)


def test_make_ddp_data_loaders_val_sampler_is_distributed(
    cfg: TrainConfig, jsonl_path: str, splits_path: str
):
    """make_ddp_data_loaders wraps the validation dataset in a DistributedSampler."""
    _, val_loader, _ = make_ddp_data_loaders(cfg, jsonl_path, splits_path, rank=0, world_size=1)
    assert isinstance(val_loader.sampler, DistributedSampler)


def test_make_ddp_data_loaders_test_sampler_is_distributed(
    cfg: TrainConfig, jsonl_path: str, splits_path: str
):
    """make_ddp_data_loaders wraps the test dataset in a DistributedSampler."""
    _, _, test_loader = make_ddp_data_loaders(cfg, jsonl_path, splits_path, rank=0, world_size=1)
    assert isinstance(test_loader.sampler, DistributedSampler)


def test_make_ddp_data_loaders_train_sampler_shuffle_true(
    cfg: TrainConfig, jsonl_path: str, splits_path: str
):
    """make_ddp_data_loaders sets shuffle=True on the train DistributedSampler."""
    train_loader, _, _ = make_ddp_data_loaders(cfg, jsonl_path, splits_path, rank=0, world_size=1)
    assert isinstance(train_loader.sampler, DistributedSampler)
    assert train_loader.sampler.shuffle is True


def test_make_ddp_data_loaders_val_sampler_shuffle_false(
    cfg: TrainConfig, jsonl_path: str, splits_path: str
):
    """make_ddp_data_loaders sets shuffle=False on the validation DistributedSampler."""
    _, val_loader, _ = make_ddp_data_loaders(cfg, jsonl_path, splits_path, rank=0, world_size=1)
    assert isinstance(val_loader.sampler, DistributedSampler)
    assert val_loader.sampler.shuffle is False


def test_make_ddp_data_loaders_test_sampler_shuffle_false(
    cfg: TrainConfig, jsonl_path: str, splits_path: str
):
    """make_ddp_data_loaders sets shuffle=False on the test DistributedSampler."""
    _, _, test_loader = make_ddp_data_loaders(cfg, jsonl_path, splits_path, rank=0, world_size=1)
    assert isinstance(test_loader.sampler, DistributedSampler)
    assert test_loader.sampler.shuffle is False


def test_make_ddp_data_loaders_train_sampler_rank(
    cfg: TrainConfig, jsonl_path: str, splits_path: str
):
    """make_ddp_data_loaders propagates rank=0 to the train DistributedSampler."""
    train_loader, _, _ = make_ddp_data_loaders(cfg, jsonl_path, splits_path, rank=0, world_size=2)
    assert isinstance(train_loader.sampler, DistributedSampler)
    assert train_loader.sampler.rank == 0


def test_make_ddp_data_loaders_train_sampler_world_size(
    cfg: TrainConfig, jsonl_path: str, splits_path: str
):
    """make_ddp_data_loaders propagates world_size=2 to the train DistributedSampler."""
    train_loader, _, _ = make_ddp_data_loaders(cfg, jsonl_path, splits_path, rank=0, world_size=2)
    assert isinstance(train_loader.sampler, DistributedSampler)
    assert train_loader.sampler.num_replicas == 2


# ---------------------------------------------------------------------------
# _FileLogProcessor
# ---------------------------------------------------------------------------


def test_file_log_processor_creates_file(tmp_path: pathlib.Path):
    """_FileLogProcessor creates the log file on disk when constructed."""
    path = str(tmp_path / "run.jsonl")
    _FileLogProcessor(path)
    assert os.path.exists(path)


def test_file_log_processor_returns_event_dict(tmp_path: pathlib.Path):
    """_FileLogProcessor.__call__ returns the event dict unchanged (pass-through)."""
    path = str(tmp_path / "run.jsonl")
    proc = _FileLogProcessor(path)
    event = {"event": "test", "value": 1}
    returned = proc(None, None, event)
    assert returned is event


def test_file_log_processor_writes_json_line(tmp_path: pathlib.Path):
    """_FileLogProcessor serialises each event dict as a JSON line in the log file."""
    path = str(tmp_path / "run.jsonl")
    proc = _FileLogProcessor(path)
    event = {"event": "train", "loss": 0.5}
    proc(None, None, event)
    with open(path) as fh:
        written = json.loads(fh.readline())
    assert written == event


def test_file_log_processor_writes_multiple_lines(tmp_path: pathlib.Path):
    """_FileLogProcessor writes one JSON line per event when called multiple times."""
    path = str(tmp_path / "run.jsonl")
    proc = _FileLogProcessor(path)
    events = [{"event": "a", "x": 1}, {"event": "b", "x": 2}]
    for ev in events:
        proc(None, None, ev)
    with open(path) as fh:
        lines = fh.readlines()
    assert len(lines) == 2
    assert [json.loads(line) for line in lines] == events


def test_file_log_processor_truncates_existing_file(tmp_path: pathlib.Path):
    """_FileLogProcessor truncates a pre-existing log file rather than appending."""
    path = str(tmp_path / "run.jsonl")
    with open(path, "w") as fh:
        fh.write('{"stale": true}\n' * 5)
    proc = _FileLogProcessor(path)
    proc(None, None, {"event": "fresh"})
    with open(path) as fh:
        lines = fh.readlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"event": "fresh"}


def test_file_log_processor_context_manager_closes_file(tmp_path: pathlib.Path):
    """_FileLogProcessor.__exit__ closes the underlying file handle."""
    path = str(tmp_path / "run.jsonl")
    proc = _FileLogProcessor(path)
    with proc:
        proc(None, None, {"event": "inside"})
    assert proc._f.closed
