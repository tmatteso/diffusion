import json
import math
import pickle

import numpy as np
import pytest
import torch

from helpers.data import ProteinDataset, make_data_loaders, make_ddp_data_loaders
from torch.utils.data.distributed import DistributedSampler
from train.train_config import TrainConfig, TrainLoaderConfig
from train.train_config import TestLoaderConfig as EvalLoaderConfig

_N_RES_DATA  = 6     # residues per synthetic entry
_MAX_SEQ     = 8     # padded / truncated to this length
_ENTRY_NAMES = ["1aa.A", "2bb.A", "3cc.A", "4dd.A", "5ee.A"]
_TRAIN_NAMES = ["1aa.A", "2bb.A", "3cc.A"]
_VAL_NAMES   = ["4dd.A"]
_TEST_NAMES  = ["5ee.A"]
_N_DEBUG     = 252   # debug_run sampler uses SubsetRandomSampler(range(252))


def _make_coords(n: int) -> dict:
    return {atom: np.random.randn(n, 3).tolist() for atom in ("N", "CA", "C", "O")}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def jsonl_path(tmp_path) -> str:
    path = tmp_path / "proteins.jsonl"
    with open(path, "w") as f:
        for name in _ENTRY_NAMES:
            entry = {"name": name, "seq": "ACDEFG"[:_N_RES_DATA], "coords": _make_coords(_N_RES_DATA)}
            f.write(json.dumps(entry) + "\n")
    return str(path)


@pytest.fixture
def splits_path(tmp_path) -> str:
    path = tmp_path / "splits.json"
    with open(path, "w") as f:
        json.dump({"train": _TRAIN_NAMES, "validation": _VAL_NAMES, "test": _TEST_NAMES}, f)
    return str(path)


@pytest.fixture
def train_dataset(jsonl_path) -> ProteinDataset:
    return ProteinDataset(jsonl_path, _TRAIN_NAMES, max_seq_length=_MAX_SEQ)


@pytest.fixture
def cfg() -> TrainConfig:
    return TrainConfig(
        train_loader=TrainLoaderConfig(batch_size=2, max_seq_length=_MAX_SEQ),
        test_loader=EvalLoaderConfig(batch_size=2, max_seq_length=_MAX_SEQ),
    )


# ---------------------------------------------------------------------------
# ProteinDataset — length and indexing
# ---------------------------------------------------------------------------

def test_protein_dataset_len(train_dataset: ProteinDataset):
    assert len(train_dataset) == len(_TRAIN_NAMES)


def test_protein_dataset_excludes_unlisted_names(jsonl_path):
    ds = ProteinDataset(jsonl_path, ["1aa.A"], max_seq_length=_MAX_SEQ)
    assert len(ds) == 1


def test_protein_dataset_empty_names(jsonl_path):
    ds = ProteinDataset(jsonl_path, [], max_seq_length=_MAX_SEQ)
    assert len(ds) == 0


# ---------------------------------------------------------------------------
# ProteinDataset — sample structure
# ---------------------------------------------------------------------------

def test_protein_dataset_sample_keys(train_dataset: ProteinDataset):
    sample = train_dataset[0]
    assert "atom_positions" in sample
    assert "atom_mask" in sample
    assert "residue_index" in sample
    assert "seq" in sample


def test_protein_dataset_atom_positions_shape(train_dataset: ProteinDataset):
    assert train_dataset[0]["atom_positions"].shape == (_MAX_SEQ, 37, 3)


def test_protein_dataset_atom_mask_shape(train_dataset: ProteinDataset):
    assert train_dataset[0]["atom_mask"].shape == (_MAX_SEQ, 37)


def test_protein_dataset_residue_index_shape(train_dataset: ProteinDataset):
    assert train_dataset[0]["residue_index"].shape == (_MAX_SEQ,)


def test_protein_dataset_tensor_fields_are_float32(train_dataset: ProteinDataset):
    sample = train_dataset[0]
    for key in ("atom_positions", "atom_mask", "residue_index"):
        assert sample[key].dtype == torch.float32, f"{key} is not float32"


def test_protein_dataset_seq_is_string(train_dataset: ProteinDataset):
    assert isinstance(train_dataset[0]["seq"], str)


def test_protein_dataset_seq_truncated_to_max_seq_length(tmp_path):
    path = str(tmp_path / "long.jsonl")
    with open(path, "w") as f:
        entry = {"name": "long.A", "seq": "A" * 100, "coords": _make_coords(100)}
        f.write(json.dumps(entry) + "\n")
    ds = ProteinDataset(path, ["long.A"], max_seq_length=10)
    assert len(ds[0]["seq"]) == 10


def test_protein_dataset_all_items_accessible(train_dataset: ProteinDataset):
    for i in range(len(train_dataset)):
        assert "atom_positions" in train_dataset[i]


# ---------------------------------------------------------------------------
# ProteinDataset — pickle / multiprocessing compatibility
# ---------------------------------------------------------------------------

def test_protein_dataset_picklable_before_open(train_dataset: ProteinDataset):
    ds2 = pickle.loads(pickle.dumps(train_dataset))
    assert len(ds2) == len(train_dataset)


def test_protein_dataset_picklable_after_open(train_dataset: ProteinDataset):
    _ = train_dataset[0]   # opens the file handle
    ds2 = pickle.loads(pickle.dumps(train_dataset))
    assert ds2[0]["atom_positions"].shape == (_MAX_SEQ, 37, 3)


# ---------------------------------------------------------------------------
# make_data_loaders — loader counts and batch shapes
# ---------------------------------------------------------------------------

def test_make_data_loaders_returns_three_loaders(cfg, jsonl_path, splits_path):
    assert len(make_data_loaders(cfg, jsonl_path, splits_path)) == 3


def test_make_data_loaders_train_len(cfg, jsonl_path, splits_path):
    train_loader, _, _ = make_data_loaders(cfg, jsonl_path, splits_path, debug_run=False)
    expected = math.ceil(len(_TRAIN_NAMES) / cfg.train_loader.batch_size)
    assert len(train_loader) == expected


def test_make_data_loaders_batch_atom_positions_shape(cfg, jsonl_path, splits_path):
    train_loader, _, _ = make_data_loaders(cfg, jsonl_path, splits_path, debug_run=False)
    batch = next(iter(train_loader))
    assert batch["atom_positions"].shape == (cfg.train_loader.batch_size, _MAX_SEQ, 37, 3)


def test_make_data_loaders_batch_seq_is_list_of_strings(cfg, jsonl_path, splits_path):
    train_loader, _, _ = make_data_loaders(cfg, jsonl_path, splits_path, debug_run=False)
    batch = next(iter(train_loader))
    assert isinstance(batch["seq"], list)
    assert all(isinstance(s, str) for s in batch["seq"])


# ---------------------------------------------------------------------------
# make_data_loaders — debug_run=True
# ---------------------------------------------------------------------------

# The debug sampler is SubsetRandomSampler(range(8)), so iterating requires
# a dataset with ≥256 entries; len() only reads sampler size and works cheaply.

@pytest.fixture
def debug_jsonl_path(tmp_path) -> str:
    path = tmp_path / "debug_proteins.jsonl"
    names = [f"{i:04d}x.A" for i in range(_N_DEBUG)]
    with open(path, "w") as f:
        for name in names:
            entry = {"name": name, "seq": "ACDEFG"[:_N_RES_DATA], "coords": _make_coords(_N_RES_DATA)}
            f.write(json.dumps(entry) + "\n")
    return str(path)


@pytest.fixture
def debug_splits_path(tmp_path) -> str:
    names = [f"{i:04d}x.A" for i in range(_N_DEBUG)]
    path = tmp_path / "debug_splits.json"
    with open(path, "w") as f:
        json.dump({"train": names, "validation": names[:1], "test": names[:1]}, f)
    return str(path)


def test_make_data_loaders_debug_train_len(cfg, debug_jsonl_path, debug_splits_path):
    train_loader, _, _ = make_data_loaders(cfg, debug_jsonl_path, debug_splits_path, debug_run=True)
    expected = math.ceil(_N_DEBUG / cfg.train_loader.batch_size)
    assert len(train_loader) == expected


def test_make_data_loaders_debug_batch_atom_positions_shape(cfg, debug_jsonl_path, debug_splits_path):
    train_loader, _, _ = make_data_loaders(cfg, debug_jsonl_path, debug_splits_path, debug_run=True)
    batch = next(iter(train_loader))
    assert batch["atom_positions"].shape == (cfg.train_loader.batch_size, _MAX_SEQ, 37, 3)


def test_make_data_loaders_debug_batch_seq_is_list_of_strings(cfg, debug_jsonl_path, debug_splits_path):
    train_loader, _, _ = make_data_loaders(cfg, debug_jsonl_path, debug_splits_path, debug_run=True)
    batch = next(iter(train_loader))
    assert isinstance(batch["seq"], list)
    assert all(isinstance(s, str) for s in batch["seq"])


# ---------------------------------------------------------------------------
# make_ddp_data_loaders
# ---------------------------------------------------------------------------

def test_make_ddp_data_loaders_returns_three_loaders(jsonl_path, splits_path):
    cfg = TrainConfig()
    train_loader, val_loader, test_loader = make_ddp_data_loaders(cfg, jsonl_path, splits_path, rank=0, world_size=1)
    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None


def test_make_ddp_data_loaders_train_sampler_is_distributed(jsonl_path, splits_path):
    cfg = TrainConfig()
    train_loader, _, _ = make_ddp_data_loaders(cfg, jsonl_path, splits_path, rank=0, world_size=1)
    assert isinstance(train_loader.sampler, DistributedSampler)


def test_make_ddp_data_loaders_val_sampler_is_distributed(jsonl_path, splits_path):
    cfg = TrainConfig()
    _, val_loader, _ = make_ddp_data_loaders(cfg, jsonl_path, splits_path, rank=0, world_size=1)
    assert isinstance(val_loader.sampler, DistributedSampler)


def test_make_ddp_data_loaders_test_sampler_is_distributed(jsonl_path, splits_path):
    cfg = TrainConfig()
    _, _, test_loader = make_ddp_data_loaders(cfg, jsonl_path, splits_path, rank=0, world_size=1)
    assert isinstance(test_loader.sampler, DistributedSampler)


def test_make_ddp_data_loaders_train_sampler_shuffle_true(jsonl_path, splits_path):
    cfg = TrainConfig()
    train_loader, _, _ = make_ddp_data_loaders(cfg, jsonl_path, splits_path, rank=0, world_size=1)
    assert train_loader.sampler.shuffle is True


def test_make_ddp_data_loaders_val_sampler_shuffle_false(jsonl_path, splits_path):
    cfg = TrainConfig()
    _, val_loader, _ = make_ddp_data_loaders(cfg, jsonl_path, splits_path, rank=0, world_size=1)
    assert val_loader.sampler.shuffle is False


def test_make_ddp_data_loaders_train_sampler_rank(jsonl_path, splits_path):
    cfg = TrainConfig()
    train_loader, _, _ = make_ddp_data_loaders(cfg, jsonl_path, splits_path, rank=0, world_size=2)
    assert train_loader.sampler.rank == 0


def test_make_ddp_data_loaders_train_sampler_world_size(jsonl_path, splits_path):
    cfg = TrainConfig()
    train_loader, _, _ = make_ddp_data_loaders(cfg, jsonl_path, splits_path, rank=0, world_size=2)
    assert train_loader.sampler.num_replicas == 2
