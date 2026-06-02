"""Tests for dataset and data loading utilities."""

import json
import pathlib
import pickle
from collections.abc import Mapping
from typing import cast

import numpy as np
import pytest
import torch
from helpers.bucketed_sampler import BucketedBatchSampler
from helpers.data import (
    ClusteredProteinDataset,
    ProteinDataset,
    make_bucketed_data_loaders,
    to_protein_batch_dynamic,
)
from helpers.featurize import ProteinBatch
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
    """Return synthetic backbone coordinates for N, CA, C, O atoms."""
    rng = np.random.default_rng()
    return {atom: rng.standard_normal((n, 3)).tolist() for atom in ("N", "CA", "C", "O")}


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


# ---------------------------------------------------------------------------
# Helpers for ClusteredProteinDataset tests
# ---------------------------------------------------------------------------


def _make_entry(name: str, seq_len: int) -> dict[str, object]:
    """Build a minimal JSONL entry with the given name and sequence length."""
    return {
        "name": name,
        "seq": "A" * seq_len,
        "coords": _make_coords(seq_len),
    }


def _write_jsonl(path: pathlib.Path, entries: list[dict[str, object]]) -> None:
    """Write entries as JSONL to path."""
    with open(path, "w") as f:
        f.writelines(json.dumps(entry) + "\n" for entry in entries)


# ---------------------------------------------------------------------------
# ClusteredProteinDataset
# ---------------------------------------------------------------------------


def test_clustered_dataset_len(tmp_path: pathlib.Path) -> None:
    """ClusteredProteinDataset.__len__ returns the number of included proteins."""
    entries = [_make_entry(f"p{i}", 8) for i in range(5)]
    _write_jsonl(tmp_path / "p.jsonl", entries)
    ds = ClusteredProteinDataset(tmp_path / "p.jsonl", [f"p{i}" for i in range(5)])
    assert len(ds) == 5


def test_clustered_dataset_item_keys(tmp_path: pathlib.Path) -> None:
    """__getitem__ returns a dict with atom_positions, atom_mask, residue_index, seq."""
    entries = [_make_entry("p1", 10)]
    _write_jsonl(tmp_path / "p.jsonl", entries)
    ds = ClusteredProteinDataset(tmp_path / "p.jsonl", ["p1"])
    item = ds[0]
    assert set(item.keys()) == {"atom_positions", "atom_mask", "residue_index", "seq"}


def test_clustered_dataset_variable_lengths(tmp_path: pathlib.Path) -> None:
    """Items have their actual length, not a fixed padded length."""
    entries = [_make_entry("p1", 8), _make_entry("p2", 16), _make_entry("p3", 32)]
    _write_jsonl(tmp_path / "p.jsonl", entries)
    ds = ClusteredProteinDataset(tmp_path / "p.jsonl", ["p1", "p2", "p3"])
    lengths = {cast(torch.Tensor, ds[i]["atom_positions"]).shape[0] for i in range(len(ds))}
    assert lengths == {8, 16, 32}


def test_clustered_dataset_truncates_to_max_seq_length(tmp_path: pathlib.Path) -> None:
    """Proteins longer than max_seq_length are truncated to max_seq_length."""
    entries = [_make_entry("p1", 600)]
    _write_jsonl(tmp_path / "p.jsonl", entries)
    ds = ClusteredProteinDataset(tmp_path / "p.jsonl", ["p1"], max_seq_length=128, token_budget=512)
    assert cast(torch.Tensor, ds[0]["atom_positions"]).shape[0] == 128


def test_clustered_dataset_pickles(tmp_path: pathlib.Path) -> None:
    """ClusteredProteinDataset survives pickle round-trip (for DataLoader workers)."""
    entries = [_make_entry("p1", 8)]
    _write_jsonl(tmp_path / "p.jsonl", entries)
    ds = ClusteredProteinDataset(tmp_path / "p.jsonl", ["p1"])
    restored = pickle.loads(pickle.dumps(ds))
    item = restored[0]
    assert item["atom_positions"].shape[0] == 8


# ---------------------------------------------------------------------------
# to_protein_batch_dynamic
# ---------------------------------------------------------------------------


def test_to_protein_batch_dynamic_pads_to_max(tmp_path: pathlib.Path) -> None:
    """to_protein_batch_dynamic pads shorter items to the longest item in the batch."""
    entries = [_make_entry("p1", 8), _make_entry("p2", 16)]
    _write_jsonl(tmp_path / "p.jsonl", entries)
    ds = ClusteredProteinDataset(tmp_path / "p.jsonl", ["p1", "p2"])
    batch = to_protein_batch_dynamic([ds[0], ds[1]])
    assert batch.atom_positions.shape == (2, 16, 37, 3)
    assert batch.atom_mask.shape == (2, 16, 37)
    assert batch.residue_index.shape == (2, 16)


def test_to_protein_batch_dynamic_uniform_lengths(tmp_path: pathlib.Path) -> None:
    """to_protein_batch_dynamic is a no-op when all items have the same length."""
    entries = [_make_entry("p1", 10), _make_entry("p2", 10)]
    _write_jsonl(tmp_path / "p.jsonl", entries)
    ds = ClusteredProteinDataset(tmp_path / "p.jsonl", ["p1", "p2"])
    batch = to_protein_batch_dynamic([ds[0], ds[1]])
    assert batch.atom_positions.shape == (2, 10, 37, 3)


# ---------------------------------------------------------------------------
# Fixtures for bucketed loader tests
# ---------------------------------------------------------------------------


@pytest.fixture
def bucketed_jsonl(tmp_path: pathlib.Path) -> str:
    """Write a JSONL with 5 proteins of varying lengths."""
    entries = [
        _make_entry("p1", 8),
        _make_entry("p2", 16),
        _make_entry("p3", 24),
        _make_entry("p4", 32),
        _make_entry("p5", 40),
    ]
    path = tmp_path / "proteins.jsonl"
    _write_jsonl(path, entries)
    return str(path)


@pytest.fixture
def bucketed_splits(tmp_path: pathlib.Path) -> str:
    """Write a splits JSON using the 5-protein JSONL names."""
    path = tmp_path / "splits.json"
    with open(path, "w") as f:
        json.dump(
            {
                "train": ["p1", "p2", "p3"],
                "validation": ["p4"],
                "test": ["p5"],
            },
            f,
        )
    return str(path)


@pytest.fixture
def bucketed_cfg() -> TrainConfig:
    """Provide a minimal TrainConfig with token_budget=512 for bucketed loader tests."""
    return TrainConfig(
        train_loader=TrainLoaderConfig(batch_size=2, max_seq_length=_MAX_SEQ, token_budget=512),
        test_loader=EvalLoaderConfig(batch_size=2, max_seq_length=_MAX_SEQ, token_budget=512),
    )


# ---------------------------------------------------------------------------
# make_bucketed_data_loaders tests
# ---------------------------------------------------------------------------


def test_bucketed_train_loader_uses_batch_sampler(
    bucketed_cfg: TrainConfig,
    bucketed_jsonl: str,
    bucketed_splits: str,
) -> None:
    """Training loader uses BucketedBatchSampler as its batch_sampler."""
    _, _, _, train_sampler = make_bucketed_data_loaders(
        cfg=bucketed_cfg,
        jsonl_path=bucketed_jsonl,
        splits_path=bucketed_splits,
        num_workers=0,
        debug_run=False,
    )
    assert isinstance(train_sampler, BucketedBatchSampler)


def test_bucketed_train_loader_yields_protein_batch(
    bucketed_cfg: TrainConfig,
    bucketed_jsonl: str,
    bucketed_splits: str,
) -> None:
    """Training loader yields ProteinBatch objects."""
    train_loader, _, _, train_sampler = make_bucketed_data_loaders(
        cfg=bucketed_cfg,
        jsonl_path=bucketed_jsonl,
        splits_path=bucketed_splits,
        num_workers=0,
        debug_run=False,
    )
    train_sampler.set_epoch(0)
    batch = next(iter(train_loader))
    assert isinstance(batch, ProteinBatch)


def test_bucketed_debug_run_train_dataset_has_252_items(
    bucketed_cfg: TrainConfig,
    debug_jsonl_path: str,
    debug_splits_path: str,
) -> None:
    """make_bucketed_data_loaders with debug_run=True yields a train dataset of _N_DEBUG items."""
    train_loader, _, _, _ = make_bucketed_data_loaders(
        cfg=bucketed_cfg,
        jsonl_path=debug_jsonl_path,
        splits_path=debug_splits_path,
        num_workers=0,
        debug_run=True,
    )
    assert len(cast(ClusteredProteinDataset, train_loader.dataset)) == _N_DEBUG


_N_FULL = 300  # dataset larger than _N_DEBUG to expose cache-poisoning bug


def test_bucketed_debug_run_not_poisoned_by_prior_full_cache(
    bucketed_cfg: TrainConfig,
    tmp_path: pathlib.Path,
) -> None:
    """debug_run=True must yield _N_DEBUG items even when cache was built by a prior full run.

    Root cause: ClusterIndex._load_from_cache() reads all cluster-file lines without
    re-applying the names filter, so a cache built from all _N_FULL proteins silently
    returns _N_FULL items for a subsequent debug_run=True call capped at _N_DEBUG.
    """
    names = [f"{i:04d}p.A" for i in range(_N_FULL)]
    jsonl_path = tmp_path / "full_proteins.jsonl"
    with open(jsonl_path, "w") as f:
        for name in names:
            entry = {
                "name": name,
                "seq": "ACDEFG"[:_N_RES_DATA],
                "coords": _make_coords(_N_RES_DATA),
            }
            f.write(json.dumps(entry) + "\n")
    splits_path = tmp_path / "full_splits.json"
    with open(splits_path, "w") as f:
        json.dump({"train": names, "validation": names[:1], "test": names[:1]}, f)

    # Seed the ClusterIndex cache with the full _N_FULL-protein dataset.
    make_bucketed_data_loaders(
        cfg=bucketed_cfg,
        jsonl_path=str(jsonl_path),
        splits_path=str(splits_path),
        num_workers=0,
        debug_run=False,
    )

    # A subsequent debug_run=True call must cap at _N_DEBUG despite the warm cache.
    train_loader, _, _, _ = make_bucketed_data_loaders(
        cfg=bucketed_cfg,
        jsonl_path=str(jsonl_path),
        splits_path=str(splits_path),
        num_workers=0,
        debug_run=True,
    )
    assert len(cast(ClusteredProteinDataset, train_loader.dataset)) == _N_DEBUG
