"""Tests for dataset and data loading utilities.

Covers ProteinDataset,
and make_bucketed_data_loaders, including length/indexing, sample structure,
pickle compatibility, and bucketed loader behaviour.
"""

import json
import pathlib
import pickle
from collections.abc import Mapping
from typing import cast

import numpy as np
import pytest
from helpers.atom_utils import Protein
from helpers.data import (
    ProteinDataset,
    ProteinShardDataset,
    ShardBudgetParameters,
    ShardDataLoader,
    make_bucketed_data_loaders,
)
from helpers.useful_objects import TrainArgs
from train.train_config import LoaderConfig as EvalLoaderConfig
from train.train_config import TrainConfig, TrainLoaderConfig

_N_RES_DATA = 6  # residues per synthetic entry
_MAX_SEQ = 8  # padded / truncated to this length
_ENTRY_NAMES = ["1aa.A", "2bb.A", "3cc.A", "4dd.A", "5ee.A"]
_TRAIN_NAMES = ["1aa.A", "2bb.A", "3cc.A"]
_VAL_NAMES = ["4dd.A"]
_TEST_NAMES = ["5ee.A"]
_N_DEBUG = 252  # debug_run sampler uses SubsetRandomSampler(range(252))
PROT_1_LEN = 8
PROT_2_LEN = 16
PROT_3_LEN = 24
PROT_4_LEN = 32
PROT_5_LEN = 40
B = 5
MAX_SEQ_LENGTH = 128


def _make_coords(n: int) -> Mapping[str, list[list[float]]]:
    """Return synthetic backbone coordinates for N, CA, C, O atoms.

    Args:
        n: Number of residues to generate coordinates for.

    Returns:
        Mapping from atom name to a list of (n, 3) coordinate lists.
    """
    rng = np.random.default_rng()
    return {
        atom: rng.standard_normal((n, 3)).tolist()
        for atom in ("N", "CA", "C", "O")
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def jsonl_path(tmp_path: pathlib.Path) -> str:
    """Write JSONL file with synthetic protein entries and return its path.

    Writes one entry per name in _ENTRY_NAMES, each with _N_RES_DATA residues.
    """
    path = tmp_path / "proteins.jsonl"
    with path.open("w") as f:
        for name in _ENTRY_NAMES:
            entry = {
                "name": name,
                "seq": "ACDEFG"[:_N_RES_DATA],
                "coords": _make_coords(_N_RES_DATA),
            }
            _ = f.write(json.dumps(entry) + "\n")
    return str(path)


@pytest.fixture
def splits_path(tmp_path: pathlib.Path) -> str:
    """Write splits JSON with train/val/test name lists and return its path.

    Uses _TRAIN_NAMES, _VAL_NAMES, and _TEST_NAMES as the split contents.
    """
    path = tmp_path / "splits.json"
    with path.open("w") as f:
        json.dump(
            {
                "train": _TRAIN_NAMES,
                "validation": _VAL_NAMES,
                "test": _TEST_NAMES,
            },
            f,
        )
    return str(path)


@pytest.fixture
def train_dataset(jsonl_path: str) -> ProteinDataset:
    """ProteinDataset loaded from the synthetic JSONL with training names.

    Loads only the entries listed in _TRAIN_NAMES with max_seq_length=_MAX_SEQ.
    """
    return ProteinDataset(jsonl_path, _TRAIN_NAMES, max_seq_length=_MAX_SEQ)


@pytest.fixture
def cfg() -> TrainConfig:
    """Minimal TrainConfig with batch_size=2 and max_seq_length=_MAX_SEQ.

    Both train and test loaders share the same batch_size and max_seq_length.
    """
    return TrainConfig(
        train_loader=TrainLoaderConfig(batch_size=2, max_seq_length=_MAX_SEQ),
        test_loader=EvalLoaderConfig(batch_size=2, max_seq_length=_MAX_SEQ),
    )


# ---------------------------------------------------------------------------
# ProteinDataset — length and indexing
# ---------------------------------------------------------------------------


def test_protein_dataset_len(train_dataset: ProteinDataset) -> None:
    """ProteinDataset length equals the number of listed training names.

    Verifies that len(dataset) matches len(_TRAIN_NAMES).
    """
    assert len(train_dataset) == len(_TRAIN_NAMES)


def test_protein_dataset_excludes_unlisted_names(jsonl_path: str) -> None:
    """ProteinDataset excludes entries whose names are not in provided list.

    Verifies that constructing dataset with only ["1aa.A"] yields single entry.
    """
    ds = ProteinDataset(jsonl_path, ["1aa.A"], max_seq_length=_MAX_SEQ)
    assert len(ds) == 1


def test_protein_dataset_empty_names(jsonl_path: str) -> None:
    """ProteinDataset with an empty names list has length 0.

    Verifies that passing an empty list to ProteinDataset yields a dataset of
    length zero.
    """
    ds = ProteinDataset(jsonl_path, [], max_seq_length=_MAX_SEQ)
    assert len(ds) == 0


# ---------------------------------------------------------------------------
# ProteinDataset — sample structure
# ---------------------------------------------------------------------------


def test_protein_dataset_sample_keys(train_dataset: ProteinDataset) -> None:
    """Checks that ProteinDataset sample is a Protein instance.

    Verifies the sample is a Protein instance and has the expected attributes.
    Also asserts ProteinDataset pads/truncates all protein fields to _MAX_SEQ.
    """
    sample = train_dataset[0]
    assert isinstance(sample, Protein)

    # aatype shape is (_MAX_SEQ,) — not one-hot encoded
    expected_shapes: dict[str, tuple[int, ...]] = {
        "atom_positions": (_MAX_SEQ, 37, 3),
        "atom_mask": (_MAX_SEQ, 37),
        "residue_index": (_MAX_SEQ,),
        "aatype": (_MAX_SEQ,),
    }
    for field, shape in expected_shapes.items():
        arr = cast(object, getattr(sample, field))
        assert isinstance(arr, np.ndarray)
        assert arr.shape == shape


def test_protein_dataset_float_fields_are_float64(
    train_dataset: ProteinDataset,
) -> None:
    """Float fields in a ProteinDataset sample have dtype float64.

    Verifies atom_positions and atom_mask are numpy float64 arrays.
    """
    sample = train_dataset[0]
    assert sample.atom_positions.dtype == np.float64
    assert sample.atom_mask.dtype == np.float64


def test_protein_dataset_aatype_is_integer_array(
    train_dataset: ProteinDataset,
) -> None:
    """The aatype field in a ProteinDataset sample is an integer numpy array.

    Verifies aatype is an ndarray with an integer dtype.
    """
    aatype = train_dataset[0].aatype
    assert isinstance(aatype, np.ndarray)
    assert np.issubdtype(aatype.dtype, np.integer)


def test_protein_dataset_aatype_truncated_to_max_seq_length(
    tmp_path: pathlib.Path,
) -> None:
    """ProteinDataset truncates aatype to max_seq_length entries.

    Verifies 100-residue entry is truncated to PROT_1_LEN entries in aatype.
    """
    path = pathlib.Path(tmp_path / "long.jsonl")
    with path.open("w", encoding="utf-8") as f:
        entry = {
            "name": "long.A",
            "seq": "A" * 100,
            "coords": _make_coords(100),
        }
        _ = f.write(json.dumps(entry) + "\n")
    ds = ProteinDataset(path, ["long.A"], max_seq_length=PROT_1_LEN)
    assert ds[0].aatype.shape[0] == PROT_1_LEN


def test_protein_dataset_all_items_accessible(
    train_dataset: ProteinDataset,
) -> None:
    """Every index in ProteinDataset is accessible and returns a Protein.

    Verifies __getitem__ succeeds for valid indices and returns a Protein.
    """
    for i in range(len(train_dataset)):
        assert isinstance(train_dataset[i], Protein)


# ---------------------------------------------------------------------------
# ProteinDataset — pickle / multiprocessing compatibility
# ---------------------------------------------------------------------------


def test_protein_dataset_picklable_before_open(
    train_dataset: ProteinDataset,
) -> None:
    """ProteinDataset can be pickled and restored before JSONL file is opened.

    Verifies the dataset length is preserved after a pickle round-trip with no
    prior access.
    """
    ds2 = cast(
        ProteinDataset,
        pickle.loads(pickle.dumps(train_dataset)),  # noqa: S301
    )
    assert len(ds2) == len(train_dataset)


def test_protein_dataset_picklable_after_open(
    train_dataset: ProteinDataset,
) -> None:
    """ProteinDataset can be pickled and restored after file handle open.

    Verifies atom_positions shape is correct after a pickle round-trip with the
    handle open.
    """
    _ = train_dataset[0]  # opens the file handle
    ds2 = cast(
        ProteinDataset,
        pickle.loads(pickle.dumps(train_dataset)),  # noqa: S301
    )
    assert ds2[0].atom_positions.shape == (_MAX_SEQ, 37, 3)


# ---------------------------------------------------------------------------
# make_data_loaders — debug_run=True
# ---------------------------------------------------------------------------

# The debug sampler is SubsetRandomSampler(range(8)), so iterating requires
# a dataset with ≥256 entries; len() only reads sampler size and works cheaply.


@pytest.fixture
def debug_jsonl_path(tmp_path: pathlib.Path) -> str:
    """Write a JSONL file with _N_DEBUG protein entries for debug-mode testing.

    Each entry has _N_RES_DATA residues; names are zero-padded integers with
    the "x.A" suffix.
    """
    path = tmp_path / "debug_proteins.jsonl"
    names = [f"{i:04d}x.A" for i in range(_N_DEBUG)]
    with path.open("w") as f:
        for name in names:
            entry = {
                "name": name,
                "seq": "ACDEFG"[:_N_RES_DATA],
                "coords": _make_coords(_N_RES_DATA),
            }
            _ = f.write(json.dumps(entry) + "\n")
    return str(path)


@pytest.fixture
def debug_splits_path(tmp_path: pathlib.Path) -> str:
    """Write splits JSON assigning _N_DEBUG entries to train for debug-mode.

    Validation and test splits receive only first entry to satisfy schema.
    """
    names = [f"{i:04d}x.A" for i in range(_N_DEBUG)]
    path = tmp_path / "debug_splits.json"
    with path.open("w") as f:
        json.dump(
            {"train": names, "validation": names[:1], "test": names[:1]},
            f,
        )
    return str(path)


# ---------------------------------------------------------------------------
# Helpers for ProteinShardDataset tests
# ---------------------------------------------------------------------------


def _make_entry(name: str, seq_len: int) -> dict[str, object]:
    """Build a minimal JSONL entry with the given name and sequence length.

    Args:
        name: Protein entry identifier.
        seq_len: Number of residues (all set to alanine 'A').

    Returns:
        Dict with name, seq, and backbone coords for JSONL serialisation.
    """
    return {
        "name": name,
        "seq": "A" * seq_len,
        "coords": _make_coords(seq_len),
    }


def _write_jsonl(path: pathlib.Path, entries: list[dict[str, object]]) -> None:
    """Write entries as JSONL to path.

    Args:
        path: Destination file path.
        entries: List of dicts to serialise, one JSON object per line.
    """
    with path.open("w") as f:
        f.writelines(json.dumps(entry) + "\n" for entry in entries)


@pytest.fixture
def shard_budget(tmp_path: pathlib.Path) -> ShardBudgetParameters:
    """Minimal ShardBudgetParameters for ProteinShardDataset tests."""
    return ShardBudgetParameters(
        shard_dir=tmp_path / "shards",
        structlog_path=tmp_path / "train.jsonl",
        token_budget=512,
        max_seq_len=MAX_SEQ_LENGTH,
        seed=0,
        n_threads=1,
        world_size=1,
        rank=0,
        n_proteins_in_shard=100,
        noise_magnitude=0,
        num_workers=1,
    )


# ---------------------------------------------------------------------------
# ProteinShardDataset
# ---------------------------------------------------------------------------


def test_shard_dataset_names_len(
    tmp_path: pathlib.Path,
    shard_budget: ShardBudgetParameters,
) -> None:
    """ProteinShardDataset.names contains one entry per included protein.

    Verifies that a dataset constructed with B entries has B names.
    """
    entries = [_make_entry(f"p{i}", PROT_1_LEN) for i in range(B)]
    _write_jsonl(tmp_path / "p.jsonl", entries)
    ds = ProteinShardDataset(
        budget_parameters=shard_budget,
        names=[f"p{i}" for i in range(B)],
        dataset_jsonl=tmp_path / "p.jsonl",
        n_clusters=1,
    )
    assert len(ds.names) == B


def test_shard_dataset_builds_shards_on_init(
    tmp_path: pathlib.Path,
    shard_budget: ShardBudgetParameters,
) -> None:
    """ProteinShardDataset writes shard tars and metadata files at first init.

    Verifies that shard_metadata.json, shard_sidecar.npz, and
    all_protein_lengths.npy  all exist in shard_dir after the dataset is
    constructed.
    """
    entries = [_make_entry("p1", PROT_1_LEN)]
    _write_jsonl(tmp_path / "p.jsonl", entries)
    ds = ProteinShardDataset(
        budget_parameters=shard_budget,
        names=["p1"],
        dataset_jsonl=tmp_path / "p.jsonl",
        n_clusters=1,
    )
    assert ds.shard_metadata_path.exists()
    assert ds.shard_sidecar_path.exists()
    assert ds.lengths_path.exists()


def test_shard_dataset_parse_protein_returns_protein(
    tmp_path: pathlib.Path,
    shard_budget: ShardBudgetParameters,
) -> None:
    """parse_protein converts raw JSONL sample to a Protein of correct length.

    Verifies that atom_positions has the expected number of residues.
    """
    entry = _make_entry("p1", PROT_1_LEN)
    _write_jsonl(tmp_path / "p.jsonl", [entry])
    ds = ProteinShardDataset(
        budget_parameters=shard_budget,
        names=["p1"],
        dataset_jsonl=tmp_path / "p.jsonl",
        n_clusters=1,
    )
    protein = ds.parse_protein({"json": entry})
    assert isinstance(protein, Protein)
    assert protein.atom_positions.shape[0] == PROT_1_LEN


def test_shard_dataset_truncates_to_max_seq_length(
    tmp_path: pathlib.Path,
    shard_budget: ShardBudgetParameters,
) -> None:
    """parse_protein truncates proteins to max_seq_len residues.

    Verifies that a 600-residue entry truncated to MAX_SEQ_LENGTH (128) atoms.
    """
    entry = _make_entry("p1", 600)
    _write_jsonl(tmp_path / "p.jsonl", [entry])
    ds = ProteinShardDataset(
        budget_parameters=shard_budget,
        names=["p1"],
        dataset_jsonl=tmp_path / "p.jsonl",
        n_clusters=1,
    )
    protein = ds.parse_protein({"json": entry})
    assert protein.atom_positions.shape[0] == MAX_SEQ_LENGTH


def test_shard_dataset_pickles(
    tmp_path: pathlib.Path,
    shard_budget: ShardBudgetParameters,
) -> None:
    """ProteinShardDataset survives a pickle round-trip for DataLoader workers.

    Verifies names are preserved after serialising and deserialising dataset.
    """
    entries = [_make_entry("p1", PROT_1_LEN)]
    _write_jsonl(tmp_path / "p.jsonl", entries)
    ds = ProteinShardDataset(
        budget_parameters=shard_budget,
        names=["p1"],
        dataset_jsonl=tmp_path / "p.jsonl",
        n_clusters=1,
    )
    restored = cast(
        ProteinShardDataset,
        pickle.loads(pickle.dumps(ds)),  # noqa: S301
    )
    assert restored.names == ["p1"]


# ---------------------------------------------------------------------------
# Fixtures for bucketed loader tests
# ---------------------------------------------------------------------------


@pytest.fixture
def bucketed_jsonl(tmp_path: pathlib.Path) -> str:
    """Write a JSONL with 5 proteins of varying lengths.

    Proteins p1-p5 have lengths 8, 16, 24, 32, and 40 residues respectively.
    """
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
    with path.open("w") as f:
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
    """Minimal TrainConfig with token_budget=512 for bucketed loader tests."""
    return TrainConfig(
        train_loader=TrainLoaderConfig(
            batch_size=2,
            max_seq_length=_MAX_SEQ,
            token_budget=512,
            num_workers=1,
            epoch_prefetch_depth=1,
        ),
        test_loader=EvalLoaderConfig(
            batch_size=2,
            max_seq_length=_MAX_SEQ,
        ),
    )


@pytest.fixture
def bucketed_train_args(
    bucketed_jsonl: str,
    bucketed_splits: str,
    tmp_path: pathlib.Path,
) -> TrainArgs:
    """TrainArgs pointing at the bucketed JSONL and splits fixtures."""
    return TrainArgs(
        dataset_jsonl=pathlib.Path(bucketed_jsonl),
        shard_dir=tmp_path / "shards",
        keys_for_splits_json=pathlib.Path(bucketed_splits),
        config=tmp_path / "config.json",
        structlog_jsonl=tmp_path / "train.jsonl",
        ddp=False,
        debug_run=False,
    )


@pytest.fixture
def debug_train_args(
    debug_jsonl_path: str,
    debug_splits_path: str,
    tmp_path: pathlib.Path,
) -> TrainArgs:
    """TrainArgs fixture with debug_run=True."""
    return TrainArgs(
        dataset_jsonl=pathlib.Path(debug_jsonl_path),
        shard_dir=tmp_path / "shards",
        keys_for_splits_json=pathlib.Path(debug_splits_path),
        config=tmp_path / "config.json",
        structlog_jsonl=tmp_path / "train.jsonl",
        ddp=False,
        debug_run=True,
    )


@pytest.fixture
def full_dataset_jsonl(tmp_path: pathlib.Path) -> pathlib.Path:
    """Write a JSONL file with _N_FULL synthetic protein entries.

    Used to seed a shard cache before a debug_run=True call so the
    cache-poisoning scenario can be exercised.
    """
    path = tmp_path / "full_proteins.jsonl"
    names = [f"{i:04d}p.A" for i in range(_N_FULL)]
    with path.open("w") as f:
        for name in names:
            entry = {
                "name": name,
                "seq": "ACDEFG"[:_N_RES_DATA],
                "coords": _make_coords(_N_RES_DATA),
            }
            _ = f.write(json.dumps(entry) + "\n")
    return path


@pytest.fixture
def full_dataset_splits(tmp_path: pathlib.Path) -> pathlib.Path:
    """Write a splits JSON assigning all _N_FULL entries to train."""
    names = [f"{i:04d}p.A" for i in range(_N_FULL)]
    path = tmp_path / "full_splits.json"
    with path.open("w") as f:
        json.dump(
            {"train": names, "validation": names[:1], "test": names[:1]},
            f,
        )
    return path


@pytest.fixture
def full_train_args(
    full_dataset_jsonl: pathlib.Path,
    full_dataset_splits: pathlib.Path,
    tmp_path: pathlib.Path,
) -> TrainArgs:
    """TrainArgs pointing at the _N_FULL dataset with debug_run=False."""
    return TrainArgs(
        dataset_jsonl=full_dataset_jsonl,
        shard_dir=tmp_path / "shards",
        keys_for_splits_json=full_dataset_splits,
        config=tmp_path / "config.json",
        structlog_jsonl=tmp_path / "train.jsonl",
        ddp=False,
        debug_run=False,
    )


@pytest.fixture
def full_debug_train_args(
    full_dataset_jsonl: pathlib.Path,
    full_dataset_splits: pathlib.Path,
    tmp_path: pathlib.Path,
) -> TrainArgs:
    """TrainArgs pointing at the _N_FULL dataset with debug_run=True."""
    return TrainArgs(
        dataset_jsonl=full_dataset_jsonl,
        shard_dir=tmp_path / "shards",
        keys_for_splits_json=full_dataset_splits,
        config=tmp_path / "config.json",
        structlog_jsonl=tmp_path / "train.jsonl",
        ddp=False,
        debug_run=True,
    )


# ---------------------------------------------------------------------------
# make_bucketed_data_loaders tests
# ---------------------------------------------------------------------------


def test_bucketed_train_loader_yields_protein_batch(
    bucketed_cfg: TrainConfig,
    bucketed_train_args: TrainArgs,
) -> None:
    """Training loader yields a list of Protein objects."""
    train_loader, _, _ = make_bucketed_data_loaders(
        cfg=bucketed_cfg,
        extra_train_args=bucketed_train_args,
    )
    batch = cast(list[Protein], next(iter(train_loader)))
    assert isinstance(batch, list)
    assert all(isinstance(p, Protein) for p in batch)


def test_bucketed_debug_run_train_dataset_has_252_items(
    bucketed_cfg: TrainConfig,
    debug_train_args: TrainArgs,
) -> None:
    """Using debug_run=True yields a train dataset of _N_DEBUG items."""
    train_loader, _, _ = make_bucketed_data_loaders(
        cfg=bucketed_cfg,
        extra_train_args=debug_train_args,
    )
    assert (
        len(cast(ShardDataLoader, train_loader).shard_dataset.names) == _N_DEBUG
    )


_N_FULL = 300  # dataset larger than _N_DEBUG to expose cache-poisoning bug


def test_bucketed_debug_run_not_poisoned_by_prior_full_cache(
    bucketed_cfg: TrainConfig,
    full_train_args: TrainArgs,
    full_debug_train_args: TrainArgs,
) -> None:
    """Must yield _N_DEBUG items even when shard cache built by full run.

    A cache built from all _N_FULL proteins must not silently return _N_FULL
    items for subsequent debug_run=True call that caps train names at _N_DEBUG.
    """
    # Seed the shard cache with the full _N_FULL-protein dataset.
    _, _, _ = make_bucketed_data_loaders(
        cfg=bucketed_cfg,
        extra_train_args=full_train_args,
    )

    # A subsequent debug_run=True call must cap at _N_DEBUG despite warm cache.
    train_loader, _, _ = make_bucketed_data_loaders(
        cfg=bucketed_cfg,
        extra_train_args=full_debug_train_args,
    )
    assert (
        len(cast(ShardDataLoader, train_loader).shard_dataset.names) == _N_DEBUG
    )
