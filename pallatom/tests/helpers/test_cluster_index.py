"""Tests for ClusterIndex."""

import json
import pathlib
from collections.abc import Mapping, Sequence

import pytest
from helpers.cluster_index import ClusterIndex

_Entry = Mapping[str, str | Mapping[str, list[list[float]]]]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_coords(n: int) -> dict[str, list[list[float]]]:
    """Build a minimal coords dict with n atoms at the origin."""
    return {atom: [[0.0, 0.0, 0.0]] * n for atom in ("N", "CA", "C", "O")}


def _write_jsonl(
    path: pathlib.Path,
    entries: Sequence[Mapping[str, str | Mapping[str, list[list[float]]]]],
) -> None:
    """Write entries as JSONL to path."""
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# assign_cluster
# ---------------------------------------------------------------------------


def test_assign_cluster_first_bin(tmp_path: pathlib.Path) -> None:
    """Length 1 maps to cluster 0 (first bin [1-8])."""
    _write_jsonl(tmp_path / "p.jsonl", [{"name": "p1", "seq": "A", "coords": _make_coords(1)}])
    idx = ClusterIndex(tmp_path / "p.jsonl", ["p1"], token_budget=512, n_clusters=64)
    assert idx.assign_cluster(1) == 0


def test_assign_cluster_last_regular_bin(tmp_path: pathlib.Path) -> None:
    """Length 512 maps to cluster 63 (last regular bin [505-512])."""
    seq = "A" * 512
    _write_jsonl(
        tmp_path / "p.jsonl",
        [{"name": "p1", "seq": seq, "coords": _make_coords(512)}],
    )
    idx = ClusterIndex(tmp_path / "p.jsonl", ["p1"], token_budget=512, n_clusters=64)
    assert idx.assign_cluster(512) == 63


def test_assign_cluster_overflow(tmp_path: pathlib.Path) -> None:
    """Length > 512 maps to overflow cluster 64."""
    seq = "A" * 513
    _write_jsonl(
        tmp_path / "p.jsonl",
        [{"name": "p1", "seq": seq, "coords": _make_coords(513)}],
    )
    idx = ClusterIndex(tmp_path / "p.jsonl", ["p1"], token_budget=512, n_clusters=64)
    assert idx.assign_cluster(513) == 64


# ---------------------------------------------------------------------------
# cluster_rep_len
# ---------------------------------------------------------------------------


def test_cluster_rep_len_regular(tmp_path: pathlib.Path) -> None:
    """cluster_rep_len[0] == 8, cluster_rep_len[63] == 512."""
    _write_jsonl(tmp_path / "p.jsonl", [{"name": "p1", "seq": "A", "coords": _make_coords(1)}])
    idx = ClusterIndex(tmp_path / "p.jsonl", ["p1"], token_budget=512, n_clusters=64)
    assert idx.cluster_rep_len[0] == 8
    assert idx.cluster_rep_len[63] == 512


def test_cluster_rep_len_overflow(tmp_path: pathlib.Path) -> None:
    """cluster_rep_len[64] == token_budget + 1 == 513."""
    _write_jsonl(tmp_path / "p.jsonl", [{"name": "p1", "seq": "A", "coords": _make_coords(1)}])
    idx = ClusterIndex(tmp_path / "p.jsonl", ["p1"], token_budget=512, n_clusters=64)
    assert idx.cluster_rep_len[64] == 513


# ---------------------------------------------------------------------------
# file creation and flat index
# ---------------------------------------------------------------------------


def test_cluster_index_creates_cluster_dir(tmp_path: pathlib.Path) -> None:
    """ClusterIndex creates the cluster directory and 65 JSONL files."""
    entries: list[_Entry] = [
        {"name": "p1", "seq": "A" * 8, "coords": _make_coords(8)},
        {"name": "p2", "seq": "A" * 16, "coords": _make_coords(16)},
    ]
    _write_jsonl(tmp_path / "proteins.jsonl", entries)
    ClusterIndex(tmp_path / "proteins.jsonl", ["p1", "p2"])
    cluster_dir = tmp_path / "proteins_clusters"
    assert cluster_dir.is_dir()
    for k in range(65):
        assert (cluster_dir / f"cluster_{k:03d}.jsonl").exists()


def test_cluster_index_flat_to_cluster_correct(tmp_path: pathlib.Path) -> None:
    """flat_to_cluster assigns each protein to the expected bucket."""
    entries: list[_Entry] = [
        {"name": "p1", "seq": "A" * 8, "coords": _make_coords(8)},  # → cluster 0
        {"name": "p2", "seq": "A" * 9, "coords": _make_coords(9)},  # → cluster 1
        {"name": "p3", "seq": "A" * 513, "coords": _make_coords(513)},  # → overflow 64
    ]
    _write_jsonl(tmp_path / "proteins.jsonl", entries)
    idx = ClusterIndex(tmp_path / "proteins.jsonl", ["p1", "p2", "p3"])
    cluster_ids = sorted(idx.flat_to_cluster[i] for i in range(len(idx)))
    assert cluster_ids == [0, 1, 64]


def test_cluster_index_len(tmp_path: pathlib.Path) -> None:
    """__len__ returns the total number of included proteins."""
    entries: list[_Entry] = [
        {"name": f"p{i}", "seq": "A" * (i + 1), "coords": _make_coords(i + 1)} for i in range(10)
    ]
    _write_jsonl(tmp_path / "proteins.jsonl", entries)
    idx = ClusterIndex(tmp_path / "proteins.jsonl", [f"p{i}" for i in range(10)])
    assert len(idx) == 10


def test_cluster_index_name_filter(tmp_path: pathlib.Path) -> None:
    """Only names in the provided list are included."""
    entries: list[_Entry] = [
        {"name": "p1", "seq": "A" * 8, "coords": _make_coords(8)},
        {"name": "p2", "seq": "A" * 8, "coords": _make_coords(8)},
        {"name": "p3", "seq": "A" * 8, "coords": _make_coords(8)},
    ]
    _write_jsonl(tmp_path / "proteins.jsonl", entries)
    idx = ClusterIndex(tmp_path / "proteins.jsonl", ["p1", "p3"])
    assert len(idx) == 2


# ---------------------------------------------------------------------------
# cache hit
# ---------------------------------------------------------------------------


def test_cluster_index_cache_hit_same_result(tmp_path: pathlib.Path) -> None:
    """Second construction from the same path reads cache and produces identical arrays."""
    entries: list[_Entry] = [
        {"name": "p1", "seq": "A" * 8, "coords": _make_coords(8)},
        {"name": "p2", "seq": "A" * 16, "coords": _make_coords(16)},
    ]
    _write_jsonl(tmp_path / "proteins.jsonl", entries)
    idx1 = ClusterIndex(tmp_path / "proteins.jsonl", ["p1", "p2"])
    idx2 = ClusterIndex(tmp_path / "proteins.jsonl", ["p1", "p2"])
    assert idx1.flat_to_cluster == idx2.flat_to_cluster
    assert idx1.flat_to_local == idx2.flat_to_local
    assert idx1.cluster_offsets == idx2.cluster_offsets


# ---------------------------------------------------------------------------
# assign_cluster edge cases
# ---------------------------------------------------------------------------


def test_assign_cluster_raises_on_zero(tmp_path: pathlib.Path) -> None:
    """assign_cluster raises ValueError for seq_len <= 0."""
    _write_jsonl(tmp_path / "p.jsonl", [{"name": "p1", "seq": "A", "coords": _make_coords(1)}])
    idx = ClusterIndex(tmp_path / "p.jsonl", ["p1"], token_budget=512, n_clusters=64)
    with pytest.raises(ValueError, match="seq_len must be positive"):
        idx.assign_cluster(0)
