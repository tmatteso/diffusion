"""Tests for _compute_batch_plan and BucketedBatchSampler."""

import json
import pathlib

import pytest
from helpers.bucketed_sampler import BucketedBatchSampler, _compute_batch_plan
from helpers.cluster_index import ClusterIndex

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rep_lens(n_clusters: int = 64, token_budget: int = 512) -> list[int]:
    """Return the cluster_rep_len list for n_clusters regular clusters + overflow."""
    bin_width = token_budget // n_clusters
    return [bin_width * (k + 1) for k in range(n_clusters)] + [token_budget + 1]


# ---------------------------------------------------------------------------
# _compute_batch_plan
# ---------------------------------------------------------------------------


def test_batch_plan_respects_token_budget() -> None:
    """No batch in the plan exceeds the token budget (using representative lengths)."""
    n = 200
    flat_to_cluster = [0] * n  # all in cluster 0, rep_len=8
    rep_lens = _rep_lens()
    batches = _compute_batch_plan(flat_to_cluster, rep_lens, n, 512, 16, seed=0)
    for batch in batches:
        total = sum(rep_lens[flat_to_cluster[i]] for i in batch)
        assert total <= 512, f"Batch budget exceeded: {total}"


def test_batch_plan_covers_all_proteins() -> None:
    """Every protein index appears exactly once across all batches."""
    n = 100
    flat_to_cluster = list(range(64)) * (n // 64) + list(range(n % 64))
    rep_lens = _rep_lens()
    batches = _compute_batch_plan(flat_to_cluster, rep_lens, n, 512, 16, seed=0)
    all_indices = sorted(i for batch in batches for i in batch)
    assert all_indices == list(range(n))


def test_batch_plan_overflow_is_singleton() -> None:
    """A protein in the overflow cluster is always a singleton batch."""
    flat_to_cluster = [64]  # one overflow protein
    rep_lens = _rep_lens()
    batches = _compute_batch_plan(flat_to_cluster, rep_lens, 1, 512, 16, seed=0)
    assert len(batches) == 1
    assert batches[0] == [0]


def test_batch_plan_different_seeds_differ() -> None:
    """Different seeds produce different orderings for the same data."""
    n = 200
    flat_to_cluster = [0] * n
    rep_lens = _rep_lens()
    batches_a = _compute_batch_plan(flat_to_cluster, rep_lens, n, 512, 16, seed=0)
    batches_b = _compute_batch_plan(flat_to_cluster, rep_lens, n, 512, 16, seed=1)
    indices_a = [i for batch in batches_a for i in batch]
    indices_b = [i for batch in batches_b for i in batch]
    assert indices_a != indices_b


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_coords(n: int) -> dict[str, list[list[float]]]:
    """Build a minimal coords dict with n atoms at the origin."""
    return {atom: [[0.0, 0.0, 0.0]] * n for atom in ("N", "CA", "C", "O")}


@pytest.fixture
def small_cluster_index(tmp_path: pathlib.Path) -> ClusterIndex:
    """ClusterIndex over 80 synthetic proteins spread across 8 clusters."""
    entries = [
        {
            "name": f"p{i}",
            "seq": "A" * ((i % 8 + 1) * 8),
            "coords": _make_coords((i % 8 + 1) * 8),
        }
        for i in range(80)
    ]
    path = tmp_path / "proteins.jsonl"
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    return ClusterIndex(path, [f"p{i}" for i in range(80)], token_budget=512, n_clusters=64)


# ---------------------------------------------------------------------------
# BucketedBatchSampler
# ---------------------------------------------------------------------------


def test_sampler_covers_all_proteins(small_cluster_index: ClusterIndex) -> None:
    """Every protein index appears exactly once per epoch."""
    sampler = BucketedBatchSampler(small_cluster_index, token_budget=512, seed=0, prefetch_epochs=1)
    sampler.set_epoch(0)
    all_indices = sorted(i for batch in sampler for i in batch)
    assert all_indices == list(range(len(small_cluster_index)))


def test_sampler_respects_token_budget(small_cluster_index: ClusterIndex) -> None:
    """No batch exceeds the token budget (by representative lengths)."""
    sampler = BucketedBatchSampler(small_cluster_index, token_budget=512, seed=0, prefetch_epochs=1)
    sampler.set_epoch(0)
    for batch in sampler:
        total = sum(
            small_cluster_index.cluster_rep_len[small_cluster_index.flat_to_cluster[i]]
            for i in batch
        )
        assert total <= 512


def test_sampler_ddp_equal_length(small_cluster_index: ClusterIndex) -> None:
    """Both DDP ranks receive the same number of batches per epoch."""
    sampler_r0 = BucketedBatchSampler(
        small_cluster_index, token_budget=512, world_size=2, rank=0, seed=0, prefetch_epochs=1
    )
    sampler_r1 = BucketedBatchSampler(
        small_cluster_index, token_budget=512, world_size=2, rank=1, seed=0, prefetch_epochs=1
    )
    sampler_r0.set_epoch(0)
    sampler_r1.set_epoch(0)
    assert len(list(sampler_r0)) == len(list(sampler_r1))


def test_sampler_set_epoch_reshuffles(small_cluster_index: ClusterIndex) -> None:
    """Different epochs produce different batch orderings."""
    sampler = BucketedBatchSampler(small_cluster_index, token_budget=512, seed=0, prefetch_epochs=3)
    sampler.set_epoch(0)
    batches_e0 = [batch[:] for batch in sampler]
    sampler.set_epoch(1)
    batches_e1 = [batch[:] for batch in sampler]
    assert batches_e0 != batches_e1


def test_sampler_len_after_set_epoch(small_cluster_index: ClusterIndex) -> None:
    """__len__ returns the correct batch count after set_epoch."""
    sampler = BucketedBatchSampler(small_cluster_index, token_budget=512, seed=0, prefetch_epochs=1)
    sampler.set_epoch(0)
    assert len(sampler) == len(list(sampler))
