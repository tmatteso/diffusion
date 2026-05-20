"""Tests for _compute_batch_plan and BucketedBatchSampler."""

from helpers.bucketed_sampler import _compute_batch_plan

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
