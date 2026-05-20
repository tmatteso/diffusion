"""BucketedBatchSampler: sortish token-budget batching with a process-pool prefetch queue."""

import random


def _compute_batch_plan(
    flat_to_cluster: list[int],
    cluster_rep_len: list[int],
    n_proteins: int,
    token_budget: int,
    chunk_multiplier: int,
    seed: int,
) -> list[list[int]]:
    """Compute one epoch's batch plan using sortish sampling and greedy token packing.

    This is a module-level function so it can be submitted to a ProcessPoolExecutor
    (must be picklable). All inputs and outputs are plain Python lists.

    Algorithm:
        1. Shuffle all indices with Random(seed).
        2. Split into chunks of chunk_size = chunk_multiplier * (token_budget // median_rep_len).
        3. Sort each chunk ascending by cluster_rep_len (shortest first).
        4. Greedy pack: accumulate proteins until adding the next would exceed token_budget.
           Overflow proteins (rep_len > token_budget) always become singleton batches.

    Args:
        flat_to_cluster:  Cluster id for each global protein index.
        cluster_rep_len:  Representative length for each cluster id.
        n_proteins:       Total number of proteins.
        token_budget:     Maximum cumulative rep_len per batch.
        chunk_multiplier: Controls sortish-window width (default 16 ~= 16 full batches).
        seed:             RNG seed; use seed + epoch to get per-epoch shuffles.

    Returns:
        List of batches; each batch is a list of flat protein indices.
    """
    rng = random.Random(seed)
    indices = list(range(n_proteins))
    rng.shuffle(indices)

    # Chunk size: ~chunk_multiplier full batches per sortish window.
    # Use the median regular cluster's rep_len as the denominator.
    n_regular = len(cluster_rep_len) - 1  # last entry is overflow
    median_rep_len = cluster_rep_len[n_regular // 2]
    avg_proteins_per_budget = max(1, token_budget // median_rep_len)
    chunk_size = chunk_multiplier * avg_proteins_per_budget

    batches: list[list[int]] = []

    for chunk_start in range(0, n_proteins, chunk_size):
        chunk = indices[chunk_start : chunk_start + chunk_size]
        chunk.sort(key=lambda i: cluster_rep_len[flat_to_cluster[i]])

        current_batch: list[int] = []
        current_budget = 0

        for i in chunk:
            rep_len = cluster_rep_len[flat_to_cluster[i]]
            if rep_len > token_budget:
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_budget = 0
                batches.append([i])
            elif current_budget + rep_len > token_budget:
                batches.append(current_batch)
                current_batch = [i]
                current_budget = rep_len
            else:
                current_batch.append(i)
                current_budget += rep_len

        if current_batch:
            batches.append(current_batch)

    return batches
