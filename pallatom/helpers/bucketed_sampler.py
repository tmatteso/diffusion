"""BucketedBatchSampler: sortish token-budget batching with a process-pool prefetch queue."""

import math
import multiprocessing as mp
import queue
import random
from collections.abc import Iterator
from concurrent.futures import Future, ProcessPoolExecutor

import torch
import torch.utils.data
from helpers.cluster_index import ClusterIndex


def compute_batch_plan(
    flat_to_cluster: list[int],
    cluster_rep_len: list[int],
    n_proteins: int,
    token_budget: int,
    chunk_multiplier: int,
    seed: int,
    max_seq_len: int,
) -> list[list[int]]:
    """Compute one epoch's batch plan using sortish sampling and greedy token packing.

    This is a module-level function so it can be submitted to a ProcessPoolExecutor
    (must be picklable). All inputs and outputs are plain Python lists.

    Algorithm:
        1. Shuffle all indices with Random(seed).
        2. Split into chunks of chunk_size = chunk_multiplier * (token_budget // median_rep_len).
        3. Sort each chunk ascending by cluster_rep_len (shortest first).
        4. Greedy pack: accumulate proteins until the padded batch cost would exceed
           token_budget. Because the chunk is sorted ascending, each new protein has the
           largest effective_len seen so far, so the padded cost is simply
           (batch_size + 1) * effective_len. Proteins whose effective_len alone exceeds
           the budget become singleton batches.

    Args:
        flat_to_cluster:  Cluster id for each global protein index.
        cluster_rep_len:  Representative length for each cluster id.
        n_proteins:       Total number of proteins.
        token_budget:     Maximum cumulative rep_len per batch.
        chunk_multiplier: Controls sortish-window width (default 16 ~= 16 full batches).
        seed:             RNG seed; use seed + epoch to get per-epoch shuffles.
        max_seq_len:      Dataset truncation cap; rep_len is capped at this value for
                          budget accounting so truncated proteins pack correctly.

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

        for i in chunk:
            rep_len = cluster_rep_len[flat_to_cluster[i]]
            effective_len = min(rep_len, max_seq_len)
            if effective_len > token_budget:
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                batches.append([i])
            elif (len(current_batch) + 1) * effective_len > token_budget:
                # Chunk is sorted ascending, so effective_len is the new batch max.
                # Check padded cost rather than sum to prevent padding blowup.
                batches.append(current_batch)
                current_batch = [i]
            else:
                current_batch.append(i)

        if current_batch:
            batches.append(current_batch)

    return batches


class BucketedBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Token-budget batch sampler with a process-pool prefetch queue.

    Implements sortish sampling: shuffles all indices, splits into chunks, sorts
    within each chunk by representative length, then greedily packs proteins into
    batches that respect the token budget. Overflow proteins (rep_len > token_budget)
    are always singleton batches.

    Batch plans are precomputed in a subprocess via ProcessPoolExecutor so that the
    next epoch's plan is ready before the current epoch finishes. A bounded queue
    of `prefetch_epochs` futures ensures the pipeline stays full.

    DDP-aware: pads the batch list to a multiple of world_size and yields only the
    batches assigned to this rank (rank-strided).

    Usage::

        sampler = BucketedBatchSampler(cluster_index, token_budget=512)
        loader  = DataLoader(dataset, batch_sampler=sampler, collate_fn=...)
        for epoch in range(n_epochs):
            sampler.set_epoch(epoch)
            for batch in loader:
                ...

    Args:
        cluster_index:    ClusterIndex exposing flat_to_cluster and cluster_rep_len.
        token_budget:     Maximum cumulative representative length per batch.
        max_seq_len:      Dataset truncation cap passed to compute_batch_plan so that
                          proteins whose rep_len exceeds it are packed at their truncated
                          length rather than as singletons.
        chunk_multiplier: Sortish window width in multiples of avg proteins per batch.
        world_size:       Number of DDP processes. Default 1 (single GPU).
        rank:             This process's DDP rank. Default 0.
        seed:             Base RNG seed; epoch is added before each call to
                          compute_batch_plan for per-epoch diversity.
        prefetch_epochs:  Queue depth — how many future epochs to precompute. Default 2.
    """

    def __init__(
        self,
        cluster_index: ClusterIndex,
        token_budget: int,
        max_seq_len: int,
        chunk_multiplier: int = 16,
        world_size: int = 1,
        rank: int = 0,
        seed: int = 0,
        prefetch_epochs: int = 2,
    ) -> None:
        self._cluster_index = cluster_index
        self._token_budget = token_budget
        self._max_seq_len = max_seq_len
        self._chunk_multiplier = chunk_multiplier
        self._world_size = world_size
        self._rank = rank
        self._seed = seed
        self._prefetch_epochs = prefetch_epochs
        self._epoch = 0
        self._current_batches: list[list[int]] = []

        self._executor = ProcessPoolExecutor(
            max_workers=1,
            mp_context=mp.get_context("spawn"),
        )
        self._queue: queue.Queue[Future[list[list[int]]]] = queue.Queue(maxsize=prefetch_epochs)

        for offset in range(prefetch_epochs):
            self._queue.put_nowait(self._submit(seed + offset))

    def _submit(self, seed: int) -> "Future[list[list[int]]]":
        """Submit a batch plan computation to the executor.

        Args:
            seed: RNG seed for this epoch's batch plan.

        Returns:
            Future resolving to the batch plan (list of batches).
        """
        return self._executor.submit(
            compute_batch_plan,
            self._cluster_index.flat_to_cluster,
            self._cluster_index.cluster_rep_len,
            len(self._cluster_index),
            self._token_budget,
            self._chunk_multiplier,
            seed,
            self._max_seq_len,
        )

    def set_epoch(self, epoch: int) -> None:
        """Swap in the precomputed batch plan and enqueue the next epoch's plan.

        Must be called once per epoch before iterating. Blocks if the prefetch
        process has not yet finished (rare — it should finish well before the
        previous epoch ends).

        Args:
            epoch: Current epoch number (0-indexed).
        """
        self._epoch = epoch
        self._current_batches = self._queue.get().result()
        self._queue.put_nowait(self._submit(self._seed + epoch + self._prefetch_epochs))

    def __iter__(self) -> Iterator[list[int]]:
        """Yield batches for this rank.

        Pads the batch list to a multiple of world_size (repeating the first batch)
        so every rank gets the same number of batches, then yields rank-strided.
        """
        all_batches = list(self._current_batches)
        remainder = len(all_batches) % self._world_size
        if remainder != 0:
            all_batches = all_batches + all_batches[: self._world_size - remainder]
        yield from all_batches[self._rank :: self._world_size]

    def __len__(self) -> int:
        """Return the number of batches this rank will yield.

        Returns an exact count after set_epoch has been called, or a rough
        estimate before the first call (safe for tqdm progress bars).
        """
        if self._current_batches:
            n_total = len(self._current_batches)
            remainder = n_total % self._world_size
            if remainder != 0:
                n_total += self._world_size - remainder
            return n_total // self._world_size
        # Estimate before first set_epoch.
        n = len(self._cluster_index)
        n_regular = len(self._cluster_index.cluster_rep_len) - 1
        median_rep = self._cluster_index.cluster_rep_len[n_regular // 2]
        avg_per_batch = max(1, self._token_budget // median_rep)
        return math.ceil(math.ceil(n / avg_per_batch) / self._world_size)

    def __del__(self) -> None:
        """Shut down the executor without blocking process exit."""
        self._executor.shutdown(wait=False)
