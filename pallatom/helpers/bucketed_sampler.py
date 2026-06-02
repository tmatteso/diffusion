"""BucketedBatchSampler: sortish token-budget batching with a process-pool prefetch queue."""

import math
import multiprocessing as mp
import os
import queue
from collections.abc import Iterator
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
import torch.utils.data
from helpers.cluster_index import ClusterIndex

# Compact batch plan: flat protein indices concatenated across all batches, plus
# exclusive end positions marking where each batch ends in the flat array.
BatchPlan = tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]


@dataclass
class _WorkerState:
    """Per-worker-process cache for large index arrays set once by _worker_init."""

    flat_to_cluster: npt.NDArray[np.int32] | None = None
    cluster_rep_len: npt.NDArray[np.int32] | None = None


_W = _WorkerState()


def _worker_init(
    flat_to_cluster: npt.NDArray[np.int32],
    cluster_rep_len: npt.NDArray[np.int32],
) -> None:
    """Store large index arrays as worker-process globals.

    Called exactly once per worker by ProcessPoolExecutor. Subsequent
    _compute_batch_plan_worker calls read these globals instead of receiving
    the arrays via pickle on every invocation — a significant memory and
    latency saving for datasets with millions of proteins.

    Args:
        flat_to_cluster: Int32 array mapping each protein to its cluster id.
        cluster_rep_len: Int32 array of representative lengths per cluster.
    """
    _W.flat_to_cluster = flat_to_cluster
    _W.cluster_rep_len = cluster_rep_len


def compute_batch_plan(
    flat_to_cluster: list[int] | npt.NDArray[np.int32],
    cluster_rep_len: list[int] | npt.NDArray[np.int32],
    n_proteins: int,
    token_budget: int,
    chunk_multiplier: int,
    seed: int,
    max_seq_len: int,
    n_threads: int = 1,
) -> list[list[int]]:
    """Compute one epoch's batch plan using sortish sampling and greedy token packing.

    This module-level function accepts explicit array arguments so it can be called
    directly in tests without touching the worker-process globals. The internal
    algorithm uses numpy for shuffle and sort (faster; numpy releases the GIL for
    sort so multiple threads run truly in parallel).

    Algorithm:
        1. Shuffle all indices with numpy default_rng(seed).
        2. Split into chunks of chunk_size = chunk_multiplier * (token_budget //
           median_rep_len).
        3. Sort each chunk ascending by effective_len (numpy argsort, GIL-free).
        4. Greedy pack each sorted chunk: accumulate until the padded batch cost
           (batch_size * max_eff_len_in_batch) would exceed token_budget. Chunks
           processed in parallel via ThreadPoolExecutor(n_threads).
        5. Return a flat list of batches in chunk order.

    Args:
        flat_to_cluster: Cluster id per protein index.
        cluster_rep_len: Representative length per cluster id.
        n_proteins: Total number of proteins.
        token_budget: Maximum padded batch cost (batch_size * max eff_len).
        chunk_multiplier: Sortish window width in multiples of avg proteins per batch.
        seed: RNG seed; vary per epoch for diversity.
        max_seq_len: Dataset truncation cap; rep_len is capped before budget accounting.
        n_threads: Worker threads for parallel chunk sort+pack (default 1).

    Returns:
        List of batches; each batch is a list of flat protein indices.
    """
    ftc = np.asarray(flat_to_cluster, dtype=np.int32)
    crl = np.asarray(cluster_rep_len, dtype=np.int32)

    rng = np.random.default_rng(seed)
    indices = np.arange(n_proteins, dtype=np.int32)
    rng.shuffle(indices)  # in-place, avoids int64 intermediate

    n_regular = len(crl) - 1
    median_rep_len = int(crl[n_regular // 2])
    avg_per_budget = max(1, token_budget // median_rep_len)
    chunk_size = chunk_multiplier * avg_per_budget

    def process_chunk(start: int) -> list[list[int]]:
        """Sort and greedily pack one sortish chunk.

        Args:
            start: Start index into the shuffled indices array.

        Returns:
            List of batches for this chunk.
        """
        chunk = indices[start : start + chunk_size]
        eff = np.minimum(crl[ftc[chunk]], max_seq_len)
        order = np.argsort(eff, kind="stable")  # releases GIL
        sorted_chunk = chunk[order].tolist()
        sorted_eff = eff[order].tolist()

        batches: list[list[int]] = []
        current: list[int] = []
        for i, eff_len in zip(sorted_chunk, sorted_eff, strict=True):
            if eff_len > token_budget:
                if current:
                    batches.append(current)
                    current = []
                batches.append([i])
            elif (len(current) + 1) * eff_len > token_budget:
                batches.append(current)
                current = [i]
            else:
                current.append(i)
        if current:
            batches.append(current)
        return batches

    chunk_starts = list(range(0, n_proteins, chunk_size))
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        packed: list[list[list[int]]] = list(pool.map(process_chunk, chunk_starts))

    return [batch for chunk_batches in packed for batch in chunk_batches]


def _compute_batch_plan_worker(
    n_proteins: int,
    token_budget: int,
    chunk_multiplier: int,
    seed: int,
    max_seq_len: int,
    n_threads: int,
) -> BatchPlan:
    """Compute one epoch's batch plan in the worker subprocess and return compact format.

    Reads flat_to_cluster and cluster_rep_len from worker-process globals set once
    by _worker_init — no re-pickling of large arrays on each call. Delegates to
    compute_batch_plan then packs the result into a compact numpy representation
    to minimise the data transferred back to the main process via pickle.

    Args:
        n_proteins: Total number of proteins.
        token_budget: Maximum padded batch cost per batch.
        chunk_multiplier: Sortish window width.
        seed: RNG seed for this epoch.
        max_seq_len: Dataset truncation cap.
        n_threads: Worker threads for parallel chunk processing.

    Returns:
        BatchPlan: (flat_indices, batch_ends) where flat_indices (int32) holds all
        protein indices concatenated across batches and batch_ends (int32) holds the
        exclusive end position of each batch within flat_indices.
    """
    assert _W.flat_to_cluster is not None
    assert _W.cluster_rep_len is not None
    batches = compute_batch_plan(
        _W.flat_to_cluster,
        _W.cluster_rep_len,
        n_proteins,
        token_budget,
        chunk_multiplier,
        seed,
        max_seq_len,
        n_threads,
    )
    total = sum(len(b) for b in batches)
    flat_indices = np.empty(total, dtype=np.int32)
    batch_ends = np.empty(len(batches), dtype=np.int32)
    pos = 0
    for idx, batch in enumerate(batches):
        n = len(batch)
        flat_indices[pos : pos + n] = batch
        pos += n
        batch_ends[idx] = pos
    return flat_indices, batch_ends


class BucketedBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Token-budget batch sampler with a multithreaded process-pool prefetch queue.

    Implements sortish sampling: shuffles all indices, splits into chunks, sorts
    within each chunk by representative length, then greedily packs proteins into
    batches that respect the token budget. Overflow proteins (rep_len > token_budget)
    are always singleton batches.

    Batch plans are precomputed in a long-lived subprocess (ProcessPoolExecutor). The
    subprocess is initialised once with the large index arrays (flat_to_cluster and
    cluster_rep_len), which are pickled only at startup rather than on every plan
    submission. Within the subprocess, a ThreadPoolExecutor sorts and packs independent
    sortish chunks in parallel; numpy argsort releases the GIL so threads run truly in
    parallel for the sort step.

    Computed plans are stored in compact numpy format (flat_indices + batch_ends) to
    reduce the memory of the prefetch queue versus storing full Python list-of-lists.

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
        cluster_index: ClusterIndex exposing flat_to_cluster and cluster_rep_len.
        token_budget: Maximum cumulative representative length per batch.
        max_seq_len: Dataset truncation cap passed to the batch plan computation.
        chunk_multiplier: Sortish window width in multiples of avg proteins per batch.
        world_size: Number of DDP processes. Default 1 (single GPU).
        rank: This process's DDP rank. Default 0.
        seed: Base RNG seed; epoch offset is added per epoch for diversity.
        prefetch_epochs: Queue depth — how many future epochs to precompute. Default 2.
        n_threads: Threads used inside the worker subprocess for parallel chunk sort
            and pack. Defaults to all available logical CPUs.
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
        prefetch_epochs: int = 5,
        n_threads: int | None = None,
    ) -> None:
        self._cluster_index = cluster_index
        self._token_budget = token_budget
        self._max_seq_len = max_seq_len
        self._chunk_multiplier = chunk_multiplier
        self._world_size = world_size
        self._rank = rank
        self._seed = seed
        self._prefetch_epochs = prefetch_epochs
        self._n_threads = n_threads if n_threads is not None else (os.cpu_count() or 4)
        self._epoch = 0
        self._current_batches: BatchPlan | None = None

        self._executor = ProcessPoolExecutor(
            max_workers=1,
            mp_context=mp.get_context("spawn"),
            initializer=_worker_init,
            initargs=(
                cluster_index.flat_to_cluster,
                cluster_index.cluster_rep_len,
            ),
        )
        self._queue: queue.Queue[Future[BatchPlan]] = queue.Queue(maxsize=prefetch_epochs)

        for offset in range(prefetch_epochs):
            self._queue.put_nowait(self._submit(seed + offset))

    def _submit(self, seed: int) -> "Future[BatchPlan]":
        """Submit a batch plan computation to the worker subprocess.

        Args:
            seed: RNG seed for this epoch's batch plan.

        Returns:
            Future resolving to a compact BatchPlan (flat_indices, batch_ends).
        """
        return self._executor.submit(
            _compute_batch_plan_worker,
            len(self._cluster_index),
            self._token_budget,
            self._chunk_multiplier,
            seed,
            self._max_seq_len,
            self._n_threads,
        )

    def set_epoch(self, epoch: int) -> None:
        """Swap in the precomputed batch plan and enqueue the next epoch's plan.

        Must be called once per epoch before iterating. Blocks if the prefetch
        subprocess has not yet finished (rare — it should finish well before the
        previous epoch ends).

        Args:
            epoch: Current epoch number (0-indexed).
        """
        self._epoch = epoch
        self._current_batches = self._queue.get().result()
        self._queue.put_nowait(self._submit(self._seed + epoch + self._prefetch_epochs))

    def __iter__(self) -> Iterator[list[int]]:
        """Yield batches for this rank.

        Pads the batch list to a multiple of world_size (repeating leading batches)
        so every rank gets the same number of steps, then yields rank-strided.
        """
        assert self._current_batches is not None, "call set_epoch before iterating"
        flat, ends = self._current_batches
        n_batches = len(ends)
        remainder = n_batches % self._world_size
        n_padded = n_batches + (0 if remainder == 0 else self._world_size - remainder)

        for global_i in range(self._rank, n_padded, self._world_size):
            # Wrap padded indices back to existing batches (repeat leading batches).
            i = global_i if global_i < n_batches else global_i - n_batches
            start = int(ends[i - 1]) if i > 0 else 0
            yield flat[start : int(ends[i])].tolist()

    def __len__(self) -> int:
        """Return the number of batches this rank will yield.

        Returns an exact count after set_epoch has been called, or a rough
        estimate before the first call (safe for tqdm progress bars).
        """
        if self._current_batches is not None:
            _, ends = self._current_batches
            n_total = len(ends)
            remainder = n_total % self._world_size
            if remainder != 0:
                n_total += self._world_size - remainder
            return n_total // self._world_size
        n = len(self._cluster_index)
        n_regular = len(self._cluster_index.cluster_rep_len) - 1
        median_rep = int(self._cluster_index.cluster_rep_len[n_regular // 2])
        avg_per_batch = max(1, self._token_budget // median_rep)
        return math.ceil(math.ceil(n / avg_per_batch) / self._world_size)

    def __del__(self) -> None:
        """Shut down the executor without blocking process exit."""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)
