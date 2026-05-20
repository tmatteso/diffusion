# Bucketed Token-Budget Batching Design

**Date:** 2026-05-20
**Status:** Approved

## Problem

The current `ProteinDataset` pads every protein to a fixed `max_seq_length=128` at item-load time.
This has two compounding costs:

1. **Truncation**: proteins longer than 128 residues lose data silently.
2. **Padding waste**: short proteins are padded to 128, so a batch of 50-residue proteins is
   ~75% padding. Pack rates are low and GPU compute is wasted on zeros.

The goal is a data system that fills every batch to a configurable **token budget** (512 residues)
with near-zero padding, while remaining DDP-compatible and measurable via explicit efficiency metrics.

---

## Architecture

```
ClusterIndex  (pallatom/helpers/cluster_index.py)
    ↓
ClusteredProteinDataset  (pallatom/helpers/data.py)
    ↓
BucketedBatchSampler  (pallatom/helpers/bucketed_sampler.py)
    ↓
DataLoader  (batch_sampler=..., collate_fn=_to_protein_batch_dynamic)
```

Val/test splits are unchanged: `ProteinDataset` + fixed `batch_size` + `_to_protein_batch`.

---

## Section 1: Metrics (implement first as baseline)

Three scalars added to the `dict[str, float]` returned by `train_step` in
`pallatom/train/train_loop.py`:

| Key | Formula | Source tensors |
|---|---|---|
| `pack_rate` | `residue_mask.sum() / (B × N_res)` | `featurized_batch.residue_mask` |
| `residues_per_sec` | `residue_mask.sum() / step_time` | timer + above |
| `atoms_per_sec` | `atom5_mask.sum() / step_time` | `featurized_batch.atom5_mask` |

`step_time` is measured with `time.perf_counter()` bracketing the model forward + backward pass.
These keys flow automatically into the existing W&B and structlog pipeline.

Implementing metrics **before** the bucketing system gives a baseline pack rate against which
the improvement can be measured.

---

## Section 2: Cluster boundaries and `ClusterIndex`

### Cluster assignment

```
Cluster k  (k = 0..63):  lengths [8k+1 .. 8(k+1)]   rep_len = 8(k+1)
Cluster 64 (overflow):   lengths > 512               rep_len = token_budget + 1, always singleton batch
```

64 bins × width 8 = 512 — covers the full token budget with uniform resolution.
Proteins longer than 512 residues are **truncated to 512** and always batched alone
(pack\_rate = 1.0 for those batches).

### `ClusterIndex`

**File:** `pallatom/helpers/cluster_index.py`

**Constructor** `ClusterIndex(jsonl_path, names, token_budget=512, n_clusters=64)`:

1. Derive `cluster_dir = jsonl_path.parent / (jsonl_path.stem + "_clusters")`
2. Check for `cluster_000.jsonl` … `cluster_064.jsonl` (65 files)
3. **Cache miss** — scan source JSONL by name-filter, assign each entry to cluster
   `(len(seq) - 1) // 8` (clamped to 64 for overflow), write per-cluster JSONLs,
   build byte-offset arrays.
4. **Cache hit** — scan each cluster JSONL to rebuild offset arrays (no protein parsing).

**Exposed attributes** (all indexed by flat global index `i`):

| Attribute | Type | Meaning |
|---|---|---|
| `flat_to_cluster` | `list[int]` | cluster id (0–64) for protein `i` |
| `flat_to_local` | `list[int]` | within-cluster row index for protein `i` |
| `cluster_rep_len` | `list[int]` | representative length for cluster `k` |
| `cluster_offsets[k]` | `list[int]` | byte offsets into `cluster_k.jsonl` |
| `cluster_file(k)` | `Path` | path to `cluster_k.jsonl` |
| `__len__` | `int` | total number of proteins |

Pickling: file handles excluded; cluster files re-opened lazily in DataLoader workers
(same pattern as existing `ProteinDataset`).

---

## Section 3: `ClusteredProteinDataset`

**File:** `pallatom/helpers/data.py`

Replaces `ProteinDataset` for the training split. Same flat-integer `__getitem__` API.

```python
ClusteredProteinDataset(
    jsonl_path: str | Path,
    names: list[str],
    token_budget: int = 512,
    n_clusters: int = 64,
)
```

**`__getitem__(idx)`**:
1. Look up `(cluster_id, local_idx)` from `ClusterIndex`
2. Seek to `cluster_offsets[cluster_id][local_idx]` in the cluster file
3. Call `make_np_example` + `center_positions` (unchanged)
4. Call `truncate_to_length(np_example, token_budget)` — **truncate only, no padding**
5. Return `{atom_positions, atom_mask, residue_index, seq}` at actual protein length

Items returned by `__getitem__` are **variable-length** — `atom_positions.shape[0]` is the true
residue count (≤ `token_budget`). Padding is deferred to the collate function.

**New helper** `truncate_to_length(np_example, max_length)` in `pallatom/helpers/atom_utils.py`:
identical to the `pad < 0` branch of `make_fixed_size`, without the padding branch.
`make_fixed_size` is kept unchanged for backward compatibility.

**Exposed for sampler:**
```python
dataset.cluster_index: ClusterIndex
```

One `io.BufferedReader | None` per cluster file (65 handles); all excluded from pickle.

---

## Section 4: `BucketedBatchSampler` with process-pool prefetch queue

**File:** `pallatom/helpers/bucketed_sampler.py`

A `torch.utils.data.Sampler[list[int]]` — each yielded item is a `list[int]` of flat dataset
indices constituting one full batch.

### `_compute_batch_plan` (module-level, picklable)

```
inputs:
    flat_to_cluster: list[int]
    cluster_rep_len: list[int]
    n_proteins: int
    token_budget: int
    chunk_multiplier: int
    seed: int   ← seed + epoch, gives per-epoch shuffle

algorithm:
    1. Shuffle all N indices with random.Random(seed)
    2. Chunk into windows of chunk_size indices, where
       chunk_size = chunk_multiplier × (token_budget // median_rep_len)
       median_rep_len = cluster_rep_len[n_clusters // 2]  (computed once at __init__)
       (chunk_multiplier=16 → ~16 full batches of diversity before sorting kicks in)
    3. Within each chunk: sort by cluster_rep_len[flat_to_cluster[i]]
    4. Greedy pack:
         current_batch = [], current_budget = 0
         for i in sorted_chunk:
             rep_len = cluster_rep_len[flat_to_cluster[i]]
             if rep_len > token_budget:          # overflow → singleton
                 flush current_batch; yield [i]
             elif current_budget + rep_len > token_budget:
                 yield current_batch
                 current_batch, current_budget = [i], rep_len
             else:
                 current_batch.append(i); current_budget += rep_len
         flush final current_batch

output: list[list[int]]
```

Pure Python lists in/out — fully picklable, no torch or file-handle dependencies.

### `BucketedBatchSampler`

```python
BucketedBatchSampler(
    cluster_index: ClusterIndex,
    token_budget: int,
    chunk_multiplier: int = 16,
    world_size: int = 1,
    rank: int = 0,
    seed: int = 0,
    prefetch_epochs: int = 2,
)
```

**Internal state:**
```
_executor: ProcessPoolExecutor(max_workers=1, mp_context=spawn)
_queue:    Queue[Future[list[list[int]]]](maxsize=prefetch_epochs)
_current_batches: list[list[int]]
```

**`__init__`** pre-fills the queue:
```python
for offset in range(prefetch_epochs):
    _queue.put_nowait(executor.submit(_compute_batch_plan, ..., seed=seed+offset))
```

**`set_epoch(epoch)`** (called by training loop before each epoch):
```python
self._current_batches = self._queue.get().result()   # blocks if not ready (rare)
self._queue.put_nowait(executor.submit(              # keep pipeline full
    _compute_batch_plan, ..., seed=seed + epoch + prefetch_epochs
))
```

**`__iter__`**: pads `_current_batches` to a multiple of `world_size` (repeat first batch),
then yields `_current_batches[rank::world_size]`.

**`__len__`**: returns `len(_current_batches) // world_size` once `set_epoch` has been called;
before that returns a rough estimate based on average proteins per budget.

**`__del__`**: `executor.shutdown(wait=False)` — non-blocking on process exit.

**Why `spawn` + 1 worker?** Batch planning is pure Python list arithmetic (milliseconds per epoch);
one worker suffices. `spawn` avoids CUDA-context fork corruption.

---

## Section 5: Dynamic collate and factory functions

### `_to_protein_batch_dynamic`

```python
def _to_protein_batch_dynamic(samples):
    max_len = max(s["atom_positions"].shape[0] for s in samples)
    # pad each sample to max_len, stack into ProteinBatch
```

Within a bucket all proteins are within 8 residues of each other, so padding is near-zero.
`ProteinBatch` shape annotations (`"B N_res 37 3"` etc.) remain valid; `N_res` varies
batch-to-batch but is uniform within a batch.

### Factory functions

Added to `pallatom/helpers/data.py`:

```python
def make_bucketed_data_loaders(
    *, cfg, jsonl_path, splits_path, num_workers, debug_run
) -> tuple[DataLoader, DataLoader, DataLoader]

def make_ddp_bucketed_data_loaders(
    cfg, jsonl_path, splits_path, rank, world_size, num_workers
) -> tuple[DataLoader, DataLoader, DataLoader]
```

- **Train**: `ClusteredProteinDataset` + `BucketedBatchSampler` +
  `DataLoader(batch_sampler=..., collate_fn=_to_protein_batch_dynamic)`
- **Val/test**: unchanged — `ProteinDataset` + fixed `batch_size` + `_to_protein_batch`

`DataLoader` receives `batch_sampler=` (not `batch_size=` + `sampler=`) — standard PyTorch
pattern for custom batch samplers.

### Config change

`LoaderConfig` in `pallatom/train/train_config.py` gains:
```python
token_budget: int = Field(default=512, gt=0)
```

`batch_size` and `max_seq_length` are kept for val/test. No breaking change to existing configs.

---

## Section 6: Training loop integration

**File:** `pallatom/train/train_loop.py`

### Metrics in `train_step`

```python
import time

t0 = time.perf_counter()
(r_denoised, ...) = model(featurized_batch)
# losses, backward, optimizer.step()
t1 = time.perf_counter()
step_time = t1 - t0

B, N_res = featurized_batch.residue_mask.shape
actual_residues = int(featurized_batch.residue_mask.sum().item())
actual_atoms    = int(featurized_batch.atom5_mask.sum().item())

return {
    ...,   # existing loss keys unchanged
    "pack_rate":        actual_residues / (B * N_res),
    "residues_per_sec": actual_residues / step_time,
    "atoms_per_sec":    actual_atoms    / step_time,
}, grad_norm
```

### Loader swap and `set_epoch`

```python
# replace make_ddp_data_loaders → make_ddp_bucketed_data_loaders
train_loader, val_loader, _ = make_ddp_bucketed_data_loaders(...)

# existing set_epoch call; cast changes to BucketedBatchSampler
cast(BucketedBatchSampler, train_loader.batch_sampler).set_epoch(epoch)
```

`set_epoch` pops the precomputed plan from the queue and enqueues the next epoch's
computation — one call drives the whole prefetch pipeline.

---

## Files changed

| File | Status | Notes |
|---|---|---|
| `pallatom/helpers/cluster_index.py` | **NEW** | `ClusterIndex` |
| `pallatom/helpers/bucketed_sampler.py` | **NEW** | `BucketedBatchSampler`, `_compute_batch_plan` |
| `pallatom/helpers/data.py` | **MOD** | `ClusteredProteinDataset`, `_to_protein_batch_dynamic`, new factory fns |
| `pallatom/helpers/atom_utils.py` | **MOD** | `truncate_to_length` helper |
| `pallatom/train/train_config.py` | **MOD** | `token_budget` field on `LoaderConfig` |
| `pallatom/train/train_loop.py` | **MOD** | timing, pack metrics, loader swap |
| `pallatom/tests/helpers/test_cluster_index.py` | **NEW** | ClusterIndex tests |
| `pallatom/tests/helpers/test_bucketed_sampler.py` | **NEW** | BucketedBatchSampler tests |
| `pallatom/tests/helpers/test_data.py` | **MOD** | ClusteredProteinDataset tests |
