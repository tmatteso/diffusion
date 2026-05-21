# Token-Budget Gradient Accumulation Design

**Date:** 2026-05-21
**Status:** Approved
**Supersedes:** `2026-05-20-gradient-accumulation-design.md`

## Overview

The existing gradient accumulation uses `accum_steps = accumulated_batch_size // batch_size`,
but `BucketedBatchSampler` yields heterogeneous batch sizes (greedy token-budget packing), so
`batch_size` is meaningless as a divisor. This design replaces the fixed-step logic with
runtime token counting: accumulate micro-batches until their total (non-padded) residue count
reaches `accumulated_token_budget`, then flush. The grad_scale is also fixed to weight each
micro-batch by its protein count, since all losses are already `.mean()`'d over the batch
dimension.

## Section 1: Config

### `TrainingParams` — rename field

```python
# Before
accumulated_batch_size: int = Field(default=32, gt=0)

# After
accumulated_token_budget: int = Field(default=2048, gt=0)
```

The old default of 32 was in *proteins*. The new default should be set in *residues* (tokens).
A sensible starting point is `token_budget * desired_accum_factor`; e.g., `token_budget=512`
and 4 micro-steps → `accumulated_token_budget=2048`.

### `TrainConfig` — remove validator

Drop `_accumulated_batch_size_gte_loader_batch_size`. It compared against `train_loader.batch_size`,
which the token-budget sampler ignores entirely.

## Section 2: `_process_accum_window` changes

### Signature

```python
def _process_accum_window(
    micro_buffer: list[ProteinBatch],
    n_proteins_per_batch: list[int],          # NEW: protein count per micro-batch
    model: nn.Module,
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    device: str,
) -> dict[str, float]:
```

`n_proteins_per_batch[i]` is `micro_buffer[i].atom_positions.shape[0]`, computed cheaply at
collection time.

### Grad scale and metric weighting

```python
total_proteins = sum(n_proteins_per_batch)
# For micro-batch i with n_proteins_i proteins:
grad_scale = total_proteins / n_proteins_i        # each micro-batch's mean loss weighted by n_i/N
weight     = n_proteins_i / total_proteins        # same weight for metric averaging
```

**Why protein count, not token count?** All losses call `.mean()` over the batch (B) dimension,
so each micro-batch's loss is already a per-protein average. Weighting by `n_i / N` recovers
the gradient of the mean over all N proteins in the window — the correct analogue of a single
large-batch step.

## Section 3: Accumulation loop

Both `train()` and `train_ddp()` use the same structure.

### State variables (per epoch)

```python
accum_tokens:      int = 0
micro_buffer:      list[ProteinBatch] = []
n_proteins_buffer: list[int] = []
```

### Per-batch logic

```python
for batch in pbar:
    n_tokens   = int(batch.atom_mask.any(dim=-1).sum().item())
    n_proteins = batch.atom_positions.shape[0]

    # Pre-flush: if adding this batch would push tokens over the threshold, flush first.
    # This prevents overshoot; a partially-filled window is flushed clean before
    # a new window starts with the current batch.
    if micro_buffer and accum_tokens + n_tokens > per_rank_token_budget:
        window_metrics = _process_accum_window(
            micro_buffer, n_proteins_buffer, model, ...
        )
        # ... grad clip, optimizer.step(), optimizer.zero_grad()
        micro_buffer, n_proteins_buffer, accum_tokens = [], [], 0
        global_step += 1

    micro_buffer.append(batch)
    n_proteins_buffer.append(n_proteins)
    accum_tokens += n_tokens
```

### Per-rank token budget

| Context | Formula |
|---------|---------|
| Single-GPU `train()` | `per_rank_token_budget = tp.accumulated_token_budget` |
| DDP `train_ddp()` | `per_rank_token_budget = tp.accumulated_token_budget // world_size` |

Each rank accumulates `accumulated_token_budget / world_size` tokens; globally all ranks
together see `accumulated_token_budget` tokens per optimizer step.

### Edge cases

- **Singleton oversized batch** (`n_tokens > per_rank_token_budget` on its own): the batch is
  collected into an empty buffer. The next batch triggers a pre-flush of this singleton.
  If it is the last batch of the epoch, it falls into the partial-window path (see below).
- **Epoch-end partial window**: unchanged — `if micro_buffer: log.warning(...)` at epoch end.
- **`no_sync()` context**: unchanged — `_process_accum_window` still gates `no_sync()` on all
  but the last micro-batch.

## Section 4: Logging

Replace the startup log line:

```python
# Before
log.info("gradient_accumulation", accum_steps=accum_steps, effective_batch_size=tp.accumulated_batch_size)

# After
log.info("gradient_accumulation", token_budget_per_rank=per_rank_token_budget,
         global_token_budget=tp.accumulated_token_budget)
```

## Section 5: Tests

### Updated fixtures

All `TrainingParams(...)` fixture calls that relied on `accumulated_batch_size` must switch to
`accumulated_token_budget`. Set it low enough (e.g., equal to `token_budget`) so the test
loaders trigger at least one optimizer step per epoch.

### Updated test: `test_train_config_accumulated_batch_size_validator`

Rename/replace with a test that confirms the old validator is gone and `TrainConfig` now
accepts any `accumulated_token_budget > 0`.

### New test: `test_process_accum_window_protein_weighted_grad_scale`

Construct two micro-batches with different `B` dimensions. Call `_process_accum_window`
with matching `n_proteins_per_batch`. Verify that `grad_scale` passed to each
`train_step` call is `total_proteins / n_i` (use a mock or monkeypatch `train_step`).

### New test: `test_train_token_budget_flush_fires_before_overshoot`

Run `train()` with a 3-batch loader where batch 1+2 exceed the token budget together.
Verify that the optimizer step fires after batch 2 (not batch 3), i.e., the pre-flush
prevents overshoot.
