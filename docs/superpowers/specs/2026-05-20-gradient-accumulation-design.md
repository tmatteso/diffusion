# Gradient Accumulation Design

**Date:** 2026-05-20
**Status:** Approved

## Overview

Add gradient accumulation to the training loop so large effective batch sizes can be achieved on hardware with limited per-GPU memory. The config expresses the target **global effective batch size** (`accumulated_batch_size = 32` by default); the number of micro-steps is derived automatically at runtime from `batch_size` and `world_size`.

## Section 1: Config

### `TrainingParams` — new field

```python
accumulated_batch_size: int = Field(default=32, gt=0)
```

Default `32` with the existing `batch_size=2` default produces `16` micro-steps on a single GPU.

### `TrainConfig` — cross-config validator

A `@model_validator(mode="after")` on `TrainConfig` validates:

```
training.accumulated_batch_size >= train_loader.batch_size
```

Raises `ValueError` if not satisfied — you cannot accumulate to fewer samples than one micro-batch.

### Derived `accum_steps` at runtime

| Context | Formula |
|---------|---------|
| Single-GPU `train()` | `max(1, accumulated_batch_size // train_loader.batch_size)` |
| DDP `train_ddp()` | `max(1, accumulated_batch_size // (train_loader.batch_size * world_size))` |

`accum_steps` is logged at rank-0 startup so users can verify the effective batch math.

## Section 2: `train_step` refactor

`train_step` becomes a pure **forward + backward** function; the outer loop owns all optimizer state transitions.

### Signature change

**Before:**
```python
def train_step(
    batch, model, tcfg, distogram_res, distogram_atom,
    optimizer,          # removed
    device,
) -> tuple[dict[str, float], float]:   # second element was grad_norm
```

**After:**
```python
def train_step(
    batch, model, tcfg, distogram_res, distogram_atom,
    device,
    grad_scale: float = 1.0,   # new: divide loss by this before .backward()
) -> dict[str, float]:          # grad_norm removed from return
```

### Behavior change

| Responsibility | Before | After |
|----------------|--------|-------|
| `optimizer.zero_grad()` | inside `train_step` | outer loop |
| `loss.backward()` | `total_loss.backward()` | `(total_loss / grad_scale).backward()` |
| `clip_grad_norm_` | inside `train_step` | outer loop |
| `optimizer.step()` | inside `train_step` | outer loop |

### Return value

Return dict keys are **unchanged** — `pack_rate`, `residues_per_sec`, and `atoms_per_sec` are measured per micro-step and averaged across the window at the epoch level. `grad_norm` is no longer returned; it is computed in the outer loop after all micro-steps.

## Section 3: Outer loop changes

Both `train()` and `train_ddp()` adopt the same accumulation pattern.

### Loop structure

```
optimizer.zero_grad()
micro_buffer = []

for batch in loader:
    micro_buffer.append(batch)
    if len(micro_buffer) < accum_steps:
        continue                          # keep collecting

    # --- full accumulation window ---
    for micro_idx, mb in enumerate(micro_buffer):
        is_last = (micro_idx == accum_steps - 1)
        ctx = model.no_sync() if (not is_last and hasattr(model, 'no_sync'))
              else contextlib.nullcontext()
        with ctx:
            step_metrics = train_step(mb, ..., grad_scale=accum_steps)
        accumulate step_metrics into window_metrics (averaging losses, summing throughput counts)

    grad_norm = clip_grad_norm_(model.parameters(), tp.grad_clip or inf)
    optimizer.step()
    optimizer.zero_grad()

    update epoch_metrics, n_batches, global_step
    clear micro_buffer

# end-of-epoch: drop partial window (< accum_steps batches), log warning on rank-0 if any dropped
```

### Key invariants

- `model.no_sync()` is gated on `hasattr(model, 'no_sync')` — the `_FakeDDP` used in tests does not implement it, so tests continue to work without modification beyond the `optimizer` argument removal.
- `global_step` counts **optimizer steps** (one weight update = one step). This is a semantic improvement over the previous "loader batches seen" count.
- Epoch-level averaging: loss metrics are divided by `accum_steps` per window then averaged across windows; throughput metrics (`residues_per_sec`, `atoms_per_sec`) sum residue/atom counts and elapsed time across all micro-steps in the window.
- Partial windows at epoch end are silently dropped; rank-0 logs a `structlog` warning with the count of dropped batches when `dropped > 0`.

## Section 4: Tests

### Updated existing tests (no new fixtures needed)

| Test | Change |
|------|--------|
| `test_train_step_returns_expected_keys` | Remove `optimizer` arg from call; `train_step` now returns `dict` not `tuple` |
| `test_train_step_pack_rate_in_range` | Same — remove `optimizer` |
| `test_train_step_throughput_metrics_positive` | Same — remove `optimizer` |
| `test_integration_gradient_flow_via_train_step` | Remove `optimizer` arg; remove `grad_norm` destructuring — backward is still called so gradients still exist |

### New tests

**`test_train_config_accumulated_batch_size_validator`**
Constructing `TrainConfig` with `training.accumulated_batch_size < train_loader.batch_size` raises `ValidationError`.

**`test_train_step_grad_scale_reduces_gradient_magnitude`**
Call `train_step` twice on the same batch (same seed, `zero_grad` between calls): once with `grad_scale=1.0`, once with `grad_scale=2.0`. The grad L2 norm with `grad_scale=2.0` is approximately half the norm with `grad_scale=1.0`.

**`test_train_accumulation_updates_params_after_full_window`**
Use a 2-batch loader and `tcfg` with `training.accumulated_batch_size = 2 * train_loader.batch_size` (so `accum_steps=2`). Capture params before training, run one epoch, assert at least one parameter changed — verifying the optimizer step fires correctly after both micro-steps.

**`test_train_ddp_accumulation_updates_params`**
Same as above but via `train_ddp` with `rank=0, world_size=1`.
