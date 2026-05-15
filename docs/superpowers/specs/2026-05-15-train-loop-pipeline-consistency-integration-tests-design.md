# Train Loop Pipeline Consistency Integration Tests — Design Spec

**Date:** 2026-05-15
**Status:** Approved

---

## Problem

The existing `test_train_loop.py` tests for `train()` and `train_ddp()` verify structural properties
only: the function returns `None`, a checkpoint file is created, some parameters changed, wandb
payloads have the right keys. No test verifies that the pipeline is *internally consistent* — that
the LR scheduler actually fires, that each loss component contributes a nonzero value, that epoch
metrics are averages and not sums, that the saved checkpoint reflects the model at the end of
training, or that the NaN-detection branch in `train_ddp` emits a warning.

The user suspects `train()` is broken in ways the existing tests cannot detect.

---

## Goal

Add 11 integration tests to a new file
`pallatom/tests/train/test_train_integration.py` that verify the full-loop pipeline consistency of
`train()` (6 tests) and `train_ddp()` (5 tests).

---

## File Changed

| File | Change |
|---|---|
| `pallatom/tests/train/test_train_loop.py` | Add 11 integration tests + 2 new fixtures at the bottom |

---

## Fixtures

All existing fixtures (`mini_batch`, `model`, `distogram_res`, `distogram_atom`, `loader`,
`tcfg`, `tcfg_multi`, `ddp_loader`, `_FakeDDP`, `_MockSampler`, `_patch_ddp`) are already defined
in `test_train_loop.py` and are reused directly.

### New fixtures added to `test_train_loop.py`

**`multi_loader`** — `DataLoader` backed by 3 identical copies of `mini_batch`, `batch_size=None`.
Used by the metrics-averaging tests (tests 4 and 11).

**`tcfg_save_every_1`** — already exists as `tcfg_save` in the file; reuse that name directly.

---

## `train()` Integration Tests (6 tests)

### 1. `test_train_lr_decreases_after_epoch`

**Purpose:** Verify that `CosineAnnealingLR.step()` is actually called each epoch.

**Setup:** Monkeypatch `torch.optim.lr_scheduler.CosineAnnealingLR` to wrap the real scheduler
and capture the LR at each `step()` call.

**Assertion:** The LR recorded after epoch 1 is strictly less than `tcfg.training.lr` (cosine
annealing always decreases on epoch 1 for `T_max > 1`). Use `tcfg_3ep` so `T_max=3`.

**Why this catches bugs:** If `scheduler.step()` is accidentally removed or never reached, the LR
stays flat and the assertion fails.

---

### 2. `test_train_loss_components_all_nonzero`

**Purpose:** Verify that no loss term is silently zeroed out by a mask, a wrong weight, or
a detached path.

**Setup:** Monkeypatch `train.train_loop.train_step` to record the returned `step_metrics` dict
on the first call, then delegate to the real `train_step`.

**Assertion:** For all 7 keys in the captured `step_metrics`, the value is `> 0`.

Keys checked: `"total loss"`, `"Kabsch aligned MSE loss"`, `"Cross Entropy loss"`,
`"smooth lddt"`, `"Residue Distogram loss"`, `"Atom Distogram loss"`, `"Intermediate loss"`.

---

### 3. `test_train_checkpoint_matches_final_model`

**Purpose:** Verify that the saved checkpoint reflects the actual model weights at the end of
training, not stale weights from an earlier epoch or a DDP wrapper.

**Setup:** Run `train()` for 1 epoch with `tcfg_1ep`. Load the checkpoint with
`torch.load(..., weights_only=True)`.

**Assertion:** For every key in `model.state_dict()`, the tensor in the loaded checkpoint equals
the tensor in the model (`torch.equal`).

---

### 4. `test_train_epoch_metrics_are_averages_not_sums`

**Purpose:** Verify that `avg_train[k] == mean(per_step_k)` for all 7 metric keys — catching
an accidental double-count, a missing normalisation, or an extra accumulation.

**Setup:** Use `multi_loader` (3 batches per epoch). Monkeypatch `train.train_loop.train_step`
to record the `step_metrics` dict from every call while delegating to the real function.

**Assertion:** For each key `k`, `abs(avg_val_captured[k] - mean(recorded[k])) < 1e-5`.

To capture `avg_train` the test monkeypatches `train.train_loop.log_epoch` to intercept the
`avg_train` argument on its first invocation.

---

### 5. `test_train_empty_train_loader_raises`

**Purpose:** Document the existing `ZeroDivisionError` when `train_loader` is empty
(line 536: `{k: v / n_batches ...}` with `n_batches == 0`).

**Setup:** Pass an empty `DataLoader` as `train_loader`.

**Assertion:** `pytest.raises(ZeroDivisionError)`.

**Note:** This is a documentation test, not a correctness test. It makes the crash explicit and
will become a regression test once the bug is fixed (assertion updated to `pytest.raises` →
check that the function returns without error).

---

### 6. `test_train_avg_val_has_total_loss_key`

**Purpose:** Verify that the dict returned by `evaluate()` and forwarded to `log_epoch` as
`avg_val` contains `"total loss"` — the key used by `log_epoch` to update `best_val_loss`.
A KeyError here would crash training silently in some error-handling configurations.

**Setup:** Monkeypatch `train.train_loop.log_epoch` to capture the `avg_val` argument.
Run `train()` for 1 epoch.

**Assertion:** `"total loss" in captured_avg_val`.

---

## `train_ddp()` Integration Tests (5 tests)

All tests use `_FakeDDP` (via the `_patch_ddp` autouse fixture) and `ddp_loader` with
`device="cpu"`.

### 7. `test_train_ddp_loss_components_all_nonzero`

Mirror of test 2 for `train_ddp`. Monkeypatch `train.train_loop.train_step` to record the first
`step_metrics`; assert all 7 values `> 0`.

---

### 8. `test_train_ddp_lr_decreases_after_epoch`

Mirror of test 1 for `train_ddp`. Use `tcfg_3ep`, capture LR at each `scheduler.step()` call,
assert LR after epoch 1 < `tcfg.training.lr`.

---

### 9. `test_train_ddp_checkpoint_unwraps_ddp_module`

**Purpose:** Verify that `log_epoch` correctly unwraps `ddp_model.module` before calling
`state_dict()`, so the checkpoint contains the inner model's weights (not the wrapper's).

**Setup:** Run `train_ddp()` for 1 epoch, rank=0. Load checkpoint.

**Assertion:** For every key in `model.state_dict()`, the tensor in the checkpoint equals
the tensor in the model (`torch.equal`). Uses the fact that `_FakeDDP.module` is the original
`model` fixture.

---

### 10. `test_train_ddp_nan_warning_emitted`

**Purpose:** Exercise the NaN-detection branch (lines 598–600 of `train_loop.py`):
```python
if rank == 0 and math.isnan(step_metrics["total loss"]):
    log.warning("nan_loss", ...)
```

**Setup:** Monkeypatch `train.train_loop.train_step` to return a `step_metrics` dict with
`"total loss": float("nan")` and zeros for all other keys, plus `grad_norm=0.0`.
Monkeypatch `train.train_loop.log` (the structlog logger) to record all `.warning(...)` calls.

**Assertion:** At least one `log.warning` call captured has event `"nan_loss"`.

---

### 11. `test_train_ddp_epoch_metrics_are_averages_not_sums`

Mirror of test 4 for `train_ddp`. Use `multi_loader` (3 batches), capture per-step metrics and
the `avg_train` passed to `log_epoch`, assert averages match.

---

## What is NOT tested

- Actual loss *decrease* over epochs (that is a learning-dynamics property, not a pipeline
  consistency property; the user requested consistency tests).
- Gradient coverage per submodule (covered by the companion
  `2026-05-15-gradient-flow-integration-tests-design.md` spec).
- Real DDP multi-process gradient synchronisation (requires torchrun; out of scope for CPU tests).
- W&B payload structure (already covered in `test_train_loop.py`).

---

## Conventions followed

Derived from `/workspaces/diffusion/CLAUDE.md` and `pyproject.toml`.

- Module-level `def test_*()` functions, no class grouping.
- `pytest` fixtures for all shared state.
- `torch.manual_seed(42)` at module level (already present in the file).
- No `einops` or `jaxtyping` needed (tests deal with control-flow, not tensor shapes).
- No new external dependencies.

### Docstrings

Every function (test, fixture, helper) must have a Google-style docstring. The `ANN` ruff waiver
for test files covers annotation rules only; `D` pydocstyle rules still apply. One-line
docstrings are fine for simple helpers and test functions.

### Type annotations

All fixture parameters and return types must be annotated. pyright `reportMissingParameterType`
fires for test files even though ruff `ANN` rules are waived. Pattern for test functions:

```python
def test_foo(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg: TrainConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One-line summary."""
    ...
```

Use fully-parameterised generics everywhere: `dict[str, float]`, `list[float]`,
`DataLoader[ProteinBatch]`, not bare `dict` / `list` / `DataLoader`.

### Import ordering

isort ordering: stdlib → third-party → first-party (no `pallatom.` prefix), alphabetical within
each group. New imports for the integration section go at the top of the existing import block.

### Line length

100 characters (Black-enforced). Wrap long argument lists at 100.
