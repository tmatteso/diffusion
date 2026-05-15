# Train Loop Pipeline Consistency Integration Tests — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 11 integration tests to `pallatom/tests/train/test_train_loop.py` that verify the
full-loop pipeline consistency of `train()` (6 tests) and `train_ddp()` (5 tests).

**Architecture:** All tests are appended to the existing `test_train_loop.py` file — no new files.
Two new fixtures (`multi_loader`, `multi_ddp_loader`) are added above the integration test
section. Each test monkeypatches internal collaborators to capture state without changing
production behaviour, then asserts correctness properties.

**Tech Stack:** pytest, torch, structlog, `unittest.mock.MagicMock`, `torch.optim.lr_scheduler`,
existing project fixtures (`_ListDataset`, `_FakeDDP`, `_MockSampler`).

---

## File Modified

- `pallatom/tests/train/test_train_loop.py` (currently 1327 lines)
  - **Extend imports** (lines 14–32): add `CosineAnnealingLR`; add `log_epoch` to `train_loop` import
  - **Add 2 fixtures** after line 1327 (end of file)
  - **Add 11 integration tests** after the fixtures

---

## Quick Reference — Existing Fixtures Reused

| Fixture | Type | What it provides |
|---|---|---|
| `model` | `MainTrunk` | Small `MainTrunk` (N_KEEP=16, K_UNIT=1) |
| `loader` | `DataLoader[ProteinBatch]` | 1 mini-batch, sequential sampler |
| `tcfg` | `TrainConfig` | 1 epoch, save_every=100, use_wandb=False |
| `tcfg_multi` | `TrainConfig` | 3 epochs, save_every=100, use_wandb=False |
| `tcfg_save` | `TrainConfig` | 1 epoch, save_every=1 |
| `ddp_loader` | `DataLoader[ProteinBatch]` | 1 mini-batch, `_MockSampler` |
| `distogram_res` | `Distogram` | Residue distogram |
| `distogram_atom` | `Distogram` | Atom distogram |

`_patch_ddp` is `autouse=True` — it replaces `DDP` with `_FakeDDP` for all tests in the file
(including these new ones). `_FakeDDP.module` is the original `model`, so checkpoint assertions
work correctly.

---

## Task 1 — Extend imports and add new fixtures

**Files:**
- Modify: `pallatom/tests/train/test_train_loop.py:14-32`

- [ ] **Step 1.1 — Add `CosineAnnealingLR` import and extend `train_loop` import**

  Replace lines 14–32 in `test_train_loop.py`:

  ```python
  import torch.distributed as dist_module
  import torch.nn as nn
  import torch.nn.parallel
  from architecture.main_trunk import MainTrunk
  from helpers.data import _to_protein_batch
  from helpers.featurize import Distogram, ProteinBatch, apply_conditioning_dropout, featurize_batch
  from torch.optim import Adam
  from torch.optim.lr_scheduler import CosineAnnealingLR
  from train.train_config import (
      CheckpointParams,
      LoggingParams,
      ModelParams,
      TrainConfig,
      TrainingParams,
  )
  from train.train_loop import (
      evaluate,
      evaluate_ddp,
      log_epoch,
      train,
      train_ddp,
      train_step,
  )
  ```

  Note: `Adam` and `train_step` are already present; this adds `CosineAnnealingLR` and `log_epoch`.

- [ ] **Step 1.2 — Add `multi_loader` and `multi_ddp_loader` fixtures at end of file**

  Append to the end of `test_train_loop.py` (after line 1327):

  ```python
  # ---------------------------------------------------------------------------
  # Fixtures for integration tests
  # ---------------------------------------------------------------------------


  @pytest.fixture
  def multi_loader(
      mini_batch: Mapping[str, torch.Tensor | list[str]],
  ) -> torch.utils.data.DataLoader[ProteinBatch]:
      """DataLoader with 3 identical mini-batches for epoch-metrics averaging tests."""
      return torch.utils.data.DataLoader(
          _ListDataset([mini_batch, mini_batch, mini_batch]), batch_size=None
      )


  @pytest.fixture
  def multi_ddp_loader(
      mini_batch: Mapping[str, torch.Tensor | list[str]],
  ) -> torch.utils.data.DataLoader[ProteinBatch]:
      """DataLoader with 3 batches and a MockSampler for train_ddp averaging tests."""
      dataset = _ListDataset([mini_batch, mini_batch, mini_batch])
      sampler = _MockSampler([mini_batch, mini_batch, mini_batch])
      return torch.utils.data.DataLoader(dataset, batch_size=None, sampler=sampler)
  ```

- [ ] **Step 1.3 — Run the full test suite to confirm nothing broken**

  ```bash
  cd /workspaces/diffusion
  python -m pytest pallatom/tests/train/test_train_loop.py -v --tb=short 2>&1 | tail -20
  ```

  Expected: all existing tests pass; the two new fixtures are collected but unused (no error).

- [ ] **Step 1.4 — Commit**

  ```bash
  git add pallatom/tests/train/test_train_loop.py
  git commit -m "test: add imports and fixtures for train loop integration tests"
  ```

---

## Task 2 — LR scheduler fires each epoch

**Files:**
- Modify: `pallatom/tests/train/test_train_loop.py` (append)

- [ ] **Step 2.1 — Append two LR-scheduler tests**

  Append to `test_train_loop.py`:

  ```python
  # ---------------------------------------------------------------------------
  # Integration — LR scheduler fires
  # ---------------------------------------------------------------------------


  def test_train_lr_decreases_after_epoch(
      model: MainTrunk,
      loader: torch.utils.data.DataLoader[ProteinBatch],
      tcfg_multi: TrainConfig,
      distogram_res: Distogram,
      distogram_atom: Distogram,
      monkeypatch: pytest.MonkeyPatch,
  ) -> None:
      """CosineAnnealingLR.step() is called each epoch and reduces the LR."""
      lrs: list[float] = []
      _real_step: Any = CosineAnnealingLR.step

      def _capturing_step(self: CosineAnnealingLR) -> None:
          _real_step(self)
          lrs.append(self.get_last_lr()[0])

      monkeypatch.setattr(CosineAnnealingLR, "step", _capturing_step)
      train(model, tcfg_multi, loader, loader, distogram_res, distogram_atom, "cpu")
      assert len(lrs) == tcfg_multi.training.num_epochs
      assert lrs[0] < tcfg_multi.training.lr


  def test_train_ddp_lr_decreases_after_epoch(
      model: MainTrunk,
      ddp_loader: torch.utils.data.DataLoader[ProteinBatch],
      tcfg_multi: TrainConfig,
      distogram_res: Distogram,
      distogram_atom: Distogram,
      monkeypatch: pytest.MonkeyPatch,
  ) -> None:
      """CosineAnnealingLR.step() is called each epoch in train_ddp and reduces the LR."""
      lrs: list[float] = []
      _real_step: Any = CosineAnnealingLR.step

      def _capturing_step(self: CosineAnnealingLR) -> None:
          _real_step(self)
          lrs.append(self.get_last_lr()[0])

      monkeypatch.setattr(CosineAnnealingLR, "step", _capturing_step)
      train_ddp(
          0, 0, 1, model, tcfg_multi, ddp_loader, ddp_loader,
          distogram_res, distogram_atom, device="cpu",
      )
      assert len(lrs) == tcfg_multi.training.num_epochs
      assert lrs[0] < tcfg_multi.training.lr
  ```

- [ ] **Step 2.2 — Run these two tests**

  ```bash
  python -m pytest pallatom/tests/train/test_train_loop.py \
      -k "test_train_lr_decreases_after_epoch or test_train_ddp_lr_decreases_after_epoch" \
      -v --tb=short
  ```

  **If both PASS:** the scheduler fires correctly. Proceed to step 2.3.

  **If a test FAILS with `AssertionError: 0 == 3` (lrs is empty):** `scheduler.step()` is never
  called. In `train_loop.py`, check that `scheduler.step()` is present inside the epoch loop
  (around line 534 for `train()`, line 609 for `train_ddp()`).

  **If a test FAILS with `AssertionError: lrs[0] >= tcfg_multi.training.lr`:** `step()` is called
  but LR doesn't change. This would indicate a wrong `T_max` being passed to the scheduler.

- [ ] **Step 2.3 — Commit**

  ```bash
  git add pallatom/tests/train/test_train_loop.py
  git commit -m "test(integration): LR scheduler fires each epoch in train and train_ddp"
  ```

---

## Task 3 — All 7 loss components are nonzero

**Files:**
- Modify: `pallatom/tests/train/test_train_loop.py` (append)

- [ ] **Step 3.1 — Append two loss-components tests**

  Append to `test_train_loop.py`:

  ```python
  # ---------------------------------------------------------------------------
  # Integration — all loss components are nonzero
  # ---------------------------------------------------------------------------

  _LOSS_KEYS = [
      "total loss",
      "Kabsch aligned MSE loss",
      "Cross Entropy loss",
      "smooth lddt",
      "Residue Distogram loss",
      "Atom Distogram loss",
      "Intermediate loss",
  ]


  def test_train_loss_components_all_nonzero(
      model: MainTrunk,
      loader: torch.utils.data.DataLoader[ProteinBatch],
      tcfg: TrainConfig,
      distogram_res: Distogram,
      distogram_atom: Distogram,
      monkeypatch: pytest.MonkeyPatch,
  ) -> None:
      """All 7 step-level loss components are positive in the first training step."""
      captured: list[dict[str, float]] = []
      _real_train_step = train_step

      def _patched(
          batch: ProteinBatch,
          m: nn.Module,
          cfg: TrainConfig,
          dr: Distogram,
          da: Distogram,
          opt: Adam,
          dev: str,
      ) -> tuple[dict[str, float], float]:
          result = _real_train_step(batch, m, cfg, dr, da, opt, dev)
          if not captured:
              captured.append(result[0])
          return result

      monkeypatch.setattr("train.train_loop.train_step", _patched)
      train(model, tcfg, loader, loader, distogram_res, distogram_atom, "cpu")

      assert captured, "train_step was never called"
      for key in _LOSS_KEYS:
          assert captured[0][key] > 0, f"'{key}' is not positive: {captured[0][key]}"


  def test_train_ddp_loss_components_all_nonzero(
      model: MainTrunk,
      ddp_loader: torch.utils.data.DataLoader[ProteinBatch],
      tcfg: TrainConfig,
      distogram_res: Distogram,
      distogram_atom: Distogram,
      monkeypatch: pytest.MonkeyPatch,
  ) -> None:
      """All 7 step-level loss components are positive in the first train_ddp step."""
      captured: list[dict[str, float]] = []
      _real_train_step = train_step

      def _patched(
          batch: ProteinBatch,
          m: nn.Module,
          cfg: TrainConfig,
          dr: Distogram,
          da: Distogram,
          opt: Adam,
          dev: str,
      ) -> tuple[dict[str, float], float]:
          result = _real_train_step(batch, m, cfg, dr, da, opt, dev)
          if not captured:
              captured.append(result[0])
          return result

      monkeypatch.setattr("train.train_loop.train_step", _patched)
      train_ddp(
          0, 0, 1, model, tcfg, ddp_loader, ddp_loader,
          distogram_res, distogram_atom, device="cpu",
      )

      assert captured, "train_step was never called"
      for key in _LOSS_KEYS:
          assert captured[0][key] > 0, f"'{key}' is not positive: {captured[0][key]}"
  ```

- [ ] **Step 3.2 — Run these two tests**

  ```bash
  python -m pytest pallatom/tests/train/test_train_loop.py \
      -k "test_train_loss_components_all_nonzero or test_train_ddp_loss_components_all_nonzero" \
      -v --tb=short
  ```

  **If a test FAILS with `assert 0.0 > 0`:** a loss component is zeroed out. The key in the
  assertion message identifies which one. In `train_loop.py::train_step` (lines 278–389), inspect
  the weight (`lp.lam`, `lp.alpha_0`…`lp.alpha_4`) for that component. Also check that the
  intermediate stack (`intermediate_denoised_coord_stack`) has length > 0 — if empty, intermediate
  loss is 0. `K_unit=1` in the test fixtures means it should have exactly 1 element.

- [ ] **Step 3.3 — Commit**

  ```bash
  git add pallatom/tests/train/test_train_loop.py
  git commit -m "test(integration): all 7 loss components are nonzero in first training step"
  ```

---

## Task 4 — Checkpoint reflects final model weights

**Files:**
- Modify: `pallatom/tests/train/test_train_loop.py` (append)

- [ ] **Step 4.1 — Append two checkpoint-accuracy tests**

  Append to `test_train_loop.py`:

  ```python
  # ---------------------------------------------------------------------------
  # Integration — checkpoint reflects final model weights
  # ---------------------------------------------------------------------------


  def test_train_checkpoint_matches_final_model(
      model: MainTrunk,
      loader: torch.utils.data.DataLoader[ProteinBatch],
      tcfg: TrainConfig,
      distogram_res: Distogram,
      distogram_atom: Distogram,
  ) -> None:
      """Checkpoint saved by train() contains the model's exact final weights."""
      train(model, tcfg, loader, loader, distogram_res, distogram_atom, "cpu")
      ckpt = torch.load(tcfg.checkpoint.checkpoint_path, weights_only=True)
      for key, tensor in model.state_dict().items():
          assert torch.equal(ckpt["model"][key], tensor), (
              f"Checkpoint mismatch at key '{key}'"
          )


  def test_train_ddp_checkpoint_unwraps_ddp_module(
      model: MainTrunk,
      ddp_loader: torch.utils.data.DataLoader[ProteinBatch],
      tcfg: TrainConfig,
      distogram_res: Distogram,
      distogram_atom: Distogram,
  ) -> None:
      """Checkpoint from train_ddp contains the inner module weights, not the DDP wrapper."""
      train_ddp(
          0, 0, 1, model, tcfg, ddp_loader, ddp_loader,
          distogram_res, distogram_atom, device="cpu",
      )
      ckpt = torch.load(tcfg.checkpoint.checkpoint_path, weights_only=True)
      for key, tensor in model.state_dict().items():
          assert torch.equal(ckpt["model"][key], tensor), (
              f"Checkpoint mismatch at key '{key}'"
          )
  ```

  **Why these pass with `_FakeDDP`:** `_patch_ddp` (autouse) replaces `DDP` with `_FakeDDP`.
  `_FakeDDP.__init__` stores the original `model` as `self.module`. In `log_epoch`, the line
  `inner = model.module if isinstance(model, DDP) else model` uses the real `DDP` class for
  the `isinstance` check — but `DDP` was patched to `_FakeDDP`, so `isinstance(ddp_model,
  _FakeDDP)` is `True` and `inner = ddp_model.module = model`. The checkpoint will contain
  `model.state_dict()` exactly.

  Wait — there is a subtlety. `log_epoch` imports `DDP` at module load time:
  ```python
  from torch.nn.parallel import DistributedDataParallel as DDP
  ```
  `_patch_ddp` patches `train.train_loop.DDP`, which is the name used in the `isinstance` check.
  So `isinstance(ddp_model, train.train_loop.DDP)` where `train.train_loop.DDP` is now `_FakeDDP`.
  This means `isinstance(ddp_model, _FakeDDP)` is `True` → `inner = ddp_model.module`. ✓

- [ ] **Step 4.2 — Run these two tests**

  ```bash
  python -m pytest pallatom/tests/train/test_train_loop.py \
      -k "test_train_checkpoint_matches_final_model or test_train_ddp_checkpoint_unwraps" \
      -v --tb=short
  ```

  **If FAILS with `KeyError: 'module.some_param'`:** the checkpoint was saved with DDP-prefixed
  keys (e.g. `module.encoder.weight` instead of `encoder.weight`). In `log_epoch`, verify that
  `inner = model.module if isinstance(model, DDP) else model` is reached before
  `inner.state_dict()` — i.e., the `isinstance` check works correctly with the patched `DDP`.

  **If FAILS with `AssertionError: Checkpoint mismatch at key '...'`:** the checkpoint was saved
  at the wrong epoch (e.g., epoch 0 stale weights). In `log_epoch`, confirm that `state_dict = 
  inner.state_dict()` is called AFTER the optimizer step, not before.

- [ ] **Step 4.3 — Commit**

  ```bash
  git add pallatom/tests/train/test_train_loop.py
  git commit -m "test(integration): checkpoint contains exact final model weights"
  ```

---

## Task 5 — Epoch metrics are averages, not sums

**Files:**
- Modify: `pallatom/tests/train/test_train_loop.py` (append)

- [ ] **Step 5.1 — Append two averaging tests**

  Append to `test_train_loop.py`:

  ```python
  # ---------------------------------------------------------------------------
  # Integration — epoch metrics are per-step averages, not accumulated sums
  # ---------------------------------------------------------------------------


  def test_train_epoch_metrics_are_averages_not_sums(
      model: MainTrunk,
      multi_loader: torch.utils.data.DataLoader[ProteinBatch],
      tcfg: TrainConfig,
      distogram_res: Distogram,
      distogram_atom: Distogram,
      monkeypatch: pytest.MonkeyPatch,
  ) -> None:
      """avg_train passed to log_epoch equals the mean of per-step metrics (not the sum)."""
      step_records: list[dict[str, float]] = []
      captured_avg: list[dict[str, float]] = []
      _real_train_step = train_step
      _real_log_epoch = log_epoch

      def _patched_step(
          batch: ProteinBatch,
          m: nn.Module,
          cfg: TrainConfig,
          dr: Distogram,
          da: Distogram,
          opt: Adam,
          dev: str,
      ) -> tuple[dict[str, float], float]:
          result = _real_train_step(batch, m, cfg, dr, da, opt, dev)
          step_records.append(result[0])
          return result

      def _patched_log_epoch(
          epoch: int,
          global_step: int,
          avg_train: dict[str, float],
          avg_val: dict[str, float],
          m: nn.Module,
          cfg: TrainConfig,
          best_val: float,
          *,
          do_log: bool = True,
      ) -> float:
          if not captured_avg:
              captured_avg.append(avg_train)
          return _real_log_epoch(
              epoch, global_step, avg_train, avg_val, m, cfg, best_val, do_log=do_log
          )

      monkeypatch.setattr("train.train_loop.train_step", _patched_step)
      monkeypatch.setattr("train.train_loop.log_epoch", _patched_log_epoch)
      train(model, tcfg, multi_loader, multi_loader, distogram_res, distogram_atom, "cpu")

      assert len(step_records) == 3, f"Expected 3 steps, got {len(step_records)}"
      avg = captured_avg[0]
      for key in _LOSS_KEYS:
          expected = sum(r[key] for r in step_records) / 3
          assert abs(avg[key] - expected) < 1e-5, (
              f"Key '{key}': avg_train={avg[key]:.8f}, mean(steps)={expected:.8f}"
          )


  def test_train_ddp_epoch_metrics_are_averages_not_sums(
      model: MainTrunk,
      multi_ddp_loader: torch.utils.data.DataLoader[ProteinBatch],
      tcfg: TrainConfig,
      distogram_res: Distogram,
      distogram_atom: Distogram,
      monkeypatch: pytest.MonkeyPatch,
  ) -> None:
      """avg_train in train_ddp equals the mean of per-step metrics (not the sum)."""
      step_records: list[dict[str, float]] = []
      captured_avg: list[dict[str, float]] = []
      _real_train_step = train_step
      _real_log_epoch = log_epoch

      def _patched_step(
          batch: ProteinBatch,
          m: nn.Module,
          cfg: TrainConfig,
          dr: Distogram,
          da: Distogram,
          opt: Adam,
          dev: str,
      ) -> tuple[dict[str, float], float]:
          result = _real_train_step(batch, m, cfg, dr, da, opt, dev)
          step_records.append(result[0])
          return result

      def _patched_log_epoch(
          epoch: int,
          global_step: int,
          avg_train: dict[str, float],
          avg_val: dict[str, float],
          m: nn.Module,
          cfg: TrainConfig,
          best_val: float,
          *,
          do_log: bool = True,
      ) -> float:
          if not captured_avg:
              captured_avg.append(avg_train)
          return _real_log_epoch(
              epoch, global_step, avg_train, avg_val, m, cfg, best_val, do_log=do_log
          )

      monkeypatch.setattr("train.train_loop.train_step", _patched_step)
      monkeypatch.setattr("train.train_loop.log_epoch", _patched_log_epoch)
      train_ddp(
          0, 0, 1, model, tcfg, multi_ddp_loader, multi_ddp_loader,
          distogram_res, distogram_atom, device="cpu",
      )

      assert len(step_records) == 3, f"Expected 3 steps, got {len(step_records)}"
      avg = captured_avg[0]
      for key in _LOSS_KEYS:
          expected = sum(r[key] for r in step_records) / 3
          assert abs(avg[key] - expected) < 1e-5, (
              f"Key '{key}': avg_train={avg[key]:.8f}, mean(steps)={expected:.8f}"
          )
  ```

- [ ] **Step 5.2 — Run these two tests**

  ```bash
  python -m pytest pallatom/tests/train/test_train_loop.py \
      -k "test_train_epoch_metrics_are_averages or test_train_ddp_epoch_metrics_are_averages" \
      -v --tb=short
  ```

  **If FAILS with large difference between `avg_train[key]` and `mean(steps[key])`:** the epoch
  metrics are being accumulated incorrectly. In `train()` (lines 506–537), look at the
  `epoch_metrics` accumulation loop — check that `n_batches` is incremented once per batch, not
  per metric key, and that the final division is `/ n_batches` not `/ len(epoch_metrics)`.

- [ ] **Step 5.3 — Commit**

  ```bash
  git add pallatom/tests/train/test_train_loop.py
  git commit -m "test(integration): epoch metrics are per-step averages in train and train_ddp"
  ```

---

## Task 6 — Empty-loader crash and avg_val key guard

**Files:**
- Modify: `pallatom/tests/train/test_train_loop.py` (append)

- [ ] **Step 6.1 — Append two guard tests**

  Append to `test_train_loop.py`:

  ```python
  # ---------------------------------------------------------------------------
  # Integration — empty-loader crash and avg_val key guard
  # ---------------------------------------------------------------------------


  def test_train_empty_train_loader_raises(
      model: MainTrunk,
      tcfg: TrainConfig,
      distogram_res: Distogram,
      distogram_atom: Distogram,
  ) -> None:
      """train() raises ZeroDivisionError when train_loader yields no batches.

      This documents the existing crash at the `{k: v / n_batches ...}` line
      (train_loop.py line ~536) when n_batches == 0. Once fixed, update this
      test to assert the function returns without error.
      """
      empty_loader: torch.utils.data.DataLoader[ProteinBatch] = torch.utils.data.DataLoader(
          _ListDataset([]), batch_size=None
      )
      with pytest.raises(ZeroDivisionError):
          train(model, tcfg, empty_loader, empty_loader, distogram_res, distogram_atom, "cpu")


  def test_train_avg_val_has_total_loss_key(
      model: MainTrunk,
      loader: torch.utils.data.DataLoader[ProteinBatch],
      tcfg: TrainConfig,
      distogram_res: Distogram,
      distogram_atom: Distogram,
      monkeypatch: pytest.MonkeyPatch,
  ) -> None:
      """avg_val forwarded to log_epoch always contains the 'total loss' key."""
      captured: list[dict[str, float]] = []
      _real_log_epoch = log_epoch

      def _patched_log_epoch(
          epoch: int,
          global_step: int,
          avg_train: dict[str, float],
          avg_val: dict[str, float],
          m: nn.Module,
          cfg: TrainConfig,
          best_val: float,
          *,
          do_log: bool = True,
      ) -> float:
          captured.append(avg_val)
          return _real_log_epoch(
              epoch, global_step, avg_train, avg_val, m, cfg, best_val, do_log=do_log
          )

      monkeypatch.setattr("train.train_loop.log_epoch", _patched_log_epoch)
      train(model, tcfg, loader, loader, distogram_res, distogram_atom, "cpu")
      assert captured, "log_epoch was never called"
      assert "total loss" in captured[0], (
          f"'total loss' missing from avg_val keys: {list(captured[0].keys())}"
      )
  ```

- [ ] **Step 6.2 — Run these two tests**

  ```bash
  python -m pytest pallatom/tests/train/test_train_loop.py \
      -k "test_train_empty_train_loader_raises or test_train_avg_val_has_total_loss_key" \
      -v --tb=short
  ```

  **`test_train_empty_train_loader_raises` — expected outcomes:**
  - PASS: `ZeroDivisionError` is raised — documents the crash (expected).
  - FAIL with different exception: some other crash happens before the division — investigate.
  - FAIL with no exception: `train()` silently handles empty loader — update the test to
    `assert result is None` once the fix is in place.

  **`test_train_avg_val_has_total_loss_key` — expected outcomes:**
  - PASS: `evaluate()` returns a dict with `"total loss"` — expected.
  - FAIL with `KeyError`: `evaluate()` changed its output keys — fix the key in `evaluate()` or
    `log_epoch`.

- [ ] **Step 6.3 — Commit**

  ```bash
  git add pallatom/tests/train/test_train_loop.py
  git commit -m "test(integration): document empty-loader crash and verify avg_val key"
  ```

---

## Task 7 — NaN warning emitted in train_ddp

**Files:**
- Modify: `pallatom/tests/train/test_train_loop.py` (append)

- [ ] **Step 7.1 — Append the NaN-warning test**

  Append to `test_train_loop.py`:

  ```python
  # ---------------------------------------------------------------------------
  # Integration — NaN detection warning in train_ddp
  # ---------------------------------------------------------------------------


  def test_train_ddp_nan_warning_emitted(
      model: MainTrunk,
      ddp_loader: torch.utils.data.DataLoader[ProteinBatch],
      tcfg: TrainConfig,
      distogram_res: Distogram,
      distogram_atom: Distogram,
      monkeypatch: pytest.MonkeyPatch,
  ) -> None:
      """train_ddp emits a structlog warning when total loss is NaN at rank 0."""
      _nan_metrics: dict[str, float] = {
          "total loss": float("nan"),
          "Kabsch aligned MSE loss": 0.0,
          "Cross Entropy loss": 0.0,
          "smooth lddt": 0.0,
          "Residue Distogram loss": 0.0,
          "Atom Distogram loss": 0.0,
          "Intermediate loss": 0.0,
      }

      def _nan_train_step(*_args: Any) -> tuple[dict[str, float], float]:
          return _nan_metrics, 0.0

      warning_events: list[str] = []
      mock_log = MagicMock()
      mock_log.warning.side_effect = lambda event, **_kwargs: warning_events.append(event)

      monkeypatch.setattr("train.train_loop.train_step", _nan_train_step)
      monkeypatch.setattr("train.train_loop.log", mock_log)

      train_ddp(
          0, 0, 1, model, tcfg, ddp_loader, ddp_loader,
          distogram_res, distogram_atom, device="cpu",
      )
      assert "nan_loss" in warning_events, (
          f"Expected 'nan_loss' warning; got: {warning_events}"
      )
  ```

  **How this works:** `_nan_train_step` returns NaN total loss without touching the model or
  optimizer. `monkeypatch.setattr("train.train_loop.log", mock_log)` replaces the structlog
  logger used by `train_ddp` and `log_epoch`. The NaN check at
  `train_loop.py:598–600` fires only when `rank == 0 and math.isnan(step_metrics["total loss"])`.
  Since we pass `rank=0`, the warning is expected.

- [ ] **Step 7.2 — Run the test**

  ```bash
  python -m pytest pallatom/tests/train/test_train_loop.py \
      -k "test_train_ddp_nan_warning_emitted" \
      -v --tb=short
  ```

  **If PASSES:** NaN detection branch works correctly.

  **If FAILS with empty `warning_events`:** the NaN guard was removed or the condition changed.
  In `train_loop.py`, check for the block:
  ```python
  if rank == 0 and math.isnan(step_metrics["total loss"]):
      log.warning("nan_loss", step=global_step, nan_components=nan_keys)
  ```
  Ensure `rank` is the first positional arg passed to `train_ddp` (it is, by convention).

- [ ] **Step 7.3 — Run full suite and commit**

  ```bash
  python -m pytest pallatom/tests/train/test_train_loop.py -v --tb=short 2>&1 | tail -30
  git add pallatom/tests/train/test_train_loop.py
  git commit -m "test(integration): NaN loss emits structlog warning in train_ddp"
  ```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task covering it |
|---|---|
| `test_train_lr_decreases_after_epoch` | Task 2 |
| `test_train_loss_components_all_nonzero` | Task 3 |
| `test_train_checkpoint_matches_final_model` | Task 4 |
| `test_train_epoch_metrics_are_averages_not_sums` | Task 5 |
| `test_train_empty_train_loader_raises` | Task 6 |
| `test_train_avg_val_has_total_loss_key` | Task 6 |
| `test_train_ddp_loss_components_all_nonzero` | Task 3 |
| `test_train_ddp_lr_decreases_after_epoch` | Task 2 |
| `test_train_ddp_checkpoint_unwraps_ddp_module` | Task 4 |
| `test_train_ddp_nan_warning_emitted` | Task 7 |
| `test_train_ddp_epoch_metrics_are_averages_not_sums` | Task 5 |
| `multi_loader` fixture | Task 1 |
| `multi_ddp_loader` fixture | Task 1 |
| `CosineAnnealingLR` import | Task 1 |
| `log_epoch` import | Task 1 |

**Type consistency:** `_LOSS_KEYS` is a module-level `list[str]` defined once in Task 3 and
reused in Tasks 5. All patched-function signatures match the actual signatures in `train_loop.py`.
`_nan_train_step` returns `tuple[dict[str, float], float]` matching `train_step`'s return type.

**Docstring check:** All 11 test functions and 2 fixture functions have Google-style one-line
docstrings. `_LOSS_KEYS` is a module-level constant — no docstring needed (not a function/class).

**Line length:** All multi-argument calls are broken at 100 characters.
