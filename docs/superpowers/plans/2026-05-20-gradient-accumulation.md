# Gradient Accumulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `accumulated_batch_size: int = 32` to `TrainConfig` and rewrite the `train()` / `train_ddp()` epoch loops to buffer micro-batches and fire the optimizer only after a complete accumulation window.

**Architecture:** `TrainingParams.accumulated_batch_size` stores the global effective batch target. `accum_steps = max(1, accumulated_batch_size // (batch_size * world_size))` is derived at runtime. `train_step` becomes a pure forward+backward function (caller owns `zero_grad`, `clip_grad_norm_`, `optimizer.step`). The loops buffer micro-batches, use `model.no_sync()` for non-final DDP steps, and drop partial windows at epoch end with a warning.

**Tech Stack:** Python 3.10+, PyTorch, Pydantic v2, pytest, structlog

---

### File Map

| File | Change |
|------|--------|
| `pallatom/train/train_config.py` | Add `accumulated_batch_size` to `TrainingParams`; add cross-config `@model_validator` to `TrainConfig` |
| `pallatom/train/train_loop.py` | Remove `optimizer` from `train_step`, add `grad_scale`; rewrite `train()` and `train_ddp()` epoch loops with micro-batch buffer |
| `pallatom/tests/train/test_train_loop.py` | Update 4 existing `train_step` call sites; add `accumulated_batch_size=2` to all 10 `TrainingParams(...)` fixture calls; add 6 new tests |

---

### Task 1: Config — `accumulated_batch_size` field and cross-config validator

**Files:**
- Modify: `pallatom/train/train_config.py`
- Modify: `pallatom/tests/train/test_train_loop.py`

- [ ] **Step 1: Write three failing tests**

Add to the import block in `pallatom/tests/train/test_train_loop.py`:

```python
from pydantic import ValidationError
```

Also add `LoaderConfig` to the existing `from train.train_config import (...)` block:

```python
from train.train_config import (
    CheckpointParams,
    LoaderConfig,
    LoggingParams,
    ModelParams,
    TrainConfig,
    TrainingParams,
)
```

Then add these three tests at the bottom of the file:

```python
def test_training_params_accumulated_batch_size_default() -> None:
    """accumulated_batch_size defaults to 32."""
    assert TrainingParams().accumulated_batch_size == 32


def test_train_config_validator_rejects_accum_lt_batch_size() -> None:
    """TrainConfig raises ValidationError when accumulated_batch_size < train_loader.batch_size."""
    with pytest.raises(ValidationError):
        TrainConfig(
            training=TrainingParams(accumulated_batch_size=1),
            train_loader=LoaderConfig(batch_size=2),
        )


def test_train_config_validator_accepts_accum_eq_batch_size() -> None:
    """TrainConfig accepts accumulated_batch_size == train_loader.batch_size."""
    cfg = TrainConfig(
        training=TrainingParams(accumulated_batch_size=2),
        train_loader=LoaderConfig(batch_size=2),
    )
    assert cfg.training.accumulated_batch_size == 2
```

- [ ] **Step 2: Run tests to see them fail**

```bash
pytest pallatom/tests/train/test_train_loop.py::test_training_params_accumulated_batch_size_default pallatom/tests/train/test_train_loop.py::test_train_config_validator_rejects_accum_lt_batch_size pallatom/tests/train/test_train_loop.py::test_train_config_validator_accepts_accum_eq_batch_size -v
```

Expected: all 3 FAIL with `AttributeError: 'TrainingParams' object has no attribute 'accumulated_batch_size'`.

- [ ] **Step 3: Add `accumulated_batch_size` to `TrainingParams`**

In `pallatom/train/train_config.py`, inside `TrainingParams`, after `resume_checkpoint`:

```python
accumulated_batch_size: int = Field(default=32, gt=0)
```

- [ ] **Step 4: Add cross-config validator to `TrainConfig`**

In `pallatom/train/train_config.py`, inside `TrainConfig`, after the field declarations:

```python
@model_validator(mode="after")
def _accumulated_batch_size_gte_loader_batch_size(self) -> "TrainConfig":
    """Validate accumulated_batch_size is at least one loader micro-batch."""
    if self.training.accumulated_batch_size < self.train_loader.batch_size:
        raise ValueError(
            f"accumulated_batch_size ({self.training.accumulated_batch_size}) must be"
            f" >= train_loader.batch_size ({self.train_loader.batch_size})"
        )
    return self
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest pallatom/tests/train/test_train_loop.py::test_training_params_accumulated_batch_size_default pallatom/tests/train/test_train_loop.py::test_train_config_validator_rejects_accum_lt_batch_size pallatom/tests/train/test_train_loop.py::test_train_config_validator_accepts_accum_eq_batch_size -v
```

Expected: all 3 PASS.

- [ ] **Step 6: Commit**

```bash
git add pallatom/train/train_config.py pallatom/tests/train/test_train_loop.py
git commit -m "feat: add accumulated_batch_size to TrainingParams with cross-config validator"
```

---

### Task 2: Refactor `train_step` — remove optimizer, add `grad_scale`; update loop callers

**Files:**
- Modify: `pallatom/train/train_loop.py`
- Modify: `pallatom/tests/train/test_train_loop.py`

- [ ] **Step 1: Update the 4 existing `train_step` call sites in tests**

In `pallatom/tests/train/test_train_loop.py`:

**1a.** Remove the `from torch.optim import Adam` import line (it will no longer be needed in the test file).

**1b.** Replace `test_train_step_returns_expected_keys`:

```python
def test_train_step_returns_expected_keys(
    protein_batch: ProteinBatch,
    model: MainTrunk,
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
) -> None:
    """train_step return dict contains exactly EXPECTED_STEP_KEYS."""
    model.zero_grad()
    metrics = train_step(protein_batch, model, tcfg, distogram_res, distogram_atom, "cpu")
    assert set(metrics.keys()) == EXPECTED_STEP_KEYS
```

**1c.** Replace `test_train_step_pack_rate_in_range`:

```python
def test_train_step_pack_rate_in_range(
    protein_batch: ProteinBatch,
    model: MainTrunk,
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
) -> None:
    """pack_rate is in (0, 1]."""
    model.zero_grad()
    metrics = train_step(protein_batch, model, tcfg, distogram_res, distogram_atom, "cpu")
    assert 0.0 < metrics["pack_rate"] <= 1.0
```

**1d.** Replace `test_train_step_throughput_metrics_positive`:

```python
def test_train_step_throughput_metrics_positive(
    protein_batch: ProteinBatch,
    model: MainTrunk,
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
) -> None:
    """residues_per_sec and atoms_per_sec are positive floats."""
    model.zero_grad()
    metrics = train_step(protein_batch, model, tcfg, distogram_res, distogram_atom, "cpu")
    assert metrics["residues_per_sec"] > 0.0
    assert metrics["atoms_per_sec"] > 0.0
```

**1e.** Replace `test_integration_gradient_flow_via_train_step`:

```python
def test_integration_gradient_flow_via_train_step(
    model: MainTrunk,
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    loader: torch.utils.data.DataLoader[ProteinBatch],
) -> None:
    """train_step() back-propagates finite nonzero grads to every MainTrunk submodule."""
    model.train()
    batch = next(iter(loader))
    model.zero_grad()
    train_step(batch, model, tcfg, distogram_res, distogram_atom, device="cpu")
    _assert_submodule_grads(model)
```

- [ ] **Step 2: Run updated tests to see them fail**

```bash
pytest pallatom/tests/train/test_train_loop.py::test_train_step_returns_expected_keys pallatom/tests/train/test_train_loop.py::test_train_step_pack_rate_in_range pallatom/tests/train/test_train_loop.py::test_train_step_throughput_metrics_positive pallatom/tests/train/test_train_loop.py::test_integration_gradient_flow_via_train_step -v
```

Expected: all 4 FAIL with `TypeError` — old signature still requires `optimizer`.

- [ ] **Step 3: Replace `train_step` in `pallatom/train/train_loop.py`**

Replace the entire `train_step` function (from `def train_step(` through its final `return {`) with:

```python
def train_step(
    batch: ProteinBatch,
    model: nn.Module,
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    device: str,
    grad_scale: float = 1.0,
) -> dict[str, float]:
    """Forward and backward pass for one micro-batch.

    The caller owns ``optimizer.zero_grad()``, ``clip_grad_norm_``, and
    ``optimizer.step()``.  Pass ``accum_steps`` as ``grad_scale`` so that
    accumulated gradients match a single large-batch backward.

    Args:
        batch: Raw protein micro-batch.
        model: Model to forward through (plain ``MainTrunk`` or DDP-wrapped).
        tcfg: Training configuration.
        distogram_res: Residue-level distogram.
        distogram_atom: Atom-level distogram.
        device: PyTorch device string.
        grad_scale: Divide total loss by this value before backward (default 1.0).

    Returns:
        Step-level metrics dict with keys matching ``EXPECTED_STEP_KEYS``.
    """
    lp = tcfg.loss
    tp = tcfg.training

    featurized_batch = featurize_batch(batch, tcfg, distogram_res, distogram_atom, device)
    featurized_batch = apply_conditioning_dropout(
        featurized_batch,
        p_distogram=tcfg.conditioning_dropout.p_distogram,
        p_atom=tcfg.conditioning_dropout.p_atom,
        p_seq=tcfg.conditioning_dropout.p_seq,
        device=device,
    )

    t0 = time.perf_counter()
    (
        r_denoised,
        f_seq_logits,
        residue_distogram_logits,
        atom_distogram_logits,
        intermediate_denoised_coord_stack,
        intermediate_pred_aa_logit_stack,
    ) = model(featurized_batch)
    r_denoised: Float[torch.Tensor, "B N_atom 3"]
    f_seq_logits: Float[torch.Tensor, "B N_res n_amino"]
    residue_distogram_logits: Float[torch.Tensor, "B N_res N_res n_bins"]
    atom_distogram_logits: Float[torch.Tensor, "B N_atom K n_atom_bins"]

    Kabsch_aligned_MSE_loss: Float[torch.Tensor, ""] = atom_loss(
        r_denoised, featurized_batch.r_gt, featurized_batch.atom5_mask
    ).mean()

    K_unit = len(intermediate_denoised_coord_stack)
    intermediate_med_loss: Float[torch.Tensor, ""] = torch.tensor(0.0, device=device)
    for k_idx, intermediate_denoised_coord in enumerate(intermediate_denoised_coord_stack):
        intermediate_denoised_coord: Float[torch.Tensor, "B N_atom 3"]
        gamma_K_minus_k: float = lp.gamma ** (K_unit - k_idx - 1)
        k_loss: Float[torch.Tensor, ""] = lp.lam * atom_loss(
            intermediate_denoised_coord, featurized_batch.r_gt, featurized_batch.atom5_mask
        ) + lp.alpha_0 * F.cross_entropy(
            rearrange(intermediate_pred_aa_logit_stack[k_idx], "b n c -> (b n) c"),
            rearrange(_mask_seq_target(featurized_batch.aa_indices), "b n -> (b n)"),
        )
        intermediate_med_loss = intermediate_med_loss + gamma_K_minus_k * k_loss
    intermediate_med_loss = (intermediate_med_loss / max(K_unit, 1)).mean()

    gt_res_bin_idx: Int[torch.Tensor, "B N_res N_res"] = featurized_batch.gt_res_distogram.argmax(
        dim=-1
    ).clamp(0, residue_distogram_logits.size(-1) - 1)
    residue_distogram_loss: Float[torch.Tensor, ""] = distogram_loss_residue(
        residue_distogram_logits,
        gt_res_bin_idx,
        featurized_batch.residue_mask,
    ).mean()

    atom_distogram_loss: Float[torch.Tensor, ""] = distogram_loss_atom(
        atom_distogram_logits,
        featurized_batch.gt_atom_distogram_sparse,
        featurized_batch.gt_atom_distogram_mask_sparse,
    ).mean()

    lddt_loss: Float[torch.Tensor, ""] = smooth_lddt_loss(
        r_denoised,
        featurized_batch.r_gt,
        featurized_batch.atom5_mask,
        cutoff=float(lp.smooth_lddt_cutoff),
    )
    CE_loss: Float[torch.Tensor, ""] = F.cross_entropy(
        rearrange(f_seq_logits, "b n c -> (b n) c"),
        rearrange(_mask_seq_target(featurized_batch.aa_indices), "b n -> (b n)"),
    )

    total_loss: Float[torch.Tensor, ""] = (
        lp.lam * Kabsch_aligned_MSE_loss
        + lp.alpha_0 * CE_loss
        + lp.alpha_1 * lddt_loss
        + lp.alpha_2 * residue_distogram_loss
        + lp.alpha_3 * atom_distogram_loss
        + lp.alpha_4 * intermediate_med_loss
    )

    (total_loss / grad_scale).backward()
    t1 = time.perf_counter()
    step_time = t1 - t0

    b_size, n_res = featurized_batch.residue_mask.shape
    actual_residues = int(featurized_batch.residue_mask.sum().item())
    actual_atoms = int(featurized_batch.atom5_mask.sum().item())

    return {
        "total loss": total_loss.item(),
        "Kabsch aligned MSE loss": Kabsch_aligned_MSE_loss.item(),
        "Cross Entropy loss": CE_loss.item(),
        "smooth lddt": lddt_loss.item(),
        "Residue Distogram loss": residue_distogram_loss.item(),
        "Atom Distogram loss": atom_distogram_loss.item(),
        "Intermediate loss": intermediate_med_loss.item(),
        "pack_rate": actual_residues / (b_size * n_res),
        "residues_per_sec": actual_residues / step_time,
        "atoms_per_sec": actual_atoms / step_time,
    }
```

Note: the `tp = tcfg.training` line is now unused inside `train_step` — remove it from the function body.

- [ ] **Step 4: Update the `train()` epoch loop to call `train_step` without optimizer**

In `pallatom/train/train_loop.py`, inside `train()`, replace the for-batch loop body (the `for batch in pbar:` block):

```python
        optimizer.zero_grad()
        for batch in pbar:
            step_metrics = train_step(
                batch, model, tcfg, distogram_res, distogram_atom, device
            )
            grad_norm: float = float(
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    tp.grad_clip if tp.grad_clip is not None else float("inf"),
                )
            )
            optimizer.step()
            optimizer.zero_grad()
            for k in epoch_metrics:
                epoch_metrics[k] += step_metrics[k]
            n_batches += 1
            global_step += 1

            if global_step % lg.log_interval == 0:
                pbar.set_postfix(
                    loss=f"{step_metrics['total loss']:.4f}", gnorm=f"{grad_norm:.3f}"
                )
```

- [ ] **Step 5: Update the `train_ddp()` epoch loop to call `train_step` without optimizer**

In `pallatom/train/train_loop.py`, inside `train_ddp()`, replace the for-batch loop body (the `for batch in pbar:` block):

```python
        optimizer.zero_grad()
        for batch in pbar:
            step_metrics = train_step(
                batch, ddp_model, tcfg, distogram_res, distogram_atom, device
            )
            if rank == 0 and math.isnan(step_metrics["total loss"]):
                nan_keys = [k for k, v in step_metrics.items() if math.isnan(v)]
                log.warning("nan_loss", step=global_step, nan_components=nan_keys)
            grad_norm: float = float(
                nn.utils.clip_grad_norm_(
                    ddp_model.parameters(),
                    tp.grad_clip if tp.grad_clip is not None else float("inf"),
                )
            )
            optimizer.step()
            optimizer.zero_grad()
            for k in epoch_metrics:
                epoch_metrics[k] += step_metrics[k]
            n_batches += 1
            global_step += 1

            if rank == 0 and global_step % lg.log_interval == 0:
                pbar.set_postfix(
                    loss=f"{step_metrics['total loss']:.4f}", gnorm=f"{grad_norm:.3f}"
                )
```

- [ ] **Step 6: Run the 4 train_step tests to verify they pass**

```bash
pytest pallatom/tests/train/test_train_loop.py::test_train_step_returns_expected_keys pallatom/tests/train/test_train_loop.py::test_train_step_pack_rate_in_range pallatom/tests/train/test_train_loop.py::test_train_step_throughput_metrics_positive pallatom/tests/train/test_train_loop.py::test_integration_gradient_flow_via_train_step -v
```

Expected: all 4 PASS.

- [ ] **Step 7: Run the full test suite**

```bash
pytest pallatom/tests/train/test_train_loop.py -v
```

Expected: all tests PASS.

- [ ] **Step 8: Commit**

```bash
git add pallatom/train/train_loop.py pallatom/tests/train/test_train_loop.py
git commit -m "refactor: remove optimizer from train_step, add grad_scale; update loop callers"
```

---

### Task 3: Update test fixtures and implement accumulation buffer in `train()`

**Files:**
- Modify: `pallatom/tests/train/test_train_loop.py`
- Modify: `pallatom/train/train_loop.py`

- [ ] **Step 1: Add `accumulated_batch_size=2` to all 10 `TrainingParams(...)` fixture calls**

In `pallatom/tests/train/test_train_loop.py`, update every `TrainingParams(...)` constructor. Without this, the default `accumulated_batch_size=32` with `train_loader.batch_size=2` gives `accum_steps=16`; test loaders hold only 1–3 batches, so no window ever completes and all parameter-update assertions break.

Update each occurrence (exact fixture/test and line locations):

`tcfg` fixture → `TrainingParams(num_epochs=1, lr=1e-4, grad_clip=1.0, accumulated_batch_size=2)`

`tcfg_multi` fixture → `TrainingParams(num_epochs=3, lr=1e-3, grad_clip=1.0, accumulated_batch_size=2)`

`tcfg_no_clip` fixture → `TrainingParams(num_epochs=1, lr=1e-4, grad_clip=None, accumulated_batch_size=2)`

`tcfg_wandb` fixture → `TrainingParams(num_epochs=1, lr=1e-4, grad_clip=1.0, accumulated_batch_size=2)`

`tcfg_wandb_3ep` fixture → `TrainingParams(num_epochs=3, lr=1e-4, grad_clip=1.0, accumulated_batch_size=2)`

`tcfg_save` fixture → `TrainingParams(num_epochs=1, lr=1e-4, grad_clip=1.0, accumulated_batch_size=2)`

`test_train_resume_runs_remaining_epochs` — `tcfg_first` → `TrainingParams(num_epochs=1, lr=1e-4, grad_clip=1.0, accumulated_batch_size=2)`

`test_train_resume_runs_remaining_epochs` — `tcfg_resume` → `TrainingParams(num_epochs=3, lr=1e-4, grad_clip=1.0, resume_checkpoint=ckpt_path, accumulated_batch_size=2)`

`test_train_resume_restores_optimizer_state` — `tcfg_first` → `TrainingParams(num_epochs=1, lr=1e-4, grad_clip=1.0, accumulated_batch_size=2)`

`test_train_resume_restores_scheduler_state` — `tcfg_first` → `TrainingParams(num_epochs=1, lr=1e-4, grad_clip=1.0, accumulated_batch_size=2)`

`test_train_resume_checkpoint_epoch_and_step` — `tcfg_first` → `TrainingParams(num_epochs=1, lr=1e-4, grad_clip=1.0, accumulated_batch_size=2)`

- [ ] **Step 2: Write a failing test for accumulation window behavior**

Add to `pallatom/tests/train/test_train_loop.py`:

```python
@pytest.fixture
def tcfg_accum(tmp_path: pathlib.Path) -> TrainConfig:
    """TrainConfig with accumulated_batch_size=4, giving accum_steps=2 with batch_size=2."""
    return TrainConfig(
        training=TrainingParams(num_epochs=1, lr=1e-4, grad_clip=1.0, accumulated_batch_size=4),
        model=ModelParams(
            f_ref_dim=_F_REF_DIM,
            n_bins=_N_BINS,
            c_atom=_C_ATOM,
            c_pair=_C_PAIR,
            c_res=_C_RES,
            c_atompair=_C_ATOMPAIR,
            K_unit=_K_UNIT,
        ),
        checkpoint=CheckpointParams(
            checkpoint_path=str(tmp_path / "best_accum.pt"),
            save_every=100,
        ),
        logging=LoggingParams(use_wandb=False, log_interval=1),
    )


def test_train_partial_window_does_not_update_params(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg_accum: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
) -> None:
    """With accum_steps=2 and only 1 batch available, the partial window is dropped and params are unchanged."""
    params_before = [p.clone().detach() for p in model.parameters()]
    train(model, tcfg_accum, loader, loader, distogram_res, distogram_atom, "cpu")
    assert all(
        torch.equal(b, a)
        for b, a in zip(params_before, model.parameters(), strict=False)
    )
```

- [ ] **Step 3: Run the new test to see it fail**

```bash
pytest pallatom/tests/train/test_train_loop.py::test_train_partial_window_does_not_update_params -v
```

Expected: FAIL — the non-buffered loop from Task 2 steps the optimizer on every batch regardless of `accum_steps`, so params DO change.

- [ ] **Step 4: Implement the accumulation buffer in `train()`**

In `pallatom/train/train_loop.py`, inside `train()`, add these two lines at the top of the function body (after `tp = tcfg.training`):

```python
    accum_steps: int = max(1, tp.accumulated_batch_size // tcfg.train_loader.batch_size)
    log.info("gradient_accumulation", accum_steps=accum_steps, effective_batch_size=tp.accumulated_batch_size)
```

Then replace the epoch loop (the entire `for epoch in range(...)` block) with:

```python
    for epoch in range(start_epoch, tp.num_epochs + 1):
        model.train()
        epoch_metrics: dict[str, float] = dict.fromkeys(
            [
                "total loss",
                "Kabsch aligned MSE loss",
                "Cross Entropy loss",
                "smooth lddt",
                "Residue Distogram loss",
                "Atom Distogram loss",
                "Intermediate loss",
                "pack_rate",
                "residues_per_sec",
                "atoms_per_sec",
            ],
            0.0,
        )
        n_batches = 0
        micro_buffer: list[ProteinBatch] = []
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{tp.num_epochs}", leave=False)

        for batch in pbar:
            micro_buffer.append(batch)
            if len(micro_buffer) < accum_steps:
                continue

            window_metrics: dict[str, float] = dict.fromkeys(
                [
                    "total loss",
                    "Kabsch aligned MSE loss",
                    "Cross Entropy loss",
                    "smooth lddt",
                    "Residue Distogram loss",
                    "Atom Distogram loss",
                    "Intermediate loss",
                    "pack_rate",
                    "residues_per_sec",
                    "atoms_per_sec",
                ],
                0.0,
            )
            for micro_idx, mb in enumerate(micro_buffer):
                is_last = micro_idx == accum_steps - 1
                ctx = (
                    model.no_sync()
                    if (not is_last and hasattr(model, "no_sync"))
                    else contextlib.nullcontext()
                )
                with ctx:
                    step_metrics = train_step(
                        mb,
                        model,
                        tcfg,
                        distogram_res,
                        distogram_atom,
                        device,
                        grad_scale=float(accum_steps),
                    )
                for k in window_metrics:
                    window_metrics[k] += step_metrics[k] / accum_steps

            grad_norm: float = float(
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    tp.grad_clip if tp.grad_clip is not None else float("inf"),
                )
            )
            optimizer.step()
            optimizer.zero_grad()

            for k in epoch_metrics:
                epoch_metrics[k] += window_metrics[k]
            n_batches += 1
            global_step += 1
            micro_buffer = []

            if global_step % lg.log_interval == 0:
                pbar.set_postfix(
                    loss=f"{window_metrics['total loss']:.4f}", gnorm=f"{grad_norm:.3f}"
                )

        if micro_buffer:
            log.warning("dropped_partial_window", n_dropped=len(micro_buffer))

        scheduler.step()

        avg_train = {k: v / max(n_batches, 1) for k, v in epoch_metrics.items()}
        avg_val = evaluate(model, test_loader, tcfg, distogram_res, distogram_atom, device)
        model.train()

        best_val_loss = log_epoch(
            epoch, global_step, avg_train, avg_val, model, optimizer, scheduler, tcfg, best_val_loss
        )
```

- [ ] **Step 5: Run the new test and the full suite**

```bash
pytest pallatom/tests/train/test_train_loop.py::test_train_partial_window_does_not_update_params pallatom/tests/train/test_train_loop.py -v
```

Expected: `test_train_partial_window_does_not_update_params` PASS; all other tests PASS.

- [ ] **Step 6: Commit**

```bash
git add pallatom/train/train_loop.py pallatom/tests/train/test_train_loop.py
git commit -m "feat: implement gradient accumulation buffer in train() with partial-window drop"
```

---

### Task 4: Implement accumulation buffer in `train_ddp()`

**Files:**
- Modify: `pallatom/train/train_loop.py`
- Modify: `pallatom/tests/train/test_train_loop.py`

- [ ] **Step 1: Write a failing test for `train_ddp()` accumulation**

Add to `pallatom/tests/train/test_train_loop.py`:

```python
def test_train_ddp_partial_window_does_not_update_params(
    model: MainTrunk,
    ddp_loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg_accum: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
) -> None:
    """With accum_steps=2 and only 1 batch, train_ddp() drops the partial window and params are unchanged."""
    params_before = [p.clone().detach() for p in model.parameters()]
    train_ddp(
        0,
        0,
        1,
        model,
        tcfg_accum,
        ddp_loader,
        ddp_loader,
        distogram_res,
        distogram_atom,
        device="cpu",
    )
    assert all(
        torch.equal(b, a)
        for b, a in zip(params_before, model.parameters(), strict=False)
    )
```

- [ ] **Step 2: Run the new test to see it fail**

```bash
pytest pallatom/tests/train/test_train_loop.py::test_train_ddp_partial_window_does_not_update_params -v
```

Expected: FAIL — the non-buffered `train_ddp()` loop still steps on every batch.

- [ ] **Step 3: Implement the accumulation buffer in `train_ddp()`**

In `pallatom/train/train_loop.py`, inside `train_ddp()`, add after `tp = tcfg.training`:

```python
    accum_steps: int = max(
        1, tp.accumulated_batch_size // (tcfg.train_loader.batch_size * world_size)
    )
    if rank == 0:
        log.info(
            "gradient_accumulation",
            accum_steps=accum_steps,
            effective_batch_size=tp.accumulated_batch_size,
        )
```

Then replace the epoch loop (the entire `for epoch in range(...)` block) with:

```python
    for epoch in range(start_epoch, tp.num_epochs + 1):
        ddp_model.train()
        cast(BucketedBatchSampler, train_loader.batch_sampler).set_epoch(epoch)
        epoch_metrics: dict[str, float] = dict.fromkeys(
            [
                "total loss",
                "Kabsch aligned MSE loss",
                "Cross Entropy loss",
                "smooth lddt",
                "Residue Distogram loss",
                "Atom Distogram loss",
                "Intermediate loss",
                "pack_rate",
                "residues_per_sec",
                "atoms_per_sec",
            ],
            0.0,
        )
        n_batches = 0
        micro_buffer: list[ProteinBatch] = []
        optimizer.zero_grad()

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:03d}/{tp.num_epochs}",
            leave=False,
            disable=(rank != 0),
        )

        for batch in pbar:
            micro_buffer.append(batch)
            if len(micro_buffer) < accum_steps:
                continue

            window_metrics: dict[str, float] = dict.fromkeys(
                [
                    "total loss",
                    "Kabsch aligned MSE loss",
                    "Cross Entropy loss",
                    "smooth lddt",
                    "Residue Distogram loss",
                    "Atom Distogram loss",
                    "Intermediate loss",
                    "pack_rate",
                    "residues_per_sec",
                    "atoms_per_sec",
                ],
                0.0,
            )
            for micro_idx, mb in enumerate(micro_buffer):
                is_last = micro_idx == accum_steps - 1
                ctx = (
                    ddp_model.no_sync()
                    if (not is_last and hasattr(ddp_model, "no_sync"))
                    else contextlib.nullcontext()
                )
                with ctx:
                    step_metrics = train_step(
                        mb,
                        ddp_model,
                        tcfg,
                        distogram_res,
                        distogram_atom,
                        device,
                        grad_scale=float(accum_steps),
                    )
                for k in window_metrics:
                    window_metrics[k] += step_metrics[k] / accum_steps

            if rank == 0 and math.isnan(window_metrics["total loss"]):
                nan_keys = [k for k, v in window_metrics.items() if math.isnan(v)]
                log.warning("nan_loss", step=global_step, nan_components=nan_keys)

            grad_norm: float = float(
                nn.utils.clip_grad_norm_(
                    ddp_model.parameters(),
                    tp.grad_clip if tp.grad_clip is not None else float("inf"),
                )
            )
            optimizer.step()
            optimizer.zero_grad()

            for k in epoch_metrics:
                epoch_metrics[k] += window_metrics[k]
            n_batches += 1
            global_step += 1
            micro_buffer = []

            if rank == 0 and global_step % lg.log_interval == 0:
                pbar.set_postfix(
                    loss=f"{window_metrics['total loss']:.4f}", gnorm=f"{grad_norm:.3f}"
                )

        if micro_buffer and rank == 0:
            log.warning("dropped_partial_window", n_dropped=len(micro_buffer))

        scheduler.step()

        avg_train = {k: v / max(n_batches, 1) for k, v in epoch_metrics.items()}
        _eff_world_size = world_size if dist.is_initialized() else 1
        avg_val = evaluate_ddp(
            _eff_world_size, ddp_model, test_loader, tcfg, distogram_res, distogram_atom, device
        )
        ddp_model.train()

        best_val_loss = log_epoch(
            epoch,
            global_step,
            avg_train,
            avg_val,
            ddp_model,
            optimizer,
            scheduler,
            tcfg,
            best_val_loss,
            do_log=(rank == 0),
        )
```

- [ ] **Step 4: Run the new test and the full suite**

```bash
pytest pallatom/tests/train/test_train_loop.py::test_train_ddp_partial_window_does_not_update_params pallatom/tests/train/test_train_loop.py -v
```

Expected: `test_train_ddp_partial_window_does_not_update_params` PASS; all other tests PASS.

- [ ] **Step 5: Commit**

```bash
git add pallatom/train/train_loop.py pallatom/tests/train/test_train_loop.py
git commit -m "feat: implement gradient accumulation buffer in train_ddp() with partial-window drop"
```

---

### Task 5: Add grad_scale magnitude test and full-window tests

**Files:**
- Modify: `pallatom/tests/train/test_train_loop.py`

- [ ] **Step 1: Add a fixture and three new tests**

Add to `pallatom/tests/train/test_train_loop.py`:

```python
@pytest.fixture
def loader_2batch(
    mini_batch: Mapping[str, Float[torch.Tensor, "..."] | list[str]]
) -> torch.utils.data.DataLoader[ProteinBatch]:
    """DataLoader with two identical mini-batches for accumulation tests."""
    return torch.utils.data.DataLoader(_ListDataset([mini_batch, mini_batch]), batch_size=None)


def test_train_step_grad_scale_halves_gradient_norm(
    protein_batch: ProteinBatch,
    model: MainTrunk,
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
) -> None:
    """Loss divided by grad_scale=2 yields approximately half the gradient norm of grad_scale=1."""
    torch.manual_seed(0)
    model.zero_grad()
    train_step(protein_batch, model, tcfg, distogram_res, distogram_atom, "cpu", grad_scale=1.0)
    norm1: float = sum(
        p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None
    ) ** 0.5

    model.zero_grad()
    train_step(protein_batch, model, tcfg, distogram_res, distogram_atom, "cpu", grad_scale=2.0)
    norm2: float = sum(
        p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None
    ) ** 0.5

    assert abs(norm2 - norm1 / 2.0) < 1e-4


def test_train_accumulation_full_window_updates_params(
    model: MainTrunk,
    loader_2batch: torch.utils.data.DataLoader[ProteinBatch],
    tcfg_accum: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
) -> None:
    """With accum_steps=2 and a 2-batch loader, train() completes one window and updates params."""
    params_before = [p.clone().detach() for p in model.parameters()]
    train(model, tcfg_accum, loader_2batch, loader_2batch, distogram_res, distogram_atom, "cpu")
    assert any(
        not torch.equal(b, a)
        for b, a in zip(params_before, model.parameters(), strict=False)
    )


def test_train_ddp_accumulation_full_window_updates_params(
    model: MainTrunk,
    tcfg_accum: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    mini_batch: Mapping[str, Float[torch.Tensor, "..."] | list[str]],
) -> None:
    """With accum_steps=2 and a 2-batch DDP loader, train_ddp() completes one window and updates params."""
    sampler = _MockSampler([mini_batch, mini_batch])
    ddp_loader_2batch: torch.utils.data.DataLoader[ProteinBatch] = torch.utils.data.DataLoader(
        _ListDataset([mini_batch, mini_batch]),
        batch_sampler=sampler,
        collate_fn=_identity_collate,
    )
    params_before = [p.clone().detach() for p in model.parameters()]
    train_ddp(
        0,
        0,
        1,
        model,
        tcfg_accum,
        ddp_loader_2batch,
        ddp_loader_2batch,
        distogram_res,
        distogram_atom,
        device="cpu",
    )
    assert any(
        not torch.equal(b, a)
        for b, a in zip(params_before, model.parameters(), strict=False)
    )
```

- [ ] **Step 2: Run the three new tests**

```bash
pytest pallatom/tests/train/test_train_loop.py::test_train_step_grad_scale_halves_gradient_norm pallatom/tests/train/test_train_loop.py::test_train_accumulation_full_window_updates_params pallatom/tests/train/test_train_loop.py::test_train_ddp_accumulation_full_window_updates_params -v
```

Expected: all 3 PASS.

- [ ] **Step 3: Run the full test suite**

```bash
pytest pallatom/tests/train/test_train_loop.py -v
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add pallatom/tests/train/test_train_loop.py
git commit -m "test: add grad_scale magnitude and accumulation window tests"
```
