# Token-Budget Gradient Accumulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the broken fixed `accum_steps` (computed from an unused `batch_size` field) with runtime token counting, and fix the grad_scale to weight each micro-batch by its protein count.

**Architecture:** `BucketedBatchSampler` yields variable-size batches; each batch's token count is `batch.atom_mask.any(dim=-1).sum()`. We pre-flush the accumulation buffer when adding a new batch would push total tokens over `accumulated_token_budget`, then also post-flush if the newly added batch alone reaches the budget. Grad_scale for each micro-batch is `total_proteins / n_proteins_i` since all losses already `.mean()` over the batch dimension.

**Tech Stack:** PyTorch, Pydantic v2, pytest, jaxtyping / beartype

---

## File Map

| File | Change |
|------|--------|
| `pallatom/train/train_config.py` | Rename `accumulated_batch_size` → `accumulated_token_budget` in `TrainingParams`; remove `_accumulated_batch_size_gte_loader_batch_size` validator from `TrainConfig` |
| `pallatom/train/train_loop.py` | Update `_process_accum_window` (new `n_proteins_per_batch` param, protein-weighted grad_scale/metrics); replace `accum_steps` logic in `train()` and `train_ddp()` with token-counting loops |
| `pallatom/tests/train/test_train_loop.py` | Write 3 new tests; remove 3 stale tests; rename `accumulated_batch_size` in all fixtures; add `_BATCH_TOKENS` constant |

---

### Task 1: Write all new and updated tests (before touching implementation)

**Files:**
- Modify: `pallatom/tests/train/test_train_loop.py`

The `mini_batch` fixture has `atom_mask=torch.ones(1, _N_KEEP, 37)` so `batch.atom_mask.any(dim=-1).sum() == _N_KEEP == 16`. Token budget values in fixtures are chosen so:
- `accumulated_token_budget=_BATCH_TOKENS` → post-add flush fires for every mini_batch (replaces old `accum_steps=1`)
- `accumulated_token_budget=2 * _BATCH_TOKENS` → two mini_batches required before flush (replaces old `accum_steps=2`)

- [ ] **Step 1: Add `_BATCH_TOKENS` constant and update imports**

After line `_K_UNIT = 1`, add:

```python
_BATCH_TOKENS: int = _N_KEEP  # atom_mask=ones → every residue is a valid token
```

Update the existing `from train.train_loop import` block to add two private names:

```python
from train.train_loop import (
    _METRIC_KEYS,
    _process_accum_window,
    evaluate,
    evaluate_ddp,
    train,
    train_ddp,
    train_step,
)
```

- [ ] **Step 2: Remove three stale tests**

Find and delete the three test functions at the bottom of the file (currently around lines 1747–1768):

```python
# DELETE all three of these:
def test_training_params_accumulated_batch_size_default() -> None: ...
def test_train_config_validator_rejects_accum_lt_batch_size() -> None: ...
def test_train_config_validator_accepts_accum_eq_batch_size() -> None: ...
```

- [ ] **Step 3: Add three new tests (at the end of the file)**

```python
# ---------------------------------------------------------------------------
# accumulated_token_budget config
# ---------------------------------------------------------------------------


def test_accumulated_token_budget_default() -> None:
    """accumulated_token_budget defaults to 2048."""
    assert TrainingParams().accumulated_token_budget == 2048


def test_accumulated_token_budget_rejects_zero() -> None:
    """TrainingParams raises ValidationError when accumulated_token_budget is zero."""
    with pytest.raises(ValidationError):
        TrainingParams(accumulated_token_budget=0)


def test_train_config_accepts_any_positive_token_budget() -> None:
    """TrainConfig no longer validates token budget against batch_size."""
    cfg = TrainConfig(training=TrainingParams(accumulated_token_budget=1))
    assert cfg.training.accumulated_token_budget == 1


# ---------------------------------------------------------------------------
# _process_accum_window — protein-weighted grad_scale
# ---------------------------------------------------------------------------


def test_process_accum_window_protein_weighted_grad_scale(
    protein_batch: ProteinBatch,
    model: MainTrunk,
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_process_accum_window passes protein-count-weighted grad_scale to each train_step call.

    Two micro-batches with n_proteins_per_batch=[1, 3] → total_proteins=4.
    Expected grad_scale: 4/1=4.0 for the first, 4/3≈1.333 for the second.
    """
    captured_scales: list[float] = []

    def _mock_train_step(
        batch: ProteinBatch,
        mdl: nn.Module,
        cfg: TrainConfig,
        dr: Distogram,
        da: Distogram,
        device: str,
        grad_scale: float = 1.0,
    ) -> dict[str, float]:
        captured_scales.append(grad_scale)
        return dict.fromkeys(_METRIC_KEYS, 0.0)

    monkeypatch.setattr("train.train_loop.train_step", _mock_train_step)

    _process_accum_window(
        micro_buffer=[protein_batch, protein_batch],
        n_proteins_per_batch=[1, 3],
        model=model,
        tcfg=tcfg,
        distogram_res=distogram_res,
        distogram_atom=distogram_atom,
        device="cpu",
    )

    assert len(captured_scales) == 2
    assert abs(captured_scales[0] - 4.0) < 1e-6        # total=4, n=1 → 4/1
    assert abs(captured_scales[1] - 4.0 / 3.0) < 1e-6  # total=4, n=3 → 4/3


# ---------------------------------------------------------------------------
# token-based accumulation — pre-flush behavior
# ---------------------------------------------------------------------------


def test_train_token_budget_preflush_fires_before_oversized_batch(
    model: MainTrunk,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    mini_batch: Mapping[str, Float[torch.Tensor, "..."] | list[str]],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    log: FilteringBoundLogger,
) -> None:
    """Pre-flush fires for batch1 when batch2 would push tokens over the budget.

    mini_batch has _BATCH_TOKENS=16 tokens. With budget=24 and a 2-batch loader:
    - batch1 (16 tokens): added, held (16 < 24)
    - batch2 (16 tokens): 16+16=32 > 24 → pre-flush batch1 alone, then batch2 added (post-add: 16<24)
    Result: _process_accum_window called exactly once (window of size 1); batch2 is partial, dropped.
    """
    window_sizes: list[int] = []
    _real_process = _process_accum_window

    def _tracking_process(
        micro_buffer: list[ProteinBatch],
        n_proteins_per_batch: list[int],
        mdl: nn.Module,
        cfg: TrainConfig,
        dr: Distogram,
        da: Distogram,
        dev: str,
    ) -> dict[str, float]:
        window_sizes.append(len(micro_buffer))
        return _real_process(micro_buffer, n_proteins_per_batch, mdl, cfg, dr, da, dev)

    monkeypatch.setattr("train.train_loop._process_accum_window", _tracking_process)

    budget: int = _BATCH_TOKENS + _BATCH_TOKENS // 2  # 24: one batch (16) fits; two (32) don't
    tcfg_budget = TrainConfig(
        training=TrainingParams(
            num_epochs=1, lr=1e-4, grad_clip=1.0, accumulated_token_budget=budget
        ),
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
            checkpoint_path=str(tmp_path / "best_flush.pt"), save_every=100
        ),
        logging=LoggingParams(use_wandb=False, log_interval=1),
    )
    loader_2b = torch.utils.data.DataLoader(
        _ListDataset([mini_batch, mini_batch]), batch_size=None
    )
    train(model, tcfg_budget, loader_2b, loader_2b, distogram_res, distogram_atom, "cpu", log)

    assert window_sizes == [1]  # pre-flush with batch1 alone; batch2 stays partial, is dropped
```

- [ ] **Step 4: Verify the new tests fail for the right reason**

```bash
cd /workspaces/diffusion && python -m pytest \
  pallatom/tests/train/test_train_loop.py::test_accumulated_token_budget_default \
  pallatom/tests/train/test_train_loop.py::test_process_accum_window_protein_weighted_grad_scale \
  pallatom/tests/train/test_train_loop.py::test_train_token_budget_preflush_fires_before_oversized_batch \
  -x -q 2>&1 | head -25
```

Expected output: FAIL — `AttributeError: 'TrainingParams' object has no attribute 'accumulated_token_budget'` (config rename not done yet). The other two will also fail for similar reasons.

- [ ] **Step 5: Commit the test changes**

```bash
git add pallatom/tests/train/test_train_loop.py
git commit -m "test: add token-budget accumulation tests, remove stale validator tests"
```

---

### Task 2: Implement all changes atomically

**Files:**
- Modify: `pallatom/train/train_config.py`
- Modify: `pallatom/train/train_loop.py`
- Modify: `pallatom/tests/train/test_train_loop.py`

All three files must change together because the config rename breaks the loop and fixtures simultaneously.

- [ ] **Step 1: Rename field in `TrainingParams` and remove validator in `TrainConfig`**

In `pallatom/train/train_config.py`:

**Change 1** — `TrainingParams` line 17:
```python
# BEFORE
accumulated_batch_size: int = Field(default=32, gt=0)

# AFTER
accumulated_token_budget: int = Field(default=2048, gt=0)
```

**Change 2** — remove the entire validator method from `TrainConfig` (the `_accumulated_batch_size_gte_loader_batch_size` method, roughly lines 161–169):
```python
# DELETE this entire block:
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

- [ ] **Step 2: Update `_process_accum_window` in `train_loop.py`**

Replace the entire function body (lines 613–664) with the protein-weighted version:

```python
def _process_accum_window(
    micro_buffer: list[ProteinBatch],
    n_proteins_per_batch: list[int],
    model: nn.Module,
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    device: str,
) -> dict[str, float]:
    """Forward + backward over one accumulation window; returns protein-weighted metrics.

    Each micro-batch's loss is divided by ``total_proteins / n_proteins_i`` so that the
    accumulated gradient is equivalent to a single large-batch backward over all proteins
    in the window. Metrics are averaged with the same protein-count weights.

    The ``no_sync()`` context manager is used on all but the last micro-batch when the model
    exposes it (DDP), so gradient all-reduces happen only once per window.

    Args:
        micro_buffer: Micro-batches to process.
        n_proteins_per_batch: Protein count per micro-batch (``batch.atom_positions.shape[0]``).
        model: Model to forward through (plain ``MainTrunk`` or DDP-wrapped).
        tcfg: Training configuration.
        distogram_res: Residue-level distogram.
        distogram_atom: Atom-level distogram.
        device: PyTorch device string.

    Returns:
        Dict of metrics averaged by protein count over all micro-batches in the window.
    """
    total_proteins: int = sum(n_proteins_per_batch)
    n_micro: int = len(micro_buffer)
    window_metrics: dict[str, float] = dict.fromkeys(_METRIC_KEYS, 0.0)
    maybe_no_sync = getattr(model, "no_sync", None)
    for micro_idx, (mb, n_proteins) in enumerate(zip(micro_buffer, n_proteins_per_batch)):
        is_last = micro_idx == n_micro - 1
        ctx = cast(
            contextlib.AbstractContextManager[None],
            (
                maybe_no_sync()
                if (not is_last and callable(maybe_no_sync))
                else contextlib.nullcontext()
            ),
        )
        grad_scale: float = total_proteins / n_proteins
        weight: float = n_proteins / total_proteins
        with ctx:
            step_metrics = train_step(
                mb,
                model,
                tcfg,
                distogram_res,
                distogram_atom,
                device,
                grad_scale=grad_scale,
            )
        for k in window_metrics:
            window_metrics[k] += step_metrics[k] * weight
    return window_metrics
```

- [ ] **Step 3: Replace the `train()` accumulation loop in `train_loop.py`**

Find lines 710–805 (approximately). Replace the `tp = tcfg.training` block through the end of the epoch body:

```python
    tp = tcfg.training
    lg = tcfg.logging
    per_rank_token_budget: int = tp.accumulated_token_budget
    log.info(
        "gradient_accumulation",
        token_budget_per_rank=per_rank_token_budget,
        global_token_budget=tp.accumulated_token_budget,
    )

    optimizer = Adam(model.parameters(), lr=tp.lr, weight_decay=tp.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=tp.num_epochs, eta_min=tp.lr * 0.01)

    best_val_loss = float("inf")
    global_step = 0
    start_epoch = 1

    if tp.resume_checkpoint is not None:
        ckpt = torch.load(tp.resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        best_val_loss = ckpt["best_val_loss"]
        global_step = ckpt["global_step"]
        start_epoch = ckpt["epoch"] + 1
        log.info("resumed from checkpoint", path=tp.resume_checkpoint, start_epoch=start_epoch)

    for epoch in range(start_epoch, tp.num_epochs + 1):
        model.train()
        epoch_metrics: dict[str, float] = dict.fromkeys(_METRIC_KEYS, 0.0)
        n_batches = 0
        micro_buffer: list[ProteinBatch] = []
        n_proteins_buffer: list[int] = []
        accum_tokens: int = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{tp.num_epochs}", leave=False)

        for batch in pbar:
            n_tokens: int = int(batch.atom_mask.any(dim=-1).sum().item())
            n_proteins: int = batch.atom_positions.shape[0]

            # Pre-flush: if adding this batch would push tokens over the budget, flush first.
            if micro_buffer and accum_tokens + n_tokens > per_rank_token_budget:
                window_metrics = _process_accum_window(
                    micro_buffer, n_proteins_buffer, model, tcfg, distogram_res, distogram_atom, device
                )
                component_norms = _component_grad_norms(model)
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
                n_proteins_buffer = []
                accum_tokens = 0
                if global_step % lg.log_interval == 0:
                    pbar.set_postfix(
                        {
                            "loss": f"{window_metrics['total loss']:.2f}",
                            "MSE_loss": f"{window_metrics['Kabsch aligned MSE loss']:.2f}",
                            "CE_loss": f"{window_metrics['Cross Entropy loss']:.2f}",
                            "smooth_lddt_loss": f"{window_metrics['smooth lddt']:.2f}",
                            "residue_distogram_loss": f"{window_metrics['Residue Distogram loss']:.2f}",
                            "atom_distogram_loss": f"{window_metrics['Atom Distogram loss']:.2f}",
                            "intermediate_loss": f"{window_metrics['Intermediate loss']:.2f}",
                            "gnorm": f"{grad_norm:.2f}",
                            **{k: f"{v:.2f}" for k, v in component_norms.items()},
                        }
                    )

            micro_buffer.append(batch)
            n_proteins_buffer.append(n_proteins)
            accum_tokens += n_tokens

            # Post-add flush: fire when this batch alone (or combined) hits the budget.
            if accum_tokens >= per_rank_token_budget:
                window_metrics = _process_accum_window(
                    micro_buffer, n_proteins_buffer, model, tcfg, distogram_res, distogram_atom, device
                )
                component_norms = _component_grad_norms(model)
                grad_norm = float(
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
                n_proteins_buffer = []
                accum_tokens = 0
                if global_step % lg.log_interval == 0:
                    pbar.set_postfix(
                        {
                            "loss": f"{window_metrics['total loss']:.2f}",
                            "MSE_loss": f"{window_metrics['Kabsch aligned MSE loss']:.2f}",
                            "CE_loss": f"{window_metrics['Cross Entropy loss']:.2f}",
                            "smooth_lddt_loss": f"{window_metrics['smooth lddt']:.2f}",
                            "residue_distogram_loss": f"{window_metrics['Residue Distogram loss']:.2f}",
                            "atom_distogram_loss": f"{window_metrics['Atom Distogram loss']:.2f}",
                            "intermediate_loss": f"{window_metrics['Intermediate loss']:.2f}",
                            "gnorm": f"{grad_norm:.2f}",
                            **{k: f"{v:.2f}" for k, v in component_norms.items()},
                        }
                    )

        if micro_buffer:
            log.warning("dropped_partial_window", n_dropped=len(micro_buffer))

        if n_batches > 0:
            scheduler.step()

        avg_train = {k: v / max(n_batches, 1) for k, v in epoch_metrics.items()}
        avg_val = evaluate(model, test_loader, tcfg, distogram_res, distogram_atom, device)
        model.train()

        best_val_loss = log_epoch(
            epoch,
            global_step,
            avg_train,
            avg_val,
            model,
            optimizer,
            scheduler,
            tcfg,
            best_val_loss,
            log,
        )
```

- [ ] **Step 4: Replace the `train_ddp()` accumulation loop in `train_loop.py`**

Find the `train_ddp()` function's setup and epoch body. Replace the `tp = tcfg.training` block:

```python
    device = device or f"cuda:{local_rank}"
    ddp_model = DDP(model, device_ids=[local_rank])

    tp = tcfg.training
    lg = tcfg.logging
    per_rank_token_budget: int = max(1, tp.accumulated_token_budget // world_size)
    if rank == 0:
        log.info(
            "gradient_accumulation",
            token_budget_per_rank=per_rank_token_budget,
            global_token_budget=tp.accumulated_token_budget,
        )
```

Replace the epoch body (from `epoch_metrics: dict[str, float]` through `if micro_buffer and rank == 0:`):

```python
        epoch_metrics: dict[str, float] = dict.fromkeys(_METRIC_KEYS, 0.0)
        n_batches = 0
        micro_buffer: list[ProteinBatch] = []
        n_proteins_buffer: list[int] = []
        accum_tokens: int = 0
        optimizer.zero_grad()

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:03d}/{tp.num_epochs}",
            leave=False,
            disable=(rank != 0),
        )

        for batch in pbar:
            n_tokens: int = int(batch.atom_mask.any(dim=-1).sum().item())
            n_proteins: int = batch.atom_positions.shape[0]

            # Pre-flush: if adding this batch would push tokens over the budget, flush first.
            if micro_buffer and accum_tokens + n_tokens > per_rank_token_budget:
                window_metrics = _process_accum_window(
                    micro_buffer, n_proteins_buffer, ddp_model, tcfg,
                    distogram_res, distogram_atom, device
                )
                if math.isnan(window_metrics["total loss"]):
                    nan_keys = [k for k, v in window_metrics.items() if math.isnan(v)]
                    if rank == 0:
                        log.warning("nan_loss", step=global_step, nan_components=nan_keys)
                component_norms = _component_grad_norms(ddp_model)
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
                n_proteins_buffer = []
                accum_tokens = 0
                if rank == 0 and global_step % lg.log_interval == 0:
                    pbar.set_postfix(
                        {
                            "loss": f"{window_metrics['total loss']:.2f}",
                            "MSE_loss": f"{window_metrics['Kabsch aligned MSE loss']:.2f}",
                            "CE_loss": f"{window_metrics['Cross Entropy loss']:.2f}",
                            "smooth_lddt_loss": f"{window_metrics['smooth lddt']:.2f}",
                            "residue_distogram_loss": f"{window_metrics['Residue Distogram loss']:.2f}",
                            "atom_distogram_loss": f"{window_metrics['Atom Distogram loss']:.2f}",
                            "intermediate_loss": f"{window_metrics['Intermediate loss']:.2f}",
                            "gnorm": f"{grad_norm:.2f}",
                            **{k: f"{v:.2f}" for k, v in component_norms.items()},
                        }
                    )

            micro_buffer.append(batch)
            n_proteins_buffer.append(n_proteins)
            accum_tokens += n_tokens

            # Post-add flush: fire when this batch alone (or combined) hits the budget.
            if accum_tokens >= per_rank_token_budget:
                window_metrics = _process_accum_window(
                    micro_buffer, n_proteins_buffer, ddp_model, tcfg,
                    distogram_res, distogram_atom, device
                )
                if math.isnan(window_metrics["total loss"]):
                    nan_keys = [k for k, v in window_metrics.items() if math.isnan(v)]
                    if rank == 0:
                        log.warning("nan_loss", step=global_step, nan_components=nan_keys)
                component_norms = _component_grad_norms(ddp_model)
                grad_norm = float(
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
                n_proteins_buffer = []
                accum_tokens = 0
                if rank == 0 and global_step % lg.log_interval == 0:
                    pbar.set_postfix(
                        {
                            "loss": f"{window_metrics['total loss']:.2f}",
                            "MSE_loss": f"{window_metrics['Kabsch aligned MSE loss']:.2f}",
                            "CE_loss": f"{window_metrics['Cross Entropy loss']:.2f}",
                            "smooth_lddt_loss": f"{window_metrics['smooth lddt']:.2f}",
                            "residue_distogram_loss": f"{window_metrics['Residue Distogram loss']:.2f}",
                            "atom_distogram_loss": f"{window_metrics['Atom Distogram loss']:.2f}",
                            "intermediate_loss": f"{window_metrics['Intermediate loss']:.2f}",
                            "gnorm": f"{grad_norm:.2f}",
                            **{k: f"{v:.2f}" for k, v in component_norms.items()},
                        }
                    )

        if micro_buffer and rank == 0:
            log.warning("dropped_partial_window", n_dropped=len(micro_buffer))

        scheduler.step()
```

Also update the `train_ddp()` docstring. Find:
```
    The ``accumulated_batch_size`` field of
    ``tcfg.training`` is divided by ``train_loader.batch_size * world_size``
    to derive the per-rank ``accum_steps``; the micro-buffer pattern is shared
    with :func:`train` via :func:`_process_accum_window`.
```
Replace with:
```
    ``tcfg.training.accumulated_token_budget`` is divided by ``world_size`` to get
    the per-rank token threshold; micro-batches accumulate until their combined token
    count hits that threshold before each optimizer step.
```

- [ ] **Step 5: Replace all `accumulated_batch_size` references in test fixtures**

Run sed to do the bulk rename:

```bash
sed -i 's/accumulated_batch_size=2/accumulated_token_budget=_BATCH_TOKENS/g' \
  /workspaces/diffusion/pallatom/tests/train/test_train_loop.py
```

Then find and update the two remaining inline `TrainingParams` calls that have multi-line forms with `accumulated_batch_size=2` (the `test_train_resume_*` tests) — verify sed caught them too. Also update `tcfg_accum` fixture:

```python
# BEFORE (in tcfg_accum fixture)
training=TrainingParams(num_epochs=1, lr=1e-4, grad_clip=1.0, accumulated_batch_size=4),

# AFTER
training=TrainingParams(num_epochs=1, lr=1e-4, grad_clip=1.0, accumulated_token_budget=2 * _BATCH_TOKENS),
```

Verify no `accumulated_batch_size` remains in the test file:

```bash
grep -n "accumulated_batch_size" /workspaces/diffusion/pallatom/tests/train/test_train_loop.py
```

Expected: no output.

- [ ] **Step 6: Run the full train test suite**

```bash
cd /workspaces/diffusion && python -m pytest pallatom/tests/train/ -x -q 2>&1 | tail -20
```

Expected: all PASS.

Behavior guarantees to verify:
- `test_train_updates_model_parameters`: `tcfg` has `accumulated_token_budget=_BATCH_TOKENS=16`. Mini-batch has 16 tokens. Post-add: `16 >= 16` → flush → params updated. ✓
- `test_train_partial_window_does_not_update_params`: `tcfg_accum` has budget `2*16=32`. 1-batch loader (16 tokens). `16 < 32`, no flush → params unchanged. ✓
- `test_train_accumulation_full_window_updates_params`: `tcfg_accum` (budget=32), 2-batch loader. Batch 1: `16 < 32`, held. Batch 2: pre-check `16+16=32 > 32`? No. Add. Post-add: `32 >= 32` → flush → params updated. ✓
- `test_train_resume_checkpoint_epoch_and_step`: asserts `global_step == 1`. 1-batch loader, budget=16, 1 flush per epoch → `global_step=1` after epoch 1. ✓
- `test_train_token_budget_preflush_fires_before_oversized_batch`: budget=24. Batch 1 (16 tokens): held. Batch 2 (16 tokens): pre-check `16+16=32 > 24` → pre-flush batch1. Add batch2 (16 tokens). Post-add: `16 >= 24`? No. End: batch2 partial, dropped. `window_sizes == [1]`. ✓

- [ ] **Step 7: Run pre-commit to catch any formatting issues**

```bash
cd /workspaces/diffusion && pre-commit run --files \
  pallatom/train/train_config.py \
  pallatom/train/train_loop.py \
  pallatom/tests/train/test_train_loop.py 2>&1 | tail -20
```

Expected: all hooks PASS. If black/ruff auto-fix, stage and include in commit.

- [ ] **Step 8: Commit**

```bash
git add pallatom/train/train_config.py pallatom/train/train_loop.py pallatom/tests/train/test_train_loop.py
git commit -m "feat: token-budget gradient accumulation with protein-weighted grad_scale"
```

---

### Task 3: Final verification

**Files:** none (read-only verification)

- [ ] **Step 1: Run the complete test suite**

```bash
cd /workspaces/diffusion && python -m pytest pallatom/tests/ -q 2>&1 | tail -10
```

Expected: all tests PASS.

- [ ] **Step 2: Confirm no `accumulated_batch_size` references remain**

```bash
grep -rn "accumulated_batch_size" /workspaces/diffusion/pallatom/ 2>&1
```

Expected: no output.
