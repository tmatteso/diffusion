# FeaturizeCollate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move `featurize_batch` from the main-process training loop into a `FeaturizeCollate` callable that runs inside DataLoader worker processes, eliminating the CPU bottleneck between GPU steps.

**Architecture:** A `FeaturizeCollate` dataclass captures `tcfg`, `distogram_res`, and `distogram_atom` at construction time and exposes `__call__(batch: list[Protein]) -> FeaturizedBatch`. `ShardDataLoader.__init__` drops its three explicit loader-tuning params and reads them from `tcfg.train_loader` instead, then builds a `FeaturizeCollate` and passes it to `super().__init__`. `make_bucketed_data_loaders` constructs the distograms from `cfg` and applies the same collate to all three loaders. `take_step` receives an already-featurized `FeaturizedBatch`; every `list[Protein]` type annotation in the training loop becomes `FeaturizedBatch`.

**Tech Stack:** PyTorch DataLoader, `@dataclasses.dataclass`, `beartype` / `jaxtyping`, `pytest` fixtures.

---

## File Map

| Action | Path |
|--------|------|
| Modify | `pallatom/helpers/data.py` |
| Modify | `pallatom/train/train_loop.py` |
| Modify | `pallatom/tests/helpers/test_data.py` |
| Modify | `pallatom/tests/train/test_train_loop.py` |

---

### Task 1: Add `FeaturizeCollate` dataclass to `data.py`

**Files:**
- Modify: `pallatom/helpers/data.py` (after `featurize_batch`, around line 726)
- Test: `pallatom/tests/helpers/test_data.py`

- [ ] **Step 1: Add `FeaturizeCollate` to the import list in `test_data.py`**

In `pallatom/tests/helpers/test_data.py`, extend the `from helpers.data import (` block to include `FeaturizeCollate`:

```python
from helpers.data import (
    DatasetSplitsManifest,
    Distogram,
    FeaturizeCollate,          # add this
    FeaturizedBatch,
    FeaturizedItem,
    ProteinDataset,
    ProteinShardDataset,
    ShardBudgetParameters,
    ShardDataLoader,
    ShardMetadata,
    apply_conditioning_dropout,
    featurize_batch,
    featurize_single_item,
    make_bucketed_data_loaders,
    ref_pos_for_residue,
    sinusoidal_encoding,
)
```

- [ ] **Step 2: Write two failing tests for `FeaturizeCollate`**

Add both tests at the end of the `# featurize_batch` block in `test_data.py` (after the existing `featurize_batch` tests, around line 1090):

```python
def test_featurize_collate_returns_featurized_batch(
    featurize_protein_batch: list[Protein],
    tcfg: TrainConfig,
    c_beta_distogram_fn: Distogram,
    atom_distogram_fn: Distogram,
) -> None:
    """FeaturizeCollate.__call__ produces a FeaturizedBatch from list[Protein]."""
    collate = FeaturizeCollate(
        tcfg=tcfg,
        distogram_res=c_beta_distogram_fn,
        distogram_atom=atom_distogram_fn,
    )
    result = collate(featurize_protein_batch)
    assert isinstance(result, FeaturizedBatch)


def test_featurize_collate_is_picklable(
    tcfg: TrainConfig,
    c_beta_distogram_fn: Distogram,
    atom_distogram_fn: Distogram,
) -> None:
    """FeaturizeCollate survives a pickle round-trip (required for num_workers > 0)."""
    collate = FeaturizeCollate(
        tcfg=tcfg,
        distogram_res=c_beta_distogram_fn,
        distogram_atom=atom_distogram_fn,
    )
    restored = pickle.loads(pickle.dumps(collate))
    assert callable(restored)
```

`pickle` is already imported in `test_data.py` at the top.

- [ ] **Step 3: Run tests to confirm they fail**

```bash
pytest pallatom/tests/helpers/test_data.py::test_featurize_collate_returns_featurized_batch pallatom/tests/helpers/test_data.py::test_featurize_collate_is_picklable -v
```

Expected: `ImportError: cannot import name 'FeaturizeCollate'`

- [ ] **Step 4: Implement `FeaturizeCollate` in `data.py`**

Insert immediately after the closing `return FeaturizedBatch(...)` of `featurize_batch` (around line 726), before `apply_conditioning_dropout`:

```python
@dataclasses.dataclass
class FeaturizeCollate:
    """Picklable collation callable wrapping featurize_batch for DataLoader workers.

    Captures the three featurization dependencies so the collate_fn contract
    ``(batch: list[T]) -> CollatedT`` is satisfied. Implemented as a dataclass
    so it survives pickle round-trips required by multi-worker DataLoaders.

    Args:
        tcfg: Training configuration supplying noise schedule parameters.
        distogram_res: Residue-level Cβ distogram module.
        distogram_atom: Atom-level sparse distogram module.
    """

    tcfg: TrainConfig
    distogram_res: Distogram
    distogram_atom: Distogram

    def __call__(self, batch: list[Protein]) -> FeaturizedBatch:
        """Featurize a pre-assembled protein batch.

        Args:
            batch: List of proteins assembled by the dataset iterator.

        Returns:
            FeaturizedBatch with noisy inputs, ground-truth positions, and labels.
        """
        return featurize_batch(
            batch, self.tcfg, self.distogram_res, self.distogram_atom
        )
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
pytest pallatom/tests/helpers/test_data.py::test_featurize_collate_returns_featurized_batch pallatom/tests/helpers/test_data.py::test_featurize_collate_is_picklable -v
```

Expected: both PASS.

- [ ] **Step 6: Commit**

```bash
git add pallatom/helpers/data.py pallatom/tests/helpers/test_data.py
git commit -m "feat: add FeaturizeCollate picklable dataclass to data.py"
```

---

### Task 2: Wire `FeaturizeCollate` into `ShardDataLoader` and `make_bucketed_data_loaders`

**Files:**
- Modify: `pallatom/helpers/data.py` (lines ~1470–1488 for `ShardDataLoader.__init__`, lines ~1855–1982 for `make_bucketed_data_loaders`)
- Test: `pallatom/tests/helpers/test_data.py`

- [ ] **Step 1: Update `test_bucketed_train_loader_yields_protein_batch` to expect `FeaturizedBatch`**

Replace the existing test (`test_bucketed_train_loader_yields_protein_batch`) with:

```python
def test_bucketed_train_loader_yields_featurized_batch(
    bucketed_cfg: TrainConfig,
    bucketed_train_args: TrainArgs,
) -> None:
    """Training loader yields FeaturizedBatch objects (not raw Protein lists)."""
    train_loader, _, _ = make_bucketed_data_loaders(
        cfg=bucketed_cfg,
        extra_train_args=bucketed_train_args,
    )
    batch = next(iter(train_loader))
    assert isinstance(batch, FeaturizedBatch)
```

- [ ] **Step 2: Run the renamed test to confirm it fails**

```bash
pytest pallatom/tests/helpers/test_data.py::test_bucketed_train_loader_yields_featurized_batch -v
```

Expected: FAIL — `AssertionError` because the loader still yields `list[Protein]`.

- [ ] **Step 3: Update `ShardDataLoader.__init__` in `data.py`**

Replace the existing `__init__` signature and body (lines ~1470–1495):

```python
def __init__(
    self,
    *,
    dataset: ProteinShardDataset,
    budget: ShardBudgetParameters,
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
) -> None:
    self.shard_dataset: ProteinShardDataset = dataset
    num_workers: int = tcfg.train_loader.num_workers
    self.prefetch_epochs: int = tcfg.train_loader.epoch_prefetch_depth
    collate = FeaturizeCollate(
        tcfg=tcfg,
        distogram_res=distogram_res,
        distogram_atom=distogram_atom,
    )
    super().__init__(  # pyright: ignore[reportUnknownMemberType]
        self.shard_dataset,
        batch_size=None,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=(
            tcfg.train_loader.batch_prefetch_depth if num_workers > 0 else None
        ),
    )
    self.budget: ShardBudgetParameters = budget
    self.world_size: int = budget.world_size
    self.base_seed: int = budget.seed
    self.epoch: int = 0
```

Also update the class declaration line to reflect the new type parameter:

```python
class ShardDataLoader(torch.utils.data.DataLoader[FeaturizedBatch]):
```

And update the docstring `Args` section to replace the three removed params with the three new ones:

```
    Args:
        dataset: Pre-constructed ProteinShardDataset to stream from.
        budget: Scalar batching and shard configuration shared with the
            dataset.
        tcfg: Training configuration; supplies num_workers,
            batch_prefetch_depth, epoch_prefetch_depth, and featurization
            parameters.
        distogram_res: Residue-level Cβ distogram module used by
            FeaturizeCollate.
        distogram_atom: Atom-level sparse distogram module used by
            FeaturizeCollate.
```

- [ ] **Step 4: Update `make_bucketed_data_loaders` in `data.py`**

a. Construct distograms from `cfg` at the top of the function body (after extracting `splits`):

```python
dr = cfg.distogram_res
da = cfg.distogram_atom
distogram_res: Distogram = Distogram(
    n_bins=dr.n_bins - 1,
    min_dist=dr.min_dist,
    max_dist=dr.max_dist,
    overflow_bin=True,
).eval()
distogram_atom: Distogram = Distogram(
    n_bins=da.n_bins,
    overflow_bin=False,
    min_dist=da.min_dist,
    max_dist=da.max_dist,
).eval()
collate: FeaturizeCollate = FeaturizeCollate(
    tcfg=cfg,
    distogram_res=distogram_res,
    distogram_atom=distogram_atom,
)
```

b. Replace the `ShardDataLoader(...)` call (lines ~1922–1928):

```python
train_loader = ShardDataLoader(
    dataset=train_set,
    budget=budget,
    tcfg=cfg,
    distogram_res=distogram_res,
    distogram_atom=distogram_atom,
)
```

c. Replace all `collate_fn=identity_collate` in the val/test `DataLoader` constructions with `collate_fn=collate` (four occurrences, lines ~1943–1974).

d. Update the return type annotation:

```python
) -> tuple[
    torch.utils.data.DataLoader[FeaturizedBatch],
    torch.utils.data.DataLoader[FeaturizedBatch],
    torch.utils.data.DataLoader[FeaturizedBatch],
]:
```

e. Update the final `cast` calls to match the new type:

```python
return (
    train_loader,
    cast(torch.utils.data.DataLoader[FeaturizedBatch], val_loader),
    cast(torch.utils.data.DataLoader[FeaturizedBatch], test_loader),
)
```

- [ ] **Step 5: Run updated test and full data test suite**

```bash
pytest pallatom/tests/helpers/test_data.py::test_bucketed_train_loader_yields_featurized_batch -v
pytest pallatom/tests/helpers/test_data.py -x -q
```

Expected: both pass. The ShardDataLoader lifecycle tests (epoch increment, len, del) use `make_bucketed_data_loaders` and require no changes.

- [ ] **Step 6: Commit**

```bash
git add pallatom/helpers/data.py pallatom/tests/helpers/test_data.py
git commit -m "feat: wire FeaturizeCollate into ShardDataLoader and make_bucketed_data_loaders"
```

---

### Task 3: Update `take_step` to accept `FeaturizedBatch`

**Files:**
- Modify: `pallatom/train/train_loop.py` (lines ~190–347)
- Test: `pallatom/tests/train/test_train_loop.py`

- [ ] **Step 1: Add `featurized_batch` fixture and update imports in `test_train_loop.py`**

Add to the `from helpers.data import` block:

```python
from helpers.data import Distogram, FeaturizedBatch, featurize_batch, make_bucketed_data_loaders
```

Add the fixture immediately after `protein_batch` (around line 423):

```python
@pytest.fixture
def featurized_batch(
    protein_batch: list[Protein],
    model_params: ModelSetup,
) -> FeaturizedBatch:
    """FeaturizedBatch produced from protein_batch for take_step tests."""
    return featurize_batch(
        protein_batch,
        model_params.tcfg,
        model_params.distogram_res,
        model_params.distogram_atom,
    )
```

- [ ] **Step 2: Update all `take_step` tests to use `featurized_batch`**

In each test that passes `batch=protein_batch` to `take_step`, replace the fixture parameter `protein_batch: list[Protein]` with `featurized_batch: FeaturizedBatch` and update the call.

Tests to update (all in `pallatom/tests/train/test_train_loop.py`):

**`test_take_step_eval_outputs`** (line ~537):
```python
def test_take_step_eval_outputs(
    featurized_batch: FeaturizedBatch,
    model_params: ModelSetup,
) -> None:
    """Verify take_step eval-mode outputs are well-formed and gradient-free."""
    model_params.model.zero_grad()
    loss_metrics, tput = take_step(
        batch=featurized_batch,
        model_params=model_params,
        train_mode=False,
    )
    assert isinstance(loss_metrics, LossMetrics)
    assert isinstance(tput, ThroughputStatistics)
    for name, v in [
        ("total_loss", loss_metrics.total_loss),
        ("Kabsch_aligned_MSE_loss", loss_metrics.Kabsch_aligned_MSE_loss),
        ("CE_loss", loss_metrics.CE_loss),
        ("smooth_lddt_loss", loss_metrics.smooth_lddt_loss),
        ("res_distogram_loss", loss_metrics.res_distogram_loss),
        ("atom_distogram_loss", loss_metrics.atom_distogram_loss),
        ("intermediate_loss", loss_metrics.intermediate_loss),
        ("RMSD", loss_metrics.RMSD),
    ]:
        assert torch.isfinite(v), f"Field '{name}' is not finite: {v}"
    assert 0.0 < tput.token_pack_rate.item() <= 1.0
    assert tput.residues_per_sec.item() > 0.0
    assert tput.atoms_per_sec.item() > 0.0
    for p in model_params.model.parameters():
        assert p.grad is None
```

**`test_take_step_train_produces_gradients`** (line ~582):
```python
def test_take_step_train_produces_gradients(
    featurized_batch: FeaturizedBatch,
    model_params: ModelSetup,
) -> None:
    """take_step(train_mode=True) back-props grads into model parameters."""
    _ = model_params.model.train()
    model_params.model.zero_grad()
    _ = take_step(
        batch=featurized_batch,
        model_params=model_params,
        train_mode=True,
    )
    assert any(p.grad is not None for p in model_params.model.parameters())
```

**`test_take_step_grad_scale_halves_gradient_norm`** (line ~597): replace `protein_batch: list[Protein]` with `featurized_batch: FeaturizedBatch` in the signature and all three `take_step(batch=protein_batch, ...)` calls with `take_step(batch=featurized_batch, ...)`.

**`test_component_grad_norms_outputs`** (line ~725): same swap.

**`test_optimizer_step_outputs`** (line ~754): same swap; also update `[protein_batch]` → `[featurized_batch]` when building `micro_buffer`.

- [ ] **Step 3: Run updated tests to confirm they fail**

```bash
pytest pallatom/tests/train/test_train_loop.py::test_take_step_eval_outputs -v
```

Expected: FAIL — `beartype` raises because `take_step` still declares `batch: list[Protein]`.

- [ ] **Step 4: Update `take_step` in `train_loop.py`**

a. Remove `featurize_batch` from the `from helpers.data import` block (line ~46). Keep `apply_conditioning_dropout`, `FeaturizedBatch`, and `make_bucketed_data_loaders`.

b. Replace the `take_step` signature and first block (lines ~191–222):

```python
@jaxtyped(typechecker=beartype)
def take_step(
    *,
    batch: FeaturizedBatch,
    model_params: ModelSetup,
    train_mode: bool,
    grad_scale: float = 1.0,
) -> tuple[LossMetrics, ThroughputStatistics]:
    """Forward and backward pass for one micro-batch.

    The caller owns ``optimizer.zero_grad()``, ``clip_grad_norm_``, and
    ``optimizer.step()``.  Pass ``accum_steps`` as ``grad_scale`` so that
    accumulated gradients match a single large-batch backward.

    Args:
        batch: Pre-featurized micro-batch produced by FeaturizeCollate.
        model_params: Bundled model, optimizer, config, and device.
        train_mode: If True, enables dropout, conditioning dropout, and backward
            pass.
        grad_scale: Divide total loss by this value before backward (default
            1.0).

    Returns:
        Tuple of step-level loss metrics and throughput statistics.
    """
    t0 = time.perf_counter()

    sigma_data = model_params.tcfg.noise.sigma_data
    lp = model_params.tcfg.loss

    cpu_batch: FeaturizedBatch = batch
    if train_mode:
        cpu_batch = apply_conditioning_dropout(
            cpu_batch,
            p_distogram=model_params.tcfg.conditioning_dropout.p_distogram,
            p_atom=model_params.tcfg.conditioning_dropout.p_atom,
            p_seq=model_params.tcfg.conditioning_dropout.p_seq,
            device="cpu",
        )

    featurized_batch: FeaturizedBatch = cpu_batch.to(
        model_params.device,
        non_blocking=True,
    )
```

The rest of `take_step` (from `with StepContext(...)` onward) is unchanged.

- [ ] **Step 5: Run updated tests to confirm they pass**

```bash
pytest pallatom/tests/train/test_train_loop.py::test_take_step_eval_outputs pallatom/tests/train/test_train_loop.py::test_take_step_train_produces_gradients pallatom/tests/train/test_train_loop.py::test_take_step_grad_scale_halves_gradient_norm pallatom/tests/train/test_train_loop.py::test_component_grad_norms_outputs pallatom/tests/train/test_train_loop.py::test_optimizer_step_outputs -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add pallatom/train/train_loop.py pallatom/tests/train/test_train_loop.py
git commit -m "refactor: take_step accepts FeaturizedBatch directly, drops featurize_batch call"
```

---

### Task 4: Propagate `FeaturizedBatch` through remaining `train_loop.py` functions

**Files:**
- Modify: `pallatom/train/train_loop.py` (multiple functions)
- Test: `pallatom/tests/train/test_train_loop.py`

- [ ] **Step 1: Update `process_accum_window` test**

In `test_train_loop.py`, find the `process_accum_window` test (around line 680). Update:

a. Replace the fixture parameter `protein_batch: list[Protein]` with `featurized_batch: FeaturizedBatch`.

b. Update `_recording_take_step` internal wrapper signature:

```python
def _recording_take_step(
    *,
    batch: FeaturizedBatch,
    model_params: ModelSetup,
    train_mode: bool,
    grad_scale: float = 1.0,
) -> tuple[LossMetrics, ThroughputStatistics]:
    captured_scales.append(grad_scale)
    return _real_take_step(
        batch=batch,
        model_params=model_params,
        train_mode=train_mode,
        grad_scale=grad_scale,
    )
```

c. Update the accumulated batch:

```python
acummulated_batch = [featurized_batch, featurized_batch]
_ = process_accum_window(acummulated_batch, [1, 3], model_params)
```

- [ ] **Step 2: Update `loaders` fixture type annotation in `test_train_loop.py`**

Replace the return type annotation of the `loaders` fixture (around line 430):

```python
) -> tuple[
    torch.utils.data.DataLoader[FeaturizedBatch],
    torch.utils.data.DataLoader[FeaturizedBatch],
]:
```

- [ ] **Step 3: Run tests to confirm they currently fail or emit type errors**

```bash
pytest pallatom/tests/train/test_train_loop.py -k "process_accum" -v
```

Expected: FAIL because `process_accum_window` still declares `micro_buffer: list[list[Protein]]` and beartype rejects `list[FeaturizedBatch]`.

- [ ] **Step 4: Update `process_accum_window` in `train_loop.py`**

Change the signature from `micro_buffer: list[list[Protein]]` to `micro_buffer: list[FeaturizedBatch]`:

```python
def process_accum_window(
    micro_buffer: list[FeaturizedBatch],
    n_proteins_per_batch: list[int],
    model_params: ModelSetup,
) -> tuple[LossMetrics, ThroughputStatistics]:
```

Update the docstring `Args` to match:
```
        micro_buffer: Pre-featurized micro-batches to process.
```

- [ ] **Step 5: Update `optimizer_step` in `train_loop.py`**

Change the signature:

```python
def optimizer_step(
    micro_buffer: list[FeaturizedBatch],
    n_proteins_per_batch: list[int],
    model_params: ModelSetup,
    global_step: int,
) -> tuple[LossMetrics, ThroughputStatistics, ComponentNorms, int]:
```

- [ ] **Step 6: Update `flush_micro_buffer` in `train_loop.py`**

Change the signature:

```python
def flush_micro_buffer(
    micro_buffer: list[FeaturizedBatch],
    n_proteins_buffer: list[int],
    model_params: ModelSetup,
    step: StepProgress,
) -> StepProgress:
```

- [ ] **Step 7: Update `evaluate` in `train_loop.py`**

Change loader type and tqdm cast:

```python
def evaluate(
    loader: torch.utils.data.DataLoader[FeaturizedBatch],
    model_params: ModelSetup,
    log: FilteringBoundLogger,
) -> tuple[LossMetrics, ThroughputStatistics]:
```

Update the tqdm declaration (around line 439):

```python
pbar: tqdm[FeaturizedBatch] = tqdm(  # pylint: disable=unsubscriptable-object
    cast(Iterable[FeaturizedBatch], loader),
    desc="evaluate",
    total=len(loader),
    leave=False,
    unit="batch",
    disable=(rank != 0),
)
```

- [ ] **Step 8: Update `train` in `train_loop.py`**

a. Change both loader type annotations:

```python
def train(
    best_val_loss: Float[torch.Tensor, ""],
    train_loader: torch.utils.data.DataLoader[FeaturizedBatch],
    test_loader: torch.utils.data.DataLoader[FeaturizedBatch],
    model_params: ModelSetup,
    log: FilteringBoundLogger,
) -> None:
```

b. Change `micro_buffer` declaration (line ~764):

```python
micro_buffer: list[FeaturizedBatch] = []
```

c. Change `train_iter` declaration and cast (lines ~771–773):

```python
train_iter: Iterator[FeaturizedBatch] = iter(
    cast(Iterable[FeaturizedBatch], train_loader),
)
```

d. Replace the two lines that extract `n_proteins` and `n_all_tokens` from the old `list[Protein]` batch (lines ~797–798):

```python
n_proteins: int = batch.r_gt.shape[0]
n_all_tokens: int = int(batch.f_pseudo_beta_mask.sum().item())
```

`batch.r_gt.shape[0]` is the batch dimension B. `batch.f_pseudo_beta_mask.sum()` gives the count of non-padded residue positions across all proteins in the batch, matching what `take_step` uses for throughput accounting.

- [ ] **Step 9: Remove `Protein` import from `train_loop.py`**

Delete the line:

```python
from helpers.atom_utils import Protein
```

It is no longer referenced in any runtime expression after the type-annotation changes above.

- [ ] **Step 10: Run full train test suite**

```bash
pytest pallatom/tests/train/test_train_loop.py -x -q
```

Expected: all pass.

- [ ] **Step 11: Commit**

```bash
git add pallatom/train/train_loop.py pallatom/tests/train/test_train_loop.py
git commit -m "refactor: propagate FeaturizedBatch through all train_loop.py functions"
```

---

### Task 5: Remove `identity_collate`

**Files:**
- Modify: `pallatom/helpers/data.py` (lines ~1401–1404)

- [ ] **Step 1: Delete `identity_collate` and its comment from `data.py`**

Remove these lines entirely (around lines 1401–1404):

```python
# this will be replaced with featurize
def identity_collate(batch: list[Protein]) -> list[Protein]:
    """Pass pre-assembled protein batches through without default stacking."""
    return batch
```

- [ ] **Step 2: Run the full test suite**

```bash
pytest pallatom/tests/ -x -q
```

Expected: all pass. `identity_collate` is not imported or referenced in any test file.

- [ ] **Step 3: Run pre-commit checks**

```bash
pre-commit run --all-files
```

Expected: all hooks pass. Key ones to watch: `basedpyright` (type consistency), `ruff` (unused imports).

- [ ] **Step 4: Commit**

```bash
git add pallatom/helpers/data.py
git commit -m "chore: remove identity_collate placeholder now that FeaturizeCollate is wired in"
```
