# DDP Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `train_ddp()` and `evaluate_ddp()` to `pallatom/train/train_loop.py` and a `make_ddp_data_loaders()` helper to `pallatom/helpers/data.py`, enabling `torchrun`-based multi-node / multi-GPU training while leaving the existing single-GPU `train()` unchanged.

**Architecture:** `torchrun` spawns one process per GPU; each calls `dist.init_process_group` then `train_ddp()` directly (no `mp.spawn`). Only `model` is DDP-wrapped; `index_embedding` stays a plain `nn.Embedding` with gradients manually all-reduced after each backward pass to avoid a type-annotation conflict with `featurize_batch`. `evaluate_ddp()` packs all metrics into a single tensor, calls `dist.all_reduce` once (guarded by `world_size > 1`), then returns global means.

**Tech Stack:** PyTorch DDP (`torch.nn.parallel.DistributedDataParallel`), `torch.distributed`, `torch.utils.data.distributed.DistributedSampler`, pytest + monkeypatch for unit tests.

---

## File map

| Action   | Path                                               | Responsibility                                  |
|----------|----------------------------------------------------|-------------------------------------------------|
| Modify   | `pallatom/helpers/data.py`                         | Add `make_ddp_data_loaders()`                   |
| Modify   | `pallatom/train/train_loop.py`                     | Add imports, `evaluate_ddp()`, `train_ddp()`, update `__main__` |
| Modify   | `pallatom/tests/helpers/test_data.py`              | Tests for `make_ddp_data_loaders()`             |
| Modify   | `pallatom/tests/train/test_train_loop.py`          | Tests for `evaluate_ddp()` and `train_ddp()`    |

---

## Task 1: `make_ddp_data_loaders` in `helpers/data.py`

**Files:**
- Modify: `pallatom/helpers/data.py`
- Test: `pallatom/tests/helpers/test_data.py`

- [ ] **Step 1: Write the failing tests**

Append to `pallatom/tests/helpers/test_data.py`:

```python
from helpers.data import make_ddp_data_loaders
from torch.utils.data.distributed import DistributedSampler

# -- make_ddp_data_loaders ---------------------------------------------------

def test_make_ddp_data_loaders_returns_three_loaders(jsonl_path, splits_path):
    cfg = TrainConfig()
    train_loader, val_loader, test_loader = make_ddp_data_loaders(cfg, jsonl_path, splits_path, rank=0, world_size=1)
    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None


def test_make_ddp_data_loaders_train_sampler_is_distributed(jsonl_path, splits_path):
    cfg = TrainConfig()
    train_loader, _, _ = make_ddp_data_loaders(cfg, jsonl_path, splits_path, rank=0, world_size=1)
    assert isinstance(train_loader.sampler, DistributedSampler)


def test_make_ddp_data_loaders_val_sampler_is_distributed(jsonl_path, splits_path):
    cfg = TrainConfig()
    _, val_loader, _ = make_ddp_data_loaders(cfg, jsonl_path, splits_path, rank=0, world_size=1)
    assert isinstance(val_loader.sampler, DistributedSampler)


def test_make_ddp_data_loaders_test_sampler_is_distributed(jsonl_path, splits_path):
    cfg = TrainConfig()
    _, _, test_loader = make_ddp_data_loaders(cfg, jsonl_path, splits_path, rank=0, world_size=1)
    assert isinstance(test_loader.sampler, DistributedSampler)


def test_make_ddp_data_loaders_train_sampler_shuffle_true(jsonl_path, splits_path):
    cfg = TrainConfig()
    train_loader, _, _ = make_ddp_data_loaders(cfg, jsonl_path, splits_path, rank=0, world_size=1)
    assert train_loader.sampler.shuffle is True


def test_make_ddp_data_loaders_val_sampler_shuffle_false(jsonl_path, splits_path):
    cfg = TrainConfig()
    _, val_loader, _ = make_ddp_data_loaders(cfg, jsonl_path, splits_path, rank=0, world_size=1)
    assert val_loader.sampler.shuffle is False


def test_make_ddp_data_loaders_train_sampler_rank(jsonl_path, splits_path):
    cfg = TrainConfig()
    train_loader, _, _ = make_ddp_data_loaders(cfg, jsonl_path, splits_path, rank=0, world_size=2)
    assert train_loader.sampler.rank == 0


def test_make_ddp_data_loaders_train_sampler_world_size(jsonl_path, splits_path):
    cfg = TrainConfig()
    train_loader, _, _ = make_ddp_data_loaders(cfg, jsonl_path, splits_path, rank=0, world_size=2)
    assert train_loader.sampler.num_replicas == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspaces/diffusion/pallatom && python -m pytest tests/helpers/test_data.py -k "ddp" -v 2>&1 | tail -20
```

Expected: `ImportError` or `AttributeError: module 'helpers.data' has no attribute 'make_ddp_data_loaders'`

- [ ] **Step 3: Implement `make_ddp_data_loaders` in `helpers/data.py`**

Add after the existing imports in `pallatom/helpers/data.py`:

```python
from torch.utils.data.distributed import DistributedSampler
```

Add after `make_data_loaders`:

```python
def make_ddp_data_loaders(
    cfg:         TrainConfig,
    jsonl_path:  str | Path,
    splits_path: str | Path,
    rank:        int,
    world_size:  int,
    num_workers: int = 0,
) -> tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    """Build train/val/test DataLoaders backed by DistributedSampler for DDP training."""
    with open(splits_path) as f:
        splits: dict[str, list[str]] = json.load(f)

    train_set = ProteinDataset(jsonl_path, splits["train"],   max_seq_length=cfg.train_loader.max_seq_length)
    val_set   = ProteinDataset(jsonl_path, splits["validation"], max_seq_length=cfg.test_loader.max_seq_length)
    test_set  = ProteinDataset(jsonl_path, splits["test"],    max_seq_length=cfg.test_loader.max_seq_length)

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler   = DistributedSampler(val_set,   num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler  = DistributedSampler(test_set,  num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.train_loader.batch_size,
        sampler=train_sampler, num_workers=num_workers, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=cfg.test_loader.batch_size,
        sampler=val_sampler, num_workers=num_workers, pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=cfg.test_loader.batch_size,
        sampler=test_sampler, num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /workspaces/diffusion/pallatom && python -m pytest tests/helpers/test_data.py -k "ddp" -v 2>&1 | tail -20
```

Expected: 8 PASSED

- [ ] **Step 5: Commit**

```bash
cd /workspaces/diffusion && git add pallatom/helpers/data.py pallatom/tests/helpers/test_data.py
git commit -m "feat: add make_ddp_data_loaders with DistributedSampler"
```

---

## Task 2: `evaluate_ddp` in `train_loop.py`

**Files:**
- Modify: `pallatom/train/train_loop.py`
- Test: `pallatom/tests/train/test_train_loop.py`

- [ ] **Step 1: Add new imports to `train_loop.py`**

At the top of `pallatom/train/train_loop.py`, add after existing imports:

```python
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from helpers.data import make_data_loaders, make_ddp_data_loaders
```

(Replace the existing `from helpers.data import make_data_loaders` line with the combined import above.)

- [ ] **Step 2: Write the failing tests**

First, add these two lines to the existing top-level import block in `pallatom/tests/train/test_train_loop.py` (alongside the existing `from train.train_loop import ...` line):

```python
import torch.distributed as dist_module
from train.train_loop import evaluate_ddp
```

Then append the following test functions to the bottom of `pallatom/tests/train/test_train_loop.py`:

# ---------------------------------------------------------------------------
# evaluate_ddp
# ---------------------------------------------------------------------------

def test_evaluate_ddp_returns_dict(model, loader, tcfg, distogram_res, distogram_atom, index_embedding):
    result = evaluate_ddp(0, 1, model, loader, tcfg, distogram_res, distogram_atom, index_embedding, "cpu")
    assert isinstance(result, dict)


def test_evaluate_ddp_returns_expected_keys(model, loader, tcfg, distogram_res, distogram_atom, index_embedding):
    result = evaluate_ddp(0, 1, model, loader, tcfg, distogram_res, distogram_atom, index_embedding, "cpu")
    assert set(result.keys()) == EXPECTED_EVAL_KEYS


def test_evaluate_ddp_all_values_are_floats(model, loader, tcfg, distogram_res, distogram_atom, index_embedding):
    result = evaluate_ddp(0, 1, model, loader, tcfg, distogram_res, distogram_atom, index_embedding, "cpu")
    for k, v in result.items():
        assert isinstance(v, float), f"'{k}' is {type(v)}, expected float"


def test_evaluate_ddp_all_losses_finite(model, loader, tcfg, distogram_res, distogram_atom, index_embedding):
    result = evaluate_ddp(0, 1, model, loader, tcfg, distogram_res, distogram_atom, index_embedding, "cpu")
    for k, v in result.items():
        assert v == v, f"NaN in '{k}'"
        assert v != float("inf"), f"Inf in '{k}'"


def test_evaluate_ddp_total_loss_non_negative(model, loader, tcfg, distogram_res, distogram_atom, index_embedding):
    result = evaluate_ddp(0, 1, model, loader, tcfg, distogram_res, distogram_atom, index_embedding, "cpu")
    assert result["total loss"] >= 0.0


def test_evaluate_ddp_rmsd_non_negative(model, loader, tcfg, distogram_res, distogram_atom, index_embedding):
    result = evaluate_ddp(0, 1, model, loader, tcfg, distogram_res, distogram_atom, index_embedding, "cpu")
    assert result["RMSD"] >= 0.0


def test_evaluate_ddp_sets_model_to_eval_mode(model, loader, tcfg, distogram_res, distogram_atom, index_embedding):
    model.train()
    evaluate_ddp(0, 1, model, loader, tcfg, distogram_res, distogram_atom, index_embedding, "cpu")
    assert not model.training


def test_evaluate_ddp_empty_loader_returns_zero_dict(model, tcfg, distogram_res, distogram_atom, index_embedding):
    empty = torch.utils.data.DataLoader([], batch_size=None, collate_fn=lambda x: x)
    result = evaluate_ddp(0, 1, model, empty, tcfg, distogram_res, distogram_atom, index_embedding, "cpu")
    assert all(v == 0.0 for v in result.values())


def test_evaluate_ddp_calls_all_reduce_once_when_world_size_gt_1(
    model, loader, tcfg, distogram_res, distogram_atom, index_embedding, monkeypatch
):
    calls = []
    monkeypatch.setattr(dist_module, "all_reduce", lambda t, op=None: calls.append(1))
    evaluate_ddp(0, 2, model, loader, tcfg, distogram_res, distogram_atom, index_embedding, "cpu")
    assert len(calls) == 1


def test_evaluate_ddp_skips_all_reduce_when_world_size_is_1(
    model, loader, tcfg, distogram_res, distogram_atom, index_embedding, monkeypatch
):
    calls = []
    monkeypatch.setattr(dist_module, "all_reduce", lambda t, op=None: calls.append(1))
    evaluate_ddp(0, 1, model, loader, tcfg, distogram_res, distogram_atom, index_embedding, "cpu")
    assert len(calls) == 0
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd /workspaces/diffusion/pallatom && python -m pytest tests/train/test_train_loop.py -k "evaluate_ddp" -v 2>&1 | tail -20
```

Expected: `ImportError: cannot import name 'evaluate_ddp'`

- [ ] **Step 4: Implement `evaluate_ddp` in `train_loop.py`**

Add after the existing `evaluate()` function, before `train()`:

```python
@torch.no_grad()
def evaluate_ddp(
    rank: int,
    world_size: int,
    model: nn.Module,
    loader,
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    index_embedding: nn.Embedding,
    device: str,
) -> dict[str, float]:
    """Distributed evaluation. Each rank processes its shard; metrics are all-reduced."""
    model.eval()
    metric_names = [
        "total loss",
        "Kabsch aligned MSE loss",
        "Cross Entropy loss",
        "Smooth LDDT loss",
        "Residue Distogram loss",
        "Atom Distogram loss",
        "Intermediate loss",
        "RMSD",
    ]
    # totals[:8] = metric sums, totals[8] = n_batches
    totals = torch.zeros(len(metric_names) + 1, device=device)
    lp = tcfg.loss

    for batch in loader:
        featurized_batch = featurize_batch(
            _to_protein_batch(batch), tcfg, distogram_res, distogram_atom, index_embedding, device
        )

        (
            r_denoised,
            f_seq_logits,
            residue_distogram_logits,
            atom_distogram_logits,
            intermediate_denoised_coord_stack,
            intermediate_pred_aa_logit_stack,
        ) = model(featurized_batch)

        Kabsch_aligned_MSE_loss: Float[torch.Tensor, ""] = atom_loss(
            r_denoised, featurized_batch.r_gt, featurized_batch.atom5_mask
        ).mean()
        CE_loss: Float[torch.Tensor, ""] = F.cross_entropy(
            rearrange(f_seq_logits, "b n c -> (b n) c"),
            rearrange(featurized_batch.aa_indices, "b n -> (b n)"),
        )

        gt_res_bin_idx: Int[torch.Tensor, "B N_res N_res"] = featurized_batch.gt_res_distogram.argmax(dim=-1).clamp(
            0, residue_distogram_logits.size(-1) - 1
        )
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
            r_denoised, featurized_batch.r_gt, featurized_batch.atom5_mask
        )

        K_unit = len(intermediate_denoised_coord_stack)
        intermediate_med_loss: Float[torch.Tensor, ""] = torch.tensor(0.0, device=device)
        for k_idx, intermediate_denoised_coord in enumerate(intermediate_denoised_coord_stack):
            intermediate_denoised_coord: Float[torch.Tensor, "B N_atom 3"]
            gamma_K_minus_k: float = lp.gamma ** (K_unit - k_idx - 1)
            intermediate_med_loss = (
                lp.lam * atom_loss(
                    intermediate_denoised_coord, featurized_batch.r_gt, featurized_batch.atom5_mask
                )
                + lp.alpha_0 * F.cross_entropy(
                    rearrange(intermediate_pred_aa_logit_stack[k_idx], "b n c -> (b n) c"),
                    rearrange(featurized_batch.aa_indices, "b n -> (b n)"),
                )
            )
            intermediate_med_loss += gamma_K_minus_k * intermediate_med_loss
        intermediate_med_loss = (intermediate_med_loss / max(K_unit, 1)).mean()

        total_loss: Float[torch.Tensor, ""] = (
            lp.lam       * Kabsch_aligned_MSE_loss
            + lp.alpha_0 * CE_loss
            + lp.alpha_1 * lddt_loss
            + lp.alpha_2 * residue_distogram_loss
            + lp.alpha_3 * atom_distogram_loss
            + lp.alpha_4 * intermediate_med_loss
        )

        (r_aligned,) = kabsch_align(
            featurized_batch.r_gt, r_denoised,
            weights=featurized_batch.atom5_mask.float(),
        )
        diff: Float[torch.Tensor, "B N_atom 3"] = r_denoised - r_aligned
        sq: Float[torch.Tensor, "B N_atom"] = (diff * diff).sum(dim=-1)
        m: Float[torch.Tensor, "B N_atom"] = featurized_batch.atom5_mask.float()
        rmsd: Float[torch.Tensor, ""] = ((sq * m).sum() / m.sum().clamp(min=1)).sqrt()

        totals[0] += total_loss.item()
        totals[1] += Kabsch_aligned_MSE_loss.item()
        totals[2] += CE_loss.item()
        totals[3] += lddt_loss.item()
        totals[4] += residue_distogram_loss.item()
        totals[5] += atom_distogram_loss.item()
        totals[6] += intermediate_med_loss.item()
        totals[7] += rmsd.item()
        totals[8] += 1.0

    if world_size > 1:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)

    n_batches = totals[8].item()
    return {k: totals[i].item() / max(n_batches, 1) for i, k in enumerate(metric_names)}
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /workspaces/diffusion/pallatom && python -m pytest tests/train/test_train_loop.py -k "evaluate_ddp" -v 2>&1 | tail -20
```

Expected: 11 PASSED

- [ ] **Step 6: Make sure existing evaluate tests still pass**

```bash
cd /workspaces/diffusion/pallatom && python -m pytest tests/train/test_train_loop.py -k "evaluate" -v 2>&1 | tail -20
```

Expected: all PASSED

- [ ] **Step 7: Commit**

```bash
cd /workspaces/diffusion && git add pallatom/train/train_loop.py pallatom/tests/train/test_train_loop.py
git commit -m "feat: add evaluate_ddp with single all_reduce per evaluation pass"
```

---

## Task 3: `train_ddp` in `train_loop.py`

**Files:**
- Modify: `pallatom/train/train_loop.py`
- Test: `pallatom/tests/train/test_train_loop.py`

- [ ] **Step 1: Write the failing tests**

First, add these two lines to the existing top-level import block in `pallatom/tests/train/test_train_loop.py`:

```python
import torch.nn.parallel
from train.train_loop import train_ddp
```

Then append the following class definitions and test functions to the bottom of the file:


class _FakeDDP(nn.Module):
    """Minimal DDP stand-in that exposes .module and works on CPU."""

    def __init__(self, module, device_ids=None):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class _MockSampler:
    """DataLoader sampler with set_epoch() for verifying epoch reseeding."""

    def __init__(self, data):
        self._data = data
        self.set_epoch_calls: list[int] = []

    def __iter__(self):
        return iter(range(len(self._data)))

    def __len__(self):
        return len(self._data)

    def set_epoch(self, epoch: int) -> None:
        self.set_epoch_calls.append(epoch)


@pytest.fixture
def ddp_loader(mini_batch):
    sampler = _MockSampler([mini_batch])
    loader = torch.utils.data.DataLoader([mini_batch], batch_size=None, sampler=sampler, collate_fn=lambda x: x)
    return loader


@pytest.fixture
def patch_ddp(monkeypatch):
    monkeypatch.setattr("train.train_loop.DDP", _FakeDDP)


# ---------------------------------------------------------------------------
# train_ddp
# ---------------------------------------------------------------------------

def test_train_ddp_returns_none(model, ddp_loader, tcfg, distogram_res, distogram_atom, index_embedding, patch_ddp):
    result = train_ddp(0, 0, 1, model, tcfg, ddp_loader, ddp_loader, distogram_res, distogram_atom, index_embedding, device="cpu")
    assert result is None


def test_train_ddp_rank0_saves_checkpoint(model, ddp_loader, tcfg, distogram_res, distogram_atom, index_embedding, patch_ddp):
    train_ddp(0, 0, 1, model, tcfg, ddp_loader, ddp_loader, distogram_res, distogram_atom, index_embedding, device="cpu")
    assert os.path.exists(tcfg.checkpoint.checkpoint_path)


def test_train_ddp_checkpoint_has_correct_keys(model, ddp_loader, tcfg, distogram_res, distogram_atom, index_embedding, patch_ddp):
    train_ddp(0, 0, 1, model, tcfg, ddp_loader, ddp_loader, distogram_res, distogram_atom, index_embedding, device="cpu")
    ckpt = torch.load(tcfg.checkpoint.checkpoint_path, weights_only=True)
    assert "model" in ckpt and "index_embedding" in ckpt


def test_train_ddp_rank1_does_not_save_checkpoint(model, ddp_loader, tcfg, distogram_res, distogram_atom, index_embedding, patch_ddp):
    train_ddp(1, 0, 2, model, tcfg, ddp_loader, ddp_loader, distogram_res, distogram_atom, index_embedding, device="cpu")
    assert not os.path.exists(tcfg.checkpoint.checkpoint_path)


def test_train_ddp_calls_set_epoch_each_epoch(model, ddp_loader, tcfg_multi, distogram_res, distogram_atom, index_embedding, patch_ddp):
    train_ddp(0, 0, 1, model, tcfg_multi, ddp_loader, ddp_loader, distogram_res, distogram_atom, index_embedding, device="cpu")
    assert ddp_loader.sampler.set_epoch_calls == [1, 2, 3]


def test_train_ddp_updates_model_parameters(model, ddp_loader, tcfg, distogram_res, distogram_atom, index_embedding, patch_ddp):
    params_before = [p.clone().detach() for p in model.parameters()]
    train_ddp(0, 0, 1, model, tcfg, ddp_loader, ddp_loader, distogram_res, distogram_atom, index_embedding, device="cpu")
    assert any(not torch.equal(b, a) for b, a in zip(params_before, model.parameters()))


def test_train_ddp_wandb_not_called_when_disabled(model, ddp_loader, tcfg, distogram_res, distogram_atom, index_embedding, patch_ddp, monkeypatch):
    mock_log = MagicMock()
    monkeypatch.setattr("train.train_loop.wandb.log", mock_log)
    train_ddp(0, 0, 1, model, tcfg, ddp_loader, ddp_loader, distogram_res, distogram_atom, index_embedding, device="cpu")
    mock_log.assert_not_called()


def test_train_ddp_wandb_called_when_rank0_and_enabled(model, ddp_loader, tcfg_wandb, distogram_res, distogram_atom, index_embedding, patch_ddp, monkeypatch):
    logged = []
    monkeypatch.setattr("train.train_loop.wandb.log", lambda data, step: logged.append(data))
    train_ddp(0, 0, 1, model, tcfg_wandb, ddp_loader, ddp_loader, distogram_res, distogram_atom, index_embedding, device="cpu")
    assert len(logged) == 1


def test_train_ddp_wandb_not_called_when_rank_nonzero(model, ddp_loader, tcfg_wandb, distogram_res, distogram_atom, index_embedding, patch_ddp, monkeypatch):
    mock_log = MagicMock()
    monkeypatch.setattr("train.train_loop.wandb.log", mock_log)
    train_ddp(1, 0, 2, model, tcfg_wandb, ddp_loader, ddp_loader, distogram_res, distogram_atom, index_embedding, device="cpu")
    mock_log.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspaces/diffusion/pallatom && python -m pytest tests/train/test_train_loop.py -k "train_ddp" -v 2>&1 | tail -20
```

Expected: `ImportError: cannot import name 'train_ddp'`

- [ ] **Step 3: Implement `train_ddp` in `train_loop.py`**

Add after `evaluate_ddp()`, before `_FileLogProcessor`:

```python
def train_ddp(
    rank: int,
    local_rank: int,
    world_size: int,
    model: MainTrunk,
    tcfg: TrainConfig,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    index_embedding: nn.Embedding,
    device: str | None = None,
) -> None:
    """DDP training loop. Launched via torchrun — one process per GPU."""
    device = device or f"cuda:{local_rank}"
    ddp_model = DDP(model, device_ids=[local_rank])

    tp = tcfg.training
    lp = tcfg.loss
    lg = tcfg.logging
    ck = tcfg.checkpoint

    optimizer = Adam(
        list(ddp_model.parameters()) + list(index_embedding.parameters()),
        lr=tp.lr,
        weight_decay=tp.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=tp.num_epochs, eta_min=tp.lr * 0.01)

    best_val_loss = float("inf")
    global_step   = 0

    for epoch in range(1, tp.num_epochs + 1):
        ddp_model.train()
        train_loader.sampler.set_epoch(epoch)
        epoch_total_loss = epoch_MSE = epoch_CE = epoch_smooth_lddt = epoch_res_dist = epoch_atom_dist = epoch_intermediate_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{tp.num_epochs}", leave=False, disable=(rank != 0))

        for batch in pbar:
            featurized_batch = featurize_batch(
                _to_protein_batch(batch), tcfg, distogram_res, distogram_atom, index_embedding, device
            )

            (
                r_denoised,
                f_seq_logits,
                residue_distogram_logits,
                atom_distogram_logits,
                intermediate_denoised_coord_stack,
                intermediate_pred_aa_logit_stack,
            ) = ddp_model(featurized_batch)
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
                intermediate_med_loss = (
                    lp.lam * atom_loss(
                        intermediate_denoised_coord, featurized_batch.r_gt, featurized_batch.atom5_mask
                    )
                    + lp.alpha_0 * F.cross_entropy(
                        rearrange(intermediate_pred_aa_logit_stack[k_idx], "b n c -> (b n) c"),
                        rearrange(featurized_batch.aa_indices, "b n -> (b n)"),
                    )
                )
                intermediate_med_loss += gamma_K_minus_k * intermediate_med_loss
            intermediate_med_loss = (intermediate_med_loss / max(K_unit, 1)).mean()

            gt_res_bin_idx: Int[torch.Tensor, "B N_res N_res"] = featurized_batch.gt_res_distogram.argmax(dim=-1).clamp(
                0, residue_distogram_logits.size(-1) - 1
            )
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
                r_denoised, featurized_batch.r_gt, featurized_batch.atom5_mask,
                cutoff=float(lp.smooth_lddt_cutoff),
            )
            CE_loss: Float[torch.Tensor, ""] = F.cross_entropy(
                rearrange(f_seq_logits, "b n c -> (b n) c"),
                rearrange(featurized_batch.aa_indices, "b n -> (b n)"),
            )

            total_loss: Float[torch.Tensor, ""] = (
                lp.lam       * Kabsch_aligned_MSE_loss
                + lp.alpha_0 * CE_loss
                + lp.alpha_1 * lddt_loss
                + lp.alpha_2 * residue_distogram_loss
                + lp.alpha_3 * atom_distogram_loss
                + lp.alpha_4 * intermediate_med_loss
            )

            optimizer.zero_grad()
            total_loss.backward()

            if world_size > 1:
                for param in index_embedding.parameters():
                    if param.grad is not None:
                        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                        param.grad.div_(world_size)

            grad_norm: Float[torch.Tensor, ""] = nn.utils.clip_grad_norm_(
                list(ddp_model.parameters()) + list(index_embedding.parameters()),
                tp.grad_clip if tp.grad_clip is not None else float("inf"),
            )
            optimizer.step()

            epoch_total_loss        += total_loss.item()
            epoch_MSE               += Kabsch_aligned_MSE_loss.item()
            epoch_CE                += CE_loss.item()
            epoch_smooth_lddt       += lddt_loss.item()
            epoch_res_dist          += residue_distogram_loss.item()
            epoch_atom_dist         += atom_distogram_loss.item()
            epoch_intermediate_loss += intermediate_med_loss.item()
            n_batches  += 1
            global_step += 1

            if rank == 0 and global_step % lg.log_interval == 0:
                pbar.set_postfix(loss=f"{total_loss.item():.4f}", gnorm=f"{grad_norm:.3f}")

        scheduler.step()

        avg_train = {k: v / n_batches for k, v in zip(
            ["total loss", "Kabsch aligned MSE loss", "Cross Entropy loss",
             "smooth lddt", "Residue Distogram loss", "Atom Distogram loss", "Intermediate loss"],
            [epoch_total_loss, epoch_MSE, epoch_CE, epoch_smooth_lddt,
             epoch_res_dist, epoch_atom_dist, epoch_intermediate_loss],
        )}

        avg_val = evaluate_ddp(
            rank, world_size, ddp_model, test_loader,
            tcfg, distogram_res, distogram_atom, index_embedding, device,
        )
        ddp_model.train()

        if rank == 0:
            log.info(
                "train",
                epoch=epoch,
                **{k.replace(" ", "_"): round(v, 6) for k, v in avg_train.items()},
            )
            log.info(
                "val",
                epoch=epoch,
                **{k.replace(" ", "_"): round(v, 6) for k, v in avg_val.items()},
            )

            if lg.use_wandb:
                wandb.log(
                    {
                        "epoch": epoch,
                        **{f"train/{k}": v for k, v in avg_train.items()},
                        **{f"val/{k}": v for k, v in avg_val.items()},
                    },
                    step=global_step,
                )

            if avg_val["total loss"] < best_val_loss:
                best_val_loss = avg_val["total loss"]
                torch.save(
                    {"model": ddp_model.module.state_dict(), "index_embedding": index_embedding.state_dict()},
                    ck.checkpoint_path,
                )

            if epoch % ck.save_every == 0:
                torch.save(
                    {"model": ddp_model.module.state_dict(), "index_embedding": index_embedding.state_dict()},
                    f"checkpoint_epoch_{epoch:03d}.pt",
                )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /workspaces/diffusion/pallatom && python -m pytest tests/train/test_train_loop.py -k "train_ddp" -v 2>&1 | tail -20
```

Expected: 9 PASSED

- [ ] **Step 5: Run full test suite to check for regressions**

```bash
cd /workspaces/diffusion/pallatom && python -m pytest tests/train/test_train_loop.py -v 2>&1 | tail -30
```

Expected: all existing tests PASSED, no regressions

- [ ] **Step 6: Commit**

```bash
cd /workspaces/diffusion && git add pallatom/train/train_loop.py pallatom/tests/train/test_train_loop.py
git commit -m "feat: add train_ddp with DDP model wrapping and manual embedding grad sync"
```

---

## Task 4: Update `__main__` in `train_loop.py`

**Files:**
- Modify: `pallatom/train/train_loop.py`

No unit tests for `__main__` — it is an integration entry point validated by running `torchrun`.

- [ ] **Step 1: Replace the `if __name__ == "__main__":` block**

Replace the entire existing `if __name__ == "__main__":` block (lines 342–424) with:

```python
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train PallAtom (DDP)")
    parser.add_argument("--data",        required=True,       help="path to proteins.jsonl")
    parser.add_argument("--splits",      required=True,       help="path to splits.json")
    parser.add_argument("--config",      default=None,        help="path to TrainConfig JSON (omit for defaults)")
    parser.add_argument("--log_file",    default=None,        help="path to write structured JSON log lines")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    device     = f"cuda:{local_rank}"
    torch.cuda.set_device(local_rank)

    _processors = [
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.stdlib.add_log_level,
        structlog.processors.StackInfoRenderer(),
    ]
    if args.log_file and rank == 0:
        _processors.append(_FileLogProcessor(args.log_file))
    _processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=_processors,
        wrapper_class=structlog.make_filtering_bound_logger(20),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )

    import traceback as _tb

    try:
        if args.config is not None:
            with open(args.config) as _f:
                tcfg = TrainConfig.model_validate_json(_f.read())
        else:
            tcfg = TrainConfig()

        train_loader, val_loader, _ = make_ddp_data_loaders(
            tcfg, args.data, args.splits,
            rank=rank, world_size=world_size,
            num_workers=args.num_workers,
        )

        mp = tcfg.model
        model = MainTrunk(
            f_ref_dim=mp.f_ref_dim,
            n_bins=mp.n_bins,
            n_atom_bins=tcfg.distogram_atom.n_bins,
            c_atom=mp.c_atom,
            c_pair=mp.c_pair,
            c_res=mp.c_res,
            c_atompair=mp.c_atompair,
            K_unit=mp.K_unit,
            sigma_data=tcfg.noise.sigma_data,
        ).to(device)

        dr = tcfg.distogram_res
        da = tcfg.distogram_atom
        distogram_res   = Distogram(n_bins=dr.n_bins, min_dist=dr.min_dist, max_dist=dr.max_dist, overflow_bin=True).to(device)
        distogram_atom  = Distogram(n_bins=da.n_bins, min_dist=da.min_dist, max_dist=da.max_dist).to(device)
        index_embedding = nn.Embedding(tcfg.model.max_residues, tcfg.model.c_res).to(device)

        if tcfg.training.pretrained_weights is not None:
            ckpt = torch.load(tcfg.training.pretrained_weights, map_location=device)
            model.load_state_dict(ckpt["model"])
            index_embedding.load_state_dict(ckpt["index_embedding"])
            if rank == 0:
                log.info("loaded pretrained weights", path=tcfg.training.pretrained_weights)

        if rank == 0 and tcfg.logging.use_wandb:
            wandb.init(project=tcfg.logging.wandb_project, config=tcfg.model_dump())

        train_ddp(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            model=model,
            tcfg=tcfg,
            train_loader=train_loader,
            test_loader=val_loader,
            distogram_res=distogram_res,
            distogram_atom=distogram_atom,
            index_embedding=index_embedding,
        )
    except Exception as _exc:
        log.error("fatal", error=str(_exc), traceback=_tb.format_exc())
        raise SystemExit(1) from _exc
    finally:
        dist.destroy_process_group()
```

- [ ] **Step 2: Run full test suite to verify no regressions**

```bash
cd /workspaces/diffusion/pallatom && python -m pytest tests/train/test_train_loop.py tests/helpers/test_data.py -v 2>&1 | tail -30
```

Expected: all tests PASSED

- [ ] **Step 3: Commit**

```bash
cd /workspaces/diffusion && git add pallatom/train/train_loop.py
git commit -m "feat: update __main__ to use torchrun DDP launch pattern"
```

---

## Launch reference

```bash
# Single node, N GPUs (debug)
torchrun --nproc_per_node=N pallatom/train/train_loop.py --data proteins.jsonl --splits splits.json

# Multi-node (NNODES nodes, GPUS GPUs per node)
torchrun \
  --nnodes=NNODES \
  --nproc_per_node=GPUS \
  --rdzv_backend=c10d \
  --rdzv_endpoint=MASTER_HOST:29500 \
  pallatom/train/train_loop.py --data proteins.jsonl --splits splits.json
```
