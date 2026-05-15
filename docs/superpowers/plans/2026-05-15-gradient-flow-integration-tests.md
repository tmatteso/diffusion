# Gradient-Flow Integration Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two integration tests that prove the full training computation graph — encoder, all K_unit decoder blocks, and every loss term — produces finite nonzero gradients at every `MainTrunk` parameter after a backward pass.

**Architecture:** Test 1 lives in `test_main_trunk.py` and builds the 7-term composite loss inline, independent of `TrainConfig`. Test 2 lives in `test_train_loop.py` and calls `train_step()` end-to-end. Both use a private `_assert_submodule_grads` helper that buckets `model.named_parameters()` by top-level attribute name and asserts each bucket has ≥1 finite nonzero gradient.

**Tech Stack:** PyTorch, pytest, einops, black (line-length 100), ruff (Google docstrings, isort), pyright basic.

**Spec:** `docs/superpowers/specs/2026-05-15-gradient-flow-integration-tests-design.md`

---

## File Map

| File | Change |
|---|---|
| `pallatom/tests/architecture/test_main_trunk.py` | Extend losses import (line 8); insert helper + test after line 425 |
| `pallatom/tests/train/test_train_loop.py` | Add `Adam` import + `train_step` to existing import block; append helper + test at end of file |

---

## Task 1 — Integration test in `test_main_trunk.py`

**Files:**
- Modify: `pallatom/tests/architecture/test_main_trunk.py`

- [ ] **Step 1: Extend the losses import at line 8**

Replace:
```python
from architecture.losses import distogram_loss_atom
```
With (alphabetical isort order):
```python
from architecture.losses import atom_loss, distogram_loss_atom, distogram_loss_residue, smooth_lddt_loss
```

- [ ] **Step 2: Insert the helper and test after the line 425 comment**

The comment at line 425 reads `# you need to add integration tests here. don't be afraid to use pytest mocks.`
Insert the following block immediately after it (before the `# ---------------------------------------------------------------------------` scatter_mean header):

```python
def _assert_submodule_grads(model: MainTrunk) -> None:
    """Assert every top-level submodule has at least one finite nonzero gradient.

    Args:
        model: Trunk module after a backward pass has been called.
    """
    buckets: dict[str, list[torch.Tensor]] = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            buckets.setdefault(name.split(".")[0], []).append(param.grad)
    for prefix, grads in buckets.items():
        assert any(
            torch.isfinite(g).all().item() and g.abs().max().item() > 0 for g in grads
        ), f"submodule '{prefix}' has no finite nonzero gradients"


def test_integration_gradient_flow_composite_loss(
    model: MainTrunk, featurized_batch: FeaturizedBatch
) -> None:
    """Composite 7-term training loss propagates finite nonzero grads to every submodule."""
    model.train()
    (
        r_denoised,
        f_seq_logits,
        residue_distogram_logits,
        atom_distogram_logits,
        intermediate_denoised_coord_stack,
        intermediate_pred_aa_logit_stack,
    ) = model(featurized_batch)

    kabsch_loss = atom_loss(
        r_denoised, featurized_batch.r_gt, featurized_batch.atom5_mask
    ).mean()
    ce_loss = F.cross_entropy(
        rearrange(f_seq_logits, "b n c -> (b n) c"),
        rearrange(featurized_batch.aa_indices, "b n -> (b n)"),
    )
    lddt = smooth_lddt_loss(
        r_denoised, featurized_batch.r_gt, featurized_batch.atom5_mask, cutoff=15.0
    )
    gt_res_bin_idx = featurized_batch.gt_res_distogram.argmax(dim=-1).clamp(
        0, residue_distogram_logits.size(-1) - 1
    )
    res_distogram_loss = distogram_loss_residue(
        residue_distogram_logits, gt_res_bin_idx, featurized_batch.residue_mask
    ).mean()
    atom_distogram_loss = distogram_loss_atom(
        atom_distogram_logits,
        featurized_batch.gt_atom_distogram_sparse,
        featurized_batch.gt_atom_distogram_mask_sparse,
    ).mean()

    K_unit = len(intermediate_denoised_coord_stack)
    intermediate_loss = torch.tensor(0.0)
    for k_idx, (inter_coords, inter_logits) in enumerate(
        zip(intermediate_denoised_coord_stack, intermediate_pred_aa_logit_stack)
    ):
        gamma: float = 0.99 ** (K_unit - k_idx - 1)
        k_loss = atom_loss(
            inter_coords, featurized_batch.r_gt, featurized_batch.atom5_mask
        ) + F.cross_entropy(
            rearrange(inter_logits, "b n c -> (b n) c"),
            rearrange(featurized_batch.aa_indices, "b n -> (b n)"),
        )
        intermediate_loss = intermediate_loss + gamma * k_loss
    intermediate_loss = (intermediate_loss / max(K_unit, 1)).mean()

    total_loss = (
        kabsch_loss
        + ce_loss
        + lddt
        + res_distogram_loss
        + atom_distogram_loss
        + intermediate_loss
    )
    total_loss.backward()

    _assert_submodule_grads(model)
```

- [ ] **Step 3: Run the new test in isolation**

```bash
cd /workspaces/diffusion && python -m pytest pallatom/tests/architecture/test_main_trunk.py::test_integration_gradient_flow_composite_loss -v
```

Expected output (all submodules covered, test passes):
```
PASSED pallatom/tests/architecture/test_main_trunk.py::test_integration_gradient_flow_composite_loss
```

If it **fails** with `"submodule 'X' has no finite nonzero gradients"`, a gradient path is broken — stop and investigate before committing.

- [ ] **Step 4: Run the full architecture test suite to confirm no regressions**

```bash
cd /workspaces/diffusion && python -m pytest pallatom/tests/architecture/test_main_trunk.py -v
```

Expected: all tests pass (including the existing `test_main_trunk_gradient_flows_to_r_input`).

- [ ] **Step 5: Commit**

```bash
cd /workspaces/diffusion && git add pallatom/tests/architecture/test_main_trunk.py
git commit -m "test: add integration gradient-flow test with composite loss to test_main_trunk"
```

---

## Task 2 — Integration test in `test_train_loop.py`

**Files:**
- Modify: `pallatom/tests/train/test_train_loop.py`

- [ ] **Step 1: Add `Adam` import to the third-party block**

After line 14 (`import torch.nn.parallel`), insert:
```python
from torch.optim import Adam
```

The third-party import block should now read:
```python
import pytest
import torch
import torch.distributed as dist_module
import torch.nn as nn
import torch.nn.parallel
from torch.optim import Adam
```

- [ ] **Step 2: Add `train_step` to the `train.train_loop` import block**

The existing block (lines 25–30) reads:
```python
from train.train_loop import (
    evaluate,
    evaluate_ddp,
    train,
    train_ddp,
)
```

Replace with (alphabetical isort, `train_step` after `train_ddp`):
```python
from train.train_loop import (
    evaluate,
    evaluate_ddp,
    train,
    train_ddp,
    train_step,
)
```

- [ ] **Step 3: Append the helper and test at the end of the file (after line 1287)**

```python


# ---------------------------------------------------------------------------
# Integration: gradient flow via train_step
# ---------------------------------------------------------------------------


def _assert_submodule_grads(model: MainTrunk) -> None:
    """Assert every top-level submodule has at least one finite nonzero gradient.

    Args:
        model: Trunk module after a backward pass has been called.
    """
    buckets: dict[str, list[torch.Tensor]] = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            buckets.setdefault(name.split(".")[0], []).append(param.grad)
    for prefix, grads in buckets.items():
        assert any(
            torch.isfinite(g).all().item() and g.abs().max().item() > 0 for g in grads
        ), f"submodule '{prefix}' has no finite nonzero gradients"


def test_integration_gradient_flow_via_train_step(
    model: MainTrunk,
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    loader: torch.utils.data.DataLoader[ProteinBatch],
) -> None:
    """train_step() back-propagates finite nonzero grads to every MainTrunk submodule.

    Gradients persist on parameters after train_step() returns because zero_grad()
    is called at the *start* of the next train_step() call, not at the end of this one.
    """
    batch = next(iter(loader))
    optimizer = Adam(model.parameters(), lr=1e-4)
    train_step(batch, model, tcfg, distogram_res, distogram_atom, optimizer, device="cpu")
    _assert_submodule_grads(model)
```

- [ ] **Step 4: Run the new test in isolation**

```bash
cd /workspaces/diffusion && python -m pytest pallatom/tests/train/test_train_loop.py::test_integration_gradient_flow_via_train_step -v
```

Expected:
```
PASSED pallatom/tests/train/test_train_loop.py::test_integration_gradient_flow_via_train_step
```

If it **fails** with `"submodule 'X' has no finite nonzero gradients"`, a gradient path is silently killed inside `train_step()` — stop and investigate.

- [ ] **Step 5: Run the full train-loop test suite to confirm no regressions**

```bash
cd /workspaces/diffusion && python -m pytest pallatom/tests/train/test_train_loop.py -v
```

Expected: all existing tests pass.

- [ ] **Step 6: Run the full test suite and pre-commit checks**

```bash
cd /workspaces/diffusion && python -m pytest pallatom/tests/ -v && pre-commit run --all-files
```

Expected: all tests pass, black/ruff/pyright all clean.

- [ ] **Step 7: Commit**

```bash
cd /workspaces/diffusion && git add pallatom/tests/train/test_train_loop.py
git commit -m "test: add integration gradient-flow test via train_step to test_train_loop"
```

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task covering it |
|---|---|
| Inline composite 7-term loss (MSE, CE, lDDT, res distogram, atom distogram, intermediate ×K_unit) | Task 1 Step 2 |
| Per-submodule bucket assertion | Task 1 Step 2 (`_assert_submodule_grads`), Task 2 Step 3 |
| Test in `test_main_trunk.py` after line 425 | Task 1 Step 2 |
| `train_step()` black-box test in `test_train_loop.py` | Task 2 Steps 1–3 |
| Google-style docstrings on all functions | Both `_assert_submodule_grads` and test functions have docstrings |
| Type annotations on helper | `_assert_submodule_grads(model: MainTrunk) -> None` |
| isort-sorted imports | Steps 1 in both tasks show sorted form |
| `einops.rearrange` not `.view`/`.reshape` | All tensor reshaping uses `rearrange(...)` |
| Black line-length 100 | All lines kept ≤100; `total_loss` broken across lines |

**Placeholder scan:** No TBDs, TODOs, or vague steps. Every step shows exact code.

**Type consistency:** `_assert_submodule_grads` has the same signature in both tasks. `torch.tensor(0.0)` intermediate accumulator in Task 1 is consistent with the `.mean()` call that follows it. `.item()` used to produce Python scalars for `any()`, avoiding ambiguous tensor bool behaviour.
