# Triangle Attention Invariance Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add six tests to `test_pair_update.py` that prove, in three layers, that `PairUpdate` (and its Triangle Attention internals) is invariant to translation and rotation of the input coordinates.

**Architecture:** Three helpers (`compute_dij`, `random_rotation`, `apply_rotation`) enable tests at each layer: Layer 1 verifies the distance matrix is unchanged, Layer 2 verifies `TransformRBF` output is unchanged, Layer 3 verifies `PairUpdate` end-to-end output is unchanged. All six tests are flat module-level functions appended to the existing test file in a single `# Geometric Invariance` section.

**Tech Stack:** PyTorch, pytest, einops (`rearrange`, `einsum`), jaxtyping + beartype

---

### Task 1: Add geometry helpers

**Files:**
- Modify: `pallatom/tests/architecture/test_pair_update.py` — add three helpers after `mean_abs_asymmetry`

- [ ] **Step 1: Add the three helpers**

Open `pallatom/tests/architecture/test_pair_update.py`. After the `mean_abs_asymmetry` function (line 33), insert:

```python
@jaxtyped(typechecker=beartype)
def compute_dij(
    r: Float[torch.Tensor, "B N_res 3"],
) -> Float[torch.Tensor, "B N_res N_res"]:
    diff = rearrange(r, "b n d -> b n 1 d") - rearrange(r, "b n d -> b 1 n d")
    return diff.norm(dim=-1)


@jaxtyped(typechecker=beartype)
def random_rotation() -> Float[torch.Tensor, "3 3"]:
    Q, _ = torch.linalg.qr(torch.randn(3, 3))
    if Q.det() < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


@jaxtyped(typechecker=beartype)
def apply_rotation(
    r: Float[torch.Tensor, "B N_res 3"],
    R: Float[torch.Tensor, "3 3"],
) -> Float[torch.Tensor, "B N_res 3"]:
    return einsum(r, R, "b n d, d e -> b n e")
```

- [ ] **Step 2: Verify the file still imports and existing tests pass**

```bash
pytest pallatom/tests/architecture/test_pair_update.py -v --tb=short
```

Expected: all existing tests PASS, no import errors.

- [ ] **Step 3: Commit**

```bash
git add pallatom/tests/architecture/test_pair_update.py
git commit -m "test(pair_update): add geometry helpers for invariance tests"
```

---

### Task 2: Layer 1 — distance invariance tests

**Files:**
- Modify: `pallatom/tests/architecture/test_pair_update.py` — append Layer 1 tests

- [ ] **Step 1: Append a new section with the two Layer 1 tests**

At the very end of `pallatom/tests/architecture/test_pair_update.py`, add:

```python
# ---------------------------------------------------------------------------
# Geometric Invariance
# Layer 1: distance computation is invariant to translation and rotation.
# Layer 2: TransformRBF output is invariant (invariant input → invariant output).
# Layer 3: PairUpdate end-to-end output is invariant.
# ---------------------------------------------------------------------------


def test_distance_translation_invariant(r_center):
    t = torch.randn(1, 1, 3)
    with torch.no_grad():
        d_orig  = compute_dij(r_center)
        d_shift = compute_dij(r_center + t)
    assert torch.allclose(d_orig, d_shift, atol=1e-5)


def test_distance_rotation_invariant(r_center):
    R = random_rotation()
    with torch.no_grad():
        d_orig = compute_dij(r_center)
        d_rot  = compute_dij(apply_rotation(r_center, R))
    assert torch.allclose(d_orig, d_rot, atol=1e-5)
```

- [ ] **Step 2: Run the two new tests**

```bash
pytest pallatom/tests/architecture/test_pair_update.py::test_distance_translation_invariant pallatom/tests/architecture/test_pair_update.py::test_distance_rotation_invariant -v
```

Expected: both PASS. Euclidean distance is invariant to rigid-body transforms by definition.

- [ ] **Step 3: Commit**

```bash
git add pallatom/tests/architecture/test_pair_update.py
git commit -m "test(pair_update): Layer 1 — distance is translation and rotation invariant"
```

---

### Task 3: Layer 2 — RBF invariance tests

**Files:**
- Modify: `pallatom/tests/architecture/test_pair_update.py` — append Layer 2 tests

- [ ] **Step 1: Append the two Layer 2 tests directly after the Layer 1 tests**

```python
def test_rbf_translation_invariant(rbf, r_center):
    t = torch.randn(1, 1, 3)
    with torch.no_grad():
        b_orig  = rbf(compute_dij(r_center))
        b_shift = rbf(compute_dij(r_center + t))
    assert torch.allclose(b_orig, b_shift, atol=1e-5)


def test_rbf_rotation_invariant(rbf, r_center):
    R = random_rotation()
    with torch.no_grad():
        b_orig = rbf(compute_dij(r_center))
        b_rot  = rbf(compute_dij(apply_rotation(r_center, R)))
    assert torch.allclose(b_orig, b_rot, atol=1e-5)
```

- [ ] **Step 2: Run the two new tests**

```bash
pytest pallatom/tests/architecture/test_pair_update.py::test_rbf_translation_invariant pallatom/tests/architecture/test_pair_update.py::test_rbf_rotation_invariant -v
```

Expected: both PASS. `TransformRBF` is a deterministic function of distance values; invariant distances → invariant RBF output.

- [ ] **Step 3: Commit**

```bash
git add pallatom/tests/architecture/test_pair_update.py
git commit -m "test(pair_update): Layer 2 — RBF bias is translation and rotation invariant"
```

---

### Task 4: Layer 3 — PairUpdate end-to-end invariance tests

**Files:**
- Modify: `pallatom/tests/architecture/test_pair_update.py` — append Layer 3 tests

- [ ] **Step 1: Append the two Layer 3 tests directly after the Layer 2 tests**

```python
def test_pair_update_translation_invariant(pair_update, z, r_center):
    t = torch.randn(1, 1, 3)
    with torch.no_grad():
        out_orig  = pair_update(z, r_center)
        out_shift = pair_update(z, r_center + t)
    assert torch.allclose(out_orig, out_shift, atol=1e-5)


def test_pair_update_rotation_invariant(pair_update, z, r_center):
    R = random_rotation()
    with torch.no_grad():
        out_orig = pair_update(z, r_center)
        out_rot  = pair_update(z, apply_rotation(r_center, R))
    assert torch.allclose(out_orig, out_rot, atol=1e-5)
```

- [ ] **Step 2: Run the two new tests**

```bash
pytest pallatom/tests/architecture/test_pair_update.py::test_pair_update_translation_invariant pallatom/tests/architecture/test_pair_update.py::test_pair_update_rotation_invariant -v
```

Expected: both PASS. The `pair_update` fixture uses `dropout=0.0` and `.eval()` mode — no stochasticity.

- [ ] **Step 3: Run the full test file to confirm no regressions**

```bash
pytest pallatom/tests/architecture/test_pair_update.py -v
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add pallatom/tests/architecture/test_pair_update.py
git commit -m "test(pair_update): Layer 3 — PairUpdate is translation and rotation invariant"
```
