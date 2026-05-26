# compute_beta Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the inline beta-calculation block in `AtomTransformer.forward` into a standalone, jaxtyped module-level function `compute_beta`, and add hard correctness tests for it.

**Architecture:** `compute_beta` is a pure, stateless function placed at module level in `atom_transformers.py` directly after `build_sparse_pairs`. `AtomTransformer.forward` replaces its ~40-line inline block with a single call. All new tests live in the existing `test_atom_transformers.py` file.

**Tech Stack:** PyTorch, einops (`rearrange`, `reduce`, `repeat`), jaxtyping + beartype, pytest.

---

## File Map

| File | Change |
|------|--------|
| `pallatom/architecture/atom_transformers.py` | Add `compute_beta` after `build_sparse_pairs` (after line 165); replace inline beta block in `AtomTransformer.forward` (lines 228–271) |
| `pallatom/tests/architecture/test_atom_transformers.py` | Add `compute_beta` to imports; add 9 unit tests; add 2 `AtomTransformer` block tests |

---

### Task 1: Write failing `compute_beta` tests

**Files:**
- Modify: `pallatom/tests/architecture/test_atom_transformers.py`

- [ ] **Step 1: Add `compute_beta` to the import block**

The existing import block at lines 6–13 becomes:

```python
from architecture.atom_transformers import (
    AtomAttentionDecoder,
    AtomFeatureEncoder,
    AtomTransformer,
    ConditionedTransitionBlock,
    DiffusionTransformer,
    build_sparse_pairs,
    compute_beta,
)
```

- [ ] **Step 2: Run tests to confirm the import fails**

```bash
pytest pallatom/tests/architecture/test_atom_transformers.py -x -q 2>&1 | head -5
```

Expected: `ImportError: cannot import name 'compute_beta' from 'architecture.atom_transformers'`

- [ ] **Step 3: Add a new `compute_beta` section after the `build_sparse_pairs` tests (after line 284)**

Insert before the `# ConditionedTransitionBlock` comment block:

```python
# ---------------------------------------------------------------------------
# compute_beta
# ---------------------------------------------------------------------------


def test_compute_beta_output_shape(
    neighbor_idx: Int[torch.Tensor, "N_atom K"],
    valid_mask: Bool[torch.Tensor, "B N_atom K"],
) -> None:
    """compute_beta returns a float tensor of shape [B, N_atom, K]."""
    ref = torch.randn(B, N_ATOM, C_ATOM)
    beta = compute_beta(neighbor_idx, valid_mask, n_queries=32, n_keys=128, ref=ref)
    assert beta.shape == (B, N_ATOM, K)


def test_compute_beta_values_are_binary(
    neighbor_idx: Int[torch.Tensor, "N_atom K"],
    valid_mask: Bool[torch.Tensor, "B N_atom K"],
) -> None:
    """Every element of beta is exactly 0.0 or -1e10; no intermediate values."""
    ref = torch.randn(B, N_ATOM, C_ATOM)
    beta = compute_beta(neighbor_idx, valid_mask, n_queries=32, n_keys=128, ref=ref)
    assert ((beta == 0.0) | (beta == -1e10)).all()


def test_compute_beta_zero_for_near_atoms() -> None:
    """An atom pair sharing a window centre receives beta=0.0.

    n_queries=4, n_keys=8: centre 0 at 1.5 (half_q=2.0, half_k=4.0).
    l=0: |0-1.5|=1.5 < 2.0 → query; m=1: |1-1.5|=0.5 < 4.0 → key → beta=0.0.
    """
    N_t, n_q, n_k, B_t = 8, 4, 8, 1
    neighbor_idx_t = repeat(torch.arange(N_t), "k -> n k", n=N_t)
    valid_mask_t = torch.ones(B_t, N_t, N_t, dtype=torch.bool)
    ref = torch.zeros(B_t, N_t, C_ATOM)
    beta = compute_beta(neighbor_idx_t, valid_mask_t, n_q, n_k, ref)
    assert beta[0, 0, 1].item() == 0.0


def test_compute_beta_large_neg_for_far_atoms() -> None:
    """An atom pair from disjoint windows receives beta=-1e10.

    n_queries=n_keys=4: centres at 1.5 and 5.5 (half_q=half_k=2.0).
    l=0 is a query for centre 0 (|0-1.5|=1.5<2.0).
    m=7: |7-1.5|=5.5 > 2.0 → not a key for centre 0.
    l=0: |0-5.5|=5.5 > 2.0 → not a query for centre 1.
    No centre admits this pair → beta=-1e10.
    """
    N_t, n_q, n_k, B_t = 8, 4, 4, 1
    neighbor_idx_t = repeat(torch.arange(N_t), "k -> n k", n=N_t)
    valid_mask_t = torch.ones(B_t, N_t, N_t, dtype=torch.bool)
    ref = torch.zeros(B_t, N_t, C_ATOM)
    beta = compute_beta(neighbor_idx_t, valid_mask_t, n_q, n_k, ref)
    assert beta[0, 0, 7].item() == -1e10


def test_compute_beta_valid_mask_forces_neg() -> None:
    """valid_mask=False gives -1e10 even if the pair is geometrically in-window."""
    N_t, n_q, n_k, B_t = 8, 4, 8, 1
    neighbor_idx_t = repeat(torch.arange(N_t), "k -> n k", n=N_t)
    valid_mask_t = torch.ones(B_t, N_t, N_t, dtype=torch.bool)
    valid_mask_t[0, 0, 1] = False  # pair (l=0, m=1) is in-window but masked
    ref = torch.zeros(B_t, N_t, C_ATOM)
    beta = compute_beta(neighbor_idx_t, valid_mask_t, n_q, n_k, ref)
    assert beta[0, 0, 1].item() == -1e10  # mask wins over geometry
    assert beta[0, 0, 0].item() == 0.0   # unmasked in-window pair is unaffected


def test_compute_beta_boundary_exclusive() -> None:
    """The window boundary is strict (<): the atom just outside is excluded.

    With n_queries=n_keys=4, centres are at 1.5 and 5.5 (half_q=half_k=2.0).
    Atom 3 is the last query for centre 0; atom 4 is the first query for centre 1.
    Pair (l=3, m=4):
      centre 0: l=3 is query (|3-1.5|=1.5<2), m=4 is NOT key (|4-1.5|=2.5>2)
      centre 1: l=3 is NOT query (|3-5.5|=2.5>2)
    → beta=-1e10.
    Pair (l=3, m=3): centre 0 admits both → beta=0.0.
    """
    N_t, n_q, n_k, B_t = 8, 4, 4, 1
    neighbor_idx_t = repeat(torch.arange(N_t), "k -> n k", n=N_t)
    valid_mask_t = torch.ones(B_t, N_t, N_t, dtype=torch.bool)
    ref = torch.zeros(B_t, N_t, C_ATOM)
    beta = compute_beta(neighbor_idx_t, valid_mask_t, n_q, n_k, ref)
    assert beta[0, 3, 4].item() == -1e10
    assert beta[0, 3, 3].item() == 0.0


def test_compute_beta_cross_window_centre_admits_pair() -> None:
    """A pair admitted by a neighbouring centre (not the query's own centre) gets 0.0.

    n_queries=4, n_keys=8 (half_q=2.0, half_k=4.0). Centres at 1.5 and 5.5.
    Pair (l=4, m=3):
      centre 0: l=4 is NOT a query (|4-1.5|=2.5>2.0)
      centre 1: l=4 IS a query (|4-5.5|=1.5<2.0), m=3 IS a key (|3-5.5|=2.5<4.0)
    → beta=0.0 because centre 1 admits both. Tests multi-centre reduce logic.
    """
    N_t, n_q, n_k, B_t = 8, 4, 8, 1
    neighbor_idx_t = repeat(torch.arange(N_t), "k -> n k", n=N_t)
    valid_mask_t = torch.ones(B_t, N_t, N_t, dtype=torch.bool)
    ref = torch.zeros(B_t, N_t, C_ATOM)
    beta = compute_beta(neighbor_idx_t, valid_mask_t, n_q, n_k, ref)
    assert beta[0, 4, 3].item() == 0.0


def test_compute_beta_all_zero_when_n_leq_n_queries(
    neighbor_idx: Int[torch.Tensor, "N_atom K"],
    valid_mask: Bool[torch.Tensor, "B N_atom K"],
) -> None:
    """When N <= n_queries every valid pair is in the single window → no -1e10 entries.

    N_ATOM=24 < n_queries=32: single centre at 15.5 (half_q=16.0) covers all atoms.
    n_keys=128, half_k=64.0: all atoms are keys too. Every valid pair gets beta=0.0.
    """
    ref = torch.randn(B, N_ATOM, C_ATOM)
    beta = compute_beta(neighbor_idx, valid_mask, n_queries=32, n_keys=128, ref=ref)
    assert (beta[valid_mask] == 0.0).all()


def test_compute_beta_wrong_shape_raises() -> None:
    """A 1-D neighbor_idx (missing K dimension) triggers TypeCheckError."""
    neighbor_idx_bad = torch.zeros(N_ATOM, dtype=torch.long)  # 1-D, not [N, K]
    valid_mask_t = torch.ones(B, N_ATOM, K, dtype=torch.bool)
    ref = torch.zeros(B, N_ATOM, C_ATOM)
    with pytest.raises(TypeCheckError):
        compute_beta(neighbor_idx_bad, valid_mask_t, 32, 128, ref)
```

- [ ] **Step 4: Run the compute_beta tests to confirm they all fail with ImportError**

```bash
pytest pallatom/tests/architecture/test_atom_transformers.py -k "compute_beta" -v 2>&1 | head -20
```

Expected: all 9 tests fail with `ImportError` (not yet defined).

---

### Task 2: Implement `compute_beta`

**Files:**
- Modify: `pallatom/architecture/atom_transformers.py`

- [ ] **Step 1: Insert `compute_beta` after `build_sparse_pairs` (after line 165)**

Add the following block between the closing `return neighbor_idx, valid_mask` of `build_sparse_pairs` and the `# ---------------------------------------------------------------------------` comment for `AtomTransformer`:

```python
@jaxtyped(typechecker=beartype)
def compute_beta(
    neighbor_idx: Int[torch.Tensor, "N K"],
    valid_mask: Bool[torch.Tensor, "B N K"],
    n_queries: int,
    n_keys: int,
    ref: Float[torch.Tensor, "B N c"],
) -> Float[torch.Tensor, "B N K"]:
    """Compute the sliding-window attention bias β (Algorithm 7, line 1).

    Returns 0.0 for atom pairs (l, m) admitted by at least one query/key window
    centre and by valid_mask; returns -1e10 everywhere else.

    Window centres are placed at ``c * n_queries + (n_queries / 2 − 0.5)``
    for ``c = 0, 1, …, ⌈N / n_queries⌉ − 1``.

    Args:
        neighbor_idx: Sparse neighbour indices [N, K].
        valid_mask: True where neighbour slot is a real atom [B, N, K].
        n_queries: Query-window full-width in atoms.
        n_keys: Key-window full-width in atoms.
        ref: Float tensor whose device and dtype are used for the output [B, N, c].

    Returns:
        Additive attention bias [B, N, K]; 0.0 for admitted pairs, -1e10 elsewhere.
    """
    B, N = ref.shape[0], ref.shape[1]
    K = neighbor_idx.shape[1]
    half_q = n_queries / 2.0
    half_k = n_keys / 2.0

    n_centres = math.ceil(N / n_queries)
    centres: Float[torch.Tensor, "n_centres"] = (
        torch.arange(n_centres, device=ref.device, dtype=torch.float32) * n_queries
        + (half_q - 0.5)
    )

    atom_idx: Float[torch.Tensor, "B N"] = repeat(
        torch.arange(N, device=ref.device, dtype=torch.float32), "N -> B N", B=B
    )
    m_idx: Float[torch.Tensor, "B N K"] = atom_idx[:, neighbor_idx]

    l_in_window: Bool[torch.Tensor, "B N n_centres"] = (
        rearrange(atom_idx, "B N -> B N 1") - rearrange(centres, "c -> 1 1 c")
    ).abs() < half_q
    m_in_window: Bool[torch.Tensor, "B N K n_centres"] = (
        rearrange(m_idx, "B N K -> B N K 1") - rearrange(centres, "c -> 1 1 1 c")
    ).abs() < half_k
    both_in_window: Bool[torch.Tensor, "B N K n_centres"] = (
        rearrange(l_in_window, "B N c -> B N 1 c") & m_in_window
    )
    in_window: Bool[torch.Tensor, "B N K"] = reduce(
        both_in_window.float(), "B N K c -> B N K", "max"
    ).bool()

    return torch.where(
        in_window & valid_mask,
        torch.zeros(B, N, K, device=ref.device, dtype=ref.dtype),
        torch.full((B, N, K), -1e10, device=ref.device, dtype=ref.dtype),
    )
```

- [ ] **Step 2: Run the nine compute_beta tests**

```bash
pytest pallatom/tests/architecture/test_atom_transformers.py -k "compute_beta" -v
```

Expected: all 9 tests PASS.

---

### Task 3: Update `AtomTransformer.forward`

**Files:**
- Modify: `pallatom/architecture/atom_transformers.py`

- [ ] **Step 1: Replace the inline beta block (lines 228–271) with a single call**

Delete everything from `B = q.shape[0]` down to and including the closing `beta = torch.where(...)` expression. Replace the entire block with:

```python
        # Algorithm 7 line 1: β_lm — sliding-window attention bias
        beta: Float[torch.Tensor, "B N K"] = compute_beta(
            neighbor_idx, valid_mask, self.n_queries, self.n_keys, q
        )
```

The complete updated `forward` method body (after the docstring) becomes:

```python
        # Algorithm 7 line 1: β_lm — sliding-window attention bias
        beta: Float[torch.Tensor, "B N K"] = compute_beta(
            neighbor_idx, valid_mask, self.n_queries, self.n_keys, q
        )

        # Algorithm 7 line 2: DiffusionTransformer — n_blocks rounds of attention + transition
        return self.blocks(q, c, p, beta, neighbor_idx=neighbor_idx)
```

- [ ] **Step 2: Run the full test file to confirm no regressions**

```bash
pytest pallatom/tests/architecture/test_atom_transformers.py -v
```

Expected: all previously passing tests still PASS (no regressions).

---

### Task 4: Write and run the `AtomTransformer` block tests

**Files:**
- Modify: `pallatom/tests/architecture/test_atom_transformers.py`

- [ ] **Step 1: Add the two block tests after `test_atom_transformer_gradient_flows`**

Insert the following two tests directly after `test_atom_transformer_gradient_flows` (around line 430), before the `AtomFeatureEncoder` section:

```python
def test_atom_transformer_block_isolation() -> None:
    """Block-0 output is unchanged when block-1 inputs are scrambled.

    With n_queries=n_keys=4 and N=8, blocks [0..3] and [4..7] map to disjoint
    window centres (1.5 and 5.5). Since exp(-1e10)=0.0 in float32, cross-block
    attention weights are exactly zero, so block-1 inputs cannot affect block-0 outputs.
    """
    n_q, N_t, B_t = 4, 8, 1
    model = AtomTransformer(C_ATOM, C_ATOMPAIR, n_blocks=1, n_heads=1,
                             n_queries=n_q, n_keys=n_q).eval()
    neighbor_idx_t = repeat(torch.arange(N_t), "k -> n k", n=N_t)
    valid_mask_t = torch.ones(B_t, N_t, N_t, dtype=torch.bool)

    torch.manual_seed(0)
    q = torch.randn(B_t, N_t, C_ATOM)
    c = torch.randn(B_t, N_t, C_ATOM)
    p = torch.randn(B_t, N_t, N_t, C_ATOMPAIR)

    with torch.no_grad():
        out1 = model(q, c, p, neighbor_idx_t, valid_mask_t)

    torch.manual_seed(1)
    q_alt = q.clone()
    c_alt = c.clone()
    q_alt[:, n_q:, :] = torch.randn(B_t, n_q, C_ATOM)
    c_alt[:, n_q:, :] = torch.randn(B_t, n_q, C_ATOM)

    with torch.no_grad():
        out2 = model(q_alt, c_alt, p, neighbor_idx_t, valid_mask_t)

    assert torch.equal(out1[:, :n_q, :], out2[:, :n_q, :])


def test_atom_transformer_gradient_isolation() -> None:
    """Backpropping from block-0 outputs produces zero gradient for block-1 inputs.

    With n_queries=n_keys=4 and N=8, the two blocks are disjoint. Softmax weights
    for cross-block pairs are exactly 0.0 in float32, so the softmax gradient
    (s_lm * (delta - s_lm) = 0 when s_lm=0) propagates zero back to block-1 inputs.
    ConditionedTransitionBlock is pointwise, so it contributes no cross-block gradient.
    """
    n_q, N_t, B_t = 4, 8, 1
    model = AtomTransformer(C_ATOM, C_ATOMPAIR, n_blocks=1, n_heads=1,
                             n_queries=n_q, n_keys=n_q).eval()
    neighbor_idx_t = repeat(torch.arange(N_t), "k -> n k", n=N_t)
    valid_mask_t = torch.ones(B_t, N_t, N_t, dtype=torch.bool)

    torch.manual_seed(0)
    q = torch.randn(B_t, N_t, C_ATOM, requires_grad=True)
    c = torch.randn(B_t, N_t, C_ATOM, requires_grad=True)
    p = torch.randn(B_t, N_t, N_t, C_ATOMPAIR)

    out = model(q, c, p, neighbor_idx_t, valid_mask_t)
    out[:, :n_q, :].sum().backward()

    assert q.grad is not None
    assert c.grad is not None
    assert q.grad[:, n_q:, :].abs().max().item() == 0.0
    assert c.grad[:, n_q:, :].abs().max().item() == 0.0
```

- [ ] **Step 2: Run the two new tests**

```bash
pytest pallatom/tests/architecture/test_atom_transformers.py -k "block_isolation or gradient_isolation" -v
```

Expected: both tests PASS.

---

### Task 5: Full suite, pre-commit, and commit

- [ ] **Step 1: Run the complete test file**

```bash
pytest pallatom/tests/architecture/test_atom_transformers.py -v
```

Expected: all tests PASS (no regressions).

- [ ] **Step 2: Run all pre-commit hooks**

```bash
pre-commit run --all-files
```

Expected: all hooks pass — black, ruff, pyright, enforce-einops, enforce-jaxtyping, pytest.

- [ ] **Step 3: Commit**

```bash
git add pallatom/architecture/atom_transformers.py \
        pallatom/tests/architecture/test_atom_transformers.py
git commit -m "refactor: extract compute_beta from AtomTransformer and add hard tests"
```

---

## Self-Review

**Spec coverage:**
- `compute_beta` module-level function with `ref` for dtype/device → Task 2 ✓
- `AtomTransformer.forward` calls `compute_beta` → Task 3 ✓
- 9 compute_beta unit tests → Task 1 ✓
- `test_compute_beta_output_shape` → Task 1 ✓
- `test_compute_beta_values_are_binary` → Task 1 ✓
- `test_compute_beta_zero_for_near_atoms` → Task 1 ✓
- `test_compute_beta_large_neg_for_far_atoms` → Task 1 ✓
- `test_compute_beta_valid_mask_forces_neg` → Task 1 ✓
- `test_compute_beta_boundary_exclusive` → Task 1 ✓
- `test_compute_beta_cross_window_centre_admits_pair` → Task 1 ✓
- `test_compute_beta_all_zero_when_n_leq_n_queries` → Task 1 ✓
- `test_compute_beta_wrong_shape_raises` → Task 1 ✓
- Block-isolation test → Task 4 ✓
- Gradient-isolation test → Task 4 ✓

**Placeholder scan:** No TBDs, TODOs, or vague steps. Every step has full code.

**Type consistency:** `compute_beta` signature is identical in Task 1 (import) and Task 2 (definition). The `n_queries` and `n_keys` args passed in Task 3 (`self.n_queries`, `self.n_keys`) match the `int` parameters defined in Task 2. The `ref` shape `"B N c"` matches across both usages.
