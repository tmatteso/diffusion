# jaxtyping Shape-Contract Enforcement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a CI gate that fails if any `@jaxtyped`-decorated function lacks a test, and add one wrong-shape negative test per decorated function/class to prove beartype enforcement is live.

**Architecture:** A new `test_jaxtyped_coverage.py` uses pure-AST analysis to audit every source file and verify each decorated name is referenced in its test counterpart. Negative tests are added inline to existing test files — one per `@jaxtyped` site — each passing a deliberately wrong-shaped tensor or wrong-typed argument and asserting `BeartypeCallHintParamViolation`.

**Tech Stack:** `pytest`, `beartype.roar.BeartypeCallHintParamViolation`, `ast`, `pathlib`, `torch`, `numpy`

---

## File map

| File | Action |
|---|---|
| `pallatom/tests/test_jaxtyped_coverage.py` | **Create** — CI gate |
| `pallatom/tests/architecture/test_atom_transformers.py` | Modify — +5 negative tests, add `BeartypeCallHintParamViolation` import |
| `pallatom/tests/architecture/test_losses.py` | Modify — +6 tests, add `BeartypeCallHintParamViolation`, `med_loss_per_block`, `_pairwise_dist` imports |
| `pallatom/tests/architecture/test_node_update.py` | Modify — +3 tests, add `AdaLN`, `BeartypeCallHintParamViolation` imports |
| `pallatom/tests/architecture/test_pair_update.py` | Modify — +4 tests, add `BeartypeCallHintParamViolation` import |
| `pallatom/tests/architecture/test_pairformer_stack.py` | Modify — +4 tests, add `BeartypeCallHintParamViolation` import |
| `pallatom/tests/architecture/test_main_trunk.py` | Modify — +3 tests, add `BeartypeCallHintParamViolation` import |
| `pallatom/tests/architecture/test_template_embedder.py` | Modify — +1 test, add `BeartypeCallHintParamViolation` import |
| `pallatom/tests/helpers/test_alignment.py` | Modify — +5 tests, add `BeartypeCallHintParamViolation`, full alignment imports |
| `pallatom/tests/helpers/test_atom_utils.py` | Modify — +5 tests, add `BeartypeCallHintParamViolation`, `np` import |
| `pallatom/tests/helpers/test_featurize.py` | Modify — +9 tests, add `BeartypeCallHintParamViolation`, several featurize imports |
| `pallatom/tests/sample/test_sampling.py` | Modify — +11 tests, add `BeartypeCallHintParamViolation` import |
| `pallatom/tests/train/test_train_loop.py` | Modify — +2 tests, add `BeartypeCallHintParamViolation` import |

---

### Task 1: CI gate — `test_jaxtyped_coverage.py`

**Files:**
- Create: `pallatom/tests/test_jaxtyped_coverage.py`

- [ ] **Step 1: Create the file**

```python
"""CI gate: every @jaxtyped-decorated function or class must be referenced in its test file."""

import ast
import pathlib


def _is_jaxtyped(decorator: ast.expr) -> bool:
    """Return True if decorator node is @jaxtyped or @jaxtyped(...)."""
    if isinstance(decorator, ast.Name):
        return decorator.id == "jaxtyped"
    if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
        return decorator.func.id == "jaxtyped"
    return False


class _JaxtypedCollector(ast.NodeVisitor):
    """Walk an AST and collect the lookup name for every @jaxtyped-decorated node.

    Rules:
    - Decorated standalone function ``foo`` → lookup name ``"foo"``.
    - Decorated method inside ``ClassName`` → lookup name ``"ClassName"``.
    - Decorated class (dataclass) ``MyClass`` → lookup name ``"MyClass"``.
    """

    def __init__(self) -> None:
        """Initialise collector with empty class stack and result set."""
        self._class_stack: list[str] = []
        self.lookup_names: set[str] = set()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Record class name if @jaxtyped; recurse into body for methods."""
        if any(_is_jaxtyped(d) for d in node.decorator_list):
            self.lookup_names.add(node.name)
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Record enclosing class for methods, function name for module-level functions."""
        if any(_is_jaxtyped(d) for d in node.decorator_list):
            if self._class_stack:
                self.lookup_names.add(self._class_stack[-1])
            else:
                self.lookup_names.add(node.name)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef


def _names_referenced_in(test_path: pathlib.Path) -> set[str]:
    """Return all identifiers referenced in a test file.

    Includes both ``ast.Name`` nodes (variables, calls, annotations in code)
    and ``ast.alias`` names (import statements).  Comments and string literals
    are excluded because they don't appear as AST nodes.
    """
    tree = ast.parse(test_path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.alias):
            names.add(node.name.split(".")[-1])
    return names


def test_every_jaxtyped_function_has_a_test() -> None:
    """Every @jaxtyped-decorated function or class must appear in the corresponding test file.

    Scans all source files under ``pallatom/`` (excluding ``tests/``), collects
    every @jaxtyped-decorated function or class, derives the expected test file
    path (``pallatom/{pkg}/{mod}.py`` → ``pallatom/tests/{pkg}/test_{mod}.py``),
    and asserts that the decorated name appears as an AST identifier in that file.
    Fails with a complete list of missing entries so all gaps can be fixed at once.
    """
    pallatom_root = pathlib.Path(__file__).parent.parent
    missing: list[str] = []

    for src_path in sorted(pallatom_root.rglob("*.py")):
        if "tests" in src_path.parts:
            continue

        collector = _JaxtypedCollector()
        collector.visit(ast.parse(src_path.read_text()))
        if not collector.lookup_names:
            continue

        rel = src_path.relative_to(pallatom_root)
        test_path = pallatom_root / "tests" / rel.parent / f"test_{rel.name}"

        if not test_path.exists():
            missing.extend(
                f"{src_path.name}: '{name}' (test file {test_path} not found)"
                for name in sorted(collector.lookup_names)
            )
            continue

        referenced = _names_referenced_in(test_path)
        for name in sorted(collector.lookup_names):
            if name not in referenced:
                missing.append(f"{src_path.name}: '{name}'")

    assert not missing, (
        "The following @jaxtyped functions/classes have no reference in their test file:\n"
        + "\n".join(f"  {m}" for m in missing)
    )
```

- [ ] **Step 2: Run to verify it passes (all 56 already covered)**

```
cd /workspaces/diffusion && python -m pytest pallatom/tests/test_jaxtyped_coverage.py -v
```

Expected: `PASSED pallatom/tests/test_jaxtyped_coverage.py::test_every_jaxtyped_function_has_a_test`

- [ ] **Step 3: Commit**

```bash
git add pallatom/tests/test_jaxtyped_coverage.py
git commit -m "test: add CI gate for @jaxtyped coverage"
```

---

### Task 2: Negative tests — `atom_transformers.py` (5 functions)

**Files:**
- Modify: `pallatom/tests/architecture/test_atom_transformers.py`

The 5 decorated functions: `ConditionedTransitionBlock.forward`, `DiffusionTransformer.forward`,
`AtomTransformer.forward`, `AtomFeatureEncoder.forward`, `AtomAttentionDecoder.forward`.

Strategy: pass the first tensor argument with wrong `ndim` (2D instead of 3D). beartype
raises on the first check before touching the model.

- [ ] **Step 1: Add `BeartypeCallHintParamViolation` to imports and append 5 tests**

Add to the imports block at the top of `test_atom_transformers.py`:

```python
from beartype.roar import BeartypeCallHintParamViolation
```

Append to the bottom of `test_atom_transformers.py`:

```python
# ---------------------------------------------------------------------------
# Shape-contract enforcement — negative tests
# ---------------------------------------------------------------------------


def test_conditioned_transition_block_forward_wrong_shape(
    conditioned_transition_block: ConditionedTransitionBlock,
) -> None:
    """Wrong a ndim (2-D instead of 3-D) triggers BeartypeCallHintParamViolation."""
    a_bad = torch.zeros(B, N_RES)  # missing c_a dim
    s = torch.zeros(B, N_RES, C_ATOM)
    with pytest.raises(BeartypeCallHintParamViolation):
        conditioned_transition_block(a_bad, s)


def test_diffusion_transformer_forward_wrong_shape(
    diffusion_transformer: DiffusionTransformer,
) -> None:
    """Wrong a ndim (2-D instead of 3-D) triggers BeartypeCallHintParamViolation."""
    a_bad = torch.zeros(B, N_RES)  # missing c_a dim
    s = torch.zeros(B, N_RES, C_ATOM)
    z = torch.zeros(B, N_RES, N_RES, C_PAIR)
    with pytest.raises(BeartypeCallHintParamViolation):
        diffusion_transformer(a_bad, s, z)


def test_atom_transformer_forward_wrong_shape(transformer: AtomTransformer) -> None:
    """Wrong q ndim (2-D instead of 3-D) triggers BeartypeCallHintParamViolation."""
    tok_idx = torch.repeat_interleave(torch.arange(N_RES), ATOMS_PER_RES)
    nbrs, mask = build_sparse_pairs(tok_idx)
    q_bad = torch.zeros(B, N_ATOM)  # missing c_atom dim
    c = torch.zeros(B, N_ATOM, C_ATOM)
    p = torch.zeros(B, N_ATOM, K, C_ATOMPAIR)
    with pytest.raises(BeartypeCallHintParamViolation):
        transformer(q_bad, c, p, nbrs, mask.unsqueeze(0).expand(B, -1, -1))


def test_atom_feature_encoder_forward_wrong_shape(encoder: AtomFeatureEncoder) -> None:
    """Wrong ref_pos ndim (2-D instead of 3-D) triggers BeartypeCallHintParamViolation."""
    tok_idx = torch.repeat_interleave(torch.arange(N_RES), ATOMS_PER_RES)
    ref_pos_bad = torch.zeros(B, N_ATOM)  # missing coordinate dim
    ref_element = torch.zeros(B, N_ATOM, E)
    ref_space_uid = torch.zeros(B, N_ATOM, dtype=torch.long)
    s_input = torch.zeros(B, N_RES, C_TOKEN)
    z_input = torch.zeros(B, N_RES, N_RES, C_PAIR)
    r_scaled = torch.zeros(B, N_ATOM, 3)
    tok = tok_idx.unsqueeze(0).expand(B, -1)
    with pytest.raises(BeartypeCallHintParamViolation):
        encoder(ref_pos_bad, ref_element, ref_space_uid, s_input, z_input, r_scaled, tok)


def test_atom_attention_decoder_forward_wrong_shape(decoder: AtomAttentionDecoder) -> None:
    """Wrong q_skip ndim (2-D instead of 3-D) triggers BeartypeCallHintParamViolation."""
    tok_idx = torch.repeat_interleave(torch.arange(N_RES), ATOMS_PER_RES)
    tok = tok_idx.unsqueeze(0).expand(B, -1)
    q_skip_bad = torch.zeros(B, N_ATOM)  # missing c_atom dim
    p_skip = torch.zeros(B, N_ATOM, K, C_ATOMPAIR)
    c_skip = torch.zeros(B, N_ATOM, C_ATOM)
    c = torch.zeros(B, N_ATOM, C_ATOM)
    s = torch.zeros(B, N_RES, C_TOKEN)
    z = torch.zeros(B, N_RES, N_RES, C_PAIR)
    with pytest.raises(BeartypeCallHintParamViolation):
        decoder(q_skip_bad, p_skip, c_skip, c, s, z, tok)
```

- [ ] **Step 2: Run the 5 new tests**

```
python -m pytest pallatom/tests/architecture/test_atom_transformers.py -k "wrong_shape" -v
```

Expected: 5 × PASSED

- [ ] **Step 3: Commit**

```bash
git add pallatom/tests/architecture/test_atom_transformers.py
git commit -m "test: add wrong-shape negative tests for atom_transformers"
```

---

### Task 3: Negative tests — `losses.py` (6 functions)

**Files:**
- Modify: `pallatom/tests/architecture/test_losses.py`

The 6 decorated functions: `atom_loss`, `med_loss_per_block`, `_pairwise_dist`,
`smooth_lddt_loss`, `distogram_loss_residue`, `distogram_loss_atom`.

Strategy for `"... N 3"` annotations: pass a tensor with last dim `4` (not `3`).
Strategy for `"... N_atom K n_bins"` or `"... N_res N_res n_bins"`: pass a 2-D tensor
(below the minimum dimensionality implied by the annotation's fixed trailing dims).

- [ ] **Step 1: Extend imports**

Change the existing losses import block in `test_losses.py` to add `med_loss_per_block`
and `_pairwise_dist`, and add the beartype import:

```python
from architecture.losses import (
    _pairwise_dist,
    atom_loss,
    distogram_loss_atom,
    distogram_loss_residue,
    med_loss,
    med_loss_per_block,
    smooth_lddt_loss,
)
from beartype.roar import BeartypeCallHintParamViolation
```

- [ ] **Step 2: Append 6 negative tests**

```python
# ---------------------------------------------------------------------------
# Shape-contract enforcement — negative tests
# ---------------------------------------------------------------------------


def test_atom_loss_wrong_shape() -> None:
    """Wrong last dim (4 instead of 3) on r_denoised triggers BeartypeCallHintParamViolation."""
    r_bad = torch.zeros(B, N, 4)  # last dim must be 3
    r_gt = torch.zeros(B, N, 3)
    with pytest.raises(BeartypeCallHintParamViolation):
        atom_loss(r_bad, r_gt)


def test_med_loss_per_block_wrong_shape() -> None:
    """Wrong last dim on r_denoised_k triggers BeartypeCallHintParamViolation."""
    r_bad = torch.zeros(B, N, 4)  # last dim must be 3
    r_gt = torch.zeros(B, N, 3)
    logits = torch.zeros(B, N, VOCAB)
    aa_gt = torch.zeros(B, N, dtype=torch.long)
    with pytest.raises(BeartypeCallHintParamViolation):
        med_loss_per_block(r_bad, r_gt, logits, aa_gt, LAM, ALPHA)


def test_pairwise_dist_wrong_shape() -> None:
    """Wrong last dim (4 instead of 3) triggers BeartypeCallHintParamViolation."""
    x_bad = torch.zeros(10, 4)  # last dim must be 3
    with pytest.raises(BeartypeCallHintParamViolation):
        _pairwise_dist(x_bad)


def test_smooth_lddt_loss_wrong_shape() -> None:
    """Wrong last dim (4 instead of 3) on r_pred triggers BeartypeCallHintParamViolation."""
    r_pred_bad = torch.zeros(N, 4)  # last dim must be 3
    r_true = torch.zeros(N, 3)
    with pytest.raises(BeartypeCallHintParamViolation):
        smooth_lddt_loss(r_pred_bad, r_true)


def test_distogram_loss_residue_wrong_shape() -> None:
    """2-D p (below min 3-D for '... N N n_bins') triggers BeartypeCallHintParamViolation."""
    p_bad = torch.zeros(N, N)  # needs at least 3 dims for "... N_res N_res n_bins"
    y = torch.zeros(N, N, dtype=torch.long)
    with pytest.raises(BeartypeCallHintParamViolation):
        distogram_loss_residue(p_bad, y)


def test_distogram_loss_atom_wrong_shape() -> None:
    """2-D q (below min 3-D for '... N K n_bins') triggers BeartypeCallHintParamViolation."""
    q_bad = torch.zeros(N_ATOMS, K)  # needs at least 3 dims for "... N_atom K n_bins"
    y = torch.zeros(N_ATOMS, K, dtype=torch.long)
    with pytest.raises(BeartypeCallHintParamViolation):
        distogram_loss_atom(q_bad, y)
```

- [ ] **Step 3: Run**

```
python -m pytest pallatom/tests/architecture/test_losses.py -k "wrong_shape" -v
```

Expected: 6 × PASSED

- [ ] **Step 4: Commit**

```bash
git add pallatom/tests/architecture/test_losses.py
git commit -m "test: add wrong-shape negative tests for losses"
```

---

### Task 4: Negative tests — `node_update.py` (3 functions)

**Files:**
- Modify: `pallatom/tests/architecture/test_node_update.py`

The 3 decorated functions: `AdaLN.forward`, `AttentionPairBias.forward`, `NodeUpdate.forward`.
`AdaLN` is not yet imported — add it.

- [ ] **Step 1: Extend imports**

```python
from architecture.node_update import AdaLN, AttentionPairBias, NodeUpdate
from beartype.roar import BeartypeCallHintParamViolation
```

- [ ] **Step 2: Append 3 negative tests**

```python
# ---------------------------------------------------------------------------
# Shape-contract enforcement — negative tests
# ---------------------------------------------------------------------------


def test_adaln_forward_wrong_shape() -> None:
    """Wrong a ndim (2-D instead of 3-D) triggers BeartypeCallHintParamViolation."""
    adaln = AdaLN(c_a=C_RES, c_s=C_RES).eval()
    a_bad = torch.zeros(B, N_RES)  # missing c_a dim
    s = torch.zeros(B, N_RES, C_RES)
    with pytest.raises(BeartypeCallHintParamViolation):
        adaln(a_bad, s)


def test_attention_pair_bias_forward_wrong_shape(
    attn: AttentionPairBias,
    z: Float[torch.Tensor, "B N_res N_res C_pair"],
) -> None:
    """Wrong a ndim (2-D instead of 3-D) triggers BeartypeCallHintParamViolation."""
    a_bad = torch.zeros(B, N_RES)  # missing c_res dim
    with pytest.raises(BeartypeCallHintParamViolation):
        attn(a_bad, None, z)


def test_node_update_forward_wrong_shape(
    node_update: NodeUpdate,
    s: Float[torch.Tensor, "B N_res C_res"],
    t: Float[torch.Tensor, "B N_res C_res"],
    z: Float[torch.Tensor, "B N_res N_res C_pair"],
) -> None:
    """Wrong s ndim (2-D instead of 3-D) triggers BeartypeCallHintParamViolation."""
    s_bad = torch.zeros(B, N_RES)  # missing c_res dim
    with pytest.raises(BeartypeCallHintParamViolation):
        node_update(s_bad, t, z)
```

- [ ] **Step 3: Run**

```
python -m pytest pallatom/tests/architecture/test_node_update.py -k "wrong_shape" -v
```

Expected: 3 × PASSED

- [ ] **Step 4: Commit**

```bash
git add pallatom/tests/architecture/test_node_update.py
git commit -m "test: add wrong-shape negative tests for node_update"
```

---

### Task 5: Negative tests — `pair_update.py` (4 functions)

**Files:**
- Modify: `pallatom/tests/architecture/test_pair_update.py`

The 4 decorated functions: `TransformRBF.forward`, `TriangleAttentionStartingNodeWithBias.forward`,
`TriangleAttentionEndingNodeWithBias.forward`, `PairUpdate.forward`.

Strategy: pass `z` with 3 dims instead of 4 (missing the channel dim).

- [ ] **Step 1: Extend imports**

```python
from beartype.roar import BeartypeCallHintParamViolation
```

- [ ] **Step 2: Append 4 negative tests**

```python
# ---------------------------------------------------------------------------
# Shape-contract enforcement — negative tests
# ---------------------------------------------------------------------------


def test_transform_rbf_forward_wrong_shape(rbf: TransformRBF) -> None:
    """Wrong d ndim (2-D instead of 3-D) triggers BeartypeCallHintParamViolation."""
    d_bad = torch.zeros(B, N_RES)  # missing last N_res dim
    with pytest.raises(BeartypeCallHintParamViolation):
        rbf(d_bad)


def test_triangle_attn_starting_node_forward_wrong_shape(
    tri_start: TriangleAttentionStartingNodeWithBias,
    b: Float[torch.Tensor, "B N_res N_res n_heads"],
) -> None:
    """Wrong z ndim (3-D instead of 4-D) triggers BeartypeCallHintParamViolation."""
    z_bad = torch.zeros(B, N_RES, N_RES)  # missing c_pair dim
    with pytest.raises(BeartypeCallHintParamViolation):
        tri_start(z_bad, b)


def test_triangle_attn_ending_node_forward_wrong_shape(
    tri_end: TriangleAttentionEndingNodeWithBias,
    b: Float[torch.Tensor, "B N_res N_res n_heads"],
) -> None:
    """Wrong z ndim (3-D instead of 4-D) triggers BeartypeCallHintParamViolation."""
    z_bad = torch.zeros(B, N_RES, N_RES)  # missing c_pair dim
    with pytest.raises(BeartypeCallHintParamViolation):
        tri_end(z_bad, b)


def test_pair_update_forward_wrong_shape(
    pair_update: PairUpdate,
    r_center: Float[torch.Tensor, "B N_res 3"],
) -> None:
    """Wrong z ndim (3-D instead of 4-D) triggers BeartypeCallHintParamViolation."""
    z_bad = torch.zeros(B, N_RES, N_RES)  # missing c_pair dim
    with pytest.raises(BeartypeCallHintParamViolation):
        pair_update(z_bad, r_center)
```

- [ ] **Step 3: Run**

```
python -m pytest pallatom/tests/architecture/test_pair_update.py -k "wrong_shape" -v
```

Expected: 4 × PASSED

- [ ] **Step 4: Commit**

```bash
git add pallatom/tests/architecture/test_pair_update.py
git commit -m "test: add wrong-shape negative tests for pair_update"
```

---

### Task 6: Negative tests — `pairformer_stack.py` (4 functions)

**Files:**
- Modify: `pallatom/tests/architecture/test_pairformer_stack.py`

The 4 decorated functions: `TriangleMultiplicationOutgoing.forward`,
`TriangleMultiplicationIncoming.forward`, `PairformerBlock.forward`, `PairformerStack.forward`.
`TriangleMultiplicationOutgoing` and `TriangleMultiplicationIncoming` have no fixtures —
instantiate them inline.

- [ ] **Step 1: Extend imports**

```python
from beartype.roar import BeartypeCallHintParamViolation
```

- [ ] **Step 2: Append 4 negative tests**

```python
# ---------------------------------------------------------------------------
# Shape-contract enforcement — negative tests
# ---------------------------------------------------------------------------


def test_triangle_mult_outgoing_forward_wrong_shape() -> None:
    """Wrong z ndim (3-D instead of 4-D) triggers BeartypeCallHintParamViolation."""
    mod = TriangleMultiplicationOutgoing(c=_C, c_hidden=_C_HIDDEN).eval()
    z_bad = torch.zeros(B, _N, _N)  # missing c_pair dim
    with pytest.raises(BeartypeCallHintParamViolation):
        mod(z_bad)


def test_triangle_mult_incoming_forward_wrong_shape() -> None:
    """Wrong z ndim (3-D instead of 4-D) triggers BeartypeCallHintParamViolation."""
    mod = TriangleMultiplicationIncoming(c=_C, c_hidden=_C_HIDDEN).eval()
    z_bad = torch.zeros(B, _N, _N)  # missing c_pair dim
    with pytest.raises(BeartypeCallHintParamViolation):
        mod(z_bad)


def test_pairformer_block_forward_wrong_shape(block: PairformerBlock) -> None:
    """Wrong z ndim (3-D instead of 4-D) triggers BeartypeCallHintParamViolation."""
    z_bad = torch.zeros(B, N_RES, N_RES)  # missing c_pair dim
    with pytest.raises(BeartypeCallHintParamViolation):
        block(None, z_bad)


def test_pairformer_stack_forward_wrong_shape(stack: PairformerStack) -> None:
    """Wrong z ndim (3-D instead of 4-D) triggers BeartypeCallHintParamViolation."""
    z_bad = torch.zeros(B, N_RES, N_RES)  # missing c_pair dim
    with pytest.raises(BeartypeCallHintParamViolation):
        stack(None, z_bad)
```

- [ ] **Step 3: Run**

```
python -m pytest pallatom/tests/architecture/test_pairformer_stack.py -k "wrong_shape" -v
```

Expected: 4 × PASSED

- [ ] **Step 4: Commit**

```bash
git add pallatom/tests/architecture/test_pairformer_stack.py
git commit -m "test: add wrong-shape negative tests for pairformer_stack"
```

---

### Task 7: Negative tests — `main_trunk.py` (3 functions)

**Files:**
- Modify: `pallatom/tests/architecture/test_main_trunk.py`

The 3 decorated functions: `scatter_mean`, `MainTrunk.embed_inputs`, `MainTrunk.forward`.

`scatter_mean` takes tensors → pass 2-D `src` (missing channel dim).
`embed_inputs` and `forward` take `FeaturizedBatch` → pass wrong type (str) to prove
beartype is active without constructing a full batch.

- [ ] **Step 1: Extend imports**

```python
from beartype.roar import BeartypeCallHintParamViolation
```

- [ ] **Step 2: Append 3 negative tests**

```python
# ---------------------------------------------------------------------------
# Shape-contract enforcement — negative tests
# ---------------------------------------------------------------------------


def test_scatter_mean_wrong_shape() -> None:
    """Wrong src ndim (2-D instead of 3-D) triggers BeartypeCallHintParamViolation."""
    src_bad = torch.zeros(B, N_ATOM)  # missing channel dim C
    index = torch.zeros(B, N_ATOM, dtype=torch.long)
    with pytest.raises(BeartypeCallHintParamViolation):
        scatter_mean(src_bad, index, N_RES, B)


def test_main_trunk_embed_inputs_wrong_type(model: MainTrunk) -> None:
    """Passing a non-FeaturizedBatch triggers BeartypeCallHintParamViolation."""
    with pytest.raises(BeartypeCallHintParamViolation):
        model.embed_inputs("not a FeaturizedBatch")  # type: ignore[arg-type]


def test_main_trunk_forward_wrong_type(model: MainTrunk) -> None:
    """Passing a non-FeaturizedBatch triggers BeartypeCallHintParamViolation."""
    with pytest.raises(BeartypeCallHintParamViolation):
        model("not a FeaturizedBatch")  # type: ignore[arg-type]
```

- [ ] **Step 3: Run**

```
python -m pytest pallatom/tests/architecture/test_main_trunk.py -k "wrong_shape or wrong_type" -v
```

Expected: 3 × PASSED

- [ ] **Step 4: Commit**

```bash
git add pallatom/tests/architecture/test_main_trunk.py
git commit -m "test: add wrong-shape/type negative tests for main_trunk"
```

---

### Task 8: Negative test — `template_embedder.py` (1 function)

**Files:**
- Modify: `pallatom/tests/architecture/test_template_embedder.py`

The 1 decorated function: `TemplateEmbedder.forward`. Pass `f_distogram` with 3 dims
instead of 4 (missing `n_bins` dim).

- [ ] **Step 1: Extend imports**

```python
from beartype.roar import BeartypeCallHintParamViolation
```

- [ ] **Step 2: Append 1 negative test**

```python
# ---------------------------------------------------------------------------
# Shape-contract enforcement — negative tests
# ---------------------------------------------------------------------------


def test_template_embedder_forward_wrong_shape(embedder: TemplateEmbedder) -> None:
    """Wrong f_distogram ndim (3-D instead of 4-D) triggers BeartypeCallHintParamViolation."""
    f_distogram_bad = torch.zeros(B, N_RES, N_RES)  # missing n_bins dim
    f_pseudo_beta_mask = torch.zeros(B, N_RES)
    z_ij = torch.zeros(B, N_RES, N_RES, C_Z)
    with pytest.raises(BeartypeCallHintParamViolation):
        embedder(f_distogram_bad, f_pseudo_beta_mask, z_ij, 0.5)
```

- [ ] **Step 3: Run**

```
python -m pytest pallatom/tests/architecture/test_template_embedder.py -k "wrong_shape" -v
```

Expected: 1 × PASSED

- [ ] **Step 4: Commit**

```bash
git add pallatom/tests/architecture/test_template_embedder.py
git commit -m "test: add wrong-shape negative test for template_embedder"
```

---

### Task 9: Negative tests — `alignment.py` (5 functions)

**Files:**
- Modify: `pallatom/tests/helpers/test_alignment.py`

The 5 decorated functions: `kabsch_rotation`, `kabsch_align`, `rmsd`, `kabsch_rmsd`,
`apply_transform`. All take `"... N 3"` tensors — pass last dim `4` to violate the literal
`3` constraint. `kabsch_rotation`, `rmsd`, `kabsch_rmsd` are not yet imported — add them.

- [ ] **Step 1: Extend imports**

```python
from beartype.roar import BeartypeCallHintParamViolation
from helpers.alignment import apply_transform, kabsch_align, kabsch_rmsd, kabsch_rotation, rmsd
```

- [ ] **Step 2: Append 5 negative tests**

```python
# ---------------------------------------------------------------------------
# Shape-contract enforcement — negative tests
# ---------------------------------------------------------------------------


def test_kabsch_rotation_wrong_shape() -> None:
    """Wrong last dim (4 instead of 3) triggers BeartypeCallHintParamViolation."""
    mobile_bad = torch.zeros(N, 4)  # last dim must be 3
    reference = torch.zeros(N, 3)
    with pytest.raises(BeartypeCallHintParamViolation):
        kabsch_rotation(mobile_bad, reference)


def test_kabsch_align_wrong_shape() -> None:
    """Wrong last dim (4 instead of 3) triggers BeartypeCallHintParamViolation."""
    mobile_bad = torch.zeros(N, 4)  # last dim must be 3
    target = torch.zeros(N, 3)
    with pytest.raises(BeartypeCallHintParamViolation):
        kabsch_align(mobile_bad, target, return_transform=False)


def test_rmsd_wrong_shape() -> None:
    """Wrong last dim (4 instead of 3) triggers BeartypeCallHintParamViolation."""
    predicted_bad = torch.zeros(N, 4)  # last dim must be 3
    reference = torch.zeros(N, 3)
    with pytest.raises(BeartypeCallHintParamViolation):
        rmsd(predicted_bad, reference)


def test_kabsch_rmsd_wrong_shape() -> None:
    """Wrong last dim (4 instead of 3) triggers BeartypeCallHintParamViolation."""
    mobile_bad = torch.zeros(N, 4)  # last dim must be 3
    target = torch.zeros(N, 3)
    with pytest.raises(BeartypeCallHintParamViolation):
        kabsch_rmsd(mobile_bad, target)


def test_apply_transform_wrong_shape() -> None:
    """Wrong coords last dim (4 instead of 3) triggers BeartypeCallHintParamViolation."""
    coords_bad = torch.zeros(N, 4)  # last dim must be 3
    R = torch.eye(3).unsqueeze(0)  # (1, 3, 3)
    t = torch.zeros(1, 1, 3)
    with pytest.raises(BeartypeCallHintParamViolation):
        apply_transform(coords_bad, R, t, t)
```

- [ ] **Step 3: Run**

```
python -m pytest pallatom/tests/helpers/test_alignment.py -k "wrong_shape" -v
```

Expected: 5 × PASSED

- [ ] **Step 4: Commit**

```bash
git add pallatom/tests/helpers/test_alignment.py
git commit -m "test: add wrong-shape negative tests for alignment"
```

---

### Task 10: Negative tests — `atom_utils.py` (4 functions + `Protein` dataclass)

**Files:**
- Modify: `pallatom/tests/helpers/test_atom_utils.py`

The 5 sites: `Protein` (dataclass), `atom37_to_atom5`, `pseudo_cb`, `get_cb_coords`,
`atom37_to_cb`. The `Protein` dataclass stores NumPy arrays, not tensors.

- [ ] **Step 1: Extend imports**

```python
import numpy as np
from beartype.roar import BeartypeCallHintParamViolation
```

- [ ] **Step 2: Append 5 negative tests**

```python
# ---------------------------------------------------------------------------
# Shape-contract enforcement — negative tests
# ---------------------------------------------------------------------------


def test_protein_wrong_shape() -> None:
    """Wrong atom_positions last dim (4 instead of 3) triggers BeartypeCallHintParamViolation."""
    n_res = 5
    with pytest.raises(BeartypeCallHintParamViolation):
        Protein(
            atom_positions=np.zeros((n_res, 37, 4), dtype=np.float64),  # last dim must be 3
            aatype=np.zeros(n_res, dtype=np.intp),
            atom_mask=np.zeros((n_res, 37), dtype=np.float64),
            residue_index=np.zeros(n_res, dtype=np.intp),
            chain_index=np.zeros(n_res, dtype=np.intp),
            b_factors=np.zeros((n_res, 37), dtype=np.float64),
        )


def test_atom37_to_atom5_wrong_shape(
    atom37_positions: Float[torch.Tensor, "B N_res 37 3"],
) -> None:
    """Wrong last dim (4 instead of 3) on atom37_positions triggers BeartypeCallHintParamViolation."""
    positions_bad = torch.zeros(B, N_RES, 37, 4)  # last dim must be 3
    atom37_mask = torch.ones(B, N_RES, 37)
    with pytest.raises(BeartypeCallHintParamViolation):
        atom37_to_atom5(positions_bad, atom37_mask)


def test_pseudo_cb_wrong_shape() -> None:
    """Wrong last dim (4 instead of 3) on n triggers BeartypeCallHintParamViolation."""
    n_bad = torch.zeros(10, 4)  # last dim must be 3
    ca = torch.zeros(10, 3)
    c = torch.zeros(10, 3)
    with pytest.raises(BeartypeCallHintParamViolation):
        pseudo_cb(n_bad, ca, c)


def test_get_cb_coords_wrong_shape(
    atom5_positions: Float[torch.Tensor, "B N_res 5 3"],
) -> None:
    """Wrong last dim (4 instead of 3) on atom5_positions triggers BeartypeCallHintParamViolation."""
    positions_bad = torch.zeros(B, N_RES, 5, 4)  # last dim must be 3
    atom5_mask = torch.ones(B, N_RES, 5)
    with pytest.raises(BeartypeCallHintParamViolation):
        get_cb_coords(positions_bad, atom5_mask)


def test_atom37_to_cb_wrong_shape() -> None:
    """Wrong last dim (4 instead of 3) on atom37_positions triggers BeartypeCallHintParamViolation."""
    positions_bad = torch.zeros(B, N_RES, 37, 4)  # last dim must be 3
    atom37_mask = torch.ones(B, N_RES, 37)
    with pytest.raises(BeartypeCallHintParamViolation):
        atom37_to_cb(positions_bad, atom37_mask)
```

- [ ] **Step 3: Run**

```
python -m pytest pallatom/tests/helpers/test_atom_utils.py -k "wrong_shape" -v
```

Expected: 5 × PASSED

- [ ] **Step 4: Commit**

```bash
git add pallatom/tests/helpers/test_atom_utils.py
git commit -m "test: add wrong-shape negative tests for atom_utils"
```

---

### Task 11: Negative tests — `featurize.py` (6 functions + 3 dataclasses)

**Files:**
- Modify: `pallatom/tests/helpers/test_featurize.py`

The 9 sites: `Distogram.forward`, `sinusoidal_encoding`, `_ref_pos_for_residue` (wrong type),
`ProteinBatch` (dataclass), `FeaturizedBatch` (dataclass), `FeaturizedItem` (dataclass),
`featurize_single_item`, `featurize_batch` (wrong type), `apply_conditioning_dropout` (wrong type).

- [ ] **Step 1: Extend imports**

```python
import dataclasses

from beartype.roar import BeartypeCallHintParamViolation
from helpers.featurize import (
    Distogram,
    FeaturizedBatch,
    FeaturizedItem,
    ProteinBatch,
    _ref_pos_for_residue,
    apply_conditioning_dropout,
    featurize_batch,
    featurize_single_item,
    sinusoidal_encoding,
)
from train.train_config import TrainConfig
```

Note: `dataclasses` and `TrainConfig` are needed for dataclass negative tests. `TrainConfig` is
already imported in the test file — skip if so; add if not.

- [ ] **Step 2: Append 9 negative tests**

```python
# ---------------------------------------------------------------------------
# Shape-contract enforcement — negative tests
# ---------------------------------------------------------------------------


def test_distogram_forward_wrong_shape(disto: Distogram) -> None:
    """Wrong coords last dim (4 instead of 3) triggers BeartypeCallHintParamViolation."""
    coords_bad = torch.zeros(N_RES, 4)  # last dim must be 3
    with pytest.raises(BeartypeCallHintParamViolation):
        disto(coords_bad)


def test_sinusoidal_encoding_wrong_shape() -> None:
    """Wrong positions ndim (3-D instead of 2-D) triggers BeartypeCallHintParamViolation."""
    positions_bad = torch.zeros(B, N_RES, 1)  # must be 2-D "batch N_res"
    with pytest.raises(BeartypeCallHintParamViolation):
        sinusoidal_encoding(positions_bad)


def test_ref_pos_for_residue_wrong_type() -> None:
    """Non-str resname triggers BeartypeCallHintParamViolation."""
    with pytest.raises(BeartypeCallHintParamViolation):
        _ref_pos_for_residue(42)  # type: ignore[arg-type]


def test_protein_batch_wrong_shape() -> None:
    """Wrong atom_positions last dim (4 instead of 3) triggers BeartypeCallHintParamViolation."""
    with pytest.raises(BeartypeCallHintParamViolation):
        ProteinBatch(
            atom_positions=torch.zeros(B, N_RES, 37, 4),  # last dim must be 3
            atom_mask=torch.ones(B, N_RES, 37),
            residue_index=torch.arange(N_RES, dtype=torch.float).unsqueeze(0).expand(B, -1),
            seq=["A" * N_RES] * B,
        )


def test_featurized_batch_wrong_shape() -> None:
    """Wrong ref_pos ndim (2-D instead of 3-D) triggers BeartypeCallHintParamViolation."""
    n_atom = N_RES * 5
    k_local = 4
    with pytest.raises(BeartypeCallHintParamViolation):
        FeaturizedBatch(
            ref_pos=torch.zeros(B, n_atom),  # must be 3-D "B N_atom 3"
            ref_element=torch.zeros(B, n_atom, 4),
            ref_space_uid=torch.zeros(B, n_atom, dtype=torch.long),
            gt_res_distogram=torch.zeros(B, N_RES, N_RES, N_BINS, dtype=torch.long),
            f_pseudo_beta_mask=torch.zeros(B, N_RES, dtype=torch.long),
            f_residue_idx=torch.zeros(B, N_RES, C_RES),
            r_input=torch.zeros(B, n_atom, 3),
            r_gt=torch.zeros(B, n_atom, 3),
            atom5_mask=torch.zeros(B, n_atom, dtype=torch.bool),
            aa_indices=torch.zeros(B, N_RES, dtype=torch.long),
            residue_mask=torch.zeros(B, N_RES, dtype=torch.bool),
            t_hat=1.0,
            t_normalized=0.5,
            tok_idx=torch.zeros(B, n_atom, dtype=torch.long),
            center_uid=torch.zeros(B, N_RES, dtype=torch.long),
            gt_atom_distogram_sparse=torch.zeros(B, n_atom, k_local, 5),
            gt_atom_distogram_mask_sparse=torch.zeros(B, n_atom, k_local, dtype=torch.bool),
        )


def test_featurized_item_wrong_shape() -> None:
    """Wrong flat_pos ndim (1-D instead of 2-D) triggers BeartypeCallHintParamViolation."""
    n_atom = N_RES * 5
    with pytest.raises(BeartypeCallHintParamViolation):
        FeaturizedItem(
            N_res=N_RES,
            flat_pos=torch.zeros(n_atom),  # must be 2-D "N_atom 3"
            atom_mask_flat=torch.zeros(n_atom, dtype=torch.bool),
            residue_mask=torch.zeros(N_RES, dtype=torch.bool),
            f_pseudo_beta=torch.zeros(N_RES, dtype=torch.long),
            gt_res_distogram=torch.zeros(N_RES, N_RES, N_BINS, dtype=torch.long),
            aa_indices=torch.zeros(N_RES, dtype=torch.long),
            ref_pos=torch.zeros(n_atom, 3),
            ref_element=torch.zeros(n_atom, 4),
            f_residue_idx=torch.zeros(N_RES, C_RES),
        )


def test_featurize_single_item_wrong_shape() -> None:
    """Wrong atom37_positions last dim (4 instead of 3) triggers BeartypeCallHintParamViolation."""
    ala_ref_pos = torch.zeros(5, 3)
    ala_ref_elem = torch.zeros(5, 4)
    positions_bad = torch.zeros(N_RES, 37, 4)  # last dim must be 3
    atom37_mask = torch.ones(N_RES, 37)
    index = torch.arange(N_RES, dtype=torch.float)
    disto = Distogram(n_bins=N_BINS, min_dist=MIN_DIST, max_dist=MAX_DIST, overflow_bin=False)
    with pytest.raises(BeartypeCallHintParamViolation):
        featurize_single_item(
            positions_bad, atom37_mask, index, "A" * N_RES, ala_ref_pos, ala_ref_elem,
            C_RES, disto,
        )


def test_featurize_batch_wrong_type() -> None:
    """Non-ProteinBatch first arg triggers BeartypeCallHintParamViolation."""
    with pytest.raises(BeartypeCallHintParamViolation):
        featurize_batch("not a batch", None, None, None)  # type: ignore[arg-type]


def test_apply_conditioning_dropout_wrong_type() -> None:
    """Non-FeaturizedBatch first arg triggers BeartypeCallHintParamViolation."""
    with pytest.raises(BeartypeCallHintParamViolation):
        apply_conditioning_dropout("not a batch", 0.5, 0.5, 0.5, "cpu")  # type: ignore[arg-type]
```

- [ ] **Step 3: Run**

```
python -m pytest pallatom/tests/helpers/test_featurize.py -k "wrong_shape or wrong_type" -v
```

Expected: 9 × PASSED

- [ ] **Step 4: Commit**

```bash
git add pallatom/tests/helpers/test_featurize.py
git commit -m "test: add wrong-shape/type negative tests for featurize"
```

---

### Task 12: Negative tests — `sampling.py` (9 functions + 2 dataclasses)

**Files:**
- Modify: `pallatom/tests/sample/test_sampling.py`

The 11 sites: `EDMPrecond.forward`, `AllAtomContext` (dataclass), `build_AA_context`,
`TemplateContext` (dataclass), `build_template_context`, `build_sampling_context`,
`EDMSampler._sigma_schedule`, `EDMSampler.sample`, `atom5_to_atom37`.

- [ ] **Step 1: Extend imports**

```python
from beartype.roar import BeartypeCallHintParamViolation
```

- [ ] **Step 2: Append 11 negative tests**

```python
# ---------------------------------------------------------------------------
# Shape-contract enforcement — negative tests
# ---------------------------------------------------------------------------


def test_edm_precond_forward_wrong_shape(edm_precond: EDMPrecond) -> None:
    """Wrong r_input ndim (2-D instead of 3-D) triggers BeartypeCallHintParamViolation."""
    r_bad = torch.zeros(1, N_ATOM)  # missing coordinate dim
    with pytest.raises(BeartypeCallHintParamViolation):
        edm_precond(r_bad, 1.0)


def test_all_atom_context_wrong_shape() -> None:
    """Wrong r_gt ndim (2-D instead of 3-D) triggers BeartypeCallHintParamViolation."""
    b_local, n_atom_local, n_res_local, k_local = 2, N_ATOM, N_RES, 4
    with pytest.raises(BeartypeCallHintParamViolation):
        AllAtomContext(
            r_gt=torch.zeros(b_local, n_atom_local),  # must be 3-D "B N_atom 3"
            atom5_mask=torch.zeros(b_local, n_atom_local, dtype=torch.bool),
            residue_mask=torch.zeros(b_local, n_res_local, dtype=torch.bool),
            gt_atom_distogram_sparse=torch.zeros(b_local, n_atom_local, k_local, 22),
            gt_atom_distogram_mask_sparse=torch.zeros(
                b_local, n_atom_local, k_local, dtype=torch.bool
            ),
            aa_indices=torch.zeros(b_local, n_res_local, dtype=torch.long),
            f_residue_idx=torch.zeros(b_local, n_res_local, 32),
        )


def test_build_aa_context_wrong_shape(atom_disto_fn: Distogram) -> None:
    """Wrong coord last dim (4 instead of 3) triggers BeartypeCallHintParamViolation."""
    coord_bad = torch.zeros(N_RES, 37, 4)  # last dim must be 3
    with pytest.raises(BeartypeCallHintParamViolation):
        build_AA_context(
            atom_37_coordinate_tensor=coord_bad,
            atom_37_mask=torch.ones(N_RES, 37),
            residue_index=torch.arange(N_RES, dtype=torch.float),
            aa_sequence="A" * N_RES,
            atom_distogram_fn=atom_disto_fn,
            batch_size=1,
            device="cpu",
            c_res=32,
        )


def test_template_context_wrong_shape() -> None:
    """Wrong f_template_distogram ndim (3-D instead of 4-D) triggers BeartypeCallHintParamViolation."""
    with pytest.raises(BeartypeCallHintParamViolation):
        TemplateContext(
            f_template_distogram=torch.zeros(1, N_RES, N_RES, dtype=torch.long),  # missing bins dim
            f_pseudo_beta_mask=torch.zeros(1, N_RES, dtype=torch.long),
        )


def test_build_template_context_wrong_type(templ_disto: Distogram) -> None:
    """Non-Protein list element triggers BeartypeCallHintParamViolation."""
    with pytest.raises(BeartypeCallHintParamViolation):
        build_template_context([42], templ_disto)  # type: ignore[list-item]


def test_build_sampling_context_wrong_shape(
    atom_disto_fn: Distogram, templ_disto: Distogram
) -> None:
    """Wrong atom_positions last dim (4 instead of 3) triggers BeartypeCallHintParamViolation."""
    positions_bad = torch.zeros(N_RES, 37, 4)  # last dim must be 3
    with pytest.raises(BeartypeCallHintParamViolation):
        build_sampling_context(
            atom_positions=positions_bad,
            atom_mask=torch.ones(N_RES, 37),
            residue_index=torch.arange(N_RES, dtype=torch.float),
            seq="A" * N_RES,
            pdb_files=[],
            atom_distogram_fn=atom_disto_fn,
            templ_distogram_fn=templ_disto,
            c_res=32,
        )


def test_edm_sampler_sigma_schedule_wrong_type(bare_sampler: EDMSampler) -> None:
    """Non-int steps triggers BeartypeCallHintParamViolation."""
    with pytest.raises(BeartypeCallHintParamViolation):
        bare_sampler._sigma_schedule(3.14, "cpu")  # type: ignore[arg-type]


def test_edm_sampler_sample_wrong_type(edm_sampler: EDMSampler) -> None:
    """Tuple with non-int element triggers BeartypeCallHintParamViolation."""
    with pytest.raises(BeartypeCallHintParamViolation):
        edm_sampler.sample(shape=(1, N_ATOM, "bad"), steps=2)  # type: ignore[arg-type]


def test_atom5_to_atom37_wrong_shape() -> None:
    """Wrong coords_5 last dim (4 instead of 3) triggers BeartypeCallHintParamViolation."""
    coords_bad = torch.zeros(N_RES, 5, 4)  # last dim must be 3
    with pytest.raises(BeartypeCallHintParamViolation):
        atom5_to_atom37(coords_bad)
```

- [ ] **Step 3: Run**

```
python -m pytest pallatom/tests/sample/test_sampling.py -k "wrong_shape or wrong_type" -v
```

Expected: 9 × PASSED (note: `test_build_template_context_wrong_type` and
`test_edm_sampler_sigma_schedule_wrong_type` and `test_edm_sampler_sample_wrong_type`
also pass since they test type, not shape)

- [ ] **Step 4: Commit**

```bash
git add pallatom/tests/sample/test_sampling.py
git commit -m "test: add wrong-shape/type negative tests for sampling"
```

---

### Task 13: Negative tests — `train_loop.py` (2 functions)

**Files:**
- Modify: `pallatom/tests/train/test_train_loop.py`

The 2 decorated functions: `evaluate`, `train`. Both take `MainTrunk`, `DataLoader`,
`TrainConfig`, `Distogram`, `Distogram`, `device: str`. Pass `device=42` (int not str)
with all other arguments from existing fixtures.

- [ ] **Step 1: Extend imports**

```python
from beartype.roar import BeartypeCallHintParamViolation
```

- [ ] **Step 2: Append 2 negative tests**

```python
# ---------------------------------------------------------------------------
# Shape-contract enforcement — negative tests
# ---------------------------------------------------------------------------


def test_evaluate_wrong_device_type(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
) -> None:
    """Non-str device triggers BeartypeCallHintParamViolation for evaluate."""
    with pytest.raises(BeartypeCallHintParamViolation):
        evaluate(model, loader, tcfg, distogram_res, distogram_atom, 42)  # type: ignore[arg-type]


def test_train_wrong_device_type(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
) -> None:
    """Non-str device triggers BeartypeCallHintParamViolation for train."""
    with pytest.raises(BeartypeCallHintParamViolation):
        train(model, tcfg, loader, loader, distogram_res, distogram_atom, 42)  # type: ignore[arg-type]
```

- [ ] **Step 3: Run**

```
python -m pytest pallatom/tests/train/test_train_loop.py -k "wrong_device_type" -v
```

Expected: 2 × PASSED

- [ ] **Step 4: Commit**

```bash
git add pallatom/tests/train/test_train_loop.py
git commit -m "test: add wrong-type negative tests for train_loop"
```

---

### Task 14: Full verification

- [ ] **Step 1: Run the complete test suite**

```
cd /workspaces/diffusion && python -m pytest pallatom/tests/ -x --tb=short -q
```

Expected: all tests pass (614 existing + ~57 new = ~671 total), 0 failures.

- [ ] **Step 2: Confirm the CI gate sees the new negative tests**

The negative tests reference function names and class names via imports and direct calls,
so the CI gate should still pass (it checks source coverage, not test count). Verify:

```
python -m pytest pallatom/tests/test_jaxtyped_coverage.py -v
```

Expected: PASSED

- [ ] **Step 3: Run pre-commit to confirm all hooks pass**

```
pre-commit run --all-files
```

Expected: all hooks pass (black, ruff, pyright, pytest).
