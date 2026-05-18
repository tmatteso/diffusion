# Design: jaxtyping Shape-Contract Enforcement

**Date:** 2026-05-18
**Status:** Approved

## Problem

`@jaxtyped(typechecker=beartype)` decorators enforce tensor-shape contracts at runtime only — they
fire when a decorated function is called with real tensors, not during static analysis (black, ruff,
pyright). This means a newly added `@jaxtyped` function with no tests is just documentation at
commit time; no pre-commit hook catches the gap.

The repo currently has **56** `@jaxtyped` decorators across source files: 50 on
functions/methods and 6 on frozen dataclasses (which validates constructor field types at
instantiation time via beartype).

| Module | Functions/methods | Dataclasses |
|---|---|---|
| `pallatom/architecture/atom_transformers.py` | 5 | — |
| `pallatom/architecture/losses.py` | 6 | — |
| `pallatom/architecture/node_update.py` | 3 | — |
| `pallatom/architecture/pair_update.py` | 4 | — |
| `pallatom/architecture/pairformer_stack.py` | 4 | — |
| `pallatom/architecture/main_trunk.py` | 3 | — |
| `pallatom/architecture/template_embedder.py` | 1 | — |
| `pallatom/helpers/alignment.py` | 5 | — |
| `pallatom/helpers/atom_utils.py` | 4 | `Protein` |
| `pallatom/helpers/featurize.py` | 6 | `ProteinBatch`, `FeaturizedBatch`, `FeaturizedItem` |
| `pallatom/sample/sampling.py` | 9 | `AllAtomContext`, `TemplateContext` |
| `pallatom/train/train_loop.py` | 2 | — |

All 56 already have positive tests (614 tests pass). The gap: no guarantee that future additions
stay covered, and no test explicitly proves each annotation has beartype teeth.

## Goals

1. **CI gate** — a test that fails if any `@jaxtyped`-decorated function or dataclass has no
   reference in the corresponding test file.
2. **Negative tests** — one explicit "wrong-shape → `BeartypeCallHintParamViolation`" test per
   decorated function/method, and one per dataclass (wrong field shape at construction), proving
   the annotation enforces at runtime.

## Approach: Static AST gate + inline negative tests

**Chosen:** Approach A from the design session — inline negative tests in existing files, plus one
new `test_jaxtyped_coverage.py` containing the CI gate. Runtime call-tracking (Approach B) and
`pytest-cov` integration (Approach C) were considered and rejected as overly fragile for a research
codebase.

## Section 1: CI Gate — `pallatom/tests/test_jaxtyped_coverage.py`

A single test function `test_every_jaxtyped_function_has_a_test`:

1. **Source scan** — Uses `ast.parse` on every `*.py` file under `pallatom/` (excluding `tests/`)
   to find functions decorated with `@jaxtyped`. Identifies each as either a standalone function
   or a method (class member). No module imports; pure AST.

2. **Test file derivation** — Maps source path to test path:
   `pallatom/{subpkg}/{mod}.py` → `pallatom/tests/{subpkg}/test_{mod}.py`

3. **Coverage check** — For each decorated function or class, parses the test file's AST and
   checks that the relevant name appears as an `ast.Name` node anywhere in the tree:
   - Standalone function `foo` → `ast.Name(id="foo")` present in test AST.
   - Class method `ClassName.forward` → `ast.Name(id="ClassName")` present in test AST.
   - Dataclass `FooDataclass` → `ast.Name(id="FooDataclass")` present in test AST.
   - AST name nodes exclude comments, docstrings, and string literals.

4. **Failure message** — Collects all uncovered functions and fails once with a list, so a single
   commit can fix all gaps at once.

### Decorator detection logic

Detects both call forms of `@jaxtyped`:

```python
# @jaxtyped(typechecker=beartype)  → ast.Call with func=ast.Name(id="jaxtyped")
# @jaxtyped                        → ast.Name(id="jaxtyped")
```

Walks all function and class definitions, including nested ones. When the decorated node is a
`ClassDef`, it's treated as a dataclass — the class name is the lookup key in the test file.

## Section 2: Negative tests

For each of the 56 `@jaxtyped` decorators, one test is added to the existing test file.

### Functions and methods (50 tests)

- **Naming:** `test_<funcname>_wrong_shape` for standalone functions;
  `test_<ClassName>_forward_wrong_shape` for methods.
- **Structure:** Constructs one deliberately wrong-shaped tensor (wrong size on the most
  characteristic dimension — e.g. wrong channel dim for single-embedding functions, wrong spatial
  dim for coordinate functions). Passes it to the decorated function inside `pytest.raises`.
- **Exception type:** `beartype.roar.BeartypeCallHintParamViolation` — the specific subclass
  beartype raises on shape/type mismatch.
- **Fixtures:** Reuses existing fixtures (e.g. `conditioned_transition_block`,
  `encoder`, `transformer`) already defined in each test file. No new fixtures needed.
- **Private helpers** (e.g. `_pairwise_dist`): Called directly; no fixture needed.

### Dataclasses (6 tests)

- **Naming:** `test_<DataclassName>_wrong_shape`.
- **Structure:** Instantiates the dataclass with one field set to a wrong-shaped tensor while all
  other fields are correct. Since frozen dataclasses require all fields, a helper builder is used
  where needed for classes with many fields (`FeaturizedBatch`, `FeaturizedItem`).
- **Exception type:** same `BeartypeCallHintParamViolation`.

### Example

```python
from beartype.roar import BeartypeCallHintParamViolation

def test_conditioned_transition_block_wrong_shape(
    conditioned_transition_block: ConditionedTransitionBlock,
) -> None:
    """Wrong a channel dimension triggers beartype."""
    a_bad = torch.zeros(B, N_RES, C_ATOM + 1)  # c_a mismatch
    s = torch.zeros(B, N_RES, C_ATOM)
    with pytest.raises(BeartypeCallHintParamViolation):
        conditioned_transition_block(a_bad, s)
```

## Section 3: Scope and file layout

### File changes

| File | Change |
|---|---|
| `pallatom/tests/test_jaxtyped_coverage.py` | **New** — CI gate (~80 lines) |
| `pallatom/tests/architecture/test_atom_transformers.py` | +5 negative tests |
| `pallatom/tests/architecture/test_losses.py` | +6 negative tests |
| `pallatom/tests/architecture/test_node_update.py` | +3 negative tests |
| `pallatom/tests/architecture/test_pair_update.py` | +4 negative tests |
| `pallatom/tests/architecture/test_pairformer_stack.py` | +4 negative tests |
| `pallatom/tests/architecture/test_main_trunk.py` | +3 negative tests |
| `pallatom/tests/architecture/test_template_embedder.py` | +1 negative test |
| `pallatom/tests/helpers/test_alignment.py` | +5 negative tests |
| `pallatom/tests/helpers/test_atom_utils.py` | +5 negative tests (4 fns + `Protein`) |
| `pallatom/tests/helpers/test_featurize.py` | +9 negative tests (6 fns + 3 dataclasses) |
| `pallatom/tests/sample/test_sampling.py` | +11 negative tests (9 fns + 2 dataclasses) |
| `pallatom/tests/train/test_train_loop.py` | +2 negative tests |

**Total:** ~56 new test functions + 1 new file.

### What we are NOT doing

- Not modifying any source files.
- Not adding positive tests (all 56 already have them).
- Not adding new fixtures (reusing existing ones throughout).
- Not adding a pre-commit hook (the CI gate lives in the pytest suite, which pre-commit already
  runs via the existing `pytest` hook in `.pre-commit-config.yaml`).

## Implementation order

1. Write `test_jaxtyped_coverage.py` (CI gate) — immediately surfaces any real gaps.
2. Add negative tests file-by-file in the same module order as the source files.
3. Verify all tests pass (`pytest pallatom/tests/ -x`).
4. Commit.
