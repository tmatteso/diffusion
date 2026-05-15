# diffusion — repo-wide conventions

## Code quality tooling

Three pre-commit hooks run on every commit: **black**, **ruff**, and **pyright**.
All three must pass before a commit is accepted.

---

### Black

- **Line length:** 100
- **Target:** Python 3.10+
- Standard Black formatting: double quotes, trailing commas in multi-line structures.

---

### Ruff

- **Line length:** 100, **target:** Python 3.10
- Notebooks (`*.ipynb`) are excluded.

**Active rule sets:**

| Code | Ruleset | Notes |
|------|---------|-------|
| `E`, `W` | pycodestyle errors/warnings | |
| `F` | pyflakes | unused imports, undefined names |
| `I` | isort | import ordering: stdlib → third-party → first-party, alphabetical within groups |
| `B` | flake8-bugbear | |
| `C4` | flake8-comprehensions | |
| `UP` | pyupgrade | modernise syntax to Python 3.10+ |
| `PL` | pylint | semantic checks |
| `D` | pydocstyle | **Google convention** — every public function/class needs a Google-style docstring |
| `N` | PEP8 naming | with ML-friendly exceptions (see below) |
| `ANN` | type-hint enforcement | **waived entirely for `pallatom/tests/**/*.py`** |
| `RET` | return consistency | |
| `SIM` | flake8-simplify | |
| `ARG` | unused arguments | |

**Key ignores (ML-friendly carve-outs):**

| Rule(s) | Reason |
|---------|--------|
| `F722`, `F821`, `UP037` | jaxtyping shape strings are runtime-parsed, not forward refs |
| `PLR2004` | magic numbers are common in ML constants |
| `PLR0913` | no 5-argument limit on functions |
| `N802`, `N803`, `N806` | uppercase names are fine (`N_atom`, `B`, `Kabsch_loss`) |
| `N812` | `import torch.nn.functional as F` is the standard idiom |
| `N817` | `DDP` abbreviation is fine |
| `PLR0402` | `import torch.nn as nn` is fine |
| `D107` | no docstring required on `__init__`; use the class-level docstring |

**Docstring convention (Google):**

```python
def my_func(x: torch.Tensor, n: int) -> torch.Tensor:
    """One-line summary ending with a period.

    Args:
        x: Description of x.
        n: Description of n.

    Returns:
        Description of the return value.
    """
```

One-line docstrings are fine when the function is self-explanatory. Omit the `Args`/`Returns`
sections only when there are no arguments or the meaning is completely obvious from the name.

---

### Pyright

- **Mode:** `basic` (not strict; ~30 fundamental checks enabled)
- **`extraPaths = ["pallatom"]`** — first-party imports resolve without the `pallatom.` prefix
  (e.g. `from architecture.main_trunk import MainTrunk`).

**Enforced as errors:**

| Setting | Meaning |
|---------|---------|
| `reportConstantRedefinition` | Reassigning a `Final` variable is an error |

**Enforced as warnings (pre-commit treats these as failures):**

| Setting | Meaning |
|---------|---------|
| `reportMissingParameterType` | All parameters need type annotations |
| `reportMissingTypeArgument` | Generics need type args — `dict[str, X]` not bare `dict` |
| `reportUnknownArgumentType` | Argument types must be resolvable |
| `reportUnknownLambdaType` | Lambda return types must be resolvable |
| `reportUninitializedInstanceVariable` | Class attributes must be initialised in `__init__` |
| `reportUnnecessaryCast` | `cast()` calls must do real work |
| `reportUnnecessaryTypeIgnoreComment` | `# type: ignore` must suppress an actual error |
| `reportIncompatibleMethodOverride` | Overrides must be covariant |
| `reportIncompatibleVariableOverride` | Same for variable overrides |
| `reportCallInDefaultInitializer` | No function calls in default argument values |

**Disabled (contamination from untyped libraries):**

| Setting | Reason |
|---------|--------|
| `reportUnknownVariableType` | torch/einops are partially untyped |
| `reportUnknownParameterType` | same |
| `reportUnknownMemberType` | same |
| `reportPrivateImportUsage` | disabled by project choice |
| `reportInvalidTypeForm` | jaxtyping shape strings |
| `reportIndexIssue` | complex PyTorch tensor indexing patterns |

---

### Practical rules for writing new code

1. **Every function and class needs a Google-style docstring** — one line is enough for simple
   helpers; full `Args`/`Returns` for anything non-obvious. This applies to test files too
   (the `ANN` waiver covers annotations only, not `D` docstring rules).

2. **Type-annotate all parameters and return types** — even in test files, because pyright's
   `reportMissingParameterType` fires regardless of the `ANN` ruff waiver.

3. **Sort imports** with isort ordering: stdlib → third-party → first-party (no `pallatom.`
   prefix), alphabetical within each group. Black/ruff will flag violations.

4. **Use `einops`** (`rearrange`, `reduce`, `repeat`, `einsum`) for all tensor reshaping and
   contraction — no `.view`, `.reshape`, `.unsqueeze`, `torch.matmul`, or `@`.

5. **Line length 100** — Black reformats automatically; keep manual line breaks at or under 100.

6. **Generics need type arguments** — write `dict[str, torch.Tensor]`, not `dict`;
   `list[str]`, not `list`.
