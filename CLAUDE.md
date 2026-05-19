# diffusion — repo-wide conventions

## Code quality tooling

All hooks are declared in `.pre-commit-config.yaml`. Configuration for the Python
tools lives in `pyproject.toml`. **Every hook listed below must pass before a commit
is accepted** — run `pre-commit run --all-files` locally to verify before pushing.

> **Authoritative sources:** `.pre-commit-config.yaml` (hook inventory) and
> `pyproject.toml` (tool configuration). The summaries below are kept in sync with
> those files — when in doubt, check them directly.

### Hook inventory

| Stage | Hook | What it checks |
|-------|------|----------------|
| `commit-msg` | **commitlint** | Commit message follows [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`, etc.) |
| `pre-commit` | **check-github-workflows** | GitHub Actions YAML is schema-valid |
| `pre-commit` | **check-taskfile** | `Taskfile.yml` is schema-valid |
| `pre-commit` | **check-yaml** | All YAML files parse without errors |
| `pre-commit` | **checkmake** | `Makefile` passes style checks |
| `pre-commit` | **black** | Python code is formatted (line-length 100, Python 3.10+) |
| `pre-commit` | **ruff** | Python linting (auto-fixes applied; remaining errors block commit) |
| `pre-commit` | **pyright** | Type-checking of `pallatom/` and `REST_APIs/` |
| `pre-commit` | **enforce-einops** | Bans raw tensor ops (see below) |
| `pre-commit` | **enforce-jaxtyping** | Bans bare `torch.Tensor` / `Tensor` annotations (see below) |
| `pre-commit` | **pytest** | Full test suite must pass |

---

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
| `RUF` | ruff-specific | `RUF100` flags unused `# noqa` comments |
| `TRY` | exception handling hygiene | |
| `PERF` | performance antipatterns | |
| `PGH` | pygrep-hooks | `# type: ignore` comments must include a specific error code |
| `FBT` | boolean trap | prefer keyword-only args or enums over bare `bool` parameters |
| `PT` | pytest style | enforces `match=` on `raises`, one statement per block, parametrize format |
| `ISC` | implicit string concatenation | catches `("foo" "bar")` copy-paste bugs |

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
| `TRY003` | long exception messages are fine in ML code |
| `FBT003` | boolean literals at call sites are unavoidable in ML code |

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
| `reportConstantRedefinition` | Reassigning a `Final` variable |
| `reportMissingParameterType` | All parameters need type annotations |
| `reportMissingTypeArgument` | Generics need type args — `dict[str, X]` not bare `dict` |
| `reportUninitializedInstanceVariable` | Class attributes must be initialised in `__init__` |
| `reportUnnecessaryCast` | `cast()` calls must do real work |
| `reportUnnecessaryTypeIgnoreComment` | `# type: ignore` must suppress an actual error |
| `reportIncompatibleMethodOverride` | Overrides must be covariant |
| `reportIncompatibleVariableOverride` | Same for variable overrides |
| `reportCallInDefaultInitializer` | No function calls in default argument values |
| `reportReturnType` | All public functions must have explicit return type annotations |
| `strictParameterNoneValue` | Prevents accessing attributes that might be `None` at runtime |

**Enforced as warnings (pre-commit treats these as failures too):**

| Setting | Meaning |
|---------|---------|
| `reportUnknownArgumentType` | Argument types must be resolvable |
| `reportUnknownLambdaType` | Lambda return types must be resolvable |
| `reportDeprecated` | Catches deprecated PyTorch/CUDA API usage early |

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

### enforce-einops (`scripts/check_einops.py`)

The following patterns are **banned** in non-comment Python lines and will fail the commit:

| Banned call | Use instead |
|-------------|-------------|
| `.reshape(` | `einops.rearrange` |
| `.view(` | `einops.rearrange` |
| `.permute(` | `einops.rearrange` |
| `.unsqueeze(` | `einops.rearrange` |
| `.squeeze(` | `einops.rearrange` |
| `torch.einsum(` | `einops.einsum` |

---

### enforce-jaxtyping (`scripts/check_jaxtyping.py`)

Bare `torch.Tensor` or `Tensor` is **banned** in all function argument annotations,
return-type annotations, and variable annotations. Every tensor annotation must be
wrapped with a jaxtyping dtype class:

```python
# ✗ banned
def foo(x: torch.Tensor) -> torch.Tensor: ...

# ✓ required
def foo(x: Float[torch.Tensor, "N 3"]) -> Float[torch.Tensor, "N 3"]: ...
```

Valid wrappers: `Bool`, `Complex`, `Float`, `Inexact`, `Int`, `Integer`, `Num`, `Shaped`.

---

### commitlint

Commit messages must follow Conventional Commits. Examples of valid prefixes:
`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`, `perf:`, `ci:`.

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

7. **NEVER use `Any`, `TypedDict`, or `np.generic`** — these are absolutely forbidden, no exceptions.
   - `Any` defeats the type checker entirely; use a concrete type, a union, or a generic instead.
   - `TypedDict` is a crutch for untyped dicts; use a proper `@dataclass` instead.
   - `np.generic` is too broad; use the specific NumPy scalar type (`np.float32`, `np.int64`, etc.).
   If you are tempted to reach for any of these, stop and redesign the data structure or annotation.
