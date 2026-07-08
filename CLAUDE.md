# diffusion — repo-wide conventions

## Code quality tooling

All hooks are declared in `.pre-commit-config.yaml`. Configuration for the Python
tools lives in `pyproject.toml`. **Every hook listed below must pass before a commit
is accepted** — run `pre-commit run --all-files` locally to verify before pushing.
**Every applicable GitHub Actions workflow must also pass before a pull request can
be merged** — see [GitHub Actions](#github-actions-required-for-merge) below.

> **Authoritative sources:** `.pre-commit-config.yaml` (hook inventory),
> `pyproject.toml` (tool configuration), and `.github/workflows/` (CI checks). The
> summaries below are kept in sync with those files — when in doubt, check them
> directly.

### Hook inventory

| Stage | Hook | What it checks |
|-------|------|----------------|
| `commit-msg` | **commitlint** | Commit message follows [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`, etc.) |
| `pre-commit` | **black** | Python code is formatted (line-length 80, Python 3.12) |
| `pre-commit` | **check-yaml** | All YAML files parse without errors |
| `pre-commit` | **check-github-workflows** | GitHub Actions YAML is schema-valid |
| `pre-commit` | **check-taskfile** | `Taskfile.yml` is schema-valid |
| `pre-commit` | **basedpyright** | Type-checking of `pallatom/` (including `pallatom/tests/`) |
| `pre-commit` | **ruff** | Python linting (auto-fixes applied; remaining errors block commit) |
| `pre-commit` | **xenon** | Cyclomatic complexity limits (`--max-absolute B --max-modules A --max-average A`) |
| `pre-commit` | **pylint** | Google-style lint |
| `pre-commit` | **enforce-einops** | Bans raw tensor ops (see below) |
| `pre-commit` | **enforce-jaxtyping** | Bans bare `torch.Tensor` / `Tensor` annotations (see below) |
| `pre-commit` | **pydoclint** | Google-style docstring linting |
| CI (`push`/`pull_request`) | **ARM64 CI** / **CUDA CI** | Rebuild the devcontainer, rerun every pre-commit hook above inside it, then run `pytest pallatom` — see below |

---

---

### Black

- **Line length:** 80
- **Target:** Python 3.12
- Standard Black formatting: double quotes, trailing commas in multi-line structures.

---

### Ruff

- **Line length:** 80, **target:** Python 3.12
- **`select = ["ALL"]`** — every ruleset ruff ships is enabled by default; new ruff
  rules apply automatically. Only the rules below are turned back off.
- Notebooks (`*.ipynb`) are excluded from ruff entirely.
- `pallatom/tests/**` gets exactly one ruleset carve-out (`S101`, see below) —
  otherwise tests are linted identically to the rest of `pallatom/`, including
  `ANN` (type hints) and `D` (docstrings).

**Key ignores (ML-friendly carve-outs):**

| Rule(s) | Reason |
|---------|--------|
| `F722`, `F821`, `UP037` | jaxtyping shape strings are runtime-parsed, not forward refs |
| `D107` | no docstring required on `__init__`; use the class-level docstring |
| `PLR0402` | `import torch.nn as nn` is fine |
| `N812` | `import torch.nn.functional as F` is the standard idiom |
| `N817` | `DDP` abbreviation is fine |
| `N803`, `N806` | uppercase argument/variable names are fine (`N_atom`, `B`) |
| `TC002`, `TC003`, `TC006` | forcing imports behind `if TYPE_CHECKING:` blocks isn't worth the churn |
| `ISC003` | conflicts with basedpyright's `reportImplicitStringConcatenation` |
| `S101` (tests only, via `per-file-ignores`) | `assert` is the correct tool inside `pytest` tests |

Everything else in `select = ["ALL"]` is **active**, including rules a curated
config would typically drop for ML code — notably `PLR2004` (magic numbers),
`N802` (function names must be lowercase), `TRY003` (long exception messages),
and `FBT003` (boolean literals at call sites). `PLR0913` (too-many-arguments)
is not ignored either; instead `[tool.ruff.lint.pylint] max-args = 6` raises
the default limit to 6 instead of disabling the check.

Import sorting (`I`) groups stdlib → third-party → first-party, alphabetical
within each group, with `known-first-party = ["diffusion"]` and
`known-third-party = ["wandb"]` pinned explicitly in `pyproject.toml` — the
`wandb` pin exists because a gitignored local `wandb/` run-log directory can
otherwise shadow the real package during isort's filesystem-based detection.

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

### Basedpyright

- **Tool:** `basedpyright` (a stricter superset of pyright)
- **Mode:** `all` — every check is an error by default; `enableBasedFeatures = true` adds
  basedpyright-specific rules on top.
- **Python version:** 3.12
- **`extraPaths = ["pallatom", "/usr/local/lib/python3.12/dist-packages"]`** — `pallatom` lets
  first-party imports resolve without the `pallatom.` prefix (e.g.
  `from architecture.main_trunk import MainTrunk`); the `dist-packages` entry points at the
  CUDA devcontainer base image's system Python, since that container gets torch via
  `--system-site-packages` rather than an install into `/opt/venv` and basedpyright doesn't
  honor that fallback on its own.
- **Excluded:** `**/*.ipynb` only — `pallatom/tests/*` **is** type-checked (no test carve-out).

**Notable basedpyright-specific rules (enabled by `enableBasedFeatures`):**

| Rule | Meaning |
|------|---------|
| `reportAny` | Variables and arguments must not have type `Any`; numpy stubs often produce `Any` for array element access — this is a known limitation |
| `reportImplicitOverride` | Methods that override a parent must be decorated with `@override` (import from `typing_extensions` for Python < 3.12) |

**Disabled (ML / library compatibility):**

| Setting | Reason |
|---------|--------|
| `reportInvalidTypeForm` | jaxtyping shape strings are runtime-parsed, not forward refs |
| `reportIndexIssue` | complex PyTorch tensor indexing patterns |
| `reportUnknownVariableType` | jaxtyping interference |
| `reportPrivateImportUsage` | disabled by project choice |

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
| `.norm(` | `torch.sqrt(einops.reduce(x**2, "... -> ", "sum"))` |
| `torch.linalg.vector_norm(` | `torch.sqrt(einops.reduce(x**2, "... -> ", "sum"))` |
| `torch.linalg.norm(` | `torch.sqrt(einops.reduce(x**2, "... -> ", "sum"))` |

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

### GitHub Actions (required for merge)

**A pull request must pass every GitHub Actions workflow before it can be merged** —
passing `pre-commit` locally is necessary but not sufficient, since CI rebuilds the
devcontainer from scratch and reruns the full check suite inside it.

| Workflow | Trigger | What it does |
|----------|---------|---------------|
| `.github/workflows/arm64.yml` (**ARM64 CI**) | every `push` to `main` and every `pull_request` | Builds the ARM64 devcontainer, then calls the reusable workflow below on `ubuntu-latest` |
| `.github/workflows/cuda.yml` (**CUDA CI**) | every `push` to `main` and every `pull_request` | Builds the CUDA devcontainer on the self-hosted GPU runner, asserts `torch.cuda.is_available()`, then calls the same reusable workflow |
| `.github/workflows/devcontainer-checks.yml` (reusable, called by both above) | — | Inside the built devcontainer: `check-yaml`, `check-github-workflows`, `check-taskfile`, `black --check`, `ruff check`, `basedpyright`, `pylint`, `xenon`, `pydoclint`, `enforce-einops`, `enforce-jaxtyping` each run as separate jobs; a final `test` job (depending on all of them) runs `pytest pallatom` |

In short: every pre-commit hook runs a second time in CI against a clean container
image on both architectures, on every push and every PR, and the full test suite
only runs after all of them pass.

---

### Practical rules for writing new code

1. **Every function and class needs a Google-style docstring** — one line is enough for simple
   helpers; full `Args`/`Returns` for anything non-obvious. This applies to test files too.

2. **Type-annotate all parameters and return types** — even in test files. There is no `ANN`
   waiver for `pallatom/tests/`; ruff's `ANN` rules and basedpyright's `reportMissingParameterType`
   both apply there exactly as they do everywhere else in `pallatom/`.

3. **Sort imports** with isort ordering: stdlib → third-party → first-party (no `pallatom.`
   prefix), alphabetical within each group. Black/ruff will flag violations.

4. **Use `einops`** (`rearrange`, `reduce`, `repeat`, `einsum`) for all tensor reshaping,
   contraction, and reduction — no `.view`, `.reshape`, `.unsqueeze`, `torch.matmul`, `@`,
   `.norm(`, `torch.linalg.vector_norm(`, or `torch.linalg.norm(`. For L2 norm use
   `torch.sqrt(reduce(x**2, "... -> ", "sum"))`.

5. **Line length 80** — Black reformats automatically; keep manual line breaks at or under 80.

6. **Generics need type arguments** — write `dict[str, torch.Tensor]`, not `dict`;
   `list[str]`, not `list`.

7. **NEVER use `Any`, `TypedDict`, or `np.generic`** — these are absolutely forbidden, no exceptions.
   - `Any` defeats the type checker entirely; use a concrete type, a union, or a generic instead.
   - `TypedDict` is a crutch for untyped dicts; use a proper `@dataclass` instead.
   - `np.generic` is too broad; use the specific NumPy scalar type (`np.float32`, `np.int64`, etc.).
   If you are tempted to reach for any of these, stop and redesign the data structure or annotation.
