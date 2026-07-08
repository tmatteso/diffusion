# Best practices for writing code in this repo

> **Authoritative sources:** [`.pre-commit-config.yaml`](../.pre-commit-config.yaml)
> (hook inventory), [`pyproject.toml`](../pyproject.toml) (tool configuration), and
> [`CLAUDE.md`](../CLAUDE.md) (full conventions). This page is a quick-reference
> summary — when in doubt, check those directly.

## Quality gates

Every commit must pass, in order: **commitlint** (Conventional Commits message),
then the pre-commit stage — **black**, **check-yaml**,
**check-github-workflows**, **check-taskfile**, **basedpyright**, **ruff**,
**xenon** (complexity limits), **pylint** (against the
[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)),
**enforce-einops**, **enforce-jaxtyping**, and **pydoclint**. Run
`pre-commit run --all-files` locally before pushing to catch everything at
once.

**Passing pre-commit locally is necessary but not sufficient.** Every PR must
also pass **ARM64 CI** and **CUDA CI** (`.github/workflows/arm64.yml` /
`cuda.yml`) before it can be merged — each rebuilds the devcontainer from
scratch on its architecture, reruns every pre-commit hook above inside the
clean image via the shared `devcontainer-checks.yml` workflow, and only then
runs `pytest pallatom` (on the self-hosted GPU runner for the CUDA leg).
A hook passing on your machine doesn't guarantee it passes in the rebuilt
container — push and watch both workflows before assuming a change is done.

Ruff runs with **`select = ["ALL"]`** — every ruleset ruff ships is active by
default, including ones a typical ML-friendly config would drop
(`PLR2004` magic numbers, `N802` lowercase function names, `TRY003` long
exception messages, `FBT003` boolean literals at call sites). `PLR0913`
isn't ignored either; instead `max-args = 6` in `pyproject.toml` raises the
limit rather than disabling the check. `pallatom/tests/**` gets exactly one
carve-out (`S101`, so `assert` is allowed) — otherwise tests are linted
identically to the rest of `pallatom/`, including `ANN` (type hints) and `D`
(docstrings). Similarly, basedpyright type-checks `pallatom/tests/*` with no
carve-out at all.

## Practical rules for writing new code

1. **Every function and class needs a Google-style docstring** — one line is
   enough for simple helpers; full `Args`/`Returns` for anything non-obvious.
   This applies to test files too — there is no docstring carve-out for
   `pallatom/tests/`.

2. **Type-annotate all parameters and return types** — even in test files.
   There is no `ANN` waiver for `pallatom/tests/`; ruff's `ANN` rules and
   basedpyright's `reportMissingParameterType` both apply there exactly as
   they do everywhere else in `pallatom/`.

3. **Sort imports** with isort ordering: stdlib → third-party → first-party
   (no `pallatom.` prefix), alphabetical within each group. Black/ruff will
   flag violations.

4. **Use `einops`** (`rearrange`, `reduce`, `repeat`, `einsum`) for all tensor
   reshaping, contraction, and reduction — no `.view`, `.reshape`,
   `.unsqueeze`, `torch.matmul`, `@`, `.norm(`, `torch.linalg.vector_norm(`,
   or `torch.linalg.norm(`. For L2 norm use
   `torch.sqrt(reduce(x**2, "... -> ", "sum"))`.

5. **Wrap every tensor annotation with a jaxtyping dtype class** — bare
   `torch.Tensor` / `Tensor` is banned in argument, return, and variable
   annotations. Use `Float`, `Int`, `Bool`, `Shaped`, etc.

6. **Line length 80** — Black reformats automatically; keep manual line
   breaks at or under 80.

7. **Generics need type arguments** — write `dict[str, torch.Tensor]`, not
   `dict`; `list[str]`, not `list`.

8. **Never use `Any`, `TypedDict`, or `np.generic`** — no exceptions.
   - `Any` defeats the type checker entirely; use a concrete type, a union,
     or a generic instead.
   - `TypedDict` is a crutch for untyped dicts; use a proper `@dataclass`
     instead.
   - `np.generic` is too broad; use the specific NumPy scalar type
     (`np.float32`, `np.int64`, etc.).

Pytest quality expectations (fixtures, parametrization, factory methods) are
documented in [pallatom/tests/README.md](../pallatom/tests/README.md).

## Commit messages

Follow [Conventional Commits](https://www.conventionalcommits.org/) for the
subject-line prefix (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`,
`chore:`, `perf:`, `ci:`, etc.), and otherwise follow the style described in
[How to Write a Git Commit Message](https://cbea.ms/git-commit/): a short
imperative-mood subject line (≤ ~50 chars), a blank line, then a body that
explains *why* the change was made, wrapped at ~72 characters.
