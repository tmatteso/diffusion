# Pytest quality expectations

Tests aren't just there to satisfy CI — they should read as documentation of
expected behavior and stay cheap to extend. In particular:

- Use **fixtures** to factor out setup/teardown and shared state instead of
  copy-pasting it into every test function. See
  [Using pytest Fixtures to Elevate Product Feature Quality](https://www.fiddler.ai/blog/using-pytest-fixtures-to-elevate-product-feature-quality).
- Use **`@pytest.mark.parametrize`** (and factory-fixture patterns for
  constructing variant inputs) instead of near-duplicate test functions that
  only differ by a literal or two. See
  [Advanced Pytest Patterns: Harnessing the Power of Parametrization and Factory Methods](https://www.fiddler.ai/blog/advanced-pytest-patterns-harnessing-the-power-of-parametrization-and-factory-methods)
  and [Pytest Parametrize: Testing](https://mjmichael.medium.com/pytest-parametrize-testing-9a7661701d23).
- Prefer one behavior asserted per test case, with parametrize IDs or
  descriptive fixture names that make a failing case identifiable from the
  test name alone (this is also what ruff's `PT` rules and pylint's Google
  style expect).

## Structure

- Write tests as **module-level functions** (`def test_...`) — never group
  them inside a class.
- Every test must take at least one fixture as input; never construct the
  object under test or the expected output inline in the test body. If a
  test doesn't use `@pytest.mark.parametrize`, its docstring should justify
  why no other edge case is worth covering.
- Brainstorm each test with **Arrange-Act-Assert**: set up exact state and
  inputs, perform the one call under test, then verify the exact outcome
  (return value, exception, or side effect).
- Prefer **factory fixtures** (fixtures that build data from arguments) over
  hardcoded data fixtures. Use `indirect=True` with `@pytest.mark.parametrize`
  to route parameters into a factory fixture. Data fixtures should return a
  single `torch.Tensor`, dataclass, or other custom object — not a `dict` or
  `tuple`. Compose small fixtures rather than writing large monolithic ones.

## Shape and dtype contracts

- Every tensor-producing function or method needs a test that verifies both
  shape and dtype, so runtime jaxtyping annotations stay honest.
- Prefer concrete, human-readable, deterministic tensor/array fixtures over
  randomly generated ones — a reviewer should be able to hand-verify the
  expected output from the fixture values alone.

## Isolation and mocking

- Tests must be reliable and order-independent — brainstorm hidden
  dependencies between tests (shared global state, network calls) and use
  `monkeypatch` to isolate them rather than relying on execution order.
- Avoid mocking wherever possible. A functional or integration test that
  exercises real code is always preferable to a heavily mocked unit test.
- Data pipelines, model forward/backward passes, file I/O, and
  training/sampling scripts all require an end-to-end test that validates
  the full workflow, not just isolated units.

## Misc conventions

- Imports at module level only — no imports inside test functions.
- Use the built-in `tmp_path` fixture for any test that touches the
  filesystem.
- Use `pytest.raises(ExceptionType, match=...)` for exception testing.
- No inline comments — if a test needs explanation, put it in a complete,
  Google-style docstring instead.

See [../../docs/best_practices.md](../../docs/best_practices.md) for the
repo's broader code quality conventions, and [CLAUDE.md](CLAUDE.md) for the
full, authoritative pallatom-specific testing rules.
