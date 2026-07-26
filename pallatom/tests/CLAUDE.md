# Test conventions

## Structure

- Write all tests as **module-level functions** (`def test_...`). Never group tests inside a class.
- Use **`pytest` fixtures** for all shared tensors and inputs. Prefer factory fixtures over hard coded data fixtures. Factory fixtures construct data fixtures from input arguments. Use indirect=True for `@pytest.mark.parametrize` to pass parameters directly from the test function to fixture factory. Data fixtures should return a single `torch.Tensor`, dataclass,  custom object. Avoid fixtures that return dicts, tuples, or other composite builtin types. Fixtures may depend on other fixtures via argument injection — prefer composing small fixtures over writing large monolithic ones.
- Brainstorm with the Arrange-Act-Assert (AAA) Pattern: Always structure your thoughts using this triad to define exactly what you are testing: Arrange: Set up the exact state of the world, initial inputs, and necessary mocks.Act: Perform the specific function, method, or API call you are testing.Assert: Verify the exact outcome (return values, exceptions, or side effects).
- Aggressively use the `@pytest.mark.parametrize` marker for tests. Instead of writing 10 different test functions for 10 different inputs, use the @pytest.mark.parametrize decorator to cover base cases, boundaries, and edge cases in a single, readable function. Add ids so failures to the parameterized tests tell a story.
- Default to the combination of a **factory fixture** + a **companion enum** + a single `@pytest.mark.parametrize`-driven test function when a test covers several named scenarios. Give each scenario an enum member (used with `indirect=True` to build the case, or passed straight to the test) instead of branching on raw values (`if`/`elif`/`match`) inside the test body — branching per scenario is what drives up cyclomatic complexity and is exactly what this pattern avoids. When a scenario needs more than one piece of expected data (e.g. an expected boundary index *and* an expected boolean outcome), bundle them into the enum member's value and expose each as a `@property` (e.g. `expectation.present_from`, `expectation.expect_distinct_pseudo_cb`) — the test reads attributes off the enum, it never compares enum identity to select behavior. See `test_atom37_to_cb` and its `BetaCarbonExpectation` enum for a worked example.
- A great test suite is reliable and order-independent. Brainstorm potential hidden dependencies between your tests. If your tests rely on global states or network calls, design them to use monkeypatch to isolate environments securely.
- Avoid Mocking as much as possible. Functional tests and integration tests are always superior to highly mocked unit tests.
- Data Pipelines, deep learning models, file input and output, training and sampling scripts, all MUST have End to End tests to validate the entire workflow.
- All tests must take at least one fixture as input. If a pytest does not use `@pytest.mark.parametrize`, justify why this test should test no other edge cases. Never construct objects to test or expected outputs to tests againt inside a test. Use pytest fixtures and `@pytest.mark.parametrize`.
## Type checking and shape contracts

- Annotate helper functions with **`jaxtyping`** (`Float`, `Int`, `Bool`, etc.) and `@jaxtyped(typechecker=beartype)` so shape contracts are verified at call time.
- Use named dimensions in jaxtyping annotations (e.g. `"B N 3"`, `"N_atoms N_atoms"`) to make shape intent explicit.
- All tensor functions and methods should have tests that verify shape and data type to enforce runtime compliance to jaxtyping annotations.
- Avoid constructing random tensor and numpy array fixtures as much as possible. Force exactness with concrete, human readable, deterministic tensor and numpy array fixtures.

## Tensor operations

- Replace all `@` matrix multiplications and `torch.matmul` calls with **`einops.einsum`**. The einsum string makes the contraction axes explicit and self-documenting.
- Use **`einops.rearrange`** instead of `view`, `reshape`, `unsqueeze`, `squeeze`, or `permute`.
- Use **`einops.reduce`** instead of `torch.sum`, `torch.mean`, `torch.max`, etc. when reducing over named axes.
- Use **`einops.repeat`** instead of `expand`, `repeat`, or `tile`.
- Prefer einops operations throughout helper functions so shape contracts interact naturally with jaxtyping annotations.

## Pytest best practices

- **Style:** Prefer functional tests over class-based tests.
- **Comments:** Code should be self-explanatory. If explanation is needed, write a multi-line Google-style docstring — no inline comments. Be complete and understandable in your docstring explanations. Always prefer longer, understandable docstrings over shorter confusing ones.
- **Imports:** Import at module level only.
- **Fixtures:** Reuse existing fixtures; create new ones only when needed. When constructing a hard coded data fixture, always attempt to construct a factory fixture. If it cannot be factory fixture, explain why in the docstring.
- **Temp files:** Use the `tmp_path` fixture for filesystem tests.
- **Assertions:** Use `pytest.raises(ExceptionType)` for exception testing.
