# Test conventions

## Structure

- Write all tests as **module-level functions** (`def test_...`). Never group tests inside a class.
- **Do not write closures.** A closure is a function defined inside another function that reads one of the outer function's local variables, such as a nested `def reduced(metric_class)` inside a worker that uses the worker's `rank`, or a fixture returning `lambda: case.convert(positions, mask)`. The captured variable is a hidden input: the inner function's signature does not show it, and the function cannot be called, imported or tested on its own. This rule covers fixtures, helpers, workers and tests alike:
  - Never define a function inside another function, whether or not it captures anything. Define it at module level and pass every input as a parameter, as `reduce_at_rank(metric_class, rank)` in `test_useful_objects.py` does.
  - Never write a `lambda` that reads a local variable of the enclosing function.
  - When a fixture must hand back a callable, such as a thunk for `pytest.raises` or a mock's `side_effect`, return `functools.partial(module_level_function, arg, ...)`. This binds the arguments explicitly and does not create a closure.
  - When a fixture needs to build different data for different cases, use an `indirect=True` factory fixture that reads `request.param`. Do not return an inner builder function.

  Existing closures in the suite are not a pattern to extend. Leaving them alone is fine, but rewrite any closure in code you touch.
- Use **`pytest` fixtures** for all shared tensors and inputs. Prefer factory fixtures over hard coded data fixtures. Factory fixtures construct data fixtures from input arguments. Use indirect=True for `@pytest.mark.parametrize` to pass parameters directly from the test function to fixture factory. Data fixtures should return a single `torch.Tensor`, dataclass,  custom object. Avoid fixtures that return dicts, tuples, or other composite builtin types. Fixtures may depend on other fixtures via argument injection — prefer composing small fixtures over writing large monolithic ones.
- Brainstorm with the Arrange-Act-Assert (AAA) Pattern: Always structure your thoughts using this triad to define exactly what you are testing: Arrange: Set up the exact state of the world, initial inputs, and necessary mocks.Act: Perform the specific function, method, or API call you are testing.Assert: Verify the exact outcome (return values, exceptions, or side effects).
- Aggressively use the `@pytest.mark.parametrize` marker for tests. Instead of writing 10 different test functions for 10 different inputs, use the @pytest.mark.parametrize decorator to cover base cases, boundaries, and edge cases in a single, readable function. Add ids so failures to the parameterized tests tell a story.
- Default to the combination of a **factory fixture** + a **companion enum** + a single `@pytest.mark.parametrize`-driven test function when a test covers several named scenarios. Give each scenario an enum member, passed with `indirect=True` to build the case, instead of branching on raw values (`if`/`elif`/`match`) inside the test body — branching per scenario is what drives up cyclomatic complexity and is exactly what this pattern avoids. When a scenario needs several pieces of data, make the member's value an instance of a frozen dataclass shared by every case (see the dataclass rule below), so the test reads named fields such as `scenario.value.expected_steps` — it never compares enum identity to select behavior. See `test_train_ddp_keeps_ranks_in_lockstep` with its `LockstepScenario` enum and `LockstepCase` and `RankResidues` dataclasses for a worked example. Older enums such as `InvalidProteinShape` and `AtomTensorScenario` still use tuple values read through `@property`s; that is the pattern being replaced, not one to copy.
- **Every parametrized case must fit on one line.** Never write a `pytest.param(` that spans multiple lines: Black explodes any case over 80 characters, one argument per line, which makes the case table hard to scan. A short case may stay as a one-line `pytest.param(..., id="...")`; otherwise it becomes a one-line enum member whose name is the id, and the `parametrize` call stays this small:

  ```python
  @dataclasses.dataclass(frozen=True)
  class RankResidues:
      rank_0: tuple[int, ...]
      rank_1: tuple[int, ...]

  @dataclasses.dataclass(frozen=True)
  class LockstepCase:
      residues: RankResidues
      budget_per_rank: int
      expected_steps: int

  class LockstepScenario(enum.Enum):
      rank_1_over_budget = LockstepCase(RankResidues((8, 8), (16, 16)), 24, 2)
      neither_over_budget = LockstepCase(RankResidues((8, 8), (8, 8)), 24, 1)

  @pytest.mark.parametrize(
      "lockstep_scenario",
      list(LockstepScenario),
      ids=[case.name for case in LockstepScenario],
      indirect=True,
  )
  ```

  - If a member does not fit in 80 characters, shorten its name and move the detail into the enum's `Members:` docstring section. Do not fall back to a multi-line case, a dict of cases, or a list of dataclass instances passed to `parametrize` in place of the enum: the enum is what gives every case its id. A frozen dataclass as each member's *value* is required (see below); it is the enum that must not be replaced.
  - Use lowercase member names, and give every member a distinct value: members with equal values silently become aliases and drop out of `list(...)`, which loses a test case without any error.
  - Build `ids` with a list comprehension as above, not a lambda: basedpyright types the lambda's parameter as `Any`.
  - When several tests share one factory, subclass a member-free base enum that exposes the factory's inputs, and give each test its own subclass holding that test's expectations. A small `indirect=True` fixture returns the member so the factory and the test read the same scenario. See `AtomTensorScenario`, its subclasses, and the `atom_scenario` fixture.
- **Make each enum member's value an instance of one frozen dataclass shared by all the cases**, not a tuple. The dataclass names and types every field once, so a member reads as `rank_1_over_budget = LockstepCase(RankResidues((8, 8), (16, 16)), 24, 2)` and the test reads `scenario.value.expected_steps`, with no `self.value[2]` index properties to keep in sync with the tuple layout. Apply the same rule inside the dataclass: when a field has structure of its own, such as batch lengths per rank, give it a nested frozen dataclass with named fields (`RankResidues` with `rank_0` and `rank_1`) instead of a tuple of tuples whose axes the reader has to guess. Document each field in the dataclass's `Attributes:` section and each case in the enum's `Members:` section. Frozen dataclasses compare and hash by value, so the distinct-values rule above still applies.
- **Do not add module-level constants for test data.** A name like `LOCKSTEP_SHORT_RES = 8` at the top of a test section separates a case's inputs from the case table, so a reader has to hunt through the file to check a member's arithmetic, and other tests can quietly start depending on it. Before adding a constant, put the value in the first of these homes that fits:
  1. **The enum member**, when the value is a scenario's input or expectation. Write it as a literal in the one-line member, and shape the value so it also shows the counts that would otherwise need their own names: `rank_1_over_budget = LockstepCase(RankResidues((8, 8), (16, 16)), 24, 2)` lists every rank's batch lengths, so the number of ranks and batches per rank needs no constant.
  2. **The enum member as well**, when every member shares the value (such as a budget that all cases are judged against). Repeating it per member keeps each row complete and the members distinct. If the value truly never varies, return it from a property on the enum, or on a member-free base enum, so it stays attached to the scenarios that use it.
  3. **The library**, when the code under test already defines the value. Import it (for example `ResidueRepresentation.atom37` from `helpers.atom_utils`, not `37`) so the test cannot drift from the code.
  4. **A fixture**, when the value is shared setup across many tests in a module (model sizes, a base config). Tests then receive it by injection rather than reading a global.

  Derive anything computable from these, such as a world size from `len(scenario.value.residues.by_rank)`, instead of naming it. A leading underscore does not make a constant acceptable, and neither does the fact that the file already has some: existing module-level constants are not a pattern to extend.
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
