# Test conventions

## Structure

- Write all tests as **module-level functions** (`def test_...`). Never group tests inside a class.
- **Do not write closures.** A closure is a function defined inside another function that reads one of the outer function's local variables, such as a nested `def reduced(metric_class)` inside a worker that uses the worker's `rank`, or a fixture returning `lambda: case.convert(positions, mask)`. The captured variable is a hidden input: the inner function's signature does not show it, and the function cannot be called, imported or tested on its own. This rule covers fixtures, helpers, workers and tests alike:
  - Never define a function inside another function, whether or not it captures anything. Define it at module level and pass every input as a parameter, as `reduce_at_rank(metric_class, rank)` in `test_useful_objects.py` does.
  - Never write a `lambda` that reads a local variable of the enclosing function.
  - When a fixture needs to build different data for different cases, make it a factory fixture that takes the case as a parameter (see the direct-parametrization rule below). Do not return an inner builder function.

  Existing closures in the suite are not a pattern to extend. Leaving them alone is fine, but rewrite any closure in code you touch.
- **Do not write thunks or use `functools.partial`.** A thunk is a zero-argument callable that wraps a call so it runs later, such as a fixture returning `lambda: constructor(**kwargs)` for the test to call inside `pytest.raises`. Both thunks and `partial` objects hide what is being called behind an indirection the reader has to unwrap. A fixture returns data (a tensor, a dataclass or a custom object), never a callable:
  - **Code expected to raise:** fixtures return the inputs, and the test makes the call that raises directly inside `with pytest.raises(...)`. That call is the test's Act step, so it is the one exception to the rule below against constructing objects inside a test. To break one field's contract, take a valid instance from a fixture and swap in the bad value with `dataclasses.replace`, which rebuilds the object through its `@jaxtyped` `__init__`. See `test_metrics_reject_non_scalar_field` in `test_useful_objects.py`, which takes the `metrics_of_ones` and `vector_field` fixtures:

    ```python
    with pytest.raises(TypeCheckError):
        _ = dataclasses.replace(
            metrics_of_ones,
            **{vector_field.name: vector_field.value},
        )
    ```
  - **Mock `side_effect`:** pass a module-level function by name (`side_effect=zero_forward`), with every input it needs taken from the call's own arguments.
  - **Library APIs that demand a callable** (for example `LambdaLR`'s `lr_lambda`): do not rebuild the call in the test with `partial`. Move the construction into a named library function (such as a `build_scheduler(optimizer, config)` in `train/train_loop.py`) and have the test call that, so the test exercises the same code as training.

  Existing thunks and `functools.partial` calls are not a pattern to extend. Rewrite any in code you touch.
- **Do not cast a class to `Callable[..., T]` to call it.** A line such as `constructor: Callable[..., TensorAccumulatorMixin] = metric_cls` exists only to call a `type[...]` with arbitrary `**kwargs`, and the `...` turns off argument checking. It is the `Any` ban in disguise. Build instances through the class's own typed API instead: explicit keyword construction (`ThroughputStatistics(avg_batch_size=..., ...)`), a classmethod such as `zero_init`, or `dataclasses.replace` on an existing instance. See `metrics_filled_with` in `test_useful_objects.py`.
- **Scenario data holds values, not functions.** Never put a `lambda`, a `Callable` field or a function reference in an enum member's value or a case dataclass, as in `InputEdit("t", lambda t: t * 2)`. Even a lambda that captures nothing hides what a case does behind code the reader has to unwrap, and its parameters are typed `Any` unless the field spells out the signature. Instead, give each kind of operation a small frozen dataclass whose settings are plain fields and whose behaviour is a method, all subclassing one abstract base the test calls through. See `InputEdit` and its subclasses `ScaleEdit`, `ShiftBinsEdit`, `SelectEdit` and `DropFirstEdit` in `architecture/test_template_embedder.py`, where the members read as data:

  ```python
  time = ScaleEdit("t", factor=2.0)
  mask_too_short = DropFirstEdit("f_pseudo_beta_mask", dim=1)
  ```

  When cases differ in which library function they call, give each function its own subclass whose method makes that call with explicit arguments. Test doubles that stand in for a model or a library call (such as `ScaledDenoiser`) hold the state they need as fields, not a stored function. Existing lambdas in scenario enums are not a pattern to extend.
- **Never give a test a `bool` parameter or fixture, and never silence ruff's boolean-trap rules with `# noqa: FBT001`.** A `bool` says nothing about what each case means, and the `noqa` is an inline comment the guide already bans. Give the two cases an enum whose member values are a frozen dataclass, parametrize it as described below, and have the test or fixture read the named field. Write the dataclass with a keyword argument, as in `ProcessGroupCase(group_active=False)`, because a positional `ProcessGroupCase(False)` trips ruff's `FBT003`. See `ProcessGroup`, `ProcessGroupCase` and the `process_group` fixture in `test_useful_objects.py`, used by `test_all_reduce_mean_keeps_values_on_one_process`. Existing `# noqa: FBT001` suppressions are not a pattern to extend.
- Use **`pytest` fixtures** for all shared tensors and inputs. Prefer factory fixtures over hard coded data fixtures. Factory fixtures construct data fixtures from input arguments. Pass parameters from the test's `@pytest.mark.parametrize` to the factory by direct parametrization, as described below. Data fixtures should return a single `torch.Tensor`, dataclass,  custom object. Avoid fixtures that return dicts, tuples, or other composite builtin types. Fixtures may depend on other fixtures via argument injection — prefer composing small fixtures over writing large monolithic ones.
- Brainstorm with the Arrange-Act-Assert (AAA) Pattern: Always structure your thoughts using this triad to define exactly what you are testing: Arrange: Set up the exact state of the world, initial inputs, and necessary mocks.Act: Perform the specific function, method, or API call you are testing.Assert: Verify the exact outcome (return values, exceptions, or side effects).
- Aggressively use the `@pytest.mark.parametrize` marker for tests. Instead of writing 10 different test functions for 10 different inputs, use the @pytest.mark.parametrize decorator to cover base cases, boundaries, and edge cases in a single, readable function. Add ids so failures to the parameterized tests tell a story.
- Default to the combination of a **factory fixture** + a **companion enum** + a single `@pytest.mark.parametrize`-driven test function when a test covers several named scenarios. Give each scenario an enum member, parametrized directly and read by the factory that builds the case, instead of branching on raw values (`if`/`elif`/`match`) inside the test body — branching per scenario is what drives up cyclomatic complexity and is exactly what this pattern avoids. When a scenario needs several pieces of data, make the member's value an instance of a frozen dataclass shared by every case (see the dataclass rule below), so the test reads named fields such as `scenario.value.expected_steps` — it never compares enum identity to select behavior. See `test_train_ddp_keeps_ranks_in_lockstep` with its `LockstepScenario` enum and `LockstepCase` and `RankResidues` dataclasses for a worked example.
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
  )
  ```

  - If a member does not fit in 80 characters, shorten its name and move the detail into the enum's `Members:` docstring section. Do not fall back to a multi-line case, a dict of cases, or a list of dataclass instances passed to `parametrize` in place of the enum: the enum is what gives every case its id. A frozen dataclass as each member's *value* is required (see below); it is the enum that must not be replaced.
  - Use lowercase member names, and give every member a distinct value: members with equal values silently become aliases and drop out of `list(...)`, which loses a test case without any error.
  - Build `ids` with a list comprehension as above, not a lambda: basedpyright types the lambda's parameter as `Any`.
  - When several tests share one factory, put the factory's inputs in a base frozen dataclass and give each test its own subclass adding that test's expectations; each test gets its own enum whose members are instances of its subclass. Every test parametrizes its enum under one shared name, and the factory takes that name typed as the union of the enums, reading only the base fields through `.value`. See `TensorSpec`, its subclasses `Atom5Spec` and `CbSpec`, and the `atom_tensors` fixture in `helpers/test_atom_utils.py`, which every conversion test feeds through `atom_scenario`.
- **Parametrize scenarios directly; do not write pass-through fixtures.** pytest makes every name in a direct `@pytest.mark.parametrize` available as a fixture to the test and to every fixture it depends on. A factory fixture therefore takes the scenario as an ordinary typed parameter, and the test lists it too when it reads an expectation:

  ```python
  @pytest.fixture
  def null_template(
      null_template_scenario: NullTemplateScenario,
  ) -> Float[torch.Tensor, "B N_res N_res n_templ_bins"]:
      return null_template_distogram(null_template_scenario.value.shape, "cpu")

  @pytest.mark.parametrize(
      "null_template_scenario",
      list(NullTemplateScenario),
      ids=[case.name for case in NullTemplateScenario],
  )
  def test_null_template_distogram_is_all_zero(
      null_template: Float[torch.Tensor, "B N_res N_res n_templ_bins"],
      null_template_scenario: NullTemplateScenario,
  ) -> None:
  ```

  - Do not write a fixture whose only job is `return cast(Scenario, request.param)`. It adds a definition and a docstring, and its `cast` exists only because `request.param` is untyped. A direct parameter gets its type from its annotation.
  - Nothing defines a directly parametrized name as a fixture, so a reader cannot jump to it. In the `Args:` entry of every fixture that takes one, say that the requesting test parametrizes it.
  - Use `indirect=True` only when the receiving fixture needs a scope wider than `function`, to build an expensive object once per scenario and share it across tests. A direct parameter lives for one test call, so a wider-scoped fixture cannot depend on it.
  - Existing pass-through fixtures and `indirect=True` factories are not a pattern to extend. This includes the `lockstep_scenario` and `process_group` fixtures behind the worked examples cited above, which predate this rule. Leaving them alone is fine, but convert any in code you touch.
- **Make each enum member's value an instance of one frozen dataclass shared by all the cases**, not a tuple. The dataclass names and types every field once, so a member reads as `rank_1_over_budget = LockstepCase(RankResidues((8, 8), (16, 16)), 24, 2)` and the test reads `scenario.value.expected_steps`, with no `self.value[2]` index properties to keep in sync with the tuple layout. Apply the same rule inside the dataclass: when a field has structure of its own, such as batch lengths per rank, give it a nested frozen dataclass with named fields (`RankResidues` with `rank_0` and `rank_1`) instead of a tuple of tuples whose axes the reader has to guess. Document each field in the dataclass's `Attributes:` section and each case in the enum's `Members:` section. Frozen dataclasses compare and hash by value, so the distinct-values rule above still applies.
- **Do not define type aliases or rename imports from this repository; write every annotation out where it is used.** Annotate a parameter as `z: Float[torch.Tensor, "B N_res N_res c"]`, not `z: Pair` with `type Pair = Float[torch.Tensor, "B N_res N_res c"]` at the top of the file, and spell out a union such as `module: TriangleMultiplicationOutgoing | PairformerStack` rather than naming it. An alias moves a tensor's dtype and shape away from the signature, so a reader has to scroll up to learn what a fixture returns or a test receives, and the jaxtyping dimension names stop being visible where they bind. The longer signature is the intended cost: Black puts one parameter per line, which keeps each annotation readable. This covers `type X = ...` statements and plain assignments such as `Coords = Float[torch.Tensor, "N 3"]` alike. The same goes for imports from this repository's own packages (`architecture`, `helpers`, `sample`, `train`, `tests`): import each name as it is defined, with no `as` rename. Write `from architecture.pairformer_stack import TriangleMultiplicationOutgoing`, not `TriangleMultiplicationOutgoing as TMO`, and `from train.train_config import LoaderConfig`, not `LoaderConfig as EvalLoaderConfig`. A rename means a reader cannot search for the class's real name, and two files can call the same class by different names. Third-party idioms that the lint config already allows, such as `import torch.nn.functional as F`, are unaffected. Existing aliases and renamed imports are not a pattern to extend.
- **Do not add module-level constants for test data.** A name like `LOCKSTEP_SHORT_RES = 8` at the top of a test section separates a case's inputs from the case table, so a reader has to hunt through the file to check a member's arithmetic, and other tests can quietly start depending on it. Before adding a constant, put the value in the first of these homes that fits:
  1. **The enum member**, when the value is a scenario's input or expectation. Write it as a literal in the one-line member, and shape the value so it also shows the counts that would otherwise need their own names: `rank_1_over_budget = LockstepCase(RankResidues((8, 8), (16, 16)), 24, 2)` lists every rank's batch lengths, so the number of ranks and batches per rank needs no constant.
  2. **The enum member as well**, when every member shares the value (such as a budget that all cases are judged against). Repeating it per member keeps each row complete and the members distinct. If the value truly never varies, return it from a property on the enum, or on a member-free base enum, so it stays attached to the scenarios that use it.
  3. **The library**, when the code under test already defines the value. Import it (for example `ResidueRepresentation.atom37` from `helpers.atom_utils`, not `37`) so the test cannot drift from the code.
  4. **A fixture**, when the value is shared setup across many tests in a module (model sizes, a base config). Tests then receive it by injection rather than reading a global.

  Derive anything computable from these, such as a world size from `len(scenario.value.residues.by_rank)`, instead of naming it. A leading underscore does not make a constant acceptable, and neither does the fact that the file already has some: existing module-level constants are not a pattern to extend.
- A great test suite is reliable and order-independent. Brainstorm potential hidden dependencies between your tests. If your tests rely on global states or network calls, design them to use monkeypatch to isolate environments securely.
- Avoid Mocking as much as possible. Functional tests and integration tests are always superior to highly mocked unit tests.
- Data Pipelines, deep learning models, file input and output, training and sampling scripts, all MUST have End to End tests to validate the entire workflow.
- All tests must take at least one fixture as input. If a pytest does not use `@pytest.mark.parametrize`, justify why this test should test no other edge cases. Never construct objects to test or expected outputs to tests againt inside a test. Use pytest fixtures and `@pytest.mark.parametrize`. The one exception is a call expected to raise: make it inside `with pytest.raises(...)` in the test body, using inputs from fixtures (see the thunk rule above).
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
