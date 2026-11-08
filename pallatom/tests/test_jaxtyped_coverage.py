"""CI gate: every @jaxtyped-decorated symbol must appear in its test file.

Parses all source files under ``pallatom/`` with the ``ast`` module,
collects every ``@jaxtyped``-decorated symbol, and verifies that each
name appears as an AST identifier in the corresponding test file under
``pallatom/tests/``.
"""

import ast
import dataclasses
import enum
import pathlib
from typing import cast, override

import pytest
from jaxtyping import jaxtyped


class JaxtypedCollector(ast.NodeVisitor):
    """Walk an AST and collect lookup name for every @jaxtyped-decorated node.

    Rules:
    - Decorated standalone function ``foo`` → lookup name ``"foo"``.
    - Decorated method inside ``ClassName`` → lookup name ``"ClassName"``.
    - Decorated class (dataclass) ``MyClass`` → lookup name ``"MyClass"``.

    The decorator name is read from ``jaxtyped.__name__`` so the
    gate follows the library rather than a hard-coded string.
    """

    def __init__(self) -> None:
        self._class_stack: list[str] = []
        self.lookup_names: set[str] = set()

    @staticmethod
    def is_jaxtyped(decorator: ast.expr) -> bool:
        """Return True if decorator node is @jaxtyped or @jaxtyped(...).

        Handles both bare ``@jaxtyped`` name references and call forms such
        as ``@jaxtyped(typechecker=beartype)``.

        Args:
            decorator: An AST expression node from a decorator list.

        Returns:
            True when the decorator resolves to the ``jaxtyped`` symbol.
        """
        target = (
            decorator.func if isinstance(decorator, ast.Call) else decorator
        )
        return isinstance(target, ast.Name) and target.id == jaxtyped.__name__

    @override
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Record class name if @jaxtyped; recurse into body for methods.

        When the class itself is decorated with ``@jaxtyped`` (a dataclass),
        the class name is added directly.  Either way, the visitor descends
        into the class body so that decorated methods are also collected.

        Args:
            node: The AST node for the class definition.
        """
        if any(self.is_jaxtyped(d) for d in node.decorator_list):
            self.lookup_names.add(node.name)
        self._class_stack.append(node.name)
        self.generic_visit(node)
        _ = self._class_stack.pop()

    def _visit_any_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        """Record enclosing class for methods, name for module-level functions.

        When inside a class body, the enclosing class name is recorded (so
        that any ``@jaxtyped`` method maps back to its class). For top-level
        functions the function's own name is recorded directly.

        Args:
            node: The AST node for either a sync or async function definition.
        """
        if any(self.is_jaxtyped(d) for d in node.decorator_list):
            self.lookup_names.add((self._class_stack or [node.name])[-1])
        self.generic_visit(node)

    @override
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Handle sync function definitions.

        Args:
            node: The AST node for the synchronous function definition.
        """
        self._visit_any_function(node)

    @override
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Handle async function definitions.

        Args:
            node: The AST node for the asynchronous function definition.
        """
        self._visit_any_function(node)


@dataclasses.dataclass(frozen=True)
class CollectorCase:
    """Source snippet and the lookup names the collector should report.

    Attributes:
        source: Python source to parse.
        expected: Sorted lookup names the collector must record.
    """

    source: str
    expected: tuple[str, ...]


class CollectorScenario(enum.Enum):
    """Decorator layouts the collector must map to lookup names.

    Members:
        bare_decorator: ``@jaxtyped`` with no call on a function.
        called_decorator: ``@jaxtyped(typechecker=...)`` on a function.
        method: A decorated method records its enclosing class name, not
            the method name.
        decorated_class: A decorated class records its own name.
        async_function: A decorated ``async def`` is recorded too.
        undecorated: Undecorated code, or another decorator, records nothing.
    """

    bare_decorator = CollectorCase("@jaxtyped\ndef f(): ...", ("f",))
    called_decorator = CollectorCase("@jaxtyped(t=b)\ndef g(): ...", ("g",))
    method = CollectorCase("class C:\n @jaxtyped\n def m(s): ...", ("C",))
    decorated_class = CollectorCase("@jaxtyped(t=b)\nclass D: ...", ("D",))
    async_function = CollectorCase("@jaxtyped\nasync def a(): ...", ("a",))
    undecorated = CollectorCase("@other\ndef h(): ...\nclass E: ...", ())


@dataclasses.dataclass(frozen=True)
class CoverageReport:
    """Result of scanning ``pallatom/`` for untested @jaxtyped symbols.

    Attributes:
        scanned_symbols: Number of @jaxtyped lookup names found in source.
        missing: One line per lookup name absent from its test file.
    """

    scanned_symbols: int
    missing: tuple[str, ...]


@pytest.fixture
def collector_scenario(request: pytest.FixtureRequest) -> CollectorScenario:
    """Return the ``CollectorScenario`` member routed in via indirect param.

    Args:
        request: Pytest request carrying the scenario in ``param``.

    Returns:
        The scenario under test.
    """
    return cast(CollectorScenario, request.param)


@pytest.fixture
def collected_names(collector_scenario: CollectorScenario) -> tuple[str, ...]:
    """Run ``JaxtypedCollector`` over the scenario's source snippet.

    Args:
        collector_scenario: Scenario supplying the source to parse.

    Returns:
        Sorted lookup names the collector recorded.
    """
    collector = JaxtypedCollector()
    collector.visit(ast.parse(collector_scenario.value.source))
    return tuple(sorted(collector.lookup_names))


@pytest.fixture
def tests_root() -> pathlib.Path:
    """Return the ``pallatom/tests`` directory that holds this file.

    Returns:
        Absolute path to the tests package.
    """
    return pathlib.Path(__file__).parent


@pytest.fixture
def jaxtyped_coverage(tests_root: pathlib.Path) -> CoverageReport:
    """Scan every non-test source file and report uncovered @jaxtyped names.

    A source file ``pallatom/{pkg}/{mod}.py`` maps to the test file
    ``pallatom/tests/{pkg}/test_{mod}.py``. A lookup name is covered when it
    appears as an ``ast.Name`` or an import alias in that test file, so
    mentions in comments and strings do not count.

    Args:
        tests_root: The ``pallatom/tests`` directory.

    Returns:
        Report with the number of scanned names and every missing entry.
    """
    pallatom_root = tests_root.parent
    missing: list[str] = []
    scanned = 0
    for src_path in sorted(pallatom_root.rglob("*.py")):
        if src_path.is_relative_to(tests_root):
            continue
        collector = JaxtypedCollector()
        collector.visit(ast.parse(src_path.read_text()))
        scanned += len(collector.lookup_names)
        rel = src_path.relative_to(pallatom_root)
        test_path = tests_root / rel.parent / f"test_{rel.name}"
        tree = ast.parse(test_path.read_text() if test_path.exists() else "")
        referenced = {
            node.id if isinstance(node, ast.Name) else node.name.split(".")[-1]
            for node in ast.walk(tree)
            if isinstance(node, ast.Name | ast.alias)
        }
        missing.extend(
            f"{rel}: '{name}' not referenced in {test_path.name}"
            for name in sorted(collector.lookup_names - referenced)
        )
    return CoverageReport(scanned_symbols=scanned, missing=tuple(missing))


@pytest.mark.parametrize(
    "collector_scenario",
    list(CollectorScenario),
    ids=[case.name for case in CollectorScenario],
    indirect=True,
)
def test_collector_maps_decorators_to_lookup_names(
    collector_scenario: CollectorScenario,
    collected_names: tuple[str, ...],
) -> None:
    """The collector records the lookup name each decorator layout implies.

    The coverage gate below is only as good as this mapping: a layout the
    collector skips is a symbol the gate silently never checks.

    Args:
        collector_scenario: Scenario holding the expected names.
        collected_names: Names the collector recorded for the scenario.
    """
    assert collected_names == collector_scenario.value.expected


def test_every_jaxtyped_symbol_is_referenced_in_its_test_file(
    jaxtyped_coverage: CoverageReport,
) -> None:
    """Every @jaxtyped function or class is referenced by its test file.

    The scan is a single repository-wide check rather than a parametrized
    one: the set of source files is discovered at run time, and one failure
    listing every gap lets them all be fixed at once. The scanned count is
    asserted too, so a scan that silently finds nothing cannot pass.

    Args:
        jaxtyped_coverage: Report from scanning ``pallatom/``.
    """
    assert jaxtyped_coverage.scanned_symbols > 0
    assert not jaxtyped_coverage.missing, "\n".join(jaxtyped_coverage.missing)
