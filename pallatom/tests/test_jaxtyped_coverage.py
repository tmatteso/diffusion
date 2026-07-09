"""CI gate: every @jaxtyped-decorated symbol must appear in its test file.

Parses all source files under ``pallatom/`` with the ``ast`` module,
collects every ``@jaxtyped``-decorated symbol, and verifies that each
name appears as an AST identifier in the corresponding test file under
``pallatom/tests/``.
"""

import ast
import pathlib
from typing import override

_JAXTYPED_NAME = "jaxtyped"
_TESTS_DIR = "tests"


def _is_jaxtyped(decorator: ast.expr) -> bool:
    """Return True if decorator node is @jaxtyped or @jaxtyped(...).

    Handles both bare ``@jaxtyped`` name references and call forms such as
    ``@jaxtyped(typechecker=beartype)``.

    Args:
        decorator: An AST expression node from a decorator list.

    Returns:
        True when the decorator resolves to the ``jaxtyped`` symbol.
    """
    if isinstance(decorator, ast.Name):
        return decorator.id == _JAXTYPED_NAME
    if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
        return decorator.func.id == _JAXTYPED_NAME
    return False


class _JaxtypedCollector(ast.NodeVisitor):
    """Walk an AST and collect lookup name for every @jaxtyped-decorated node.

    Rules:
    - Decorated standalone function ``foo`` → lookup name ``"foo"``.
    - Decorated method inside ``ClassName`` → lookup name ``"ClassName"``.
    - Decorated class (dataclass) ``MyClass`` → lookup name ``"MyClass"``.

    """

    def __init__(self) -> None:
        self._class_stack: list[str] = []
        self.lookup_names: set[str] = set()

    @override
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Record class name if @jaxtyped; recurse into body for methods.

        When the class itself is decorated with ``@jaxtyped`` (a dataclass),
        the class name is added directly.  Either way, the visitor descends
        into the class body so that decorated methods are also collected.

        Args:
            node: The AST node for the class definition.
        """
        if any(_is_jaxtyped(d) for d in node.decorator_list):
            self.lookup_names.add(node.name)
        self._class_stack.append(node.name)
        self.generic_visit(node)
        _ = self._class_stack.pop()

    def _visit_any_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        """Record enclosing class for methods, name for module-level functions.

        When inside a class body, the enclosing class name is recorded (so that
        any ``@jaxtyped`` method maps back to its class). For top-level
        functions the function's own name is recorded directly.

        Args:
            node: The AST node for either a sync or async function definition.
        """
        if any(_is_jaxtyped(d) for d in node.decorator_list):
            if self._class_stack:
                self.lookup_names.add(self._class_stack[-1])
            else:
                self.lookup_names.add(node.name)
        self.generic_visit(node)

    @override
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Handle sync function definitions.

        Delegates to ``_visit_any_function`` to record any ``@jaxtyped``
        decoration on synchronous ``def`` statements.

        Args:
            node: The AST node for the synchronous function definition.
        """
        self._visit_any_function(node)

    @override
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Handle async function definitions.

        Delegates to ``_visit_any_function`` to record any ``@jaxtyped``
        decoration on asynchronous ``async def`` statements.

        Args:
            node: The AST node for the asynchronous function definition.
        """
        self._visit_any_function(node)


def _names_referenced_in(test_path: pathlib.Path) -> set[str]:
    """Return all identifiers referenced in a test file.

    Includes both ``ast.Name`` nodes (variables, calls, annotations in code)
    and ``ast.alias`` names (import statements).  Comments and string literals
    are excluded because they don't appear as AST nodes.

    Args:
        test_path: Absolute path to the test file to scan.

    Returns:
        Set of identifier strings found in the file's AST.
    """
    tree = ast.parse(test_path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.alias):
            names.add(node.name.split(".")[-1])
    return names


def test_every_jaxtyped_function_has_a_test() -> None:
    """Every @jaxtyped-decorated function/class has corresponding test file.

    Scans all source files under ``pallatom/`` (excluding ``tests/``), collects
    every @jaxtyped-decorated function or class, derives the expected test file
    path (``pallatom/{pkg}/{mod}.py`` →
    ``pallatom/tests/{pkg}/test_{mod}.py``), and asserts that the decorated
    name appears as an AST identifier in that file. Fails with a complete list
    of missing entries so all gaps can be fixed at once.
    """
    pallatom_root = pathlib.Path(__file__).parent.parent
    missing: list[str] = []

    for src_path in sorted(pallatom_root.rglob("*.py")):
        if _TESTS_DIR in src_path.parts:
            continue

        collector = _JaxtypedCollector()
        collector.visit(ast.parse(src_path.read_text()))
        if not collector.lookup_names:
            continue

        rel = src_path.relative_to(pallatom_root)
        test_path = pallatom_root / "tests" / rel.parent / f"test_{rel.name}"

        if not test_path.exists():
            missing.extend(
                f"{src_path.name}: '{name}' (test file {test_path} not found)"
                for name in sorted(collector.lookup_names)
            )
            continue

        referenced = _names_referenced_in(test_path)
        missing.extend(
            f"{src_path.name}: '{name}'"
            for name in sorted(collector.lookup_names)
            if name not in referenced
        )

    assert not missing, (
        "The following @jaxtyped functions/classes"
        + " have no reference in their test file:\n"
        + "\n".join(f"  {m}" for m in missing)
    )
