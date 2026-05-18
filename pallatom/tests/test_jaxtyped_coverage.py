"""CI gate: every @jaxtyped-decorated function or class must be referenced in its test file."""

import ast
import pathlib


def _is_jaxtyped(decorator: ast.expr) -> bool:
    """Return True if decorator node is @jaxtyped or @jaxtyped(...)."""
    if isinstance(decorator, ast.Name):
        return decorator.id == "jaxtyped"
    if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
        return decorator.func.id == "jaxtyped"
    return False


class _JaxtypedCollector(ast.NodeVisitor):
    """Walk an AST and collect the lookup name for every @jaxtyped-decorated node.

    Rules:
    - Decorated standalone function ``foo`` → lookup name ``"foo"``.
    - Decorated method inside ``ClassName`` → lookup name ``"ClassName"``.
    - Decorated class (dataclass) ``MyClass`` → lookup name ``"MyClass"``.
    """

    def __init__(self) -> None:
        """Initialise collector with empty class stack and result set."""
        self._class_stack: list[str] = []
        self.lookup_names: set[str] = set()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Record class name if @jaxtyped; recurse into body for methods."""
        if any(_is_jaxtyped(d) for d in node.decorator_list):
            self.lookup_names.add(node.name)
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def _visit_any_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Record enclosing class for methods, function name for module-level functions."""
        if any(_is_jaxtyped(d) for d in node.decorator_list):
            if self._class_stack:
                self.lookup_names.add(self._class_stack[-1])
            else:
                self.lookup_names.add(node.name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Handle sync function definitions."""
        self._visit_any_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Handle async function definitions."""
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
    """Every @jaxtyped-decorated function or class must appear in the corresponding test file.

    Scans all source files under ``pallatom/`` (excluding ``tests/``), collects
    every @jaxtyped-decorated function or class, derives the expected test file
    path (``pallatom/{pkg}/{mod}.py`` → ``pallatom/tests/{pkg}/test_{mod}.py``),
    and asserts that the decorated name appears as an AST identifier in that file.
    Fails with a complete list of missing entries so all gaps can be fixed at once.
    """
    pallatom_root = pathlib.Path(__file__).parent.parent
    missing: list[str] = []

    for src_path in sorted(pallatom_root.rglob("*.py")):
        if "tests" in src_path.parts:
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
        "The following @jaxtyped functions/classes have no reference in their test file:\n"
        + "\n".join(f"  {m}" for m in missing)
    )
