"""Enforce jaxtyping annotations: bare torch.Tensor (or Tensor) is not allowed.

Every tensor annotation must be wrapped by a jaxtyping dtype class,
e.g. Float[torch.Tensor, "B N C"].  Bare torch.Tensor in function
arguments, return types, or variable annotations is rejected.
"""

import ast
import sys
from pathlib import Path

# All dtype wrapper names exported by jaxtyping.
_JAXTYPING_WRAPPERS = frozenset(
    ["Bool", "Complex", "Float", "Inexact", "Int", "Integer", "Num", "Shaped"],
)


def _is_tensor_node(node: ast.expr) -> bool:
    """Return True if node is the literal torch.Tensor or bare Tensor."""
    if isinstance(node, ast.Attribute):
        return (
            node.attr == "Tensor"
            and isinstance(node.value, ast.Name)
            and node.value.id == "torch"
        )
    return isinstance(node, ast.Name) and node.id == "Tensor"


def _collect_bare_tensors(node: ast.expr, out: list[tuple[int, int]]) -> None:
    """Recursively find bare Tensor references that are not jaxtyping-wrapped."""
    if _is_tensor_node(node):
        out.append((node.lineno, node.col_offset))
        return

    if isinstance(node, ast.Subscript):
        # Float[torch.Tensor, "..."] — stop; the Tensor is properly wrapped.
        if (
            isinstance(node.value, ast.Name)
            and node.value.id in _JAXTYPING_WRAPPERS
        ):
            return
        _collect_bare_tensors(node.value, out)
        _collect_bare_tensors(node.slice, out)
    elif isinstance(node, ast.Tuple | ast.List):
        for elt in node.elts:
            _collect_bare_tensors(elt, out)
    elif isinstance(node, ast.BinOp):  # X | Y union syntax (Python 3.10+)
        _collect_bare_tensors(node.left, out)
        _collect_bare_tensors(node.right, out)


def _has_jaxtyped_decorator(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
) -> bool:
    """Return True if the node has @jaxtyped(typechecker=beartype) in its decorator list."""
    for dec in node.decorator_list:
        if not isinstance(dec, ast.Call):
            continue
        func = dec.func
        if not (isinstance(func, ast.Name) and func.id == "jaxtyped"):
            continue
        for kw in dec.keywords:
            if (
                kw.arg == "typechecker"
                and isinstance(kw.value, ast.Name)
                and kw.value.id == "beartype"
            ):
                return True
    return False


class _AnnotationVisitor(ast.NodeVisitor):
    """Collect (line, col, description) for every bare Tensor annotation.

    Only annotations inside functions/classes decorated with
    @jaxtyped(typechecker=beartype) are checked.
    """

    def __init__(self) -> None:
        self.violations: list[tuple[int, int, str]] = []
        self._jaxtyped_depth: int = 0

    def _check(self, annotation: ast.expr | None, description: str) -> None:
        if annotation is None:
            return
        found: list[tuple[int, int]] = []
        _collect_bare_tensors(annotation, found)
        for line, col in found:
            self.violations.append((line, col, description))

    def _visit_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        has_jaxtyped = _has_jaxtyped_decorator(node)
        if has_jaxtyped:
            all_args = (
                node.args.posonlyargs + node.args.args + node.args.kwonlyargs
            )
            for arg in all_args:
                self._check(arg.annotation, f"argument '{arg.arg}'")
            if node.args.vararg:
                self._check(
                    node.args.vararg.annotation,
                    f"argument '*{node.args.vararg.arg}'",
                )
            if node.args.kwarg:
                self._check(
                    node.args.kwarg.annotation,
                    f"argument '**{node.args.kwarg.arg}'",
                )
            self._check(node.returns, "return type")
            self._jaxtyped_depth += 1
        self.generic_visit(node)
        if has_jaxtyped:
            self._jaxtyped_depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit a synchronous function definition."""
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit an asynchronous function definition."""
        self._visit_function(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit a class definition."""
        has_jaxtyped = _has_jaxtyped_decorator(node)
        if has_jaxtyped:
            self._jaxtyped_depth += 1
        self.generic_visit(node)
        if has_jaxtyped:
            self._jaxtyped_depth -= 1

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Visit a variable annotation."""
        if self._jaxtyped_depth > 0:
            self._check(node.annotation, "variable annotation")
        self.generic_visit(node)


def check_file(path: Path) -> list[str]:
    """Return one error string per bare Tensor annotation found in path."""
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []

    visitor = _AnnotationVisitor()
    visitor.visit(tree)

    errors = []
    for line, col, desc in visitor.violations:
        errors.append(
            f"{path}:{line}:{col + 1}: bare Tensor in {desc}"
            ' — use Float[torch.Tensor, "..."] or similar jaxtyping wrapper',
        )
    return errors


def main() -> None:
    """Entry point for pre-commit: receives changed Python files as argv."""
    py_files = [Path(f) for f in sys.argv[1:] if f.endswith(".py")]
    all_errors: list[str] = []
    for f in py_files:
        all_errors.extend(check_file(f))
    if all_errors:
        print(
            "bare torch.Tensor annotations found — wrap with jaxtyping dtype:\n",
        )
        for err in all_errors:
            print(f"  {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
