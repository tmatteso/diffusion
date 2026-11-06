"""Ratchet the test rules in pallatom/tests/CLAUDE.md.

Each rule counts its violations per test file and compares them with the
ceilings in scripts/test_quality_baseline.json. A count above its ceiling
fails, so a file can't get worse. A count below its ceiling also fails until
the baseline is rewritten, so fixed violations can't come back.

Rules:
    fixtures: a test takes no fixture and has no usefixtures mark.
    random: a test file calls one of torch's random tensor constructors.

Rewrite the baseline with --write-baseline after fixing violations.
"""

import argparse
import ast
import json
import sys
from pathlib import Path

_TESTS_DIR = Path("pallatom/tests")
_BASELINE = Path("scripts/test_quality_baseline.json")
_RANDOM_CALLS = frozenset(
    [
        "rand",
        "rand_like",
        "randint",
        "randint_like",
        "randn",
        "randn_like",
        "randperm",
    ],
)
_ADVICE = {
    "fixtures": "take at least one fixture",
    "random": "use concrete, human-readable tensors, not torch.rand*",
}

Baseline = dict[str, dict[str, int]]


def _fixtureless_tests(tree: ast.Module) -> list[int]:
    """Return the line of every test with no fixture.

    Args:
        tree: The parsed test file.

    Returns:
        Line numbers of test functions with no parameters and no
        usefixtures mark.
    """
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if not node.name.startswith("test_"):
            continue
        params = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
        marks = [ast.unparse(mark) for mark in node.decorator_list]
        if not params and not any("usefixtures" in mark for mark in marks):
            lines.append(node.lineno)
    return lines


def _random_calls(tree: ast.Module) -> list[int]:
    """Return the line of every torch random tensor constructor call.

    Args:
        tree: The parsed test file.

    Returns:
        Line numbers of calls such as torch.randn(...).
    """
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _RANDOM_CALLS
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
    ]


_RULES = {"fixtures": _fixtureless_tests, "random": _random_calls}


def _scan(rule: str) -> dict[str, list[int]]:
    """Return the violating lines of each test file that has any.

    Args:
        rule: The rule to apply.

    Returns:
        Violating line numbers, keyed by POSIX path.
    """
    found: dict[str, list[int]] = {}
    for path in sorted(_TESTS_DIR.rglob("test_*.py")):
        lines = _RULES[rule](ast.parse(path.read_text(encoding="utf-8")))
        if lines:
            found[path.as_posix()] = lines
    return found


def _load_baseline() -> Baseline:
    """Return the stored ceilings, or an empty baseline if none exists."""
    if not _BASELINE.exists():
        return {}
    baseline: Baseline = json.loads(_BASELINE.read_text(encoding="utf-8"))
    return baseline


def _write_baseline() -> None:
    """Rewrite the baseline from the current count of every rule."""
    baseline = {
        rule: {path: len(lines) for path, lines in _scan(rule).items()}
        for rule in sorted(_RULES)
    }
    text = json.dumps(baseline, indent=2, sort_keys=True)
    _ = _BASELINE.write_text(f"{text}\n", encoding="utf-8")


def _check(rule: str) -> list[str]:
    """Compare a rule's counts with its ceilings.

    Args:
        rule: The rule to apply.

    Returns:
        One error per file whose count differs from its ceiling.
    """
    found = _scan(rule)
    ceilings = _load_baseline().get(rule, {})
    errors: list[str] = []
    for path in sorted(found.keys() | ceilings.keys()):
        lines = found.get(path, [])
        ceiling = ceilings.get(path, 0)
        if len(lines) > ceiling:
            where = ", ".join(str(line) for line in lines)
            errors.append(
                f"{path}: {len(lines)} {rule} violations, ceiling {ceiling}"
                f" (lines {where}) — {_ADVICE[rule]}",
            )
        elif len(lines) < ceiling:
            errors.append(
                f"{path}: {len(lines)} {rule} violations, ceiling {ceiling}"
                " — run scripts/check_test_quality.py --write-baseline",
            )
    return errors


def main() -> None:
    """Check one rule, or rewrite the baseline for all of them."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    _ = group.add_argument("--rule", choices=sorted(_RULES))
    _ = group.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args()
    rule: str | None = args.rule

    if rule is None:
        _write_baseline()
        return
    errors = _check(rule)
    if errors:
        print("\n".join(errors))
        sys.exit(1)


if __name__ == "__main__":
    main()
