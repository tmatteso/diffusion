"""Tests for the scripts/check_test_quality.py ratchet."""

import json
import pathlib
import subprocess
import sys
from dataclasses import dataclass

import pytest

_SCRIPT = pathlib.Path(__file__).parents[2] / "scripts/check_test_quality.py"
_TEST_FILE = "pallatom/tests/test_sample.py"
_NO_FIXTURE = "def test_a() -> None:\n    assert True\n"
_WITH_FIXTURE = "def test_a(tmp_path) -> None:\n    assert tmp_path\n"
_USEFIXTURES = (
    "import pytest\n"
    '@pytest.mark.usefixtures("tmp_path")\n'
    "def test_a() -> None:\n    assert True\n"
)
_HELPER_ONLY = "def helper() -> None:\n    pass\n"
_RANDN = "import torch\nx = torch.randn(2)\n"
_RANDN_LIKE = "import torch\nx = torch.randn_like(torch.zeros(2))\n"
_RANDOM_STRING = 'x = "torch.randn(2)"\n'


@dataclass
class Project:
    """A throwaway repository layout for running the script against."""

    root: pathlib.Path

    def write_test(self, source: str) -> None:
        """Write the single test file.

        Args:
            source: Python source for the file.
        """
        path = self.root / _TEST_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = path.write_text(source, encoding="utf-8")

    def write_ceiling(self, rule: str, ceiling: int) -> None:
        """Write a baseline holding one ceiling for the test file.

        Args:
            rule: The rule the ceiling applies to.
            ceiling: The allowed violation count.
        """
        path = self.root / "scripts/test_quality_baseline.json"
        path.parent.mkdir(exist_ok=True)
        text = json.dumps({rule: {_TEST_FILE: ceiling}})
        _ = path.write_text(text, encoding="utf-8")

    def run(self, *args: str) -> int:
        """Run the script from the project root.

        Args:
            *args: Command-line arguments for the script.

        Returns:
            The script's exit code.
        """
        return subprocess.run(  # noqa: S603
            [sys.executable, str(_SCRIPT), *args],
            cwd=self.root,
            check=False,
            capture_output=True,
        ).returncode


@pytest.fixture
def project(tmp_path: pathlib.Path) -> Project:
    """Return an empty project rooted in a temporary directory."""
    return Project(tmp_path)


@pytest.mark.parametrize(
    ("rule", "source", "ceiling", "expected"),
    [
        pytest.param("fixtures", _NO_FIXTURE, 0, 1, id="new_fixtureless_test"),
        pytest.param("fixtures", _NO_FIXTURE, 1, 0, id="fixtureless_at_limit"),
        pytest.param("fixtures", _WITH_FIXTURE, 0, 0, id="takes_a_fixture"),
        pytest.param("fixtures", _USEFIXTURES, 0, 0, id="usefixtures_mark"),
        pytest.param("fixtures", _HELPER_ONLY, 0, 0, id="non_test_function"),
        pytest.param("fixtures", _WITH_FIXTURE, 1, 1, id="fixed_needs_rewrite"),
        pytest.param("random", _RANDN, 0, 1, id="new_randn_call"),
        pytest.param("random", _RANDN_LIKE, 0, 1, id="new_randn_like_call"),
        pytest.param("random", _RANDN, 1, 0, id="randn_at_limit"),
        pytest.param("random", _RANDOM_STRING, 0, 0, id="string_is_not_a_call"),
    ],
)
def test_rule_exit_code(
    project: Project,
    rule: str,
    source: str,
    ceiling: int,
    expected: int,
) -> None:
    """Exit non-zero exactly when a count differs from its ceiling."""
    project.write_test(source)
    project.write_ceiling(rule, ceiling)

    assert project.run("--rule", rule) == expected


@pytest.mark.parametrize(
    ("rule", "source"),
    [
        pytest.param("fixtures", _NO_FIXTURE, id="fixtures"),
        pytest.param("random", _RANDN, id="random"),
    ],
)
def test_write_baseline_accepts_current_counts(
    project: Project,
    rule: str,
    source: str,
) -> None:
    """Pass after --write-baseline records the existing violations."""
    project.write_test(source)
    project.write_ceiling(rule, 0)

    assert project.run("--write-baseline") == 0
    assert project.run("--rule", rule) == 0
