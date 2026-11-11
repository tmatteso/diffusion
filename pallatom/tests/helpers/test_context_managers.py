"""Tests for context manager helpers.

Covers StructlogConfig end to end: which events reach the JSONL file and the
console on each rank, truncation of a stale file, line buffering, and closing
the file on both clean and exceptional exits.
"""

import dataclasses
import enum
import json
import pathlib
from typing import cast

import pytest
import structlog
from helpers.context_managers import StructlogConfig
from structlog.typing import FilteringBoundLogger


class Reach(enum.Flag):
    """Where one logged event ended up.

    Members:
        nowhere: Dropped: not in the log file and not printed.
        file: Written to the log file as a JSON line.
        console: Printed to stdout by ``ConsoleRenderer``.
        both: Written to the log file and printed.
    """

    nowhere = 0
    file = enum.auto()
    console = enum.auto()
    both = file | console


@dataclasses.dataclass(frozen=True)
class LoggedRun:
    """Where the two events logged through a StructlogConfig ended up.

    ``log_two_events`` logs ``info_event`` at INFO level and then
    ``warning_event`` at WARNING level, exits the config, and reads back
    both the log file and the captured console output.

    Attributes:
        info: Where the INFO event ended up.
        warning: Where the WARNING event ended up.
        file_created: Whether the log file exists after the run.
        file_lines: Number of JSON lines in the log file; 0 when there is
            no file.
    """

    info: Reach
    warning: Reach
    file_created: bool
    file_lines: int


@dataclasses.dataclass(frozen=True)
class RankCase:
    """A process rank and where StructlogConfig must send each event.

    Attributes:
        rank: Global rank; the config gets ``is_rank_zero=(rank == 0)``.
        info: Where the INFO event must end up.
        warning: Where the WARNING event must end up.
        file_created: Whether the rank must create the log file.
    """

    rank: int
    info: Reach
    warning: Reach
    file_created: bool


class RankScenario(enum.Enum):
    """Ranks whose logging output StructlogConfig gates differently.

    Members:
        rank_zero: Writes every event to the file and prints INFO and
            above.
        other_rank: Creates no file and prints only WARNING and above, so
            failures on non-zero ranks stay visible.
    """

    rank_zero = RankCase(0, Reach.both, Reach.both, file_created=True)
    other_rank = RankCase(1, Reach.nowhere, Reach.console, file_created=False)


def event_reach(event: str, in_file: set[str], console: str) -> Reach:
    """Combine where ``event`` was found.

    Args:
        event: Event name to look for.
        in_file: Event names read back from the log file.
        console: Captured stdout and stderr.

    Returns:
        The file and console flags the event earned.
    """
    file = Reach.file if event in in_file else Reach.nowhere
    shown = Reach.console if event in console else Reach.nowhere
    return file | shown


def log_two_events(
    config: StructlogConfig,
    path: pathlib.Path,
    capture: pytest.CaptureFixture[str],
) -> LoggedRun:
    """Log one INFO and one WARNING event through ``config``.

    Enters the config, logs ``info_event`` at INFO and ``warning_event`` at
    WARNING through ``structlog.get_logger()``, exits, then reads back the
    log file and the captured console output.

    Args:
        config: Unentered config to log through.
        path: File the config writes to.
        capture: Captures console output from ``ConsoleRenderer``.

    Returns:
        Where each event ended up, and what the file holds.
    """
    with config:
        log = cast(FilteringBoundLogger, structlog.get_logger())
        log.info("info_event")
        log.warning("warning_event")
    captured = capture.readouterr()
    console = captured.out + captured.err
    created = path.exists()
    lines = path.read_text().splitlines() if created else []
    in_file = {cast(str, json.loads(line)["event"]) for line in lines}
    return LoggedRun(
        info=event_reach("info_event", in_file, console),
        warning=event_reach("warning_event", in_file, console),
        file_created=created,
        file_lines=len(lines),
    )


@pytest.fixture
def log_path(tmp_path: pathlib.Path) -> pathlib.Path:
    """Return the JSONL path StructlogConfig writes to; not created yet.

    Args:
        tmp_path: Per-test temporary directory.

    Returns:
        Path to ``run.jsonl`` inside ``tmp_path``.
    """
    return tmp_path / "run.jsonl"


@pytest.fixture
def stale_log_path(log_path: pathlib.Path) -> pathlib.Path:
    """Pre-populate ``log_path`` with five stale JSON lines.

    Args:
        log_path: Path to fill with stale content.

    Returns:
        The same path, now holding lines from an earlier run.
    """
    _ = log_path.write_text('{"event": "stale"}\n' * 5)
    return log_path


@pytest.fixture
def rank_config(
    rank_scenario: RankScenario,
    log_path: pathlib.Path,
) -> StructlogConfig:
    """Build an unentered StructlogConfig for the scenario's rank.

    Args:
        rank_scenario: Scenario supplying the rank; the requesting test
            parametrizes it.
        log_path: Path the config writes to.

    Returns:
        The config, not yet entered.
    """
    return StructlogConfig(
        is_rank_zero=rank_scenario.value.rank == 0,
        log_file=log_path,
    )


@pytest.fixture
def rank_zero_config(log_path: pathlib.Path) -> StructlogConfig:
    """Build an unentered rank-zero StructlogConfig writing to ``log_path``.

    Args:
        log_path: Path the config writes to.

    Returns:
        The config, not yet entered.
    """
    return StructlogConfig(is_rank_zero=True, log_file=log_path)


@pytest.fixture
def rank_run(
    rank_config: StructlogConfig,
    log_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> LoggedRun:
    """Log the two events through the scenario's rank config.

    Args:
        rank_config: Config built for the scenario's rank.
        log_path: File the config writes to.
        capsys: Captures console output.

    Returns:
        Where each event ended up.
    """
    return log_two_events(rank_config, log_path, capsys)


@pytest.fixture
def stale_config(stale_log_path: pathlib.Path) -> StructlogConfig:
    """Build an unentered rank-zero config over a file with stale lines.

    Args:
        stale_log_path: Log path already holding stale lines.

    Returns:
        The config, not yet entered.
    """
    return StructlogConfig(is_rank_zero=True, log_file=stale_log_path)


@pytest.fixture
def stale_run(
    stale_config: StructlogConfig,
    stale_log_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> LoggedRun:
    """Log the two events through a config whose file starts stale.

    Args:
        stale_config: Rank-zero config over the stale file.
        stale_log_path: File the config writes to.
        capsys: Captures console output.

    Returns:
        Where each event ended up, and how many lines the file holds.
    """
    return log_two_events(stale_config, stale_log_path, capsys)


@pytest.fixture
def lines_before_exit(
    rank_zero_config: StructlogConfig,
    log_path: pathlib.Path,
) -> int:
    """Log one event, then count the file's lines while still entered.

    Args:
        rank_zero_config: Config to log through.
        log_path: File the config writes to.

    Returns:
        Number of JSON lines on disk before ``__exit__`` closes the file.
    """
    with rank_zero_config:
        cast(FilteringBoundLogger, structlog.get_logger()).info("info_event")
        return len(log_path.read_text().splitlines())


@pytest.mark.parametrize(
    "rank_scenario",
    list(RankScenario),
    ids=[case.name for case in RankScenario],
)
def test_structlog_config_routes_events_by_rank(
    rank_scenario: RankScenario,
    rank_run: LoggedRun,
) -> None:
    """Each rank sends events to the file and console its scenario expects.

    Rank zero writes every event as a JSON line and prints INFO and above;
    any other rank creates no file and prints only WARNING and above.

    Args:
        rank_scenario: Scenario holding where each event must end up.
        rank_run: Where each event ended up for the scenario's rank.
    """
    case = rank_scenario.value
    logged = rank_run
    assert logged.info == case.info
    assert logged.warning == case.warning
    assert logged.file_created == case.file_created


def test_structlog_config_truncates_existing_file(
    stale_run: LoggedRun,
) -> None:
    """A pre-existing log file is truncated on entry rather than appended to.

    The file starts with five stale lines; afterwards it must hold exactly
    one line per event just logged, and nothing else. Only the rank-zero
    case applies, since no other rank opens the file.

    Args:
        stale_run: What logging through the stale file left behind.
    """
    logged = stale_run
    in_file = [Reach.file in reach for reach in (logged.info, logged.warning)]
    assert all(in_file)
    assert logged.file_lines == sum(in_file)


def test_structlog_config_file_is_line_buffered(
    lines_before_exit: int,
) -> None:
    """Each JSON line reaches disk as soon as it is logged, before exit.

    A crashed run therefore keeps every line logged before the crash. The
    single rank-zero case is the only one with a file to buffer.

    Args:
        lines_before_exit: JSON lines on disk while the config was entered.
    """
    assert lines_before_exit == 1


def test_structlog_config_write_log_line_before_enter_is_passthrough(
    rank_zero_config: StructlogConfig,
    log_path: pathlib.Path,
) -> None:
    """``write_log_line`` with no open file returns the event dict as is.

    The processor protocol requires it to return the same dict; with no file
    open yet it must neither raise nor create one. No other state applies:
    once entered, the end-to-end routing test covers the processor.

    Args:
        rank_zero_config: Config that has not been entered.
        log_path: File the config would write to.
    """
    event: dict[str, object] = {"event": "no_file"}
    assert rank_zero_config.write_log_line(None, None, event) is event
    assert not log_path.exists()


def test_structlog_config_closes_file_on_clean_exit(
    rank_zero_config: StructlogConfig,
) -> None:
    """Leaving the block normally closes the log file handle.

    Args:
        rank_zero_config: Config to enter and exit.
    """
    with rank_zero_config:
        pass
    assert rank_zero_config.f is not None
    assert rank_zero_config.f.closed


def test_structlog_config_closes_file_and_propagates_exception(
    rank_zero_config: StructlogConfig,
) -> None:
    """An exception inside the block propagates and the file is still closed.

    ``__exit__`` must not swallow the error, and must release the handle on
    the error path as well as the clean one.

    Args:
        rank_zero_config: Config to enter and exit.

    Raises:
        RuntimeError: Raised inside the block to exercise the error path;
            caught by ``pytest.raises``.
    """
    with pytest.raises(RuntimeError), rank_zero_config:
        raise RuntimeError
    assert rank_zero_config.f is not None
    assert rank_zero_config.f.closed
