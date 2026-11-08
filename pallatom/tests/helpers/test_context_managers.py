"""Tests for context manager helpers.

Covers StructlogConfig end to end: which events reach the JSONL file and the
console on each rank, truncation of a stale file, line buffering, and closing
the file on both clean and exceptional exits.
"""

import dataclasses
import enum
import json
import pathlib
from collections.abc import Callable
from typing import cast

import pytest
import structlog
from helpers.context_managers import StructlogConfig
from structlog.typing import FilteringBoundLogger


@dataclasses.dataclass(frozen=True)
class LoggedRun:
    """What one INFO and one WARNING event left behind after a logging run.

    The INFO event is named ``quiet`` and the WARNING event ``loud``; the
    names avoid ``info`` and ``warning``, which ``ConsoleRenderer`` prints as
    level labels and would match on every line.

    Attributes:
        file_events: ``event`` field of each JSON line in the log file, in
            order, or ``None`` when no log file was created.
        console_events: Which of ``quiet`` and ``loud`` appear
            in captured stdout/stderr, in that order.
    """

    file_events: tuple[str, ...] | None
    console_events: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class RankCase:
    """Global rank of the process and the output StructlogConfig must produce.

    Attributes:
        rank: Global rank; ``is_rank_zero`` is passed as ``rank == 0``.
        expected: Output after logging ``quiet`` then ``loud``.
    """

    rank: int
    expected: LoggedRun


class RankScenario(enum.Enum):
    """Ranks whose logging output StructlogConfig gates differently.

    Members:
        rank_zero: Writes every event to the file and shows INFO and above
            on the console.
        other_rank: Creates no file and shows only WARNING and above, so
            failures on non-zero ranks stay visible.
    """

    rank_zero = RankCase(0, LoggedRun(("quiet", "loud"), ("quiet", "loud")))
    other_rank = RankCase(1, LoggedRun(None, ("loud",)))


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
def rank_scenario(request: pytest.FixtureRequest) -> RankScenario:
    """Return the ``RankScenario`` member routed in via indirect param.

    Args:
        request: Pytest request carrying the scenario in ``param``.

    Returns:
        The scenario under test.
    """
    return cast(RankScenario, request.param)


@pytest.fixture
def rank_config(
    rank_scenario: RankScenario,
    log_path: pathlib.Path,
) -> StructlogConfig:
    """Build an unentered StructlogConfig for the scenario's rank.

    Args:
        rank_scenario: Scenario supplying the rank.
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
def run_logging(
    log_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> Callable[[StructlogConfig], LoggedRun]:
    """Factory that logs one INFO and one WARNING event through a config.

    The returned callable enters the config, logs ``quiet`` at INFO and
    ``loud`` at WARNING through ``structlog.get_logger()``, exits,
    then reads back the log file and the captured console output.

    Args:
        log_path: Path the configs passed in write to.
        capsys: Captures console output from ``ConsoleRenderer``.

    Returns:
        Callable mapping a config to the ``LoggedRun`` it produced.
    """

    def run(config: StructlogConfig) -> LoggedRun:
        """Log through ``config`` and collect what it wrote.

        Args:
            config: Unentered config to log through.

        Returns:
            Events found in the file and on the console.
        """
        with config:
            log = cast(FilteringBoundLogger, structlog.get_logger())
            log.info("quiet")
            log.warning("loud")
        captured = capsys.readouterr()
        console = captured.out + captured.err
        file_events = (
            tuple(
                cast(str, json.loads(line)["event"])
                for line in log_path.read_text().splitlines()
            )
            if log_path.exists()
            else None
        )
        return LoggedRun(
            file_events=file_events,
            console_events=tuple(
                event for event in ("quiet", "loud") if event in console
            ),
        )

    return run


@pytest.fixture
def events_before_exit(
    rank_zero_config: StructlogConfig,
    log_path: pathlib.Path,
) -> tuple[str, ...]:
    """Log one event, then read the file while the config is still entered.

    Args:
        rank_zero_config: Config to log through.
        log_path: File the config writes to.

    Returns:
        ``event`` field of each JSON line on disk before ``__exit__`` closes
        the file.
    """
    with rank_zero_config:
        cast(FilteringBoundLogger, structlog.get_logger()).info("quiet")
        lines = log_path.read_text().splitlines()
    return tuple(cast(str, json.loads(line)["event"]) for line in lines)


@pytest.mark.parametrize(
    "rank_scenario",
    list(RankScenario),
    ids=[case.name for case in RankScenario],
    indirect=True,
)
def test_structlog_config_routes_events_by_rank(
    rank_scenario: RankScenario,
    rank_config: StructlogConfig,
    run_logging: Callable[[StructlogConfig], LoggedRun],
) -> None:
    """Each rank sends events to the file and console its scenario expects.

    Rank zero writes every event as a JSON line and prints INFO and above;
    any other rank creates no file and prints only WARNING and above.

    Args:
        rank_scenario: Scenario holding the expected output.
        rank_config: Config built for the scenario's rank.
        run_logging: Factory that logs through a config.
    """
    assert run_logging(rank_config) == rank_scenario.value.expected


def test_structlog_config_truncates_existing_file(
    stale_log_path: pathlib.Path,
    run_logging: Callable[[StructlogConfig], LoggedRun],
) -> None:
    """A pre-existing log file is truncated on entry rather than appended to.

    Only the rank-zero case applies, since no other rank opens the file.

    Args:
        stale_log_path: Log path already holding stale lines.
        run_logging: Factory that logs through a config.
    """
    config = StructlogConfig(is_rank_zero=True, log_file=stale_log_path)
    logged = run_logging(config)
    assert logged.file_events == ("quiet", "loud")


def test_structlog_config_file_is_line_buffered(
    events_before_exit: tuple[str, ...],
) -> None:
    """Each JSON line reaches disk as soon as it is logged, before exit.

    A crashed run therefore keeps every line logged before the crash. The
    single rank-zero case is the only one with a file to buffer.

    Args:
        events_before_exit: Events read while the config was entered.
    """
    assert events_before_exit == ("quiet",)


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
