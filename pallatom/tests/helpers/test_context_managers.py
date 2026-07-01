"""Tests for context manager helpers.

Covers StructlogConfig: file creation, JSON line writing, truncation, file
handle lifecycle, rank-zero gating, processor chain composition, and exception
propagation.
"""

import json
import pathlib
from typing import cast

import pytest
import structlog
from helpers.context_managers import StructlogConfig


def test_structlog_config_creates_log_file(tmp_path: pathlib.Path) -> None:
    """StructlogConfig creates the log file on disk when entering the context.

    Verifies that new JSONL file exists at the given path after ``with`` block
    exits when ``is_rank_zero=True``.
    """
    path = tmp_path / "run.jsonl"
    with StructlogConfig(is_rank_zero=True, log_file=path):
        pass
    assert path.exists()


def test_structlog_config_write_log_line_returns_event_dict(
    tmp_path: pathlib.Path,
) -> None:
    """_write_log_line passes the event dict through unchanged.

    Verifies that ``write_log_line`` returns exact same dict object that was
    passed in, satisfying the structlog processor protocol.
    """
    path = tmp_path / "run.jsonl"
    cfg = StructlogConfig(is_rank_zero=True, log_file=path)
    with cfg:
        event: dict[str, object] = {"event": "test", "value": 1}
        returned = cfg.write_log_line(None, None, event)
    assert returned is event


def test_structlog_config_writes_json_line(tmp_path: pathlib.Path) -> None:
    """_write_log_line serialises the event dict as a JSON line in log file.

    Verifies that the written line round-trips through ``json.loads`` back to
    original event dict.
    """
    path = tmp_path / "run.jsonl"
    cfg = StructlogConfig(is_rank_zero=True, log_file=path)
    with cfg:
        event: dict[str, object] = {"event": "train", "loss": 0.5}
        _ = cfg.write_log_line(None, None, event)
    with path.open() as fh:
        written = cast(dict[str, object], json.loads(fh.readline()))
    assert written == event


def test_structlog_config_writes_multiple_lines(tmp_path: pathlib.Path) -> None:
    """_write_log_line writes one JSON line per call.

    Verifies that two successive calls produce exactly two lines in log file,
    each deserialising back to the corresponding event dict.
    """
    path = tmp_path / "run.jsonl"
    cfg = StructlogConfig(is_rank_zero=True, log_file=path)
    events: list[dict[str, object]] = [
        {"event": "a", "x": 1},
        {"event": "b", "x": 2},
    ]
    with cfg:
        for ev in events:
            _ = cfg.write_log_line(None, None, ev)
    with path.open() as fh:
        lines = fh.readlines()
    assert len(lines) == len(events)
    assert [json.loads(line) for line in lines] == events


def test_structlog_config_truncates_existing_file(
    tmp_path: pathlib.Path,
) -> None:
    """StructlogConfig truncates a pre-existing log file rather than appending.

    Verifies that stale lines written before the context manager is entered are
    discarded, leaving only the lines written inside the ``with`` block.
    """
    path = tmp_path / "run.jsonl"
    with path.open("w") as fh:
        _ = fh.write('{"stale": true}\n' * 5)
    cfg = StructlogConfig(is_rank_zero=True, log_file=path)
    with cfg:
        _ = cfg.write_log_line(None, None, {"event": "fresh"})
    with path.open() as fh:
        lines = fh.readlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"event": "fresh"}


def test_structlog_config_exit_closes_file(tmp_path: pathlib.Path) -> None:
    """StructlogConfig.__exit__ closes the underlying file handle.

    Verifies that ``cfg.f`` is not ``None`` after the context exits and that
    file object reports ``closed == True``.
    """
    path = tmp_path / "run.jsonl"
    cfg = StructlogConfig(is_rank_zero=True, log_file=path)
    with cfg:
        _ = cfg.write_log_line(None, None, {"event": "inside"})
    assert cfg.f is not None
    assert cfg.f.closed


def test_structlog_config_no_file_when_not_rank_zero(
    tmp_path: pathlib.Path,
) -> None:
    """StructlogConfig does not open a log file when is_rank_zero is False.

    Verifies that no file is created at given path when the context manager is
    used by a non-rank-zero process.
    """
    path = tmp_path / "run.jsonl"
    with StructlogConfig(is_rank_zero=False, log_file=path):
        pass
    assert not path.exists()


def test_structlog_config_f_is_none_before_enter(
    tmp_path: pathlib.Path,
) -> None:
    """StructlogConfig.f is None after construction, before entering context.

    Verifies file handle attribute is not opened during ``__init__``, only
    during ``__enter__``.
    """
    path = tmp_path / "run.jsonl"
    cfg = StructlogConfig(is_rank_zero=True, log_file=path)
    assert cfg.f is None


def test_structlog_config_enter_returns_self(tmp_path: pathlib.Path) -> None:
    """StructlogConfig.__enter__ returns instance itself enabling `as` binding.

    Verifies that the object bound with ``with cfg as result`` is the identical
    ``StructlogConfig`` instance, not a copy or wrapper.
    """
    path = tmp_path / "run.jsonl"
    cfg = StructlogConfig(is_rank_zero=True, log_file=path)
    with cfg as result:
        assert result is cfg


def test_structlog_config_write_log_line_safe_when_f_is_none(
    tmp_path: pathlib.Path,
) -> None:
    """_write_log_line returns event dict without error when no file is open.

    Verifies that calling ``write_log_line`` before entering context manager
    (when ``cfg.f`` is ``None``) does not raise and still returns event dict.
    """
    cfg = StructlogConfig(is_rank_zero=True, log_file=tmp_path / "run.jsonl")
    event: dict[str, object] = {"event": "no_file", "step": 0}
    result = cfg.write_log_line(None, None, event)
    assert result is event


def test_structlog_config_exception_propagates(tmp_path: pathlib.Path) -> None:
    """StructlogConfig.__exit__ does not suppress exceptions raised inside.

    Verifies ``ValueError`` raised inside ``with`` block propagates to the
    caller unchanged.
    """
    path = tmp_path / "run.jsonl"
    with (
        pytest.raises(ValueError),  # noqa: PT011
        StructlogConfig(is_rank_zero=True, log_file=path),
    ):
        raise ValueError


def test_structlog_config_file_closed_on_exception(
    tmp_path: pathlib.Path,
) -> None:
    """StructlogConfig closes log file when exception is raised inside block.

    Verifies that ``cfg.f`` is closed after ``RuntimeError`` propagates out of
    ``with`` block, confirming file handle is properly released on error paths.
    """
    path = tmp_path / "run.jsonl"
    cfg = StructlogConfig(is_rank_zero=True, log_file=path)
    with pytest.raises(RuntimeError), cfg:
        raise RuntimeError
    assert cfg.f is not None
    assert cfg.f.closed


def test_structlog_config_f_is_none_rank_zero_false(
    tmp_path: pathlib.Path,
) -> None:
    """StructlogConfig.f stays None when is_rank_zero is False.

    Verifies that a non-rank-zero process never opens the log file, leaving
    ``cfg.f`` as ``None`` after the context exits.
    """
    cfg = StructlogConfig(is_rank_zero=False, log_file=tmp_path / "run.jsonl")
    with cfg:
        pass
    assert cfg.f is None


def test_structlog_config_processors_include_console_renderer_rank_zero(
    tmp_path: pathlib.Path,
) -> None:
    """ConsoleRenderer in structlog processor chain when is_rank_zero is True.

    Verifies ``structlog.dev.ConsoleRenderer`` appears among the configured
    processors while inside the context manager on the rank-zero process.
    """
    path = tmp_path / "run.jsonl"
    with StructlogConfig(is_rank_zero=True, log_file=path):
        processor_types = [
            type(p)
            for p in cast(
                list[object],
                structlog.get_config()["processors"],
            )
        ]
    assert structlog.dev.ConsoleRenderer in processor_types


def test_structlog_config_processors_no_console_renderer_not_rank_zero(
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Non-rank-zero suppresses INFO but emits WARNING and above.

    Verifies the minimum log level is 30 (WARNING): an INFO event must produce
    no output while a WARNING event must appear in the captured output.
    """
    with StructlogConfig(is_rank_zero=False, log_file=tmp_path / "run.jsonl"):
        log = structlog.get_logger()
        log.info("below_threshold")
        log.warning("above_threshold")
    captured = capsys.readouterr()
    assert "below_threshold" not in captured.out + captured.err
    assert "above_threshold" in captured.out + captured.err


def test_structlog_config_write_log_line_in_processors_with_file(
    tmp_path: pathlib.Path,
) -> None:
    """_write_log_line in processor chain when is_rank_zero=True.

    Verifies that ``cfg.write_log_line`` is registered as a structlog
    processor when both rank-zero status and a valid log path are provided.
    """
    path = tmp_path / "run.jsonl"
    cfg = StructlogConfig(is_rank_zero=True, log_file=path)
    with cfg:
        processors = cast(list[object], structlog.get_config()["processors"])
    assert cfg.write_log_line in processors


def test_structlog_config_write_log_line_not_in_processors_not_rank_zero(
    tmp_path: pathlib.Path,
) -> None:
    """_write_log_line is absent from processors when is_rank_zero is False.

    Verifies that ``cfg.write_log_line`` is not added to the structlog processor
    chain on non-rank-zero processes.
    """
    cfg = StructlogConfig(is_rank_zero=False, log_file=tmp_path / "run.jsonl")
    with cfg:
        processors = cast(list[object], structlog.get_config()["processors"])
    assert cfg.write_log_line not in processors


def test_structlog_config_file_is_line_buffered(tmp_path: pathlib.Path) -> None:
    """StructlogConfig opens the log file with line buffering (buffering=1).

    Verifies that ``cfg.f.line_buffering`` is ``True``, ensuring each written
    JSON line is flushed to disk immediately without an explicit ``flush()``
    call.
    """
    path = tmp_path / "run.jsonl"
    cfg = StructlogConfig(is_rank_zero=True, log_file=path)
    with cfg:
        assert cfg.f is not None
        assert cfg.f.line_buffering
