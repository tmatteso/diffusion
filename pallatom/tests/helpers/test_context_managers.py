"""Tests for context manager helpers."""

import json
import os
import pathlib

import pytest
import structlog
from helpers.context_managers import StructlogConfig


def test_structlog_config_creates_log_file(tmp_path: pathlib.Path) -> None:
    """StructlogConfig creates the log file on disk when entering the context."""
    path = str(tmp_path / "run.jsonl")
    with StructlogConfig(is_rank_zero=True, log_file=path):
        pass
    assert os.path.exists(path)


def test_structlog_config_write_log_line_returns_event_dict(tmp_path: pathlib.Path) -> None:
    """_write_log_line passes the event dict through unchanged."""
    path = str(tmp_path / "run.jsonl")
    cfg = StructlogConfig(is_rank_zero=True, log_file=path)
    with cfg:
        event: dict[str, object] = {"event": "test", "value": 1}
        returned = cfg.write_log_line(None, None, event)
    assert returned is event


def test_structlog_config_writes_json_line(tmp_path: pathlib.Path) -> None:
    """_write_log_line serialises the event dict as a JSON line in the log file."""
    path = str(tmp_path / "run.jsonl")
    cfg = StructlogConfig(is_rank_zero=True, log_file=path)
    with cfg:
        event: dict[str, object] = {"event": "train", "loss": 0.5}
        cfg.write_log_line(None, None, event)
    with open(path) as fh:
        written = json.loads(fh.readline())
    assert written == event


def test_structlog_config_writes_multiple_lines(tmp_path: pathlib.Path) -> None:
    """_write_log_line writes one JSON line per call."""
    path = str(tmp_path / "run.jsonl")
    cfg = StructlogConfig(is_rank_zero=True, log_file=path)
    events: list[dict[str, object]] = [{"event": "a", "x": 1}, {"event": "b", "x": 2}]
    with cfg:
        for ev in events:
            cfg.write_log_line(None, None, ev)
    with open(path) as fh:
        lines = fh.readlines()
    assert len(lines) == 2
    assert [json.loads(line) for line in lines] == events


def test_structlog_config_truncates_existing_file(tmp_path: pathlib.Path) -> None:
    """StructlogConfig truncates a pre-existing log file rather than appending."""
    path = str(tmp_path / "run.jsonl")
    with open(path, "w") as fh:
        fh.write('{"stale": true}\n' * 5)
    cfg = StructlogConfig(is_rank_zero=True, log_file=path)
    with cfg:
        cfg.write_log_line(None, None, {"event": "fresh"})
    with open(path) as fh:
        lines = fh.readlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"event": "fresh"}


def test_structlog_config_exit_closes_file(tmp_path: pathlib.Path) -> None:
    """StructlogConfig.__exit__ closes the underlying file handle."""
    path = str(tmp_path / "run.jsonl")
    cfg = StructlogConfig(is_rank_zero=True, log_file=path)
    with cfg:
        cfg.write_log_line(None, None, {"event": "inside"})
    assert cfg.f is not None
    assert cfg.f.closed


def test_structlog_config_no_file_when_not_rank_zero(tmp_path: pathlib.Path) -> None:
    """StructlogConfig does not open a log file when is_rank_zero is False."""
    path = str(tmp_path / "run.jsonl")
    with StructlogConfig(is_rank_zero=False, log_file=path):
        pass
    assert not os.path.exists(path)


def test_structlog_config_f_is_none_before_enter(tmp_path: pathlib.Path) -> None:
    """StructlogConfig.f is None immediately after construction, before entering the context."""
    path = str(tmp_path / "run.jsonl")
    cfg = StructlogConfig(is_rank_zero=True, log_file=path)
    assert cfg.f is None


def test_structlog_config_enter_returns_self(tmp_path: pathlib.Path) -> None:
    """StructlogConfig.__enter__ returns the instance itself, enabling `as` binding."""
    path = str(tmp_path / "run.jsonl")
    cfg = StructlogConfig(is_rank_zero=True, log_file=path)
    with cfg as result:
        assert result is cfg


def test_structlog_config_no_file_opened_when_log_file_none() -> None:
    """StructlogConfig.f stays None when log_file is None, even for rank zero."""
    cfg = StructlogConfig(is_rank_zero=True, log_file=None)
    with cfg:
        assert cfg.f is None


def test_structlog_config_write_log_line_safe_when_f_is_none() -> None:
    """_write_log_line returns the event dict without error when no file is open."""
    cfg = StructlogConfig(is_rank_zero=True, log_file=None)
    event: dict[str, object] = {"event": "no_file", "step": 0}
    result = cfg.write_log_line(None, None, event)
    assert result is event


def test_structlog_config_exception_propagates(tmp_path: pathlib.Path) -> None:
    """StructlogConfig.__exit__ does not suppress exceptions raised inside the block."""
    path = str(tmp_path / "run.jsonl")
    with pytest.raises(ValueError, match="boom"):  # noqa: SIM117
        with StructlogConfig(is_rank_zero=True, log_file=path):
            raise ValueError("boom")


def test_structlog_config_file_closed_on_exception(tmp_path: pathlib.Path) -> None:
    """StructlogConfig closes the log file even when an exception is raised inside the block."""
    path = str(tmp_path / "run.jsonl")
    cfg = StructlogConfig(is_rank_zero=True, log_file=path)
    with pytest.raises(RuntimeError):  # noqa: SIM117
        with cfg:
            raise RuntimeError("abort")
    assert cfg.f is not None
    assert cfg.f.closed


def test_structlog_config_f_is_none_rank_zero_false_no_log_file() -> None:
    """StructlogConfig.f stays None with is_rank_zero=False and no log_file."""
    cfg = StructlogConfig(is_rank_zero=False, log_file=None)
    with cfg:
        pass
    assert cfg.f is None


def test_structlog_config_processors_include_console_renderer_rank_zero(
    tmp_path: pathlib.Path,
) -> None:
    """ConsoleRenderer is in the structlog processor chain when is_rank_zero is True."""
    path = str(tmp_path / "run.jsonl")
    with StructlogConfig(is_rank_zero=True, log_file=path):
        processor_types = [type(p) for p in structlog.get_config()["processors"]]
    assert structlog.dev.ConsoleRenderer in processor_types


def test_structlog_config_processors_no_console_renderer_not_rank_zero() -> None:
    """ConsoleRenderer is absent from the structlog processor chain when is_rank_zero is False."""
    with StructlogConfig(is_rank_zero=False, log_file=None):
        processor_types = [type(p) for p in structlog.get_config()["processors"]]
    assert structlog.dev.ConsoleRenderer not in processor_types


def test_structlog_config_write_log_line_in_processors_with_file(
    tmp_path: pathlib.Path,
) -> None:
    """_write_log_line is in the processor chain when is_rank_zero=True and log_file is set."""
    path = str(tmp_path / "run.jsonl")
    cfg = StructlogConfig(is_rank_zero=True, log_file=path)
    with cfg:
        processors = structlog.get_config()["processors"]
    assert cfg.write_log_line in processors


def test_structlog_config_write_log_line_not_in_processors_without_file() -> None:
    """_write_log_line is absent from processors when log_file is None."""
    cfg = StructlogConfig(is_rank_zero=True, log_file=None)
    with cfg:
        processors = structlog.get_config()["processors"]
    assert cfg.write_log_line not in processors


def test_structlog_config_processor_count_rank_zero_with_file(tmp_path: pathlib.Path) -> None:
    """Five processors for rank_zero=True with a log file.

    Processors: timestamp, log_level, stack_info, write_log_line, console.
    """
    path = str(tmp_path / "run.jsonl")
    with StructlogConfig(is_rank_zero=True, log_file=path):
        count = len(structlog.get_config()["processors"])
    assert count == 5


def test_structlog_config_processor_count_rank_zero_no_file() -> None:
    """Four processors for rank_zero=True without a log file.

    Processors: timestamp, log_level, stack_info, console.
    """
    with StructlogConfig(is_rank_zero=True, log_file=None):
        count = len(structlog.get_config()["processors"])
    assert count == 4


def test_structlog_config_processor_count_not_rank_zero() -> None:
    """Three processors configured for rank_zero=False: timestamp, log_level, stack_info only."""
    with StructlogConfig(is_rank_zero=False, log_file=None):
        count = len(structlog.get_config()["processors"])
    assert count == 3


def test_structlog_config_file_is_line_buffered(tmp_path: pathlib.Path) -> None:
    """StructlogConfig opens the log file with line buffering (buffering=1)."""
    path = str(tmp_path / "run.jsonl")
    cfg = StructlogConfig(is_rank_zero=True, log_file=path)
    with cfg:
        assert cfg.f is not None
        assert cfg.f.line_buffering
