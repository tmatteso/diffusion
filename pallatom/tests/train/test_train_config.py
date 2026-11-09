"""Tests for the validators in train/train_config.py."""

import pathlib

import pytest
from pydantic import ValidationError
from train.train_config import TrainingParams


@pytest.fixture
def checkpoint_file(tmp_path: pathlib.Path) -> pathlib.Path:
    """Provide a checkpoint path; the file itself is never read.

    Args:
        tmp_path: Pytest-provided temporary directory.

    Returns:
        A path under ``tmp_path`` to use as a checkpoint source.
    """
    return tmp_path / "checkpoint.pt"


def test_training_params_rejects_two_checkpoint_sources(
    checkpoint_file: pathlib.Path,
) -> None:
    """Setting both pretrained_weights and resume_checkpoint is rejected.

    Not parametrized: setting both fields is the only invalid combination.
    The three valid combinations (either one alone, or neither) are covered
    by ``test_restore_training_state_reads_configured_source`` in
    ``test_train_loop.py``, which builds a config for each. Pydantic wraps
    the validator's ``ConflictingCheckpointSourcesError`` in a
    ``ValidationError``, so the test matches on that error's message.
    """
    with pytest.raises(ValidationError, match="set only one of"):
        _ = TrainingParams(
            pretrained_weights=checkpoint_file,
            resume_checkpoint=checkpoint_file,
        )
