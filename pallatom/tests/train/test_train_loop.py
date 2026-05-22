"""Tests for the training loop."""

import json
import math
import os
import pathlib
from collections.abc import Mapping
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
import structlog
import torch
import torch.distributed as dist_module
import torch.nn as nn
import torch.nn.parallel
from architecture.main_trunk import MainTrunk
from einops import rearrange
from helpers.bucketed_sampler import BucketedBatchSampler
from helpers.data import _to_protein_batch, make_bucketed_data_loaders
from helpers.featurize import Distogram, ProteinBatch, apply_conditioning_dropout, featurize_batch
from jaxtyping import Float, TypeCheckError
from pydantic import ValidationError
from structlog.typing import FilteringBoundLogger
from train.train_config import (
    CheckpointParams,
    LoggingParams,
    ModelParams,
    TrainConfig,
    TrainingParams,
)
from train.train_loop import (
    _METRIC_KEYS,
    _process_accum_window,
    evaluate,
    evaluate_ddp,
    train,
    train_ddp,
    train_step,
)

torch.manual_seed(42)

_N_KEEP = 16
_C_RES = 32
_C_ATOM = 32
_C_PAIR = 32
_C_ATOMPAIR = 16
_F_REF_DIM = 35
_N_BINS = 8
_N_ATOM_BINS = 5
_K_UNIT = 1
_BATCH_TOKENS: int = _N_KEEP  # atom_mask=ones → every residue is a valid token

EXPECTED_EVAL_KEYS = frozenset(
    {
        "total loss",
        "Kabsch aligned MSE loss",
        "Cross Entropy loss",
        "Smooth LDDT loss",
        "Residue Distogram loss",
        "Atom Distogram loss",
        "Intermediate loss",
        "RMSD",
    }
)

EXPECTED_STEP_KEYS = frozenset(
    {
        "total loss",
        "Kabsch aligned MSE loss",
        "Cross Entropy loss",
        "smooth lddt",
        "Residue Distogram loss",
        "Atom Distogram loss",
        "Intermediate loss",
        "pack_rate",
        "residues_per_sec",
        "atoms_per_sec",
    }
)

EXPECTED_CHECKPOINT_KEYS = frozenset(
    {"model", "optimizer", "scheduler", "epoch", "global_step", "best_val_loss"}
)


class _ListDataset(torch.utils.data.Dataset[ProteinBatch]):
    def __init__(self, items: list[Mapping[str, Float[torch.Tensor, "..."] | list[str]]]) -> None:
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> ProteinBatch:
        item = self._items[idx]
        return ProteinBatch(
            atom_positions=cast("torch.Tensor", item["atom_positions"]),
            atom_mask=cast("torch.Tensor", item["atom_mask"]),
            residue_index=cast("torch.Tensor", item["residue_index"]),
            seq=cast("list[str]", item["seq"]),
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def distogram_res() -> Distogram:
    """Provide a residue-level Distogram in eval mode."""
    return Distogram(n_bins=_N_BINS, min_dist=3.25, max_dist=50.75, overflow_bin=False).eval()


@pytest.fixture
def distogram_atom() -> Distogram:
    """Provide an atom-level Distogram in eval mode."""
    return Distogram(n_bins=_N_ATOM_BINS, min_dist=0.0, max_dist=10.0, overflow_bin=False).eval()


@pytest.fixture
def model() -> MainTrunk:
    """Provide a small MainTrunk instance for training loop tests."""
    return MainTrunk(
        f_ref_dim=_F_REF_DIM,
        n_bins=_N_BINS,
        n_atom_bins=_N_ATOM_BINS,
        c_atom=_C_ATOM,
        c_pair=_C_PAIR,
        c_res=_C_RES,
        c_atompair=_C_ATOMPAIR,
        K_unit=_K_UNIT,
    )


@pytest.fixture
def tcfg(tmp_path: pathlib.Path) -> TrainConfig:
    """Provide a single-epoch TrainConfig with a temporary checkpoint path."""
    return TrainConfig(
        training=TrainingParams(
            num_epochs=1, lr=1e-4, grad_clip=1.0, accumulated_token_budget=_BATCH_TOKENS
        ),
        model=ModelParams(
            f_ref_dim=_F_REF_DIM,
            n_bins=_N_BINS,
            c_atom=_C_ATOM,
            c_pair=_C_PAIR,
            c_res=_C_RES,
            c_atompair=_C_ATOMPAIR,
            K_unit=_K_UNIT,
        ),
        checkpoint=CheckpointParams(
            checkpoint_path=str(tmp_path / "best.pt"),
            save_every=100,
        ),
        logging=LoggingParams(use_wandb=False, log_interval=1),
    )


@pytest.fixture
def single_sample() -> Mapping[str, Float[torch.Tensor, "..."] | str]:
    """Provide single unbatched protein sample (no batch dimension) for _to_protein_batch tests."""
    return {
        "atom_positions": torch.randn(_N_KEEP, 37, 3),
        "atom_mask": torch.ones(_N_KEEP, 37),
        "residue_index": torch.arange(_N_KEEP, dtype=torch.float32),
        "seq": ("ACDEFGHIKLMNPQRSTVWY" * (_N_KEEP // 20 + 1))[:_N_KEEP],
    }


@pytest.fixture
def mini_batch() -> Mapping[str, Float[torch.Tensor, "..."] | list[str]]:
    """Provide a single-item mini-batch of random tensors with _N_KEEP residues."""
    return {
        "atom_positions": torch.randn(1, _N_KEEP, 37, 3),
        "atom_mask": torch.ones(1, _N_KEEP, 37),
        "residue_index": rearrange(torch.arange(_N_KEEP, dtype=torch.float32), "n -> 1 n"),
        "seq": [("ACDEFGHIKLMNPQRSTVWY" * (_N_KEEP // 20 + 1))[:_N_KEEP]],
    }


@pytest.fixture
def loader(
    mini_batch: Mapping[str, Float[torch.Tensor, "..."] | list[str]]
) -> torch.utils.data.DataLoader[ProteinBatch]:
    """Provide a DataLoader that yields the mini_batch as a single batch."""
    return torch.utils.data.DataLoader(_ListDataset([mini_batch]), batch_size=None)


@pytest.fixture
def tcfg_multi(tmp_path: pathlib.Path) -> TrainConfig:
    """Provide a 3-epoch TrainConfig for multi-epoch loop tests."""
    return TrainConfig(
        training=TrainingParams(
            num_epochs=3, lr=1e-3, grad_clip=1.0, accumulated_token_budget=_BATCH_TOKENS
        ),
        model=ModelParams(
            f_ref_dim=_F_REF_DIM,
            n_bins=_N_BINS,
            c_atom=_C_ATOM,
            c_pair=_C_PAIR,
            c_res=_C_RES,
            c_atompair=_C_ATOMPAIR,
            K_unit=_K_UNIT,
        ),
        checkpoint=CheckpointParams(
            checkpoint_path=str(tmp_path / "best_multi.pt"),
            save_every=100,
        ),
        logging=LoggingParams(use_wandb=False, log_interval=1),
    )


@pytest.fixture
def tcfg_no_clip(tmp_path: pathlib.Path) -> TrainConfig:
    """Provide a TrainConfig with grad_clip=None to exercise the unclipped gradient path."""
    return TrainConfig(
        training=TrainingParams(
            num_epochs=1, lr=1e-4, grad_clip=None, accumulated_token_budget=_BATCH_TOKENS
        ),
        model=ModelParams(
            f_ref_dim=_F_REF_DIM,
            n_bins=_N_BINS,
            c_atom=_C_ATOM,
            c_pair=_C_PAIR,
            c_res=_C_RES,
            c_atompair=_C_ATOMPAIR,
            K_unit=_K_UNIT,
        ),
        checkpoint=CheckpointParams(
            checkpoint_path=str(tmp_path / "best_no_clip.pt"),
            save_every=100,
        ),
        logging=LoggingParams(use_wandb=False, log_interval=1),
    )


@pytest.fixture
def tcfg_wandb(tmp_path: pathlib.Path) -> TrainConfig:
    """Provide a 1-epoch TrainConfig with W&B logging enabled for wandb-branch tests."""
    return TrainConfig(
        training=TrainingParams(
            num_epochs=1, lr=1e-4, grad_clip=1.0, accumulated_token_budget=_BATCH_TOKENS
        ),
        model=ModelParams(
            f_ref_dim=_F_REF_DIM,
            n_bins=_N_BINS,
            c_atom=_C_ATOM,
            c_pair=_C_PAIR,
            c_res=_C_RES,
            c_atompair=_C_ATOMPAIR,
            K_unit=_K_UNIT,
        ),
        checkpoint=CheckpointParams(
            checkpoint_path=str(tmp_path / "best_wandb.pt"),
            save_every=100,
        ),
        logging=LoggingParams(use_wandb=True, log_interval=1),
    )


@pytest.fixture
def tcfg_wandb_3ep(tmp_path: pathlib.Path) -> TrainConfig:
    """Provide a 3-epoch TrainConfig with W&B logging enabled for multi-epoch wandb tests."""
    return TrainConfig(
        training=TrainingParams(
            num_epochs=3, lr=1e-4, grad_clip=1.0, accumulated_token_budget=_BATCH_TOKENS
        ),
        model=ModelParams(
            f_ref_dim=_F_REF_DIM,
            n_bins=_N_BINS,
            c_atom=_C_ATOM,
            c_pair=_C_PAIR,
            c_res=_C_RES,
            c_atompair=_C_ATOMPAIR,
            K_unit=_K_UNIT,
        ),
        checkpoint=CheckpointParams(
            checkpoint_path=str(tmp_path / "best_wandb_3ep.pt"),
            save_every=100,
        ),
        logging=LoggingParams(use_wandb=True, log_interval=1),
    )


@pytest.fixture
def tcfg_save(tmp_path: pathlib.Path) -> TrainConfig:
    """Provide a TrainConfig with save_every=1 to exercise the periodic epoch checkpoint path."""
    return TrainConfig(
        training=TrainingParams(
            num_epochs=1, lr=1e-4, grad_clip=1.0, accumulated_token_budget=_BATCH_TOKENS
        ),
        model=ModelParams(
            f_ref_dim=_F_REF_DIM,
            n_bins=_N_BINS,
            c_atom=_C_ATOM,
            c_pair=_C_PAIR,
            c_res=_C_RES,
            c_atompair=_C_ATOMPAIR,
            K_unit=_K_UNIT,
        ),
        checkpoint=CheckpointParams(
            checkpoint_path=str(tmp_path / "best_save_every.pt"),
            save_every=1,
        ),
        logging=LoggingParams(use_wandb=False, log_interval=1),
    )


@pytest.fixture
def protein_batch() -> ProteinBatch:
    """Provide a single-item ProteinBatch with _N_KEEP residues for train_step tests."""
    return ProteinBatch(
        atom_positions=torch.randn(1, _N_KEEP, 37, 3),
        atom_mask=torch.ones(1, _N_KEEP, 37),
        residue_index=rearrange(torch.arange(_N_KEEP, dtype=torch.float32), "n -> 1 n"),
        seq=[("ACDEFGHIKLMNPQRSTVWY" * (_N_KEEP // 20 + 1))[:_N_KEEP]],
    )


def test_train_step_returns_expected_keys(
    protein_batch: ProteinBatch,
    model: MainTrunk,
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
) -> None:
    """train_step return dict contains exactly EXPECTED_STEP_KEYS."""
    model.zero_grad()
    metrics = train_step(protein_batch, model, tcfg, distogram_res, distogram_atom, "cpu")
    assert set(metrics.keys()) == EXPECTED_STEP_KEYS


def test_train_step_pack_rate_in_range(
    protein_batch: ProteinBatch,
    model: MainTrunk,
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
) -> None:
    """pack_rate is in (0, 1]."""
    model.zero_grad()
    metrics = train_step(protein_batch, model, tcfg, distogram_res, distogram_atom, "cpu")
    assert 0.0 < metrics["pack_rate"] <= 1.0


def test_train_step_throughput_metrics_positive(
    protein_batch: ProteinBatch,
    model: MainTrunk,
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
) -> None:
    """residues_per_sec and atoms_per_sec are positive floats."""
    model.zero_grad()
    metrics = train_step(protein_batch, model, tcfg, distogram_res, distogram_atom, "cpu")
    assert metrics["residues_per_sec"] > 0.0
    assert metrics["atoms_per_sec"] > 0.0


# ---------------------------------------------------------------------------
# _to_protein_batch
# ---------------------------------------------------------------------------


def test_to_protein_batch_returns_protein_batch(
    single_sample: Mapping[str, Float[torch.Tensor, "..."] | str]
) -> None:
    """Ensures _to_protein_batch returns a ProteinBatch dataclass instance."""
    assert isinstance(_to_protein_batch([single_sample]), ProteinBatch)


def test_to_protein_batch_atom_positions_shape(
    single_sample: Mapping[str, Float[torch.Tensor, "..."] | str]
) -> None:
    """Ensures _to_protein_batch stacks atom_positions into batched tensor w/ leading batch dim."""
    result = _to_protein_batch([single_sample])
    assert result.atom_positions.shape == torch.Size(
        [1, *cast("torch.Tensor", single_sample["atom_positions"]).shape]
    )


def test_to_protein_batch_atom_mask_shape(
    single_sample: Mapping[str, Float[torch.Tensor, "..."] | str],
) -> None:
    """Ensures _to_protein_batch stacks atom_mask into a batched tensor with a leading batch dim."""
    result = _to_protein_batch([single_sample])
    assert result.atom_mask.shape == torch.Size(
        [1, *cast("torch.Tensor", single_sample["atom_mask"]).shape]
    )


def test_to_protein_batch_residue_index_shape(
    single_sample: Mapping[str, Float[torch.Tensor, "..."] | str]
) -> None:
    """Ensures _to_protein_batch stacks residue_index into batched tensor with leading batch dim."""
    result = _to_protein_batch([single_sample])
    assert result.residue_index.shape == torch.Size(
        [1, *cast("torch.Tensor", single_sample["residue_index"]).shape]
    )


def test_to_protein_batch_seq_is_list(
    single_sample: Mapping[str, Float[torch.Tensor, "..."] | str],
) -> None:
    """Ensures _to_protein_batch wraps the sequence field into a Python list."""
    result = _to_protein_batch([single_sample])
    assert isinstance(result.seq, list)


def test_to_protein_batch_seq_elements_are_strings(
    single_sample: Mapping[str, Float[torch.Tensor, "..."] | str]
) -> None:
    """Ensures _to_protein_batch contains only str elements in the seq list."""
    result = _to_protein_batch([single_sample])
    assert all(isinstance(s, str) for s in result.seq)


def test_to_protein_batch_seq_length_matches_batch_size(
    single_sample: Mapping[str, Float[torch.Tensor, "..."] | str]
) -> None:
    """Ensures _to_protein_batch seq list has one entry per sample passed in."""
    samples = [single_sample]
    result = _to_protein_batch(samples)
    assert len(result.seq) == len(samples)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def test_evaluate_returns_dict(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
) -> None:
    """Ensures evaluate returns a plain dict."""
    result = evaluate(model, loader, tcfg, distogram_res, distogram_atom, "cpu")
    assert isinstance(result, dict)


def test_evaluate_returns_expected_keys(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
) -> None:
    """Ensures evaluate result contains exactly the expected metric keys."""
    result = evaluate(model, loader, tcfg, distogram_res, distogram_atom, "cpu")
    assert set(result.keys()) == EXPECTED_EVAL_KEYS


def test_evaluate_all_values_are_floats(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
) -> None:
    """Every value in the evaluate result dict is a Python float."""
    result = evaluate(model, loader, tcfg, distogram_res, distogram_atom, "cpu")
    for k, v in result.items():
        assert isinstance(v, float), f"'{k}' is {type(v)}, expected float"


def test_evaluate_all_losses_finite(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
) -> None:
    """Every metric value in the evaluate result is finite (no NaN or Inf)."""
    result = evaluate(model, loader, tcfg, distogram_res, distogram_atom, "cpu")
    for k, v in result.items():
        # math.isfinite returns False if the value is NaN or Infinity
        assert math.isfinite(v), f"Value for '{k}' is not finite: {v}"


def test_evaluate_total_loss_non_negative(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
) -> None:
    """Ensures evaluate total loss is non-negative for any valid model and input."""
    result = evaluate(model, loader, tcfg, distogram_res, distogram_atom, "cpu")
    assert result["total loss"] >= 0.0


def test_evaluate_rmsd_non_negative(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
) -> None:
    """Ensures evaluate RMSD is non-negative (it is the square root of a mean squared distance)."""
    result = evaluate(model, loader, tcfg, distogram_res, distogram_atom, "cpu")
    assert result["RMSD"] >= 0.0


def test_evaluate_sets_model_to_eval_mode(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
) -> None:
    """Ensures evaluate puts the model in eval mode and leaves it there after returning."""
    model.train()
    evaluate(model, loader, tcfg, distogram_res, distogram_atom, "cpu")
    assert not model.training


def test_evaluate_empty_loader_returns_zero_dict(
    model: MainTrunk,
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
) -> None:
    """Ensures evaluate returns all-zero metrics when the loader contains no batches."""
    empty = torch.utils.data.DataLoader(_ListDataset([]), batch_size=None)
    result = evaluate(model, empty, tcfg, distogram_res, distogram_atom, "cpu")
    assert all(v == 0.0 for v in result.values())


def test_evaluate_uses_no_grad(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
) -> None:
    """Ensures evaluate accumulates no gradients into model parameters."""
    evaluate(model, loader, tcfg, distogram_res, distogram_atom, "cpu")
    for p in model.parameters():
        assert p.grad is None


@pytest.fixture
def log() -> FilteringBoundLogger:
    """Provide a structlog logger for training loop tests."""
    return structlog.get_logger()


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def test_train_returns_none(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    log: FilteringBoundLogger,
) -> None:
    """Ensures train returns None (the function is side-effect-only)."""
    assert train(model, tcfg, loader, loader, distogram_res, distogram_atom, "cpu", log) is None


def test_train_saves_best_checkpoint(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    log: FilteringBoundLogger,
) -> None:
    """Ensures train writes a checkpoint file at the path configured in tcfg.checkpoint."""
    train(model, tcfg, loader, loader, distogram_res, distogram_atom, "cpu", log)
    assert os.path.exists(tcfg.checkpoint.checkpoint_path)


def test_train_checkpoint_is_valid_state_dict(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    log: FilteringBoundLogger,
) -> None:
    """The checkpoint written by train contains model, optimizer, scheduler, and training state."""
    train(model, tcfg, loader, loader, distogram_res, distogram_atom, "cpu", log)
    ckpt = torch.load(tcfg.checkpoint.checkpoint_path, weights_only=True)
    assert isinstance(ckpt, dict)
    assert ckpt.keys() >= EXPECTED_CHECKPOINT_KEYS


def test_train_updates_model_parameters(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    log: FilteringBoundLogger,
) -> None:
    """Ensures train modifies at least one model parameter via gradient descent."""
    params_before = [p.clone().detach() for p in model.parameters()]
    train(model, tcfg, loader, loader, distogram_res, distogram_atom, "cpu", log)
    params_after = list(model.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(params_before, params_after, strict=False))


def test_train_model_in_train_mode_after(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    log: FilteringBoundLogger,
) -> None:
    """Ensures train leaves the model in train mode after the last epoch completes."""
    model.eval()
    train(model, tcfg, loader, loader, distogram_res, distogram_atom, "cpu", log)
    assert model.training


def test_train_runs_all_epochs(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg_multi: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    monkeypatch: pytest.MonkeyPatch,
    log: FilteringBoundLogger,
) -> None:
    """Ensures train calls evaluate once per epoch and produces finite losses for all epochs."""
    losses: list[float] = []
    _real_evaluate = evaluate

    def _patched_evaluate(
        m: MainTrunk,
        ldr: torch.utils.data.DataLoader[ProteinBatch],
        cfg: TrainConfig,
        dr: Distogram,
        da: Distogram,
        dev: str,
    ) -> Mapping[str, float]:
        result = _real_evaluate(m, ldr, cfg, dr, da, dev)
        losses.append(result["total loss"])
        return result

    monkeypatch.setattr("train.train_loop.evaluate", _patched_evaluate)
    train(model, tcfg_multi, loader, loader, distogram_res, distogram_atom, "cpu", log)

    assert len(losses) == 3
    # math.isfinite returns False if the value is NaN or Infinity
    assert all(math.isfinite(v) for v in losses)


def test_train_no_grad_clip_runs(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg_no_clip: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    log: FilteringBoundLogger,
) -> None:
    """Ensures train completes without error when grad_clip is None (gradient clipping disabled)."""
    result = train(model, tcfg_no_clip, loader, loader, distogram_res, distogram_atom, "cpu", log)
    assert result is None


def test_train_no_grad_clip_saves_checkpoint(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg_no_clip: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    log: FilteringBoundLogger,
) -> None:
    """Ensures train writes a checkpoint even when grad_clip is None."""
    train(model, tcfg_no_clip, loader, loader, distogram_res, distogram_atom, "cpu", log)
    assert os.path.exists(tcfg_no_clip.checkpoint.checkpoint_path)


def test_train_save_every_creates_epoch_checkpoint(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg_save: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    log: FilteringBoundLogger,
) -> None:
    """Ensures train writes a per-epoch checkpoint file when save_every=1."""
    monkeypatch.chdir(tmp_path)
    train(model, tcfg_save, loader, loader, distogram_res, distogram_atom, "cpu", log)
    assert os.path.exists(tmp_path / "checkpoint_epoch_001.pt")


def test_train_save_every_checkpoint_is_valid_state_dict(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg_save: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    log: FilteringBoundLogger,
) -> None:
    """Per-epoch checkpoint contains model, optimizer, scheduler, and training state."""
    monkeypatch.chdir(tmp_path)
    train(model, tcfg_save, loader, loader, distogram_res, distogram_atom, "cpu", log)
    ckpt = torch.load(tmp_path / "checkpoint_epoch_001.pt", weights_only=True)
    assert isinstance(ckpt, dict)
    assert ckpt.keys() >= EXPECTED_CHECKPOINT_KEYS


# ---------------------------------------------------------------------------
# evaluate - multi-batch averaging
# ---------------------------------------------------------------------------


def test_evaluate_multi_batch_returns_expected_keys(
    model: MainTrunk,
    mini_batch: Mapping[str, Float[torch.Tensor, "..."] | list[str]],
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
) -> None:
    """Ensures evaluate with a 2-batch loader still returns exactly the expected metric keys."""
    loader2 = torch.utils.data.DataLoader(_ListDataset([mini_batch, mini_batch]), batch_size=None)
    result = evaluate(model, loader2, tcfg, distogram_res, distogram_atom, "cpu")
    assert set(result.keys()) == EXPECTED_EVAL_KEYS


def test_evaluate_multi_batch_n_batches_counted(
    model: MainTrunk,
    mini_batch: Mapping[str, Float[torch.Tensor, "..."] | list[str]],
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
) -> None:
    """Ensures evaluate averages correctly over 3 batches, producing finite loss values."""
    loader3 = torch.utils.data.DataLoader(_ListDataset([mini_batch] * 3), batch_size=None)
    result = evaluate(model, loader3, tcfg, distogram_res, distogram_atom, "cpu")
    for k, v in result.items():
        # math.isfinite returns False if the value is NaN or Infinity
        assert math.isfinite(v), f"Value for '{k}' is not finite: {v}"


# ---------------------------------------------------------------------------
# train - wandb logging branch
# ---------------------------------------------------------------------------


def test_train_calls_wandb_log_when_enabled(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg_wandb: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    monkeypatch: pytest.MonkeyPatch,
    log: FilteringBoundLogger,
) -> None:
    """Train calls wandb.log exactly once per epoch when use_wandb=True."""
    logged: list[dict[str, float]] = []
    monkeypatch.setattr("train.train_loop.wandb.log", lambda data, **_: logged.append(data))
    train(model, tcfg_wandb, loader, loader, distogram_res, distogram_atom, "cpu", log)
    assert len(logged) == 1


def test_train_wandb_log_not_called_when_disabled(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    monkeypatch: pytest.MonkeyPatch,
    log: FilteringBoundLogger,
) -> None:
    """Train does not call wandb.log when use_wandb=False."""
    mock_log = MagicMock()
    monkeypatch.setattr("train.train_loop.wandb.log", mock_log)
    train(model, tcfg, loader, loader, distogram_res, distogram_atom, "cpu", log)
    mock_log.assert_not_called()


def test_train_wandb_log_payload_has_epoch_key(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg_wandb: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    monkeypatch: pytest.MonkeyPatch,
    log: FilteringBoundLogger,
) -> None:
    """The W&B payload logged by train includes an 'epoch' key set to the current epoch number."""
    payloads: list[Mapping[str, object]] = []
    monkeypatch.setattr("train.train_loop.wandb.log", lambda data, **_: payloads.append(data))
    train(model, tcfg_wandb, loader, loader, distogram_res, distogram_atom, "cpu", log)
    assert "epoch" in payloads[0]
    assert payloads[0]["epoch"] == 1


def test_train_wandb_log_payload_has_train_keys(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg_wandb: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    monkeypatch: pytest.MonkeyPatch,
    log: FilteringBoundLogger,
) -> None:
    """The W&B payload logged by train contains at least one key prefixed with 'train/'."""
    payloads: list[Mapping[str, object]] = []
    monkeypatch.setattr("train.train_loop.wandb.log", lambda data, **_: payloads.append(data))
    train(model, tcfg_wandb, loader, loader, distogram_res, distogram_atom, "cpu", log)
    assert any(k.startswith("train/") for k in payloads[0])


def test_train_wandb_log_payload_has_val_keys(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg_wandb: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    monkeypatch: pytest.MonkeyPatch,
    log: FilteringBoundLogger,
) -> None:
    """The W&B payload logged by train contains at least one key prefixed with 'val/'."""
    payloads: list[Mapping[str, object]] = []
    monkeypatch.setattr("train.train_loop.wandb.log", lambda data, **_: payloads.append(data))
    train(model, tcfg_wandb, loader, loader, distogram_res, distogram_atom, "cpu", log)
    assert any(k.startswith("val/") for k in payloads[0])


def test_train_wandb_log_step_is_positive(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg_wandb: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    monkeypatch: pytest.MonkeyPatch,
    log: FilteringBoundLogger,
) -> None:
    """The global_step passed to wandb.log is at least 1 after the first training step."""
    steps: list[int] = []

    def _log(_data: object, step: int | None = None) -> None:
        assert step is not None
        steps.append(step)

    monkeypatch.setattr("train.train_loop.wandb.log", _log)
    train(model, tcfg_wandb, loader, loader, distogram_res, distogram_atom, "cpu", log)
    assert steps[0] >= 1


def test_train_wandb_log_called_once_per_epoch(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg_wandb_3ep: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    monkeypatch: pytest.MonkeyPatch,
    log: FilteringBoundLogger,
) -> None:
    """Train calls wandb.log exactly once per epoch for a 3-epoch run."""
    logged: list[dict[str, float]] = []
    monkeypatch.setattr("train.train_loop.wandb.log", lambda data, **_: logged.append(data))
    train(model, tcfg_wandb_3ep, loader, loader, distogram_res, distogram_atom, "cpu", log)
    assert len(logged) == 3


def test_train_wandb_log_epoch_increments(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg_wandb_3ep: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    monkeypatch: pytest.MonkeyPatch,
    log: FilteringBoundLogger,
) -> None:
    """The epoch number in W&B payloads increments from 1 to 3 across a 3-epoch run."""
    payloads: list[Mapping[str, object]] = []
    monkeypatch.setattr("train.train_loop.wandb.log", lambda data, **_: payloads.append(data))
    train(model, tcfg_wandb_3ep, loader, loader, distogram_res, distogram_atom, "cpu", log)
    epochs = [p["epoch"] for p in payloads]
    assert epochs == [1, 2, 3]


def test_train_wandb_log_train_total_loss_is_float(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg_wandb: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    monkeypatch: pytest.MonkeyPatch,
    log: FilteringBoundLogger,
) -> None:
    """The 'train/total loss' value logged to W&B is a Python float."""
    payloads: list[Mapping[str, object]] = []
    monkeypatch.setattr("train.train_loop.wandb.log", lambda data, **_: payloads.append(data))
    train(model, tcfg_wandb, loader, loader, distogram_res, distogram_atom, "cpu", log)
    assert isinstance(payloads[0]["train/total loss"], float)


def test_train_wandb_log_val_total_loss_is_float(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg_wandb: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    monkeypatch: pytest.MonkeyPatch,
    log: FilteringBoundLogger,
) -> None:
    """The 'val/total loss' value logged to W&B is a Python float."""
    payloads: list[Mapping[str, object]] = []
    monkeypatch.setattr("train.train_loop.wandb.log", lambda data, **_: payloads.append(data))
    train(model, tcfg_wandb, loader, loader, distogram_res, distogram_atom, "cpu", log)
    assert isinstance(payloads[0]["val/total loss"], float)


def test_train_wandb_log_steps_increase_across_epochs(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg_wandb_3ep: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    monkeypatch: pytest.MonkeyPatch,
    log: FilteringBoundLogger,
) -> None:
    """Global steps passed to wandb.log are strictly increasing and unique across epochs."""
    steps: list[int] = []

    def _capture_step(_data: object, step: int | None = None) -> None:
        assert step is not None
        steps.append(step)

    monkeypatch.setattr("train.train_loop.wandb.log", _capture_step)
    train(model, tcfg_wandb_3ep, loader, loader, distogram_res, distogram_atom, "cpu", log)
    assert steps == sorted(steps)
    assert len(set(steps)) == 3


# ---------------------------------------------------------------------------
# evaluate_ddp
# ---------------------------------------------------------------------------


def test_evaluate_ddp_returns_dict(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
) -> None:
    """Ensures evaluate_ddp returns a plain dict for world_size=1."""
    result = evaluate_ddp(1, model, loader, tcfg, distogram_res, distogram_atom, "cpu")
    assert isinstance(result, dict)


def test_evaluate_ddp_returns_expected_keys(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
) -> None:
    """Ensures evaluate_ddp result contains exactly the expected metric keys."""
    result = evaluate_ddp(1, model, loader, tcfg, distogram_res, distogram_atom, "cpu")
    assert set(result.keys()) == EXPECTED_EVAL_KEYS


def test_evaluate_ddp_all_values_are_floats(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
) -> None:
    """Every value in the evaluate_ddp result dict is a Python float."""
    result = evaluate_ddp(1, model, loader, tcfg, distogram_res, distogram_atom, "cpu")
    for k, v in result.items():
        assert isinstance(v, float), f"'{k}' is {type(v)}, expected float"


def test_evaluate_ddp_all_losses_finite(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
) -> None:
    """Every metric value in evaluate_ddp is finite (no NaN or Inf)."""
    result = evaluate_ddp(1, model, loader, tcfg, distogram_res, distogram_atom, "cpu")
    for k, v in result.items():
        # math.isfinite returns False if the value is NaN or Infinity
        assert math.isfinite(v), f"Value for '{k}' is not finite: {v}"


def test_evaluate_ddp_total_loss_non_negative(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
) -> None:
    """Ensures evaluate_ddp total loss is non-negative."""
    result = evaluate_ddp(1, model, loader, tcfg, distogram_res, distogram_atom, "cpu")
    assert result["total loss"] >= 0.0


def test_evaluate_ddp_rmsd_non_negative(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
) -> None:
    """Ensures evaluate_ddp RMSD is non-negative."""
    result = evaluate_ddp(1, model, loader, tcfg, distogram_res, distogram_atom, "cpu")
    assert result["RMSD"] >= 0.0


def test_evaluate_ddp_sets_model_to_eval_mode(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
) -> None:
    """Ensures evaluate_ddp puts the model in eval mode and leaves it there after returning."""
    model.train()
    evaluate_ddp(1, model, loader, tcfg, distogram_res, distogram_atom, "cpu")
    assert not model.training


def test_evaluate_ddp_empty_loader_returns_zero_dict(
    model: MainTrunk,
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
) -> None:
    """Ensures evaluate_ddp returns all-zero metrics when the loader contains no batches."""
    empty = torch.utils.data.DataLoader(_ListDataset([]), batch_size=None)
    result = evaluate_ddp(1, model, empty, tcfg, distogram_res, distogram_atom, "cpu")
    assert all(v == 0.0 for v in result.values())


def test_evaluate_ddp_calls_all_reduce_once_when_world_size_gt_1(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensures evaluate_ddp calls dist.all_reduce exactly once to aggregate metrics across ranks."""
    calls = []
    monkeypatch.setattr(dist_module, "all_reduce", lambda *_args, **_kwargs: calls.append(1))
    evaluate_ddp(2, model, loader, tcfg, distogram_res, distogram_atom, "cpu")
    assert len(calls) == 1


def test_evaluate_ddp_skips_all_reduce_when_world_size_is_1(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensures evaluate_ddp does not call dist.all_reduce when world_size=1."""
    calls = []
    monkeypatch.setattr(dist_module, "all_reduce", lambda *_args, **_kwargs: calls.append(1))
    evaluate_ddp(1, model, loader, tcfg, distogram_res, distogram_atom, "cpu")
    assert len(calls) == 0


class _FakeDDP(nn.Module):
    """Minimal DDP stand-in that exposes .module and works on CPU."""

    def __init__(self, module: nn.Module, **_kwargs: Any) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)


@pytest.fixture(autouse=True)
def _patch_ddp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace DDP with _FakeDDP so train_ddp tests run without a real process group."""
    monkeypatch.setattr("train.train_loop.DDP", _FakeDDP)


class _MockSampler:
    """Batch sampler with set_epoch() for verifying epoch reseeding in train_ddp tests.

    Implements the batch-sampler protocol: __iter__ yields lists of indices so it can
    be passed as batch_sampler to DataLoader.
    """

    def __init__(self, data: list[Mapping[str, Float[torch.Tensor, "..."] | list[str]]]) -> None:
        self._data = data
        self.set_epoch_calls: list[int] = []

    def __iter__(self):
        return iter([[i] for i in range(len(self._data))])

    def __len__(self) -> int:
        return len(self._data)

    def set_epoch(self, epoch: int) -> None:
        """Record epoch for later assertion."""
        self.set_epoch_calls.append(epoch)


def _identity_collate(
    items: list[ProteinBatch],
) -> ProteinBatch:
    """Return the single ProteinBatch item from a singleton batch list.

    Args:
        items: A one-element list produced by the DataLoader when batch_sampler
               yields a single-index list.

    Returns:
        The unwrapped ProteinBatch.
    """
    return items[0]


@pytest.fixture
def ddp_loader(
    mini_batch: Mapping[str, Float[torch.Tensor, "..."] | list[str]]
) -> torch.utils.data.DataLoader[ProteinBatch]:
    """Provide a DataLoader with _MockSampler as batch_sampler to track set_epoch calls."""
    sampler = _MockSampler([mini_batch])
    return torch.utils.data.DataLoader(
        _ListDataset([mini_batch]), batch_sampler=sampler, collate_fn=_identity_collate
    )


# ---------------------------------------------------------------------------
# train_ddp
# ---------------------------------------------------------------------------


def test_train_ddp_returns_none(
    model: MainTrunk,
    ddp_loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    log: FilteringBoundLogger,
) -> None:
    """Ensures train_ddp returns None (side-effect-only function)."""
    result = train_ddp(
        0,
        0,
        1,
        model,
        tcfg,
        ddp_loader,
        ddp_loader,
        distogram_res,
        distogram_atom,
        log,
        device="cpu",
    )
    assert result is None


def test_train_ddp_rank0_saves_checkpoint(
    model: MainTrunk,
    ddp_loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    log: FilteringBoundLogger,
) -> None:
    """Ensures train_ddp writes a checkpoint when rank=0."""
    train_ddp(
        0,
        0,
        1,
        model,
        tcfg,
        ddp_loader,
        ddp_loader,
        distogram_res,
        distogram_atom,
        log,
        device="cpu",
    )
    assert os.path.exists(tcfg.checkpoint.checkpoint_path)


def test_train_ddp_checkpoint_has_correct_keys(
    model: MainTrunk,
    ddp_loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    log: FilteringBoundLogger,
) -> None:
    """train_ddp checkpoint contains model, optimizer, scheduler, and training state."""
    train_ddp(
        0,
        0,
        1,
        model,
        tcfg,
        ddp_loader,
        ddp_loader,
        distogram_res,
        distogram_atom,
        log,
        device="cpu",
    )
    ckpt = torch.load(tcfg.checkpoint.checkpoint_path, weights_only=True)
    assert ckpt.keys() >= EXPECTED_CHECKPOINT_KEYS


def test_train_ddp_rank1_does_not_save_checkpoint(
    model: MainTrunk,
    ddp_loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    log: FilteringBoundLogger,
) -> None:
    """Ensures train_ddp does not write a checkpoint when rank != 0."""
    train_ddp(
        1,
        0,
        2,
        model,
        tcfg,
        ddp_loader,
        ddp_loader,
        distogram_res,
        distogram_atom,
        log,
        device="cpu",
    )
    assert not os.path.exists(tcfg.checkpoint.checkpoint_path)


def test_train_ddp_calls_set_epoch_each_epoch(
    model: MainTrunk,
    ddp_loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg_multi: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    log: FilteringBoundLogger,
) -> None:
    """Ensures train_ddp calls sampler.set_epoch(epoch) once per epoch in ascending order."""
    train_ddp(
        0,
        0,
        1,
        model,
        tcfg_multi,
        ddp_loader,
        ddp_loader,
        distogram_res,
        distogram_atom,
        log,
        device="cpu",
    )
    assert cast("_MockSampler", ddp_loader.batch_sampler).set_epoch_calls == [1, 2, 3]


def test_train_ddp_updates_model_parameters(
    model: MainTrunk,
    ddp_loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    log: FilteringBoundLogger,
) -> None:
    """Ensures train_ddp modifies at least one model parameter via gradient descent."""
    params_before = [p.clone().detach() for p in model.parameters()]
    train_ddp(
        0,
        0,
        1,
        model,
        tcfg,
        ddp_loader,
        ddp_loader,
        distogram_res,
        distogram_atom,
        log,
        device="cpu",
    )
    changed = (
        not torch.equal(b, a) for b, a in zip(params_before, model.parameters(), strict=False)
    )
    assert any(changed)


def test_train_ddp_wandb_not_called_when_disabled(
    model: MainTrunk,
    ddp_loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    monkeypatch: pytest.MonkeyPatch,
    log: FilteringBoundLogger,
) -> None:
    """Ensures train_ddp does not call wandb.log when use_wandb=False."""
    mock_log = MagicMock()
    monkeypatch.setattr("train.train_loop.wandb.log", mock_log)
    train_ddp(
        0,
        0,
        1,
        model,
        tcfg,
        ddp_loader,
        ddp_loader,
        distogram_res,
        distogram_atom,
        log,
        device="cpu",
    )
    mock_log.assert_not_called()


def test_train_ddp_wandb_called_when_rank0_and_enabled(
    model: MainTrunk,
    ddp_loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg_wandb: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    monkeypatch: pytest.MonkeyPatch,
    log: FilteringBoundLogger,
) -> None:
    """Ensures train_ddp calls wandb.log when rank=0 and use_wandb=True."""
    logged = []
    monkeypatch.setattr("train.train_loop.wandb.log", lambda data, **_: logged.append(data))
    train_ddp(
        0,
        0,
        1,
        model,
        tcfg_wandb,
        ddp_loader,
        ddp_loader,
        distogram_res,
        distogram_atom,
        log,
        device="cpu",
    )
    assert len(logged) == 1


def test_train_ddp_wandb_not_called_when_rank_nonzero(
    model: MainTrunk,
    ddp_loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg_wandb: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    monkeypatch: pytest.MonkeyPatch,
    log: FilteringBoundLogger,
) -> None:
    """Ensures train_ddp does not call wandb.log when rank != 0 even if use_wandb=True."""
    mock_log = MagicMock()
    monkeypatch.setattr("train.train_loop.wandb.log", mock_log)
    train_ddp(
        1,
        0,
        2,
        model,
        tcfg_wandb,
        ddp_loader,
        ddp_loader,
        distogram_res,
        distogram_atom,
        log,
        device="cpu",
    )
    mock_log.assert_not_called()


def test_evaluate_ddp_world_size1_matches_evaluate(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
) -> None:
    """Ensures evaluate_ddp with world_size=1 produces identical results to evaluate."""
    torch.manual_seed(42)
    r1 = evaluate(model, loader, tcfg, distogram_res, distogram_atom, "cpu")
    torch.manual_seed(42)
    r2 = evaluate_ddp(1, model, loader, tcfg, distogram_res, distogram_atom, "cpu")
    for k in r1:
        assert (
            abs(r1[k] - r2[k]) < 1e-5
        ), f"Mismatch for '{k}': evaluate={r1[k]}, evaluate_ddp={r2[k]}"


# ---------------------------------------------------------------------------
# conditioning dropout integration
# ---------------------------------------------------------------------------


def test_training_step_with_full_dropout_produces_finite_outputs(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
) -> None:
    """A forward pass with 50% conditioning dropout produces finite denoised coords and logits."""
    batch = next(iter(loader))
    featurized = featurize_batch(batch, tcfg, distogram_res, distogram_atom, device="cpu")
    featurized = apply_conditioning_dropout(
        featurized,
        p_distogram=0.5,
        p_atom=0.5,
        p_seq=0.5,
        device="cpu",
    )
    with torch.no_grad():
        r_denoised, f_seq_logits, *_ = model(featurized)
    assert torch.isfinite(r_denoised).all()
    assert torch.isfinite(f_seq_logits).all()


def test_evaluate_loop_unchanged_by_dropout_config(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
) -> None:
    """Ensures evaluate returns valid float metrics regardless of dropout config."""
    result = evaluate(model, loader, tcfg, distogram_res, distogram_atom, device="cpu")
    assert all(isinstance(v, float) for v in result.values())


# ---------------------------------------------------------------------------
# Integration: gradient flow via train_step
# ---------------------------------------------------------------------------


def _assert_submodule_grads(model: MainTrunk) -> None:
    """Assert every top-level submodule has at least one finite nonzero gradient.

    Args:
        model: Trunk module after a backward pass has been called.
    """
    buckets: dict[str, list[Float[torch.Tensor, "..."]]] = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            buckets.setdefault(name.split(".")[0], []).append(param.grad)
    assert buckets, "no parameters have gradients — backward was not called"
    for prefix, grads in buckets.items():
        assert any(
            torch.isfinite(g).all().item() and g.abs().max().item() > 0 for g in grads
        ), f"submodule '{prefix}' has no finite nonzero gradients"


def test_integration_gradient_flow_via_train_step(
    model: MainTrunk,
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    loader: torch.utils.data.DataLoader[ProteinBatch],
) -> None:
    """train_step() back-propagates finite nonzero grads to every MainTrunk submodule."""
    model.train()
    batch = next(iter(loader))
    model.zero_grad()
    train_step(batch, model, tcfg, distogram_res, distogram_atom, device="cpu")
    _assert_submodule_grads(model)


# ---------------------------------------------------------------------------
# Shape-contract enforcement — negative tests
# ---------------------------------------------------------------------------


def test_evaluate_wrong_device_type(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
) -> None:
    """Non-str device triggers TypeCheckError for evaluate."""
    with pytest.raises(TypeCheckError):
        evaluate(model, loader, tcfg, distogram_res, distogram_atom, 42)  # type: ignore[arg-type]


def test_train_wrong_device_type(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    log: FilteringBoundLogger,
) -> None:
    """Non-str device triggers TypeCheckError for train."""
    with pytest.raises(TypeCheckError):
        train(model, tcfg, loader, loader, distogram_res, distogram_atom, 42, log)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Checkpoint resume
# ---------------------------------------------------------------------------


def test_train_resume_runs_remaining_epochs(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    distogram_res: Distogram,
    distogram_atom: Distogram,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    log: FilteringBoundLogger,
) -> None:
    """Resuming from a 1-epoch checkpoint with num_epochs=3 runs exactly 2 more epochs."""
    ckpt_path = str(tmp_path / "best.pt")
    tcfg_first = TrainConfig(
        training=TrainingParams(
            num_epochs=1, lr=1e-4, grad_clip=1.0, accumulated_token_budget=_BATCH_TOKENS
        ),
        model=ModelParams(
            f_ref_dim=_F_REF_DIM,
            n_bins=_N_BINS,
            c_atom=_C_ATOM,
            c_pair=_C_PAIR,
            c_res=_C_RES,
            c_atompair=_C_ATOMPAIR,
            K_unit=_K_UNIT,
        ),
        checkpoint=CheckpointParams(checkpoint_path=ckpt_path, save_every=100),
        logging=LoggingParams(use_wandb=False, log_interval=1),
    )
    train(model, tcfg_first, loader, loader, distogram_res, distogram_atom, "cpu", log)

    eval_epochs: list[int] = []
    _real_evaluate = evaluate

    def _counting_evaluate(
        m: MainTrunk,
        ldr: torch.utils.data.DataLoader[ProteinBatch],
        cfg: TrainConfig,
        dr: Distogram,
        da: Distogram,
        dev: str,
    ) -> Mapping[str, float]:
        result = _real_evaluate(m, ldr, cfg, dr, da, dev)
        eval_epochs.append(len(eval_epochs) + 1)
        return result

    monkeypatch.setattr("train.train_loop.evaluate", _counting_evaluate)
    tcfg_resume = TrainConfig(
        training=TrainingParams(
            num_epochs=3,
            lr=1e-4,
            grad_clip=1.0,
            resume_checkpoint=ckpt_path,
            accumulated_token_budget=_BATCH_TOKENS,
        ),
        model=ModelParams(
            f_ref_dim=_F_REF_DIM,
            n_bins=_N_BINS,
            c_atom=_C_ATOM,
            c_pair=_C_PAIR,
            c_res=_C_RES,
            c_atompair=_C_ATOMPAIR,
            K_unit=_K_UNIT,
        ),
        checkpoint=CheckpointParams(
            checkpoint_path=str(tmp_path / "best_resume.pt"), save_every=100
        ),
        logging=LoggingParams(use_wandb=False, log_interval=1),
    )
    train(model, tcfg_resume, loader, loader, distogram_res, distogram_atom, "cpu", log)
    assert len(eval_epochs) == 2


def test_train_resume_restores_optimizer_state(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    distogram_res: Distogram,
    distogram_atom: Distogram,
    tmp_path: pathlib.Path,
    log: FilteringBoundLogger,
) -> None:
    """Optimizer state (exp_avg) loaded from checkpoint is non-zero after one resumed step."""
    ckpt_path = str(tmp_path / "opt.pt")
    tcfg_first = TrainConfig(
        training=TrainingParams(
            num_epochs=1, lr=1e-4, grad_clip=1.0, accumulated_token_budget=_BATCH_TOKENS
        ),
        model=ModelParams(
            f_ref_dim=_F_REF_DIM,
            n_bins=_N_BINS,
            c_atom=_C_ATOM,
            c_pair=_C_PAIR,
            c_res=_C_RES,
            c_atompair=_C_ATOMPAIR,
            K_unit=_K_UNIT,
        ),
        checkpoint=CheckpointParams(checkpoint_path=ckpt_path, save_every=100),
        logging=LoggingParams(use_wandb=False, log_interval=1),
    )
    train(model, tcfg_first, loader, loader, distogram_res, distogram_atom, "cpu", log)

    saved_ckpt = torch.load(ckpt_path, weights_only=True)
    opt_state = saved_ckpt["optimizer"]["state"]
    assert len(opt_state) > 0
    first_param_state = next(iter(opt_state.values()))
    assert "exp_avg" in first_param_state
    assert first_param_state["exp_avg"].abs().max().item() > 0


def test_train_resume_restores_scheduler_state(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    distogram_res: Distogram,
    distogram_atom: Distogram,
    tmp_path: pathlib.Path,
    log: FilteringBoundLogger,
) -> None:
    """Scheduler last_epoch in saved checkpoint matches the epoch at which it was written."""
    ckpt_path = str(tmp_path / "sched.pt")
    tcfg_first = TrainConfig(
        training=TrainingParams(
            num_epochs=1, lr=1e-4, grad_clip=1.0, accumulated_token_budget=_BATCH_TOKENS
        ),
        model=ModelParams(
            f_ref_dim=_F_REF_DIM,
            n_bins=_N_BINS,
            c_atom=_C_ATOM,
            c_pair=_C_PAIR,
            c_res=_C_RES,
            c_atompair=_C_ATOMPAIR,
            K_unit=_K_UNIT,
        ),
        checkpoint=CheckpointParams(checkpoint_path=ckpt_path, save_every=100),
        logging=LoggingParams(use_wandb=False, log_interval=1),
    )
    train(model, tcfg_first, loader, loader, distogram_res, distogram_atom, "cpu", log)

    saved_ckpt = torch.load(ckpt_path, weights_only=True)
    assert saved_ckpt["scheduler"]["last_epoch"] == 1


def test_train_resume_checkpoint_epoch_and_step(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    distogram_res: Distogram,
    distogram_atom: Distogram,
    tmp_path: pathlib.Path,
    log: FilteringBoundLogger,
) -> None:
    """Saved checkpoint records the correct epoch number and global_step."""
    ckpt_path = str(tmp_path / "meta.pt")
    tcfg_first = TrainConfig(
        training=TrainingParams(
            num_epochs=1, lr=1e-4, grad_clip=1.0, accumulated_token_budget=_BATCH_TOKENS
        ),
        model=ModelParams(
            f_ref_dim=_F_REF_DIM,
            n_bins=_N_BINS,
            c_atom=_C_ATOM,
            c_pair=_C_PAIR,
            c_res=_C_RES,
            c_atompair=_C_ATOMPAIR,
            K_unit=_K_UNIT,
        ),
        checkpoint=CheckpointParams(checkpoint_path=ckpt_path, save_every=100),
        logging=LoggingParams(use_wandb=False, log_interval=1),
    )
    train(model, tcfg_first, loader, loader, distogram_res, distogram_atom, "cpu", log)

    saved_ckpt = torch.load(ckpt_path, weights_only=True)
    assert saved_ckpt["epoch"] == 1
    assert saved_ckpt["global_step"] == 1  # one batch in the loader
    assert isinstance(saved_ckpt["best_val_loss"], float)


# ---------------------------------------------------------------------------
# BucketedBatchSampler integration
# ---------------------------------------------------------------------------

_N_RES_BUCKET = 6
_ENTRY_NAMES_BUCKET = ["1aa.A", "2bb.A", "3cc.A", "4dd.A", "5ee.A"]
_TRAIN_NAMES_BUCKET = ["1aa.A", "2bb.A", "3cc.A"]
_VAL_NAMES_BUCKET = ["4dd.A"]
_TEST_NAMES_BUCKET = ["5ee.A"]


@pytest.fixture
def jsonl_path(tmp_path: pathlib.Path) -> str:
    """Write a temporary JSONL file with synthetic protein entries and return its path."""
    path = tmp_path / "proteins.jsonl"
    with open(path, "w") as f:
        for name in _ENTRY_NAMES_BUCKET:
            entry = {
                "name": name,
                "seq": "ACDEFG"[:_N_RES_BUCKET],
                "coords": {
                    atom: np.random.randn(_N_RES_BUCKET, 3).tolist()
                    for atom in ("N", "CA", "C", "O")
                },
            }
            f.write(json.dumps(entry) + "\n")
    return str(path)


@pytest.fixture
def splits_path(tmp_path: pathlib.Path) -> str:
    """Write a temporary splits JSON with train/val/test name lists and return its path."""
    path = tmp_path / "splits.json"
    with open(path, "w") as f:
        json.dump(
            {
                "train": _TRAIN_NAMES_BUCKET,
                "validation": _VAL_NAMES_BUCKET,
                "test": _TEST_NAMES_BUCKET,
            },
            f,
        )
    return str(path)


def test_train_one_epoch_with_bucketed_loader(
    model: MainTrunk,
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    jsonl_path: str,
    splits_path: str,
    log: FilteringBoundLogger,
) -> None:
    """train() runs one epoch with a bucketed DataLoader without error."""
    train_loader, val_loader, _ = make_bucketed_data_loaders(
        cfg=tcfg,
        jsonl_path=jsonl_path,
        splits_path=splits_path,
        num_workers=0,
        debug_run=False,
    )
    sampler = train_loader.batch_sampler
    assert isinstance(sampler, BucketedBatchSampler)
    sampler.set_epoch(0)
    result = train(model, tcfg, train_loader, val_loader, distogram_res, distogram_atom, "cpu", log)
    assert result is None


@pytest.fixture
def tcfg_accum(tmp_path: pathlib.Path) -> TrainConfig:
    """TrainConfig with budget=2*_BATCH_TOKENS, requiring 2 micro-batches per optimizer step."""
    return TrainConfig(
        training=TrainingParams(
            num_epochs=1, lr=1e-4, grad_clip=1.0, accumulated_token_budget=2 * _BATCH_TOKENS
        ),
        model=ModelParams(
            f_ref_dim=_F_REF_DIM,
            n_bins=_N_BINS,
            c_atom=_C_ATOM,
            c_pair=_C_PAIR,
            c_res=_C_RES,
            c_atompair=_C_ATOMPAIR,
            K_unit=_K_UNIT,
        ),
        checkpoint=CheckpointParams(
            checkpoint_path=str(tmp_path / "best_accum.pt"),
            save_every=100,
        ),
        logging=LoggingParams(use_wandb=False, log_interval=1),
    )


def test_train_partial_window_at_epoch_end_updates_params(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg_accum: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    log: FilteringBoundLogger,
) -> None:
    """Partial window flushed at epoch end even when budget is not reached (budget=2x batch size).

    With 1 batch of _BATCH_TOKENS tokens and budget=2*_BATCH_TOKENS, no pre-flush fires during
    the epoch, but the remaining buffer is always flushed at epoch end so params must change.
    """
    params_before = [p.clone().detach() for p in model.parameters()]
    train(model, tcfg_accum, loader, loader, distogram_res, distogram_atom, "cpu", log)
    assert any(
        not torch.equal(b, a) for b, a in zip(params_before, model.parameters(), strict=False)
    )


def test_train_ddp_partial_window_at_epoch_end_updates_params(
    model: MainTrunk,
    ddp_loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg_accum: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    log: FilteringBoundLogger,
) -> None:
    """Partial window flushed at epoch end in train_ddp even when budget is not reached.

    With 1 batch of _BATCH_TOKENS tokens and budget=2*_BATCH_TOKENS, the buffer is flushed
    at epoch end, so params must change.
    """
    params_before = [p.clone().detach() for p in model.parameters()]
    train_ddp(
        0,
        0,
        1,
        model,
        tcfg_accum,
        ddp_loader,
        ddp_loader,
        distogram_res,
        distogram_atom,
        log,
        device="cpu",
    )
    assert any(
        not torch.equal(b, a) for b, a in zip(params_before, model.parameters(), strict=False)
    )


@pytest.fixture
def loader_2batch(
    mini_batch: Mapping[str, Float[torch.Tensor, "..."] | list[str]]
) -> torch.utils.data.DataLoader[ProteinBatch]:
    """DataLoader with two identical mini-batches for accumulation tests."""
    return torch.utils.data.DataLoader(_ListDataset([mini_batch, mini_batch]), batch_size=None)


def test_train_step_grad_scale_halves_gradient_norm(
    protein_batch: ProteinBatch,
    model: MainTrunk,
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
) -> None:
    """Loss divided by grad_scale=2 yields approximately half the gradient norm of grad_scale=1."""
    torch.manual_seed(0)
    model.zero_grad()
    train_step(protein_batch, model, tcfg, distogram_res, distogram_atom, "cpu", grad_scale=1.0)
    norm1: float = (
        sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    )

    torch.manual_seed(0)
    model.zero_grad()
    train_step(protein_batch, model, tcfg, distogram_res, distogram_atom, "cpu", grad_scale=2.0)
    norm2: float = (
        sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    )

    assert abs(norm2 - norm1 / 2.0) < 1e-3


def test_train_accumulation_full_window_updates_params(
    model: MainTrunk,
    loader_2batch: torch.utils.data.DataLoader[ProteinBatch],
    tcfg_accum: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    log: FilteringBoundLogger,
) -> None:
    """With accum_steps=2 and a 2-batch loader, train() completes one window and updates params."""
    params_before = [p.clone().detach() for p in model.parameters()]
    train(
        model, tcfg_accum, loader_2batch, loader_2batch, distogram_res, distogram_atom, "cpu", log
    )
    assert any(
        not torch.equal(b, a) for b, a in zip(params_before, model.parameters(), strict=False)
    )


def test_train_ddp_accumulation_full_window_updates_params(
    model: MainTrunk,
    tcfg_accum: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    mini_batch: Mapping[str, Float[torch.Tensor, "..."] | list[str]],
    log: FilteringBoundLogger,
) -> None:
    """With accum_steps=2 and a 2-batch DDP loader, train_ddp() updates params after one window."""
    sampler = _MockSampler([mini_batch, mini_batch])
    ddp_loader_2batch: torch.utils.data.DataLoader[ProteinBatch] = torch.utils.data.DataLoader(
        _ListDataset([mini_batch, mini_batch]),
        batch_sampler=sampler,
        collate_fn=_identity_collate,
    )
    params_before = [p.clone().detach() for p in model.parameters()]
    train_ddp(
        rank=0,
        local_rank=0,
        world_size=1,
        model=model,
        tcfg=tcfg_accum,
        train_loader=ddp_loader_2batch,
        test_loader=ddp_loader_2batch,
        distogram_res=distogram_res,
        distogram_atom=distogram_atom,
        log=log,
        device="cpu",
    )
    assert any(
        not torch.equal(b, a) for b, a in zip(params_before, model.parameters(), strict=False)
    )


# ---------------------------------------------------------------------------
# Regression: train metrics must be non-zero when budget exceeds total tokens
# ---------------------------------------------------------------------------


def test_train_metrics_nonzero_when_budget_exceeds_total_tokens(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    distogram_res: Distogram,
    distogram_atom: Distogram,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    log: FilteringBoundLogger,
) -> None:
    """Train metrics logged to W&B are non-zero when the entire dataset fits below the budget.

    Regression test: with a large token budget, no pre-flush fires during the epoch, so the
    only optimizer step is the end-of-epoch flush. If that flush is silently dropped,
    train/total loss is logged as 0.0 — the model is never updated.
    """
    large_budget = 10 * _BATCH_TOKENS  # far larger than the 1-batch loader's token count
    tcfg_large = TrainConfig(
        training=TrainingParams(
            num_epochs=1, lr=1e-4, grad_clip=1.0, accumulated_token_budget=large_budget
        ),
        model=ModelParams(
            f_ref_dim=_F_REF_DIM,
            n_bins=_N_BINS,
            c_atom=_C_ATOM,
            c_pair=_C_PAIR,
            c_res=_C_RES,
            c_atompair=_C_ATOMPAIR,
            K_unit=_K_UNIT,
        ),
        checkpoint=CheckpointParams(
            checkpoint_path=str(tmp_path / "best_large.pt"), save_every=100
        ),
        logging=LoggingParams(use_wandb=True, log_interval=1),
    )
    payloads: list[dict[str, object]] = []
    monkeypatch.setattr("train.train_loop.wandb.log", lambda data, **_: payloads.append(data))
    train(model, tcfg_large, loader, loader, distogram_res, distogram_atom, "cpu", log)

    assert len(payloads) == 1
    train_total_loss = payloads[0]["train/total loss"]
    assert isinstance(train_total_loss, float)
    assert (
        train_total_loss > 0.0
    ), "train/total loss is 0.0 — end-of-epoch flush was dropped so no training occurred"


def test_train_ddp_metrics_nonzero_when_budget_exceeds_total_tokens(
    model: MainTrunk,
    ddp_loader: torch.utils.data.DataLoader[ProteinBatch],
    distogram_res: Distogram,
    distogram_atom: Distogram,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    log: FilteringBoundLogger,
) -> None:
    """train_ddp train metrics are non-zero when the entire dataset fits below the budget.

    Same regression as test_train_metrics_nonzero_when_budget_exceeds_total_tokens but for
    the DDP path.
    """
    large_budget = 10 * _BATCH_TOKENS
    tcfg_large = TrainConfig(
        training=TrainingParams(
            num_epochs=1, lr=1e-4, grad_clip=1.0, accumulated_token_budget=large_budget
        ),
        model=ModelParams(
            f_ref_dim=_F_REF_DIM,
            n_bins=_N_BINS,
            c_atom=_C_ATOM,
            c_pair=_C_PAIR,
            c_res=_C_RES,
            c_atompair=_C_ATOMPAIR,
            K_unit=_K_UNIT,
        ),
        checkpoint=CheckpointParams(
            checkpoint_path=str(tmp_path / "best_large_ddp.pt"), save_every=100
        ),
        logging=LoggingParams(use_wandb=True, log_interval=1),
    )
    payloads: list[dict[str, object]] = []
    monkeypatch.setattr("train.train_loop.wandb.log", lambda data, **_: payloads.append(data))
    train_ddp(
        0,
        0,
        1,
        model,
        tcfg_large,
        ddp_loader,
        ddp_loader,
        distogram_res,
        distogram_atom,
        log,
        device="cpu",
    )

    assert len(payloads) == 1
    train_total_loss = payloads[0]["train/total loss"]
    assert isinstance(train_total_loss, float)
    assert (
        train_total_loss > 0.0
    ), "train/total loss is 0.0 — end-of-epoch flush was dropped so no training occurred"


# ---------------------------------------------------------------------------
# accumulated_token_budget config
# ---------------------------------------------------------------------------


def test_accumulated_token_budget_rejects_zero() -> None:
    """TrainingParams raises ValidationError when accumulated_token_budget is zero."""
    with pytest.raises(ValidationError):
        TrainingParams(accumulated_token_budget=0)


def test_train_config_accepts_any_positive_token_budget() -> None:
    """TrainConfig no longer validates token budget against batch_size."""
    cfg = TrainConfig(training=TrainingParams(accumulated_token_budget=1))
    assert cfg.training.accumulated_token_budget == 1


# ---------------------------------------------------------------------------
# _process_accum_window — protein-weighted grad_scale
# ---------------------------------------------------------------------------


def test_process_accum_window_protein_weighted_grad_scale(
    protein_batch: ProteinBatch,
    model: MainTrunk,
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_process_accum_window passes protein-count-weighted grad_scale to each train_step call.

    Two micro-batches with n_proteins_per_batch=[1, 3] → total_proteins=4.
    Expected grad_scale: 4/1=4.0 for the first, 4/3≈1.333 for the second.
    """
    captured_scales: list[float] = []

    def _mock_train_step(
        _batch: ProteinBatch,
        _mdl: nn.Module,
        _cfg: TrainConfig,
        _dr: Distogram,
        _da: Distogram,
        _device: str,
        grad_scale: float = 1.0,
    ) -> dict[str, float]:
        captured_scales.append(grad_scale)
        return dict.fromkeys(_METRIC_KEYS, 0.0)

    monkeypatch.setattr("train.train_loop.train_step", _mock_train_step)

    _process_accum_window(
        micro_buffer=[protein_batch, protein_batch],
        n_proteins_per_batch=[1, 3],
        model=model,
        tcfg=tcfg,
        distogram_res=distogram_res,
        distogram_atom=distogram_atom,
        device="cpu",
    )

    assert len(captured_scales) == 2
    assert abs(captured_scales[0] - 4.0) < 1e-6  # total=4, n=1 → 4/1
    assert abs(captured_scales[1] - 4.0 / 3.0) < 1e-6  # total=4, n=3 → 4/3


# ---------------------------------------------------------------------------
# token-based accumulation — pre-flush behavior
# ---------------------------------------------------------------------------


def test_train_token_budget_preflush_fires_before_oversized_batch(
    model: MainTrunk,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    mini_batch: Mapping[str, Float[torch.Tensor, "..."] | list[str]],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    log: FilteringBoundLogger,
) -> None:
    """Pre-flush fires for batch1 when batch2 would push tokens over the budget.

    mini_batch has _BATCH_TOKENS=16 tokens. With budget=24 and a 2-batch loader:
    - batch1 (16 tokens): added, held (16 < 24)
    - batch2 (16 tokens): 16+16=32 > 24 → pre-flush batch1 alone, then batch2 added (16<24)
    - epoch end: batch2 still in buffer → flushed as a second window.
    Result: _process_accum_window called twice, each with window size 1.
    """
    window_sizes: list[int] = []
    _real_process = _process_accum_window

    def _tracking_process(
        micro_buffer: list[ProteinBatch],
        n_proteins_per_batch: list[int],
        mdl: nn.Module,
        cfg: TrainConfig,
        dr: Distogram,
        da: Distogram,
        dev: str,
    ) -> dict[str, float]:
        window_sizes.append(len(micro_buffer))
        return _real_process(micro_buffer, n_proteins_per_batch, mdl, cfg, dr, da, dev)

    monkeypatch.setattr("train.train_loop._process_accum_window", _tracking_process)

    budget: int = _BATCH_TOKENS + _BATCH_TOKENS // 2  # 24: one batch (16) fits; two (32) don't
    tcfg_budget = TrainConfig(
        training=TrainingParams(
            num_epochs=1, lr=1e-4, grad_clip=1.0, accumulated_token_budget=budget
        ),
        model=ModelParams(
            f_ref_dim=_F_REF_DIM,
            n_bins=_N_BINS,
            c_atom=_C_ATOM,
            c_pair=_C_PAIR,
            c_res=_C_RES,
            c_atompair=_C_ATOMPAIR,
            K_unit=_K_UNIT,
        ),
        checkpoint=CheckpointParams(
            checkpoint_path=str(tmp_path / "best_flush.pt"), save_every=100
        ),
        logging=LoggingParams(use_wandb=False, log_interval=1),
    )
    loader_2b = torch.utils.data.DataLoader(_ListDataset([mini_batch, mini_batch]), batch_size=None)
    train(model, tcfg_budget, loader_2b, loader_2b, distogram_res, distogram_atom, "cpu", log)

    assert window_sizes == [1, 1]  # pre-flush with batch1; batch2 flushed at epoch end
