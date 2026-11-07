"""Tests for the training loop.

Covers to_protein_batch_dynamic, wavg helpers, take_step, process_accum_window,
component_grad_norms, optimizer_step, evaluate, save/load checkpoint,
log_epoch, and the end-to-end train() function including gradient
accumulation and W&B logging.
"""

import contextlib
import copy
import dataclasses
import functools
import hashlib
import json
import multiprocessing
import pathlib
from collections.abc import Callable, Generator
from datetime import timedelta
from enum import Enum
from multiprocessing.queues import SimpleQueue
from typing import cast

import numpy as np
import pytest
import structlog
import torch
import torch.distributed as dist
import torch.nn as nn
from architecture.main_trunk import EMA, MainTrunk
from einops import rearrange, reduce
from helpers.atom_utils import RESTYPE_NUM, RESTYPE_NUM_NO_X, Protein
from helpers.data import (
    Distogram,
    FeaturizedBatch,
    featurize_batch,
    make_bucketed_data_loaders,
)
from helpers.useful_objects import (
    ComponentNorms,
    EpochMetrics,
    LossMetrics,
    ModelSetup,
    TensorAccumulatorMixin,
    ThroughputStatistics,
    manual_seed,
)
from pydantic import ValidationError
from structlog.typing import FilteringBoundLogger
from torch.multiprocessing.spawn import spawn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from train.train_config import (
    AtomDistogramParams,
    CheckpointParams,
    LoaderConfig,
    LoggingParams,
    ModelParams,
    NoiseScheduleParams,
    ResidueDistogramParams,
    TemplateDistogramParams,
    TrainArgs,
    TrainConfig,
    TrainingParams,
    TrainLoaderConfig,
)
from train.train_loop import (
    alphafold_lr_lambda,
    component_grad_norms,
    evaluate,
    load_checkpoint,
    log_epoch,
    optimizer_step,
    process_accum_window,
    save_checkpoint,
    swapped_in_ema_weights,
    take_step,
    train,
)

_ = manual_seed(42)


_N_KEEP = 16
_C_RES = 32
_C_ATOM = 32
_C_PAIR = 32
_C_ATOMPAIR = 16
_F_REF_DIM = 35
_N_BINS = 8
_N_ATOM_BINS = 5
_K_UNIT = 1
_BATCH_TOKENS: int = _N_KEEP

N_BLOCKS_ATOM_TRANSFORMER_ENCODER = 3
N_HEADS_ATOM_TRANSFORMER_ENCODER = 4
N_BLOCKS_ATOM_TRANSFORMER_DECODER = 3
N_HEADS_ATOM_TRANSFORMER_DECODER = 4
N_PAIRFORMER_BLOCKS_TEMPLATE_EMBEDDER = 2
N_PAIFORMER_HEADS_TEMPLATE_EMBEDDER = 16
SIGMA_DATA = 16.0
N_AMINO = RESTYPE_NUM_NO_X
RESIDUE_NUMBER = 50

EPOCH = 7
LEARNING_RATE = 1e-4
TOLERANCE = 1e-5
TIGHT_TOLERANCE = 1e-3
EXPECTED_CHECKPOINT_KEYS = frozenset(
    {"model", "optimizer", "scheduler", "ema", "best_val_loss"},
)
_ADAM_EXP_AVG_KEY = "exp_avg"


def _make_model_params(
    trunk: MainTrunk | DDP,
    train_cfg: TrainConfig,
    dgram_template: Distogram,
    dgram_residue: Distogram,
    dgram_atom: Distogram,
) -> ModelSetup:
    """Construct a ModelSetup with Adam optimizer, AlphaFold-style scheduler.

    Args:
        trunk: The MainTrunk to wrap.
        train_cfg: Training configuration.
        dgram_template: Self-conditioning template distogram module.
        dgram_residue: Residue-level distogram module.
        dgram_atom: Atom-level distogram module.

    Returns:
        Fully initialised ModelSetup.
    """
    optimizer = Adam(trunk.parameters(), lr=train_cfg.training.lr)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=functools.partial(
            alphafold_lr_lambda,
            warmup_steps=train_cfg.training.lr_warmup_steps,
            decay_steps=train_cfg.training.lr_decay_steps,
            decay_factor=train_cfg.training.lr_decay_factor,
        ),
    )
    return ModelSetup(
        model=trunk,
        tcfg=train_cfg,
        distogram_template=dgram_template,
        distogram_residue=dgram_residue,
        distogram_atom=dgram_atom,
        device=torch.device("cpu"),
        optimizer=optimizer,
        scheduler=scheduler,
        ema=EMA(trunk, decay=train_cfg.training.ema_decay),
    )


def _make_model() -> MainTrunk:
    """Build a small MainTrunk for training loop tests.

    Returns:
        MainTrunk built with reduced channel widths and block counts to keep
        tests fast on CPU.
    """
    return MainTrunk(
        model_params=ModelParams(
            f_ref_dim=_F_REF_DIM,
            c_atom=_C_ATOM,
            c_pair=_C_PAIR,
            c_res=_C_RES,
            c_atompair=_C_ATOMPAIR,
            K_unit=_K_UNIT,
            n_blocks_atom_transformer_encoder=N_BLOCKS_ATOM_TRANSFORMER_ENCODER,
            n_heads_atom_transformer_encoder=N_HEADS_ATOM_TRANSFORMER_ENCODER,
            n_blocks_atom_transformer_decoder=N_BLOCKS_ATOM_TRANSFORMER_DECODER,
            n_heads_atom_transformer_decoder=N_HEADS_ATOM_TRANSFORMER_DECODER,
            n_pairformer_blocks_template_embedder=N_PAIRFORMER_BLOCKS_TEMPLATE_EMBEDDER,
            n_paiformer_heads_template_embedder=N_HEADS_ATOM_TRANSFORMER_DECODER,
            n_amino=N_AMINO,
            max_residues=RESIDUE_NUMBER,
        ),
        template_distogram_params=TemplateDistogramParams(
            n_bins=_N_BINS,
            min_dist=3.25,
            max_dist=50.75,
        ),
        residue_distogram_params=ResidueDistogramParams(
            n_bins=_N_BINS,
            min_dist=2.0,
            max_dist=22.0,
        ),
        atom_distogram_params=AtomDistogramParams(
            n_bins=_N_ATOM_BINS,
            min_dist=0.0,
            max_dist=10.0,
        ),
        noise_params=NoiseScheduleParams(
            sigma_data=SIGMA_DATA,
        ),
    )


def _make_distogram_template() -> Distogram:
    """Build the self-conditioning template Distogram in eval mode.

    Returns:
        Distogram configured for template Cβ distances, set to eval mode.
    """
    return Distogram(
        n_bins=_N_BINS,
        min_dist=3.25,
        max_dist=50.75,
        overflow_bin=False,
    ).eval()


def _make_distogram_residue() -> Distogram:
    """Build the residue-level Distogram in eval mode.

    Returns:
        Distogram configured for residue Cβ distances, set to eval mode.
    """
    return Distogram(
        n_bins=_N_BINS,
        min_dist=2.0,
        max_dist=22.0,
        overflow_bin=False,
    ).eval()


def _make_distogram_atom() -> Distogram:
    """Build the atom-level Distogram in eval mode.

    Returns:
        Distogram configured for short-range atom distances, set to eval mode.
    """
    return Distogram(
        n_bins=_N_ATOM_BINS,
        min_dist=0.0,
        max_dist=10.0,
        overflow_bin=False,
    ).eval()


def _make_tcfg(
    checkpoint_path: pathlib.Path,
    *,
    num_epochs: int = 1,
    grad_clip: float | None = 1.0,
    accumulated_token_budget: int = _BATCH_TOKENS,
    use_wandb: bool = False,
) -> TrainConfig:
    """Build a TrainConfig for the small test model.

    Args:
        checkpoint_path: Where the best checkpoint is written.
        num_epochs: Number of training epochs.
        grad_clip: Gradient clipping threshold, or None to disable clipping.
        accumulated_token_budget: Tokens accumulated before an optimizer step.
        use_wandb: Whether to log to W&B.

    Returns:
        TrainConfig with the module's small model channel widths.
    """
    return TrainConfig(
        training=TrainingParams(
            num_epochs=num_epochs,
            lr=LEARNING_RATE,
            grad_clip=grad_clip,
            accumulated_token_budget=accumulated_token_budget,
        ),
        model=ModelParams(
            f_ref_dim=_F_REF_DIM,
            c_atom=_C_ATOM,
            c_pair=_C_PAIR,
            c_res=_C_RES,
            c_atompair=_C_ATOMPAIR,
            K_unit=_K_UNIT,
        ),
        checkpoint=CheckpointParams(
            checkpoint_path=checkpoint_path,
            save_every=100,
        ),
        logging=LoggingParams(use_wandb=use_wandb),
    )


def _make_loaders(
    work_dir: pathlib.Path,
    dataset_jsonl: str,
    splits_json: str,
) -> tuple[
    torch.utils.data.DataLoader[FeaturizedBatch],
    torch.utils.data.DataLoader[FeaturizedBatch],
]:
    """Build train and test loaders over the synthetic proteins.

    Args:
        work_dir: Directory for shards, config, and structured logs.
        dataset_jsonl: Path to the synthetic proteins JSONL file.
        splits_json: Path to the train/val/test splits JSON.

    Returns:
        Train and test loaders streaming the synthetic proteins.
    """
    cfg = TrainConfig(
        training=TrainingParams(
            num_epochs=1,
            lr=LEARNING_RATE,
            grad_clip=1.0,
            accumulated_token_budget=_BATCH_TOKENS,
        ),
        model=ModelParams(
            f_ref_dim=_F_REF_DIM,
            c_atom=_C_ATOM,
            c_pair=_C_PAIR,
            c_res=_C_RES,
            c_atompair=_C_ATOMPAIR,
            K_unit=_K_UNIT,
        ),
        distogram_template=TemplateDistogramParams(
            n_bins=_N_BINS,
            min_dist=3.25,
            max_dist=50.75,
        ),
        distogram_residue=ResidueDistogramParams(
            n_bins=_N_BINS,
            min_dist=2.0,
            max_dist=22.0,
        ),
        distogram_atom=AtomDistogramParams(
            n_bins=_N_ATOM_BINS,
            min_dist=0.0,
            max_dist=10.0,
        ),
        checkpoint=CheckpointParams(
            checkpoint_path=work_dir / "loader_best.pt",
            save_every=100,
        ),
        logging=LoggingParams(use_wandb=False),
        test_loader=LoaderConfig(batch_size=2),
        train_loader=TrainLoaderConfig(
            token_budget=63,
            n_shards=1,
            num_workers=1,
            epoch_prefetch_depth=1,
            batch_prefetch_depth=1,
            n_threads=1,
        ),
    )
    args = TrainArgs(
        dataset_jsonl=pathlib.Path(dataset_jsonl),
        shard_dir=work_dir / "shards",
        keys_for_splits_json=pathlib.Path(splits_json),
        config=work_dir / "config.json",
        structlog_jsonl=work_dir / "structlog.jsonl",
        ddp=False,
        debug_run=False,
    )
    train_loader, test_loader, _ = make_bucketed_data_loaders(
        cfg=cfg,
        extra_train_args=args,
    )
    return train_loader, test_loader


def _make_epoch_metrics(
    epoch: int,
    global_step: int = 0,
    loss_metrics: LossMetrics | None = None,
    throughput: ThroughputStatistics | None = None,
    component_norms: ComponentNorms | None = None,
) -> EpochMetrics:
    """Build a minimal EpochMetrics with zero-valued fields for log_epoch tests.

    Args:
        epoch: Epoch number to embed.
        global_step: Global step count.
        loss_metrics: LossMetrics with all-zero fields; built internally
            if None.
        throughput: ThroughputStatistics with all-zero fields; built
            internally if None.
        component_norms: ComponentNorms with all-zero fields; built
            internally if None.

    Returns:
        EpochMetrics suitable for passing to log_epoch.
    """
    z = torch.tensor(0.0)
    lm = loss_metrics or LossMetrics(
        total_loss=z,
        Kabsch_aligned_MSE_loss=z,
        CE_loss=z,
        smooth_lddt_loss=z,
        res_distogram_loss=z,
        atom_distogram_loss=z,
        intermediate_loss=z,
        RMSD=z,
    )
    tp = throughput or ThroughputStatistics(
        avg_batch_size=z,
        token_pack_rate=z,
        residues_per_sec=z,
        atoms_per_sec=z,
    )
    cn = component_norms or ComponentNorms(
        template_embedder=z,
        atom_encoder=z,
        atom_decoders=z,
        residue_distogram_head=z,
        atom_distogram_head=z,
        inter_proj_seq=z,
        inter_seq_logits=z,
        proj_seq=z,
        seq_logits=z,
    )
    return EpochMetrics(
        epoch=epoch,
        global_step=global_step,
        train_loss_metrics=lm,
        train_throughput_stats=tp,
        train_gradient_norms=cn,
        val_loss_metrics=lm,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def zero_loss_metrics() -> LossMetrics:
    """Build a LossMetrics with all fields set to zero.

    Returns:
        LossMetrics with torch.tensor(0.0) in every field.
    """
    z = torch.tensor(0.0)
    return LossMetrics(
        total_loss=z,
        Kabsch_aligned_MSE_loss=z,
        CE_loss=z,
        smooth_lddt_loss=z,
        res_distogram_loss=z,
        atom_distogram_loss=z,
        intermediate_loss=z,
        RMSD=z,
    )


@pytest.fixture
def zero_throughput() -> ThroughputStatistics:
    """Build a ThroughputStatistics with all fields set to zero.

    Returns:
        ThroughputStatistics with torch.tensor(0.0) in every field.
    """
    z = torch.tensor(0.0)
    return ThroughputStatistics(
        avg_batch_size=z,
        token_pack_rate=z,
        residues_per_sec=z,
        atoms_per_sec=z,
    )


@pytest.fixture
def zero_component_norms() -> ComponentNorms:
    """Build a ComponentNorms with all fields set to zero.

    Returns:
        ComponentNorms with torch.tensor(0.0) in every field.
    """
    z = torch.tensor(0.0)
    return ComponentNorms(
        template_embedder=z,
        atom_encoder=z,
        atom_decoders=z,
        residue_distogram_head=z,
        atom_distogram_head=z,
        inter_proj_seq=z,
        inter_seq_logits=z,
        proj_seq=z,
        seq_logits=z,
    )


@pytest.fixture
def distogram_template() -> Distogram:
    """Provide a self-conditioning template Distogram in eval mode.

    Returns:
        Distogram configured for template Cβ distances, set to eval mode.
    """
    return _make_distogram_template()


@pytest.fixture
def distogram_residue() -> Distogram:
    """Provide a residue-level Distogram in eval mode.

    Returns:
        Distogram configured for residue Cβ distances, set to eval mode.
    """
    return _make_distogram_residue()


@pytest.fixture
def distogram_atom() -> Distogram:
    """Provide an atom-level Distogram in eval mode.

    Returns:
        Distogram configured for short-range atom distances, set to eval mode.
    """
    return _make_distogram_atom()


@pytest.fixture
def model() -> MainTrunk:
    """Provide a small MainTrunk instance for training loop tests.

    Returns:
        MainTrunk built by ``_make_model``.
    """
    return _make_model()


@pytest.fixture
def tcfg(tmp_path: pathlib.Path) -> TrainConfig:
    """Provide a single-epoch TrainConfig with a temporary checkpoint path.

    Args:
        tmp_path: Pytest-provided temporary directory for checkpoint files.

    Returns:
        TrainConfig configured for one epoch with Adam learning rate and
        W&B disabled, writing checkpoints under pytest's tmp_path.
    """
    return _make_tcfg(tmp_path / "best.pt")


@pytest.fixture
def model_params(
    model: MainTrunk,
    tcfg: TrainConfig,
    distogram_template: Distogram,
    distogram_residue: Distogram,
    distogram_atom: Distogram,
) -> ModelSetup:
    """Provide a ModelSetup bundling model, optimizer, scheduler, and config.

    Args:
        model: Fixture providing the MainTrunk model.
        tcfg: Fixture providing the single-epoch TrainConfig.
        distogram_template: Fixture providing the self-conditioning template
            distogram head.
        distogram_residue: Fixture providing the residue-level distogram head.
        distogram_atom: Fixture providing the atom-level distogram head.

    Returns:
        ModelSetup wrapping the model fixture with an Adam optimizer and
        AlphaFold-style warmup/decay LambdaLR scheduler configured from
        tcfg.
    """
    return _make_model_params(
        model,
        tcfg,
        distogram_template,
        distogram_residue,
        distogram_atom,
    )


@pytest.fixture
def single_sample() -> Protein:
    """Provide a single unbatched protein sample."""
    rng = np.random.default_rng()
    return Protein(
        atom_positions=rng.standard_normal((_N_KEEP, 37, 3)),
        atom_mask=np.ones((_N_KEEP, 37)),
        residue_index=np.arange(_N_KEEP, dtype=np.intp),
        aatype=rng.integers(low=0, high=RESTYPE_NUM, size=(_N_KEEP,)),
        chain_index=np.zeros(_N_KEEP, dtype=np.intp),
        b_factors=np.zeros((_N_KEEP, 37)),
    )


@pytest.fixture
def protein_batch(single_sample: Protein) -> list[Protein]:
    """Provide a list of Protein objects for take_step tests."""
    return [single_sample] * 10


@pytest.fixture
def featurized_batch(
    protein_batch: list[Protein],
    model_params: ModelSetup,
) -> FeaturizedBatch:
    """FeaturizedBatch produced from protein_batch for take_step tests."""
    return featurize_batch(
        protein_batch,
        model_params.tcfg,
        model_params.distogram_template,
        model_params.distogram_residue,
        model_params.distogram_atom,
    )


@pytest.fixture
def loaders(
    tmp_path: pathlib.Path,
    jsonl_path: str,
    splits_path: str,
) -> tuple[
    torch.utils.data.DataLoader[FeaturizedBatch],
    torch.utils.data.DataLoader[FeaturizedBatch],
]:
    """Provide loaders built by make_bucketed_data_loaders.

    Args:
        tmp_path: Pytest-provided temporary directory for shards and config.
        jsonl_path: Fixture providing path to synthetic proteins JSONL file.
        splits_path: Fixture providing path to the train/val/test splits JSON.

    Returns:
        Train and test loaders streaming the synthetic proteins.
    """
    return _make_loaders(tmp_path, jsonl_path, splits_path)


@pytest.fixture
def log() -> FilteringBoundLogger:
    """Provide a structlog logger for training loop tests.

    Returns:
        A structlog bound logger suitable for passing to train(), evaluate(),
        save_checkpoint(), and related functions.
    """
    return cast(FilteringBoundLogger, structlog.get_logger())


@pytest.fixture
def wandb_payloads(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """Patch wandb.log and return the list of captured payload dicts.

    Args:
        monkeypatch: Pytest fixture for patching attributes at test scope.

    Returns:
        List that accumulates one dict per wandb.log call; inspect after the
        system-under-test runs to assert on logged keys and values.
    """
    captured: list[dict[str, object]] = []
    monkeypatch.setattr("train.train_loop.wandb.log", captured.append)
    return captured


@pytest.fixture
def wandb_call_counter(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Patch wandb.log and return a list that grows by one entry per call.

    Args:
        monkeypatch: Pytest fixture for patching attributes at test scope.

    Returns:
        List whose length equals the number of times wandb.log was invoked;
        use len() to assert on call count without inspecting payload contents.
    """
    called: list[object] = []
    monkeypatch.setattr("train.train_loop.wandb.log", called.append)
    return called


# ---------------------------------------------------------------------------
# take_step — eval mode
# ---------------------------------------------------------------------------


def test_take_step_eval_outputs(
    featurized_batch: FeaturizedBatch,
    model_params: ModelSetup,
) -> None:
    """Verify take_step eval-mode outputs are well-formed and gradient-free.

    Checks:
        - Returns a LossMetrics and a ThroughputStatistics instance.
        - Every LossMetrics field is finite.
        - RMSD is non-negative.
        - token_pack_rate is in (0, 1].
        - residues_per_sec and atoms_per_sec are positive.
        - No model parameter accumulates a gradient.
    """
    model_params.model.zero_grad()
    loss_metrics, tput = take_step(
        batch=featurized_batch,
        model_params=model_params,
        train_mode=False,
    )
    assert isinstance(loss_metrics, LossMetrics)
    assert isinstance(tput, ThroughputStatistics)
    for name, v in [
        ("total_loss", loss_metrics.total_loss),
        ("Kabsch_aligned_MSE_loss", loss_metrics.Kabsch_aligned_MSE_loss),
        ("CE_loss", loss_metrics.CE_loss),
        ("smooth_lddt_loss", loss_metrics.smooth_lddt_loss),
        ("res_distogram_loss", loss_metrics.res_distogram_loss),
        ("atom_distogram_loss", loss_metrics.atom_distogram_loss),
        ("intermediate_loss", loss_metrics.intermediate_loss),
        ("RMSD", loss_metrics.RMSD),
    ]:
        assert torch.isfinite(v), f"Field '{name}' is not finite: {v}"
    assert 0.0 < tput.token_pack_rate.item() <= 1.0
    assert tput.residues_per_sec.item() > 0.0
    assert tput.atoms_per_sec.item() > 0.0
    for p in model_params.model.parameters():
        assert p.grad is None


# ---------------------------------------------------------------------------
# take_step — train mode
# ---------------------------------------------------------------------------


def test_take_step_train_produces_gradients(
    featurized_batch: FeaturizedBatch,
    model_params: ModelSetup,
) -> None:
    """take_step(train_mode=True) back-props grads into model parameters."""
    _ = model_params.model.train()
    model_params.model.zero_grad()
    _ = take_step(
        batch=featurized_batch,
        model_params=model_params,
        train_mode=True,
    )
    assert any(p.grad is not None for p in model_params.model.parameters())


def test_take_step_grad_scale_halves_gradient_norm(
    featurized_batch: FeaturizedBatch,
    model_params: ModelSetup,
) -> None:
    """grad_scale=2 yields approx half the gradient L2 norm of grad_scale=1.

    Verifies that doubling grad_scale halves the effective gradient magnitude,
    confirming that take_step applies gradient scaling inversely.
    """
    _ = model_params.model.train()

    _ = manual_seed(0)
    model_params.model.zero_grad()
    _ = take_step(
        batch=featurized_batch,
        model_params=model_params,
        train_mode=True,
        grad_scale=1.0,
    )
    norm1 = cast(
        float,
        sum(
            cast(float, reduce(p.grad**2, "... -> ", "sum").item())
            for p in model_params.model.parameters()
            if p.grad is not None
        )
        ** 0.5,
    )

    _ = manual_seed(0)
    model_params.model.zero_grad()
    _ = take_step(
        batch=featurized_batch,
        model_params=model_params,
        train_mode=True,
        grad_scale=2.0,
    )
    norm2 = cast(
        float,
        sum(
            cast(float, reduce(p.grad**2, "... -> ", "sum").item())
            for p in model_params.model.parameters()
            if p.grad is not None
        )
        ** 0.5,
    )

    assert abs(norm2 - norm1 / 2.0) < TIGHT_TOLERANCE


# ---------------------------------------------------------------------------
# process_accum_window
# ---------------------------------------------------------------------------


def test_process_accum_window_outputs(
    featurized_batch: FeaturizedBatch,
    model_params: ModelSetup,
) -> None:
    """process_accum_window -> finite LossMetrics, ThroughputStatistics."""
    _ = model_params.model.train()
    model_params.model.zero_grad()
    loss_metrics, tput = process_accum_window(
        [featurized_batch],
        [1],
        model_params,
    )
    assert isinstance(loss_metrics, LossMetrics)
    assert isinstance(tput, ThroughputStatistics)
    for name, v in [
        ("total_loss", loss_metrics.total_loss),
        ("Kabsch_aligned_MSE_loss", loss_metrics.Kabsch_aligned_MSE_loss),
        ("CE_loss", loss_metrics.CE_loss),
        ("smooth_lddt_loss", loss_metrics.smooth_lddt_loss),
        ("res_distogram_loss", loss_metrics.res_distogram_loss),
        ("atom_distogram_loss", loss_metrics.atom_distogram_loss),
        ("intermediate_loss", loss_metrics.intermediate_loss),
        ("RMSD", loss_metrics.RMSD),
    ]:
        assert torch.isfinite(v), f"Field '{name}' is not finite: {v}"


def test_process_accum_window_protein_weighted_grad_scale(
    featurized_batch: FeaturizedBatch,
    model_params: ModelSetup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """process_accum_window grad_scale is protein-count-weighted.

    Verifies that each micro-batch's grad_scale equals total_proteins /
    batch_proteins, so batches with fewer proteins receive a larger scale.
    With n_proteins=[1, 3] the total is 4, yielding grad_scale 4/1=4.0 for
    the first batch and 4/3≈1.333 for the second.
    """
    captured_scales: list[float] = []
    _real_take_step = take_step

    def _recording_take_step(
        *,
        batch: FeaturizedBatch,
        model_params: ModelSetup,  # pylint: disable=redefined-outer-name
        train_mode: bool,
        grad_scale: float = 1.0,
    ) -> tuple[LossMetrics, ThroughputStatistics]:
        captured_scales.append(grad_scale)
        return _real_take_step(
            batch=batch,
            model_params=model_params,
            train_mode=train_mode,
            grad_scale=grad_scale,
        )

    monkeypatch.setattr("train.train_loop.take_step", _recording_take_step)
    _ = model_params.model.train()
    model_params.model.zero_grad()
    acummulated_batch = [featurized_batch, featurized_batch]
    _ = process_accum_window(acummulated_batch, [1, 3], model_params)

    assert len(captured_scales) == len(acummulated_batch)
    assert abs(captured_scales[0] - 4.0) < TOLERANCE
    assert abs(captured_scales[1] - 4.0 / 3.0) < TOLERANCE


# ---------------------------------------------------------------------------
# component_grad_norms
# ---------------------------------------------------------------------------


def test_component_grad_norms_outputs(
    featurized_batch: FeaturizedBatch,
    model_params: ModelSetup,
) -> None:
    """Component_grad_norms returns a ComponentNorms with all positive values.

    Checks:
        - Returns a ComponentNorms instance after a backward pass.
        - Every per-component gradient norm is positive.
    """
    _ = model_params.model.train()
    model_params.model.zero_grad()
    _ = take_step(
        batch=featurized_batch,
        model_params=model_params,
        train_mode=True,
    )
    norms = component_grad_norms(model_params.model)
    assert isinstance(norms, ComponentNorms)
    for f in dataclasses.fields(norms):
        v = cast(torch.Tensor, getattr(norms, f.name))
        assert v.item() > 0.0, f"Component norm '{f.name}' is not positive: {v}"


# ---------------------------------------------------------------------------
# optimizer_step
# ---------------------------------------------------------------------------


def test_optimizer_step_outputs(
    featurized_batch: FeaturizedBatch,
    model_params: ModelSetup,
) -> None:
    """Verify optimizer_step returns valid outputs and updates model parameters.

    Checks:
        - Returns a LossMetrics instance as the first element.
        - Returns a ComponentNorms instance as the third element.
        - Returns global_step + 1 as the fourth element.
        - Modifies at least one model parameter via gradient descent.
    """
    params_before = [
        p.clone().detach() for p in model_params.model.parameters()
    ]
    _ = model_params.model.train()
    model_params.model.zero_grad()
    global_step = 5
    loss_metrics, _, norms, new_step = optimizer_step(
        [featurized_batch],
        [1],
        model_params,
        global_step=global_step,
    )
    assert isinstance(loss_metrics, LossMetrics)
    assert isinstance(norms, ComponentNorms)
    assert new_step == global_step + 1
    assert any(
        not torch.equal(b, a)
        for b, a in zip(
            params_before,
            model_params.model.parameters(),
            strict=False,
        )
    )


def test_optimizer_step_updates_ema_shadow(
    featurized_batch: FeaturizedBatch,
    model_params: ModelSetup,
) -> None:
    """optimizer_step advances the EMA shadow using the pre-step global_step.

    Verifies the shadow after one optimizer_step call matches the
    bias-corrected decay formula applied to the model's post-step
    parameters, confirming optimizer_step actually wires ``EMA.update``
    into the training loop rather than only stepping the optimizer.
    """
    name, _ = next(model_params.model.named_parameters())
    shadow_before = model_params.ema.shadow[name].clone()

    _ = model_params.model.train()
    model_params.model.zero_grad()
    global_step = 5
    _ = optimizer_step(
        [featurized_batch],
        [1],
        model_params,
        global_step=global_step,
    )

    param_after = cast(
        dict[str, torch.Tensor],
        model_params.model.state_dict(),
    )[name].clone()
    decay = model_params.tcfg.training.ema_decay
    effective_decay = min(decay, (global_step + 1) / (global_step + 10))
    expected = (
        effective_decay * shadow_before + (1 - effective_decay) * param_after
    )
    assert torch.allclose(
        model_params.ema.shadow[name],
        expected,
        atol=TOLERANCE,
    )


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def test_evaluate_outputs(
    loaders: tuple[
        torch.utils.data.DataLoader[FeaturizedBatch],
        torch.utils.data.DataLoader[FeaturizedBatch],
    ],
    model_params: ModelSetup,
    log: FilteringBoundLogger,
) -> None:
    """Evaluate returns well-formed, finite outputs and leaves no gradients.

    Checks:
        - Returns a LossMetrics instance as its first element.
        - Returns a ThroughputStatistics instance as its second element.
        - Every LossMetrics field is finite.
        - total_loss is non-negative.
        - RMSD is non-negative.
        - No model parameter accumulates a gradient.
    """
    _, test_loader = loaders
    loss_metrics, tput = evaluate(test_loader, model_params, log)
    assert isinstance(loss_metrics, LossMetrics)
    assert isinstance(tput, ThroughputStatistics)
    for name, v in [
        ("total_loss", loss_metrics.total_loss),
        ("Kabsch_aligned_MSE_loss", loss_metrics.Kabsch_aligned_MSE_loss),
        ("CE_loss", loss_metrics.CE_loss),
        ("smooth_lddt_loss", loss_metrics.smooth_lddt_loss),
        ("res_distogram_loss", loss_metrics.res_distogram_loss),
        ("atom_distogram_loss", loss_metrics.atom_distogram_loss),
        ("intermediate_loss", loss_metrics.intermediate_loss),
        ("RMSD", loss_metrics.RMSD),
    ]:
        assert torch.isfinite(v), f"Field '{name}' is not finite: {v}"
    assert loss_metrics.total_loss.item() >= 0.0
    assert loss_metrics.RMSD.item() >= 0.0
    for p in model_params.model.parameters():
        assert p.grad is None


# ---------------------------------------------------------------------------
# swapped_in_ema_weights
# ---------------------------------------------------------------------------


def test_swapped_in_ema_weights_loads_shadow_and_restores_raw_after(
    model_params: ModelSetup,
) -> None:
    """Entering swaps in the EMA shadow; exiting restores the raw weights.

    Advances the EMA shadow away from the model's raw weights first (via a
    manual perturbation) so the swap is actually observable, then checks
    the model matches the shadow inside the context and matches the
    original raw weights again once it exits. ``state_dict()`` is called
    fresh at each checkpoint rather than reused, since ``swapped_in_ema_
    weights`` now reassigns ``.data`` in place (zero-copy) instead of
    copying values -- a state_dict captured before the swap holds
    ``.detach()``-ed views of the pre-swap storage and would look
    unchanged forever if reused, regardless of whether the swap actually
    happened.
    """
    with torch.no_grad():
        for p in model_params.model.parameters():
            _ = p.add_(1.0)
    model_params.ema.update(model_params.model, step=0)

    raw_sd = {
        k: v.clone()
        for k, v in cast(
            dict[str, torch.Tensor],
            model_params.model.state_dict(),
        ).items()
    }
    ema_sd = {k: v.clone() for k, v in model_params.ema.state_dict().items()}

    with swapped_in_ema_weights(model_params):
        live_sd = cast(dict[str, torch.Tensor], model_params.model.state_dict())
        for k, v in live_sd.items():
            assert torch.equal(v, ema_sd[k]), f"Param '{k}' not swapped to EMA"

    live_sd_after = cast(
        dict[str, torch.Tensor],
        model_params.model.state_dict(),
    )
    for k, v in live_sd_after.items():
        assert torch.equal(v, raw_sd[k]), f"Param '{k}' not restored after swap"


def test_swapped_in_ema_weights_restores_raw_weights_on_exception(
    model_params: ModelSetup,
) -> None:
    """Raw weights are restored even when the wrapped block raises.

    Ensures a mid-block failure (e.g. an eval-time error) can't leave the
    model stuck on EMA weights for subsequent training steps. Perturbs the
    model and advances the EMA shadow first so raw and shadow values
    genuinely differ -- otherwise this would pass trivially even if the
    ``finally`` swap-back never ran, since raw and shadow would already be
    identical.
    """
    with torch.no_grad():
        for p in model_params.model.parameters():
            _ = p.add_(1.0)
    model_params.ema.update(model_params.model, step=0)

    raw_sd = {
        k: v.clone()
        for k, v in cast(
            dict[str, torch.Tensor],
            model_params.model.state_dict(),
        ).items()
    }

    eval_failure_message = "simulated eval failure"
    with (
        pytest.raises(RuntimeError, match=eval_failure_message),
        swapped_in_ema_weights(model_params),
    ):
        raise RuntimeError(eval_failure_message)

    live_sd = cast(dict[str, torch.Tensor], model_params.model.state_dict())
    for k, v in live_sd.items():
        assert torch.equal(
            v,
            raw_sd[k],
        ), f"Param '{k}' not restored after error"


def test_swapped_in_ema_weights_unwraps_ddp(
    model_params: ModelSetup,
) -> None:
    """Swap operates on the unwrapped .module when the model is DDP-wrapped.

    Mirrors test_save_checkpoint_strips_ddp_wrapper_prefix: wraps the model
    in a single-process gloo DDP group and confirms the swap-in/swap-out
    still works against the underlying MainTrunk rather than erroring on a
    "module."-prefix key mismatch.
    """
    with torch.no_grad():
        for p in model_params.model.parameters():
            _ = p.add_(1.0)
    model_params.ema.update(model_params.model, step=0)
    raw_sd = {
        k: v.clone()
        for k, v in cast(
            dict[str, torch.Tensor],
            model_params.model.state_dict(),
        ).items()
    }
    ema_sd = {k: v.clone() for k, v in model_params.ema.state_dict().items()}

    with _single_process_gloo_group():
        ddp_model_params = dataclasses.replace(
            model_params,
            model=DDP(model_params.model),
        )
        with swapped_in_ema_weights(ddp_model_params):
            live_sd = cast(
                dict[str, torch.Tensor],
                model_params.model.state_dict(),
            )
            for k, v in live_sd.items():
                assert torch.equal(v, ema_sd[k])

        live_sd_after = cast(
            dict[str, torch.Tensor],
            model_params.model.state_dict(),
        )
        for k, v in live_sd_after.items():
            assert torch.equal(v, raw_sd[k])


def test_swapped_in_ema_weights_unwraps_ddp_and_restores_on_exception(
    model_params: ModelSetup,
) -> None:
    """DDP-wrapped raw weights are restored even when the wrapped block raises.

    Combines test_swapped_in_ema_weights_restores_raw_weights_on_exception
    and test_swapped_in_ema_weights_unwraps_ddp: a mid-block failure under a
    DDP-wrapped model must still restore the raw weights via ``.module``
    rather than leaving the model stuck on EMA weights or erroring on the
    "module."-prefix key mismatch.
    """
    with torch.no_grad():
        for p in model_params.model.parameters():
            _ = p.add_(1.0)
    model_params.ema.update(model_params.model, step=0)
    raw_sd = {
        k: v.clone()
        for k, v in cast(
            dict[str, torch.Tensor],
            model_params.model.state_dict(),
        ).items()
    }

    eval_failure_message = "simulated eval failure under ddp"
    with _single_process_gloo_group():
        ddp_model_params = dataclasses.replace(
            model_params,
            model=DDP(model_params.model),
        )
        with (
            pytest.raises(RuntimeError, match=eval_failure_message),
            swapped_in_ema_weights(ddp_model_params),
        ):
            raise RuntimeError(eval_failure_message)

        live_sd = cast(
            dict[str, torch.Tensor],
            model_params.model.state_dict(),
        )
        for k, v in live_sd.items():
            assert torch.equal(
                v,
                raw_sd[k],
            ), f"Param '{k}' not restored after error"


# ---------------------------------------------------------------------------
# save_checkpoint / load_checkpoint
# ---------------------------------------------------------------------------


def test_save_checkpoint_rank0_writes_file_with_expected_keys(
    model_params: ModelSetup,
    log: FilteringBoundLogger,
) -> None:
    """Verify save_checkpoint on rank 0 writes a file with the expected keys.

    Checks:
        - A checkpoint file is created at the configured path.
        - Checkpoint contains model, optimizer, scheduler, best_val_loss keys.
    """
    save_checkpoint(
        model_params=model_params,
        rank=0,
        log=log,
        best_val_loss=torch.tensor(0.5),
    )
    assert model_params.tcfg.checkpoint.checkpoint_path.exists()
    ckpt = cast(
        dict[str, object],
        torch.load(
            model_params.tcfg.checkpoint.checkpoint_path,
            weights_only=True,
        ),
    )
    assert isinstance(ckpt, dict)
    assert ckpt.keys() >= EXPECTED_CHECKPOINT_KEYS


def test_save_checkpoint_rank1_skips_write(
    model_params: ModelSetup,
    log: FilteringBoundLogger,
) -> None:
    """save_checkpoint does not write a file on rank != 0."""
    save_checkpoint(
        model_params=model_params,
        rank=1,
        log=log,
        best_val_loss=torch.tensor(1.0),
    )
    assert not model_params.tcfg.checkpoint.checkpoint_path.exists()


def test_load_checkpoint_restores_model_state(
    model_params: ModelSetup,
    log: FilteringBoundLogger,
) -> None:
    """load_checkpoint restores model weights matching the saved state dict."""
    save_checkpoint(
        model_params=model_params,
        rank=0,
        log=log,
        best_val_loss=torch.tensor(0.42),
    )
    original_sd = cast(
        dict[str, torch.Tensor],
        torch.load(
            model_params.tcfg.checkpoint.checkpoint_path,
            weights_only=True,
        )["model"],
    )
    for p in model_params.model.parameters():
        _ = nn.init.zeros_(p)
    restored, _ = load_checkpoint(model_params=model_params, rank=0, log=log)
    restored_sd = cast(dict[str, torch.Tensor], restored.model.state_dict())
    for k, v in restored_sd.items():
        assert torch.allclose(v, original_sd[k]), f"Param '{k}' not restored"


def test_load_checkpoint_restores_ema_state(
    model_params: ModelSetup,
    log: FilteringBoundLogger,
) -> None:
    """load_checkpoint restores EMA shadow weights matching the saved state.

    Mirrors test_load_checkpoint_restores_model_state but for the EMA
    shadow: advances it away from its initial (model-matching) values,
    saves, zeroes the in-memory shadow, then confirms load_checkpoint
    restores the exact saved values rather than the assertion trivially
    passing.
    """
    with torch.no_grad():
        for p in model_params.model.parameters():
            _ = p.add_(1.0)
    model_params.ema.update(model_params.model, step=0)
    save_checkpoint(
        model_params=model_params,
        rank=0,
        log=log,
        best_val_loss=torch.tensor(0.42),
    )
    original_ema_sd = {
        k: v.clone() for k, v in model_params.ema.state_dict().items()
    }

    for tensor in model_params.ema.shadow.values():
        _ = tensor.zero_()

    restored, _ = load_checkpoint(model_params=model_params, rank=0, log=log)
    for k, v in restored.ema.state_dict().items():
        assert torch.allclose(
            v,
            original_ema_sd[k],
        ), f"EMA param '{k}' not restored"


def test_checkpoint_round_trip_preserves_weights_and_loss(
    model_params: ModelSetup,
    log: FilteringBoundLogger,
) -> None:
    """save→load→save is lossless: second checkpoint matches first.

    Saves a checkpoint, reloads it into a fresh model, saves again, and
    confirms that the model weights and best_val_loss are identical across
    both saved files.
    """
    best_val_loss = torch.tensor(0.271)
    path = model_params.tcfg.checkpoint.checkpoint_path

    save_checkpoint(
        model_params=model_params,
        rank=0,
        log=log,
        best_val_loss=best_val_loss,
    )
    model_sd = cast(dict[str, torch.Tensor], model_params.model.state_dict())
    first_sd = {k: v.clone() for k, v in model_sd.items()}

    for p in model_params.model.parameters():
        _ = nn.init.zeros_(p)

    restored, loaded_loss = load_checkpoint(
        model_params=model_params,
        rank=0,
        log=log,
    )
    save_checkpoint(
        model_params=restored,
        rank=0,
        log=log,
        best_val_loss=loaded_loss,
    )

    second_ckpt = cast(
        dict[str, object],
        torch.load(path, weights_only=True),
    )
    second_ckpt_model = cast(
        dict[str, torch.Tensor],
        second_ckpt["model"],
    )
    for k, v in first_sd.items():
        assert torch.allclose(
            v,
            second_ckpt_model[k],
        ), f"Param '{k}' differs after round trip"
    second_ckpt_loss = cast(torch.Tensor, second_ckpt["best_val_loss"])
    assert abs(second_ckpt_loss.item() - best_val_loss.item()) < TOLERANCE


def test_checkpoint_round_trip_preserves_ema_state(
    model_params: ModelSetup,
    log: FilteringBoundLogger,
) -> None:
    """save->load->save is lossless for EMA shadow weights.

    Mirrors test_checkpoint_round_trip_preserves_weights_and_loss but for
    the EMA shadow: saves a checkpoint, reloads it, saves again, and
    confirms the EMA weights are identical across both saved files.
    """
    path = model_params.tcfg.checkpoint.checkpoint_path
    with torch.no_grad():
        for p in model_params.model.parameters():
            _ = p.add_(1.0)
    model_params.ema.update(model_params.model, step=0)

    save_checkpoint(
        model_params=model_params,
        rank=0,
        log=log,
        best_val_loss=torch.tensor(0.271),
    )
    first_ema_sd = {
        k: v.clone() for k, v in model_params.ema.state_dict().items()
    }

    for tensor in model_params.ema.shadow.values():
        _ = tensor.zero_()

    restored, loaded_loss = load_checkpoint(
        model_params=model_params,
        rank=0,
        log=log,
    )
    save_checkpoint(
        model_params=restored,
        rank=0,
        log=log,
        best_val_loss=loaded_loss,
    )

    second_ckpt = cast(
        dict[str, object],
        torch.load(path, weights_only=True),
    )
    second_ckpt_ema = cast(dict[str, torch.Tensor], second_ckpt["ema"])
    for k, v in first_ema_sd.items():
        assert torch.allclose(
            v,
            second_ckpt_ema[k],
        ), f"EMA param '{k}' differs after round trip"


def test_checkpoint_round_trip_preserves_optimizer_and_scheduler_state(
    model_params: ModelSetup,
    log: FilteringBoundLogger,
) -> None:
    """save->load restores Adam moments and scheduler step count exactly.

    Populates optimizer/scheduler state with a manual step, saves it,
    corrupts the in-memory state, then confirms load_checkpoint restores the
    original values rather than the assertions trivially passing.
    """
    for p in model_params.model.parameters():
        p.grad = torch.ones_like(p)
    _ = (
        model_params.optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    )
    model_params.scheduler.step()
    model_params.optimizer.zero_grad()

    save_checkpoint(
        model_params=model_params,
        rank=0,
        log=log,
        best_val_loss=torch.tensor(0.1),
    )
    saved_opt_state = cast(
        dict[int, dict[str, torch.Tensor]],
        cast(
            dict[str, object],
            model_params.optimizer.state_dict(),
        )["state"],
    )
    saved_exp_avg = {
        i: state[_ADAM_EXP_AVG_KEY].clone()
        for i, state in saved_opt_state.items()
    }
    saved_last_epoch = cast(
        int,
        cast(dict[str, object], model_params.scheduler.state_dict())[
            "last_epoch"
        ],
    )

    live_opt_state = cast(
        dict[torch.nn.Parameter, dict[str, torch.Tensor]],
        model_params.optimizer.state,
    )
    for state in live_opt_state.values():
        _ = state[_ADAM_EXP_AVG_KEY].zero_()
    model_params.scheduler.last_epoch = -1

    restored, _ = load_checkpoint(model_params=model_params, rank=0, log=log)

    restored_opt_state = cast(
        dict[int, dict[str, torch.Tensor]],
        cast(
            dict[str, object],
            restored.optimizer.state_dict(),
        )["state"],
    )
    for i, exp_avg in saved_exp_avg.items():
        assert torch.allclose(
            restored_opt_state[i][_ADAM_EXP_AVG_KEY],
            exp_avg,
        ), f"Adam exp_avg for param {i} not restored"
    restored_last_epoch = cast(
        int,
        cast(dict[str, object], restored.scheduler.state_dict())["last_epoch"],
    )
    assert restored_last_epoch == saved_last_epoch


@contextlib.contextmanager
def _single_process_gloo_group() -> Generator[None]:
    """Init and tear down a single-process gloo group for DDP-wrapping tests."""
    # HashStore is a private torch.distributed C-extension symbol that
    # isn't officially re-exported; getattr() sidesteps the static
    # private-import check, whose result is otherwise inconsistent
    # across torch's per-platform packaging.
    dist.init_process_group(
        backend="gloo",
        store=getattr(  # noqa: B009  # pyright: ignore[reportAny]
            dist,
            "HashStore",
        )(),
        rank=0,
        world_size=1,
    )
    try:
        yield
    finally:
        dist.destroy_process_group()


def test_save_checkpoint_strips_ddp_wrapper_prefix(
    model_params: ModelSetup,
    log: FilteringBoundLogger,
) -> None:
    """Checkpoint saved from a DDP-wrapped model has unprefixed model keys.

    Wraps model_params.model in DDP under a single-process gloo group (no
    network ports or GPUs needed) and confirms save_checkpoint unwraps
    ``.module`` before calling state_dict(), so the saved keys match the
    plain model and never carry a "module." prefix.
    """
    plain_keys = set(model_params.model.state_dict().keys())
    with _single_process_gloo_group():
        ddp_model_params = dataclasses.replace(
            model_params,
            model=DDP(model_params.model),
        )
        save_checkpoint(
            model_params=ddp_model_params,
            rank=0,
            log=log,
            best_val_loss=torch.tensor(0.3),
        )

    ckpt = cast(
        dict[str, object],
        torch.load(
            model_params.tcfg.checkpoint.checkpoint_path,
            weights_only=True,
        ),
    )
    model_sd = cast(dict[str, torch.Tensor], ckpt["model"])
    assert not any(k.startswith("module.") for k in model_sd)
    assert model_sd.keys() == plain_keys


# ---------------------------------------------------------------------------
# log_epoch
# ---------------------------------------------------------------------------


def _tcfg_with_wandb(*, base: TrainConfig, use_wandb: bool) -> TrainConfig:
    """Return a copy of base with use_wandb overridden.

    Args:
        base: Base TrainConfig to copy from.
        use_wandb: Whether to enable W&B logging.

    Returns:
        New TrainConfig with logging.use_wandb set to use_wandb.
    """
    return TrainConfig(
        training=base.training,
        model=base.model,
        checkpoint=base.checkpoint,
        logging=LoggingParams(use_wandb=use_wandb),
    )


def test_log_epoch_do_log_false_skips_wandb(
    model_params: ModelSetup,
    log: FilteringBoundLogger,
    wandb_call_counter: list[object],
) -> None:
    """log_epoch(do_log=False) does not call wandb.log if use_wandb=True."""
    mp = dataclasses.replace(
        model_params,
        tcfg=_tcfg_with_wandb(base=model_params.tcfg, use_wandb=True),
    )
    log_epoch(
        epoch_metrics=_make_epoch_metrics(epoch=1, global_step=1),
        model_params=mp,
        log=log,
        do_log=False,
    )
    assert len(wandb_call_counter) == 0


def test_log_epoch_wandb_disabled_skips_call(
    model_params: ModelSetup,
    log: FilteringBoundLogger,
    wandb_call_counter: list[object],
) -> None:
    """log_epoch(use_wandb=False) does not call wandb.log if do_log=True."""
    log_epoch(
        epoch_metrics=_make_epoch_metrics(epoch=1, global_step=1),
        model_params=model_params,
        log=log,
        do_log=True,
    )
    assert len(wandb_call_counter) == 0


def test_log_epoch_wandb_enabled_calls_once(
    model_params: ModelSetup,
    log: FilteringBoundLogger,
    wandb_payloads: list[dict[str, object]],
) -> None:
    """log_epoch with do_log=True and use_wandb=True calls wandb.log once."""
    mp = dataclasses.replace(
        model_params,
        tcfg=_tcfg_with_wandb(base=model_params.tcfg, use_wandb=True),
    )
    log_epoch(
        epoch_metrics=_make_epoch_metrics(epoch=3, global_step=1),
        model_params=mp,
        log=log,
        do_log=True,
    )
    assert len(wandb_payloads) == 1


def test_log_epoch_wandb_payload_keys(
    model_params: ModelSetup,
    log: FilteringBoundLogger,
    wandb_payloads: list[dict[str, object]],
) -> None:
    """Verify W&B payload from log_epoch has expected epoch and key prefixes.

    Checks:
        - The 'epoch' key equals the epoch embedded in EpochMetrics.
        - At least one key is prefixed with 'train/'.
        - At least one key is prefixed with 'val/'.
    """
    mp = dataclasses.replace(
        model_params,
        tcfg=_tcfg_with_wandb(base=model_params.tcfg, use_wandb=True),
    )
    log_epoch(
        epoch_metrics=_make_epoch_metrics(epoch=EPOCH, global_step=1),
        model_params=mp,
        log=log,
        do_log=True,
    )
    assert wandb_payloads[0]["epoch"] == EPOCH
    assert any(k.startswith("train/") for k in wandb_payloads[0])
    assert any(k.startswith("val/") for k in wandb_payloads[0])


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def test_train_token_budget_preflush_fires_before_oversized_batch(
    model_params: ModelSetup,
    jsonl_path: str,
    splits_path: str,
    log: FilteringBoundLogger,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """Pre-flush fires when the next batch would push tokens over the budget.

    Loader yields 4 batches of _N_RES_BUCKET tokens each with budget=7:
    - batch1 (buffer empty): added, no pre-flush
    - batch2: tokens+tokens > 7 → pre-flush batch1, then batch2 added
    - batch3: tokens+tokens > 7 → pre-flush batch2, then batch3 added
    - batch4: tokens+tokens > 7 → pre-flush batch3, then batch4 added
    - epoch end: batch4 remains in buffer → end-of-epoch flush of batch4
    Result: four optimizer windows each of size 1; no batch is dropped.
    """
    window_sizes: list[int] = []
    _real_process = process_accum_window

    def _tracking_process(
        micro_buffer: list[FeaturizedBatch],
        n_proteins_per_batch: list[int],
        model_params: ModelSetup,  # pylint: disable=redefined-outer-name
    ) -> tuple[LossMetrics, ThroughputStatistics]:
        window_sizes.append(len(micro_buffer))
        return _real_process(
            micro_buffer,
            n_proteins_per_batch,
            model_params,
        )

    monkeypatch.setattr(
        "train.train_loop.process_accum_window",
        _tracking_process,
    )

    budget: int = 7  # tokens+tokens > 7 forces a pre-flush on every 2nd batch
    tcfg_b = TrainConfig(
        training=TrainingParams(
            num_epochs=1,
            lr=1e-4,
            grad_clip=1.0,
            accumulated_token_budget=budget,
        ),
        model=model_params.tcfg.model,
        distogram_template=TemplateDistogramParams(
            n_bins=_N_BINS,
            min_dist=3.25,
            max_dist=50.75,
        ),
        distogram_residue=ResidueDistogramParams(
            n_bins=_N_BINS,
            min_dist=2.0,
            max_dist=22.0,
        ),
        distogram_atom=AtomDistogramParams(
            n_bins=_N_ATOM_BINS,
            min_dist=0.0,
            max_dist=10.0,
        ),
        checkpoint=CheckpointParams(
            checkpoint_path=tmp_path / "flush.pt",
            save_every=100,
        ),
        logging=LoggingParams(use_wandb=False),
        test_loader=LoaderConfig(batch_size=1),
        train_loader=TrainLoaderConfig(
            token_budget=_N_RES_BUCKET,
            noise_magnitude=0,
            n_shards=1,
            num_workers=1,
            epoch_prefetch_depth=1,
            batch_prefetch_depth=1,
            n_threads=1,
        ),
    )
    mp_b = _make_model_params(
        model_params.model,
        tcfg_b,
        model_params.distogram_template,
        model_params.distogram_residue,
        model_params.distogram_atom,
    )
    args = TrainArgs(
        dataset_jsonl=pathlib.Path(jsonl_path),
        shard_dir=tmp_path / "shards",
        keys_for_splits_json=pathlib.Path(splits_path),
        config=tmp_path / "config.json",
        structlog_jsonl=tmp_path / "structlog.jsonl",
        ddp=False,
        debug_run=False,
    )
    train_loader, test_loader, _ = make_bucketed_data_loaders(
        cfg=tcfg_b,
        extra_train_args=args,
    )
    train(
        best_val_loss=torch.tensor(float("inf")),
        train_loader=train_loader,
        test_loader=test_loader,
        model_params=mp_b,
        log=log,
    )
    assert window_sizes == [1, 1, 1, 1]


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_accumulated_token_budget_rejects_zero() -> None:
    """TrainingParams raises ValidationError when token budget zero."""
    with pytest.raises(ValidationError):
        _ = TrainingParams(accumulated_token_budget=0)


def test_train_config_accepts_any_positive_token_budget() -> None:
    """TrainConfig accepts any positive accumulated_token_budget."""
    cfg = TrainConfig(training=TrainingParams(accumulated_token_budget=1))
    assert cfg.training.accumulated_token_budget == 1


_WARMUP_DECAY_BASE_LR = 2.0
_WARMUP_DECAY_WARMUP_STEPS = 4
_WARMUP_DECAY_DECAY_STEPS = 3
_WARMUP_DECAY_DECAY_FACTOR = 0.5


@pytest.fixture
def warmup_decay_optimizer() -> Adam:
    """Adam optimizer over a single dummy parameter at a fixed base LR."""
    param = nn.Parameter(torch.zeros(1))
    return Adam([param], lr=_WARMUP_DECAY_BASE_LR)


@pytest.mark.parametrize(
    "n_steps",
    [0, 1, 3, 4, 5, 7, 10],
    ids=[
        "construction_only_first_warmup_value",
        "one_step_into_warmup",
        "last_warmup_step",
        "warmup_just_ended_no_decay",
        "one_step_past_warmup_before_decay_boundary",
        "first_decay_boundary_reached",
        "second_decay_boundary_reached",
    ],
)
def test_lambda_lr_matches_alphafold_lr_lambda_through_real_scheduler(
    warmup_decay_optimizer: Adam,
    n_steps: int,
) -> None:
    """A real LambdaLR driven by alphafold_lr_lambda reproduces its outputs.

    Steps a genuine torch.optim.lr_scheduler.LambdaLR n_steps times beyond
    its construction-time initial value, then checks the optimizer's live
    LR equals base_lr times alphafold_lr_lambda evaluated at the same step
    count -- exercising the real PyTorch scheduler machinery end to end
    rather than just the bare multiplier function in isolation.
    """
    scheduler = LambdaLR(
        warmup_decay_optimizer,
        lr_lambda=functools.partial(
            alphafold_lr_lambda,
            warmup_steps=_WARMUP_DECAY_WARMUP_STEPS,
            decay_steps=_WARMUP_DECAY_DECAY_STEPS,
            decay_factor=_WARMUP_DECAY_DECAY_FACTOR,
        ),
    )
    for _ in range(n_steps):
        scheduler.step()
    expected = _WARMUP_DECAY_BASE_LR * alphafold_lr_lambda(
        n_steps,
        warmup_steps=_WARMUP_DECAY_WARMUP_STEPS,
        decay_steps=_WARMUP_DECAY_DECAY_STEPS,
        decay_factor=_WARMUP_DECAY_DECAY_FACTOR,
    )
    actual = cast(float, warmup_decay_optimizer.param_groups[0]["lr"])
    expected_approx = pytest.approx(  # pyright: ignore[reportUnknownMemberType]
        expected,
        rel=TOLERANCE,
    )
    assert actual == expected_approx


_N_RES_BUCKET = 42
_ENTRY_NAMES_BUCKET = ["1aa.A", "2bb.A", "3cc.A", "4dd.A", "5ee.A", "6ff.A"]
_TRAIN_NAMES_BUCKET = ["1aa.A", "2bb.A", "3cc.A", "6ff.A"]
_VAL_NAMES_BUCKET = ["4dd.A"]
_TEST_NAMES_BUCKET = ["5ee.A"]


@pytest.fixture(scope="module")
def jsonl_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Write JSONL file with synthetic protein entries and return its path."""
    path = tmp_path_factory.mktemp("dataset") / "proteins.jsonl"
    with path.open("w") as f:
        for name in _ENTRY_NAMES_BUCKET:
            entry = {
                "name": name,
                "seq": "A" * _N_RES_BUCKET,
                "coords": {
                    atom: np.zeros((_N_RES_BUCKET, 3)).tolist()
                    for atom in ("N", "CA", "C", "O")
                },
            }
            _ = f.write(json.dumps(entry) + "\n")
    return str(path)


@pytest.fixture(scope="module")
def splits_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Write splits JSON with train/val/test name lists and return its path."""
    path = tmp_path_factory.mktemp("splits") / "splits.json"
    with path.open("w") as f:
        json.dump(
            {
                "train": _TRAIN_NAMES_BUCKET,
                "validation": _VAL_NAMES_BUCKET,
                "test": _TEST_NAMES_BUCKET,
            },
            f,
        )
    return str(path)


class TrainScenario(Enum):
    """Configurations train() runs under, each exercised by one shared run.

    A real ``train()`` epoch takes seconds, so ``recorded_train_runs`` runs
    each scenario once per module and every test asserts on the result.
    Each member is a ``(num_epochs, budget_multiplier, grad_clip, use_wandb)``
    tuple read through properties; the budget multiplier scales
    ``_BATCH_TOKENS``. Tests share this one enum rather than each subclassing
    a base enum, because members of separate subclasses are separate
    parameters and would each trigger their own ``train()`` run.

    Every budget here, including ``large_token_budget``, is below the 168
    tokens of the two synthetic train batches (84 tokens each). So each
    epoch flushes the first batch when the second arrives, and flushes the
    second at the end of the epoch: two optimizer steps per epoch.

    Members:
        default_budget: One epoch with clipping on and W&B off.
        no_grad_clip: As ``default_budget`` with gradient clipping disabled.
        large_token_budget: A budget above one batch's tokens, with W&B on.
        multi_epoch: Two epochs with W&B on, the fewest that can show the
            logged epoch advancing.
    """

    default_budget = (1, 1, 1.0, False)
    no_grad_clip = (1, 1, None, False)
    large_token_budget = (1, 10, 1.0, True)
    multi_epoch = (2, 1, 1.0, True)

    @property
    def num_epochs(self) -> int:
        """Number of training epochs."""
        return self.value[0]

    @property
    def accumulated_token_budget(self) -> int:
        """Tokens accumulated before an optimizer step."""
        return self.value[1] * _BATCH_TOKENS

    @property
    def grad_clip(self) -> float | None:
        """Gradient clipping threshold, or None to disable clipping."""
        return self.value[2]

    @property
    def use_wandb(self) -> bool:
        """Whether W&B logging is enabled."""
        return self.value[3]


@dataclasses.dataclass(frozen=True)
class TrainRun:
    """What one train() run did, recorded by ``_record_train_run``.

    Attributes:
        parameters_changed: True if any model parameter changed in the run.
        reduced_metrics: Every instance all_reduce_mean() was called on.
        wandb_payloads: One dict per wandb.log call.
        checkpoint: The saved checkpoint, loaded back from disk.
        num_epochs: Number of epochs the run was configured for.
    """

    parameters_changed: bool
    reduced_metrics: list[TensorAccumulatorMixin]
    wandb_payloads: list[dict[str, object]]
    checkpoint: dict[str, dict[str, object]]
    num_epochs: int


def _record_train_run(
    scenario: TrainScenario,
    work_dir: pathlib.Path,
    dataset_jsonl: str,
    splits_json: str,
) -> TrainRun:
    """Run train() once for a scenario and record what it did.

    Arrange: build a fresh model and config, spy on all_reduce_mean(), and
    capture wandb.log. Function-scoped ``monkeypatch`` cannot serve a
    module-scoped cache, so the patches use ``MonkeyPatch.context``.
    Act: run train() on the real loaders.

    Args:
        scenario: Configuration to run.
        work_dir: Directory for the checkpoint, shards, and logs.
        dataset_jsonl: Path to the synthetic proteins JSONL file.
        splits_json: Path to the train/val/test splits JSON.

    Returns:
        TrainRun recording the scenario's outcome.
    """
    _ = manual_seed(0)
    trunk = _make_model()
    setup = _make_model_params(
        trunk,
        _make_tcfg(
            work_dir / "best.pt",
            num_epochs=scenario.num_epochs,
            grad_clip=scenario.grad_clip,
            accumulated_token_budget=scenario.accumulated_token_budget,
            use_wandb=scenario.use_wandb,
        ),
        _make_distogram_template(),
        _make_distogram_residue(),
        _make_distogram_atom(),
    )
    train_loader, test_loader = _make_loaders(
        work_dir,
        dataset_jsonl,
        splits_json,
    )
    params_before = [p.clone().detach() for p in trunk.parameters()]
    reduced_metrics: list[TensorAccumulatorMixin] = []
    logged_payloads: list[dict[str, object]] = []
    original_reduce = TensorAccumulatorMixin.all_reduce_mean

    def _spy(self: TensorAccumulatorMixin) -> None:
        """Record the instance, then run the real all_reduce_mean()."""
        reduced_metrics.append(self)
        original_reduce(self)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(TensorAccumulatorMixin, "all_reduce_mean", _spy)
        patch.setattr("train.train_loop.wandb.log", logged_payloads.append)
        train(
            best_val_loss=torch.tensor(float("inf")),
            train_loader=train_loader,
            test_loader=test_loader,
            model_params=setup,
            log=cast(FilteringBoundLogger, structlog.get_logger()),
        )
    return TrainRun(
        parameters_changed=any(
            not torch.equal(before, after)
            for before, after in zip(
                params_before,
                trunk.parameters(),
                strict=True,
            )
        ),
        reduced_metrics=reduced_metrics,
        wandb_payloads=logged_payloads,
        checkpoint=cast(
            dict[str, dict[str, object]],
            torch.load(work_dir / "best.pt", weights_only=True),
        ),
        num_epochs=scenario.num_epochs,
    )


@pytest.fixture(scope="module")
def recorded_train_runs(
    tmp_path_factory: pytest.TempPathFactory,
    jsonl_path: str,
    splits_path: str,
) -> Callable[[TrainScenario], TrainRun]:
    """Provide a lookup that runs each TrainScenario at most once per module.

    The cache is explicit because a parametrized module-scoped fixture does
    not guarantee one run per scenario: pytest groups parameters by their
    position in each ``parametrize`` list, so the same member in two lists
    is torn down and built again.

    Args:
        tmp_path_factory: Factory for each run's temporary directory.
        jsonl_path: Fixture providing path to synthetic proteins JSONL file.
        splits_path: Fixture providing path to the train/val/test splits JSON.

    Returns:
        Callable returning the scenario's TrainRun, running it on first use.
    """
    runs: dict[TrainScenario, TrainRun] = {}

    def _lookup(scenario: TrainScenario) -> TrainRun:
        """Return the scenario's recorded run, running it on first use."""
        if scenario not in runs:
            runs[scenario] = _record_train_run(
                scenario,
                tmp_path_factory.mktemp(scenario.name),
                jsonl_path,
                splits_path,
            )
        return runs[scenario]

    return _lookup


@pytest.fixture
def train_run(
    request: pytest.FixtureRequest,
    recorded_train_runs: Callable[[TrainScenario], TrainRun],
) -> TrainRun:
    """Provide the shared recorded run for a scenario.

    Factory fixture: receives a ``TrainScenario`` through ``request.param``.
    The result is a frozen snapshot rather than live training state, so the
    tests sharing it stay order-independent.

    Args:
        request: Pytest request carrying the ``TrainScenario``.
        recorded_train_runs: Module-wide lookup of recorded runs.

    Returns:
        TrainRun recording the scenario's outcome.
    """
    return recorded_train_runs(cast(TrainScenario, request.param))


@pytest.mark.parametrize(
    "train_run",
    list(TrainScenario),
    ids=[case.name for case in TrainScenario],
    indirect=True,
)
def test_train_updates_model_parameters(train_run: TrainRun) -> None:
    """Train changes model parameters under every scenario.

    Args:
        train_run: Recorded run for the scenario.
    """
    assert train_run.parameters_changed


@pytest.mark.parametrize(
    "train_run",
    [pytest.param(TrainScenario.default_budget, id="default_budget")],
    indirect=True,
)
def test_train_calls_all_reduce_mean_on_all_epoch_metrics(
    train_run: TrainRun,
) -> None:
    """train() calls all_reduce_mean() on every metric feeding EpochMetrics.

    Arrange: ``train_run`` wraps TensorAccumulatorMixin.all_reduce_mean with
    a spy that records every instance before delegating to the real
    implementation (which no-ops here, since no distributed process group
    is active -- the reduction arithmetic itself is proven separately, with
    real multi-rank processes, by
    test_all_reduce_mean_averages_across_real_ranks in
    test_useful_objects.py). This test only proves train() invokes the
    reduction on the right objects, not the arithmetic.
    Act: run train() for one epoch.
    Assert: the spy was invoked on exactly four metric instances: two
    LossMetrics (train and val), one ThroughputStatistics, and one
    ComponentNorms -- matching every metric field of EpochMetrics.

    One scenario is enough: the reduction calls do not depend on the
    configuration.

    Args:
        train_run: Recorded one-epoch run.
    """
    calls = train_run.reduced_metrics
    expected_loss_metrics_calls = 2  # train_loss_metrics + val_loss_metrics
    expected_throughput_calls = 1  # train_throughput_stats
    expected_norms_calls = 1  # train_gradient_norms
    expected_total_calls = (
        expected_loss_metrics_calls
        + expected_throughput_calls
        + expected_norms_calls
    )

    assert len(calls) == expected_total_calls
    assert (
        sum(isinstance(c, LossMetrics) for c in calls)
        == expected_loss_metrics_calls
    )
    assert (
        sum(isinstance(c, ThroughputStatistics) for c in calls)
        == expected_throughput_calls
    )
    assert (
        sum(isinstance(c, ComponentNorms) for c in calls)
        == expected_norms_calls
    )


@pytest.mark.parametrize(
    "train_run",
    [
        pytest.param(TrainScenario.default_budget, id="default_budget"),
        pytest.param(TrainScenario.no_grad_clip, id="no_grad_clip"),
    ],
    indirect=True,
)
def test_train_wandb_not_called_when_disabled(train_run: TrainRun) -> None:
    """Train does not call wandb.log when use_wandb=False.

    Args:
        train_run: Recorded run with W&B disabled.
    """
    assert train_run.wandb_payloads == []


@pytest.mark.parametrize(
    "train_run",
    [pytest.param(TrainScenario.large_token_budget, id="large_token_budget")],
    indirect=True,
)
def test_train_logs_nonzero_loss_under_large_token_budget(
    train_run: TrainRun,
) -> None:
    """train/total_loss logged to W&B is non-zero under a large token budget.

    The budget is well above one batch's tokens but below the two batches'
    combined total (see ``TrainScenario``), so an optimizer step still runs.
    A budget above that total would train nothing, because ``train()`` has
    no end-of-epoch flush.

    Args:
        train_run: Recorded run with W&B enabled.
    """
    assert len(train_run.wandb_payloads) == 1
    train_loss = train_run.wandb_payloads[0]["train/total_loss"]
    assert isinstance(train_loss, float)
    assert train_loss > 0.0, "train/total_loss is 0 — no optimizer step ran"


@pytest.mark.parametrize(
    "train_run",
    [pytest.param(TrainScenario.default_budget, id="default_budget")],
    indirect=True,
)
def test_train_optimizer_state_non_zero_after_one_step(
    train_run: TrainRun,
) -> None:
    """Adam exp_avg in saved checkpoint is non-zero after one training step.

    Args:
        train_run: Recorded one-epoch run.
    """
    opt_state = cast(
        dict[int, dict[str, torch.Tensor]],
        train_run.checkpoint["optimizer"]["state"],
    )
    assert len(opt_state) > 0
    first_state = next(iter(opt_state.values()))
    assert _ADAM_EXP_AVG_KEY in first_state
    assert first_state[_ADAM_EXP_AVG_KEY].abs().max().item() > 0


@pytest.mark.parametrize(
    "train_run",
    [pytest.param(TrainScenario.default_budget, id="default_budget")],
    indirect=True,
)
def test_train_scheduler_state_matches_optimizer_step_count(
    train_run: TrainRun,
) -> None:
    """Checkpointed scheduler last_epoch equals the optimizer step count.

    The LR schedule is per optimizer step (AlphaFold 3-style warmup), so
    the scheduler advances once per optimizer step, not once per epoch.
    Adam's per-parameter ``step`` counter records the same number of
    optimizer steps, so the two must agree in the saved checkpoint, and
    both must be positive after a training epoch.

    Args:
        train_run: Recorded one-epoch run.
    """
    opt_state = cast(
        dict[int, dict[str, torch.Tensor]],
        train_run.checkpoint["optimizer"]["state"],
    )
    adam_steps = int(next(iter(opt_state.values()))["step"].item())
    assert adam_steps > 0
    assert train_run.checkpoint["scheduler"]["last_epoch"] == adam_steps


@pytest.mark.parametrize(
    "train_run",
    [pytest.param(TrainScenario.multi_epoch, id="multi_epoch")],
    indirect=True,
)
def test_train_wandb_epoch_increments_across_epochs(
    train_run: TrainRun,
) -> None:
    """W&B logs exactly once per epoch, with the epoch number advancing.

    Args:
        train_run: Recorded two-epoch run with W&B enabled.
    """
    assert [p["epoch"] for p in train_run.wandb_payloads] == [1, 2]


def test_integration_gradient_flow_through_all_submodules(
    featurized_batch: FeaturizedBatch,
    model_params: ModelSetup,
) -> None:
    """take_step(train_mode) back-props nonzero grads to every submodule."""
    _ = model_params.model.train()
    model_params.model.zero_grad()
    _ = take_step(
        batch=featurized_batch,
        model_params=model_params,
        train_mode=True,
    )

    buckets: dict[str, list[torch.Tensor]] = {}
    for name, param in model_params.model.named_parameters():
        if param.grad is not None:
            buckets.setdefault(name.split(".")[0], []).append(param.grad)

    assert buckets, "no parameters have gradients — backward was not called"
    for prefix, grads in buckets.items():
        assert any(
            torch.isfinite(g).all().item() and g.abs().max().item() > 0
            for g in grads
        ), f"submodule '{prefix}' has no finite nonzero gradients"


# ---------------------------------------------------------------------------
# DDP: collective flush decision keeps ranks in lockstep
# ---------------------------------------------------------------------------

LOCKSTEP_SHORT_RES = 8
LOCKSTEP_LONG_RES = 16
LOCKSTEP_BUDGET_PER_RANK = LOCKSTEP_SHORT_RES + LOCKSTEP_LONG_RES
LOCKSTEP_BATCHES_PER_RANK = 2
LOCKSTEP_ATOM_TYPES = 37


@dataclasses.dataclass(frozen=True)
class DDPLockstepRun:
    """Inputs for a CPU gloo DDP run in which each rank trains on its own data.

    Built by the ``lockstep_run`` factory fixture and pickled by
    ``torch.multiprocessing.spawn`` into one process per rank, where
    ``train_one_rank`` consumes it. The world size is the number of entries
    in ``batches_by_rank``.

    Attributes:
        setup: Model, optimizer, scheduler, EMA and config shared by every
            rank before training; each rank deep-copies it.
        batches_by_rank: The training batches each rank iterates over,
            indexed by rank. Every rank holds the same number of batches, as
            ``DistributedSampler`` guarantees in real training.
        store_path: File backing the gloo rendezvous store, so the test does
            not have to reserve a free TCP port.
    """

    setup: ModelSetup
    batches_by_rank: tuple[tuple[FeaturizedBatch, ...], ...]
    store_path: pathlib.Path


def straight_chain_protein(n_res: int) -> Protein:
    """Build a deterministic, human-readable protein of ``n_res`` residues.

    Residue ``i`` sits at ``(3.8 * i, 0, 0)`` angstroms, the typical spacing
    of consecutive alpha carbons, and atom slot ``a`` of every residue is
    offset from it by ``(0.1 * a, 0.3 * sin(a), 0.3 * cos(a))`` so the atoms
    are not coplanar. Residue types cycle through the standard amino acids
    in order, every atom is present, and all residues are on one chain. The
    token count the training loop sees for this protein is ``n_res``, which
    is the only property the lockstep test depends on.

    Args:
        n_res: Number of residues.

    Returns:
        The protein, with every array fully determined by ``n_res``.
    """
    residue_index = np.arange(n_res, dtype=np.float64)
    atom_index = np.arange(LOCKSTEP_ATOM_TYPES, dtype=np.float64)
    residue_origins = np.stack(
        [3.8 * residue_index, np.zeros(n_res), np.zeros(n_res)],
        axis=-1,
    )
    atom_offsets = np.stack(
        [0.1 * atom_index, 0.3 * np.sin(atom_index), 0.3 * np.cos(atom_index)],
        axis=-1,
    )
    return Protein(
        atom_positions=rearrange(residue_origins, "n c -> n 1 c")
        + rearrange(atom_offsets, "a c -> 1 a c"),
        atom_mask=np.ones((n_res, LOCKSTEP_ATOM_TYPES)),
        residue_index=np.arange(n_res, dtype=np.intp),
        aatype=np.arange(n_res, dtype=np.intp) % RESTYPE_NUM_NO_X,
        chain_index=np.zeros(n_res, dtype=np.intp),
        b_factors=np.zeros((n_res, LOCKSTEP_ATOM_TYPES)),
    )


def train_one_rank(
    rank: int,
    run: DDPLockstepRun,
    reports: SimpleQueue[tuple[int, str]],
) -> None:
    """Train for one epoch as one rank of a CPU gloo group and report back.

    This is the per-process entry point handed to
    ``torch.multiprocessing.spawn``. It lives at module level because
    ``spawn`` can only run a picklable callable in the child process, and
    fixtures cannot run there.

    The setup is deep-copied first. ``spawn`` pickles tensors through shared
    memory, so without the copy every rank would train the very same
    parameter storage and the cross-rank weight comparison in the test would
    pass trivially. ``copy.deepcopy`` keeps the optimizer, scheduler and EMA
    bound to the copied model.

    After training, the rank reports its optimizer step count (the
    scheduler steps once per optimizer step) and a SHA-256 digest of its
    parameters, so the test can compare ranks exactly.

    Args:
        rank: This process's rank, supplied by ``spawn``.
        run: The shared run inputs; this rank trains on
            ``run.batches_by_rank[rank]``.
        reports: Queue receiving ``(optimizer_steps, weight_digest)``.
    """
    torch.set_num_threads(2)
    dist.init_process_group(
        backend="gloo",
        init_method=run.store_path.as_uri(),
        rank=rank,
        world_size=len(run.batches_by_rank),
        timeout=timedelta(seconds=60),
    )
    setup = copy.deepcopy(run.setup)
    loader: torch.utils.data.DataLoader[FeaturizedBatch] = (
        torch.utils.data.DataLoader(
            cast(
                torch.utils.data.Dataset[FeaturizedBatch],
                cast(object, run.batches_by_rank[rank]),
            ),
            batch_size=None,
        )
    )
    train(
        best_val_loss=torch.tensor(float("inf")),
        train_loader=loader,
        test_loader=loader,
        model_params=setup,
        log=cast(FilteringBoundLogger, structlog.get_logger()),
    )
    weights = hashlib.sha256()
    for param in setup.model.parameters():
        weights.update(param.detach().numpy().tobytes())
    reports.put((setup.scheduler.last_epoch, weights.hexdigest()))
    dist.destroy_process_group()


@pytest.fixture
def lockstep_setup(model_params: ModelSetup) -> ModelSetup:
    """Provide ``model_params`` with a per-rank budget of short + long tokens.

    ``train()`` divides ``accumulated_token_budget`` by the world size, so
    the global budget is ``LOCKSTEP_BUDGET_PER_RANK`` times the two ranks
    used by ``lockstep_run``. With that per-rank budget of 24 tokens, two
    short 8-residue batches fit in one window (16 <= 24) but two long
    16-residue batches do not (32 > 24). This is a plain data fixture rather
    than a factory fixture because every lockstep scenario needs the same
    budget; the scenarios differ only in which proteins each rank holds.

    Args:
        model_params: Fixture providing the small model, optimizer,
            scheduler, EMA and base TrainConfig.

    Returns:
        A ModelSetup sharing every component of ``model_params`` except the
        TrainConfig, whose token budget is replaced.
    """
    base = model_params.tcfg
    return dataclasses.replace(
        model_params,
        tcfg=base.model_copy(
            update={
                "training": base.training.model_copy(
                    update={
                        "accumulated_token_budget": 2
                        * LOCKSTEP_BUDGET_PER_RANK,
                    },
                ),
            },
        ),
    )


class LockstepScenario(Enum):
    """Per-rank protein lengths and the optimizer steps every rank must take.

    Each member is ``(residues_by_rank, expected_steps)``, read through
    properties. Rank ``r`` trains on two one-protein batches of
    ``residues_by_rank[r]`` residues under a per-rank budget of 24 tokens.
    Before its second batch, a rank holding 16-residue proteins wants to
    flush (16 + 16 > 24) and a rank holding 8-residue proteins does not
    (8 + 8 <= 24). Every scenario ends with one epoch-end flush of the
    buffered micro-batches.

    Members:
        rank_1_over_budget: Only rank 1 wants to flush, so the vote must
            flush rank 0 too: one pre-flush plus the epoch-end flush.
        rank_0_over_budget: Only rank 0 wants to flush; catches a vote that
            listens to only one rank.
        neither_over_budget: No rank wants to flush, so the only step is the
            epoch-end flush of both batches.
        both_over_budget: Both ranks want to flush at the same batch.
    """

    rank_1_over_budget = ((LOCKSTEP_SHORT_RES, LOCKSTEP_LONG_RES), 2)
    rank_0_over_budget = ((LOCKSTEP_LONG_RES, LOCKSTEP_SHORT_RES), 2)
    neither_over_budget = ((LOCKSTEP_SHORT_RES, LOCKSTEP_SHORT_RES), 1)
    both_over_budget = ((LOCKSTEP_LONG_RES, LOCKSTEP_LONG_RES), 2)

    @property
    def residues_by_rank(self) -> tuple[int, ...]:
        """Residue count of every protein on each rank, indexed by rank."""
        return self.value[0]

    @property
    def expected_steps(self) -> int:
        """Optimizer steps every rank must take in the epoch."""
        return self.value[1]


@pytest.fixture
def lockstep_scenario(request: pytest.FixtureRequest) -> LockstepScenario:
    """Expose the indirect scenario to both the factory and the test.

    Args:
        request: Pytest fixture request whose ``.param`` attribute is a
            LockstepScenario member.

    Returns:
        The scenario member for the current test case.
    """
    return cast(LockstepScenario, request.param)


@pytest.fixture
def lockstep_run(
    lockstep_scenario: LockstepScenario,
    lockstep_setup: ModelSetup,
    tmp_path: pathlib.Path,
) -> DDPLockstepRun:
    """Build a DDP run in which each rank holds proteins of its own length.

    Factory fixture: rank ``r`` gets ``LOCKSTEP_BATCHES_PER_RANK``
    one-protein batches of
    ``straight_chain_protein(lockstep_scenario.residues_by_rank[r])``, so
    every rank yields the same number of batches while the ranks' token
    counts differ.

    Args:
        lockstep_scenario: Scenario naming each rank's residue count.
        lockstep_setup: Fixture providing the model setup with the lockstep
            token budget.
        tmp_path: Pytest-provided directory for the rendezvous store and the
            checkpoint that rank 0 writes.

    Returns:
        DDPLockstepRun ready to hand to ``torch.multiprocessing.spawn``.
    """
    return DDPLockstepRun(
        setup=lockstep_setup,
        batches_by_rank=tuple(
            (
                featurize_batch(
                    [straight_chain_protein(n_res)],
                    lockstep_setup.tcfg,
                    lockstep_setup.distogram_template,
                    lockstep_setup.distogram_residue,
                    lockstep_setup.distogram_atom,
                ),
            )
            * LOCKSTEP_BATCHES_PER_RANK
            for n_res in lockstep_scenario.residues_by_rank
        ),
        store_path=tmp_path / "gloo_store",
    )


@pytest.mark.parametrize(
    "lockstep_scenario",
    list(LockstepScenario),
    ids=[case.name for case in LockstepScenario],
    indirect=True,
)
def test_train_ddp_keeps_ranks_in_lockstep(
    lockstep_scenario: LockstepScenario,
    lockstep_run: DDPLockstepRun,
) -> None:
    """Ranks with different token counts take the same steps and match.

    Regression test for an end-of-epoch DDP deadlock. Each rank used to
    decide on its own when its next batch would overflow the token budget,
    so a rank holding long proteins took more optimizer steps than a rank
    holding short ones. Each extra step's gradient all-reduce had no
    partner, and the job hung until the collective timeout killed it. The
    flush decision is now a vote: if any rank's next batch would overflow,
    every rank flushes. Leftover micro-batches are flushed at epoch end
    instead of being dropped. ``LockstepScenario`` covers the vote being
    won by rank 1 alone, by rank 0 alone, by nobody, and by both ranks.

    Arrange: ``lockstep_run`` gives each rank two one-protein batches of the
    scenario's residue count, under a per-rank budget of 24 tokens.
    Act: ``torch.multiprocessing.spawn`` runs ``train_one_rank`` in two real
    OS processes forming a CPU gloo group; each trains one epoch and reports
    its optimizer step count and a digest of its weights.
    Assert: every rank reports the scenario's step count, and all ranks'
    weight digests are identical. A deadlock makes gloo abort the stuck
    collective after 60 seconds, and ``spawn`` then raises, so the test
    fails instead of hanging.

    Args:
        lockstep_scenario: Scenario naming the expected step count.
        lockstep_run: Factory-fixture-built run for this scenario.
    """
    world_size = len(lockstep_run.batches_by_rank)
    reports: SimpleQueue[tuple[int, str]] = multiprocessing.get_context(
        "spawn",
    ).SimpleQueue()

    _ = spawn(train_one_rank, args=(lockstep_run, reports), nprocs=world_size)

    reported = {reports.get() for _ in range(world_size)}
    assert {steps for steps, _ in reported} == {
        lockstep_scenario.expected_steps,
    }
    assert len({digest for _, digest in reported}) == 1
