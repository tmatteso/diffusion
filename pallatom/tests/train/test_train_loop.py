"""Tests for the training loop.

Covers to_protein_batch_dynamic, wavg helpers, take_step, process_accum_window,
component_grad_norms, optimizer_step, evaluate, save/load checkpoint,
log_epoch, and the end-to-end train() function including gradient
accumulation and W&B logging.
"""

import contextlib
import dataclasses
import json
import pathlib
from collections.abc import Generator
from typing import cast

import numpy as np
import pytest
import structlog
import torch
import torch.distributed as dist
import torch.nn as nn
from architecture.main_trunk import EMA, MainTrunk
from einops import reduce
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
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
    """Construct a ModelSetup with Adam optimizer, StepLR scheduler.

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
    scheduler = StepLR(
        optimizer,
        step_size=train_cfg.training.lr_decay_steps,
        gamma=train_cfg.training.lr_decay_factor,
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
    return Distogram(
        n_bins=_N_BINS,
        min_dist=3.25,
        max_dist=50.75,
        overflow_bin=False,
    ).eval()


@pytest.fixture
def distogram_residue() -> Distogram:
    """Provide a residue-level Distogram in eval mode.

    Returns:
        Distogram configured for residue Cβ distances, set to eval mode.
    """
    return Distogram(
        n_bins=_N_BINS,
        min_dist=2.0,
        max_dist=22.0,
        overflow_bin=False,
    ).eval()


@pytest.fixture
def distogram_atom() -> Distogram:
    """Provide an atom-level Distogram in eval mode.

    Returns:
        Distogram configured for short-range atom distances, set to eval mode.
    """
    return Distogram(
        n_bins=_N_ATOM_BINS,
        min_dist=0.0,
        max_dist=10.0,
        overflow_bin=False,
    ).eval()


@pytest.fixture
def model() -> MainTrunk:
    """Provide a small MainTrunk instance for training loop tests.

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


@pytest.fixture
def tcfg(tmp_path: pathlib.Path) -> TrainConfig:
    """Provide a single-epoch TrainConfig with a temporary checkpoint path.

    Args:
        tmp_path: Pytest-provided temporary directory for checkpoint files.

    Returns:
        TrainConfig configured for one epoch with Adam learning rate and
        W&B disabled, writing checkpoints under pytest's tmp_path.
    """
    return TrainConfig(
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
        checkpoint=CheckpointParams(
            checkpoint_path=tmp_path / "best.pt",
            save_every=100,
        ),
        logging=LoggingParams(use_wandb=False),
    )


@pytest.fixture
def tcfg3(model_params: ModelSetup, tmp_path: pathlib.Path) -> TrainConfig:
    """Three-epoch TrainConfig sharing model params from model_params fixture.

    Args:
        model_params: Fixture providing bundled model, optimizer, and config.
        tmp_path: Pytest-provided temporary directory for checkpoint files.

    Returns:
        TrainConfig identical to tcfg except num_epochs=3, used for multi-epoch
        set_epoch and W&B epoch-increment tests.
    """
    return TrainConfig(
        training=TrainingParams(
            num_epochs=3,
            lr=LEARNING_RATE,
            grad_clip=1.0,
            accumulated_token_budget=_BATCH_TOKENS,
        ),
        model=model_params.tcfg.model,
        checkpoint=CheckpointParams(
            checkpoint_path=tmp_path / "best.pt",
            save_every=100,
        ),
        logging=LoggingParams(use_wandb=False),
    )


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
        StepLR scheduler configured from tcfg.
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
    """Provide a ShardDataLoader built by make_bucketed_data_loaders.

    Args:
        tmp_path: Pytest-provided temporary directory for shards and config.
        jsonl_path: Fixture providing path to synthetic proteins JSONL file.
        splits_path: Fixture providing path to the train/val/test splits JSON.

    Returns:
        ShardDataLoader streaming synthetic test proteins for one epoch.
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
            checkpoint_path=tmp_path / "loader_best.pt",
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
        dataset_jsonl=pathlib.Path(jsonl_path),
        shard_dir=tmp_path / "shards",
        keys_for_splits_json=pathlib.Path(splits_path),
        config=tmp_path / "config.json",
        structlog_jsonl=tmp_path / "structlog.jsonl",
        ddp=False,
        debug_run=False,
    )
    train_loader, test_loader, _ = make_bucketed_data_loaders(
        cfg=cfg,
        extra_train_args=args,
    )
    return train_loader, test_loader


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
    """save->load restores Adam moments and StepLR step count exactly.

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


def test_train_updates_model_parameters(
    model_params: ModelSetup,
    loaders: tuple[
        torch.utils.data.DataLoader[FeaturizedBatch],
        torch.utils.data.DataLoader[FeaturizedBatch],
    ],
    log: FilteringBoundLogger,
) -> None:
    """Train modifies at least one model parameter via gradient descent."""
    train_loader, test_loader = loaders
    params_before = [
        p.clone().detach() for p in model_params.model.parameters()
    ]
    train(
        best_val_loss=torch.tensor(float("inf")),
        train_loader=train_loader,
        test_loader=test_loader,
        model_params=model_params,
        log=log,
    )
    assert any(
        not torch.equal(b, a)
        for b, a in zip(
            params_before,
            model_params.model.parameters(),
            strict=False,
        )
    )


def test_train_calls_all_reduce_mean_on_all_epoch_metrics(
    model_params: ModelSetup,
    loaders: tuple[
        torch.utils.data.DataLoader[FeaturizedBatch],
        torch.utils.data.DataLoader[FeaturizedBatch],
    ],
    log: FilteringBoundLogger,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """train() calls all_reduce_mean() on every metric feeding EpochMetrics.

    Arrange: wrap TensorAccumulatorMixin.all_reduce_mean with a spy that
    records every instance it's called on before delegating to the real
    implementation (which no-ops here, since no distributed process group
    is active in this test -- the reduction arithmetic itself is proven
    separately, with real multi-rank processes, by
    test_all_reduce_mean_averages_across_real_ranks in
    test_useful_objects.py). This test only proves train() invokes the
    reduction on the right objects, not the arithmetic.
    Act: run train() for one epoch.
    Assert: the spy was invoked on exactly four metric instances: two
    LossMetrics (train and val), one ThroughputStatistics, and one
    ComponentNorms -- matching every metric field of EpochMetrics.
    """
    train_loader, test_loader = loaders
    calls: list[TensorAccumulatorMixin] = []
    original = TensorAccumulatorMixin.all_reduce_mean

    def _spy(self: TensorAccumulatorMixin) -> None:
        calls.append(self)
        original(self)

    monkeypatch.setattr(TensorAccumulatorMixin, "all_reduce_mean", _spy)

    train(
        best_val_loss=torch.tensor(float("inf")),
        train_loader=train_loader,
        test_loader=test_loader,
        model_params=model_params,
        log=log,
    )

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


def test_train_no_grad_clip_completes(
    model_params: ModelSetup,
    loaders: tuple[
        torch.utils.data.DataLoader[FeaturizedBatch],
        torch.utils.data.DataLoader[FeaturizedBatch],
    ],
    log: FilteringBoundLogger,
    tmp_path: pathlib.Path,
) -> None:
    """Train completes without error when grad_clip is None."""
    train_loader, test_loader = loaders
    tcfg_nc = TrainConfig(
        training=TrainingParams(
            num_epochs=1,
            lr=1e-4,
            grad_clip=None,
            accumulated_token_budget=_BATCH_TOKENS,
        ),
        model=model_params.tcfg.model,
        checkpoint=CheckpointParams(
            checkpoint_path=tmp_path / "no_clip.pt",
            save_every=100,
        ),
        logging=LoggingParams(use_wandb=False),
    )
    mp_nc = _make_model_params(
        model_params.model,
        tcfg_nc,
        model_params.distogram_template,
        model_params.distogram_residue,
        model_params.distogram_atom,
    )
    assert (
        train(
            best_val_loss=torch.tensor(float("inf")),
            train_loader=train_loader,
            test_loader=test_loader,
            model_params=mp_nc,
            log=log,
        )
        is None
    )


def test_train_partial_window_at_epoch_end_updates_params(
    model_params: ModelSetup,
    loaders: tuple[
        torch.utils.data.DataLoader[FeaturizedBatch],
        torch.utils.data.DataLoader[FeaturizedBatch],
    ],
    log: FilteringBoundLogger,
    tmp_path: pathlib.Path,
) -> None:
    """Partial window flushed at epoch end when token budget is not reached.

    With all loader batches fitting under 2*_BATCH_TOKENS, no pre-flush fires,
    but the end-of-epoch flush still triggers one optimizer step so params must
    change.
    """
    train_loader, test_loader = loaders
    tcfg_acc = TrainConfig(
        training=TrainingParams(
            num_epochs=1,
            lr=1e-4,
            grad_clip=1.0,
            accumulated_token_budget=2 * _BATCH_TOKENS,
        ),
        model=model_params.tcfg.model,
        checkpoint=CheckpointParams(
            checkpoint_path=tmp_path / "accum.pt",
            save_every=100,
        ),
        logging=LoggingParams(use_wandb=False),
    )
    mp_acc = _make_model_params(
        model_params.model,
        tcfg_acc,
        model_params.distogram_template,
        model_params.distogram_residue,
        model_params.distogram_atom,
    )
    params_before = [p.clone().detach() for p in mp_acc.model.parameters()]
    train(
        best_val_loss=torch.tensor(float("inf")),
        train_loader=train_loader,
        test_loader=test_loader,
        model_params=mp_acc,
        log=log,
    )
    assert any(
        not torch.equal(b, a)
        for b, a in zip(params_before, mp_acc.model.parameters(), strict=False)
    )


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
    - epoch end: batch4 remains in buffer (no end-of-epoch flush)
    Result: three optimizer windows each of size 1.
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
    assert window_sizes == [1, 1, 1]


def test_train_wandb_called_once_per_epoch_when_enabled(
    model_params: ModelSetup,
    loaders: tuple[
        torch.utils.data.DataLoader[FeaturizedBatch],
        torch.utils.data.DataLoader[FeaturizedBatch],
    ],
    log: FilteringBoundLogger,
    wandb_payloads: list[dict[str, object]],
    tmp_path: pathlib.Path,
) -> None:
    """Train calls wandb.log exactly once per epoch when use_wandb=True."""
    train_loader, test_loader = loaders
    tcfg_w = TrainConfig(
        training=TrainingParams(
            num_epochs=1,
            lr=1e-4,
            grad_clip=1.0,
            accumulated_token_budget=_BATCH_TOKENS,
        ),
        model=model_params.tcfg.model,
        checkpoint=CheckpointParams(
            checkpoint_path=tmp_path / "wandb.pt",
            save_every=100,
        ),
        logging=LoggingParams(use_wandb=True),
    )
    mp_w = _make_model_params(
        model_params.model,
        tcfg_w,
        model_params.distogram_template,
        model_params.distogram_residue,
        model_params.distogram_atom,
    )
    train(
        best_val_loss=torch.tensor(float("inf")),
        train_loader=train_loader,
        test_loader=test_loader,
        model_params=mp_w,
        log=log,
    )
    assert len(wandb_payloads) == 1


def test_train_wandb_not_called_when_disabled(
    model_params: ModelSetup,
    loaders: tuple[
        torch.utils.data.DataLoader[FeaturizedBatch],
        torch.utils.data.DataLoader[FeaturizedBatch],
    ],
    log: FilteringBoundLogger,
    wandb_call_counter: list[object],
) -> None:
    """Train does not call wandb.log when use_wandb=False."""
    train_loader, test_loader = loaders
    train(
        best_val_loss=torch.tensor(float("inf")),
        train_loader=train_loader,
        test_loader=test_loader,
        model_params=model_params,
        log=log,
    )
    assert len(wandb_call_counter) == 0


def test_train_metrics_nonzero_when_budget_exceeds_total_tokens(
    model_params: ModelSetup,
    loaders: tuple[
        torch.utils.data.DataLoader[FeaturizedBatch],
        torch.utils.data.DataLoader[FeaturizedBatch],
    ],
    log: FilteringBoundLogger,
    wandb_payloads: list[dict[str, object]],
    tmp_path: pathlib.Path,
) -> None:
    """train/total_loss logged to W&B non-zero when dataset fits under budget.

    Regression: if the end-of-epoch flush is dropped, no training occurs and
        loss = 0.
    """
    train_loader, test_loader = loaders
    tcfg_large = TrainConfig(
        training=TrainingParams(
            num_epochs=1,
            lr=1e-4,
            grad_clip=1.0,
            accumulated_token_budget=10 * _BATCH_TOKENS,
        ),
        model=model_params.tcfg.model,
        checkpoint=CheckpointParams(
            checkpoint_path=tmp_path / "large.pt",
            save_every=100,
        ),
        logging=LoggingParams(use_wandb=True),
    )
    mp_large = _make_model_params(
        model_params.model,
        tcfg_large,
        model_params.distogram_template,
        model_params.distogram_residue,
        model_params.distogram_atom,
    )
    train(
        best_val_loss=torch.tensor(float("inf")),
        train_loader=train_loader,
        test_loader=test_loader,
        model_params=mp_large,
        log=log,
    )
    assert len(wandb_payloads) == 1
    train_loss = wandb_payloads[0]["train/total_loss"]
    assert isinstance(train_loss, float)
    assert (
        train_loss > 0.0
    ), "train/total_loss is 0 — end-of-epoch flush was dropped"


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


_N_RES_BUCKET = 42
_ENTRY_NAMES_BUCKET = ["1aa.A", "2bb.A", "3cc.A", "4dd.A", "5ee.A", "6ff.A"]
_TRAIN_NAMES_BUCKET = ["1aa.A", "2bb.A", "3cc.A", "6ff.A"]
_VAL_NAMES_BUCKET = ["4dd.A"]
_TEST_NAMES_BUCKET = ["5ee.A"]


@pytest.fixture
def jsonl_path(tmp_path: pathlib.Path) -> str:
    """Write JSONL file with synthetic protein entries and return its path."""
    path = tmp_path / "proteins.jsonl"
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


@pytest.fixture
def splits_path(tmp_path: pathlib.Path) -> str:
    """Write splits JSON with train/val/test name lists and return its path."""
    path = tmp_path / "splits.json"
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


def test_train_one_epoch_with_bucketed_loader(
    model_params: ModelSetup,
    loaders: tuple[
        torch.utils.data.DataLoader[FeaturizedBatch],
        torch.utils.data.DataLoader[FeaturizedBatch],
    ],
    log: FilteringBoundLogger,
) -> None:
    """train() completes one epoch with a real ShardDataLoader."""
    train_loader, test_loader = loaders
    result = train(
        best_val_loss=torch.tensor(float("inf")),
        train_loader=train_loader,
        test_loader=test_loader,
        model_params=model_params,
        log=log,
    )
    assert result is None


def test_train_accum_full_window_updates_params(
    model_params: ModelSetup,
    loaders: tuple[
        torch.utils.data.DataLoader[FeaturizedBatch],
        torch.utils.data.DataLoader[FeaturizedBatch],
    ],
    log: FilteringBoundLogger,
    tmp_path: pathlib.Path,
) -> None:
    """Train loop completes one accumulation window under a token budget.

    With an accumulated token budget set to twice the per-batch token count
    and all loader batches fitting under that budget, the training loop
    should flush at epoch end and update model parameters.
    """
    train_loader, test_loader = loaders
    _ = manual_seed(0)
    tcfg_acc = TrainConfig(
        training=TrainingParams(
            num_epochs=1,
            lr=1e-4,
            grad_clip=1.0,
            accumulated_token_budget=2 * _BATCH_TOKENS,
        ),
        model=model_params.tcfg.model,
        checkpoint=CheckpointParams(
            checkpoint_path=tmp_path / "full_accum.pt",
            save_every=100,
        ),
        logging=LoggingParams(use_wandb=False),
    )
    mp_acc = _make_model_params(
        model_params.model,
        tcfg_acc,
        model_params.distogram_template,
        model_params.distogram_residue,
        model_params.distogram_atom,
    )
    params_before = [p.clone().detach() for p in mp_acc.model.parameters()]
    train(
        best_val_loss=torch.tensor(float("inf")),
        train_loader=train_loader,
        test_loader=test_loader,
        model_params=mp_acc,
        log=log,
    )
    assert any(
        not torch.equal(b, a)
        for b, a in zip(params_before, mp_acc.model.parameters(), strict=False)
    )


def test_train_optimizer_state_non_zero_after_one_step(
    model_params: ModelSetup,
    loaders: tuple[
        torch.utils.data.DataLoader[FeaturizedBatch],
        torch.utils.data.DataLoader[FeaturizedBatch],
    ],
    log: FilteringBoundLogger,
) -> None:
    """Adam exp_avg in saved checkpoint is non-zero after one training step."""
    train_loader, test_loader = loaders
    train(
        best_val_loss=torch.tensor(float("inf")),
        train_loader=train_loader,
        test_loader=test_loader,
        model_params=model_params,
        log=log,
    )
    ckpt = cast(
        dict[str, object],
        torch.load(
            model_params.tcfg.checkpoint.checkpoint_path,
            weights_only=True,
        ),
    )
    optimizer_ckpt = cast(dict[str, object], ckpt["optimizer"])
    opt_state = cast(
        dict[int, dict[str, torch.Tensor]],
        optimizer_ckpt["state"],
    )
    assert len(opt_state) > 0
    first_state = next(iter(opt_state.values()))
    assert _ADAM_EXP_AVG_KEY in first_state
    assert first_state[_ADAM_EXP_AVG_KEY].abs().max().item() > 0


def test_train_scheduler_state_reflects_completed_epoch(
    model_params: ModelSetup,
    loaders: tuple[
        torch.utils.data.DataLoader[FeaturizedBatch],
        torch.utils.data.DataLoader[FeaturizedBatch],
    ],
    log: FilteringBoundLogger,
) -> None:
    """Scheduler last_epoch in checkpoint equals number of epochs completed."""
    train_loader, test_loader = loaders
    train(
        best_val_loss=torch.tensor(float("inf")),
        train_loader=train_loader,
        test_loader=test_loader,
        model_params=model_params,
        log=log,
    )
    ckpt = cast(
        dict[str, dict[str, object]],
        torch.load(
            model_params.tcfg.checkpoint.checkpoint_path,
            weights_only=True,
        ),
    )
    assert (
        ckpt["scheduler"]["last_epoch"] == model_params.tcfg.training.num_epochs
    )


def test_train_wandb_epoch_increments_across_epochs(
    model_params: ModelSetup,
    loaders: tuple[
        torch.utils.data.DataLoader[FeaturizedBatch],
        torch.utils.data.DataLoader[FeaturizedBatch],
    ],
    tcfg3: TrainConfig,
    log: FilteringBoundLogger,
    wandb_payloads: list[dict[str, object]],
) -> None:
    """W&B epoch values logged across a 3-epoch run are [1, 2, 3]."""
    train_loader, test_loader = loaders
    mp3 = _make_model_params(
        model_params.model,
        _tcfg_with_wandb(base=tcfg3, use_wandb=True),
        model_params.distogram_template,
        model_params.distogram_residue,
        model_params.distogram_atom,
    )
    train(
        best_val_loss=torch.tensor(float("inf")),
        train_loader=train_loader,
        test_loader=test_loader,
        model_params=mp3,
        log=log,
    )
    assert [p["epoch"] for p in wandb_payloads] == [1, 2, 3]


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
