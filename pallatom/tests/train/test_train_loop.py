"""Tests for the training loop.

Covers to_protein_batch_dynamic, wavg helpers, take_step, process_accum_window,
component_grad_norms, optimizer_step, evaluate, save/load checkpoint,
log_epoch, and the end-to-end train() function including gradient
accumulation and W&B logging.
"""

import dataclasses
import json
import pathlib
from typing import cast

import numpy as np
import pytest
import structlog
import torch
import torch.nn as nn
from architecture.main_trunk import MainTrunk
from einops import reduce
from helpers.atom_utils import RESTYPE_NUM, RESTYPE_NUM_NO_X, Protein
from helpers.data import make_bucketed_data_loaders
from helpers.featurize import Distogram
from helpers.useful_objects import (
    ComponentNorms,
    EpochMetrics,
    LossMetrics,
    ModelSetup,
    ThroughputStatistics,
    TrainArgs,
    manual_seed,
)
from pydantic import ValidationError
from structlog.typing import FilteringBoundLogger
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from train.train_config import (
    AtomDistogramParams,
    CheckpointParams,
    LoggingParams,
    ModelParams,
    NoiseScheduleParams,
    ResidueDistogramParams,
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
    {"model", "optimizer", "scheduler", "best_val_loss"},
)
_ADAM_EXP_AVG_KEY = "exp_avg"


def _make_model_params(
    trunk: MainTrunk | DDP,
    train_cfg: TrainConfig,
    dgram_res: Distogram,
    dgram_atom: Distogram,
) -> ModelSetup:
    """Construct a ModelSetup with Adam optimizer, CosineAnnealingLR scheduler.

    Args:
        trunk: The MainTrunk to wrap.
        train_cfg: Training configuration.
        dgram_res: Residue-level distogram module.
        dgram_atom: Atom-level distogram module.

    Returns:
        Fully initialised ModelSetup.
    """
    optimizer = Adam(trunk.parameters(), lr=train_cfg.training.lr)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=train_cfg.training.num_epochs,
        eta_min=train_cfg.training.lr * 0.01,
    )
    return ModelSetup(
        model=trunk,
        tcfg=train_cfg,
        distogram_res=dgram_res,
        distogram_atom=dgram_atom,
        device=torch.device("cpu"),
        optimizer=optimizer,
        scheduler=scheduler,
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
def distogram_res() -> Distogram:
    """Provide a residue-level Distogram in eval mode.

    Returns:
        Distogram configured for residue Cβ distances, set to eval mode.
    """
    return Distogram(
        n_bins=_N_BINS,
        min_dist=3.25,
        max_dist=50.75,
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
        res_distogram_params=ResidueDistogramParams(
            n_bins=_N_BINS,
            min_dist=3.25,
            max_dist=50.75,
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
    distogram_res: Distogram,
    distogram_atom: Distogram,
) -> ModelSetup:
    """Provide a ModelSetup bundling model, optimizer, scheduler, and config.

    Args:
        model: Fixture providing the MainTrunk model.
        tcfg: Fixture providing the single-epoch TrainConfig.
        distogram_res: Fixture providing the residue-level distogram head.
        distogram_atom: Fixture providing the atom-level distogram head.

    Returns:
        ModelSetup wrapping the model fixture with an Adam optimizer and
        CosineAnnealingLR scheduler configured from tcfg.
    """
    return _make_model_params(model, tcfg, distogram_res, distogram_atom)


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
def loaders(
    tmp_path: pathlib.Path,
    jsonl_path: str,
    splits_path: str,
) -> tuple[
    torch.utils.data.DataLoader[list[Protein]],
    torch.utils.data.DataLoader[Protein],
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
        checkpoint=CheckpointParams(
            checkpoint_path=tmp_path / "loader_best.pt",
            save_every=100,
        ),
        logging=LoggingParams(use_wandb=False),
        train_loader=TrainLoaderConfig(
            token_budget=64,
            n_clusters=1,
            n_proteins_in_shard=5,
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
    protein_batch: list[Protein],
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
        batch=protein_batch,
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
    protein_batch: list[Protein],
    model_params: ModelSetup,
) -> None:
    """take_step(train_mode=True) back-props grads into model parameters."""
    _ = model_params.model.train()
    model_params.model.zero_grad()
    _ = take_step(
        batch=protein_batch,
        model_params=model_params,
        train_mode=True,
    )
    assert any(p.grad is not None for p in model_params.model.parameters())


def test_take_step_grad_scale_halves_gradient_norm(
    protein_batch: list[Protein],
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
        batch=protein_batch,
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
        batch=protein_batch,
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
    protein_batch: list[Protein],
    model_params: ModelSetup,
) -> None:
    """process_accum_window -> finite LossMetrics, ThroughputStatistics."""
    _ = model_params.model.train()
    model_params.model.zero_grad()
    loss_metrics, tput = process_accum_window(
        [protein_batch],
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
    protein_batch: list[Protein],
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
        batch: list[Protein],
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
    acummulated_batch = [protein_batch, protein_batch]
    _ = process_accum_window(acummulated_batch, [1, 3], model_params)

    assert len(captured_scales) == len(acummulated_batch)
    assert abs(captured_scales[0] - 4.0) < TOLERANCE
    assert abs(captured_scales[1] - 4.0 / 3.0) < TOLERANCE


# ---------------------------------------------------------------------------
# component_grad_norms
# ---------------------------------------------------------------------------


def test_component_grad_norms_outputs(
    protein_batch: list[Protein],
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
        batch=protein_batch,
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
    protein_batch: list[Protein],
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
        [protein_batch],
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


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def test_evaluate_outputs(
    loaders: tuple[
        torch.utils.data.DataLoader[list[Protein]],
        torch.utils.data.DataLoader[Protein],
    ],
    model_params: ModelSetup,
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
    loss_metrics, tput = evaluate(test_loader, model_params)
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
        torch.utils.data.DataLoader[list[Protein]],
        torch.utils.data.DataLoader[Protein],
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


def test_train_no_grad_clip_completes(
    model_params: ModelSetup,
    loaders: tuple[
        torch.utils.data.DataLoader[list[Protein]],
        torch.utils.data.DataLoader[Protein],
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
        model_params.distogram_res,
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
        torch.utils.data.DataLoader[list[Protein]],
        torch.utils.data.DataLoader[Protein],
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
        model_params.distogram_res,
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
    loaders: tuple[
        torch.utils.data.DataLoader[list[Protein]],
        torch.utils.data.DataLoader[Protein],
    ],
    log: FilteringBoundLogger,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """Pre-flush fires when the next batch would push tokens over the budget.

    Loader yields 3 batches of 6 tokens each with budget=7:
    - batch1 (buffer empty): added, no pre-flush
    - batch2: 6+6=12 > 7 → pre-flush batch1, then batch2 added
    - batch3: 6+6=12 > 7 → pre-flush batch2, then batch3 added
    - epoch end: batch3 flushed
    Result: three optimizer windows each of size 1.
    """
    train_loader, test_loader = loaders
    window_sizes: list[int] = []
    _real_process = process_accum_window

    def _tracking_process(
        micro_buffer: list[list[Protein]],
        n_proteins_per_batch: list[int],
        setup: ModelSetup,
    ) -> tuple[LossMetrics, ThroughputStatistics]:
        window_sizes.append(len(micro_buffer))
        return _real_process(micro_buffer, n_proteins_per_batch, setup)

    monkeypatch.setattr(
        "train.train_loop.process_accum_window",
        _tracking_process,
    )

    budget: int = 7  # 6+6=12 > 7 forces a pre-flush on every 2nd batch
    tcfg_b = TrainConfig(
        training=TrainingParams(
            num_epochs=1,
            lr=1e-4,
            grad_clip=1.0,
            accumulated_token_budget=budget,
        ),
        model=model_params.tcfg.model,
        checkpoint=CheckpointParams(
            checkpoint_path=tmp_path / "flush.pt",
            save_every=100,
        ),
        logging=LoggingParams(use_wandb=False),
    )
    mp_b = _make_model_params(
        model_params.model,
        tcfg_b,
        model_params.distogram_res,
        model_params.distogram_atom,
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
        torch.utils.data.DataLoader[list[Protein]],
        torch.utils.data.DataLoader[Protein],
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
        model_params.distogram_res,
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
        torch.utils.data.DataLoader[list[Protein]],
        torch.utils.data.DataLoader[Protein],
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
        torch.utils.data.DataLoader[list[Protein]],
        torch.utils.data.DataLoader[Protein],
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
        model_params.distogram_res,
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


_N_RES_BUCKET = 6
_ENTRY_NAMES_BUCKET = ["1aa.A", "2bb.A", "3cc.A", "4dd.A", "5ee.A"]
_TRAIN_NAMES_BUCKET = ["1aa.A", "2bb.A", "3cc.A"]
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
                "seq": "ACDEFG"[:_N_RES_BUCKET],
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
        torch.utils.data.DataLoader[list[Protein]],
        torch.utils.data.DataLoader[Protein],
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
        torch.utils.data.DataLoader[list[Protein]],
        torch.utils.data.DataLoader[Protein],
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
        model_params.distogram_res,
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
        torch.utils.data.DataLoader[list[Protein]],
        torch.utils.data.DataLoader[Protein],
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
        torch.utils.data.DataLoader[list[Protein]],
        torch.utils.data.DataLoader[Protein],
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
        torch.utils.data.DataLoader[list[Protein]],
        torch.utils.data.DataLoader[Protein],
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
        model_params.distogram_res,
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
    protein_batch: list[Protein],
    model_params: ModelSetup,
) -> None:
    """take_step(train_mode) back-props nonzero grads to every submodule."""
    _ = model_params.model.train()
    model_params.model.zero_grad()
    _ = take_step(
        batch=protein_batch,
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
