"""End-to-end tests of a training step on Self-Flow batches.

Featurizes real proteins with Self-Flow on and off, runs ``take_step`` with
a small ``MainTrunk``, and checks the step trains and logs the noise-group
diagnostic.
"""

import dataclasses
import enum

import numpy as np
import pytest
import torch
from architecture.main_trunk import EMA, MainTrunk
from einops import rearrange
from helpers.atom_utils import Protein
from helpers.data import (
    Distogram,
    FeaturizedBatch,
    build_distogram_module,
    featurize_batch,
)
from helpers.useful_objects import LossMetrics, ModelSetup, manual_seed
from torch.optim import Adam
from train.train_config import ModelParams, SelfFlowParams, TrainConfig
from train.train_loop import build_scheduler, take_step


@dataclasses.dataclass(frozen=True)
class StepCase:
    """One training-step configuration and the low-group share it must log.

    Attributes:
        enabled: Whether Self-Flow is on for featurization.
        share_range: Smallest and largest acceptable low-group weight share.
    """

    enabled: bool
    share_range: tuple[float, float]


class SelfFlowStep(enum.Enum):
    """Training steps with Self-Flow off and on.

    Members:
        off: One uniform noise level, so every residue is in the low group
            and both shares are exactly 1.
        on: Seed 1 mixes two levels in each protein, so the low group holds
            some but not all of the loss weight.
    """

    off = StepCase(enabled=False, share_range=(1.0, 1.0))
    on = StepCase(enabled=True, share_range=(0.0, 0.99))


@pytest.fixture
def step_config(self_flow_step_scenario: SelfFlowStep) -> TrainConfig:
    """Small-trunk training config with Self-Flow as the scenario sets it.

    Three decoder units keep the step fast on CPU and are the fewest that
    give Self-Flow distinct student and teacher units. Half the residues
    are masked when Self-Flow is on.

    Args:
        self_flow_step_scenario: Self-Flow on or off; the requesting test
            parametrizes it.

    Returns:
        The training configuration.
    """
    return TrainConfig(
        model=ModelParams(K_unit=3),
        self_flow=SelfFlowParams(
            enabled=self_flow_step_scenario.value.enabled,
            mask_ratio=0.5,
        ),
    )


@pytest.fixture
def template_distogram(step_config: TrainConfig) -> Distogram:
    """Template distogram module for the step config.

    Args:
        step_config: Training configuration.

    Returns:
        The template distogram module.
    """
    return build_distogram_module(step_config.distogram_template).eval()


@pytest.fixture
def residue_distogram(step_config: TrainConfig) -> Distogram:
    """Residue distogram module for the step config.

    Args:
        step_config: Training configuration.

    Returns:
        The residue distogram module.
    """
    return build_distogram_module(step_config.distogram_residue).eval()


@pytest.fixture
def atom_distogram(step_config: TrainConfig) -> Distogram:
    """Atom distogram module for the step config.

    Args:
        step_config: Training configuration.

    Returns:
        The atom distogram module.
    """
    return build_distogram_module(step_config.distogram_atom).eval()


@pytest.fixture
def model_setup(
    step_config: TrainConfig,
    template_distogram: Distogram,
    residue_distogram: Distogram,
    atom_distogram: Distogram,
) -> ModelSetup:
    """Seeded small trunk with its optimizer, schedule and EMA on CPU.

    Args:
        step_config: Training configuration.
        template_distogram: Template distogram module.
        residue_distogram: Residue distogram module.
        atom_distogram: Atom distogram module.

    Returns:
        The bundled training objects.
    """
    _ = manual_seed(0)
    trunk = MainTrunk(
        model_params=step_config.model,
        template_distogram_params=step_config.distogram_template,
        residue_distogram_params=step_config.distogram_residue,
        atom_distogram_params=step_config.distogram_atom,
        noise_params=step_config.noise,
    )
    optimizer = Adam(trunk.parameters(), lr=step_config.training.lr)
    return ModelSetup(
        model=trunk,
        tcfg=step_config,
        distogram_template=template_distogram,
        distogram_residue=residue_distogram,
        distogram_atom=atom_distogram,
        device=torch.device("cpu"),
        optimizer=optimizer,
        scheduler=build_scheduler(optimizer, step_config.training),
        ema=EMA(trunk, decay=step_config.training.ema_decay),
    )


@pytest.fixture
def step_batch(
    step_config: TrainConfig,
    template_distogram: Distogram,
    residue_distogram: Distogram,
    atom_distogram: Distogram,
) -> FeaturizedBatch:
    """Seeded batch of two all-alanine proteins, six and four residues long.

    Args:
        step_config: Training configuration.
        template_distogram: Template distogram module.
        residue_distogram: Residue distogram module.
        atom_distogram: Atom distogram module.

    Returns:
        The featurized batch; it has a teacher view when Self-Flow is on.
    """
    proteins = [
        Protein(
            atom_positions=rearrange(
                0.5 * np.arange(n_res * 37 * 3, dtype=np.float64),
                "(n a d) -> n a d",
                a=37,
                d=3,
            ),
            aatype=np.zeros(n_res, dtype=np.intp),
            atom_mask=np.ones((n_res, 37), dtype=np.float64),
            residue_index=np.arange(n_res, dtype=np.intp),
            chain_index=np.zeros(n_res, dtype=np.intp),
            b_factors=np.zeros((n_res, 37), dtype=np.float64),
        )
        for n_res in (6, 4)
    ]
    _ = manual_seed(1)
    return featurize_batch(
        proteins,
        step_config,
        template_distogram,
        residue_distogram,
        atom_distogram,
    )


@pytest.fixture
def step_metrics(
    step_batch: FeaturizedBatch,
    model_setup: ModelSetup,
) -> LossMetrics:
    """Loss metrics of one training step on ``step_batch``.

    Args:
        step_batch: Batch to train on.
        model_setup: Model and optimizer bundle.

    Returns:
        The step's loss metrics.
    """
    metrics, _ = take_step(
        batch=step_batch,
        model_params=model_setup,
        train_mode=True,
    )
    return metrics


@pytest.mark.parametrize(
    "self_flow_step_scenario",
    list(SelfFlowStep),
    ids=[case.name for case in SelfFlowStep],
)
def test_take_step_logs_noise_group_shares(
    step_metrics: LossMetrics,
    self_flow_step_scenario: SelfFlowStep,
) -> None:
    """A real training step trains and logs the low-noise group's shares.

    Arrange: two proteins featurized with Self-Flow off or on, and a small
    trunk with self-conditioning on (the default).
    Act: one ``take_step`` in train mode, backward pass included.
    Assert: the total loss is finite, and the low group's weight share is
    inside the scenario's range: exactly 1 with one noise level, strictly
    below 1 when two levels are mixed. The loss share is a valid fraction.

    Args:
        step_metrics: Metrics of the step.
        self_flow_step_scenario: Expected share range; parametrized.
    """
    lowest, highest = self_flow_step_scenario.value.share_range
    weight_share = float(step_metrics.low_sigma_weight_share)
    loss_share = float(step_metrics.low_sigma_loss_share)
    assert bool(torch.isfinite(step_metrics.total_loss))
    assert lowest <= weight_share <= highest
    assert 0.0 <= loss_share <= 1.0
