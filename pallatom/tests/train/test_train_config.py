"""Tests for training configuration models."""

import math

import pytest
from pydantic import ValidationError
from train.train_config import (
    AtomDistogramParams,
    CheckpointParams,
    LoaderConfig,
    LoggingParams,
    LossParams,
    ModelParams,
    NoiseScheduleParams,
    ResidueDistogramParams,
    TrainConfig,
    TrainingParams,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def training() -> TrainingParams:
    """Provide default TrainingParams."""
    return TrainingParams()


@pytest.fixture
def model() -> ModelParams:
    """Provide default ModelParams."""
    return ModelParams()


@pytest.fixture
def noise() -> NoiseScheduleParams:
    """Provide default NoiseScheduleParams."""
    return NoiseScheduleParams()


@pytest.fixture
def distogram_res() -> ResidueDistogramParams:
    """Provide default ResidueDistogramParams."""
    return ResidueDistogramParams()


@pytest.fixture
def distogram_atom() -> AtomDistogramParams:
    """Provide default AtomDistogramParams."""
    return AtomDistogramParams()


@pytest.fixture
def loss() -> LossParams:
    """Provide default LossParams."""
    return LossParams()


@pytest.fixture
def checkpoint() -> CheckpointParams:
    """Provide default CheckpointParams."""
    return CheckpointParams()


@pytest.fixture
def logging_() -> LoggingParams:
    """Provide default LoggingParams."""
    return LoggingParams()


@pytest.fixture
def loader() -> LoaderConfig:
    """Provide default LoaderConfig."""
    return LoaderConfig()


@pytest.fixture
def cfg(
    training: TrainingParams,
    model: ModelParams,
    noise: NoiseScheduleParams,
    distogram_res: ResidueDistogramParams,
    distogram_atom: AtomDistogramParams,
    loss: LossParams,
    checkpoint: CheckpointParams,
    logging_: LoggingParams,
    loader: LoaderConfig,
) -> TrainConfig:
    """Provide a TrainConfig composed from individual default sub-config fixtures."""
    return TrainConfig(
        training=training,
        model=model,
        noise=noise,
        distogram_res=distogram_res,
        distogram_atom=distogram_atom,
        loss=loss,
        checkpoint=checkpoint,
        logging=logging_,
        loader=loader,
    )


# ---------------------------------------------------------------------------
# TrainConfig — construction and sub-model types
# ---------------------------------------------------------------------------


def test_train_config_default_constructs():
    """TrainConfig can be instantiated with all default values."""
    assert isinstance(TrainConfig(), TrainConfig)


def test_train_config_sub_models_are_correct_types(cfg: TrainConfig):
    """Each sub-config field of TrainConfig holds an instance of its expected Pydantic model."""
    assert isinstance(cfg.training, TrainingParams)
    assert isinstance(cfg.model, ModelParams)
    assert isinstance(cfg.noise, NoiseScheduleParams)
    assert isinstance(cfg.distogram_res, ResidueDistogramParams)
    assert isinstance(cfg.distogram_atom, AtomDistogramParams)
    assert isinstance(cfg.loss, LossParams)
    assert isinstance(cfg.checkpoint, CheckpointParams)
    assert isinstance(cfg.logging, LoggingParams)
    assert isinstance(cfg.loader, LoaderConfig)


# ---------------------------------------------------------------------------
# TrainConfig — immutability
# ---------------------------------------------------------------------------


def test_train_config_is_frozen(cfg: TrainConfig):
    """TrainConfig raises ValidationError when a top-level field is reassigned."""
    with pytest.raises(ValidationError):
        cfg.training = TrainingParams()


def test_training_params_is_frozen(training: TrainingParams):
    """TrainingParams raises ValidationError when any field is mutated after construction."""
    with pytest.raises(ValidationError):
        training.lr = 1e-2


def test_model_params_is_frozen(model: ModelParams):
    """ModelParams raises ValidationError when any field is mutated after construction."""
    with pytest.raises(ValidationError):
        model.c_res = 512


# ---------------------------------------------------------------------------
# TrainConfig — serialization round-trip
# ---------------------------------------------------------------------------


def test_train_config_round_trips_through_dict(cfg: TrainConfig):
    """TrainConfig survives a model_dump / model_validate round-trip unchanged."""
    assert TrainConfig.model_validate(cfg.model_dump()) == cfg


# ---------------------------------------------------------------------------
# TrainingParams — defaults and field constraints
# ---------------------------------------------------------------------------


def test_training_rejects_nonpositive_grad_clip():
    """TrainingParams raises ValidationError when grad_clip is zero or negative."""
    with pytest.raises(ValidationError):
        TrainingParams(grad_clip=0.0)


def test_training_accepts_none_grad_clip():
    """TrainingParams accepts grad_clip=None to disable gradient clipping."""
    assert TrainingParams(grad_clip=None).grad_clip is None


# ---------------------------------------------------------------------------
# ModelParams — defaults and field constraints
# ---------------------------------------------------------------------------


def test_model_rejects_zero_c_res():
    """ModelParams raises ValidationError when c_res is zero (embedding width must be positive)."""
    with pytest.raises(ValidationError):
        ModelParams(c_res=0)


def test_model_rejects_zero_k_unit():
    """ModelParams raises ValidationError when K_unit zero (at least one decoder unit required)."""
    with pytest.raises(ValidationError):
        ModelParams(K_unit=0)


# ---------------------------------------------------------------------------
# NoiseScheduleParams — defaults, field constraints, cross-field validator
# ---------------------------------------------------------------------------


def test_noise_sigma_min_lt_sigma_max(noise: NoiseScheduleParams):
    """Default sigma_min is strictly less than sigma_max."""
    assert noise.sigma_min < noise.sigma_max


def test_noise_rejects_nonpositive_sigma_data():
    """NoiseScheduleParams raises ValidationError when sigma_data is zero."""
    with pytest.raises(ValidationError):
        NoiseScheduleParams(sigma_data=0.0)


def test_noise_rejects_nonpositive_sigma_min():
    """NoiseScheduleParams raises ValidationError when sigma_min is zero."""
    with pytest.raises(ValidationError):
        NoiseScheduleParams(sigma_min=0.0)


def test_noise_rejects_sigma_min_equal_to_sigma_max():
    """NoiseScheduleParams cross-field validator rejects sigma_min == sigma_max."""
    with pytest.raises(ValidationError):
        NoiseScheduleParams(sigma_min=80.0, sigma_max=80.0)


def test_noise_rejects_sigma_min_greater_than_sigma_max():
    """NoiseScheduleParams cross-field validator rejects sigma_min > sigma_max."""
    with pytest.raises(ValidationError):
        NoiseScheduleParams(sigma_min=100.0, sigma_max=80.0)


# ---------------------------------------------------------------------------
# AtomDistogramParams — defaults
# ---------------------------------------------------------------------------


def test_atom_distogram_default_values(distogram_atom: AtomDistogramParams):
    """AtomDistogramParams defaults match the expected atom-level binning settings."""
    assert math.isclose(distogram_atom.min_dist, 0.0)
    assert math.isclose(distogram_atom.max_dist, 10.0)
    assert math.isclose(distogram_atom.n_bins, 22)


def test_atom_distogram_min_dist_lt_max_dist(distogram_atom: AtomDistogramParams):
    """Default min_dist is strictly less than max_dist for atom distograms."""
    assert distogram_atom.min_dist < distogram_atom.max_dist


# ---------------------------------------------------------------------------
# LossParams — defaults and field constraints
# ---------------------------------------------------------------------------


def test_loss_default_values(loss: LossParams):
    """LossParams defaults match the expected multi-objective loss weight baseline."""
    assert math.isclose(loss.lam, 1.0)
    assert math.isclose(loss.alpha_0, 0.25)
    assert math.isclose(loss.alpha_1, 1.0)
    assert math.isclose(loss.gamma, 0.99)
    assert math.isclose(loss.smooth_lddt_cutoff, 15)


def test_loss_rejects_nonpositive_lam():
    """LossParams raises ValidationError when lam is zero (coordinate loss must be weighted)."""
    with pytest.raises(ValidationError):
        LossParams(lam=0.0)


def test_loss_rejects_gamma_of_one():
    """LossParams raises ValidationError when gamma=1.0 (no discount between decoder units)."""
    with pytest.raises(ValidationError):
        LossParams(gamma=1.0)


def test_loss_rejects_gamma_of_zero():
    """LossParams raises ValidationError when gamma=0.0 (only last decoder unit contributes)."""
    with pytest.raises(ValidationError):
        LossParams(gamma=0.0)


def test_loss_rejects_negative_alpha():
    """LossParams raises ValidationError when alpha_0 is negative."""
    with pytest.raises(ValidationError):
        LossParams(alpha_0=-0.1)


def test_loss_rejects_nonpositive_smooth_lddt_cutoff():
    """LossParams raises ValidationError when smooth_lddt_cutoff is zero."""
    with pytest.raises(ValidationError):
        LossParams(smooth_lddt_cutoff=0)
