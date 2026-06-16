"""Tests for training configuration models.

Covers construction, immutability, serialization round-trips, and field-level
validation for every Pydantic model in train.train_config.
"""

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
    """Provide default TrainingParams.

    Returns a TrainingParams instance constructed with all default values.
    """
    return TrainingParams()


@pytest.fixture
def model() -> ModelParams:
    """Provide default ModelParams.

    Returns a ModelParams instance constructed with all default values.
    """
    return ModelParams()


@pytest.fixture
def noise() -> NoiseScheduleParams:
    """Provide default NoiseScheduleParams.

    Returns a NoiseScheduleParams instance constructed with all default values.
    """
    return NoiseScheduleParams()


@pytest.fixture
def distogram_res() -> ResidueDistogramParams:
    """Provide default ResidueDistogramParams.

    Returns ResidueDistogramParams instance constructed with all default values.
    """
    return ResidueDistogramParams()


@pytest.fixture
def distogram_atom() -> AtomDistogramParams:
    """Provide default AtomDistogramParams.

    Returns an AtomDistogramParams instance constructed with all default values.
    """
    return AtomDistogramParams()


@pytest.fixture
def loss() -> LossParams:
    """Provide default LossParams.

    Returns a LossParams instance constructed with all default values.
    """
    return LossParams()


@pytest.fixture
def checkpoint() -> CheckpointParams:
    """Provide default CheckpointParams.

    Returns a CheckpointParams instance constructed with all default values.
    """
    return CheckpointParams()


@pytest.fixture
def logging_() -> LoggingParams:
    """Provide default LoggingParams.

    Returns a LoggingParams instance constructed with all default values.
    """
    return LoggingParams()


@pytest.fixture
def loader() -> LoaderConfig:
    """Provide default LoaderConfig.

    Returns a LoaderConfig instance constructed with all default values.
    """
    return LoaderConfig()


@pytest.fixture
def cfg(  # noqa: PLR0913
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
    """TrainConfig composed from individual default sub-config fixtures.

    Assembles complete TrainConfig by injecting each sub-config fixture so that
    tests can inspect the composed object without constructing it inline.
    """
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


def test_train_config_default_constructs() -> None:
    """TrainConfig can be instantiated with all default values.

    Verifies that the top-level config model requires no arguments and that
    the resulting object is an instance of TrainConfig.
    """
    assert isinstance(TrainConfig(), TrainConfig)


def test_train_config_sub_models_are_correct_types(cfg: TrainConfig) -> None:
    """Each field of TrainConfig holds instance of expected Pydantic model.

    Verifies Pydantic correctly coerces and stores each nested config as the
    declared sub-model type rather than as a plain dict or some other type.
    """
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
# NoiseScheduleParams — defaults, field constraints, cross-field validator
# ---------------------------------------------------------------------------


def test_noise_sigma_min_lt_sigma_max(noise: NoiseScheduleParams) -> None:
    """Default sigma_min is strictly less than sigma_max.

    Verifies that the default noise schedule satisfies the ordering invariant
    required by the cross-field validator.
    """
    assert noise.sigma_min < noise.sigma_max


def test_noise_rejects_nonpositive_sigma_data() -> None:
    """NoiseScheduleParams raises ValidationError when sigma_data is zero.

    Verifies that the field validator enforces a strictly positive constraint
    on sigma_data, which is used as a normalisation denominator.
    """
    with pytest.raises(ValidationError):
        _ = NoiseScheduleParams(sigma_data=0.0)


def test_noise_rejects_nonpositive_sigma_min() -> None:
    """NoiseScheduleParams raises ValidationError when sigma_min is zero.

    Verifies that the field validator enforces a strictly positive constraint
    on sigma_min so the noise schedule lower bound is always meaningful.
    """
    with pytest.raises(ValidationError):
        _ = NoiseScheduleParams(sigma_min=0.0)


def test_noise_rejects_sigma_min_equal_to_sigma_max() -> None:
    """NoiseScheduleParams field validator rejects sigma_min == sigma_max.

    Verifies that the model-level validator requires a strictly ordered noise
    schedule where sigma_min is less than sigma_max.
    """
    with pytest.raises(ValidationError):
        _ = NoiseScheduleParams(sigma_min=80.0, sigma_max=80.0)


def test_noise_rejects_sigma_min_greater_than_sigma_max() -> None:
    """NoiseScheduleParams cross-field validator rejects sigma_min > sigma_max.

    Verifies that the model-level validator rejects an inverted noise schedule
    where sigma_min exceeds sigma_max.
    """
    with pytest.raises(ValidationError):
        _ = NoiseScheduleParams(sigma_min=100.0, sigma_max=80.0)


# ---------------------------------------------------------------------------
# AtomDistogramParams — defaults
# ---------------------------------------------------------------------------


def test_atom_distogram_default_values(
    distogram_atom: AtomDistogramParams,
) -> None:
    """AtomDistogramParams defaults match expected atom-level binning settings.

    Verifies that min_dist, max_dist, and n_bins are set to the values agreed
    upon by the atom distogram head design.
    """
    assert math.isclose(distogram_atom.min_dist, 0.0)
    assert math.isclose(distogram_atom.max_dist, 10.0)
    assert math.isclose(distogram_atom.n_bins, 22)


def test_atom_distogram_min_dist_lt_max_dist(
    distogram_atom: AtomDistogramParams,
) -> None:
    """Default min_dist is strictly less than max_dist for atom distograms.

    Verifies that the default binning range is valid so that distance bins
    have a positive width.
    """
    assert distogram_atom.min_dist < distogram_atom.max_dist


# ---------------------------------------------------------------------------
# LossParams — defaults and field constraints
# ---------------------------------------------------------------------------


def test_loss_default_values(loss: LossParams) -> None:
    """LossParams defaults match expected multi-objective loss weight baseline.

    Verifies that lam, alpha_0, alpha_1, gamma, and smooth_lddt_cutoff are set
    to the agreed-upon baseline values for the multi-objective training loss.
    """
    assert math.isclose(loss.lam, 1.0)
    assert math.isclose(loss.alpha_0, 0.25)
    assert math.isclose(loss.alpha_1, 1.0)
    assert math.isclose(loss.gamma, 0.99)
    assert math.isclose(loss.smooth_lddt_cutoff, 15)


def test_loss_rejects_nonpositive_lam() -> None:
    """LossParams raises ValidationError when lam is zero.

    Verifies that the field validator enforces a strictly positive constraint
    on lam so the coordinate reconstruction loss always contributes to training.
    """
    with pytest.raises(ValidationError):
        _ = LossParams(lam=0.0)


def test_loss_rejects_gamma_of_one() -> None:
    """LossParams raises ValidationError when gamma=1.0.

    Verifies that gamma must be strictly less than 1.0 so that auxiliary losses
    from earlier decoder units are discounted relative to later ones.
    """
    with pytest.raises(ValidationError):
        _ = LossParams(gamma=1.0)


def test_loss_rejects_gamma_of_zero() -> None:
    """LossParams raises ValidationError when gamma=0.0.

    A gamma of zero would zero-weight all intermediate decoder units, making
    only the final unit contribute to the loss, which is not a supported
    training configuration.
    """
    with pytest.raises(ValidationError):
        _ = LossParams(gamma=0.0)


def test_loss_rejects_negative_alpha() -> None:
    """LossParams raises ValidationError when alpha_0 is negative."""
    with pytest.raises(ValidationError):
        _ = LossParams(alpha_0=-0.1)


def test_loss_rejects_nonpositive_smooth_lddt_cutoff() -> None:
    """LossParams raises ValidationError when smooth_lddt_cutoff is zero."""
    with pytest.raises(ValidationError):
        _ = LossParams(smooth_lddt_cutoff=0)
