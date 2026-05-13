"""Tests for training configuration models."""

import pytest
from pydantic import ValidationError
from train.train_config import (
    AtomDistogramParams,
    CheckpointParams,
    ConditioningDropoutConfig,
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


def test_train_config_accepts_nested_overrides():
    """TrainConfig forwards keyword overrides into the appropriate sub-config models."""
    cfg = TrainConfig(training=TrainingParams(lr=1e-3, num_epochs=100))
    assert cfg.training.lr == 1e-3
    assert cfg.training.num_epochs == 100


# ---------------------------------------------------------------------------
# TrainConfig — immutability
# ---------------------------------------------------------------------------


def test_train_config_is_frozen(cfg: TrainConfig):
    """TrainConfig raises ValidationError when a top-level field is reassigned."""
    with pytest.raises(ValidationError):
        cfg.training = TrainingParams()  # type: ignore[misc]


def test_training_params_is_frozen(training: TrainingParams):
    """TrainingParams raises ValidationError when any field is mutated after construction."""
    with pytest.raises(ValidationError):
        training.lr = 1e-2  # type: ignore[misc]


def test_model_params_is_frozen(model: ModelParams):
    """ModelParams raises ValidationError when any field is mutated after construction."""
    with pytest.raises(ValidationError):
        model.c_res = 512  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TrainConfig — serialization round-trip
# ---------------------------------------------------------------------------


def test_train_config_model_dump_is_dict(cfg: TrainConfig):
    """TrainConfig.model_dump() returns a plain dict."""
    assert isinstance(cfg.model_dump(), dict)


def test_train_config_round_trips_through_dict(cfg: TrainConfig):
    """TrainConfig survives a model_dump / model_validate round-trip unchanged."""
    assert TrainConfig.model_validate(cfg.model_dump()) == cfg


def test_train_config_round_trips_custom_values():
    """Custom field values are preserved after a model_dump / model_validate round-trip."""
    cfg = TrainConfig(training=TrainingParams(lr=1e-5), loader=LoaderConfig(batch_size=64))
    cfg2 = TrainConfig.model_validate(cfg.model_dump())
    assert cfg2.training.lr == 1e-5
    assert cfg2.loader.batch_size == 64


# ---------------------------------------------------------------------------
# TrainingParams — defaults and field constraints
# ---------------------------------------------------------------------------


def test_training_default_values(training: TrainingParams):
    """TrainingParams defaults match the expected hyper-parameter baseline values."""
    assert training.num_epochs == 50
    assert training.lr == pytest.approx(3e-4)
    assert training.weight_decay == pytest.approx(1e-4)
    assert training.grad_clip == pytest.approx(2.0)


def test_training_rejects_nonpositive_lr():
    """TrainingParams raises ValidationError when lr is zero or negative."""
    with pytest.raises(ValidationError):
        TrainingParams(lr=0.0)


def test_training_rejects_negative_weight_decay():
    """TrainingParams raises ValidationError when weight_decay is negative."""
    with pytest.raises(ValidationError):
        TrainingParams(weight_decay=-0.1)


def test_training_rejects_nonpositive_grad_clip():
    """TrainingParams raises ValidationError when grad_clip is zero or negative."""
    with pytest.raises(ValidationError):
        TrainingParams(grad_clip=0.0)


def test_training_accepts_none_grad_clip():
    """TrainingParams accepts grad_clip=None to disable gradient clipping."""
    assert TrainingParams(grad_clip=None).grad_clip is None


def test_training_accepts_positive_grad_clip():
    """TrainingParams stores a positive grad_clip value without error."""
    assert TrainingParams(grad_clip=10.0).grad_clip == 10.0


# ---------------------------------------------------------------------------
# ModelParams — defaults and field constraints
# ---------------------------------------------------------------------------


def test_model_default_values(model: ModelParams):
    """ModelParams defaults match the expected compact model architecture dimensions."""
    assert model.f_ref_dim == 35
    assert model.n_bins == 39
    assert model.c_atom == 16
    assert model.c_pair == 16
    assert model.c_res == 32
    assert model.c_atompair == 2
    assert model.K_unit == 3


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


def test_noise_default_values(noise: NoiseScheduleParams):
    """NoiseScheduleParams defaults match the pre-computed EDM noise schedule values."""
    assert noise.sigma_data == pytest.approx(19.2368)
    assert noise.sigma_max == pytest.approx(42.3689)
    assert noise.sigma_min == pytest.approx(3.807123)
    assert noise.P_mean == pytest.approx(2.5416)
    assert noise.P_std == pytest.approx(1.2048)


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
# ResidueDistogramParams — defaults, field constraints, cross-field validator
# ---------------------------------------------------------------------------


def test_residue_distogram_default_values(distogram_res: ResidueDistogramParams):
    """ResidueDistogramParams defaults match the expected Cβ distogram binning settings."""
    assert distogram_res.min_dist == pytest.approx(3.25)
    assert distogram_res.max_dist == pytest.approx(50.75)
    assert distogram_res.n_bins == 38
    assert distogram_res.tok_emb_dim == 32


def test_residue_distogram_min_dist_lt_max_dist(distogram_res: ResidueDistogramParams):
    """Default min_dist is strictly less than max_dist for residue distograms."""
    assert distogram_res.min_dist < distogram_res.max_dist


def test_residue_distogram_rejects_zero_n_bins():
    """ResidueDistogramParams raises ValidationError when n_bins is zero."""
    with pytest.raises(ValidationError):
        ResidueDistogramParams(n_bins=0)


def test_residue_distogram_rejects_min_equal_to_max():
    """ResidueDistogramParams cross-field validator rejects min_dist == max_dist."""
    with pytest.raises(ValidationError):
        ResidueDistogramParams(min_dist=10.0, max_dist=10.0)


def test_residue_distogram_rejects_min_greater_than_max():
    """ResidueDistogramParams cross-field validator rejects min_dist > max_dist."""
    with pytest.raises(ValidationError):
        ResidueDistogramParams(min_dist=20.0, max_dist=10.0)


def test_residue_distogram_accepts_zero_min_dist():
    """ResidueDistogramParams accepts min_dist=0.0 for distance ranges starting at the origin."""
    assert ResidueDistogramParams(min_dist=0.0, max_dist=10.0).min_dist == 0.0


# ---------------------------------------------------------------------------
# AtomDistogramParams — defaults
# ---------------------------------------------------------------------------


def test_atom_distogram_default_values(distogram_atom: AtomDistogramParams):
    """AtomDistogramParams defaults match the expected atom-level binning settings."""
    assert distogram_atom.min_dist == pytest.approx(0.0)
    assert distogram_atom.max_dist == pytest.approx(10.0)
    assert distogram_atom.n_bins == 22


def test_atom_distogram_min_dist_lt_max_dist(distogram_atom: AtomDistogramParams):
    """Default min_dist is strictly less than max_dist for atom distograms."""
    assert distogram_atom.min_dist < distogram_atom.max_dist


# ---------------------------------------------------------------------------
# LossParams — defaults and field constraints
# ---------------------------------------------------------------------------


def test_loss_default_values(loss: LossParams):
    """LossParams defaults match the expected multi-objective loss weight baseline."""
    assert loss.lam == pytest.approx(1.0)
    assert loss.alpha_0 == pytest.approx(0.25)
    assert loss.alpha_1 == pytest.approx(1.0)
    assert loss.gamma == pytest.approx(0.99)
    assert loss.smooth_lddt_cutoff == 15


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


# ---------------------------------------------------------------------------
# CheckpointParams — defaults and field constraints
# ---------------------------------------------------------------------------


def test_checkpoint_default_values(checkpoint: CheckpointParams):
    """CheckpointParams defaults match the expected path and save-frequency settings."""
    assert checkpoint.checkpoint_path == "pallatom_best_best.pt"
    assert checkpoint.save_every == 1


def test_checkpoint_accepts_save_every_zero():
    """CheckpointParams accepts save_every=0 to disable periodic epoch checkpoints."""
    assert CheckpointParams(save_every=0).save_every == 0


def test_checkpoint_rejects_negative_save_every():
    """CheckpointParams raises ValidationError when save_every is negative."""
    with pytest.raises(ValidationError):
        CheckpointParams(save_every=-1)


# ---------------------------------------------------------------------------
# LoggingParams — defaults and field constraints
# ---------------------------------------------------------------------------


def test_logging_default_values(logging_: LoggingParams):
    """LoggingParams defaults match the expected logging frequency and W&B settings."""
    assert logging_.log_interval == 1
    assert logging_.use_wandb is True
    assert logging_.wandb_project == "pallatom-training"


def test_logging_rejects_zero_log_interval():
    """LoggingParams raises ValidationError when log_interval is zero."""
    with pytest.raises(ValidationError):
        LoggingParams(log_interval=0)


def test_logging_accepts_wandb_enabled():
    """LoggingParams stores a custom W&B project name when use_wandb=True."""
    p = LoggingParams(use_wandb=True, wandb_project="my-project")
    assert p.use_wandb is True
    assert p.wandb_project == "my-project"


# ---------------------------------------------------------------------------
# LoaderConfig — defaults and field constraints
# ---------------------------------------------------------------------------


def test_loader_default_values(loader: LoaderConfig):
    """LoaderConfig defaults match the expected sequence length and batch size settings."""
    assert loader.max_seq_length == 128
    assert loader.batch_size == 2


def test_loader_rejects_zero_batch_size():
    """LoaderConfig raises ValidationError when batch_size is zero."""
    with pytest.raises(ValidationError):
        LoaderConfig(batch_size=0)


def test_loader_rejects_zero_max_seq_length():
    """LoaderConfig raises ValidationError when max_seq_length is zero."""
    with pytest.raises(ValidationError):
        LoaderConfig(max_seq_length=0)


# ---------------------------------------------------------------------------
# ConditioningDropoutConfig — defaults, field constraints
# ---------------------------------------------------------------------------


def test_conditioning_dropout_config_defaults():
    """ConditioningDropoutConfig defaults set all dropout probabilities to 0.15."""
    cfg = ConditioningDropoutConfig()
    assert cfg.p_distogram == 0.15
    assert cfg.p_atom == 0.15
    assert cfg.p_seq == 0.15


def test_conditioning_dropout_config_custom_values():
    """ConditioningDropoutConfig stores independently specified probabilities for each modality."""
    cfg = ConditioningDropoutConfig(p_distogram=0.3, p_atom=0.1, p_seq=0.2)
    assert cfg.p_distogram == 0.3
    assert cfg.p_atom == 0.1
    assert cfg.p_seq == 0.2


def test_conditioning_dropout_config_rejects_negative():
    """ConditioningDropoutConfig raises ValidationError when a probability is negative."""
    with pytest.raises(ValidationError):
        ConditioningDropoutConfig(p_distogram=-0.1)


def test_conditioning_dropout_config_rejects_above_one():
    """ConditioningDropoutConfig raises ValidationError when a probability exceeds 1."""
    with pytest.raises(ValidationError):
        ConditioningDropoutConfig(p_seq=1.1)


def test_train_config_has_conditioning_dropout():
    """TrainConfig exposes a conditioning_dropout field of type ConditioningDropoutConfig."""
    cfg = TrainConfig()
    assert hasattr(cfg, "conditioning_dropout")
    assert isinstance(cfg.conditioning_dropout, ConditioningDropoutConfig)


def test_train_config_conditioning_dropout_defaults():
    """TrainConfig.conditioning_dropout uses the default p_distogram of 0.15."""
    cfg = TrainConfig()
    assert cfg.conditioning_dropout.p_distogram == 0.15
