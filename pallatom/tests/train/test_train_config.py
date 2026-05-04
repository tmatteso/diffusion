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
    return TrainingParams()


@pytest.fixture
def model() -> ModelParams:
    return ModelParams()


@pytest.fixture
def noise() -> NoiseScheduleParams:
    return NoiseScheduleParams()


@pytest.fixture
def distogram_res() -> ResidueDistogramParams:
    return ResidueDistogramParams()


@pytest.fixture
def distogram_atom() -> AtomDistogramParams:
    return AtomDistogramParams()


@pytest.fixture
def loss() -> LossParams:
    return LossParams()


@pytest.fixture
def checkpoint() -> CheckpointParams:
    return CheckpointParams()


@pytest.fixture
def logging_() -> LoggingParams:
    return LoggingParams()


@pytest.fixture
def loader() -> LoaderConfig:
    return LoaderConfig()


@pytest.fixture
def cfg(
    training:      TrainingParams,
    model:         ModelParams,
    noise:         NoiseScheduleParams,
    distogram_res: ResidueDistogramParams,
    distogram_atom: AtomDistogramParams,
    loss:          LossParams,
    checkpoint:    CheckpointParams,
    logging_:      LoggingParams,
    loader:        LoaderConfig,
) -> TrainConfig:
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
    assert isinstance(TrainConfig(), TrainConfig)


def test_train_config_sub_models_are_correct_types(cfg: TrainConfig):
    assert isinstance(cfg.training,      TrainingParams)
    assert isinstance(cfg.model,         ModelParams)
    assert isinstance(cfg.noise,         NoiseScheduleParams)
    assert isinstance(cfg.distogram_res, ResidueDistogramParams)
    assert isinstance(cfg.distogram_atom, AtomDistogramParams)
    assert isinstance(cfg.loss,          LossParams)
    assert isinstance(cfg.checkpoint,    CheckpointParams)
    assert isinstance(cfg.logging,       LoggingParams)
    assert isinstance(cfg.loader,        LoaderConfig)


def test_train_config_accepts_nested_overrides():
    cfg = TrainConfig(training=TrainingParams(lr=1e-3, num_epochs=100))
    assert cfg.training.lr == 1e-3
    assert cfg.training.num_epochs == 100


# ---------------------------------------------------------------------------
# TrainConfig — immutability
# ---------------------------------------------------------------------------

def test_train_config_is_frozen(cfg: TrainConfig):
    with pytest.raises(Exception):
        cfg.training = TrainingParams()  # type: ignore[misc]


def test_training_params_is_frozen(training: TrainingParams):
    with pytest.raises(Exception):
        training.lr = 1e-2  # type: ignore[misc]


def test_model_params_is_frozen(model: ModelParams):
    with pytest.raises(Exception):
        model.c_res = 512  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TrainConfig — serialization round-trip
# ---------------------------------------------------------------------------

def test_train_config_model_dump_is_dict(cfg: TrainConfig):
    assert isinstance(cfg.model_dump(), dict)


def test_train_config_round_trips_through_dict(cfg: TrainConfig):
    assert TrainConfig.model_validate(cfg.model_dump()) == cfg


def test_train_config_round_trips_custom_values():
    cfg = TrainConfig(training=TrainingParams(lr=1e-5), loader=LoaderConfig(batch_size=64))
    cfg2 = TrainConfig.model_validate(cfg.model_dump())
    assert cfg2.training.lr == 1e-5
    assert cfg2.loader.batch_size == 64


# ---------------------------------------------------------------------------
# TrainingParams — defaults and field constraints
# ---------------------------------------------------------------------------

def test_training_default_values(training: TrainingParams):
    assert training.num_epochs == 50
    assert training.lr == pytest.approx(3e-4)
    assert training.weight_decay == pytest.approx(1e-4)
    assert training.grad_clip == pytest.approx(2.0)


def test_training_rejects_nonpositive_lr():
    with pytest.raises(ValidationError):
        TrainingParams(lr=0.0)


def test_training_rejects_negative_weight_decay():
    with pytest.raises(ValidationError):
        TrainingParams(weight_decay=-0.1)


def test_training_rejects_nonpositive_grad_clip():
    with pytest.raises(ValidationError):
        TrainingParams(grad_clip=0.0)


def test_training_accepts_none_grad_clip():
    assert TrainingParams(grad_clip=None).grad_clip is None


def test_training_accepts_positive_grad_clip():
    assert TrainingParams(grad_clip=10.0).grad_clip == 10.0


# ---------------------------------------------------------------------------
# ModelParams — defaults and field constraints
# ---------------------------------------------------------------------------

def test_model_default_values(model: ModelParams):
    assert model.f_ref_dim  == 35
    assert model.n_bins     == 39
    assert model.c_atom     == 16
    assert model.c_pair     == 16
    assert model.c_res      == 32
    assert model.c_atompair == 2
    assert model.K_unit     == 3


def test_model_rejects_zero_c_res():
    with pytest.raises(ValidationError):
        ModelParams(c_res=0)


def test_model_rejects_zero_k_unit():
    with pytest.raises(ValidationError):
        ModelParams(K_unit=0)


# ---------------------------------------------------------------------------
# NoiseScheduleParams — defaults, field constraints, cross-field validator
# ---------------------------------------------------------------------------

def test_noise_default_values(noise: NoiseScheduleParams):
    assert noise.sigma_data == pytest.approx(16.0)
    assert noise.sigma_max  == pytest.approx(80.0)
    assert noise.sigma_min  == pytest.approx(0.002)
    assert noise.P_mean     == pytest.approx(0.0)
    assert noise.P_std      == pytest.approx(1.0)


def test_noise_sigma_min_lt_sigma_max(noise: NoiseScheduleParams):
    assert noise.sigma_min < noise.sigma_max


def test_noise_rejects_nonpositive_sigma_data():
    with pytest.raises(ValidationError):
        NoiseScheduleParams(sigma_data=0.0)


def test_noise_rejects_nonpositive_sigma_min():
    with pytest.raises(ValidationError):
        NoiseScheduleParams(sigma_min=0.0)


def test_noise_rejects_sigma_min_equal_to_sigma_max():
    with pytest.raises(ValidationError):
        NoiseScheduleParams(sigma_min=80.0, sigma_max=80.0)


def test_noise_rejects_sigma_min_greater_than_sigma_max():
    with pytest.raises(ValidationError):
        NoiseScheduleParams(sigma_min=100.0, sigma_max=80.0)


# ---------------------------------------------------------------------------
# ResidueDistogramParams — defaults, field constraints, cross-field validator
# ---------------------------------------------------------------------------

def test_residue_distogram_default_values(distogram_res: ResidueDistogramParams):
    assert distogram_res.min_dist    == pytest.approx(3.25)
    assert distogram_res.max_dist    == pytest.approx(50.75)
    assert distogram_res.n_bins      == 38
    assert distogram_res.tok_emb_dim == 32


def test_residue_distogram_min_dist_lt_max_dist(distogram_res: ResidueDistogramParams):
    assert distogram_res.min_dist < distogram_res.max_dist


def test_residue_distogram_rejects_zero_n_bins():
    with pytest.raises(ValidationError):
        ResidueDistogramParams(n_bins=0)


def test_residue_distogram_rejects_min_equal_to_max():
    with pytest.raises(ValidationError):
        ResidueDistogramParams(min_dist=10.0, max_dist=10.0)


def test_residue_distogram_rejects_min_greater_than_max():
    with pytest.raises(ValidationError):
        ResidueDistogramParams(min_dist=20.0, max_dist=10.0)


def test_residue_distogram_accepts_zero_min_dist():
    assert ResidueDistogramParams(min_dist=0.0, max_dist=10.0).min_dist == 0.0


# ---------------------------------------------------------------------------
# AtomDistogramParams — defaults
# ---------------------------------------------------------------------------

def test_atom_distogram_default_values(distogram_atom: AtomDistogramParams):
    assert distogram_atom.min_dist == pytest.approx(0.0)
    assert distogram_atom.max_dist == pytest.approx(10.0)
    assert distogram_atom.n_bins   == 22


def test_atom_distogram_min_dist_lt_max_dist(distogram_atom: AtomDistogramParams):
    assert distogram_atom.min_dist < distogram_atom.max_dist


# ---------------------------------------------------------------------------
# LossParams — defaults and field constraints
# ---------------------------------------------------------------------------

def test_loss_default_values(loss: LossParams):
    assert loss.lam                == pytest.approx(1.0)
    assert loss.alpha_0            == pytest.approx(0.25)
    assert loss.alpha_1            == pytest.approx(1.0)
    assert loss.gamma              == pytest.approx(0.99)
    assert loss.smooth_lddt_cutoff == 15


def test_loss_rejects_nonpositive_lam():
    with pytest.raises(ValidationError):
        LossParams(lam=0.0)


def test_loss_rejects_gamma_of_one():
    with pytest.raises(ValidationError):
        LossParams(gamma=1.0)


def test_loss_rejects_gamma_of_zero():
    with pytest.raises(ValidationError):
        LossParams(gamma=0.0)


def test_loss_rejects_negative_alpha():
    with pytest.raises(ValidationError):
        LossParams(alpha_0=-0.1)


def test_loss_rejects_nonpositive_smooth_lddt_cutoff():
    with pytest.raises(ValidationError):
        LossParams(smooth_lddt_cutoff=0)


# ---------------------------------------------------------------------------
# CheckpointParams — defaults and field constraints
# ---------------------------------------------------------------------------

def test_checkpoint_default_values(checkpoint: CheckpointParams):
    assert checkpoint.checkpoint_path == "pallatom_best_best.pt"
    assert checkpoint.save_every == 1


def test_checkpoint_accepts_save_every_zero():
    assert CheckpointParams(save_every=0).save_every == 0


def test_checkpoint_rejects_negative_save_every():
    with pytest.raises(ValidationError):
        CheckpointParams(save_every=-1)


# ---------------------------------------------------------------------------
# LoggingParams — defaults and field constraints
# ---------------------------------------------------------------------------

def test_logging_default_values(logging_: LoggingParams):
    assert logging_.log_interval  == 1
    assert logging_.use_wandb     is True
    assert logging_.wandb_project == "pallatom-training"


def test_logging_rejects_zero_log_interval():
    with pytest.raises(ValidationError):
        LoggingParams(log_interval=0)


def test_logging_accepts_wandb_enabled():
    p = LoggingParams(use_wandb=True, wandb_project="my-project")
    assert p.use_wandb is True
    assert p.wandb_project == "my-project"


# ---------------------------------------------------------------------------
# LoaderConfig — defaults and field constraints
# ---------------------------------------------------------------------------

def test_loader_default_values(loader: LoaderConfig):
    assert loader.max_seq_length == 128
    assert loader.batch_size     == 2


def test_loader_rejects_zero_batch_size():
    with pytest.raises(ValidationError):
        LoaderConfig(batch_size=0)


def test_loader_rejects_zero_max_seq_length():
    with pytest.raises(ValidationError):
        LoaderConfig(max_seq_length=0)
