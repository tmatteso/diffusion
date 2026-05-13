"""Pydantic configuration models for training, noise schedule, and model hyperparameters."""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TrainingParams(BaseModel):
    """Optimizer and training loop hyperparameters."""

    model_config = ConfigDict(frozen=True)

    num_epochs: int = 50
    lr: float = Field(3e-4, gt=0)
    weight_decay: float = Field(1e-4, ge=0)
    grad_clip: float | None = Field(2.0, gt=0)
    pretrained_weights: str | None = None


class ModelParams(BaseModel):
    """Architecture channel dimensions and capacity parameters."""

    model_config = ConfigDict(frozen=True)

    f_ref_dim: int = Field(35, gt=0)  # 5 atoms × 7 features
    n_bins: int = Field(39, gt=0)  # distogram bins for TemplateEmbedder (38 + 1 overflow bin)
    c_atom: int = Field(16, gt=0)
    c_pair: int = Field(16, gt=0)
    c_res: int = Field(32, gt=0)
    c_atompair: int = Field(2, gt=0)
    K_unit: int = Field(3, gt=0)
    max_residues: int = Field(256, gt=0)


class NoiseScheduleParams(BaseModel):
    """EDM diffusion noise schedule params (sigma bounds, data scale, and sampling distribution)."""

    model_config = ConfigDict(frozen=True)

    sigma_data: float = Field(19.2368, gt=0)
    sigma_max: float = Field(42.3689, gt=0)
    sigma_min: float = Field(3.807123, gt=0)
    P_mean: float = Field(2.5416, gt=0)
    P_std: float = Field(1.2048, gt=0)

    @model_validator(mode="after")
    def sigma_min_lt_max(self) -> "NoiseScheduleParams":
        """Validate that sigma_min is strictly less than sigma_max."""
        if self.sigma_min >= self.sigma_max:
            raise ValueError(f"sigma_min ({self.sigma_min}) must be < sigma_max ({self.sigma_max})")
        return self


class ResidueDistogramParams(BaseModel):
    """Binning configuration for the residue-level Cβ distance distogram."""

    model_config = ConfigDict(frozen=True)

    min_dist: float = Field(3.25, ge=0)
    max_dist: float = Field(50.75, gt=0)
    n_bins: int = Field(38, gt=0)
    tok_emb_dim: int = Field(32, gt=0)

    @model_validator(mode="after")
    def min_lt_max(self) -> "ResidueDistogramParams":
        """Validate that min_dist is strictly less than max_dist."""
        if self.min_dist >= self.max_dist:
            raise ValueError(f"min_dist ({self.min_dist}) must be < max_dist ({self.max_dist})")
        return self


class AtomDistogramParams(ResidueDistogramParams):
    """Binning configuration for the sparse atom-pair distance distogram."""

    min_dist: float = Field(0.0, ge=0)
    max_dist: float = Field(10.0, gt=0)
    n_bins: int = Field(22, gt=0)


class LossParams(BaseModel):
    """Weights and thresholds for the composite training loss."""

    model_config = ConfigDict(frozen=True)

    # polar residues get weight 2.0, others 1.0 in the basic L0 loss
    lam: float = Field(1.0, gt=0)
    alpha_0: float = Field(0.25, ge=0)
    alpha_1: float = Field(1.0, ge=0)
    alpha_2: float = Field(0.5, ge=0)
    alpha_3: float = Field(0.5, ge=0)
    alpha_4: float = Field(1.0, ge=0)
    gamma: float = Field(0.99, gt=0, lt=1)
    smooth_lddt_cutoff: int = Field(15, gt=0)


class CheckpointParams(BaseModel):
    """Checkpoint file path and save frequency."""

    model_config = ConfigDict(frozen=True)

    checkpoint_path: str = "pallatom_best_best.pt"
    save_every: int = Field(1, ge=0)


class LoggingParams(BaseModel):
    """Logging frequency and W&B project configuration."""

    model_config = ConfigDict(frozen=True)

    log_interval: int = Field(1, ge=1)
    use_wandb: bool = True
    wandb_project: str = "pallatom-training"


class LoaderConfig(BaseModel):
    """DataLoader sequence length cap and batch size."""

    model_config = ConfigDict(frozen=True)

    max_seq_length: int = Field(128, gt=0)  # was 256
    batch_size: int = Field(2, gt=0)  # was 2


TrainLoaderConfig = LoaderConfig
TestLoaderConfig = LoaderConfig


class ConditioningDropoutConfig(BaseModel):
    """Per-conditioning-signal dropout probabilities used during training."""

    model_config = ConfigDict(frozen=True)

    p_distogram: float = Field(0.15, ge=0.0, le=1.0)
    p_atom: float = Field(0.15, ge=0.0, le=1.0)
    p_seq: float = Field(0.15, ge=0.0, le=1.0)


class TrainConfig(BaseModel):
    """Top-level frozen config aggregating all training sub-configs."""

    model_config = ConfigDict(frozen=True)

    training: TrainingParams = Field(default_factory=TrainingParams)
    model: ModelParams = Field(default_factory=ModelParams)
    noise: NoiseScheduleParams = Field(default_factory=NoiseScheduleParams)
    distogram_res: ResidueDistogramParams = Field(default_factory=ResidueDistogramParams)
    distogram_atom: AtomDistogramParams = Field(default_factory=AtomDistogramParams)
    loss: LossParams = Field(default_factory=LossParams)
    checkpoint: CheckpointParams = Field(default_factory=CheckpointParams)
    logging: LoggingParams = Field(default_factory=LoggingParams)
    loader: LoaderConfig = Field(default_factory=LoaderConfig)
    train_loader: LoaderConfig = Field(default_factory=LoaderConfig)
    test_loader: LoaderConfig = Field(default_factory=LoaderConfig)
    conditioning_dropout: ConditioningDropoutConfig = Field(
        default_factory=ConditioningDropoutConfig
    )
