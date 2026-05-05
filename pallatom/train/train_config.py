from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator


class TrainingParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    num_epochs:        int            = 50
    lr:                float          = Field(3e-4, gt=0)
    weight_decay:      float          = Field(1e-4, ge=0)
    grad_clip:         Optional[float] = Field(2.0, gt=0)
    pretrained_weights: Optional[str]  = None


class ModelParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    f_ref_dim:     int = Field(35, gt=0)   # 5 atoms × 7 features
    n_bins:        int = Field(39, gt=0)   # distogram bins for TemplateEmbedder (38 + 1 overflow bin)
    c_atom:        int = Field(16, gt=0)
    c_pair:        int = Field(16, gt=0)
    c_res:         int = Field(32, gt=0)
    c_atompair:    int = Field(2, gt=0)
    K_unit:        int = Field(3, gt=0)
    max_residues:  int = Field(256, gt=0)


class NoiseScheduleParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    sigma_data: float = Field(19.2368, gt=0)
    sigma_max:  float = Field(42.3689, gt=0)
    sigma_min:  float = Field(3.807123, gt=0)
    P_mean:     float = Field(2.5416, gt=0)
    P_std:      float = Field(1.2048, gt=0)

    @model_validator(mode="after")
    def sigma_min_lt_max(self) -> "NoiseScheduleParams":
        if self.sigma_min >= self.sigma_max:
            raise ValueError(f"sigma_min ({self.sigma_min}) must be < sigma_max ({self.sigma_max})")
        return self


class ResidueDistogramParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    min_dist:    float = Field(3.25, ge=0)
    max_dist:    float = Field(50.75, gt=0)
    n_bins:      int   = Field(38, gt=0)
    tok_emb_dim: int   = Field(32, gt=0)

    @model_validator(mode="after")
    def min_lt_max(self) -> "ResidueDistogramParams":
        if self.min_dist >= self.max_dist:
            raise ValueError(f"min_dist ({self.min_dist}) must be < max_dist ({self.max_dist})")
        return self

class AtomDistogramParams(ResidueDistogramParams):
    min_dist: float = Field(0.0, ge=0)
    max_dist: float = Field(10.0, gt=0)
    n_bins:   int   = Field(22, gt=0)


class LossParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    # polar residues get weight 2.0, others 1.0 in the basic L0 loss
    lam:                float = Field(1.0, gt=0)
    alpha_0:            float = Field(0.25, ge=0)
    alpha_1:            float = Field(1.0, ge=0)
    alpha_2:            float = Field(0.5, ge=0)
    alpha_3:            float = Field(0.5, ge=0)
    alpha_4:            float = Field(1.0, ge=0)
    gamma:              float = Field(0.99, gt=0, lt=1)
    smooth_lddt_cutoff: int   = Field(15, gt=0)


class CheckpointParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    checkpoint_path: str = "pallatom_best_best.pt"
    save_every:      int = Field(1, ge=0)


class LoggingParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    log_interval:  int  = Field(1, ge=1)
    use_wandb:     bool = True
    wandb_project: str  = "pallatom-training"


class LoaderConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_seq_length: int = Field(128, gt=0) # was 256
    batch_size:     int = Field(2, gt=0) # was 2


TrainLoaderConfig = LoaderConfig
TestLoaderConfig  = LoaderConfig


class TrainConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    training:     TrainingParams      = Field(default_factory=TrainingParams)
    model:        ModelParams         = Field(default_factory=ModelParams)
    noise:        NoiseScheduleParams = Field(default_factory=NoiseScheduleParams)
    distogram_res:  ResidueDistogramParams = Field(default_factory=ResidueDistogramParams)
    distogram_atom: AtomDistogramParams    = Field(default_factory=AtomDistogramParams)
    loss:         LossParams          = Field(default_factory=LossParams)
    checkpoint:   CheckpointParams    = Field(default_factory=CheckpointParams)
    logging:      LoggingParams       = Field(default_factory=LoggingParams)
    loader:       LoaderConfig        = Field(default_factory=LoaderConfig)
    train_loader: LoaderConfig        = Field(default_factory=LoaderConfig)
    test_loader:  LoaderConfig        = Field(default_factory=LoaderConfig)
