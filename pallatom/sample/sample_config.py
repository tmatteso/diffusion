"""Pydantic configuration models for conditional sampling."""

from pydantic import BaseModel, ConfigDict, Field
from train.train_config import (
    AtomDistogramParams,
    ModelParams,
    NoiseScheduleParams,
    ResidueDistogramParams,
)


class SamplerParams(BaseModel):
    """Hyperparameters for the stochastic DDIM sampler."""

    model_config = ConfigDict(frozen=True)

    rho: float = Field(default=7.0, gt=0)
    S_churn: float = Field(default=0.0, ge=0)
    S_tmin: float = Field(default=0.0, ge=0)
    S_tmax: float = Field(default=1e38, gt=0)
    S_noise: float = Field(default=1.003, gt=0)
    ddim_steps: int = Field(default=40, gt=1)


class GenerationParams(BaseModel):
    """Parameters controlling the size and count of generated structures."""

    model_config = ConfigDict(frozen=True)

    n_res: int = Field(default=100, gt=0)
    n_samples: int = Field(default=1, gt=0)


class SampleCheckpointParams(BaseModel):
    """Path configuration for loading a model checkpoint during sampling."""

    model_config = ConfigDict(frozen=True)

    checkpoint_path: str = "pallatom_best.pt"


class SampleOutputParams(BaseModel):
    """Path configuration for writing sampling results to disk."""

    model_config = ConfigDict(frozen=True)

    output_path: str = "samples.json"


class SampleConfig(BaseModel):
    """Top-level configuration for a conditional sampling run."""

    model_config = ConfigDict(frozen=True)

    model: ModelParams = Field(default_factory=ModelParams)
    distogram_res: ResidueDistogramParams = Field(default_factory=ResidueDistogramParams)
    distogram_atom: AtomDistogramParams = Field(default_factory=AtomDistogramParams)
    noise: NoiseScheduleParams = Field(default_factory=NoiseScheduleParams)
    sampler: SamplerParams = Field(default_factory=SamplerParams)
    generation: GenerationParams = Field(default_factory=GenerationParams)
    checkpoint: SampleCheckpointParams = Field(default_factory=SampleCheckpointParams)
    output: SampleOutputParams = Field(default_factory=SampleOutputParams)
