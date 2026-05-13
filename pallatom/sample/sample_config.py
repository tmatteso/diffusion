"""Pydantic configuration models for conditional sampling."""

from pydantic import BaseModel, ConfigDict, Field
from train.train_config import ModelParams, NoiseScheduleParams


class SamplerParams(BaseModel):
    """Hyperparameters for the stochastic DDIM sampler."""

    model_config = ConfigDict(frozen=True)

    rho: float = Field(7.0, gt=0)
    S_churn: float = Field(0.0, ge=0)
    S_tmin: float = Field(0.0, ge=0)
    S_tmax: float = Field(1e38, gt=0)
    S_noise: float = Field(1.003, gt=0)
    ddim_steps: int = Field(40, gt=1)


class GenerationParams(BaseModel):
    """Parameters controlling the size and count of generated structures."""

    model_config = ConfigDict(frozen=True)

    n_res: int = Field(100, gt=0)
    n_samples: int = Field(1, gt=0)


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
    noise: NoiseScheduleParams = Field(default_factory=NoiseScheduleParams)
    sampler: SamplerParams = Field(default_factory=SamplerParams)
    generation: GenerationParams = Field(default_factory=GenerationParams)
    checkpoint: SampleCheckpointParams = Field(default_factory=SampleCheckpointParams)
    output: SampleOutputParams = Field(default_factory=SampleOutputParams)
