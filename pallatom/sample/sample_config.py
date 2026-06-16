"""Pydantic configuration models for conditional sampling.

This module defines frozen Pydantic models that group all hyperparameters
required to run the stochastic DDIM sampler, including noise-schedule settings,
generation size, model checkpoint paths, and output paths.
"""

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field
from train.train_config import (
    AtomDistogramParams,
    ModelParams,
    NoiseScheduleParams,
    ResidueDistogramParams,
)


class SamplerParams(BaseModel):
    """Hyperparameters for the stochastic DDIM sampler.

    Attributes:
        model_config: Pydantic config — frozen to prevent mutation after init.
        rho: Exponent controlling the non-linear time-step spacing; higher
            values concentrate steps near low noise levels.
        S_churn: Amount of stochasticity injected at each step; 0 recovers
            deterministic DDIM.
        S_tmin: Lower bound of the noise-level range in which churn is applied.
        S_tmax: Upper bound of the noise-level range in which churn is applied.
        S_noise: Scale factor applied to injected Gaussian noise during churn.
        ddim_steps: Total number of denoising steps.
        eta_step_scale: Multiplicative scale for the effective step size used in
            the eta (coordinate-update) calculation.
        seq_temperature: Softmax temperature for sampling amino-acid sequence
            tokens from the sequence logits.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    rho: float = Field(default=7.0, gt=0)
    S_churn: float = Field(default=0.2, ge=0)
    S_tmin: float = Field(default=0.01, ge=0)
    S_tmax: float = Field(default=1.0, gt=0)
    S_noise: float = Field(default=1.003, gt=0)
    ddim_steps: int = Field(default=200, gt=1)
    eta_step_scale: float = Field(default=2.25, gt=1)
    seq_temperature: float = Field(default=0.1, gt=0)


class GenerationParams(BaseModel):
    """Parameters controlling the size and count of generated structures.

    Attributes:
        model_config: Pydantic config — frozen to prevent mutation after init.
        n_res: Number of residues in each generated protein structure.
        n_samples: Number of independent structures to generate per run.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    n_res: int = Field(default=100, gt=0)
    n_samples: int = Field(default=1, gt=0)


class SampleCheckpointParams(BaseModel):
    """Path configuration for loading a model checkpoint during sampling.

    Attributes:
        model_config: Pydantic config — frozen to prevent mutation after init.
        checkpoint_path: Filesystem path to the saved model checkpoint file
            (``.pt``) that will be loaded before the sampling loop begins.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    checkpoint_path: str = "pallatom_best.pt"


class SampleOutputParams(BaseModel):
    """Path configuration for writing sampling results to disk.

    Attributes:
        model_config: Pydantic config — frozen to prevent mutation after init.
        output_path: Filesystem path where the JSON file containing generated
            structures and sequences will be written.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    output_path: str = "samples.json"


class SampleConfig(BaseModel):
    """Top-level configuration for a conditional sampling run.

    Aggregates all sub-configurations required to fully specify a sampling
    experiment: model architecture, distogram parameters, noise schedule,
    sampler hyperparameters, generation size, checkpoint location, and output
    destination.

    Attributes:
        model_config: Pydantic config — frozen to prevent mutation after init.
        model: Architecture hyperparameters for the denoising model.
        distogram_res: Residue-level distogram head parameters.
        distogram_atom: Atom-level distogram head parameters.
        noise: Noise schedule parameters (sigma range, sigma_data, etc.).
        sampler: DDIM sampler hyperparameters.
        generation: Number of residues and samples to generate.
        checkpoint: Path to the model checkpoint to load.
        output: Path where generated results are written.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    model: ModelParams = Field(default_factory=ModelParams)
    distogram_res: ResidueDistogramParams = Field(
        default_factory=ResidueDistogramParams,
    )
    distogram_atom: AtomDistogramParams = Field(
        default_factory=AtomDistogramParams,
    )
    noise: NoiseScheduleParams = Field(default_factory=NoiseScheduleParams)
    sampler: SamplerParams = Field(default_factory=SamplerParams)
    generation: GenerationParams = Field(default_factory=GenerationParams)
    checkpoint: SampleCheckpointParams = Field(
        default_factory=SampleCheckpointParams,
    )
    output: SampleOutputParams = Field(default_factory=SampleOutputParams)
