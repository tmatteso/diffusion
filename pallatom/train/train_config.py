"""Pydantic configuration models for training, noise, and model hyperparameters.

Contains frozen Pydantic models that collectively define every knob available
during a training run: optimizer settings, model capacity, EDM noise schedule,
distogram binning, loss weights, checkpointing, logging, data-loading, and
conditioning dropout.
"""

import dataclasses
import pathlib
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator


class InvalidDistanceError(ValueError):
    """Raised when min_dist is not strictly less than max_dist."""

    def __init__(self, min_dist: float, max_dist: float) -> None:
        super().__init__(
            f"min_dist ({min_dist}) must be < max_dist ({max_dist})",
        )


class InvalidSigmaRangeError(ValueError):
    """Raised when sigma_min is not strictly less than sigma_max."""

    def __init__(self, sigma_min: float, sigma_max: float) -> None:
        super().__init__(
            f"sigma_min ({sigma_min}) must be < sigma_max ({sigma_max})",
        )


class InsufficientShardsError(ValueError):
    """Raised when num_workers exceeds n_shards.

    The FFD plan assigns shards round-robin (worker w streams
    shards[w::num_workers]).  If num_workers > n_shards some workers receive
    zero shards, causing WebDataset to raise ValueError during iteration.
    """

    def __init__(self, num_workers: int, n_shards: int) -> None:
        super().__init__(
            f"num_workers ({num_workers}) must be <= n_shards ({n_shards}); "
            + "workers with no shards cause WebDataset to raise at runtime",
        )


class TrainingParams(BaseModel):
    """Optimizer and training loop hyperparameters.

    Attributes:
        model_config: Pydantic config — frozen to prevent mutation after init.
        num_epochs: Total number of training epochs.
        lr: Peak learning rate for the optimizer.
        weight_decay: L2 regularization coefficient applied by the optimizer.
        grad_clip: Maximum gradient norm for clipping; disabled when ``None``.
        pretrained_weights: Path to a pretrained checkpoint to load weights from
            before training begins; skipped when ``None``.
        resume_checkpoint: Path to a full training checkpoint (weights +
            optimizer state) to resume from; skipped when ``None``.
        accumulated_token_budget: Target token count across
            gradient-accumulation steps before a parameter update is applied.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    num_epochs: int = 50
    lr: float = Field(default=3e-4, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    grad_clip: float | None = Field(default=2.0, gt=0)
    pretrained_weights: pathlib.Path | None = None
    resume_checkpoint: pathlib.Path | None = None
    accumulated_token_budget: int = Field(default=4096, gt=0)


class ModelParams(BaseModel):
    """Architecture channel dimensions and capacity parameters.

    Attributes:
        model_config: Pydantic config — frozen to prevent mutation after init.
        window_size: Residue-level neighbour window used by the atom transformer
            to build sparse pairs (default 32).
        f_ref_dim: Dimensionality of the reference atom feature vector
            (5 atoms by 7 features = 35).
        c_atom: Hidden dimension for atom-level single embeddings.
        c_pair: Hidden dimension for trunk residue-pair embeddings.
        c_res: Hidden dimension for residue-level single embeddings.
        c_atompair: Hidden dimension for sparse atom-pair embeddings.
        K_unit: Number of decoder units in the diffusion trunk.
        max_residues: Maximum sequence length the model is built to handle.
        n_amino: Amino-acid vocabulary size (canonical residues only).
        n_blocks_atom_transformer_encoder: Transformer depth for atom encoder.
        n_heads_atom_transformer_encoder: Attention heads in the atom encoder.
        n_blocks_atom_transformer_decoder: Transformer depth for atom decoder.
        n_heads_atom_transformer_decoder: Attention heads in the atom decoder.
        n_pairformer_blocks_template_embedder: Pairformer depth inside the
            template embedder.
        n_paiformer_heads_template_embedder: Attention heads inside the
            template embedder pairformer (16, as per AlphaFold 3).
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    window_size: int = Field(default=32, gt=0)
    f_ref_dim: int = Field(default=35, gt=0)  # 5 atoms by 7 features
    c_atom: int = Field(default=4, gt=0)
    c_pair: int = Field(default=4, gt=0)
    c_res: int = Field(default=8, gt=0)
    c_atompair: int = Field(default=2, gt=0)
    K_unit: int = Field(default=8, gt=0)
    max_residues: int = Field(default=128, gt=0)
    n_amino: int = Field(default=20, gt=0)  # canonical AA for now
    n_blocks_atom_transformer_encoder: int = Field(default=3, gt=0)
    n_heads_atom_transformer_encoder: int = Field(default=4, gt=0)
    n_blocks_atom_transformer_decoder: int = Field(default=3, gt=0)
    n_heads_atom_transformer_decoder: int = Field(default=4, gt=0)
    n_pairformer_blocks_template_embedder: int = Field(default=2, gt=0)
    n_paiformer_heads_template_embedder: int = Field(
        default=16,
        gt=0,
    )  # as per AF3


class NoiseScheduleParams(BaseModel):
    """EDM noise schedule params (sigma bounds, data scale and distribution).

    Attributes:
        model_config: Pydantic config — frozen to prevent mutation after init.
        sigma_data: Expected standard deviation of the clean data distribution,
            used to scale the preconditioning factors.
        sigma_max: Maximum noise level; denoising starts from this sigma.
        sigma_min: Minimum noise level; denoising terminates at this sigma.
        P_mean: Mean of the log-normal distribution used to sample training
            sigmas.
        P_std: Standard deviation of the log-normal distribution used to sample
            training sigmas.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    sigma_data: float = Field(default=16.0, gt=0)
    sigma_max: float = Field(default=160, gt=0)
    sigma_min: float = Field(default=4 * 10 ** (-4), gt=0)
    P_mean: float = Field(default=-1.2)
    P_std: float = Field(default=1.5, gt=0)

    @model_validator(mode="after")
    def sigma_min_lt_max(self) -> "NoiseScheduleParams":
        """Validate that sigma_min is strictly less than sigma_max.

        Returns:
            The validated model instance.

        Raises:
            InvalidSigmaRangeError: If ``sigma_min`` is greater than or equal
                to ``sigma_max``.
        """
        if self.sigma_min >= self.sigma_max:
            raise InvalidSigmaRangeError(self.sigma_min, self.sigma_max)
        return self


class ResidueDistogramParams(BaseModel):
    """Binning configuration for the residue-level Cβ distance distogram.

    Attributes:
        model_config: Pydantic config — frozen to prevent mutation after init.
        min_dist: Lower edge of the first distance bin in ångströms.
        max_dist: Upper edge of the last distance bin in ångströms.
        n_bins: Total number of distance bins, including the overflow bin.
        tok_emb_dim: Dimensionality of the learned bin-token embedding.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    min_dist: float = Field(default=3.25, ge=0)
    max_dist: float = Field(default=50.75, gt=0)
    n_bins: int = Field(
        default=39,
        gt=0,
    )  # 38 + 1. need another arg here for overflow
    tok_emb_dim: int = Field(default=32, gt=0)

    @model_validator(mode="after")
    def min_lt_max(self) -> "ResidueDistogramParams":
        """Validate that min_dist is strictly less than max_dist.

        Returns:
            The validated model instance.

        Raises:
            InvalidDistanceError: If ``min_dist`` is greater than or equal to
                ``max_dist``.
        """
        if self.min_dist >= self.max_dist:
            raise InvalidDistanceError(self.min_dist, self.max_dist)
        return self


class AtomDistogramParams(ResidueDistogramParams):
    """Binning configuration for the sparse atom-pair distance distogram.

    Overrides residue distogram defaults with tighter distance bounds suited
    for covalent and near-covalent atom-atom distances.

    Attributes:
        min_dist: Lower edge of the first distance bin (default 0.0 Å).
        max_dist: Upper edge of the last distance bin (default 10.0 Å).
        n_bins: Total number of distance bins (default 22).
    """

    min_dist: float = Field(default=0.0, ge=0)
    max_dist: float = Field(default=10.0, gt=0)
    n_bins: int = Field(default=22, gt=0)


class LossParams(BaseModel):
    """Weights and thresholds for the composite training loss.

    Attributes:
        model_config: Pydantic config — frozen to prevent mutation after init.
        lam: Scale factor applied to the diffusion denoising loss term.
        alpha_0: Weight for the auxiliary coordinate loss at intermediate steps.
        alpha_1: Weight for the sequence cross-entropy loss term.
        alpha_2: Weight for the residue-level distogram loss.
        alpha_3: Weight for the atom-pair distogram loss.
        alpha_4: Weight for the smooth lDDT structural quality loss.
        gamma: Exponential moving average decay used when computing auxiliary
            losses across decoder units.
        smooth_lddt_cutoff: Distance cutoff in ångströms used when computing the
            smooth lDDT score.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    # polar residues get weight 2.0, others 1.0 in the basic L0 loss
    lam: float = Field(default=1.0, gt=0)
    alpha_0: float = Field(default=0.25, ge=0)
    alpha_1: float = Field(default=1.0, ge=0)
    alpha_2: float = Field(default=0.5, ge=0)
    alpha_3: float = Field(default=0.5, ge=0)
    alpha_4: float = Field(default=1.0, ge=0)
    gamma: float = Field(default=0.99, gt=0, lt=1)
    smooth_lddt_cutoff: int = Field(default=15, gt=0)


class CheckpointParams(BaseModel):
    """Checkpoint file path and save frequency.

    Attributes:
        model_config: Pydantic config — frozen to prevent mutation after init.
        checkpoint_path: File path where the best model checkpoint is written.
        save_every: Save a checkpoint every this many epochs; set to 0 to
            disable periodic saving (best-model saving is unaffected).
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    checkpoint_path: pathlib.Path = pathlib.Path("pallatom_best.pt")
    save_every: int = Field(default=1, ge=0)


class LoggingParams(BaseModel):
    """Logging frequency and W&B project configuration.

    Attributes:
        model_config: Pydantic config — frozen to prevent mutation after init.
        use_wandb: When ``True``, metrics are streamed to Weights & Biases.
        wandb_project: Name of the W&B project to log runs under.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    use_wandb: bool = True
    wandb_project: str = "pallatom-training"


class LoaderConfig(BaseModel):
    """Base DataLoader configuration shared across dataset splits.

    Attributes:
        model_config: Pydantic config — frozen to prevent mutation after init.
        max_seq_length: Maximum number of residues per sample; longer chains
            are truncated or excluded.
        batch_size: Number of samples per batch.
        num_workers: Number of DataLoader worker processes.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    max_seq_length: int = Field(default=128, gt=0)  # was 256
    batch_size: int = Field(default=2, gt=0)  # was 2
    num_workers: int = Field(default=4, gt=0)


class TrainLoaderConfig(LoaderConfig):
    """DataLoader configuration for the training split.

    Extends `LoaderConfig` with training-specific data-pipeline settings.

    Attributes:
        model_config: Pydantic config — frozen to prevent mutation after init.
        token_budget: Upper bound on the total number of residues across all
            samples in a single batch step.
        batch_prefetch_depth: Number of batches to prefetch from the shard
            worker.
        epoch_prefetch_depth: Number of epochs to prefetch shard plans for.
        seed: Random seed for the training data sampler.
        n_threads: Number of worker threads used by the DataLoader.
        n_shards: Total number of shard tars in the prepared training split.
            Must be >= ``num_workers`` so every DataLoader worker receives at
            least one shard in the FFD round-robin assignment.
        noise_magnitude: Half-width of uniform noise added to each protein's
            length before sorting; controls cross-epoch FFD batch diversity.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    token_budget: int = Field(default=384, gt=0)
    batch_prefetch_depth: int = Field(default=4, gt=0)
    epoch_prefetch_depth: int = Field(default=3, gt=0)
    seed: int = Field(default=0)
    n_threads: int = Field(default=32, gt=0)
    n_shards: int = Field(default=16, gt=0)
    noise_magnitude: int = Field(default=10, ge=0)

    @model_validator(mode="after")
    def workers_le_shards(self) -> "TrainLoaderConfig":
        """Validate that num_workers does not exceed n_shards.

        Returns:
            The validated model instance.

        Raises:
            InsufficientShardsError: If ``num_workers`` is greater than
                ``n_shards``.
        """
        if self.num_workers > self.n_shards:
            raise InsufficientShardsError(self.num_workers, self.n_shards)
        return self


class ConditioningDropoutConfig(BaseModel):
    """Per-conditioning-signal dropout probabilities used during training.

    Each field controls the probability that the corresponding conditioning
    input is zeroed out for a given sample, encouraging the model to operate
    robustly in the absence of any single conditioning signal.

    Attributes:
        model_config: Pydantic config — frozen to prevent mutation after init.
        p_distogram: Dropout probability for the residue-level Cβ distogram
            conditioning signal.
        p_atom: Dropout probability for the atom-pair distogram conditioning
            signal.
        p_seq: Dropout probability for the sequence conditioning signal.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    p_distogram: float = Field(default=0.15, ge=0.0, le=1.0)
    p_atom: float = Field(default=0.15, ge=0.0, le=1.0)
    p_seq: float = Field(default=0.15, ge=0.0, le=1.0)


class TrainConfig(BaseModel):
    """Top-level frozen config aggregating all training sub-configs.

    Composes every sub-configuration object into a single immutable structure
    that is passed through the training pipeline, ensuring consistent
    hyperparameters across all components.

    Attributes:
        model_config: Pydantic config — frozen to prevent mutation after init.
        training: Optimizer and training loop settings.
        model: Architecture channel dimensions and capacity.
        noise: EDM noise schedule parameters.
        distogram_res: Binning config for the residue-level Cβ distogram.
        distogram_atom: Binning config for the atom-pair distogram.
        loss: Weights and thresholds for the composite training loss.
        checkpoint: Checkpoint path and save frequency.
        logging: Logging and W&B project settings.
        loader: Default DataLoader settings (sequence length, batch size,
            token budget).
        train_loader: DataLoader settings used for the training split.
        test_loader: DataLoader settings used for the evaluation split.
        conditioning_dropout: Dropout probabilities for each conditioning
            signal.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    training: TrainingParams = Field(default_factory=TrainingParams)
    model: ModelParams = Field(default_factory=ModelParams)
    noise: NoiseScheduleParams = Field(default_factory=NoiseScheduleParams)
    distogram_res: ResidueDistogramParams = Field(
        default_factory=ResidueDistogramParams,
    )
    distogram_atom: AtomDistogramParams = Field(
        default_factory=AtomDistogramParams,
    )
    loss: LossParams = Field(default_factory=LossParams)
    checkpoint: CheckpointParams = Field(default_factory=CheckpointParams)
    logging: LoggingParams = Field(default_factory=LoggingParams)
    loader: LoaderConfig = Field(default_factory=LoaderConfig)
    train_loader: TrainLoaderConfig = Field(default_factory=TrainLoaderConfig)
    test_loader: LoaderConfig = Field(default_factory=LoaderConfig)
    conditioning_dropout: ConditioningDropoutConfig = Field(
        default_factory=ConditioningDropoutConfig,
    )


@dataclasses.dataclass
class TrainArgs:
    """Parsed command-line arguments for the training entry point.

    Attributes:
        dataset_jsonl: Path to the proteins JSONL dataset file.
        shard_dir: Directory containing the pre-built shard tars.
        keys_for_splits_json: Path to the train/val/test splits JSON.
        config: Path to TrainConfig JSON.
        structlog_jsonl: Path to write structured JSON log lines.
        ddp: If True, use DistributedDataParallel training.
        debug_run: If True, restrict to 252 proteins for fast iteration.
    """

    dataset_jsonl: Path
    shard_dir: Path
    keys_for_splits_json: Path
    config: Path
    structlog_jsonl: Path
    ddp: bool
    debug_run: bool
