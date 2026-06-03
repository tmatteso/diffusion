"""Frozen dataclasses bundling mutable training objects for the training loop."""

import dataclasses
from typing import NoReturn

import torch
from architecture.main_trunk import MainTrunk
from beartype import beartype
from helpers.featurize import Distogram
from jaxtyping import Float, jaxtyped
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from train.train_config import TrainConfig


def manual_seed(seed: int) -> torch.Generator:
    """Set the seed for generating random numbers.

    Args:
        seed: The desired seed.

    Returns:
        A torch.Generator whose state is set.
    """
    return torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]


@dataclasses.dataclass(frozen=True)
class ModelSetup:
    """Frozen bundle of all mutable training objects passed through the training loop.

    Attributes:
        model: The trunk model, either unwrapped or wrapped in DDP for multi-GPU training.
        tcfg: Training configuration holding hyperparameters, paths, and loss weights.
        distogram_res: Residue-level distogram head for computing pairwise distance logits.
        distogram_atom: Atom-level distogram head for computing sparse atom-pair distance logits.
        device: Target device string (e.g. ``"cuda:0"`` or ``"cpu"``).
        optimizer: Adam optimizer whose state is saved and restored on checkpoint.
        scheduler: Cosine annealing LR scheduler tied to the optimizer.
    """

    model: MainTrunk | DDP
    tcfg: TrainConfig
    distogram_res: Distogram
    distogram_atom: Distogram
    device: str
    optimizer: Adam
    scheduler: CosineAnnealingLR


@jaxtyped(typechecker=beartype)
@dataclasses.dataclass(frozen=True)
class LossMetrics:
    """Mean per-metric losses for one evaluation pass."""

    total_loss: Float[torch.Tensor, ""]
    Kabsch_aligned_MSE_loss: Float[torch.Tensor, ""]
    CE_loss: Float[torch.Tensor, ""]
    smooth_lddt_loss: Float[torch.Tensor, ""]
    res_distogram_loss: Float[torch.Tensor, ""]
    atom_distogram_loss: Float[torch.Tensor, ""]
    intermediate_loss: Float[torch.Tensor, ""]
    RMSD: Float[torch.Tensor, ""]


@jaxtyped(typechecker=beartype)
@dataclasses.dataclass(frozen=True)
class ThroughputStatistics:
    """Throughput statistics for one training step."""

    avg_batch_size: Float[torch.Tensor, ""]
    token_pack_rate: Float[torch.Tensor, ""]
    residues_per_sec: Float[torch.Tensor, ""]
    atoms_per_sec: Float[torch.Tensor, ""]

    # n_non_pad_tokens: int = int(batch.atom_mask.any(dim=-1).sum().item())
    # n_proteins: int = batch.atom_positions.shape[0]
    # max_seq_len: int = batch.atom_positions.shape[1]
    # n_all_tokens: int = n_proteins * max_seq_len
    # token_pack_rate: float = n_non_pad_tokens / n_all_tokens


@jaxtyped(typechecker=beartype)
@dataclasses.dataclass(frozen=True)
class ComponentNorms:
    """Per-component gradient L2 norms."""

    template_embedder: Float[torch.Tensor, ""]
    atom_encoder: Float[torch.Tensor, ""]
    atom_decoders: Float[torch.Tensor, ""]
    residue_distogram_head: Float[torch.Tensor, ""]
    atom_distogram_head: Float[torch.Tensor, ""]
    inter_proj_seq: Float[torch.Tensor, ""]
    inter_seq_logits: Float[torch.Tensor, ""]
    proj_seq: Float[torch.Tensor, ""]
    seq_logits: Float[torch.Tensor, ""]


@dataclasses.dataclass(frozen=True)
class EpochMetrics:
    """Aggregated metrics collected at the end of a training epoch.

    Attributes:
        epoch: Zero-based epoch index.
        global_step: Total optimizer steps taken across all epochs up to this point.
        train_loss_metrics: Mean loss components (total, FAPE, sequence, distogram) over the
            training split for this epoch.
        train_throughput_stats: Mean throughput statistics (tokens/sec, batch size, etc.) over
            the training split for this epoch.
        train_gradient_norms: Per-component gradient norms recorded during the training pass.
        val_loss_metrics: Mean loss components over the validation split for this epoch.
    """

    epoch: int
    global_step: int
    train_loss_metrics: LossMetrics
    train_throughput_stats: ThroughputStatistics
    train_gradient_norms: ComponentNorms
    val_loss_metrics: LossMetrics


@dataclasses.dataclass
class StepProgress:
    """Mutable accumulator for one epoch's optimizer steps.

    Attributes:
        global_step: Running optimizer step count across all epochs.
        rank: Process rank; used to gate logging to rank 0.
        pbar: tqdm progress bar for the current epoch.
        step_loss_metrics: Per-step loss metrics appended on each flush.
        step_throughput_stats: Per-step throughput stats appended on each flush.
        step_component_norms: Per-step gradient norms appended on each flush.
        step_n_proteins: Per-step protein counts appended on each flush.
    """

    global_step: int
    rank: int
    pbar: "tqdm[NoReturn]"
    step_loss_metrics: list[LossMetrics]
    step_throughput_stats: list[ThroughputStatistics]
    step_component_norms: list[ComponentNorms]
    step_n_proteins: list[int]
