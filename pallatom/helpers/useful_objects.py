"""Frozen dataclasses bundling mutable training objects for the training loop.

Contains ModelSetup, LossMetrics, ThroughputStatistics, ComponentNorms,
EpochMetrics, and StepProgress -- each grouping related scalars or tensors
that are threaded through training and evaluation passes.
"""

import dataclasses
from typing import NoReturn, cast

import torch
import torch.distributed as dist
from architecture.main_trunk import MainTrunk
from beartype import beartype
from einops import reduce
from helpers.featurize import Distogram
from jaxtyping import Float, jaxtyped
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from train.train_config import TrainConfig
from typing_extensions import Self


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
    """Frozen bundle of all mutable training objects passed through the loop.

    Attributes:
        model: The trunk model, either unwrapped or wrapped in DDP for
            multi-GPU training.
        tcfg: Training configuration holding hyperparameters, paths, and loss
            weights.
        distogram_res: Residue-level distogram head for computing pairwise
            distance logits.
        distogram_atom: Atom-level distogram head for computing sparse
            atom-pair distance logits.
        device: Target device string (e.g. ``"cuda:0"`` or ``"cpu"``).
        optimizer: Adam optimizer whose state is saved and restored on
            checkpoint.
        scheduler: Cosine annealing LR scheduler tied to the optimizer.
    """

    model: MainTrunk | DDP
    tcfg: TrainConfig
    distogram_res: Distogram
    distogram_atom: Distogram
    device: torch.device
    optimizer: Adam
    scheduler: CosineAnnealingLR


@dataclasses.dataclass
class TensorAccumulatorMixin:
    """Mixin for dataclasses whose fields are all scalar tensors.

    Provides field-wise arithmetic operators and helpers for accumulating
    and averaging metrics across batches and DDP ranks.  Subclasses must
    be decorated with ``@dataclasses.dataclass``; all fields are expected
    to be ``Float[torch.Tensor, ""]`` scalars.
    """

    def to_float_dict(self, prefix: str = "") -> dict[str, float]:
        """Return a plain Python float for every field.

        Args:
            prefix: Optional string prepended to every key, e.g.
                ``"train/"`` to namespace wandb log entries.

        Returns:
            Mapping of ``{prefix + field_name: float_value}``.
        """
        return {
            f"{prefix}{f.name}": cast(
                torch.Tensor,
                getattr(self, f.name),
            ).item()
            for f in dataclasses.fields(self)
        }

    @classmethod
    def zero_init(cls, device: torch.device) -> Self:
        """Construct an instance with every field set to 0.0.

        Args:
            device: Device on which the zero tensors are allocated.

        Returns:
            New instance with all scalar fields initialised to zero.
        """
        return cls(
            **{
                f.name: torch.tensor(0.0, device=device)
                for f in dataclasses.fields(cls)
            },
        )

    def __iadd__(self, other: Self) -> Self:
        """Add another instance field-wise in place.

        Args:
            other: Instance of the same type whose fields are added.

        Returns:
            Self after in-place update.
        """
        for f in dataclasses.fields(self):
            setattr(
                self,
                f.name,
                getattr(self, f.name) + getattr(other, f.name),
            )
        return self

    def __itruediv__(self, scalar: float) -> Self:
        """Divide every field by a scalar in place.

        Args:
            scalar: Value to divide each field tensor by.

        Returns:
            Self after in-place update.
        """
        for f in dataclasses.fields(self):
            setattr(self, f.name, getattr(self, f.name) / scalar)
        return self

    def __mul__(self, scalar: float) -> Self:
        """Return a new instance with every field multiplied by a scalar.

        Args:
            scalar: Value to multiply each field tensor by.

        Returns:
            New instance of the same type with scaled fields.
        """
        return type(self)(
            **{
                f.name: getattr(self, f.name) * scalar
                for f in dataclasses.fields(self)
            },
        )

    @classmethod
    def weighted_avg(cls, instances: list[Self], n_proteins: list[int]) -> Self:
        """Compute a protein-count-weighted average across instances.

        Args:
            instances: List of same-type instances to average over.
            n_proteins: Protein count for each instance, used as weights.

        Returns:
            New instance where each field is the weighted mean across
            all instances.
        """
        total: int = sum(n_proteins)
        first = dataclasses.fields(cls)[0]
        tensor = cast(torch.Tensor, getattr(instances[0], first.name))
        device: torch.device = tensor.device

        weights: Float[torch.Tensor, "N"] = torch.tensor(
            n_proteins,
            dtype=torch.float32,
            device=device,
        )
        return cls(
            **{
                f.name: reduce(
                    torch.stack([getattr(inst, f.name) for inst in instances])
                    * weights,
                    "N ->",
                    "sum",
                )
                / total
                for f in dataclasses.fields(cls)
            },
        )

    def all_reduce_(self) -> None:
        """Sum all tensor fields across DDP ranks in place."""
        for f in dataclasses.fields(self):
            _ = dist.all_reduce(  # pyright: ignore[reportUnknownMemberType]
                cast(torch.Tensor, getattr(self, f.name)),
                op=dist.ReduceOp.SUM,
            )


@jaxtyped(typechecker=beartype)
@dataclasses.dataclass
class LossMetrics(TensorAccumulatorMixin):
    """Mean per-metric losses for one evaluation pass.

    Attributes:
        total_loss: Weighted sum of all individual loss components.
        Kabsch_aligned_MSE_loss: MSE between predicted and ground-truth
            atom positions after Kabsch alignment.
        CE_loss: Cross-entropy loss for sequence prediction.
        smooth_lddt_loss: Smooth local distance difference test loss.
        res_distogram_loss: Residue-level distogram cross-entropy loss.
        atom_distogram_loss: Atom-level sparse distogram cross-entropy loss.
        intermediate_loss: Auxiliary loss from intermediate decoder outputs.
        RMSD: Root-mean-square deviation of predicted vs ground-truth positions.
    """

    total_loss: Float[torch.Tensor, ""]
    Kabsch_aligned_MSE_loss: Float[torch.Tensor, ""]
    CE_loss: Float[torch.Tensor, ""]
    smooth_lddt_loss: Float[torch.Tensor, ""]
    res_distogram_loss: Float[torch.Tensor, ""]
    atom_distogram_loss: Float[torch.Tensor, ""]
    intermediate_loss: Float[torch.Tensor, ""]
    RMSD: Float[torch.Tensor, ""]


@jaxtyped(typechecker=beartype)
@dataclasses.dataclass
class ThroughputStatistics(TensorAccumulatorMixin):
    """Throughput statistics for one training step.

    Attributes:
        avg_batch_size: Mean number of proteins in the batch.
        token_pack_rate: Ratio of non-padding tokens to total token slots.
        residues_per_sec: Number of residues processed per second.
        atoms_per_sec: Number of atoms processed per second.
    """

    avg_batch_size: Float[torch.Tensor, ""]
    token_pack_rate: Float[torch.Tensor, ""]
    residues_per_sec: Float[torch.Tensor, ""]
    atoms_per_sec: Float[torch.Tensor, ""]


@jaxtyped(typechecker=beartype)
@dataclasses.dataclass
class ComponentNorms(TensorAccumulatorMixin):
    """Per-component gradient L2 norms.

    Each field holds the total L2 norm of the gradients for the named model
    sub-module, recorded after the backward pass and before the optimizer step.

    Attributes:
        template_embedder: Gradient norm for the template embedder sub-module.
        atom_encoder: Gradient norm for the atom feature encoder.
        atom_decoders: Gradient norm across all atom attention decoder blocks.
        residue_distogram_head: Gradient norm for the residue-level
            distogram head.
        atom_distogram_head: Gradient norm for the atom-level distogram head.
        inter_proj_seq: Gradient norm for the intermediate sequence
            projection layer.
        inter_seq_logits: Gradient norm for the intermediate sequence
            logit head.
        proj_seq: Gradient norm for the final sequence projection layer.
        seq_logits: Gradient norm for the final sequence logit head.
    """

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
        global_step: Total optimizer steps taken across all epochs up to this
            point.
        train_loss_metrics: Mean loss components (total, FAPE, sequence,
            distogram) over the
            training split for this epoch.
        train_throughput_stats: Mean throughput statistics (tokens/sec, batch
            size, etc.) over
            the training split for this epoch.
        train_gradient_norms: Per-component gradient norms recorded during the
            training pass.
        val_loss_metrics: Mean loss components over the validation split for
            this epoch.
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
        step_throughput_stats: Per-step throughput stats appended on each
            flush.
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
