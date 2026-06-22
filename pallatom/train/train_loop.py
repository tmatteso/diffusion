"""Training loop implementation for the diffusion model.

Provides functions for checkpoint I/O, gradient accumulation, evaluation,
metric logging, and the main per-epoch training loop.
"""

import argparse
import contextlib
import dataclasses
import math
import os
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import NoReturn, cast

import structlog
import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from architecture.losses import (
    atom_loss,
    distogram_loss_atom,
    distogram_loss_residue,
    med_loss,
    seq_ce_loss,
    smooth_lddt_loss,
)
from architecture.main_trunk import MainTrunk, PredictedOutputs
from beartype import beartype
from einops import reduce
from helpers.alignment import kabsch_align
from helpers.context_managers import (
    DDPNoSync,
    DistProcessGroup,
    FatalOnError,
    StepContext,
    StructlogConfig,
)
from helpers.data import (
    Distogram,
    FeaturizedBatch,
    apply_conditioning_dropout,
    make_bucketed_data_loaders,
)
from helpers.useful_objects import (
    ComponentNorms,
    EpochMetrics,
    LossMetrics,
    ModelSetup,
    StepProgress,
    ThroughputStatistics,
)
from jaxtyping import Float, Int, jaxtyped
from structlog.typing import FilteringBoundLogger
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from train.train_config import TrainArgs, TrainConfig


@dataclasses.dataclass
class Checkpoint:
    """Serialisable snapshot of model, optimizer, and scheduler state.

    Attributes:
        model: Model parameter state dict.
        optimizer: Optimizer state dict.
        scheduler: LR scheduler state dict.
        best_val_loss: Best validation loss seen so far.
    """

    model: dict[str, Float[torch.Tensor, "..."]]
    optimizer: dict[str, Float[torch.Tensor, "..."]]
    scheduler: dict[str, Float[torch.Tensor, "..."]]
    best_val_loss: Float[torch.Tensor, ""]


def load_checkpoint(
    model_params: ModelSetup,
    rank: int,
    log: FilteringBoundLogger,
) -> tuple[ModelSetup, Float[torch.Tensor, ""]]:
    """Load a checkpoint and restore all mutable training state.

    Args:
        model_params: ModelSetup holding the model, optimizer, and scheduler to
            restore.
        rank: Current process rank; info log emitted only on rank 0.
        log: Bound structlog logger.

    Returns:
        The same ``model_params`` with model, optimizer, and scheduler state
        restored.
    """
    path: Path = model_params.tcfg.checkpoint.checkpoint_path
    raw: dict[str, object] = cast(
        dict[str, object],
        torch.load(path, map_location=model_params.device, weights_only=True),
    )
    ckpt = Checkpoint(
        model=cast(dict[str, Float[torch.Tensor, "..."]], raw["model"]),
        optimizer=cast(
            dict[str, Float[torch.Tensor, "..."]],
            raw["optimizer"],
        ),
        scheduler=cast(
            dict[str, Float[torch.Tensor, "..."]],
            raw["scheduler"],
        ),
        best_val_loss=cast(Float[torch.Tensor, ""], raw["best_val_loss"]),
    )
    _ = (
        cast(MainTrunk, model_params.model.module).load_state_dict(
            ckpt.model,
        )
        if isinstance(model_params.model, DDP)
        else model_params.model.load_state_dict(ckpt.model)
    )
    model_params.optimizer.load_state_dict(ckpt.optimizer)
    model_params.scheduler.load_state_dict(ckpt.scheduler)
    if rank == 0:
        log.info("resumed from checkpoint", path=path)
    return model_params, ckpt.best_val_loss


def save_checkpoint(
    model_params: ModelSetup,
    rank: int,
    log: FilteringBoundLogger,
    best_val_loss: Float[torch.Tensor, ""],
) -> None:
    """Save a checkpoint with all mutable training state.

    Only rank 0 writes to disk; other ranks return immediately so there are no
    concurrent-write races under DDP. The DDP wrapper is stripped before
    calling
    ``state_dict()`` so the checkpoint is loadable by ``load_checkpoint``
    regardless
    of whether the next run uses DDP.

    Args:
        model_params: ModelSetup holding the model, optimizer, and scheduler to
            save.
        rank: Current process rank; I/O is performed only on rank 0.
        log: Bound structlog logger.
        best_val_loss: best val loss encountered so far.
    """
    path: Path = model_params.tcfg.checkpoint.checkpoint_path
    if rank != 0:
        return
    if isinstance(model_params.model, DDP):
        inner = cast(
            MainTrunk,
            model_params.model.module,
        )
    else:
        inner: MainTrunk = model_params.model
    ckpt = Checkpoint(
        model=cast(
            dict[str, Float[torch.Tensor, "..."]],
            inner.state_dict(),
        ),
        optimizer=cast(
            dict[str, Float[torch.Tensor, "..."]],
            model_params.optimizer.state_dict(),
        ),
        scheduler=cast(
            dict[str, Float[torch.Tensor, "..."]],
            model_params.scheduler.state_dict(),
        ),
        best_val_loss=best_val_loss,
    )
    torch.save(
        {
            "model": ckpt.model,
            "optimizer": ckpt.optimizer,
            "scheduler": ckpt.scheduler,
            "best_val_loss": ckpt.best_val_loss,
        },
        path,
    )
    log.info("saved checkpoint", path=path)


@jaxtyped(typechecker=beartype)
def take_step(
    *,
    batch: FeaturizedBatch,
    model_params: ModelSetup,
    train_mode: bool,
    grad_scale: float = 1.0,
) -> tuple[LossMetrics, ThroughputStatistics]:
    """Forward and backward pass for one micro-batch.

    The caller owns ``optimizer.zero_grad()``, ``clip_grad_norm_``, and
    ``optimizer.step()``.  Pass ``accum_steps`` as ``grad_scale`` so that
    accumulated gradients match a single large-batch backward.

    Args:
        batch: Pre-featurized micro-batch produced by FeaturizeCollate.
        model_params: Bundled model, optimizer, config, and device.
        train_mode: If True, enables dropout, conditioning dropout, and backward
            pass.
        grad_scale: Divide total loss by this value before backward (default
            1.0).

    Returns:
        Tuple of step-level loss metrics and throughput statistics.
    """
    t0 = time.perf_counter()

    sigma_data = model_params.tcfg.noise.sigma_data

    lp = model_params.tcfg.loss

    cpu_batch: FeaturizedBatch = batch
    if train_mode:
        cpu_batch = apply_conditioning_dropout(
            cpu_batch,
            p_distogram=model_params.tcfg.conditioning_dropout.p_distogram,
            p_atom=model_params.tcfg.conditioning_dropout.p_atom,
            p_seq=model_params.tcfg.conditioning_dropout.p_seq,
            device="cpu",
        )

    featurized_batch: FeaturizedBatch = cpu_batch.to(
        model_params.device,
        non_blocking=True,
    )

    with StepContext(model=model_params.model, train_mode=train_mode):
        # Elucidating diffusion model loss weighting:
        lambda_sigma_loss_weight: Float[torch.Tensor, "B"] = (
            featurized_batch.t_hat**2 + sigma_data**2
        ) / (featurized_batch.t_hat * sigma_data) ** 2

        pred_outputs = cast(
            PredictedOutputs,
            model_params.model(featurized_batch),
        )

        Kabsch_aligned_MSE_loss: Float[torch.Tensor, ""] = atom_loss(
            pred_outputs.r_denoised,
            featurized_batch.r_gt,
            featurized_batch.atom5_mask,
            lambda_sigma_weight=lambda_sigma_loss_weight,
        ).mean()

        intermediate_loss = med_loss(
            r_denoised_blocks=pred_outputs.intermediate_denoised_coord_stack,
            logits_aa_blocks=pred_outputs.intermediate_pred_aa_logit_stack,
            batch=featurized_batch,
            loss_params=model_params.tcfg.loss,
            lambda_sigma_weight=lambda_sigma_loss_weight,
        )
        gt_res_bin_idx: Int[
            torch.Tensor,
            "B N_res N_res",
        ] = featurized_batch.gt_res_distogram.argmax(dim=-1).clamp(
            0,
            pred_outputs.residue_distogram_logits.size(-1) - 1,
        )

        residue_distogram_loss: Float[torch.Tensor, ""] = (
            distogram_loss_residue(
                pred_outputs.residue_distogram_logits,
                gt_res_bin_idx,
                featurized_batch.f_pseudo_beta_mask.bool(),
            ).mean()
        )

        atom_distogram_loss: Float[torch.Tensor, ""] = distogram_loss_atom(
            pred_outputs.atom_distogram_logits,
            featurized_batch.gt_atom_distogram_sparse,
            featurized_batch.gt_atom_distogram_mask_sparse,
        ).mean()

        lddt_loss: Float[torch.Tensor, ""] = smooth_lddt_loss(
            pred_outputs.r_denoised,
            featurized_batch.r_gt,
            featurized_batch.atom5_mask,
            cutoff=float(lp.smooth_lddt_cutoff),
        )
        CE_loss: Float[torch.Tensor, ""] = seq_ce_loss(
            pred_outputs.seq_logits,
            featurized_batch.aa_indices,
        )

        total_loss: Float[torch.Tensor, ""] = (
            lp.lam * Kabsch_aligned_MSE_loss
            + lp.alpha_0 * CE_loss
            + lp.alpha_1 * lddt_loss
            + lp.alpha_2 * residue_distogram_loss
            + lp.alpha_3 * atom_distogram_loss
            + lp.alpha_4 * intermediate_loss
        )
    if train_mode:
        torch.autograd.backward([total_loss / grad_scale])

    (r_aligned,) = kabsch_align(  # pylint: disable=unpacking-non-sequence
        featurized_batch.r_gt,
        pred_outputs.r_denoised,
        weights=featurized_batch.atom5_mask.float(),
        return_transform=False,
    )

    diff: Float[torch.Tensor, "B N_atom 3"] = (
        pred_outputs.r_denoised - r_aligned
    )
    sq: Float[torch.Tensor, "B N_atom"] = (diff * diff).sum(dim=-1)
    m: Float[torch.Tensor, "B N_atom"] = featurized_batch.atom5_mask.float()
    rmsd: Float[torch.Tensor, ""] = (
        (sq * m).sum() / m.sum().clamp(min=1)
    ).sqrt()

    t1 = time.perf_counter()
    step_time = t1 - t0

    b_size, n_res = featurized_batch.f_residue_idx.shape
    actual_residues = int(featurized_batch.f_pseudo_beta_mask.sum().item())
    actual_atoms = int(featurized_batch.atom5_mask.sum().item())

    return LossMetrics(
        total_loss=total_loss,
        Kabsch_aligned_MSE_loss=Kabsch_aligned_MSE_loss,
        CE_loss=CE_loss,
        smooth_lddt_loss=lddt_loss,
        res_distogram_loss=residue_distogram_loss,
        atom_distogram_loss=atom_distogram_loss,
        intermediate_loss=intermediate_loss,
        RMSD=rmsd,
    ), ThroughputStatistics(
        avg_batch_size=torch.tensor(float(b_size)),
        token_pack_rate=torch.tensor(actual_residues / (b_size * n_res)),
        residues_per_sec=torch.tensor(actual_residues / step_time),
        atoms_per_sec=torch.tensor(actual_atoms / step_time),
    )


def process_accum_window(
    micro_buffer: list[FeaturizedBatch],
    n_proteins_per_batch: list[int],
    model_params: ModelSetup,
) -> tuple[LossMetrics, ThroughputStatistics]:
    """Forward + backward over one accumulation window; returns protein metrics.

    Each micro-batch's loss is scaled by ``total_proteins / n_proteins_i`` so
    that the
    accumulated gradient is equivalent to a single large-batch backward over
    all proteins
    in the window.  Metrics are averaged with the same protein-count weights.

    The ``no_sync()`` context manager is used on all but the last micro-batch
    when the
    model exposes it (DDP), so gradient all-reduces happen only once per
    window.

    Args:
        micro_buffer: Micro-batches to process.
        n_proteins_per_batch: Protein count per micro-batch
            (``batch.atom_positions.shape[0]``).
        model_params: Model and associated configuration (plain ``MainTrunk``
            or DDP-wrapped).

    Returns:
        Protein-count-weighted loss metrics and throughput statistics
        aggregated over all
        micro-batches in the window.
    """
    total_proteins: int = sum(n_proteins_per_batch)
    n_micro: int = len(micro_buffer)
    loss_sums: LossMetrics = LossMetrics.zero_init(model_params.device)
    tput_sums: ThroughputStatistics = ThroughputStatistics.zero_init(
        model_params.device,
    )

    for micro_idx, (mb, n_proteins) in enumerate(
        zip(micro_buffer, n_proteins_per_batch, strict=False),
    ):
        is_last = micro_idx == n_micro - 1
        grad_scale: float = total_proteins / n_proteins
        with DDPNoSync(model_params.model, is_last=is_last):
            loss_metrics, throughput_statistics = take_step(
                batch=mb,
                model_params=model_params,
                grad_scale=grad_scale,
                train_mode=True,
            )
        loss_sums += loss_metrics * n_proteins
        tput_sums += throughput_statistics * n_proteins

    loss_sums /= total_proteins
    tput_sums /= total_proteins

    return (
        loss_sums,
        tput_sums,
    )


@torch.no_grad()
def evaluate(
    loader: torch.utils.data.DataLoader[FeaturizedBatch],
    model_params: ModelSetup,
    log: FilteringBoundLogger,
) -> tuple[LossMetrics, ThroughputStatistics]:
    """Full-dataset evaluation pass.

    Args:
        loader: DataLoader yielding ProteinBatch batches for evaluation.
        model_params: Bundled model, optimizer, config, and device.
        log: Bound structlog logger.

    Returns:
        Tuple of mean LossMetrics and mean ThroughputStatistics over the full
        dataset, averaged across all batches and DDP ranks.
    """
    loss_sums = LossMetrics.zero_init(model_params.device)
    tput_sums = ThroughputStatistics.zero_init(model_params.device)

    n_batches = 0
    is_ddp: bool = dist.is_initialized()
    world_size: int = dist.get_world_size() if is_ddp else 1
    rank: int = dist.get_rank() if is_ddp else 0

    log.info("evaluate_start", n_batches=len(loader))

    pbar: tqdm[FeaturizedBatch] = (  # pylint: disable=unsubscriptable-object
        tqdm(
            cast(Iterable[FeaturizedBatch], loader),
            desc="evaluate",
            total=len(loader),
            leave=False,
            unit="batch",
            disable=(rank != 0),
        )
    )
    with DistProcessGroup.guard():
        for batch in pbar:
            loss_metrics, throughput_statistics = take_step(
                batch=batch,
                model_params=model_params,
                grad_scale=0.0,
                train_mode=False,
            )
            loss_sums += loss_metrics
            tput_sums += throughput_statistics
            n_batches += 1
    pbar.close()

    n = max(n_batches, 1)
    divisor = n * world_size

    if is_ddp:
        loss_sums.all_reduce_()
        tput_sums.all_reduce_()

    log.info(
        "evaluate_complete",
        n_batches=n_batches,
        world_size=world_size,
    )

    loss_sums /= divisor
    tput_sums /= divisor

    return (loss_sums, tput_sums)


def component_grad_norms(model: MainTrunk | DDP) -> ComponentNorms:
    """Return gradient L2 norm for each named MainTrunk sub-module.

    Computes per-component norms from pre-clip gradients.  Call this after the
    backward pass and before ``clip_grad_norm_``.

    Args:
        model: Plain ``MainTrunk`` or DDP-wrapped model; ``.module`` is
            unwrapped
            automatically.

    Returns:
        ComponentNorms with the L2 norm of all gradients in each sub-module
        (0.0 when no grads exist).
    """
    if isinstance(model, DDP):
        inner = cast(MainTrunk, getattr(model, "module"))  # noqa: B009
    else:
        inner: MainTrunk = model

    def _norm(attr: str) -> Float[torch.Tensor, ""]:
        sub_module = cast(nn.Module, getattr(inner, attr))
        grad_norms: list[torch.Tensor] = [
            torch.sqrt(reduce(parameter.grad.detach() ** 2, "... -> ", "sum"))
            for parameter in sub_module.parameters()
            if parameter.grad is not None
        ]
        return torch.sqrt(reduce(torch.stack(grad_norms) ** 2, "n -> ", "sum"))

    return ComponentNorms(
        template_embedder=_norm("template_embedder"),
        atom_encoder=_norm("atom_encoder"),
        atom_decoders=_norm("atom_decoders"),
        residue_distogram_head=_norm("residue_distogram_head"),
        atom_distogram_head=_norm("atom_distogram_head"),
        inter_proj_seq=_norm("inter_proj_seq"),
        inter_seq_logits=_norm("inter_seq_logits"),
        proj_seq=_norm("proj_seq"),
        seq_logits=_norm("seq_logits"),
    )


def optimizer_step(
    micro_buffer: list[FeaturizedBatch],
    n_proteins_buffer: list[int],
    model_params: ModelSetup,
    global_step: int,
) -> tuple[LossMetrics, ThroughputStatistics, ComponentNorms, int]:
    """Run one accumulation window, clip gradients, and step the optimizer.

    Processes all micro-batches in ``micro_buffer`` via
    ``process_accum_window``
    (which accumulates gradients across micro-batches without stepping),
    captures
    per-component gradient L2 norms before clipping, clips the total gradient
    norm
    to ``training_cfg.grad_clip``, steps the optimizer, and zeros gradients.
    The LR scheduler is not stepped here.

    Args:
        micro_buffer: Micro-batches accumulated for this optimizer step.
        n_proteins_buffer: Protein count per micro-batch, aligned with
            ``micro_buffer``.
        model_params: Model, optimizer, scheduler, and training configuration.
        global_step: Current global step count before this flush.

    Returns:
        A 4-tuple ``(loss_metrics, throughput_statistics, component_norms,
        next_step)``
        where ``loss_metrics`` contains mean per-metric losses over the window,
        ``throughput_statistics`` contains batch size and tokens-per-second
        stats,
        ``component_norms`` contains per-module gradient L2 norms measured
        before clipping,
        and ``next_step`` is ``global_step + 1``.
    """
    loss_metrics, throughput_statistics = process_accum_window(
        micro_buffer=micro_buffer,
        n_proteins_per_batch=n_proteins_buffer,
        model_params=model_params,
    )
    component_norms = component_grad_norms(model_params.model)
    tp = model_params.tcfg.training
    _ = nn.utils.clip_grad_norm_(
        model_params.model.parameters(),
        tp.grad_clip if tp.grad_clip is not None else float("inf"),
    )

    _ = (
        model_params.optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    )
    model_params.optimizer.zero_grad()
    return loss_metrics, throughput_statistics, component_norms, global_step + 1


def log_epoch(
    *,
    epoch_metrics: EpochMetrics,
    model_params: ModelSetup,
    log: FilteringBoundLogger,
    do_log: bool = True,
) -> None:
    """Log metrics and save checkpoints for one completed epoch.

    Writes structlog entries for train and val splits, optionally pushes to
    W&B, saves the best-validation checkpoint, and saves a periodic epoch
    checkpoint when ``tcfg.checkpoint.save_every`` divides ``epoch``. Handles
    both plain ``nn.Module`` and DDP-wrapped models (accesses ``.module`` when
    present). Pass ``do_log=False`` on non-rank-0 workers to skip all I/O.

    Checkpoints include model weights, optimizer state, scheduler state, and
    epoch number so training can be resumed exactly via
    ``TrainingParams.resume_checkpoint``.

    Args:
        epoch_metrics: Aggregated metrics for this epoch, including epoch
            number,
            global step, train/val loss metrics, throughput statistics, and
            gradient norms.
        model_params: Model, optimizer, scheduler, and training configuration
            for this run.
        log: Bound structlog logger.
        do_log: If ``False``, skip all I/O (use on non-rank-0 workers).
    """
    if not do_log:
        return  # do nothing for non rank 0 workers
    lg = model_params.tcfg.logging

    val_dict = epoch_metrics.val_loss_metrics.to_float_dict()
    train_dict = epoch_metrics.train_loss_metrics.to_float_dict()
    thru_stats_dict = epoch_metrics.train_throughput_stats.to_float_dict()
    gradient_norm_dict = epoch_metrics.train_gradient_norms.to_float_dict()

    log.info(
        "train",
        epoch=epoch_metrics.epoch,
        **train_dict,
    )
    log.info(
        "throughput_statistics",
        epoch=epoch_metrics.epoch,
        **thru_stats_dict,
    )
    log.info(
        "gradient_norms",
        epoch=epoch_metrics.epoch,
        **gradient_norm_dict,
    )
    log.info(
        "val",
        epoch=epoch_metrics.epoch,
        **val_dict,
    )

    if lg.use_wandb:
        wandb.log(
            {
                "epoch": epoch_metrics.epoch,
                "global_step": epoch_metrics.global_step,
                **{f"train/{k}": v for k, v in train_dict.items()},
                **{
                    f"throughput_statistics/{k}": v
                    for k, v in thru_stats_dict.items()
                },
                **{
                    f"gradient_norms/{k}": v
                    for k, v in gradient_norm_dict.items()
                },
                **{f"val/{k}": v for k, v in val_dict.items()},
            },
        )


def flush_micro_buffer(
    micro_buffer: list[FeaturizedBatch],
    n_proteins_buffer: list[int],
    model_params: ModelSetup,
    step: StepProgress,
) -> StepProgress:
    """Run one optimizer step over buffered micro-batches.

    Appends loss metrics, throughput stats, gradient norms, and protein
    counts to the corresponding lists in ``step``, increments
    ``step.global_step``, and returns ``step``.

    Args:
        micro_buffer: Accumulated micro-batches to flush.
        n_proteins_buffer: Per-micro-batch protein counts.
        model_params: Bundled model, optimizer, scheduler, and config.
        step: Per-epoch accumulator holding running totals and the
            progress bar.

    Returns:
        The same ``step`` object with updated ``global_step`` and
        appended metrics.
    """
    loss_metrics, throughput_stats, component_norms, new_global_step = (
        optimizer_step(
            micro_buffer=micro_buffer,
            n_proteins_buffer=n_proteins_buffer,
            model_params=model_params,
            global_step=step.global_step,
        )
    )
    loss_dict: dict[str, float] = loss_metrics.to_float_dict()

    step.pbar.update(
        len(n_proteins_buffer),
    )  # pyright: ignore[reportUnusedCallResult]
    if step.rank == 0:
        step.pbar.set_postfix(  # pyright: ignore[reportUnknownMemberType]
            {k: f"{v:.2f}" for k, v in loss_dict.items()},
        )
    step.step_loss_metrics.append(loss_metrics)
    step.step_throughput_stats.append(throughput_stats)
    step.step_component_norms.append(component_norms)
    step.step_n_proteins.append(sum(n_proteins_buffer))
    return dataclasses.replace(step, global_step=new_global_step)


def collect_distributed_vars() -> tuple[int, int, int]:
    """Return (rank, world_size, local_rank) for the current DDP context."""
    if not dist.is_initialized():
        return 0, 1, 0
    return (
        dist.get_rank(),
        dist.get_world_size(),
        int(os.environ.get("LOCAL_RANK", "0")),
    )


def train(
    best_val_loss: Float[torch.Tensor, ""],
    train_loader: torch.utils.data.DataLoader[FeaturizedBatch],
    test_loader: torch.utils.data.DataLoader[FeaturizedBatch],
    model_params: ModelSetup,
    log: FilteringBoundLogger,
) -> None:
    """Training loop for the MainTrunk diffusion model; works with DDP.

    When called inside ``with DistProcessGroup():`` the loop detects an
    initialised process group, wraps the model in DDP, and divides the token
    budget by world_size.  Without a process group it runs as a single-device
    loop with rank 0 / world_size 1.

    ``tcfg.training.accumulated_token_budget`` is divided by ``world_size`` to
    get the per-rank token threshold; micro-batches accumulate until their
    combined token count hits that threshold before each optimizer step.

    Args:
        best_val_loss: Incumbent best validation loss; updated and returned
            implicitly via ``log_epoch``.
        train_loader: DataLoader for training batches.
        test_loader: DataLoader for evaluation batches.
        model_params: Bundled model, optimizer, scheduler, and config.
        log: Bound structlog logger.
    """
    rank: int
    world_size: int
    local_rank: int
    rank, world_size, local_rank = collect_distributed_vars()

    if dist.is_initialized():
        ddp_wrapped: DDP = DDP(model_params.model, device_ids=[local_rank])
        model_params = dataclasses.replace(model_params, model=ddp_wrapped)

    tp = model_params.tcfg.training
    per_rank_token_budget: int = max(
        1,
        tp.accumulated_token_budget // world_size,
    )
    if rank == 0:
        log.info("training", ddp=dist.is_initialized())
        log.info(
            "gradient_accumulation",
            token_budget_per_rank=per_rank_token_budget,
            global_token_budget=tp.accumulated_token_budget,
        )

    global_step = 0

    for epoch in range(1, tp.num_epochs + 1):
        _ = (
            model_params.model.train()
        )  # returns "DistributedDataParallel | MainTrunk"
        n_batches = 0
        micro_buffer: list[FeaturizedBatch] = []
        n_proteins_buffer: list[int] = []
        accum_tokens: int = 0
        model_params.optimizer.zero_grad()

        # Calling iter() triggers ShardDataLoader.__iter__, which dequeues
        # the current-epoch plan and updates _cached_len before we read it.
        train_iter: Iterator[FeaturizedBatch] = iter(
            cast(Iterable[FeaturizedBatch], train_loader),
        )
        estimated_steps: int = math.ceil(
            len(train_loader)
            * model_params.tcfg.train_loader.token_budget
            / per_rank_token_budget,
        )
        pbar: tqdm[NoReturn] = tqdm(  # pylint: disable=unsubscriptable-object
            desc=f"Epoch {epoch:03d}/{tp.num_epochs}",
            total=estimated_steps,
            leave=True,
            unit="step",
            disable=(rank != 0),
        )
        step = StepProgress(
            global_step=global_step,
            rank=rank,
            pbar=pbar,
            step_loss_metrics=[],
            step_throughput_stats=[],
            step_component_norms=[],
            step_n_proteins=[],
        )

        for batch in train_iter:

            n_proteins: int = int(batch.r_gt.shape[0])
            n_all_tokens: int = int(
                batch.r_gt.shape[0] * batch.r_gt.shape[1],
            )

            # if adding this batch would push tokens over the budget, flush.
            if (
                micro_buffer
                and accum_tokens + n_all_tokens > per_rank_token_budget
            ):
                step = flush_micro_buffer(
                    micro_buffer,
                    n_proteins_buffer,
                    model_params,
                    step,
                )
                n_batches += 1
                micro_buffer, n_proteins_buffer, accum_tokens = [], [], 0

            micro_buffer.append(batch)
            n_proteins_buffer.append(n_proteins)
            accum_tokens += n_all_tokens

        # Flush any remaining micro-batches at epoch end,
        # regardless of token count.
        if micro_buffer:
            step = flush_micro_buffer(
                micro_buffer,
                n_proteins_buffer,
                model_params,
                step,
            )
            n_batches += 1
            micro_buffer = []

        model_params.scheduler.step()

        epoch_val_metrics, _ = evaluate(
            loader=test_loader,
            model_params=model_params,
            log=log,
        )

        global_step = step.global_step
        epoch_metrics = EpochMetrics(
            epoch=epoch,
            global_step=global_step,
            train_loss_metrics=LossMetrics.weighted_avg(
                step.step_loss_metrics,
                step.step_n_proteins,
            ),
            train_throughput_stats=ThroughputStatistics.weighted_avg(
                step.step_throughput_stats,
                step.step_n_proteins,
            ),
            train_gradient_norms=ComponentNorms.weighted_avg(
                step.step_component_norms,
                step.step_n_proteins,
            ),
            val_loss_metrics=epoch_val_metrics,
        )
        log_epoch(
            epoch_metrics=epoch_metrics,
            model_params=model_params,
            log=log,
            do_log=(rank == 0),
        )
        if epoch_metrics.val_loss_metrics.total_loss < best_val_loss:
            best_val_loss = epoch_metrics.val_loss_metrics.total_loss
            save_checkpoint(
                best_val_loss=best_val_loss,
                model_params=model_params,
                rank=rank,
                log=log,
            )
    pbar.close()  # pyright: ignore[reportPossiblyUnboundVariable]


def _parse_args() -> TrainArgs:
    """Build the argument parser and return a fully typed TrainArgs.

    Returns:
        Fully typed TrainArgs populated from sys.argv.
    """
    parser = argparse.ArgumentParser(description="Train PallAtom")
    parser.add_argument(  # pyright: ignore[reportUnusedCallResult]
        "--dataset_jsonl",
        required=True,
        type=Path,
        help="path to proteins.jsonl",
    )
    parser.add_argument(  # pyright: ignore[reportUnusedCallResult]
        "--shard_dir",
        required=True,
        type=Path,
        help="path to generate sharded protein dataset",
    )
    parser.add_argument(  # pyright: ignore[reportUnusedCallResult]
        "--keys_for_splits_json",
        required=True,
        type=Path,
        help="path to splits.json",
    )
    parser.add_argument(  # pyright: ignore[reportUnusedCallResult]
        "--config",
        required=True,
        type=Path,
        help="path to TrainConfig JSON",
    )
    parser.add_argument(  # pyright: ignore[reportUnusedCallResult]
        "--structlog_jsonl",
        required=True,
        type=Path,
        help="path to write structured JSON log",
    )
    parser.add_argument(  # pyright: ignore[reportUnusedCallResult]
        "--ddp",
        action="store_true",
        help="DistributedDataParallel training",
    )
    parser.add_argument(  # pyright: ignore[reportUnusedCallResult]
        "--debug_run",
        action="store_true",
        help="restrict to 252 proteins for fast iteration",
    )
    ns = parser.parse_args()
    return TrainArgs(
        dataset_jsonl=cast("Path", ns.dataset_jsonl),
        shard_dir=cast("Path", ns.shard_dir),
        keys_for_splits_json=cast("Path", ns.keys_for_splits_json),
        config=cast("Path", ns.config),
        structlog_jsonl=cast("Path", ns.structlog_jsonl),
        ddp=cast("bool", ns.ddp),
        debug_run=cast("bool", ns.debug_run),
    )


def main(args: TrainArgs, tcfg: TrainConfig) -> None:
    """Entry point for training; dispatches to DDP or not based on args.ddp.

    Uses ExitStack so both paths share all setup code after context init.
    DistProcessGroup is only entered under --ddp; device/rank/world_size are
    resolved to plain values before the shared path begins.
    """
    with contextlib.ExitStack() as stack:
        if args.ddp:
            dpg = stack.enter_context(DistProcessGroup("nccl"))
            rank: int = dpg.rank
            is_rank_zero: bool = dpg.is_rank_zero
            device: torch.device = torch.device(dpg.device)
        else:
            rank = 0
            is_rank_zero = True
            device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu",
            )

        _ = stack.enter_context(
            StructlogConfig(
                is_rank_zero=is_rank_zero,
                log_file=args.structlog_jsonl,
            ),
        )
        _ = stack.enter_context(FatalOnError())

        log = cast(FilteringBoundLogger, structlog.get_logger())

        train_loader, val_loader, _ = make_bucketed_data_loaders(
            cfg=tcfg,
            extra_train_args=args,
        )

        mp = tcfg.model
        tp = tcfg.training
        model: MainTrunk = MainTrunk(
            model_params=mp,
            res_distogram_params=tcfg.distogram_res,
            atom_distogram_params=tcfg.distogram_atom,
            noise_params=tcfg.noise,
        ).to(device)

        optimizer = Adam(
            model.parameters(),
            lr=tp.lr,
            weight_decay=tp.weight_decay,
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=tp.num_epochs,
            eta_min=tp.lr * 0.01,
        )

        dr = tcfg.distogram_res
        da = tcfg.distogram_atom
        distogram_res = Distogram(
            n_bins=dr.n_bins - 1,
            min_dist=dr.min_dist,
            max_dist=dr.max_dist,
            overflow_bin=True,
        )
        distogram_atom = Distogram(
            n_bins=da.n_bins,
            min_dist=da.min_dist,
            max_dist=da.max_dist,
            overflow_bin=False,
        )

        model_params = ModelSetup(
            model=model,
            tcfg=tcfg,
            distogram_res=distogram_res,
            distogram_atom=distogram_atom,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        if tcfg.training.pretrained_weights is not None:
            model_params, best_val_loss = load_checkpoint(
                model_params=model_params,
                rank=rank,
                log=log,
            )
        else:
            best_val_loss = torch.tensor(float("inf"))

        if is_rank_zero and tcfg.logging.use_wandb:
            _ = wandb.init(
                project=tcfg.logging.wandb_project,
                config=tcfg.model_dump(),
            )

        train(
            best_val_loss=best_val_loss,
            train_loader=train_loader,
            test_loader=val_loader,
            model_params=model_params,
            log=log,
        )


if __name__ == "__main__":
    _args = _parse_args()
    with _args.config.open(encoding="utf-8") as file:
        _tcfg = TrainConfig.model_validate_json(file.read())
    main(_args, _tcfg)
