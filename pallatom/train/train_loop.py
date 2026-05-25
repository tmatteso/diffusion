"""Training loop implementation for the diffusion model."""

import argparse
import contextlib
import dataclasses
import math
import os
import time
from typing import NoReturn, cast

import structlog
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import wandb
from architecture.losses import (
    atom_loss,
    distogram_loss_atom,
    distogram_loss_residue,
    smooth_lddt_loss,
)
from architecture.main_trunk import MainTrunk, PredictedOutputs
from beartype import beartype
from einops import rearrange, reduce
from helpers.alignment import kabsch_align
from helpers.bucketed_sampler import BucketedBatchSampler
from helpers.context_managers import DistProcessGroup, FatalOnError, StepContext, StructlogConfig
from helpers.data import make_bucketed_data_loaders
from helpers.featurize import (
    Distogram,
    FeaturizedBatch,
    ProteinBatch,
    apply_conditioning_dropout,
    featurize_batch,
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
from train.train_config import TrainConfig


def _mask_seq_target(aa_indices: Int[torch.Tensor, "B N_res"]) -> Int[torch.Tensor, "B N_res"]:
    """Replace mask-token index 20 with -100 so CE loss ignores dropped positions."""
    return aa_indices.masked_fill(aa_indices == 20, -100)


def load_checkpoint(
    model_params: ModelSetup,
    rank: int,
    log: FilteringBoundLogger,
) -> tuple[ModelSetup, Float[torch.Tensor, ""]]:
    """Load a checkpoint and restore all mutable training state.

    Args:
        model_params: ModelSetup holding the model, optimizer, and scheduler to restore.
        rank: Current process rank; info log emitted only on rank 0.
        log: Bound structlog logger.

    Returns:
        The same ``model_params`` with model, optimizer, and scheduler state restored.
    """
    path: str = model_params.tcfg.checkpoint.checkpoint_path
    ckpt = torch.load(path, map_location=model_params.device)
    if isinstance(model_params.model, DDP):
        model_params.model.module.load_state_dict(ckpt["model"])
    else:
        model_params.model.load_state_dict(ckpt["model"])

    model_params.optimizer.load_state_dict(ckpt["optimizer"])
    model_params.scheduler.load_state_dict(ckpt["scheduler"])
    best_val_loss = ckpt["best_val_loss"]
    if rank == 0:
        log.info("resumed from checkpoint", path=path)
    return model_params, best_val_loss


def save_checkpoint(
    model_params: ModelSetup,
    rank: int,
    log: FilteringBoundLogger,
    best_val_loss: Float[torch.Tensor, ""],
) -> None:
    """Save a checkpoint with all mutable training state.

    Only rank 0 writes to disk; other ranks return immediately so there are no
    concurrent-write races under DDP.  The DDP wrapper is stripped before calling
    ``state_dict()`` so the checkpoint is loadable by ``load_checkpoint`` regardless
    of whether the next run uses DDP.

    Args:
        model_params: ModelSetup holding the model, optimizer, and scheduler to save.
        rank: Current process rank; I/O is performed only on rank 0.
        log: Bound structlog logger.
        best_val_loss: best val loss encountered so far.
    """
    path: str = model_params.tcfg.checkpoint.checkpoint_path
    if rank != 0:
        return
    inner: nn.Module = (
        model_params.model.module if isinstance(model_params.model, DDP) else model_params.model
    )
    ckpt = {
        "model": inner.state_dict(),
        "optimizer": model_params.optimizer.state_dict(),
        "scheduler": model_params.scheduler.state_dict(),
        "best_val_loss": best_val_loss,
    }
    torch.save(ckpt, path)
    log.info("saved checkpoint", path=path)


@jaxtyped(typechecker=beartype)
def take_step(
    *,
    batch: ProteinBatch,
    model_params: ModelSetup,
    train: bool,
    grad_scale: float = 1.0,
) -> tuple[LossMetrics, ThroughputStatistics]:
    """Forward and backward pass for one micro-batch.

    The caller owns ``optimizer.zero_grad()``, ``clip_grad_norm_``, and
    ``optimizer.step()``.  Pass ``accum_steps`` as ``grad_scale`` so that
    accumulated gradients match a single large-batch backward.

    Args:
        batch: Raw protein micro-batch.
        model_params: Bundled model, optimizer, config, and device.
        train: If True, enables dropout, conditioning dropout, and backward pass.
        grad_scale: Divide total loss by this value before backward (default 1.0).

    Returns:
        Tuple of step-level loss metrics and throughput statistics.
    """
    featurized_batch: FeaturizedBatch = featurize_batch(
        batch, model_params.tcfg, model_params.distogram_res, model_params.distogram_atom
    )
    lp = model_params.tcfg.loss

    if train:
        # Save orig amino-acid indices before conditioning dropout replaces some with <MASK> (20).
        orig_aa_indices: Int[torch.Tensor, "B N_res"] = featurized_batch.aa_indices.clone()
        featurized_batch: FeaturizedBatch = apply_conditioning_dropout(
            featurized_batch,
            p_distogram=model_params.tcfg.conditioning_dropout.p_distogram,
            p_atom=model_params.tcfg.conditioning_dropout.p_atom,
            p_seq=model_params.tcfg.conditioning_dropout.p_seq,
            device=model_params.device,
        )
        # MLM-style CE targets: original amino acid at positions where dropout placed <MASK> (20),
        # -100 (ignore_index) at positions the model can still see and at padding.
        # This forces the model to predict sequence from structure, not copy visible conditioning.
        mask_positions: Int[torch.Tensor, "B N_res"] = featurized_batch.aa_indices == 20
        aa_targets: Int[torch.Tensor, "B N_res"] = orig_aa_indices.masked_fill(
            ~mask_positions, -100
        )
        # Flat targets reused for final and intermediate CE; count guards against p_seq=0 edge case.
        flat_aa_targets: Int[torch.Tensor, "B_N_res"] = rearrange(aa_targets, "b n -> (b n)")
        n_masked_aa: Int[torch.Tensor, ""] = (flat_aa_targets >= 0).sum().clamp(min=1)

    t0 = time.perf_counter()

    with StepContext(model=model_params.model, train=train):
        pred_outputs: PredictedOutputs = model_params.model(featurized_batch)

        Kabsch_aligned_MSE_loss: Float[torch.Tensor, ""] = atom_loss(
            pred_outputs.r_denoised, featurized_batch.r_gt, featurized_batch.atom5_mask
        ).mean()

        K_unit: int = len(pred_outputs.intermediate_denoised_coord_stack)
        intermediate_med_loss: Float[torch.Tensor, ""] = torch.tensor(
            0.0, device=model_params.device
        )
        for k_idx, intermediate_denoised_coord in enumerate(
            pred_outputs.intermediate_denoised_coord_stack
        ):
            intermediate_denoised_coord: Float[torch.Tensor, "B N_atom 3"]
            gamma_K_minus_k: float = lp.gamma ** (K_unit - k_idx - 1)
            if train:
                inter_ce: Float[torch.Tensor, ""] = (
                    F.cross_entropy(
                        rearrange(
                            pred_outputs.intermediate_pred_aa_logit_stack[k_idx], "b n c -> (b n) c"
                        ),
                        flat_aa_targets,  # (b n)
                        reduction="sum",
                    )
                    / n_masked_aa
                )
            else:
                inter_ce: Float[torch.Tensor, ""] = F.cross_entropy(
                    rearrange(
                        pred_outputs.intermediate_pred_aa_logit_stack[k_idx], "b n c -> (b n) c"
                    ),
                    rearrange(_mask_seq_target(featurized_batch.aa_indices), "b n -> (b n)"),
                )

            k_loss: Float[torch.Tensor, ""] = (
                lp.lam
                * atom_loss(
                    intermediate_denoised_coord, featurized_batch.r_gt, featurized_batch.atom5_mask
                )
                + lp.alpha_0 * inter_ce
            )
            intermediate_med_loss = intermediate_med_loss + gamma_K_minus_k * k_loss
        intermediate_med_loss = (intermediate_med_loss / max(K_unit, 1)).mean()

        gt_res_bin_idx: Int[torch.Tensor, "B N_res N_res"] = (
            featurized_batch.gt_res_distogram.argmax(dim=-1).clamp(
                0, pred_outputs.residue_distogram_logits.size(-1) - 1
            )
        )

        residue_distogram_loss: Float[torch.Tensor, ""] = distogram_loss_residue(
            pred_outputs.residue_distogram_logits,
            gt_res_bin_idx,
            featurized_batch.residue_mask,
        ).mean()

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
        CE_loss: Float[torch.Tensor, ""] = F.cross_entropy(
            rearrange(pred_outputs.seq_logits, "b n c -> (b n) c"),
            rearrange(_mask_seq_target(featurized_batch.aa_indices), "b n -> (b n)"),
        )

        total_loss: Float[torch.Tensor, ""] = (
            lp.lam * Kabsch_aligned_MSE_loss
            + lp.alpha_0 * CE_loss
            + lp.alpha_1 * lddt_loss
            + lp.alpha_2 * residue_distogram_loss
            + lp.alpha_3 * atom_distogram_loss
            + lp.alpha_4 * intermediate_med_loss
        )
    if train:
        (total_loss / grad_scale).backward()

    r_aligned: Float[torch.Tensor, "B N_atom 3"]
    (r_aligned,) = kabsch_align(
        featurized_batch.r_gt,
        pred_outputs.r_denoised,
        weights=featurized_batch.atom5_mask.float(),
        return_transform=False,
    )

    diff: Float[torch.Tensor, "B N_atom 3"] = pred_outputs.r_denoised - r_aligned
    sq: Float[torch.Tensor, "B N_atom"] = (diff * diff).sum(dim=-1)
    m: Float[torch.Tensor, "B N_atom"] = featurized_batch.atom5_mask.float()
    rmsd: Float[torch.Tensor, ""] = ((sq * m).sum() / m.sum().clamp(min=1)).sqrt()

    t1 = time.perf_counter()
    step_time = t1 - t0

    b_size, n_res = featurized_batch.residue_mask.shape
    actual_residues = int(featurized_batch.residue_mask.sum().item())
    actual_atoms = int(featurized_batch.atom5_mask.sum().item())

    return LossMetrics(
        total_loss=total_loss,
        Kabsch_aligned_MSE_loss=Kabsch_aligned_MSE_loss,
        CE_loss=CE_loss,
        smooth_lddt_loss=lddt_loss,
        res_distogram_loss=residue_distogram_loss,
        atom_distogram_loss=atom_distogram_loss,
        intermediate_loss=intermediate_med_loss,
        RMSD=rmsd,
    ), ThroughputStatistics(
        avg_batch_size=torch.tensor(float(b_size)),
        token_pack_rate=torch.tensor(actual_residues / (b_size * n_res)),
        residues_per_sec=torch.tensor(actual_residues / step_time),
        atoms_per_sec=torch.tensor(actual_atoms / step_time),
    )


def process_accum_window(
    micro_buffer: list[ProteinBatch], n_proteins_per_batch: list[int], model_params: ModelSetup
) -> tuple[LossMetrics, ThroughputStatistics]:
    """Forward + backward over one accumulation window; returns protein-weighted metrics.

    Each micro-batch's loss is scaled by ``total_proteins / n_proteins_i`` so that the
    accumulated gradient is equivalent to a single large-batch backward over all proteins
    in the window.  Metrics are averaged with the same protein-count weights.

    The ``no_sync()`` context manager is used on all but the last micro-batch when the
    model exposes it (DDP), so gradient all-reduces happen only once per window.

    Args:
        micro_buffer: Micro-batches to process.
        n_proteins_per_batch: Protein count per micro-batch (``batch.atom_positions.shape[0]``).
        model_params: Model and associated configuration (plain ``MainTrunk`` or DDP-wrapped).

    Returns:
        Protein-count-weighted loss metrics and throughput statistics aggregated over all
        micro-batches in the window.
    """
    total_proteins: int = sum(n_proteins_per_batch)
    n_micro: int = len(micro_buffer)
    loss_sums: dict[str, torch.Tensor] = {
        f.name: torch.tensor(0.0) for f in dataclasses.fields(LossMetrics)
    }
    tput_sums: dict[str, torch.Tensor] = {
        f.name: torch.tensor(0.0) for f in dataclasses.fields(ThroughputStatistics)
    }

    maybe_no_sync = getattr(model_params.model, "no_sync", None)

    for micro_idx, (mb, n_proteins) in enumerate(
        zip(micro_buffer, n_proteins_per_batch, strict=False)
    ):
        is_last = micro_idx == n_micro - 1
        ctx = cast(
            contextlib.AbstractContextManager[None],  # this context manager needs to be refactored.
            (
                maybe_no_sync()
                if (not is_last and callable(maybe_no_sync))
                else contextlib.nullcontext()
            ),
        )
        grad_scale: float = total_proteins / n_proteins
        with ctx:
            loss_metrics, throughput_statistics = take_step(
                batch=mb,
                model_params=model_params,
                grad_scale=grad_scale,
                train=True,
            )
        for f in dataclasses.fields(loss_metrics):
            loss_sums[f.name] += getattr(loss_metrics, f.name) * n_proteins
        for f in dataclasses.fields(throughput_statistics):
            tput_sums[f.name] += getattr(throughput_statistics, f.name) * n_proteins

    for key, value in loss_sums.items():
        loss_sums[key] = value / total_proteins
    for key, value in tput_sums.items():
        tput_sums[key] = value / total_proteins

    return (
        LossMetrics(**loss_sums),
        ThroughputStatistics(**tput_sums),
    )


@torch.no_grad()
@jaxtyped(typechecker=beartype)
def evaluate(
    loader: torch.utils.data.DataLoader[ProteinBatch], model_params: ModelSetup
) -> tuple[LossMetrics, ThroughputStatistics]:
    """Full-dataset evaluation pass. Returns mean loss per metric."""
    loss_sums: dict[str, Float[torch.Tensor, ""]] = {
        f.name: torch.tensor(0.0) for f in dataclasses.fields(LossMetrics)
    }
    tput_sums: dict[str, Float[torch.Tensor, ""]] = {
        f.name: torch.tensor(0.0) for f in dataclasses.fields(ThroughputStatistics)
    }
    n_batches = 0

    for batch in loader:
        loss_metrics, throughput_statistics = take_step(
            batch=batch,
            model_params=model_params,
            grad_scale=0.0,
            train=False,
        )
        for f in dataclasses.fields(loss_metrics):
            loss_sums[f.name] += getattr(loss_metrics, f.name)
        for f in dataclasses.fields(throughput_statistics):
            tput_sums[f.name] += getattr(throughput_statistics, f.name)
        n_batches += 1

    if isinstance(model_params.model, DDP):
        for t in loss_sums.values():
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        for t in tput_sums.values():
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        # shouldn't these then be divided by the worldsize?

    n = max(n_batches, 1)
    return (
        LossMetrics(**{k: v / n for k, v in loss_sums.items()}),
        ThroughputStatistics(**{k: v / n for k, v in tput_sums.items()}),
    )


def component_grad_norms(model: MainTrunk | DDP) -> ComponentNorms:
    """Return gradient L2 norm for each named MainTrunk sub-module.

    Computes per-component norms from pre-clip gradients.  Call this after the
    backward pass and before ``clip_grad_norm_``.

    Args:
        model: Plain ``MainTrunk`` or DDP-wrapped model; ``.module`` is unwrapped
            automatically.

    Returns:
        ComponentNorms with the L2 norm of all gradients in each sub-module
        (0.0 when no grads exist).
    """
    inner: nn.Module = model.module if isinstance(model, DDP) else model

    def _norm(attr: str) -> Float[torch.Tensor, ""]:
        sub_module: nn.Module = getattr(inner, attr)
        grad_norms: list[torch.Tensor] = [
            torch.linalg.vector_norm(parameter.grad.detach(), 2.0)
            for parameter in sub_module.parameters()
            if parameter.grad is not None
        ]
        return torch.linalg.vector_norm(torch.stack(grad_norms), 2.0)

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
    micro_buffer: list[ProteinBatch],
    n_proteins_buffer: list[int],
    model_params: ModelSetup,
    global_step: int,
) -> tuple[LossMetrics, ThroughputStatistics, ComponentNorms, int]:
    """Run one accumulation window, clip gradients, and step the optimizer.

    Processes all micro-batches in ``micro_buffer`` via ``process_accum_window``
    (which accumulates gradients across micro-batches without stepping), captures
    per-component gradient L2 norms before clipping, clips the total gradient norm
    to ``training_cfg.grad_clip``, steps the optimizer, and zeros gradients.
    The LR scheduler is not stepped here.

    Args:
        micro_buffer: Micro-batches accumulated for this optimizer step.
        n_proteins_buffer: Protein count per micro-batch, aligned with ``micro_buffer``.
        model_params: Model, optimizer, scheduler, and training configuration.
        global_step: Current global step count before this flush.

    Returns:
        A 4-tuple ``(loss_metrics, throughput_statistics, component_norms, next_step)``
        where ``loss_metrics`` contains mean per-metric losses over the window,
        ``throughput_statistics`` contains batch size and tokens-per-second stats,
        ``component_norms`` contains per-module gradient L2 norms measured before clipping,
        and ``next_step`` is ``global_step + 1``.
    """
    loss_metrics, throughput_statistics = process_accum_window(
        micro_buffer=micro_buffer, n_proteins_per_batch=n_proteins_buffer, model_params=model_params
    )
    component_norms = component_grad_norms(model_params.model)
    tp = model_params.tcfg.training
    nn.utils.clip_grad_norm_(
        model_params.model.parameters(),
        tp.grad_clip if tp.grad_clip is not None else float("inf"),
    )

    model_params.optimizer.step()
    model_params.optimizer.zero_grad()
    return loss_metrics, throughput_statistics, component_norms, global_step + 1


@jaxtyped(typechecker=beartype)
def _wavg(values: list[Float[torch.Tensor, ""]], n_proteins: list[int]) -> Float[torch.Tensor, ""]:
    """Protein-count-weighted average of scalar tensors.

    Args:
        values: Per-step scalar tensors to average.
        n_proteins: Protein count for each step, used as weights.

    Returns:
        Weighted average scalar tensor.
    """
    total: int = sum(n_proteins)
    weights: Float[torch.Tensor, "N"] = torch.tensor(
        n_proteins, dtype=torch.float32, device=values[0].device
    )
    stacked: Float[torch.Tensor, "N"] = torch.stack(values)
    return reduce(stacked * weights, "N ->", "sum") / total


def _wavg_loss_metrics(steps: list[LossMetrics], n_proteins: list[int]) -> LossMetrics:
    """Protein-count-weighted average of per-step LossMetrics.

    Args:
        steps: Per-step LossMetrics instances.
        n_proteins: Protein count for each step.

    Returns:
        Single LossMetrics with all fields weighted-averaged.
    """
    return LossMetrics(
        total_loss=_wavg([m.total_loss for m in steps], n_proteins),
        Kabsch_aligned_MSE_loss=_wavg([m.Kabsch_aligned_MSE_loss for m in steps], n_proteins),
        CE_loss=_wavg([m.CE_loss for m in steps], n_proteins),
        smooth_lddt_loss=_wavg([m.smooth_lddt_loss for m in steps], n_proteins),
        res_distogram_loss=_wavg([m.res_distogram_loss for m in steps], n_proteins),
        atom_distogram_loss=_wavg([m.atom_distogram_loss for m in steps], n_proteins),
        intermediate_loss=_wavg([m.intermediate_loss for m in steps], n_proteins),
        RMSD=_wavg([m.RMSD for m in steps], n_proteins),
    )


def _wavg_throughput_stats(
    steps: list[ThroughputStatistics], n_proteins: list[int]
) -> ThroughputStatistics:
    """Protein-count-weighted average of per-step ThroughputStatistics.

    Args:
        steps: Per-step ThroughputStatistics instances.
        n_proteins: Protein count for each step.

    Returns:
        Single ThroughputStatistics with all fields weighted-averaged.
    """
    return ThroughputStatistics(
        avg_batch_size=_wavg([m.avg_batch_size for m in steps], n_proteins),
        token_pack_rate=_wavg([m.token_pack_rate for m in steps], n_proteins),
        residues_per_sec=_wavg([m.residues_per_sec for m in steps], n_proteins),
        atoms_per_sec=_wavg([m.atoms_per_sec for m in steps], n_proteins),
    )


def _wavg_component_norms(steps: list[ComponentNorms], n_proteins: list[int]) -> ComponentNorms:
    """Protein-count-weighted average of per-step ComponentNorms.

    Args:
        steps: Per-step ComponentNorms instances.
        n_proteins: Protein count for each step.

    Returns:
        Single ComponentNorms with all fields weighted-averaged.
    """
    return ComponentNorms(
        template_embedder=_wavg([m.template_embedder for m in steps], n_proteins),
        atom_encoder=_wavg([m.atom_encoder for m in steps], n_proteins),
        atom_decoders=_wavg([m.atom_decoders for m in steps], n_proteins),
        residue_distogram_head=_wavg([m.residue_distogram_head for m in steps], n_proteins),
        atom_distogram_head=_wavg([m.atom_distogram_head for m in steps], n_proteins),
        inter_proj_seq=_wavg([m.inter_proj_seq for m in steps], n_proteins),
        inter_seq_logits=_wavg([m.inter_seq_logits for m in steps], n_proteins),
        proj_seq=_wavg([m.proj_seq for m in steps], n_proteins),
        seq_logits=_wavg([m.seq_logits for m in steps], n_proteins),
    )


def log_epoch(
    *,
    epoch_metrics: EpochMetrics,
    model_params: ModelSetup,
    log: FilteringBoundLogger,
    do_log: bool = True,
) -> None:
    """Log metrics and save checkpoints for one completed epoch.

    Writes structlog entries for train and val splits, optionally pushes to W&B, saves
    the best-validation checkpoint, and saves a periodic epoch checkpoint when
    ``tcfg.checkpoint.save_every`` divides ``epoch``.  Handles both plain ``nn.Module``
    and DDP-wrapped models (accesses ``.module`` when present).  Pass ``do_log=False``
    on non-rank-0 workers to skip all I/O.

    Checkpoints include model weights, optimizer state, scheduler state, and epoch number
    so training can be resumed exactly via ``TrainingParams.resume_checkpoint``.

    Args:
        epoch_metrics: Aggregated metrics for this epoch, including epoch number,
            global step, train/val loss metrics, throughput statistics, and gradient norms.
        model_params: Model, optimizer, scheduler, and training configuration for this run.
        best_val_loss: Best validation loss seen before this epoch.
        log: Bound structlog logger.
        do_log: If ``False``, skip all I/O (use on non-rank-0 workers).

    Returns:
        The (possibly updated) best validation loss.
    """
    if not do_log:
        return  # do nothing for non rank 0 workers
    lg = model_params.tcfg.logging

    def _to_float_dict(dc: object) -> dict[str, float]:
        return {f.name: getattr(dc, f.name).item() for f in dataclasses.fields(dc)}  # type: ignore[arg-type]

    val_dict = _to_float_dict(epoch_metrics.val_loss_metrics)
    train_dict = (
        _to_float_dict(epoch_metrics.train_loss_metrics)
        | _to_float_dict(epoch_metrics.train_throughput_stats)
        | _to_float_dict(epoch_metrics.train_gradient_norms)
    )

    log.info(
        "train",
        epoch=epoch_metrics.epoch,
        **{k.replace(" ", "_"): round(v, 6) for k, v in train_dict.items()},
    )
    log.info(
        "val",
        epoch=epoch_metrics.epoch,
        **{k: round(v, 6) for k, v in val_dict.items()},
    )

    if lg.use_wandb:
        wandb.log(
            {
                "epoch": epoch_metrics.epoch,
                "global_step": epoch_metrics.global_step,
                **{f"train/{k}": v for k, v in train_dict.items()},
                **{f"val/{k}": v for k, v in val_dict.items()},
            }
        )


def _flush_micro_buffer(
    micro_buffer: list[ProteinBatch],
    n_proteins_buffer: list[int],
    model_params: ModelSetup,
    log: FilteringBoundLogger,
    step: StepProgress,
) -> StepProgress:
    """Run one optimizer step over the buffered micro-batches, append results, and update pbar.

    Appends loss metrics, throughput stats, gradient norms, and protein counts to the
    corresponding lists in ``step``, increments ``step.global_step``, and returns ``step``.

    Args:
        micro_buffer: Accumulated micro-batches to flush.
        n_proteins_buffer: Per-micro-batch protein counts.
        model_params: Bundled model, optimizer, scheduler, and config.
        log: Bound structlog logger.
        step: Per-epoch accumulator holding running totals and the progress bar.

    Returns:
        The same ``step`` object with updated ``global_step`` and appended metrics.
    """
    loss_metrics, throughput_stats, component_norms, new_global_step = optimizer_step(
        micro_buffer=micro_buffer,
        n_proteins_buffer=n_proteins_buffer,
        model_params=model_params,
        global_step=step.global_step,
    )
    loss_dict = {
        f.name: getattr(loss_metrics, f.name).item() for f in dataclasses.fields(loss_metrics)
    }
    if any(math.isnan(v) for v in loss_dict.values()) and step.rank == 0:
        log.error("nan_loss", step=new_global_step, **{k: f"{v:.2f}" for k, v in loss_dict.items()})
    step.pbar.update(1)
    if step.rank == 0:
        step.pbar.set_postfix({k: f"{v:.2f}" for k, v in loss_dict.items()})
    step.step_loss_metrics.append(loss_metrics)
    step.step_throughput_stats.append(throughput_stats)
    step.step_component_norms.append(component_norms)
    step.step_n_proteins.append(sum(n_proteins_buffer))
    return dataclasses.replace(step, global_step=new_global_step)


def train(
    best_val_loss: Float[torch.Tensor, ""],
    train_loader: torch.utils.data.DataLoader[ProteinBatch],
    test_loader: torch.utils.data.DataLoader[ProteinBatch],
    model_params: ModelSetup,
    log: FilteringBoundLogger,
) -> None:
    """Training loop for the MainTrunk diffusion model; works with or without DDP.

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
        train_loader: DataLoader for training batches; its
            ``BucketedBatchSampler`` must support ``set_epoch``.
        test_loader: DataLoader for evaluation batches.
        model_params: Bundled model, optimizer, scheduler, and config.
        log: Bound structlog logger.
    """
    is_ddp: bool = dist.is_initialized()
    rank: int = dist.get_rank() if is_ddp else 0
    world_size: int = dist.get_world_size() if is_ddp else 1
    local_rank: int = int(os.environ.get("LOCAL_RANK", "0")) if is_ddp else 0

    if is_ddp:
        ddp_wrapped: DDP = DDP(model_params.model, device_ids=[local_rank])
        model_params = dataclasses.replace(model_params, model=ddp_wrapped)

    tp = model_params.tcfg.training
    per_rank_token_budget: int = max(1, tp.accumulated_token_budget // world_size)
    if rank == 0:
        log.info("training", ddp=is_ddp)
        log.info(
            "gradient_accumulation",
            token_budget_per_rank=per_rank_token_budget,
            global_token_budget=tp.accumulated_token_budget,
        )

    global_step = 0

    for epoch in range(1, tp.num_epochs + 1):
        model_params.model.train()
        cast(BucketedBatchSampler, train_loader.batch_sampler).set_epoch(epoch)
        n_batches = 0
        micro_buffer: list[ProteinBatch] = []
        n_proteins_buffer: list[int] = []
        accum_tokens: int = 0
        model_params.optimizer.zero_grad()

        estimated_steps: int = math.ceil(
            len(train_loader) * model_params.tcfg.train_loader.token_budget / per_rank_token_budget
        )
        pbar: tqdm[NoReturn] = tqdm(
            desc=f"Epoch {epoch:03d}/{tp.num_epochs}",
            total=estimated_steps,
            leave=False,
            unit="step",
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

        for batch in train_loader:
            n_proteins: int = batch.atom_positions.shape[0]
            n_all_tokens: int = n_proteins * batch.atom_positions.shape[1]

            # Pre-flush: if adding this batch would push tokens over the budget, flush first.
            if micro_buffer and accum_tokens + n_all_tokens > per_rank_token_budget:
                step = _flush_micro_buffer(micro_buffer, n_proteins_buffer, model_params, log, step)
                n_batches += 1
                micro_buffer, n_proteins_buffer, accum_tokens = [], [], 0

            micro_buffer.append(batch)
            n_proteins_buffer.append(n_proteins)
            accum_tokens += n_all_tokens

        # Flush any remaining micro-batches at epoch end, regardless of token count.
        if micro_buffer:
            step = _flush_micro_buffer(micro_buffer, n_proteins_buffer, model_params, log, step)
            n_batches += 1
            micro_buffer = []

        model_params.scheduler.step()

        epoch_val_metrics, _ = evaluate(
            loader=test_loader,
            model_params=model_params,
        )

        global_step = step.global_step
        epoch_metrics = EpochMetrics(
            epoch=epoch,
            global_step=global_step,
            train_loss_metrics=_wavg_loss_metrics(step.step_loss_metrics, step.step_n_proteins),
            train_throughput_stats=_wavg_throughput_stats(
                step.step_throughput_stats, step.step_n_proteins
            ),
            train_gradient_norms=_wavg_component_norms(
                step.step_component_norms, step.step_n_proteins
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

    pbar.close()


def main(args: argparse.Namespace, tcfg: TrainConfig) -> None:
    """Entry point for training; dispatches to DDP or single-device based on args.ddp.

    Uses ExitStack so both paths share all setup code after context initialisation.
    DistProcessGroup is only entered under --ddp; device/rank/world_size are resolved
    to plain values before the shared path begins.
    """
    with contextlib.ExitStack() as stack:
        if args.ddp:
            dpg = stack.enter_context(DistProcessGroup("nccl"))
            rank: int = dpg.rank
            is_rank_zero: bool = dpg.is_rank_zero
            device: str = dpg.device
        else:
            rank = 0
            is_rank_zero = True
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        stack.enter_context(StructlogConfig(is_rank_zero, args.log_file))
        stack.enter_context(FatalOnError())

        log = structlog.get_logger()

        train_loader, val_loader, _ = make_bucketed_data_loaders(
            cfg=tcfg,
            jsonl_path=args.data,
            splits_path=args.splits,
            num_workers=args.num_workers,
            debug_run=args.debug_run,
        )

        mp = tcfg.model
        tp = tcfg.training
        model: MainTrunk = MainTrunk(
            f_ref_dim=mp.f_ref_dim,
            n_bins=mp.n_bins,
            n_atom_bins=tcfg.distogram_atom.n_bins,
            c_atom=mp.c_atom,
            c_pair=mp.c_pair,
            c_res=mp.c_res,
            c_atompair=mp.c_atompair,
            K_unit=mp.K_unit,
            sigma_data=tcfg.noise.sigma_data,
        ).to(device)

        optimizer = Adam(model.parameters(), lr=tp.lr, weight_decay=tp.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=tp.num_epochs, eta_min=tp.lr * 0.01)

        dr = tcfg.distogram_res
        da = tcfg.distogram_atom
        distogram_res = Distogram(
            n_bins=dr.n_bins, min_dist=dr.min_dist, max_dist=dr.max_dist, overflow_bin=True
        ).to(device)
        distogram_atom = Distogram(
            n_bins=da.n_bins, min_dist=da.min_dist, max_dist=da.max_dist, overflow_bin=False
        ).to(device)

        model_params = ModelSetup(
            model=model,
            tcfg=tcfg,
            distogram_res=distogram_res,
            distogram_atom=distogram_atom,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        model_params, best_val_loss = load_checkpoint(
            model_params=model_params,
            rank=rank,
            log=log,
        )

        if is_rank_zero and tcfg.logging.use_wandb:
            wandb.init(project=tcfg.logging.wandb_project, config=tcfg.model_dump())

        train(
            best_val_loss=best_val_loss,
            train_loader=train_loader,
            test_loader=val_loader,
            model_params=model_params,
            log=log,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PallAtom")
    parser.add_argument("--data", required=True, help="path to proteins.jsonl")
    parser.add_argument("--splits", required=True, help="path to splits.json")
    parser.add_argument("--config", help="path to TrainConfig JSON (omit for defaults)")
    parser.add_argument("--log_file", default=None, help="path to write structured JSON log lines")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ddp", action="store_true", help="use DistributedDataParallel training")
    parser.add_argument(
        "--debug_run", action="store_true", help="restrict to 252 proteins for fast iteration"
    )
    args = parser.parse_args()
    with open(args.config) as _f:
        tcfg = TrainConfig.model_validate_json(_f.read())
    main(args, tcfg)
