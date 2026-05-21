"""Training loop implementation for the diffusion model."""

import argparse
import contextlib
import math
import time
from typing import cast

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
from architecture.main_trunk import MainTrunk
from beartype import beartype
from einops import rearrange
from helpers.alignment import kabsch_align
from helpers.bucketed_sampler import BucketedBatchSampler
from helpers.context_managers import DistProcessGroup, FatalOnError, StructlogConfig
from helpers.data import (
    make_bucketed_data_loaders,
    make_ddp_bucketed_data_loaders,
)
from helpers.featurize import Distogram, ProteinBatch, apply_conditioning_dropout, featurize_batch
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


@torch.no_grad()
@jaxtyped(typechecker=beartype)
def evaluate(
    model: MainTrunk,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    device: str,
) -> dict[str, float]:
    """Full-dataset evaluation pass. Returns mean loss per metric."""
    model.eval()
    totals = {
        "total loss": 0.0,
        "Kabsch aligned MSE loss": 0.0,
        "Cross Entropy loss": 0.0,
        "Smooth LDDT loss": 0.0,
        "Residue Distogram loss": 0.0,
        "Atom Distogram loss": 0.0,
        "Intermediate loss": 0.0,
        "RMSD": 0.0,
    }
    n_batches = 0
    lp = tcfg.loss

    for batch in loader:
        featurized_batch = featurize_batch(batch, tcfg, distogram_res, distogram_atom, device)

        (
            r_denoised,
            f_seq_logits,
            residue_distogram_logits,
            atom_distogram_logits,
            intermediate_denoised_coord_stack,
            intermediate_pred_aa_logit_stack,
        ) = model(featurized_batch)
        r_denoised: Float[torch.Tensor, "B N_atom 3"]
        f_seq_logits: Float[torch.Tensor, "B N_res n_amino"]
        residue_distogram_logits: Float[torch.Tensor, "B N_res N_res n_bins"]
        atom_distogram_logits: Float[torch.Tensor, "B N_atom K n_atom_bins"]

        Kabsch_aligned_MSE_loss: Float[torch.Tensor, ""] = atom_loss(
            r_denoised, featurized_batch.r_gt, featurized_batch.atom5_mask
        ).mean()
        CE_loss: Float[torch.Tensor, ""] = F.cross_entropy(
            rearrange(f_seq_logits, "b n c -> (b n) c"),
            rearrange(_mask_seq_target(featurized_batch.aa_indices), "b n -> (b n)"),
        )

        gt_res_bin_idx: Int[torch.Tensor, "B N_res N_res"] = (
            featurized_batch.gt_res_distogram.argmax(dim=-1).clamp(
                0, residue_distogram_logits.size(-1) - 1
            )
        )
        residue_distogram_loss: Float[torch.Tensor, ""] = distogram_loss_residue(
            residue_distogram_logits,
            gt_res_bin_idx,
            featurized_batch.residue_mask,
        ).mean()

        atom_distogram_loss: Float[torch.Tensor, ""] = distogram_loss_atom(
            atom_distogram_logits,
            featurized_batch.gt_atom_distogram_sparse,
            featurized_batch.gt_atom_distogram_mask_sparse,
        ).mean()

        lddt_loss: Float[torch.Tensor, ""] = smooth_lddt_loss(
            r_denoised, featurized_batch.r_gt, featurized_batch.atom5_mask
        )

        K_unit = len(intermediate_denoised_coord_stack)
        intermediate_med_loss: Float[torch.Tensor, ""] = torch.tensor(0.0, device=device)

        for k_idx, intermediate_denoised_coord in enumerate(intermediate_denoised_coord_stack):
            intermediate_denoised_coord: Float[torch.Tensor, "B N_atom 3"]
            gamma_K_minus_k: float = lp.gamma ** (K_unit - k_idx - 1)
            k_loss: Float[torch.Tensor, ""] = lp.lam * atom_loss(
                intermediate_denoised_coord, featurized_batch.r_gt, featurized_batch.atom5_mask
            ) + lp.alpha_0 * F.cross_entropy(
                rearrange(intermediate_pred_aa_logit_stack[k_idx], "b n c -> (b n) c"),
                rearrange(_mask_seq_target(featurized_batch.aa_indices), "b n -> (b n)"),
            )
            intermediate_med_loss = intermediate_med_loss + gamma_K_minus_k * k_loss
        intermediate_med_loss = (intermediate_med_loss / max(K_unit, 1)).mean()

        total_loss: Float[torch.Tensor, ""] = (
            lp.lam * Kabsch_aligned_MSE_loss
            + lp.alpha_0 * CE_loss
            + lp.alpha_1 * lddt_loss
            + lp.alpha_2 * residue_distogram_loss
            + lp.alpha_3 * atom_distogram_loss
            + lp.alpha_4 * intermediate_med_loss
        )
        (r_aligned,) = kabsch_align(
            featurized_batch.r_gt,
            r_denoised,
            weights=featurized_batch.atom5_mask.float(),
            return_transform=False,
        )
        r_aligned: Float[torch.Tensor, "B N_atom 3"]
        diff: Float[torch.Tensor, "B N_atom 3"] = r_denoised - r_aligned
        sq: Float[torch.Tensor, "B N_atom"] = (diff * diff).sum(dim=-1)
        m: Float[torch.Tensor, "B N_atom"] = featurized_batch.atom5_mask.float()
        rmsd: Float[torch.Tensor, ""] = ((sq * m).sum() / m.sum().clamp(min=1)).sqrt()

        totals["total loss"] += total_loss.item()
        totals["Kabsch aligned MSE loss"] += Kabsch_aligned_MSE_loss.item()
        totals["Cross Entropy loss"] += CE_loss.item()
        totals["Smooth LDDT loss"] += lddt_loss.item()
        totals["Residue Distogram loss"] += residue_distogram_loss.item()
        totals["Atom Distogram loss"] += atom_distogram_loss.item()
        totals["Intermediate loss"] += intermediate_med_loss.item()
        totals["RMSD"] += rmsd.item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


@torch.no_grad()
def evaluate_ddp(
    world_size: int,
    model: nn.Module,
    loader: torch.utils.data.DataLoader[ProteinBatch],
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    device: str,
) -> dict[str, float]:
    """Distributed evaluation. Each rank processes its shard; metrics are all-reduced."""
    model.eval()
    metric_names = [
        "total loss",
        "Kabsch aligned MSE loss",
        "Cross Entropy loss",
        "Smooth LDDT loss",
        "Residue Distogram loss",
        "Atom Distogram loss",
        "Intermediate loss",
        "RMSD",
    ]
    # totals[:8] = metric sums, totals[8] = n_batches
    totals = torch.zeros(len(metric_names) + 1, device=device)
    lp = tcfg.loss

    for batch in loader:
        featurized_batch = featurize_batch(batch, tcfg, distogram_res, distogram_atom, device)

        (
            r_denoised,
            f_seq_logits,
            residue_distogram_logits,
            atom_distogram_logits,
            intermediate_denoised_coord_stack,
            intermediate_pred_aa_logit_stack,
        ) = model(featurized_batch)

        Kabsch_aligned_MSE_loss: Float[torch.Tensor, ""] = atom_loss(
            r_denoised, featurized_batch.r_gt, featurized_batch.atom5_mask
        ).mean()
        CE_loss: Float[torch.Tensor, ""] = F.cross_entropy(
            rearrange(f_seq_logits, "b n c -> (b n) c"),
            rearrange(_mask_seq_target(featurized_batch.aa_indices), "b n -> (b n)"),
        )

        gt_res_bin_idx: Int[torch.Tensor, "B N_res N_res"] = (
            featurized_batch.gt_res_distogram.argmax(dim=-1).clamp(
                0, residue_distogram_logits.size(-1) - 1
            )
        )
        residue_distogram_loss: Float[torch.Tensor, ""] = distogram_loss_residue(
            residue_distogram_logits,
            gt_res_bin_idx,
            featurized_batch.residue_mask,
        ).mean()

        atom_distogram_loss: Float[torch.Tensor, ""] = distogram_loss_atom(
            atom_distogram_logits,
            featurized_batch.gt_atom_distogram_sparse,
            featurized_batch.gt_atom_distogram_mask_sparse,
        ).mean()

        lddt_loss: Float[torch.Tensor, ""] = smooth_lddt_loss(
            r_denoised, featurized_batch.r_gt, featurized_batch.atom5_mask
        )

        K_unit = len(intermediate_denoised_coord_stack)
        intermediate_med_loss: Float[torch.Tensor, ""] = torch.tensor(0.0, device=device)
        for k_idx, intermediate_denoised_coord in enumerate(intermediate_denoised_coord_stack):
            intermediate_denoised_coord: Float[torch.Tensor, "B N_atom 3"]
            gamma_K_minus_k: float = lp.gamma ** (K_unit - k_idx - 1)
            k_loss: Float[torch.Tensor, ""] = lp.lam * atom_loss(
                intermediate_denoised_coord, featurized_batch.r_gt, featurized_batch.atom5_mask
            ) + lp.alpha_0 * F.cross_entropy(
                rearrange(intermediate_pred_aa_logit_stack[k_idx], "b n c -> (b n) c"),
                rearrange(_mask_seq_target(featurized_batch.aa_indices), "b n -> (b n)"),
            )
            intermediate_med_loss = intermediate_med_loss + gamma_K_minus_k * k_loss
        intermediate_med_loss = (intermediate_med_loss / max(K_unit, 1)).mean()

        total_loss: Float[torch.Tensor, ""] = (
            lp.lam * Kabsch_aligned_MSE_loss
            + lp.alpha_0 * CE_loss
            + lp.alpha_1 * lddt_loss
            + lp.alpha_2 * residue_distogram_loss
            + lp.alpha_3 * atom_distogram_loss
            + lp.alpha_4 * intermediate_med_loss
        )

        (r_aligned,) = kabsch_align(
            featurized_batch.r_gt,
            r_denoised,
            weights=featurized_batch.atom5_mask.float(),
            return_transform=False,
        )
        diff: Float[torch.Tensor, "B N_atom 3"] = r_denoised - r_aligned
        sq: Float[torch.Tensor, "B N_atom"] = (diff * diff).sum(dim=-1)
        m: Float[torch.Tensor, "B N_atom"] = featurized_batch.atom5_mask.float()
        rmsd: Float[torch.Tensor, ""] = ((sq * m).sum() / m.sum().clamp(min=1)).sqrt()

        totals[0] += total_loss.item()
        totals[1] += Kabsch_aligned_MSE_loss.item()
        totals[2] += CE_loss.item()
        totals[3] += lddt_loss.item()
        totals[4] += residue_distogram_loss.item()
        totals[5] += atom_distogram_loss.item()
        totals[6] += intermediate_med_loss.item()
        totals[7] += rmsd.item()
        totals[8] += 1.0

    if world_size > 1:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)

    n_batches = totals[8].item()
    return {k: totals[i].item() / max(n_batches, 1) for i, k in enumerate(metric_names)}


def train_step(
    batch: ProteinBatch,
    model: nn.Module,
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    device: str,
    grad_scale: float = 1.0,
) -> dict[str, float]:
    """Forward and backward pass for one micro-batch.

    The caller owns ``optimizer.zero_grad()``, ``clip_grad_norm_``, and
    ``optimizer.step()``.  Pass ``accum_steps`` as ``grad_scale`` so that
    accumulated gradients match a single large-batch backward.

    Args:
        batch: Raw protein micro-batch.
        model: Model to forward through (plain ``MainTrunk`` or DDP-wrapped).
        tcfg: Training configuration.
        distogram_res: Residue-level distogram.
        distogram_atom: Atom-level distogram.
        device: PyTorch device string.
        grad_scale: Divide total loss by this value before backward (default 1.0).

    Returns:
        Step-level metrics dict with keys matching ``EXPECTED_STEP_KEYS``.
    """
    lp = tcfg.loss

    featurized_batch = featurize_batch(batch, tcfg, distogram_res, distogram_atom, device)
    # Save original amino-acid indices before conditioning dropout replaces some with <MASK> (20).
    orig_aa_indices: Int[torch.Tensor, "B N_res"] = featurized_batch.aa_indices.clone()

    featurized_batch = apply_conditioning_dropout(
        featurized_batch,
        p_distogram=tcfg.conditioning_dropout.p_distogram,
        p_atom=tcfg.conditioning_dropout.p_atom,
        p_seq=tcfg.conditioning_dropout.p_seq,
        device=device,
    )
    # MLM-style CE targets: original amino acid at positions where dropout placed <MASK> (20),
    # -100 (ignore_index) at positions the model can still see and at padding.
    # This forces the model to predict sequence from structure, not copy visible conditioning.
    mask_positions: Int[torch.Tensor, "B N_res"] = featurized_batch.aa_indices == 20
    aa_targets: Int[torch.Tensor, "B N_res"] = orig_aa_indices.masked_fill(~mask_positions, -100)
    # Flat targets reused for final and intermediate CE; count guards against p_seq=0 edge case.
    flat_aa_targets: Int[torch.Tensor, "B_N_res"] = rearrange(aa_targets, "b n -> (b n)")
    n_masked_aa: Int[torch.Tensor, ""] = (flat_aa_targets >= 0).sum().clamp(min=1)

    t0 = time.perf_counter()
    (
        r_denoised,
        f_seq_logits,
        residue_distogram_logits,
        atom_distogram_logits,
        intermediate_denoised_coord_stack,
        intermediate_pred_aa_logit_stack,
    ) = model(featurized_batch)
    r_denoised: Float[torch.Tensor, "B N_atom 3"]
    f_seq_logits: Float[torch.Tensor, "B N_res n_amino"]
    residue_distogram_logits: Float[torch.Tensor, "B N_res N_res n_bins"]
    atom_distogram_logits: Float[torch.Tensor, "B N_atom K n_atom_bins"]

    Kabsch_aligned_MSE_loss: Float[torch.Tensor, ""] = atom_loss(
        r_denoised, featurized_batch.r_gt, featurized_batch.atom5_mask
    ).mean()

    K_unit = len(intermediate_denoised_coord_stack)
    intermediate_med_loss: Float[torch.Tensor, ""] = torch.tensor(0.0, device=device)
    for k_idx, intermediate_denoised_coord in enumerate(intermediate_denoised_coord_stack):
        intermediate_denoised_coord: Float[torch.Tensor, "B N_atom 3"]
        gamma_K_minus_k: float = lp.gamma ** (K_unit - k_idx - 1)
        inter_ce: Float[torch.Tensor, ""] = (
            F.cross_entropy(
                rearrange(intermediate_pred_aa_logit_stack[k_idx], "b n c -> (b n) c"),
                flat_aa_targets,
                reduction="sum",
            )
            / n_masked_aa
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

    gt_res_bin_idx: Int[torch.Tensor, "B N_res N_res"] = featurized_batch.gt_res_distogram.argmax(
        dim=-1
    ).clamp(0, residue_distogram_logits.size(-1) - 1)
    residue_distogram_loss: Float[torch.Tensor, ""] = distogram_loss_residue(
        residue_distogram_logits,
        gt_res_bin_idx,
        featurized_batch.residue_mask,
    ).mean()

    atom_distogram_loss: Float[torch.Tensor, ""] = distogram_loss_atom(
        atom_distogram_logits,
        featurized_batch.gt_atom_distogram_sparse,
        featurized_batch.gt_atom_distogram_mask_sparse,
    ).mean()

    lddt_loss: Float[torch.Tensor, ""] = smooth_lddt_loss(
        r_denoised,
        featurized_batch.r_gt,
        featurized_batch.atom5_mask,
        cutoff=float(lp.smooth_lddt_cutoff),
    )
    CE_loss: Float[torch.Tensor, ""] = F.cross_entropy(
        rearrange(f_seq_logits, "b n c -> (b n) c"),
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

    (total_loss / grad_scale).backward()
    t1 = time.perf_counter()
    step_time = t1 - t0

    b_size, n_res = featurized_batch.residue_mask.shape
    actual_residues = int(featurized_batch.residue_mask.sum().item())
    actual_atoms = int(featurized_batch.atom5_mask.sum().item())

    return {
        "total loss": total_loss.item(),
        "Kabsch aligned MSE loss": Kabsch_aligned_MSE_loss.item(),
        "Cross Entropy loss": CE_loss.item(),
        "smooth lddt": lddt_loss.item(),
        "Residue Distogram loss": residue_distogram_loss.item(),
        "Atom Distogram loss": atom_distogram_loss.item(),
        "Intermediate loss": intermediate_med_loss.item(),
        "pack_rate": actual_residues / (b_size * n_res),
        "residues_per_sec": actual_residues / step_time,
        "atoms_per_sec": actual_atoms / step_time,
    }


def log_epoch(
    epoch: int,
    global_step: int,
    avg_train: dict[str, float],
    avg_val: dict[str, float],
    model: nn.Module,
    optimizer: Adam,
    scheduler: CosineAnnealingLR,
    tcfg: TrainConfig,
    best_val_loss: float,
    log: FilteringBoundLogger,
    *,
    do_log: bool = True,
) -> float:
    """Log metrics and save checkpoints for one completed epoch.

    Writes structlog entries for train and val splits, optionally pushes to W&B, saves
    the best-validation checkpoint, and saves a periodic epoch checkpoint when
    ``tcfg.checkpoint.save_every`` divides ``epoch``.  Handles both plain ``nn.Module``
    and DDP-wrapped models (accesses ``.module`` when present).  Pass ``do_log=False``
    on non-rank-0 workers to skip all I/O.

    Checkpoints include model weights, optimizer state, scheduler state, epoch number,
    global step, and best validation loss so training can be resumed exactly via
    ``TrainingParams.resume_checkpoint``.

    Args:
        epoch: Current epoch number (1-indexed).
        global_step: Total optimizer steps taken so far.
        avg_train: Per-metric average training losses for this epoch.
        avg_val: Per-metric average validation losses for this epoch.
        model: The model (plain or DDP-wrapped) being trained.
        optimizer: Adam optimizer used for this run.
        scheduler: Cosine annealing scheduler for this run.
        tcfg: Training configuration supplying logging and checkpoint settings.
        best_val_loss: Best validation loss seen before this epoch.
        log: Bound structlog logger.
        do_log: If ``False``, skip all I/O (use on non-rank-0 workers).

    Returns:
        The (possibly updated) best validation loss.
    """
    if not do_log:
        return best_val_loss

    ck = tcfg.checkpoint
    lg = tcfg.logging

    log.info(
        "train",
        epoch=epoch,
        **{k.replace(" ", "_"): round(v, 6) for k, v in avg_train.items()},
    )
    log.info(
        "val",
        epoch=epoch,
        **{k.replace(" ", "_"): round(v, 6) for k, v in avg_val.items()},
    )

    if lg.use_wandb:
        wandb.log(
            {
                "epoch": epoch,
                **{f"train/{k}": v for k, v in avg_train.items()},
                **{f"val/{k}": v for k, v in avg_val.items()},
            },
            step=global_step,
        )

    inner: nn.Module = model.module if isinstance(model, DDP) else model
    checkpoint = {
        "model": inner.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
    }

    if avg_val["total loss"] < best_val_loss:
        best_val_loss = avg_val["total loss"]
        checkpoint["best_val_loss"] = best_val_loss
        torch.save(checkpoint, ck.checkpoint_path)

    if ck.save_every > 0 and epoch % ck.save_every == 0:
        torch.save(checkpoint, f"checkpoint_epoch_{epoch:03d}.pt")

    return best_val_loss


def _load_ddp_checkpoint(
    path: str,
    ddp_model: DDP,
    optimizer: Adam,
    scheduler: CosineAnnealingLR,
    device: str,
    rank: int,
    log: FilteringBoundLogger,
) -> tuple[float, int, int]:
    """Load a DDP checkpoint and restore all mutable training state.

    Args:
        path: Filesystem path to the checkpoint file.
        ddp_model: DDP-wrapped model; weights are loaded into ``ddp_model.module``.
        optimizer: Adam optimizer whose state will be restored.
        scheduler: Cosine annealing scheduler whose state will be restored.
        device: Target device for ``torch.load``.
        rank: Current process rank; info log emitted only on rank 0.
        log: Bound structlog logger.

    Returns:
        A ``(best_val_loss, global_step, start_epoch)`` triple.
    """
    ckpt = torch.load(path, map_location=device)
    ddp_model.module.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    best_val_loss: float = ckpt["best_val_loss"]
    global_step: int = ckpt["global_step"]
    start_epoch: int = ckpt["epoch"] + 1
    if rank == 0:
        log.info("resumed from checkpoint", path=path, start_epoch=start_epoch)
    return best_val_loss, global_step, start_epoch


_METRIC_KEYS: list[str] = [
    "total loss",
    "Kabsch aligned MSE loss",
    "Cross Entropy loss",
    "smooth lddt",
    "Residue Distogram loss",
    "Atom Distogram loss",
    "Intermediate loss",
    "pack_rate",
    "residues_per_sec",
    "atoms_per_sec",
]


# (model attribute name, short pbar key) for per-component gradient norm tracking.
_COMPONENT_NAMES: list[tuple[str, str]] = [
    ("template_embedder", "gn_templ"),
    ("atom_encoder", "gn_enc"),
    ("atom_decoders", "gn_dec"),
    ("residue_distogram_head", "gn_rdist"),
    ("atom_distogram_head", "gn_adist"),
    ("inter_proj_seq", "gn_iproj"),
    ("inter_seq_logits", "gn_ilogit"),
    ("proj_seq", "gn_proj"),
    ("seq_logits", "gn_logit"),
]


def _component_grad_norms(model: nn.Module) -> dict[str, float]:
    """Return gradient L2 norm for each named MainTrunk sub-module.

    Computes per-component norms from pre-clip gradients.  Call this after the
    backward pass and before ``clip_grad_norm_``.

    Args:
        model: Plain ``MainTrunk`` or DDP-wrapped model; ``.module`` is unwrapped
            automatically.

    Returns:
        Dict mapping the short pbar key (from ``_COMPONENT_NAMES``) to the L2 norm
        of all gradients in that sub-module (0.0 when no grads exist).
    """
    inner: nn.Module = model.module if isinstance(model, DDP) else model
    norms: dict[str, float] = {}
    for attr, key in _COMPONENT_NAMES:
        sub: nn.Module = getattr(inner, attr)
        grad_norms: list[torch.Tensor] = [
            torch.linalg.vector_norm(p.grad.detach(), 2.0)
            for p in sub.parameters()
            if p.grad is not None
        ]
        norms[key] = (
            float(torch.linalg.vector_norm(torch.stack(grad_norms), 2.0).item())
            if grad_norms
            else 0.0
        )
    return norms


def _process_accum_window(
    micro_buffer: list[ProteinBatch],
    n_proteins_per_batch: list[int],
    model: nn.Module,
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    device: str,
) -> dict[str, float]:
    """Forward + backward over one accumulation window; returns protein-weighted metrics.

    Each micro-batch's loss is scaled by ``total_proteins / n_proteins_i`` so that the
    accumulated gradient is equivalent to a single large-batch backward over all proteins
    in the window.  Metrics are averaged with the same protein-count weights.

    The ``no_sync()`` context manager is used on all but the last micro-batch when the
    model exposes it (DDP), so gradient all-reduces happen only once per window.

    Args:
        micro_buffer: Micro-batches to process.
        n_proteins_per_batch: Protein count per micro-batch (``batch.atom_positions.shape[0]``).
        model: Model to forward through (plain ``MainTrunk`` or DDP-wrapped).
        tcfg: Training configuration.
        distogram_res: Residue-level distogram.
        distogram_atom: Atom-level distogram.
        device: PyTorch device string.

    Returns:
        Dict of metrics averaged by protein count over all micro-batches in the window.
    """
    total_proteins: int = sum(n_proteins_per_batch)
    n_micro: int = len(micro_buffer)
    window_metrics: dict[str, float] = dict.fromkeys(_METRIC_KEYS, 0.0)
    maybe_no_sync = getattr(model, "no_sync", None)
    for micro_idx, (mb, n_proteins) in enumerate(
        zip(micro_buffer, n_proteins_per_batch, strict=False)
    ):
        is_last = micro_idx == n_micro - 1
        ctx = cast(
            contextlib.AbstractContextManager[None],
            (
                maybe_no_sync()
                if (not is_last and callable(maybe_no_sync))
                else contextlib.nullcontext()
            ),
        )
        grad_scale: float = total_proteins / n_proteins
        weight: float = n_proteins / total_proteins
        with ctx:
            step_metrics = train_step(
                mb,
                model,
                tcfg,
                distogram_res,
                distogram_atom,
                device,
                grad_scale=grad_scale,
            )
        for k in window_metrics:
            window_metrics[k] += step_metrics[k] * weight
    return window_metrics


def _optimizer_step(
    micro_buffer: list[ProteinBatch],
    n_proteins_buffer: list[int],
    model: nn.Module,
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    device: str,
    optimizer: Adam,
    epoch_metrics: dict[str, float],
    global_step: int,
) -> tuple[dict[str, float], dict[str, float], float, int]:
    """Run one accumulation window, clip gradients, and step the optimizer.

    Mutates ``epoch_metrics`` in-place. Returns the window metrics, per-component
    gradient norms, the clipped gradient norm, and the incremented global step.

    Args:
        micro_buffer: Micro-batches accumulated for this optimizer step.
        n_proteins_buffer: Protein count per micro-batch.
        model: Model (plain or DDP-wrapped).
        tcfg: Training configuration.
        distogram_res: Residue-level distogram.
        distogram_atom: Atom-level distogram.
        device: PyTorch device string.
        optimizer: The optimizer to step.
        epoch_metrics: Running per-epoch metric sums, updated in place.
        global_step: Current global step count before this flush.

    Returns:
        Tuple of (window_metrics, component_norms, grad_norm, global_step + 1).
    """
    window_metrics = _process_accum_window(
        micro_buffer, n_proteins_buffer, model, tcfg, distogram_res, distogram_atom, device
    )
    component_norms = _component_grad_norms(model)
    tp = tcfg.training
    grad_norm = float(
        nn.utils.clip_grad_norm_(
            model.parameters(),
            tp.grad_clip if tp.grad_clip is not None else float("inf"),
        )
    )
    optimizer.step()
    optimizer.zero_grad()
    for k in epoch_metrics:
        epoch_metrics[k] += window_metrics[k]
    return window_metrics, component_norms, grad_norm, global_step + 1


def train(  # noqa: PLR0915
    model: MainTrunk,
    tcfg: TrainConfig,
    train_loader: torch.utils.data.DataLoader[ProteinBatch],
    test_loader: torch.utils.data.DataLoader[ProteinBatch],
    distogram_res: Distogram,
    distogram_atom: Distogram,
    device: str,
    log: FilteringBoundLogger,
) -> None:
    """Single-GPU training loop for the MainTrunk diffusion model.

    Runs a standard train/eval loop for ``tcfg.training.num_epochs`` epochs.
    Each training step featurizes a raw protein batch with a freshly sampled noise
    level sigmq, applies per-modality conditioning dropout for classifier-free guidance,
    forwards through the model, and back-propagates a weighted sum of seven losses:

    - **MSE** (Kabsch-aligned coordinate error)
    - **Cross-entropy** (sequence prediction)
    - **Smooth lDDT** (local distance difference test)
    - **Residue distogram** (Cβ pairwise distances)
    - **Atom distogram** (sparse local atom-pair distances)
    - **Intermediate coordinate loss** (auxiliary loss over each decoder unit,
      weighted by gamma^(K-k) to emphasise later units)
    - **Intermediate sequence loss** (auxiliary CE over each decoder unit)

    After each epoch the model is evaluated on ``test_loader`` via :func:`evaluate`,
    metrics are logged with structlog (and optionally W&B), and the best-validation
    checkpoint is saved.  Periodic epoch checkpoints are also written if
    ``tcfg.checkpoint.save_every`` is set.

    Args:
        model: MainTrunk to train; must already be on ``device``.
        tcfg: Training configuration supplying optimizer hyper-parameters, loss
            weights, conditioning-dropout probabilities, logging, and checkpoint
            settings.
        train_loader: DataLoader yielding raw protein batches for training.
        test_loader: DataLoader yielding raw protein batches for evaluation.
        distogram_res: Callable producing Cβ residue-level distograms.
        distogram_atom: Callable producing atom-level distograms.
        device: PyTorch device string (e.g. ``"cuda:0"`` or ``"cpu"``).
        log: Bound structlog logger.
    """
    tp = tcfg.training
    lg = tcfg.logging
    per_rank_token_budget: int = tp.accumulated_token_budget
    log.info("training with DDP", DDP=False)
    log.info(
        "gradient_accumulation",
        token_budget_per_rank=per_rank_token_budget,
        global_token_budget=tp.accumulated_token_budget,
    )

    optimizer = Adam(model.parameters(), lr=tp.lr, weight_decay=tp.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=tp.num_epochs, eta_min=tp.lr * 0.01)

    best_val_loss = float("inf")
    global_step = 0
    start_epoch = 1

    if tp.resume_checkpoint is not None:
        ckpt = torch.load(tp.resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        best_val_loss = ckpt["best_val_loss"]
        global_step = ckpt["global_step"]
        start_epoch = ckpt["epoch"] + 1
        log.info("resumed from checkpoint", path=tp.resume_checkpoint, start_epoch=start_epoch)

    for epoch in range(start_epoch, tp.num_epochs + 1):
        model.train()
        epoch_metrics: dict[str, float] = dict.fromkeys(_METRIC_KEYS, 0.0)
        n_batches = 0
        micro_buffer: list[ProteinBatch] = []
        n_proteins_buffer: list[int] = []
        accum_tokens: int = 0
        optimizer.zero_grad()

        if isinstance(train_loader.batch_sampler, BucketedBatchSampler):
            train_loader.batch_sampler.set_epoch(epoch)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{tp.num_epochs}", leave=False)

        for batch in pbar:
            n_tokens: int = int(batch.atom_mask.any(dim=-1).sum().item())
            n_proteins: int = batch.atom_positions.shape[0]
            max_seq_len: int = batch.atom_positions.shape[1]
            log.info(
                "batch statistics",
                batch_token_count=n_tokens,
                batch_size=n_proteins,
                max_seq_len_in_batch=max_seq_len,
            )

            # Pre-flush: if adding this batch would push tokens over the budget, flush first.
            if micro_buffer and accum_tokens + n_tokens > per_rank_token_budget:
                log.info(f"Too many tokens in batch, given budget {per_rank_token_budget}.")
                window_metrics, component_norms, grad_norm, global_step = _optimizer_step(
                    micro_buffer,
                    n_proteins_buffer,
                    model,
                    tcfg,
                    distogram_res,
                    distogram_atom,
                    device,
                    optimizer,
                    epoch_metrics,
                    global_step,
                )
                n_batches += 1
                micro_buffer, n_proteins_buffer, accum_tokens = [], [], 0
                if global_step % lg.log_interval == 0:
                    pbar.set_postfix(
                        {
                            "loss": f"{window_metrics['total loss']:.2f}",
                            "MSE_loss": f"{window_metrics['Kabsch aligned MSE loss']:.2f}",
                            "CE_loss": f"{window_metrics['Cross Entropy loss']:.2f}",
                            "smooth_lddt_loss": f"{window_metrics['smooth lddt']:.2f}",
                            "residue_distogram_loss": (
                                f"{window_metrics['Residue Distogram loss']:.2f}"
                            ),
                            "atom_distogram_loss": f"{window_metrics['Atom Distogram loss']:.2f}",
                            "intermediate_loss": f"{window_metrics['Intermediate loss']:.2f}",
                            "gnorm": f"{grad_norm:.2f}",
                            **{k: f"{v:.2f}" for k, v in component_norms.items()},
                        }
                    )

            micro_buffer.append(batch)
            n_proteins_buffer.append(n_proteins)
            accum_tokens += n_tokens

        # Flush any remaining micro-batches at epoch end, regardless of token count.
        if micro_buffer:
            log.info("Flushing remaining batch at epoch end")
            _, _, _, global_step = _optimizer_step(
                micro_buffer,
                n_proteins_buffer,
                model,
                tcfg,
                distogram_res,
                distogram_atom,
                device,
                optimizer,
                epoch_metrics,
                global_step,
            )
            n_batches += 1
            micro_buffer = []

        if n_batches > 0:
            scheduler.step()

        avg_train = {k: v / max(n_batches, 1) for k, v in epoch_metrics.items()}
        avg_val = evaluate(model, test_loader, tcfg, distogram_res, distogram_atom, device)
        model.train()

        best_val_loss = log_epoch(
            epoch,
            global_step,
            avg_train,
            avg_val,
            model,
            optimizer,
            scheduler,
            tcfg,
            best_val_loss,
            log,
        )


def train_ddp(  # noqa: PLR0915
    rank: int,
    local_rank: int,
    world_size: int,
    model: MainTrunk,
    tcfg: TrainConfig,
    train_loader: torch.utils.data.DataLoader[ProteinBatch],
    test_loader: torch.utils.data.DataLoader[ProteinBatch],
    distogram_res: Distogram,
    distogram_atom: Distogram,
    log: FilteringBoundLogger,
    device: str | None = None,
) -> None:
    """DDP training loop for the MainTrunk diffusion model.

    Mirrors :func:`train` but runs under PyTorch DDP — one process per GPU,
    launched via ``torchrun``.  ``tcfg.training.accumulated_token_budget`` is
    divided by ``world_size`` to get the per-rank token threshold; micro-batches
    accumulate until their combined token count hits that threshold before each
    optimizer step.

    Args:
        rank: Global process rank (0 = primary).
        local_rank: Device-local rank used to select ``cuda:<local_rank>``.
        world_size: Total number of DDP processes.
        model: ``MainTrunk`` to train; must already be on the correct device.
        tcfg: Training configuration (hyper-parameters, loss weights, logging).
        train_loader: DataLoader for training batches; its
            ``BucketedBatchSampler`` must support ``set_epoch``.
        test_loader: DataLoader for evaluation batches.
        distogram_res: Callable producing Cβ residue-level distograms.
        distogram_atom: Callable producing atom-level distograms.
        log: Bound structlog logger injected by the DDP launcher.
        device: Device string override; defaults to ``"cuda:<local_rank>"``.
    """
    device = device or f"cuda:{local_rank}"
    ddp_model = DDP(model, device_ids=[local_rank])

    tp = tcfg.training
    lg = tcfg.logging
    per_rank_token_budget: int = max(1, tp.accumulated_token_budget // world_size)
    if rank == 0:
        log.info("training with DDP", DDP=True)
        log.info(
            "gradient_accumulation",
            token_budget_per_rank=per_rank_token_budget,
            global_token_budget=tp.accumulated_token_budget,
        )

    optimizer = Adam(ddp_model.parameters(), lr=tp.lr, weight_decay=tp.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=tp.num_epochs, eta_min=tp.lr * 0.01)

    best_val_loss = float("inf")
    global_step = 0
    start_epoch = 1

    if tp.resume_checkpoint is not None:
        best_val_loss, global_step, start_epoch = _load_ddp_checkpoint(
            tp.resume_checkpoint, ddp_model, optimizer, scheduler, device, rank, log
        )

    for epoch in range(start_epoch, tp.num_epochs + 1):
        ddp_model.train()
        cast(BucketedBatchSampler, train_loader.batch_sampler).set_epoch(epoch)
        epoch_metrics: dict[str, float] = dict.fromkeys(_METRIC_KEYS, 0.0)
        n_batches = 0
        micro_buffer: list[ProteinBatch] = []
        n_proteins_buffer: list[int] = []
        accum_tokens: int = 0
        optimizer.zero_grad()

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:03d}/{tp.num_epochs}",
            leave=False,
            disable=(rank != 0),
        )

        for batch in pbar:
            n_tokens: int = int(batch.atom_mask.any(dim=-1).sum().item())
            n_proteins: int = batch.atom_positions.shape[0]
            max_seq_len: int = batch.atom_positions.shape[1]
            if rank == 0:
                log.info(
                    "batch statistics",
                    batch_token_count=n_tokens,
                    batch_size=n_proteins,
                    max_seq_len_in_batch=max_seq_len,
                )

            # Pre-flush: if adding this batch would push tokens over the budget, flush first.
            if micro_buffer and accum_tokens + n_tokens > per_rank_token_budget:
                window_metrics, component_norms, grad_norm, global_step = _optimizer_step(
                    micro_buffer,
                    n_proteins_buffer,
                    ddp_model,
                    tcfg,
                    distogram_res,
                    distogram_atom,
                    device,
                    optimizer,
                    epoch_metrics,
                    global_step,
                )
                if math.isnan(window_metrics["total loss"]) and rank == 0:
                    log.warning(
                        "nan_loss",
                        step=global_step,
                        nan_components=[k for k, v in window_metrics.items() if math.isnan(v)],
                    )
                n_batches += 1
                micro_buffer, n_proteins_buffer, accum_tokens = [], [], 0
                if rank == 0 and global_step % lg.log_interval == 0:
                    pbar.set_postfix(
                        {
                            "loss": f"{window_metrics['total loss']:.2f}",
                            "MSE_loss": f"{window_metrics['Kabsch aligned MSE loss']:.2f}",
                            "CE_loss": f"{window_metrics['Cross Entropy loss']:.2f}",
                            "smooth_lddt_loss": f"{window_metrics['smooth lddt']:.2f}",
                            "residue_distogram_loss": (
                                f"{window_metrics['Residue Distogram loss']:.2f}"
                            ),
                            "atom_distogram_loss": f"{window_metrics['Atom Distogram loss']:.2f}",
                            "intermediate_loss": f"{window_metrics['Intermediate loss']:.2f}",
                            "gnorm": f"{grad_norm:.2f}",
                            **{k: f"{v:.2f}" for k, v in component_norms.items()},
                        }
                    )

            micro_buffer.append(batch)
            n_proteins_buffer.append(n_proteins)
            accum_tokens += n_tokens

        # Flush any remaining micro-batches at epoch end, regardless of token count.
        if micro_buffer:
            window_metrics, _, _, global_step = _optimizer_step(
                micro_buffer,
                n_proteins_buffer,
                ddp_model,
                tcfg,
                distogram_res,
                distogram_atom,
                device,
                optimizer,
                epoch_metrics,
                global_step,
            )
            if math.isnan(window_metrics["total loss"]) and rank == 0:
                log.warning(
                    "nan_loss",
                    step=global_step,
                    nan_components=[k for k, v in window_metrics.items() if math.isnan(v)],
                )
            n_batches, micro_buffer = n_batches + 1, []

        scheduler.step()

        avg_train = {k: v / max(n_batches, 1) for k, v in epoch_metrics.items()}
        avg_val = evaluate_ddp(
            world_size if dist.is_initialized() else 1,
            ddp_model,
            test_loader,
            tcfg,
            distogram_res,
            distogram_atom,
            device,
        )
        ddp_model.train()

        best_val_loss = log_epoch(
            epoch,
            global_step,
            avg_train,
            avg_val,
            ddp_model,
            optimizer,
            scheduler,
            tcfg,
            best_val_loss,
            log,
            do_log=(rank == 0),
        )


# still missing the queue
def main(args: argparse.Namespace, tcfg: TrainConfig) -> None:
    """Entry point for training; dispatches to DDP or single-device based on args.ddp."""
    if args.ddp:
        with (
            DistProcessGroup("nccl") as dpg,
            StructlogConfig(dpg.is_rank_zero, args.log_file),
            FatalOnError(),
        ):
            log = structlog.get_logger()
            train_loader, val_loader, _ = make_ddp_bucketed_data_loaders(
                tcfg,
                args.data,
                args.splits,
                rank=dpg.rank,
                world_size=dpg.world_size,
                num_workers=args.num_workers,
            )

            mp = tcfg.model
            model = MainTrunk(
                f_ref_dim=mp.f_ref_dim,
                n_bins=mp.n_bins,
                n_atom_bins=tcfg.distogram_atom.n_bins,
                c_atom=mp.c_atom,
                c_pair=mp.c_pair,
                c_res=mp.c_res,
                c_atompair=mp.c_atompair,
                K_unit=mp.K_unit,
                sigma_data=tcfg.noise.sigma_data,
            ).to(dpg.device)

            dr = tcfg.distogram_res
            da = tcfg.distogram_atom
            distogram_res = Distogram(
                n_bins=dr.n_bins, min_dist=dr.min_dist, max_dist=dr.max_dist, overflow_bin=True
            ).to(dpg.device)
            distogram_atom = Distogram(
                n_bins=da.n_bins, min_dist=da.min_dist, max_dist=da.max_dist, overflow_bin=False
            ).to(dpg.device)

            if tcfg.training.pretrained_weights is not None:
                ckpt = torch.load(tcfg.training.pretrained_weights, map_location=dpg.device)
                model.load_state_dict(ckpt["model"])
                if dpg.is_rank_zero:
                    log.info("loaded pretrained weights", path=tcfg.training.pretrained_weights)

            if dpg.is_rank_zero and tcfg.logging.use_wandb:
                wandb.init(project=tcfg.logging.wandb_project, config=tcfg.model_dump())

            train_ddp(
                rank=dpg.rank,
                local_rank=dpg.local_rank,
                world_size=dpg.world_size,
                model=model,
                tcfg=tcfg,
                train_loader=train_loader,
                test_loader=val_loader,
                distogram_res=distogram_res,
                distogram_atom=distogram_atom,
                log=log,
            )
    else:
        with (
            StructlogConfig(is_rank_zero=True, log_file=args.log_file),
            FatalOnError(),
        ):
            log = structlog.get_logger()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            train_loader, val_loader, _ = make_bucketed_data_loaders(
                cfg=tcfg,
                jsonl_path=args.data,
                splits_path=args.splits,
                num_workers=args.num_workers,
                debug_run=args.debug_run,
            )

            mp = tcfg.model
            model = MainTrunk(
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

            dr = tcfg.distogram_res
            da = tcfg.distogram_atom
            distogram_res = Distogram(
                n_bins=dr.n_bins, min_dist=dr.min_dist, max_dist=dr.max_dist, overflow_bin=True
            ).to(device)
            distogram_atom = Distogram(
                n_bins=da.n_bins, min_dist=da.min_dist, max_dist=da.max_dist, overflow_bin=False
            ).to(device)

            if tcfg.training.pretrained_weights is not None:
                ckpt = torch.load(tcfg.training.pretrained_weights, map_location=device)
                model.load_state_dict(ckpt["model"])
                log.info("loaded pretrained weights", path=tcfg.training.pretrained_weights)

            if tcfg.logging.use_wandb:
                wandb.init(project=tcfg.logging.wandb_project, config=tcfg.model_dump())

            train(
                model=model,
                tcfg=tcfg,
                train_loader=train_loader,
                test_loader=val_loader,
                distogram_res=distogram_res,
                distogram_atom=distogram_atom,
                device=device,
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
