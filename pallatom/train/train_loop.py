import os

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
from helpers.data import make_ddp_data_loaders
from helpers.featurize import Distogram, ProteinBatch, featurize_batch
from jaxtyping import Float, Int, jaxtyped
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from train.train_config import TrainConfig

log = structlog.get_logger()


def _to_protein_batch(batch: dict) -> ProteinBatch:
    """Convert a DataLoader dict batch to a ProteinBatch dataclass."""
    return ProteinBatch(
        atom_positions=batch["atom_positions"],
        atom_mask=batch["atom_mask"],
        residue_index=batch["residue_index"],
        seq=list(batch["seq"]),
    )


@torch.no_grad()
@jaxtyped(typechecker=beartype)
def evaluate(
    model: MainTrunk,
    loader,
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    index_embedding: nn.Embedding,
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
        featurized_batch = featurize_batch(
            _to_protein_batch(batch), tcfg, distogram_res, distogram_atom, index_embedding, device
        )

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
            rearrange(featurized_batch.aa_indices, "b n -> (b n)"),
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
                rearrange(featurized_batch.aa_indices, "b n -> (b n)"),
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
    rank: int,
    world_size: int,
    model: nn.Module,
    loader,
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    index_embedding: nn.Embedding,
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
        featurized_batch = featurize_batch(
            _to_protein_batch(batch), tcfg, distogram_res, distogram_atom, index_embedding, device
        )

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
            rearrange(featurized_batch.aa_indices, "b n -> (b n)"),
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
                rearrange(featurized_batch.aa_indices, "b n -> (b n)"),
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


@jaxtyped(typechecker=beartype)
def train(
    model: MainTrunk,
    tcfg: TrainConfig,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    index_embedding: nn.Embedding,
    device: str,
) -> None:
    tp = tcfg.training
    lp = tcfg.loss
    lg = tcfg.logging
    ck = tcfg.checkpoint

    optimizer = Adam(
        list(model.parameters()) + list(index_embedding.parameters()),
        lr=tp.lr,
        weight_decay=tp.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=tp.num_epochs, eta_min=tp.lr * 0.01)

    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(1, tp.num_epochs + 1):
        model.train()
        epoch_total_loss = epoch_MSE = epoch_CE = epoch_smooth_lddt = epoch_res_dist = (
            epoch_atom_dist
        ) = epoch_intermediate_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{tp.num_epochs}", leave=False)

        for batch in pbar:
            featurized_batch = featurize_batch(
                _to_protein_batch(batch),
                tcfg,
                distogram_res,
                distogram_atom,
                index_embedding,
                device,
            )

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
                k_loss: Float[torch.Tensor, ""] = lp.lam * atom_loss(
                    intermediate_denoised_coord, featurized_batch.r_gt, featurized_batch.atom5_mask
                ) + lp.alpha_0 * F.cross_entropy(
                    rearrange(intermediate_pred_aa_logit_stack[k_idx], "b n c -> (b n) c"),
                    rearrange(featurized_batch.aa_indices, "b n -> (b n)"),
                )
                intermediate_med_loss = intermediate_med_loss + gamma_K_minus_k * k_loss
            intermediate_med_loss = (intermediate_med_loss / max(K_unit, 1)).mean()

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
                r_denoised,
                featurized_batch.r_gt,
                featurized_batch.atom5_mask,
                cutoff=float(lp.smooth_lddt_cutoff),
            )
            CE_loss: Float[torch.Tensor, ""] = F.cross_entropy(
                rearrange(f_seq_logits, "b n c -> (b n) c"),
                rearrange(featurized_batch.aa_indices, "b n -> (b n)"),
            )

            total_loss: Float[torch.Tensor, ""] = (
                lp.lam * Kabsch_aligned_MSE_loss
                + lp.alpha_0 * CE_loss
                + lp.alpha_1 * lddt_loss
                + lp.alpha_2 * residue_distogram_loss
                + lp.alpha_3 * atom_distogram_loss
                + lp.alpha_4 * intermediate_med_loss
            )

            optimizer.zero_grad()
            total_loss.backward()

            grad_norm: Float[torch.Tensor, ""] = nn.utils.clip_grad_norm_(
                model.parameters(),
                tp.grad_clip if tp.grad_clip is not None else float("inf"),
            )
            optimizer.step()

            epoch_total_loss += total_loss.item()
            epoch_MSE += Kabsch_aligned_MSE_loss.item()
            epoch_CE += CE_loss.item()
            epoch_smooth_lddt += lddt_loss.item()
            epoch_res_dist += residue_distogram_loss.item()
            epoch_atom_dist += atom_distogram_loss.item()
            epoch_intermediate_loss += intermediate_med_loss.item()
            n_batches += 1
            global_step += 1

            if global_step % lg.log_interval == 0:
                pbar.set_postfix(loss=f"{total_loss.item():.4f}", gnorm=f"{grad_norm:.3f}")

        scheduler.step()

        avg_train = {
            k: v / n_batches
            for k, v in zip(
                [
                    "total loss",
                    "Kabsch aligned MSE loss",
                    "Cross Entropy loss",
                    "smooth lddt",
                    "Residue Distogram loss",
                    "Atom Distogram loss",
                    "Intermediate loss",
                ],
                [
                    epoch_total_loss,
                    epoch_MSE,
                    epoch_CE,
                    epoch_smooth_lddt,
                    epoch_res_dist,
                    epoch_atom_dist,
                    epoch_intermediate_loss,
                ],
                strict=False,
            )
        }

        avg_val = evaluate(
            model, test_loader, tcfg, distogram_res, distogram_atom, index_embedding, device
        )
        model.train()

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

        if avg_val["total loss"] < best_val_loss:
            best_val_loss = avg_val["total loss"]
            torch.save(
                {"model": model.state_dict(), "index_embedding": index_embedding.state_dict()},
                ck.checkpoint_path,
            )

        if epoch % ck.save_every == 0:
            torch.save(
                {"model": model.state_dict(), "index_embedding": index_embedding.state_dict()},
                f"checkpoint_epoch_{epoch:03d}.pt",
            )


def train_ddp(
    rank: int,
    local_rank: int,
    world_size: int,
    model: MainTrunk,
    tcfg: TrainConfig,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    index_embedding: nn.Embedding,
    device: str | None = None,
) -> None:
    """DDP training loop. Launched via torchrun — one process per GPU."""
    device = device or f"cuda:{local_rank}"
    ddp_model = DDP(model, device_ids=[local_rank])

    tp = tcfg.training
    lp = tcfg.loss
    lg = tcfg.logging
    ck = tcfg.checkpoint

    optimizer = Adam(
        list(ddp_model.parameters()) + list(index_embedding.parameters()),
        lr=tp.lr,
        weight_decay=tp.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=tp.num_epochs, eta_min=tp.lr * 0.01)

    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(1, tp.num_epochs + 1):
        ddp_model.train()
        train_loader.sampler.set_epoch(epoch)
        epoch_total_loss = epoch_MSE = epoch_CE = epoch_smooth_lddt = epoch_res_dist = (
            epoch_atom_dist
        ) = epoch_intermediate_loss = 0.0
        n_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:03d}/{tp.num_epochs}",
            leave=False,
            disable=(rank != 0),
        )

        for batch in pbar:
            featurized_batch = featurize_batch(
                _to_protein_batch(batch),
                tcfg,
                distogram_res,
                distogram_atom,
                index_embedding,
                device,
            )

            (
                r_denoised,
                f_seq_logits,
                residue_distogram_logits,
                atom_distogram_logits,
                intermediate_denoised_coord_stack,
                intermediate_pred_aa_logit_stack,
            ) = ddp_model(featurized_batch)
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
                k_loss: Float[torch.Tensor, ""] = lp.lam * atom_loss(
                    intermediate_denoised_coord, featurized_batch.r_gt, featurized_batch.atom5_mask
                ) + lp.alpha_0 * F.cross_entropy(
                    rearrange(intermediate_pred_aa_logit_stack[k_idx], "b n c -> (b n) c"),
                    rearrange(featurized_batch.aa_indices, "b n -> (b n)"),
                )
                intermediate_med_loss = intermediate_med_loss + gamma_K_minus_k * k_loss
            intermediate_med_loss = (intermediate_med_loss / max(K_unit, 1)).mean()

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
                r_denoised,
                featurized_batch.r_gt,
                featurized_batch.atom5_mask,
                cutoff=float(lp.smooth_lddt_cutoff),
            )
            CE_loss: Float[torch.Tensor, ""] = F.cross_entropy(
                rearrange(f_seq_logits, "b n c -> (b n) c"),
                rearrange(featurized_batch.aa_indices, "b n -> (b n)"),
            )

            total_loss: Float[torch.Tensor, ""] = (
                lp.lam * Kabsch_aligned_MSE_loss
                + lp.alpha_0 * CE_loss
                + lp.alpha_1 * lddt_loss
                + lp.alpha_2 * residue_distogram_loss
                + lp.alpha_3 * atom_distogram_loss
                + lp.alpha_4 * intermediate_med_loss
            )

            if rank == 0 and torch.isnan(total_loss):
                components = {
                    "mse": Kabsch_aligned_MSE_loss,
                    "ce": CE_loss,
                    "lddt": lddt_loss,
                    "res_dist": residue_distogram_loss,
                    "atom_dist": atom_distogram_loss,
                    "inter": intermediate_med_loss,
                }
                nan_keys = [k for k, v in components.items() if torch.isnan(v)]
                log.warning("nan_loss", step=global_step, nan_components=nan_keys)

            optimizer.zero_grad()
            total_loss.backward()

            if world_size > 1 and dist.is_initialized():
                for param in index_embedding.parameters():
                    if param.grad is not None:
                        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                        param.grad.div_(world_size)

            grad_norm: Float[torch.Tensor, ""] = nn.utils.clip_grad_norm_(
                list(ddp_model.parameters()) + list(index_embedding.parameters()),
                tp.grad_clip if tp.grad_clip is not None else float("inf"),
            )
            optimizer.step()

            epoch_total_loss += total_loss.item()
            epoch_MSE += Kabsch_aligned_MSE_loss.item()
            epoch_CE += CE_loss.item()
            epoch_smooth_lddt += lddt_loss.item()
            epoch_res_dist += residue_distogram_loss.item()
            epoch_atom_dist += atom_distogram_loss.item()
            epoch_intermediate_loss += intermediate_med_loss.item()
            n_batches += 1
            global_step += 1

            if rank == 0 and global_step % lg.log_interval == 0:
                pbar.set_postfix(loss=f"{total_loss.item():.4f}", gnorm=f"{grad_norm:.3f}")

        scheduler.step()

        avg_train = {
            k: v / n_batches
            for k, v in zip(
                [
                    "total loss",
                    "Kabsch aligned MSE loss",
                    "Cross Entropy loss",
                    "smooth lddt",
                    "Residue Distogram loss",
                    "Atom Distogram loss",
                    "Intermediate loss",
                ],
                [
                    epoch_total_loss,
                    epoch_MSE,
                    epoch_CE,
                    epoch_smooth_lddt,
                    epoch_res_dist,
                    epoch_atom_dist,
                    epoch_intermediate_loss,
                ],
                strict=False,
            )
        }

        _eff_world_size = world_size if dist.is_initialized() else 1
        avg_val = evaluate_ddp(
            rank,
            _eff_world_size,
            ddp_model,
            test_loader,
            tcfg,
            distogram_res,
            distogram_atom,
            index_embedding,
            device,
        )
        ddp_model.train()

        if rank == 0:
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

            if avg_val["total loss"] < best_val_loss:
                best_val_loss = avg_val["total loss"]
                torch.save(
                    {
                        "model": ddp_model.module.state_dict(),
                        "index_embedding": index_embedding.state_dict(),
                    },
                    ck.checkpoint_path,
                )

            if epoch % ck.save_every == 0:
                torch.save(
                    {
                        "model": ddp_model.module.state_dict(),
                        "index_embedding": index_embedding.state_dict(),
                    },
                    f"checkpoint_epoch_{epoch:03d}.pt",
                )


class _FileLogProcessor:
    """Structlog processor that appends JSON lines to a file, then passes the event dict through."""

    def __init__(self, path: str) -> None:
        import json as _json

        self._f = open(path, "w", buffering=1)  # line-buffered
        self._json = _json

    def __call__(self, logger, method, event_dict):
        self._f.write(self._json.dumps(event_dict) + "\n")
        return event_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train PallAtom (DDP)")
    parser.add_argument("--data", required=True, help="path to proteins.jsonl")
    parser.add_argument("--splits", required=True, help="path to splits.json")
    parser.add_argument(
        "--config", default=None, help="path to TrainConfig JSON (omit for defaults)"
    )
    parser.add_argument("--log_file", default=None, help="path to write structured JSON log lines")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    try:
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)

        _processors = [
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.stdlib.add_log_level,
            structlog.processors.StackInfoRenderer(),
        ]
        if args.log_file and rank == 0:
            _processors.append(_FileLogProcessor(args.log_file))
        if rank == 0:
            _processors.append(structlog.dev.ConsoleRenderer())

        structlog.configure(
            processors=_processors,
            wrapper_class=structlog.make_filtering_bound_logger(20),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
        )

        import traceback as _tb

        try:
            if args.config is not None:
                with open(args.config) as _f:
                    tcfg = TrainConfig.model_validate_json(_f.read())
            else:
                tcfg = TrainConfig()

            train_loader, val_loader, _ = make_ddp_data_loaders(
                tcfg,
                args.data,
                args.splits,
                rank=rank,
                world_size=world_size,
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
            ).to(device)

            dr = tcfg.distogram_res
            da = tcfg.distogram_atom
            distogram_res = Distogram(
                n_bins=dr.n_bins, min_dist=dr.min_dist, max_dist=dr.max_dist, overflow_bin=True
            ).to(device)
            distogram_atom = Distogram(
                n_bins=da.n_bins, min_dist=da.min_dist, max_dist=da.max_dist
            ).to(device)
            index_embedding = nn.Embedding(tcfg.model.max_residues, tcfg.model.c_res).to(device)

            if tcfg.training.pretrained_weights is not None:
                ckpt = torch.load(tcfg.training.pretrained_weights, map_location=device)
                model.load_state_dict(ckpt["model"])
                index_embedding.load_state_dict(ckpt["index_embedding"])
                if rank == 0:
                    log.info("loaded pretrained weights", path=tcfg.training.pretrained_weights)

            if rank == 0 and tcfg.logging.use_wandb:
                wandb.init(project=tcfg.logging.wandb_project, config=tcfg.model_dump())

            train_ddp(
                rank=rank,
                local_rank=local_rank,
                world_size=world_size,
                model=model,
                tcfg=tcfg,
                train_loader=train_loader,
                test_loader=val_loader,
                distogram_res=distogram_res,
                distogram_atom=distogram_atom,
                index_embedding=index_embedding,
            )
        except Exception as _exc:
            log.error("fatal", error=str(_exc), traceback=_tb.format_exc())
            raise SystemExit(1) from _exc
    finally:
        dist.destroy_process_group()
