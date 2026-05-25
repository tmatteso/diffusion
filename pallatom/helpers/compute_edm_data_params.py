"""Script to compute and cache EDM noise schedule data parameters."""

import argparse
from collections.abc import Mapping

import numpy as np
import torch
from einops import rearrange, reduce
from helpers.data import make_bucketed_data_loaders
from helpers.featurize import ProteinBatch
from jaxtyping import Float
from train.train_config import TrainConfig


def compute_sigma_data(train_loader: torch.utils.data.DataLoader[ProteinBatch]) -> float:
    """RMS displacement of valid atom positions from the CA centroid."""
    sum_sq = 0.0
    count = 0

    for batch in train_loader:
        atom_positions: Float[torch.Tensor, "B N_res 37 3"] = batch.atom_positions
        atom_mask: Float[torch.Tensor, "B N_res 37"] = batch.atom_mask
        sq = reduce(atom_positions**2, "b n a d -> b n a", "sum")
        sum_sq += reduce(sq * atom_mask, "b n a -> ", "sum").item()
        count += atom_mask.sum().item()

    return (sum_sq / count) ** 0.5


def compute_edm_noise_params(
    train_loader: torch.utils.data.DataLoader[ProteinBatch],
) -> Mapping[str, float]:
    """Fit sigma_max, sigma_min, P_mean, P_std from the training coordinate distribution.

    - sigma_max : 99th-percentile per-atom distance from the CA centroid
    - sigma_min : max(1st-percentile pairwise CA distance, 2e-3)
    - P_mean    : midpoint of [log sigma_min, log sigma_max]
    - P_std     : half-width of that interval (±1 std spans the full range)
    """
    dists_from_center: list[Float[torch.Tensor, "..."]] = []
    pairwise_dists: list[Float[torch.Tensor, "..."]] = []

    for batch in train_loader:
        atom_positions: Float[torch.Tensor, "B N_res 37 3"] = batch.atom_positions
        atom_mask: Float[torch.Tensor, "B N_res 37"] = batch.atom_mask
        dist = atom_positions.norm(dim=-1)  # (B, N_res, 37)
        valid = atom_mask.bool()
        dists_from_center.append(dist[valid].cpu().float())

        for b in range(atom_positions.shape[0]):
            ca_b = atom_positions[b, :, 1, :]  # (N_res, 3)
            cam_b = atom_mask[b, :, 1].bool()  # (N_res,)
            ca_valid = ca_b[cam_b]  # (M, 3)
            if ca_valid.shape[0] > 1:
                diff = rearrange(ca_valid, "m d -> m 1 d") - rearrange(
                    ca_valid, "m d -> 1 m d"
                )  # (M, M, 3)
                d = diff.norm(dim=-1)  # (M, M)
                mask = torch.triu(torch.ones_like(d, dtype=torch.bool), diagonal=1)
                triu_vals = d[mask].cpu().float()
                # cap per-protein to avoid exceeding torch.quantile's ~16M element limit
                if triu_vals.shape[0] > 1000:
                    triu_vals = triu_vals[torch.randperm(triu_vals.shape[0])[:1000]]
                pairwise_dists.append(triu_vals)

    all_dists = torch.cat(dists_from_center)
    all_pairwise = torch.cat(pairwise_dists)

    # torch.quantile fails above ~16.7 M elements; subsample if needed
    _MAX_Q = 2_000_000
    if all_dists.shape[0] > _MAX_Q:
        all_dists = all_dists[torch.randperm(all_dists.shape[0])[:_MAX_Q]]
    if all_pairwise.shape[0] > _MAX_Q:
        all_pairwise = all_pairwise[torch.randperm(all_pairwise.shape[0])[:_MAX_Q]]

    sigma_max = torch.quantile(all_dists, 0.99).item()
    sigma_min = max(torch.quantile(all_pairwise, 0.01).item(), 2e-3)

    log_min = np.log(sigma_min)
    log_max = np.log(sigma_max)
    P_mean = (log_min + log_max) / 2.0
    P_std = (log_max - log_min) / 2.0

    return {"sigma_max": sigma_max, "sigma_min": sigma_min, "P_mean": P_mean, "P_std": P_std}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute EDM noise parameters from training data")
    parser.add_argument("--data", required=True, help="path to proteins.jsonl")
    parser.add_argument("--splits", required=True, help="path to splits.json")
    parser.add_argument(
        "--config", default=None, help="path to TrainConfig JSON (omit for defaults)"
    )
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    if args.config is not None:
        with open(args.config) as f:
            tcfg = TrainConfig.model_validate_json(f.read())
    else:
        tcfg = TrainConfig()

    train_loader, _, _ = make_bucketed_data_loaders(
        cfg=tcfg,
        jsonl_path=args.data,
        splits_path=args.splits,
        num_workers=args.num_workers,
        debug_run=False,
    )

    sigma_data = compute_sigma_data(train_loader)
    print(f"sigma_data = {sigma_data:.4f}")

    params = compute_edm_noise_params(train_loader)
    print(f"sigma_max  = {params['sigma_max']:.4f}")
    print(f"sigma_min  = {params['sigma_min']:.6f}")
    print(f"P_mean     = {params['P_mean']:.4f}")
    print(f"P_std      = {params['P_std']:.4f}")
