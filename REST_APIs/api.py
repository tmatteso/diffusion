"""FastAPI service for unconditional protein structure sampling from PallAtom."""

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator

# ── resolve pallatom onto the path ────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "pallatom"))

from architecture.main_trunk import MainTrunk  # noqa: E402
from helpers.atom_utils import Protein, to_pdb  # noqa: E402
from helpers.featurize import Distogram  # noqa: E402
from sample.sampling import (  # noqa: E402
    NATOM,
    EDMPrecond,
    EDMSampler,
    atom5_to_atom37,
    build_sampling_context,
)
from train.train_config import ModelParams, NoiseScheduleParams  # noqa: E402

# ── startup configuration (override via environment variables) ────────────────
CHECKPOINT_PATH: str = os.environ.get(
    "CHECKPOINT_PATH",
    os.path.join(_HERE, "..", "pallatom", "pallatom_best_best.pt"),
)
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

_state: dict = {}
_semaphore: asyncio.Semaphore


# ── model loading ─────────────────────────────────────────────────────────────


def _load_model(
    checkpoint_path: str,
    mp: ModelParams,
    noise: NoiseScheduleParams,
    device: str,
) -> MainTrunk:
    model = MainTrunk(
        f_ref_dim=mp.f_ref_dim,
        n_bins=mp.n_bins,
        c_atom=mp.c_atom,
        c_pair=mp.c_pair,
        c_res=mp.c_res,
        c_atompair=mp.c_atompair,
        K_unit=mp.K_unit,
        sigma_data=noise.sigma_data,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _semaphore
    _semaphore = asyncio.Semaphore(1)

    if not os.path.exists(CHECKPOINT_PATH):
        raise RuntimeError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    mp = ModelParams()
    noise = NoiseScheduleParams()
    model = _load_model(CHECKPOINT_PATH, mp, noise, DEVICE)
    atom_disto = Distogram(n_bins=22, min_dist=2.0, max_dist=22.0).to(DEVICE)
    templ_disto = Distogram(
        n_bins=mp.n_bins - 1, min_dist=3.25, max_dist=50.75, overflow_bin=True
    ).to(DEVICE)
    _state.update(
        model=model,
        mp=mp,
        noise=noise,
        atom_disto=atom_disto,
        templ_disto=templ_disto,
    )
    yield
    _state.clear()


app = FastAPI(
    title="PallAtom Diffusion API",
    description="Unconditional protein backbone sampling via EDM diffusion.",
    lifespan=lifespan,
)


# ── request / response schemas ────────────────────────────────────────────────

_VALID_AA: frozenset[str] = frozenset("ARNDCQEGHILKMFPSTWYVX")


class SampleRequest(BaseModel):
    # --- target ---
    n_res: int = Field(..., gt=0, le=512, description="Number of residues to generate")

    # --- conditioning (all optional) ---
    sequence: str | None = Field(
        None,
        description=(
            "Amino-acid sequence of length n_res. "
            "Standard 20 AAs plus 'X' for unknown. "
            "Omit for no sequence conditioning ('X' * n_res is used internally)."
        ),
    )
    structure_pdb: str | None = Field(
        None,
        description=(
            "PDB string for atom-level conditioning. "
            "Residues present in the PDB fill r_gt/atom5_mask; uncovered positions are zeroed. "
            "Must cover ≤ n_res residues."
        ),
    )
    template_pdb: str | None = Field(
        None,
        description=(
            "PDB string for template-distogram conditioning. "
            "May cover fewer than n_res residues (padded with zeros)."
        ),
    )

    # --- sampler ---
    n_samples: int = Field(
        1, gt=0, le=10, description="Number of structures to generate in parallel"
    )
    ddim_steps: int = Field(40, gt=1, description="ODE solver steps (more = higher quality)")
    rho: float = Field(7.0, gt=0, description="Karras noise-schedule exponent")
    S_churn: float = Field(0.0, ge=0, description="Stochasticity per step (0 = deterministic)")
    S_noise: float = Field(1.003, gt=0, description="Noise scaling for stochastic steps")

    @field_validator("sequence")
    @classmethod
    def sequence_valid_characters(cls, v: str | None) -> str | None:
        if v is not None:
            invalid = set(v) - _VALID_AA
            if invalid:
                raise ValueError(f"Invalid characters in sequence: {sorted(invalid)!r}")
        return v

    @model_validator(mode="after")
    def sequence_length_matches_n_res(self) -> "SampleRequest":
        if self.sequence is not None and len(self.sequence) != self.n_res:
            raise ValueError(f"sequence length {len(self.sequence)} must equal n_res {self.n_res}")
        return self


class SampleResponse(BaseModel):
    pdb_strings: list[str]
    n_res: int
    n_samples: int
    device: str


# ── synchronous sampling work (runs in a thread-pool executor) ────────────────


def _run_sampling(req: SampleRequest) -> list[str]:
    model: MainTrunk = _state["model"]
    index_embedding: nn.Embedding = _state["index_embedding"]
    mp: ModelParams = _state["mp"]
    noise: NoiseScheduleParams = _state["noise"]

    N_res = req.n_res
    N_atom = N_res * NATOM
    B = req.n_samples

    context = build_sampling_context(
        N_res,
        index_embedding,
        batch_size=B,
        n_templ_bins=mp.n_bins,
        device=DEVICE,
    )
    edm_precond = EDMPrecond(
        model,
        context,
        sigma_min=noise.sigma_min,
        sigma_max=noise.sigma_max,
    ).to(DEVICE)
    edm_precond.eval()

    edm_sampler = EDMSampler(
        edm_precond,
        sigma_min=noise.sigma_min,
        sigma_max=noise.sigma_max,
        rho=req.rho,
        S_churn=req.S_churn,
        S_tmin=0.0,
        S_tmax=float("inf"),
        S_noise=req.S_noise,
    )

    coords_batch = edm_sampler.sample(
        shape=(B, N_atom, 3),
        steps=req.ddim_steps,
        device=DEVICE,
    )

    pdb_strings: list[str] = []
    for b in range(B):
        coords_np = rearrange(coords_batch[b].cpu().numpy(), "(n a) d -> n a d", n=N_res, a=NATOM)
        x_37, mask_37 = atom5_to_atom37(coords_np)
        prot = Protein(
            atom_positions=x_37,
            atom_mask=mask_37,
            residue_index=np.arange(N_res, dtype=np.int32),
            aatype=np.zeros(N_res, dtype=np.int32),
            chain_index=np.zeros(N_res, dtype=np.int32),
            b_factors=np.ones((N_res, 37), dtype=np.float32),
        )
        pdb_strings.append(to_pdb(prot))

    return pdb_strings


# ── endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "device": DEVICE, "model_loaded": bool(_state)}


@app.post("/sample", response_model=SampleResponse)
async def sample(request: SampleRequest) -> SampleResponse:
    """
    Generate unconditional protein backbone structures.

    Returns one PDB string per requested sample.
    Requests are serialised (one sampling job runs at a time).
    """
    async with _semaphore:
        loop = asyncio.get_event_loop()
        try:
            pdb_strings = await loop.run_in_executor(None, partial(_run_sampling, request))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return SampleResponse(
        pdb_strings=pdb_strings,
        n_res=request.n_res,
        n_samples=request.n_samples,
        device=DEVICE,
    )
