"""FastAPI service for unconditional protein structure sampling from PallAtom."""

import asyncio
import os
import sys
import tempfile
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
from helpers.atom_utils import Protein, protein_from_pdb, to_pdb  # noqa: E402
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
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    index_embedding = nn.Embedding(mp.max_residues, mp.c_res).to(DEVICE)
    index_embedding.load_state_dict(ckpt["index_embedding"])
    index_embedding.eval()
    atom_disto = Distogram(n_bins=22, min_dist=2.0, max_dist=22.0).to(DEVICE)
    templ_disto = Distogram(
        n_bins=mp.n_bins - 1, min_dist=3.25, max_dist=50.75, overflow_bin=True
    ).to(DEVICE)
    _state.update(
        model=model,
        mp=mp,
        noise=noise,
        index_embedding=index_embedding,
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


def _protein_from_pdb_string(pdb_string: str):
    """Write pdb_string to a temp file, parse it, delete the temp file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
        f.write(pdb_string)
        path = f.name
    try:
        return protein_from_pdb(path)
    finally:
        os.unlink(path)


def _run_sampling(req: SampleRequest) -> list[str]:
    model: MainTrunk = _state["model"]
    noise: NoiseScheduleParams = _state["noise"]
    index_embedding: nn.Embedding = _state["index_embedding"]
    atom_disto: Distogram = _state["atom_disto"]
    templ_disto: Distogram = _state["templ_disto"]

    N_res = req.n_res
    B = req.n_samples
    N_atom = N_res * NATOM

    # ── atom-level conditioning ───────────────────────────────────────────────
    if req.structure_pdb is not None:
        prot = _protein_from_pdb_string(req.structure_pdb)
        n_pdb: int = prot.atom_positions.shape[0]
        if n_pdb > N_res:
            raise ValueError(
                f"structure_pdb has {n_pdb} residues but n_res={N_res}; "
                "structure_pdb must cover ≤ n_res residues"
            )
        atom_positions = torch.zeros(N_res, 37, 3)
        atom_mask = torch.zeros(N_res, 37)
        atom_positions[:n_pdb] = torch.tensor(prot.atom_positions, dtype=torch.float32)
        atom_mask[:n_pdb] = torch.tensor(prot.atom_mask, dtype=torch.float32)
        pdb_idx = torch.tensor(prot.residue_index, dtype=torch.float32)
        if n_pdb < N_res:
            last = int(pdb_idx[-1].item()) if n_pdb > 0 else -1
            tail = torch.arange(last + 1, last + 1 + (N_res - n_pdb), dtype=torch.float32)
            residue_index = torch.cat([pdb_idx, tail])
        else:
            residue_index = pdb_idx
    else:
        atom_positions = torch.zeros(N_res, 37, 3)
        atom_mask = torch.zeros(N_res, 37)
        residue_index = torch.arange(N_res, dtype=torch.float32)

    # ── sequence ─────────────────────────────────────────────────────────────
    seq: str = req.sequence if req.sequence is not None else "X" * N_res

    # ── template-distogram conditioning ──────────────────────────────────────
    pdb_files: list[str] = []
    tmp_path: str | None = None
    try:
        if req.template_pdb is not None:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
                f.write(req.template_pdb)
                tmp_path = f.name
            pdb_files = [tmp_path]

        context = build_sampling_context(
            atom_positions=atom_positions,
            atom_mask=atom_mask,
            residue_index=residue_index,
            seq=seq,
            pdb_files=pdb_files,
            index_embedding=index_embedding,
            atom_distogram_fn=atom_disto,
            templ_distogram_fn=templ_disto,
            batch_size=B,
            device=DEVICE,
        )
    finally:
        if tmp_path is not None:
            os.unlink(tmp_path)

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
        prot_out = Protein(
            atom_positions=x_37,
            atom_mask=mask_37,
            residue_index=np.arange(N_res, dtype=np.int32),
            aatype=np.zeros(N_res, dtype=np.int32),
            chain_index=np.zeros(N_res, dtype=np.int32),
            b_factors=np.ones((N_res, 37), dtype=np.float32),
        )
        pdb_strings.append(to_pdb(prot_out))

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
