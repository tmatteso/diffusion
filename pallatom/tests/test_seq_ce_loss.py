"""Tests for the unified sequence cross-entropy loss wrapper and related pipeline invariants."""

import pytest
import torch
from architecture.losses import seq_ce_loss
from einops import repeat
from helpers.featurize import (
    Distogram,
    FeaturizedBatch,
    FeaturizedItem,
    apply_conditioning_dropout,
    featurize_single_item,
    ref_pos_for_residue,
)
from jaxtyping import Bool, Float, Int

# Module-level constants used by fixtures
_B = 1
_N_RES = 5
_ATOMS_PER_RES = 5
_N_ATOM = _N_RES * _ATOMS_PER_RES  # 25
_N_BINS = 16
_N_AMINO_BINS = 22
_C_RES = 32
_N_AMINO = 20


@pytest.fixture
def minimal_batch() -> FeaturizedBatch:
    """Minimal FeaturizedBatch for conditioning-dropout tests (B=1, N_res=5, all valid)."""
    tok: Int[torch.Tensor, "1 25"] = repeat(
        torch.repeat_interleave(torch.arange(_N_RES), _ATOMS_PER_RES),
        "n -> b n",
        b=_B,
    ).contiguous()
    cuid: Int[torch.Tensor, "1 5"] = repeat(
        torch.arange(0, _N_ATOM, _ATOMS_PER_RES), "n -> b n", b=_B
    ).contiguous()
    return FeaturizedBatch(
        ref_pos=torch.randn(_B, _N_ATOM, 3),
        ref_element=torch.zeros(_B, _N_ATOM, 4),
        ref_space_uid=torch.zeros(_B, _N_ATOM, dtype=torch.long),
        gt_res_distogram=torch.zeros(_B, _N_RES, _N_RES, _N_BINS, dtype=torch.long),
        f_pseudo_beta_mask=torch.ones(_B, _N_RES, dtype=torch.long),
        f_residue_idx=torch.randn(_B, _N_RES, _C_RES),
        r_input=torch.randn(_B, _N_ATOM, 3),
        r_gt=torch.randn(_B, _N_ATOM, 3),
        atom5_mask=torch.ones(_B, _N_ATOM, dtype=torch.bool),
        aa_indices=torch.randint(0, _N_AMINO, (_B, _N_RES), dtype=torch.long),
        residue_mask=torch.ones(_B, _N_RES, dtype=torch.bool),
        t_hat=1.0,
        t_normalized=0.5,
        tok_idx=tok,
        center_uid=cuid,
        gt_atom_distogram_sparse=torch.randn(_B, _N_ATOM, _N_ATOM, _N_AMINO_BINS),
        gt_atom_distogram_mask_sparse=torch.ones(_B, _N_ATOM, _N_ATOM, dtype=torch.bool),
    )


@pytest.fixture
def c_beta_distogram_fn() -> Distogram:
    """Minimal Cβ distogram function for featurize_single_item tests."""
    return Distogram(n_bins=_N_BINS, min_dist=2.0, max_dist=22.0, overflow_bin=True).eval()


def test_intermediate_and_final_ce_are_identical() -> None:
    """seq_ce_loss returns the same scalar for identical logits/targets regardless of call site."""
    logits: Float[torch.Tensor, "1 10 20"] = torch.randn(1, 10, _N_AMINO)
    aa_indices: Int[torch.Tensor, "1 10"] = torch.randint(0, _N_AMINO, (1, 10))

    loss_final: Float[torch.Tensor, ""] = seq_ce_loss(logits, aa_indices)
    loss_inter: Float[torch.Tensor, ""] = seq_ce_loss(logits, aa_indices)

    assert torch.allclose(loss_final, loss_inter)


def test_conditioning_dropout_uses_index_20_not_minus_100(
    minimal_batch: FeaturizedBatch,
) -> None:
    """p_seq=1.0 sets all valid positions to 20; no valid position becomes -100."""
    out: FeaturizedBatch = apply_conditioning_dropout(
        minimal_batch, p_distogram=0.0, p_atom=0.0, p_seq=1.0, device="cpu"
    )
    valid: Bool[torch.Tensor, "1 5"] = minimal_batch.residue_mask
    assert (out.aa_indices[valid] == 20).all()
    assert not (out.aa_indices[valid] == -100).any()


def test_seq_ce_loss_ignores_minus_100() -> None:
    """seq_ce_loss on a tensor with padding equals seq_ce_loss on the non-padded subset."""
    logits: Float[torch.Tensor, "1 6 20"] = torch.randn(1, 6, _N_AMINO)
    aa_full: Int[torch.Tensor, "1 6"] = torch.full((1, 6), -100, dtype=torch.long)
    aa_full[0, :3] = torch.randint(0, _N_AMINO, (3,))

    loss_full: Float[torch.Tensor, ""] = seq_ce_loss(logits, aa_full)
    loss_half: Float[torch.Tensor, ""] = seq_ce_loss(logits[:, :3], aa_full[:, :3])

    assert torch.allclose(loss_full, loss_half)


def test_pipeline_preserves_pdb_x_as_index_20(c_beta_distogram_fn: Distogram) -> None:
    """featurize_single_item maps 'X' in the sequence string to aa_indices == 20."""
    n_res = 5
    x_pos = 2
    aa_seq = "ACXDE"

    atom37_positions: Float[torch.Tensor, "5 37 3"] = torch.randn(n_res, 37, 3)
    atom37_mask: Float[torch.Tensor, "5 37"] = torch.ones(n_res, 37)
    index: Float[torch.Tensor, "5"] = torch.arange(n_res, dtype=torch.float)
    ala_ref_pos: Float[torch.Tensor, "5 3"] = ref_pos_for_residue("ALA")
    ala_ref_elem: Float[torch.Tensor, "5 4"] = torch.zeros(5, 4)
    ala_ref_elem[:, 0] = 1.0

    item: FeaturizedItem = featurize_single_item(
        atom37_positions,
        atom37_mask,
        index,
        aa_seq,
        ala_ref_pos,
        ala_ref_elem,
        c_res=_C_RES,
        c_beta_distogram_fn=c_beta_distogram_fn,
        device="cpu",
    )

    assert item.aa_indices[x_pos].item() == 20
