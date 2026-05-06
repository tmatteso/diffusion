import pytest
import torch
import torch.nn.functional as F
from beartype import beartype
from einops import einsum, rearrange, reduce, repeat
from jaxtyping import Bool, Float, Int, jaxtyped

from architecture.main_trunk import (
    AtomDistogramHead,
    MainTrunk,
    RelativePositionEncoding,
    ResidueDistogramHead,
    TimeFourierEmbedding,
    scatter_mean,
    sinusoidal_encoding,
)
from helpers.featurize import FeaturizedBatch

torch.manual_seed(42)

B = 2
N_RES = 50
ATOMS_PER_RES = 3
N_ATOM = N_RES * ATOMS_PER_RES   # 150
E = 4                             # element one-hot dim
C_RES = 32
C_PAIR = 32
C_ATOM = 32
C_ATOMPAIR = 16
N_BINS = 38
N_ATOM_BINS = 22   # distinct from N_BINS; must match AtomDistogramParams.n_bins
K_UNIT = 2
# WINDOW_SIZE=32 (half=16); interior residues span 31 residues × 3 atoms/res = 93 atom neighbours
K_SPARSE = 93
F_REF_DIM = ATOMS_PER_RES * (3 + E)  # encoder groups all sibling atoms: n_per_res*(pos_dim+elem_dim)


# ---------------------------------------------------------------------------
# Typed helpers
# ---------------------------------------------------------------------------

@jaxtyped(typechecker=beartype)
def sq_dist_matrix(
    x: Float[torch.Tensor, "N D"],
) -> Float[torch.Tensor, "N N"]:
    diff = rearrange(x, "n d -> n 1 d") - rearrange(x, "n d -> 1 n d")
    return einsum(diff, diff, "n m d, n m d -> n m")


@jaxtyped(typechecker=beartype)
def mean_abs_asymmetry(
    x: Float[torch.Tensor, "B N N D"],
) -> Float[torch.Tensor, ""]:
    diff = x - rearrange(x, "b i j d -> b j i d")
    return reduce(diff.abs(), "b i j d -> ", "mean")


# ---------------------------------------------------------------------------
# Model fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def model():
    return MainTrunk(
        f_ref_dim=F_REF_DIM,
        n_bins=N_BINS,
        n_atom_bins=N_ATOM_BINS,
        c_atom=C_ATOM,
        c_pair=C_PAIR,
        c_res=C_RES,
        c_atompair=C_ATOMPAIR,
        n_blocks=1,
        n_heads=2,
        K_unit=K_UNIT,
    ).eval()


# ---------------------------------------------------------------------------
# Input tensor fixtures  (all have leading B dim)
# ---------------------------------------------------------------------------

@pytest.fixture
def ref_pos():
    return torch.randn(B, N_ATOM, 3)


@pytest.fixture
def ref_element():
    return F.one_hot(torch.randint(0, E, (B, N_ATOM)), num_classes=E).float()


@pytest.fixture
def ref_space_uid():
    return torch.zeros(B, N_ATOM, dtype=torch.long)


@pytest.fixture
def f_distogram():
    return F.one_hot(torch.randint(0, N_BINS, (B, N_RES, N_RES)), num_classes=N_BINS).float()


@pytest.fixture
def f_pseudo_beta_mask():
    return torch.ones(B, N_RES)


@pytest.fixture
def f_residue_idx():
    return torch.randn(B, N_RES, C_RES)


@pytest.fixture
def r_input():
    return torch.randn(B, N_ATOM, 3)


@pytest.fixture
def tok_idx():
    single = torch.repeat_interleave(torch.arange(N_RES), ATOMS_PER_RES)
    return single.unsqueeze(0).expand(B, -1).contiguous()


@pytest.fixture
def center_uid():
    single = torch.arange(0, N_ATOM, ATOMS_PER_RES)
    return single.unsqueeze(0).expand(B, -1).contiguous()


@pytest.fixture
def gt_atom_distogram_sparse():
    return torch.randn(B, N_ATOM, K_SPARSE, N_ATOM_BINS)


@pytest.fixture
def gt_atom_distogram_mask_sparse():
    return torch.ones(B, N_ATOM, K_SPARSE, dtype=torch.bool)


# ---------------------------------------------------------------------------
# FeaturizedBatch fixture and forward helper
# ---------------------------------------------------------------------------

@pytest.fixture
def featurized_batch(
    ref_pos,
    ref_element,
    ref_space_uid,
    f_distogram,
    f_pseudo_beta_mask,
    f_residue_idx,
    r_input,
    tok_idx,
    center_uid,
    gt_atom_distogram_sparse,
    gt_atom_distogram_mask_sparse,
) -> FeaturizedBatch:
    return FeaturizedBatch(
        ref_pos=ref_pos,
        ref_element=ref_element,
        ref_space_uid=ref_space_uid,
        gt_res_distogram=f_distogram.long(),
        f_pseudo_beta_mask=f_pseudo_beta_mask.long(),
        f_residue_idx=f_residue_idx,
        r_input=r_input,
        r_gt=torch.zeros_like(r_input),
        atom5_mask=torch.ones(B, N_ATOM, dtype=torch.bool),
        aa_indices=torch.zeros(B, N_RES, dtype=torch.long),
        residue_mask=torch.ones(B, N_RES, dtype=torch.bool),
        t_hat=1.0,
        t_normalized=0.5,
        tok_idx=tok_idx,
        center_uid=center_uid,
        gt_atom_distogram_sparse=gt_atom_distogram_sparse,
        gt_atom_distogram_mask_sparse=gt_atom_distogram_mask_sparse,
    )


def _forward(model: MainTrunk, batch: FeaturizedBatch):
    with torch.no_grad():
        return model(batch)


# ---------------------------------------------------------------------------
# sinusoidal_encoding
# ---------------------------------------------------------------------------

def test_sinusoidal_encoding_output_shape():
    positions = repeat(torch.arange(N_RES, dtype=torch.float32), "n -> b n", b=2)
    out = sinusoidal_encoding(positions, dim=C_RES)
    assert out.shape == (2, N_RES, C_RES)
    assert torch.isfinite(out).all()


def test_sinusoidal_encoding_varies_across_positions():
    positions = repeat(torch.arange(N_RES, dtype=torch.float32), "n -> 1 n")
    enc = rearrange(sinusoidal_encoding(positions, dim=C_RES), "1 n d -> n d")
    off_diag = sq_dist_matrix(enc) + torch.eye(N_RES) * 1e10
    assert off_diag.min().item() > 0


# ---------------------------------------------------------------------------
# TimeFourierEmbedding
# ---------------------------------------------------------------------------

def test_time_fourier_embedding_output_shape(model):
    out = model.time_fourier(torch.randn(N_RES))
    assert out.shape == (N_RES, C_RES)
    assert torch.isfinite(out).all()


def test_time_fourier_embedding_gradient_flows_to_freqs():
    emb = TimeFourierEmbedding(C_RES)
    reduce(emb(torch.randn(N_RES)), "n d -> ", "sum").backward()
    assert emb.freqs.grad is not None
    assert torch.isfinite(emb.freqs.grad).all()


# ---------------------------------------------------------------------------
# RelativePositionEncoding
# ---------------------------------------------------------------------------

def test_rel_pos_enc_output_shape(model):
    out = model.rel_pos_enc(N_RES, torch.device("cpu"))
    assert out.shape == (N_RES, N_RES, C_PAIR)
    assert torch.isfinite(out).all()


def test_rel_pos_enc_deterministic(model):
    out1 = model.rel_pos_enc(N_RES, torch.device("cpu"))
    out2 = model.rel_pos_enc(N_RES, torch.device("cpu"))
    assert torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# ResidueDistogramHead
# ---------------------------------------------------------------------------

def test_residue_distogram_head_output_shape():
    head = ResidueDistogramHead(C_PAIR, n_bins=N_BINS)
    logits = head(torch.randn(B, N_RES, N_RES, C_PAIR))
    assert logits.shape == (B, N_RES, N_RES, N_BINS)
    assert torch.isfinite(logits).all()


def test_residue_distogram_head_output_symmetric():
    head = ResidueDistogramHead(C_PAIR, n_bins=N_BINS)
    logits = head(torch.randn(B, N_RES, N_RES, C_PAIR))
    assert mean_abs_asymmetry(logits).item() < 1e-5


# ---------------------------------------------------------------------------
# AtomDistogramHead
# ---------------------------------------------------------------------------

def test_atom_distogram_head_output_shapes():
    head = AtomDistogramHead(C_ATOMPAIR, n_bins=N_BINS, atoms_per_res=ATOMS_PER_RES)
    logits, mask = head(torch.randn(N_ATOM, N_ATOM, C_ATOMPAIR))
    assert logits.shape == (N_ATOM, N_ATOM, N_BINS)
    assert mask.shape == (N_ATOM, N_ATOM)
    assert mask.dtype == torch.bool
    assert torch.isfinite(logits).all()


def test_atom_distogram_head_mask_includes_diagonal():
    head = AtomDistogramHead(C_ATOMPAIR, n_bins=N_BINS, atoms_per_res=ATOMS_PER_RES)
    _, mask = head(torch.randn(N_ATOM, N_ATOM, C_ATOMPAIR))
    assert mask.diagonal().all()


def test_atom_distogram_head_mask_symmetric():
    head = AtomDistogramHead(C_ATOMPAIR, n_bins=N_BINS, atoms_per_res=ATOMS_PER_RES)
    _, mask = head(torch.randn(N_ATOM, N_ATOM, C_ATOMPAIR))
    assert torch.equal(mask, rearrange(mask, "i j -> j i"))


# ---------------------------------------------------------------------------
# MainTrunk.forward — output shapes and values
# ---------------------------------------------------------------------------

def test_main_trunk_r_denoised_shape_finite(model, featurized_batch):
    r_denoised, *_ = _forward(model, featurized_batch)
    assert r_denoised.shape == (B, N_ATOM, 3)
    assert torch.isfinite(r_denoised).all()


def test_main_trunk_seq_logits_shape_finite(model, featurized_batch):
    _, f_seq_logits, *_ = _forward(model, featurized_batch)
    assert f_seq_logits.shape == (B, N_RES, 20)
    assert torch.isfinite(f_seq_logits).all()


def test_main_trunk_residue_distogram_shape_finite(model, featurized_batch):
    _, _, res_logits, *_ = _forward(model, featurized_batch)
    assert res_logits.shape == (B, N_RES, N_RES, N_BINS)
    assert torch.isfinite(res_logits).all()


def test_main_trunk_atom_distogram_shape_finite(model, featurized_batch):
    _, _, _, atom_logits, *_ = _forward(model, featurized_batch)
    assert atom_logits.ndim == 4
    assert atom_logits.shape[0] == B
    assert atom_logits.shape[1] == N_ATOM
    assert atom_logits.shape[3] == N_ATOM_BINS
    assert torch.isfinite(atom_logits).all()


def test_main_trunk_atom_distogram_bins_match_ground_truth(model, featurized_batch):
    _, _, _, atom_logits, *_ = _forward(model, featurized_batch)
    assert atom_logits.shape[-1] == featurized_batch.gt_atom_distogram_sparse.shape[-1]


def test_main_trunk_distogram_loss_atom_computable(model, featurized_batch):
    from architecture.losses import distogram_loss_atom
    _, _, _, atom_logits, *_ = _forward(model, featurized_batch)
    loss = distogram_loss_atom(
        atom_logits,
        featurized_batch.gt_atom_distogram_sparse,
        featurized_batch.gt_atom_distogram_mask_sparse,
    ).mean()
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_main_trunk_intermediate_stack_lengths(model, featurized_batch):
    _, _, _, _, coord_stack, aa_stack = _forward(model, featurized_batch)
    assert len(coord_stack) == K_UNIT
    assert len(aa_stack) == K_UNIT


def test_main_trunk_intermediate_coords_shape_finite(model, featurized_batch):
    _, _, _, _, coord_stack, _ = _forward(model, featurized_batch)
    for r in coord_stack:
        assert r.shape == (B, N_ATOM, 3)
        assert torch.isfinite(r).all()


def test_main_trunk_gradient_flows_to_r_input(model, featurized_batch):
    r_input_g = torch.randn(B, N_ATOM, 3, requires_grad=True)
    batch_g = FeaturizedBatch(
        **{k: v for k, v in featurized_batch.__dict__.items() if k != "r_input"},
        r_input=r_input_g,
    )
    r_denoised, *_ = model(batch_g)
    reduce(r_denoised, "b n d -> ", "sum").backward()
    assert r_input_g.grad is not None
    assert torch.isfinite(r_input_g.grad).all()

# you need to add integration tests here. don't be afraid to use pytest mocks.


# ---------------------------------------------------------------------------
# scatter_mean
# ---------------------------------------------------------------------------

_SM_B = 2
_SM_N_RES = 4
_SM_N_ATOM = 8   # 2 atoms per residue
_SM_C = 6


@pytest.fixture
def sm_src() -> Float[torch.Tensor, "B N_atom C"]:
    return torch.arange(_SM_B * _SM_N_ATOM * _SM_C, dtype=torch.float32).reshape(
        _SM_B, _SM_N_ATOM, _SM_C
    )


@pytest.fixture
def sm_tok_idx() -> Int[torch.Tensor, "B N_atom"]:
    # 2 atoms per residue: [0, 0, 1, 1, 2, 2, 3, 3]
    single = torch.repeat_interleave(torch.arange(_SM_N_RES), 2)
    return repeat(single, "n -> b n", b=_SM_B)


@pytest.fixture
def sm_index(sm_tok_idx) -> Int[torch.Tensor, "B N_atom"]:
    # batch-offset flat index: tok_idx + b * N_RES
    offset = repeat(torch.arange(_SM_B) * _SM_N_RES, "b -> b n", n=_SM_N_ATOM)
    return sm_tok_idx + offset


def test_scatter_mean_output_shape(sm_src, sm_index):
    out = scatter_mean(sm_src, sm_index, _SM_B * _SM_N_RES, _SM_B)
    assert out.shape == (_SM_B, _SM_N_RES, _SM_C)


def test_scatter_mean_output_finite(sm_src, sm_index):
    out = scatter_mean(sm_src, sm_index, _SM_B * _SM_N_RES, _SM_B)
    assert torch.isfinite(out).all()


def test_scatter_mean_known_values_uniform():
    # B=1, 2 atoms per residue, C=1; manually verify pair-wise means
    src = torch.tensor([[[0.], [2.], [4.], [6.]]])  # (1, 4, 1)
    index = torch.tensor([[0, 0, 1, 1]])             # (1, 4)
    out = scatter_mean(src, index, 2, 1)
    expected = torch.tensor([[[1.], [5.]]])           # mean(0,2)=1, mean(4,6)=5
    assert torch.allclose(out, expected)


def test_scatter_mean_known_values_nonuniform():
    # B=1, residue 0 has 3 atoms, residue 1 has 2 atoms, C=1
    src = torch.tensor([[[1.], [3.], [5.], [7.], [9.]]])  # (1, 5, 1)
    index = torch.tensor([[0, 0, 0, 1, 1]])               # (1, 5)
    out = scatter_mean(src, index, 2, 1)
    expected = torch.tensor([[[3.], [8.]]])                # mean(1,3,5)=3, mean(7,9)=8
    assert torch.allclose(out, expected)


def test_scatter_mean_one_atom_per_residue():
    # 1:1 atom-to-residue mapping — output must equal src exactly
    B_sm, N_sm, C_sm = 2, 4, 6
    src = torch.randn(B_sm, N_sm, C_sm)
    base   = repeat(torch.arange(N_sm), "n -> b n", b=B_sm)
    offset = repeat(torch.arange(B_sm) * N_sm, "b -> b n", n=N_sm)
    index  = base + offset   # [[0,1,2,3],[4,5,6,7]]
    out = scatter_mean(src, index, B_sm * N_sm, B_sm)
    assert torch.allclose(out, src)


def test_scatter_mean_constant_src_returns_that_constant():
    # Every atom in every residue holds the same value; mean must equal it
    B_sm, N_tgt, atoms_per, C_sm = 2, 2, 3, 4
    N_src = N_tgt * atoms_per
    val = 3.14
    src = torch.full((B_sm, N_src, C_sm), val)
    tok_idx = torch.repeat_interleave(torch.arange(N_tgt), atoms_per)
    tok_idx = repeat(tok_idx, "n -> b n", b=B_sm)
    offset  = repeat(torch.arange(B_sm) * N_tgt, "b -> b n", n=N_src)
    index   = tok_idx + offset
    out = scatter_mean(src, index, B_sm * N_tgt, B_sm)
    assert torch.allclose(out, torch.full((B_sm, N_tgt, C_sm), val))


def test_scatter_mean_batch_items_independent():
    # Zeroing batch item 1's src must leave batch item 0's output unchanged
    B_sm, N_src, N_tgt, C_sm = 2, 8, 4, 6
    src  = torch.randn(B_sm, N_src, C_sm)
    src0 = src.clone()
    src0[1] = 0.0

    tok_base = torch.repeat_interleave(torch.arange(N_tgt), 2)  # [0,0,1,1,2,2,3,3]
    tok_idx  = repeat(tok_base, "n -> b n", b=B_sm)
    offset   = repeat(torch.arange(B_sm) * N_tgt, "b -> b n", n=N_src)
    index    = tok_idx + offset

    out_full = scatter_mean(src,  index, B_sm * N_tgt, B_sm)
    out_zero = scatter_mean(src0, index, B_sm * N_tgt, B_sm)
    assert torch.allclose(out_full[0], out_zero[0])


def test_scatter_mean_multichannel_mean_matches_per_channel():
    # Verify that scatter_mean is equivalent to running per-channel means manually
    B_sm, N_tgt, atoms_per, C_sm = 1, 3, 4, 5
    N_src = N_tgt * atoms_per
    src = torch.randn(B_sm, N_src, C_sm)
    tok_idx = torch.repeat_interleave(torch.arange(N_tgt), atoms_per)
    index   = repeat(tok_idx, "n -> b n", b=B_sm)

    out = scatter_mean(src, index, B_sm * N_tgt, B_sm)

    # Per-residue means computed with reshape+reduce as ground truth
    src_grouped: Float[torch.Tensor, "B N_tgt atoms_per C"] = rearrange(
        src, "b (n a) c -> b n a c", n=N_tgt, a=atoms_per
    )
    expected: Float[torch.Tensor, "B N_tgt C"] = reduce(src_grouped, "b n a c -> b n c", "mean")
    assert torch.allclose(out, expected, atol=1e-6)
