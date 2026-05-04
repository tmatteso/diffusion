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
