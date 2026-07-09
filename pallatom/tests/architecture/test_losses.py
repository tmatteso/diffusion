"""Tests for diffusion model loss functions.

Covers atom_loss, med_loss, smooth_lddt_loss, distogram_loss_residue,
distogram_loss_atom, and seq_ce_loss, including shape contracts, gradient
flow, numerical correctness, boundary conditions, padding masking,
all-masked edge case, conditioning-dropout token conventions, and the
featurize pipeline's handling of unknown amino acid 'X'.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from architecture.errors import (
    AtomResidueCountMismatchError,
    BlockCountMismatchError,
)
from architecture.losses import (
    NONPOLAR_RESIDUE_WEIGHT,
    POLAR_RESIDUE_WEIGHT,
    POLAR_RESIDUES,
    RESIDUE_TYPE_WEIGHTS,
    atom_loss,
    distogram_loss_atom,
    distogram_loss_residue,
    med_loss,
    pairwise_dist,
    residue_type_weight,
    seq_ce_loss,
    smooth_lddt_loss,
)
from beartype import beartype
from einops import einsum, rearrange, repeat
from helpers.alignment import kabsch_align
from helpers.atom_utils import (
    RESTYPE_NUM_NO_X,
    RESTYPES_NO_X,
    Protein,
    restype_order,
)
from helpers.data import (
    Distogram,
    FeaturizedBatch,
    FeaturizedItem,
    build_distogram_module,
    featurize_single_item,
)
from helpers.useful_objects import manual_seed
from jaxtyping import Bool, Float, Int, TypeCheckError, jaxtyped
from train.train_config import (
    AtomDistogramParams,
    LossParams,
    ResidueDistogramParams,
    TemplateDistogramParams,
    TrainConfig,
)

_ = manual_seed(42)

B, N = 4, 50
K = 6
VOCAB = 20
LAM, ALPHA, GAMMA = 1.0, 0.1, 0.99
L_RES, B_RES = 32, 64
ATOMS_PER_RES, B_ATOM = 14, 22
N_ATOMS = 8 * ATOMS_PER_RES  # 112
TOLERANCE = 1e-5
TIGHT_TOLERANCE = 1e-3
K_NEIGH = 4
N_ATOM_BINS = 10
N_TEMPL_BINS = 38

_B = 1
_N_RES = 5
_ATOMS_PER_RES = 5
_N_ATOM = _N_RES * _ATOMS_PER_RES  # 25
_N_BINS = 16
_N_AMINO_BINS = 22
_N_AMINO = RESTYPE_NUM_NO_X
PADDING_DROPOUT_TOKEN = -100

# ---------------------------------------------------------------------------
# Typed helpers
# ---------------------------------------------------------------------------


@jaxtyped(typechecker=beartype)
def make_noisy(
    base: Float[torch.Tensor, "... N 3"],
    sigma: float,
) -> Float[torch.Tensor, "... N 3"]:
    """Add zero-mean Gaussian noise with s.d. sigma to base coordinates.

    Args:
        base: Input coordinate tensor of shape (..., N, 3).
        sigma: Standard deviation of the Gaussian noise to add.

    Returns:
        Noisy coordinate tensor with the same shape as base.
    """
    return base + sigma * torch.randn_like(base)


@jaxtyped(typechecker=beartype)
def pairwise_sq_dist(
    x: Float[torch.Tensor, "N 3"],
) -> Float[torch.Tensor, "N N"]:
    """Compute the symmetric N by N pairwise squared Euclidean distance matrix.

    Args:
        x: Point cloud of shape (N, 3).

    Returns:
        Symmetric (N, N) matrix of squared distances; diagonal is zero.
    """
    diff = rearrange(x, "n d -> n 1 d") - rearrange(x, "n d -> 1 n d")
    return einsum(diff, diff, "n m d, n m d -> n m")


@jaxtyped(typechecker=beartype)
def local_window_mask(
    N_atoms: int,
    half_window: int,
) -> Bool[torch.Tensor, "N_atoms N_atoms"]:
    """Boolean mask where entry (i,j) is True iff |i-j| <= half_window.

    Args:
        N_atoms: Total number of atoms (side length of the square mask).
        half_window: Maximum index distance to include in the local window.

    Returns:
        Boolean (N_atoms, N_atoms) mask that is True within the window.
    """
    idx = torch.arange(N_atoms)
    dist = (rearrange(idx, "n -> n 1") - rearrange(idx, "m -> 1 m")).abs()
    return dist <= half_window


@jaxtyped(typechecker=beartype)
def to_onehot(
    indices: Int[torch.Tensor, "... L"],
    n_classes: int,
) -> Float[torch.Tensor, "... L n_classes"]:
    """Convert integer indices to float one-hot vectors of length n_classes.

    Args:
        indices: Integer class indices of shape (..., L).
        n_classes: Total number of classes for the one-hot encoding.

    Returns:
        Float one-hot tensor of shape (..., L, n_classes).
    """
    return F.one_hot(indices, num_classes=n_classes).float()


@jaxtyped(typechecker=beartype)
def block_decay_weights(
    n_blocks: int,
    gamma: float,
) -> Float[torch.Tensor, "n_blocks"]:
    """Exponentially decaying weights gamma^(K-k) for k=1..K, last weight = 1.

    Args:
        n_blocks: Number of decoder blocks.
        gamma: Exponential decay base in (0, 1].

    Returns:
        Weight vector of length n_blocks; the final entry is always 1.0.
    """
    ks = torch.arange(1, n_blocks + 1, dtype=torch.float32)
    return gamma ** (n_blocks - ks)


@jaxtyped(typechecker=beartype)
def ce_via_einsum(
    logits: Float[torch.Tensor, "... B"],
    targets: Float[torch.Tensor, "... B"],
) -> Float[torch.Tensor, "..."]:
    """Compute cross-entropy -sum(targets * log_softmax(logits)) via einsum.

    Args:
        logits: Raw (un-normalised) log scores of shape (..., B).
        targets: Soft target distribution of shape (..., B); must sum to 1
            along last dim.

    Returns:
        Scalar cross-entropy for each element in the leading dimensions.
    """
    log_p = F.log_softmax(logits, dim=-1)
    return -einsum(targets, log_p, "... b, ... b -> ...")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def coords() -> Float[torch.Tensor, "B N_atoms 3"]:
    """Provide a random (B, N, 3) coordinate tensor.

    Returns:
        Random float tensor of shape (B, N, 3).
    """
    return torch.randn(B, N, 3)


@pytest.fixture
def uniform_aa_indices() -> Int[torch.Tensor, "B N_atoms"]:
    """Amino-acid indices of a single nonpolar residue type ('A' = 0).

    Used by atom_loss tests that don't care about residue-type weighting;
    a uniform residue type makes that weighting a constant factor that
    doesn't affect relative comparisons between losses.

    Returns:
        Integer tensor of shape (B, N), all entries 0.
    """
    return torch.zeros(B, N, dtype=torch.long)


@pytest.fixture
def noisy_coords(
    coords: Float[torch.Tensor, "B N_atoms 3"],
) -> Float[torch.Tensor, "B N_atoms 3"]:
    """Provide coords with sigma=0.5 Gaussian noise added.

    Args:
        coords: Clean coordinate tensor of shape (B, N, 3).

    Returns:
        Perturbed coordinate tensor with the same shape.
    """
    return make_noisy(coords, sigma=0.5)


@pytest.fixture
def half_mask() -> Bool[torch.Tensor, "B N_atoms"]:
    """Bool mask where first N//2 atoms are valid and the rest are masked out.

    Returns:
        Bool tensor of shape (B, N); True for first half of atoms per sample.
    """
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[:, N // 2 :] = False
    return mask


@pytest.fixture
def rotation() -> Float[torch.Tensor, "3 3"]:
    """Random 3 by 3 orthogonal rotation matrix obtained via QR decomposition.

    Returns:
        Orthogonal float tensor of shape (3, 3).
    """
    R, _ = torch.qr(torch.randn(3, 3))
    return R


@pytest.fixture
def r_gt() -> Float[torch.Tensor, "B N_atoms 3"]:
    """Ground-truth atom coordinates [B, N, 3] for structure-loss tests.

    Returns:
        Random float tensor of shape (B, N, 3).
    """
    return torch.randn(B, N, 3)


@pytest.fixture
def aa_gt() -> Int[torch.Tensor, "B N_atoms"]:
    """Ground-truth amino-acid class indices drawn uniformly from vocabulary.

    Returns:
        Integer tensor of shape (B, N) with values in [0, VOCAB).
    """
    return torch.randint(0, VOCAB, (B, N))


@pytest.fixture
def r_blocks() -> Float[torch.Tensor, "K B N_atoms 3"]:
    """Stack of K random coordinate tensors for per-decoder-unit predictions.

    Returns:
        Float tensor of shape (K, B, N, 3).
    """
    return torch.stack([torch.randn(B, N, 3) for _ in range(K)])


@pytest.fixture
def aa_blocks() -> Float[torch.Tensor, "K B N_atoms VOCAB"]:
    """Stack of K random amino-acid logit tensors for med_loss tests.

    Returns:
        Float tensor of shape (K, B, N, VOCAB).
    """
    return torch.stack([torch.randn(B, N, VOCAB) for _ in range(K)])


@pytest.fixture
def batch(
    r_gt: Float[torch.Tensor, "B N_atoms 3"],
    aa_gt: Int[torch.Tensor, "B N_atoms"],
) -> FeaturizedBatch:
    """FeaturizedBatch with valid r_gt and aa_indices for med_loss tests.

    Non-essential fields are filled with zeros; r_gt and aa_indices carry the
    fixture values so structure/sequence losses are well-defined.

    Args:
        r_gt: Ground-truth coordinates of shape (B, N, 3).
        aa_gt: Ground-truth amino-acid indices of shape (B, N).

    Returns:
        FeaturizedBatch with all required fields populated.
    """
    return FeaturizedBatch(
        ref_pos=torch.randn(B, N, 3),
        ref_element=torch.zeros(B, N, 4),
        ref_space_uid=torch.zeros(B, N, dtype=torch.long),
        gt_res_distogram_indices=torch.zeros(B, N, N, dtype=torch.long),
        noised_res_distogram=torch.zeros(B, N, N, N_TEMPL_BINS),
        f_pseudo_beta_mask=torch.zeros(B, N, dtype=torch.long),
        f_residue_idx=torch.zeros(B, N, dtype=torch.long),
        r_gt=r_gt,
        r_gt_noised=r_gt.clone(),
        atom5_mask=torch.ones(B, N, dtype=torch.bool),
        aa_indices=aa_gt,
        t_hat=torch.zeros(B),
        t_normalized=torch.zeros(B, N, N),
        tok_idx=torch.zeros(B, N, dtype=torch.long),
        center_uid=torch.zeros(B, N, dtype=torch.long),
        gt_atom_distogram_sparse=torch.zeros(B, N, K_NEIGH, dtype=torch.long),
        gt_atom_distogram_mask_sparse=torch.zeros(
            B,
            N,
            K_NEIGH,
            dtype=torch.bool,
        ),
    )


@pytest.fixture
def res_logits() -> Float[torch.Tensor, "B L_res L_res B_res"]:
    """Random residue distogram logits [B, L_RES, L_RES, B_RES].

    Returns:
        Float tensor of shape (B, L_RES, L_RES, B_RES).
    """
    return torch.randn(B, L_RES, L_RES, B_RES)


@pytest.fixture
def res_bin_idx() -> Int[torch.Tensor, "B L_res L_res"]:
    """Random ground-truth residue distance bin indices [B, L_RES, L_RES].

    Returns:
        Integer tensor of shape (B, L_RES, L_RES) with values in [0, B_RES).
    """
    return torch.randint(0, B_RES, (B, L_RES, L_RES))


@pytest.fixture
def res_onehot(
    res_bin_idx: Int[torch.Tensor, "B L_res L_res"],
) -> Float[torch.Tensor, "B L_res L_res B_res"]:
    """One-hot encoded residue distance bins derived from res_bin_idx.

    Args:
        res_bin_idx: Integer bin indices of shape (B, L_RES, L_RES).

    Returns:
        Float one-hot tensor of shape (B, L_RES, L_RES, B_RES).
    """
    return to_onehot(res_bin_idx, B_RES)


@pytest.fixture
def res_mask() -> Bool[torch.Tensor, "B L_res"]:
    """Boolean residue mask where only first half of residues are unmasked.

    Returns:
        Boolean tensor of shape (B, L_RES); True for first L_RES//2 residues.
    """
    mask = torch.ones(B, L_RES, dtype=torch.float)
    mask[:, L_RES // 2 :] = False
    return mask


@pytest.fixture
def atom_logits() -> Float[torch.Tensor, "B N_atoms N_atoms B_atom"]:
    """Random atom distogram logits [B, N_ATOMS, N_ATOMS, B_ATOM].

    Returns:
        Float tensor of shape (B, N_ATOMS, N_ATOMS, B_ATOM).
    """
    return torch.randn(B, N_ATOMS, N_ATOMS, B_ATOM)


@pytest.fixture
def atom_bin_idx() -> Int[torch.Tensor, "B N_atoms N_atoms"]:
    """Random ground-truth atom distance bin indices [B, N_ATOMS, N_ATOMS].

    Returns:
        Int tensor of shape (B, N_ATOMS, N_ATOMS) with values in [0, B_ATOM).
    """
    return torch.randint(0, B_ATOM, (B, N_ATOMS, N_ATOMS))


@pytest.fixture
def atom_onehot(
    atom_bin_idx: Int[torch.Tensor, "B N_atoms N_atoms"],
) -> Float[torch.Tensor, "B N_atoms N_atoms B_atom"]:
    """One-hot encoded atom dist bins derived from atom_bin_idx.

    Args:
        atom_bin_idx: Integer bin indices of shape (B, N_ATOMS, N_ATOMS).

    Returns:
        Float one-hot tensor of shape (B, N_ATOMS, N_ATOMS, B_ATOM).
    """
    return to_onehot(atom_bin_idx, B_ATOM)


@pytest.fixture
def atom_local_mask() -> Bool[torch.Tensor, "B N_atoms N_atoms"]:
    """Bool local-window mask is True within ±2*ATOMS_PER_RES of diagonal.

    Returns:
        Bool tensor of shape (B, N_ATOMS, N_ATOMS) with local window applied.
    """
    lmask = local_window_mask(N_ATOMS, 2 * ATOMS_PER_RES)
    return repeat(lmask, "n m -> b n m", b=B)


# ---------------------------------------------------------------------------
# atom_loss
# ---------------------------------------------------------------------------


def test_atom_loss_perfect_near_zero(
    coords: Float[torch.Tensor, "B N_atoms 3"],
    uniform_aa_indices: Int[torch.Tensor, "B N_atoms"],
) -> None:
    """Same coords for both pred and GT yields near zero loss.

    Verifies atom_loss(x, x) returns a per-sample loss vector of shape (B,)
    whose maximum is below numerical tolerance when prediction equals ground
    truth.
    """
    loss = atom_loss(
        coords,
        coords,
        aa_indices=uniform_aa_indices,
        lambda_sigma_weight=torch.ones(B),
    )
    assert loss.shape == (B,)
    assert loss.max().item() < TOLERANCE


def test_atom_loss_known_translation_near_zero(
    coords: Float[torch.Tensor, "B N_atoms 3"],
    uniform_aa_indices: Int[torch.Tensor, "B N_atoms"],
) -> None:
    """Gobal translation removed by alignment, leaving near-zero loss.

    Verifies atom_loss is translation-invariant by confirming the loss remains
    below a tight tolerance after applying a constant displacement to the
    predictions.
    """
    translation = rearrange(torch.tensor([10.0, 5.0, -3.0]), "d -> 1 1 d")
    r_translated = coords + translation
    loss = atom_loss(
        r_translated,
        coords,
        aa_indices=uniform_aa_indices,
        lambda_sigma_weight=torch.ones(B),
    )
    assert (loss < TIGHT_TOLERANCE).all()


def test_atom_loss_full_mask_matches_no_mask(
    coords: Float[torch.Tensor, "B N_atoms 3"],
    noisy_coords: Float[torch.Tensor, "B N_atoms 3"],
    uniform_aa_indices: Int[torch.Tensor, "B N_atoms"],
) -> None:
    """All-True mask equal to calling atom_loss without a mask argument.

    Verifies mask=all-True code path produces numerically identical results
    to the unmasked default, confirming mask has no unintended side effects.
    """
    mask_all = torch.ones(B, N, dtype=torch.float)
    w = torch.ones(B)
    assert torch.allclose(
        atom_loss(
            coords,
            noisy_coords,
            mask=mask_all,
            aa_indices=uniform_aa_indices,
            lambda_sigma_weight=w,
        ),
        atom_loss(
            coords,
            noisy_coords,
            aa_indices=uniform_aa_indices,
            lambda_sigma_weight=w,
        ),
        atol=TOLERANCE,
    )


def test_atom_loss_increases_with_noise_level(
    coords: Float[torch.Tensor, "B N_atoms 3"],
    uniform_aa_indices: Int[torch.Tensor, "B N_atoms"],
) -> None:
    """Loss monotone in noise magnitude, larger perturbations -> higher loss.

    Verifies atom_loss correctly reflects the magnitude of displacement by
    checking that loss_low < loss_mid < loss_high for noise levels 0.1, 1.0,
    and 5.0.
    """
    _ = manual_seed(0)
    noise_dir = torch.randn(B, N, 3)
    w = torch.ones(B)
    loss_low = atom_loss(
        coords + 0.1 * noise_dir,
        coords,
        aa_indices=uniform_aa_indices,
        lambda_sigma_weight=w,
    )
    loss_mid = atom_loss(
        coords + 1.0 * noise_dir,
        coords,
        aa_indices=uniform_aa_indices,
        lambda_sigma_weight=w,
    )
    loss_high = atom_loss(
        coords + 5.0 * noise_dir,
        coords,
        aa_indices=uniform_aa_indices,
        lambda_sigma_weight=w,
    )
    assert (loss_low < loss_mid).all()
    assert (loss_mid < loss_high).all()


def test_atom_loss_gradient_flow() -> None:
    """Gradients flow to predicted coords but not to ground-truth coords.

    Verifies that torch.autograd.grad on r_pred produces finite gradients,
    and that backward leaves r_gt.grad as None, confirming ground-truth
    coordinates are treated as constants.
    """
    aa_indices = torch.zeros(1, N, dtype=torch.long)
    r_pred = torch.randn(1, N, 3, requires_grad=True)
    w1 = torch.ones(1)
    (grad,) = torch.autograd.grad(
        atom_loss(
            r_pred,
            torch.randn(1, N, 3),
            aa_indices=aa_indices,
            lambda_sigma_weight=w1,
        ),
        r_pred,
    )
    assert torch.isfinite(grad).all()
    r_pred2 = torch.randn(1, N, 3, requires_grad=True)
    r_gt2 = torch.randn(1, N, 3, requires_grad=True)
    torch.autograd.backward(
        [
            atom_loss(
                r_pred2,
                r_gt2,
                aa_indices=aa_indices,
                lambda_sigma_weight=w1,
            ),
        ],
    )
    assert r_gt2.grad is None


def test_atom_loss_rotation_invariant(
    coords: Float[torch.Tensor, "B N_atoms 3"],
    rotation: Float[torch.Tensor, "3 3"],
    r_gt: Float[torch.Tensor, "B N_atoms 3"],
    uniform_aa_indices: Int[torch.Tensor, "B N_atoms"],
) -> None:
    """Global rotation to GT structure does not change loss after alignment.

    Verifies rotation invariance by rotating r_gt by a random orthogonal
    matrix and confirming the loss is unchanged up to a generous numerical
    tolerance.
    """
    r_gt_rot = einsum(r_gt, rotation, "b n d, d e -> b n e")
    w = torch.ones(B)
    assert torch.allclose(
        atom_loss(
            coords,
            r_gt,
            aa_indices=uniform_aa_indices,
            lambda_sigma_weight=w,
        ),
        atom_loss(
            coords,
            r_gt_rot,
            aa_indices=uniform_aa_indices,
            lambda_sigma_weight=w,
        ),
        atol=1e-4,
    )


def test_pairwise_sq_dist_shape_symmetric() -> None:
    """The helper returns an [N, N] symmetric matrix with zero diagonal.

    Verifies pairwise_sq_dist produces correct shape, that d(i,i)=0 for all i,
    and that the matrix equals its transpose within numerical tolerance.
    """
    x = torch.randn(N, 3)
    D = pairwise_sq_dist(x)
    assert D.shape == (N, N)
    assert D.diagonal().abs().max().item() < TOLERANCE
    assert torch.allclose(D, D.T, atol=TOLERANCE)


# ---------------------------------------------------------------------------
# med_loss
# ---------------------------------------------------------------------------


def test_med_loss_scalar_finite_positive(
    batch: FeaturizedBatch,
    r_blocks: Float[torch.Tensor, "K B N_atoms 3"],
    aa_blocks: Float[torch.Tensor, "K B N_atoms VOCAB"],
) -> None:
    """med_loss returns a finite positive scalar for typical random inputs.

    Verifies that med_loss produces a 0-dim tensor that is finite and strictly
    positive when given K blocks of random coordinate and logit predictions.
    """
    loss = med_loss(
        list(r_blocks),
        list(aa_blocks),
        batch,
        LossParams(lam=LAM, alpha_0=ALPHA, gamma=GAMMA),
        lambda_sigma_weight=torch.ones(B),
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_med_loss_perfect_struct_near_zero(
    batch: FeaturizedBatch,
    aa_blocks: Float[torch.Tensor, "K B N_atoms VOCAB"],
) -> None:
    """If every block preds GT coords exactly and alpha_0=0, loss vanishes.

    Verifies setting alpha_0=0 and supplying perfect coordinate predictions
    drives med_loss below numerical tolerance, isolating the structural
    component to zero.
    """
    loss = med_loss(
        [batch.r_gt.clone() for _ in range(K)],
        list(aa_blocks),
        batch,
        LossParams(lam=LAM, alpha_0=0.0, gamma=GAMMA),
        lambda_sigma_weight=torch.ones(B),
    )
    assert loss.item() < TOLERANCE


def test_med_loss_block_weights_strictly_increasing() -> None:
    """Enfoces block weights are strictly increasing and final weight is 1.0.

    Verifies block_decay_weights helper generates a monotonically increasing
    schedule with exactly K entries and a terminal value of 1.0 (matching
    exponential schedule)
    """
    w = block_decay_weights(K, GAMMA)
    assert w.shape == (K,)
    assert (w[1:] > w[:-1]).all()
    assert abs(w[-1].item() - 1.0) < TOLERANCE


def test_med_loss_mismatched_blocks_raises(
    batch: FeaturizedBatch,
    r_blocks: Float[torch.Tensor, "K B N_atoms 3"],
    aa_blocks: Float[torch.Tensor, "K B N_atoms VOCAB"],
) -> None:
    """Different lengths for r_blocks, aa_blocks raise BlockCountMismatchError.

    Verifies input validation guard in med_loss by dropping one element from
    aa_blocks and confirming that a BlockCountMismatchError is raised.
    """
    with pytest.raises(
        BlockCountMismatchError,
        match="structure decoder blocks",
    ):
        _ = med_loss(
            list(r_blocks),
            list(aa_blocks)[:-1],
            batch,
            LossParams(lam=LAM, alpha_0=ALPHA),
            lambda_sigma_weight=torch.ones(B),
        )


def test_med_loss_gradient_flows_to_first_block() -> None:
    """Gradients from loss go back through earliest decoder block coordinates.

    Verifies end-to-end differentiability by requesting gradients w.r.t.
    first block's coordinates and confirming all entries of the resulting
    gradient are finite.
    """
    r0: Float[torch.Tensor, "1 N 3"] = torch.randn(1, N, 3, requires_grad=True)
    r_blocks_g: list[Float[torch.Tensor, "1 N 3"]] = [r0] + [
        torch.randn(1, N, 3) for _ in range(K - 1)
    ]
    aa_blocks_g: list[Float[torch.Tensor, "1 N VOCAB"]] = [
        torch.randn(1, N, VOCAB) for _ in range(K)
    ]
    batch_1 = FeaturizedBatch(
        ref_pos=torch.randn(1, N, 3),
        ref_element=torch.zeros(1, N, 4),
        ref_space_uid=torch.zeros(1, N, dtype=torch.long),
        gt_res_distogram_indices=torch.zeros(1, N, N, dtype=torch.long),
        noised_res_distogram=torch.zeros(1, N, N, N_TEMPL_BINS),
        f_pseudo_beta_mask=torch.zeros(1, N, dtype=torch.long),
        f_residue_idx=torch.zeros(1, N, dtype=torch.long),
        r_gt=torch.randn(1, N, 3),
        r_gt_noised=torch.randn(1, N, 3),
        atom5_mask=torch.ones(1, N, dtype=torch.bool),
        aa_indices=torch.randint(0, VOCAB, (1, N)),
        t_hat=torch.zeros(1),
        t_normalized=torch.zeros(1, N, N),
        tok_idx=torch.zeros(1, N, dtype=torch.long),
        center_uid=torch.zeros(1, N, dtype=torch.long),
        gt_atom_distogram_sparse=torch.zeros(1, N, K_NEIGH, dtype=torch.long),
        gt_atom_distogram_mask_sparse=torch.zeros(
            1,
            N,
            K_NEIGH,
            dtype=torch.bool,
        ),
    )
    (grad,) = torch.autograd.grad(
        med_loss(
            r_blocks_g,
            aa_blocks_g,
            batch_1,
            LossParams(lam=LAM, alpha_0=ALPHA, gamma=GAMMA),
            lambda_sigma_weight=torch.ones(1),
        ),
        r0,
    )
    assert torch.isfinite(grad).all()


def test_med_loss_hyperparams_scale_contributions(
    batch: FeaturizedBatch,
    r_blocks: Float[torch.Tensor, "K B N_atoms 3"],
    aa_blocks: Float[torch.Tensor, "K B N_atoms VOCAB"],
) -> None:
    """Lambda and alpha_0 independently gate structural and sequence losses.

    Verifies that near-zero lam removes the structural contribution and that
    alpha_0=0 removes the sequence CE contribution, each yielding a strictly
    smaller total loss than the nominal configuration.
    """
    r_list, aa_list = list(r_blocks), list(aa_blocks)
    w = torch.ones(B)
    loss_nominal = med_loss(
        r_list,
        aa_list,
        batch,
        LossParams(lam=LAM, alpha_0=ALPHA, gamma=GAMMA),
        lambda_sigma_weight=w,
    )
    loss_lam_small = med_loss(
        r_list,
        aa_list,
        batch,
        LossParams(lam=1e-6, alpha_0=ALPHA, gamma=GAMMA),
        lambda_sigma_weight=w,
    )
    assert loss_lam_small.item() < loss_nominal.item()
    loss_alpha_zero = med_loss(
        r_list,
        aa_list,
        batch,
        LossParams(lam=LAM, alpha_0=0.0, gamma=GAMMA),
        lambda_sigma_weight=w,
    )
    assert loss_alpha_zero.item() < loss_nominal.item()


# ---------------------------------------------------------------------------
# smooth_lddt_loss
# ---------------------------------------------------------------------------


def test_smooth_lddt_identical_coords_expected_value() -> None:
    """If Pred and true same, loss equals 1 - mean of four sigmoid scores at 0.

    Verifies closed-form expected value when all pairwise distance differences
    are zero, so sigmoid arguments collapse to the four fixed threshold values.
    """
    r = torch.randn(10, 3) * 0.1
    loss = smooth_lddt_loss(r, r)
    expected = 1.0 - 0.25 * sum(
        torch.sigmoid(torch.tensor(t)).item() for t in [0.5, 1.0, 2.0, 4.0]
    )
    assert abs(loss.item() - expected) < TIGHT_TOLERANCE


def test_smooth_lddt_noisy_exceeds_identical(
    coords: Float[torch.Tensor, "B N_atoms 3"],
) -> None:
    """Adding noise to predicted coords strictly increases smooth-lDDT loss.

    Verifies monotonicity property by confirming that a noisy prediction incurs
    a strictly higher loss than the perfect (identical) prediction.
    """
    loss_identical = smooth_lddt_loss(coords, coords)
    loss_noisy = smooth_lddt_loss(make_noisy(coords, sigma=2.0), coords)
    assert loss_noisy.item() > loss_identical.item()


def test_smooth_lddt_bounded_in_unit_interval(
    coords: Float[torch.Tensor, "B N_atoms 3"],
    half_mask: Bool[torch.Tensor, "B N_atoms"],
) -> None:
    """smooth_lddt_loss is bounded in [0, 1] with and without a mask.

    Verifies valid output range under large (sigma=5.0) perturbations without
    a mask and under moderate (sigma=1.0) perturbations with half the atoms
    masked, ensuring the sigmoid-based formulation keeps values in [0, 1] in
    both cases.
    """
    loss_full = smooth_lddt_loss(make_noisy(coords, sigma=5.0), coords)
    assert 0.0 <= loss_full.item() <= 1.0 + 1e-6
    loss_masked = smooth_lddt_loss(
        make_noisy(coords, sigma=1.0),
        coords,
        mask=half_mask,
    )
    assert torch.isfinite(loss_masked)
    assert 0.0 <= loss_masked.item() <= 1.0 + 1e-6


def test_smooth_lddt_gradient_flows() -> None:
    """The smooth-lDDT loss is differentiable w.r.t predicted coordinates.

    Verifies torch.autograd.grad succeeds and all gradient entries are finite,
    confirming no non-differentiable operations block gradient flow.
    """
    r_true = torch.randn(N, 3)
    r_pred = torch.randn(N, 3, requires_grad=True)
    (grad,) = torch.autograd.grad(smooth_lddt_loss(r_pred, r_true), r_pred)
    assert torch.isfinite(grad).all()


def test_smooth_lddt_pairwise_sq_dist_matches_einsum() -> None:
    """Pairwise_sq_dist produces same result as einsum over difference vectors.

    Verifies helper against a reference implementation that explicitly computes
    pairwise differences and contracts with einsum, up to 1e-6 absolute
    tolerance.
    """
    x = torch.randn(N, 3)
    diff = rearrange(x, "n d -> n 1 d") - rearrange(x, "n d -> 1 n d")
    sq_ref = einsum(diff, diff, "n m d, n m d -> n m")
    assert torch.allclose(pairwise_sq_dist(x), sq_ref, atol=1e-6)


# ---------------------------------------------------------------------------
# distogram_loss_residue
# ---------------------------------------------------------------------------


def test_distogram_residue_onehot_index_same_loss(
    res_logits: Float[torch.Tensor, "B L_res L_res B_res"],
    res_bin_idx: Int[torch.Tensor, "B L_res L_res"],
    res_onehot: Float[torch.Tensor, "B L_res L_res B_res"],
) -> None:
    """Passing one-hot targets vs. integer bin indices yield identical loss.

    Verifies distogram_loss_residue accepts both target formats and produces
    the same result, confirming the two code paths are equivalent.
    """
    assert torch.allclose(
        distogram_loss_residue(res_logits, res_onehot),
        distogram_loss_residue(res_logits, res_bin_idx),
        atol=1e-4,
    )


def test_distogram_residue_perfect_logits_near_zero(
    res_onehot: Float[torch.Tensor, "B L_res L_res B_res"],
) -> None:
    """Extremely high logits concentrated on correct bin drive CE to zero.

    Verifies that near-perfect logits (scaled by 1e6 from the one-hot targets)
    produce a maximum per-sample loss below TIGHT_TOLERANCE.
    """
    assert (
        distogram_loss_residue(res_onehot * 1e6, res_onehot).max().item()
        < TIGHT_TOLERANCE
    )


def test_distogram_residue_mask_changes_loss(
    res_logits: Float[torch.Tensor, "B L_res L_res B_res"],
    res_onehot: Float[torch.Tensor, "B L_res L_res B_res"],
    res_mask: Bool[torch.Tensor, "B L_res"],
) -> None:
    """Masking out half residues yields a different loss from unmasked case.

    Verifies that the residue mask is applied meaningfully by confirming masked
    loss is finite and numerically distinct from the unmasked loss.
    """
    loss_full = distogram_loss_residue(res_logits, res_onehot)
    loss_masked = distogram_loss_residue(res_logits, res_onehot, mask=res_mask)
    assert torch.isfinite(loss_masked).all()
    assert not torch.allclose(loss_full, loss_masked)


def test_distogram_residue_output_shape_and_gradient(
    res_logits: Float[torch.Tensor, "B L_res L_res B_res"],
    res_bin_idx: Int[torch.Tensor, "B L_res L_res"],
    res_onehot: Float[torch.Tensor, "B L_res L_res B_res"],
) -> None:
    """Batched input returns shape B and loss is differentiable w.r.t. logits.

    Verifies that distogram_loss_residue reduces over spatial dimensions to
    one scalar per batch element, and that torch.autograd.grad succeeds with
    all finite gradient entries.
    """
    assert distogram_loss_residue(res_logits, res_bin_idx).shape == (B,)
    logits_g = torch.randn(L_RES, L_RES, B_RES, requires_grad=True)
    (grad,) = torch.autograd.grad(
        distogram_loss_residue(logits_g, res_onehot[0]),
        logits_g,
    )
    assert torch.isfinite(grad).all()


def test_distogram_residue_ce_einsum_matches(
    res_logits: Float[torch.Tensor, "B L_res L_res B_res"],
    res_onehot: Float[torch.Tensor, "B L_res L_res B_res"],
) -> None:
    """Loss matches manual mean-CE over log_softmax probabilities.

    Verifies numerical correctness by comparing distogram_loss_residue against
    reference implementation that uses ce_via_einsum summed over all pairs and
    normalised.
    """
    logits, y = res_logits[0], res_onehot[0]
    loss_ref = distogram_loss_residue(logits, y).item()
    loss_manual = ce_via_einsum(logits, y).sum() / (L_RES * L_RES)
    assert abs(loss_ref - loss_manual.item()) < TOLERANCE


# ---------------------------------------------------------------------------
# distogram_loss_atom
# ---------------------------------------------------------------------------


def test_distogram_atom_uniform_worse_than_perfect(
    atom_onehot: Float[torch.Tensor, "B N_atoms N_atoms B_atom"],
    atom_local_mask: Bool[torch.Tensor, "B N_atoms N_atoms"],
) -> None:
    """Zero logits (uniform dist) produce higher loss for sample.

    Verifies that distogram_loss_atom correctly ranks predictions by comparing
    a uniform baseline (zero logits) against near-perfect logits
    (one-hot * 1e6).
    """
    uniform = torch.zeros_like(
        atom_onehot,
    )  # zero logits → uniform distribution after softmax
    loss_perfect = distogram_loss_atom(
        atom_onehot * 1e6,
        atom_onehot,
        atom_local_mask,
    )
    loss_uniform = distogram_loss_atom(uniform, atom_onehot, atom_local_mask)
    assert (loss_uniform > loss_perfect).all()


def test_distogram_atom_local_and_full_differ(
    atom_logits: Float[torch.Tensor, "B N_atoms N_atoms B_atom"],
    atom_onehot: Float[torch.Tensor, "B N_atoms N_atoms B_atom"],
    atom_local_mask: Bool[torch.Tensor, "B N_atoms N_atoms"],
) -> None:
    """Local-window mask changes pairs, masked and unmasked losses not equal.

    Verifies that atom_local_mask meaningfully restricts the set of atom pairs
    considered in loss, producing a numerically different result from no mask.
    """
    assert not torch.allclose(
        distogram_loss_atom(atom_logits, atom_onehot, atom_local_mask),
        distogram_loss_atom(atom_logits, atom_onehot),
    )


def test_distogram_atom_perfect_logits_near_zero(
    atom_onehot: Float[torch.Tensor, "B N_atoms N_atoms B_atom"],
    atom_local_mask: Bool[torch.Tensor, "B N_atoms N_atoms"],
) -> None:
    """Large logits on correct bin drive atom distogram CE to zero.

    Verifies that near-perfect logits (scaled by 1e6 from the one-hot targets)
    produce a maximum per-sample atom distogram loss below TIGHT_TOLERANCE.
    """
    assert (
        distogram_loss_atom(atom_onehot * 1e6, atom_onehot, atom_local_mask)
        .max()
        .item()
        < TIGHT_TOLERANCE
    )


def test_distogram_atom_gradient_flows(
    atom_onehot: Float[torch.Tensor, "B N_atoms N_atoms B_atom"],
    atom_local_mask: Bool[torch.Tensor, "B N_atoms N_atoms"],
) -> None:
    """The atom distogram loss is differentiable with respect to the logits.

    Verifies torch.autograd.grad succeeds and all gradient entries are finite,
    confirming end-to-end differentiability through the atom distogram
    cross-entropy.
    """
    logits_g = torch.randn(N_ATOMS, N_ATOMS, B_ATOM, requires_grad=True)
    (grad,) = torch.autograd.grad(
        distogram_loss_atom(logits_g, atom_onehot[0], atom_local_mask[0]),
        logits_g,
    )
    assert torch.isfinite(grad).all()


def test_distogram_atom_local_mask_shape_diagonal() -> None:
    """local_window_mask returns a square bool mask with diagonal always True.

    Verifies the shape, dtype, and diagonal properties of the local_window_mask
    helper, ensuring every atom is always within its own local window.
    """
    mask = local_window_mask(N_ATOMS, 2 * ATOMS_PER_RES)
    assert mask.shape == (N_ATOMS, N_ATOMS)
    assert mask.dtype == torch.bool
    assert mask.diagonal().all()


def test_distogram_atom_ce_einsum_matches(
    atom_logits: Float[torch.Tensor, "B N_atoms N_atoms B_atom"],
    atom_onehot: Float[torch.Tensor, "B N_atoms N_atoms B_atom"],
) -> None:
    """Atom distogram loss matches manual mean-CE over log-softmax values.

    Verifies numerical correctness by comparing distogram_loss_atom against a
    reference implementation that uses ce_via_einsum summed over all atom pairs
    and normalised.
    """
    logits, y = atom_logits[0], atom_onehot[0]
    loss_ref = distogram_loss_atom(logits, y).item()
    loss_manual = ce_via_einsum(logits, y).sum() / (N_ATOMS * N_ATOMS)
    assert abs(loss_ref - loss_manual.item()) < TOLERANCE


def test_distogram_atom_onehot_index_same_loss(
    atom_logits: Float[torch.Tensor, "B N_atoms N_atoms B_atom"],
    atom_onehot: Float[torch.Tensor, "B N_atoms N_atoms B_atom"],
    atom_bin_idx: Int[torch.Tensor, "B N_atoms N_atoms"],
    atom_local_mask: Bool[torch.Tensor, "B N_atoms N_atoms"],
) -> None:
    """One-hot and integer bin index targets yield same atom distogram losses.

    Verifies that distogram_loss_atom accepts both target formats and produces
    the same result, confirming the two code paths are numerically equivalent.
    """
    assert torch.allclose(
        distogram_loss_atom(atom_logits, atom_onehot, atom_local_mask),
        distogram_loss_atom(atom_logits, atom_bin_idx, atom_local_mask),
        atol=1e-4,
    )


# ---------------------------------------------------------------------------
# Shape-contract enforcement — negative tests
# ---------------------------------------------------------------------------


def test_atom_loss_wrong_shape() -> None:
    """Wrong last dim (4 instead of 3) on r_denoised triggers TypeCheckError.

    Verifies that the jaxtyping shape contract rejects coordinate tensors whose
    last dimension is 4 rather than the required 3.
    """
    r_bad = torch.zeros(B, N, 4)  # last dim must be 3
    r_target = torch.zeros(B, N, 3)
    aa_indices = torch.zeros(B, N, dtype=torch.long)
    with pytest.raises(TypeCheckError):
        _ = atom_loss(
            r_bad,
            r_target,
            aa_indices=aa_indices,
            lambda_sigma_weight=torch.ones(B),
        )


def test_atom_loss_residue_count_mismatch_raises() -> None:
    """N_res not a multiple of N_aa raises AtomResidueCountMismatchError.

    Verifies that supplying an aa_indices tensor whose residue count doesn't
    evenly divide the number of points in r_denoised/r_gt is rejected, since
    there would be no well-defined way to broadcast residue-type weights
    onto the points.
    """
    r = torch.zeros(B, N, 3)
    aa_indices = torch.zeros(B, N - 1, dtype=torch.long)
    with pytest.raises(AtomResidueCountMismatchError):
        _ = atom_loss(
            r,
            r,
            aa_indices=aa_indices,
            lambda_sigma_weight=torch.ones(B),
        )


@pytest.mark.parametrize(
    ("aa_letter", "expected_weight"),
    [
        pytest.param("S", POLAR_RESIDUE_WEIGHT, id="polar-serine"),
        pytest.param("D", POLAR_RESIDUE_WEIGHT, id="polar-aspartate"),
        pytest.param("K", POLAR_RESIDUE_WEIGHT, id="polar-lysine"),
        pytest.param("A", NONPOLAR_RESIDUE_WEIGHT, id="nonpolar-alanine"),
        pytest.param("V", NONPOLAR_RESIDUE_WEIGHT, id="nonpolar-valine"),
    ],
)
def test_atom_loss_weights_deviation_by_residue_polarity(
    aa_letter: str,
    expected_weight: float,
) -> None:
    """atom_loss weights one residue's contribution by its polarity.

    Verifies the residue-type weighting end to end by independently
    recomputing the Kabsch-aligned squared residuals with the
    ``kabsch_align`` primitive (the same primitive atom_loss uses
    internally) and combining them with a hand-built weight vector — 2.0
    for the residue under test if it is polar, 1.0 otherwise, 1.0 for
    every other (always-nonpolar 'A') residue — then checking that
    atom_loss's output matches this independently-derived reference.
    """
    _ = manual_seed(3)
    n_res = 6
    aa_indices = torch.full((1, n_res), restype_order["A"], dtype=torch.long)
    aa_indices[0, 0] = restype_order[aa_letter]

    gt_coords = torch.randn(1, n_res, 3)
    pred_coords = gt_coords + 0.3 * torch.randn(1, n_res, 3)

    with torch.no_grad():
        (r_aligned,) = kabsch_align(  # pylint: disable=unpacking-non-sequence
            gt_coords,
            pred_coords,
            weights=None,
            return_transform=False,
        )
    diff = pred_coords - r_aligned
    sq = einsum(diff, diff, "b n d, b n d -> b n")
    weights = torch.tensor(
        [[expected_weight, *([NONPOLAR_RESIDUE_WEIGHT] * (n_res - 1))]],
    )
    expected = einsum(sq, weights, "b n, b n -> b") / (
        3.0 * weights.sum(dim=-1)
    )

    loss = atom_loss(
        pred_coords,
        gt_coords,
        aa_indices=aa_indices,
        lambda_sigma_weight=torch.ones(1),
    )
    assert torch.allclose(loss, expected, atol=1e-4)


@pytest.mark.parametrize(
    ("aa_letters", "expected_weights"),
    [
        pytest.param(
            ["A", "S", "D", "V", "K", "G"],
            [
                NONPOLAR_RESIDUE_WEIGHT,
                POLAR_RESIDUE_WEIGHT,
                POLAR_RESIDUE_WEIGHT,
                NONPOLAR_RESIDUE_WEIGHT,
                POLAR_RESIDUE_WEIGHT,
                NONPOLAR_RESIDUE_WEIGHT,
            ],
            id="mixed-polarity",
        ),
        pytest.param(
            ["X", "X"],
            [NONPOLAR_RESIDUE_WEIGHT, NONPOLAR_RESIDUE_WEIGHT],
            id="unknown-clamped-in-range",
        ),
    ],
)
def test_residue_type_weight_matches_polarity_table(
    aa_letters: list[str],
    expected_weights: list[float],
) -> None:
    """residue_type_weight looks up the correct weight for each residue.

    Verifies the helper directly: mixed polar/nonpolar residues map to
    their table weights, and the out-of-vocabulary 'X' token (index 20,
    used for unknown/conditioning-dropped residues) is clamped into range
    rather than raising an index error.
    """
    aa_indices = torch.tensor(
        [[restype_order[letter] for letter in aa_letters]],
    )
    weight = residue_type_weight(aa_indices)
    assert torch.allclose(weight, torch.tensor([expected_weights]))


def test_residue_type_weights_cover_all_canonical_residues() -> None:
    """Every canonical residue is classified as exactly polar or nonpolar.

    Verifies RESIDUE_TYPE_WEIGHTS assigns POLAR_RESIDUE_WEIGHT to exactly
    the residues listed in POLAR_RESIDUES and NONPOLAR_RESIDUE_WEIGHT to
    every other residue in RESTYPES_NO_X, with no residue left
    unclassified and no unexpected weight values.
    """
    assert len(RESIDUE_TYPE_WEIGHTS) == RESTYPE_NUM_NO_X
    for restype, weight in zip(
        RESTYPES_NO_X,
        RESIDUE_TYPE_WEIGHTS,
        strict=True,
    ):
        expected = (
            POLAR_RESIDUE_WEIGHT
            if restype in POLAR_RESIDUES
            else NONPOLAR_RESIDUE_WEIGHT
        )
        assert weight == expected


def test_atom_loss_padded_residue_ignored_regardless_of_polarity() -> None:
    """A masked-out residue contributes zero weight even if it is polar.

    Verifies that padding (mask=0) always overrides residue-type weighting:
    a large deviation placed on a polar residue that is masked out must not
    affect the loss at all.
    """
    n_res = 4
    aa_indices = torch.full(
        (1, n_res),
        restype_order["A"],
        dtype=torch.long,
    )
    aa_indices[0, 0] = restype_order["D"]  # polar, but will be masked out
    mask = torch.ones(1, n_res)
    mask[0, 0] = 0.0

    gt_coords = torch.zeros(1, n_res, 3)
    pred_coords = gt_coords.clone()
    pred_coords[0, 0, 0] = 100.0  # large deviation on the masked-out residue

    loss = atom_loss(
        pred_coords,
        gt_coords,
        mask=mask,
        aa_indices=aa_indices,
        lambda_sigma_weight=torch.ones(1),
    )
    assert loss.item() < TOLERANCE


def testpairwise_dist_wrong_shape() -> None:
    """Wrong last dim (4 instead of 3) triggers TypeCheckError.

    Verifies that the jaxtyping shape contract on pairwise_dist rejects input
    tensors whose last dimension is 4 rather than the required 3.
    """
    x_bad = torch.zeros(10, 4)  # last dim must be 3
    with pytest.raises(TypeCheckError):
        _ = pairwise_dist(x_bad)


def test_smooth_lddt_loss_wrong_shape() -> None:
    """Wrong last dim (4 instead of 3) on r_pred triggers TypeCheckError.

    Verifies that the jaxtyping shape contract on smooth_lddt_loss rejects
    predicted coordinate tensors whose last dimension is 4 rather than 3.
    """
    r_pred_bad = torch.zeros(N, 4)  # last dim must be 3
    r_true = torch.zeros(N, 3)
    with pytest.raises(TypeCheckError):
        _ = smooth_lddt_loss(r_pred_bad, r_true)


def test_distogram_loss_residue_wrong_shape() -> None:
    """2-D p (below min 3-D for '... N N n_bins') triggers TypeCheckError.

    Verifies that distogram_loss_residue rejects a 2-D logit tensor that lacks
    required minimum of 3 dimensions for '... N_res N_res n_bins' annotation.
    """
    p_bad = torch.zeros(
        N,
        N,
    )  # needs at least 3 dims for "... N_res N_res n_bins"
    y = torch.zeros(N, N, dtype=torch.long)
    with pytest.raises(TypeCheckError):
        _ = distogram_loss_residue(p_bad, y)


def test_distogram_loss_atom_wrong_shape() -> None:
    """2-D q (below min 3-D for '... N K n_bins') triggers TypeCheckError.

    Verifies that distogram_loss_atom rejects a 2-D logit tensor that lacks
    the required minimum of 3 dimensions for '... N_atom K n_bins' annotation.
    """
    q_bad = torch.zeros(
        N_ATOMS,
        K,
    )  # needs at least 3 dims for "... N_atom K n_bins"
    y = torch.zeros(N_ATOMS, K, dtype=torch.long)
    with pytest.raises(TypeCheckError):
        _ = distogram_loss_atom(q_bad, y)


# ---------------------------------------------------------------------------
# seq_ce_loss
# ---------------------------------------------------------------------------


def test_seq_ce_loss_scalar_output() -> None:
    """seq_ce_loss returns a 0-dim scalar for batched logits and indices.

    Verifies that the output tensor has ndim == 0 and is finite, confirming
    that seq_ce_loss correctly reduces over the batch and sequence dimensions.
    """
    logits: Float[torch.Tensor, "B N VOCAB"] = torch.randn(B, N, VOCAB)
    indices: Int[torch.Tensor, "B N"] = torch.randint(0, VOCAB, (B, N))
    loss: Float[torch.Tensor, ""] = seq_ce_loss(logits, indices)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_seq_ce_loss_wrong_rank() -> None:
    """3-D logits with a 1-D index tensor triggers TypeCheckError.

    Verifies that the jaxtyping shape contract on seq_ce_loss rejects a rank
    mismatch where logits are (B, N, VOCAB) but indices are only (N,).
    """
    logits_bad: Float[torch.Tensor, "B N VOCAB"] = torch.randn(B, N, VOCAB)
    indices_bad: Int[torch.Tensor, "N"] = torch.randint(0, VOCAB, (N,))
    with pytest.raises(TypeCheckError):
        _ = seq_ce_loss(logits_bad, indices_bad)


# ---------------------------------------------------------------------------
# seq_ce_loss — pipeline invariants and conditioning-dropout conventions
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_batch() -> FeaturizedBatch:
    """FeaturizedBatch for conditioning-dropout tests (B=1, N_res=5).

    All residues are marked valid via f_pseudo_beta_mask so dropout logic
    operates on every position without any padding interference.
    """
    tok: Int[torch.Tensor, "1 25"] = repeat(
        torch.repeat_interleave(torch.arange(_N_RES), _ATOMS_PER_RES),
        "n -> b n",
        b=_B,
    ).contiguous()
    cuid: Int[torch.Tensor, "1 25"] = repeat(
        torch.arange(0, _N_ATOM, _ATOMS_PER_RES),
        "n -> b (n a)",
        b=_B,
        a=_ATOMS_PER_RES,
    ).contiguous()
    return FeaturizedBatch(
        ref_pos=torch.randn(_B, _N_ATOM, 3),
        ref_element=torch.zeros(_B, _N_ATOM, 4),
        ref_space_uid=torch.zeros(_B, _N_ATOM, dtype=torch.long),
        gt_res_distogram_indices=torch.zeros(
            _B,
            _N_RES,
            _N_RES,
            dtype=torch.long,
        ),
        noised_res_distogram=torch.zeros(
            _B,
            _N_RES,
            _N_RES,
            _N_BINS,
        ),
        f_pseudo_beta_mask=torch.ones(_B, _N_RES, dtype=torch.long),
        f_residue_idx=torch.zeros(_B, _N_RES, dtype=torch.long),
        r_gt=torch.randn(_B, _N_ATOM, 3),
        r_gt_noised=torch.randn(_B, _N_ATOM, 3),
        atom5_mask=torch.ones(_B, _N_ATOM, dtype=torch.bool),
        aa_indices=torch.randint(0, _N_AMINO, (_B, _N_RES), dtype=torch.long),
        t_hat=torch.randn(_B),
        t_normalized=torch.randn(_B, _N_RES, _N_RES),
        tok_idx=tok,
        center_uid=cuid,
        gt_atom_distogram_sparse=torch.randn(
            _B,
            _N_ATOM,
            _N_ATOM,
            _N_AMINO_BINS,
        ),
        gt_atom_distogram_mask_sparse=torch.ones(
            _B,
            _N_ATOM,
            _N_ATOM,
            dtype=torch.bool,
        ),
    )


@pytest.fixture
def template_distogram_fn() -> Distogram:
    """Minimal self-conditioning template distogram function for tests.

    Built from TemplateDistogramParams (overflow_bin=True by default) so
    distances beyond max_dist are clipped into the last bin, matching the
    template Cβ distance convention used in the full pipeline.
    """
    return build_distogram_module(
        TemplateDistogramParams(n_bins=_N_BINS, min_dist=2.0, max_dist=22.0),
    ).eval()


@pytest.fixture
def residue_distogram_fn() -> Distogram:
    """Minimal residue-level Cβ distogram function for tests.

    Built from ResidueDistogramParams (overflow_bin=True by default) so
    distances beyond max_dist are clipped into the last bin, matching the
    residue_distogram_loss ground-truth convention used in the full
    pipeline.
    """
    return build_distogram_module(
        ResidueDistogramParams(n_bins=_N_BINS, min_dist=2.0, max_dist=22.0),
    ).eval()


@pytest.fixture
def atom_distogram_fn() -> Distogram:
    """Minimal atom distogram function for featurize_single_item tests.

    Built from AtomDistogramParams (overflow_bin=True by default) so
    distances beyond max_dist are clipped into the last bin, matching the
    atom-level distance convention used in the full pipeline.
    """
    return build_distogram_module(
        AtomDistogramParams(n_bins=_N_BINS, min_dist=2.0, max_dist=22.0),
    ).eval()


def test_intermediate_and_final_ce_are_identical() -> None:
    """seq_ce_loss returns the same scalar for identical logits/targets.

    Verifies that calling seq_ce_loss twice with the same inputs produces
    bit-identical results, confirming the function is deterministic and has no
    hidden mutable state.
    """
    logits: Float[torch.Tensor, "1 10 20"] = torch.randn(1, 10, _N_AMINO)
    aa_indices: Int[torch.Tensor, "1 10"] = torch.randint(0, _N_AMINO, (1, 10))

    loss_final: Float[torch.Tensor, ""] = seq_ce_loss(logits, aa_indices)
    loss_inter: Float[torch.Tensor, ""] = seq_ce_loss(logits, aa_indices)

    assert torch.allclose(loss_final, loss_inter)


def test_seq_ce_loss_ignores_minus_100() -> None:
    """seq_ce_loss on a tensor with padding is same as on the non-padded subset.

    Verifies that positions with aa_indices == -100 are silently excluded from
    the loss computation, so that padding tokens do not bias the gradient
    signal.
    """
    logits: Float[torch.Tensor, "1 6 20"] = torch.randn(1, 6, _N_AMINO)
    aa_full: Int[torch.Tensor, "1 6"] = torch.full(
        (1, 6),
        PADDING_DROPOUT_TOKEN,
        dtype=torch.long,
    )
    aa_full[0, :3] = torch.randint(0, _N_AMINO, (3,))

    loss_full: Float[torch.Tensor, ""] = seq_ce_loss(logits, aa_full)
    loss_half: Float[torch.Tensor, ""] = seq_ce_loss(
        logits[:, :3],
        aa_full[:, :3],
    )

    assert torch.allclose(loss_full, loss_half)


def test_pipeline_preserves_pdb_x_as_index_20(
    template_distogram_fn: Distogram,
    residue_distogram_fn: Distogram,
    atom_distogram_fn: Distogram,
) -> None:
    """featurize_single_item maps aatype==20 (unknown 'X') to aa_indices==20.

    Verifies that the featurize pipeline preserves the unknown-residue sentinel
    (index 20) for PDB 'X' residues, ensuring downstream seq_ce_loss can ignore
    them via the same masking path as conditioned unknowns.
    """
    n_res = 5
    x_pos = 2
    aa_seq = "ACXDE"

    aatype = np.array(
        [restype_order.get(c, 20) for c in aa_seq],
        dtype=np.intp,
    )
    prot = Protein(
        atom_positions=np.zeros((n_res, 37, 3), dtype=np.float64),
        aatype=aatype,
        atom_mask=np.ones((n_res, 37), dtype=np.float64),
        residue_index=np.arange(n_res, dtype=np.intp),
        chain_index=np.zeros(n_res, dtype=np.intp),
        b_factors=np.zeros((n_res, 37), dtype=np.float64),
    )

    item: FeaturizedItem = featurize_single_item(
        prot,
        template_distogram_fn,
        residue_distogram_fn,
        atom_distogram_fn,
        TrainConfig(),
        max_seq_len_in_batch=n_res,
    )

    assert item.aa_indices[x_pos].item() == _N_AMINO


def test_seq_ce_loss_all_ignored_returns_zero() -> None:
    """seq_ce_loss returns 0.0 when every position is masked (no valid targets).

    Verifies the edge case where all aa_indices equal the unknown sentinel (20),
    meaning there are no valid cross-entropy targets; the loss must be exactly
    0.0 rather than NaN or an error.
    """
    logits: Float[torch.Tensor, "1 5 20"] = torch.randn(1, 5, _N_AMINO)
    aa_all_x: Int[torch.Tensor, "1 5"] = torch.full(
        (1, 5),
        _N_AMINO,
        dtype=torch.long,
    )

    loss: Float[torch.Tensor, ""] = seq_ce_loss(logits, aa_all_x)

    assert loss.item() == 0.0
