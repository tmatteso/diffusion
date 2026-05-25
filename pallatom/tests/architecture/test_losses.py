"""Tests for diffusion model loss functions."""

import pytest
import torch
import torch.nn.functional as F
from architecture.losses import (
    atom_loss,
    distogram_loss_atom,
    distogram_loss_residue,
    med_loss,
    med_loss_per_block,
    pairwise_dist,
    smooth_lddt_loss,
)
from beartype import beartype
from einops import einsum, rearrange, repeat
from helpers.useful_objects import manual_seed
from jaxtyping import Bool, Float, Int, TypeCheckError, jaxtyped

manual_seed(42)

B, N = 4, 50
K = 6
VOCAB = 20
LAM, ALPHA, GAMMA = 1.0, 0.1, 0.99
L_RES, B_RES = 32, 64
ATOMS_PER_RES, B_ATOM = 14, 22
N_ATOMS = 8 * ATOMS_PER_RES  # 112


# ---------------------------------------------------------------------------
# Typed helpers
# ---------------------------------------------------------------------------


@jaxtyped(typechecker=beartype)
def make_noisy(
    base: Float[torch.Tensor, "... N 3"],
    sigma: float,
) -> Float[torch.Tensor, "... N 3"]:
    """Add zero-mean Gaussian noise with standard deviation sigma to base coordinates."""
    return base + sigma * torch.randn_like(base)


@jaxtyped(typechecker=beartype)
def pairwise_sq_dist(
    x: Float[torch.Tensor, "N 3"],
) -> Float[torch.Tensor, "N N"]:
    """Compute the symmetric N by N pairwise squared Euclidean distance matrix."""
    diff = rearrange(x, "n d -> n 1 d") - rearrange(x, "n d -> 1 n d")
    return einsum(diff, diff, "n m d, n m d -> n m")


@jaxtyped(typechecker=beartype)
def local_window_mask(
    N_atoms: int,
    half_window: int,
) -> Bool[torch.Tensor, "N_atoms N_atoms"]:
    """Return a boolean mask where entry (i,j) is True iff |i-j| <= half_window."""
    idx = torch.arange(N_atoms)
    dist = (rearrange(idx, "n -> n 1") - rearrange(idx, "m -> 1 m")).abs()
    return dist <= half_window


@jaxtyped(typechecker=beartype)
def to_onehot(
    indices: Int[torch.Tensor, "... L"],
    n_classes: int,
) -> Float[torch.Tensor, "... L n_classes"]:
    """Convert integer indices to float one-hot vectors of length n_classes."""
    return F.one_hot(indices, num_classes=n_classes).float()


@jaxtyped(typechecker=beartype)
def block_decay_weights(
    K: int,
    gamma: float,
) -> Float[torch.Tensor, "K"]:
    """Return exponentially decaying weights gamma^(K-k) for k=1..K, with the last weight = 1."""
    ks = torch.arange(1, K + 1, dtype=torch.float32)
    return gamma ** (K - ks)


@jaxtyped(typechecker=beartype)
def ce_via_einsum(
    logits: Float[torch.Tensor, "... B"],
    targets: Float[torch.Tensor, "... B"],
) -> Float[torch.Tensor, "..."]:
    """Compute cross-entropy -sum(targets * log_softmax(logits)) via einsum."""
    log_p = F.log_softmax(logits, dim=-1)
    return -einsum(targets, log_p, "... b, ... b -> ...")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def coords() -> Float[torch.Tensor, "B N_atoms 3"]:
    """Provide a random (B, N, 3) coordinate tensor."""
    return torch.randn(B, N, 3)


@pytest.fixture
def noisy_coords(coords: Float[torch.Tensor, "B N_atoms 3"]) -> Float[torch.Tensor, "B N_atoms 3"]:
    """Provide coords with sigma=0.5 Gaussian noise added."""
    return make_noisy(coords, sigma=0.5)


@pytest.fixture
def half_mask() -> Bool[torch.Tensor, "B N_atoms"]:
    """Boolean mask [B, N] where the first N//2 atoms are valid and the rest are masked out."""
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[:, N // 2 :] = False
    return mask


@pytest.fixture
def rotation() -> Float[torch.Tensor, "3 3"]:
    """A random 3 by 3 orthogonal rotation matrix obtained via QR decomposition."""
    R, _ = torch.qr(torch.randn(3, 3))
    return R


@pytest.fixture
def r_gt() -> Float[torch.Tensor, "B N_atoms 3"]:
    """Ground-truth atom coordinates [B, N, 3] for structure-loss tests."""
    return torch.randn(B, N, 3)


@pytest.fixture
def aa_gt() -> Int[torch.Tensor, "B N_atoms"]:
    """Ground-truth amino-acid class indices [B, N] drawn uniformly from the vocabulary."""
    return torch.randint(0, VOCAB, (B, N))


@pytest.fixture
def r_blocks() -> Float[torch.Tensor, "K B N_atoms 3"]:
    """Stack of K random coordinate tensors [K, B, N, 3] simulating per-decoder-unit predictions."""
    return torch.stack([torch.randn(B, N, 3) for _ in range(K)])


@pytest.fixture
def aa_blocks() -> Float[torch.Tensor, "K B N_atoms VOCAB"]:
    """Stack of K random amino-acid logit tensors [K, B, N, VOCAB] for med_loss tests."""
    return torch.stack([torch.randn(B, N, VOCAB) for _ in range(K)])


@pytest.fixture
def res_logits() -> Float[torch.Tensor, "B L_res L_res B_res"]:
    """Random residue distogram logits [B, L_RES, L_RES, B_RES]."""
    return torch.randn(B, L_RES, L_RES, B_RES)


@pytest.fixture
def res_bin_idx() -> Int[torch.Tensor, "B L_res L_res"]:
    """Random ground-truth residue distance bin indices [B, L_RES, L_RES]."""
    return torch.randint(0, B_RES, (B, L_RES, L_RES))


@pytest.fixture
def res_onehot(
    res_bin_idx: Int[torch.Tensor, "B L_res L_res"],
) -> Float[torch.Tensor, "B L_res L_res B_res"]:
    """One-hot encoded residue distance bins [B, L_RES, L_RES, B_RES] derived from res_bin_idx."""
    return to_onehot(res_bin_idx, B_RES)


@pytest.fixture
def res_mask() -> Bool[torch.Tensor, "B L_res"]:
    """Boolean residue mask [B, L_RES] where only the first half of residues are unmasked."""
    mask = torch.ones(B, L_RES, dtype=torch.bool)
    mask[:, L_RES // 2 :] = False
    return mask


@pytest.fixture
def atom_logits() -> Float[torch.Tensor, "B N_atoms N_atoms B_atom"]:
    """Random atom distogram logits [B, N_ATOMS, N_ATOMS, B_ATOM]."""
    return torch.randn(B, N_ATOMS, N_ATOMS, B_ATOM)


@pytest.fixture
def atom_bin_idx() -> Int[torch.Tensor, "B N_atoms N_atoms"]:
    """Random ground-truth atom distance bin indices [B, N_ATOMS, N_ATOMS]."""
    return torch.randint(0, B_ATOM, (B, N_ATOMS, N_ATOMS))


@pytest.fixture
def atom_onehot(
    atom_bin_idx: Int[torch.Tensor, "B N_atoms N_atoms"],
) -> Float[torch.Tensor, "B N_atoms N_atoms B_atom"]:
    """One-hot encoded atom dist bins [B, N_ATOMS, N_ATOMS, B_ATOM] derived from atom_bin_idx."""
    return to_onehot(atom_bin_idx, B_ATOM)


@pytest.fixture
def atom_local_mask() -> Bool[torch.Tensor, "B N_atoms N_atoms"]:
    """Bool local-window mask [B, N_ATOMS, N_ATOMS] — True within ±2*ATOMS_PER_RES of diagonal."""
    lmask = local_window_mask(N_ATOMS, 2 * ATOMS_PER_RES)
    return repeat(lmask, "n m -> b n m", b=B)


# ---------------------------------------------------------------------------
# atom_loss
# ---------------------------------------------------------------------------


def test_atom_loss_perfect_near_zero(coords: Float[torch.Tensor, "B N_atoms 3"]):
    """Passing identical coords as both pred and GT yields loss indistinguishable from zero."""
    loss = atom_loss(coords, coords)
    assert loss.shape == (B,)
    assert loss.max().item() < 1e-5


def test_atom_loss_known_translation_near_zero(coords: Float[torch.Tensor, "B N_atoms 3"]):
    """A pure global translation should be removed by alignment, leaving near-zero loss."""
    translation = rearrange(torch.tensor([10.0, 5.0, -3.0]), "d -> 1 1 d")
    r_translated = coords + translation
    loss = atom_loss(r_translated, coords)
    assert (loss < 1e-4).all()


def test_atom_loss_full_mask_matches_no_mask(
    coords: Float[torch.Tensor, "B N_atoms 3"], noisy_coords: Float[torch.Tensor, "B N_atoms 3"]
):
    """Passing an all-True mask must be equivalent to calling atom_loss without a mask argument."""
    mask_all = torch.ones(B, N, dtype=torch.bool)
    assert torch.allclose(
        atom_loss(coords, noisy_coords, mask=mask_all),
        atom_loss(coords, noisy_coords),
        atol=1e-5,
    )


def test_atom_loss_increases_with_noise_level(coords: Float[torch.Tensor, "B N_atoms 3"]):
    """Loss is strictly monotone in the noise magnitude — larger perturbations yield higher loss."""
    g = torch.Generator()
    g.manual_seed(0)
    noise_dir = torch.randn(B, N, 3, generator=g)
    loss_low = atom_loss(coords + 0.1 * noise_dir, coords)
    loss_mid = atom_loss(coords + 1.0 * noise_dir, coords)
    loss_high = atom_loss(coords + 5.0 * noise_dir, coords)
    assert (loss_low < loss_mid).all()
    assert (loss_mid < loss_high).all()


def test_atom_loss_gradient_flows_through_pred():
    """The loss is differentiable with respect to the predicted coordinates."""
    r_pred = torch.randn(N, 3, requires_grad=True)
    (grad,) = torch.autograd.grad(atom_loss(r_pred, torch.randn(N, 3)), r_pred)
    assert torch.isfinite(grad).all()


def test_atom_loss_gt_receives_no_gradient():
    """Ground-truth coordinates are treated as constants — no gradient should flow through them."""
    r_pred = torch.randn(N, 3, requires_grad=True)
    r_gt = torch.randn(N, 3, requires_grad=True)
    torch.autograd.backward([atom_loss(r_pred, r_gt)])
    assert r_gt.grad is None


def test_atom_loss_rotation_invariant(
    coords: Float[torch.Tensor, "B N_atoms 3"], rotation: Float[torch.Tensor, "3 3"]
):
    """Applying a global rotation to the GT structure does not change the loss after alignment."""
    r_gt = torch.randn(B, N, 3)
    r_gt_rot = einsum(r_gt, rotation, "b n d, d e -> b n e")
    assert torch.allclose(atom_loss(coords, r_gt), atom_loss(coords, r_gt_rot), atol=1e-4)


def test_pairwise_sq_dist_shape_symmetric():
    """The helper returns an [N, N] symmetric matrix with zero diagonal."""
    x = torch.randn(N, 3)
    D = pairwise_sq_dist(x)
    assert D.shape == (N, N)
    assert D.diagonal().abs().max().item() < 1e-5
    assert torch.allclose(D, D.T, atol=1e-5)


# ---------------------------------------------------------------------------
# med_loss
# ---------------------------------------------------------------------------


def test_med_loss_scalar_finite_positive(
    r_gt: Float[torch.Tensor, "B N_atoms 3"],
    aa_gt: Int[torch.Tensor, "B N_atoms"],
    r_blocks: Float[torch.Tensor, "K B N_atoms 3"],
    aa_blocks: Float[torch.Tensor, "K B N_atoms VOCAB"],
):
    """med_loss returns a finite positive scalar for typical random inputs."""
    loss = med_loss(
        list(r_blocks),
        r_gt,
        list(aa_blocks),
        aa_gt,
        lam=LAM,
        alpha_0=ALPHA,
        gamma=GAMMA,
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_med_loss_perfect_struct_near_zero(
    r_gt: Float[torch.Tensor, "B N_atoms 3"],
    aa_gt: Int[torch.Tensor, "B N_atoms"],
    aa_blocks: Float[torch.Tensor, "K B N_atoms VOCAB"],
):
    """When every block predicts GT coordinates exactly and alpha_0=0, structural loss vanishes."""
    loss = med_loss(
        [r_gt.clone() for _ in range(K)],
        r_gt,
        list(aa_blocks),
        aa_gt,
        lam=LAM,
        alpha_0=0.0,
        gamma=GAMMA,
    )
    assert loss.item() < 1e-5


def test_med_loss_block_weights_strictly_increasing():
    """Block weights are strictly increasing and final weight is 1.0, matching exponential sched."""
    w = block_decay_weights(K, GAMMA)
    assert w.shape == (K,)
    assert (w[1:] > w[:-1]).all()
    assert abs(w[-1].item() - 1.0) < 1e-6


def test_med_loss_mismatched_blocks_raises(
    r_gt: Float[torch.Tensor, "B N_atoms 3"],
    aa_gt: Int[torch.Tensor, "B N_atoms"],
    r_blocks: Float[torch.Tensor, "K B N_atoms 3"],
    aa_blocks: Float[torch.Tensor, "K B N_atoms VOCAB"],
):
    """Passing lists of different lengths for r_blocks and aa_blocks must raise ValueError."""
    with pytest.raises(ValueError, match="r_denoised_blocks has"):
        med_loss(
            list(r_blocks),
            r_gt,
            list(aa_blocks)[:-1],
            aa_gt,
            lam=LAM,
            alpha_0=ALPHA,
        )


def test_med_loss_gradient_flows_to_first_block(
    r_gt: Float[torch.Tensor, "B N_atoms 3"], aa_gt: Int[torch.Tensor, "B N_atoms"]
):
    """Gradients propagate from aggregated loss back through earliest decoder block coordinates."""
    r0 = torch.randn(N, 3, requires_grad=True)
    r_blocks_g = [r0] + [torch.randn(N, 3) for _ in range(K - 1)]
    aa_blocks_g = [torch.randn(N, VOCAB) for _ in range(K)]
    (grad,) = torch.autograd.grad(
        med_loss(
            r_blocks_g,
            r_gt[0],
            aa_blocks_g,
            aa_gt[0],
            lam=LAM,
            alpha_0=ALPHA,
            gamma=GAMMA,
        ),
        r0,
    )
    assert torch.isfinite(grad).all()


def test_med_loss_lam_zero_less_than_lam_positive(
    r_gt: Float[torch.Tensor, "B N_atoms 3"],
    aa_gt: Int[torch.Tensor, "B N_atoms"],
    r_blocks: Float[torch.Tensor, "K B N_atoms 3"],
    aa_blocks: Float[torch.Tensor, "K B N_atoms VOCAB"],
):
    """Setting lam=0 removes sequence cross-entropy term, giving a strictly smaller total loss."""
    r_list, aa_list = list(r_blocks), list(aa_blocks)
    loss_lam0 = med_loss(r_list, r_gt, aa_list, aa_gt, lam=0.0, alpha_0=ALPHA, gamma=GAMMA)
    loss_lam1 = med_loss(r_list, r_gt, aa_list, aa_gt, lam=LAM, alpha_0=ALPHA, gamma=GAMMA)
    assert loss_lam0.item() < loss_lam1.item()


def test_med_loss_alpha_zero_less_than_alpha_positive(
    r_gt: Float[torch.Tensor, "B N_atoms 3"],
    aa_gt: Int[torch.Tensor, "B N_atoms"],
    r_blocks: Float[torch.Tensor, "K B N_atoms 3"],
    aa_blocks: Float[torch.Tensor, "K B N_atoms VOCAB"],
):
    """Setting alpha_0=0 removes smooth-lDDT penalty term, giving a strictly smaller total loss."""
    r_list, aa_list = list(r_blocks), list(aa_blocks)
    loss_a0 = med_loss(r_list, r_gt, aa_list, aa_gt, lam=LAM, alpha_0=0.0, gamma=GAMMA)
    loss_a1 = med_loss(r_list, r_gt, aa_list, aa_gt, lam=LAM, alpha_0=ALPHA, gamma=GAMMA)
    assert loss_a0.item() < loss_a1.item()


# ---------------------------------------------------------------------------
# smooth_lddt_loss
# ---------------------------------------------------------------------------


def test_smooth_lddt_identical_coords_expected_value():
    """Pred and true are same, smooth_lddt_loss equals 1 minus mean of four sigmoid scores at 0."""
    r = torch.randn(10, 3) * 0.1
    loss = smooth_lddt_loss(r, r)
    expected = 1.0 - 0.25 * sum(torch.sigmoid(torch.tensor(t)).item() for t in [0.5, 1.0, 2.0, 4.0])
    assert abs(loss.item() - expected) < 1e-4


def test_smooth_lddt_noisy_exceeds_identical(coords: Float[torch.Tensor, "B N_atoms 3"]):
    """Adding substantial noise to the predicted coords strictly increases the smooth-lDDT loss."""
    loss_identical = smooth_lddt_loss(coords, coords)
    loss_noisy = smooth_lddt_loss(make_noisy(coords, sigma=2.0), coords)
    assert loss_noisy.item() > loss_identical.item()


def test_smooth_lddt_in_unit_interval(coords: Float[torch.Tensor, "B N_atoms 3"]):
    """The smooth-lDDT loss is bounded in [0, 1] regardless of the noise magnitude."""
    loss = smooth_lddt_loss(make_noisy(coords, sigma=5.0), coords)
    assert 0.0 <= loss.item() <= 1.0 + 1e-6


def test_smooth_lddt_masked_bounded(
    coords: Float[torch.Tensor, "B N_atoms 3"], half_mask: Bool[torch.Tensor, "B N_atoms"]
):
    """Masking half the atoms still yields a finite loss in the valid [0, 1] range."""
    loss = smooth_lddt_loss(make_noisy(coords, sigma=1.0), coords, mask=half_mask)
    assert torch.isfinite(loss)
    assert 0.0 <= loss.item() <= 1.0 + 1e-6


def test_smooth_lddt_gradient_flows():
    """The smooth-lDDT loss is differentiable with respect to the predicted coordinates."""
    r_true = torch.randn(N, 3)
    r_pred = torch.randn(N, 3, requires_grad=True)
    (grad,) = torch.autograd.grad(smooth_lddt_loss(r_pred, r_true), r_pred)
    assert torch.isfinite(grad).all()


def test_smooth_lddt_pairwise_sq_dist_matches_einsum():
    """The pairwise_sq_dist helper produces same result as direct einsum over difference vectors."""
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
):
    """Passing one-hot targets vs. integer bin indices yield numerically identical loss values."""
    assert torch.allclose(
        distogram_loss_residue(res_logits, res_onehot),
        distogram_loss_residue(res_logits, res_bin_idx),
        atol=1e-4,
    )


def test_distogram_residue_perfect_logits_near_zero(
    res_onehot: Float[torch.Tensor, "B L_res L_res B_res"],
):
    """Extremely high logits concentrated on the correct bin drive the cross-entropy near zero."""
    assert distogram_loss_residue(res_onehot * 1e6, res_onehot).max().item() < 1e-3


def test_distogram_residue_mask_changes_loss(
    res_logits: Float[torch.Tensor, "B L_res L_res B_res"],
    res_onehot: Float[torch.Tensor, "B L_res L_res B_res"],
    res_mask: Bool[torch.Tensor, "B L_res"],
):
    """Masking out half residues yields a finite but different loss from unmasked case."""
    loss_full = distogram_loss_residue(res_logits, res_onehot)
    loss_masked = distogram_loss_residue(res_logits, res_onehot, mask=res_mask)
    assert torch.isfinite(loss_masked).all()
    assert not torch.allclose(loss_full, loss_masked)


def test_distogram_residue_batched_output_shape(
    res_logits: Float[torch.Tensor, "B L_res L_res B_res"],
    res_bin_idx: Int[torch.Tensor, "B L_res L_res"],
):
    """Batched input returns a per-sample loss vector of shape [B]."""
    assert distogram_loss_residue(res_logits, res_bin_idx).shape == (B,)


def test_distogram_residue_gradient_flows(res_onehot: Float[torch.Tensor, "B L_res L_res B_res"]):
    """The residue distogram loss is differentiable with respect to the logits."""
    logits_g = torch.randn(L_RES, L_RES, B_RES, requires_grad=True)
    (grad,) = torch.autograd.grad(distogram_loss_residue(logits_g, res_onehot[0]), logits_g)
    assert torch.isfinite(grad).all()


def test_distogram_residue_ce_einsum_matches(
    res_logits: Float[torch.Tensor, "B L_res L_res B_res"],
    res_onehot: Float[torch.Tensor, "B L_res L_res B_res"],
):
    """The loss matches a manual mean-CE computed via einsum over log_softmax probabilities."""
    logits, y = res_logits[0], res_onehot[0]
    loss_ref = distogram_loss_residue(logits, y).item()
    loss_manual = ce_via_einsum(logits, y).sum() / (L_RES * L_RES)
    assert abs(loss_ref - loss_manual.item()) < 1e-5


def test_distogram_residue_unbatched_scalar():
    """Unbatched [L, L, B] logits yield a scalar (0-dim) loss rather than a length-1 vector."""
    logits = torch.randn(L_RES, L_RES, B_RES)
    assert distogram_loss_residue(logits, torch.randint(0, B_RES, (L_RES, L_RES))).ndim == 0


# ---------------------------------------------------------------------------
# distogram_loss_atom
# ---------------------------------------------------------------------------


def test_distogram_atom_uniform_worse_than_perfect(
    atom_onehot: Float[torch.Tensor, "B N_atoms N_atoms B_atom"],
    atom_local_mask: Bool[torch.Tensor, "B N_atoms N_atoms"],
):
    """Zero logits (uniform dist) produce higher loss than near-perfect logits for sample."""
    uniform = torch.zeros_like(atom_onehot)  # zero logits → uniform distribution after softmax
    loss_perfect = distogram_loss_atom(atom_onehot * 1e6, atom_onehot, atom_local_mask)
    loss_uniform = distogram_loss_atom(uniform, atom_onehot, atom_local_mask)
    assert (loss_uniform > loss_perfect).all()


def test_distogram_atom_local_and_full_differ(
    atom_logits: Float[torch.Tensor, "B N_atoms N_atoms B_atom"],
    atom_onehot: Float[torch.Tensor, "B N_atoms N_atoms B_atom"],
    atom_local_mask: Bool[torch.Tensor, "B N_atoms N_atoms"],
):
    """Local-window mask changes pairs, so masked and unmasked losses are not equal."""
    assert not torch.allclose(
        distogram_loss_atom(atom_logits, atom_onehot, atom_local_mask),
        distogram_loss_atom(atom_logits, atom_onehot),
    )


def test_distogram_atom_perfect_logits_near_zero(
    atom_onehot: Float[torch.Tensor, "B N_atoms N_atoms B_atom"],
    atom_local_mask: Bool[torch.Tensor, "B N_atoms N_atoms"],
):
    """Extremely high logits concentrated on correct bin drive the atom distogram CE near zero."""
    assert distogram_loss_atom(atom_onehot * 1e6, atom_onehot, atom_local_mask).max().item() < 1e-3


def test_distogram_atom_gradient_flows(
    atom_onehot: Float[torch.Tensor, "B N_atoms N_atoms B_atom"],
    atom_local_mask: Bool[torch.Tensor, "B N_atoms N_atoms"],
):
    """The atom distogram loss is differentiable with respect to the logits."""
    logits_g = torch.randn(N_ATOMS, N_ATOMS, B_ATOM, requires_grad=True)
    (grad,) = torch.autograd.grad(
        distogram_loss_atom(logits_g, atom_onehot[0], atom_local_mask[0]), logits_g
    )
    assert torch.isfinite(grad).all()


def test_distogram_atom_local_mask_shape_diagonal():
    """local_window_mask returns a square bool mask with the diagonal always True."""
    mask = local_window_mask(N_ATOMS, 2 * ATOMS_PER_RES)
    assert mask.shape == (N_ATOMS, N_ATOMS)
    assert mask.dtype == torch.bool
    assert mask.diagonal().all()


def test_distogram_atom_ce_einsum_matches(
    atom_logits: Float[torch.Tensor, "B N_atoms N_atoms B_atom"],
    atom_onehot: Float[torch.Tensor, "B N_atoms N_atoms B_atom"],
):
    """Atom distogram loss matches a manual mean-CE computed via einsum over log-softmax values."""
    logits, y = atom_logits[0], atom_onehot[0]
    loss_ref = distogram_loss_atom(logits, y).item()
    loss_manual = ce_via_einsum(logits, y).sum() / (N_ATOMS * N_ATOMS)
    assert abs(loss_ref - loss_manual.item()) < 1e-5


def test_distogram_atom_onehot_index_same_loss(
    atom_logits: Float[torch.Tensor, "B N_atoms N_atoms B_atom"],
    atom_onehot: Float[torch.Tensor, "B N_atoms N_atoms B_atom"],
    atom_bin_idx: Int[torch.Tensor, "B N_atoms N_atoms"],
    atom_local_mask: Bool[torch.Tensor, "B N_atoms N_atoms"],
):
    """One-hot targets and integer bin index targets yield identical atom distogram losses."""
    assert torch.allclose(
        distogram_loss_atom(atom_logits, atom_onehot, atom_local_mask),
        distogram_loss_atom(atom_logits, atom_bin_idx, atom_local_mask),
        atol=1e-4,
    )


# ---------------------------------------------------------------------------
# Shape-contract enforcement — negative tests
# ---------------------------------------------------------------------------


def test_atom_loss_wrong_shape() -> None:
    """Wrong last dim (4 instead of 3) on r_denoised triggers TypeCheckError."""
    r_bad = torch.zeros(B, N, 4)  # last dim must be 3
    r_gt = torch.zeros(B, N, 3)
    with pytest.raises(TypeCheckError):
        atom_loss(r_bad, r_gt)


def test_med_loss_per_block_wrong_shape() -> None:
    """Wrong last dim on r_denoised_k triggers TypeCheckError."""
    r_bad = torch.zeros(B, N, 4)  # last dim must be 3
    r_gt = torch.zeros(B, N, 3)
    logits = torch.zeros(B, N, VOCAB)
    aa_gt = torch.zeros(B, N, dtype=torch.long)
    with pytest.raises(TypeCheckError):
        med_loss_per_block(r_bad, r_gt, logits, aa_gt, LAM, ALPHA)


def testpairwise_dist_wrong_shape() -> None:
    """Wrong last dim (4 instead of 3) triggers TypeCheckError."""
    x_bad = torch.zeros(10, 4)  # last dim must be 3
    with pytest.raises(TypeCheckError):
        pairwise_dist(x_bad)


def test_smooth_lddt_loss_wrong_shape() -> None:
    """Wrong last dim (4 instead of 3) on r_pred triggers TypeCheckError."""
    r_pred_bad = torch.zeros(N, 4)  # last dim must be 3
    r_true = torch.zeros(N, 3)
    with pytest.raises(TypeCheckError):
        smooth_lddt_loss(r_pred_bad, r_true)


def test_distogram_loss_residue_wrong_shape() -> None:
    """2-D p (below min 3-D for '... N N n_bins') triggers TypeCheckError."""
    p_bad = torch.zeros(N, N)  # needs at least 3 dims for "... N_res N_res n_bins"
    y = torch.zeros(N, N, dtype=torch.long)
    with pytest.raises(TypeCheckError):
        distogram_loss_residue(p_bad, y)


def test_distogram_loss_atom_wrong_shape() -> None:
    """2-D q (below min 3-D for '... N K n_bins') triggers TypeCheckError."""
    q_bad = torch.zeros(N_ATOMS, K)  # needs at least 3 dims for "... N_atom K n_bins"
    y = torch.zeros(N_ATOMS, K, dtype=torch.long)
    with pytest.raises(TypeCheckError):
        distogram_loss_atom(q_bad, y)
