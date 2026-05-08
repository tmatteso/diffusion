import pytest
import torch
import torch.nn.functional as F
from architecture.losses import (
    atom_loss,
    distogram_loss_atom,
    distogram_loss_residue,
    med_loss,
    med_loss_per_block,
    smooth_lddt_loss,
)
from beartype import beartype
from einops import einsum, rearrange
from jaxtyping import Bool, Float, Int, jaxtyped

torch.manual_seed(42)

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
    return base + sigma * torch.randn_like(base)


@jaxtyped(typechecker=beartype)
def pairwise_sq_dist(
    x: Float[torch.Tensor, "N 3"],
) -> Float[torch.Tensor, "N N"]:
    diff = rearrange(x, "n d -> n 1 d") - rearrange(x, "n d -> 1 n d")
    return einsum(diff, diff, "n m d, n m d -> n m")


@jaxtyped(typechecker=beartype)
def local_window_mask(
    N_atoms: int,
    half_window: int,
) -> Bool[torch.Tensor, "N_atoms N_atoms"]:
    idx = torch.arange(N_atoms)
    dist = (rearrange(idx, "n -> n 1") - rearrange(idx, "m -> 1 m")).abs()
    return dist <= half_window


@jaxtyped(typechecker=beartype)
def to_onehot(
    indices: Int[torch.Tensor, "... L"],
    n_classes: int,
) -> Float[torch.Tensor, "... L n_classes"]:
    return F.one_hot(indices, num_classes=n_classes).float()


@jaxtyped(typechecker=beartype)
def block_decay_weights(
    K: int,
    gamma: float,
) -> Float[torch.Tensor, "K"]:
    ks = torch.arange(1, K + 1, dtype=torch.float32)
    return gamma ** (K - ks)


@jaxtyped(typechecker=beartype)
def ce_via_einsum(
    logits: Float[torch.Tensor, "... B"],
    targets: Float[torch.Tensor, "... B"],
) -> Float[torch.Tensor, "..."]:
    log_p = F.log_softmax(logits, dim=-1)
    return -einsum(targets, log_p, "... b, ... b -> ...")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def coords():
    return torch.randn(B, N, 3)


@pytest.fixture
def noisy_coords(coords):
    return make_noisy(coords, sigma=0.5)


@pytest.fixture
def half_mask():
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[:, N // 2 :] = False
    return mask


@pytest.fixture
def rotation():
    R, _ = torch.linalg.qr(torch.randn(3, 3))
    return R


@pytest.fixture
def r_gt():
    return torch.randn(B, N, 3)


@pytest.fixture
def aa_gt():
    return torch.randint(0, VOCAB, (B, N))


@pytest.fixture
def r_blocks():
    return torch.stack([torch.randn(B, N, 3) for _ in range(K)])


@pytest.fixture
def aa_blocks():
    return torch.stack([torch.randn(B, N, VOCAB) for _ in range(K)])


@pytest.fixture
def res_logits():
    return torch.randn(B, L_RES, L_RES, B_RES)


@pytest.fixture
def res_bin_idx():
    return torch.randint(0, B_RES, (B, L_RES, L_RES))


@pytest.fixture
def res_onehot(res_bin_idx):
    return to_onehot(res_bin_idx, B_RES)


@pytest.fixture
def res_mask():
    mask = torch.ones(B, L_RES, dtype=torch.bool)
    mask[:, L_RES // 2 :] = False
    return mask


@pytest.fixture
def atom_logits():
    return torch.randn(B, N_ATOMS, N_ATOMS, B_ATOM)


@pytest.fixture
def atom_bin_idx():
    return torch.randint(0, B_ATOM, (B, N_ATOMS, N_ATOMS))


@pytest.fixture
def atom_onehot(atom_bin_idx):
    return to_onehot(atom_bin_idx, B_ATOM)


@pytest.fixture
def atom_local_mask():
    lmask = local_window_mask(N_ATOMS, 2 * ATOMS_PER_RES)
    return lmask.unsqueeze(0).expand(B, -1, -1)


# ---------------------------------------------------------------------------
# atom_loss
# ---------------------------------------------------------------------------


def test_atom_loss_perfect_near_zero(coords):
    loss = atom_loss(coords, coords)
    assert loss.shape == (B,)
    assert loss.max().item() < 1e-5


def test_atom_loss_known_translation_near_zero(coords):
    translation = rearrange(torch.tensor([10.0, 5.0, -3.0]), "d -> 1 1 d")
    r_translated = coords + translation
    loss = atom_loss(r_translated, coords)
    assert (loss < 1e-4).all()


def test_atom_loss_full_mask_matches_no_mask(coords, noisy_coords):
    mask_all = torch.ones(B, N, dtype=torch.bool)
    assert torch.allclose(
        atom_loss(coords, noisy_coords, mask=mask_all),
        atom_loss(coords, noisy_coords),
        atol=1e-5,
    )


def test_atom_loss_increases_with_noise_level(coords):
    g = torch.Generator()
    g.manual_seed(0)
    noise_dir = torch.randn(B, N, 3, generator=g)
    loss_low  = atom_loss(coords + 0.1 * noise_dir, coords)
    loss_mid  = atom_loss(coords + 1.0 * noise_dir, coords)
    loss_high = atom_loss(coords + 5.0 * noise_dir, coords)
    assert (loss_low < loss_mid).all()
    assert (loss_mid < loss_high).all()


def test_atom_loss_gradient_flows_through_pred():
    r_pred = torch.randn(N, 3, requires_grad=True)
    atom_loss(r_pred, torch.randn(N, 3)).backward()
    assert r_pred.grad is not None
    assert torch.isfinite(r_pred.grad).all()


def test_atom_loss_gt_receives_no_gradient():
    r_pred = torch.randn(N, 3, requires_grad=True)
    r_gt = torch.randn(N, 3, requires_grad=True)
    atom_loss(r_pred, r_gt).backward()
    assert r_gt.grad is None


def test_atom_loss_rotation_invariant(coords, rotation):
    r_gt = torch.randn(B, N, 3)
    r_gt_rot = einsum(r_gt, rotation, "b n d, d e -> b n e")
    assert torch.allclose(atom_loss(coords, r_gt), atom_loss(coords, r_gt_rot), atol=1e-4)


def test_pairwise_sq_dist_shape_symmetric():
    x = torch.randn(N, 3)
    D = pairwise_sq_dist(x)
    assert D.shape == (N, N)
    assert D.diagonal().abs().max().item() < 1e-5
    assert torch.allclose(D, D.T, atol=1e-5)


# ---------------------------------------------------------------------------
# med_loss
# ---------------------------------------------------------------------------


def test_med_loss_scalar_finite_positive(r_gt, aa_gt, r_blocks, aa_blocks):
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


def test_med_loss_perfect_struct_near_zero(r_gt, aa_gt, aa_blocks):
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
    w = block_decay_weights(K, GAMMA)
    assert w.shape == (K,)
    assert (w[1:] > w[:-1]).all()
    assert abs(w[-1].item() - 1.0) < 1e-6


def test_med_loss_mismatched_blocks_raises(r_gt, aa_gt, r_blocks, aa_blocks):
    with pytest.raises(ValueError):
        med_loss(
            list(r_blocks),
            r_gt,
            list(aa_blocks)[:-1],
            aa_gt,
            lam=LAM,
            alpha_0=ALPHA,
        )


def test_med_loss_gradient_flows_to_first_block(r_gt, aa_gt):
    r0 = torch.randn(N, 3, requires_grad=True)
    r_blocks_g = [r0] + [torch.randn(N, 3) for _ in range(K - 1)]
    aa_blocks_g = [torch.randn(N, VOCAB) for _ in range(K)]
    med_loss(
        r_blocks_g,
        r_gt[0],
        aa_blocks_g,
        aa_gt[0],
        lam=LAM,
        alpha_0=ALPHA,
        gamma=GAMMA,
    ).backward()
    assert r0.grad is not None
    assert torch.isfinite(r0.grad).all()


def test_med_loss_lam_zero_less_than_lam_positive(r_gt, aa_gt, r_blocks, aa_blocks):
    r_list, aa_list = list(r_blocks), list(aa_blocks)
    loss_lam0 = med_loss(r_list, r_gt, aa_list, aa_gt, lam=0.0,  alpha_0=ALPHA, gamma=GAMMA)
    loss_lam1 = med_loss(r_list, r_gt, aa_list, aa_gt, lam=LAM, alpha_0=ALPHA, gamma=GAMMA)
    assert loss_lam0.item() < loss_lam1.item()


def test_med_loss_alpha_zero_less_than_alpha_positive(r_gt, aa_gt, r_blocks, aa_blocks):
    r_list, aa_list = list(r_blocks), list(aa_blocks)
    loss_a0 = med_loss(r_list, r_gt, aa_list, aa_gt, lam=LAM, alpha_0=0.0,   gamma=GAMMA)
    loss_a1 = med_loss(r_list, r_gt, aa_list, aa_gt, lam=LAM, alpha_0=ALPHA, gamma=GAMMA)
    assert loss_a0.item() < loss_a1.item()


# ---------------------------------------------------------------------------
# smooth_lddt_loss
# ---------------------------------------------------------------------------


def test_smooth_lddt_identical_coords_expected_value():
    r = torch.randn(10, 3) * 0.1
    loss = smooth_lddt_loss(r, r)
    expected = 1.0 - 0.25 * sum(torch.sigmoid(torch.tensor(t)).item() for t in [0.5, 1.0, 2.0, 4.0])
    assert abs(loss.item() - expected) < 1e-4


def test_smooth_lddt_noisy_exceeds_identical(coords):
    loss_identical = smooth_lddt_loss(coords, coords)
    loss_noisy = smooth_lddt_loss(make_noisy(coords, sigma=2.0), coords)
    assert loss_noisy.item() > loss_identical.item()


def test_smooth_lddt_in_unit_interval(coords):
    loss = smooth_lddt_loss(make_noisy(coords, sigma=5.0), coords)
    assert 0.0 <= loss.item() <= 1.0 + 1e-6


def test_smooth_lddt_masked_bounded(coords, half_mask):
    loss = smooth_lddt_loss(make_noisy(coords, sigma=1.0), coords, mask=half_mask)
    assert torch.isfinite(loss)
    assert 0.0 <= loss.item() <= 1.0 + 1e-6


def test_smooth_lddt_gradient_flows():
    r_true = torch.randn(N, 3)
    r_pred = torch.randn(N, 3, requires_grad=True)
    smooth_lddt_loss(r_pred, r_true).backward()
    assert r_pred.grad is not None
    assert torch.isfinite(r_pred.grad).all()


def test_smooth_lddt_pairwise_sq_dist_matches_einsum():
    x = torch.randn(N, 3)
    diff = rearrange(x, "n d -> n 1 d") - rearrange(x, "n d -> 1 n d")
    sq_ref = einsum(diff, diff, "n m d, n m d -> n m")
    assert torch.allclose(pairwise_sq_dist(x), sq_ref, atol=1e-6)


# ---------------------------------------------------------------------------
# distogram_loss_residue
# ---------------------------------------------------------------------------


def test_distogram_residue_onehot_index_same_loss(res_logits, res_bin_idx, res_onehot):
    assert torch.allclose(
        distogram_loss_residue(res_logits, res_onehot),
        distogram_loss_residue(res_logits, res_bin_idx),
        atol=1e-4,
    )


def test_distogram_residue_perfect_logits_near_zero(res_onehot):
    assert distogram_loss_residue(res_onehot * 1e6, res_onehot).max().item() < 1e-3


def test_distogram_residue_mask_changes_loss(res_logits, res_onehot, res_mask):
    loss_full = distogram_loss_residue(res_logits, res_onehot)
    loss_masked = distogram_loss_residue(res_logits, res_onehot, mask=res_mask)
    assert torch.isfinite(loss_masked).all()
    assert not torch.allclose(loss_full, loss_masked)


def test_distogram_residue_batched_output_shape(res_logits, res_bin_idx):
    assert distogram_loss_residue(res_logits, res_bin_idx).shape == (B,)


def test_distogram_residue_gradient_flows(res_onehot):
    logits_g = torch.randn(L_RES, L_RES, B_RES, requires_grad=True)
    distogram_loss_residue(logits_g, res_onehot[0]).backward()
    assert logits_g.grad is not None
    assert torch.isfinite(logits_g.grad).all()


def test_distogram_residue_ce_einsum_matches(res_logits, res_onehot):
    logits, y = res_logits[0], res_onehot[0]
    loss_ref = distogram_loss_residue(logits, y).item()
    loss_manual = ce_via_einsum(logits, y).sum() / (L_RES * L_RES)
    assert abs(loss_ref - loss_manual.item()) < 1e-5


def test_distogram_residue_unbatched_scalar():
    logits = torch.randn(L_RES, L_RES, B_RES)
    assert distogram_loss_residue(logits, torch.randint(0, B_RES, (L_RES, L_RES))).ndim == 0


# ---------------------------------------------------------------------------
# distogram_loss_atom
# ---------------------------------------------------------------------------


def test_distogram_atom_uniform_worse_than_perfect(atom_onehot, atom_local_mask):
    uniform = torch.zeros_like(atom_onehot)   # zero logits → uniform distribution after softmax
    loss_perfect = distogram_loss_atom(atom_onehot * 1e6, atom_onehot, atom_local_mask)
    loss_uniform = distogram_loss_atom(uniform, atom_onehot, atom_local_mask)
    assert (loss_uniform > loss_perfect).all()


def test_distogram_atom_local_and_full_differ(atom_logits, atom_onehot, atom_local_mask):
    assert not torch.allclose(
        distogram_loss_atom(atom_logits, atom_onehot, atom_local_mask),
        distogram_loss_atom(atom_logits, atom_onehot),
    )


def test_distogram_atom_perfect_logits_near_zero(atom_onehot, atom_local_mask):
    assert distogram_loss_atom(atom_onehot * 1e6, atom_onehot, atom_local_mask).max().item() < 1e-3


def test_distogram_atom_gradient_flows(atom_onehot, atom_local_mask):
    logits_g = torch.randn(N_ATOMS, N_ATOMS, B_ATOM, requires_grad=True)
    distogram_loss_atom(logits_g, atom_onehot[0], atom_local_mask[0]).backward()
    assert logits_g.grad is not None
    assert torch.isfinite(logits_g.grad).all()


def test_distogram_atom_local_mask_shape_diagonal():
    mask = local_window_mask(N_ATOMS, 2 * ATOMS_PER_RES)
    assert mask.shape == (N_ATOMS, N_ATOMS)
    assert mask.dtype == torch.bool
    assert mask.diagonal().all()


def test_distogram_atom_ce_einsum_matches(atom_logits, atom_onehot):
    logits, y = atom_logits[0], atom_onehot[0]
    loss_ref = distogram_loss_atom(logits, y).item()
    loss_manual = ce_via_einsum(logits, y).sum() / (N_ATOMS * N_ATOMS)
    assert abs(loss_ref - loss_manual.item()) < 1e-5


def test_distogram_atom_onehot_index_same_loss(
    atom_logits, atom_onehot, atom_bin_idx, atom_local_mask
):
    assert torch.allclose(
        distogram_loss_atom(atom_logits, atom_onehot, atom_local_mask),
        distogram_loss_atom(atom_logits, atom_bin_idx, atom_local_mask),
        atol=1e-4,
    )
