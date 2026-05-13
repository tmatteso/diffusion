"""Tests for EDM noise schedule data parameter computation."""

import math

import pytest
import torch
from helpers.compute_edm_data_params import compute_edm_noise_params, compute_sigma_data
from jaxtyping import Float

torch.manual_seed(42)

B = 2
N_RES = 6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _batch(atom_positions: torch.Tensor, atom_mask: torch.Tensor) -> dict:
    return {"atom_positions": atom_positions, "atom_mask": atom_mask}


def _loader(*batches: dict) -> list[dict]:
    return list(batches)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def constant_positions() -> Float[torch.Tensor, "B N_res 37 3"]:
    """All atoms at (2.0, 0.0, 0.0); every norm = 2.0."""
    pos = torch.zeros(B, N_RES, 37, 3)
    pos[..., 0] = 2.0
    return pos


@pytest.fixture
def all_ones_mask() -> Float[torch.Tensor, "B N_res 37"]:
    """Provide an all-ones atom mask of shape (B, N_RES, 37) — every atom is present."""
    return torch.ones(B, N_RES, 37)


@pytest.fixture
def spread_positions() -> Float[torch.Tensor, "B N_res 37 3"]:
    """Residue i: all atoms at (i, 0, 0). CA pairwise distances are |i - j|, minimum = 1."""
    pos = torch.zeros(B, N_RES, 37, 3)
    for i in range(N_RES):
        pos[:, i, :, 0] = float(i)
    return pos


@pytest.fixture
def spread_loader(
    spread_positions: Float[torch.Tensor, "B N_res 37 3"],
    all_ones_mask: Float[torch.Tensor, "B N_res 37"],
) -> list[dict]:
    """Provide a single-batch DataLoader-like list using the spread_positions fixture."""
    return _loader(_batch(spread_positions, all_ones_mask))


# ---------------------------------------------------------------------------
# compute_sigma_data
# ---------------------------------------------------------------------------


def test_compute_sigma_data_known_value(
    constant_positions: Float[torch.Tensor, "B N_res 37 3"],
    all_ones_mask: Float[torch.Tensor, "B N_res 37"],
) -> None:
    """compute_sigma_data returns 2.0 when all atom norms are 2.0."""
    # All norms = 2.0 → RMS = 2.0
    result = compute_sigma_data(_loader(_batch(constant_positions, all_ones_mask)))
    assert math.isclose(result, 2.0, rel_tol=1e-5)


def test_compute_sigma_data_atoms_at_origin():
    """compute_sigma_data returns 0 when all atoms are at the origin."""
    pos = torch.zeros(2, 4, 37, 3)  # all atoms at origin: every norm = 0
    mask = torch.ones(2, 4, 37)
    result = compute_sigma_data(_loader(_batch(pos, mask)))
    assert math.isclose(result, 0.0, abs_tol=1e-8)


def test_compute_sigma_data_mask_excludes_zero_atoms():
    """Only atom-0 (at norm 3.0) is valid; remaining atoms at origin are masked out."""
    pos = torch.zeros(2, 4, 37, 3)
    pos[:, :, 0, 0] = 3.0
    mask = torch.zeros(2, 4, 37)
    mask[:, :, 0] = 1.0
    result = compute_sigma_data(_loader(_batch(pos, mask)))
    assert math.isclose(result, 3.0, rel_tol=1e-5)


def test_compute_sigma_data_scales_with_radius():
    """compute_sigma_data doubles when all positions are doubled (RMS scales linearly)."""
    pos = torch.randn(2, 4, 37, 3)
    mask = torch.ones(2, 4, 37)
    s1 = compute_sigma_data(_loader(_batch(pos, mask)))
    s2 = compute_sigma_data(_loader(_batch(pos * 2, mask)))
    assert math.isclose(s2, 2.0 * s1, rel_tol=1e-5)


def test_compute_sigma_data_multi_batch_matches_single():
    """compute_sigma_data is consistent whether data is split across batches or not."""
    pos = torch.randn(4, 4, 37, 3)
    mask = torch.ones(4, 4, 37)
    single = compute_sigma_data(_loader(_batch(pos, mask)))
    split = compute_sigma_data([_batch(pos[:2], mask[:2]), _batch(pos[2:], mask[2:])])
    assert math.isclose(single, split, rel_tol=1e-5)


def test_compute_sigma_data_weighted_by_mask_count():
    """sigma_data with half the atoms masked should match full atoms at same positions."""
    pos = torch.zeros(1, 4, 2, 3)  # only 2 atom slots needed
    pos[:, :, 0, 0] = 5.0  # valid atom at norm 5
    pos[:, :, 1, 0] = 5.0  # second atom also at norm 5

    mask_full = torch.ones(1, 4, 2)
    mask_half = torch.zeros(1, 4, 2)
    mask_half[:, :, 0] = 1.0

    s_full = compute_sigma_data(_loader(_batch(pos, mask_full)))
    s_half = compute_sigma_data(_loader(_batch(pos, mask_half)))
    assert math.isclose(s_full, s_half, rel_tol=1e-5)


# ---------------------------------------------------------------------------
# compute_edm_noise_params
# ---------------------------------------------------------------------------


def test_compute_edm_noise_params_returns_dict(spread_loader: list[dict]) -> None:
    """compute_edm_noise_params returns a plain dict for downstream consumption."""
    assert isinstance(compute_edm_noise_params(spread_loader), dict)


def test_compute_edm_noise_params_has_required_keys(spread_loader: list[dict]) -> None:
    """compute_edm_noise_params output includes all four noise schedule keys."""
    params = compute_edm_noise_params(spread_loader)
    assert {"sigma_max", "sigma_min", "P_mean", "P_std"} <= params.keys()


def test_compute_edm_noise_params_sigma_max_positive(spread_loader: list[dict]) -> None:
    """sigma_max is strictly positive for a non-degenerate dataset."""
    assert compute_edm_noise_params(spread_loader)["sigma_max"] > 0.0


def test_compute_edm_noise_params_sigma_min_ge_floor(spread_loader: list[dict]) -> None:
    """sigma_min is always at least 2e-3, the hard floor imposed to avoid underflow."""
    assert compute_edm_noise_params(spread_loader)["sigma_min"] >= 2e-3


def test_compute_edm_noise_params_sigma_max_gt_sigma_min(spread_loader: list[dict]) -> None:
    """sigma_max exceeds sigma_min so the log-uniform noise distribution is well-defined."""
    params = compute_edm_noise_params(spread_loader)
    assert params["sigma_max"] > params["sigma_min"]


def test_compute_edm_noise_params_P_std_positive(spread_loader: list[dict]) -> None:
    """P_std is strictly positive, ensuring the log-normal noise sampler has non-zero width."""
    assert compute_edm_noise_params(spread_loader)["P_std"] > 0.0


def test_compute_edm_noise_params_P_mean_formula(spread_loader: list[dict]) -> None:
    """P_mean equals the midpoint of (log sigma_min, log sigma_max) as specified by EDM."""
    params = compute_edm_noise_params(spread_loader)
    expected = (math.log(params["sigma_min"]) + math.log(params["sigma_max"])) / 2.0
    assert math.isclose(params["P_mean"], expected, rel_tol=1e-5)


def test_compute_edm_noise_params_P_std_formula(spread_loader: list[dict]) -> None:
    """P_std equals half the log-range (log sigma_max − log sigma_min) / 2 as specified by EDM."""
    params = compute_edm_noise_params(spread_loader)
    expected = (math.log(params["sigma_max"]) - math.log(params["sigma_min"])) / 2.0
    assert math.isclose(params["P_std"], expected, rel_tol=1e-5)


def test_compute_edm_noise_params_sigma_min_floor_applied():
    """When all CA positions coincide, pairwise = 0 so sigma_min clamps to 2e-3."""
    pos = torch.zeros(2, 4, 37, 3)
    pos[:, :, 0, 0] = 10.0  # non-CA atom at norm 10 so sigma_max > 0
    # CA (index 1) stays at origin → all pairwise distances = 0
    mask = torch.ones(2, 4, 37)
    params = compute_edm_noise_params(_loader(_batch(pos, mask)))
    assert params["sigma_min"] == pytest.approx(2e-3)


def test_compute_edm_noise_params_sigma_max_bounded_by_max_norm(
    spread_positions: Float[torch.Tensor, "B N_res 37 3"],
    all_ones_mask: Float[torch.Tensor, "B N_res 37"],
) -> None:
    """sigma_max (99th pct) cannot exceed the maximum atom norm present."""
    max_norm = float(N_RES - 1)  # residue N_RES-1 sits at (N_RES-1, 0, 0)
    params = compute_edm_noise_params(_loader(_batch(spread_positions, all_ones_mask)))
    assert params["sigma_max"] <= max_norm + 1e-6


def test_compute_edm_noise_params_sigma_min_bounded_by_min_pairwise(
    spread_positions: Float[torch.Tensor, "B N_res 37 3"],
    all_ones_mask: Float[torch.Tensor, "B N_res 37"],
) -> None:
    """sigma_min (1st pct) cannot exceed the minimum pairwise CA distance (1 Å here)."""
    params = compute_edm_noise_params(_loader(_batch(spread_positions, all_ones_mask)))
    assert params["sigma_min"] <= 1.0 + 1e-6


def test_compute_edm_noise_params_multi_batch_consistent():
    """Splitting data across two batches produces the same result as one batch."""
    pos = torch.randn(4, N_RES, 37, 3).abs() + 0.5  # keep norms positive
    mask = torch.ones(4, N_RES, 37)
    single = compute_edm_noise_params(_loader(_batch(pos, mask)))
    split = compute_edm_noise_params([_batch(pos[:2], mask[:2]), _batch(pos[2:], mask[2:])])
    for key in ("sigma_max", "sigma_min", "P_mean", "P_std"):
        assert math.isclose(single[key], split[key], rel_tol=1e-4), key
