"""Tests for the pairformer stack.

Covers TriangleMultiplicationOutgoing and TriangleMultiplicationIncoming
(the exact output positions a single pair entry reaches, the gates, the
LayerNorm scale invariance of the projections, ``c_hidden``, and how the two
contraction directions relate), PairformerBlock with and without the single
track, PairformerStack as a composition of its blocks, and the output,
gradient, batch-independence and shape contracts shared by all of them.

Inputs come from ``tests.patterns.sine_pattern`` and every module is
built under a fixed seed, so each test is reproducible on its own.
"""

import dataclasses
import enum
from collections.abc import Callable
from typing import cast

import pytest
import torch
from architecture.errors import InvalidPairHeadDimensionError
from architecture.pairformer_stack import PairformerBlock, PairformerStack
from architecture.pairformer_stack import (
    TriangleMultiplicationIncoming as TMI,
)
from architecture.pairformer_stack import (
    TriangleMultiplicationOutgoing as TMO,
)
from einops import rearrange, reduce
from helpers.useful_objects import manual_seed
from jaxtyping import Bool, Float, TypeCheckError
from tests.patterns import sine_pattern

type Pair = Float[torch.Tensor, "B N_res N_res c"]
type PairModule = TMO | TMI | PairformerStack


@dataclasses.dataclass(frozen=True)
class PairDims:
    """Sizes shared by every module and input in this file.

    Attributes:
        batch: Batch size ``B``.
        n_res: Residues ``N_res``.
        c: Pair embedding width; divisible by ``n_heads``.
        c_hidden: Hidden width of the triangle multiplications.
        c_res: Single embedding width for the single-track block.
        n_heads: Attention heads; triangle attention uses ``n_heads // 4``.
        n_blocks: Blocks in the stack.
    """

    batch: int
    n_res: int
    c: int
    c_hidden: int
    c_res: int
    n_heads: int
    n_blocks: int


@dataclasses.dataclass(frozen=True)
class TriangleCase:
    """A triangle multiplication direction and where ``z[0, 1]`` reaches.

    Perturbing the single entry ``z[:, 0, 1, :]`` changes exactly the
    output entries in ``rows``, ``cols`` and ``points``. Outgoing computes
    ``m[i, j] = Σ_k a[i, k] b[k, j]``, so entry ``(0, 1)`` feeds ``a`` in
    row 0 and ``b`` in column 1. Incoming computes
    ``m[i, j] = Σ_k a[k, i] b[j, k]``, so it feeds row 1 and column 0, plus
    the outer gate at ``(0, 1)`` itself.

    Attributes:
        module_cls: The triangle multiplication class.
        rows: Output rows that change in full.
        cols: Output columns that change in full.
        points: Further single output entries ``(i, j)`` that change.
    """

    module_cls: type[TMO | TMI]
    rows: tuple[int, ...]
    cols: tuple[int, ...]
    points: tuple[tuple[int, int], ...]


class TriangleScenario(enum.Enum):
    """The two triangle multiplication contraction directions.

    Members:
        outgoing: ``TriangleMultiplicationOutgoing`` (Algorithm 17 step 2).
        incoming: ``TriangleMultiplicationIncoming`` (Algorithm 17 step 3).
    """

    outgoing = TriangleCase(TMO, (0,), (1,), ())
    incoming = TriangleCase(TMI, (1,), (0,), ((0, 1),))


@dataclasses.dataclass(frozen=True)
class SymmetryCase:
    """Whether the pair input is symmetric and what that implies.

    Attributes:
        symmetric_input: Average ``z`` with its transpose before the call.
        directions_agree: Whether outgoing and incoming must then match.
    """

    symmetric_input: bool
    directions_agree: bool


class SymmetryScenario(enum.Enum):
    """Pair inputs for comparing the two triangle directions.

    Members:
        asymmetric: ``z`` as built, so the two directions differ.
        symmetric: ``z`` symmetrised, so the two directions agree.
    """

    asymmetric = SymmetryCase(symmetric_input=False, directions_agree=False)
    symmetric = SymmetryCase(symmetric_input=True, directions_agree=True)


@dataclasses.dataclass(frozen=True)
class MalformedPair:
    """A reshaping of a valid pair tensor that breaks the shape contract.

    Attributes:
        malform: Maps a valid ``(B, N, N, c)`` tensor to a malformed one.
    """

    malform: Callable[[Pair], Float[torch.Tensor, "..."]]


class MalformedScenario(enum.Enum):
    """Pair tensors every module in this file must reject.

    Members:
        missing_batch: ``(N, N, c)`` with no batch axis.
        missing_channel: ``(B, N, N)`` with no channel axis.
        non_square: ``(B, N, N - 1, c)``; the two residue axes differ.
    """

    missing_batch = MalformedPair(lambda z: z[0])
    missing_channel = MalformedPair(lambda z: z[..., 0])
    non_square = MalformedPair(lambda z: z[:, :, 1:])


@pytest.fixture
def dims() -> PairDims:
    """Return the small sizes every test builds its modules and inputs from.

    Returns:
        Sizes with ``c`` divisible by ``n_heads`` and ``n_heads >= 4``.
    """
    return PairDims(
        batch=2,
        n_res=6,
        c=16,
        c_hidden=8,
        c_res=8,
        n_heads=4,
        n_blocks=3,
    )


@pytest.fixture
def z(dims: PairDims) -> Pair:
    """Return a deterministic, asymmetric pair embedding.

    Args:
        dims: Tensor sizes.

    Returns:
        ``(B, N_res, N_res, c)`` tensor with distinct batch items.
    """
    return sine_pattern((dims.batch, dims.n_res, dims.n_res, dims.c), 0.37)


@pytest.fixture
def s(
    dims: PairDims,
) -> Float[torch.Tensor, "B N_res c_res"]:
    """Return a deterministic single embedding for the single-track block.

    Args:
        dims: Tensor sizes.

    Returns:
        ``(B, N_res, c_res)`` tensor.
    """
    return sine_pattern((dims.batch, dims.n_res, dims.c_res), 0.53)


@pytest.fixture
def triangle(triangle_scenario: TriangleScenario, dims: PairDims) -> TMO | TMI:
    """Build the scenario's triangle multiplication under a fixed seed.

    Args:
        triangle_scenario: Direction to build; the requesting test
            parametrizes it.
        dims: Module sizes.

    Returns:
        The module in eval mode.
    """
    _ = manual_seed(0)
    module_cls = triangle_scenario.value.module_cls
    return module_cls(c=dims.c, c_hidden=dims.c_hidden).eval()


@pytest.fixture
def tmo(dims: PairDims) -> TMO:
    """Build a TriangleMultiplicationOutgoing under a fixed seed.

    Args:
        dims: Module sizes.

    Returns:
        The module in eval mode.
    """
    _ = manual_seed(0)
    return TMO(c=dims.c, c_hidden=dims.c_hidden).eval()


@pytest.fixture
def tmi(tmo: TMO, dims: PairDims) -> TMI:
    """Build a TriangleMultiplicationIncoming sharing ``tmo``'s weights.

    The two classes have identical parameters, so copying the state dict
    leaves the contraction direction as the only difference between them.

    Args:
        tmo: Outgoing module to copy weights from.
        dims: Module sizes.

    Returns:
        The module in eval mode.
    """
    module = TMI(c=dims.c, c_hidden=dims.c_hidden).eval()
    _ = module.load_state_dict(tmo.state_dict())
    return module


@pytest.fixture
def block(dims: PairDims) -> PairformerBlock:
    """Build a pair-only PairformerBlock under a fixed seed.

    Args:
        dims: Module sizes.

    Returns:
        The block in eval mode, so dropout is off.
    """
    _ = manual_seed(0)
    return PairformerBlock(dims.c, None, dims.n_heads).eval()


@pytest.fixture
def single_track_block(dims: PairDims) -> PairformerBlock:
    """Build a PairformerBlock that also updates ``s``, under a fixed seed.

    Args:
        dims: Module sizes.

    Returns:
        The block in eval mode.
    """
    _ = manual_seed(0)
    return PairformerBlock(dims.c, dims.c_res, dims.n_heads).eval()


@pytest.fixture
def stack(dims: PairDims) -> PairformerStack:
    """Build a pair-only PairformerStack under a fixed seed.

    Args:
        dims: Module sizes.

    Returns:
        The stack in eval mode.
    """
    _ = manual_seed(0)
    return PairformerStack(dims.c, dims.n_blocks, dims.n_heads).eval()


def pair_forward(module: PairModule, pair: Pair) -> Pair:
    """Run ``module`` on ``pair`` alone and return its output pair tensor.

    ``PairformerStack`` takes the single embedding first, and passing
    ``s=None`` makes it process and return the pair tensor alone; the two
    triangle modules take the pair tensor only.

    Args:
        module: Triangle module or pair-only stack under test.
        pair: Pair embedding.

    Returns:
        The module's output pair tensor, shaped like ``pair``.
    """
    if isinstance(module, PairformerStack):
        return module(None, pair)
    return module(pair)


@pytest.fixture
def pair_module(
    request: pytest.FixtureRequest,
    pair_module_name: str,
) -> PairModule:
    """Return the module fixture named by ``pair_module_name``.

    A pair-only ``PairformerBlock`` is covered through the stack, which
    ``test_stack_applies_its_blocks_in_order`` ties to its blocks.

    Args:
        request: Pytest request used to look up the named fixture.
        pair_module_name: ``"tmo"``, ``"tmi"`` or ``"stack"``, the name
            of the fixture to use; the requesting test parametrizes it.

    Returns:
        The module under test.
    """
    return cast(PairModule, request.getfixturevalue(pair_module_name))


@pytest.fixture
def changed_mask(
    triangle_scenario: TriangleScenario,
    dims: PairDims,
) -> Bool[torch.Tensor, "N_res N_res"]:
    """Build the output positions the scenario says ``z[0, 1]`` reaches.

    Args:
        triangle_scenario: Holds the rows, columns and points; the
            requesting test parametrizes it.
        dims: Supplies ``N_res``.

    Returns:
        Boolean ``(N_res, N_res)`` mask, True where the output must change.
    """
    case = triangle_scenario.value
    mask = torch.zeros(dims.n_res, dims.n_res, dtype=torch.bool)
    mask[list(case.rows), :] = True
    mask[:, list(case.cols)] = True
    for i, j in case.points:
        mask[i, j] = True
    return mask


@pytest.fixture
def direction_input(symmetry_scenario: SymmetryScenario, z: Pair) -> Pair:
    """Return ``z``, or ``z`` averaged with its transpose, per the scenario.

    Args:
        symmetry_scenario: Whether the input must be symmetric; the
            requesting test parametrizes it.
        z: Asymmetric pair embedding.

    Returns:
        A pair embedding that is symmetric in ``i`` and ``j`` exactly
        when the scenario asks for one.
    """
    if not symmetry_scenario.value.symmetric_input:
        return z
    return (z + rearrange(z, "b i j c -> b j i c")) / 2


@pytest.mark.parametrize(
    "pair_module_name",
    ["tmo", "tmi", "stack"],
    ids=["outgoing", "incoming", "stack"],
)
def test_output_shape_dtype_and_finite(
    z: Pair,
    pair_module: PairModule,
) -> None:
    """Each module maps ``(B, N, N, c)`` float32 to the same, all finite.

    Args:
        z: Valid pair embedding.
        pair_module: Module under test.
    """
    with torch.no_grad():
        out = pair_forward(pair_module, z)
    assert out.shape == z.shape
    assert out.dtype == z.dtype
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    "pair_module_name",
    ["tmo", "tmi", "stack"],
    ids=["outgoing", "incoming", "stack"],
)
def test_gradient_reaches_input(z: Pair, pair_module: PairModule) -> None:
    """The summed output has a finite, nonzero gradient with respect to ``z``.

    Args:
        z: Valid pair embedding.
        pair_module: Module under test.
    """
    leaf = z.clone().requires_grad_()
    out = reduce(pair_forward(pair_module, leaf), "b i j c -> ", "sum")
    (grad,) = torch.autograd.grad(out, leaf)
    assert torch.isfinite(grad).all()
    assert torch.count_nonzero(grad) > 0


@pytest.mark.parametrize(
    "pair_module_name",
    ["tmo", "tmi", "stack"],
    ids=["outgoing", "incoming", "stack"],
)
def test_batch_items_are_independent(z: Pair, pair_module: PairModule) -> None:
    """Running each batch item alone matches its slice of the batched run.

    Args:
        z: Pair embedding whose batch items differ.
        pair_module: Module under test.
    """
    with torch.no_grad():
        batched = pair_forward(pair_module, z)
        separate = torch.cat(
            [pair_forward(pair_module, z[i : i + 1]) for i in range(len(z))],
        )
    torch.testing.assert_close(batched, separate, atol=1e-5, rtol=0)


@pytest.mark.parametrize(
    "pair_module_name",
    ["tmo", "tmi", "stack"],
    ids=["outgoing", "incoming", "stack"],
)
@pytest.mark.parametrize(
    "malformed_scenario",
    list(MalformedScenario),
    ids=[case.name for case in MalformedScenario],
)
def test_forward_rejects_malformed_pair(
    z: Pair,
    pair_module: PairModule,
    malformed_scenario: MalformedScenario,
) -> None:
    """Each module's jaxtyping contract rejects every malformed layout.

    Args:
        z: Valid pair embedding to cut the malformed one from.
        pair_module: Module under test.
        malformed_scenario: How to break the shape.
    """
    malformed = malformed_scenario.value.malform(z)
    with pytest.raises(TypeCheckError):
        _ = pair_forward(pair_module, malformed)


@pytest.mark.parametrize(
    "triangle_scenario",
    list(TriangleScenario),
    ids=[case.name for case in TriangleScenario],
)
def test_triangle_single_entry_reaches_exactly_its_row_and_column(
    triangle: TMO | TMI,
    z: Pair,
    changed_mask: Bool[torch.Tensor, "N_res N_res"],
) -> None:
    """Perturbing ``z[0, 1]`` changes exactly the positions the scenario lists.

    This pins down the contraction direction: outgoing and incoming reach
    different rows and columns, so implementing either as the other fails.
    The perturbation varies across channels, because LayerNorm would cancel
    a uniform shift.

    Args:
        triangle: Module under test.
        z: Valid pair embedding.
        changed_mask: Output positions that must change.
    """
    perturbed = z.clone()
    perturbed[:, 0, 1] += 5 * sine_pattern((z.shape[-1],), 1.3)
    with torch.no_grad():
        moved = ~torch.isclose(triangle(perturbed), triangle(z), atol=1e-6)
    changed = reduce(moved.float(), "b i j c -> i j", "max") > 0
    assert torch.equal(changed, changed_mask)


@pytest.mark.parametrize(
    "triangle_scenario",
    list(TriangleScenario),
    ids=[case.name for case in TriangleScenario],
)
def test_triangle_closed_outer_gate_suppresses_output(
    triangle: TMO | TMI,
    z: Pair,
) -> None:
    """With the outer gate pinned near 0, the output is near 0 everywhere.

    Args:
        triangle: Module under test; its gate is pinned here.
        z: Valid pair embedding.
    """
    with torch.no_grad():
        _ = triangle.gate.weight.fill_(0.0)
        _ = triangle.gate.bias.fill_(-100.0)
        out = triangle(z)
    torch.testing.assert_close(out, torch.zeros_like(out), atol=1e-6, rtol=0)


@pytest.mark.parametrize(
    "triangle_scenario",
    list(TriangleScenario),
    ids=[case.name for case in TriangleScenario],
)
def test_triangle_outer_gate_sees_unnormalised_input(
    triangle: TMO | TMI,
    z: Pair,
) -> None:
    """With inner gates pinned open, scaling ``z`` by 100 changes the output.

    The projections see ``LayerNorm(z)``, which is scale-invariant, so the
    change can only come from the outer gate reading raw ``z``.

    Args:
        triangle: Module under test; its inner gates are pinned here.
        z: Valid pair embedding.
    """
    with torch.no_grad():
        for gate in (triangle.gate_a, triangle.gate_b):
            _ = gate.weight.fill_(0.0)
            _ = gate.bias.fill_(100.0)
        base = triangle(z)
        scaled = triangle(100.0 * z)
    assert not torch.allclose(base, scaled, atol=1e-4)


@pytest.mark.parametrize(
    "triangle_scenario",
    list(TriangleScenario),
    ids=[case.name for case in TriangleScenario],
)
def test_triangle_projections_are_scale_invariant(
    triangle: TMO | TMI,
    z: Pair,
) -> None:
    """With the outer gate pinned open, ``z`` and ``2 z`` give equal output.

    Pinning the outer gate removes the only path that reads raw ``z``; the
    rest sees ``LayerNorm(z)``, which ignores scale.

    Args:
        triangle: Module under test; its outer gate is pinned here.
        z: Valid pair embedding.
    """
    with torch.no_grad():
        _ = triangle.gate.weight.fill_(0.0)
        _ = triangle.gate.bias.fill_(100.0)
        base = triangle(z)
        scaled = triangle(2.0 * z)
    torch.testing.assert_close(base, scaled, atol=1e-4, rtol=0)


@pytest.mark.parametrize(
    "triangle_scenario",
    list(TriangleScenario),
    ids=[case.name for case in TriangleScenario],
)
@pytest.mark.parametrize(
    ("c_hidden", "expected_hidden"),
    [pytest.param(None, 16, id="default"), pytest.param(4, 4, id="custom")],
)
def test_triangle_c_hidden_sets_internal_width_only(
    triangle_scenario: TriangleScenario,
    dims: PairDims,
    z: Pair,
    c_hidden: int | None,
    expected_hidden: int,
) -> None:
    """``c_hidden`` sizes the projections; omitted, it defaults to ``c``.

    The output width stays ``c`` either way.

    Args:
        triangle_scenario: Direction to build.
        dims: Supplies ``c = 16``.
        z: Valid pair embedding.
        c_hidden: Value passed to the constructor.
        expected_hidden: Internal width the module must use.
    """
    module = triangle_scenario.value.module_cls(c=dims.c, c_hidden=c_hidden)
    with torch.no_grad():
        out = module.eval()(z)
    assert module.proj_a.weight.shape == (expected_hidden, dims.c)
    assert module.norm_out.normalized_shape == (expected_hidden,)
    assert module.proj_out.weight.shape == (dims.c, expected_hidden)
    assert out.shape == z.shape


@pytest.mark.parametrize(
    "symmetry_scenario",
    list(SymmetryScenario),
    ids=[case.name for case in SymmetryScenario],
)
def test_directions_agree_only_on_symmetric_input(
    tmo: TMO,
    tmi: TMI,
    symmetry_scenario: SymmetryScenario,
    direction_input: Pair,
) -> None:
    """With shared weights, outgoing equals incoming iff ``z`` is symmetric.

    For ``z[i, k] = z[k, i]``, LayerNorm keeps the symmetry, so
    ``a[k, i] = a[i, k]`` and ``b[j, k] = b[k, j]`` and the incoming sum
    ``Σ_k a[k, i] b[j, k]`` becomes the outgoing ``Σ_k a[i, k] b[k, j]``.
    The outer gate reads the same ``z[i, j]`` in both.

    Args:
        tmo: Outgoing module.
        tmi: Incoming module with ``tmo``'s weights.
        symmetry_scenario: Whether the input is symmetric and whether the
            two directions must then agree.
        direction_input: Pair embedding built for ``symmetry_scenario``.
    """
    with torch.no_grad():
        out_tmo = tmo(direction_input)
        out_tmi = tmi(direction_input)
    agree = torch.allclose(out_tmo, out_tmi, atol=1e-4)
    assert agree == symmetry_scenario.value.directions_agree


def test_incoming_equals_outgoing_of_transpose_with_gates_pinned(
    tmo: TMO,
    tmi: TMI,
    z: Pair,
) -> None:
    """``incoming(z) == outgoing(zᵀ)`` once all gates are constant.

    With every gate pinned, incoming gives ``Σ_k p_a(zn[k, i]) p_b(zn[j, k])``
    and outgoing on ``zᵀ`` gives ``Σ_k p_a(zn[k, i]) p_b(zn[j, k])`` too,
    since ``LN(zᵀ[i, k]) = LN(z[k, i])``. This fails if either contraction
    were implemented as the other. Not parametrized: it is one identity
    relating the two directions.

    Args:
        tmo: Outgoing module; its gates are pinned here.
        tmi: Incoming module with ``tmo``'s weights; gates pinned here.
        z: Asymmetric pair embedding.
    """
    with torch.no_grad():
        for module in (tmo, tmi):
            for gate in (module.gate, module.gate_a, module.gate_b):
                _ = gate.weight.fill_(0.0)
                _ = gate.bias.fill_(100.0)
        incoming = tmi(z)
        outgoing_t = tmo(rearrange(z, "b i j c -> b j i c"))
    torch.testing.assert_close(incoming, outgoing_t, atol=1e-5, rtol=0)


def test_block_pair_only_changes_z_and_returns_no_single(
    block: PairformerBlock,
    z: Pair,
) -> None:
    """With ``s=None`` the block returns ``(None, z_out)`` with ``z_out != z``.

    Not parametrized: the single-track case is tested separately because
    it returns a different structure.

    Args:
        block: Pair-only block.
        z: Valid pair embedding.
    """
    with torch.no_grad():
        s_out, z_out = block(None, z)
    assert s_out is None
    assert z_out.shape == z.shape
    assert not torch.allclose(z_out, z)


def test_block_single_track_updates_s(
    dims: PairDims,
    single_track_block: PairformerBlock,
    s: Float[torch.Tensor, "B N_res c_res"],
    z: Pair,
) -> None:
    """With ``c_res`` set, the block also returns an updated, finite ``s``.

    Not parametrized: one block with a single track covers the branch that
    adds attention with pair bias and a second transition to ``s``.

    Args:
        dims: Expected sizes.
        single_track_block: Block built with ``c_res``.
        s: Valid single embedding.
        z: Valid pair embedding.
    """
    with torch.no_grad():
        s_out, z_out = single_track_block(s, z)
    assert s_out is not None
    assert s_out.shape == (dims.batch, dims.n_res, dims.c_res)
    assert torch.isfinite(s_out).all()
    assert not torch.allclose(s_out, s)
    assert z_out.shape == z.shape


def test_stack_applies_its_blocks_in_order(
    dims: PairDims,
    stack: PairformerStack,
    z: Pair,
) -> None:
    """The stack equals running its own blocks one after another.

    Shows the activation-checkpointing wrapper adds nothing and that the
    stack holds ``n_blocks`` blocks. Not parametrized: the composition is
    the same for any block count.

    Args:
        dims: Supplies ``n_blocks``.
        stack: Pair-only stack.
        z: Valid pair embedding.
    """
    expected = z
    with torch.no_grad():
        for blk in stack.blocks:
            _, expected = blk(None, expected)
        out = stack(None, z)
    assert len(stack.blocks) == dims.n_blocks
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=0)


def test_block_rejects_heads_that_do_not_divide_width(dims: PairDims) -> None:
    """``c_pair`` not divisible by ``n_heads`` raises at construction.

    Not parametrized: the check is a single modulus, so one indivisible
    pair covers it.

    Args:
        dims: Supplies a valid ``n_heads``.
    """
    with pytest.raises(InvalidPairHeadDimensionError, match="divisible"):
        _ = PairformerBlock(dims.n_heads + 2, None, dims.n_heads)
