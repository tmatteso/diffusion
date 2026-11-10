"""Tests for the pair_update architecture module.

Covers TransformRBF values against the Gaussian it documents; the output,
gradient and shape contracts of the triangle attentions, Transition and
PairUpdate; which axis each triangle attention keeps independent and that
it reads its bias; the masks DropoutRowwise and DropoutColumnwise draw; and
PairUpdate's invariance to rigid motions of the centre-atom coordinates.

Inputs come from ``tests.patterns.sine_pattern`` and every module is
built under a fixed seed, so each test is reproducible on its own.
"""

import abc
import dataclasses
import enum
import math
from typing import cast, override

import pytest
import torch
from architecture.pair_update import (
    DropoutColumnwise,
    DropoutRowwise,
    PairUpdate,
    ParamsForRBF,
    TransformRBF,
    Transition,
    TriangleAttentionEndingNodeWithBias,
    TriangleAttentionStartingNodeWithBias,
)
from einops import einsum, rearrange, reduce
from helpers.useful_objects import manual_seed
from jaxtyping import Float, TypeCheckError
from tests.patterns import sine_pattern

type AnyFloat = Float[torch.Tensor, "..."]
type Pair = Float[torch.Tensor, "B N_res N_res c"]
type TriangleAttention = (
    TriangleAttentionStartingNodeWithBias | TriangleAttentionEndingNodeWithBias
)


@dataclasses.dataclass(frozen=True)
class PairDims:
    """Sizes shared by every module and input in this file.

    Attributes:
        batch: Batch size ``B``.
        n_res: Residues ``N_res``.
        c: Pair embedding width; divisible by ``n_heads``.
        n_heads: Attention heads.
    """

    batch: int
    n_res: int
    c: int
    n_heads: int


@dataclasses.dataclass(frozen=True)
class PairProbe(abc.ABC):
    """One module reduced to a call on the pair embedding ``z``.

    Each subclass holds a module and its fixed inputs as named fields, and
    ``run`` calls the module with ``z`` swapped in, so nothing is
    captured.
    """

    @abc.abstractmethod
    def run(self, pair: AnyFloat) -> AnyFloat:
        """Call the module on ``pair`` with every other input fixed.

        Args:
            pair: Pair embedding, or any tensor the module accepts.

        Returns:
            The module's output.
        """


@dataclasses.dataclass(frozen=True)
class TriangleProbe(PairProbe):
    """A triangle attention probed over ``z`` with a fixed bias.

    Attributes:
        module: Module under test, in eval mode.
        bias: Fixed attention bias.
    """

    module: TriangleAttention
    bias: Pair

    @override
    def run(self, pair: AnyFloat) -> AnyFloat:
        """Run ``module(pair, bias)``.

        Args:
            pair: Pair embedding.

        Returns:
            The attention output.
        """
        return self.module(pair, self.bias)


@dataclasses.dataclass(frozen=True)
class TransitionProbe(PairProbe):
    """The Transition block, which takes ``z`` alone.

    Attributes:
        module: Module under test, in eval mode.
    """

    module: Transition

    @override
    def run(self, pair: AnyFloat) -> AnyFloat:
        """Run ``module(pair)``.

        Args:
            pair: Pair or single embedding.

        Returns:
            The block's output.
        """
        return self.module(pair)


@dataclasses.dataclass(frozen=True)
class PairUpdateProbe(PairProbe):
    """PairUpdate probed over ``z`` with fixed centre-atom coordinates.

    Attributes:
        module: Module under test, in eval mode.
        r_center: Fixed centre-atom coordinates.
    """

    module: PairUpdate
    r_center: Float[torch.Tensor, "B N_res 3"]

    @override
    def run(self, pair: AnyFloat) -> AnyFloat:
        """Run ``module(pair, r_center)``.

        Args:
            pair: Pair embedding.

        Returns:
            The updated pair embedding.
        """
        return self.module(pair, self.r_center)


@dataclasses.dataclass(frozen=True)
class RbfCase:
    """A distance from the first RBF centre and that centre's feature.

    Attributes:
        offset_in_sigmas: Distance from the first centre, in sigmas.
        expected: Value of the first centre's feature.
        formula_bug: Whether the case hits the known precedence bug in
            ``TransformRBF``, so the ``rbf_scenario`` fixture marks it
            strict ``xfail``.
    """

    offset_in_sigmas: float
    expected: float
    formula_bug: bool


class RbfScenario(enum.Enum):
    """Distances at which the RBF feature has a known value.

    Members:
        at_centre: On the centre, the feature is 1.
        one_sigma: One ``sigma`` away, the feature is ``exp(-1/2)``. Fails
            today: ``TransformRBF`` computes
            ``exp(-(d - c)**2 / 2 * sigma**2)``, so operator precedence
            multiplies by ``sigma**2`` instead of dividing by
            ``2 * sigma**2``.
    """

    at_centre = RbfCase(0.0, 1.0, formula_bug=False)
    one_sigma = RbfCase(1.0, math.exp(-0.5), formula_bug=True)


@dataclasses.dataclass(frozen=True)
class TriangleCase:
    """A triangle attention direction and the axis it attends within.

    Attributes:
        module_cls: The attention class.
        axis: Axis of ``z`` whose index is held fixed while attending: 1
            (rows) for the starting node, 2 (columns) for the ending node.
            Output slices along this axis depend only on the matching
            input slice.
    """

    module_cls: type[TriangleAttention]
    axis: int


class TriangleScenario(enum.Enum):
    """The two triangle attention directions.

    Members:
        starting: ``TriangleAttentionStartingNodeWithBias``; row ``i``
            attends over its columns.
        ending: ``TriangleAttentionEndingNodeWithBias``; column ``j``
            attends over its rows.
    """

    starting = TriangleCase(TriangleAttentionStartingNodeWithBias, 1)
    ending = TriangleCase(TriangleAttentionEndingNodeWithBias, 2)


@dataclasses.dataclass(frozen=True)
class DropoutCase:
    """A structured dropout and the axis whose slices it drops whole.

    Attributes:
        module_cls: The dropout class.
        axis: Each ``(batch, index along axis)`` slice is kept or dropped
            as a unit.
    """

    module_cls: type[DropoutRowwise | DropoutColumnwise]
    axis: int


class DropoutScenario(enum.Enum):
    """The two structured dropouts.

    Members:
        rowwise: ``DropoutRowwise`` drops whole rows (axis 1).
        columnwise: ``DropoutColumnwise`` drops whole columns (axis 2).
    """

    rowwise = DropoutCase(DropoutRowwise, 1)
    columnwise = DropoutCase(DropoutColumnwise, 2)


@dataclasses.dataclass(frozen=True)
class RigidCase:
    """A rigid motion ``x -> x R + t`` applied to every coordinate.

    Integer rotation matrices keep the rotated coordinates exact.

    Attributes:
        rotation: Proper rotation matrix ``R``, row by row.
        translation: Offset ``t``.
    """

    rotation: tuple[tuple[int, int, int], ...]
    translation: tuple[float, float, float]


class RigidScenario(enum.Enum):
    """Rigid motions PairUpdate's output must not see.

    Members:
        translate: Pure shift by ``(3, -2, 5)``.
        rotate_z: Quarter turn about the z axis, no shift.
        screw: Third of a turn about ``(1, 1, 1)``, which cycles the axes,
            plus the shift.
    """

    translate = RigidCase(((1, 0, 0), (0, 1, 0), (0, 0, 1)), (3, -2, 5))
    rotate_z = RigidCase(((0, -1, 0), (1, 0, 0), (0, 0, 1)), (0, 0, 0))
    screw = RigidCase(((0, 0, 1), (1, 0, 0), (0, 1, 0)), (3, -2, 5))


@pytest.fixture
def dims() -> PairDims:
    """Return the small sizes every test builds its modules and inputs from.

    Returns:
        Sizes with ``c`` divisible by ``n_heads``.
    """
    return PairDims(batch=2, n_res=8, c=32, n_heads=4)


@pytest.fixture
def z(dims: PairDims) -> Pair:
    """Return a deterministic pair embedding.

    Args:
        dims: Tensor sizes.

    Returns:
        ``(B, N_res, N_res, c)`` tensor.
    """
    return sine_pattern((dims.batch, dims.n_res, dims.n_res, dims.c), 0.37)


@pytest.fixture
def bias(dims: PairDims) -> Pair:
    """Return a deterministic triangle attention bias, unrelated to ``z``.

    Args:
        dims: Tensor sizes.

    Returns:
        ``(B, N_res, N_res, c)`` tensor.
    """
    return sine_pattern((dims.batch, dims.n_res, dims.n_res, dims.c), 0.53)


@pytest.fixture
def r_center(
    dims: PairDims,
) -> Float[torch.Tensor, "B N_res 3"]:
    """Return centre-atom coordinates spread over tens of angstroms.

    Scaling the pattern by 10 puts pairwise distances inside the RBF
    centres' 3.25 to 50.75 angstrom range.

    Args:
        dims: Tensor sizes.

    Returns:
        ``(B, N_res, 3)`` coordinates.
    """
    return 10 * sine_pattern((dims.batch, dims.n_res, 3), 0.71)


@pytest.fixture
def pair_update(dims: PairDims) -> PairUpdate:
    """Build a PairUpdate without dropout under a fixed seed.

    Args:
        dims: Module sizes.

    Returns:
        The module in eval mode.
    """
    _ = manual_seed(0)
    return PairUpdate(dims.c, n_heads=dims.n_heads, dropout=0.0).eval()


@pytest.fixture
def triangle_scenario(request: pytest.FixtureRequest) -> TriangleScenario:
    """Return the ``TriangleScenario`` member routed in via indirect param.

    Args:
        request: Pytest request carrying the scenario in ``param``.

    Returns:
        The scenario under test.
    """
    return cast(TriangleScenario, request.param)


@pytest.fixture
def triangle(
    triangle_scenario: TriangleScenario,
    dims: PairDims,
) -> TriangleAttention:
    """Build the scenario's triangle attention under a fixed seed.

    Args:
        triangle_scenario: Direction to build.
        dims: Module sizes.

    Returns:
        The module in eval mode.
    """
    _ = manual_seed(0)
    module_cls = triangle_scenario.value.module_cls
    return module_cls(dims.c, n_heads=dims.n_heads).eval()


@pytest.fixture
def starting_probe(dims: PairDims, bias: Pair) -> TriangleProbe:
    """Probe the starting-node triangle attention with a fixed bias.

    Args:
        dims: Module sizes.
        bias: Attention bias.

    Returns:
        Probe over ``z``.
    """
    _ = manual_seed(0)
    module = TriangleAttentionStartingNodeWithBias(dims.c, dims.n_heads)
    return TriangleProbe(module=module.eval(), bias=bias)


@pytest.fixture
def ending_probe(dims: PairDims, bias: Pair) -> TriangleProbe:
    """Probe the ending-node triangle attention with a fixed bias.

    Args:
        dims: Module sizes.
        bias: Attention bias.

    Returns:
        Probe over ``z``.
    """
    _ = manual_seed(0)
    module = TriangleAttentionEndingNodeWithBias(dims.c, dims.n_heads)
    return TriangleProbe(module=module.eval(), bias=bias)


@pytest.fixture
def transition_probe(dims: PairDims) -> TransitionProbe:
    """Probe the Transition feed-forward block.

    Args:
        dims: Module sizes.

    Returns:
        Probe over ``z``.
    """
    _ = manual_seed(0)
    return TransitionProbe(module=Transition(dims.c).eval())


@pytest.fixture
def pair_update_probe(
    pair_update: PairUpdate,
    r_center: Float[torch.Tensor, "B N_res 3"],
) -> PairUpdateProbe:
    """Probe PairUpdate with fixed centre-atom coordinates.

    Args:
        pair_update: Module under test.
        r_center: Centre-atom coordinates.

    Returns:
        Probe over ``z``.
    """
    return PairUpdateProbe(module=pair_update, r_center=r_center)


@pytest.fixture
def flat_input(dims: PairDims) -> Float[torch.Tensor, "N_res c"]:
    """Return a 2-D ``(N_res, c)`` input with no pair layout.

    Args:
        dims: Tensor sizes.

    Returns:
        A sine pattern of shape ``(N_res, c)``.
    """
    return sine_pattern((dims.n_res, dims.c), 0.41)


@pytest.fixture
def rbf_scenario(request: pytest.FixtureRequest) -> RbfScenario:
    """Return the ``RbfScenario`` member, marking known failures ``xfail``.

    A case flagged ``formula_bug`` gets a strict ``xfail`` marker, so it
    must fail today and the marker must come off once ``TransformRBF``
    is fixed. Marks cannot ride on enum members, so the fixture applies
    the marker.

    Args:
        request: Pytest request carrying the scenario in ``param``.

    Returns:
        The scenario under test.
    """
    scenario = cast(RbfScenario, request.param)
    if scenario.value.formula_bug:
        request.applymarker(
            pytest.mark.xfail(
                strict=True,
                reason="TransformRBF precedence bug, see RbfScenario",
            ),
        )
    return scenario


@pytest.fixture
def pair_probe(request: pytest.FixtureRequest) -> PairProbe:
    """Return the probe fixture named by the indirect param.

    Args:
        request: Pytest request carrying a ``*_probe`` fixture name.

    Returns:
        The named probe.
    """
    return cast(PairProbe, request.getfixturevalue(cast(str, request.param)))


@pytest.fixture
def dropout_scenario(request: pytest.FixtureRequest) -> DropoutScenario:
    """Return the ``DropoutScenario`` member routed in via indirect param.

    Args:
        request: Pytest request carrying the scenario in ``param``.

    Returns:
        The scenario under test.
    """
    return cast(DropoutScenario, request.param)


@pytest.fixture
def rigid_scenario(request: pytest.FixtureRequest) -> RigidScenario:
    """Return the ``RigidScenario`` member routed in via indirect param.

    Args:
        request: Pytest request carrying the scenario in ``param``.

    Returns:
        The scenario under test.
    """
    return cast(RigidScenario, request.param)


@pytest.mark.parametrize(
    "rbf_scenario",
    list(RbfScenario),
    ids=[case.name for case in RbfScenario],
    indirect=True,
)
def test_transform_rbf_is_a_gaussian_of_width_sigma(
    rbf_scenario: RbfScenario,
) -> None:
    """At ``sigma`` from a centre, that centre's feature is ``exp(-1/2)``.

    The documented basis is ``exp(-(d - c)**2 / (2 * sigma**2))``, so a
    distance on a centre gives 1 and one ``sigma`` away gives
    ``exp(-0.5)``. The second case fails today; ``rbf_scenario`` marks it
    strict ``xfail`` so the marker must be removed once the formula is
    fixed.

    Args:
        rbf_scenario: Distance from the first centre and expected value.
    """
    case = rbf_scenario.value
    params = ParamsForRBF()
    distance = params.d_min + case.offset_in_sigmas * params.sigma
    out = TransformRBF(params)(torch.full((1, 1, 1), distance))
    assert out.shape == (1, 1, 1, params.n_rbf)
    torch.testing.assert_close(out[0, 0, 0, 0], torch.tensor(case.expected))


def test_transform_rbf_rejects_distances_missing_an_axis(
    dims: PairDims,
) -> None:
    """A ``(B, N_res)`` distance tensor fails the ``(B, N, N)`` contract.

    Not parametrized: ``forward`` takes a single tensor argument.

    Args:
        dims: Tensor sizes.
    """
    with pytest.raises(TypeCheckError):
        _ = TransformRBF(ParamsForRBF())(torch.zeros(dims.batch, dims.n_res))


@pytest.mark.parametrize(
    "pair_probe",
    ["starting_probe", "ending_probe", "transition_probe", "pair_update_probe"],
    indirect=True,
)
def test_output_shape_and_finite(z: Pair, pair_probe: PairProbe) -> None:
    """Each module maps the pair embedding to a finite tensor of its shape.

    Args:
        z: Valid pair embedding.
        pair_probe: Module call under test.
    """
    with torch.no_grad():
        out = pair_probe.run(z)
    assert out.shape == z.shape
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    "pair_probe",
    ["starting_probe", "ending_probe", "transition_probe", "pair_update_probe"],
    indirect=True,
)
def test_gradient_reaches_pair_embedding(
    z: Pair,
    pair_probe: PairProbe,
) -> None:
    """The summed output has a finite, nonzero gradient with respect to ``z``.

    Args:
        z: Valid pair embedding.
        pair_probe: Module call under test.
    """
    leaf = z.clone().requires_grad_()
    out = reduce(pair_probe.run(leaf), "b i j c -> ", "sum")
    (grad,) = torch.autograd.grad(out, leaf)
    assert torch.isfinite(grad).all()
    assert torch.count_nonzero(grad) > 0


@pytest.mark.parametrize(
    "pair_probe",
    ["starting_probe", "ending_probe", "pair_update_probe"],
    indirect=True,
)
def test_forward_rejects_pair_missing_channel_axis(
    z: Pair,
    pair_probe: PairProbe,
) -> None:
    """Each module's contract rejects a ``(B, N, N)`` pair tensor.

    Transition is left out: it takes any ``(..., c)`` tensor, which the
    next test relies on.

    Args:
        z: Valid pair embedding to cut the malformed one from.
        pair_probe: Module call under test.
    """
    malformed = z[..., 0]
    with pytest.raises(TypeCheckError):
        _ = pair_probe.run(malformed)


def test_transition_accepts_any_leading_shape(
    flat_input: Float[torch.Tensor, "N_res c"],
    transition_probe: TransitionProbe,
) -> None:
    """Transition maps an ``(N, c)`` tensor to the same shape.

    It is applied to both single and pair embeddings, so it must not assume
    the pair layout. Not parametrized: the pair layout is covered by the
    shape test above.

    Args:
        flat_input: Input without a pair layout.
        transition_probe: Transition under test.
    """
    with torch.no_grad():
        out = transition_probe.run(flat_input)
    assert out.shape == flat_input.shape


@pytest.mark.parametrize(
    "triangle_scenario",
    list(TriangleScenario),
    ids=[case.name for case in TriangleScenario],
    indirect=True,
)
def test_triangle_attention_keeps_its_axis_independent(
    triangle_scenario: TriangleScenario,
    triangle: TriangleAttention,
    z: Pair,
    bias: Pair,
) -> None:
    """Changing slice 0 of ``z`` along the fixed axis leaves the rest alone.

    The starting node's queries, keys, values and gates for row ``i`` all
    come from ``z[:, i]``, and the ending node's for column ``j`` from
    ``z[:, :, j]``, so editing that slice cannot reach any other.

    Args:
        triangle_scenario: Direction and fixed axis.
        triangle: Module under test.
        z: Valid pair embedding.
        bias: Attention bias, held fixed.
    """
    axis = triangle_scenario.value.axis
    edited = z.clone()
    _ = edited.select(axis, 0).neg_()
    with torch.no_grad():
        base = triangle(z, bias).narrow(axis, 1, z.shape[axis] - 1)
        changed = triangle(edited, bias).narrow(axis, 1, z.shape[axis] - 1)
    torch.testing.assert_close(base, changed, atol=1e-5, rtol=0)


@pytest.mark.parametrize(
    "triangle_scenario",
    list(TriangleScenario),
    ids=[case.name for case in TriangleScenario],
    indirect=True,
)
def test_triangle_attention_reads_its_bias(
    triangle: TriangleAttention,
    z: Pair,
    bias: Pair,
) -> None:
    """Negating the bias, ``z`` fixed, changes the attention output.

    Args:
        triangle: Module under test.
        z: Valid pair embedding.
        bias: Attention bias.
    """
    with torch.no_grad():
        base = triangle(z, bias)
        flipped = triangle(z, -bias)
    assert not torch.allclose(base, flipped)


@pytest.mark.parametrize(
    "dropout_scenario",
    list(DropoutScenario),
    ids=[case.name for case in DropoutScenario],
    indirect=True,
)
def test_dropout_is_identity_in_eval(
    dropout_scenario: DropoutScenario,
    z: Pair,
) -> None:
    """In eval mode the dropout returns its input unchanged.

    Args:
        dropout_scenario: Dropout under test.
        z: Valid pair embedding.
    """
    drop = dropout_scenario.value.module_cls(p=0.5).eval()
    assert torch.equal(drop(z), z)


@pytest.mark.parametrize(
    "dropout_scenario",
    list(DropoutScenario),
    ids=[case.name for case in DropoutScenario],
    indirect=True,
)
def test_dropout_drops_whole_slices_and_rescales(
    dropout_scenario: DropoutScenario,
    dims: PairDims,
) -> None:
    """In train mode each slice along the axis is all 0 or all ``1/(1-p)``.

    Feeding ones makes the drawn mask visible. With ``p = 0.5`` and a fixed
    seed the 16 slices include both kept and dropped ones, so neither
    "drop nothing" nor "drop everything" passes.

    Args:
        dropout_scenario: Dropout and the axis it drops along.
        dims: Tensor sizes.
    """
    case = dropout_scenario.value
    ones = torch.ones(dims.batch, dims.n_res, dims.n_res, dims.c)
    _ = manual_seed(0)
    p = 0.5
    out = case.module_cls(p=p).train()(ones)
    slices = rearrange(out.movedim(case.axis, 1), "b n ... -> b n (...)")
    lowest = reduce(slices, "b n k -> b n", "min")
    highest = reduce(slices, "b n k -> b n", "max")
    kept, dropped = lowest == 1 / (1 - p), lowest == 0
    assert torch.equal(lowest, highest)
    assert bool((kept | dropped).all())
    assert bool(kept.any())
    assert bool(dropped.any())


def test_pair_update_is_not_identity(
    pair_update_probe: PairUpdateProbe,
    z: Pair,
) -> None:
    """PairUpdate's residual updates are nonzero at initialisation.

    Not parametrized: a degenerate initialisation would zero the updates
    for every input.

    Args:
        pair_update_probe: PairUpdate under test.
        z: Valid pair embedding.
    """
    with torch.no_grad():
        out = pair_update_probe.run(z)
    assert not torch.allclose(out, z)


def test_pair_update_coordinate_gradient_is_finite(
    pair_update: PairUpdate,
    z: Pair,
    r_center: Float[torch.Tensor, "B N_res 3"],
) -> None:
    """The output's gradient with respect to ``r_center`` is finite.

    Every ``d_ii`` is 0, where the derivative of a bare ``sqrt`` is
    infinite; the clamp inside the distance keeps it finite. Not
    parametrized: the diagonal is present for any coordinates.

    Args:
        pair_update: Module under test.
        z: Valid pair embedding.
        r_center: Centre-atom coordinates.
    """
    leaf = r_center.clone().requires_grad_()
    out = reduce(pair_update(z, leaf), "b i j c -> ", "sum")
    (grad,) = torch.autograd.grad(out, leaf)
    assert torch.isfinite(grad).all()


@pytest.mark.parametrize(
    "rigid_scenario",
    list(RigidScenario),
    ids=[case.name for case in RigidScenario],
    indirect=True,
)
def test_pair_update_is_invariant_to_rigid_motion(
    rigid_scenario: RigidScenario,
    pair_update: PairUpdate,
    z: Pair,
    r_center: Float[torch.Tensor, "B N_res 3"],
) -> None:
    """Moving every centre atom rigidly leaves PairUpdate's output unchanged.

    Coordinates enter only through pairwise distances, which rigid motions
    preserve.

    Args:
        rigid_scenario: Rotation and translation to apply.
        pair_update: Module under test.
        z: Valid pair embedding.
        r_center: Centre-atom coordinates.
    """
    case = rigid_scenario.value
    rotation = torch.tensor(case.rotation, dtype=torch.float32)
    moved = einsum(r_center, rotation, "b n d, d e -> b n e") + torch.tensor(
        case.translation,
    )
    with torch.no_grad():
        base = pair_update(z, r_center)
        after = pair_update(z, moved)
    torch.testing.assert_close(base, after, atol=1e-5, rtol=0)
