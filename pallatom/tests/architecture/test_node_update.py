"""Tests for node update modules.

Covers AdaLN, AttentionPairBias and NodeUpdate: output shape, dtype and
finiteness on the dense and sparse paths; which inputs condition the
output; gradient flow to inputs and to every parameter; the ``beta`` bias
and ``neighbor_idx`` sparse path; dropout; recycling stability; and the
constructor and jaxtyping contracts.

Inputs are deterministic sine patterns and every module is built under a
fixed seed inside its fixture, so each test is reproducible on its own.
"""

import dataclasses
import enum
from collections.abc import Callable
from typing import cast

import pytest
import torch
from architecture.node_update import (
    AdaLN,
    AttentionPairBias,
    InvalidSingleHeadDimensionError,
    NodeUpdate,
)
from einops import rearrange, reduce, repeat
from helpers.useful_objects import manual_seed
from jaxtyping import Float, Int, TypeCheckError

type Single = Float[torch.Tensor, "B N_res c_res"]
type Pair = Float[torch.Tensor, "B N_res N_j c_pair"]
type ModuleCall = Callable[[Single, Single, Pair], Single]
type PatternFactory = Callable[
    [tuple[int, ...], float],
    Float[torch.Tensor, "..."],
]


@dataclasses.dataclass(frozen=True)
class NodeDims:
    """Sizes shared by every module and input in this file.

    Attributes:
        batch: Batch size ``B``.
        n_res: Residues ``N_res`` on the dense path.
        c_res: Single embedding width; divisible by ``n_heads``.
        c_pair: Pair embedding width.
        n_heads: Attention heads.
    """

    batch: int
    n_res: int
    c_res: int
    c_pair: int
    n_heads: int


@dataclasses.dataclass(frozen=True)
class NodeModules:
    """The three modules under test, built from the same ``NodeDims``.

    Attributes:
        adaln: ``AdaLN(c_res, c_res)``.
        attn: ``AttentionPairBias`` in eval mode.
        node: ``NodeUpdate`` with dropout 0, in eval mode.
    """

    adaln: AdaLN
    attn: AttentionPairBias
    node: NodeUpdate


@dataclasses.dataclass(frozen=True)
class NodeInputs:
    """Dense-path inputs; ``z`` is ``(B, N_res, N_res, c_pair)``.

    Attributes:
        s: Single embedding ``(B, N_res, c_res)``.
        t: Time conditioning ``(B, N_res, c_res)``.
        z: Pair embedding.
    """

    s: Single
    t: Single
    z: Pair


@dataclasses.dataclass(frozen=True)
class SparseInputs:
    """Sparse-path inputs where each query sees ``K < N_res`` neighbours.

    This is the layout that once crashed with ``attn (B, H, N, N) +
    pair_bias (B, H, N, K)``; ``K != N`` keeps that mismatch in play.

    Attributes:
        a: Node embedding ``(B, N_res, c_res)``.
        s: Conditioning ``(B, N_res, c_res)``.
        z: Pair embedding over neighbours ``(B, N_res, K, c_pair)``.
        neighbor_idx: Key index for each query slot ``(B, N_res, K)``.
    """

    a: Single
    s: Single
    z: Pair
    neighbor_idx: Int[torch.Tensor, "B N_res K"]


@dataclasses.dataclass(frozen=True)
class MalformedCall:
    """A module call whose first argument has lost its channel axis.

    Attributes:
        call: Invokes one module with the malformed argument.
    """

    call: Callable[[NodeModules, NodeInputs], object]


class MalformedScenario(enum.Enum):
    """Calls that break a ``forward`` shape contract.

    Members:
        adaln: ``AdaLN(a, s)`` with ``a`` as ``(B, N_res)``.
        attention: ``AttentionPairBias(a, None, z)`` with ``a`` 2-D.
        node_update: ``NodeUpdate(s, t, z)`` with ``s`` 2-D.
    """

    adaln = MalformedCall(lambda m, x: m.adaln(x.s[..., 0], x.t))
    attention = MalformedCall(lambda m, x: m.attn(x.s[..., 0], None, x.z))
    node_update = MalformedCall(lambda m, x: m.node(x.s[..., 0], x.t, x.z))


@pytest.fixture
def dims() -> NodeDims:
    """Return the small sizes every dense-path test uses.

    Returns:
        Sizes with ``c_res`` divisible by ``n_heads``.
    """
    return NodeDims(batch=2, n_res=8, c_res=32, c_pair=32, n_heads=4)


@pytest.fixture
def modules(dims: NodeDims) -> NodeModules:
    """Build AdaLN, AttentionPairBias and NodeUpdate under a fixed seed.

    Args:
        dims: Module sizes.

    Returns:
        The three modules, attention and node update in eval mode.
    """
    _ = manual_seed(0)
    return NodeModules(
        adaln=AdaLN(c_a=dims.c_res, c_s=dims.c_res),
        attn=AttentionPairBias(dims.c_res, dims.c_pair, dims.n_heads).eval(),
        node=NodeUpdate(
            dims.c_res,
            dims.c_pair,
            n_heads=dims.n_heads,
            dropout=0.0,
        ).eval(),
    )


@pytest.fixture
def dropout_node(dims: NodeDims) -> NodeUpdate:
    """Build a NodeUpdate with dropout 0.5 under a fixed seed.

    Args:
        dims: Module sizes.

    Returns:
        The module, in eval mode.
    """
    _ = manual_seed(0)
    return NodeUpdate(dims.c_res, dims.c_pair, dims.n_heads, 0.5).eval()


@pytest.fixture
def make_pattern() -> PatternFactory:
    """Return a factory for deterministic sine-pattern tensors.

    ``make(shape, freq)`` fills ``shape`` with ``sin(freq * k)`` over the
    flat element index ``k``. Distinct frequencies give inputs that are
    unrelated to each other, with values in ``[-1, 1]``.

    Returns:
        The factory.
    """

    def make(shape: tuple[int, ...], freq: float) -> Float[torch.Tensor, "..."]:
        """Build one sine-pattern tensor.

        Args:
            shape: Output shape.
            freq: Angular step between consecutive elements.

        Returns:
            Float32 tensor of ``shape``.
        """
        names = [f"d{axis}" for axis in range(len(shape))]
        axes = " ".join(names)
        flat = torch.sin(freq * torch.arange(torch.Size(shape).numel()))
        return rearrange(
            flat.float(),
            f"({axes}) -> {axes}",
            **dict(zip(names, shape, strict=True)),
        )

    return make


@pytest.fixture
def inputs(
    dims: NodeDims,
    make_pattern: PatternFactory,
) -> NodeInputs:
    """Build dense-path inputs from three unrelated sine patterns.

    Args:
        dims: Input sizes.
        make_pattern: Sine-pattern factory.

    Returns:
        ``s``, ``t`` and ``z`` for ``N_j = N_res``.
    """
    b, n = dims.batch, dims.n_res
    return NodeInputs(
        s=make_pattern((b, n, dims.c_res), 0.37),
        t=make_pattern((b, n, dims.c_res), 0.53),
        z=make_pattern((b, n, n, dims.c_pair), 0.71),
    )


@pytest.fixture
def sparse_inputs(
    dims: NodeDims,
    make_pattern: PatternFactory,
) -> SparseInputs:
    """Build sparse-path inputs with 20 nodes and 6 neighbours each.

    Query ``i`` attends to keys ``i, i+1, ..., i+5`` modulo 20.

    Args:
        dims: Channel sizes and batch.
        make_pattern: Sine-pattern factory.

    Returns:
        Inputs for ``AttentionPairBias`` with ``neighbor_idx``.
    """
    b, n, k = dims.batch, 20, 6
    window = (
        rearrange(torch.arange(n), "n -> n 1")
        + rearrange(torch.arange(k), "k -> 1 k")
    ) % n
    return SparseInputs(
        a=make_pattern((b, n, dims.c_res), 0.37),
        s=make_pattern((b, n, dims.c_res), 0.53),
        z=make_pattern((b, n, k, dims.c_pair), 0.71),
        neighbor_idx=repeat(window, "n k -> b n k", b=b),
    )


@pytest.fixture
def module_call(
    request: pytest.FixtureRequest,
    modules: NodeModules,
) -> ModuleCall:
    """Return the ``(s, t, z)`` module named by the indirect param.

    Args:
        request: Pytest request carrying ``"attn"`` or ``"node"``.
        modules: Modules to pick from.

    Returns:
        The chosen module as a callable.
    """
    return cast(ModuleCall, getattr(modules, cast(str, request.param)))


@pytest.fixture
def malformed_scenario(request: pytest.FixtureRequest) -> MalformedScenario:
    """Return the ``MalformedScenario`` member routed in via indirect param.

    Args:
        request: Pytest request carrying the scenario in ``param``.

    Returns:
        The scenario under test.
    """
    return cast(MalformedScenario, request.param)


@pytest.mark.parametrize(
    "module_call",
    ["attn", "node"],
    ids=["attention_pair_bias", "node_update"],
    indirect=True,
)
def test_dense_output_shape_dtype_and_finite(
    dims: NodeDims,
    inputs: NodeInputs,
    module_call: ModuleCall,
) -> None:
    """Each module returns a finite float32 ``(B, N_res, c_res)`` tensor.

    Args:
        dims: Expected sizes.
        inputs: Dense inputs.
        module_call: Module under test.
    """
    with torch.no_grad():
        out = module_call(inputs.s, inputs.t, inputs.z)
    assert out.shape == (dims.batch, dims.n_res, dims.c_res)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("with_beta", [False, True], ids=["no_beta", "beta"])
def test_sparse_output_shape_and_finite(
    dims: NodeDims,
    modules: NodeModules,
    sparse_inputs: SparseInputs,
    with_beta: bool,  # noqa: FBT001
) -> None:
    """Sparse attention with ``K < N_res`` returns finite ``(B, N, c_res)``.

    Covers both the bare sparse path and the path that also adds a
    ``beta`` bias aligned with the neighbour axis.

    Args:
        dims: Channel sizes.
        modules: Holds the attention module.
        sparse_inputs: Inputs with 6 of 20 neighbours per query.
        with_beta: Whether to pass a zero ``beta`` of shape ``(B, N, K)``.
    """
    x = sparse_inputs
    beta = torch.zeros(x.neighbor_idx.shape) if with_beta else None
    with torch.no_grad():
        out = modules.attn(x.a, x.s, x.z, beta, x.neighbor_idx)
    assert out.shape == (dims.batch, x.a.shape[1], dims.c_res)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("field", ["t", "z"], ids=["time", "pair"])
def test_attention_output_depends_on_conditioning(
    modules: NodeModules,
    inputs: NodeInputs,
    field: str,
) -> None:
    """Negating ``t`` or ``z``, all else fixed, changes the attention output.

    ``s`` is left out: ``test_node_update_is_not_identity`` and the gradient
    tests already show the output depends on it.

    Args:
        modules: Holds the attention module.
        inputs: Dense inputs.
        field: Input to negate.
    """
    old = cast(Float[torch.Tensor, "..."], getattr(inputs, field))
    edited = dataclasses.replace(inputs, **{field: -old})
    with torch.no_grad():
        base = modules.attn(inputs.s, inputs.t, inputs.z)
        changed = modules.attn(edited.s, edited.t, edited.z)
    assert not torch.allclose(base, changed)


def test_node_update_is_not_identity(
    modules: NodeModules,
    inputs: NodeInputs,
) -> None:
    """NodeUpdate's residual update is nonzero at initialisation.

    Not parametrized: a degenerate initialisation would zero the update for
    every input, so one input exposes it.

    Args:
        modules: Holds the node update.
        inputs: Dense inputs.
    """
    with torch.no_grad():
        out = modules.node(inputs.s, inputs.t, inputs.z)
    assert not torch.allclose(out, inputs.s)


@pytest.mark.parametrize("field", ["s", "t", "z"])
def test_node_update_gradient_reaches_input(
    modules: NodeModules,
    inputs: NodeInputs,
    field: str,
) -> None:
    """The summed NodeUpdate output has a finite, nonzero gradient per input.

    Args:
        modules: Holds the node update.
        inputs: Dense inputs.
        field: Input to differentiate with respect to.
    """
    leaf = cast(torch.Tensor, getattr(inputs, field)).clone().requires_grad_()
    x = dataclasses.replace(inputs, **{field: leaf})
    out = reduce(modules.node(x.s, x.t, x.z), "b n c -> ", "sum")
    (grad,) = torch.autograd.grad(out, leaf)
    assert torch.isfinite(grad).all()
    assert torch.count_nonzero(grad) > 0


def test_sparse_gradient_reaches_node_embedding(
    modules: NodeModules,
    sparse_inputs: SparseInputs,
) -> None:
    """Gradients flow through the sparse gather back to ``a``.

    Not parametrized: the gather on ``neighbor_idx`` is the only part of
    the sparse path that differs from the dense one the input-gradient
    test already covers.

    Args:
        modules: Holds the attention module.
        sparse_inputs: Inputs with 6 of 20 neighbours per query.
    """
    x = sparse_inputs
    a = x.a.clone().requires_grad_()
    out = modules.attn(a, x.s, x.z, neighbor_idx=x.neighbor_idx)
    (grad,) = torch.autograd.grad(reduce(out, "b n c -> ", "sum"), a)
    assert torch.isfinite(grad).all()
    assert torch.count_nonzero(grad) > 0


@pytest.mark.parametrize(
    "module_call",
    ["attn", "node"],
    ids=["attention_pair_bias", "node_update"],
    indirect=True,
)
def test_every_parameter_receives_a_finite_gradient(
    inputs: NodeInputs,
    module_call: ModuleCall,
) -> None:
    """No parameter is skipped by the forward pass when ``s`` is provided.

    Regression test for the DDP unused-parameter crash: DDP fails when any
    parameter gets no gradient.

    Args:
        inputs: Dense inputs.
        module_call: Module under test.
    """
    module = cast(torch.nn.Module, module_call)
    out = reduce(module_call(inputs.s, inputs.t, inputs.z), "... -> ", "sum")
    torch.autograd.backward([out])
    missing = [
        name
        for name, param in module.named_parameters()
        if param.grad is None or not torch.isfinite(param.grad).all()
    ]
    assert not missing


def test_attention_norm_a_has_no_learnable_parameters(
    modules: NodeModules,
) -> None:
    """``norm_a`` is non-affine, so it holds no parameters for DDP to miss.

    Every caller passes a real ``s``, so the ``s=None`` path through
    ``norm_a`` never runs in training. Learnable parameters there would
    never receive gradients, and DDP would crash on them. Not parametrized:
    this is a fixed structural property of the module.

    Args:
        modules: Holds the attention module.
    """
    assert not list(modules.attn.norm_a.parameters())


def test_node_update_dropout_is_active_only_in_train(
    dropout_node: NodeUpdate,
    inputs: NodeInputs,
) -> None:
    """With dropout 0.5, eval output is repeatable and train output differs.

    Eval mode must turn ``DropoutRowwise`` into the identity; train mode
    must actually drop rows, while keeping the output finite.

    Args:
        dropout_node: NodeUpdate with dropout 0.5, in eval mode.
        inputs: Dense inputs.
    """
    x = inputs
    with torch.no_grad():
        first = dropout_node(x.s, x.t, x.z)
        second = dropout_node(x.s, x.t, x.z)
        trained = dropout_node.train()(x.s, x.t, x.z)
    assert torch.equal(first, second)
    assert torch.isfinite(trained).all()
    assert not torch.allclose(first, trained)


def test_attention_beta_fully_masked_stays_finite(
    modules: NodeModules,
    inputs: NodeInputs,
) -> None:
    """A ``beta`` of -1e10 on every key does not produce NaN.

    Guards against ``softmax([-inf, ...]) = NaN`` when -1e10 overflows to
    -inf under float16 autocast. Not parametrized: masking every key is the
    extreme case; any partial mask is less severe.

    Args:
        modules: Holds the attention module.
        inputs: Dense inputs.
    """
    beta = torch.full(inputs.z.shape[:-1], -1e10)
    with torch.no_grad():
        out = modules.attn(inputs.s, inputs.t, inputs.z, beta=beta)
    assert torch.isfinite(out).all()


def test_attention_beta_single_open_key_changes_output(
    modules: NodeModules,
    inputs: NodeInputs,
) -> None:
    """Masking every key but key 0 changes the output from an open ``beta``.

    Not parametrized: which key is left open does not matter, only that
    ``beta`` reaches the softmax.

    Args:
        modules: Holds the attention module.
        inputs: Dense inputs.
    """
    open_beta = torch.zeros(inputs.z.shape[:-1])
    one_key = torch.full_like(open_beta, -1e10)
    one_key[..., 0] = 0.0
    with torch.no_grad():
        out_open = modules.attn(inputs.s, inputs.t, inputs.z, beta=open_beta)
        out_one = modules.attn(inputs.s, inputs.t, inputs.z, beta=one_key)
    assert not torch.allclose(out_open, out_one)


@pytest.mark.parametrize("shift", [-3.0, 5.0], ids=["negative", "positive"])
def test_attention_beta_constant_shift_is_invariant(
    modules: NodeModules,
    inputs: NodeInputs,
    make_pattern: PatternFactory,
    shift: float,
) -> None:
    """Adding a constant to every ``beta`` entry leaves the output unchanged.

    Softmax is shift-invariant: ``exp(x+c)/sum(exp(x+c)) = exp(x)/sum(exp(x))``.

    Args:
        modules: Holds the attention module.
        inputs: Dense inputs.
        make_pattern: Builds a non-uniform base ``beta``.
        shift: Constant added to ``beta``.
    """
    beta = make_pattern(inputs.z.shape[:-1], 0.29)
    x = inputs
    with torch.no_grad():
        base = modules.attn(x.s, x.t, x.z, beta=beta)
        shifted = modules.attn(x.s, x.t, x.z, beta=beta + shift)
    torch.testing.assert_close(base, shifted, atol=1e-5, rtol=0)


def test_attention_sparse_with_all_keys_matches_dense(
    modules: NodeModules,
    inputs: NodeInputs,
) -> None:
    """``neighbor_idx`` listing every key in order equals the dense path.

    Not parametrized: the identity neighbour list is the one layout for
    which the sparse and dense paths must agree exactly.

    Args:
        modules: Holds the attention module.
        inputs: Dense inputs.
    """
    b, n = inputs.s.shape[:2]
    every_key = repeat(torch.arange(n), "k -> b n k", b=b, n=n)
    x = inputs
    with torch.no_grad():
        dense = modules.attn(x.s, x.t, x.z)
        sparse = modules.attn(x.s, x.t, x.z, neighbor_idx=every_key)
    torch.testing.assert_close(dense, sparse, atol=1e-5, rtol=0)


def test_attention_zero_pair_projection_ignores_z(
    modules: NodeModules,
    inputs: NodeInputs,
) -> None:
    """With ``z_to_b`` zeroed, scaling ``z`` tenfold leaves the output alone.

    Shows the pair channel reaches the output only through ``z_to_b``. Not
    parametrized: zero weights make every ``z`` equivalent, so one change
    to ``z`` suffices.

    Args:
        modules: Holds the attention module; its weights are zeroed here.
        inputs: Dense inputs.
    """
    _ = torch.nn.init.zeros_(modules.attn.z_to_b.weight)
    x = inputs
    with torch.no_grad():
        base = modules.attn(x.s, x.t, x.z)
        scaled = modules.attn(x.s, x.t, x.z * 10.0)
    torch.testing.assert_close(base, scaled, atol=1e-6, rtol=0)


def test_attention_residual_update_is_contractive(
    modules: NodeModules,
    inputs: NodeInputs,
) -> None:
    """At init the attention update is smaller in norm than ``s`` itself.

    NodeUpdate applies ``s = s + attn(s, t, z)``; a gain of 1 or more at
    init predicts norm blow-up across the decoder's recycling blocks. Not
    parametrized: the gain is a property of the initialisation.

    Args:
        modules: Holds the attention module.
        inputs: Dense inputs.
    """
    with torch.no_grad():
        delta = modules.attn(inputs.s, inputs.t, inputs.z)
    delta_norm = torch.sqrt(reduce(delta**2, "... -> ", "sum"))
    s_norm = torch.sqrt(reduce(inputs.s**2, "... -> ", "sum"))
    assert delta_norm < s_norm


def test_attention_recycling_stays_bounded_with_finite_gradients(
    modules: NodeModules,
    inputs: NodeInputs,
) -> None:
    """Twenty rounds of ``s = s + attn(s, t, z)`` stay bounded and trainable.

    Simulates the decoder recycling loop. RMS may grow, but by less than 5x;
    an expansive operator grows exponentially. Backpropagating through all
    twenty rounds must leave every parameter gradient finite. Not
    parametrized: twenty rounds is well past the decoder's eight blocks.

    Args:
        modules: Holds the attention module.
        inputs: Dense inputs.
    """
    s = inputs.s
    for _ in range(20):
        s = s + modules.attn(s, inputs.t, inputs.z)
    rms_before = torch.sqrt(reduce(inputs.s**2, "... -> ", "mean"))
    rms_after = torch.sqrt(reduce(s**2, "... -> ", "mean"))
    torch.autograd.backward([reduce(s**2, "... -> ", "mean")])
    assert rms_after < 5 * rms_before
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in modules.attn.parameters()
    )


def test_attention_rejects_heads_that_do_not_divide_width(
    dims: NodeDims,
) -> None:
    """``c_res`` not divisible by ``n_heads`` raises at construction.

    Not parametrized: the check is a single modulus, so one indivisible
    pair covers it.

    Args:
        dims: Supplies a valid ``c_pair`` and ``n_heads``.
    """
    with pytest.raises(InvalidSingleHeadDimensionError, match="divisible"):
        _ = AttentionPairBias(dims.n_heads + 1, dims.c_pair, dims.n_heads)


@pytest.mark.parametrize(
    "malformed_scenario",
    list(MalformedScenario),
    ids=[case.name for case in MalformedScenario],
    indirect=True,
)
def test_forward_rejects_input_missing_channel_axis(
    malformed_scenario: MalformedScenario,
    modules: NodeModules,
    inputs: NodeInputs,
) -> None:
    """Each module's jaxtyping contract rejects a 2-D first argument.

    Args:
        malformed_scenario: Call with a malformed argument.
        modules: Modules to call.
        inputs: Dense inputs the malformed argument is cut from.
    """
    with pytest.raises(TypeCheckError):
        _ = malformed_scenario.value.call(modules, inputs)
