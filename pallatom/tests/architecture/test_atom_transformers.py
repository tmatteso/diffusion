"""Tests for atom-level transformer layers.

Covers ``build_sparse_pairs`` against the window definition, ``compute_beta``
admission rules and fill values, ``scatter_mean`` on hand-computed cases,
the block isolation of AtomTransformer's sliding windows, and the output,
gradient and shape contracts of ConditionedTransitionBlock,
DiffusionTransformer, AtomTransformer, AtomFeatureEncoder and
AtomAttentionDecoder, including the sparse ``K < N`` layout that once
crashed with a pair-bias shape mismatch.

Float inputs come from ``tests.patterns.sine_pattern`` and every module
is built under a fixed seed, so each test is reproducible on its own.
"""

import abc
import dataclasses
import enum
from typing import cast, override

import pytest
import torch
import torch.nn.functional as F
from architecture.atom_transformers import (
    AtomAttentionDecoder,
    AtomFeatureEncoder,
    AtomTransformer,
    ConditionedTransitionBlock,
    DiffusionTransformer,
    build_sparse_pairs,
    compute_beta,
    scatter_mean,
)
from einops import rearrange, reduce, repeat
from helpers.useful_objects import manual_seed
from jaxtyping import Bool, Float, Int, TypeCheckError
from tests.patterns import sine_pattern
from train.train_config import ModelParams


@dataclasses.dataclass(frozen=True)
class AtomDims:
    """Sizes shared by every module and input in this file.

    Attributes:
        batch: Batch size ``B``.
        n_res: Residues ``N_res``.
        atoms_per_res: Atoms per residue.
        n_elements: Width ``E`` of the element one-hot.
        c_atom: Atom single embedding width.
        c_atompair: Atom pair embedding width.
        c_token: Residue single embedding width.
        c_pair: Residue pair embedding width.
        n_heads: Attention heads.
    """

    batch: int
    n_res: int
    atoms_per_res: int
    n_elements: int
    c_atom: int
    c_atompair: int
    c_token: int
    c_pair: int
    n_heads: int

    @property
    def n_atom(self) -> int:
        """Total atoms ``N_atom = n_res * atoms_per_res``.

        Returns:
            Atom count.
        """
        return self.n_res * self.atoms_per_res


@dataclasses.dataclass(frozen=True)
class SparsePairs:
    """Batched output of ``build_sparse_pairs``.

    Attributes:
        neighbor_idx: Neighbour atom index per slot ``(B, N_atom, K)``.
        valid_mask: True where the slot is a real neighbour.
    """

    neighbor_idx: Int[torch.Tensor, "B N_atom K"]
    valid_mask: Bool[torch.Tensor, "B N_atom K"]


@dataclasses.dataclass(frozen=True)
class AtomInputs:
    """Every encoder, decoder and atom-transformer input, from one layout.

    Attributes:
        tok_idx: Residue index per atom ``(B, N_atom)``.
        pairs: Sparse neighbour layout for ``tok_idx``.
        q: Atom queries ``(B, N_atom, c_atom)``.
        c: Atom context ``(B, N_atom, c_atom)``.
        p: Sparse atom pairs ``(B, N_atom, K, c_atompair)``.
        ref_pos: Reference positions ``(B, N_atom, 3)``.
        ref_element: Element one-hot ``(B, N_atom, E)``.
        ref_space_uid: Chain id per atom ``(B, N_atom)``; all zero.
        r_scaled: Scaled noisy positions ``(B, N_atom, 3)``.
        s_input: Trunk single embedding ``(B, N_res, c_token)``.
        z_input: Trunk pair embedding ``(B, N_res, N_res, c_pair)``.
    """

    tok_idx: Int[torch.Tensor, "B N_atom"]
    pairs: SparsePairs
    q: Float[torch.Tensor, "B N_atom c_atom"]
    c: Float[torch.Tensor, "B N_atom c_atom"]
    p: Float[torch.Tensor, "B N_atom K c_atompair"]
    ref_pos: Float[torch.Tensor, "B N_atom 3"]
    ref_element: Float[torch.Tensor, "B N_atom E"]
    ref_space_uid: Int[torch.Tensor, "B N_atom"]
    r_scaled: Float[torch.Tensor, "B N_atom 3"]
    s_input: Float[torch.Tensor, "B N_res c_token"]
    z_input: Float[torch.Tensor, "B N_res N_res c_pair"]


@dataclasses.dataclass(frozen=True)
class NeighbourSets:
    """Expected neighbours of every atom in a layout.

    Attributes:
        per_atom: Neighbour atom indices of atom ``l`` at position ``l``.
    """

    per_atom: tuple[frozenset[int], ...]


@dataclasses.dataclass(frozen=True)
class ModuleProbe(abc.ABC):
    """One module call reduced to a single differentiable input and output.

    Each subclass holds a module and every input it needs except the one
    under test, and ``run`` calls the module with that one input swapped
    in. Every fixed input is a named field, so nothing is captured.

    Attributes:
        first: The input the gradient and malformed-shape tests act on.
        out_shape: Shape the probed output must have.
    """

    first: Float[torch.Tensor, "..."]
    out_shape: tuple[int, ...]

    @abc.abstractmethod
    def run(
        self,
        first: Float[torch.Tensor, "..."],
    ) -> Float[torch.Tensor, "..."]:
        """Call the module with ``first`` in place of the probed input.

        Args:
            first: Value for the probed input.

        Returns:
            The probed output.
        """


@dataclasses.dataclass(frozen=True)
class TransitionProbe(ModuleProbe):
    """ConditionedTransitionBlock probed over its atom embedding ``a``.

    Attributes:
        block: Module under test.
        s: Fixed conditioning embedding.
    """

    block: ConditionedTransitionBlock
    s: Float[torch.Tensor, "..."]

    @override
    def run(
        self,
        first: Float[torch.Tensor, "..."],
    ) -> Float[torch.Tensor, "..."]:
        """Run ``block(first, s)``.

        Args:
            first: Atom embedding ``a``.

        Returns:
            The block's output.
        """
        return self.block(first, self.s)


@dataclasses.dataclass(frozen=True)
class DiffusionProbe(ModuleProbe):
    """DiffusionTransformer probed over its atom embedding ``a``.

    Attributes:
        model: Module under test.
        s: Fixed conditioning embedding.
        z: Fixed pair embedding, dense or sparse.
        beta: Fixed attention bias aligned with ``z``.
        neighbor_idx: Neighbour index for the sparse layout, or ``None``
            for dense pairs.
    """

    model: DiffusionTransformer
    s: Float[torch.Tensor, "..."]
    z: Float[torch.Tensor, "..."]
    beta: Float[torch.Tensor, "..."]
    neighbor_idx: Int[torch.Tensor, "B N K"] | None

    @override
    def run(
        self,
        first: Float[torch.Tensor, "..."],
    ) -> Float[torch.Tensor, "..."]:
        """Run the transformer with ``first`` as ``a``.

        Args:
            first: Atom embedding ``a``.

        Returns:
            The transformer's output.
        """
        return self.model(
            first,
            self.s,
            self.z,
            self.beta,
            neighbor_idx=self.neighbor_idx,
        )


@dataclasses.dataclass(frozen=True)
class AtomTransformerProbe(ModuleProbe):
    """AtomTransformer probed over its queries ``q``.

    Attributes:
        model: Module under test.
        inputs: Fixed context, pairs and neighbour layout.
    """

    model: AtomTransformer
    inputs: AtomInputs

    @override
    def run(
        self,
        first: Float[torch.Tensor, "..."],
    ) -> Float[torch.Tensor, "..."]:
        """Run the transformer with ``first`` as ``q``.

        Args:
            first: Atom queries ``q``.

        Returns:
            The transformer's output.
        """
        x = self.inputs
        return self.model(
            first,
            x.c,
            x.p,
            x.pairs.neighbor_idx,
            x.pairs.valid_mask,
        )


@dataclasses.dataclass(frozen=True)
class EncoderProbe(ModuleProbe):
    """AtomFeatureEncoder probed from ``r_scaled`` to ``q_skip``.

    Attributes:
        encoder: Module under test.
        inputs: Every other encoder input.
    """

    encoder: AtomFeatureEncoder
    inputs: AtomInputs

    @override
    def run(
        self,
        first: Float[torch.Tensor, "..."],
    ) -> Float[torch.Tensor, "..."]:
        """Run the encoder with ``first`` as ``r_scaled``.

        Args:
            first: Scaled noisy positions.

        Returns:
            The encoder's ``q_skip`` output.
        """
        x = self.inputs
        _, q_skip, *_ = self.encoder(
            x.ref_pos,
            x.ref_element,
            x.ref_space_uid,
            x.s_input,
            x.z_input,
            first,
            x.tok_idx,
        )
        return q_skip


@dataclasses.dataclass(frozen=True)
class DecoderProbe(ModuleProbe):
    """AtomAttentionDecoder probed from ``q_skip`` to its position update.

    Attributes:
        decoder: Module under test.
        inputs: Atom-level inputs; ``q``, ``p`` and ``c`` stand in for the
            encoder's skip outputs.
    """

    decoder: AtomAttentionDecoder
    inputs: AtomInputs

    @override
    def run(
        self,
        first: Float[torch.Tensor, "..."],
    ) -> Float[torch.Tensor, "..."]:
        """Run the decoder with ``first`` as ``q_skip``.

        Args:
            first: Atom skip queries.

        Returns:
            The decoder's ``r_update`` output.
        """
        x = self.inputs
        _, _, r_update, _ = self.decoder(
            first,
            x.p,
            x.c,
            x.c,
            x.s_input,
            x.z_input,
            x.tok_idx,
        )
        return r_update


@dataclasses.dataclass(frozen=True)
class LayoutCase:
    """A residue layout for ``build_sparse_pairs`` and its neighbour count.

    Attributes:
        n_res: Residues in the chain.
        atoms_per_res: Atoms per residue.
        window_size: Full window span in residues.
        expected_k: Largest neighbour count of any atom.
    """

    n_res: int
    atoms_per_res: int
    window_size: int
    expected_k: int


class LayoutScenario(enum.Enum):
    """Layouts with and without atoms outside each other's window.

    Members:
        one_window: 8 residues fit in a 128 window, so all 24 atoms are
            neighbours of each other.
        chain_window_8: 64 one-atom residues, half window 4: up to 7
            neighbours, fewer at the chain ends.
        pairs_window_4: 20 two-atom residues, half window 2: 3 residues of
            2 atoms each, so ``K = 6 < N = 40``.
    """

    one_window = LayoutCase(8, 3, 128, 24)
    chain_window_8 = LayoutCase(64, 1, 8, 7)
    pairs_window_4 = LayoutCase(20, 2, 4, 6)


@dataclasses.dataclass(frozen=True)
class BetaPointCase:
    """One atom pair on an 8-atom dense layout and whether it is admitted.

    Window centres sit at ``c * n_queries + n_queries / 2 - 0.5``. A pair
    ``(l, m)`` is admitted when some centre has ``l`` within
    ``n_queries / 2`` and ``m`` within ``n_keys / 2`` of it, strictly.

    Attributes:
        n_queries: Query window width.
        n_keys: Key window width.
        query: Atom ``l``.
        key: Atom ``m``.
        expected_beta: 0.0 when the pair is admitted, else the float32
            fill value -1e10.
    """

    n_queries: int
    n_keys: int
    query: int
    key: int
    expected_beta: float


class BetaPointScenario(enum.Enum):
    """Hand-derived admission decisions for ``compute_beta``.

    Members:
        shared_centre: q=4, k=8. Centre 1.5 has ``|0-1.5| < 2`` and
            ``|1-1.5| < 4``.
        disjoint_windows: q=k=4. Atom 7 is no key of centre 1.5 and atom 0
            no query of centre 5.5.
        boundary_is_strict: q=k=4. Atom 3 queries centre 1.5 but
            ``|4-1.5| = 2.5`` is not below 2.
        boundary_self: q=k=4. Centre 1.5 admits ``(3, 3)``.
        other_centre_admits: q=4, k=8. Atom 4 queries centre 5.5, which has
            ``|3-5.5| = 2.5 < 4``.
        forward_admitted: q=2, k=8. Atom 0 queries centre 0.5 and
            ``|4-0.5| = 3.5 < 4``.
        backward_blocked: q=2, k=8. Atom 4 queries centre 4.5 but
            ``|0-4.5| = 4.5`` is not below 4: beta is not symmetric.
    """

    shared_centre = BetaPointCase(4, 8, 0, 1, 0.0)
    disjoint_windows = BetaPointCase(4, 4, 0, 7, -1e10)
    boundary_is_strict = BetaPointCase(4, 4, 3, 4, -1e10)
    boundary_self = BetaPointCase(4, 4, 3, 3, 0.0)
    other_centre_admits = BetaPointCase(4, 8, 4, 3, 0.0)
    forward_admitted = BetaPointCase(2, 8, 0, 4, 0.0)
    backward_blocked = BetaPointCase(2, 8, 4, 0, -1e10)


@dataclasses.dataclass(frozen=True)
class ScatterCase:
    """A ``scatter_mean`` input and its hand-computed mean, one channel.

    Attributes:
        src: Values per batch item and source position.
        index: Flat segment per position, already offset by ``b * N_tgt``.
        expected: Mean per batch item and target segment.
    """

    src: tuple[tuple[float, ...], ...]
    index: tuple[tuple[int, ...], ...]
    expected: tuple[tuple[float, ...], ...]


class ScatterScenario(enum.Enum):
    """Segment layouts for ``scatter_mean``.

    Members:
        uniform_pairs: Two atoms per segment: ``mean(0, 2) = 1``,
            ``mean(4, 6) = 5``.
        uneven: Three then two atoms: ``mean(1, 3, 5) = 3``,
            ``mean(7, 9) = 8``.
        one_per_segment: Each atom its own segment, so values pass through.
        batch_offset: Batch 1's segment index is offset by ``N_tgt = 1``;
            each batch item averages only its own atoms.
    """

    uniform_pairs = ScatterCase(((0, 2, 4, 6),), ((0, 0, 1, 1),), ((1, 5),))
    uneven = ScatterCase(((1, 3, 5, 7, 9),), ((0, 0, 0, 1, 1),), ((3, 8),))
    one_per_segment = ScatterCase(((4, -2, 7),), ((0, 1, 2),), ((4, -2, 7),))
    batch_offset = ScatterCase(((1, 3), (5, 9)), ((0, 0), (1, 1)), ((2,), (7,)))


@pytest.fixture
def dims() -> AtomDims:
    """Return the small sizes every test builds its modules and inputs from.

    Returns:
        Sizes with every attention width divisible by ``n_heads``.
    """
    return AtomDims(
        batch=1,
        n_res=8,
        atoms_per_res=3,
        n_elements=4,
        c_atom=32,
        c_atompair=16,
        c_token=32,
        c_pair=32,
        n_heads=4,
    )


@pytest.fixture
def model_params(dims: AtomDims) -> ModelParams:
    """Build ModelParams at the test sizes, one block per atom transformer.

    Args:
        dims: Model sizes.

    Returns:
        ModelParams whose ``f_ref_dim`` matches ``atoms_per_res * (3 + E)``.
    """
    return ModelParams(
        f_ref_dim=dims.atoms_per_res * (3 + dims.n_elements),
        c_res=dims.c_token,
        c_pair=dims.c_pair,
        c_atom=dims.c_atom,
        c_atompair=dims.c_atompair,
        n_blocks_atom_transformer_encoder=1,
        n_heads_atom_transformer_encoder=dims.n_heads,
        n_blocks_atom_transformer_decoder=1,
        n_heads_atom_transformer_decoder=dims.n_heads,
    )


@pytest.fixture
def atom_inputs(dims: AtomDims) -> AtomInputs:
    """Build every atom-level input from deterministic patterns.

    Atoms are grouped ``atoms_per_res`` to a residue, atom ``l`` has element
    ``l % E``, and the sparse layout uses the default window, which covers
    all 8 residues.

    Args:
        dims: Input sizes.

    Returns:
        The bundled inputs.
    """
    b, n, r = dims.batch, dims.n_atom, dims.n_res
    tok = torch.repeat_interleave(torch.arange(r), dims.atoms_per_res)
    nbrs, mask = build_sparse_pairs(tok)
    k = nbrs.shape[1]
    elements = torch.arange(n) % dims.n_elements
    return AtomInputs(
        tok_idx=repeat(tok, "n -> b n", b=b),
        pairs=SparsePairs(
            neighbor_idx=repeat(nbrs, "n k -> b n k", b=b),
            valid_mask=repeat(mask, "n k -> b n k", b=b),
        ),
        q=sine_pattern((b, n, dims.c_atom), 0.31),
        c=sine_pattern((b, n, dims.c_atom), 0.43),
        p=sine_pattern((b, n, k, dims.c_atompair), 0.59),
        ref_pos=sine_pattern((b, n, 3), 0.67),
        ref_element=repeat(
            F.one_hot(elements, dims.n_elements).float(),
            "n e -> b n e",
            b=b,
        ),
        ref_space_uid=torch.zeros(b, n, dtype=torch.long),
        r_scaled=sine_pattern((b, n, 3), 0.79),
        s_input=sine_pattern((b, r, dims.c_token), 0.83),
        z_input=sine_pattern((b, r, r, dims.c_pair), 0.97),
    )


@pytest.fixture
def sparse_inputs(dims: AtomDims) -> AtomInputs:
    """Build atom inputs on 20 two-atom residues with a window of 4.

    Each atom then has ``K = 6`` neighbours out of ``N = 40``, the layout
    in which dense ``(N, N)`` logits plus a ``(N, K)`` pair bias crashed.
    Only the atom-level fields are meaningful; the residue-level fields are
    sized for 20 residues but unused by the sparse probes.

    Args:
        dims: Channel sizes.

    Returns:
        The bundled inputs.
    """
    b, r, per = dims.batch, 20, 2
    n = r * per
    tok = torch.repeat_interleave(torch.arange(r), per)
    nbrs, mask = build_sparse_pairs(tok, window_size=4)
    k = nbrs.shape[1]
    return AtomInputs(
        tok_idx=repeat(tok, "n -> b n", b=b),
        pairs=SparsePairs(
            neighbor_idx=repeat(nbrs, "n k -> b n k", b=b),
            valid_mask=repeat(mask, "n k -> b n k", b=b),
        ),
        q=sine_pattern((b, n, dims.c_atom), 0.31),
        c=sine_pattern((b, n, dims.c_atom), 0.43),
        p=sine_pattern((b, n, k, dims.c_atompair), 0.59),
        ref_pos=sine_pattern((b, n, 3), 0.67),
        ref_element=torch.zeros(b, n, dims.n_elements),
        ref_space_uid=torch.zeros(b, n, dtype=torch.long),
        r_scaled=sine_pattern((b, n, 3), 0.79),
        s_input=sine_pattern((b, r, dims.c_token), 0.83),
        z_input=sine_pattern((b, r, r, dims.c_pair), 0.97),
    )


@pytest.fixture
def transition_probe(dims: AtomDims) -> TransitionProbe:
    """Probe ConditionedTransitionBlock on residue-level inputs.

    Args:
        dims: Module and input sizes.

    Returns:
        Probe over the atom embedding ``a``.
    """
    _ = manual_seed(0)
    shape = (dims.batch, dims.n_res, dims.c_atom)
    return TransitionProbe(
        first=sine_pattern(shape, 0.37),
        out_shape=shape,
        block=ConditionedTransitionBlock(dims.c_atom, dims.c_atom).eval(),
        s=sine_pattern(shape, 0.53),
    )


@pytest.fixture
def diffusion_dense_probe(dims: AtomDims) -> DiffusionProbe:
    """Probe a two-block DiffusionTransformer on dense residue pairs.

    Args:
        dims: Module and input sizes.

    Returns:
        Probe over the atom embedding ``a``.
    """
    _ = manual_seed(0)
    b, n = dims.batch, dims.n_res
    shape = (b, n, dims.c_atom)
    return DiffusionProbe(
        first=sine_pattern(shape, 0.37),
        out_shape=shape,
        model=DiffusionTransformer(
            dims.c_atom,
            dims.c_atom,
            dims.c_pair,
            N_block=2,
            N_head=dims.n_heads,
        ).eval(),
        s=sine_pattern(shape, 0.53),
        z=sine_pattern((b, n, n, dims.c_pair), 0.71),
        beta=torch.zeros(b, n, n),
        neighbor_idx=None,
    )


@pytest.fixture
def diffusion_sparse_probe(
    dims: AtomDims,
    sparse_inputs: AtomInputs,
) -> DiffusionProbe:
    """Probe DiffusionTransformer on the ``K < N`` sparse layout.

    Args:
        dims: Module sizes.
        sparse_inputs: Inputs with 6 of 40 neighbours per atom.

    Returns:
        Probe over the atom embedding ``a``.
    """
    _ = manual_seed(0)
    x = sparse_inputs
    return DiffusionProbe(
        first=x.q,
        out_shape=tuple(x.q.shape),
        model=DiffusionTransformer(
            dims.c_atom,
            dims.c_atom,
            dims.c_atompair,
            N_block=1,
            N_head=dims.n_heads,
        ).eval(),
        s=x.c,
        z=x.p,
        beta=torch.zeros(x.pairs.neighbor_idx.shape),
        neighbor_idx=x.pairs.neighbor_idx,
    )


@pytest.fixture
def atom_transformer_probe(
    dims: AtomDims,
    atom_inputs: AtomInputs,
) -> AtomTransformerProbe:
    """Probe a two-block AtomTransformer on the 24-atom layout.

    Args:
        dims: Module sizes.
        atom_inputs: Atom-level inputs.

    Returns:
        Probe over the queries ``q``.
    """
    _ = manual_seed(0)
    model = AtomTransformer(dims.c_atom, dims.c_atompair, 2, dims.n_heads)
    return AtomTransformerProbe(
        first=atom_inputs.q,
        out_shape=tuple(atom_inputs.q.shape),
        model=model.eval(),
        inputs=atom_inputs,
    )


@pytest.fixture
def atom_transformer_sparse_probe(
    dims: AtomDims,
    sparse_inputs: AtomInputs,
) -> AtomTransformerProbe:
    """Probe AtomTransformer on the ``K < N`` sparse layout.

    Args:
        dims: Module sizes.
        sparse_inputs: Inputs with 6 of 40 neighbours per atom.

    Returns:
        Probe over the queries ``q``.
    """
    _ = manual_seed(0)
    model = AtomTransformer(dims.c_atom, dims.c_atompair, 1, dims.n_heads)
    return AtomTransformerProbe(
        first=sparse_inputs.q,
        out_shape=tuple(sparse_inputs.q.shape),
        model=model.eval(),
        inputs=sparse_inputs,
    )


@pytest.fixture
def encoder(dims: AtomDims, model_params: ModelParams) -> AtomFeatureEncoder:
    """Build an AtomFeatureEncoder at the test sizes under a fixed seed.

    Args:
        dims: Module sizes.
        model_params: Shared model configuration.

    Returns:
        The encoder in eval mode.
    """
    _ = manual_seed(0)
    return AtomFeatureEncoder(
        c=dims.c_token,
        d=dims.c_atompair,
        m=dims.c_atom,
        model_params=model_params,
    ).eval()


@pytest.fixture
def decoder(model_params: ModelParams) -> AtomAttentionDecoder:
    """Build an AtomAttentionDecoder at the test sizes under a fixed seed.

    Args:
        model_params: Shared model configuration.

    Returns:
        The decoder in eval mode.
    """
    _ = manual_seed(0)
    return AtomAttentionDecoder(model_params=model_params).eval()


@pytest.fixture
def encoder_probe(
    encoder: AtomFeatureEncoder,
    atom_inputs: AtomInputs,
) -> EncoderProbe:
    """Probe the encoder from ``r_scaled`` to its ``q_skip`` output.

    Args:
        encoder: Encoder under test.
        atom_inputs: Atom-level inputs.

    Returns:
        Probe over the scaled noisy positions.
    """
    return EncoderProbe(
        first=atom_inputs.r_scaled,
        out_shape=tuple(atom_inputs.q.shape),
        encoder=encoder,
        inputs=atom_inputs,
    )


@pytest.fixture
def decoder_probe(
    decoder: AtomAttentionDecoder,
    atom_inputs: AtomInputs,
) -> DecoderProbe:
    """Probe the decoder from ``q_skip`` to its position update.

    Args:
        decoder: Decoder under test.
        atom_inputs: Atom-level inputs; ``q``, ``p``, ``c`` stand in for
            the encoder's skip outputs.

    Returns:
        Probe over the skip queries.
    """
    return DecoderProbe(
        first=atom_inputs.q,
        out_shape=tuple(atom_inputs.r_scaled.shape),
        decoder=decoder,
        inputs=atom_inputs,
    )


@pytest.fixture
def module_probe(
    request: pytest.FixtureRequest,
    module_probe_name: str,
) -> ModuleProbe:
    """Return the probe fixture named by ``module_probe_name``.

    Args:
        request: Pytest request used to look up the named fixture.
        module_probe_name: A ``*_probe`` fixture name; the requesting test
            parametrizes it.

    Returns:
        The named probe.
    """
    return cast(ModuleProbe, request.getfixturevalue(module_probe_name))


@pytest.fixture
def layout_tok_idx(layout_scenario: LayoutScenario) -> Int[torch.Tensor, "N"]:
    """Build the scenario's residue index per atom.

    Args:
        layout_scenario: Layout to build; the requesting test
            parametrizes it.

    Returns:
        ``(N,)`` residue index, ``atoms_per_res`` atoms to a residue.
    """
    case = layout_scenario.value
    return torch.repeat_interleave(torch.arange(case.n_res), case.atoms_per_res)


@pytest.fixture
def window_neighbours(
    layout_scenario: LayoutScenario,
    layout_tok_idx: Int[torch.Tensor, "N"],
) -> NeighbourSets:
    """List each atom's neighbours straight from the window definition.

    An independent, loop-based reference: atom ``m`` neighbours ``l`` iff
    ``|tok[l] - tok[m]| < window_size // 2``.

    Args:
        layout_scenario: Supplies the window size; the requesting test
            parametrizes it.
        layout_tok_idx: Residue index per atom.

    Returns:
        One frozenset of neighbour indices per atom.
    """
    half = layout_scenario.value.window_size // 2
    tok = [int(t) for t in layout_tok_idx]
    return NeighbourSets(
        per_atom=tuple(
            frozenset(m for m, t_m in enumerate(tok) if abs(t_l - t_m) < half)
            for t_l in tok
        ),
    )


@pytest.fixture
def dense_pairs() -> SparsePairs:
    """Build a two-item batch of 8 atoms where every slot is every atom.

    Slot ``k`` of atom ``l`` holds atom ``k``, so ``beta[b, l, k]`` reads
    as the bias for the pair ``(l, k)``.

    Returns:
        Identity neighbour layout, all slots valid.
    """
    n = 8
    return SparsePairs(
        neighbor_idx=repeat(torch.arange(n), "k -> b n k", b=2, n=n),
        valid_mask=torch.ones(2, n, n, dtype=torch.bool),
    )


@pytest.fixture
def windowed_transformer(dims: AtomDims) -> AtomTransformer:
    """Build a one-block, one-head AtomTransformer with windows of 4 atoms.

    On 8 atoms the two windows, centred at 1.5 and 5.5, share no pair.

    Args:
        dims: Channel sizes.

    Returns:
        The transformer in eval mode.
    """
    _ = manual_seed(0)
    return AtomTransformer(
        dims.c_atom,
        dims.c_atompair,
        n_blocks=1,
        n_heads=1,
        n_queries=4,
        n_keys=4,
    ).eval()


@pytest.fixture
def windowed_inputs(
    dims: AtomDims,
    dense_pairs: SparsePairs,
) -> AtomInputs:
    """Build one-item inputs on the 8-atom dense layout.

    Args:
        dims: Channel sizes.
        dense_pairs: Identity layout; only batch item 0 is used.

    Returns:
        Inputs whose atom-level fields drive ``windowed_transformer``.
    """
    n = 8
    pairs = SparsePairs(
        neighbor_idx=dense_pairs.neighbor_idx[:1],
        valid_mask=dense_pairs.valid_mask[:1],
    )
    return AtomInputs(
        tok_idx=rearrange(torch.arange(n), "n -> 1 n"),
        pairs=pairs,
        q=sine_pattern((1, n, dims.c_atom), 0.31),
        c=sine_pattern((1, n, dims.c_atom), 0.43),
        p=sine_pattern((1, n, n, dims.c_atompair), 0.59),
        ref_pos=torch.zeros(1, n, 3),
        ref_element=torch.zeros(1, n, dims.n_elements),
        ref_space_uid=torch.zeros(1, n, dtype=torch.long),
        r_scaled=torch.zeros(1, n, 3),
        s_input=torch.zeros(1, n, dims.c_token),
        z_input=torch.zeros(1, n, n, dims.c_pair),
    )


@pytest.mark.parametrize(
    "layout_scenario",
    list(LayoutScenario),
    ids=[case.name for case in LayoutScenario],
)
def test_build_sparse_pairs_matches_window_definition(
    layout_scenario: LayoutScenario,
    layout_tok_idx: Int[torch.Tensor, "N"],
    window_neighbours: NeighbourSets,
) -> None:
    """Each atom's valid slots hold exactly its in-window atoms.

    Also checks ``K`` is the largest neighbour count, the dtypes, and that
    padding slots hold an in-range index, so a gather never reads out of
    bounds.

    Args:
        layout_scenario: Layout and expected ``K``.
        layout_tok_idx: Residue index per atom.
        window_neighbours: Reference neighbour set per atom.
    """
    case = layout_scenario.value
    nbrs, mask = build_sparse_pairs(layout_tok_idx, case.window_size)
    n = layout_tok_idx.shape[0]
    found = tuple(
        frozenset(int(m) for m in row[valid])
        for row, valid in zip(nbrs, mask, strict=True)
    )
    assert nbrs.shape == mask.shape == (n, case.expected_k)
    assert (nbrs.dtype, mask.dtype) == (torch.long, torch.bool)
    assert found == window_neighbours.per_atom
    assert bool(((nbrs >= 0) & (nbrs < n)).all())


@pytest.mark.parametrize(
    "beta_point_scenario",
    list(BetaPointScenario),
    ids=[case.name for case in BetaPointScenario],
)
def test_compute_beta_admits_pairs_sharing_a_window_centre(
    beta_point_scenario: BetaPointScenario,
    dense_pairs: SparsePairs,
) -> None:
    """``beta[l, m]`` is 0 when some centre admits both atoms, else -1e10.

    Args:
        beta_point_scenario: Window widths, pair, and decision.
        dense_pairs: 8-atom identity layout with every slot valid.
    """
    case = beta_point_scenario.value
    ref = torch.zeros(2, 8, 1)
    beta = compute_beta(
        dense_pairs.neighbor_idx,
        dense_pairs.valid_mask,
        case.n_queries,
        case.n_keys,
        ref,
    )
    assert beta[0, case.query, case.key].item() == case.expected_beta


@pytest.mark.parametrize(
    ("dtype", "fill"),
    [
        pytest.param(torch.float32, -1e10, id="float32"),
        pytest.param(torch.float16, -32752.0, id="float16"),
    ],
)
def test_compute_beta_values_are_zero_or_dtype_safe_fill(
    dense_pairs: SparsePairs,
    dtype: torch.dtype,
    fill: float,
) -> None:
    """Beta holds only 0 and the fill value, in ``ref``'s dtype, never inf.

    The fill is -1e10 where the dtype can hold it and half the dtype's
    minimum otherwise, since -1e10 overflows float16 to -inf: float16's
    minimum is -65504, so its fill is -32752.

    Args:
        dense_pairs: 8-atom identity layout with every slot valid.
        dtype: Dtype of ``ref``.
        fill: Value blocked pairs must hold.
    """
    ref = torch.zeros(2, 8, 1, dtype=dtype)
    beta = compute_beta(
        dense_pairs.neighbor_idx,
        dense_pairs.valid_mask,
        4,
        4,
        ref,
    )
    assert beta.shape == dense_pairs.neighbor_idx.shape
    assert beta.dtype == dtype
    assert bool(((beta == 0) | (beta == fill)).all())
    assert bool((beta == 0).any())
    assert bool((beta == fill).any())


def test_compute_beta_invalid_slots_get_fill_per_batch_item(
    dense_pairs: SparsePairs,
) -> None:
    """Masked slots are blocked; unmasked ones keep the geometric decision.

    Batch item 1 masks the in-window pair ``(0, 1)`` and all of atom 2's
    slots; batch item 0 masks nothing. Item 1 must equal item 0 where valid
    and the fill elsewhere, and item 0 must be unaffected by item 1's mask.
    Not parametrized: one mask with a single slot and a full row covers
    both shapes of masking.

    Args:
        dense_pairs: 8-atom identity layout with every slot valid.
    """
    mask = dense_pairs.valid_mask.clone()
    mask[1, 0, 1] = False
    mask[1, 2, :] = False
    ref = torch.zeros(2, 8, 1)
    idx = dense_pairs.neighbor_idx
    open_beta = compute_beta(idx[:1], mask[:1], 4, 8, ref[:1])
    beta = compute_beta(idx, mask, 4, 8, ref)
    expected_item_1 = torch.where(mask[1], open_beta[0], -1e10)
    assert torch.equal(beta[0], open_beta[0])
    assert torch.equal(beta[1], expected_item_1)


def test_compute_beta_admits_everything_when_one_window_covers_all(
    atom_inputs: AtomInputs,
) -> None:
    """With ``N <= n_queries`` every valid pair is admitted.

    24 atoms fit in one 32-atom query window whose 128-atom key window
    covers them too, the production widths on a short chain. Not
    parametrized: any wider window admits the same set.

    Args:
        atom_inputs: 24-atom layout with every slot valid.
    """
    pairs = atom_inputs.pairs
    beta = compute_beta(
        pairs.neighbor_idx,
        pairs.valid_mask,
        32,
        128,
        atom_inputs.q,
    )
    assert bool((beta[pairs.valid_mask] == 0).all())


def test_compute_beta_rejects_unbatched_neighbor_index(
    atom_inputs: AtomInputs,
) -> None:
    """A ``neighbor_idx`` with only the atom axis fails the shape contract.

    Not parametrized: the other inputs share ``neighbor_idx``'s named axes,
    so breaking one of them is enough.

    Args:
        atom_inputs: Supplies a valid mask and ``ref``.
    """
    pairs = atom_inputs.pairs
    flat = pairs.neighbor_idx[0, :, 0]
    with pytest.raises(TypeCheckError):
        _ = compute_beta(flat, pairs.valid_mask, 32, 128, atom_inputs.q)


@pytest.mark.parametrize(
    "scatter_scenario",
    list(ScatterScenario),
    ids=[case.name for case in ScatterScenario],
)
def test_scatter_mean_averages_each_segment(
    scatter_scenario: ScatterScenario,
) -> None:
    """Each output segment is the mean of the source values mapped to it.

    Args:
        scatter_scenario: Source, flat index, and expected means.
    """
    case = scatter_scenario.value
    src = rearrange(torch.tensor(case.src, dtype=torch.float32), "b n -> b n 1")
    index = torch.tensor(case.index)
    expected = torch.tensor(case.expected, dtype=torch.float32)
    b, n_tgt = expected.shape
    out = scatter_mean(src, index, b * n_tgt, b)
    torch.testing.assert_close(out, rearrange(expected, "b n -> b n 1"))


@pytest.fixture
def channel_source() -> Float[torch.Tensor, "1 12 5"]:
    """Return a 12-atom, 5-channel source whose channels all differ.

    Returns:
        A sine pattern of shape ``(1, 12, 5)``.
    """
    return sine_pattern((1, 12, 5), 0.41)


def test_scatter_mean_averages_every_channel_independently(
    channel_source: Float[torch.Tensor, "1 12 5"],
) -> None:
    """With 5 channels, each channel's segment mean matches a grouped mean.

    Three segments of four atoms each; the reference groups the atoms with
    ``rearrange`` and averages with ``reduce``. Not parametrized: the
    one-channel scenarios already vary the segment layout.

    Args:
        channel_source: Source with five distinct channels.
    """
    src = channel_source
    index = rearrange(torch.repeat_interleave(torch.arange(3), 4), "n -> 1 n")
    grouped = rearrange(src, "b (n a) c -> b n a c", a=4)
    expected = reduce(grouped, "b n a c -> b n c", "mean")
    out = scatter_mean(src, index, 3, 1)
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=0)


def test_scatter_mean_rejects_source_without_channel_axis(
    atom_inputs: AtomInputs,
) -> None:
    """A 2-D ``src`` fails the ``(B, N_src, C)`` shape contract.

    Not parametrized: ``index`` has no channel axis to break.

    Args:
        atom_inputs: Supplies a valid index and a source to flatten.
    """
    flat = atom_inputs.q[..., 0]
    with pytest.raises(TypeCheckError):
        _ = scatter_mean(flat, atom_inputs.tok_idx, 8, 1)


@pytest.mark.parametrize(
    "module_probe_name",
    [
        "transition_probe",
        "diffusion_dense_probe",
        "diffusion_sparse_probe",
        "atom_transformer_probe",
        "atom_transformer_sparse_probe",
        "encoder_probe",
        "decoder_probe",
    ],
)
def test_probed_output_shape_and_finite(module_probe: ModuleProbe) -> None:
    """Each module's probed output has its expected shape and is finite.

    The sparse probes run the ``K < N`` layout that used to raise a shape
    mismatch between dense logits and the sparse pair bias.

    Args:
        module_probe: Module call under test.
    """
    with torch.no_grad():
        out = module_probe.run(module_probe.first)
    assert tuple(out.shape) == module_probe.out_shape
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    "module_probe_name",
    [
        "transition_probe",
        "diffusion_dense_probe",
        "diffusion_sparse_probe",
        "atom_transformer_probe",
        "atom_transformer_sparse_probe",
        "encoder_probe",
        "decoder_probe",
    ],
)
def test_probed_gradient_reaches_first_input(module_probe: ModuleProbe) -> None:
    """The summed probed output has a finite, nonzero input gradient.

    Args:
        module_probe: Module call under test.
    """
    leaf = module_probe.first.clone().requires_grad_()
    out = reduce(module_probe.run(leaf), "... -> ", "sum")
    (grad,) = torch.autograd.grad(out, leaf)
    assert torch.isfinite(grad).all()
    assert torch.count_nonzero(grad) > 0


@pytest.mark.parametrize(
    "module_probe_name",
    [
        "transition_probe",
        "diffusion_dense_probe",
        "diffusion_sparse_probe",
        "atom_transformer_probe",
        "atom_transformer_sparse_probe",
        "encoder_probe",
        "decoder_probe",
    ],
)
def test_probed_input_missing_last_axis_is_rejected(
    module_probe: ModuleProbe,
) -> None:
    """Dropping the probed input's last axis fails the module's contract.

    Args:
        module_probe: Module call under test.
    """
    malformed = module_probe.first[..., 0]
    with pytest.raises(TypeCheckError):
        _ = module_probe.run(malformed)


def test_atom_feature_encoder_output_shapes_and_finite(
    dims: AtomDims,
    encoder: AtomFeatureEncoder,
    atom_inputs: AtomInputs,
) -> None:
    """All five encoder outputs have their shapes and are finite.

    ``p_skip`` has one slot per sparse neighbour, so its ``K`` must match
    ``build_sparse_pairs`` on the same residue index. Not parametrized: the
    probe tests cover the encoder's other contracts.

    Args:
        dims: Expected sizes.
        encoder: Encoder under test.
        atom_inputs: Atom-level inputs.
    """
    x = atom_inputs
    b, n, k = x.pairs.neighbor_idx.shape
    with torch.no_grad():
        outputs = encoder(
            x.ref_pos,
            x.ref_element,
            x.ref_space_uid,
            x.s_input,
            x.z_input,
            x.r_scaled,
            x.tok_idx,
        )
    assert [tuple(t.shape) for t in outputs] == [
        (b, dims.n_res, dims.c_token),
        (b, n, dims.c_atom),
        (b, n, dims.c_atom),
        (b, n, k, dims.c_atompair),
        (b, n, dims.c_atom),
    ]
    assert all(torch.isfinite(t).all() for t in outputs)


def test_atom_feature_encoder_keeps_residues_distinct(
    encoder: AtomFeatureEncoder,
    atom_inputs: AtomInputs,
) -> None:
    """No two residues get the same pooled embedding ``s_i``.

    Guards against the atom-to-residue pooling collapsing residues. Not
    parametrized: distinct inputs per residue are enough to expose it.

    Args:
        encoder: Encoder under test.
        atom_inputs: Atom-level inputs that differ per residue.
    """
    x = atom_inputs
    with torch.no_grad():
        s_i, *_ = encoder(
            x.ref_pos,
            x.ref_element,
            x.ref_space_uid,
            x.s_input,
            x.z_input,
            x.r_scaled,
            x.tok_idx,
        )
    assert torch.pdist(s_i[0]).min() > 0


def test_atom_attention_decoder_output_shapes_and_finite(
    dims: AtomDims,
    decoder: AtomAttentionDecoder,
    atom_inputs: AtomInputs,
) -> None:
    """The decoder returns a finite ``(B, N, 3)`` update and atom context.

    Not parametrized: the probe tests cover the decoder's other contracts.

    Args:
        dims: Expected sizes.
        decoder: Decoder under test.
        atom_inputs: Atom-level inputs.
    """
    x = atom_inputs
    with torch.no_grad():
        _, _, r_update, c_out = decoder(
            x.q,
            x.p,
            x.c,
            x.c,
            x.s_input,
            x.z_input,
            x.tok_idx,
        )
    assert r_update.shape == (dims.batch, dims.n_atom, 3)
    assert c_out.shape == (dims.batch, dims.n_atom, dims.c_atom)
    assert torch.isfinite(r_update).all()
    assert torch.isfinite(c_out).all()


def test_atom_transformer_window_ignores_atoms_outside_it(
    windowed_transformer: AtomTransformer,
    windowed_inputs: AtomInputs,
) -> None:
    """Replacing atoms 4-7 leaves the output for atoms 0-3 bit-identical.

    The windows centred at 1.5 and 5.5 share no pair, and ``exp(-1e10)`` is
    exactly 0 in float32, so the second window has no path into the first.
    Not parametrized: the two-window split is the smallest layout with a
    window boundary.

    Args:
        windowed_transformer: Transformer with 4-atom windows.
        windowed_inputs: 8-atom inputs.
    """
    x = windowed_inputs
    pairs = x.pairs
    q_alt, c_alt = x.q.clone(), x.c.clone()
    q_alt[:, 4:] = -q_alt[:, 4:] + 1
    c_alt[:, 4:] = -c_alt[:, 4:] + 1
    with torch.no_grad():
        out = windowed_transformer(
            x.q,
            x.c,
            x.p,
            pairs.neighbor_idx,
            pairs.valid_mask,
        )
        out_alt = windowed_transformer(
            q_alt,
            c_alt,
            x.p,
            pairs.neighbor_idx,
            pairs.valid_mask,
        )
    assert torch.equal(out[:, :4], out_alt[:, :4])


def test_atom_transformer_window_blocks_gradient_from_outside_it(
    windowed_transformer: AtomTransformer,
    windowed_inputs: AtomInputs,
) -> None:
    """Gradients of atoms 0-3's outputs are exactly zero for atoms 4-7.

    Cross-window softmax weights are exactly 0, so their gradient
    ``s * (delta - s)`` is too, and the transition block is per-atom. Not
    parametrized: the companion forward test uses the same layout.

    Args:
        windowed_transformer: Transformer with 4-atom windows.
        windowed_inputs: 8-atom inputs.
    """
    x = windowed_inputs
    q = x.q.clone().requires_grad_()
    c = x.c.clone().requires_grad_()
    pairs = x.pairs
    out = windowed_transformer(q, c, x.p, pairs.neighbor_idx, pairs.valid_mask)
    total = reduce(out[:, :4], "b n c -> ", "sum")
    grad_q, grad_c = torch.autograd.grad(total, [q, c])
    assert torch.count_nonzero(grad_q[:, 4:]) == 0
    assert torch.count_nonzero(grad_c[:, 4:]) == 0
    assert torch.count_nonzero(grad_q[:, :4]) > 0
