"""Atom-level transformer layers for local and global attention over atoms.

Contains scatter_mean, build_sparse_pairs, compute_beta,
ConditionedTransitionBlock, DiffusionTransformer, AtomTransformer,
AtomFeatureEncoder, and AtomAttentionDecoder.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from architecture.layers import (
    LayerNorm,
    LinearNoBias,
    TypedLinear,
    TypedModuleList,
    TypedSequential,
)
from architecture.node_update import AdaLN, AttentionPairBias
from beartype import beartype
from einops import einsum, rearrange, reduce, repeat
from jaxtyping import Bool, Float, Int, jaxtyped
from train.train_config import ModelParams
from typing_extensions import override

# Local attention window size, matching AlphaFold 3 atom transformer.
# Each atom l attends only to atoms m where
# |tok_idx[l] - tok_idx[m]| < WINDOW_SIZE // 2,
# giving each atom at most WINDOW_SIZE // 2 neighbours on each side.
WINDOW_SIZE: int = 128


@jaxtyped(typechecker=beartype)
def scatter_mean(
    src: Float[torch.Tensor, "B N_src C"],
    index: Int[torch.Tensor, "B N_src"],
    num_segments: int,
    B: int,
) -> Float[torch.Tensor, "B N_target C"]:
    """Per-segment mean pooling via scatter.

    Maps atom-level features to residue-level by averaging all atoms that share
    the same flat segment index.  `index` must already encode the batch offset
    (i.e. atom j in batch item b maps to index[b, j] = tok_idx[b, j] + b *
    N_tgt).

    Args:
        src: Source features of shape ``(B, N_src, C)``.
        index: Flat segment indices of shape ``(B, N_src)``, already offset by
            ``b * N_tgt`` so that they index into a ``(B * N_tgt)`` flat
            buffer.
        num_segments: Total number of flat segments, i.e. ``B * N_tgt``.
        B: Batch size, used to reshape the result back to ``(B, N_tgt, C)``.

    Returns:
        Mean-pooled features of shape ``(B, N_target, C)``.
    """
    C: int = src.size(-1)
    device = src.device

    flat_index: Int[torch.Tensor, "BN_src"] = rearrange(index, "b n -> (b n)")
    flat_src: Float[torch.Tensor, "BN_src C"] = rearrange(
        src,
        "b n c -> (b n) c",
    )

    sum_flat: Float[torch.Tensor, "BN_target C"] = torch.zeros(
        num_segments,
        C,
        device=device,
    )
    _ = sum_flat.scatter_add_(0, repeat(flat_index, "n -> n c", c=C), flat_src)

    cnt_flat: Float[torch.Tensor, "BN_target 1"] = torch.zeros(
        num_segments,
        1,
        device=device,
    )
    _ = cnt_flat.scatter_add_(
        0,
        rearrange(flat_index, "n -> n 1"),
        torch.ones(flat_index.size(0), 1, device=device),
    )

    return rearrange(sum_flat / cnt_flat.clamp(min=1), "(b n) c -> b n c", b=B)


class ConditionedTransitionBlock(nn.Module):
    """Gated transition block conditioned on sequence embedding via adaLN-Zero.

    Applies adaptive layer normalisation (AdaLN) to the atom features using the
    sequence embedding, expands into an intermediate SwiGLU-style
    representation,
    then projects back with a sigmoid gate derived from the sequence embedding.

    Args:
        c_a: Atom single-embedding dimension.
        c_s: Sequence (conditioning) embedding dimension.
        expansion: Hidden-dimension multiplier; intermediate width is
            ``expansion * c_a``.
    """

    def __init__(self, c_a: int, c_s: int, expansion: int = 2) -> None:
        super().__init__()
        self.adaln: AdaLN = AdaLN(c_a=c_a, c_s=c_s)
        self.a_to_b_1: LinearNoBias = LinearNoBias(c_a, expansion * c_a)
        self.a_to_b_2: LinearNoBias = LinearNoBias(c_a, expansion * c_a)
        self.s_to_a: TypedLinear = TypedLinear(c_s, c_a)  # biasinit=-2.0
        _ = nn.init.constant_(self.s_to_a.bias, -2.0)
        self.b_to_a: LinearNoBias = LinearNoBias(expansion * c_a, c_a)

    @override
    def __call__(
        self,
        a: Float[torch.Tensor, "B N_res c_a"],
        s: Float[torch.Tensor, "B N_res c_s"],
    ) -> Float[torch.Tensor, "B N_res c_a"]:
        """Call forward; typed override so call-site return types are not Any.

        See ``forward`` for full documentation of arguments and return values.
        """
        return self.forward(a, s)

    @override
    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        a: Float[torch.Tensor, "B N_res c_a"],
        s: Float[torch.Tensor, "B N_res c_s"],
    ) -> Float[torch.Tensor, "B N_res c_a"]:
        """Apply the conditioned transition to atom features.

        Computes:
            1. ``a ← AdaLN(a, s)``
            2. ``b ← swish(Linear(a)) ⊙ Linear(a)`` (SwiGLU gate, ``b ∈
            R^{n·c_a}``)
            3. ``a ← sigmoid(Linear(s, biasinit=-2.0)) ⊙ Linear(b)``
            (adaLN-Zero output gate)

        Args:
            a: Atom single embeddings of shape ``(B, N_res, c_a)``.
            s: Sequence conditioning embeddings of shape ``(B, N_res, c_s)``.

        Returns:
            Updated atom embeddings of shape ``(B, N_res, c_a)``.
        """
        a = self.adaln(a, s)
        b: Float[torch.Tensor, "B N_res c_b"] = F.silu(
            self.a_to_b_1(a),
        ) * self.a_to_b_2(a)
        return F.sigmoid(self.s_to_a(s)) * self.b_to_a(b)


class DiffusionTransformer(nn.Module):
    """Transformer that refines atom embeddings with pair-biased attention.

    Runs ``N_block`` rounds of pair-biased attention followed by a conditioned
    transition block, matching the AlphaFold 3 DiffusionTransformer loop:

        for n in 1 … N_block:
            b  = AttentionPairBias_n(a, s, z, β)
            a  = b + ConditionedTransitionBlock_n(a, s)

    Each block has its own independent weights (no weight tying).

    Args:
        c_a: Atom single-embedding dimension.
        c_s: Sequence (conditioning) embedding dimension.
        c_pair: Pair embedding dimension fed into attention pair bias.
        N_block: Number of transformer blocks to apply.
        N_head: Number of attention heads.
    """

    def __init__(
        self,
        c_a: int,
        c_s: int,
        c_pair: int,
        N_block: int,
        N_head: int,
    ) -> None:
        super().__init__()
        self.N_block: int = N_block
        self.attn_pair_bias_blocks: TypedModuleList[AttentionPairBias] = (
            TypedModuleList(
                [
                    AttentionPairBias(c_res=c_a, c_pair=c_pair, n_heads=N_head)
                    for _ in range(N_block)
                ],
            )
        )
        self.cond_trans_blocks: TypedModuleList[ConditionedTransitionBlock] = (
            TypedModuleList(
                [
                    ConditionedTransitionBlock(c_a=c_a, c_s=c_s, expansion=2)
                    for _ in range(N_block)
                ],
            )
        )

    @override
    def __call__(
        self,
        a: Float[torch.Tensor, "B N_res c_a"],
        s: Float[torch.Tensor, "B N_res c_s"],
        z: Float[torch.Tensor, "B N_res N_j c_pair"],
        beta: Float[torch.Tensor, "B N_res N_j"] | None = None,
        neighbor_idx: Int[torch.Tensor, "B N_res N_j"] | None = None,
    ) -> Float[torch.Tensor, "B N_res c_a"]:
        """Call forward; typed override so call-site return types are not Any.

        See ``forward`` for full documentation of arguments and return values.
        """
        return self.forward(a, s, z, beta, neighbor_idx)

    @override
    @jaxtyped(typechecker=beartype)  # q,c, p = a,s,z
    def forward(
        self,
        a: Float[torch.Tensor, "B N_res c_a"],
        s: Float[torch.Tensor, "B N_res c_s"],
        z: Float[torch.Tensor, "B N_res N_j c_pair"],
        beta: Float[torch.Tensor, "B N_res N_j"] | None = None,
        neighbor_idx: Int[torch.Tensor, "B N_res N_j"] | None = None,
    ) -> Float[torch.Tensor, "B N_res c_a"]:
        """Run N_block rounds of attention and transition over atom embeddings.

        Args:
            a: Atom single embeddings of shape ``(B, N_res, c_a)``.
            s: Sequence conditioning embeddings of shape ``(B, N_res, c_s)``.
            z: Pair embeddings — dense ``(B, N_res, N_res, c_pair)`` or
                sparse ``(B, N_res, K, c_pair)``.
            beta: Optional additive attention bias matching the pair dimension
                of ``z``.
            neighbor_idx: Sparse neighbour indices ``(B, N_res, K)``; pass when
                z is sparse so
                attention is computed over K neighbours rather than all N
                positions.

        Returns:
            Refined atom embeddings of shape ``(B, N_res, c_a)``.
        """
        for attn_block, trans_block in zip(
            self.attn_pair_bias_blocks,
            self.cond_trans_blocks,
            strict=True,
        ):
            b: Float[torch.Tensor, "B N_res c_a"] = attn_block(
                a=a,
                s=s,
                z=z,
                beta=beta,
                neighbor_idx=neighbor_idx,
            )
            a = b + trans_block(a, s)
        return a


# ---------------------------------------------------------------------------
# Sparse-pair index builder
# ---------------------------------------------------------------------------


def build_sparse_pairs(
    tok_idx: Int[torch.Tensor, "N"],
    window_size: int = WINDOW_SIZE,
) -> tuple[
    Int[torch.Tensor, "N K"],
    Bool[torch.Tensor, "N K"],
]:
    """For each atom l, collect indices of all atoms m within window.

    Uses an O(N²) boolean intermediate (one byte per pair) to build the index,
    then returns O(N by K) long + bool tensors where K << N for large proteins.

    Args:
        tok_idx: [N] residue index (0-based) for each atom.
        window_size: Total window span in residues (default 32). Atom l
            attends to m iff ``|tok_idx[l] - tok_idx[m]|`` < window_size // 2.

    Returns:
        neighbor_idx: [N, K] atom indices of each neighbour; padding slots → 0
        valid_mask: [N, K] True where the slot is a real neighbour (not
            padding)

    K = maximum neighbours any single atom has; varies with sequence length and
    atoms-per-residue. For a 32-residue window with ~14 atoms/residue, K ≈ 448.
    """
    N = tok_idx.size(0)
    half = window_size // 2

    # [N, N] bool — True where m is within half residues of l
    diff = (
        rearrange(tok_idx, "n -> n 1") - rearrange(tok_idx, "n -> 1 n")
    ).abs()  # [N, N]
    in_window = diff < half  # [N, N]

    K = int(in_window.sum(dim=1).max().item())

    # For each row l, sort so valid neighbours appear first
    # (out-of-window → sentinel N)
    col = repeat(
        torch.arange(N, device=tok_idx.device),
        "k -> n k",
        n=N,
    )  # [N, N]
    sentinel = col.masked_fill(~in_window, N)
    neighbor_idx = sentinel.sort(dim=1).values[
        :,
        :K,
    ]  # [N, K], valid entries < N

    valid_mask = neighbor_idx < N  # [N, K]
    # padding slots → safe index 0
    neighbor_idx = neighbor_idx.clamp(max=N - 1)

    return neighbor_idx, valid_mask


# ---------------------------------------------------------------------------
# compute_beta — Algorithm 7 line 1
# ---------------------------------------------------------------------------


@jaxtyped(typechecker=beartype)
def compute_beta(
    neighbor_idx: Int[torch.Tensor, "B N K"],
    valid_mask: Bool[torch.Tensor, "B N K"],
    n_queries: int,
    n_keys: int,
    ref: Float[torch.Tensor, "B N c"],
) -> Float[torch.Tensor, "B N K"]:
    """Compute the sliding-window attention bias β (Algorithm 7, line 1).

    Returns 0.0 for atom pairs (l, m) admitted by at least one query/key window
    centre and by valid_mask; returns -1e10 everywhere else.

    Window centres are placed at ``c * n_queries + (n_queries / 2 - 0.5)``
    for ``c = 0, 1, …, ⌈N / n_queries⌉ - 1``.

    Args:
        neighbor_idx: Sparse neighbour indices [B, N, K].
        valid_mask: True where neighbour slot is a real atom [B, N, K].
        n_queries: Query-window full-width in atoms.
        n_keys: Key-window full-width in atoms.
        ref: Float tensor whose device and dtype are used for the output [B, N,
            c].

    Returns:
        Additive attention bias [B, N, K]; 0.0 for admitted pairs, -1e10
        elsewhere.
    """
    B, N = ref.shape[0], ref.shape[1]
    K = neighbor_idx.shape[2]
    half_q = n_queries / 2.0
    half_k = n_keys / 2.0

    n_centres = math.ceil(N / n_queries)
    centres: Float[torch.Tensor, "n_centres"] = torch.arange(
        n_centres,
        device=ref.device,
        dtype=torch.float32,
    ) * n_queries + (half_q - 0.5)

    atom_idx: Float[torch.Tensor, "B N"] = repeat(
        torch.arange(N, device=ref.device, dtype=torch.float32),
        "N -> B N",
        B=B,
    )
    # neighbor_idx is (B, N, K) — each entry is an atom index in [0, N).
    # We want m_idx[b, n, k] = atom_idx[b, neighbor_idx[b, n, k]].
    # arange(B)[:, None, None] broadcasts to (B, N, K) to select correct batch
    # row, while neighbor_idx selects the atom column — together they do
    # per-batch gather.
    m_idx: Float[torch.Tensor, "B N K"] = atom_idx[
        torch.arange(B, device=ref.device)[:, None, None],
        neighbor_idx,
    ]

    l_in_window: Bool[torch.Tensor, "B N n_centres"] = (
        rearrange(atom_idx, "B N -> B N 1") - rearrange(centres, "c -> 1 1 c")
    ).abs() < half_q
    m_in_window: Bool[torch.Tensor, "B N K n_centres"] = (
        rearrange(m_idx, "B N K -> B N K 1")
        - rearrange(centres, "c -> 1 1 1 c")
    ).abs() < half_k
    both_in_window: Bool[torch.Tensor, "B N K n_centres"] = (
        rearrange(l_in_window, "B N c -> B N 1 c") & m_in_window
    )
    in_window: Bool[torch.Tensor, "B N K"] = reduce(
        both_in_window.float(),
        "B N K c -> B N K",
        "max",
    ).bool()

    mask_fill = max(-1e10, torch.finfo(ref.dtype).min / 2)
    return torch.where(
        in_window & valid_mask,
        torch.zeros(B, N, K, device=ref.device, dtype=ref.dtype),
        torch.full((B, N, K), mask_fill, device=ref.device, dtype=ref.dtype),
    )


# ---------------------------------------------------------------------------
# AtomTransformer — stack of sparse blocks
# ---------------------------------------------------------------------------


class AtomTransformer(nn.Module):
    """Implements Algorithm 7 (AtomTransformer).

    Constructs the sequence-local β mask once — 0 for atom pairs that share a
    sliding window centre, -1e10 elsewhere — then runs n_blocks rounds of
    sparse pair-biased attention (the DiffusionTransformer loop), passing beta
    as an additive bias to every block.

    n_queries and n_keys are independent widths, not interchangeable. Block
    centres are spaced every n_queries atoms, and atom l is assigned to
    whichever centre c satisfies |l - c| < n_queries / 2. Once assigned, l
    may attend to any atom m with |m - c| < n_keys / 2 — note the key window
    is centred on the block centre, not on l itself. An atom sitting exactly
    at its block's centre sees a full, symmetric window of n_keys / 2 atoms
    on each side. An atom at the edge of its block (offset by close to
    n_queries / 2 from the centre) sees an asymmetric window shifted away
    from it: roughly n_keys/2 - n_queries/2 atoms on the near side and
    n_keys/2 + n_queries/2 atoms on the far side. Keeping n_keys comfortably
    larger than n_queries (AF3: n_queries=32, n_keys=128, so
    half_k - half_q = 48) guarantees every atom, even one at a block edge,
    still sees dozens of neighbours on every side. If n_keys < n_queries,
    half_k - half_q goes negative: an atom near a block edge can end up with
    a key window that doesn't reach back to cover even its own position, so
    its entire attention row is masked to -1e10 by compute_beta and falls
    back to a near-uniform softmax over whatever padding/neighbour slots
    remain in {valid_mask} — not just reduced context, but potentially no
    valid local context at all. Callers must therefore keep
    n_keys >= n_queries.

    Args:
        c_atom: Atom single-embedding dimension.
        c_atompair: Atom-pair embedding dimension.
        n_blocks: Number of transformer blocks (N_block in the paper, default
            3).
        n_heads: Number of attention heads (N_head).
        n_queries: Query-window half-width in atoms (N_queries, default 32).
        n_keys: Key-window full-width in atoms (N_keys, default 128).
    """

    def __init__(
        self,
        c_atom: int,
        c_atompair: int,
        n_blocks: int = 3,
        n_heads: int = 4,
        n_queries: int = 32,
        n_keys: int = 128,
    ) -> None:
        super().__init__()
        self.n_queries: int = n_queries
        self.n_keys: int = n_keys
        self.blocks: DiffusionTransformer = DiffusionTransformer(
            c_a=c_atom,
            c_s=c_atom,
            c_pair=c_atompair,
            N_block=n_blocks,
            N_head=n_heads,
        )

    @override
    def __call__(
        self,
        q: Float[torch.Tensor, "B N c_atom"],
        c: Float[torch.Tensor, "B N c_atom"],
        p: Float[torch.Tensor, "B N K c_atompair"],  # K should be 128 now!
        neighbor_idx: Int[torch.Tensor, "B N K"],
        valid_mask: Bool[torch.Tensor, "B N K"],
    ) -> Float[torch.Tensor, "B N c_atom"]:
        """Call forward; typed override so call-site return types are not Any.

        See ``forward`` for full documentation of arguments and return values.
        """
        return self.forward(q, c, p, neighbor_idx, valid_mask)

    @override
    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        q: Float[torch.Tensor, "B N c_atom"],
        c: Float[torch.Tensor, "B N c_atom"],
        p: Float[torch.Tensor, "B N K c_atompair"],  # K should be 128 now!
        neighbor_idx: Int[torch.Tensor, "B N K"],
        valid_mask: Bool[torch.Tensor, "B N K"],
    ) -> Float[torch.Tensor, "B N c_atom"]:
        """Run Algorithm 7: construct beta then run DiffusionTransformer loop.

        Args:
            q: Atom query embeddings [B, N, c_atom].
            c: Atom context embeddings [B, N, c_atom].
            p: Sparse pair embeddings [B, N, K, c_atompair].
            neighbor_idx: Neighbour atom indices [B, N, K].
            valid_mask: Live-pair mask [B, N, K].

        Returns:
            q : updated atom embeddings            [B, N, c_atom]
        """
        # Algorithm 7 line 1: β_lm — sliding-window attention bias
        beta: Float[torch.Tensor, "B N K"] = compute_beta(
            neighbor_idx,
            valid_mask,
            self.n_queries,
            self.n_keys,
            q,
        )

        # Algorithm 7 line 2:
        # DiffusionTransformer — n_blocks rounds of attention + transition
        return self.blocks(q, c, p, beta, neighbor_idx=neighbor_idx)


# ---------------------------------------------------------------------------
# AtomFeatureEncoder — Algorithm 4
# ---------------------------------------------------------------------------


class AtomFeatureEncoder(nn.Module):
    """Encodes per-atom reference features into per-residue embeddings.

    All N by N pair tensors (d_lm, p_lm, z_gathered) are replaced by sparse
    [B, N, K, *] tensors indexed over the N by K live pairs within the
    32-residue
    window. K is the maximum number of window-neighbours any atom has.

    Parameters
    ----------
    f_ref_dim   : per-atom f^ref feature size (ref_pos_dim + ref_element_dim)
    c_token     : trunk single dim   (s_input)
    c_pair      : trunk pair dim     (z_input)
    c           : output per-residue dim
    d           : atom-pair embedding dim
    m           : atom single embedding dim
    n_blocks    : AtomTransformer blocks
    n_heads     : AtomTransformer heads
    window_size : local attention window in residues (default 32)

    Weight shapes:
        proj_fref_c              : [f_ref_dim, m]
        proj_d_vec               : [3, d]
        proj_inv_sq              : [1, d]
        proj_v                   : [1, d]
        proj_cl_pair / cm_pair   : [m, d]
        proj_r_scaled            : [3, m]
        proj_s_init              : [c_token, m]
        proj_z_init              : [c_pair, d]
        mlp_p layers             : [d, d] each
        proj_agg                 : [m, c]
    """

    def __init__(
        self,
        c: int,
        d: int,
        m: int,
        model_params: ModelParams,
    ) -> None:
        super().__init__()
        self.c: int = c
        self.d: int = d
        self.m: int = m
        self.window_size: int = model_params.window_size
        self.f_ref_dim: int = model_params.f_ref_dim
        self.c_res: int = model_params.c_res
        self.c_pair: int = model_params.c_pair
        self.n_blocks: int = model_params.n_blocks_atom_transformer_encoder
        self.n_heads: int = model_params.n_heads_atom_transformer_encoder

        self.proj_fref_c: LinearNoBias = LinearNoBias(
            self.f_ref_dim,
            self.m,
        )  # [f_ref_dim, m]

        self.proj_d_vec: LinearNoBias = LinearNoBias(3, self.d)  # [3, d]
        self.proj_inv_sq: LinearNoBias = LinearNoBias(1, self.d)  # [1, d]
        self.proj_v: LinearNoBias = LinearNoBias(1, self.d)  # [1, d]
        self.proj_cl_pair: LinearNoBias = LinearNoBias(self.m, self.d)  # [m, d]
        self.proj_cm_pair: LinearNoBias = LinearNoBias(self.m, self.d)  # [m, d]

        self.proj_r_scaled: LinearNoBias = LinearNoBias(3, self.m)  # [3, m]

        self.norm_s_init: LayerNorm = LayerNorm(self.c_res)
        self.proj_s_init: LinearNoBias = LinearNoBias(
            self.c_res,
            self.m,
        )  # [c_token, m]

        self.norm_z_init: LayerNorm = LayerNorm(self.c_pair)
        self.proj_z_init: LinearNoBias = LinearNoBias(self.c_pair, self.d)

        self.mlp_p: TypedSequential = TypedSequential(  # [d, d] each layer
            nn.ReLU(),
            LinearNoBias(self.d, self.d),
            nn.ReLU(),
            LinearNoBias(self.d, self.d),
            nn.ReLU(),
            LinearNoBias(self.d, self.d),
        )

        self.transformer: AtomTransformer = AtomTransformer(
            c_atom=self.m,
            c_atompair=self.d,
            n_blocks=self.n_blocks,
            n_heads=self.n_heads,
            n_queries=self.window_size,
            n_keys=self.window_size * 4,
        )

        self.proj_agg: LinearNoBias = LinearNoBias(self.m, self.c)  # [m, c]

    @override
    def __call__(
        self,
        ref_pos: Float[torch.Tensor, "B N_atom 3"],
        ref_element: Float[torch.Tensor, "B N_atom E"],
        ref_space_uid: Int[torch.Tensor, "B N_atom"],
        s_input: Float[torch.Tensor, "B N_res c_token"],
        z_input: Float[torch.Tensor, "B N_res N_res c_pair"],
        r_scaled: Float[torch.Tensor, "B N_atom 3"],
        tok_idx: Int[torch.Tensor, "B N_atom"],
    ) -> tuple[
        Float[torch.Tensor, "B N_res c_res"],
        Float[torch.Tensor, "B N_atom c_atom"],
        Float[torch.Tensor, "B N_atom c_atom"],
        Float[torch.Tensor, "B N_atom K c_atompair"],
        Float[torch.Tensor, "B N_atom c_atom"],
    ]:
        """Call forward; typed override so call-site return types are not Any.

        See ``forward`` for full documentation of arguments and return values.
        """
        return self.forward(
            ref_pos,
            ref_element,
            ref_space_uid,
            s_input,
            z_input,
            r_scaled,
            tok_idx,
        )

    @override
    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        ref_pos: Float[torch.Tensor, "B N_atom 3"],
        ref_element: Float[torch.Tensor, "B N_atom E"],
        ref_space_uid: Int[torch.Tensor, "B N_atom"],
        s_input: Float[torch.Tensor, "B N_res c_token"],
        z_input: Float[torch.Tensor, "B N_res N_res c_pair"],
        r_scaled: Float[torch.Tensor, "B N_atom 3"],
        tok_idx: Int[torch.Tensor, "B N_atom"],
    ) -> tuple[
        Float[torch.Tensor, "B N_res c_res"],
        Float[torch.Tensor, "B N_atom c_atom"],
        Float[torch.Tensor, "B N_atom c_atom"],
        Float[torch.Tensor, "B N_atom K c_atompair"],
        Float[torch.Tensor, "B N_atom c_atom"],
    ]:
        """Args:.

            ref_pos: [B, N_atom, 3] reference atom positions
            ref_element: [B, N_atom, E] element one-hot features
            ref_space_uid: [B, N_atom] chain/space identifier per atom
            s_input       : [B, N_res, c_token]         trunk single embeddings
            z_input       : [B, N_res, N_res, c_pair]   trunk pair embeddings
            r_scaled      : [B, N_atom, 3]              scaled noisy positions
            tok_idx: [B, N_atom] residue index per atom (0-based)

        Returns (all sparse where applicable):
            s_i    : [B, N_res, c]              aggregated residue embeddings
            q_skip: [B, N_atom, m] atom skip queries (post-transformer)
            c_skip : [B, N_atom, m]             atom skip context
            p_skip: [B, N_atom, K, d] sparse atom-pair skip embeddings
            c_l    : [B, N_atom, m]             updated atom context
        """
        B = ref_pos.size(0)
        N_atom = ref_pos.size(1)
        N_token = s_input.size(1)

        # ------------------------------------------------------------------
        # Build sparse pair index per batch item and stack into (B, N, K).
        # Every sequence has 5 atoms per residue, so K is identical across
        # batch items and torch.stack works without padding.
        # ------------------------------------------------------------------
        per_batch_nidx: list[Int[torch.Tensor, "N_atom K"]] = []
        per_batch_wvalid: list[Bool[torch.Tensor, "N_atom K"]] = []
        for b in range(B):
            nidx_b, wvalid_b = build_sparse_pairs(tok_idx[b], self.window_size)
            per_batch_nidx.append(nidx_b)
            per_batch_wvalid.append(wvalid_b)

        neighbor_idx: Int[torch.Tensor, "B N_atom K"] = torch.stack(
            per_batch_nidx,
        )
        window_valid: Bool[torch.Tensor, "B N_atom K"] = torch.stack(
            per_batch_wvalid,
        )
        K = neighbor_idx.size(2)

        # Reusable batch selectors for per-batch advanced indexing.
        # arange(B)[:, None] → (B, 1) broadcasts against (B, N);
        # sliced to (B, 1, 1) broadcasts against (B, N, K).
        batch_idx_2d: Int[torch.Tensor, "B 1"] = torch.arange(
            B,
            device=ref_pos.device,
        )[:, None]
        batch_idx_3d: Int[torch.Tensor, "B 1 1"] = batch_idx_2d[:, :, None]

        # Chain constraint: same ref_space_uid per batch item — [B, N_atom, K]
        chain_valid: Bool[torch.Tensor, "B N_atom K"] = (
            rearrange(ref_space_uid, "b n -> b n 1")
            == ref_space_uid[batch_idx_3d, neighbor_idx]
        )
        valid_mask: Bool[torch.Tensor, "B N_atom K"] = (
            window_valid & chain_valid
        )

        # ------------------------------------------------------------------
        # Step 1: f^ref — each atom sees all sibling atoms' pos+element
        # ------------------------------------------------------------------
        n_per_res = N_atom // N_token
        ref_pos_grouped: Float[
            torch.Tensor,
            "B N_atom ref_pos_flat",
        ] = rearrange(ref_pos, "b (n a) d -> b n (a d)", a=n_per_res)[
            batch_idx_2d,
            tok_idx,
        ]
        ref_elem_grouped: Float[
            torch.Tensor,
            "B N_atom ref_elem_flat",
        ] = rearrange(ref_element, "b (n a) d -> b n (a d)", a=n_per_res)[
            batch_idx_2d,
            tok_idx,
        ]
        f_ref: Float[torch.Tensor, "B N_atom f_ref_dim"] = torch.cat(
            [ref_pos_grouped, ref_elem_grouped],
            dim=-1,
        )

        # ------------------------------------------------------------------
        # Step 2: c_l = LinearNoBias(f^ref)              [B, N_atom, m]
        # ------------------------------------------------------------------
        c_l: Float[torch.Tensor, "B N_atom m"] = self.proj_fref_c(f_ref)
        c_skip: Float[torch.Tensor, "B N_atom m"] = c_l.clone()  # step 4

        # ------------------------------------------------------------------
        # Steps 5-10: sparse atom-pair embeddings p_lm   [B, N_atom, K, d]
        # ------------------------------------------------------------------

        # Step 5: d_lm = ref_pos[b, l] - ref_pos[b, m]  [B, N_atom, K, 3]
        d_lm: Float[torch.Tensor, "B N_atom K 3"] = (
            rearrange(ref_pos, "b n d -> b n 1 d")
            - ref_pos[batch_idx_3d, neighbor_idx]
        )

        # Step 6: v_lm = same-chain flag                 [B, N_atom, K]
        v_lm: Float[torch.Tensor, "B N_atom K"] = chain_valid.float()

        # Step 7: p_lm = proj(d_lm) * v_lm              [B, N_atom, K, d]
        p_lm: Float[torch.Tensor, "B N_atom K d"] = self.proj_d_vec(
            d_lm,
        ) * rearrange(v_lm, "b n k -> b n k 1")

        # Step 8: p_lm += proj(1/(1+||d||²)) * v_lm
        inv_sq: Float[torch.Tensor, "B N_atom K 1"] = 1.0 / (
            1.0
            + rearrange(
                einsum(d_lm, d_lm, "b n k d, b n k d -> b n k"),
                "b n k -> b n k 1",
            )
        )
        p_lm = p_lm + self.proj_inv_sq(inv_sq) * rearrange(
            v_lm,
            "b n k -> b n k 1",
        )

        # Step 9: p_lm += proj(v_lm) * v_lm
        p_lm = p_lm + self.proj_v(
            rearrange(v_lm, "b n k -> b n k 1"),
        ) * rearrange(v_lm, "b n k -> b n k 1")

        # Step 10: p_lm += proj(ReLU(c_l)) + proj(ReLU(c_m))
        cl_proj: Float[torch.Tensor, "B N_atom d"] = self.proj_cl_pair(
            F.relu(c_l),
        )
        cm_proj: Float[torch.Tensor, "B N_atom d"] = self.proj_cm_pair(
            F.relu(c_l),
        )
        cm_nbr: Float[torch.Tensor, "B N_atom K d"] = cm_proj[
            batch_idx_3d,
            neighbor_idx,
        ]
        p_lm = p_lm + rearrange(cl_proj, "b n d -> b n 1 d") + cm_nbr

        # Zero padding slots
        p_lm = p_lm * rearrange(valid_mask, "b n k -> b n k 1")
        p_skip: Float[torch.Tensor, "B N_atom K d"] = p_lm.clone()  # step 11

        # ------------------------------------------------------------------
        # Step 12: q_skip = c_l + proj(r_scaled)        [B, N_atom, m]
        # ------------------------------------------------------------------
        q_skip: Float[torch.Tensor, "B N_atom m"] = c_l + self.proj_r_scaled(
            r_scaled,
        )

        # ------------------------------------------------------------------
        # Step 13: c_l += proj(LayerNorm(s_input[tok_idx]))
        # ------------------------------------------------------------------
        c_l = c_l + self.proj_s_init(
            self.norm_s_init(s_input[batch_idx_2d, tok_idx]),
        )

        # ------------------------------------------------------------------
        # Step 14: p_lm += proj(LayerNorm(z_input[tok_l, tok_m]))
        # ------------------------------------------------------------------
        # tok_idx[b, neighbor_idx[b, n, k]]: residue idx of each neighbour atom
        tok_nbr: Int[torch.Tensor, "B N_atom K"] = tok_idx[
            batch_idx_3d,
            neighbor_idx,
        ]
        tok_l: Int[torch.Tensor, "B N_atom K"] = repeat(
            tok_idx,
            "b n -> b n k",
            k=K,
        )
        # z_input[b, tok_l[b,n,k], tok_nbr[b,n,k]] — needs explicit batch index
        batch_idx_full: Int[torch.Tensor, "B N_atom K"] = repeat(
            torch.arange(B, device=z_input.device),
            "b -> b n k",
            n=N_atom,
            k=K,
        )
        z_gathered: Float[torch.Tensor, "B N_atom K c_pair"] = z_input[
            batch_idx_full,
            tok_l,
            tok_nbr,
        ]
        p_lm = p_lm + self.proj_z_init(self.norm_z_init(z_gathered))

        # ------------------------------------------------------------------
        # Step 15: p_lm += MLP(p_lm)   residual 3-layer ReLU
        # ------------------------------------------------------------------
        p_lm = p_lm + self.mlp_p(p_lm)

        # ------------------------------------------------------------------
        # Step 16: AtomTransformer with 32 / 128 sparse window described in AF3
        # ------------------------------------------------------------------
        q_skip = self.transformer(
            q_skip,
            c_l,
            p_lm,
            neighbor_idx,
            valid_mask,
        )

        # ------------------------------------------------------------------
        # Step 17: s_i = mean_{l: tok_idx(l)=i} ReLU(proj(q_skip))
        # ------------------------------------------------------------------
        proj_q: Float[torch.Tensor, "B N_atom c"] = F.relu(
            self.proj_agg(q_skip),
        )

        tok_offset: Int[torch.Tensor, "B N_atom"] = (
            tok_idx
            + repeat(
                torch.arange(B, device=tok_idx.device),
                "b -> b n",
                n=N_atom,
            )
            * N_token
        )
        s_i: Float[torch.Tensor, "B N_res c"] = scatter_mean(
            proj_q,
            tok_offset,
            B * N_token,
            B,
        )

        return s_i, q_skip, c_skip, p_skip, c_l


# ---------------------------------------------------------------------------
# AtomAttentionDecoder — Algorithm 5
# ---------------------------------------------------------------------------


class AtomAttentionDecoder(nn.Module):
    """Decodes trunk embeddings back to per-atom position updates.

    Like the encoder, all pair tensors are sparse [B, N, K, *].
    Builds its own neighbour index from tok_idx; no dense mask required.

    Parameters
    ----------
    c_token     : trunk single dim  (s)
    c_pair      : trunk pair dim    (z)
    c_atom      : atom single dim   (q, c)
    c_atompair  : atom-pair dim     (p)
    n_blocks    : AtomTransformer blocks (default 3)
    n_heads     : attention heads        (default 4)
    window_size : local window in residues (default 32)

    Weight shapes:
        proj_s_q / proj_s_c : [c_token, c_atom]      (no bias)
        proj_z              : [c_pair, c_atompair]    (no bias)
        proj_r              : [c_atom, 3]             (no bias)
        mlp_p layers        : [c_atompair, c_atompair] each (no bias)
    """

    def __init__(
        self,
        model_params: ModelParams,
    ) -> None:
        super().__init__()
        self.c_res: int = model_params.c_res
        self.c_pair: int = model_params.c_pair
        self.c_atom: int = model_params.c_atom
        self.c_atompair: int = model_params.c_atompair
        self.n_blocks: int = model_params.n_blocks_atom_transformer_decoder
        self.n_heads: int = model_params.n_heads_atom_transformer_decoder

        self.window_size: int = model_params.window_size
        self.n_queries: int = self.window_size
        self.n_keys: int = self.window_size * 4

        self.norm_s_q: LayerNorm = LayerNorm(self.c_res)
        self.proj_s_q: LinearNoBias = LinearNoBias(
            self.c_res,
            self.c_atom,
        )  # [c_res, c_atom]

        self.norm_z: LayerNorm = LayerNorm(self.c_pair)
        self.proj_z: LinearNoBias = LinearNoBias(
            self.c_pair,
            self.c_atompair,
        )  # [c_pair, c_atompair]

        self.mlp_p: TypedSequential = (
            TypedSequential(  # [c_atompair, c_atompair] each
                nn.ReLU(),
                LinearNoBias(self.c_atompair, self.c_atompair),
                nn.ReLU(),
                LinearNoBias(self.c_atompair, self.c_atompair),
                nn.ReLU(),
                LinearNoBias(self.c_atompair, self.c_atompair),
            )
        )

        self.transformer: AtomTransformer = AtomTransformer(
            c_atom=self.c_atom,
            c_atompair=self.c_atompair,
            n_blocks=self.n_blocks,
            n_heads=self.n_heads,
            n_queries=self.n_queries,
            n_keys=self.n_keys,
        )

        self.norm_q_out: LayerNorm = LayerNorm(self.c_atom)
        self.proj_r: LinearNoBias = LinearNoBias(self.c_atom, 3)  # [c_atom, 3]

        self.norm_s_c: LayerNorm = LayerNorm(self.c_res)
        self.proj_s_c: LinearNoBias = LinearNoBias(
            self.c_res,
            self.c_atom,
        )  # [c_res, c_atom]

    @override
    def __call__(
        self,
        q_skip: Float[torch.Tensor, "B N_atom c_atom"],
        p_skip: Float[torch.Tensor, "B N_atom K c_atompair"],
        c_skip: Float[torch.Tensor, "B N_atom c_atom"],
        c: Float[torch.Tensor, "B N_atom c_atom"],
        s: Float[torch.Tensor, "B N_res c_token"],
        z: Float[torch.Tensor, "B N_res N_res c_pair"],
        tok_idx: Int[torch.Tensor, "B N_atom"],
    ) -> tuple[
        Float[torch.Tensor, "B N_atom c_atom"],
        Float[torch.Tensor, "B N_atom K c_atompair"],
        Float[torch.Tensor, "B N_atom 3"],
        Float[torch.Tensor, "B N_atom c_atom"],
    ]:
        """Call forward; typed override so call-site return types are not Any.

        See ``forward`` for full documentation of arguments and return values.
        """
        return self.forward(q_skip, p_skip, c_skip, c, s, z, tok_idx)

    @override
    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        q_skip: Float[torch.Tensor, "B N_atom c_atom"],
        p_skip: Float[torch.Tensor, "B N_atom K c_atompair"],
        c_skip: Float[torch.Tensor, "B N_atom c_atom"],
        c: Float[torch.Tensor, "B N_atom c_atom"],
        s: Float[torch.Tensor, "B N_res c_token"],
        z: Float[torch.Tensor, "B N_res N_res c_pair"],
        tok_idx: Int[torch.Tensor, "B N_atom"],
    ) -> tuple[
        Float[torch.Tensor, "B N_atom c_atom"],
        Float[torch.Tensor, "B N_atom K c_atompair"],
        Float[torch.Tensor, "B N_atom 3"],
        Float[torch.Tensor, "B N_atom c_atom"],
    ]:
        """Run Algorithm 8: cross-attention decoder block.

        Args:
            q_skip: Atom skip queries [B, N_atom, c_atom].
            p_skip: Sparse pair skip [B, N_atom, K, c_atompair] (from encoder).
            c_skip: Atom skip context [B, N_atom, c_atom].
            c: Atom context [B, N_atom, c_atom].
            s: Trunk single embeds [B, N_res, c_token].
            z: Trunk pair embeds [B, N_res, N_res, c_pair].
            tok_idx: Residue index per atom [B, N_atom].

        Returns:
            q: Atom query embeddings [B, N_atom, c_atom].
            p: Sparse pair embeddings [B, N_atom, K, c_atompair].
            r_update: Per-atom position update [B, N_atom, 3].
            c_out: Updated atom context [B, N_atom, c_atom].
        """
        B = tok_idx.size(0)
        N_atom = tok_idx.size(1)

        # Build sparse pair index per batch item and stack — K is identical
        # across items since every residue has 5 atoms.
        per_batch_nidx: list[Int[torch.Tensor, "N_atom K"]] = []
        per_batch_wvalid: list[Bool[torch.Tensor, "N_atom K"]] = []
        for b in range(B):
            nidx_b, wvalid_b = build_sparse_pairs(tok_idx[b], self.window_size)
            per_batch_nidx.append(nidx_b)
            per_batch_wvalid.append(wvalid_b)

        neighbor_idx: Int[torch.Tensor, "B N_atom K"] = torch.stack(
            per_batch_nidx,
        )
        valid_mask: Bool[torch.Tensor, "B N_atom K"] = torch.stack(
            per_batch_wvalid,
        )
        K = neighbor_idx.size(2)

        batch_idx_2d: Int[torch.Tensor, "B 1"] = torch.arange(
            B,
            device=tok_idx.device,
        )[:, None]
        batch_idx_3d: Int[torch.Tensor, "B 1 1"] = batch_idx_2d[:, :, None]

        # Step 1: q = proj(LayerNorm(s[tok_idx])) + q_skip [B, N_atom, c_atom]
        q: Float[torch.Tensor, "B N_atom c_atom"] = (
            self.proj_s_q(self.norm_s_q(s[batch_idx_2d, tok_idx])) + q_skip
        )

        # Step 2: p = proj(LayerNorm(z[tok_l, tok_m])) + p_skip
        tok_nbr: Int[torch.Tensor, "B N_atom K"] = tok_idx[
            batch_idx_3d,
            neighbor_idx,
        ]
        tok_l: Int[torch.Tensor, "B N_atom K"] = repeat(
            tok_idx,
            "b n -> b n k",
            k=K,
        )
        batch_idx_full: Int[torch.Tensor, "B N_atom K"] = repeat(
            torch.arange(B, device=z.device),
            "b -> b n k",
            n=N_atom,
            k=K,
        )
        z_gathered: Float[torch.Tensor, "B N_atom K c_pair"] = z[
            batch_idx_full,
            tok_l,
            tok_nbr,
        ]
        p: Float[torch.Tensor, "B N_atom K c_atompair"] = (
            self.proj_z(self.norm_z(z_gathered)) + p_skip
        )

        # Step 3: p += MLP(p) residual; zero padding slots
        p = (p + self.mlp_p(p)) * rearrange(valid_mask, "b n k -> b n k 1")

        # Step 4: AtomTransformer — 32-residue sparse window
        q = self.transformer(
            q,
            c,
            p,
            neighbor_idx,
            valid_mask,
        )

        # Step 5: r_update = proj(LayerNorm(q))             [B, N_atom, 3]
        r_update: Float[torch.Tensor, "B N_atom 3"] = self.proj_r(
            self.norm_q_out(q),
        )

        # Step 6: c_out = proj(LayerNorm(s[tok_idx])) + c_skip
        c_out: Float[torch.Tensor, "B N_atom c_atom"] = (
            self.proj_s_c(self.norm_s_c(s[batch_idx_2d, tok_idx])) + c_skip
        )

        return q, p, r_update, c_out
