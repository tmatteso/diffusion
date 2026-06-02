"""Atom-level transformer layers for local and global attention over atoms."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from architecture.layers import LinearNoBias
from architecture.node_update import AdaLN, AttentionPairBias
from beartype import beartype
from einops import einsum, rearrange, reduce, repeat
from jaxtyping import Bool, Float, Int, jaxtyped

# Local attention window size (residues), matching AlphaFold 3 atom transformer.
# Each atom l attends only to atoms m where |tok_idx[l] - tok_idx[m]| < WINDOW_SIZE // 2,
# giving each atom at most WINDOW_SIZE // 2 neighbours on each side.
WINDOW_SIZE: int = 128


class ConditionedTransitionBlock(nn.Module):
    """Gated transition block conditioned on a sequence embedding via adaLN-Zero.

    Applies adaptive layer normalisation (AdaLN) to the atom features using the
    sequence embedding, expands into an intermediate SwiGLU-style representation,
    then projects back with a sigmoid gate derived from the sequence embedding.

    Args:
        c_a: Atom single-embedding dimension.
        c_s: Sequence (conditioning) embedding dimension.
        expansion: Hidden-dimension multiplier; intermediate width is ``expansion * c_a``.
    """

    def __init__(self, c_a: int, c_s: int, expansion: int = 2) -> None:
        super().__init__()
        self.adaln = AdaLN(c_a=c_a, c_s=c_s)
        self.a_to_b_1 = LinearNoBias(c_a, expansion * c_a)
        self.a_to_b_2 = LinearNoBias(c_a, expansion * c_a)
        self.s_to_a = nn.Linear(c_s, c_a)  # biasinit=-2.0
        nn.init.constant_(self.s_to_a.bias, -2.0)
        self.b_to_a = LinearNoBias(expansion * c_a, c_a)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        a: Float[torch.Tensor, "B N_res c_a"],
        s: Float[torch.Tensor, "B N_res c_s"],
    ) -> Float[torch.Tensor, "B N_res c_a"]:
        """Apply the conditioned transition to atom features.

        Computes:
            1. ``a ← AdaLN(a, s)``
            2. ``b ← swish(Linear(a)) ⊙ Linear(a)``  (SwiGLU gate, ``b ∈ R^{n·c_a}``)
            3. ``a ← sigmoid(Linear(s, biasinit=-2.0)) ⊙ Linear(b)``  (adaLN-Zero output gate)

        Args:
            a: Atom single embeddings of shape ``(B, N_res, c_a)``.
            s: Sequence conditioning embeddings of shape ``(B, N_res, c_s)``.

        Returns:
            Updated atom embeddings of shape ``(B, N_res, c_a)``.
        """
        a = self.adaln(a, s)
        b: Float[torch.Tensor, "B N_res c_b"] = F.silu(self.a_to_b_1(a)) * self.a_to_b_2(a)
        return F.sigmoid(self.s_to_a(s)) * self.b_to_a(b)


class DiffusionTransformer(nn.Module):
    """Iterative transformer that refines atom embeddings with pair-biased attention.

    Runs ``N_block`` rounds of pair-biased attention followed by a conditioned
    transition block, matching the AlphaFold 3 DiffusionTransformer loop:

        for n in 1 … N_block:
            b  = AttentionPairBias(a, s, z, β)
            a  = b + ConditionedTransitionBlock(a, s)

    Args:
        c_a: Atom single-embedding dimension.
        c_s: Sequence (conditioning) embedding dimension.
        c_pair: Pair embedding dimension fed into attention pair bias.
        N_block: Number of transformer blocks to apply.
        N_head: Number of attention heads.
    """

    def __init__(self, c_a: int, c_s: int, c_pair: int, N_block: int, N_head: int) -> None:
        super().__init__()
        self.N_block = N_block
        self.attn_pair_bias = AttentionPairBias(c_res=c_a, c_pair=c_pair, n_heads=N_head)
        self.cond_trans_block = ConditionedTransitionBlock(c_a=c_a, c_s=c_s, expansion=2)

    @jaxtyped(typechecker=beartype)  # q,c, p = a,s,z
    def forward(
        self,
        a: Float[torch.Tensor, "B N_res c_a"],
        s: Float[torch.Tensor, "B N_res c_s"],
        z: Float[torch.Tensor, "B N_res N_j c_pair"],
        beta: Float[torch.Tensor, "B N_res N_j"] | None = None,
        neighbor_idx: Int[torch.Tensor, "N_res N_j"] | None = None,
    ) -> Float[torch.Tensor, "B N_res c_a"]:
        """Run N_block rounds of attention and transition over atom embeddings.

        Args:
            a: Atom single embeddings of shape ``(B, N_res, c_a)``.
            s: Sequence conditioning embeddings of shape ``(B, N_res, c_s)``.
            z: Pair embeddings — dense ``(B, N_res, N_res, c_pair)`` or
                sparse ``(B, N_res, K, c_pair)``.
            beta: Optional additive attention bias matching the pair dimension of ``z``.
            neighbor_idx: Sparse neighbour indices ``(N_res, K)``; pass when z is sparse so
                attention is computed over K neighbours rather than all N positions.

        Returns:
            Refined atom embeddings of shape ``(B, N_res, c_a)``.
        """
        for _ in range(self.N_block):
            a = a + self.attn_pair_bias(a=a, s=s, z=z, beta=beta, neighbor_idx=neighbor_idx)
            a = a + self.cond_trans_block(a, s)
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
    """For each atom l, collect indices of all atoms m within the 32-residue window.

    Uses an O(N²) boolean intermediate (one byte per pair) to build the index,
    then returns O(N by K) long + bool tensors where K << N for large proteins.

    Args:
        tok_idx     : [N]  residue index (0-based) for each atom
        window_size : total window span in residues (default 32)
                      atom l attends to m iff ``|tok_idx[l] - tok_idx[m]|`` < window_size // 2

    Returns:
        neighbor_idx : [N, K]  atom indices of each neighbour; padding slots → 0
        valid_mask   : [N, K]  True where the slot is a real neighbour (not padding)

    K = maximum neighbours any single atom has; varies with sequence length and
    atoms-per-residue. For a 32-residue window with ~14 atoms/residue, K ≈ 448.
    """
    N = tok_idx.size(0)
    half = window_size // 2

    # [N, N] bool — True where m is within half residues of l
    diff = (rearrange(tok_idx, "n -> n 1") - rearrange(tok_idx, "n -> 1 n")).abs()  # [N, N]
    in_window = diff < half  # [N, N]

    K = int(in_window.sum(dim=1).max().item())

    # For each row l, sort so valid neighbours appear first (out-of-window → sentinel N)
    col = repeat(torch.arange(N, device=tok_idx.device), "k -> n k", n=N)  # [N, N]
    sentinel = col.masked_fill(~in_window, N)
    neighbor_idx = sentinel.sort(dim=1).values[:, :K]  # [N, K], valid entries < N

    valid_mask = neighbor_idx < N  # [N, K]
    neighbor_idx = neighbor_idx.clamp(max=N - 1)  # padding slots → safe index 0

    return neighbor_idx, valid_mask


# ---------------------------------------------------------------------------
# compute_beta — Algorithm 7 line 1
# ---------------------------------------------------------------------------


@jaxtyped(typechecker=beartype)
def compute_beta(
    neighbor_idx: Int[torch.Tensor, "N K"],
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
        neighbor_idx: Sparse neighbour indices [N, K].
        valid_mask: True where neighbour slot is a real atom [B, N, K].
        n_queries: Query-window full-width in atoms.
        n_keys: Key-window full-width in atoms.
        ref: Float tensor whose device and dtype are used for the output [B, N, c].

    Returns:
        Additive attention bias [B, N, K]; 0.0 for admitted pairs, -1e10 elsewhere.
    """
    B, N = ref.shape[0], ref.shape[1]
    K = neighbor_idx.shape[1]
    half_q = n_queries / 2.0
    half_k = n_keys / 2.0

    n_centres = math.ceil(N / n_queries)
    centres: Float[torch.Tensor, "n_centres"] = torch.arange(
        n_centres, device=ref.device, dtype=torch.float32
    ) * n_queries + (half_q - 0.5)

    atom_idx: Float[torch.Tensor, "B N"] = repeat(
        torch.arange(N, device=ref.device, dtype=torch.float32), "N -> B N", B=B
    )
    m_idx: Float[torch.Tensor, "B N K"] = atom_idx[:, neighbor_idx]

    l_in_window: Bool[torch.Tensor, "B N n_centres"] = (
        rearrange(atom_idx, "B N -> B N 1") - rearrange(centres, "c -> 1 1 c")
    ).abs() < half_q
    m_in_window: Bool[torch.Tensor, "B N K n_centres"] = (
        rearrange(m_idx, "B N K -> B N K 1") - rearrange(centres, "c -> 1 1 1 c")
    ).abs() < half_k
    both_in_window: Bool[torch.Tensor, "B N K n_centres"] = (
        rearrange(l_in_window, "B N c -> B N 1 c") & m_in_window
    )
    in_window: Bool[torch.Tensor, "B N K"] = reduce(
        both_in_window.float(), "B N K c -> B N K", "max"
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
    sparse pair-biased attention (the DiffusionTransformer loop), passing beta as
    an additive bias to every block.

    Args:
        c_atom: Atom single-embedding dimension.
        c_atompair: Atom-pair embedding dimension.
        n_blocks: Number of transformer blocks (N_block in the paper, default 3).
        n_heads: Number of attention heads (N_head).
        n_queries: Query-window half-width in atoms (N_queries, default 32).
        n_keys: Key-window full-width in atoms (N_keys, default 128).
        window_size: Residue-level neighbour window used to build sparse pairs.
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
        self.n_queries = n_queries
        self.n_keys = n_keys
        self.blocks = DiffusionTransformer(
            c_a=c_atom, c_s=c_atom, c_pair=c_atompair, N_block=n_blocks, N_head=n_heads
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        q: Float[torch.Tensor, "B N c_atom"],
        c: Float[torch.Tensor, "B N c_atom"],
        p: Float[torch.Tensor, "B N K c_atompair"],  # K should be 128 now!
        neighbor_idx: Int[torch.Tensor, "N K"],
        valid_mask: Bool[torch.Tensor, "B N K"],
    ) -> Float[torch.Tensor, "B N c_atom"]:
        """Run Algorithm 7: construct beta then run the DiffusionTransformer loop.

        Args:
            q            : atom query embeddings   [B, N, c_atom]
            c            : atom context embeddings [B, N, c_atom]
            p            : sparse pair embeddings  [B, N, K, c_atompair]
            neighbor_idx : neighbour atom indices  [N, K]
            valid_mask   : live-pair mask          [B, N, K]

        Returns:
            q : updated atom embeddings            [B, N, c_atom]
        """
        # Algorithm 7 line 1: β_lm — sliding-window attention bias
        beta: Float[torch.Tensor, "B N K"] = compute_beta(
            neighbor_idx, valid_mask, self.n_queries, self.n_keys, q
        )

        # Algorithm 7 line 2: DiffusionTransformer — n_blocks rounds of attention + transition
        return self.blocks(q, c, p, beta, neighbor_idx=neighbor_idx)


# ---------------------------------------------------------------------------
# AtomFeatureEncoder — Algorithm 4
# ---------------------------------------------------------------------------


class AtomFeatureEncoder(nn.Module):
    """Encodes per-atom reference features into per-residue embeddings.

    All N by N pair tensors (d_lm, p_lm, z_gathered) are replaced by sparse
    [B, N, K, *] tensors indexed over the N by K live pairs within the 32-residue
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
        f_ref_dim: int,
        c_token: int,
        c_pair: int,
        c: int,
        d: int,
        m: int,
        n_blocks: int,
        n_heads: int,
        window_size: int = WINDOW_SIZE,
    ) -> None:
        super().__init__()
        self.m = m
        self.window_size = window_size

        self.proj_fref_c = LinearNoBias(f_ref_dim, m)  # [f_ref_dim, m]

        self.proj_d_vec = LinearNoBias(3, d)  # [3, d]
        self.proj_inv_sq = LinearNoBias(1, d)  # [1, d]
        self.proj_v = LinearNoBias(1, d)  # [1, d]
        self.proj_cl_pair = LinearNoBias(m, d)  # [m, d]
        self.proj_cm_pair = LinearNoBias(m, d)  # [m, d]

        self.proj_r_scaled = LinearNoBias(3, m)  # [3, m]

        self.norm_s_init = nn.LayerNorm(c_token)
        self.proj_s_init = LinearNoBias(c_token, m)  # [c_token, m]

        self.norm_z_init = nn.LayerNorm(c_pair)
        self.proj_z_init = LinearNoBias(c_pair, d)  # [c_pair, d]

        self.mlp_p = nn.Sequential(  # [d, d] each layer
            LinearNoBias(d, d),
            nn.ReLU(),
            LinearNoBias(d, d),
            nn.ReLU(),
            LinearNoBias(d, d),
            nn.ReLU(),
            LinearNoBias(d, d),
        )

        self.transformer = AtomTransformer(
            c_atom=m,
            c_atompair=d,
            n_blocks=n_blocks,
            n_heads=n_heads,
            n_queries=window_size * 4,
            n_keys=window_size,
        )

        self.proj_agg = LinearNoBias(m, c)  # [m, c]

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

            ref_pos       : [B, N_atom, 3]              reference atom positions
            ref_element   : [B, N_atom, E]              element one-hot features
            ref_space_uid : [B, N_atom]                 chain/space identifier per atom
            s_input       : [B, N_res, c_token]         trunk single embeddings
            z_input       : [B, N_res, N_res, c_pair]   trunk pair embeddings
            r_scaled      : [B, N_atom, 3]              scaled noisy positions
            tok_idx       : [B, N_atom]                 residue index per atom (0-based)

        Returns (all sparse where applicable):
            s_i    : [B, N_res, c]              aggregated residue embeddings
            q_skip : [B, N_atom, m]             atom skip queries (post-transformer)
            c_skip : [B, N_atom, m]             atom skip context
            p_skip : [B, N_atom, K, d]          sparse atom-pair skip embeddings
            c_l    : [B, N_atom, m]             updated atom context
        """
        B = ref_pos.size(0)
        N_atom = ref_pos.size(1)
        N_token = s_input.size(1)

        # tok_idx is identical across batch items; use [0] for index builds.
        tok: Int[torch.Tensor, "N_atom"] = tok_idx[0]  # [N_atom] This bothers me.

        # ------------------------------------------------------------------
        # Build sparse pair index — O(N²) bool then O(N by K) permanently
        # ------------------------------------------------------------------
        neighbor_idx: Int[torch.Tensor, "N_atom K"]
        window_valid: Bool[torch.Tensor, "N_atom K"]
        neighbor_idx, window_valid = build_sparse_pairs(tok, self.window_size)
        K = neighbor_idx.size(1)

        # Chain constraint: same ref_space_uid per batch item — [B, N_atom, K]
        chain_valid: Bool[torch.Tensor, "B N_atom K"] = (
            rearrange(ref_space_uid, "b n -> b n 1") == ref_space_uid[:, neighbor_idx]
        )
        valid_mask: Bool[torch.Tensor, "B N_atom K"] = (
            rearrange(window_valid, "n k -> 1 n k") & chain_valid
        )

        # ------------------------------------------------------------------
        # Step 1: f^ref — each atom sees all sibling atoms' pos+element
        # ------------------------------------------------------------------
        n_per_res = N_atom // N_token
        ref_pos_grouped: Float[torch.Tensor, "B N_atom ref_pos_flat"] = rearrange(
            ref_pos, "b (n a) d -> b n (a d)", a=n_per_res
        )[:, tok]
        ref_elem_grouped: Float[torch.Tensor, "B N_atom ref_elem_flat"] = rearrange(
            ref_element, "b (n a) d -> b n (a d)", a=n_per_res
        )[:, tok]
        f_ref: Float[torch.Tensor, "B N_atom f_ref_dim"] = torch.cat(
            [ref_pos_grouped, ref_elem_grouped], dim=-1
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
            rearrange(ref_pos, "b n d -> b n 1 d") - ref_pos[:, neighbor_idx]
        )

        # Step 6: v_lm = same-chain flag                 [B, N_atom, K]
        v_lm: Float[torch.Tensor, "B N_atom K"] = chain_valid.float()

        # Step 7: p_lm = proj(d_lm) * v_lm              [B, N_atom, K, d]
        p_lm: Float[torch.Tensor, "B N_atom K d"] = self.proj_d_vec(d_lm) * rearrange(
            v_lm, "b n k -> b n k 1"
        )

        # Step 8: p_lm += proj(1/(1+||d||²)) * v_lm
        inv_sq: Float[torch.Tensor, "B N_atom K 1"] = 1.0 / (
            1.0
            + rearrange(
                einsum(d_lm, d_lm, "b n k d, b n k d -> b n k"),
                "b n k -> b n k 1",
            )
        )
        p_lm = p_lm + self.proj_inv_sq(inv_sq) * rearrange(v_lm, "b n k -> b n k 1")

        # Step 9: p_lm += proj(v_lm) * v_lm
        p_lm = p_lm + self.proj_v(rearrange(v_lm, "b n k -> b n k 1")) * rearrange(
            v_lm, "b n k -> b n k 1"
        )

        # Step 10: p_lm += proj(ReLU(c_l)) + proj(ReLU(c_m))
        cl_proj: Float[torch.Tensor, "B N_atom d"] = self.proj_cl_pair(F.relu(c_l))
        cm_proj: Float[torch.Tensor, "B N_atom d"] = self.proj_cm_pair(F.relu(c_l))
        cm_nbr: Float[torch.Tensor, "B N_atom K d"] = cm_proj[:, neighbor_idx]
        p_lm = p_lm + rearrange(cl_proj, "b n d -> b n 1 d") + cm_nbr

        # Zero padding slots
        p_lm = p_lm * rearrange(valid_mask, "b n k -> b n k 1")
        p_skip: Float[torch.Tensor, "B N_atom K d"] = p_lm.clone()  # step 11

        # ------------------------------------------------------------------
        # Step 12: q_skip = c_l + proj(r_scaled)        [B, N_atom, m]
        # ------------------------------------------------------------------
        q_skip: Float[torch.Tensor, "B N_atom m"] = c_l + self.proj_r_scaled(r_scaled)

        # ------------------------------------------------------------------
        # Step 13: c_l += proj(LayerNorm(s_input[tok_idx]))
        # ------------------------------------------------------------------
        c_l = c_l + self.proj_s_init(self.norm_s_init(s_input[:, tok]))

        # ------------------------------------------------------------------
        # Step 14: p_lm += proj(LayerNorm(z_input[tok_l, tok_m]))
        # ------------------------------------------------------------------
        tok_nbr_shared: Int[torch.Tensor, "N_atom K"] = tok[neighbor_idx]
        tok_l_shared: Int[torch.Tensor, "N_atom K"] = repeat(tok, "n -> n k", k=K)
        z_gathered: Float[torch.Tensor, "B N_atom K c_pair"] = z_input[
            :, tok_l_shared, tok_nbr_shared
        ]
        p_lm = p_lm + self.proj_z_init(self.norm_z_init(z_gathered))

        # ------------------------------------------------------------------
        # Step 15: p_lm += MLP(p_lm)   residual 3-layer ReLU
        # ------------------------------------------------------------------
        p_lm = p_lm + self.mlp_p(p_lm)

        # ------------------------------------------------------------------
        # Step 16: AtomTransformer with 32 / 128 sparse window described in AF3
        # ------------------------------------------------------------------
        # this function signature has changed. it will also have changed in AtomDecoder.
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
        proj_q: Float[torch.Tensor, "B N_atom c"] = F.relu(self.proj_agg(q_skip))
        C = proj_q.size(-1)

        # Vectorized scatter: offset tok_idx by b*N_token to flatten B by N_token → (B*N_token)
        tok_offset: Int[torch.Tensor, "B N_atom"] = (
            tok_idx + repeat(torch.arange(B, device=tok_idx.device), "b -> b n", n=N_atom) * N_token
        )
        s_i_flat = torch.zeros(B * N_token, C, device=proj_q.device, dtype=proj_q.dtype)
        s_i_flat.scatter_add_(
            0,
            repeat(rearrange(tok_offset, "b n -> (b n)"), "bn -> bn c", c=C),
            rearrange(proj_q, "b n c -> (b n) c"),
        )
        counts_flat = torch.zeros(B * N_token, 1, device=proj_q.device, dtype=proj_q.dtype)
        counts_flat.scatter_add_(
            0,
            rearrange(tok_offset, "b n -> (b n) 1"),
            torch.ones(B * N_atom, 1, device=proj_q.device, dtype=proj_q.dtype),
        )
        s_i: Float[torch.Tensor, "B N_res c"] = rearrange(
            s_i_flat / counts_flat.clamp(min=1), "(b n) c -> b n c", b=B
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
        c_token: int,
        c_pair: int,
        c_atom: int,
        c_atompair: int,
        n_blocks: int = 3,
        n_heads: int = 4,
        window_size: int = WINDOW_SIZE,
    ) -> None:
        super().__init__()
        self.n_keys = window_size
        self.n_queries = window_size * 4

        self.norm_s_q = nn.LayerNorm(c_token)
        self.proj_s_q = LinearNoBias(c_token, c_atom)  # [c_token, c_atom]

        self.norm_z = nn.LayerNorm(c_pair)
        self.proj_z = LinearNoBias(c_pair, c_atompair)  # [c_pair, c_atompair]

        self.mlp_p = nn.Sequential(  # [c_atompair, c_atompair] each
            nn.ReLU(),
            LinearNoBias(c_atompair, c_atompair),
            nn.ReLU(),
            LinearNoBias(c_atompair, c_atompair),
            nn.ReLU(),
            LinearNoBias(c_atompair, c_atompair),
        )

        self.transformer = AtomTransformer(
            c_atom=c_atom,
            c_atompair=c_atompair,
            n_blocks=n_blocks,
            n_heads=n_heads,
            n_queries=window_size * 4,
            n_keys=window_size,
        )

        self.norm_q_out = nn.LayerNorm(c_atom)
        self.proj_r = LinearNoBias(c_atom, 3)  # [c_atom, 3]

        self.norm_s_c = nn.LayerNorm(c_token)
        self.proj_s_c = LinearNoBias(c_token, c_atom)  # [c_token, c_atom]

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
        """Args:.

            q_skip  : atom skip queries      [B, N_atom, c_atom]
            p_skip  : sparse pair skip       [B, N_atom, K, c_atompair]  (from encoder)
            c_skip  : atom skip context      [B, N_atom, c_atom]
            c       : atom context           [B, N_atom, c_atom]
            s       : trunk single embeds    [B, N_res, c_token]
            z       : trunk pair embeds      [B, N_res, N_res, c_pair]
            tok_idx : residue index per atom [B, N_atom]

        Returns:
            q        : atom query embeddings   [B, N_atom, c_atom]
            p        : sparse pair embeddings  [B, N_atom, K, c_atompair]
            r_update : per-atom position update  [B, N_atom, 3]
            c_out    : updated atom context      [B, N_atom, c_atom]
        """
        # tok_idx identical across batch items; use [0] for index builds
        tok = tok_idx[0]  # [N_atom]

        # Build sparse pair index (window only; chain info lives in p_skip)
        neighbor_idx: Int[torch.Tensor, "N_atom K"]
        window_valid: Bool[torch.Tensor, "N_atom K"]
        neighbor_idx, window_valid = build_sparse_pairs(tok, self.n_keys)
        K_built = neighbor_idx.size(1)
        B = tok_idx.size(0)
        valid_mask: Bool[torch.Tensor, "B N_atom K"] = repeat(window_valid, "n k -> b n k", b=B)

        # Step 1: q = proj(LayerNorm(s[tok_idx])) + q_skip   [B, N_atom, c_atom]
        q: Float[torch.Tensor, "B N_atom c_atom"] = self.proj_s_q(self.norm_s_q(s[:, tok])) + q_skip

        # Step 2: p = proj(LayerNorm(z[tok_l, tok_m])) + p_skip  [B, N_atom, K, c_atompair]
        tok_nbr_shared: Int[torch.Tensor, "N_atom K"] = tok[neighbor_idx]
        tok_l_shared: Int[torch.Tensor, "N_atom K"] = repeat(tok, "n -> n k", k=K_built)
        z_gathered: Float[torch.Tensor, "B N_atom K c_pair"] = z[:, tok_l_shared, tok_nbr_shared]
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
        r_update: Float[torch.Tensor, "B N_atom 3"] = self.proj_r(self.norm_q_out(q))

        # Step 6: c_out = proj(LayerNorm(s[tok_idx])) + c_skip
        c_out: Float[torch.Tensor, "B N_atom c_atom"] = (
            self.proj_s_c(self.norm_s_c(s[:, tok])) + c_skip
        )

        return q, p, r_update, c_out
