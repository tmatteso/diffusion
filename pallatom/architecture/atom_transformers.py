import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from einops import rearrange
from jaxtyping import Bool, Float, Int, jaxtyped

# Local attention window size (residues), matching AlphaFold 3 atom transformer.
# Each atom l attends only to atoms m where |tok_idx[l] - tok_idx[m]| < WINDOW_SIZE // 2,
# giving each atom at most WINDOW_SIZE // 2 neighbours on each side.
WINDOW_SIZE: int = 32


class LinearNoBias(nn.Linear):
    """Linear layer with no bias term."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=False)


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
    """
    For each atom l, collect indices of all atoms m within the 32-residue window.

    Uses an O(N²) boolean intermediate (one byte per pair) to build the index,
    then returns O(N × K) long + bool tensors where K << N for large proteins.

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
    diff = (tok_idx.unsqueeze(1) - tok_idx.unsqueeze(0)).abs()  # [N, N]
    in_window = diff < half                                       # [N, N]

    K = int(in_window.sum(dim=1).max().item())

    # For each row l, sort so valid neighbours appear first (out-of-window → sentinel N)
    col = torch.arange(N, device=tok_idx.device).unsqueeze(0).expand(N, -1)  # [N, N]
    sentinel = col.masked_fill(~in_window, N)
    neighbor_idx = sentinel.sort(dim=1).values[:, :K]   # [N, K], valid entries < N

    valid_mask   = neighbor_idx < N                      # [N, K]
    neighbor_idx = neighbor_idx.clamp(max=N - 1)        # padding slots → safe index 0

    return neighbor_idx, valid_mask


# ---------------------------------------------------------------------------
# AtomTransformerBlock — sparse pair-biased cross-attention
# ---------------------------------------------------------------------------

class AtomTransformerBlock(nn.Module):
    """
    Single block of the AtomTransformer with a 32-residue local attention window.

    All pair tensors are sparse [B, N, K, *] — no N² allocation.

    Weight shapes:
        to_q / to_k / to_v / to_out : [c_atom, c_atom]       (no bias)
        pair_bias                    : [c_atompair, n_heads]   (no bias)
        ff1                          : [c_atom, c_atom * 4]    (no bias)
        ff2                          : [c_atom * 4, c_atom]    (no bias)

    Sparse attention shapes (window_size=32, K = live neighbours per atom):
        Q, K_proj, V : [B, N, n_heads, head_dim]
        K_nbr, V_nbr : [B, N, K, n_heads, head_dim]  — gathered over K neighbours
        scores       : [B, N, K, n_heads]             — O(N × K), not O(N²)
        pair bias    : [B, N, K, n_heads]
    """

    def __init__(
        self,
        c_atom: int,
        c_atompair: int,
        n_heads: int,
        window_size: int = WINDOW_SIZE,
    ):
        super().__init__()
        assert c_atom % n_heads == 0, "c_atom must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = c_atom // n_heads
        self.window_size = window_size

        self.norm_q = nn.LayerNorm(c_atom)
        self.norm_c = nn.LayerNorm(c_atom)

        # [c_atom, c_atom] each — no bias, as in AF3
        self.to_q   = LinearNoBias(c_atom, c_atom)
        self.to_k   = LinearNoBias(c_atom, c_atom)
        self.to_v   = LinearNoBias(c_atom, c_atom)
        self.to_out = LinearNoBias(c_atom, c_atom)

        # [c_atompair, n_heads] — one scalar bias per head per live pair
        self.pair_bias = LinearNoBias(c_atompair, n_heads)
        self.norm_pair = nn.LayerNorm(c_atompair)

        # [c_atom, c_atom*4] then [c_atom*4, c_atom]
        self.norm_ff = nn.LayerNorm(c_atom)
        self.ff1 = LinearNoBias(c_atom, c_atom * 4)
        self.ff2 = LinearNoBias(c_atom * 4, c_atom)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        q:            Float[torch.Tensor, "B N c_atom"],
        c:            Float[torch.Tensor, "B N c_atom"],
        p:            Float[torch.Tensor, "B N K c_atompair"],
        neighbor_idx: Int[torch.Tensor, "N K"],
        valid_mask:   Bool[torch.Tensor, "B N K"],
    ) -> Float[torch.Tensor, "B N c_atom"]:
        """
        Args:
            q            : atom query embeddings      [B, N, c_atom]
            c            : atom context embeddings    [B, N, c_atom]
            p            : sparse pair embeddings     [B, N, K, c_atompair]
                           p[b, l, k] = features for pair (l, neighbor_idx[l, k])
            neighbor_idx : window neighbour indices   [N, K]
            valid_mask   : True = live pair           [B, N, K]
                           encodes both window and chain constraints

        Returns:
            q : updated atom embeddings               [B, N, c_atom]
        """
        H, D = self.n_heads, self.head_dim

        # Project to Q, K, V — each [B, N, n_heads, head_dim]
        Q:      Float[torch.Tensor, "B N n_heads head_dim"] = rearrange(self.to_q(self.norm_q(q)), "b n (h d) -> b n h d", h=H)
        K_proj: Float[torch.Tensor, "B N n_heads head_dim"] = rearrange(self.to_k(self.norm_c(c)), "b n (h d) -> b n h d", h=H)
        V:      Float[torch.Tensor, "B N n_heads head_dim"] = rearrange(self.to_v(self.norm_c(c)), "b n (h d) -> b n h d", h=H)

        # Gather K neighbours for keys and values — [B, N, K, n_heads, head_dim]
        # PyTorch advanced indexing: K_proj[:, neighbor_idx] with neighbor_idx [N, K]
        # broadcasts over B to give [B, N, K, H, D]
        K_nbr: Float[torch.Tensor, "B N K n_heads head_dim"] = K_proj[:, neighbor_idx]
        V_nbr: Float[torch.Tensor, "B N K n_heads head_dim"] = V[:, neighbor_idx]

        # Attention scores — [B, N, K, n_heads]
        scores: Float[torch.Tensor, "B N K n_heads"] = torch.einsum("b n h d, b n k h d -> b n k h", Q, K_nbr) / math.sqrt(D)

        # Pair bias — [B, N, K, n_heads]
        scores = scores + self.pair_bias(self.norm_pair(p))

        # Mask padding and out-of-window slots
        scores = scores.masked_fill(
            rearrange(~valid_mask, "b n k -> b n k 1"), float("-inf")
        )

        # Softmax over K neighbours (not N) — O(N × K) not O(N²)
        attn: Float[torch.Tensor, "B N K n_heads"] = F.softmax(scores, dim=2)

        # Weighted sum — [B, N, n_heads, head_dim] → [B, N, c_atom]
        out: Float[torch.Tensor, "B N n_heads head_dim"] = torch.einsum("b n k h, b n k h d -> b n h d", attn, V_nbr)
        out: Float[torch.Tensor, "B N c_atom"]           = rearrange(out, "b n h d -> b n (h d)")

        q = q + self.to_out(out)

        # Feed-forward
        q = q + self.ff2(F.relu(self.ff1(self.norm_ff(q))))

        return q


# ---------------------------------------------------------------------------
# AtomTransformer — stack of sparse blocks
# ---------------------------------------------------------------------------

class AtomTransformer(nn.Module):
    """
    Stack of n_blocks AtomTransformerBlocks, all operating on sparse [B, N, K, *] pairs.
    """

    def __init__(
        self,
        c_atom: int,
        c_atompair: int,
        n_blocks: int = 3,
        n_heads: int = 4,
        window_size: int = WINDOW_SIZE,
    ):
        super().__init__()
        self.window_size = window_size
        self.blocks = nn.ModuleList([
            AtomTransformerBlock(c_atom, c_atompair, n_heads, window_size=window_size)
            for _ in range(n_blocks)
        ])

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        q:            Float[torch.Tensor, "B N c_atom"],
        c:            Float[torch.Tensor, "B N c_atom"],
        p:            Float[torch.Tensor, "B N K c_atompair"],
        neighbor_idx: Int[torch.Tensor, "N K"],
        valid_mask:   Bool[torch.Tensor, "B N K"],
    ) -> Float[torch.Tensor, "B N c_atom"]:
        """
        Args:
            q            : atom query embeddings   [B, N, c_atom]
            c            : atom context embeddings [B, N, c_atom]
            p            : sparse pair embeddings  [B, N, K, c_atompair]
            neighbor_idx : neighbour atom indices  [N, K]
            valid_mask   : live-pair mask          [B, N, K]

        Returns:
            q : updated atom embeddings            [B, N, c_atom]
        """
        for block in self.blocks:
            q = block(q, c, p, neighbor_idx, valid_mask)
        return q


# ---------------------------------------------------------------------------
# AtomFeatureEncoder — Algorithm 4
# ---------------------------------------------------------------------------

class AtomFeatureEncoder(nn.Module):
    """
    Encodes per-atom reference features into per-residue embeddings.

    All N×N pair tensors (d_lm, p_lm, z_gathered) are replaced by sparse
    [B, N, K, *] tensors indexed over the N × K live pairs within the 32-residue
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
    ):
        super().__init__()
        self.m = m
        self.window_size = window_size

        self.proj_fref_c = LinearNoBias(f_ref_dim, m)          # [f_ref_dim, m]

        self.proj_d_vec   = LinearNoBias(3, d)                  # [3, d]
        self.proj_inv_sq  = LinearNoBias(1, d)                  # [1, d]
        self.proj_v       = LinearNoBias(1, d)                  # [1, d]
        self.proj_cl_pair = LinearNoBias(m, d)                  # [m, d]
        self.proj_cm_pair = LinearNoBias(m, d)                  # [m, d]

        self.proj_r_scaled = LinearNoBias(3, m)                 # [3, m]

        self.norm_s_init = nn.LayerNorm(c_token)
        self.proj_s_init = LinearNoBias(c_token, m)             # [c_token, m]

        self.norm_z_init = nn.LayerNorm(c_pair)
        self.proj_z_init = LinearNoBias(c_pair, d)              # [c_pair, d]

        self.mlp_p = nn.Sequential(                             # [d, d] each layer
            LinearNoBias(d, d), nn.ReLU(),
            LinearNoBias(d, d), nn.ReLU(),
            LinearNoBias(d, d), nn.ReLU(),
            LinearNoBias(d, d),
        )

        self.transformer = AtomTransformer(
            c_atom=m, c_atompair=d, n_blocks=n_blocks,
            n_heads=n_heads, window_size=window_size,
        )

        self.proj_agg = LinearNoBias(m, c)                      # [m, c]

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        ref_pos:       Float[torch.Tensor, "B N_atom 3"],
        ref_element:   Float[torch.Tensor, "B N_atom E"],
        ref_space_uid: Int[torch.Tensor,   "B N_atom"],
        s_input:       Float[torch.Tensor, "B N_res c_token"],
        z_input:       Float[torch.Tensor, "B N_res N_res c_pair"],
        r_scaled:      Float[torch.Tensor, "B N_atom 3"],
        tok_idx:       Int[torch.Tensor,   "B N_atom"],
    ) -> tuple[
        Float[torch.Tensor, "B N_res c_res"],
        Float[torch.Tensor, "B N_atom c_atom"],
        Float[torch.Tensor, "B N_atom c_atom"],
        Float[torch.Tensor, "B N_atom K c_atompair"],
        Float[torch.Tensor, "B N_atom c_atom"],
    ]:
        """
        Args:
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
        B       = ref_pos.size(0)
        N_atom  = ref_pos.size(1)
        N_token = s_input.size(1)

        # tok_idx is identical across batch items; use [0] for index builds
        tok = tok_idx[0]  # [N_atom]

        # ------------------------------------------------------------------
        # Build sparse pair index — O(N²) bool then O(N × K) permanently
        # ------------------------------------------------------------------
        neighbor_idx: Int[torch.Tensor, "N_atom K"]
        window_valid: Bool[torch.Tensor, "N_atom K"]
        neighbor_idx, window_valid = build_sparse_pairs(tok, self.window_size)
        K = neighbor_idx.size(1)

        # Chain constraint: same ref_space_uid per batch item — [B, N_atom, K]
        chain_valid: Bool[torch.Tensor, "B N_atom K"] = (
            ref_space_uid.unsqueeze(-1) == ref_space_uid[:, neighbor_idx]
        )
        valid_mask: Bool[torch.Tensor, "B N_atom K"] = window_valid.unsqueeze(0) & chain_valid

        # ------------------------------------------------------------------
        # Step 1: f^ref — each atom sees all sibling atoms' pos+element
        # ------------------------------------------------------------------
        n_per_res = N_atom // N_token
        ref_pos_grouped:  Float[torch.Tensor, "B N_atom ref_pos_flat"]  = rearrange(ref_pos,     "b (n a) d -> b n (a d)", a=n_per_res)[:, tok]
        ref_elem_grouped: Float[torch.Tensor, "B N_atom ref_elem_flat"] = rearrange(ref_element, "b (n a) d -> b n (a d)", a=n_per_res)[:, tok]
        f_ref: Float[torch.Tensor, "B N_atom f_ref_dim"] = torch.cat([ref_pos_grouped, ref_elem_grouped], dim=-1)

        # ------------------------------------------------------------------
        # Step 2: c_l = LinearNoBias(f^ref)              [B, N_atom, m]
        # ------------------------------------------------------------------
        c_l:    Float[torch.Tensor, "B N_atom m"] = self.proj_fref_c(f_ref)
        c_skip: Float[torch.Tensor, "B N_atom m"] = c_l.clone()                  # step 4

        # ------------------------------------------------------------------
        # Steps 5-10: sparse atom-pair embeddings p_lm   [B, N_atom, K, d]
        # ------------------------------------------------------------------

        # Step 5: d_lm = ref_pos[b, l] − ref_pos[b, m]  [B, N_atom, K, 3]
        d_lm: Float[torch.Tensor, "B N_atom K 3"] = (
            rearrange(ref_pos, "b n d -> b n 1 d") - ref_pos[:, neighbor_idx]
        )

        # Step 6: v_lm = same-chain flag                 [B, N_atom, K]
        v_lm: Float[torch.Tensor, "B N_atom K"] = chain_valid.float()

        # Step 7: p_lm = proj(d_lm) * v_lm              [B, N_atom, K, d]
        p_lm: Float[torch.Tensor, "B N_atom K d"] = self.proj_d_vec(d_lm) * rearrange(v_lm, "b n k -> b n k 1")

        # Step 8: p_lm += proj(1/(1+||d||²)) * v_lm
        inv_sq: Float[torch.Tensor, "B N_atom K 1"] = 1.0 / (
            1.0 + torch.einsum("b n k d, b n k d -> b n k", d_lm, d_lm).unsqueeze(-1)
        )
        p_lm = p_lm + self.proj_inv_sq(inv_sq) * rearrange(v_lm, "b n k -> b n k 1")

        # Step 9: p_lm += proj(v_lm) * v_lm
        p_lm = p_lm + self.proj_v(rearrange(v_lm, "b n k -> b n k 1")) * rearrange(v_lm, "b n k -> b n k 1")

        # Step 10: p_lm += proj(ReLU(c_l)) + proj(ReLU(c_m))
        cl_proj: Float[torch.Tensor, "B N_atom d"]   = self.proj_cl_pair(F.relu(c_l))
        cm_proj: Float[torch.Tensor, "B N_atom d"]   = self.proj_cm_pair(F.relu(c_l))
        cm_nbr:  Float[torch.Tensor, "B N_atom K d"] = cm_proj[:, neighbor_idx]
        p_lm    = p_lm + rearrange(cl_proj, "b n d -> b n 1 d") + cm_nbr

        # Zero padding slots
        p_lm   = p_lm * rearrange(valid_mask, "b n k -> b n k 1")
        p_skip: Float[torch.Tensor, "B N_atom K d"] = p_lm.clone()               # step 11

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
        tok_l_shared:   Int[torch.Tensor, "N_atom K"] = tok.unsqueeze(1).expand(-1, K)
        z_gathered: Float[torch.Tensor, "B N_atom K c_pair"] = z_input[:, tok_l_shared, tok_nbr_shared]
        p_lm = p_lm + self.proj_z_init(self.norm_z_init(z_gathered))

        # ------------------------------------------------------------------
        # Step 15: p_lm += MLP(p_lm)   residual 3-layer ReLU
        # ------------------------------------------------------------------
        p_lm = p_lm + self.mlp_p(p_lm)

        # ------------------------------------------------------------------
        # Step 16: AtomTransformer with 32-residue sparse window
        # ------------------------------------------------------------------
        q_skip: Float[torch.Tensor, "B N_atom m"] = self.transformer(
            q_skip, c_l, p_lm, neighbor_idx, valid_mask,
        )

        # ------------------------------------------------------------------
        # Step 17: s_i = mean_{l: tok_idx(l)=i} ReLU(proj(q_skip))
        # ------------------------------------------------------------------
        proj_q: Float[torch.Tensor, "B N_atom c"] = F.relu(self.proj_agg(q_skip))
        C = proj_q.size(-1)

        # Vectorized scatter: offset tok_idx by b*N_token to flatten B×N_token → (B*N_token)
        tok_offset: Int[torch.Tensor, "B N_atom"] = (
            tok_idx + torch.arange(B, device=tok_idx.device).unsqueeze(1) * N_token
        )
        s_i_flat = torch.zeros(B * N_token, C, device=proj_q.device, dtype=proj_q.dtype)
        s_i_flat.scatter_add_(
            0,
            rearrange(tok_offset, "b n -> (b n)").unsqueeze(1).expand(-1, C),
            rearrange(proj_q, "b n c -> (b n) c"),
        )
        counts_flat = torch.zeros(B * N_token, 1, device=proj_q.device, dtype=proj_q.dtype)
        counts_flat.scatter_add_(
            0,
            rearrange(tok_offset, "b n -> (b n)").unsqueeze(1),
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
    """
    Decodes trunk embeddings back to per-atom position updates.

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
    ):
        super().__init__()
        self.window_size = window_size

        self.norm_s_q = nn.LayerNorm(c_token)
        self.proj_s_q = LinearNoBias(c_token, c_atom)   # [c_token, c_atom]

        self.norm_z   = nn.LayerNorm(c_pair)
        self.proj_z   = LinearNoBias(c_pair, c_atompair) # [c_pair, c_atompair]

        self.mlp_p = nn.Sequential(                      # [c_atompair, c_atompair] each
            LinearNoBias(c_atompair, c_atompair), nn.ReLU(),
            LinearNoBias(c_atompair, c_atompair), nn.ReLU(),
            LinearNoBias(c_atompair, c_atompair), nn.ReLU(),
            LinearNoBias(c_atompair, c_atompair),
        )

        self.transformer = AtomTransformer(
            c_atom=c_atom, c_atompair=c_atompair,
            n_blocks=n_blocks, n_heads=n_heads, window_size=window_size,
        )

        self.norm_q_out = nn.LayerNorm(c_atom)
        self.proj_r     = LinearNoBias(c_atom, 3)        # [c_atom, 3]

        self.norm_s_c = nn.LayerNorm(c_token)
        self.proj_s_c = LinearNoBias(c_token, c_atom)    # [c_token, c_atom]

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        q_skip:  Float[torch.Tensor, "B N_atom c_atom"],
        p_skip:  Float[torch.Tensor, "B N_atom K c_atompair"],
        c_skip:  Float[torch.Tensor, "B N_atom c_atom"],
        c:       Float[torch.Tensor, "B N_atom c_atom"],
        s:       Float[torch.Tensor, "B N_res c_token"],
        z:       Float[torch.Tensor, "B N_res N_res c_pair"],
        tok_idx: Int[torch.Tensor,   "B N_atom"],
    ) -> tuple[
        Float[torch.Tensor, "B N_atom 3"],
        Float[torch.Tensor, "B N_atom c_atom"],
    ]:
        """
        Args:
            q_skip  : atom skip queries      [B, N_atom, c_atom]
            p_skip  : sparse pair skip       [B, N_atom, K, c_atompair]  (from encoder)
            c_skip  : atom skip context      [B, N_atom, c_atom]
            c       : atom context           [B, N_atom, c_atom]
            s       : trunk single embeds    [B, N_res, c_token]
            z       : trunk pair embeds      [B, N_res, N_res, c_pair]
            tok_idx : residue index per atom [B, N_atom]

        Returns:
            r_update : per-atom position update  [B, N_atom, 3]
            c_out    : updated atom context      [B, N_atom, c_atom]
        """
        # tok_idx identical across batch items; use [0] for index builds
        tok = tok_idx[0]  # [N_atom]

        # Build sparse pair index (window only; chain info lives in p_skip)
        neighbor_idx: Int[torch.Tensor, "N_atom K"]
        window_valid: Bool[torch.Tensor, "N_atom K"]
        neighbor_idx, window_valid = build_sparse_pairs(tok, self.window_size)
        K_built = neighbor_idx.size(1)
        B = tok_idx.size(0)
        valid_mask: Bool[torch.Tensor, "B N_atom K"] = window_valid.unsqueeze(0).expand(B, -1, -1)

        # Step 1: q = proj(LayerNorm(s[tok_idx])) + q_skip   [B, N_atom, c_atom]
        q: Float[torch.Tensor, "B N_atom c_atom"] = self.proj_s_q(self.norm_s_q(s[:, tok])) + q_skip

        # Step 2: p = proj(LayerNorm(z[tok_l, tok_m])) + p_skip  [B, N_atom, K, c_atompair]
        tok_nbr_shared: Int[torch.Tensor, "N_atom K"] = tok[neighbor_idx]
        tok_l_shared:   Int[torch.Tensor, "N_atom K"] = tok.unsqueeze(1).expand(-1, K_built)
        z_gathered: Float[torch.Tensor, "B N_atom K c_pair"]     = z[:, tok_l_shared, tok_nbr_shared]
        p:          Float[torch.Tensor, "B N_atom K c_atompair"] = self.proj_z(self.norm_z(z_gathered)) + p_skip

        # Step 3: p += MLP(p) residual; zero padding slots
        p = (p + self.mlp_p(p)) * rearrange(valid_mask, "b n k -> b n k 1")

        # Step 4: AtomTransformer — 32-residue sparse window
        q: Float[torch.Tensor, "B N_atom c_atom"] = self.transformer(
            q, c, p, neighbor_idx, valid_mask,
        )

        # Step 5: r_update = proj(LayerNorm(q))             [B, N_atom, 3]
        r_update: Float[torch.Tensor, "B N_atom 3"] = self.proj_r(self.norm_q_out(q))

        # Step 6: c_out = proj(LayerNorm(s[tok_idx])) + c_skip
        c_out: Float[torch.Tensor, "B N_atom c_atom"] = self.proj_s_c(self.norm_s_c(s[:, tok])) + c_skip

        return r_update, c_out
