# pallatom conventions

## Testing

- All files outside of the data folder must have tests in pallatom/tests. Do not write smoke tests outside of the pallatom/tests directory. Do not write smoke tests in ephemeral bash scripts.

## Type checking and shape contracts

- Annotate helper functions with **`jaxtyping`** (`Float`, `Int`, `Bool`, etc.) and `@jaxtyped(typechecker=beartype)` so shape contracts are verified at call time.
- Use named dimensions in jaxtyping annotations (e.g. `"N_atom 3"`, `"N_res c_res"`) to make shape intent explicit.

## Tensor operations

- Replace all `@` matrix multiplications and `torch.matmul` calls with **`einops.einsum`**. The einsum string makes the contraction axes explicit and self-documenting.
- Use **`einops.rearrange`** instead of `view`, `reshape`, `unsqueeze`, `squeeze`, or `permute`.
- Use **`einops.reduce`** instead of `torch.sum`, `torch.mean`, `torch.max`, etc. when reducing over named axes.
- Use **`einops.repeat`** instead of `expand`, `repeat`, or `tile`.
- Prefer einops operations throughout helper functions so shape contracts interact naturally with jaxtyping annotations.

---

## MainTrunk tensor reference

All tensors are **unbatched** (no leading `B` dimension) unless otherwise noted. The batch dimension is squeezed/unsqueezed at the `TemplateEmbedder` boundary.

### Named dimensions

| Symbol      | Meaning                                                                 | Typical value |
|-------------|-------------------------------------------------------------------------|---------------|
| `N_atom`    | Total atoms in the packed sequence                                      | varies        |
| `N_res`     | Number of residues / tokens (`N_token` in some comments)               | varies        |
| `K`         | Sparse neighbour count per atom (32-residue local window)               | ≤ 448         |
| `c_res`     | Residue single-embedding dim                                            | 256           |
| `c_pair`    | Trunk pair-embedding dim                                                | 128           |
| `c_atom`    | Atom single-embedding dim                                               | 128           |
| `c_atompair`| Atom-pair embedding dim                                                 | 16            |
| `E`         | Element feature dim (C, N, O, UNK one-hot)                              | 4             |
| `n_bins`    | Distogram distance bins                                                 | 38            |
| `n_amino`   | Amino-acid vocabulary size                                              | 20            |

### Inputs

| Tensor               | jaxtyping annotation                                      | Description |
|----------------------|-----------------------------------------------------------|-------------|
| `ref_pos`            | `Float[Tensor, "N_atom 3"]`                               | Reference atom positions from the ground-truth structure, used to build initial atom-pair features. |
| `ref_element`        | `Float[Tensor, "N_atom E"]`                               | Float one-hot element identity per atom (C / N / O / UNK), concatenated into `f_ref` inside `AtomFeatureEncoder`. |
| `ref_space_uid`      | `Int[Tensor, "N_atom"]`                                   | Chain / space identifier per atom; controls which atom pairs are considered covalently bonded for relative position encoding. |
| `f_distogram`        | `Float[Tensor, "N_res N_res n_bins"]`                     | Template pairwise Cβ distance distribution over `n_bins` bins, one matrix per structure. |
| `f_pseudo_beta_mask` | `Float[Tensor, "N_res"]`                                  | Binary mask (0 / 1) indicating residues that have a valid pseudo-β carbon in the template. |
| `f_residue_idx`      | `Float[Tensor, "N_res c_res"]`                            | Sinusoidal encoding of the per-residue index, projected to `c_res` dims to seed `s_init`. |
| `r_input`            | `Float[Tensor, "N_atom 3"]`                               | Noisy atom positions at the current diffusion step (input to the denoiser). |
| `t_hat`              | `float`                                                   | Noise level σ of the input noise (scalar, not a tensor). |
| `t`                  | `float`                                                   | Diffusion time in `[0, 1)` passed to `TemplateEmbedder` for time-conditional template weighting. |
| `tok_idx`            | `Int[Tensor, "N_atom"]`                                   | Maps each atom to its parent residue index in `[0, N_res)`. |
| `center_uid`         | `Int[Tensor, "N_atom"]`                                   | For each atom, the index of its residue's center atom; broadcast per-residue center into the atom dimension, used in step 15 to extract `r_center`. |

### Key intermediate tensors

| Tensor       | jaxtyping annotation                             | Description |
|--------------|--------------------------------------------------|-------------|
| `r_scaled`   | `Float[Tensor, "N_atom 3"]`                      | `r_input` normalised by `sqrt(σ_data² + t̂²)`; fed into `AtomFeatureEncoder` as the conditioning signal. |
| `s_init`     | `Float[Tensor, "N_res c_res"]`                   | Initial residue single embedding: projected `f_residue_idx` plus time Fourier embedding `t_i`. |
| `t_i`        | `Float[Tensor, "N_res c_res"]`                   | Time Fourier embedding of `¼·log(t̂/σ_data)`, broadcast to every residue. |
| `z_ij`       | `Float[Tensor, "N_res N_res c_pair"]`            | Trunk pair embedding; initialised from relative position encoding + template embedding, then updated by `PairUpdate` each decoder unit. |
| `q_skip`     | `Float[Tensor, "N_atom c_atom"]`                 | Atom single embedding from the last encoder block, passed as a skip connection into every `AtomAttentionDecoder`. |
| `c_skip`     | `Float[Tensor, "N_atom c_atom"]`                 | Atom context embedding from the encoder (distinct from `q_skip`), also skip-connected to each decoder. |
| `p_skip`     | `Float[Tensor, "N_atom K c_atompair"]`           | Sparse atom-pair embedding from the encoder (local `K`-neighbour window), skip-connected to each decoder and used by `AtomDistogramHead`. |
| `c_l`        | `Float[Tensor, "N_atom c_atom"]`                 | Atom-level single embedding updated by each `AtomAttentionDecoder` block; accumulates cross-residue context. |
| `s_i`        | `Float[Tensor, "N_res c_res"]`                   | Residue single embedding updated by each `NodeUpdate` block in the decoder loop. |
| `r_updates`  | `Float[Tensor, "N_atom 3"]`                      | Accumulated 3D position updates from all decoder units; combined with `r_input` to form `r_denoised`. |
| `r_center`   | `Float[Tensor, "N_res 3"]`                       | Denoised position of the center atom per residue (`r_denoised[center_uid]`); fed into `PairUpdate` to refresh `z_ij`. |
| `a_i`        | `Float[Tensor, "N_res c_res"]`                   | Per-residue mean of projected `q_skip` atom features (pooled with `tok_idx`); input to the sequence head. |

### Outputs

| Tensor                              | jaxtyping annotation                             | Description |
|-------------------------------------|--------------------------------------------------|-------------|
| `r_denoised`                        | `Float[Tensor, "N_atom 3"]`                      | Denoised atom positions at the current diffusion step. |
| `f_seq_logits`                      | `Float[Tensor, "N_res n_amino"]`                 | Amino-acid sequence logits (20 classes) derived from pooled atom features. |
| `residue_distogram_logits`          | `Float[Tensor, "N_res N_res n_bins"]`            | Pairwise residue Cβ distance bin logits predicted from the symmetrised trunk pair embedding `z_ij`. |
| `atom_distogram_logits`             | `Float[Tensor, "N_atom K n_bins"]`               | Pairwise atom distance bin logits, sparse over the local `K`-neighbour window. |
| `intermediate_denoised_coord_stack` | `list[Float[Tensor, "N_atom 3"]]`                | Denoised coordinates after each decoder unit (length `K_unit`); used for auxiliary loss. |
| `intermediate_pred_aa_logit_stack`  | `list[Float[Tensor, "N_atom c_atom"]]`           | Atom-level single features `c_l` after each decoder unit (length `K_unit`); used for auxiliary loss. |
