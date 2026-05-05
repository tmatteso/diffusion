Architecture
============

Visual overview of the **pallatom** denoising network.
Diagrams follow the ``forward()`` call order; tensor shapes use the
named-dimension conventions documented in ``pallatom/CLAUDE.md``
(e.g. ``N_atom``, ``N_res``, ``c_pair``, ``K``).

MainTrunk (Algorithm 2)
-----------------------

:class:`~pallatom.architecture.main_trunk.MainTrunk` takes a
:class:`~pallatom.helpers.featurize.FeaturizedBatch` and returns denoised atom
positions plus auxiliary sequence and distogram logits.

.. mermaid::

   flowchart TD
       classDef inp  fill:#dbeafe,stroke:#3b82f6,color:#1e293b,font-size:13px,padding:8px
       classDef proc fill:#d1fae5,stroke:#059669,color:#1e293b,font-size:13px,padding:8px
       classDef skip fill:#fef3c7,stroke:#d97706,color:#1e293b,font-size:13px,padding:8px
       classDef out  fill:#fce7f3,stroke:#db2777,color:#1e293b,font-size:13px,padding:8px

       I1(["r_input  [B, N_atom, 3]"]):::inp
       I2(["f_residue_idx  [B, N_res, c_res]"]):::inp
       I3(["f_distogram · pseudo_beta_mask"]):::inp
       I4(["ref_pos · ref_element · ref_uid"]):::inp
       I5(["t_hat · t_normalized"]):::inp

       I1  --> P1["r_scaled = r_input / sqrt(sigma_data^2 + t_hat^2)"]:::proc
       I2  --> P2["s_init = Linear(f_residue_idx)"]:::proc
       I5  --> P3["t_i = TimeFourierEmbedding(0.25 * log(t_hat / sigma_data))"]:::proc
       P2  --> P4["s_init += t_i"]:::proc
       P3  --> P4

       P4  --> P5["z_ij = RelativePositionEncoding"]:::proc
       I3  --> P6["z_ij += TemplateEmbedder(f_distogram, z_ij, t)"]:::proc
       P5  --> P6

       P1  --> P7["AtomFeatureEncoder  (Algorithm 4)"]:::proc
       I4  --> P7
       P4  --> P7
       P6  --> P7
       P7  --> SK1(["s_i  [B, N_res, c_res]"]):::skip
       P7  --> SK2(["q_skip · c_skip  [B, N_atom, c_atom]"]):::skip
       P7  --> SK3(["p_skip  [B, N_atom, K, c_atompair]"]):::skip
       P7  --> SK4(["c_l  [B, N_atom, c_atom]"]):::skip

       SK1 --> P8["s_i += proj(LN(s_init))"]:::proc
       P4  --> P8

       P8  --> DL["Decoder Loop x K_unit"]:::proc
       SK2 --> DL
       SK4 --> DL
       P6  --> DL

       DL  --> O1(["r_denoised  [B, N_atom, 3]"]):::out
       DL  --> O2(["f_seq_logits  [B, N_res, 20]"]):::out
       DL  --> O3(["residue_distogram_logits  [B, N_res, N_res, n_bins]"]):::out
       SK3 --> O4(["atom_distogram_logits  [B, N_atom, K, n_atom_bins]"]):::out

----

Decoder Loop (x K\_unit)
------------------------

Each iteration of the ``K_unit`` loop refines all three representations:
residue single ``s_i``, pair ``z_ij``, and atom context ``c_l``.
The EDM blend formula ensures the denoised coordinates are always on the
correct noise-level manifold.

.. mermaid::

   flowchart TD
       classDef proc fill:#d1fae5,stroke:#059669,color:#1e293b,font-size:13px,padding:8px
       classDef acc  fill:#fef3c7,stroke:#d97706,color:#1e293b,font-size:13px,padding:8px
       classDef out  fill:#fce7f3,stroke:#db2777,color:#1e293b,font-size:13px,padding:8px

       IN(["s_i · z_ij · t_i · q_skip · c_skip · p_skip · c_l\nr_input · r_updates = 0"]):::acc

       IN  --> NU["NodeUpdate\n  s_i = Update(s_i, t_i, z_ij)"]:::proc
       NU  --> AD["AtomAttentionDecoder\n  r_update, c_l = Decode(q_skip, p_skip, c_skip, c_l, s_i, z_ij)"]:::proc
       AD  --> RU["r_updates += r_update"]:::acc
       RU  --> RD["r_denoised = EDM blend\n  sigma^2 * r_input + sigma*t_hat * r_updates\n  ——————————————\n  sigma^2 + t_hat^2"]:::acc
       RD  --> IH(["intermediate stack\n  r_denoised_k · aa_logits_k"]):::out
       RD  --> RC["r_center = r_denoised at center_uid"]:::acc
       RC  --> PU["PairUpdate\n  z_ij = Update(z_ij, r_center)"]:::proc
       PU  -.->|"repeat for next k"| NU

       PU  --> FN(["final: r_denoised · z_ij · q_skip"]):::acc

----

AtomFeatureEncoder (Algorithm 4)
---------------------------------

Builds per-atom and per-residue embeddings using a sparse
:class:`~pallatom.architecture.atom_transformers.AtomTransformer`.
All pair tensors are ``[B, N_atom, K, *]`` — the ``N_atom x N_atom`` dense
grid is never materialised.

.. mermaid::

   flowchart TD
       classDef inp  fill:#dbeafe,stroke:#3b82f6,color:#1e293b,font-size:13px,padding:8px
       classDef proc fill:#d1fae5,stroke:#059669,color:#1e293b,font-size:13px,padding:8px
       classDef skip fill:#fef3c7,stroke:#d97706,color:#1e293b,font-size:13px,padding:8px
       classDef out  fill:#fce7f3,stroke:#db2777,color:#1e293b,font-size:13px,padding:8px

       A1(["ref_pos · ref_element"]):::inp
       A2(["s_input  [B, N_res, c_res]"]):::inp
       A3(["z_input  [B, N_res, N_res, c_pair]"]):::inp
       A4(["r_scaled  [B, N_atom, 3]"]):::inp
       A5(["tok_idx  [B, N_atom]"]):::inp

       A5  --> SP["Build sparse pairs\ntok_idx -> N x K neighbor index\n(32-residue local window)"]:::proc

       A1  --> FR["f_ref = tile(ref_pos, ref_element) per atom"]:::proc
       FR  --> CL["c_l = Linear(f_ref)  [B, N_atom, c_atom]"]:::proc
       CL  --> CS(["c_skip  saved"]):::skip

       CL & A4 --> QS["q_skip = c_l + proj(r_scaled)"]:::proc
       QS  --> QSS(["q_skip  saved"]):::skip

       SP & CL --> PM["p_lm = atom-pair features\n(distance · chain validity · c_l projections)"]:::proc
       A3  --> PM

       A2 & CL --> CL2["c_l += proj(LN(s_input[tok_idx]))"]:::proc
       PM  --> PML["p_lm += proj(LN(z_input[tok_l, tok_m]))\np_lm += MLP(p_lm)"]:::proc
       PML --> PS(["p_skip  saved"]):::skip

       QSS & CL2 & PML & SP --> AT["AtomTransformer\n3 x AtomTransformerBlock\n(sparse K-neighbor attention)"]:::proc

       AT  --> MP["s_i = mean-pool(ReLU(proj(q_skip)), tok_idx)"]:::proc

       MP  --> O1(["s_i  [B, N_res, c_res]"]):::out
       AT  --> O2(["q_skip  [B, N_atom, c_atom]"]):::out
       CS  --> O3(["c_skip  [B, N_atom, c_atom]"]):::out
       PS  --> O4(["p_skip  [B, N_atom, K, c_atompair]"]):::out
       CL2 --> O5(["c_l  [B, N_atom, c_atom]"]):::out
