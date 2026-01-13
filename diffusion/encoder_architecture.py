import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.init import trunc_normal_
from torch.nn.utils import weight_norm

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention using PyTorch's native Flash Attention via SDPA.

    Uses scaled_dot_product_attention with Flash Attention backend for
    efficient computation with reduced memory usage.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        """
        Args:
            dim: Input/output dimension
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in QKV projection
            attn_drop: Dropout rate for attention weights
            proj_drop: Dropout rate for output projection
        """
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # Attention dropout
        self.attn_drop = attn_drop

        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            attn_mask: Optional attention mask [batch_size, seq_len, seq_len]

        Returns:
            Output tensor [batch_size, seq_len, dim]
        """
        B, N, C = x.shape

        # Project to Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, D]
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is [B, H, N, D]

        # Use PyTorch's scaled_dot_product_attention with Flash Attention backend
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop if self.training else 0.0,
                scale=self.scale,
            )

        # Reshape back to [batch_size, seq_len, dim]
        out = out.transpose(1, 2).reshape(B, N, C)

        # Output projection
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class MLP(nn.Module):
    """
    Feed-Forward Network (MLP) with GELU activation.

    Standard two-layer MLP with expansion ratio (typically 4x).
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        activation: nn.Module = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ):
        """
        Args:
            in_features: Input dimension
            hidden_features: Hidden dimension (if None, 4x in_features)
            out_features: Output dimension (if None, same as in_features)
            activation: Activation function
            drop: Dropout rate
            bias: Whether to use bias in linear layers
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = activation()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, in_features]

        Returns:
            Output tensor [batch_size, seq_len, out_features]
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.

    Used for regularization in deep networks by randomly dropping
    entire residual branches during training.
    """

    def __init__(self, drop_prob: float = 0.0):
        """
        Args:
            drop_prob: Probability of dropping a path
        """
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor

        Returns:
            Output tensor with paths randomly dropped during training
        """
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}"


class TransformerBlock(nn.Module):
    """
    Transformer Block with Multi-Head Self-Attention and Feed-Forward Network.

    Uses Flash Attention via PyTorch's scaled_dot_product_attention for efficiency.
    Follows pre-norm architecture with residual connections.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        activation: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        """
        Args:
            dim: Input/output dimension
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dim to embedding dim
            qkv_bias: Whether to use bias in QKV projection
            drop: Dropout rate for MLP
            attn_drop: Dropout rate for attention
            drop_path: Stochastic depth rate
            activation: Activation function for MLP
            norm_layer: Normalization layer
        """
        super().__init__()

        # Layer norms
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        # Multi-Head Self-Attention
        self.attn = MultiHeadSelfAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        # Stochastic Depth (DropPath)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Feed-Forward Network
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            activation=activation,
            drop=drop,
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            attn_mask: Optional attention mask

        Returns:
            Output tensor [batch_size, seq_len, dim]
        """
        # Pre-norm: LayerNorm before attention/MLP
        x = x + self.drop_path(self.attn(self.norm1(x), attn_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x



class ResidueEmbedding(nn.Module):
    """
    Convert protein structure to per-residue embeddings.

    Projects each residue's atom coordinates to an embedding vector.
    Input shape: [batch_size, n_residues, n_atoms, 3]
    Output shape: [batch_size, n_residues, embed_dim]
    """

    def __init__(
        self,
        n_atoms: int = 37,
        atom_dim: int = 3,
        embed_dim: int = 768,
    ):
        """
        Args:
            n_atoms: Number of atoms per residue (default 37 for all-atom)
            atom_dim: Dimension per atom (default 3 for x,y,z coordinates)
            embed_dim: Output embedding dimension
        """
        super().__init__()
        self.n_atoms = n_atoms
        self.atom_dim = atom_dim
        self.embed_dim = embed_dim

        # Input dimension: n_atoms * atom_dim
        self.input_dim = n_atoms * atom_dim

        # Linear projection from flattened residue to embedding
        self.proj = nn.Linear(self.input_dim, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Protein structure [n_residues, n_atoms, 3]

        Returns:
            embeddings: Residue embeddings [n_residues, embed_dim]
        """
        N, A, D = x.shape
        assert A == self.n_atoms, f"Expected {self.n_atoms} atoms, got {A}"
        assert D == self.atom_dim, f"Expected {self.atom_dim} dimensions, got {D}"

        # Flatten each residue: [N, A * D]
        x = x.reshape(N, self.input_dim)

        # Project to embedding dimension
        x = self.proj(x)  # [N, embed_dim]

        return x


def pack_sequences(
    sequences: list[torch.Tensor],
    padding_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """
    Pack multiple variable-length sequences into a single sequence (Krell et al. 2022).

    Instead of padding each sequence to max_len and processing them separately,
    concatenate all sequences and use attention masks to prevent cross-sequence attention.
    This reduces wasted computation on padding tokens.

    Args:
        sequences: List of tensors, each of shape [seq_len_i, dim]
        padding_value: Value to use for any necessary padding

    Returns:
        packed: Concatenated sequences [total_len, dim]
        attention_mask: Block diagonal mask [total_len, total_len]
                       False = attend, True = don't attend
        lengths: List of sequence lengths for unpacking
    """
    lengths = [seq.shape[0] for seq in sequences]
    total_len = sum(lengths)
    dim = sequences[0].shape[-1]

    # Concatenate all sequences
    packed = torch.cat(sequences, dim=0)  # [total_len, dim]

    # Create block diagonal attention mask
    # mask[i, j] = True means position i cannot attend to position j
    attention_mask = torch.ones(total_len, total_len, dtype=torch.bool, device=packed.device)

    start_idx = 0
    for length in lengths:
        end_idx = start_idx + length
        # Allow attention within this sequence
        attention_mask[start_idx:end_idx, start_idx:end_idx] = False
        start_idx = end_idx

    return packed, attention_mask, lengths


def unpack_sequences(
    packed: torch.Tensor,
    lengths: list[int],
) -> list[torch.Tensor]:
    """
    Unpack concatenated sequences back into individual sequences.

    Args:
        packed: Packed sequences [total_len, dim]
        lengths: List of original sequence lengths

    Returns:
        sequences: List of tensors, each of shape [seq_len_i, dim]
    """
    sequences = []
    start_idx = 0
    for length in lengths:
        end_idx = start_idx + length
        sequences.append(packed[start_idx:end_idx])
        start_idx = end_idx

    return sequences


def pack_protein_batch(
    structures: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """
    Pack a batch of variable-length protein structures using sequence packing.

    Args:
        structures: List of protein structures, each [n_residues_i, n_atoms, 3]

    Returns:
        packed: Concatenated structures [total_residues, n_atoms, 3]
        attention_mask: Block diagonal mask [total_residues, total_residues]
        lengths: List of sequence lengths for unpacking
    """
    lengths = [s.shape[0] for s in structures]
    total_len = sum(lengths)
    n_atoms = structures[0].shape[1]
    atom_dim = structures[0].shape[2]

    # Concatenate all structures
    packed = torch.cat(structures, dim=0)  # [total_residues, n_atoms, 3]

    # Create block diagonal attention mask
    attention_mask = torch.ones(total_len, total_len, dtype=torch.bool, device=packed.device)

    start_idx = 0
    for length in lengths:
        end_idx = start_idx + length
        # Allow attention within this protein
        attention_mask[start_idx:end_idx, start_idx:end_idx] = False
        start_idx = end_idx

    return packed, attention_mask, lengths


class ProjectionHead(nn.Module):
    """
    Projection head for DINOv2 (3-layer MLP with bottleneck).

    Projects embeddings to a lower-dimensional space for computing
    contrastive losses. Follows the DINOv2 architecture:
    - Three linear layers with GELU activation
    - Bottleneck dimension before final projection
    - L2 normalization before the last layer
    - Weight-normalized final layer
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        activation: nn.Module = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ):
        """
        Args:
            input_dim: Input dimension (embed_dim from backbone)
            output_dim: Output dimension (projection space, typically 65536 for DINOv2)
            hidden_dim: Hidden dimension for layers 1-2 (default 2048)
            bottleneck_dim: Bottleneck dimension before final layer (default 256)
            activation: Activation function
            drop: Dropout rate
            bias: Whether to use bias in linear layers
        """
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.act = activation()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.drop2 = nn.Dropout(drop)
        self.fc3 = nn.Linear(hidden_dim, bottleneck_dim, bias=bias)
        self.apply(self._init_weights)

        # Weight-normalized last layer (no bias)
        self.last_layer = weight_norm(nn.Linear(bottleneck_dim, output_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [..., input_dim]

        Returns:
            Output tensor [..., output_dim]
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.drop2(x)
        x = self.fc3(x)
        # L2 normalize before last layer
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = F.normalize(x, dim=-1, p=2, eps=eps)
        x = self.last_layer(x)
        return x

class PackedSequenceEncoder(nn.Module):
    """
    Encoder that processes packed sequences efficiently (Krell et al. 2022).

    Combines ResidueEmbedding with sequence packing for efficient batch processing
    of variable-length protein structures without padding overhead.

    Includes a shared projection head for DINOv2 that projects embeddings
    for both DINO loss (on [CLS] tokens) and iBOT loss (on patch tokens).
    """

    def __init__(
        self,
        n_atoms: int = 37,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        # Projection head parameters
        proj_output_dim: int = 65536,
        proj_hidden_dim: int = 2048,
        proj_bottleneck_dim: int = 256,
    ):
        """
        Args:
            n_atoms: Number of atoms per residue
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            qkv_bias: Whether to use bias in QKV projection
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            drop_path_rate: Stochastic depth rate
            norm_layer: Normalization layer
            proj_output_dim: Output dimension for projection head
            proj_hidden_dim: Hidden dimension for projection head
            proj_bottleneck_dim: Bottleneck dimension for projection head
        """
        super().__init__()

        self.embed_dim = embed_dim

        # Residue embedding
        self.residue_embed = ResidueEmbedding(
            n_atoms=n_atoms,
            embed_dim=embed_dim,
        )

        # Stochastic depth decay rule (linearly increase drop_path across blocks)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Build transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])

        # Final layer norm
        self.norm = norm_layer(embed_dim)

        # Shared projection head for both DINO and iBOT losses
        self.projection_head = ProjectionHead(
            input_dim=embed_dim,
            output_dim=proj_output_dim,
            hidden_dim=proj_hidden_dim,
            bottleneck_dim=proj_bottleneck_dim,
        )

    def forward(
        self,
        structures: list[torch.Tensor],
        return_projected: bool = True,
    ) -> dict[str, list[torch.Tensor]]:
        """
        Args:
            structures: List of protein structures, each [n_residues_i, n_atoms, 3]
            return_projected: Whether to return projected embeddings

        Returns:
            Dictionary containing:
                'backbone': List of backbone embeddings [n_residues_i, embed_dim]
                'projected': List of projected embeddings [n_residues_i, proj_output_dim] (if return_projected=True)
        """
        # Pack sequences
        packed, attention_mask, lengths = pack_protein_batch(structures)

        # Embed residues (no batch dimension needed)
        x = self.residue_embed(packed)  # [total_residues, embed_dim]

        # Add batch dimension for transformer
        x = x.unsqueeze(0)  # [1, total_residues, embed_dim]

        # Process through transformer blocks
        for block in self.blocks:
            x = block(x, attn_mask=attention_mask)

        # Apply final layer norm
        x = self.norm(x)

        # Remove batch dimension
        x = x.squeeze(0)  # [total_residues, embed_dim]

        # Unpack sequences
        backbone_outputs = unpack_sequences(x, lengths)

        # Prepare return dictionary
        outputs = {'backbone': backbone_outputs}

        # Apply projection head if requested
        if return_projected:
            # Project each sequence through the projection head
            projected_outputs = [self.projection_head(seq) for seq in backbone_outputs]
            outputs['projected'] = projected_outputs

        return outputs

