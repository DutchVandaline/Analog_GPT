import torch
import torch.nn as nn
from tqdm.auto import tqdm

from torch import nn
from transformers import AutoTokenizer


class MLPBlock(nn.Module):
    def __init__(self,
                 embedding_dim: int = 768,
                 mlp_size: int = 3072,
                 dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim,
                      out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size,
                      out_features=embedding_dim),
            nn.Dropout(p=dropout))

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x


class MaskedMultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self,
                 embedding_dim: int = 768,
                 num_heads: int = 12,
                 attn_dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embedding_dim)

        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                    num_heads=num_heads,
                                                    dropout=attn_dropout,
                                                    batch_first=True)

    def forward(self, x, key_padding_mask=None):
        # Normalize input
        x = self.layer_norm(x)

        # Masked Self-Attention
        batch_size, seq_len, _ = x.size()

        # Create causal mask (lower triangular matrix for self-attention)
        casual_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))

        # Apply MultiheadAttention
        attn_output, _ = self.multihead_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=casual_mask,  # Causal mask for self-attention
            key_padding_mask=key_padding_mask,  # Padding mask
            need_weights=True,
            average_attn_weights=False,
        )

        return attn_output


class TransformerDecoderBlock(nn.Module):
    def __init__(self,
                 embedding_dim: int = 768,
                 num_heads: int = 12,
                 mlp_size: int = 3072,
                 mlp_dropout: float = 0.1,
                 attn_dropout: float = 0.1):
        super().__init__()

        # Create Masked Self-Attention block (for autoregressive behavior)
        self.masked_msa_block = MaskedMultiHeadSelfAttentionBlock(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
        )

        # Create Feed-Forward block (MLP)
        self.mlp_block = MLPBlock(
            embedding_dim=embedding_dim,
            mlp_size=mlp_size,
            dropout=mlp_dropout
        )

        # Layer normalization for each block
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x, key_padding_mask=None):
        # print(f"Before self-attention: {x.isnan().any()}")

        attn_output = self.masked_msa_block(x, key_padding_mask)
        x_residual1 = attn_output + x
        # print(f"After self-attention: {x_residual1.isnan().any()}")

        # Apply Feed-Forward block (MLP) with residual connection
        mlp_output = self.mlp_block(x_residual1)
        x_residual2 = mlp_output + x_residual1
        # print(f"After feed-forward: {x_residual2.isnan().any()}")

        return x_residual2


class GPTDecoder(nn.Module):
    def __init__(self,
                 num_layers: int = 12,
                 embedding_dim: int = 768,
                 num_heads: int = 12,
                 mlp_size: int = 3072,
                 mlp_dropout: float = 0.1,
                 attn_dropout: float = 0.1):
        super().__init__()

        # Stack multiple transformer decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_size=mlp_size,
                mlp_dropout=mlp_dropout,
                attn_dropout=attn_dropout
            ) for _ in range(num_layers)
        ])

    def forward(self, x, key_padding_mask=None):
        for layer in self.decoder_layers:
            x = layer(x, key_padding_mask)
        return x