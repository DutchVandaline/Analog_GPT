import torch
import torch.nn
from torch import nn

from Models.DecoderBlock import GPTDecoder

device = "cuda" if torch.cuda.is_available() else "cpu"
class AnalogGPT(nn.Module):
    def __init__(self,
                 vocab_size: int,  # Vocabulary size
                 max_seq_len: int = 256,  # Maximum sequence length
                 embedding_dim: int = 768,
                 num_layers: int = 12,
                 num_heads: int = 12,
                 mlp_size: int = 3072,
                 mlp_dropout: float = 0.1,
                 attn_dropout: float = 0.1):
        super().__init__()

        # Token Embedding
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)

        # Positional Embedding
        self.positional_embedding = nn.Parameter(torch.zeros(1, max_seq_len, embedding_dim))

        # Decoder stack
        self.decoder = GPTDecoder(
            num_layers=num_layers,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            mlp_size=mlp_size,
            mlp_dropout=mlp_dropout,
            attn_dropout=attn_dropout
        )

        # Output projection to vocab size
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_ids, key_padding_mask=None):
        # Step 1: Embed tokens and add positional embeddings
        x = self.token_embedding(input_ids)  # Shape: [batch_size, seq_len, embedding_dim]
        # print(f"Token Embedding (x) shape: {x.shape}")
        # print(f"x contains NaN after token embedding: {x.isnan().any()}")

        seq_len = input_ids.size(1)
        x = x + self.positional_embedding[:, :seq_len, :]  # Add positional embedding
        # print(f"x after adding positional embedding: {x.shape}")
        # print(f"x contains NaN after positional embedding: {x.isnan().any()}")

        # Step 2: Check for NaN in input (if key_padding_mask is provided)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.to(torch.bool)  # Ensure the mask is of bool type
            # print(f"key_padding_mask contains NaN: {key_padding_mask.isnan().any()}")

        # Step 3: Pass through decoder stack
        x = self.decoder(x, key_padding_mask)
        # print(f"x after decoder: {x.shape}")
        # print(f"x contains NaN after decoder: {x.isnan().any()}")

        # Step 4: Output projection to vocab size
        logits = self.output_layer(x)
        # print(f"logits shape: {logits.shape}")
        # print(f"logits contains NaN: {logits.isnan().any()}")

        return logits