import torch
import torch.nn as nn
import math

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, seq_len, embed_dim):
        super().__init__()
        self.register_buffer('positional_encoding', self._get_sinusoidal_embedding(seq_len, embed_dim))

    @staticmethod
    def _get_sinusoidal_embedding(seq_len, embed_dim):
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        pe = torch.zeros(seq_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, seq_len, embed_dim)

    def forward(self, x):
        return x + self.positional_encoding[:, :x.size(1), :].to(x.device)


class Temporal_Encoder(nn.Module):
    def __init__(self, img_size=256, num_slices=20, embed_dim=512, num_heads=8, num_layers=6, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.num_slices = num_slices
        self.embed_dim = embed_dim
        self.img_size = img_size

        # Flatten each 256x256 slice into a vector and project to embed_dim
        self.patch_embed = nn.Linear(img_size * img_size, embed_dim)

        # Sinusoidal positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(num_slices, embed_dim)

        # Transformer encoder layers (batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True  # Important: batch-first mode
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final projection: collapse across tokens (20) and map to 256×256
        self.final_proj = nn.Linear(embed_dim, img_size * img_size)

    def forward(self, x):
        """
        x: (B, 20, 256, 256) input
        returns: (B, 1, 256, 256) single-channel output
        """
        B, T, H, W = x.shape
        assert T == self.num_slices, f"Expected {self.num_slices} slices, got {T}"

        # Flatten each slice (B, T, H*W)
        x = x.view(B, T, H * W)

        # Project to embedding dimension
        x = self.patch_embed(x)  # (B, T, embed_dim)

        # Add sinusoidal positional encoding
        x = self.pos_encoding(x)

        # Transformer encoding
        x = self.transformer_encoder(x)  # (B, T, E)

        # Aggregate tokens (mean-pooling across 20 time slices)
        x = x.mean(dim=1)  # (B, E)

        # Map to 256×256 and reshape
        x = self.final_proj(x)  # (B, 256*256)
        x = x.view(B, 1, self.img_size, self.img_size)  # (B, 1, 256, 256)
        return x

