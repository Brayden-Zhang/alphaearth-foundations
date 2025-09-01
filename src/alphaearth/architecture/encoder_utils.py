
from typing import Tuple
import torch
from torch import nn
import numpy as np

import torch.nn.functional as F 
from einops import rearrange, repeat


class IndividualSourceEncoder(nn.Module):
    """
    Individual source encoders that transform inputs to the same latent space
    as shown in Figure 2A.
    In reality, this will differ between individual sources (optical vs climate vs etc.)
    """
    
    def __init__(self, source_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(source_dim, latent_dim // 2),
            nn.GELU(),
            nn.LayerNorm(latent_dim // 2),
            nn.Linear(latent_dim // 2, latent_dim),
            nn.LayerNorm(latent_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform source data to latent space."""
        return self.encoder(x)


class SinusoidalTimeEncoding(nn.Module):
    """Sinusoidal time encoding for temporal conditioning as per paper."""
    
    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Convert millisecond timestamps to sinusoidal encodings."""
        half_dim = self.dim // 2
        
        # Create frequency embeddings
        freq = torch.exp(
            torch.arange(half_dim, device=timestamps.device) * (np.log(self.max_period) / (half_dim - 1))
        )  # (half_dim,)

        # Ensure timestamps has the right shape to broadcast: (B, T, 1)
        if timestamps.dim() == 1:
            timestamps = rearrange(timestamps, 'b -> b 1')
        t3 = rearrange(timestamps, 'b t -> b t 1')
        f3 = rearrange(freq, 'd -> 1 1 d')

        sin_embeddings = torch.sin(t3 * f3)  # (B, T, half_dim)
        cos_embeddings = torch.cos(t3 * f3)  # (B, T, half_dim)
        embeddings = torch.cat([sin_embeddings, cos_embeddings], dim=-1)  # (B, T, dim)
        
        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1))
        
        if timestamps.shape[1] == 1:
            embeddings = rearrange(embeddings, 'b 1 d -> b d')  # (B, 1, D) -> (B, D)
            
        return embeddings


class SummaryPeriodEncoder(nn.Module):
    """
    Time-conditional summarizer query constructor (Section S2.2.1):
      - encodes [t_s, t_e) and duration (t_e - t_s)
      - returns ONE query per sample to pool a per-pixel time series
    Output: q ∈ R^{B × C}
    """

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "dim should be even"
        self.dim = dim

        # Encode a single scalar time into dim//2 features
        self.time_enc = SinusoidalTimeEncoding(self.dim // 2)

        # Fuse (start, end, duration) encodings -> query
        in_dim = 3 * (self.dim // 2)  # enc(ts) || enc(te) || enc(te - ts)
        self.fuse = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )

        self.q_bias = nn.Parameter(torch.zeros(dim))

    def forward(self, valid_period: torch.Tensor) -> torch.Tensor:
        """
        valid_period: (B, 2) with [t_s, t_e) in a consistent unit
        returns: q (B, dim)  — one query per sample
        """
        assert valid_period.dim() == 2 and valid_period.size(1) == 2
        t_s = valid_period[:, 0]
        t_e = valid_period[:, 1]
        dur = t_e - t_s

        enc_s = self.time_enc(t_s)  # (B, dim//2)
        enc_e = self.time_enc(t_e)  # (B, dim//2)
        enc_d = self.time_enc(dur)  # (B, dim//2)

        q = torch.cat([enc_s, enc_e, enc_d], dim=-1)  # (B, 3*dim//2)
        q = self.fuse(q) + self.q_bias  # (B, dim)
        return q
