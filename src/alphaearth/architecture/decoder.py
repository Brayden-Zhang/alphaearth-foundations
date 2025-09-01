

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from alphaearth.architecture.encoder_utils import SinusoidalTimeEncoding




class VonMisesFisherDecoder(nn.Module):
    """
    von Mises-Fisher decoder 
    
    Model outputs are treated as the mean direction of a von Mises-Fisher distribution,
    and decoding proceeds by sampling this distribution, and concatenating it with 
    sensor geometry metadata and a timecode indicating the relative position in the 
    valid period to decode.
    """
    
    def __init__(self, embedding_dim: int, source_dims: Dict[str, int], 
                 geometry_dim: int = 16):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.source_dims = source_dims
        self.geometry_dim = geometry_dim
        
        # Concentration parameter for von Mises-Fisher distribution
        self.log_kappa = nn.Parameter(torch.log(torch.tensor(10.0)))
        
        # Decoders for each source
        self.source_decoders = nn.ModuleDict()
        for source, dim in source_dims.items():
            input_dim = embedding_dim + geometry_dim + embedding_dim  # embedding + geometry + timecode
            self.source_decoders[source] = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.GELU(),
                nn.LayerNorm(512),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.LayerNorm(256),
                nn.Linear(256, dim)
            )
        
        # Geometry metadata encoder
        self.geometry_encoder = nn.Linear(geometry_dim, geometry_dim)
        
        # Timecode encoder for relative position in valid period
        self.timecode_encoder = SinusoidalTimeEncoding(embedding_dim)
    
    def sample_von_mises_fisher(self, mu: torch.Tensor, kappa: float,
                                num_samples: int = 1) -> torch.Tensor:
        """
        Approximate sampling from vMF on S^(d-1) centered at mu with concentration kappa.

        For num_samples == 1, returns mu directly (deterministic decode). 

        """
        # mu: (B, H, W, D)
        B, H, W, D = mu.shape
        if num_samples == 1:
            return rearrange(mu, 'b h w d -> b h w 1 d')

        # For num_samples > 1, draws
        # Gaussian noise on the tangent plane and renormalizes; this is a coarse approximation

        # Noise scale inversely related to kappa
        noise_scale = (1.0 / (kappa + 1e-6)).clamp(min=1e-3)
        mu_flat = rearrange(mu, 'b h w d -> (b h w) d')
        samples = []
        for _ in range(num_samples):
            eps = torch.randn_like(mu_flat) * noise_scale
            x = mu_flat + eps
            x = F.normalize(x, p=2, dim=-1)
            samples.append(x)
        samples_tensor = torch.stack(samples, dim=1)  # (BHW, S, D)
        return rearrange(samples_tensor, '(b h w) s d -> b h w s d', b=B, h=H, w=W)
    
    def forward(self, embeddings: torch.Tensor, geometry_metadata: torch.Tensor,
                timestamps: torch.Tensor, valid_period: Tuple[float, float],
                source: str, num_samples: int = 1) -> torch.Tensor:
        """
        Decode using von Mises-Fisher sampling.
        
        Args:
            embeddings: Mean directions on unit sphere (B, L, L, 64)
            geometry_metadata: Sensor geometry metadata (B, geometry_dim)
            timestamps: Current timestamps (B,)
            valid_period: Valid period (start_time, end_time)
            source: Source to decode
            num_samples: Number of samples to draw
            
        Returns:
            Decoded source data (B, num_samples, L, L, source_dim)
        """
        B = embeddings.shape[0]
        L = embeddings.shape[1]
        
        # Sample from von Mises-Fisher distribution
        kappa = torch.exp(self.log_kappa)
        vmf_samples = self.sample_von_mises_fisher(embeddings, kappa, num_samples)
        
        start_time, end_time = valid_period
        relative_pos = (timestamps - start_time) / (end_time - start_time + 1e-6)
        relative_pos = relative_pos.clamp(0, 1)
        
        # Encode timecode for relative position
        timecodes = self.timecode_encoder(relative_pos.unsqueeze(-1))  # (B, embedding_dim)
        
        geo_encoded = self.geometry_encoder(geometry_metadata)  # (B, geometry_dim)
        
        # Expand geometry and timecodes to match spatial dimensions
        geo_encoded = repeat(geo_encoded, 'b d -> b l1 l2 d', l1=L, l2=L)
        timecodes = repeat(timecodes, 'b d -> b l1 l2 d', l1=L, l2=L)
        
        decoded_samples = []
        for i in range(num_samples):
            # Get VMF sample for this iteration
            vmf_sample = vmf_samples[:, :, :, i, :]  # (B, L, L, 64)
            
            # Concatenate vmf sample + geometry + timecodes
            decoder_input = torch.cat([
                vmf_sample,      # (B, L, L, 64)
                geo_encoded,     # (B, L, L, 16)
                timecodes        # (B, L, L, 64)
            ], dim=-1)  # (B, L, L, 64+16+64)
            
            decoder_input_flat = rearrange(decoder_input, 'b l1 l2 d -> (b l1 l2) d')
            
            decoded_flat = self.source_decoders[source](decoder_input_flat)  # (B*L*L, source_dim)
            
            # Reshape back to spatial grid
            decoded = rearrange(decoded_flat, '(b l1 l2) c -> b l1 l2 c', b=B, l1=L, l2=L)
            decoded_samples.append(decoded)
        
        decoded_samples = torch.stack(decoded_samples, dim=1)  # (B, num_samples, L, L, source_dim)
        
        return decoded_samples
