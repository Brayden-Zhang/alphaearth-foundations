
from einops import rearrange
import torch
from torch.functional import F 
import torch.nn as nn

from alphaearth.architecture.STPBlock import STPBlock
from alphaearth.architecture.laplacian_pyramid_exchange import LearnedSpatialResampling


class STPEncoder(nn.Module):
    """
    Space Time Precision encoder 
 
    """
    
    def __init__(self, input_channels: int, d_s: int = 1024, d_t: int = 512, d_p: int = 128, num_blocks: int = 15):
        super().__init__()
        self.space_dim = d_s
        self.time_dim = d_t
        self.precision_dim = d_p
        
        # Individual source encoders transform inputs to same latent space
        self.input_projection = nn.Linear(input_channels, self.precision_dim)
        
        # Pathway-specific projections
        self.space_projection = nn.Linear(self.precision_dim, self.space_dim)
        self.time_projection = nn.Linear(self.precision_dim, self.time_dim)
        
        # STP blocks as per paper
        self.blocks = nn.ModuleList([STPBlock(d_s, d_t, d_p) for _ in range(num_blocks)])
        
        # Final learned spatial resampling to precision resolution
        self.final_space_resample = LearnedSpatialResampling(self.space_dim, self.precision_dim, 8.0)
        self.final_time_resample = LearnedSpatialResampling(self.time_dim, self.precision_dim, 4.0)
        
        # Output norm
        self.norm = nn.LayerNorm(self.precision_dim)
        
    def forward(self, x: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, T, H, W, C) - preprocessed multi-source data
            timestamps: Millisecond timestamps (B, T)
            
        Returns:
            Features at 1/2L resolution: (B, T, H/2, W/2, precision_dim)
        """
        B, T, H, W, C = x.shape
        
        # Project inputs to common latent space
        x_proj = self.input_projection(x)
        
        # Initialize features at different resolutions using pathway projections
        # Space pathway: project to space_dim and downsample to 1/16L
        space_features = self.space_projection(x_proj)
        space_features = F.adaptive_avg_pool2d(
            rearrange(space_features, 'b t h w c -> (b t) c h w'),
            (H // 16, W // 16)
        )
        space_features = rearrange(space_features, '(b t) c h w -> b t h w c', b=B, t=T)
        
        # Time pathway: project to time_dim and downsample to 1/8L  
        time_features = self.time_projection(x_proj)
        time_features = F.adaptive_avg_pool2d(
            rearrange(time_features, 'b t h w c -> (b t) c h w'),
            (H // 8, W // 8)
        )
        time_features = rearrange(time_features, '(b t) c h w -> b t h w c', b=B, t=T)
        
        # Precision pathway: keep at precision_dim and downsample to 1/2L
        precision_features = F.adaptive_avg_pool2d(
            rearrange(x_proj, 'b t h w c -> (b t) c h w'),
            (H // 2, W // 2)
        )
        precision_features = rearrange(precision_features, '(b t) c h w -> b t h w c', b=B, t=T)

        # Apply STP blocks
        for block in self.blocks:
            space_features, time_features, precision_features = block(
                space_features, time_features, precision_features, timestamps
            )
        
        # Final learned spatial resampling to precision resolution
        space_2d = rearrange(space_features, 'b t h w c -> (b t) c h w')
        time_2d = rearrange(time_features, 'b t h w c -> (b t) c h w')
        precision_2d = rearrange(precision_features, 'b t h w c -> (b t) c h w')
        
        # Resample space and time pathways to precision resolution
        space_resampled = self.final_space_resample(space_2d)
        time_resampled = self.final_time_resample(time_2d)
        
        # Ensure all pathways have the same spatial dimensions
        target_H, target_W = precision_2d.shape[2:]
        
        if space_resampled.shape[2:] != (target_H, target_W):
            space_resampled = F.interpolate(
                space_resampled, size=(target_H, target_W), mode='bilinear', align_corners=False
            )
        
        if time_resampled.shape[2:] != (target_H, target_W):
            time_resampled = F.interpolate(
                time_resampled, size=(target_H, target_W), mode='bilinear', align_corners=False
            )
        
        # Combine all pathways at precision resolution
        final_features = space_resampled + time_resampled + precision_2d
        
        # Reshape back and normalize
        final_features = rearrange(final_features, '(b t) c h w -> b t h w c', b=B, t=T)
        
        return self.norm(final_features)
