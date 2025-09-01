import torch
from torch import nn
from torch.functional import F
from typing import Tuple
from einops import rearrange
from alphaearth.architecture.laplacian_pyramid_exchange import LearnedSpatialResampling
from alphaearth.architecture.stp_operators import SpaceOperator, PrecisionOperator, TimeOperator


class STPBlock(nn.Module):
    """Single STP block with three simultaneous operators and pyramid exchanges."""
    
    def __init__(self, space_dim: int = 1024, time_dim: int = 512, precision_dim: int = 128):
        super().__init__()
        self.space_dim = space_dim
        self.time_dim = time_dim
        self.precision_dim = precision_dim
        
        self.space_op = SpaceOperator(self.space_dim)
        self.time_op = TimeOperator(self.time_dim)
        self.precision_op = PrecisionOperator(self.precision_dim)
        
        # Pyramid exchange resampling (learned Laplacian pyramid rescaling)
        self.space_to_time = LearnedSpatialResampling(self.space_dim, self.time_dim, 2.0)
        self.space_to_precision = LearnedSpatialResampling(self.space_dim, self.precision_dim, 8.0)
        self.time_to_space = LearnedSpatialResampling(self.time_dim, self.space_dim, 0.5)
        self.time_to_precision = LearnedSpatialResampling(self.time_dim, self.precision_dim, 4.0)
        self.precision_to_space = LearnedSpatialResampling(self.precision_dim, self.space_dim, 0.125)
        self.precision_to_time = LearnedSpatialResampling(self.precision_dim, self.time_dim, 0.25)
        
    def forward(self, space_x: torch.Tensor, time_x: torch.Tensor, precision_x: torch.Tensor, 
                timestamps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # Apply operators
        space_out = self.space_op(space_x)
        time_out = self.time_op(time_x, timestamps)
        precision_out = self.precision_op(precision_x)
        
        B, T = space_out.shape[:2]
        
        # Get spatial dimensions
        space_H, space_W = space_out.shape[2:4]
        time_H, time_W = time_out.shape[2:4]
        precision_H, precision_W = precision_out.shape[2:4]
        
        # Pyramid exchanges - reshape to (BT, C, H, W) for spatial ops
        space_2d = rearrange(space_out, 'b t h w c -> (b t) c h w')
        time_2d = rearrange(time_out, 'b t h w c -> (b t) c h w')
        precision_2d = rearrange(precision_out, 'b t h w c -> (b t) c h w')
        
        # Exchange information between scales with proper resampling        
        # time_to_space: time_H -> space_H (downsample by 0.5)
        time_to_space_resampled = self.time_to_space(time_2d)
        # Ensure output matches space dimensions
        if time_to_space_resampled.shape[2:] != (space_H, space_W):
            time_to_space_resampled = F.interpolate(
                time_to_space_resampled, size=(space_H, space_W), mode='bilinear', align_corners=False
            )
        
        # precision_to_space: precision_H -> space_H (upsample by 0.125)
        precision_to_space_resampled = self.precision_to_space(precision_2d)
     
        
        # space_to_time: space_H -> time_H (upsample by 2.0)
        space_to_time_resampled = self.space_to_time(space_2d)
        # Ensure output matches time dimensions
        if space_to_time_resampled.shape[2:] != (time_H, time_W):
            space_to_time_resampled = F.interpolate(
                space_to_time_resampled, size=(time_H, time_W), mode='bilinear', align_corners=False
            )
        
        # precision_to_time: precision_H -> time_H (upsample by 0.25)
        precision_to_time_resampled = self.precision_to_time(precision_2d)
        # Ensure output matches time dimensions
        if precision_to_time_resampled.shape[2:] != (time_H, time_W):
            precision_to_time_resampled = F.interpolate(
                precision_to_time_resampled, size=(time_H, time_W), mode='bilinear', align_corners=False
            )
        
        # space_to_precision: space_H -> precision_H (downsample by 8.0)
        space_to_precision_resampled = self.space_to_precision(space_2d)
        # Ensure output matches precision dimensions
        if space_to_precision_resampled.shape[2:] != (precision_H, precision_W):
            space_to_precision_resampled = F.interpolate(
                space_to_precision_resampled, size=(precision_H, precision_W), mode='bilinear', align_corners=False
            )
        
        # time_to_precision: time_H -> precision_H (downsample by 4.0)
        time_to_precision_resampled = self.time_to_precision(time_2d)
        # Ensure output matches precision dimensions
        if time_to_precision_resampled.shape[2:] != (precision_H, precision_W):
            time_to_precision_resampled = F.interpolate(
                time_to_precision_resampled, size=(precision_H, precision_W), mode='bilinear', align_corners=False
            )
        
        # Combine with proper spatial dimensions
        space_exchange = space_2d + time_to_space_resampled + precision_to_space_resampled
        time_exchange = time_2d + space_to_time_resampled + precision_to_time_resampled
        precision_exchange = precision_2d + space_to_precision_resampled + time_to_precision_resampled
        
        # Reshape back to (B, T, H, W, C)
        space_out = rearrange(space_exchange, '(b t) c h w -> b t h w c', b=B, t=T)
        time_out = rearrange(time_exchange, '(b t) c h w -> b t h w c', b=B, t=T)
        precision_out = rearrange(precision_exchange, '(b t) c h w -> b t h w c', b=B, t=T)
        
        return space_out, time_out, precision_out

