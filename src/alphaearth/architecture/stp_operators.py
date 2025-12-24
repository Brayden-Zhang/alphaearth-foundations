import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from alphaearth.architecture.encoder_utils import SinusoidalTimeEncoding


class SpaceOperator(nn.Module):
    """Space operator: ViT-like spatial self-attention at 1/16L resolution."""
    
    def __init__(self, dim: int = 1024, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W, C = x.shape
        x_flat = rearrange(x, 'b t h w c -> (b t) (h w) c')
        
        # Self-attention
        residual = x_flat
        x_norm = self.norm1(x_flat)
        
        qkv = self.qkv(x_norm)
        qkv = rearrange(qkv, 'bt hw (three heads d) -> three bt heads hw d', 
                       three=3, heads=self.num_heads, d=self.head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5) # .transpose (-2,1) for swapping (HW, d) -> (d, HW) --> (HW, d) @ (d, HW) = (HW, HW)
        attn = F.softmax(attn, dim=-1)
        
        x_attn = attn @ v
        x_attn = rearrange(x_attn, 'bt heads hw d -> bt hw (heads d)')
        x_flat = residual + self.proj(x_attn)
        x_flat = x_flat + self.mlp(self.norm2(x_flat))
        
        return rearrange(x_flat, '(b t) (h w) c -> b t h w c', b=B, t=T, h=H, w=W)


class TimeOperator(nn.Module):
    """Time operator: time-axial self-attention at 1/8L resolution.

        dim: Channel size per token
    """

    def __init__(self, dim: int = 512 , num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
        self.time_encoding = SinusoidalTimeEncoding(dim)
        
    def forward(self, x: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H, W, C) video tensor at 1/8L resolution. 
        B, T, H, W, C = x.shape
        
        if timestamps.dim() == 1:
            timestamps = timestamps.view(B, T)
        
        if timestamps.shape[1] != T:
            if timestamps.shape[1] > T:
                timestamps = timestamps[:, :T]
            else:
                last_ts = timestamps[:, -1:]
                padding = last_ts.repeat(1, T - timestamps.shape[1])
                timestamps = torch.cat([timestamps, padding], dim=1)
        
        time_enc = self.time_encoding(timestamps)
        
        if time_enc.dim() == 2:
            time_enc = time_enc.unsqueeze(1).expand(B, T, C)
        elif time_enc.shape[1] != T:
            if time_enc.shape[1] > T:
                time_enc = time_enc[:, :T, :]
            else:
                last_enc = time_enc[:, -1:, :]
                padding = last_enc.repeat(1, T - time_enc.shape[1], 1)
                time_enc = torch.cat([time_enc, padding], dim=1)
        
        time_enc = time_enc.unsqueeze(2).unsqueeze(3)
        x = x + time_enc
        x_flat = rearrange(x, 'b t h w c -> (b h w) t c')

        # Self-attention across time
        residual = x_flat
        x_norm = self.norm1(x_flat)
        
        qkv = self.qkv(x_norm)
        qkv = rearrange(qkv, 'bhw t (three heads d) -> three bhw heads t d',
                       three=3, heads=self.num_heads, d=self.head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        x_attn = attn @ v
        x_attn = rearrange(x_attn, 'bhw heads t d -> bhw t (heads d)')
        x_flat = residual + self.proj(x_attn)
        x_flat = x_flat + self.mlp(self.norm2(x_flat))
        
        return rearrange(x_flat, '(b h w) t c -> b t h w c', b=B, h=H, w=W)


class PrecisionOperator(nn.Module):
    """Precision operator: 3x3 convolutions at 1/2L resolution."""
    
    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim
        num_groups = min(32, dim // 4) if dim >= 4 else 1
        self.norm1 = nn.GroupNorm(num_groups, dim)
        self.norm2 = nn.GroupNorm(num_groups, dim * 4)  # After conv1 expansion
        
        self.conv1 = nn.Conv2d(dim, dim * 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim * 4, dim, kernel_size=3, padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H, W, C) at 1/2L resolution
        B, T, _, _, _ = x.shape
        x_conv = rearrange(x, 'b t h w c -> (b t) c h w')
        
        # Residual connection
        residual = x_conv
        
        # 3x3 convolutions with residual
        x_conv = self.conv1(self.norm1(x_conv))  # norm1: C -> conv1: C*4
        x_conv = F.gelu(x_conv)
        x_conv = self.conv2(self.norm2(x_conv))  # norm2: C*4 -> conv2: C
        x_conv = residual + x_conv
        
        return rearrange(x_conv, '(b t) c h w -> b t h w c', b=B, t=T)
