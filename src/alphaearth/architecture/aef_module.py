"""
AlphaEarth Foundations Model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import math

from einops import rearrange

from alphaearth.architecture.encoder import STPEncoder
from alphaearth.architecture.decoder import VonMisesFisherDecoder
from alphaearth.architecture.encoder_utils import IndividualSourceEncoder, SummaryPeriodEncoder


class TimePooling(nn.Module):
    """
    Single-query multi-head attention over time at each (h,w).
    Inputs:
      feats: (B, T, H, W, C)
      q:     (B, C)           — from SummaryPeriodEncoder
      mask:  (B, T) optional  — 1 for valid frames, 0 for padded/missing
    Output:
      z:     (B, H, W, C)
    """
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.kv = nn.Linear(dim, 2 * dim)     # to K,V
        self.q_proj = nn.Linear(dim, dim)     # to Q
        self.out = nn.Linear(dim, dim)

    def forward(self, feats: torch.Tensor, q: torch.Tensor, mask: torch.Tensor | None = None):
        B, T, H, W, C = feats.shape
        BHW = B * H * W

        # keys/values from temporal features at each (h,w)
        x = feats.view(BHW, T, C)                                      # (BHW, T, C)
        kv = self.kv(x).view(BHW, T, 2, self.num_heads, self.head_dim)
        K, V = kv[:, :, 0], kv[:, :, 1]                                # (BHW, T, heads, d)
        K = K.permute(0, 2, 1, 3)                                      # (BHW, heads, T, d)
        V = V.permute(0, 2, 1, 3)                                      # (BHW, heads, T, d)

        # single query per sample, broadcast to all (h,w)
        qh = self.q_proj(q).view(B, self.num_heads, self.head_dim)     # (B, heads, d)
        qh = qh.unsqueeze(1).expand(B, H * W, self.num_heads, self.head_dim) \
               .reshape(BHW, self.num_heads, 1, self.head_dim)         # (BHW, heads, 1, d)

        # scaled dot-product attention over time
        logits = (qh * K).sum(-1) / (self.head_dim ** 0.5)             # (BHW, heads, 1, T)
        logits = logits.squeeze(2)                                      # (BHW, heads, T)

        if mask is not None:
            mask_flat = mask.unsqueeze(1).unsqueeze(1)                 # (B,1,1,T)
            mask_flat = mask_flat.expand(B, H * W, self.num_heads, T)  # (B,HW,heads,T)
            mask_flat = mask_flat.reshape(BHW, self.num_heads, T)      # (BHW,heads,T)
            logits = logits.masked_fill(mask_flat == 0, float('-inf'))

        attn = torch.softmax(logits, dim=-1)
        
        if attn.dim() == 2:
            attn = attn.unsqueeze(-1)
        
        z = torch.einsum('bht,bhtd->bhd', attn, V)
        z = z.reshape(BHW, self.num_heads * self.head_dim)
        z = self.out(z).view(B, H, W, C)                               # (B,H,W,C)
        return z


class TemporalSummarizer(nn.Module):
    """
    Time-conditional summarization using a single query per sample and
    multi-head attention over time at each spatial location.
    Produces per-pixel 64D unit vectors (S^63).
    """

    def __init__(self, feature_dim: int, embed_dim: int = 64, num_heads: int = 8):
        super().__init__()
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim

        self.summarizer_q = SummaryPeriodEncoder(dim=feature_dim)
        self.time_pool = TimePooling(dim=feature_dim, num_heads=num_heads)
        self.proj_64 = nn.Linear(feature_dim, embed_dim, bias=False)

    def forward(self, feats: torch.Tensor, timestamps: torch.Tensor,
                valid_periods: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            feats: (B, T, H, W, C)
            timestamps: (B, T) in consistent units (kept for API; mask can derive from it)
            valid_periods: (B, 2) [t_s, t_e)
            mask: optional (B, T) validity mask
        Returns:
            embeddings: (B, H, W, 64), unit-norm per vector
        """
        # Build single query per sample
        q = self.summarizer_q(valid_periods)                 # (B, C)
        # Pool over time at each (h,w)
        z = self.time_pool(feats, q, mask=mask)              # (B, H, W, C)
        # Project to 64D and L2-normalize
        mu = self.proj_64(z)                                 # (B, H, W, 64)
        mu = F.normalize(mu, p=2, dim=-1)
        return mu


class AlphaEarthFoundations(nn.Module):
    """
    Complete AlphaEarth Foundations model. 
    Architecture 
    - Preprocessing with normalization and sinusoidal timecodes
    - Individual source encoders
    - Teacher network: STP encoder generating embeddings
    - Student network: learns from teacher via consistency loss  
    - Text network: enables video-text contrastive learning
    - von Mises-Fisher decoder for source reconstruction
    """
    def __init__(self,
                 model_size: str = "small",
                 input_sources: Optional[Dict[str, int]] = None,
                 decode_sources: Optional[Dict[str, int]] = None,
                 per_source_latent: int = 32,
                 enable_text_align: bool = False):
        """
        Args:
            model_size: 
            input_sources: mapping of source name -> channel count (inputs only)
            decode_sources: mapping of source name -> channel count to decode
        """
        super().__init__()

        # Default to Sentinel-2 only for now (5 bands), matching the provided dataloader
        if input_sources is None:
            input_sources = {"sentinel2": 5}
        if decode_sources is None:
            decode_sources = {"sentinel2": input_sources.get("sentinel2", 5)}

        self.input_sources = input_sources
        self.decode_sources = decode_sources
        self.enable_text_align = enable_text_align

        # Choose dimensions as per paper (S2.4), scaled down 
        d_p, d_t, d_s, num_blocks = 64, 256, 512, 6

        # Per-source encoders (Preprocessing box in Fig. 2A)
        self.source_encoders = nn.ModuleDict()
        for name, c in input_sources.items():
            self.source_encoders[name] = IndividualSourceEncoder(c, per_source_latent)

        total_in_channels = per_source_latent * len(input_sources)
        self.encoder = STPEncoder(
            input_channels=total_in_channels,
            d_s=d_s, d_t=d_t, d_p=d_p,
            num_blocks=num_blocks,
        )

        # Summarizer produces 64D embeddings on S^63
        self.summarizer = TemporalSummarizer(feature_dim=d_p, embed_dim=64)

        # VMF implicit decoder for sources
        self.decoder = VonMisesFisherDecoder(
            embedding_dim=64,               
            source_dims=self.decode_sources,
            geometry_dim=16,
        )

    def _stack_inputs(self, source_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Per-source encode, then concatenate along channel axis -> (B, T, H, W, C_total)."""
        xs: List[torch.Tensor] = []
        for name, _c in self.input_sources.items():
            x = source_data[name]  # (B, T, H, W, C)
            B, T, H, W, C = x.shape
            flat = rearrange(x, 'b t h w c -> (b t h w) c')
            enc = self.source_encoders[name](flat)
            enc = rearrange(enc, '(b t h w) c -> b t h w c', b=B, t=T, h=H, w=W)
            xs.append(enc)
        x_cat = torch.cat(xs, dim=-1)
        return x_cat

    def _perturb_inputs(self,
                        source_data: Dict[str, torch.Tensor],
                        timestamps: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Apply teacher-student input perturbations (S2.2.5): random source drops and time drops.
        Returns perturbed copies of source_data and timestamps for the student.
        """
        student_sources: Dict[str, torch.Tensor] = {}
        student_ts: Dict[str, torch.Tensor] = {}

        drop_prob = {name: (0.0 if name == 'sentinel2' else 0.3) for name in self.input_sources.keys()} # only keep S2 always for reconstruction

        # Choose time perturbation strategy
        # 0: random frame drops, 1: drop latter half-year, 2: drop former half-year
        strat = torch.randint(low=0, high=3, size=(1,)).item()

        for name, x in source_data.items():
            ts = timestamps[name]
            B, T, H, W, C = x.shape
            keep_mask = torch.ones(B, T, dtype=torch.bool, device=x.device)

            # Randomly drop entire source for student with prob
            if torch.rand(()) < drop_prob.get(name, 0.2):
                keep_mask[:] = False
            else:
                if strat == 0:
                    # Random per-frame drops: percentages by source (approximate paper ratios)
                    frac = 0.5 if name == 'sentinel2' else 0.3
                    drops = torch.rand(B, T, device=x.device) < frac
                    keep_mask = ~drops
                else:
                    # Half-year drop
                    t0 = ts.min(dim=1).values
                    t1 = ts.max(dim=1).values
                    mid = (t0 + t1) / 2.0
                    if strat == 1:
                        # drop latter half
                        keep_mask = ts <= mid.unsqueeze(1)
                    else:
                        # drop former half
                        keep_mask = ts >= mid.unsqueeze(1)

            # Apply keep_mask: zero out dropped frames and repeat last timestamp to keep shapes
            keep_mask_4d = keep_mask.view(B, T, 1, 1, 1)
            x_pert = x * keep_mask_4d
            # timestamps: keep original shape but clamp dropped segments to nearest kept
            ts_pert = ts.clone()

            student_sources[name] = x_pert
            student_ts[name] = ts_pert

        return student_sources, student_ts

    def forward(self,
                source_data: Dict[str, torch.Tensor],
                timestamps: Dict[str, torch.Tensor],
                valid_periods: List[Tuple[float, float]],
                geometry_metadata: Optional[torch.Tensor] = None,
                num_decode_samples: int = 1) -> Dict[str, Any]:
        """
        End-to-end forward pass: encoder -> temporal summarization -> implicit decoding.

        Args:
            source_data: dict of tensors per input source (B, T, H, W, C)
            timestamps: dict of (B, T) ms; only the inputs' timestamps are required
            valid_periods: list of (start_ms, end_ms) length B
            geometry_metadata: optional (B, 16) sensor geometry; zeros if None
            num_decode_samples: number of VMF samples to draw during decoding

        Returns:
            Dict with keys:
              - embeddings: (B, H', W', 64) unit vectors (S^63)
              - reconstructions: dict of decoded sources -> (B, S, H', W', C_src)
        """
        # Prepare inputs
        x = self._stack_inputs(source_data)  # (B, T, H, W, C)
        # Use timestamps from the first input source (they are aligned by collate)
        first_src = next(iter(self.input_sources.keys()))
        ts = timestamps[first_src]  # (B, T)

        # Encode (teacher) with STP to precision resolution (B, T, H/2, W/2, d_p)
        feats_teacher = self.encoder(x, ts)

        # Build student inputs via perturbation, then encode student
        student_srcs, student_ts_dict = self._perturb_inputs(source_data, timestamps)
        x_student = self._stack_inputs(student_srcs)
        ts_student = student_ts_dict[first_src]
        feats_student = self.encoder(x_student, ts_student)

        # Summarize across time to 64D embeddings per pixel
        # valid_periods provided as list of tuples -> tensor (B, 2)
        if isinstance(valid_periods, list):
            vp = torch.tensor(valid_periods, dtype=feats_teacher.dtype, device=feats_teacher.device)
        else:
            vp = valid_periods.to(feats_teacher.dtype).to(feats_teacher.device)
        mu_t = self.summarizer(feats_teacher, ts, vp)      # (B, H', W', 64)
        mu_s = self.summarizer(feats_student, ts_student, vp)  # (B, H', W', 64)

        B, H2, W2, _ = mu_t.shape
        if geometry_metadata is None:
            geometry_metadata = torch.zeros(B, 16, dtype=feats_teacher.dtype, device=feats_teacher.device)

        # For decoding timestamps, use the middle time of the support for each sample
        ts_center = ts.mean(dim=1)  # (B,)

        reconstructions: Dict[str, torch.Tensor] = {}
        B, T, H, W, _ = x.shape
        for src, _ch in self.decode_sources.items():
            recon = self.decoder(
                embeddings=mu_t,
                geometry_metadata=geometry_metadata,
                timestamps=ts_center,
                valid_period=(vp[:, 0], vp[:, 1]) if isinstance(vp, tuple) else (vp[:, 0], vp[:, 1]),
                source=src,
                num_samples=num_decode_samples,
            )  # (B, S, H/2, W/2, C_src)
            
            # Upsample to original resolution
            B_recon, S, H_recon, W_recon, C_recon = recon.shape
            recon_2d = rearrange(recon, 'b s h w c -> (b s) c h w')
            recon_2d = F.interpolate(recon_2d, size=(H, W), mode='bilinear', align_corners=False)
            recon = rearrange(recon_2d, '(b s) c h w -> b s h w c', b=B_recon, s=S)
            reconstructions[src] = recon

        # Image-level pooled embeddings (for text alignment)
        img_embed_t = mu_t.mean(dim=(1, 2))  # (B, 64)
        img_embed_s = mu_s.mean(dim=(1, 2))  # (B, 64)

        out: Dict[str, Any] = {
            'embeddings': mu_t,
            'teacher_embeddings': mu_t,
            'student_embeddings': mu_s,
            'image_embeddings': img_embed_t,
            'reconstructions': reconstructions,
        }

        if self.enable_text_align:
            out['needs_text'] = True

        return out
