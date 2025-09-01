from typing import Any, Dict, List, Optional, Tuple
import itertools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from alphaearth.architecture.aef_module import AlphaEarthFoundations
from alphaearth.loss_function import AEFLoss



class Trainer:
    """
    Trainer for AEF.
    - Uses AEFLoss with reconstruction + uniformity + consistency + optional text loss
    - Expects batches from the provided dataloaders in src/alphaearth/data.py
    """

    def __init__(self,
                 model: AlphaEarthFoundations,
                 dataloader,
                 text_adapter = None, 
                 lr: float = 1e-4,
                 device: Optional[str] = None,
                 output_dir: Optional[str] = None):
        self.model = model
        self.dataloader = dataloader
        self.text_adapter = text_adapter
        self.loss_fn = AEFLoss()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        if self.text_adapter is not None:
            self.text_adapter.to(self.device)
        params = list(self.model.parameters())
        if self.text_adapter is not None and any(p.requires_grad for p in self.text_adapter.parameters()):
            params += [p for p in self.text_adapter.parameters() if p.requires_grad]
        self.optim = torch.optim.Adam(params, lr=lr)

        self.output_dir = output_dir
        # compatibility flags used in the example script
        self.max_steps = 1000
        self.warmup_steps = 0

    def _prepare_reconstruction_targets(self, batch: Dict[str, Any], pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Build a target for reconstruction from the closest input frame to the center of the time range.
        Downsample to the prediction resolution (H', W').
        pred: (B, S, H', W', C)
        """
        src_key = next(iter(batch['source_data'].keys()))  # default 'sentinel2'
        x = batch['source_data'][src_key].to(self.device)  # (B, T, H, W, C)
        ts = batch['timestamps'][src_key].to(self.device)  # (B, T)
        B, T, H, W, C = x.shape
        # choose nearest to mean timestamp
        center = ts.mean(dim=1, keepdim=True)  # (B, 1)
        idx = (ts - center).abs().argmin(dim=1)  # (B,)
        # gather per batch
        batch_indices = torch.arange(B, device=self.device)
        target = x[batch_indices, idx]  # (B, H, W, C)
        # downsample to pred H', W'
        H2, W2 = pred.shape[2], pred.shape[3]
        target_2d = rearrange(target, 'b h w c -> b c h w')
        target_2d = F.interpolate(target_2d, size=(H2, W2), mode='bilinear', align_corners=False)
        target = rearrange(target_2d, 'b c h w -> b h w c')
        return {src_key: target}

    def train(self, max_steps: Optional[int] = None, log_every: int = 20):
        steps = max_steps or self.max_steps
        self.model.train()
        data_iter = itertools.cycle(self.dataloader)

        for step in range(1, steps + 1):
            batch = next(data_iter)
            source_data: Dict[str, torch.Tensor] = {
                k: v.to(self.device) for k, v in batch['source_data'].items()
            }
            timestamps: Dict[str, torch.Tensor] = {
                k: v.to(self.device) for k, v in batch['timestamps'].items()
            }
            valid_periods: List[Tuple[float, float]] = batch['valid_periods']

            
            out = self.model(source_data, timestamps, valid_periods)

            # Predictions for reconstruction: pick first sample S=0 per source and build targets
            predictions: Dict[str, torch.Tensor] = {}
            for src, rec in out['reconstructions'].items():
                # rec: (B, S, H', W', C)
                predictions[src] = rec[:, 0]  # (B, H', W', C)

            targets: Dict[str, torch.Tensor] = {}
            if predictions:
                # Build targets from inputs for the first source using temporal center
                # Use shape of any pred to downsample target
                some_src = next(iter(predictions.keys()))
                targets = self._prepare_reconstruction_targets(batch, predictions[some_src].unsqueeze(1))

            # Optional text embeddings for text-image alignment loss
            text_embeddings = None
            if self.text_adapter is not None and 'texts' in batch:
                text_embeddings = self.text_adapter.encode(batch['texts'], device=self.device)

            outputs_for_loss: Dict[str, Any] = {
                'embeddings': out['embeddings'],
                'teacher_embeddings': out['teacher_embeddings'],
                'student_embeddings': out['student_embeddings'],
                'image_embeddings': out['image_embeddings'],
                'predictions': predictions,
                'targets': targets,
                'masks': {k: torch.ones_like(v[..., :1], device=self.device) for k, v in predictions.items()},
            }
            if text_embeddings is not None and text_embeddings.shape[0] == out['image_embeddings'].shape[0]:
                outputs_for_loss['text_embeddings'] = text_embeddings

            losses = self.loss_fn(outputs_for_loss)
            loss = losses['total']

            self.optim.zero_grad(set_to_none=True)
            loss.backward()
            self.optim.step()

            if step % log_every == 0:
                recon = float(losses.get('reconstruction', torch.tensor(0.0)))
                uni = float(losses.get('uniformity', torch.tensor(0.0)))
                cons = float(losses.get('consistency', torch.tensor(0.0)))
                clip = float(losses.get('clip', torch.tensor(0.0)))
                print(f"step {step:05d} \n  total {float(loss):.4f} | recon {recon:.4f} | uni {uni:.4f} | cons {cons:.4f} | clip {clip:.4f}")


def create_trainer(model: AlphaEarthFoundations,
                   dataloader,
                   text_adapter = None, 
                   output_dir: Optional[str] = None) -> Trainer:
    return Trainer(model=model, dataloader=dataloader, text_adapter=text_adapter, output_dir=output_dir)
