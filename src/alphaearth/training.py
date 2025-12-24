from typing import Any, Dict, List, Optional, Tuple
import itertools
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from alphaearth.architecture.aef_module import AlphaEarthFoundations
from alphaearth.loss_function import AEFLoss



class Trainer:
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
        self.max_steps = 1000
        self.warmup_steps = 0
        # Track losses for visualization
        self.loss_history = {
            'steps': [],
            'total': [],
            'reconstruction': [],
            'uniformity': [],
            'consistency': [],
            'clip': [],
        }

    def _prepare_reconstruction_targets(self, batch: Dict[str, Any], pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        src_key = next(iter(batch['source_data'].keys()))
        x = batch['source_data'][src_key].to(self.device)
        ts = batch['timestamps'][src_key].to(self.device)
        B, T, H, W, C = x.shape
        center = ts.mean(dim=1, keepdim=True)
        idx = (ts - center).abs().argmin(dim=1)
        batch_indices = torch.arange(B, device=self.device)
        target = x[batch_indices, idx]
        H2, W2 = pred.shape[2], pred.shape[3]
        target_2d = rearrange(target, 'b h w c -> b c h w')
        target_2d = F.interpolate(target_2d, size=(H2, W2), mode='bilinear', align_corners=False)
        target = rearrange(target_2d, 'b c h w -> b h w c')
        return {src_key: target}

    def train(self, max_steps: Optional[int] = None, log_every: int = 20):
        steps = max_steps or self.max_steps
        self.model.train()
        data_iter = itertools.cycle(self.dataloader)

        pbar = tqdm(range(1, steps + 1), desc="Training", unit="step")
        start_time = time.time()
        
        for step in pbar:
            batch = next(data_iter)
            step_start_time = time.time()
            
            source_data: Dict[str, torch.Tensor] = {
                k: v.to(self.device) for k, v in batch['source_data'].items()
            }
            timestamps: Dict[str, torch.Tensor] = {
                k: v.to(self.device) for k, v in batch['timestamps'].items()
            }
            valid_periods: List[Tuple[float, float]] = batch['valid_periods']

            
            out = self.model(source_data, timestamps, valid_periods)

            predictions: Dict[str, torch.Tensor] = {}
            for src, rec in out['reconstructions'].items():
                predictions[src] = rec[:, 0]

            targets: Dict[str, torch.Tensor] = {}
            if predictions:
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

            self.loss_history['steps'].append(step)
            self.loss_history['total'].append(float(loss))
            self.loss_history['reconstruction'].append(float(losses.get('reconstruction', torch.tensor(0.0))))
            self.loss_history['uniformity'].append(float(losses.get('uniformity', torch.tensor(0.0))))
            self.loss_history['consistency'].append(float(losses.get('consistency', torch.tensor(0.0))))
            self.loss_history['clip'].append(float(losses.get('clip', torch.tensor(0.0))))
            
            recon_loss = float(losses.get('reconstruction', torch.tensor(0.0)))
            pbar.set_postfix({
                'recon_loss': f'{recon_loss:.4f}',
                'total_loss': f'{float(loss):.4f}'
            })
            
            if step % log_every == 0:
                recon = float(losses.get('reconstruction', torch.tensor(0.0)))
                uni = float(losses.get('uniformity', torch.tensor(0.0)))
                cons = float(losses.get('consistency', torch.tensor(0.0)))
                clip = float(losses.get('clip', torch.tensor(0.0)))
                elapsed = time.time() - start_time
                steps_per_sec = step / elapsed if elapsed > 0 else 0
                remaining_steps = steps - step
                eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                eta_hours = eta_seconds / 3600
                print(f"\nstep {step:05d}/{steps:05d} ({step/steps*100:.1f}%) | "
                      f"total {float(loss):.4f} | recon {recon:.4f} | uni {uni:.4f} | cons {cons:.4f} | clip {clip:.4f} | "
                      f"ETA: {eta_hours:.2f}h ({steps_per_sec:.2f} steps/s)")
            
            if self.output_dir:
                self._save_checkpoint(step)
                self._save_loss_plots(step)
                
                if step % 1000 == 0 or step == steps:
                    self._save_reconstructions(out, predictions, targets, step)
        
        pbar.close()
        total_time = time.time() - start_time
        total_hours = total_time / 3600
        print(f"\nTraining completed in {total_hours:.2f} hours ({total_time:.0f} seconds)")
    
    def _save_checkpoint(self, step: int):
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
        }
        torch.save(checkpoint, output_path / 'checkpoint_latest.pt')
    
    def _save_reconstructions(self, out: Dict[str, Any], predictions: Dict[str, torch.Tensor], 
                             targets: Dict[str, torch.Tensor], step: int):
        output_path = Path(self.output_dir) / 'reconstructions'
        output_path.mkdir(parents=True, exist_ok=True)
        
        for src_name in predictions.keys():
            pred = predictions[src_name].detach().cpu()
            target = targets[src_name].detach().cpu()
            
            B = min(pred.shape[0], 4)
            fig, axes = plt.subplots(B, 2, figsize=(8, 4 * B))
            axes = axes.reshape(B, 2) if B == 1 else axes
            
            for b in range(B):
                pred_b = pred[b].numpy()
                target_b = target[b].numpy()
                
                pred_rgb = pred_b[..., :3]
                target_rgb = target_b[..., :3]
                
                pred_rgb = np.clip((pred_rgb - pred_rgb.min()) / (pred_rgb.max() - pred_rgb.min() + 1e-8), 0, 1)
                target_rgb = np.clip((target_rgb - target_rgb.min()) / (target_rgb.max() - target_rgb.min() + 1e-8), 0, 1)
                
                axes[b, 0].imshow(target_rgb)
                axes[b, 0].set_title('Target')
                axes[b, 0].axis('off')
                
                axes[b, 1].imshow(pred_rgb)
                axes[b, 1].set_title('Reconstruction')
                axes[b, 1].axis('off')
            
            plt.tight_layout()
            plt.savefig(output_path / f'{src_name}_step_{step:05d}.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    def _save_loss_plots(self, step: int):
        """Save training loss plots focusing on reconstruction loss."""
        output_path = Path(self.output_dir) / 'plots'
        output_path.mkdir(parents=True, exist_ok=True)
        
        steps = np.array(self.loss_history['steps'])
        recon_loss = np.array(self.loss_history['reconstruction'])
        
        # Plot 1: Reconstruction loss
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(steps, recon_loss, label='Reconstruction Loss', linewidth=2, color='C0')
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Reconstruction Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        plt.tight_layout()
        plt.savefig(output_path / 'reconstruction_loss.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Smoothed reconstruction loss (moving average) for better visualization
        window_size = min(50, len(steps) // 10 + 1)
        
        def smooth(values, window):
            smoothed = []
            for i in range(len(values)):
                start = max(0, i - window // 2)
                end = min(len(values), i + window // 2 + 1)
                smoothed.append(np.mean(values[start:end]))
            return smoothed
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.plot(steps, recon_loss, label='Reconstruction Loss (raw)', alpha=0.3, linewidth=1, color='C0')
        ax.plot(steps, smooth(recon_loss, window_size), 
               label=f'Reconstruction Loss (smoothed, window={window_size})', linewidth=2, color='C0')
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Reconstruction Loss (Smoothed)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        plt.tight_layout()
        plt.savefig(output_path / 'reconstruction_loss_smoothed.png', dpi=150, bbox_inches='tight')
        plt.close()


def create_trainer(model: AlphaEarthFoundations,
                   dataloader,
                   text_adapter = None,
                   lr: float = 1e-4,
                   device: Optional[str] = None,
                   output_dir: Optional[str] = None) -> Trainer:
    return Trainer(model=model, dataloader=dataloader, text_adapter=text_adapter, lr=lr, device=device, output_dir=output_dir)
