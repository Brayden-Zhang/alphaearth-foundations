
from typing import Any, Dict
from einops import rearrange
import torch
import torch.nn as nn
from torch.functional import F
import math

"""
AEF loss = Reconstruction loss + Uniformity loss + Consistency loss + Text loss
"""

class AEFLoss:
    """
    AlphaEarth Foundations loss implementation following Equation 3 in the paper:
    """
    
    def __init__(self,
                 reconstruction_weight: float = 1.0,  # a = 1.0
                 uniformity_weight: float = 0.05,    # b = 0.05
                 consistency_weight: float = 0.02,   # c = 0.02
                 text_weight: float = 0.001):        # d = 0.001
        
        self.reconstruction_weight = reconstruction_weight
        self.uniformity_weight = uniformity_weight
        self.consistency_weight = consistency_weight
        self.text_weight = text_weight
        
        # Source-specific loss configurations from Table S2
        self.source_configs = {
            'sentinel2': {'weight': 1.0, 'loss_fn': F.l1_loss},
        }
    
    def reconstruction_loss(self, predictions: Dict[str, torch.Tensor], 
                          targets: Dict[str, torch.Tensor],
                          masks: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute reconstruction loss for all sources

        Compares predicted observation y_i' with ground truth y_i for each source i --> this leads the model to force the embeddings to carry enough information to be able to reconstruct the raw EO inputs. 
        For continuous sources: use L1 Loss; for categorical sources: use Cross-entropy.

        """
        
        total_loss = 0.0
        
        for source in predictions:
            if source in targets:
                config = self.source_configs.get(source, {'weight': 1.0, 'loss_fn': F.l1_loss})
                mask = masks.get(source, torch.ones_like(targets[source]))
                
                # Apply the mask and compute reconstruction loss
                pred_masked = predictions[source] * mask
                target_masked = targets[source] * mask
                
                if config['loss_fn'] == F.cross_entropy:
                    # For categorical sources 
                    loss = config['loss_fn'](pred_masked, target_masked.long())
                else:
                    # For continuous sources
                    loss = config['loss_fn'](pred_masked, target_masked)

                # weight the loss by source
                total_loss += config['weight'] * loss
        
        return total_loss
    
    def batch_uniformity_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute batch uniformity objective (Equation 4) --> objective: to have the embeddings be uniformly distributed.
        Takes the embeddings, rotates & shuffles them across the batch and then minimizes the absolute dot product between matched pairs
        """
        # embeddings: (B, H, W, D) or (B, T, H, W, D); flatten to N vectors in D
        x = embeddings
        if x.dim() == 5:
            B, T, H, W, D = x.shape
            x = rearrange(x, 'b t h w d -> (b t h w) d')
        elif x.dim() == 4:
            B, H, W, D = x.shape
            x = rearrange(x, 'b h w d -> (b h w) d')
        else:
            # (N, D)
            pass

        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        # Rotate (roll) sample pairs to approximate u' in the paper
        x_prime = torch.roll(x, shifts=1, dims=0)
        dots = (x * x_prime).sum(dim=-1).abs()  # |u · u'|
        return dots.mean()

    
    def consistency_loss(self, teacher_embeddings: torch.Tensor, 
                        student_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute teacher-student consistency loss (Equation 5)."""
        # 1 - mu · mu_s over 2, averaged over all pixels
        mu = torch.nn.functional.normalize(teacher_embeddings, p=2, dim=-1)
        mu_s = torch.nn.functional.normalize(student_embeddings, p=2, dim=-1)
        dots = (mu * mu_s).sum(dim=-1)
        return ((1.0 - dots) * 0.5).mean()
    
    def clip_loss(self, image_embeddings: torch.Tensor, 
                  text_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute CLIP-style contrastive loss."""
        # Expect (B, D) vs (B, D)
        img = torch.nn.functional.normalize(image_embeddings, p=2, dim=-1)
        txt = torch.nn.functional.normalize(text_embeddings, p=2, dim=-1)
        logits = img @ txt.t()  # (B, B)
        targets = torch.arange(img.size(0), device=img.device)
        loss_i = torch.nn.functional.cross_entropy(logits, targets)
        loss_t = torch.nn.functional.cross_entropy(logits.t(), targets)
        return 0.5 * (loss_i + loss_t)  
    
    def __call__(self, outputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Compute total loss following Equation 3."""
        
        losses = {}
        
        if 'predictions' in outputs and 'targets' in outputs:
            recon_loss = self.reconstruction_loss(
                outputs['predictions'],
                outputs['targets'],
                outputs.get('masks', {})
            )
            losses['reconstruction'] = recon_loss
        else:
            losses['reconstruction'] = torch.tensor(0.0)
        
        if 'embeddings' in outputs:
            uniformity_loss = self.batch_uniformity_loss(outputs['embeddings'])
            losses['uniformity'] = uniformity_loss
        else:
            losses['uniformity'] = torch.tensor(0.0, device=next(iter(outputs.values())).device if outputs else 'cpu')

        if 'teacher_embeddings' in outputs and 'student_embeddings' in outputs:
            consistency_loss = self.consistency_loss(
                outputs['teacher_embeddings'],
                outputs['student_embeddings']
            )
            losses['consistency'] = consistency_loss
        else:
            losses['consistency'] = torch.tensor(0.0, device=losses['reconstruction'].device)
        
        if 'image_embeddings' in outputs and 'text_embeddings' in outputs:
            clip_loss = self.clip_loss(
                outputs['image_embeddings'],
                outputs['text_embeddings']
            )
            losses['clip'] = clip_loss
        else:
            losses['clip'] = torch.tensor(0.0, device=losses['reconstruction'].device)
        
        total_loss = (
            self.reconstruction_weight * losses['reconstruction'] +
            self.uniformity_weight * losses['uniformity'] +
            self.consistency_weight * losses['consistency'] +
            self.text_weight * losses['clip']
        )
        
        losses['total'] = total_loss
        
        return losses
