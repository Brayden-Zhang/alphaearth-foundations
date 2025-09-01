from torch import nn
import torch  

class LearnedSpatialResampling(nn.Module):
    """Learned Laplacian pyramid rescaling for spatial pyramid exchanges. Using Conv2d for simplicity"""
    
    def __init__(self, in_channels: int, out_channels: int, scale_factor: float):
        super().__init__()
        self.scale_factor = scale_factor
        
        if scale_factor > 1:
            # Upsampling: scale_factor > 1 means output is larger
            # For scale_factor = 2.0, output size = input_size * 2
            self.conv = nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=4, stride=2, padding=1
            )
        elif scale_factor < 1:
            # Downsampling: scale_factor < 1 means output is smaller
            # For scale_factor = 0.5, output size = input_size / 2
            stride = int(1 / scale_factor)
            self.conv = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=stride * 2 - 1,
                stride=stride,
                padding=stride - 1
            )
        else:
            # Same resolution
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
