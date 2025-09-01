"""
AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data.

This package implements the AlphaEarth Foundations architecture as described in the paper.
"""

from .architecture.aef_module import AlphaEarthFoundations
from .architecture.encoder import STPEncoder
from .architecture.STPBlock import STPBlock
from .data import AEFDataset

__all__ = [
    "AlphaEarthFoundations",
    "STPEncoder", 
    "STPBlock",
    "AEFDataset",
]
