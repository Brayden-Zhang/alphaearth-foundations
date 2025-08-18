# AlphaEarth Foundations

A PyTorch implementation of the AlphaEarth geospatial foundation model from Google DeepMind, which generates Earth embeddings for global environmental monitoring and analysis.
Accompanying the paper is a global dataset of embeddings from 2017 to 2024, available through Earth Engine. The goal of these embeddings is to serve as a highly general geospatial representation for a huge amount of downstream applications, without the need for retraining. 

> [!NOTE]
> This model is a work in progress and was not actually trained on the full dataset, it is just a framework that provides a general base for the paper's architecture. The code is simplified compared to the DeepMind's actual implementation (in JAX). 

### Key Features & Innovations

- **Continuous Time Support**: First EO featurization approach to support continuous time, allowing for temporal interpolation and extrapolation.
- **Space Time Precision (STP) Architecture**: Multi-resolution encoder with spatial (1/16L), temporal (1/8L), and precision (1/2L) operators - designed to maintain localized representations while also modeling long-distance relationships across time and space. 
- **von Mises-Fisher Embeddings**: 64-byte embeddings distributed on unit sphere S^63, very compact representation. 
- **Comprehensive Evaluation**: 15-task evaluation suite across 11 datasets, comparing against baseline vision transformers and other geospatial foundation models such as Clay, SatCLIP, and Prithvi.

## Architecture

### Space Time Precision (STP) Encoder

The STP encoder processes multi-temporal, multi-source data through three simultaneous operators:
- **Space Operator**: ViT-like spatial self-attention (1/16L resolution)
- **Time Operator**: Time-axial self-attention (1/8L resolution) 
- **Precision Operator**: 3x3 convolutions (1/2L resolution)

### Teacher-Student-Text Framework

1. **Teacher Video Embedding Model**: Main model with implicit decoders
2. **Student Video Embedding Model**: Shares parameters with teacher for contrastive learning
3. **Text Alignment Model**: Enables text-image contrastive learning


## Data Sources

The model is trained on many data sources including:
- **Optical**: Sentinel-2, Landsat 8/9. *Note: for simplicty, my implementation only supports Sentinel-2 and Landsat 8/9.*
- **Radar**: Sentinel-1, PALSAR2
- **LiDAR**: GEDI
- **Environmental**: GLO-30, ERA5-Land, GRACE
- **Annotated/Text**: NLCD, Wikipedia

## Installation

```bash
# Clone the repository
git clone https://github.com/brayden-zhang/alphaearth-foundations.git
cd alphaearth-foundations

# Install dependencies
uv pip install -r requirements.txt

# Install the package 
uv pip install -e .
```


## Paper Citation

```bibtex
@article{brown2025alphaearth,
  title={AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data},
  author={Brown, Christopher F and Kazmierski, Michal R and Pasquarella, Valerie J and others},
  journal={arXiv preprint arXiv:2507.22291},
  year={2025}
}
```
