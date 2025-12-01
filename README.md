
<div align="right">
  <details>
    <summary >🌐 Language</summary>
    <div>
      <div align="center">
        <a href="https://openaitx.github.io/view.html?user=Brayden-Zhang&project=alphaearth-foundations&lang=en">English</a>
        | <a href="https://openaitx.github.io/view.html?user=Brayden-Zhang&project=alphaearth-foundations&lang=zh-CN">简体中文</a>
        | <a href="https://openaitx.github.io/view.html?user=Brayden-Zhang&project=alphaearth-foundations&lang=zh-TW">繁體中文</a>
        | <a href="https://openaitx.github.io/view.html?user=Brayden-Zhang&project=alphaearth-foundations&lang=ja">日本語</a>
        | <a href="https://openaitx.github.io/view.html?user=Brayden-Zhang&project=alphaearth-foundations&lang=ko">한국어</a>
        | <a href="https://openaitx.github.io/view.html?user=Brayden-Zhang&project=alphaearth-foundations&lang=hi">हिन्दी</a>
        | <a href="https://openaitx.github.io/view.html?user=Brayden-Zhang&project=alphaearth-foundations&lang=th">ไทย</a>
        | <a href="https://openaitx.github.io/view.html?user=Brayden-Zhang&project=alphaearth-foundations&lang=fr">Français</a>
        | <a href="https://openaitx.github.io/view.html?user=Brayden-Zhang&project=alphaearth-foundations&lang=de">Deutsch</a>
        | <a href="https://openaitx.github.io/view.html?user=Brayden-Zhang&project=alphaearth-foundations&lang=es">Español</a>
        | <a href="https://openaitx.github.io/view.html?user=Brayden-Zhang&project=alphaearth-foundations&lang=it">Italiano</a>
        | <a href="https://openaitx.github.io/view.html?user=Brayden-Zhang&project=alphaearth-foundations&lang=ru">Русский</a>
        | <a href="https://openaitx.github.io/view.html?user=Brayden-Zhang&project=alphaearth-foundations&lang=pt">Português</a>
        | <a href="https://openaitx.github.io/view.html?user=Brayden-Zhang&project=alphaearth-foundations&lang=nl">Nederlands</a>
        | <a href="https://openaitx.github.io/view.html?user=Brayden-Zhang&project=alphaearth-foundations&lang=pl">Polski</a>
        | <a href="https://openaitx.github.io/view.html?user=Brayden-Zhang&project=alphaearth-foundations&lang=ar">العربية</a>
        | <a href="https://openaitx.github.io/view.html?user=Brayden-Zhang&project=alphaearth-foundations&lang=fa">فارسی</a>
        | <a href="https://openaitx.github.io/view.html?user=Brayden-Zhang&project=alphaearth-foundations&lang=tr">Türkçe</a>
        | <a href="https://openaitx.github.io/view.html?user=Brayden-Zhang&project=alphaearth-foundations&lang=vi">Tiếng Việt</a>
        | <a href="https://openaitx.github.io/view.html?user=Brayden-Zhang&project=alphaearth-foundations&lang=id">Bahasa Indonesia</a>
        | <a href="https://openaitx.github.io/view.html?user=Brayden-Zhang&project=alphaearth-foundations&lang=as">অসমীয়া</
      </div>
    </div>
  </details>
</div>

# AlphaEarth Foundations

A PyTorch implementation of the AlphaEarth geospatial foundation model from Google DeepMind, which generates Earth embeddings for global environmental monitoring and analysis.
Accompanying the paper is a global dataset of embeddings from 2017 to 2024, available through Earth Engine. The goal of these embeddings is to serve as a highly general geospatial representation for a huge amount of downstream applications, without the need for retraining. 

> [!NOTE]
> This model is a work in progress and was not actually trained on the full dataset, it is just a framework that provides a general base for the paper's architecture. The code is simplified compared to the DeepMind's actual implementation (in JAX). 

### Key parts of the methodology

- **Continuous Time Support**: First EO featurization approach to support continuous time, allowing for temporal interpolation and extrapolation.
- **Space Time Precision (STP) Architecture**: Multi-resolution encoder with spatial (1/16L), temporal (1/8L), and precision (1/2L) operators - designed to maintain localized representations while also modeling long-distance relationships across time and space. 
- **von Mises-Fisher Embeddings**: 64-byte embeddings distributed on unit sphere S^63, very compact representation. 


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
- **Optical**: Sentinel-2, Landsat 8/9. *Note: for simplicty, my implementation only supports Sentinel-2, but it should be relatively straightforward to add new datasets to the training*
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

How to run a training step:
```
python -m alphaearth.run_train
```

## Paper Citation

```bibtex
@misc{brown2025alphaearthfoundationsembeddingfield,
      title={AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data}, 
      author={Christopher F. Brown and Michal R. Kazmierski and Valerie J. Pasquarella and William J. Rucklidge and Masha Samsikova and Chenhui Zhang and Evan Shelhamer and Estefania Lahera and Olivia Wiles and Simon Ilyushchenko and Noel Gorelick and Lihui Lydia Zhang and Sophia Alj and Emily Schechter and Sean Askay and Oliver Guinan and Rebecca Moore and Alexis Boukouvalas and Pushmeet Kohli},
      year={2025},
      eprint={2507.22291},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.22291}, 
}
```
