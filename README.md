# AlphaEarth Foundations

A PyTorch implementation of the AlphaEarth geospatial foundation model from Google DeepMind, which generates Earth embeddings for global environmental monitoring and analysis.
Accompanying the paper is a global dataset of embeddings from 2017 to 2024, available through Earth Engine. The goal of these embeddings is to serve as a highly general geospatial representation for a huge amount of downstream applications, without the need for retraining. 

> [!NOTE]
> I trained this model on 1/40th of the Landsat subset of the OlmoEarth pretrain dataset instead of the AlphaEarth Foundations dataset. Due to resource limitations, I only used a batch size of 16 and a max number of steps of 20000, instead of the 256 batch size and 100000 steps in the paper.

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

How to run a training step using the OlmoEarth pretrain dataset:
```
python -m alphaearth.run_train_olmoearth \
    --data_dir ./data/olmoearth_pretrain_dataset/10_landsat_monthly \
    --batch_size 32 \
    --num_workers 4 \
    --patch_size 256 \
    --max_steps 20000 \
    --output_dir ./outputs_olmoearth
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
