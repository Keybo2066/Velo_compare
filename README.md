# WTKO RNA Velocity + Contrastive Learning Pipeline

A comprehensive pipeline for comparing Wild-Type (WT) and Knockout (KO) cells using RNA velocity analysis and contrastive learning techniques.

## Overview

This pipeline integrates **RNA velocity analysis** with **contrastive learning** to enable direct comparison of cellular dynamics between WT and KO conditions. The framework uses a Variational Autoencoder (VAE) with contrastive learning to learn shared latent representations where cells of the same type from different conditions are brought together while maintaining biological meaningful separations.

### Key Features

- 🧬 **RNA Velocity Analysis**: High-precision velocity estimation using VELOVI
- 🤖 **Contrastive Learning**: Novel VAE architecture for WT/KO comparison
- 🎯 **Cell Type Alignment**: Symmetric contrastive loss for cross-condition alignment
- 📊 **Grid-based Visualization**: Velocity field visualization in latent space
- 🔄 **End-to-End Pipeline**: From raw data to publication-ready visualizations

## Repository Structure

```
├── source/                    # Core implementation
│   ├── models.py             # WTKOContrastiveVAE and loss functions
│   ├── trainers.py           # Training and inference utilities
│   ├── data.py               # Data loading and preprocessing
│   └── utils.py              # Visualization and analysis functions
├── tutorials/                # Usage examples and tutorials
│   └── sample_use.ipynb      # Complete pipeline demonstration
├── requirements.txt          # Package dependencies
├── setup.py                  # Package installation script
├── LICENSE                   # License information
└── README.md                 # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster training)

### Quick Installation

```bash
# Clone the repository
git clone https://github.com/Keybo2066/wtko-pipeline.git
cd wtko-pipeline

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Import and Setup

```python
import sys
sys.path.append('path/to/wtko-pipeline/source')

from models import WTKOContrastiveVAE
from trainers import WTKOTrainer
from data import create_wt_ko_dataloaders
```

### 2. Load and Preprocess Data

```python
import scanpy as sc

# Load your single-cell data
adata = sc.read_h5ad('your_data.h5ad')

# Separate WT and KO cells
adata_wt = adata[adata.obs['condition'] == 'WT'].copy()
adata_ko = adata[adata.obs['condition'] == 'KO'].copy()
```

### 3. Train the Model

```python
# Initialize model
model = WTKOContrastiveVAE(
    input_dim=n_genes,
    latent_dim=10,
    lambda_contrast=10,
    lambda_align=10
)

# Train
trainer = WTKOTrainer(model)
history = trainer.train(wt_loader, ko_loader, num_epochs=400)
```

### 4. Visualize Results

```python
# Get latent representations
wt_latent, wt_labels = trainer.get_latent_representations(wt_loader)
ko_latent, ko_labels = trainer.get_latent_representations(ko_loader)

# Generate visualizations
from utils import plot_combined_latent_space
plot_combined_latent_space(wt_embedding, ko_embedding, wt_labels, ko_labels, cell_type_names)
```

## Detailed Usage

For a complete walkthrough, see the tutorial notebook:
- [`tutorials/sample_use.ipynb`](tutorials/sample_use.ipynb) - Complete pipeline demonstration

## Data Requirements

### Input Data Format

Your single-cell data should be provided as an AnnData object (`.h5ad` file) with the following requirements:

#### Required `obs` (cell metadata) columns:
- **Cell type annotations**: Column containing cell type labels (e.g., `'cell_type'`, `'haem_subclust_grouped'`)
- **Condition labels**: Column distinguishing WT from KO cells (e.g., `'tomato'` with `'neg'`/`'pos'` values)

#### Required `layers` (expression matrices):
- **`'Ms'`**: Spliced mRNA counts
- **`'Mu'`**: Unspliced mRNA counts

#### Required `var` (gene metadata):
- Gene symbols as index or in a specific column

### Data Preprocessing

The pipeline includes automatic preprocessing steps:
1. **Quality filtering**: Remove low-quality cells and genes
2. **MURK gene removal**: Filter mitochondrial and ribosomal genes
3. **Normalization**: Log-normalization and scaling
4. **Gene filtering**: Velocity-based gene selection

### Example Data Structure

```python
AnnData object with n_obs × n_vars = 50000 × 20000
    obs: 'cell_type', 'condition', 'tomato', ...
    var: 'gene_symbol', 'highly_variable', ...
    layers: 'Ms', 'Mu', 'velocity'
    obsm: 'X_pca', 'X_umap'
```

### Sample Data

*(Note: Add information about sample datasets or data availability here)*

<!-- 
TODO: Add sample data information
- Link to example datasets
- Data download instructions
- Expected file sizes and formats
-->

## Model Architecture

### WTKOContrastiveVAE

The core model combines several key components:

1. **Encoder Network**: Maps gene expression to latent space
2. **Decoder Network**: Reconstructs gene expression from latent representations  
3. **Contrastive Learning**: Aligns same cell types across WT/KO conditions
4. **Cluster Alignment**: Maintains cell type structure in latent space

### Loss Functions

- **Reconstruction Loss**: MSE between input and reconstructed expression
- **Contrastive Loss**: Symmetric alignment of WT/KO cell types
- **Cluster Alignment Loss**: Intra-cell-type cohesion
- **KL Divergence**: Regularization term

## Parameters and Configuration

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `latent_dim` | 10 | Dimensionality of latent space |
| `lambda_contrast` | 10 | Weight for contrastive loss |
| `lambda_align` | 10 | Weight for cluster alignment loss |
| `tau` | 0.3 | Temperature parameter for contrastive learning |
| `hidden_dims` | (256, 128, 64) | Hidden layer dimensions |
| `dropout_prob` | 0.2 | Dropout probability |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_epochs` | 400 | Number of training epochs |
| `lr` | 1e-3 | Learning rate |
| `weight_decay` | 1e-3 | L2 regularization |
| `batch_size` | 256 | Batch size for training |

## Visualization Options

The pipeline provides several visualization methods:

1. **Latent Space UMAP**: 2D projection of learned representations
2. **Combined Visualization**: WT/KO cells in shared space
3. **Grid-based Velocity Fields**: Velocity vectors in latent space


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Acknowledgments

- [scVelo](https://scvelo.readthedocs.io/) for RNA velocity analysis foundations
- [VELOVI](https://docs.scvi-tools.org/en/stable/api/reference/scvi.external.VELOVI.html) for advanced velocity inference
- [scanpy](https://scanpy.readthedocs.io/) for single-cell analysis ecosystem
- PyTorch community for deep learning framework

## Support

- **Issues**: Please report bugs via [GitHub Issues](https://github.com/Keybo2066/wtko-pipeline/issues)
- **Contact**: [ctmk0009@mail4.doshisha.ac.jp](mailto:ctmk0009@mail4.doshisha.ac.jp)

## Changelog

### Version 1.0.0 (Current)
- Initial release
- Complete WTKO contrastive learning pipeline
- Tutorial notebook and documentation
- Grid-based velocity visualization

---

**Last Updated**: July 2025
