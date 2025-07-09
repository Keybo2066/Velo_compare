# RNA Velocity Cross-Condition Analysis Pipeline

A comprehensive pipeline for comparing cellular dynamics between two experimental conditions using RNA velocity analysis and contrastive learning techniques.

## Overview

This pipeline integrates RNA velocity analysis with deep learning techniques to enable direct comparison of cellular dynamics between different experimental conditions. The framework uses a Variational Autoencoder (VAE) to learn shared latent representations where cells of the same type from different conditions are aligned while maintaining biologically meaningful separations.

### Key Features

- 🧬 **RNA Velocity Analysis**: High-precision velocity estimation using VELOVI
- 🤖 **Contrastive Learning**: Novel VAE architecture for cross-condition comparison
- 🎯 **Cell Type Alignment**: Symmetric contrastive loss for cross-condition alignment
- 📊 **Grid-based Visualization**: Velocity field visualization in latent space
- 🔄 **End-to-End Pipeline**: From raw data to publication-ready visualizations

## Repository Structure

```
├── source/                    # Core implementation
│   ├── models.py             # CrossConditionContrastiveVAE and loss functions
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
git clone https://github.com/Keybo2066/Velo_compare.git
cd Velo_compare

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Import and Setup

```python
import sys
sys.path.append('path/to/Velo_compare/source')

from models import CrossConditionContrastiveVAE
from trainers import CrossConditionTrainer
from data import create_condition_dataloaders
```

### 2. Load and Preprocess Data

```python
import scanpy as sc

# Load your single-cell data
adata = sc.read_h5ad('your_data.h5ad')

# Separate conditions (e.g., treated vs control, timepoint1 vs timepoint2)
adata_condition1 = adata[adata.obs['condition'] == 'condition1'].copy()
adata_condition2 = adata[adata.obs['condition'] == 'condition2'].copy()
```

### 3. Train the Model

```python
# Initialize model
model = CrossConditionContrastiveVAE(
    input_dim=n_genes,
    latent_dim=10,
    lambda_contrast=10,
    lambda_align=10
)

# Train
trainer = CrossConditionTrainer(model)
history = trainer.train(condition1_loader, condition2_loader, num_epochs=400)
```

### 4. Visualize Results

```python
# Get latent representations
condition1_latent, condition1_labels = trainer.get_latent_representations(condition1_loader)
condition2_latent, condition2_labels = trainer.get_latent_representations(condition2_loader)

# Generate visualizations
from utils import plot_combined_latent_space
plot_combined_latent_space(condition1_embedding, condition2_embedding, 
                          condition1_labels, condition2_labels, cell_type_names)
```

## Data Requirements

### Input Data Format

Your single-cell data should be provided as an AnnData object (`.h5ad` file) with the following requirements:

#### Required `obs` (cell metadata) columns:
- **Cell type annotations**: Column containing cell type labels (e.g., `'cell_type'`, `'haem_subclust_grouped'`)
- **Condition labels**: Column distinguishing between experimental conditions (e.g., `'treatment'` with `'control'`/`'treated'` values)

#### Required `layers` (expression matrices):
- **`'Ms'`**: Spliced mRNA counts
- **`'Mu'`**: Unspliced mRNA counts


### Data Preprocessing

The pipeline includes preprocessing steps:
1. **Quality filtering**: Remove low-quality cells and genes
2. **Normalization**: Log-normalization and scaling
3. **Gene filtering**: Velocity-based gene selection

### Example Data Structure

```python
AnnData object with n_obs × n_vars = 50000 × 20000
    obs: 'cell_type', 'condition', 'treatment_status', ...
    var: 'gene_symbol', 'highly_variable', ...
    layers: 'Ms', 'Mu', 'velocity'
    obsm: 'X_pca', 'X_umap'
```

### Sample Data

This pipeline was developed and tested using the following datasets:

#### Primary Dataset: Gata1 Chimeric Embryo scRNA-seq
- **GEO Accession**: [GSE167576](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE167576)
- **Description**: Single-cell RNA sequencing comparing knockout and control conditions in mouse embryos at E8.5
- **Publication**: Barile et al. (2021) "Coordinated changes in gene expression kinetics underlie both mouse and human erythroid maturation" *Genome Biology* 22:197
- **DOI**: [10.1186/s13059-021-02414-y](https://doi.org/10.1186/s13059-021-02414-y)

#### Dataset Features:
- **Species**: Mouse (*Mus musculus*)
- **Development Stage**: E8.5 embryos
- **Cell Types**: Hematopoietic lineages (erythroid, megakaryocyte, myeloid, etc.)
- **Conditions**: Control vs knockout conditions
- **Technology**: 10X Genomics scRNA-seq (version 3 chemistry)
- **Cell Count**: ~16,000 cells (8,420 condition1 + 7,944 condition2)

#### Data Structure:
```
Required obs columns:
- 'treatment_status': 'control' / 'knockout' 
- 'haem_subclust_grouped': Cell type annotations

Required layers:
- 'Ms': Spliced mRNA counts
- 'Mu': Unspliced mRNA counts
```

For detailed experimental protocols and data processing methods, please refer to the original publication.

## Model Architecture

### CrossConditionContrastiveVAE

The core model combines several key components:

1. **Encoder Network**: Maps gene expression to latent space
2. **Decoder Network**: Reconstructs gene expression from latent representations  
3. **Contrastive Learning**: Aligns same cell types across different conditions
4. **Cluster Alignment**: Maintains cell type structure in latent space

### Loss Functions

- **Reconstruction Loss**: MSE between input and reconstructed expression
- **Contrastive Loss**: Symmetric alignment of cell types across conditions
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
2. **Combined Visualization**: Cross-condition cells in shared space
3. **Grid-based Velocity Fields**: Velocity vectors in latent space

## License

This project is licensed under the MIT License - see the [LICENCE](LICENCE) file for details.


## Support

- **Issues**: Please report bugs via [GitHub Issues](https://github.com/Keybo2066/cross-condition-pipeline/issues)
- **Contact**: [ctmk0009@mail4.doshisha.ac.jp](mailto:ctmk0009@mail4.doshisha.ac.jp)

## Changelog

### Version 1.0.0 (Current)
- Initial release
- Complete cross-condition contrastive learning pipeline
- Tutorial notebook and documentation
- Grid-based velocity visualization

---

**Last Updated**: July 2025
