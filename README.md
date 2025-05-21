# WT-KO Contrastive VAE

球面上の対照学習を用いた野生型(WT)とノックアウト(KO)の単一細胞RNA-seq解析用変分オートエンコーダ

## 概要

このパッケージは、WTとKO間での細胞タイプの対応関係を保持しながら、scRNA-seqデータを低次元空間に埋め込むための対照的変分オートエンコーダ（Contrastive VAE）を実装しています。主な特徴：

- **球面VAE**: 潜在空間を単位球面上に制約することで安定した表現学習を実現
- **対照学習**: 同一細胞タイプのWT/KO細胞を潜在空間で近づける損失関数
- **クラスタ整列**: 同一細胞タイプの細胞をクラスタ中心に引き寄せる正則化
- **RNA velocity統合**: 細胞状態変化の方向性を考慮した解析
- **包括的可視化**: UMAPプロット、グリッドベース速度場など多様な可視化ツール

## インストール

### PyPI からのインストール (公開後)
```bash
pip install wt-ko-contrastive-vae
```

### ソースからのインストール
```bash
git clone https://github.com/yourusername/wt-ko-contrastive-vae.git
cd wt-ko-contrastive-vae
pip install -e .
```

## 使用例

### 基本的な使用方法

```python
import numpy as np
import torch
from wt_ko_contrastive_vae import WTKOContrastiveVAE, WTKOTrainer, create_wt_ko_dataloaders

# データの準備
n_cells = 1000
n_genes = 2000
n_cell_types = 5

# ダミーデータの作成
wt_data = np.random.rand(n_cells, n_genes).astype(np.float32)
ko_data = np.random.rand(n_cells, n_genes).astype(np.float32)
wt_labels = np.random.randint(0, n_cell_types, n_cells)
ko_labels = np.random.randint(0, n_cell_types, n_cells)

# データローダー作成
wt_loader, ko_loader = create_wt_ko_dataloaders(
    wt_data, wt_labels, ko_data, ko_labels, batch_size=64
)

# モデル初期化
model = WTKOContrastiveVAE(
    input_dim=n_genes,
    latent_dim=10,
    hidden_dims=(128, 64),
    lambda_contrast=1.0,
    lambda_align=0.5
)

# モデル学習
trainer = WTKOTrainer(model)
history = trainer.train(
    wt_loader=wt_loader,
    ko_loader=ko_loader,
    num_epochs=50,
    lr=1e-3
)

# 潜在表現の取得
wt_latent, wt_labels = trainer.get_latent_representations(wt_loader)
ko_latent, ko_labels = trainer.get_latent_representations(ko_loader)
```

### 可視化

```python
import umap
from wt_ko_contrastive_vae import plot_latent_space, plot_combined_latent_space

# UMAPで2次元に投影
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
combined_latent = np.vstack([wt_latent.cpu().numpy(), ko_latent.cpu().numpy()])
combined_embedding = reducer.fit_transform(combined_latent)

# 投影結果を分割
wt_embedding = combined_embedding[:len(wt_latent)]
ko_embedding = combined_embedding[len(wt_latent):]

# 細胞タイプの名前
cell_type_names = [f"CellType_{i}" for i in range(n_cell_types)]

# 二枚のUMAP図
plot_latent_space(
    wt_embedding=wt_embedding,
    ko_embedding=ko_embedding,
    wt_labels=wt_labels.cpu().numpy(),
    ko_labels=ko_labels.cpu().numpy(),
    cell_type_names=cell_type_names
)

# WT/KOを一枚のUMAP図に
plot_combined_latent_space(
    wt_embedding=wt_embedding,
    ko_embedding=ko_embedding,
    wt_labels=wt_labels.cpu().numpy(),
    ko_labels=ko_labels.cpu().numpy(),
    cell_type_names=cell_type_names
)
```

### RNA velocityとの統合

```python
from wt_ko_contrastive_vae import plot_latent_space_with_grid_velocity

# 速度ベクトルのシミュレーション（実際のデータではscVelo等から取得）
wt_velocity = np.random.randn(wt_embedding.shape[0], 2) * 0.1
ko_velocity = np.random.randn(ko_embedding.shape[0], 2) * 0.1

# グリッドベースの速度場可視化
plot_latent_space_with_grid_velocity(
    wt_embedding=wt_embedding,
    ko_embedding=ko_embedding,
    wt_labels=wt_labels.cpu().numpy(),
    ko_labels=ko_labels.cpu().numpy(),
    cell_type_names=cell_type_names,
    wt_velocity=wt_velocity,
    ko_velocity=ko_velocity,
    grid_size=20,
    min_cells=5
)
```

## 要件

- Python >= 3.8
- PyTorch >= 1.10.0
- NumPy >= 1.20.0
- Pandas >= 1.3.0
- Matplotlib >= 3.4.0
- Seaborn >= 0.11.0
- scikit-learn >= 1.0.0
- UMAP-learn >= 0.5.0
- AnnData >= 0.8.0
- scikit-learn >= 1.0.0

詳細な依存関係については `requirements.txt` を参照してください。

## 引用

本パッケージを研究で使用される場合は、以下のように引用してください：

```bibtex
@software{wt_ko_contrastive_vae,
  title={WT-KO Contrastive VAE},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/wt-ko-contrastive-vae}
}
```

## ライセンス

MIT ライセンスのもとで公開されています。詳細は [LICENSE](LICENSE) ファイルを参照してください。

## 開発者情報

- 作者名
- 所属機関
- メールアドレス

## 貢献

貢献は大歓迎です！バグレポート、改善提案、プルリクエストは [GitHub Issues](https://github.com/yourusername/wt-ko-contrastive-vae/issues) にてお待ちしています。
