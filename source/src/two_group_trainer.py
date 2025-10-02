"""
VGAE + クラスター整列の学習
"""
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from .two_group_model import WTKO_VGAE, vgae_loss, cluster_alignment_loss
from .two_group_utils import build_combined_graph, normalize_adjacency


class WTKO_VGAE_Trainer:
    """
    2群VGAE + クラスター整列の学習クラス
    
    RNA velocity対照学習VAEのWTKOTrainerと同様のインターフェース
    """
    def __init__(self, model, lambda_align=1.0, beta=1.0, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            model: WTKO_VGAE - モデル
            lambda_align: float - クラスター整列損失の重み
            beta: float - KL損失の重み
            device: str - デバイス
        """
        self.model = model.to(device)
        self.lambda_align = lambda_align
        self.beta = beta
        self.device = device
        
        self.history = {
            'total_loss': [],
            'vgae_loss': [],
            'recon_loss': [],
            'kl_loss': [],
            'align_loss': []
        }
    
    def train(self, X_WT, X_KO, labels_WT, labels_KO, 
              num_epochs=200, lr=0.01, weight_decay=0.0,
              k_intra=10, k_cross=5,
              save_path=None, verbose=True):
        """
        モデルの学習
        
        Args:
            X_WT: np.ndarray, shape (N_WT, G) - WT群の遺伝子発現量
            X_KO: np.ndarray, shape (N_KO, G) - KO群の遺伝子発現量
            labels_WT: np.ndarray, shape (N_WT,) - WT群のラベル
            labels_KO: np.ndarray, shape (N_KO,) - KO群のラベル
            num_epochs: int - エポック数
            lr: float - 学習率
            weight_decay: float - 重み減衰
            k_intra: int - 群内KNN近傍数
            k_cross: int - 群間KNN近傍数
            save_path: str or None - モデル保存パス
            verbose: bool - 進捗表示
        
        Returns:
            history: dict - 学習履歴
        """
        print("="*60)
        print("VGAE + Cluster Alignment Training")
        print("="*60)
        print(f"WT cells: {X_WT.shape[0]}, KO cells: {X_KO.shape[0]}")
        print(f"Genes: {X_WT.shape[1]}")
        print(f"Latent dim: {self.model.latent_dim}")
        print(f"Lambda align: {self.lambda_align}")
        print(f"k_intra: {k_intra}, k_cross: {k_cross}")
        print("="*60)
        
        # グラフ構築（一度だけ、固定）
        print("\nBuilding graph structure...")
        A_combined, common_types = build_combined_graph(
            X_WT, X_KO, labels_WT, labels_KO,
            k_intra=k_intra, k_cross=k_cross
        )
        
        # 正規化
        print("Normalizing adjacency matrix...")
        A_norm = normalize_adjacency(A_combined)
        
        # pos_weightの計算（スパースグラフ対応）
        num_edges = A_combined.sum()
        num_total = A_combined.shape[0] ** 2
        pos_weight = (num_total - num_edges) / num_edges
        print(f"pos_weight: {pos_weight:.2f}")
        
        # Tensorに変換
        X_combined = np.vstack([X_WT, X_KO])
        labels_combined = np.concatenate([labels_WT, labels_KO])
        
        X_combined = torch.tensor(X_combined, dtype=torch.float32).to(self.device)
        A_combined = torch.tensor(A_combined, dtype=torch.float32).to(self.device)
        A_norm = torch.tensor(A_norm, dtype=torch.float32).to(self.device)
        labels_combined = torch.tensor(labels_combined, dtype=torch.long).to(self.device)
        
        # グラフは固定（勾配不要）
        A_combined.requires_grad = False
        A_norm.requires_grad = False
        
        # オプティマイザ
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        # 学習ループ
        print(f"\nTraining for {num_epochs} epochs...")
        pbar = tqdm(range(num_epochs), desc="Training") if verbose else range(num_epochs)
        
        for epoch in pbar:
            self.model.train()
            optimizer.zero_grad()
            
            # 順伝播
            A_recon, mu, logvar, z = self.model(X_combined, A_norm)
            
            # VGAE損失
            loss_vgae, loss_recon, loss_kl = vgae_loss(
                A_combined, A_recon, mu, logvar, 
                pos_weight=pos_weight, beta=self.beta
            )
            
            # クラスター整列損失
            loss_align = cluster_alignment_loss(
                z, labels_combined, 
                normalize=self.model.normalize_embedding
            )
            
            # 総損失
            loss_total = loss_vgae + self.lambda_align * loss_align
            
            # 最初のエポックで損失のスケールを表示
            if epoch == 0 and verbose:
                print(f"\nInitial loss scales:")
                print(f"  Recon loss: {loss_recon.item():.2f}")
                print(f"  KL loss: {loss_kl.item():.2f}")
                print(f"  Align loss: {loss_align.item():.2f}")
                print(f"  Weighted align (×{self.lambda_align}): {(self.lambda_align * loss_align).item():.2f}")
                print(f"  Total VGAE loss: {loss_vgae.item():.2f}")
                print(f"  Total loss: {loss_total.item():.2f}\n")
            
            # NaN/Inf チェック
            if torch.isnan(loss_total) or torch.isinf(loss_total):
                print(f"\n⚠️ Warning: Loss became NaN/Inf at epoch {epoch}")
                print(f"  Recon loss: {loss_recon.item():.4f}")
                print(f"  KL loss: {loss_kl.item():.4f}")
                print(f"  Align loss: {loss_align.item():.4f}")
                print(f"  Total loss: {loss_total.item():.4f}")
                print("  Training stopped to prevent further divergence.")
                break
            
            # 逆伝播
            loss_total.backward()
            
            # 勾配クリッピング（重要！）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 履歴に記録
            self.history['total_loss'].append(loss_total.item())
            self.history['vgae_loss'].append(loss_vgae.item())
            self.history['recon_loss'].append(loss_recon.item())
            self.history['kl_loss'].append(loss_kl.item())
            self.history['align_loss'].append(loss_align.item())
            
            # 進捗表示
            if verbose and (epoch + 1) % 10 == 0:
                pbar.set_postfix({
                    'total': f'{loss_total.item():.2f}',
                    'vgae': f'{loss_vgae.item():.2f}',
                    'align': f'{loss_align.item():.4f}'
                })
        
        print("\nTraining completed!")
        
        # モデル保存
        if save_path:
            torch.save(self.model.state_dict(), save_path + '.pth')
            print(f"Model saved to {save_path}.pth")
        
        return self.history
    
    def evaluate(self, X_WT, X_KO, labels_WT, labels_KO, A_norm):
        """
        モデルの評価
        
        Args:
            X_WT: torch.Tensor or np.ndarray
            X_KO: torch.Tensor or np.ndarray
            labels_WT: torch.Tensor or np.ndarray
            labels_KO: torch.Tensor or np.ndarray
            A_norm: torch.Tensor or np.ndarray
        
        Returns:
            metrics: dict - 評価指標
        """
        self.model.eval()
        
        with torch.no_grad():
            # Tensorに変換
            if isinstance(X_WT, np.ndarray):
                X_WT = torch.tensor(X_WT, dtype=torch.float32).to(self.device)
            if isinstance(X_KO, np.ndarray):
                X_KO = torch.tensor(X_KO, dtype=torch.float32).to(self.device)
            if isinstance(A_norm, np.ndarray):
                A_norm = torch.tensor(A_norm, dtype=torch.float32).to(self.device)
            
            # エンコード
            z_WT, z_KO = self.model.encode_separate(X_WT, X_KO, A_norm)
            
            # 同一セルタイプの平均距離
            intra_dist = self._compute_intra_celltype_distance(
                z_WT, z_KO, labels_WT, labels_KO
            )
            
            # 異なるセルタイプの平均距離
            inter_dist = self._compute_inter_celltype_distance(
                z_WT, z_KO, labels_WT, labels_KO
            )
        
        metrics = {
            'intra_celltype_distance': intra_dist,
            'inter_celltype_distance': inter_dist,
            'separation_ratio': inter_dist / intra_dist if intra_dist > 0 else 0
        }
        
        return metrics
    
    def _compute_intra_celltype_distance(self, z_WT, z_KO, labels_WT, labels_KO):
        """同一セルタイプのWT-KO間平均距離（高速版）"""
        if isinstance(labels_WT, np.ndarray):
            labels_WT = torch.tensor(labels_WT).to(self.device)
        if isinstance(labels_KO, np.ndarray):
            labels_KO = torch.tensor(labels_KO).to(self.device)
        
        common_labels = set(labels_WT.cpu().numpy()) & set(labels_KO.cpu().numpy())
        
        if len(common_labels) == 0:
            return 0.0
        
        total_dist = 0.0
        total_pairs = 0
        
        for label in common_labels:
            mask_WT = (labels_WT == label)
            mask_KO = (labels_KO == label)
            
            z_WT_c = z_WT[mask_WT]  # (n_wt, d)
            z_KO_c = z_KO[mask_KO]  # (n_ko, d)
            
            n_wt = z_WT_c.shape[0]
            n_ko = z_KO_c.shape[0]
            
            # サンプリング（メモリ節約のため）
            max_samples = 500
            if n_wt > max_samples:
                indices = torch.randperm(n_wt, device=self.device)[:max_samples]
                z_WT_c = z_WT_c[indices]
                n_wt = max_samples
            if n_ko > max_samples:
                indices = torch.randperm(n_ko, device=self.device)[:max_samples]
                z_KO_c = z_KO_c[indices]
                n_ko = max_samples
            
            # ベクトル化: (n_wt, 1, d) - (1, n_ko, d) = (n_wt, n_ko, d)
            diff = z_WT_c.unsqueeze(1) - z_KO_c.unsqueeze(0)
            distances = torch.norm(diff, dim=2)  # (n_wt, n_ko)
            
            total_dist += distances.sum().item()
            total_pairs += n_wt * n_ko
        
        return total_dist / total_pairs if total_pairs > 0 else 0.0

    def _compute_inter_celltype_distance(self, z_WT, z_KO, labels_WT, labels_KO):
        """異なるセルタイプのWT-KO間平均距離（高速版）"""
        if isinstance(labels_WT, np.ndarray):
            labels_WT = torch.tensor(labels_WT).to(self.device)
        if isinstance(labels_KO, np.ndarray):
            labels_KO = torch.tensor(labels_KO).to(self.device)
        
        n_wt = z_WT.shape[0]
        n_ko = z_KO.shape[0]
        
        # サンプリング
        max_samples = 1000
        if n_wt > max_samples:
            indices_wt = torch.randperm(n_wt, device=self.device)[:max_samples]
            z_WT_sample = z_WT[indices_wt]
            labels_WT_sample = labels_WT[indices_wt]
        else:
            z_WT_sample = z_WT
            labels_WT_sample = labels_WT
        
        if n_ko > max_samples:
            indices_ko = torch.randperm(n_ko, device=self.device)[:max_samples]
            z_KO_sample = z_KO[indices_ko]
            labels_KO_sample = labels_KO[indices_ko]
        else:
            z_KO_sample = z_KO
            labels_KO_sample = labels_KO
        
        # ベクトル化: 全ペアの距離を一度に計算
        diff = z_WT_sample.unsqueeze(1) - z_KO_sample.unsqueeze(0)  # (n_wt, n_ko, d)
        distances = torch.norm(diff, dim=2)  # (n_wt, n_ko)
        
        # 異なる細胞タイプのマスク
        labels_WT_expanded = labels_WT_sample.unsqueeze(1)  # (n_wt, 1)
        labels_KO_expanded = labels_KO_sample.unsqueeze(0)  # (1, n_ko)
        mask = (labels_WT_expanded != labels_KO_expanded)  # (n_wt, n_ko)
        
        if mask.sum() == 0:
            return 0.0
        
        # マスクを適用して平均
        inter_distances = distances[mask]
        return inter_distances.mean().item()
    
    def plot_training_history(self, save_path=None):
        """
        学習履歴をプロット
        
        Args:
            save_path: str or None - 保存パス
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Total Loss
        axes[0, 0].plot(self.history['total_loss'])
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # VGAE Loss
        axes[0, 1].plot(self.history['vgae_loss'], label='VGAE')
        axes[0, 1].plot(self.history['recon_loss'], label='Recon', alpha=0.7)
        axes[0, 1].plot(self.history['kl_loss'], label='KL', alpha=0.7)
        axes[0, 1].set_title('VGAE Components')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Alignment Loss
        axes[1, 0].plot(self.history['align_loss'])
        axes[1, 0].set_title('Cluster Alignment Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True)
        
        # Loss Components Comparison
        axes[1, 1].plot(self.history['vgae_loss'], label='VGAE')
        axes[1, 1].plot([x * self.lambda_align for x in self.history['align_loss']], 
                       label=f'Align (×{self.lambda_align})')
        axes[1, 1].set_title('Loss Components (weighted)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path + '_history.png', dpi=300, bbox_inches='tight')
            print(f"Training history saved to {save_path}_history.png")
        
        plt.show()
