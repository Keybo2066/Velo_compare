"""
VGAE + クラスター整列による2群比較モデル
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class GCNLayer(nn.Module):
    """
    単一Graph Convolutional Network層
    """
    def __init__(self, input_dim, output_dim):
        super(GCNLayer, self).__init__()
        
        # 重み行列の初期化（既存VGAEと同じ方法）
        init_range = np.sqrt(6.0 / (input_dim + output_dim))
        self.weight = nn.Parameter(
            torch.FloatTensor(input_dim, output_dim).uniform_(-init_range, init_range)
        )
        self.bias = nn.Parameter(torch.zeros(output_dim))
    
    def forward(self, X, A_norm):
        """
        Args:
            X: torch.Tensor, shape (N, input_dim)
            A_norm: torch.Tensor, shape (N, N) - 正規化隣接行列
        
        Returns:
            out: torch.Tensor, shape (N, output_dim)
        """
        # Ã X W + b
        support = torch.matmul(X, self.weight)
        output = torch.matmul(A_norm, support) + self.bias
        return output


class WTKO_VGAE(nn.Module):
    """
    2群比較用VGAE + クラスター整列
    
    既存のVGAEクラスと同様のインターフェース
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, lambda_align=1.0, 
                 normalize_embedding=True):
        super(WTKO_VGAE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lambda_align = lambda_align
        self.normalize_embedding = normalize_embedding
        
        # 第1層GCN（共有）
        self.gcn1 = GCNLayer(input_dim, hidden_dim)
        
        # 第2層GCN（μとlog_σに分岐）
        self.gcn_mu = GCNLayer(hidden_dim, latent_dim)
        self.gcn_logvar = GCNLayer(hidden_dim, latent_dim)
    
    def encode(self, X, A_norm):
        """
        GCNエンコーダ
        
        Args:
            X: torch.Tensor, shape (N, input_dim)
            A_norm: torch.Tensor, shape (N, N)
        
        Returns:
            mu: torch.Tensor, shape (N, latent_dim)
            logvar: torch.Tensor, shape (N, latent_dim)
        """
        # 第1層: Ã ReLU(Ã X W_0) W_1
        hidden = torch.relu(self.gcn1(X, A_norm))
        
        # 第2層: μとlog_σ
        mu = self.gcn_mu(hidden, A_norm)
        logvar = self.gcn_logvar(hidden, A_norm)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        再パラメータ化トリック
        
        Args:
            mu: torch.Tensor, shape (N, latent_dim)
            logvar: torch.Tensor, shape (N, latent_dim)
        
        Returns:
            z: torch.Tensor, shape (N, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(mu.device)  # デバイスを明示
        z = mu + eps * std
        
        # 単位球面に正規化（オプション）
        if self.normalize_embedding:
            z = F.normalize(z, p=2, dim=1)
        
        return z
    
    @staticmethod
    def decode(z):
        """
        内積デコーダ: A_recon = sigmoid(Z Z^T)
        
        Args:
            z: torch.Tensor, shape (N, latent_dim)
        
        Returns:
            A_recon: torch.Tensor, shape (N, N)
        """
        A_recon = torch.sigmoid(torch.matmul(z, z.t()))
        return A_recon
    
    def forward(self, X, A_norm):
        """
        順伝播
        
        Args:
            X: torch.Tensor, shape (N, input_dim)
            A_norm: torch.Tensor, shape (N, N)
        
        Returns:
            A_recon: torch.Tensor, shape (N, N)
            mu: torch.Tensor, shape (N, latent_dim)
            logvar: torch.Tensor, shape (N, latent_dim)
            z: torch.Tensor, shape (N, latent_dim)
        """
        mu, logvar = self.encode(X, A_norm)
        z = self.reparameterize(mu, logvar)
        A_recon = self.decode(z)
        
        return A_recon, mu, logvar, z
    
    def encode_separate(self, X_WT, X_KO, A_norm):
        """
        WT群とKO群を分離してエンコード
        
        Args:
            X_WT: torch.Tensor, shape (N_WT, input_dim)
            X_KO: torch.Tensor, shape (N_KO, input_dim)
            A_norm: torch.Tensor, shape (N_WT+N_KO, N_WT+N_KO)
        
        Returns:
            z_WT: torch.Tensor, shape (N_WT, latent_dim)
            z_KO: torch.Tensor, shape (N_KO, latent_dim)
        """
        X_combined = torch.cat([X_WT, X_KO], dim=0)
        mu, logvar = self.encode(X_combined, A_norm)
        z = self.reparameterize(mu, logvar)
        
        N_WT = X_WT.shape[0]
        z_WT = z[:N_WT]
        z_KO = z[N_WT:]
        
        return z_WT, z_KO


def vgae_loss(A_true, A_recon, mu, logvar, pos_weight=None, beta=1.0):
    """
    VGAE損失関数: 再構成損失 + KL損失
    
    Args:
        A_true: torch.Tensor, shape (N, N) - 真の隣接行列
        A_recon: torch.Tensor, shape (N, N) - 再構成隣接行列
        mu: torch.Tensor, shape (N, latent_dim)
        logvar: torch.Tensor, shape (N, latent_dim)
        pos_weight: float or None - ポジティブエッジの重み
        beta: float - KL損失の重み
    
    Returns:
        loss: torch.Tensor, scalar
        recon_loss: torch.Tensor, scalar
        kl_loss: torch.Tensor, scalar
    """
    # 再構成損失（バイナリクロスエントロピー）
    N = A_true.size(0)  # ノード数
    
    if pos_weight is not None:
        # 重み付きBCE（スパースグラフ対応）
        weight = A_true * pos_weight + (1 - A_true)
        recon_loss = F.binary_cross_entropy(A_recon, A_true, weight=weight, reduction='sum')
    else:
        recon_loss = F.binary_cross_entropy(A_recon, A_true, reduction='sum')
    
    # ノード数で正規化（スケールを抑える）
    recon_loss = recon_loss / N
    
    # KL損失（ノード数で正規化）
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / N
    
    # 総損失
    loss = recon_loss + beta * kl_loss
    
    return loss, recon_loss, kl_loss


def cluster_alignment_loss(z, labels, normalize=True):
    """
    クラスター整列損失
    
    同一セルタイプの細胞を重心に引き寄せる
    
    Args:
        z: torch.Tensor, shape (N, latent_dim) - 潜在表現
        labels: torch.Tensor, shape (N,) - セルタイプラベル
        normalize: bool - 重心を単位球面上に正規化するか
    
    Returns:
        loss: torch.Tensor, scalar
    """
    # デバイスを揃える
    labels = labels.to(z.device)
    
    unique_labels = torch.unique(labels)
    num_clusters = len(unique_labels)
    
    total_loss = 0.0
    
    for label in unique_labels:
        # セルタイプcに属する細胞のインデックス
        mask = (labels == label)
        z_c = z[mask]
        
        if z_c.shape[0] == 0:
            continue
        
        # 重心の計算
        mu_c = z_c.mean(dim=0)
        
        # 単位球面上に正規化（オプション）
        if normalize:
            mu_c = F.normalize(mu_c, p=2, dim=0)
        
        # 各細胞と重心の距離（平均化）
        distances = torch.norm(z_c - mu_c, dim=1).pow(2)
        total_loss += distances.mean()  # sum() → mean() で安定化
    
    # 平均化
    loss = total_loss / num_clusters
    
    return loss
