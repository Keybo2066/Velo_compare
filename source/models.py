import torch
import torch.nn as nn
import torch.nn.functional as F

# EPS定数
EPS = 1e-6

def get_fully_connected_layers(input_dim, output_dim, hidden_dims, norm_type='batch', dropout_prob=0.1):
    """完全連結層のシーケンスを作成するヘルパー関数"""
    layers = []
    dims = [input_dim] + list(hidden_dims) + [output_dim]
    
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        
        # 最後の層以外に正規化とドロップアウトを追加
        if i < len(dims) - 2:
            if norm_type == 'batch':
                layers.append(nn.BatchNorm1d(dims[i+1]))
            elif norm_type == 'layer':
                layers.append(nn.LayerNorm(dims[i+1]))
            
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
    
    return nn.Sequential(*layers)

def symmetric_contrastive_loss(z_wt, z_ko, y_wt, y_ko, tau):
    """WTとKOの同一細胞タイプ間の対照損失"""
    # 共通の細胞タイプを特定
    shared_labels = set(y_wt.unique().cpu().tolist()).intersection(set(y_ko.unique().cpu().tolist()))
    
    total_loss = 0.0
    num_shared_types = 0
    
    for c in shared_labels:
        # 同じ細胞タイプのインデックスを抽出
        idx_wt = (y_wt == c).nonzero(as_tuple=True)[0]
        idx_ko = (y_ko == c).nonzero(as_tuple=True)[0]
        
        if len(idx_wt) == 0 or len(idx_ko) == 0:
            continue
            
        # 同じ細胞タイプの潜在表現を抽出
        zc_wt = z_wt[idx_wt]
        zc_ko = z_ko[idx_ko]
        
        # WT -> KO方向の非対称損失
        sim_matrix = torch.mm(zc_wt, zc_ko.T) / tau
        loss_wt_ko = -torch.log(
            torch.exp(sim_matrix) / 
            torch.exp(sim_matrix).sum(dim=1, keepdim=True)
        ).mean()
        
        # KO -> WT方向の非対称損失
        sim_matrix = torch.mm(zc_ko, zc_wt.T) / tau
        loss_ko_wt = -torch.log(
            torch.exp(sim_matrix) / 
            torch.exp(sim_matrix).sum(dim=1, keepdim=True)
        ).mean()
        
        # 対称的損失
        cell_type_loss = (loss_wt_ko + loss_ko_wt) / 2
        total_loss += cell_type_loss
        num_shared_types += 1
    
    if num_shared_types == 0:
        return torch.tensor(0.0, device=z_wt.device)
    
    return total_loss / num_shared_types

def cluster_alignment_loss(z, y):
    """同一細胞タイプの潜在表現をクラスタ中心に引き寄せる正則化項"""
    unique_labels = torch.unique(y)
    total_loss = 0.0
    
    for c in unique_labels:
        # 同じ細胞タイプのインデックスを抽出
        idx = (y == c).nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            continue
            
        # 同じ細胞タイプの潜在表現を抽出
        zc = z[idx]
        
        # クラスタ中心（重心）を計算
        mu_c = zc.mean(dim=0)
        
        # 中心からの距離の二乗和
        loss = ((zc - mu_c.unsqueeze(0)) ** 2).sum() / len(idx)
        total_loss += loss
    
    return total_loss / len(unique_labels)

class WTKOContrastiveVAE(nn.Module):
    """WTとKOのデータを対照学習するVAEモデル"""
    def __init__(self, input_dim, latent_dim=10, hidden_dims=(128,), 
                 tau=0.1, lambda_contrast=1.0, lambda_align=0.5, 
                 dropout_prob=0.1, norm_type='batch'):
        super().__init__()
        
        # エンコーダネットワーク
        self.encoder = get_fully_connected_layers(
            input_dim, latent_dim, hidden_dims, 
            norm_type=norm_type, dropout_prob=dropout_prob
        )
        
        # 濃度パラメータネットワーク
        self.kappa_encoder = get_fully_connected_layers(
            input_dim, 1, hidden_dims, 
            norm_type=norm_type, dropout_prob=dropout_prob
        )
        
        # デコーダネットワーク
        self.decoder = get_fully_connected_layers(
            latent_dim, input_dim, hidden_dims,
            norm_type=norm_type, dropout_prob=dropout_prob
        )
        
        # ハイパーパラメータ
        self.tau = tau
        self.lambda_contrast = lambda_contrast
        self.lambda_align = lambda_align
        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def encode(self, x):
        """エンコーダを通して分布のパラメータを取得"""
        mu = self.encoder(x)
        # L2正規化して単位球面上に拘束
        mu = F.normalize(mu, p=2, dim=1)
        # 濃度パラメータを計算
        kappa = F.softplus(self.kappa_encoder(x)) + EPS
        return mu, kappa
    
    def sample_z(self, mu, kappa):
        """球面分布からサンプリング (学習時は確率的、推論時は決定的)"""
        # 訓練時はランダム性を加え、推論時は平均値を使用
        if self.training:
            # 単純な近似: 平均方向にランダムなノイズを加える
            noise = torch.randn_like(mu) * torch.exp(-kappa)
            z = mu + noise
            z = F.normalize(z, p=2, dim=1)
            return None, z
        else:
            return None, mu
    
    def decode(self, z):
        """デコーダを通して再構成"""
        return self.decoder(z)
    
    def forward(self, data_dict):
        """モデルの順伝播"""
        # データの展開
        wt_x = data_dict['wt_x']
        ko_x = data_dict['ko_x']
        wt_labels = data_dict['wt_labels']
        ko_labels = data_dict['ko_labels']
        
        # エンコード
        wt_mu, wt_kappa = self.encode(wt_x)
        ko_mu, ko_kappa = self.encode(ko_x)
        
        # サンプリング
        _, wt_z = self.sample_z(wt_mu, wt_kappa)
        _, ko_z = self.sample_z(ko_mu, ko_kappa)
        
        # デコード
        wt_x_recon = self.decode(wt_z)
        ko_x_recon = self.decode(ko_z)
        
        # 再構成損失
        recon_loss = F.mse_loss(wt_x_recon, wt_x) + F.mse_loss(ko_x_recon, ko_x)
        
        # KLダイバージェンス (簡略化)
        kl_loss = (wt_kappa.mean() + ko_kappa.mean()) * 0.01
        
        # コントラスト損失
        contrast_loss = symmetric_contrastive_loss(wt_z, ko_z, wt_labels, ko_labels, self.tau)
        
        # クラスタ整列損失
        combined_z = torch.cat([wt_z, ko_z], dim=0)
        combined_labels = torch.cat([wt_labels, ko_labels], dim=0)
        align_loss = cluster_alignment_loss(combined_z, combined_labels)
        
        # 総損失
        total_loss = recon_loss + kl_loss + self.lambda_contrast * contrast_loss + self.lambda_align * align_loss
        #total_loss = recon_loss + kl_loss + self.lambda_contrast * contrast_loss 
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'contrast_loss': contrast_loss,
            'align_loss': align_loss,
            'wt_z': wt_z,
            'ko_z': ko_z
        }
    
    def encode_data(self, x):
        """データをエンコードして潜在表現を取得（推論用）"""
        mu, kappa = self.encode(x)
        _, z = self.sample_z(mu, kappa)
        return z
    
    def reconstruct_data(self, x):
        """データを再構成（推論用）"""
        z = self.encode_data(x)
        return self.decode(z)