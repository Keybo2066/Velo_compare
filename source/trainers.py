import torch
import torch.optim as optim
from tqdm.notebook import tqdm
import os

class WTKOTrainer:
    """WTKOContrastiveVAEモデルを訓練するためのクラス"""
    def __init__(self, model, device=None):
        self.model = model
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def train(self, 
              wt_loader, 
              ko_loader, 
              num_epochs=100, 
              lr=1e-3, 
              weight_decay=1e-5, 
              save_path=None, 
              verbose=True):
        """モデルの学習を実行"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # 学習履歴
        history = {
            'loss': [],
            'recon_loss': [],
            'kl_loss': [],
            'contrast_loss': [],
            'align_loss': []
        }
        
        # メインの進捗バー（エポック用）
        epoch_bar = tqdm(range(num_epochs), desc="Training")
        
        # 学習ループ
        for epoch in epoch_bar:
            self.model.train()
            epoch_losses = {k: 0.0 for k in history.keys()}
            num_batches = min(len(wt_loader), len(ko_loader))
            
            # データローダーの反復子を取得
            wt_iter = iter(wt_loader)
            ko_iter = iter(ko_loader)
            
            for _ in range(num_batches):
                # バッチデータの取得
                try:
                    wt_batch = next(wt_iter)
                    ko_batch = next(ko_iter)
                except StopIteration:
                    break
                
                # デバイスに転送
                wt_x = wt_batch[0].to(self.device)
                wt_labels = wt_batch[1].to(self.device)
                ko_x = ko_batch[0].to(self.device)
                ko_labels = ko_batch[1].to(self.device)
                
                # データ辞書の作成
                data_dict = {
                    'wt_x': wt_x,
                    'ko_x': ko_x,
                    'wt_labels': wt_labels,
                    'ko_labels': ko_labels
                }
                
                # 勾配のリセット
                optimizer.zero_grad()
                
                # 順伝播
                output = self.model(data_dict)
                
                # 損失計算
                loss = output['loss']
                
                # 逆伝播と最適化
                loss.backward()
                optimizer.step()
                
                # 損失の記録
                for k in epoch_losses.keys():
                    epoch_losses[k] += output[k].item() if k in output else 0.0
            
            # エポック平均損失
            for k in epoch_losses.keys():
                epoch_losses[k] /= num_batches
                history[k].append(epoch_losses[k])
            
            # 進捗バーの説明を更新
            epoch_bar.set_postfix({
                'loss': f"{epoch_losses['loss']:.4f}",
                'recon': f"{epoch_losses['recon_loss']:.4f}",
                'contrast': f"{epoch_losses['contrast_loss']:.4f}"
            })
            
            # モデル保存（オプション）
            if save_path is not None and (epoch + 1) % 10 == 0:
                save_dir = os.path.dirname(save_path)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_file = f"{save_path}_epoch_{epoch+1}.pt"
                torch.save(self.model.state_dict(), save_file)
        
        # 最終モデル保存（オプション）
        if save_path is not None:
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_file = f"{save_path}_final.pt"
            torch.save(self.model.state_dict(), save_file)
            
        return history
    
    def get_latent_representations(self, loader):
        """データの潜在表現を取得"""
        self.model.eval()
        latent_reps = []
        labels = []
        
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(self.device)
                if len(batch) > 1:
                    y = batch[1].to(self.device)
                    labels.append(y)
                
                z = self.model.encode_data(x)
                latent_reps.append(z)
        
        latent_reps = torch.cat(latent_reps, dim=0)
        
        if len(labels) > 0:
            labels = torch.cat(labels, dim=0)
            return latent_reps, labels
        
        return latent_reps