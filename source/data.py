import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class WTKODataset(Dataset):
    """WTとKOのscRNA-seqデータセット"""
    def __init__(self, data, labels=None):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        return self.data[idx]

def create_wt_ko_dataloaders(wt_data, wt_labels, ko_data, ko_labels, batch_size=64, shuffle=True):
    """WTとKOのデータローダーを作成"""
    # NumPy配列をTensorに変換
    if isinstance(wt_data, np.ndarray):
        wt_data = torch.tensor(wt_data, dtype=torch.float32)
    if isinstance(wt_labels, np.ndarray) and wt_labels is not None:
        wt_labels = torch.tensor(wt_labels, dtype=torch.long)
    if isinstance(ko_data, np.ndarray):
        ko_data = torch.tensor(ko_data, dtype=torch.float32)
    if isinstance(ko_labels, np.ndarray) and ko_labels is not None:
        ko_labels = torch.tensor(ko_labels, dtype=torch.long)
    
    # データセット作成
    wt_dataset = WTKODataset(wt_data, wt_labels)
    ko_dataset = WTKODataset(ko_data, ko_labels)
    
    # データローダー作成
    wt_loader = DataLoader(
        wt_dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False
    )
    
    ko_loader = DataLoader(
        ko_dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False
    )
    
    return wt_loader, ko_loader
