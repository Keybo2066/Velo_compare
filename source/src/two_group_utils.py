"""
2群scRNA-seq比較のためのグラフ構築ユーティリティ
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp


def build_knn_graph(X, k=10, symmetric=True):
    """
    群内KNNグラフを構築
    
    Args:
        X: np.ndarray, shape (N, G) - 遺伝子発現量行列
        k: int - 近傍数
        symmetric: bool - 対称化するか
    
    Returns:
        A: np.ndarray, shape (N, N) - 隣接行列（対角要素=1を含む）
    """
    N = X.shape[0]
    
    # KNN探索
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    # 隣接行列の構築
    A = np.zeros((N, N))
    for i in range(N):
        for j in indices[i, 1:]:  # 自分自身を除く
            A[i, j] = 1
    
    # 対称化
    if symmetric:
        A = np.maximum(A, A.T)
    
    # 自己ループを追加
    A = A + np.eye(N)
    
    return A


def create_cross_edges_knn(X_WT, X_KO, labels_WT, labels_KO, k=5):
    """
    群間エッジを構築（同一セルタイプ内でKNN接続）
    
    Args:
        X_WT: np.ndarray, shape (N_WT, G) - WT群の遺伝子発現量
        X_KO: np.ndarray, shape (N_KO, G) - KO群の遺伝子発現量
        labels_WT: np.ndarray, shape (N_WT,) - WT群のセルタイプラベル
        labels_KO: np.ndarray, shape (N_KO,) - KO群のセルタイプラベル
        k: int - 近傍数
    
    Returns:
        A_cross: np.ndarray, shape (N_WT, N_KO) - 群間隣接行列
        common_types: set - 共通セルタイプの集合
    """
    N_WT, N_KO = len(labels_WT), len(labels_KO)
    A_cross = np.zeros((N_WT, N_KO))
    
    # 共通セルタイプを特定
    common_types = set(labels_WT) & set(labels_KO)
    
    if len(common_types) == 0:
        print("Warning: No common cell types found!")
        return A_cross, common_types
    
    # 各セルタイプごとに群間エッジを構築
    for cell_type in common_types:
        idx_WT = np.where(labels_WT == cell_type)[0]
        idx_KO = np.where(labels_KO == cell_type)[0]
        
        if len(idx_WT) == 0 or len(idx_KO) == 0:
            continue
        
        # WT→KOの最近傍
        X_WT_ct = X_WT[idx_WT]
        X_KO_ct = X_KO[idx_KO]
        
        k_actual = min(k, len(idx_KO))
        nbrs = NearestNeighbors(n_neighbors=k_actual, algorithm='auto').fit(X_KO_ct)
        distances, indices = nbrs.kneighbors(X_WT_ct)
        
        for i, wt_idx in enumerate(idx_WT):
            for j in indices[i]:
                ko_idx = idx_KO[j]
                A_cross[wt_idx, ko_idx] = 1
        
        # KO→WTの最近傍（対称化）
        k_actual = min(k, len(idx_WT))
        nbrs = NearestNeighbors(n_neighbors=k_actual, algorithm='auto').fit(X_WT_ct)
        distances, indices = nbrs.kneighbors(X_KO_ct)
        
        for j, ko_idx in enumerate(idx_KO):
            for i in indices[j]:
                wt_idx = idx_WT[i]
                A_cross[wt_idx, ko_idx] = 1
    
    return A_cross, common_types


def build_combined_graph(X_WT, X_KO, labels_WT, labels_KO, 
                         k_intra=10, k_cross=5):
    """
    統合グラフを構築
    
    Args:
        X_WT: np.ndarray, shape (N_WT, G)
        X_KO: np.ndarray, shape (N_KO, G)
        labels_WT: np.ndarray, shape (N_WT,)
        labels_KO: np.ndarray, shape (N_KO,)
        k_intra: int - 群内KNN近傍数
        k_cross: int - 群間KNN近傍数
    
    Returns:
        A_combined: np.ndarray, shape (N_WT+N_KO, N_WT+N_KO)
        common_types: set
    """
    print(f"Building combined graph: k_intra={k_intra}, k_cross={k_cross}")
    
    N_WT, N_KO = X_WT.shape[0], X_KO.shape[0]
    N_total = N_WT + N_KO
    
    # 群内グラフ
    print("Building intra-group graphs...")
    A_WT = build_knn_graph(X_WT, k=k_intra)
    A_KO = build_knn_graph(X_KO, k=k_intra)
    
    # 群間エッジ
    print("Building cross-group edges...")
    A_cross, common_types = create_cross_edges_knn(
        X_WT, X_KO, labels_WT, labels_KO, k=k_cross
    )
    
    print(f"Common cell types: {common_types}")
    print(f"Cross edges: {A_cross.sum():.0f} connections")
    
    # 統合
    A_combined = np.zeros((N_total, N_total))
    A_combined[:N_WT, :N_WT] = A_WT
    A_combined[N_WT:, N_WT:] = A_KO
    A_combined[:N_WT, N_WT:] = A_cross
    A_combined[N_WT:, :N_WT] = A_cross.T
    
    print(f"Combined graph shape: {A_combined.shape}")
    print(f"Total edges: {A_combined.sum():.0f}")
    
    return A_combined, common_types


def normalize_adjacency(A):
    """
    隣接行列を正規化: D^(-1/2) A D^(-1/2)
    
    Args:
        A: np.ndarray, shape (N, N) - 隣接行列
    
    Returns:
        A_norm: np.ndarray, shape (N, N) - 正規化隣接行列
    """
    # 次数行列
    D = np.array(A.sum(axis=1)).flatten()
    D[D == 0] = 1  # ゼロ除算を防ぐ
    D_inv_sqrt = 1.0 / np.sqrt(D)
    
    # NaN/Infチェック
    D_inv_sqrt = np.nan_to_num(D_inv_sqrt, nan=0.0, posinf=0.0, neginf=0.0)
    
    D_inv_sqrt_mat = np.diag(D_inv_sqrt)
    
    # D^(-1/2) A D^(-1/2)
    A_norm = D_inv_sqrt_mat @ A @ D_inv_sqrt_mat
    
    # 最終チェック
    A_norm = np.nan_to_num(A_norm, nan=0.0, posinf=0.0, neginf=0.0)
    
    return A_norm
