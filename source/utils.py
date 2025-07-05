# UMAP可視化関連の関数

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binned_statistic_2d
import matplotlib.patches as mpatches

def plot_latent_space(wt_embedding, ko_embedding, wt_labels, ko_labels, cell_type_names, lambda_contrast=1.0, lambda_align=1.0):
    """
    WTとKO細胞の潜在空間を分割表示
    
    Parameters:
    -----------
    wt_embedding : numpy.ndarray
        WT細胞のUMAP埋め込み (n_cells, 2)
    ko_embedding : numpy.ndarray
        KO細胞のUMAP埋め込み (n_cells, 2)
    wt_labels : numpy.ndarray
        WT細胞のラベル (n_cells,)
    ko_labels : numpy.ndarray
        KO細胞のラベル (n_cells,)
    cell_type_names : list
        細胞タイプの名前リスト
    lambda_contrast : float, optional
        コントラスト損失の重み
    lambda_align : float, optional
        整列損失の重み
    """
    # カラーマップの設定
    n_cell_types = len(cell_type_names)
    colors = plt.cm.tab10(np.linspace(0, 1, n_cell_types))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # メインタイトルにパラメータ情報を追加
    fig.suptitle(f'Latent Space Visualization (λ_contrast={lambda_contrast}, λ_align={lambda_align})', 
                fontsize=16)
    
    # WT細胞の可視化
    for i, ct in enumerate(range(n_cell_types)):
        mask = wt_labels == ct
        if mask.sum() > 0:
            ax1.scatter(
                wt_embedding[mask, 0], 
                wt_embedding[mask, 1],
                c=[colors[i]],
                label=cell_type_names[ct],
                s=50,
                alpha=0.7
            )
    
    ax1.set_title('WT Cells - Latent Space (UMAP)', fontsize=14)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # KO細胞の可視化
    for i, ct in enumerate(range(n_cell_types)):
        mask = ko_labels == ct
        if mask.sum() > 0:
            ax2.scatter(
                ko_embedding[mask, 0], 
                ko_embedding[mask, 1],
                c=[colors[i]],
                label=cell_type_names[ct],
                s=50,
                alpha=0.7
            )
    
    ax2.set_title('KO Cells - Latent Space (UMAP)', fontsize=14)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # タイトル用のスペースを確保
    plt.show()

def plot_combined_latent_space(wt_embedding, ko_embedding, wt_labels, ko_labels, cell_type_names):
    """
    WTとKO細胞を同一空間に可視化（色＝細胞タイプ、マーカー＝WT/KO）
    
    Parameters:
    -----------
    wt_embedding : numpy.ndarray
        WT細胞のUMAP埋め込み (n_cells, 2)
    ko_embedding : numpy.ndarray
        KO細胞のUMAP埋め込み (n_cells, 2)
    wt_labels : numpy.ndarray
        WT細胞のラベル (n_cells,)
    ko_labels : numpy.ndarray
        KO細胞のラベル (n_cells,)
    cell_type_names : list
        細胞タイプの名前リスト
    """
    n_cell_types = len(cell_type_names)
    colors = plt.cm.tab10(np.linspace(0, 1, n_cell_types))

    # 結合
    combined_embedding = np.vstack([wt_embedding, ko_embedding])
    combined_labels = np.concatenate([wt_labels, ko_labels])
    combined_group = np.array(['WT'] * len(wt_labels) + ['KO'] * len(ko_labels))
    
    # 描画
    plt.figure(figsize=(10, 8))
    
    # 凡例用のハンドル
    legend_handles = []
    
    # 細胞タイプごとの描画
    for i in range(n_cell_types):
        for group, marker in zip(['WT', 'KO'], ['o', 's']):
            mask = (combined_labels == i) & (combined_group == group)
            if mask.sum() > 0:
                scatter = plt.scatter(
                    combined_embedding[mask, 0],
                    combined_embedding[mask, 1],
                    c=[colors[i]],
                    s=50,
                    alpha=0.7,
                    marker=marker,
                    edgecolor='k' if group == 'KO' else 'none',
                    linewidths=0.5
                )
                
                # 明示的な凡例エントリの追加
                legend_handles.append(
                    mpatches.Patch(color=colors[i], label=f"{cell_type_names[i]} ({group})")
                )
    
    # 凡例を整理（重複排除）
    handles_dict = {h.get_label(): h for h in legend_handles}
    legend_handles = list(handles_dict.values())
    
    plt.title('WT and KO Cells - Combined Latent Space (UMAP)', fontsize=14)
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def compute_grid_velocities(X, V, grid_size=20, min_cells=5):
    """
    グリッドベースの速度場を計算
    
    Parameters:
    -----------
    X : numpy.ndarray
        細胞の座標 (n_cells, 2)
    V : numpy.ndarray
        細胞の速度ベクトル (n_cells, 2)
    grid_size : int, optional
        グリッドのサイズ
    min_cells : int, optional
        グリッド内の最小細胞数
        
    Returns:
    --------
    X_grid : numpy.ndarray
        グリッドの中心座標
    V_grid : numpy.ndarray
        グリッドの平均速度ベクトル
    cell_counts : numpy.ndarray
        各グリッドの細胞数
    grid_edges : tuple
        グリッドのエッジ座標
    mask : numpy.ndarray
        有効なグリッドのマスク
    """
    # 座標の範囲を取得
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    
    # マージンを追加
    margin_x = (x_max - x_min) * 0.1
    margin_y = (y_max - y_min) * 0.1
    x_min -= margin_x
    x_max += margin_x
    y_min -= margin_y
    y_max += margin_y
    
    # グリッドのエッジを定義
    x_edges = np.linspace(x_min, x_max, grid_size + 1)
    y_edges = np.linspace(y_min, y_max, grid_size + 1)
    
    # 各グリッドの中心座標を計算
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    xx, yy = np.meshgrid(x_centers, y_centers)
    X_grid = np.column_stack([xx.ravel(), yy.ravel()])
    
    # 各グリッドの統計量を計算
    vx_mean, _, _, _ = binned_statistic_2d(
        X[:, 0], X[:, 1], V[:, 0], 
        statistic='mean', bins=[x_edges, y_edges]
    )
    
    vy_mean, _, _, _ = binned_statistic_2d(
        X[:, 0], X[:, 1], V[:, 1], 
        statistic='mean', bins=[x_edges, y_edges]
    )
    
    cell_counts, _, _, _ = binned_statistic_2d(
        X[:, 0], X[:, 1], V[:, 0], 
        statistic='count', bins=[x_edges, y_edges]
    )
    
    # グリッドベースの速度ベクトルを作成
    V_grid = np.zeros((grid_size * grid_size, 2))
    V_grid[:, 0] = vx_mean.T.ravel()
    V_grid[:, 1] = vy_mean.T.ravel()
    
    # NaNを0に置換
    V_grid = np.nan_to_num(V_grid)
    cell_counts = cell_counts.T.ravel()
    
    # 細胞数が少ないグリッドをマスク
    mask = cell_counts >= min_cells
    
    return X_grid[mask], V_grid[mask], cell_counts[mask], (x_edges, y_edges), mask

def plot_latent_space_with_grid_velocity(wt_embedding, ko_embedding, wt_labels, ko_labels, 
                                         cell_type_names, wt_velocity, ko_velocity,
                                         grid_size=20, min_cells=5, arrow_scale=30, 
                                         show_grid=True, show_cell_counts=False):
    """
    グリッドベースの速度場を可視化
    
    Parameters:
    -----------
    wt_embedding, ko_embedding : numpy.ndarray
        WT/KO細胞のUMAP埋め込み (n_cells, 2)
    wt_labels, ko_labels : numpy.ndarray
        WT/KO細胞のラベル (n_cells,)
    cell_type_names : list
        細胞タイプの名前リスト
    wt_velocity, ko_velocity : numpy.ndarray
        WT/KO細胞の速度ベクトル (n_cells, 2)
    grid_size : int, optional
        グリッドのサイズ
    min_cells : int, optional
        グリッド内の最小細胞数
    arrow_scale : float, optional
        矢印のスケール
    show_grid : bool, optional
        グリッド線を表示するか
    show_cell_counts : bool, optional
        各グリッドの細胞数を表示するか
    """
    # カラーマップの設定
    n_cell_types = len(cell_type_names)
    colors = plt.cm.tab10(np.linspace(0, 1, n_cell_types))
    
    # 可視化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    
    # WT細胞の可視化
    for i, ct in enumerate(range(n_cell_types)):
        mask = wt_labels == ct
        if np.sum(mask) > 0:
            ax1.scatter(
                wt_embedding[mask, 0], 
                wt_embedding[mask, 1],
                c=[colors[i]],
                label=cell_type_names[ct],
                s=30,
                alpha=0.6,
                edgecolor='none'
            )
    
    # WTのグリッドベース速度場を計算
    X_grid_wt, V_grid_wt, counts_wt, edges_wt, mask_wt = compute_grid_velocities(
        wt_embedding, wt_velocity, grid_size=grid_size, min_cells=min_cells
    )
    
    # グリッド線の表示
    if show_grid:
        x_edges, y_edges = edges_wt
        for x in x_edges:
            ax1.axvline(x, color='gray', alpha=0.2, linewidth=0.5)
        for y in y_edges:
            ax1.axhline(y, color='gray', alpha=0.2, linewidth=0.5)
    
    # 細胞数の表示（オプション）
    if show_cell_counts:
        for i, (x, y) in enumerate(X_grid_wt):
            ax1.text(x, y, str(int(counts_wt[i])), 
                    fontsize=8, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.5))
    
    # 速度ベクトルの描画（矢印）
    for i in range(len(X_grid_wt)):
        ax1.arrow(
            X_grid_wt[i, 0], X_grid_wt[i, 1],
            V_grid_wt[i, 0] * arrow_scale, V_grid_wt[i, 1] * arrow_scale,
            head_width=0.3,
            head_length=0.2,
            fc='black',
            ec='black',
            alpha=0.7,
            length_includes_head=True
        )
    
    # 軸メモリの設定
    ax1.set_title('WT Cells - Grid-based Velocity Field', fontsize=14)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.set_xlabel('UMAP 1')
    ax1.set_ylabel('UMAP 2')
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    
    # KO細胞の可視化（同様の処理）
    for i, ct in enumerate(range(n_cell_types)):
        mask = ko_labels == ct
        if np.sum(mask) > 0:
            ax2.scatter(
                ko_embedding[mask, 0], 
                ko_embedding[mask, 1],
                c=[colors[i]],
                label=cell_type_names[ct],
                s=30,
                alpha=0.6,
                edgecolor='none'
            )
    
    # KOのグリッドベース速度場を計算
    X_grid_ko, V_grid_ko, counts_ko, edges_ko, mask_ko = compute_grid_velocities(
        ko_embedding, ko_velocity, grid_size=grid_size, min_cells=min_cells
    )
    
    # グリッド線の表示
    if show_grid:
        x_edges, y_edges = edges_ko
        for x in x_edges:
            ax2.axvline(x, color='gray', alpha=0.2, linewidth=0.5)
        for y in y_edges:
            ax2.axhline(y, color='gray', alpha=0.2, linewidth=0.5)
    
    # 細胞数の表示（オプション）
    if show_cell_counts:
        for i, (x, y) in enumerate(X_grid_ko):
            ax2.text(x, y, str(int(counts_ko[i])), 
                    fontsize=8, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.5))
    
    # 速度ベクトルの描画（矢印）
    for i in range(len(X_grid_ko)):
        ax2.arrow(
            X_grid_ko[i, 0], X_grid_ko[i, 1],
            V_grid_ko[i, 0] * arrow_scale, V_grid_ko[i, 1] * arrow_scale,
            head_width=0.3,
            head_length=0.2,
            fc='black',
            ec='black',
            alpha=0.7,
            length_includes_head=True
        )
    
    # 軸メモリの設定
    ax2.set_title('KO Cells - Grid-based Velocity Field', fontsize=14)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.set_xlabel('UMAP 1')
    ax2.set_ylabel('UMAP 2')
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # 統計情報を表示
    print(f"WT: {len(X_grid_wt)} grids with velocity vectors")
    print(f"KO: {len(X_grid_ko)} grids with velocity vectors")
    print(f"Average cells per grid (WT): {counts_wt.mean():.1f}")
    print(f"Average cells per grid (KO): {counts_ko.mean():.1f}")
    
    # 速度の統計情報
    v_mag_wt = np.linalg.norm(V_grid_wt, axis=1)
    v_mag_ko = np.linalg.norm(V_grid_ko, axis=1)
    print(f"Average velocity magnitude (WT): {v_mag_wt.mean():.6f}")
    print(f"Average velocity magnitude (KO): {v_mag_ko.mean():.6f}")
    
    return X_grid_wt, V_grid_wt, counts_wt, X_grid_ko, V_grid_ko, counts_ko

def plot_grid_statistics(V_grid_wt, V_grid_ko, counts_wt, counts_ko):
    """
    グリッドごとの統計情報を可視化
    
    Parameters:
    -----------
    V_grid_wt, V_grid_ko : numpy.ndarray
        WT/KOのグリッド速度ベクトル
    counts_wt, counts_ko : numpy.ndarray
        WT/KOの各グリッドの細胞数
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 速度の大きさの分布
    v_mag_wt = np.linalg.norm(V_grid_wt, axis=1)
    v_mag_ko = np.linalg.norm(V_grid_ko, axis=1)
    
    axes[0, 0].hist(v_mag_wt, bins=20, alpha=0.7, label='WT', color='blue')
    axes[0, 0].hist(v_mag_ko, bins=20, alpha=0.7, label='KO', color='red')
    axes[0, 0].set_xlabel('Velocity Magnitude')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Distribution of Grid Velocity Magnitudes')
    axes[0, 0].legend()
    
    # 細胞数の分布
    axes[0, 1].hist(counts_wt, bins=20, alpha=0.7, label='WT', color='blue')
    axes[0, 1].hist(counts_ko, bins=20, alpha=0.7, label='KO', color='red')
    axes[0, 1].set_xlabel('Cell Count')
    axes[0, 1].set_ylabel('Number of Grids')
    axes[0, 1].set_title('Distribution of Cells per Grid')
    axes[0, 1].legend()
    
    # 速度成分の散布図
    axes[1, 0].scatter(V_grid_wt[:, 0], V_grid_wt[:, 1], 
                      alpha=0.6, label='WT', color='blue', s=50)
    axes[1, 0].scatter(V_grid_ko[:, 0], V_grid_ko[:, 1], 
                      alpha=0.6, label='KO', color='red', s=50)
    axes[1, 0].set_xlabel('Velocity X Component')
    axes[1, 0].set_ylabel('Velocity Y Component')
    axes[1, 0].set_title('Grid Velocity Components')
    axes[1, 0].legend()
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 0].axvline(x=0, color='k', linestyle='--', alpha=0.5)
    
    # 細胞数 vs 速度の大きさ
    axes[1, 1].scatter(counts_wt, v_mag_wt, alpha=0.6, label='WT', color='blue', s=50)
    axes[1, 1].scatter(counts_ko, v_mag_ko, alpha=0.6, label='KO', color='red', s=50)
    axes[1, 1].set_xlabel('Cell Count per Grid')
    axes[1, 1].set_ylabel('Velocity Magnitude')
    axes[1, 1].set_title('Cell Count vs Velocity Magnitude')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()