# utils/data_loader.py

import torch
import numpy as np
from torch_geometric.datasets import BitcoinOTC
from torch_geometric.utils import degree, to_undirected, negative_sampling
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import math
import gc
import pandas as pd # 用于方便地处理时间戳和分组
import datetime # 用于处理日期


from .preprocess_utils import *

def generate_node_features(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """生成节点特征 (使用节点度)。"""
    if edge_index.numel() == 0: # 如果没有边，所有节点度数为0
        return torch.zeros((num_nodes, 1), dtype=torch.float, device=edge_index.device)
    deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)
    return deg.view(-1, 1)

def edge_index_to_set(edge_index: torch.Tensor) -> set:
    """将 edge_index 转换为无向边的集合，方便进行集合运算。"""
    if edge_index.numel() == 0: return set()
    edges = edge_index.cpu().numpy()
    edges_sorted = np.sort(edges, axis=0)
    return set(map(tuple, edges_sorted.T))

def set_to_edge_index(edge_set: set, device: torch.device) -> torch.Tensor:
    """将边的集合转换回 edge_index Tensor (确保输出是无向的)。"""
    if not edge_set: return torch.empty((2, 0), dtype=torch.long, device=device)
    edges_list = list(edge_set)
    if not edges_list: return torch.empty((2, 0), dtype=torch.long, device=device)
    edges = torch.tensor(edges_list, dtype=torch.long, device=device).t()
    return to_undirected(edges)

# --- 核心数据加载函数 ---
def load_event_data_by_time_window(
    data_path: str,
    time_window_days: int,
    feature_generator=generate_node_features,
    K_unibasis: int = 10, # 只保留 K_unibasis
    min_nodes_remap: bool = True,
    dataset_name: str = "generic_event_data"
) -> Tuple[List[Dict[str, torch.Tensor]], int, int]:

    print(f"开始加载 {dataset_name} 数据集从: {data_path}")
    print(f"  时间窗口: {time_window_days} 天, UniBasis K (for Homophily): {K_unibasis}")

    try:
        df = pd.read_csv(data_path, sep=r'\s+', header=None, comment='%',
                         names=['src', 'dst', 'weight', 'timestamp'])
    except (pd.errors.ParserError, ValueError): # ValueError for wrong number of columns
        try:
            print(f"  尝试解析为3列 (src, dst, timestamp) for {data_path}...")
            df = pd.read_csv(data_path, sep=r'\s+', header=None, comment='%',
                            names=['src', 'dst', 'timestamp'])
            df['weight'] = 1 # 默认权重为1
        except Exception as e:
            raise ValueError(f"无法解析数据文件 {data_path}: {e}")
    except FileNotFoundError:
        raise FileNotFoundError(f"数据文件未找到: {data_path}")
    print(f"  原始数据共 {len(df)} 条边/事件。")

    if min_nodes_remap:
        all_nodes_series = pd.concat([df['src'], df['dst']]).unique()
        node_map = {node_id: i for i, node_id in enumerate(all_nodes_series)}
        df['src'] = df['src'].map(node_map)
        df['dst'] = df['dst'].map(node_map)
        num_nodes = len(all_nodes_series)
        print(f"  节点 ID 已重新映射，总节点数: {num_nodes}")
    else:
        num_nodes = int(max(df['src'].max(), df['dst'].max()) + 1) if not df.empty else 0
        print(f"  使用原始节点 ID，推断总节点数: {num_nodes}")

    if num_nodes == 0:
        print("警告：数据中没有节点。")
        return [], 0, 0


    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.sort_values(by='datetime')
    min_timestamp = df['datetime'].min(); max_timestamp = df['datetime'].max()
    if pd.isna(min_timestamp) or pd.isna(max_timestamp):
        if len(df) > 0: raise ValueError("数据中缺少有效的时间戳。")
        else: # 如果 df 为空，直接返回空
            print("警告：数据文件为空或没有有效时间戳。")
            return [], num_nodes, 0


    print(f"  数据覆盖从 {min_timestamp} 到 {max_timestamp}。")
    time_window_delta = datetime.timedelta(days=time_window_days)
    snapshot_boundaries = []
    current_boundary_start = min_timestamp
    while current_boundary_start <= max_timestamp:
        snapshot_boundaries.append(current_boundary_start)
        current_boundary_start += time_window_delta
    if not snapshot_boundaries or snapshot_boundaries[-1] <= max_timestamp : # 确保至少有一个结束边界
         snapshot_boundaries.append((snapshot_boundaries[-1] if snapshot_boundaries else max_timestamp) + time_window_delta)
    print(f"  共生成 {max(0, len(snapshot_boundaries) - 1)} 个时间窗口用于构建快照。")

    snapshots_intermediate: List[Dict[str, torch.Tensor]] = []
    initial_feature_dim = -1

    print("步骤 1: 构建图快照, 计算P_t, X_t, 和共享同配基...")
    for i in tqdm(range(len(snapshot_boundaries) - 1), desc=f"构建 {dataset_name} 快照"):
        start_time_win = snapshot_boundaries[i]
        end_time_win = snapshot_boundaries[i+1]
        edges_in_window_df = df[(df['datetime'] >= start_time_win) & (df['datetime'] < end_time_win)]
        edges_for_snapshot_np = edges_in_window_df[['src', 'dst']].values.astype(np.int64)

        device = torch.device("cpu") # 初始数据都在 CPU 上处理

        if edges_for_snapshot_np.shape[0] == 0:
            edge_index_t = torch.empty((2, 0), dtype=torch.long, device=device)
        else:
            edge_index_t = torch.from_numpy(edges_for_snapshot_np.T).to(device)

        edge_index_t_undirected = to_undirected(edge_index_t, num_nodes=num_nodes)
        p_matrix_t = compute_propagation_matrix(edge_index_t_undirected, num_nodes, dtype=torch.float) # 指定 dtype
        initial_features_t = feature_generator(edge_index_t_undirected, num_nodes).to(device)

        if initial_feature_dim == -1 and initial_features_t.numel() > 0:
            initial_feature_dim = initial_features_t.shape[1]
        
        # 确保 initial_feature_dim 被设置
        current_initial_feature_dim = initial_feature_dim if initial_feature_dim != -1 else 1

        if initial_features_t.numel() == 0 and num_nodes > 0: # 如果快照没边但有节点
            initial_features_t = torch.zeros((num_nodes, current_initial_feature_dim), dtype=torch.float, device=device)


        if initial_features_t.numel() > 0 and p_matrix_t.numel() > 0 and p_matrix_t._nnz() > 0 :
            homophily_bases_t_list = compute_homophily_bases(p_matrix_t, initial_features_t, K_unibasis)
        else: # 如果没有特征或边，同配基就是K_unibasis+1个初始特征（可能全零）
            homophily_bases_t_list = [initial_features_t.clone() for _ in range(K_unibasis + 1)]

        snapshots_intermediate.append({
            'initial_features': initial_features_t,
            'p_matrix': p_matrix_t,
            'homophily_bases': homophily_bases_t_list,
            'current_edges': edge_index_t_undirected # 当前快照的边，用于下一个的标签
        })
    
    if initial_feature_dim == -1 : initial_feature_dim = 1 # 如果所有快照都没边，则默认特征维度为1


    snapshots_data_final: List[Dict[str, torch.Tensor]] = []
    print("\n步骤 2: 准备链接预测标签 (预测下一快照所有边)...")
    for t in tqdm(range(len(snapshots_intermediate) - 1), desc=f"准备 {dataset_name} 标签"):
        current_snap_info = snapshots_intermediate[t]
        next_snap_info = snapshots_intermediate[t+1]
        device = current_snap_info['initial_features'].device

        pos_edge_index_label = next_snap_info['current_edges']
        num_pos_samples = pos_edge_index_label.size(1)
        num_neg_to_sample = num_pos_samples if num_pos_samples > 0 else max(100, num_nodes // 10) # 保证有负样本

        neg_edge_index_label = negative_sampling(
            edge_index=pos_edge_index_label, num_nodes=num_nodes,
            num_neg_samples=num_neg_to_sample, method='sparse'
        ).to(device)

        snapshots_data_final.append({
            'initial_features': current_snap_info['initial_features'],
            'p_matrix': current_snap_info['p_matrix'],
            'homophily_bases': current_snap_info['homophily_bases'],
            'pos_edge_index': pos_edge_index_label,
            'neg_edge_index': neg_edge_index_label
        })

    print(f"\n{dataset_name} 数据处理完成。共生成 {len(snapshots_data_final)} 个 (特征, 标签对) 快照。")
    print(f"  节点数量: {num_nodes}")
    print(f"  初始节点特征维度 F: {initial_feature_dim}")
    return snapshots_data_final, num_nodes, initial_feature_dim


def load_uci(
    data_path: str,
    K_unibasis: int, # 保留 K_unibasis
    feature_generator=generate_node_features,
    min_nodes_remap: bool = True
) -> Tuple[List[Dict[str, torch.Tensor]], int, int]:
    return load_event_data_by_time_window(
        data_path=data_path, time_window_days=1,
        feature_generator=feature_generator,
        K_unibasis=K_unibasis, # 传递 K_unibasis
        min_nodes_remap=min_nodes_remap,
        dataset_name="UC Irvine Messages"
    )

def load_enron_email_data(
    data_path: str, time_window_days: int, K_unibasis: int, tau: float,
    h_hat_channel1: Optional[float], h_hat_channel2: Optional[float],
    feature_generator=generate_node_features, min_nodes_remap: bool = True
) -> Tuple[List[Dict[str, torch.Tensor]], int, int]:
    return load_event_data_by_time_window(
        data_path=data_path, time_window_days=time_window_days, # Enron 按指定天数
        feature_generator=feature_generator, K_unibasis=K_unibasis,
        min_nodes_remap=min_nodes_remap, dataset_name="Enron Email"
    )


def load_bitcoin_otc_data(
    root: str, edge_window_size: int, K_unibasis: int, tau: float,
    h_hat_channel1: Optional[float], h_hat_channel2: Optional[float],
    feature_generator=generate_node_features
) -> Tuple[List[Dict[str, torch.Tensor]], int, int]:
    """
    加载并适配 BitcoinOTC 数据集以输出类似格式。
    """
    print(f"加载 BitcoinOTC 数据集 (edge_window_size={edge_window_size})...")
    dataset = BitcoinOTC(root=root, edge_window_size=edge_window_size)
    num_nodes = dataset[0].num_nodes
    initial_feature_dim = -1
    snapshots_intermediate_btc: List[Dict[str, torch.Tensor]] = []

    print("步骤 1: 处理 BitcoinOTC 快照并计算同配基...")
    for t, data_t in enumerate(tqdm(dataset, desc="处理 BitcoinOTC 快照")):
        edge_index_t = to_undirected(data_t.edge_index, num_nodes=num_nodes) # 已经是快照的边
        device = edge_index_t.device

        p_matrix_t = compute_propagation_matrix(edge_index_t, num_nodes, dtype=torch.float)
        initial_features_t = feature_generator(edge_index_t, num_nodes).to(device)

        if initial_feature_dim == -1 and initial_features_t.numel() > 0:
            initial_feature_dim = initial_features_t.shape[1]
        
        current_initial_feature_dim = initial_feature_dim if initial_feature_dim != -1 else 1
        if initial_features_t.numel() == 0 and num_nodes > 0:
             initial_features_t = torch.zeros((num_nodes, current_initial_feature_dim), dtype=torch.float, device=device)


        if initial_features_t.numel() > 0 and p_matrix_t.numel() > 0 and p_matrix_t._nnz() > 0:
            homophily_bases_t_list = compute_homophily_bases(p_matrix_t, initial_features_t, K_unibasis)
        else:
            homophily_bases_t_list = [initial_features_t.clone() for _ in range(K_unibasis + 1)]

        snapshots_intermediate_btc.append({
            'initial_features': initial_features_t,
            'p_matrix': p_matrix_t,
            'homophily_bases': homophily_bases_t_list,
            'current_edges': edge_index_t # 当前快照的边
        })
    if initial_feature_dim == -1 : initial_feature_dim = 1

    snapshots_data_final_btc: List[Dict[str, torch.Tensor]] = []
    print("\n步骤 2: 准备 BitcoinOTC 链接预测标签...")
    for t in tqdm(range(len(snapshots_intermediate_btc) - 1), desc="准备 BitcoinOTC 标签"):
        # ... (与 load_event_data_by_time_window 中准备标签的逻辑完全相同) ...
        current_snap_info = snapshots_intermediate_btc[t]
        next_snap_info = snapshots_intermediate_btc[t+1]
        device = current_snap_info['initial_features'].device
        pos_edge_index_label = next_snap_info['current_edges']
        num_pos_samples = pos_edge_index_label.size(1); num_neg_to_sample = num_pos_samples if num_pos_samples > 0 else max(100, num_nodes // 10)
        neg_edge_index_label = negative_sampling(edge_index=pos_edge_index_label, num_nodes=num_nodes, num_neg_samples=num_neg_to_sample, method='sparse').to(device)
        snapshots_data_final_btc.append({
            'initial_features': current_snap_info['initial_features'],
            'p_matrix': current_snap_info['p_matrix'],
            'homophily_bases': current_snap_info['homophily_bases'],
            'pos_edge_index': pos_edge_index_label,
            'neg_edge_index': neg_edge_index_label
        })

    print(f"\nBitcoinOTC 数据处理完成。共生成 {len(snapshots_data_final_btc)} 个 (特征, 标签对) 快照。")
    print(f"  节点数量: {num_nodes}")
    print(f"  初始节点特征维度 F: {initial_feature_dim}")
    return snapshots_data_final_btc, num_nodes, initial_feature_dim

def get_dynamic_data_splits(num_time_steps: int,
                            train_ratio: float = 0.7,
                            val_ratio: float = 0.15
                           ) -> Tuple[List[int], List[int], List[int]]:
    T = num_time_steps
    T_train = int(T * train_ratio)
    T_val = int(T * val_ratio)
    T_test = T - T_train - T_val

    if T_train <= 0 or T_val <= 0 or T_test <= 0:
        raise ValueError("划分比例导致某个集合的时间步数量小于等于 0")

    # 时间步索引从 0 开始
    train_steps = list(range(T_train))
    val_steps = list(range(T_train, T_train + T_val))
    test_steps = list(range(T_train + T_val, T))

    print(f"时间步划分: Train={len(train_steps)} ({train_steps[0]}-{train_steps[-1]}), "
          f"Val={len(val_steps)} ({val_steps[0]}-{val_steps[-1]}), "
          f"Test={len(test_steps)} ({test_steps[0]}-{test_steps[-1]})")

    return train_steps, val_steps, test_steps

