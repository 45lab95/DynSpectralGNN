import torch
import numpy as np
import pandas as pd
import os
import datetime
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import argparse

# 假设 utils 目录与此脚本在同一父目录下，或者 utils 已被正确安装或添加到 PYTHONPATH
# 如果直接在 dynspec 根目录运行，可以 from utils. ...
# 如果此脚本在 utils 外部，例如 dynspec/scripts/preprocess_dynamic_data.py
# 则需要调整导入路径，例如 from ..utils.preprocess_utils import ...
try:
    from utils.preprocess_utils import *
    from utils.data_loader import *
except ImportError:
    print("请确保 preprocess_utils.py, unibasis_utils.py (或其合并版本) 和 data_loader.py 在正确的路径下或已安装。")
    print("如果此脚本在项目根目录，尝试使用: ")
    print("from utils.preprocess_utils import ...")
    print("from utils.unibasis_utils import ...") # 如果它独立存在
    print("from utils.data_loader import generate_node_features")
    exit()


def preprocess_and_save_snapshots(
    data_path: str,
    output_dir: str, # 保存 .pt 文件的目录
    dataset_name: str,
    time_window_days: int,
    K_unibasis: int,
    tau_unibasis: float, # UniBasis tau, 所有视图共享
    h_hat_views: Dict[str, float], # 例如 {'ch1': 0.2, 'ch1_prime': 0.1, ...}
    feature_generator=generate_node_features,
    min_nodes_remap: bool = True
):
    """
    加载事件数据，按时间窗口聚合，计算所有视图的 UniBasis 特征，
    并为每个 (输入特征, 标签) 对保存为一个 .pt 文件。
    """
    print(f"开始预处理 {dataset_name} 数据从: {data_path}")
    print(f"  输出目录: {output_dir}")
    print(f"  时间窗口: {time_window_days} 天, UniBasis K: {K_unibasis}, Tau: {tau_unibasis}")
    print(f"  H_hat 视图配置: {h_hat_views}")

    os.makedirs(output_dir, exist_ok=True)

    # 1. 读取和初步处理原始数据 (与 data_loader 类似)
    try:
        df = pd.read_csv(data_path, sep=r'\s+', header=None, comment='%',
                         names=['src', 'dst', 'weight', 'timestamp'])
    except (pd.errors.ParserError, ValueError):
        try:
            df = pd.read_csv(data_path, sep=r'\s+', header=None, comment='%',
                            names=['src', 'dst', 'timestamp'])
            df['weight'] = 1
        except Exception as e: raise ValueError(f"无法解析 {data_path}: {e}")
    except FileNotFoundError: raise FileNotFoundError(f"未找到 {data_path}")

    if min_nodes_remap:
        all_nodes_series = pd.concat([df['src'], df['dst']]).unique()
        node_map = {node_id: i for i, node_id in enumerate(all_nodes_series)}
        df['src'] = df['src'].map(node_map); df['dst'] = df['dst'].map(node_map)
        num_nodes = len(all_nodes_series)
    else:
        num_nodes = int(max(df['src'].max(), df['dst'].max()) + 1) if not df.empty else 0
    if num_nodes == 0: print("警告：无节点。"); return

    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.sort_values(by='datetime')
    min_timestamp = df['datetime'].min(); max_timestamp = df['datetime'].max()
    if pd.isna(min_timestamp): print("警告：无有效时间戳。"); return

    time_window_delta = datetime.timedelta(days=time_window_days)
    snapshot_boundaries = []
    current_boundary_start = min_timestamp
    while current_boundary_start <= max_timestamp:
        snapshot_boundaries.append(current_boundary_start)
        current_boundary_start += time_window_delta
    if not snapshot_boundaries or snapshot_boundaries[-1] <= max_timestamp:
         snapshot_boundaries.append((snapshot_boundaries[-1] if snapshot_boundaries else max_timestamp) + time_window_delta)
    print(f"  共 {max(0, len(snapshot_boundaries) - 1)} 个原始时间窗口。")

    # --- 存储每个原始时间窗口的 X_t, P_t, HM_list_t 和 current_edges ---
    raw_snapshots_info: List[Dict[str, Any]] = []
    initial_feature_dim = -1
    print("步骤 1: 计算每个原始快照的 X_t, P_t, HomophilyBases...")
    for i in tqdm(range(len(snapshot_boundaries) - 1), desc=f"预计算 {dataset_name} 原始快照"):
        start_time_win = snapshot_boundaries[i]; end_time_win = snapshot_boundaries[i+1]
        edges_in_window_df = df[(df['datetime'] >= start_time_win) & (df['datetime'] < end_time_win)]
        edges_np = edges_in_window_df[['src', 'dst']].values.astype(np.int64)
        edge_index_t = torch.from_numpy(edges_np.T) if edges_np.shape[0] > 0 else torch.empty((2,0), dtype=torch.long)
        edge_index_t_undirected = to_undirected(edge_index_t, num_nodes=num_nodes)
        
        p_matrix_t = compute_propagation_matrix(edge_index_t_undirected, num_nodes, dtype=torch.float)
        initial_features_t = feature_generator(edge_index_t_undirected, num_nodes)

        if initial_feature_dim == -1 and initial_features_t.numel() > 0:
            initial_feature_dim = initial_features_t.shape[1]
        current_initial_feature_dim = initial_feature_dim if initial_feature_dim != -1 else 1
        if initial_features_t.numel() == 0 and num_nodes > 0:
            initial_features_t = torch.zeros((num_nodes, current_initial_feature_dim), dtype=torch.float)

        if initial_features_t.numel() > 0 and p_matrix_t.numel() > 0 and p_matrix_t._nnz() > 0:
            homophily_bases_t = compute_homophily_bases(p_matrix_t, initial_features_t, K_unibasis)
        else:
            homophily_bases_t = [initial_features_t.clone() for _ in range(K_unibasis + 1)]

        raw_snapshots_info.append({
            'initial_features': initial_features_t, # X_t
            'p_matrix': p_matrix_t,                 # P_t
            'homophily_bases': homophily_bases_t,   # HM_list_t
            'current_edges': edge_index_t_undirected # E_t (用于生成 t+1 的标签)
        })
    if initial_feature_dim == -1 : initial_feature_dim = 1


    # --- 步骤 2: 为每个 (输入特征组, 目标标签) 对计算并保存所有 UniBasis 视图 ---
    num_saved_samples = 0
    print(f"\n步骤 2: 计算并保存每个样本的所有 UniBasis 视图到 {output_dir}...")
    # 迭代到倒数第二个原始快照，因为最后一个没有“下一个”作为标签
    for t in tqdm(range(len(raw_snapshots_info) - 1), desc=f"保存 {dataset_name} 预处理样本"):
        current_input_info = raw_snapshots_info[t] # 这是 t 时刻的输入信息
        next_snapshot_info = raw_snapshots_info[t+1]   # 这是 t+1 时刻的图，用于提供标签

        # 获取当前输入所需的 X_t, P_t, HM_list_t
        x_t_input = current_input_info['initial_features']
        p_t_input = current_input_info['p_matrix']
        hm_list_t_input = current_input_info['homophily_bases']

        # 为所有视图计算 UniBasis 原始特征
        unibasis_views_dict: Dict[str, torch.Tensor] = {}
        for view_name, h_hat_val in h_hat_views.items():
            # 注意：compute_unibasis_for_snapshot 返回 (拼接后的特征, 单个基维度)
            # 我们只需要拼接后的特征
            unibasis_for_view, _ = compute_unibasis_for_snapshot(
                p_matrix=p_t_input,
                initial_features=x_t_input,
                K=K_unibasis,
                tau=tau_unibasis,
                h_hat=h_hat_val,
                homophily_bases_list=hm_list_t_input
            )
            unibasis_views_dict[f'unibasis_features_{view_name}'] = unibasis_for_view.cpu() # 存到CPU

        # 准备链接预测标签
        pos_edge_label = next_snapshot_info['current_edges'].cpu()
        num_pos = pos_edge_label.size(1)
        num_neg = num_pos if num_pos > 0 else max(100, num_nodes // 10)
        neg_edge_label = negative_sampling(
            edge_index=pos_edge_label, num_nodes=num_nodes,
            num_neg_samples=num_neg, method='sparse'
        ).cpu()

        # 打包当前样本的所有数据
        sample_data_to_save = {
            **unibasis_views_dict, # 包含所有视图的 UniBasis 特征
            'pos_edge_index': pos_edge_label,
            'neg_edge_index': neg_edge_label,
            # (可选) 保存原始 X_t, P_t 如果模型内部还需要
            # 'initial_features': x_t_input.cpu(),
            # 'p_matrix': p_t_input.cpu(), # 保存稀疏P矩阵可能需要特殊处理或转稠密(不推荐)
        }

        # 保存到 .pt 文件
        # 文件名可以包含时间步 t，或者使用一个统一的索引
        sample_filename = f"snapshot_sample_{t:04d}.pt" # 例如 snapshot_sample_0000.pt
        save_file_path = os.path.join(output_dir, sample_filename)
        torch.save(sample_data_to_save, save_file_path)
        num_saved_samples += 1

    print(f"\n预处理完成！共保存 {num_saved_samples} 个样本到 {output_dir}")
    print(f"  节点数量: {num_nodes}")
    print(f"  初始特征维度 F: {initial_feature_dim}")
    # 这个 initial_feature_dim 是 X_t 的维度，也是单个 UniBasis 基的维度

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="预处理动态图数据并保存 UniBasis 视图")
    parser.add_argument('--data_path', type=str, default="data/uci/out.opsahl-ucsocial", help="原始数据文件路径 (例如 enron_edges.tsv)")
    parser.add_argument('--output_dir', type=str,default = "data/uci" , help="保存 .pt 文件的输出目录")
    parser.add_argument('--dataset_name', type=str, default="uci", help="数据集名称 (用于日志)")
    parser.add_argument('--time_window_days', type=int, default=1, help="聚合快照的时间窗口 (天)")
    parser.add_argument('--K', type=int, default=10, help="UniBasis 阶数")
    parser.add_argument('--tau', type=float, default=0.5, help="UniBasis 混合系数 tau")
    # 为五个视图定义 h_hat 值
    parser.add_argument('--h_hat_ch1', type=float, default=0.3)
    parser.add_argument('--h_hat_ch1_prime', type=float, default=0.1)
    parser.add_argument('--h_hat_ch2', type=float, default=0.7)
    parser.add_argument('--h_hat_ch2_prime', type=float, default=0.9)
    parser.add_argument('--h_hat_anchor', type=float, default=0.5)
    parser.add_argument('--no_remap_nodes', action='store_true', help="不执行节点ID重映射 (假设原始ID已合格)")

    cli_args = parser.parse_args()

    h_hat_config = {
        'ch1': cli_args.h_hat_ch1,
        'ch1_prime': cli_args.h_hat_ch1_prime,
        'ch2': cli_args.h_hat_ch2,
        'ch2_prime': cli_args.h_hat_ch2_prime,
        'anchor': cli_args.h_hat_anchor
    }

    preprocess_and_save_snapshots(
        data_path=cli_args.data_path,
        output_dir=cli_args.output_dir,
        dataset_name=cli_args.dataset_name,
        time_window_days=cli_args.time_window_days,
        K_unibasis=cli_args.K,
        tau_unibasis=cli_args.tau,
        h_hat_views=h_hat_config,
        min_nodes_remap=not cli_args.no_remap_nodes
    )