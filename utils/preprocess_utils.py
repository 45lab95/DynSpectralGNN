import torch
from torch_geometric.utils import add_self_loops, degree
from torch_sparse import coalesce 
import torch.nn.functional as F
import math
import gc
from typing import Tuple, List, Optional

def compute_propagation_matrix(edge_index: torch.Tensor,
                               num_nodes: int,
                               add_self_loops_flag: bool = True,
                               dtype: torch.dtype = torch.float) -> torch.Tensor:

    device = edge_index.device 
    if add_self_loops_flag:
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    row, col = edge_index[0], edge_index[1]
    deg = degree(col, num_nodes=num_nodes, dtype=dtype) 
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    p_matrix = torch.sparse_coo_tensor(edge_index, edge_weight,
                                        torch.Size([num_nodes, num_nodes]),
                                        dtype=dtype,
                                        device=device)

    return p_matrix

def calculate_homophily(edge_index: torch.Tensor,
                        labels: torch.Tensor,
                        num_nodes: int) -> float:
    """
    计算给定图快照的边同配率 (edge homophily ratio)。
    同配率 = (连接相同标签节点的边数) / (总边数)
    """
    num_edges = edge_index.size(1)
    if num_edges == 0:
        return 0.0 
    row, col = edge_index[0], edge_index[1]
    labels_row = labels[row]
    labels_col = labels[col]

    same_label_edges = (labels_row == labels_col).sum().item()

    homophily_ratio = same_label_edges / num_edges

    return homophily_ratio

def compute_homophily_bases(p_matrix: torch.Tensor,
                            features: torch.Tensor,
                            K: int) -> List[torch.Tensor]:
    """
    计算同配基序列 [X, PX, P^2X, ..., P^K X]。
    """
    # 确保所有操作在同一设备
    device = features.device
    p_matrix = p_matrix.to(device) # 确保 p_matrix 也在正确设备

    homophily_bases_list = [features.clone()] # HM_0 = X_t, 使用 clone 避免修改原始 features
    hm_k = features.clone()
    for _ in range(K): # 迭代 K 次，得到 P^1X 到 P^KX
        hm_k = torch.spmm(p_matrix, hm_k)
        homophily_bases_list.append(hm_k.clone()) # 存储 hm_k 的副本
    return homophily_bases_list


def compute_unibasis_for_snapshot(
    p_matrix: torch.Tensor,           # P_t
    initial_features: torch.Tensor, # 原始特征 X_t
    K: int,
    tau: float,
    h_hat: float,
    homophily_bases_list: Optional[List[torch.Tensor]] = None # 可选的预计算同配基
) -> Tuple[torch.Tensor, int]:
    """
    为单个时间步计算 UniBasis 基向量。
    如果提供了 homophily_bases_list，则不再内部计算同配基。
    核心异配基和组合逻辑与 UniFilter 原始实现保持一致。
    """
    num_nodes, base_feature_dim = initial_features.shape
    device = initial_features.device
    p_matrix = p_matrix.to(device) # 确保 p_matrix 在正确设备

    cosval = math.cos(math.pi * (1.0 - h_hat) / 2.0)

    # --- 获取或计算同配基 ---
    if homophily_bases_list is None:
        hm_bases = compute_homophily_bases(p_matrix, initial_features, K)
    else:
        if len(homophily_bases_list) != K + 1:
            raise ValueError(f"提供的同配基列表长度 ({len(homophily_bases_list)}) "
                             f"与 K+1 ({K+1}) 不匹配。")
        hm_bases = [b.to(device) for b in homophily_bases_list] # 确保设备一致

    # --- 初始化异配基计算所需变量 (与你提供的代码一致) ---
    norm_feat = torch.norm(initial_features, dim=0, keepdim=True)
    norm_feat = torch.clamp(norm_feat, min=1e-8) # 使用你代码中的 1e-8
    u_0 = initial_features / norm_feat

    v_last = u_0.clone() # 使用 clone
    v_second_last = torch.zeros_like(v_last, device=device)
    basis_sum = u_0.clone() # S_0 = u_0, 使用 clone

    # --- 存储最终的 UniBasis 基向量 B_k ---
    # B_0 = τ*HM_0 + (1-τ)*u_0  (HM_0 是 hm_bases[0]，即 initial_features)
    unibasis_list = [(hm_bases[0].clone() * tau + u_0.clone() * (1.0 - tau))] # 使用 clone

    # --- 迭代计算 k=1 到 K (与你提供的代码一致) ---
    for k_iter_idx in range(1, K + 1): # k_iter_idx 是列表索引, 对应多项式阶数 k
        # --- 计算正交基 v_k ---
        v_k_temp = torch.spmm(p_matrix, v_last)
        project_1 = torch.einsum('nd,nd->d', v_k_temp, v_last)
        project_2 = torch.einsum('nd,nd->d', v_k_temp, v_second_last)
        v_k_orth = v_k_temp - project_1.unsqueeze(0) * v_last - project_2.unsqueeze(0) * v_second_last # 确保广播正确
        norm_vk = torch.norm(v_k_orth, dim=0, keepdim=True)
        norm_vk = torch.clamp(norm_vk, min=1e-8)
        v_k = v_k_orth / norm_vk

        # --- 获取当前同配基 HM_k ---
        hm_k_current = hm_bases[k_iter_idx] # hm_bases[0] 是 X, hm_bases[1] 是 PX, ..., hm_bases[K] 是 P^K X

        # --- 计算异配基 u_k ---
        H_k_approx = basis_sum / k_iter_idx # S_{k-1} / k (k_iter_idx 从1开始)
        last_unibasis_for_tk = unibasis_list[-1] # B_{k-1}

        term1_numerator = torch.einsum('nd,nd->d', H_k_approx, last_unibasis_for_tk)
        # 检查 cosval 是否过小以避免除零或数值不稳定
        term1_sq = torch.square(term1_numerator / cosval) if abs(cosval) > 1e-9 else torch.zeros_like(term1_numerator)

        term2 = ((k_iter_idx - 1) * cosval + 1) / k_iter_idx
        Tf_sq = torch.clamp(term1_sq - term2, min=0.0)
        Tf = torch.sqrt(Tf_sq)

        u_k_unnormalized = H_k_approx + torch.mul(Tf.unsqueeze(0), v_k) # 确保广播正确
        norm_uk = torch.norm(u_k_unnormalized, dim=0, keepdim=True)
        norm_uk = torch.clamp(norm_uk, min=1e-8)
        u_k = u_k_unnormalized / norm_uk

        # --- 组合 UniBasis B_k ---
        norm_hmk_current = torch.norm(hm_k_current, dim=0, keepdim=True)
        norm_hmk_current = torch.clamp(norm_hmk_current, min=1e-8)
        hm_k_normalized = hm_k_current / norm_hmk_current

        b_k = hm_k_normalized * tau + u_k * (1.0 - tau)
        unibasis_list.append(b_k)

        # --- 更新状态变量 ---
        basis_sum += u_k
        v_second_last = v_last.clone() # 更新 v_{k-2}
        v_last = v_k.clone()       # 更新 v_{k-1}

    # 清理操作可以保留
    del v_last, v_second_last, basis_sum, u_0, norm_feat # hm_k_current 是局部变量
    if homophily_bases_list is None:
        del hm_bases
    gc.collect()

    unibasis_features_concatenated = torch.cat(unibasis_list, dim=1)

    return unibasis_features_concatenated, base_feature_dim