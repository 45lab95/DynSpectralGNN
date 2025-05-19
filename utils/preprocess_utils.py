import torch
from torch_geometric.utils import add_self_loops, degree
from torch_sparse import coalesce 

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