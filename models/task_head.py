# models/task_heads.py

import torch
import torch.nn as nn
from typing import Optional

# 导入任务头的基类
from .base_model import BaseTaskHead

class LinkPredictorHead(BaseTaskHead): # 继承自 BaseTaskHead
    """
    用于链接预测的任务头。
    接收一对节点的表示，预测它们之间存在链接的 logit。
    """
    def __init__(self,
                 node_embedding_dim: int, # 输入来自 Backbone 的单个节点嵌入维度
                 hidden_dim: Optional[int] = None,
                 output_dim: int = 1): # 通常链接预测输出一个 logit
        super().__init__() # 调用父类初始化

        self.node_embedding_dim = node_embedding_dim
        self.output_dim = output_dim

        # 输入到 MLP 的维度是两个节点嵌入拼接后的维度
        mlp_input_dim = node_embedding_dim * 2

        if hidden_dim:
            self.mlp = nn.Sequential(
                nn.Linear(mlp_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        else:
            self.mlp = nn.Linear(mlp_input_dim, output_dim)


    def forward(self,
                node_representations: torch.Tensor, # [num_all_nodes, node_embedding_dim]
                target_edges: torch.Tensor # [2, num_predict_edges]
                ) -> torch.Tensor:
        current_device = node_representations.device
        target_edges_on_device = target_edges.to(current_device)
        src_node_indices = target_edges[0]
        dst_node_indices = target_edges[1]

        src_repr = node_representations[src_node_indices] # [num_predict_edges, node_embedding_dim]
        dst_repr = node_representations[dst_node_indices] # [num_predict_edges, node_embedding_dim]

        combined_edge_repr = torch.cat([src_repr, dst_repr], dim=-1) # [num_predict_edges, node_embedding_dim * 2]
        logits = self.mlp(combined_edge_repr) # [num_predict_edges, output_dim]

        return logits.squeeze(-1) if self.output_dim == 1 else logits