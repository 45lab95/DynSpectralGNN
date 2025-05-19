# models/dynamic_freq_gnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import math


# 导入项目模块
from .base_model import BaseDynamicBackbone, BaseTaskHead
from .dyn_spectral import DynSpectralBackbone # 已更新为集成对比学习头
from .task_head import LinkPredictorHead
# from .filters import Combination # DynSpectral 顶层不再直接需要 Combination

class DynSpectral(nn.Module):
    """
    顶层动态谱图模型 (DynSpectral)。
    适配集成了双通道对比学习的 DynSpectralBackbone。
    """
    def __init__(self,
                 device: torch.device,
                 # --- Backbone (DynSpectralBackbone) 参数 ---
                 initial_feature_dim: int, # 初始节点特征 X_t 的维度
                 K: int,
                 tau: float,
                 # --- 对比学习头特定参数 (传递给 Backbone) ---
                 h_hat_ch1: float, h_hat_ch1_prime: float,
                 h_hat_ch2: float, h_hat_ch2_prime: float,
                 h_hat_anchor: float,
                 combination_dropout_ch1: float,
                 combination_dropout_ch2: float,
                 combination_dropout_anchor: float,
                 contrastive_temperature: float = 0.1,
                 contrastive_loss_interval: int = 1, # 新增
                 # --- 主 LSTM 通道参数 (传递给 Backbone) ---
                 lstm_hidden_dim: int, # 单个 LSTM 的隐藏维度
                 lstm_layers: int = 1,
                 lstm_dropout: float = 0.0,
                 # --- Task Head (LinkPredictorHead) 参数 ---
                 link_pred_hidden_dim: Optional[int] = None
                 ):
        super().__init__()
        self.device = device

        # 1. 实例化主干网络 (Backbone)
        self.backbone: BaseDynamicBackbone = DynSpectralBackbone(
            device=device,
            initial_feature_dim=initial_feature_dim, # 传递初始特征维度
            K=K,
            tau=tau,
            h_hat_ch1=h_hat_ch1, h_hat_ch1_prime=h_hat_ch1_prime,
            h_hat_ch2=h_hat_ch2, h_hat_ch2_prime=h_hat_ch2_prime,
            h_hat_anchor=h_hat_anchor,
            combination_dropout_ch1=combination_dropout_ch1,
            combination_dropout_ch2=combination_dropout_ch2,
            combination_dropout_anchor=combination_dropout_anchor,
            contrastive_temperature=contrastive_temperature,
            contrastive_loss_interval=contrastive_loss_interval, # 传递
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_layers=lstm_layers,
            lstm_dropout=lstm_dropout
        ).to(device)

        # 2. 实例化任务头 (Task Head)
        # LinkPredictorHead 的输入维度是 Backbone 输出的拼接维度 (lstm_hidden_dim * 2)
        task_head_input_dim = lstm_hidden_dim * 2
        self.task_head: BaseTaskHead = LinkPredictorHead(
            node_embedding_dim=task_head_input_dim,
            hidden_dim=link_pred_hidden_dim
        ).to(device)

        self.reset_parameters() # 可选

    def forward(self,
                snapshots_data: List[Dict[str, torch.Tensor]], # 现在包含 'features', 'p_matrix'
                target_edges: Optional[torch.Tensor] = None,
                **kwargs: Any
                ) -> Tuple[torch.Tensor, torch.Tensor]: # 返回 (任务 logits, 对比损失)
        """
        模型的前向传播。

        Args:
            snapshots_data (List[Dict[str, torch.Tensor]]):
                动态图快照数据列表，包含 'features' (X_t) 和 'p_matrix' (P_t)。
            target_edges (Optional[torch.Tensor]): 链接预测的目标边。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - task_logits (torch.Tensor): 主任务的预测 logits。
                - contrastive_loss (torch.Tensor): 从 Backbone 计算得到的对比损失。
        """

        node_representations, contrastive_loss_from_backbone = self.backbone(
            snapshots_data, **kwargs
        )

        if isinstance(self.task_head, LinkPredictorHead):
            if target_edges is None:
                raise ValueError("LinkPredictorHead 需要 target_edges 参数。")
            task_logits = self.task_head(node_representations, target_edges, **kwargs)
        else:
            task_logits = self.task_head(node_representations, target_edges=target_edges, **kwargs)

        return task_logits, contrastive_loss_from_backbone 
        
    def reset_parameters(self):
        print("DynSpectral (Wrapper - w/ CL): 重置参数...")
        if hasattr(self.backbone, 'reset_parameters'):
            self.backbone.reset_parameters()
        if hasattr(self.task_head, 'reset_parameters'):
            self.task_head.reset_parameters()