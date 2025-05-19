# models/dynamic_freq_gnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any


# 导入项目模块
from .base_model import BaseDynamicBackbone, BaseTaskHead # 导入基类
from .dyn_spectral import DynSpectralBackbone    # 导入我们实现的 Backbone
from .task_head import LinkPredictorHead                # 导入链接预测头
from .sequence_encoder import LSTMWrapper # DynSpectralBackbone 会用到

class DynSpectral(nn.Module):
    """
    顶层动态谱图模型 (DynSpectral)。
    适配双通道 DynSpectralBackbone。
    """
    def __init__(self,
                 device: torch.device,
                 # --- Backbone (DynSpectralBackbone - 双通道) 参数 ---
                 unibasis_base_feature_dim: int,
                 K: int,
                 combination_dropout_ch1: float, # 通道1 Combination dropout
                 combination_dropout_ch2: float, # 通道2 Combination dropout
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
            unibasis_base_feature_dim=unibasis_base_feature_dim,
            K=K,
            combination_dropout_ch1=combination_dropout_ch1, # 传递 ch1 dropout
            combination_dropout_ch2=combination_dropout_ch2, # 传递 ch2 dropout
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_layers=lstm_layers,
            lstm_dropout=lstm_dropout
        ).to(device)

        # 2. 实例化任务头 (Task Head)
        # LinkPredictorHead 的输入维度现在是 Backbone 输出的拼接维度 (lstm_hidden_dim * 2)
        task_head_input_dim = lstm_hidden_dim * 2
        self.task_head: BaseTaskHead = LinkPredictorHead(
            node_embedding_dim=task_head_input_dim, # 使用拼接后的维度
            hidden_dim=link_pred_hidden_dim
        ).to(device)

        self.reset_parameters() # 可选

    def forward(self,
                snapshots_data: List[Dict[str, torch.Tensor]],
                target_edges: Optional[torch.Tensor] = None,
                **kwargs: Any
                ) -> torch.Tensor:

        node_representations = self.backbone(snapshots_data, **kwargs)

        if isinstance(self.task_head, LinkPredictorHead):
            if target_edges is None:
                raise ValueError("LinkPredictorHead 需要 target_edges 参数。")
            logits = self.task_head(node_representations, target_edges, **kwargs)
        else:
            logits = self.task_head(node_representations, target_edges=target_edges, **kwargs)

        return logits

    def reset_parameters(self):
        print("DynSpectral (Wrapper - 双通道): 重置参数...")
        if hasattr(self.backbone, 'reset_parameters'):
            self.backbone.reset_parameters()
        if hasattr(self.task_head, 'reset_parameters'):
            self.task_head.reset_parameters()