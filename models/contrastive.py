# models/contrastive_learning.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

# 导入我们之前放在 filters.py (或 model.py) 的 Combination 类
# 假设 Combination 类在 filters.py 中
from .filters import Combination
# 导入 UniBasis 计算工具
# 假设 utils/preprocess_utils.py 包含 compute_unibasis_for_snapshot 和 compute_homophily_bases
from utils.preprocess_utils import compute_unibasis_for_snapshot, compute_homophily_bases


class UniBasisViewGenerator(nn.Module):
    """
    使用特定的 h_hat 和 Combination 层生成一个 UniBasis 视图的表示。
    """
    def __init__(self,
                 base_feature_dim: int, 
                 K: int,
                 combination_dropout: float,
                 ):
        super().__init__()
        self.base_feature_dim = base_feature_dim
        self.K = K
        # 每个 ViewGenerator 拥有自己的 Combination 层
        self.combination_layer = Combination(
            channels=base_feature_dim,
            level=K + 1,
            dropout=combination_dropout
        )

    def forward(self,
                p_matrix: torch.Tensor,
                initial_features: torch.Tensor,
                h_hat_for_view: float, # 当前视图使用的 h_hat
                tau_for_view: float,   # 当前视图使用的 tau
                precomputed_homophily_bases: List[torch.Tensor] # 预计算的同配基
               ) -> torch.Tensor:
        """
        Args:
            p_matrix: 传播矩阵 P_t
            initial_features: 初始特征 X_t
            h_hat_for_view: 此视图的 h_hat
            tau_for_view: 此视图的 tau
            precomputed_homophily_bases: 共享的同配基列表 [X, PX, ..., P^K X]

        Returns:
            torch.Tensor: 组合后的视图表示 [N, F]
        """
        num_nodes = initial_features.shape[0]

        # 1. 计算此视图的 UniBasis (使用预计算的同配基)
        unibasis_raw_features, _ = compute_unibasis_for_snapshot(
            p_matrix=p_matrix,
            initial_features=initial_features,
            K=self.K,
            tau=tau_for_view,
            h_hat=h_hat_for_view,
            homophily_bases_list=precomputed_homophily_bases
        ) # shape: [N, (K+1)*F]

        # 2. Reshape 为 Combination 层期望的格式
        try:
            unibasis_reshaped = unibasis_raw_features.view(
                num_nodes, self.K + 1, self.base_feature_dim
            )
        except RuntimeError as e:
            print(f"ViewGenerator: Reshape UniBasis 错误 for h_hat={h_hat_for_view}: {e}")
            print(f"  原始形状: {unibasis_raw_features.shape}")
            print(f"  目标形状: ({num_nodes}, {self.K + 1}, {self.base_feature_dim})")
            raise e

        view_representation = self.combination_layer(unibasis_reshaped) # [N, F]

        return view_representation


class InfoNCEContrastiveLossModule(nn.Module):

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss() # InfoNCE 可以用交叉熵实现

    def forward(self,
                query_repr: torch.Tensor,           # [N, F] 或 [Batch, F]
                positive_key_repr: torch.Tensor,    # [N, F] 或 [Batch, F]
                negative_keys_repr_list: List[torch.Tensor] # List of [N, F] or [Batch, F]
               ) -> torch.Tensor:
        """
        Args:
            query_repr: 查询表示。
            positive_key_repr: 正键表示。
            negative_keys_repr_list: 包含一个或多个负键表示的列表。
            temperature: 温度参数。

        Returns:
            torch.Tensor: InfoNCE 损失值 (标量)。
        """

        device = query_repr.device
        positive_key_repr = positive_key_repr.to(device)
        negative_keys_repr_list = [neg_key.to(device) for neg_key in negative_keys_repr_list]


        l_pos = torch.einsum('nc,nc->n', query_repr, positive_key_repr).unsqueeze(-1)

        if negative_keys_repr_list:
            l_neg_list = []
            for neg_key in negative_keys_repr_list:
                l_neg_list.append(torch.einsum('nc,nc->n', query_repr, neg_key).unsqueeze(-1))
            l_neg = torch.cat(l_neg_list, dim=1) 
            logits = torch.cat([l_pos, l_neg], dim=1)
        else:
            logits = l_pos


        # 除以温度
        logits = logits / self.temperature

        # 目标标签：对于每个 query，第一个 logit (与 positive_key) 对应的类别是 0
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=device)

        # 计算交叉熵损失
        loss = self.criterion(logits, labels)
        return loss


class DualChannelContrastiveHead(nn.Module):

    def __init__(self,
                 base_feature_dim: int, K: int, tau: float, # 通用 UniBasis 参数
                 h_hat_ch1: float, h_hat_ch1_prime: float,
                 h_hat_ch2: float, h_hat_ch2_prime: float,
                 h_hat_anchor: float,
                 combination_dropout_ch1: float, # 假设 ch1 和 ch1_prime 用这个
                 combination_dropout_ch2: float, # 假设 ch2 和 ch2_prime 用这个
                 combination_dropout_anchor: float,
                 temperature: float = 0.1):
        super().__init__()
        self.base_feature_dim = base_feature_dim
        self.K = K
        self.tau = tau # 通用的 tau

        # 实例化五个 UniBasisViewGenerator
        self.view_gen_ch1 = UniBasisViewGenerator(base_feature_dim, K, combination_dropout_ch1)
        self.view_gen_ch1_prime = UniBasisViewGenerator(base_feature_dim, K, combination_dropout_ch1) # prime 可以共享 dropout
        self.view_gen_ch2 = UniBasisViewGenerator(base_feature_dim, K, combination_dropout_ch2)
        self.view_gen_ch2_prime = UniBasisViewGenerator(base_feature_dim, K, combination_dropout_ch2)
        self.view_gen_anchor = UniBasisViewGenerator(base_feature_dim, K, combination_dropout_anchor)

        # 保存 h_hat 值
        self.h_hats = {
            'ch1': h_hat_ch1, 'ch1_prime': h_hat_ch1_prime,
            'ch2': h_hat_ch2, 'ch2_prime': h_hat_ch2_prime,
            'anchor': h_hat_anchor
        }

        # 实例化 InfoNCE 损失模块
        self.infonce_loss = InfoNCEContrastiveLossModule(temperature=temperature)

    def forward(self,
                precomputed_homophily_bases: List[torch.Tensor],
                p_matrix: torch.Tensor,
                initial_features: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # 生成五个视图的表示
        z1 = self.view_gen_ch1(p_matrix, initial_features, self.h_hats['ch1'], self.tau, precomputed_homophily_bases)
        z1_prime = self.view_gen_ch1_prime(p_matrix, initial_features, self.h_hats['ch1_prime'], self.tau, precomputed_homophily_bases)
        z2 = self.view_gen_ch2(p_matrix, initial_features, self.h_hats['ch2'], self.tau, precomputed_homophily_bases)
        z2_prime = self.view_gen_ch2_prime(p_matrix, initial_features, self.h_hats['ch2_prime'], self.tau, precomputed_homophily_bases)
        za = self.view_gen_anchor(p_matrix, initial_features, self.h_hats['anchor'], self.tau, precomputed_homophily_bases)

        # 计算对比损失
        # loss1: z1 作为 query, z1_prime 是 positive, [z2, z2_prime, za] 是 negatives
        loss1 = self.infonce_loss(query_repr=z1,
                                  positive_key_repr=z1_prime,
                                  negative_keys_repr_list=[z2, z2_prime, za])

        # loss2: z2 作为 query, z2_prime 是 positive, [z1, z1_prime, za] 是 negatives
        loss2 = self.infonce_loss(query_repr=z2,
                                  positive_key_repr=z2_prime,
                                  negative_keys_repr_list=[z1, z1_prime, za])

        total_contrastive_loss = loss1 + loss2

        # 返回用于主 LSTM 通道的表示 z1, z2 和对比损失
        return z1, z2, total_contrastive_loss

