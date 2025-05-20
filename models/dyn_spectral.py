import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Any
from .base_model import BaseDynamicBackbone
from .sequence_encoder import LSTMWrapper
from .contrastive import DualChannelContrastiveHead
from utils.preprocess_utils import compute_homophily_bases

class DynSpectralBackbone(BaseDynamicBackbone):
    """
    动态谱图主干网络 (集成双通道对比学习头)。
    """
    def __init__(self,
                 device: torch.device,
                 initial_feature_dim: int, 
                 K: int,                   
                 tau: float,                 
                 h_hat_ch1: float, h_hat_ch1_prime: float,
                 h_hat_ch2: float, h_hat_ch2_prime: float,
                 h_hat_anchor: float,
                 combination_dropout_ch1: float,
                 combination_dropout_ch2: float,
                 combination_dropout_anchor: float,
                 lstm_hidden_dim: int, 
                 contrastive_temperature: float = 0.1,
                 
                 lstm_layers: int = 1,
                 lstm_dropout: float = 0.0,
                 contrastive_loss_interval: int = 1 
                 ):
        super().__init__(device)

        self.initial_feature_dim = initial_feature_dim # X_t 的维度 F
        self.K = K
        self.lstm_hidden_dim = lstm_hidden_dim
        self.contrastive_loss_interval = max(1, contrastive_loss_interval) # 至少为1

        self.contrastive_head = DualChannelContrastiveHead(
            base_feature_dim=initial_feature_dim, K=K, tau=tau,
            h_hat_ch1=h_hat_ch1, h_hat_ch1_prime=h_hat_ch1_prime,
            h_hat_ch2=h_hat_ch2, h_hat_ch2_prime=h_hat_ch2_prime,
            h_hat_anchor=h_hat_anchor,
            combination_dropout_ch1=combination_dropout_ch1,
            combination_dropout_ch2=combination_dropout_ch2,
            combination_dropout_anchor=combination_dropout_anchor,
            temperature=contrastive_temperature
        ).to(device)

        lstm_input_dim_single_channel = initial_feature_dim 
        self.lstm_encoder1 = LSTMWrapper(
            input_size=lstm_input_dim_single_channel,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout
        ).to(device)

        self.lstm_encoder2 = LSTMWrapper(
            input_size=lstm_input_dim_single_channel,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout
        ).to(device)

        self.reset_parameters()

    def forward(self,
                snapshots_data: List[Dict[str, torch.Tensor]], # 现在包含 'features' 和 'p_matrix'
                **kwargs: Any
                ) -> Tuple[torch.Tensor, torch.Tensor]: # 返回 (最终节点表示, 总对比损失)
        """
        Args:
            snapshots_data (List[Dict[str, torch.Tensor]]):
                包含 T 个时间步图快照数据的列表。每个字典需要包含:
                {'features': 初始特征 X_t [N, F_initial],
                 'p_matrix': 传播矩阵 P_t [N, N] (sparse)}
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - final_combined_node_repr (torch.Tensor): [N, lstm_hidden_dim * 2]
                - total_contrastive_loss (torch.Tensor): 标量
        """
        num_snapshots = len(snapshots_data)
        if num_snapshots == 0: raise ValueError("输入快照列表不能为空")

        lstm_inputs_ch1_seq = []
        lstm_inputs_ch2_seq = []
        accumulated_contrastive_loss = torch.tensor(0.0, device=self.device)
        num_contrastive_calcs = 0
        
        # --- 迭代时间步 ---
        for t in range(num_snapshots):

            # print(f"  Backbone t={t}: input snapshot device - initial_features: {snapshots_data[t]['initial_features'].device}, p_matrix: {snapshots_data[t]['p_matrix'].device}, homophily_bases[0]: {snapshots_data[t]['homophily_bases'][0].device if snapshots_data[t]['homophily_bases'] else 'N/A'}")


            initial_features_t = snapshots_data[t]['initial_features'].to(self.device)
            p_matrix_t = snapshots_data[t]['p_matrix'].to(self.device)
            precomputed_homophily_bases_t = [                                      
                b.to(self.device) for b in snapshots_data[t]['homophily_bases']
            ]
            z1_t, z2_t, contrastive_loss_t = self.contrastive_head(
                precomputed_homophily_bases=precomputed_homophily_bases_t,
                p_matrix=p_matrix_t, 
                initial_features=initial_features_t
            )
            
            # print(f"  Backbone t={t}: after .to(device) - initial_features: {initial_features_t.device}, p_matrix: {p_matrix_t.device}, homophily_bases[0]: {precomputed_homophily_bases_t[0].device if precomputed_homophily_bases_t else 'N/A'}")


            lstm_inputs_ch1_seq.append(z1_t)
            lstm_inputs_ch2_seq.append(z2_t)

            if (t + 1) % self.contrastive_loss_interval == 0:
                accumulated_contrastive_loss += contrastive_loss_t
                num_contrastive_calcs += 1
        if num_contrastive_calcs > 0:
            total_contrastive_loss_for_sequence = accumulated_contrastive_loss / num_contrastive_calcs
        else:
            total_contrastive_loss_for_sequence = torch.tensor(0.0, device=self.device)


        if not lstm_inputs_ch1_seq:
             raise ValueError("LSTM 输入序列为空，无法继续。")

        # 通道 1 LSTM
        lstm_input_ch1 = torch.stack(lstm_inputs_ch1_seq, dim=0).permute(1, 0, 2) # [N, T, F_initial]
        lstm_output_seq1, _ = self.lstm_encoder1(lstm_input_ch1)
        final_node_repr1 = lstm_output_seq1[:, -1, :] # [N, lstm_hidden_dim]

        # 通道 2 LSTM
        lstm_input_ch2 = torch.stack(lstm_inputs_ch2_seq, dim=0).permute(1, 0, 2) # [N, T, F_initial]
        lstm_output_seq2, _ = self.lstm_encoder2(lstm_input_ch2)
        final_node_repr2 = lstm_output_seq2[:, -1, :] # [N, lstm_hidden_dim]

        # --- 拼接结果 ---
        final_combined_node_repr = torch.cat([final_node_repr1, final_node_repr2], dim=-1)

        return final_combined_node_repr, total_contrastive_loss_for_sequence

    def reset_parameters(self):
        print("DynSpectralBackbone (w/ Contrastive Head): 重置参数...")
        if hasattr(self.contrastive_head, 'reset_parameters'): 
            self.contrastive_head.reset_parameters()
        for lstm_module in [self.lstm_encoder1, self.lstm_encoder2]:
            if hasattr(lstm_module, 'reset_parameters'):
                lstm_module.reset_parameters()