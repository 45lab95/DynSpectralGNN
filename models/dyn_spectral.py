import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Any

# 导入基类和我们需要的模块
from .base_model import BaseDynamicBackbone
from .filters import Combination # 从我们修改后的 model.py 导入 Combination
from .sequence_encoder import LSTMWrapper
class DynSpectralBackbone(BaseDynamicBackbone):
    """
    动态谱图主干网络 (修改为支持双通道 UniBasis 和 LSTM)。
    """
    def __init__(self,
                 device: torch.device,
                 unibasis_base_feature_dim: int,
                 K: int,
                 # --- 修改为双通道参数 ---
                 combination_dropout_ch1: float, # 通道1 Combination dropout
                 combination_dropout_ch2: float, # 通道2 Combination dropout
                 # --- LSTM 参数 (假设两个 LSTM 结构相同) ---
                 lstm_hidden_dim: int,
                 lstm_layers: int = 1,
                 lstm_dropout: float = 0.0):
        super().__init__(device)

        self.unibasis_base_feature_dim = unibasis_base_feature_dim
        self.K = K
        self.lstm_hidden_dim = lstm_hidden_dim # 单个 LSTM 的隐藏维度

        # 1. 实例化两个 Combination 层
        self.combination1 = Combination(
            channels=unibasis_base_feature_dim,
            level=K + 1,
            dropout=combination_dropout_ch1 # 使用通道1的 dropout
        ).to(device)

        self.combination2 = Combination(
            channels=unibasis_base_feature_dim,
            level=K + 1,
            dropout=combination_dropout_ch2 # 使用通道2的 dropout
        ).to(device)

        # 2. 实例化两个 LSTM 包装器
        lstm_input_dim_single_channel = unibasis_base_feature_dim # 每个 LSTM 的输入是 F

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

        self.reset_parameters() # 调用初始化

    def forward(self,
                snapshots_data: List[Dict[str, torch.Tensor]],
                **kwargs: Any
                ) -> torch.Tensor:
        num_snapshots = len(snapshots_data)
        if num_snapshots == 0: raise ValueError("输入快照列表不能为空")

        num_nodes = snapshots_data[0]['unibasis_features_ch1'].shape[0]
        # 维度检查和 current_F 的获取 (与之前类似，但现在针对任一通道即可)
        total_unibasis_dim_ch1 = snapshots_data[0]['unibasis_features_ch1'].shape[1]
        current_F = self.unibasis_base_feature_dim
        if total_unibasis_dim_ch1 % (self.K + 1) != 0:
            raise ValueError(f"CH1 UniBasis 特征维度 ({total_unibasis_dim_ch1}) "
                             f"无法被 K+1 ({self.K+1}) 整除。")
        calculated_F_ch1 = total_unibasis_dim_ch1 // (self.K + 1)
        if calculated_F_ch1 != current_F:
            print(f"警告 (DynSpectralBackbone CH1): 计算得到的单个基维度 F ({calculated_F_ch1}) "
                  f"与模型初始化时的维度 ({current_F}) 不符。将使用初始化维度。")
        lstm_inputs_ch1_seq = []
        lstm_inputs_ch2_seq = []

        # --- 迭代时间步，并行处理双通道 ---
        for t in range(num_snapshots):
            unibasis_features_t_ch1 = snapshots_data[t]['unibasis_features_ch1'].to(self.device)
            unibasis_features_t_ch2 = snapshots_data[t]['unibasis_features_ch2'].to(self.device)

            # --- 通道 1 ---
            try:
                unibasis_t_reshaped_ch1 = unibasis_features_t_ch1.view(
                    num_nodes, self.K + 1, current_F
                )
            except RuntimeError as e: # ... (错误处理) ...
                print(f"DynSpectralBackbone CH1: 时间步 {t} reshape UniBasis 错误: {e}"); raise e
            combined_repr_t_ch1 = self.combination1(unibasis_t_reshaped_ch1) # [N, F]
            lstm_inputs_ch1_seq.append(combined_repr_t_ch1)

            # --- 通道 2 ---
            try:
                unibasis_t_reshaped_ch2 = unibasis_features_t_ch2.view(
                    num_nodes, self.K + 1, current_F
                )
            except RuntimeError as e: # ... (错误处理) ...
                print(f"DynSpectralBackbone CH2: 时间步 {t} reshape UniBasis 错误: {e}"); raise e
            combined_repr_t_ch2 = self.combination2(unibasis_t_reshaped_ch2) # [N, F]
            lstm_inputs_ch2_seq.append(combined_repr_t_ch2)

        # --- 准备 LSTM 输入并进行时序编码 ---
        if not lstm_inputs_ch1_seq or not lstm_inputs_ch2_seq:
             raise ValueError("LSTM 输入序列为空，无法继续。")

        # 通道 1 LSTM
        lstm_input_tensor_ch1_seq_first = torch.stack(lstm_inputs_ch1_seq, dim=0)
        lstm_input_tensor_ch1_batch_first = lstm_input_tensor_ch1_seq_first.permute(1, 0, 2)
        lstm_output_sequence1, _ = self.lstm_encoder1(lstm_input_tensor_ch1_batch_first)
        final_node_repr1 = lstm_output_sequence1[:, -1, :] # [N, lstm_hidden_dim]

        # 通道 2 LSTM
        lstm_input_tensor_ch2_seq_first = torch.stack(lstm_inputs_ch2_seq, dim=0)
        lstm_input_tensor_ch2_batch_first = lstm_input_tensor_ch2_seq_first.permute(1, 0, 2)
        lstm_output_sequence2, _ = self.lstm_encoder2(lstm_input_tensor_ch2_batch_first)
        final_node_repr2 = lstm_output_sequence2[:, -1, :] # [N, lstm_hidden_dim]

        # --- 拼接两个通道的最终节点表示 ---
        final_combined_node_repr = torch.cat([final_node_repr1, final_node_repr2], dim=-1)
        # 输出形状: [N, lstm_hidden_dim * 2]

        return final_combined_node_repr

    def reset_parameters(self):
        print("DynSpectralBackbone: 重置参数 (双通道)...")
        if hasattr(self.combination1, 'reset_parameters'):
            self.combination1.reset_parameters()
        if hasattr(self.combination2, 'reset_parameters'):
            self.combination2.reset_parameters()
        # LSTM 默认有初始化，LSTMWrapper 未显式重写
        # 如果需要特定初始化，可以在这里为 self.lstm_encoder1.lstm 和 self.lstm_encoder2.lstm 初始化