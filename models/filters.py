# models/filters.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class BaseFilterLayer(nn.Module):
    """滤波器层的基类 (可选，用于定义通用接口和初始化)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        """使用 Kaiming Uniform 初始化权重"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))


    def forward(self, x: torch.Tensor, filter_matrix: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement the forward method.")


class LowPassFilterLayer(BaseFilterLayer):
    """
    简单的低通滤波器层 (GCN-like)(弃用)
    执行操作: H' = σ(P * X * W)
    其中 P 是传播矩阵 (如对称归一化邻接矩阵 A_sym)。
    """
    def __init__(self, in_channels, out_channels, activation: bool = True):
        super().__init__(in_channels, out_channels)
        self.activation = activation

    def forward(self, x: torch.Tensor, p_matrix: torch.Tensor) -> torch.Tensor:

        # 1. 邻居聚合 (应用传播矩阵): P * X
        support = torch.spmm(p_matrix, x)

        # 2. 特征变换: (P * X) * W
        output = torch.mm(support, self.weight)

        if self.activation:
            output = F.relu(output)

        return output


class HighPassFilterLayer(BaseFilterLayer):
    """
    简单的高通滤波器层。(弃用)
    执行操作: H' = σ((I - P) * X * W')
    其中 P 是传播矩阵 (如对称归一化邻接矩阵 A_sym)。
    注意: (I - P) X = X - P X
    """
    def __init__(self, in_channels, out_channels, activation: bool = True):
        super().__init__(in_channels, out_channels)
        self.activation = activation


    def forward(self, x: torch.Tensor, p_matrix: torch.Tensor) -> torch.Tensor:
        # 1. 计算 P * X
        support_p = torch.spmm(p_matrix, x)

        # 2. 计算 (I - P) * X = X - P * X
        support_i_minus_p = x - support_p

        # 3. 特征变换: ((I - P) * X) * W'
        output = torch.mm(support_i_minus_p, self.weight) # 使用 self.weight 作为 W'

        # 4.  应用激活函数
        if self.activation:
            output = F.relu(output)

        return output


class Combination(nn.Module):
    '''
    A mod combination the bases of polynomial filters.
    '''
    def __init__(self, channels, level, dropout, sole=False):
        super().__init__()
        self.dropout = dropout
        self.K=level
        
        self.comb_weight = nn.Parameter(torch.ones((1, level, 1)))
        self.reset_parameters()            

    def reset_parameters(self):
        bound = 1.0/self.K
        TEMP = np.random.uniform(bound, bound, self.K)  
        self.comb_weight=nn.Parameter(torch.FloatTensor(TEMP).view(-1,self.K, 1))

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = x * self.comb_weight
        x = torch.sum(x, dim=1)
        return x
