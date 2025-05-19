import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple, Optional

class BaseDynamicBackbone(nn.Module):
    def __init__(self, device: torch.device):

        super().__init__()
        self.device = device

    def forward(self,
                snapshots_data: List[Dict[str, torch.Tensor]],
                # **kwargs 可以用来传递特定于模型的额外参数
                **kwargs
                ) -> Any: # 返回值可以是任何类型，具体由子类决定
        raise NotImplementedError("Subclasses must implement the forward method.")

    def get_final_node_representations(self,
                                       model_output: Any,
                                       # **kwargs
                                       ) -> torch.Tensor:
        if isinstance(model_output, torch.Tensor):
            return model_output
        else:
            raise NotImplementedError("Subclasses must implement get_final_node_representations "
                                      "if forward does not directly return node representations.")

    def reset_parameters(self):
        for module in self.modules():
            if hasattr(module, 'reset_parameters'):
                if module != self:
                    module.reset_parameters()


class BaseTaskHead(nn.Module):
    """
    所有任务头的基类。
    """
    def __init__(self):
        super().__init__()

    def forward(self,
                node_representations: torch.Tensor, # 来自 Backbone 的节点表示
                target_edges: Optional[torch.Tensor] = None, # 对于链接预测
                # **kwargs
                ) -> torch.Tensor: # 通常返回 logits
        raise NotImplementedError("Subclasses must implement the forward method.")
