# utils/torch_utils.py

import torch
import numpy as np
import random
import os

def set_seed(seed: int):
    """
    设置随机种子以保证实验可复现性。

    Args:
        seed (int): 要设置的随机种子。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # 如果使用 CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    print(f"随机种子已设置为: {seed}")

def get_device(device_arg: str = 'auto') -> torch.device:

    if device_arg.lower() == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"自动选择设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    elif 'cuda' in device_arg.lower():
        if not torch.cuda.is_available():
            print(f"警告: 请求使用 CUDA ({device_arg})，但 CUDA 不可用。将使用 CPU。")
            device = torch.device("cpu")
        else:
            try:
                device = torch.device(device_arg)
                # 尝试访问设备以确保其有效
                _ = torch.tensor([1]).to(device)
                print(f"使用指定 CUDA 设备: {device_arg}")
            except (RuntimeError, AssertionError) as e:
                print(f"警告: 指定的 CUDA 设备 '{device_arg}' 无效或不可用: {e}。将使用默认 CUDA 设备 cuda:0。")
                try:
                    device = torch.device("cuda:0")
                    _ = torch.tensor([1]).to(device) # 再次检查默认设备
                except (RuntimeError, AssertionError):
                     print(f"警告: 默认 CUDA 设备 cuda:0 也无法使用。将使用 CPU。")
                     device = torch.device("cpu")

    else: # 假设是 'cpu'
        device = torch.device("cpu")
        print("使用 CPU 设备。")

    # 打印 CUDA 相关信息 (如果使用 CUDA)
    if device.type == 'cuda':
        print(f"  CUDA 设备名称: {torch.cuda.get_device_name(device)}")
        print(f"  CUDA 可用内存: {torch.cuda.get_device_properties(device).total_memory / (1024**3):.2f} GB")

    return device

