# utils/metrics.py

import numpy as np
from sklearn.metrics import roc_auc_score as sk_roc_auc_score
from sklearn.metrics import average_precision_score as sk_average_precision_score
from sklearn.metrics import f1_score as sk_f1_score

def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:

    # 检查 y_true 中是否同时包含正负样本，否则 AUC 无定义
    if len(np.unique(y_true)) < 2:
        print("警告: roc_auc_score - 真实标签只包含一个类别，AUC 未定义，返回 0.5")
        return 0.5
    try:
        return float(sk_roc_auc_score(y_true, y_score))
    except ValueError as e:
        # 处理一些 sklearn 可能抛出的错误，例如输入为空
        print(f"计算 ROC AUC 时出错: {e}")
        return 0.0 # 或者返回 NaN 或其他指示错误的值

def average_precision_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    
    if np.sum(y_true) == 0:
        print("警告: average_precision_score - 真实标签中没有正样本，AP 未定义，返回 0.0")
        return 0.0
    try:
        return float(sk_average_precision_score(y_true, y_score))
    except ValueError as e:
        print(f"计算 Average Precision 时出错: {e}")
        return 0.0


def f1_score(y_true: np.ndarray, y_pred_probs: np.ndarray, threshold: float = 0.5) -> float:
    y_pred_binary = (y_pred_probs >= threshold).astype(int)

    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred_binary)) < 2 :
        if np.array_equal(y_true, y_pred_binary):
            return 1.0
        else: 
            if np.sum(y_true) == 0 and np.sum(y_pred_binary) == 0: 
                return 1.0 
    try:
        return float(sk_f1_score(y_true, y_pred_binary, zero_division=0))
    except ValueError:
        return 0.0