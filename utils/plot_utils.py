# utils/plot_utils.py

import matplotlib.pyplot as plt
import os
from typing import List, Dict, Optional

def plot_training_curves(history: Dict[str, List[float]],
                         title: str = "Training Curves",
                         save_dir: str = "plots",
                         filename: str = "training_curves.png",
                         epoch_offset: int = 0):

    if not history:
        print("绘图错误：历史记录字典为空。")
        return

    # 检查必要的键是否存在
    required_keys = ['train_loss', 'val_auc', 'val_ap', 'val_f1'] # 添加 val_f1
    if not all(key in history for key in required_keys):
        print(f"绘图错误：历史记录字典缺少必要的键。需要: {required_keys}")
        return
    if not history['train_loss'] or not history['val_auc'] or not history['val_ap']:
         print(f"绘图警告：历史记录中一个或多个指标列表为空，无法绘图。")
         return


    epochs = history.get('epoch', list(range(epoch_offset + 1, epoch_offset + 1 + len(history['train_loss']))))
    num_epochs = len(epochs)

    # 确保所有列表长度一致 (如果 epoch 是后来添加的)
    if len(history['train_loss']) != num_epochs or \
       len(history['val_auc']) != num_epochs or \
       len(history['val_ap']) != num_epochs or \
       len(history['val_f1']) != num_epochs:
           min_len = min(len(history['train_loss']), len(history['val_auc']), len(history['val_ap']),len(history['val_f1']))
           print(f"绘图警告：历史记录列表长度不一致，将截取到最短长度 {min_len}。")
           epochs = epochs[:min_len]
           history['train_loss'] = history['train_loss'][:min_len]
           history['val_auc'] = history['val_auc'][:min_len]
           history['val_ap'] = history['val_ap'][:min_len]
           num_epochs = min_len
           if num_epochs == 0:
                print(f"绘图错误：截取后列表长度为 0，无法绘图。")
                return


    # --- 创建图表 ---
    # 创建一个包含两个子图的图表 (一个用于 Loss, 一个用于 AUC/AP)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # --- 绘制 Loss 曲线 ---
    ax1 = axes[0]
    ax1.plot(epochs, history['train_loss'], label='Train Loss', marker='o', linestyle='-')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss over Epochs")
    ax1.legend()
    ax1.grid(True)

    # --- 绘制 AUC, AP, F1 曲线 (ax2) ---
    ax2 = axes[1]
    ax2.plot(epochs, history['val_auc'], label='Validation AUC', marker='s', linestyle='-')
    ax2.plot(epochs, history['val_ap'], label='Validation AP', marker='^', linestyle='--')
    ax2.plot(epochs, history['val_f1'], label='Validation F1', marker='x', linestyle=':') # 添加 F1
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title("Validation Metrics over Epochs")
    ax2.legend()
    ax2.grid(True)
    # 可以设置 Y 轴范围以便更好地查看细微变化，例如:
    # ax2.set_ylim(min(min(history['val_auc']), min(history['val_ap'])) - 0.01, 1.0)


    # --- 设置总标题并调整布局 ---
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局防止标题重叠

    # --- 保存图表 ---
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    try:
        plt.savefig(save_path)
        print(f"训练曲线图已保存至: {save_path}")
    except Exception as e:
        print(f"保存绘图时出错: {e}")

    # 关闭图表以释放内存 (如果在非交互式环境运行，这很重要)
    plt.close(fig)

