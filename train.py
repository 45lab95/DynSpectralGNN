# train.py (最终版，适配 DynSpectral)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import argparse
import os
from typing import List, Dict, Tuple
from tqdm import tqdm # 可选，用于进度条

# 导入项目模块
from utils.data_loader import *
from utils.metrics import roc_auc_score, average_precision_score
from utils.metrics import f1_score
from utils.torch_utils import set_seed, get_device
from models.model import DynSpectral # 导入新的顶层模型
# 导入绘图工具
import matplotlib
matplotlib.use('Agg') # 确保在无 GUI 环境下可用
import matplotlib.pyplot as plt
from utils.plot_utils import plot_training_curves

def train_one_epoch(model: DynSpectral,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module,
                    current_input_sequence: List[Dict[str, torch.Tensor]],
                    current_target_edges: Dict[str, torch.Tensor],
                    device: torch.device,
                    lambda_contrastive: float
                    ) -> Tuple[float, float, float]:
    model.train()
    optimizer.zero_grad()
    model_input_snapshots = []
    for snap_data_dict in current_input_sequence:
        model_input_snapshots.append({
            'initial_features': snap_data_dict['initial_features'], # CPU
            'p_matrix': snap_data_dict['p_matrix'],                 # CPU
            'homophily_bases': snap_data_dict['homophily_bases']    # CPU
        })

    target_pos_edges = current_target_edges['pos_edge_index'].to(device)
    target_neg_edges = current_target_edges['neg_edge_index'].to(device)

    if target_pos_edges.numel() == 0 and target_neg_edges.numel() == 0: 
        return 0.0, 0.0, 0.0

    predict_edge_index = torch.cat([target_pos_edges, target_neg_edges], dim=1)

    if predict_edge_index.numel() == 0:
        return 0.0, 0.0, 0.0


    task_logits, contrastive_loss = model(model_input_snapshots, target_edges=predict_edge_index)

    pos_labels = torch.ones(target_pos_edges.size(1), device=device)
    neg_labels = torch.zeros(target_neg_edges.size(1), device=device)
    labels = torch.cat([pos_labels, neg_labels])
    
    if task_logits.size(0) != labels.size(0): # 增加一个检查
        print(f"错误: Logits ({task_logits.shape}) 和 Labels ({labels.shape}) 形状不匹配。")
        print(f"  target_pos_edges: {target_pos_edges.shape}, target_neg_edges: {target_neg_edges.shape}")
        print(f"  predict_edge_index: {predict_edge_index.shape}")
        return 0.0,0.0,0.0


    main_task_loss = criterion(task_logits, labels)
    total_loss_for_sample = main_task_loss + lambda_contrastive * contrastive_loss

    total_loss_for_sample.backward()
    optimizer.step()

    return total_loss_for_sample.item(), main_task_loss.item(), contrastive_loss.item()


@torch.no_grad()
def evaluate(model: DynSpectral,
             # 输入变为固定的历史窗口和一系列目标
             history_input_snapshots: List[Dict[str, torch.Tensor]], # 长度为 W_eval
             target_edges_list_for_eval: List[Dict[str, torch.Tensor]], # 包含多个未来时间步的目标
             device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    all_logits = []
    all_labels = []
    model_input_for_all_predictions = []
    for snap_data_dict in history_input_snapshots:
        model_input_for_all_predictions.append({
            'initial_features': snap_data_dict['initial_features'],
            'p_matrix': snap_data_dict['p_matrix'],
            'homophily_bases': snap_data_dict['homophily_bases']
        })


    for target_edges_dict in target_edges_list_for_eval: # 迭代每个目标时间步
        target_pos_edges = target_edges_dict['pos_edge_index'].to(device)
        target_neg_edges = target_edges_dict['neg_edge_index'].to(device)

        if target_pos_edges.numel() == 0 or target_neg_edges.numel() == 0: continue

        predict_edge_index = torch.cat([target_pos_edges, target_neg_edges], dim=1)
        # 每次都使用相同的 history_input_snapshots 进行预测
        task_logits, _ = model(model_input_for_all_predictions, target_edges=predict_edge_index)

        pos_labels = torch.ones(target_pos_edges.size(1), device=device)
        neg_labels = torch.zeros(target_neg_edges.size(1), device=device)
        labels = torch.cat([pos_labels, neg_labels])

        all_logits.append(task_logits.cpu())
        all_labels.append(labels.cpu())

    if not all_logits: return 0.0, 0.0, 0.0
    final_logits = torch.cat(all_logits).numpy(); final_labels = torch.cat(all_labels).numpy()
    auc = roc_auc_score(final_labels, final_logits)
    ap = average_precision_score(final_labels, final_logits)
    probs = 1 / (1 + np.exp(-final_logits))
    f1 = f1_score(final_labels, probs)
    return auc, ap, f1


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Dual Channel Dynamic Spectral GNN with UniBasis')
    # 数据集参数
    parser.add_argument('--dataset_name', type=str, default='bitcoin_otc', choices=['bitcoin_otc', 'uci', 'enron'], # 添加 enron
                        help='要使用的数据集名称')
    parser.add_argument('--bitcoin_otc_root', type=str, default='./bitcoin_otc_pyg_raw',
                        help='BitcoinOTC PyG 数据集根目录')
    parser.add_argument('--uc_irvine_path', type=str, default='data/uci/out.opsahl-ucsocial',
                        help='UC Irvine messages 原始数据文件路径')
    parser.add_argument('--enron_path', type=str, default='data/enron_email/enron_edges.tsv', 
                        help='Enron email 原始数据文件路径')
    parser.add_argument('--edge_window_size', type=int, default=10,
                        help='BitcoinOTC 边窗口大小 (对 UC Irvine/Enron 无效，它们使用 time_window_days)')
    parser.add_argument('--time_window_days', type=int, default=1,
                        help='事件型数据集（如UC Irvine, Enron）聚合快照的时间窗口天数')


    parser.add_argument('--initial_feature_dim', type=int, default=1, help='初始节点特征维度 (例如，度数为1)') # 需要这个
    parser.add_argument('--K', type=int, default=5, help='UniBasis 多项式阶数')
    parser.add_argument('--tau', type=float, default=0.5, help='UniBasis 同配/异配混合系数 τ')
    # --- 新增/修改：对比学习相关 h_hat 和 dropout ---
    parser.add_argument('--h_hat1', type=float, default=0.2, help='通道1 目标同配率 ĥ1 (偏异配)')
    parser.add_argument('--h_hat1_prime', type=float, default=0.1, help='通道1 对比正样本同配率 ĥ1\' (更异配)')
    parser.add_argument('--h_hat2', type=float, default=0.8, help='通道2 目标同配率 ĥ2 (偏同配)')
    parser.add_argument('--h_hat2_prime', type=float, default=0.9, help='通道2 对比正样本同配率 ĥ2\' (更同配)')
    parser.add_argument('--h_hat_anchor', type=float, default=0.5, help='锚点通道同配率 ĥ_anchor')
    parser.add_argument('--combination_dropout_ch1', type=float, default=0.3, help='通道1/1\' Combination dropout')
    parser.add_argument('--combination_dropout_ch2', type=float, default=0.3, help='通道2/2\' Combination dropout')
    parser.add_argument('--combination_dropout_anchor', type=float, default=0.3, help='锚点 Combination dropout')
    parser.add_argument('--contrastive_temperature', type=float, default=0.1, help='InfoNCE 温度参数')
    parser.add_argument('--lambda_contrastive', type=float, default=0.1, help='对比损失的权重')
    parser.add_argument('--contrastive_loss_interval', type=int, default=1, help='每隔多少时间步计算对比损失')
    # LSTM (Backbone) 参数
    parser.add_argument('--lstm_hidden', type=int, default=64, help='单个 LSTM 隐藏层维度')
    parser.add_argument('--lstm_layers', type=int, default=1, help='LSTM 层数')
    parser.add_argument('--lstm_dropout', type=float, default=0.0, help='LSTM 层间 dropout')
    # LinkPredictorHead (Task Head) 参数
    parser.add_argument('--link_pred_hidden', type=int, default=64, help='链接预测头 MLP 隐藏维度')
    # 训练参数
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--lr_comb1', type=float, default=0.005, help='通道1 Combination 层学习率')
    parser.add_argument('--lr_comb2', type=float, default=0.005, help='通道2 Combination 层学习率')
    parser.add_argument('--lr_comb_anchor', type=float, default=0.005, help='锚点 Combination 层学习率')
    parser.add_argument('--wd_comb_anchor', type=float, default=0.0, help='锚点 Combination 层权重衰减')
    parser.add_argument('--lr_lstm1', type=float, default=0.01, help='通道1 LSTM 层学习率')
    parser.add_argument('--lr_lstm2', type=float, default=0.01, help='通道2 LSTM 层学习率')
    parser.add_argument('--lr_pred_head', type=float, default=0.01, help='预测头学习率')
    parser.add_argument('--wd_comb1', type=float, default=0.0, help='通道1 Combination 层权重衰减')
    parser.add_argument('--wd_comb2', type=float, default=0.0, help='通道2 Combination 层权重衰减')
    parser.add_argument('--wd_lstm1', type=float, default=0.0, help='通道1 LSTM 层权重衰减')
    parser.add_argument('--wd_lstm2', type=float, default=0.0, help='通道2 LSTM 层权重衰减')
    parser.add_argument('--wd_pred_head', type=float, default=0.0, help='预测头权重衰减')
    parser.add_argument('--patience', type=int, default=50, help='早停耐心值')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='训练集时间步比例')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='验证集时间步比例')
    # 其他
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='auto', help='计算设备 (auto, cpu, cuda:0)')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='模型保存目录')
    parser.add_argument('--model_name_prefix', type=str, default='dynspectral_dual_unibasis', help='模型文件名前缀')
    parser.add_argument('--history_window_size', type=int, default=10,
                        help='用于LSTM输入的历史快照窗口大小 (W)')

    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    print("--- 配置参数 ---")
    for arg, value in sorted(vars(args).items()): print(f"  {arg}: {value}")

    set_seed(args.seed); device = get_device(args.device)
    args.model_name = f"{args.model_name_prefix}_{args.dataset_name.lower()}_K{args.K}_W{args.history_window_size}.pt"
    plot_save_dir = "plots"; plot_filename = f"curves_{args.model_name.replace('.pt', '')}.png"
    os.makedirs(args.save_dir, exist_ok=True); os.makedirs(plot_save_dir, exist_ok=True)
    model_save_path = os.path.join(args.save_dir, args.model_name)

    print("\n--- 加载数据 ---")
    if args.dataset_name.lower() == 'bitcoin_otc':
        snapshots_data, num_nodes, initial_feature_dim_actual = load_bitcoin_otc_data(
            root=args.bitcoin_otc_root, edge_window_size=args.edge_window_size,
            K_unibasis=args.K, feature_generator=generate_node_features
            # tau, h_hats 现在是模型参数，不由 data_loader 直接使用
        )
    elif args.dataset_name.lower() in ['uc_irvine', 'uci']:
        snapshots_data, num_nodes, initial_feature_dim_actual = load_uci(
            data_path=args.uc_irvine_path, K_unibasis=args.K,
            feature_generator=generate_node_features
        )
    elif args.dataset_name.lower() == 'enron':
        snapshots_data, num_nodes, initial_feature_dim_actual = load_event_data_by_time_window(
            data_path=args.enron_path, time_window_days=args.time_window_days,
            K_unibasis=args.K, feature_generator=generate_node_features, dataset_name="Enron Email"
        )
    else:
        raise ValueError(f"未知或不支持的数据集名称: {args.dataset_name}")

    if args.initial_feature_dim != initial_feature_dim_actual:
        print(f"警告: 参数指定的 initial_feature_dim ({args.initial_feature_dim}) 与数据加载器返回的 ({initial_feature_dim_actual}) 不符。将使用加载器返回的维度。")
    final_initial_feature_dim = initial_feature_dim_actual
    
    num_total_labeled_snapshots = len(snapshots_data)
    if num_total_labeled_snapshots < args.history_window_size + 1: # 至少 W 用于历史, 1 用于标签
        raise ValueError(f"数据快照总数 ({num_total_labeled_snapshots}) 过少 (少于 W+1={args.history_window_size + 1})，无法进行有效的训练。")

    train_steps_indices, val_steps_indices, test_steps_indices = get_dynamic_data_splits(
        num_total_labeled_snapshots, args.train_ratio, args.val_ratio
    )

    print("\n--- 准备滑动窗口训练样本 ---")
    training_samples = []
    for target_idx_in_all_snapshots in train_steps_indices:
        if target_idx_in_all_snapshots < args.history_window_size: continue
        input_start_idx = target_idx_in_all_snapshots - args.history_window_size
        input_end_idx = target_idx_in_all_snapshots
        current_input_sequence = [snapshots_data[i] for i in range(input_start_idx, input_end_idx)]
        current_target_edges = snapshots_data[target_idx_in_all_snapshots] # 整个字典包含 pos/neg
        if current_target_edges['pos_edge_index'].numel() > 0 or current_target_edges['neg_edge_index'].numel() > 0:
            training_samples.append({'inputs': current_input_sequence, 'targets': current_target_edges})
    print(f"生成了 {len(training_samples)} 个滑动窗口训练样本。")
    if not training_samples: print("错误：未能生成任何训练样本。"); return

    val_history_start_idx = train_steps_indices[-1] - args.history_window_size + 1 if train_steps_indices else 0
    val_history_end_idx = train_steps_indices[-1] + 1 if train_steps_indices else 0
    if val_history_start_idx < 0 : val_history_start_idx = 0
    val_input_history = [snapshots_data[i] for i in range(val_history_start_idx, val_history_end_idx)] if val_history_end_idx > val_history_start_idx else []
    val_target_edges_for_eval = [snapshots_data[i] for i in val_steps_indices if snapshots_data[i]['pos_edge_index'].numel() > 0 or snapshots_data[i]['neg_edge_index'].numel() > 0]


    test_history_start_idx = val_steps_indices[-1] - args.history_window_size + 1 if val_steps_indices else (train_steps_indices[-1] - args.history_window_size + 1 if train_steps_indices else 0)
    test_history_end_idx = val_steps_indices[-1] + 1 if val_steps_indices else (train_steps_indices[-1] + 1 if train_steps_indices else 0)
    if test_history_start_idx < 0 : test_history_start_idx = 0
    test_input_history = [snapshots_data[i] for i in range(test_history_start_idx, test_history_end_idx)] if test_history_end_idx > test_history_start_idx else []
    test_target_edges_for_eval = [snapshots_data[i] for i in test_steps_indices if snapshots_data[i]['pos_edge_index'].numel() > 0 or snapshots_data[i]['neg_edge_index'].numel() > 0]

    print(f"Val input history length: {len(val_input_history)}, Val target steps: {len(val_target_edges_for_eval)}")
    print(f"Test input history length: {len(test_input_history)}, Test target steps: {len(test_target_edges_for_eval)}")

    print("\n--- 初始化模型 ---")
    model = DynSpectral(
        device=device, initial_feature_dim=final_initial_feature_dim, K=args.K, tau=args.tau,
        h_hat_ch1=args.h_hat1, h_hat_ch1_prime=args.h_hat1_prime, h_hat_ch2=args.h_hat2,
        h_hat_ch2_prime=args.h_hat2_prime, h_hat_anchor=args.h_hat_anchor,
        combination_dropout_ch1=args.combination_dropout_ch1,
        combination_dropout_ch2=args.combination_dropout_ch2,
        combination_dropout_anchor=args.combination_dropout_anchor,
        contrastive_temperature=args.contrastive_temperature,
        contrastive_loss_interval=args.contrastive_loss_interval,
        lstm_hidden_dim=args.lstm_hidden, lstm_layers=args.lstm_layers, lstm_dropout=args.lstm_dropout,
        link_pred_hidden_dim=args.link_pred_hidden
    )
    print(model); num_params = sum(p.numel() for p in model.parameters() if p.requires_grad); print(f"模型总可训练参数数量: {num_params}")

    optimizer = optim.Adam([
        {'params': model.backbone.contrastive_head.view_gen_ch1.parameters(), 'lr': args.lr_comb1, 'weight_decay': args.wd_comb1},
        {'params': model.backbone.contrastive_head.view_gen_ch1_prime.parameters(), 'lr': args.lr_comb1, 'weight_decay': args.wd_comb1},
        {'params': model.backbone.contrastive_head.view_gen_ch2.parameters(), 'lr': args.lr_comb2, 'weight_decay': args.wd_comb2},
        {'params': model.backbone.contrastive_head.view_gen_ch2_prime.parameters(), 'lr': args.lr_comb2, 'weight_decay': args.wd_comb2},
        {'params': model.backbone.contrastive_head.view_gen_anchor.parameters(), 'lr': args.lr_comb_anchor, 'weight_decay': args.wd_comb_anchor},
        {'params': model.backbone.lstm_encoder1.parameters(), 'lr': args.lr_lstm1, 'weight_decay': args.wd_lstm1},
        {'params': model.backbone.lstm_encoder2.parameters(), 'lr': args.lr_lstm2, 'weight_decay': args.wd_lstm2},
        {'params': model.task_head.parameters(), 'lr': args.lr_pred_head, 'weight_decay': args.wd_pred_head}
    ])
    print("\n优化器已更新参数组。")
    criterion = nn.BCEWithLogitsLoss()
    training_history = { 'epoch': [], 'train_loss': [], 'main_loss': [], 'cl_loss':[], 'val_auc': [], 'val_ap': [] ,'val_f1':[]}

    print("\n--- 开始训练 (滑动窗口) ---")
    best_val_auc = 0.0; patience_counter = 0; start_time = time.time()
    if not training_samples: print("错误：没有有效的训练样本。"); return

    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        epoch_start_time = time.time()
        epoch_total_loss_sum, epoch_main_loss_sum, epoch_cl_loss_sum, num_samples_in_epoch = 0.0, 0.0, 0.0, 0
        np.random.shuffle(training_samples)
        for sample in tqdm(training_samples, desc=f"Epoch {epoch+1} Batches", leave=False): 
            total_s, main_s, cl_s = train_one_epoch(
                model, optimizer, criterion, sample['inputs'], sample['targets'], device, args.lambda_contrastive
            )
            if total_s > 0 or main_s > 0 or cl_s > 0 : # 如果有有效损失
                epoch_total_loss_sum += total_s; epoch_main_loss_sum += main_s; epoch_cl_loss_sum += cl_s
                num_samples_in_epoch +=1
        
        avg_epoch_total_loss = epoch_total_loss_sum / num_samples_in_epoch if num_samples_in_epoch > 0 else 0.0
        avg_epoch_main_loss = epoch_main_loss_sum / num_samples_in_epoch if num_samples_in_epoch > 0 else 0.0
        avg_epoch_cl_loss = epoch_cl_loss_sum / num_samples_in_epoch if num_samples_in_epoch > 0 else 0.0

        val_auc, val_ap, val_f1 = (0.0, 0.0, 0.0)
        if val_target_edges_for_eval and val_input_history:
            val_auc, val_ap, val_f1 = evaluate(model, val_input_history, val_target_edges_for_eval, device)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Time: {epoch_time:.2f}s | "
              f"AvgTotalLoss: {avg_epoch_total_loss:.4f} (Main: {avg_epoch_main_loss:.4f}, CL: {avg_epoch_cl_loss:.4f}) | "
              f"Val Avg AUC: {val_auc:.4f} | Val Avg AP: {val_ap:.4f} | Val Avg F1: {val_f1:.4f}")
        training_history['epoch'].append(epoch + 1); training_history['train_loss'].append(avg_epoch_total_loss)
        training_history['main_loss'].append(avg_epoch_main_loss); training_history['cl_loss'].append(avg_epoch_cl_loss)
        training_history['val_auc'].append(val_auc); training_history['val_ap'].append(val_ap); training_history['val_f1'].append(val_f1)

        if val_auc > best_val_auc and val_target_edges_for_eval:
            best_val_auc = val_auc; patience_counter = 0
            print(f"  New best validation AUC found! Saving model to {model_save_path}")
            torch.save(model.state_dict(), model_save_path)
        elif val_target_edges_for_eval:
            patience_counter += 1
        if patience_counter >= args.patience and val_target_edges_for_eval:
            print(f"  Validation AUC did not improve for {args.patience} epochs. Early stopping.")
            break
    
    total_train_time = time.time() - start_time
    print(f"\n--- 训练完成 --- Total Time: {total_train_time:.2f}s")
    try:
        plot_training_curves(training_history, title=f"Training Curves for {args.model_name.replace('.pt', '')}", save_dir=plot_save_dir, filename=plot_filename)
    except Exception as e: print(f"绘制训练曲线时发生错误: {e}")

    print("\n--- 开始测试 ---")
    if os.path.exists(model_save_path) and test_target_edges_for_eval and test_input_history:
        print(f"Loading best model from {model_save_path}")
        model.load_state_dict(torch.load(model_save_path, map_location=device)) # 添加 map_location
        test_auc, test_ap, test_f1 = evaluate(model, test_input_history, test_target_edges_for_eval, device)
        print(f"Test Results --> Avg AUC: {test_auc:.4f} | Avg AP: {test_ap:.4f} | Avg F1: {test_f1:.4f}")
    elif not os.path.exists(model_save_path):
        print(f"错误：找不到保存的最佳模型 '{model_save_path}'，无法进行测试。")
    else:
        print("警告：测试集目标边或历史输入为空，跳过测试。")


if __name__ == "__main__":
    main()