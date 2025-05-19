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
                    train_snapshots: List[Dict[str, torch.Tensor]],
                    train_target_edges: List[Dict[str, torch.Tensor]],
                    device: torch.device,
                    lambda_contrastive: float # <--- 添加这个参数的声明
                    ) -> Tuple[float, float, float]:
    """执行一个训练轮次 (包含对比损失)"""
    model.train()
    optimizer.zero_grad()
    total_epoch_loss = 0.0
    total_main_task_loss = 0.0
    total_contrastive_loss_val = 0.0
    num_processed_steps = 0

    for t in tqdm(range(len(train_snapshots)), desc="Training Steps", leave=False, disable=True):
        model_input_snapshots = []
        for i in range(t + 1):
            snapshot_data_for_model = {
                'initial_features': train_snapshots[i]['initial_features'].to(device),
                'p_matrix': train_snapshots[i]['p_matrix'].to(device),
                'homophily_bases': [b.to(device) for b in train_snapshots[i]['homophily_bases']]
            }
            model_input_snapshots.append(snapshot_data_for_model)

        target_edges_t_plus_1 = train_target_edges[t]
        target_pos_edges = target_edges_t_plus_1['pos_edge_index'].to(device)
        target_neg_edges = target_edges_t_plus_1['neg_edge_index'].to(device)

        if target_pos_edges.numel() == 0 or target_neg_edges.numel() == 0: continue

        predict_edge_index = torch.cat([target_pos_edges, target_neg_edges], dim=1)
        task_logits, contrastive_loss = model(model_input_snapshots, target_edges=predict_edge_index)

        pos_labels = torch.ones(target_pos_edges.size(1), device=device)
        neg_labels = torch.zeros(target_neg_edges.size(1), device=device)
        labels = torch.cat([pos_labels, neg_labels])

        main_task_loss = criterion(task_logits, labels)
        # 使用传入的 lambda_contrastive
        current_step_total_loss = main_task_loss + lambda_contrastive * contrastive_loss

        current_step_total_loss.backward()
        total_epoch_loss += current_step_total_loss.item()
        total_main_task_loss += main_task_loss.item()
        total_contrastive_loss_val += contrastive_loss.item()
        num_processed_steps += 1

    if num_processed_steps > 0:
        optimizer.step()
        optimizer.zero_grad()
        avg_total_loss = total_epoch_loss / num_processed_steps
        avg_main_task_loss = total_main_task_loss / num_processed_steps
        avg_contrastive_loss = total_contrastive_loss_val / num_processed_steps
    else:
        avg_total_loss = 0.0
        avg_main_task_loss = 0.0
        avg_contrastive_loss = 0.0

    return avg_total_loss, avg_main_task_loss, avg_contrastive_loss

@torch.no_grad()
def evaluate(model: DynSpectral,
             input_snapshots: List[Dict[str, torch.Tensor]], # 包含 ch1 和 ch2 特征
             target_edges_list: List[Dict[str, torch.Tensor]],
             device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    all_logits = []
    all_labels = []

    model_input_snapshots_for_eval = []
    for i in range(len(input_snapshots)):
         snapshot_data_for_model = {
            'initial_features': input_snapshots[i]['initial_features'], # 保持在 CPU
            'p_matrix': input_snapshots[i]['p_matrix'],                 # 保持在 CPU
            'homophily_bases': input_snapshots[i]['homophily_bases']   # 保持在 CPU
         }
         model_input_snapshots.append(snapshot_data_for_model)

    for t in tqdm(range(len(target_edges_list)), desc="Evaluating Steps", leave=False, disable=True):
        target_edges_t = target_edges_list[t]
        target_pos_edges = target_edges_t['pos_edge_index'].to(device)
        target_neg_edges = target_edges_t['neg_edge_index'].to(device)
        if target_pos_edges.numel() == 0 or target_neg_edges.numel() == 0: continue
        predict_edge_index = torch.cat([target_pos_edges, target_neg_edges], dim=1)
        logits = model(model_input_snapshots, target_edges=predict_edge_index)
        pos_labels = torch.ones(target_pos_edges.size(1), device=device); neg_labels = torch.zeros(target_neg_edges.size(1), device=device)
        labels = torch.cat([pos_labels, neg_labels])
        all_logits.append(logits.cpu()); all_labels.append(labels.cpu())

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
    parser.add_argument('--K', type=int, default=10, help='UniBasis 多项式阶数')
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

    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()

    print("--- 配置参数 ---");
    set_seed(args.seed); device = get_device(args.device)
    args.model_name = f"{args.model_name_prefix}_{args.dataset_name.lower()}.pt"
    plot_save_dir = "plots"; plot_filename = f"curves_{args.model_name.replace('.pt', '')}.png"
    os.makedirs(args.save_dir, exist_ok=True); model_save_path = os.path.join(args.save_dir, args.model_name)


    print("\n--- 加载数据 ---")
    # feature_dim_F 现在代表初始 X_t 的维度
    if args.dataset_name.lower() == 'bitcoin_otc':
        snapshots_data, num_nodes, initial_feature_dim_actual = load_bitcoin_otc_data(
            root=args.bitcoin_otc_root, edge_window_size=args.edge_window_size,
            feature_generator=generate_node_features,
            # K, tau, h_hats 不再由 data_loader 直接使用，而是传递给模型
        )
    elif args.dataset_name.lower() == 'uci':
        snapshots_data, num_nodes, initial_feature_dim_actual = load_uci(
        data_path=args.uc_irvine_path,
        K_unibasis=args.K, # 传递 K
        feature_generator=generate_node_features
    )
    elif args.dataset_name.lower() == 'enron':
        snapshots_data, num_nodes, initial_feature_dim_actual = load_enron_email_data(
            data_path=args.enron_path, time_window_days=args.time_window_days,
            feature_generator=generate_node_features, dataset_name="Enron Email"
        )
    else:
        raise ValueError(f"未知或不支持的数据集名称: {args.dataset_name}")
    
    if args.initial_feature_dim != initial_feature_dim_actual: # args.initial_feature_dim 是从命令行读取的
        print(f"警告: 参数指定的 initial_feature_dim ({args.initial_feature_dim}) 与数据加载器返回的 ({initial_feature_dim_actual}) 不符。将使用后者。")
    final_initial_feature_dim = initial_feature_dim_actual
    
    num_time_steps = len(snapshots_data); 
    train_steps_idx, val_steps_idx, test_steps_idx = get_dynamic_data_splits(
        num_time_steps, args.train_ratio, args.val_ratio
    )

    train_snapshots = [snapshots_data[i] for i in train_steps_idx]
    train_target_edges = [{'pos_edge_index': snapshots_data[i+1]['pos_edge_index'], 'neg_edge_index': snapshots_data[i+1]['neg_edge_index']} for i in train_steps_idx]
    val_input_snapshots = [snapshots_data[i] for i in train_steps_idx] # 验证集输入是整个训练历史
    val_target_edges = [{'pos_edge_index': snapshots_data[i]['pos_edge_index'], 'neg_edge_index': snapshots_data[i]['neg_edge_index']} for i in val_steps_idx]
    test_input_snapshots = [snapshots_data[i] for i in range(test_steps_idx[0])] # 测试集输入是到测试开始前的所有历史
    test_target_edges = [{'pos_edge_index': snapshots_data[i]['pos_edge_index'], 'neg_edge_index': snapshots_data[i]['neg_edge_index']} for i in test_steps_idx]

    print(f"\n数据准备完成: Train input steps={len(train_snapshots)}, Train target steps={len(train_target_edges)}")
    print(f"Val input steps={len(val_input_snapshots)}, Val target steps={len(val_target_edges)}")
    print(f"Test input steps={len(test_input_snapshots)}, Test target steps={len(test_target_edges)}")

    # --- 初始化模型 (传入双通道 dropout) ---
    print("\n--- 初始化模型 ---")
    model = DynSpectral(
        device=device,
        # --- 修改这里：使用正确的参数名 ---
        initial_feature_dim=final_initial_feature_dim, # 将 unibasis_base_feature_dim 改为 initial_feature_dim
        # --- 修改结束 ---
        K=args.K,
        tau=args.tau,
        h_hat_ch1=args.h_hat1, h_hat_ch1_prime=args.h_hat1_prime,
        h_hat_ch2=args.h_hat2, h_hat_ch2_prime=args.h_hat2_prime,
        h_hat_anchor=args.h_hat_anchor,
        combination_dropout_ch1=args.combination_dropout_ch1,
        combination_dropout_ch2=args.combination_dropout_ch2,
        combination_dropout_anchor=args.combination_dropout_anchor,
        contrastive_temperature=args.contrastive_temperature,
        contrastive_loss_interval=args.contrastive_loss_interval,
        lstm_hidden_dim=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        lstm_dropout=args.lstm_dropout,
        link_pred_hidden_dim=args.link_pred_hidden
    ).to(device)
    print(model); num_params = sum(p.numel() for p in model.parameters() if p.requires_grad); print(f"模型总可训练参数数量: {num_params}")


    # --- 设置优化器 (使用更细致的参数组) ---
    optimizer = optim.Adam([
        {'params': model.backbone.contrastive_head.view_gen_ch1.parameters(), 'lr': args.lr_comb1, 'weight_decay': args.wd_comb1},
        {'params': model.backbone.contrastive_head.view_gen_ch1_prime.parameters(), 'lr': args.lr_comb1, 'weight_decay': args.wd_comb1}, # 可以共享LR/WD
        {'params': model.backbone.contrastive_head.view_gen_ch2.parameters(), 'lr': args.lr_comb2, 'weight_decay': args.wd_comb2},
        {'params': model.backbone.contrastive_head.view_gen_ch2_prime.parameters(), 'lr': args.lr_comb2, 'weight_decay': args.wd_comb2},
        {'params': model.backbone.contrastive_head.view_gen_anchor.parameters(), 'lr': args.lr_comb_anchor, 'weight_decay': args.wd_comb_anchor}, # 为 anchor 单独设置
        {'params': model.backbone.lstm_encoder1.parameters(), 'lr': args.lr_lstm1, 'weight_decay': args.wd_lstm1},
        {'params': model.backbone.lstm_encoder2.parameters(), 'lr': args.lr_lstm2, 'weight_decay': args.wd_lstm2},
        {'params': model.task_head.parameters(), 'lr': args.lr_pred_head, 'weight_decay': args.wd_pred_head}
    ])
    print("\n优化器已设置参数组。")

    criterion = nn.BCEWithLogitsLoss()
    training_history = { 'epoch': [], 'train_loss': [], 'val_auc': [], 'val_ap': [] ,'val_f1':[]}

    print("\n--- 开始训练 ---") 
    best_val_auc = 0.0
    patience_counter = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        # loss = train_one_epoch(model, optimizer, criterion, train_snapshots, train_target_edges, device)
        total_loss_epoch, main_loss_epoch, cl_loss_epoch = train_one_epoch(
            model, optimizer, criterion,
            train_snapshots, train_target_edges, device,
            args.lambda_contrastive # 传递对比损失权重
        )
        val_auc, val_ap, val_f1 = evaluate(model, val_input_snapshots, val_target_edges, device)
        epoch_time = time.time() - epoch_start_time 

        print(f"Epoch {epoch+1}/{args.epochs} | Time: {epoch_time:.2f}s | "
              f"Total Loss: {total_loss_epoch:.4f} (Main: {main_loss_epoch:.4f}, CL: {cl_loss_epoch:.4f}) | "
              f"Val Avg AUC: {val_auc:.4f} | Val Avg AP: {val_ap:.4f} | Val Avg F1: {val_f1:.4f}")
        training_history['epoch'].append(epoch + 1); training_history['train_loss'].append(total_loss_epoch);training_history['main_loss'].append(main_loss_epoch);training_history['cl_loss'].append(cl_loss_epoch); training_history['val_auc'].append(val_auc); training_history['val_ap'].append(val_ap);training_history['val_f1'].append(val_f1)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            print(f"  New best validation AUC found! Saving model to {model_save_path}")
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Validation AUC did not improve for {args.patience} epochs. Early stopping.")
                break

    total_train_time = time.time() - start_time
    print(f"\n--- 训练完成 --- Total Time: {total_train_time:.2f}s")

    try:
        plot_training_curves(training_history, title=f"Training Curves ({args.model_name.replace('.pt', '')})", save_dir=plot_save_dir, filename=plot_filename)
    except Exception as e: print(f"绘制训练曲线时发生错误: {e}")

    print("\n--- 开始测试 ---")
    if os.path.exists(model_save_path):
        print(f"Loading best model from {model_save_path}")
        model.load_state_dict(torch.load(model_save_path))
        test_auc, test_ap, test_f1 = evaluate(model, test_input_snapshots, test_target_edges, device) # 接收 F1
        print(f"Test Results --> Avg AUC: {test_auc:.4f} | Avg AP: {test_ap:.4f} | Avg F1: {test_f1:.4f}")
    else:
        print(f"错误：找不到保存的最佳模型 '{model_save_path}'，无法进行测试。")

if __name__ == "__main__":
    main()