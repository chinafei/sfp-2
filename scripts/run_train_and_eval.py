import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# 确保包引入
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dual_tower_model import PenaltyBiddingLoss
from scripts.train_multi_day import MultiDayFusionModel, MultiDayDataset

def train_and_eval():
    print(">>> 1. 生成训练与测试数据 (包含 D日, D-1日, D-2日 预测数据)...")
    # 生成 200 条训练数据和 50 条测试验证数据
    train_dataset = MultiDayDataset(num_samples=200)
    test_dataset = MultiDayDataset(num_samples=50)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)
    
    print(">>> 2. 初始化 MultiDayFusionModel (采用 learnable 自注意力权重融合策略)...")
    model = MultiDayFusionModel(seq_input_dim=12, static_input_dim=4, weight_strategy='learnable')
    
    # 调高一点学习率以便在 10 个 Epoch 内看到明显的收敛过程
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = PenaltyBiddingLoss(penalty_weight=10.0)
    
    # 【归一化策略核心】：定义实际物理值的归一化缩放系数
    # 容量真实数值约3000~4000，价格约5~25，通过除以常数将 Target 压到 1.0 附近
    # [容量尺度: 4000, 边际价尺度: 10, 出清价尺度: 10]
    target_scales = torch.tensor([4000.0, 10.0, 10.0], dtype=torch.float32)
    
    print("\n>>> 3. 开始训练 (10 Epochs, 期间进行归一化以防止大数量级下的梯度爆炸)...")
    model.train()
    for epoch in range(10):
        epoch_loss = 0
        for xs0, xs1, xs2, xst0, xst1, xst2, y, pmin, pmax in train_loader:
            optimizer.zero_grad()
            
            # --- 训练时：对目标标签进行归一化(除以常数尺度) ---
            y_scaled = y / target_scales.view(1, 1, 3)
            # 限价下限和上限（索引1，2）受到价格尺度(10.0)的同比例缩放
            pmin_scaled = pmin / 10.0  
            pmax_scaled = pmax / 10.0
            
            # 网络直接学习输出处于 0~1 的小幅度归一化区间数据
            preds_scaled = model(xs0, xs1, xs2, xst0, xst1, xst2)
            
            # 使用缩放后的体系计算 Loss 惩罚，防止出现百万级别的 MSE
            loss = criterion(preds_scaled, y_scaled, pmin_scaled, pmax_scaled)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"  Epoch {epoch+1:02d} | 归一化状态下的平均 Loss: {epoch_loss/len(train_loader):.4f}")
        
    print("\n>>> 4. 开始在测试集上预测，恢复到实际值水平，并分析实际偏差 (Bias)...")
    model.eval()
    with torch.no_grad():
        # 取全量测试集进行偏差测算
        xs0, xs1, xs2, xst0, xst1, xst2, y_true_phys, pmin, pmax = next(iter(test_loader))
        
        # 此时输出的预测依然是【归一化水平】的数值 [0~1.x 左近]
        out_raw_scaled = model(xs0, xs1, xs2, xst0, xst1, xst2)
        
        # --- 输出时：反归一化，恢复至真实的兆瓦与元量级 ---
        out_raw_phys = out_raw_scaled * target_scales.view(1, 1, 3)
        
        # 手动执行限价截断规则：使用【真实的 pmin, pmax 物理数值】对价格限制进行钳位
        out_clipped = out_raw_phys.clone()
        p_min_exp = pmin.unsqueeze(-1)
        p_max_exp = pmax.unsqueeze(-1)
        
        # 仅对索引 1(边际排序价) 和 2(市场出清价) 进行上下限硬截断保护
        out_clipped[..., 1:] = torch.max(torch.min(out_raw_phys[..., 1:], p_max_exp), p_min_exp)
        
    # 计算 Bias: (预测 - 真实) 在实际物理值下比对
    bias = out_clipped.numpy() - y_true_phys.numpy()
    
    mean_bias = np.mean(bias, axis=0) 
    mae = np.mean(np.abs(bias), axis=0) 
    
    segments = ['T1', 'T2', 'T3', 'T4', 'T5']
    metrics = ['调频需求容量 (MW)', '边际排序价 (元/MW)', '市场出清价 (元/MW)']
    
    results = []
    for i, seg in enumerate(segments):
        for j, metric in enumerate(metrics):
            results.append({
                '时段': seg,
                '指标': metric,
                '平均偏差 (Bias)': round(mean_bias[i, j], 3),
                '平均绝对误差 (MAE)': round(mae[i, j], 3)
            })
            
    df_result = pd.DataFrame(results)
    
    print("\n=================================== 各时段与各指标 实际物理值预测偏差 (Bias) 评估报表 ===================================")
    print(df_result.to_string(index=False))
    print("=========================================================================================================================")
    print("注：Bias = 预测实际值 - 真实物理值。容量 MAE 显示真实的 MW 级方差；价格受保护在异常波动时自动收敛于限价附近。")

if __name__ == '__main__':
    train_and_eval()