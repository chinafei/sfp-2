import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.train_multi_day import MultiDayFusionModel, MultiDayDataset

def predict_single_day():
    print(">>> 正在初始化模型与加载单日模拟特征...")
    # 初始化网络(这里直接使用初始权重作为展现，真实环境可 torch.load 加载)
    model = MultiDayFusionModel(seq_input_dim=12, static_input_dim=4, weight_strategy='learnable')
    model.eval()

    # 随机生成/抓取 1 天的数据
    dataset = MultiDayDataset(num_samples=1)
    xs0, xs1, xs2, xst0, xst1, xst2, y_true, pmin, pmax = dataset[0]

    # 添加 batch 维度 [1, ...]
    xs0, xs1, xs2 = xs0.unsqueeze(0), xs1.unsqueeze(0), xs2.unsqueeze(0)
    xst0, xst1, xst2 = xst0.unsqueeze(0), xst1.unsqueeze(0), xst2.unsqueeze(0)
    
    print("\n>>> 开始进行单日二次调频辅助服务报价/出清预测...")
    target_scales = torch.tensor([4000.0, 10.0, 10.0], dtype=torch.float32)

    with torch.no_grad():
        out_raw_scaled = model(xs0, xs1, xs2, xst0, xst1, xst2)
        out_raw_phys = out_raw_scaled * target_scales.view(1, 1, 3)
        
        # 执行限价截断逻辑
        out_clipped = out_raw_phys.clone()
        p_min_exp = pmin.unsqueeze(0).unsqueeze(-1) # [1, 5, 1]
        p_max_exp = pmax.unsqueeze(0).unsqueeze(-1)
        
        # 对索引1(边际价)和2(出清价)应用原值的限价约束
        out_clipped[..., 1:] = torch.max(torch.min(out_raw_phys[..., 1:], p_max_exp), p_min_exp)

    # 整理结果为 DataFrame 显示
    segments = ['T1', 'T2', 'T3', 'T4', 'T5']
    
    # 转为 numpy 以便打印 (去除 batch 维度)
    y_true_np = y_true.numpy()
    preds_np = out_clipped[0].numpy()
    pmin_np = pmin.numpy()
    pmax_np = pmax.numpy()

    results = []
    for i, seg in enumerate(segments):
        # target_idx => 0: 容量, 1: 边际价, 2: 出清价
        limit_str = f"[{int(pmin_np[i])}, {int(pmax_np[i])}]"
        
        results.append({
            '时段': seg,
            '价格政策上下限': limit_str,
            '目标': '调频需求容量 (MW)',
            '真实观测值': round(y_true_np[i, 0], 2),
            '模型预测值': round(preds_np[i, 0], 2)
        })
        results.append({
            '时段': seg,
            '价格政策上下限': limit_str,
            '目标': '边际排序价 (元/MW)',
            '真实观测值': round(y_true_np[i, 1], 2),
            '模型预测值': round(preds_np[i, 1], 2)
        })
        results.append({
            '时段': seg,
            '价格政策上下限': limit_str,
            '目标': '市场出清价 (元/MW)',
            '真实观测值': round(y_true_np[i, 2], 2),
            '模型预测值': round(preds_np[i, 2], 2)
        })

    df = pd.DataFrame(results)
    
    print("\n============================== 单日预测结果面板 (Sample) ==============================")
    # 格式化输出对齐
    df_styled = df.set_index(['时段', '价格政策上下限', '目标'])
    print(df_styled.to_string())
    print("=======================================================================================")
    print("说明：价格指标（边际排序价、市场出清价）已被强制约束在所列的限价政策上下限之内；容量不受限制。")

if __name__ == "__main__":
    predict_single_day()