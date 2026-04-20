import os
import sys
import numpy as np
import pandas as pd
import torch

# 确保能包引入
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dual_tower_model import DualTowerBiddingModel

def generate_mock_test_data(num_samples=10, seq_dim=12, static_dim=4):
    """
    生成模拟的测试集特征与真实标签，用以验证输出偏差
    """
    # 特征
    x_seq = np.random.randn(num_samples, 96, seq_dim)
    x_static = np.random.randn(num_samples, static_dim)
    
    # 模拟真实的二次调频价格 (Target)
    y_true = np.random.rand(num_samples, 5) * 8 + 6
    
    # 时段限价设定
    # T1/T2/T5: [5, 10], T3/T4: [10, 15]
    p_min = np.tile([5., 5., 10., 10., 5.], (num_samples, 1))
    p_max = np.tile([10., 10., 15., 15., 10.], (num_samples, 1))
    
    return torch.tensor(x_seq, dtype=torch.float32), \
           torch.tensor(x_static, dtype=torch.float32), \
           torch.tensor(y_true, dtype=torch.float32), \
           torch.tensor(p_min, dtype=torch.float32), \
           torch.tensor(p_max, dtype=torch.float32)

def evaluate_bias():
    print(">>> 开始预测值输出偏差 (Bias) 测试评估...\n")
    
    # 定义及加载模型 (这里使用未训练初始化的权重作为演示计算逻辑)
    model = DualTowerBiddingModel(seq_input_dim=12, static_input_dim=4, hidden_dim=64)
    model.eval()
    
    # 1. 准备测试数据
    num_test_samples = 20
    xs, xst, y_true, pmin, pmax = generate_mock_test_data(num_samples=num_test_samples)
    
    # 2. 推理并限制在报价阈值之内
    with torch.no_grad():
        preds_clipped = model.predict(xs, xst, pmin, pmax)
        
    # 3. 计算各时段平均的残差 (Bias = 预测值 - 真实值)
    preds_np = preds_clipped.numpy()
    y_true_np = y_true.numpy()
    
    bias_matrix = preds_np - y_true_np
    
    # 各时段平均偏差 (Mean Bias)
    mean_bias = np.mean(bias_matrix, axis=0)
    # 各时段绝对偏差 (MAE)
    mae = np.mean(np.abs(bias_matrix), axis=0)
    
    # 打印报表
    segments = ['T1', 'T2', 'T3', 'T4', 'T5']
    results = []
    
    for i, seg in enumerate(segments):
        results.append({
            '时段 (Segment)': seg,
            '平均偏差 (Bias)': round(mean_bias[i], 3),
            '均绝对误差 (MAE)': round(mae[i], 3),
            '平均预测价': round(np.mean(preds_np[:, i]), 3),
            '平均真实价': round(np.mean(y_true_np[:, i]), 3)
        })
        
    df_metrics = pd.DataFrame(results)
    
    print("========== 各时段预测偏差评估结果 ==========")
    print(df_metrics.to_string(index=False))
    print("============================================\n")
    print("注：Bias = 预测价 - 真实价，负数表示预测偏低。由于此为未做大型训练的初始模型仅做计算演示，数值可能有较大随机性。")

if __name__ == '__main__':
    evaluate_bias()
