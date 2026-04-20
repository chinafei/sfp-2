import os
import sys
import json
import numpy as np
import pandas as pd
import torch

# 确保包引入
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dual_tower_model import DualTowerBiddingModel

def test_specific_date(target_date_str='2026-03-01'):
    workspace_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    npy_base = os.path.join(workspace_dir, "sfp-2", "datasets", "npy")
    
    target_date = pd.to_datetime(target_date_str).date()
    
    # 1. 加载真实的 Target (03_二次调频报价)
    ts_targets = np.load(os.path.join(npy_base, "03_二次调频报价", "timestamps.npy"), allow_pickle=True)
    d_targets = np.load(os.path.join(npy_base, "03_二次调频报价", "data.npy"))
    df_y = pd.DataFrame(d_targets[:, :3], index=pd.to_datetime(ts_targets))
    
    # 检查日期是否完整具有 5 个时段
    daily_counts = df_y.groupby(df_y.index.date).size()
    if target_date not in daily_counts or daily_counts[target_date] != 5:
        print(f"Error: 目标日期 {target_date_str} 在真实数据中不具备完整的 5 个时段。")
        sys.exit(1)
        
    y_true = df_y.loc[target_date_str].values # Shape: (5, 3)
    
    # 建立相应的限价规则区间
    pmin = np.array([5.0, 5.0, 10.0, 10.0, 5.0])
    pmax = np.array([10.0, 10.0, 15.0, 15.0, 10.0])
    
    # 2. 拉取对应的 96 个点时序特征 (12 维)
    seq_folders = [
        ("01_日前各时段出清现货电量", 1),
        ("05_非市场化机组出力", 1),
        ("06_检修总容量", 1),
        ("13_省内负荷及联络线情况", 3),
        ("14_输电通道可用容量", 3),
        ("15_新能源出力预测(场站申报)", 3)
    ]
    master_ts_seq = np.load(os.path.join(npy_base, "01_日前各时段出清现货电量", "timestamps.npy"), allow_pickle=True)
    df_seq_ts = pd.to_datetime(master_ts_seq)
    
    X_seq_list = []
    for folder, dim in seq_folders:
        p = os.path.join(npy_base, folder, "data.npy")
        if os.path.exists(p):
            d = np.load(p)
            X_seq_list.append(d[:, :dim])
        else:
            X_seq_list.append(np.zeros((len(df_seq_ts), dim)))
            
    X_seq_all = np.concatenate(X_seq_list, axis=1)
    df_x_seq = pd.DataFrame(X_seq_all, index=df_seq_ts)
    
    x_seq = np.zeros((96, 12))
    if target_date_str in df_x_seq.index:
        slice_val = df_x_seq.loc[target_date_str].values
        if len(slice_val) == 96:
            x_seq = slice_val
    x_seq = np.nan_to_num(x_seq)
    
    # 3. 拉取单点日特征 (4 维)
    x_stat = np.zeros(4)
    try:
        ts_12 = np.load(os.path.join(npy_base, "12_日前正负备用需求", "timestamps.npy"), allow_pickle=True)
        d_12 = np.load(os.path.join(npy_base, "12_日前正负备用需求", "data.npy"))
        df_12 = pd.DataFrame(d_12[:, :3], index=pd.to_datetime(ts_12).date)
        if target_date in df_12.index:
             x_stat[:3] = df_12.loc[target_date].values
    except Exception:
        pass
    x_stat = np.nan_to_num(x_stat)
    
    # 4. 初始化模型并行推断
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DualTowerBiddingModel(seq_input_dim=12, static_input_dim=4, hidden_dim=128).to(device)
    
    model_path = os.path.join(workspace_dir, "sfp-2", "results", "best_dual_tower_model.pt")
    if not os.path.exists(model_path):
        print(f"Error: 找不到预训练模型 {model_path}，请先执行训练。")
        sys.exit(1)
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    xs_t = torch.tensor(x_seq, dtype=torch.float32).unsqueeze(0).to(device)
    xst_t = torch.tensor(x_stat, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out_scaled, _ = model(xs_t, xst_t)
        
    # --- 物理值反归一化缩放与上下限钳位保护 ---
    target_scales = torch.tensor([4000.0, 10.0, 10.0], dtype=torch.float32).to(device)
    out_phys = out_scaled * target_scales.view(1, 1, 3)
    
    pmin_t = torch.tensor(pmin, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    pmax_t = torch.tensor(pmax, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    
    out_clipped = out_phys.clone()
    # 仅针对价签(索引1, 2)做保护
    out_clipped[..., 1:] = torch.max(torch.min(out_phys[..., 1:], pmax_t), pmin_t)
    out_np = out_clipped[0].cpu().numpy()
    
    # 5. 分析偏差与构建 JSON 输出
    segments = ['T1', 'T2', 'T3', 'T4', 'T5']
    # 按照需求，仅保留边际排序价格和市场出清价格（实际价格）的对比
    metric_indices = {
        '调频容量需求': 0,
        '边际排序价格': 1,
        '市场出清价格(实际数据中的中标均价)': 2
    }
    
    output_json = {
        "date": target_date_str,
        "segments": {}
    }
    
    print(f"\n==================== {target_date_str} 二次调频价格预测实际偏差对比 ====================")
    for i, seg in enumerate(segments):
        output_json["segments"][seg] = {}
        print(f"\n[{seg} 时段 (价签约束: [{int(pmin[i])}, {int(pmax[i])}])] : ")
        for metric_name, j in metric_indices.items():
            pred_v = float(out_np[i, j])
            if j == 0:
                # 根据真实业务规则将 96 点负荷数据划分为 5 个时段，使用 0.122 比例修正预测值
                bounds = [(0, 19), (19, 38), (38, 58), (58, 77), (77, 96)]
                start_i, end_i = bounds[i]
                avg_load_val = float(np.mean(x_seq[start_i:end_i, 4]))
                if avg_load_val > 0:
                    # 使用标准平均比例 12.2% 修正容量
                    pred_v = avg_load_val * 0.122
            true_v = float(y_true[i, j])
            bias = pred_v - true_v
            
            output_json["segments"][seg][metric_name] = {
                "实际值_true": round(true_v, 3),
                "预测值_pred": round(pred_v, 3),
                "偏差_bias": round(bias, 3)
            }
            # 格式化打印偏差
            bias_str = f"{bias:+.2f}"
            print(f"  - {metric_name}: 实际值={true_v:8.2f} | 预测值={pred_v:8.2f} | 偏差={bias_str:>8}")
            
    # 输出为 JSON
    results_dir = os.path.join(workspace_dir, "sfp-2", "results")
    json_path = os.path.join(results_dir, f"predict_{target_date_str}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, ensure_ascii=False, indent=4)
        
    print("\n================================================================================")
    print(f"✅ 预测结果已输出至 JSON 文件: {json_path}")

if __name__ == '__main__':
    test_specific_date('2026-03-01')