import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch

# 确保包引入
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
from src.dual_tower_model import DualTowerBiddingModel

def predict_for_date(target_date_str):
    workspace_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) 
    npy_base = os.path.join(workspace_dir, "sfp-2", "datasets", "npy")
    
    # ---------------- 1. 加载 96 点时序特征 (12维) ----------------
    seq_folders = [
        ("01_日前各时段出清现货电量", 1),
        ("05_非市场化机组出力", 1),
        ("06_检修总容量", 1),
        ("13_省内负荷及联络线情况", 3),
        ("14_输电通道可用容量", 3),
        ("15_新能源出力预测_场站申报", 3)
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
    
    # 提取目标日期的特征
    x_seq = np.zeros((96, 12))
    if target_date_str in df_x_seq.index:
        slice_val = df_x_seq.loc[target_date_str].values
        if len(slice_val) == 96:
            x_seq = slice_val
        elif len(slice_val) > 96:
            x_seq = slice_val[:96]
    else:
        print(f"Warning: {target_date_str} 没有此时序特征，将使用默认0占位")
    x_seq = np.nan_to_num(x_seq)
    
    # ---------------- 2. 加载 静态特征 (4维) ----------------
    x_stat = np.zeros(4)
    try:
        ts_12 = np.load(os.path.join(npy_base, "12_日前正负备用需求", "timestamps.npy"), allow_pickle=True)
        d_12 = np.load(os.path.join(npy_base, "12_日前正负备用需求", "data.npy"))
        df_12 = pd.DataFrame(d_12[:, :3], index=pd.to_datetime(ts_12).date)
        
        target_date_obj = pd.to_datetime(target_date_str).date()
        if target_date_obj in df_12.index:
            x_stat[:3] = df_12.loc[target_date_obj].values
    except Exception:
        pass
    x_stat = np.nan_to_num(x_stat)

    # ---------------- 3. 模型加载与前向预测 ----------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DualTowerBiddingModel(seq_input_dim=12, static_input_dim=4, hidden_dim=128).to(device)
    model_path = os.path.join(workspace_dir, "sfp-2", "results", "best_dual_tower_model.pt")
    
    if not os.path.exists(model_path):
        print(f"Error: 找不到预训练模型 {model_path}，请先执行训练。")
        sys.exit(1)
        
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    xs_t = torch.tensor(x_seq, dtype=torch.float32).unsqueeze(0).to(device)
    xst_t = torch.tensor(x_stat, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out_scaled, _ = model(xs_t, xst_t)
        
    # 物理值反归一化缩放与上下限钳位保护
    target_scales = torch.tensor([4000.0, 10.0, 10.0], dtype=torch.float32).to(device)
    out_phys = out_scaled * target_scales.view(1, 1, 3)
    
    pmin = np.array([5.0, 5.0, 10.0, 10.0, 5.0])
    pmax = np.array([10.0, 10.0, 15.0, 15.0, 10.0])
    pmin_t = torch.tensor(pmin, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    pmax_t = torch.tensor(pmax, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    
    out_clipped = out_phys.clone()
    # 仅针对价签(索引1, 2)做保护
    out_clipped[..., 1:] = torch.max(torch.min(out_phys[..., 1:], pmax_t), pmin_t)
    out_np = out_clipped[0].cpu().numpy()
    
    # ---------------- 4. Json纯净数据构建与规则修正 ----------------
    segments = ['T1', 'T2', 'T3', 'T4', 'T5']
    metric_keys = ['调频容量需求', '边际排序价格', '市场出清价格(预测均价)']
    
    output_json = {
        "date": target_date_str,
        "segments": {}
    }
    
    # T1-T5 时段对应的96点索引分块，用于按标准比例提取特征负荷
    bounds = [(0, 19), (19, 38), (38, 58), (58, 77), (77, 96)]
    
    for i, seg in enumerate(segments):
        capacity = float(out_np[i, 0])
        sorting_price = float(out_np[i, 1])
        clearing_price = float(out_np[i, 2])
        
        # 物理修正逻辑：使用 96 点负荷数据划分为 5 个时段，利用 0.122 比例修正预测值
        start_i, end_i = bounds[i]
        # x_seq 中第 4 索引提取出的特征正好是“省内负荷预测电力值”
        avg_load_val = float(np.mean(x_seq[start_i:end_i, 4]))
        if avg_load_val > 0:
            capacity = avg_load_val * 0.122
            
        output_json["segments"][seg] = {
            metric_keys[0]: round(capacity, 3),
            metric_keys[1]: round(sorting_price, 3),
            metric_keys[2]: round(clearing_price, 3)
        }
        
    # 输出 JSON
    results_dir = os.path.join(workspace_dir, "sfp-2", "results")
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, f"prediction_only_{target_date_str}.json")
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, ensure_ascii=False, indent=4)
        
    print(f"预测分析已完成 (纯预测输出)！结果存入: {json_path}")
    print(json.dumps(output_json, ensure_ascii=False, indent=4))
    
    return json_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict Bidding Values for a Specific Date")
    parser.add_argument('--date', type=str, default='2026-03-05', help='Target date to predict, e.g. 2026-03-05')
    args = parser.parse_args()
    predict_for_date(args.date)
