import os
import sys
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class BiddingDataset(Dataset):
    def __init__(self, x_seq, x_static, y_targets, p_min, p_max):
        import torch
        self.x_seq = torch.tensor(x_seq, dtype=torch.float32)
        self.x_static = torch.tensor(x_static, dtype=torch.float32)
        self.y_targets = torch.tensor(y_targets, dtype=torch.float32)
        self.p_min = torch.tensor(p_min, dtype=torch.float32)
        self.p_max = torch.tensor(p_max, dtype=torch.float32)
        
    def __len__(self):
        return len(self.y_targets)
        
    def __getitem__(self, idx):
        return self.x_seq[idx], self.x_static[idx], self.y_targets[idx], self.p_min[idx], self.p_max[idx]

def load_real_data(workspace_dir):
    """
    Reads available actual numpy arrays from datasets/npy, aligns them structurally
    to (N, 96, seq_dim), static (N, static_dim), and targets (N, 5, 3), and creates Dataset blocks.
    """
    import logging
    npy_base = os.path.join(workspace_dir, "datasets", "npy")
    
    # 1. Load targets (03_二次调频报价) and identify valid D-days with 5 T-periods
    target_ts_path = os.path.join(npy_base, "03_二次调频报价", "timestamps.npy")
    target_d_path = os.path.join(npy_base, "03_二次调频报价", "data.npy")
    if not os.path.exists(target_d_path):
        logging.warning("No real datasets found. Falling back to simulated data.")
        return None, None, 12, 4
        
    ts_targets = np.load(target_ts_path, allow_pickle=True)
    d_targets = np.load(target_d_path)
    
    # index: [cap, marginal_p, clearing_p, ..., ...] we only need first 3
    df_y = pd.DataFrame(d_targets[:, :3], index=pd.to_datetime(ts_targets))
    
    # Filter days that have exactly 5 periods
    daily_counts = df_y.groupby(df_y.index.date).size()
    valid_days = daily_counts[daily_counts == 5].index.tolist()
    logging.info(f"Found {len(valid_days)} target days with complete 5 periods.")
    
    # Process Targets Arrays
    Y = np.zeros((len(valid_days), 5, 3))
    P_MIN = np.zeros((len(valid_days), 5))
    P_MAX = np.zeros((len(valid_days), 5))
    for i, date in enumerate(valid_days):
        day_str = date.strftime('%Y-%m-%d')
        y_slice = df_y.loc[day_str].values  # shape (5, 3)
        Y[i] = y_slice
        # 简化限价: T3(idx2), T4(idx3) -> [10, 15], 其它 [5, 10]
        P_MIN[i] = [5.0, 5.0, 10.0, 10.0, 5.0]
        P_MAX[i] = [10.0, 10.0, 15.0, 15.0, 10.0]
        
    # 2. Gather sequence features (Shape per file: ~43488, N) 
    # Match valid dates index to construct (N, 96, 12)
    seq_folders = [
        ("01_日前各时段出清现货电量", 1),
        ("05_非市场化机组出力", 1),
        ("06_检修总容量", 1),
        ("13_省内负荷及联络线情况", 3),
        ("14_输电通道可用容量", 3),
        ("15_新能源出力预测(场站申报)", 3)
    ]
    
    X_seq_list = []
    # Using 01 as master timestamp for sequences since 96 ticks per day
    master_ts_seq = np.load(os.path.join(npy_base, "01_日前各时段出清现货电量", "timestamps.npy"), allow_pickle=True)
    df_seq_ts = pd.to_datetime(master_ts_seq)
    
    for folder, dim in seq_folders:
        p = os.path.join(npy_base, folder, "data.npy")
        if os.path.exists(p):
            d = np.load(p)
            X_seq_list.append(d[:, :dim]) # force select first dim cols
        else:
            X_seq_list.append(np.zeros((len(df_seq_ts), dim)))
            
    X_seq_all = np.concatenate(X_seq_list, axis=1) # shape (~43488, 12)
    df_x_seq = pd.DataFrame(X_seq_all, index=df_seq_ts)
    
    X_SEQ = np.zeros((len(valid_days), 96, 12))
    for i, date in enumerate(valid_days):
        day_str = date.strftime('%Y-%m-%d')
        if day_str in df_x_seq.index:
            try:
                slice_val = df_x_seq.loc[day_str].values
                if len(slice_val) == 96:
                    X_SEQ[i] = slice_val
                    continue
            except Exception:
                pass
        # Fallback if missing or length not 96
        X_SEQ[i] = 0.0

    # Fill NaNs with 0 in X_SEQ and Y
    X_SEQ = np.nan_to_num(X_SEQ)
    Y = np.nan_to_num(Y)

    # 3. Static features (using 12_日前正负备用需求(3) 和 a mock dim=1 -> Total 4)
    X_STAT = np.zeros((len(valid_days), 4))
    try:
        ts_12 = np.load(os.path.join(npy_base, "12_日前正负备用需求", "timestamps.npy"), allow_pickle=True)
        d_12 = np.load(os.path.join(npy_base, "12_日前正负备用需求", "data.npy"))
        df_12 = pd.DataFrame(d_12[:, :3], index=pd.to_datetime(ts_12).date)
        for i, date in enumerate(valid_days):
            if date in df_12.index:
                X_STAT[i, :3] = df_12.loc[date].values
    except Exception:
        pass
    X_STAT = np.nan_to_num(X_STAT)

    # 4. Split Train / Validation (80% / 20%)
    split_idx = int(len(X_SEQ) * 0.8)
    
    train_dataset = BiddingDataset(X_SEQ[:split_idx], X_STAT[:split_idx], Y[:split_idx], P_MIN[:split_idx], P_MAX[:split_idx])
    val_dataset = BiddingDataset(X_SEQ[split_idx:], X_STAT[split_idx:], Y[split_idx:], P_MIN[split_idx:], P_MAX[split_idx:])
    
    logging.info(f"Real data mapped: {len(X_SEQ[:split_idx])} Train, {len(X_SEQ[split_idx:])} Val.")
    return train_dataset, val_dataset, 12, 4

