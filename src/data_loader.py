"""
统一数据加载器 — 训练和预测共用同一份 NPY 读取逻辑。

修复的 bug:
  1. 目录名 "15_新能源出力预测_场站申报" -> "15_新能源出力预测(场站申报)"
  2. 静态特征 date index 类型不一致问题
  3. 96点序列特征缺失时静默回退为 0
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ─────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────

SEQ_FOLDERS = [
    ("01_日前各时段出清现货电量",  1),
    ("05_非市场化机组出力",         1),
    ("06_检修总容量",                 1),
    ("13_省内负荷及联络线情况",       3),
    ("14_输电通道可用容量",          3),
    ("15_新能源出力预测(场站申报)",  3),   # 注意: 括号不是下划线
]
SEQ_DIM  = 12   # 1+1+1+3+3+3
STATIC_DIM = 4   # 3 from dataset 12 + 1 padding

# 每个时段的 96点分块边界 (用于物理修正)
SEGMENT_BOUNDS = [(0, 19), (19, 38), (38, 58), (58, 77), (77, 96)]

# 各时段价格上下限
SEGMENT_PMIN = [5.0, 5.0, 10.0, 10.0, 5.0]
SEGMENT_PMAX = [10.0, 10.0, 15.0, 15.0, 10.0]

# 物理值反归一化缩放
TARGET_SCALES = np.array([4000.0, 10.0, 10.0])

# ─────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────

class BiddingDataset(Dataset):
    def __init__(self, x_seq, x_static, y_targets, p_min, p_max):
        self.x_seq    = torch.tensor(x_seq,    dtype=torch.float32)
        self.x_static = torch.tensor(x_static, dtype=torch.float32)
        self.y_targets = torch.tensor(y_targets, dtype=torch.float32)
        self.p_min    = torch.tensor(p_min,    dtype=torch.float32)
        self.p_max    = torch.tensor(p_max,    dtype=torch.float32)

    def __len__(self):
        return len(self.y_targets)

    def __getitem__(self, idx):
        return (self.x_seq[idx], self.x_static[idx],
                self.y_targets[idx], self.p_min[idx], self.p_max[idx])


# ─────────────────────────────────────────────────────────
# NPY 基础加载
# ─────────────────────────────────────────────────────────

class NPYDataLoader:
    """
    统一 NPY 数据加载类。提供:
      - load_features_for_date(): 预测单日所需的 (96, 12) + (4,) tensor
      - load_training_data():     训练所需的 (N, 96, 12) + (N, 4) + (N, 5, 3)
      - 检查目录完整性
    """

    def __init__(self, npy_base: str):
        self.npy_base = npy_base

    # ── 内部: 加载 96点序列特征 ──────────────────────────

    def _load_seq_features(self) -> pd.DataFrame:
        """返回 DataFrame，index=datetime，columns=(96, 12)"""
        master_ts_path = os.path.join(
            self.npy_base, "01_日前各时段出清现货电量", "timestamps.npy"
        )
        master_ts = np.load(master_ts_path, allow_pickle=True)
        df_ts = pd.to_datetime(master_ts)

        parts = []
        for folder, dim in SEQ_FOLDERS:
            p = os.path.join(self.npy_base, folder, "data.npy")
            if os.path.exists(p):
                d = np.load(p)
                parts.append(d[:, :dim])
            else:
                parts.append(np.zeros((len(df_ts), dim)))

        X_all = np.concatenate(parts, axis=1)  # (~43488, 12)
        return pd.DataFrame(X_all, index=df_ts)

    # ── 内部: 加载静态特征 ───────────────────────────────

    def _load_static_features(self) -> pd.DataFrame:
        """返回 DataFrame，index=date (datetime.date)，columns=(4,)"""
        ts_path = os.path.join(
            self.npy_base, "12_日前正负备用需求", "timestamps.npy"
        )
        d_path = os.path.join(
            self.npy_base, "12_日前正负备用需求", "data.npy"
        )
        if not os.path.exists(d_path):
            return pd.DataFrame(columns=["s0", "s1", "s2"])

        ts = np.load(ts_path, allow_pickle=True)
        d  = np.load(d_path)
        # index 统一为 date 对象
        df = pd.DataFrame(d[:, :3], index=pd.to_datetime(ts).date)
        df.columns = ["s0", "s1", "s2"]
        return df

    # ── 公开 API: 预测单日 ───────────────────────────────

    def load_features_for_date(self, target_date_str: str):
        """
        加载指定日期的特征。

        Returns:
            x_seq:    (96, SEQ_DIM) float32 array
            x_static: (STATIC_DIM,) float32 array
            warnings: list of str，描述数据缺失情况
        """
        warnings = []

        df_seq = self._load_seq_features()
        df_stat = self._load_static_features()

        # ── 96点序列 ────────────────────────────────────
        x_seq = np.zeros((96, SEQ_DIM), dtype=np.float32)
        target_dt = pd.to_datetime(target_date_str)

        if target_date_str in df_seq.index:
            slice_val = df_seq.loc[target_date_str].values
            if len(slice_val) == 96:
                x_seq = slice_val.astype(np.float32)
            elif len(slice_val) > 96:
                x_seq = slice_val[:96].astype(np.float32)
            else:
                warnings.append(
                    f"  [!] {target_date_str} 序列长度 {len(slice_val)} < 96，用 0 填充"
                )
        else:
            warnings.append(
                f"  [!] {target_date_str} 在 96点特征中不存在，用 0 填充"
            )

        x_seq = np.nan_to_num(x_seq)

        # ── 静态特征 ────────────────────────────────────
        x_static = np.zeros(STATIC_DIM, dtype=np.float32)
        target_date = target_dt.date()

        if target_date in df_stat.index:
            x_static[:3] = df_stat.loc[target_date].values[:3]
        else:
            warnings.append(
                f"  [!] {target_date} 在静态特征中不存在，用 0 填充"
            )

        x_static = np.nan_to_num(x_static)

        # ── 应用归一化 (如果有保存的 normalizer)
        norm_path = os.path.join(os.path.dirname(self.npy_base), "results", "normalizer.json")
        if not os.path.exists(norm_path):
            norm_path = os.path.join(os.path.dirname(self.npy_base), "checkpoints", "normalizer.json")
        if not os.path.exists(norm_path):
            # 尝试相对于 npy_base 的上两级 (sfp-2/datasets/npy → sfp-2/)
            sfp2_dir = os.path.dirname(os.path.dirname(self.npy_base))
            norm_path = os.path.join(sfp2_dir, "results", "normalizer.json")
            if not os.path.exists(norm_path):
                norm_path = os.path.join(sfp2_dir, "checkpoints", "normalizer.json")
        if os.path.exists(norm_path):
            import json as _json
            with open(norm_path) as f:
                norm = _json.load(f)
            seq_mean = np.array(norm["seq_mean"], dtype=np.float32)
            seq_std = np.array(norm["seq_std"], dtype=np.float32)
            stat_mean = np.array(norm["stat_mean"], dtype=np.float32)
            stat_std = np.array(norm["stat_std"], dtype=np.float32)
            x_seq = (x_seq - seq_mean) / seq_std
            x_static = (x_static - stat_mean) / stat_std

        return x_seq, x_static, warnings

    # ── 公开 API: 训练数据 ───────────────────────────────

    def load_training_data(self):
        """
        加载全部训练数据（供训练流程调用）。

        Returns:
            train_dataset, val_dataset, seq_dim, static_dim
        """
        import logging

        # 1. 加载目标 (03_二次调频报价)
        target_ts_path = os.path.join(
            self.npy_base, "03_二次调频报价", "timestamps.npy"
        )
        target_d_path = os.path.join(
            self.npy_base, "03_二次调频报价", "data.npy"
        )
        if not os.path.exists(target_d_path):
            logging.warning("数据集 03_二次调频报价 不存在，退回到模拟数据")
            return None, None, SEQ_DIM, STATIC_DIM

        ts_targets = np.load(target_ts_path, allow_pickle=True)
        d_targets = np.load(target_d_path)

        df_y = pd.DataFrame(d_targets[:, :3], index=pd.to_datetime(ts_targets))
        daily_counts = df_y.groupby(df_y.index.date).size()
        valid_days = daily_counts[daily_counts == 5].index.tolist()
        logging.info(f"找到 {len(valid_days)} 个有效的 5时段 目标日")

        Y = np.zeros((len(valid_days), 5, 3), dtype=np.float32)
        P_MIN = np.tile(SEGMENT_PMIN, (len(valid_days), 1)).astype(np.float32)
        P_MAX = np.tile(SEGMENT_PMAX, (len(valid_days), 1)).astype(np.float32)

        for i, date in enumerate(valid_days):
            day_str = date.strftime("%Y-%m-%d")
            Y[i] = df_y.loc[day_str].values[:5, :3]

        # 2. 加载 96点序列特征
        df_seq = self._load_seq_features()
        X_SEQ = np.zeros((len(valid_days), 96, SEQ_DIM), dtype=np.float32)

        for i, date in enumerate(valid_days):
            day_str = date.strftime("%Y-%m-%d")
            if day_str in df_seq.index:
                sv = df_seq.loc[day_str].values
                if len(sv) >= 96:
                    X_SEQ[i] = sv[:96]
        X_SEQ = np.nan_to_num(X_SEQ)

        # 特征标准化 (按列 z-score，避免大尺度特征主导)
        seq_shape = X_SEQ.shape  # (N, 96, 12)
        X_flat = X_SEQ.reshape(-1, seq_shape[-1])
        self._seq_mean = X_flat.mean(axis=0).astype(np.float32)
        self._seq_std = (X_flat.std(axis=0) + 1e-8).astype(np.float32)
        X_SEQ = ((X_SEQ - self._seq_mean) / self._seq_std).astype(np.float32)

        # 静态特征标准化
        # 3. 加载静态特征
        df_stat = self._load_static_features()
        X_STAT = np.zeros((len(valid_days), STATIC_DIM), dtype=np.float32)

        for i, date in enumerate(valid_days):
            if date in df_stat.index:
                X_STAT[i, :3] = df_stat.loc[date].values[:3]
        X_STAT = np.nan_to_num(X_STAT)
        self._stat_mean = X_STAT.mean(axis=0).astype(np.float32)
        self._stat_std = (X_STAT.std(axis=0) + 1e-8).astype(np.float32)
        X_STAT = ((X_STAT - self._stat_mean) / self._stat_std).astype(np.float32)

        # 4. 过滤含 NaN 的样本
        valid_mask = ~np.isnan(Y).reshape(len(Y), -1).any(axis=1)
        if not valid_mask.all():
            n_dropped = (~valid_mask).sum()
            logging.warning(f"过滤 {n_dropped} 个含 NaN 的目标样本")
            X_SEQ = X_SEQ[valid_mask]
            X_STAT = X_STAT[valid_mask]
            Y = Y[valid_mask]
            P_MIN = P_MIN[valid_mask]
            P_MAX = P_MAX[valid_mask]

        # 5. 切分 80/20
        split_idx = int(len(X_SEQ) * 0.8)
        train_ds = BiddingDataset(
            X_SEQ[:split_idx], X_STAT[:split_idx],
            Y[:split_idx], P_MIN[:split_idx], P_MAX[:split_idx]
        )
        val_ds = BiddingDataset(
            X_SEQ[split_idx:], X_STAT[split_idx:],
            Y[split_idx:], P_MIN[split_idx:], P_MAX[split_idx:]
        )
        logging.info(
            f"训练数据: {split_idx} 条, 验证数据: {len(X_SEQ) - split_idx} 条"
        )
        return train_ds, val_ds, SEQ_DIM, STATIC_DIM

    # ── 公开 API: 目录完整性检查 ─────────────────────────

    def check_health(self) -> dict:
        """返回数据健康检查报告"""
        report = {"ok": True, "missing": [], "warnings": []}

        for folder, dim in SEQ_FOLDERS:
            p = os.path.join(self.npy_base, folder, "data.npy")
            if not os.path.exists(p):
                report["ok"] = False
                report["missing"].append(f"{folder} (data.npy)")

        for folder in ["03_二次调频报价", "12_日前正负备用需求"]:
            p = os.path.join(self.npy_base, folder, "data.npy")
            if not os.path.exists(p):
                report["warnings"].append(f"{folder} (data.npy)")

        return report


# ─────────────────────────────────────────────────────────
# 参数持久化（训练时写，预测时读）
# ─────────────────────────────────────────────────────────

def save_model_params(results_dir: str, params: dict):
    """将模型参数（hidden_dim 等）写入 params.json"""
    path = os.path.join(results_dir, "params.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    print(f"  [✓] 模型参数 → {path}")


def save_normalizer(results_dir: str, loader: NPYDataLoader):
    """保存训练阶段计算出的特征标准化参数。"""
    required_attrs = ["_seq_mean", "_seq_std", "_stat_mean", "_stat_std"]
    missing = [attr for attr in required_attrs if not hasattr(loader, attr)]
    if missing:
        raise RuntimeError(f"标准化参数尚未生成，缺失: {missing}")

    path = os.path.join(results_dir, "normalizer.json")
    payload = {
        "seq_mean": loader._seq_mean.tolist(),
        "seq_std": loader._seq_std.tolist(),
        "stat_mean": loader._stat_mean.tolist(),
        "stat_std": loader._stat_std.tolist(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  [✓] 标准化参数 → {path}")


def load_model_params(results_dir: str) -> dict:
    """从 params.json 读取模型参数（缺失则返回默认值）"""
    path = os.path.join(results_dir, "params.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"hidden_dim": 64}  # 保守默认值
