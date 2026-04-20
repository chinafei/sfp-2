#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按指标/时段部分标签评测历史强基线预测器。

不同于 evaluate_history_baseline.py 的完整三目标日期口径，本脚本会保留
2025-12 之后只有部分目标缺失的日期，并在每个指标上分别过滤 NaN。
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SFP2_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(SFP2_DIR, "results")
NPY_BASE = os.path.join(SFP2_DIR, "datasets", "npy")

sys.path.insert(0, SFP2_DIR)

from src.history_baseline_predictor import HistoryBaselinePredictor, TARGET_KEYS  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="部分标签口径评测历史强基线预测器")
    parser.add_argument("--recent-days", type=int, default=30)
    parser.add_argument("--report-path", default="")
    return parser.parse_args()


def load_targets() -> pd.DataFrame:
    ts_path = os.path.join(NPY_BASE, "03_二次调频报价", "timestamps.npy")
    data_path = os.path.join(NPY_BASE, "03_二次调频报价", "data.npy")
    ts = np.load(ts_path, allow_pickle=True)
    data = np.load(data_path)
    return pd.DataFrame(data[:, :3], index=pd.to_datetime(ts), columns=TARGET_KEYS)


def build_available_days(df_targets: pd.DataFrame):
    daily_counts = df_targets.groupby(df_targets.index.date).size()
    days = []
    for day, count in sorted(daily_counts.items()):
        if count != 5:
            continue
        target = df_targets.loc[day.strftime("%Y-%m-%d")].values[:5, :3]
        if not np.isnan(target).all():
            days.append(day)
    return days


def summarize(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not mask.any():
        return {"count": 0, "mae": None, "rmse": None, "bias": None, "mape_pct": None}
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    err = y_pred - y_true
    return {
        "count": int(mask.sum()),
        "mae": round(float(np.mean(np.abs(err))), 3),
        "rmse": round(float(np.sqrt(np.mean(err ** 2))), 3),
        "bias": round(float(np.mean(err)), 3),
        "mape_pct": round(float(np.mean(np.abs(err) / np.clip(np.abs(y_true), 1e-6, None)) * 100.0), 2),
    }


def main():
    args = parse_args()
    targets = load_targets()
    days = build_available_days(targets)
    predictor = HistoryBaselinePredictor(NPY_BASE)

    split_idx = int(len(days) * 0.8)
    val_days = set(days[split_idx:])
    recent_days = set(days[-min(args.recent_days, len(days)):])

    rows = []
    for day in days:
        day_str = day.strftime("%Y-%m-%d")
        rows.append(
            {
                "date": day_str,
                "split": "validation" if day in val_days else "train",
                "is_recent": day in recent_days,
                "true": targets.loc[day_str].values[:5, :3].astype(np.float32),
                "pred": predictor.predict_matrix(day_str),
            }
        )

    def flatten(block_rows):
        y_true = np.concatenate([row["true"] for row in block_rows], axis=0)
        y_pred = np.concatenate([row["pred"] for row in block_rows], axis=0)
        return {
            "capacity": summarize(y_true[:, 0], y_pred[:, 0]),
            "sort_price": summarize(y_true[:, 1], y_pred[:, 1]),
            "clear_price": summarize(y_true[:, 2], y_pred[:, 2]),
        }

    report = {
        "generated_at": datetime.now().isoformat(),
        "predictor": {
            "type": "HistoryBaselinePredictor",
            "strategies": predictor.strategies,
            "available_days_last": predictor.available_days[-1].strftime("%Y-%m-%d"),
            "complete_days_last": predictor.complete_days[-1].strftime("%Y-%m-%d"),
        },
        "dataset": {
            "available_days": len(days),
            "train_days": split_idx,
            "validation_days": len(days) - split_idx,
            "recent_days": len(recent_days),
            "first_day": days[0].strftime("%Y-%m-%d"),
            "last_day": days[-1].strftime("%Y-%m-%d"),
        },
        "overall": flatten(rows),
        "validation_tail_20pct": flatten([row for row in rows if row["split"] == "validation"]),
        "recent_last_n_days": flatten([row for row in rows if row["is_recent"]]),
    }

    if args.report_path:
        output_path = args.report_path
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(RESULTS_DIR, f"history_baseline_partial_eval_{stamp}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n[✓] 部分标签口径评测已写入: {output_path}")


if __name__ == "__main__":
    main()
