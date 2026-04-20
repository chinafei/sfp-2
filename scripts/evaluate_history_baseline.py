#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评测历史强基线预测器，并和当前双塔模型结果对齐对比。
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

from src.history_baseline_predictor import (  # noqa: E402
    HistoryBaselinePredictor,
    TARGET_KEYS,
)


def parse_args():
    parser = argparse.ArgumentParser(description="评测历史强基线预测器")
    parser.add_argument(
        "--recent-days",
        type=int,
        default=30,
        help="最近窗口大小",
    )
    parser.add_argument(
        "--report-path",
        default="",
        help="输出 JSON 路径",
    )
    return parser.parse_args()


def load_targets() -> pd.DataFrame:
    ts_path = os.path.join(NPY_BASE, "03_二次调频报价", "timestamps.npy")
    data_path = os.path.join(NPY_BASE, "03_二次调频报价", "data.npy")
    ts = np.load(ts_path, allow_pickle=True)
    data = np.load(data_path)
    return pd.DataFrame(data[:, :3], index=pd.to_datetime(ts), columns=TARGET_KEYS)


def build_effective_days(df_targets: pd.DataFrame):
    daily_counts = df_targets.groupby(df_targets.index.date).size()
    candidate_days = sorted([d for d, c in daily_counts.items() if c == 5])
    result = []
    for day in candidate_days:
        y = df_targets.loc[day.strftime("%Y-%m-%d")].values[:5, :3]
        if not np.isnan(y).any():
            result.append(day)
    return result


def summarize(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not mask.any():
        return {"mae": None, "rmse": None, "bias": None, "mape_pct": None}
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    err = y_pred - y_true
    denom = np.clip(np.abs(y_true), 1e-6, None)
    return {
        "mae": round(float(np.mean(np.abs(err))), 3),
        "rmse": round(float(np.sqrt(np.mean(err ** 2))), 3),
        "bias": round(float(np.mean(err)), 3),
        "mape_pct": round(float(np.mean(np.abs(err) / denom) * 100.0), 2),
    }


def main():
    args = parse_args()
    targets = load_targets()
    days = build_effective_days(targets)
    predictor = HistoryBaselinePredictor(NPY_BASE)

    split_idx = int(len(days) * 0.8)
    val_days = days[split_idx:]
    recent_days = days[-min(args.recent_days, len(days)):]

    rows = []
    for day in days:
        day_str = day.strftime("%Y-%m-%d")
        y_true = targets.loc[day_str].values[:5, :3]
        y_pred = predictor.predict_matrix(day_str)
        rows.append(
            {
                "date": day_str,
                "split": "validation" if day in val_days else "train",
                "is_recent": day in recent_days,
                "true": y_true,
                "pred": y_pred,
            }
        )

    def flatten(block_rows):
        if not block_rows:
            return {
                "capacity": {"mae": None, "rmse": None, "bias": None, "mape_pct": None},
                "sort_price": {"mae": None, "rmse": None, "bias": None, "mape_pct": None},
                "clear_price": {"mae": None, "rmse": None, "bias": None, "mape_pct": None},
            }
        y_true = np.concatenate([r["true"] for r in block_rows], axis=0)
        y_pred = np.concatenate([r["pred"] for r in block_rows], axis=0)
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
        },
        "dataset": {
            "effective_days": len(days),
            "train_days": len(days[:split_idx]),
            "validation_days": len(val_days),
            "recent_days": len(recent_days),
        },
        "overall": flatten(rows),
        "validation_tail_20pct": flatten([r for r in rows if r["split"] == "validation"]),
        "recent_last_n_days": flatten([r for r in rows if r["is_recent"]]),
    }

    if args.report_path:
        output_path = args.report_path
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(RESULTS_DIR, f"history_baseline_eval_{stamp}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n[✓] 历史基线评测已写入: {output_path}")


if __name__ == "__main__":
    main()
