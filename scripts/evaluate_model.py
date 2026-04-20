#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SFP-2 正式评测脚本

目标:
  1. 固化与当前 API 一致的离线评测口径
  2. 对比多个强基线，避免只看模型能否出数
  3. 输出可归档的 JSON 报告，支撑后续模型迭代

默认评测对象:
  - 当前 results/ 下的 best_dual_tower_model.pt
  - 当前 results/ 下的 params.json / normalizer.json

评测集合:
  - validation_tail_20pct: 与训练一致的时间顺序 80/20 尾部验证集
  - recent_last_n_days: 最近 N 个有效目标日，默认 30 天
  - overall: 全部有效样本

基线:
  - prev_day: 前一有效日同一时段
  - mean_7d: 最近 7 个有效日同一时段均值
  - same_weekday_recent4: 最近 4 个同星期几有效日同一时段均值
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SFP2_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(SFP2_DIR, "results")
NPY_BASE = os.path.join(SFP2_DIR, "datasets", "npy")

sys.path.insert(0, SFP2_DIR)

from src.data_loader import (  # noqa: E402
    NPYDataLoader,
    SEGMENT_BOUNDS,
    SEGMENT_PMIN,
    SEGMENT_PMAX,
    STATIC_DIM,
    SEQ_DIM,
)
from src.dual_tower_model import DualTowerBiddingModel  # noqa: E402


SEGMENTS = ["T1", "T2", "T3", "T4", "T5"]
TARGET_KEYS = ["capacity", "sort_price", "clear_price"]
TARGET_SCALES = np.array([4000.0, 10.0, 10.0], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="SFP-2 正式离线评测")
    parser.add_argument(
        "--model-path",
        default=os.path.join(RESULTS_DIR, "best_dual_tower_model.pt"),
        help="待评测模型路径",
    )
    parser.add_argument(
        "--params-path",
        default=os.path.join(RESULTS_DIR, "params.json"),
        help="模型参数路径",
    )
    parser.add_argument(
        "--recent-days",
        type=int,
        default=30,
        help="最近有效样本窗口大小",
    )
    parser.add_argument(
        "--report-path",
        default="",
        help="JSON 报告输出路径；留空则自动写入 results/",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="推理设备",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_hidden_dim(params_path: str) -> int:
    if not os.path.exists(params_path):
        return 64
    with open(params_path, encoding="utf-8") as f:
        params = json.load(f)
    return int(params.get("hidden_dim", 64))


def load_targets() -> pd.DataFrame:
    ts_path = os.path.join(NPY_BASE, "03_二次调频报价", "timestamps.npy")
    data_path = os.path.join(NPY_BASE, "03_二次调频报价", "data.npy")
    ts = np.load(ts_path, allow_pickle=True)
    data = np.load(data_path)
    return pd.DataFrame(data[:, :3], index=pd.to_datetime(ts), columns=TARGET_KEYS)


def build_effective_days(df_targets: pd.DataFrame) -> List[datetime.date]:
    daily_counts = df_targets.groupby(df_targets.index.date).size()
    candidate_days = sorted([d for d, count in daily_counts.items() if count == 5])

    effective_days = []
    for day in candidate_days:
        day_str = day.strftime("%Y-%m-%d")
        y_true = df_targets.loc[day_str].values[:5, :3]
        if not np.isnan(y_true).any():
            effective_days.append(day)
    return effective_days


def build_model(model_path: str, hidden_dim: int, device: torch.device) -> DualTowerBiddingModel:
    model = DualTowerBiddingModel(
        seq_input_dim=SEQ_DIM,
        static_input_dim=STATIC_DIM,
        hidden_dim=hidden_dim,
    ).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_one_day(
    model: DualTowerBiddingModel,
    loader: NPYDataLoader,
    day_str: str,
    device: torch.device,
) -> Dict:
    x_seq, x_static, warnings = loader.load_features_for_date(day_str)

    xs_t = torch.tensor(x_seq, dtype=torch.float32).unsqueeze(0).to(device)
    xst_t = torch.tensor(x_static, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        out_scaled, _ = model(xs_t, xst_t)

    out_phys = out_scaled[0].cpu().numpy() * TARGET_SCALES.reshape(1, 3)
    out_api = out_phys.copy()
    out_api[:, 1:] = np.clip(
        out_api[:, 1:],
        np.array(SEGMENT_PMIN, dtype=np.float32).reshape(5, 1),
        np.array(SEGMENT_PMAX, dtype=np.float32).reshape(5, 1),
    )

    out_rule = out_api.copy()
    for i, (start_idx, end_idx) in enumerate(SEGMENT_BOUNDS):
        avg_load = float(np.mean(x_seq[start_idx:end_idx, 4]))
        if avg_load > 0:
            out_rule[i, 0] = avg_load * 0.122

    return {
        "api_output": out_api,
        "raw_output": out_phys,
        "rule_capacity_output": out_rule,
        "warnings": warnings,
    }


def previous_effective_day_lookup(days: List[datetime.date]) -> Dict[str, Optional[str]]:
    lookup = {}
    for idx, day in enumerate(days):
        current = day.strftime("%Y-%m-%d")
        if idx == 0:
            lookup[current] = None
        else:
            lookup[current] = days[idx - 1].strftime("%Y-%m-%d")
    return lookup


def history_window_lookup(days: List[datetime.date], window_size: int) -> Dict[str, List[str]]:
    lookup = {}
    for idx, day in enumerate(days):
        start = max(0, idx - window_size)
        lookup[day.strftime("%Y-%m-%d")] = [
            d.strftime("%Y-%m-%d") for d in days[start:idx]
        ]
    return lookup


def same_weekday_lookup(days: List[datetime.date], max_history: int) -> Dict[str, List[str]]:
    lookup = {}
    history_by_weekday = defaultdict(list)
    for day in days:
        weekday = day.weekday()
        key = day.strftime("%Y-%m-%d")
        history = history_by_weekday[weekday][-max_history:]
        lookup[key] = [d.strftime("%Y-%m-%d") for d in history]
        history_by_weekday[weekday].append(day)
    return lookup


def safe_stats(values: np.ndarray) -> Dict[str, Optional[float]]:
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        return {"mean": None, "std": None}
    return {
        "mean": round(float(valid.mean()), 3),
        "std": round(float(valid.std()), 3),
    }


def summarize_metrics(df: pd.DataFrame, true_col: str, pred_col: str) -> Dict:
    valid = df[[true_col, pred_col]].dropna()
    if valid.empty:
        return {
            "count": 0,
            "mae": None,
            "rmse": None,
            "bias": None,
            "mape_pct": None,
            "smape_pct": None,
        }

    error = valid[pred_col] - valid[true_col]
    denom = np.clip(np.abs(valid[true_col].to_numpy()), 1e-6, None)
    smape_denom = np.clip(
        np.abs(valid[pred_col].to_numpy()) + np.abs(valid[true_col].to_numpy()),
        1e-6,
        None,
    )

    return {
        "count": int(len(valid)),
        "mae": round(float(np.mean(np.abs(error))), 3),
        "rmse": round(float(math.sqrt(np.mean(np.square(error)))), 3),
        "bias": round(float(np.mean(error)), 3),
        "mape_pct": round(float(np.mean(np.abs(error) / denom) * 100.0), 2),
        "smape_pct": round(float(np.mean(2.0 * np.abs(error) / smape_denom) * 100.0), 2),
    }


def build_metric_block(df: pd.DataFrame) -> Dict:
    return {
        "capacity": summarize_metrics(df, "true_capacity", "pred_capacity"),
        "sort_price": summarize_metrics(df, "true_sort_price", "pred_sort_price"),
        "clear_price": summarize_metrics(df, "true_clear_price", "pred_clear_price"),
        "baseline_prev_day_capacity": summarize_metrics(df, "true_capacity", "base_prev_capacity"),
        "baseline_prev_day_sort_price": summarize_metrics(df, "true_sort_price", "base_prev_sort_price"),
        "baseline_prev_day_clear_price": summarize_metrics(df, "true_clear_price", "base_prev_clear_price"),
        "baseline_mean_7d_capacity": summarize_metrics(df, "true_capacity", "base_7d_capacity"),
        "baseline_mean_7d_sort_price": summarize_metrics(df, "true_sort_price", "base_7d_sort_price"),
        "baseline_mean_7d_clear_price": summarize_metrics(df, "true_clear_price", "base_7d_clear_price"),
        "baseline_same_weekday_recent4_capacity": summarize_metrics(df, "true_capacity", "base_weekday_capacity"),
        "baseline_same_weekday_recent4_sort_price": summarize_metrics(df, "true_sort_price", "base_weekday_sort_price"),
        "baseline_same_weekday_recent4_clear_price": summarize_metrics(df, "true_clear_price", "base_weekday_clear_price"),
        "capacity_rule_0122": summarize_metrics(df, "true_capacity", "pred_rule_capacity"),
    }


def average_history(day_values: Dict[str, np.ndarray], history_days: List[str]) -> np.ndarray:
    if not history_days:
        return np.full((5, 3), np.nan, dtype=np.float32)
    stacks = [day_values[day] for day in history_days if day in day_values]
    if not stacks:
        return np.full((5, 3), np.nan, dtype=np.float32)
    return np.nanmean(np.stack(stacks, axis=0), axis=0)


def collect_rows(
    model: DualTowerBiddingModel,
    loader: NPYDataLoader,
    targets_df: pd.DataFrame,
    days: List[datetime.date],
    recent_days: List[datetime.date],
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    split_idx = int(len(days) * 0.8)
    val_days = set(d.strftime("%Y-%m-%d") for d in days[split_idx:])
    recent_day_keys = set(d.strftime("%Y-%m-%d") for d in recent_days)

    target_cache = {}
    for day in days:
        day_str = day.strftime("%Y-%m-%d")
        target_cache[day_str] = targets_df.loc[day_str].values[:5, :3].astype(np.float32)

    prev_lookup = previous_effective_day_lookup(days)
    hist_7d_lookup = history_window_lookup(days, 7)
    weekday_lookup = same_weekday_lookup(days, 4)

    for day in days:
        day_str = day.strftime("%Y-%m-%d")
        pred = predict_one_day(model, loader, day_str, device)
        y_true = target_cache[day_str]

        prev_target = average_history(target_cache, [prev_lookup[day_str]] if prev_lookup[day_str] else [])
        hist_7d_target = average_history(target_cache, hist_7d_lookup[day_str])
        weekday_target = average_history(target_cache, weekday_lookup[day_str])

        for idx, segment in enumerate(SEGMENTS):
            rows.append(
                {
                    "date": day_str,
                    "segment": segment,
                    "split": "validation" if day_str in val_days else "train",
                    "is_recent": day_str in recent_day_keys,
                    "warning_count": len(pred["warnings"]),
                    "true_capacity": float(y_true[idx, 0]),
                    "true_sort_price": float(y_true[idx, 1]),
                    "true_clear_price": float(y_true[idx, 2]),
                    "pred_capacity": float(pred["api_output"][idx, 0]),
                    "pred_sort_price": float(pred["api_output"][idx, 1]),
                    "pred_clear_price": float(pred["api_output"][idx, 2]),
                    "pred_raw_capacity": float(pred["raw_output"][idx, 0]),
                    "pred_raw_sort_price": float(pred["raw_output"][idx, 1]),
                    "pred_raw_clear_price": float(pred["raw_output"][idx, 2]),
                    "pred_rule_capacity": float(pred["rule_capacity_output"][idx, 0]),
                    "base_prev_capacity": float(prev_target[idx, 0]),
                    "base_prev_sort_price": float(prev_target[idx, 1]),
                    "base_prev_clear_price": float(prev_target[idx, 2]),
                    "base_7d_capacity": float(hist_7d_target[idx, 0]),
                    "base_7d_sort_price": float(hist_7d_target[idx, 1]),
                    "base_7d_clear_price": float(hist_7d_target[idx, 2]),
                    "base_weekday_capacity": float(weekday_target[idx, 0]),
                    "base_weekday_sort_price": float(weekday_target[idx, 1]),
                    "base_weekday_clear_price": float(weekday_target[idx, 2]),
                }
            )

    return pd.DataFrame(rows)


def build_price_diagnostics(df: pd.DataFrame) -> Dict:
    out = {}
    for segment in SEGMENTS:
        sub = df[df["segment"] == segment]
        lower = 10.0 if segment in ("T3", "T4") else 5.0
        upper = 15.0 if segment in ("T3", "T4") else 10.0
        out[segment] = {
            "sort_at_lower_pct": round(float(np.mean(np.isclose(sub["pred_sort_price"], lower)) * 100.0), 2),
            "sort_at_upper_pct": round(float(np.mean(np.isclose(sub["pred_sort_price"], upper)) * 100.0), 2),
            "clear_at_lower_pct": round(float(np.mean(np.isclose(sub["pred_clear_price"], lower)) * 100.0), 2),
            "clear_at_upper_pct": round(float(np.mean(np.isclose(sub["pred_clear_price"], upper)) * 100.0), 2),
        }
    return out


def build_distribution_block(df: pd.DataFrame) -> Dict:
    return {
        "capacity": {
            "true": safe_stats(df["true_capacity"].to_numpy()),
            "pred": safe_stats(df["pred_capacity"].to_numpy()),
        },
        "sort_price": {
            "true": safe_stats(df["true_sort_price"].to_numpy()),
            "pred": safe_stats(df["pred_sort_price"].to_numpy()),
        },
        "clear_price": {
            "true": safe_stats(df["true_clear_price"].to_numpy()),
            "pred": safe_stats(df["pred_clear_price"].to_numpy()),
        },
    }


def build_sample_days(df: pd.DataFrame, sample_days: List[str]) -> List[Dict]:
    samples = []
    for day in sample_days:
        sub = df[df["date"] == day]
        if sub.empty:
            continue
        samples.append(
            {
                "date": day,
                "capacity_mae": summarize_metrics(sub, "true_capacity", "pred_capacity")["mae"],
                "sort_price_mae": summarize_metrics(sub, "true_sort_price", "pred_sort_price")["mae"],
                "clear_price_mae": summarize_metrics(sub, "true_clear_price", "pred_clear_price")["mae"],
            }
        )
    return samples


def build_report(df: pd.DataFrame, days: List[datetime.date], recent_days: List[datetime.date], args) -> Dict:
    validation_df = df[df["split"] == "validation"].copy()
    recent_day_keys = set(d.strftime("%Y-%m-%d") for d in recent_days)
    recent_df = df[df["date"].isin(recent_day_keys)].copy()
    raw_price = df[["pred_raw_sort_price", "pred_raw_clear_price"]].to_numpy()
    clipped_price = df[["pred_sort_price", "pred_clear_price"]].to_numpy()
    clipped_points = int((~np.isclose(raw_price, clipped_price)).sum())
    total_price_points = int(raw_price.size)

    report = {
        "generated_at": datetime.now().isoformat(),
        "model": {
            "model_path": os.path.abspath(args.model_path),
            "params_path": os.path.abspath(args.params_path),
            "recent_days": args.recent_days,
        },
        "dataset": {
            "effective_days": len(days),
            "train_days": int(len(days) * 0.8),
            "validation_days": len(days) - int(len(days) * 0.8),
            "first_day": days[0].strftime("%Y-%m-%d"),
            "last_day": days[-1].strftime("%Y-%m-%d"),
            "recent_window_first_day": recent_days[0].strftime("%Y-%m-%d"),
            "recent_window_last_day": recent_days[-1].strftime("%Y-%m-%d"),
        },
        "diagnostics": {
            "days_with_feature_warnings": int((df.groupby("date")["warning_count"].max() > 0).sum()),
            "price_clip_points": clipped_points,
            "price_total_points": total_price_points,
            "price_clip_rate_pct": round(float(clipped_points / max(total_price_points, 1) * 100.0), 2),
            "validation_price_saturation_pct": build_price_diagnostics(validation_df),
            "recent_price_saturation_pct": build_price_diagnostics(recent_df),
            "validation_distribution": build_distribution_block(validation_df),
            "recent_distribution": build_distribution_block(recent_df),
        },
        "overall": build_metric_block(df),
        "validation_tail_20pct": build_metric_block(validation_df),
        "recent_last_n_days": build_metric_block(recent_df),
        "validation_by_segment": {},
        "recent_daily_samples": build_sample_days(
            recent_df,
            sorted(recent_day_keys)[-min(5, len(recent_day_keys)):],
        ),
    }

    for segment in SEGMENTS:
        sub = validation_df[validation_df["segment"] == segment].copy()
        report["validation_by_segment"][segment] = {
            "capacity": summarize_metrics(sub, "true_capacity", "pred_capacity"),
            "sort_price": summarize_metrics(sub, "true_sort_price", "pred_sort_price"),
            "clear_price": summarize_metrics(sub, "true_clear_price", "pred_clear_price"),
        }

    return report


def main():
    args = parse_args()
    device = resolve_device(args.device)
    hidden_dim = load_hidden_dim(args.params_path)

    loader = NPYDataLoader(NPY_BASE)
    targets_df = load_targets()
    days = build_effective_days(targets_df)
    if not days:
        raise RuntimeError("没有找到可用于评测的有效目标日")

    recent_days = days[-min(args.recent_days, len(days)):]
    model = build_model(args.model_path, hidden_dim, device)

    df = collect_rows(model, loader, targets_df, days, recent_days, device)
    report = build_report(df, days, recent_days, args)

    if args.report_path:
        output_path = args.report_path
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(RESULTS_DIR, f"evaluation_{stamp}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n[✓] 评测报告已写入: {output_path}")


if __name__ == "__main__":
    main()
