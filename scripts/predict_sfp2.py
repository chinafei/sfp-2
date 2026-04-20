#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SFP-2 统一入口脚本

用法（需在 llm-on-ray 环境下运行）:
  conda run -n llm-on-ray python sfp-2/scripts/predict_sfp2.py --date 2026-03-05

参数:
  --date         预测目标日期 (YYYY-MM-DD)。不指定则跳过预测。
  --build        构建/更新 NPY 数据集
  --no-incremental  禁用增量构建（全量重跑 NPY，默认增量）
  --train        执行训练（默认关闭）
  --predictor    预测后端: auto | history_baseline | dual_tower
  --epochs       全量训练 epoch 数 (default: 50)
  --patience     Early stopping patience (default: 5)
  --tune_trials  Optuna 调参 trial 数 (default: 15)
"""

import os
import sys
import json
import argparse
import subprocess
import datetime

# ── 工作目录 ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SFP2_DIR   = os.path.dirname(SCRIPT_DIR)
PROJECT_DIR = os.path.dirname(SFP2_DIR)
RESULTS_DIR = os.path.join(SFP2_DIR, "results")
NPY_BASE   = os.path.join(SFP2_DIR, "datasets", "npy")

sys.path.insert(0, SFP2_DIR)
from src.data_loader import (
    NPYDataLoader, SEQ_DIM, STATIC_DIM,
    SEGMENT_PMIN, SEGMENT_PMAX,
    load_model_params,
)
from src.history_baseline_predictor import HistoryBaselinePredictor

import torch

# ─────────────────────────────────────────────────────────
# 子步骤 1: 构建 NPY
# ─────────────────────────────────────────────────────────

def step_build(incremental: bool):
    build_script = os.path.join(SCRIPT_DIR, "build_npy_datasets.py")
    if not os.path.exists(build_script):
        print(f"[!] 构建脚本不存在: {build_script}")
        return False

    # 检查 openpyxl 是否可用（读取 xlsx 必需）
    try:
        import importlib
        importlib.import_module("openpyxl")
    except ImportError:
        print("[!] openpyxl 未安装，NPY 构建需要此依赖")
        print("    conda install -n llm-on-ray openpyxl -c conda-forge")
        print("    或: pip install openpyxl")
        print("    [跳过] NPY 构建（依赖缺失）")
        return True  # 不算失败，NPY 已存在即可继续

    print("\n" + "=" * 60)
    print(" [1/3] 构建 NPY 数据集" +
          (" (增量模式)" if incremental else " (全量模式)"))
    print("=" * 60)

    cmd = [sys.executable, build_script]
    if not incremental:
        cmd.append("--no-incremental")

    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    if result.returncode != 0:
        print(f"[!] build_npy_datasets.py 失败 (exit {result.returncode})")
        return False

    print("  [✓] NPY 数据集构建完成\n")
    return True


# ─────────────────────────────────────────────────────────
# 子步骤 2: 训练
# ─────────────────────────────────────────────────────────

def step_train(epochs: int, patience: int, tune_trials: int):
    train_script = os.path.join(SCRIPT_DIR, "train_and_tune.py")
    if not os.path.exists(train_script):
        print(f"[!] 训练脚本不存在: {train_script}")
        return False

    print("\n" + "=" * 60)
    print(" [2/3] 训练模型 (Optuna 调参 → 全量训练)")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    cmd = [
        sys.executable, train_script,
        "--epochs",      str(epochs),
        "--patience",    str(patience),
        "--tune_trials", str(tune_trials),
    ]
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    if result.returncode != 0:
        print(f"[!] 训练失败 (exit {result.returncode})")
        return False

    print("  [✓] 训练完成\n")
    return True


# ─────────────────────────────────────────────────────────
# 子步骤 3: 预测
# ─────────────────────────────────────────────────────────

def predict_with_dual_tower(loader, target_date_str: str):
    """使用 DualTower 预测。"""
    x_seq, x_static, warnings = loader.load_features_for_date(target_date_str)
    for w in warnings:
        print(w)

    params = load_model_params(RESULTS_DIR)
    hidden_dim = params.get("hidden_dim", 64)
    print(f"  [i] hidden_dim={hidden_dim} (来自 params.json)")

    from src.dual_tower_model import DualTowerBiddingModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualTowerBiddingModel(
        seq_input_dim=SEQ_DIM,
        static_input_dim=STATIC_DIM,
        hidden_dim=hidden_dim,
    ).to(device)

    model_path = os.path.join(RESULTS_DIR, "best_dual_tower_model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到预训练模型: {model_path}")

    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    xs_t = torch.tensor(x_seq, dtype=torch.float32).unsqueeze(0).to(device)
    xst_t = torch.tensor(x_static, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        out_scaled, _ = model(xs_t, xst_t)

    scales_t = torch.tensor([4000.0, 10.0, 10.0], dtype=torch.float32).to(device)
    out_phys = out_scaled * scales_t.view(1, 1, 3)

    pmin_t = torch.tensor(SEGMENT_PMIN, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    pmax_t = torch.tensor(SEGMENT_PMAX, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    out_clipped = out_phys.clone()
    out_clipped[..., 1:] = torch.max(
        torch.min(out_phys[..., 1:], pmax_t), pmin_t
    )
    out_np = out_clipped[0].cpu().numpy()

    segments = ["T1", "T2", "T3", "T4", "T5"]
    metric_keys = ["调频容量需求", "边际排序价格", "市场出清价格(预测均价)"]
    output = {"date": target_date_str, "segments": {}}

    for i, seg in enumerate(segments):
        output["segments"][seg] = {
            metric_keys[0]: round(float(out_np[i, 0]), 3),
            metric_keys[1]: round(float(out_np[i, 1]), 3),
            metric_keys[2]: round(float(out_np[i, 2]), 3),
        }

    return output, f"dual_tower(hidden_dim={hidden_dim})"


def predict_with_history_baseline(target_date_str: str):
    """使用历史强基线预测。"""
    predictor = HistoryBaselinePredictor(NPY_BASE)
    raw_output = predictor.predict(target_date_str)
    output = {"date": target_date_str, "segments": {}}

    for seg, values in raw_output["segments"].items():
        output["segments"][seg] = {
            "调频容量需求": values["调频容量需求"],
            "边际排序价格": values["边际排序价格"],
            "市场出清价格(预测均价)": values["市场出清价格_预测均价"],
        }

    return output, "history_baseline(prev_day_partial_fallback)"


def write_prediction_files(target_date_str: str, predictor_backend: str, output: dict) -> str:
    """写入兼容结果文件和后端专属结果文件。"""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    enriched_output = {
        "date": output["date"],
        "predictor": predictor_backend,
        "segments": output["segments"],
    }

    backend_json_path = os.path.join(
        RESULTS_DIR,
        f"prediction_{target_date_str}_{predictor_backend}.json",
    )
    with open(backend_json_path, "w", encoding="utf-8") as f:
        json.dump(enriched_output, f, ensure_ascii=False, indent=4)

    json_path = os.path.join(RESULTS_DIR, f"prediction_{target_date_str}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(enriched_output, f, ensure_ascii=False, indent=4)

    return json_path


def step_predict(target_date_str: str, predictor_backend: str):
    print("\n" + "=" * 60)
    print(f" [3/3] 预测 {target_date_str}")
    print("=" * 60)

    # 健康检查
    loader = NPYDataLoader(NPY_BASE)
    health = loader.check_health()
    if not health["ok"]:
        print(f"[!] 数据健康检查未通过，缺失: {health['missing']}")
        print("    请先运行 --build 补充数据")
        return False
    for w in health.get("warnings", []):
        print(f"  [!] {w}")

    if predictor_backend == "auto":
        predictor_backend = "history_baseline"
        print("  [i] predictor=history_baseline (默认强基线)")
    else:
        print(f"  [i] predictor={predictor_backend}")

    try:
        if predictor_backend == "history_baseline":
            output, model_version = predict_with_history_baseline(target_date_str)
        elif predictor_backend == "dual_tower":
            output, model_version = predict_with_dual_tower(loader, target_date_str)
        else:
            print(f"[!] 不支持的 predictor: {predictor_backend}")
            return False
    except FileNotFoundError as e:
        print(f"[!] {e}")
        print("    dual_tower 需要 results/best_dual_tower_model.pt")
        return False

    json_path = write_prediction_files(target_date_str, predictor_backend, output)

    print(f"\n  预测完成！结果存入: {json_path}")
    print(f"  [i] model_version={model_version}")
    print(json.dumps(output, ensure_ascii=False, indent=4))
    return True


# ─────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SFP-2 统一入口: 构建 NPY → 训练 → 预测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 纯预测
  conda run -n llm-on-ray python sfp-2/scripts/predict_sfp2.py --date 2026-03-05

  # 构建 + 预测
  conda run -n llm-on-ray python sfp-2/scripts/predict_sfp2.py --date 2026-03-05 --build

  # 构建 + 训练 + 预测
  conda run -n llm-on-ray python sfp-2/scripts/predict_sfp2.py --date 2026-03-05 --build --train

  # 只重建 + 训练
  conda run -n llm-on-ray python sfp-2/scripts/predict_sfp2.py --build --train --epochs 30
        """,
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="预测目标日期 (YYYY-MM-DD)。不指定则跳过预测。")
    parser.add_argument(
        "--build", action="store_true",
        help="构建/更新 NPY 数据集")
    parser.add_argument(
        "--no-incremental", dest="incremental", action="store_false", default=True,
        help="禁用增量构建（全量重跑 NPY）")
    parser.add_argument(
        "--train", action="store_true",
        help="执行训练（Optuna 调参 + 全量训练）。默认关闭。")
    parser.add_argument(
        "--predictor", type=str, default="auto",
        choices=["auto", "history_baseline", "dual_tower"],
        help="预测后端: auto | history_baseline | dual_tower")
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="全量训练 epoch 数 (default: 50)")
    parser.add_argument(
        "--patience", type=int, default=5,
        help="Early stopping patience (default: 5)")
    parser.add_argument(
        "--tune_trials", type=int, default=15,
        help="Optuna 调参 trial 数 (default: 15)")

    args = parser.parse_args()

    print("=" * 60)
    print(f" SFP-2 统一入口  ({datetime.datetime.now():%Y-%m-%d %H:%M:%S})")
    print("=" * 60)
    print(f"  --date       = {args.date}")
    print(f"  --build      = {args.build}")
    print(f"  --incremental= {args.incremental}")
    print(f"  --train      = {args.train}")
    print(f"  --predictor  = {args.predictor}")
    print(f"  --epochs     = {args.epochs}")
    print()

    # --train 自动触发 --build
    if args.train and not args.build:
        args.build = True
        print("  [i] --train 自动触发 --build\n")

    ok = True

    # 步骤 1: 构建
    if args.build:
        ok = step_build(incremental=args.incremental) and ok
    else:
        print("  [跳过] --build\n")

    # 步骤 2: 训练
    if args.train:
        ok = step_train(
            epochs=args.epochs,
            patience=args.patience,
            tune_trials=args.tune_trials,
        ) and ok
    else:
        print("  [跳过] --train（使用已有 checkpoint）\n")

    # 步骤 3: 预测
    if args.date:
        ok = step_predict(args.date, args.predictor) and ok
    else:
        print("  [跳过] --date 未指定，预测步骤结束")
        if not args.train:
            print("\n  提示: 指定 --date 进行预测")
            print("        指定 --train 进行训练")
            print("        指定 --build 构建 NPY 数据集")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
