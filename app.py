#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SFP-2 预测服务 API

Docker 中运行时:
  /app/
    traindata/          ← 宿主机挂载（只读）
    sfp-2/
      datasets/npy/    ← 容器内构建（增量），可挂载到宿主机持久化
      results/         ← 预测结果，可挂载到宿主机
      checkpoints/     ← 宿主机挂载（模型权重）
"""

import os
import sys
import json
import logging
import threading
from datetime import datetime, date
from typing import Optional

# ── 路径配置 ──────────────────────────────────────────────
APP_DIR   = os.path.dirname(os.path.abspath(__file__))
SFP2_DIR  = APP_DIR
NPY_BASE  = os.path.join(SFP2_DIR, "datasets", "npy")
RESULTS_DIR = os.path.join(SFP2_DIR, "results")
CHECKPOINT_DIR = os.path.join(SFP2_DIR, "checkpoints")

sys.path.insert(0, SFP2_DIR)

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import numpy as np
import torch

# ── 日志 ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── FastAPI 应用 ─────────────────────────────────────────
app = FastAPI(
    title="SFP-2 二次调频预测服务",
    description="基于 DualTower 模型预测调频容量需求、边际排序价格、市场出清均价",
    version="1.0.0",
)

# ── 全局状态 ──────────────────────────────────────────────
class State:
    model = None
    model_loaded = False
    model_params = {}
    data_loader = None
    hidden_dim = 64
    seq_dim = 12
    static_dim = 4
    startup_done = False
    startup_error = None

state = State()

# ── Pydantic 模型 ────────────────────────────────────────

class PredictionSegment(BaseModel):
    调频容量需求: float
    边际排序价格: float
    市场出清价格_预测均价: float

class PredictionResponse(BaseModel):
    date: str
    segments: dict
    model_version: str
    generated_at: str


# ── 内部: 构建 NPY（后台） ────────────────────────────────

def build_npy_background():
    """后台线程：启动时增量构建 NPY"""
    try:
        import subprocess
        script = os.path.join(APP_DIR, "scripts", "build_npy_datasets.py")
        if not os.path.exists(script):
            logger.warning(f"[NPY Build] 构建脚本不存在: {script}")
            return
        result = subprocess.run(
            [sys.executable, script],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0:
            logger.info("[NPY Build] NPY 构建完成")
        else:
            logger.warning(f"[NPY Build] 构建失败: {result.stderr[-500:]}")
    except Exception as e:
        logger.warning(f"[NPY Build] 构建跳过: {e}")


def build_npy_on_demand():
    """按需构建 NPY（当日数据刚挂载时调用）"""
    try:
        import subprocess
        script = os.path.join(APP_DIR, "scripts", "build_npy_datasets.py")
        result = subprocess.run(
            [sys.executable, script],
            capture_output=True, text=True, timeout=3600,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-500:])
        logger.info("[NPY Build] 按需构建完成")
    except Exception as e:
        logger.warning(f"[NPY Build] 按需构建失败: {e}")


# ── 内部: 加载模型 ────────────────────────────────────────

def load_model():
    """加载 DualTower 模型和参数"""
    from src.data_loader import load_model_params, SEQ_DIM, STATIC_DIM, NPYDataLoader
    from src.dual_tower_model import DualTowerBiddingModel

    logger.info("[Model] 加载中...")

    # 数据加载器健康检查
    loader = NPYDataLoader(NPY_BASE)
    health = loader.check_health()
    if not health["ok"]:
        raise RuntimeError(f"NPY 数据缺失: {health['missing']}")

    # 加载参数
    params_path = os.path.join(CHECKPOINT_DIR, "params.json")
    fallback_params = os.path.join(RESULTS_DIR, "params.json")

    if os.path.exists(params_path):
        with open(params_path) as f:
            params = json.load(f)
    elif os.path.exists(fallback_params):
        with open(fallback_params) as f:
            params = json.load(f)
    else:
        logger.warning("[Model] 未找到 params.json，使用默认值 hidden_dim=64")
        params = {"hidden_dim": 64}

    hidden_dim = params.get("hidden_dim", 64)
    state.hidden_dim = hidden_dim

    # 加载模型权重
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_dual_tower_model.pt")
    fallback_ckpt = os.path.join(RESULTS_DIR, "best_dual_tower_model.pt")

    if os.path.exists(checkpoint_path):
        ckpt = checkpoint_path
    elif os.path.exists(fallback_ckpt):
        ckpt = fallback_ckpt
    else:
        raise FileNotFoundError(
            "未找到模型文件 best_dual_tower_model.pt，"
            "请将模型文件放入 checkpoints/ 或 results/ 目录"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualTowerBiddingModel(
        seq_input_dim=SEQ_DIM,
        static_input_dim=STATIC_DIM,
        hidden_dim=hidden_dim,
    ).to(device)
    model.load_state_dict(
        torch.load(ckpt, map_location=device, weights_only=True)
    )
    model.eval()

    state.model = model
    state.model_loaded = True
    state.data_loader = loader
    state.model_params = params
    logger.info(f"[Model] 加载完成 (hidden_dim={hidden_dim}, device={device})")


# ── 内部: 执行预测 ────────────────────────────────────────

SEGMENT_PMIN = [5.0, 5.0, 10.0, 10.0, 5.0]
SEGMENT_PMAX = [10.0, 10.0, 15.0, 15.0, 10.0]


def do_predict(target_date_str: str):
    """执行单日预测，返回 dict"""
    if not state.model_loaded:
        raise HTTPException(503, "模型未加载，服务尚未就绪")

    loader = state.data_loader
    x_seq, x_static, warnings = loader.load_features_for_date(target_date_str)

    if warnings:
        for w in warnings:
            logger.warning(w)

    device = next(state.model.parameters()).device
    xs_t  = torch.tensor(x_seq,    dtype=torch.float32).unsqueeze(0).to(device)
    xst_t = torch.tensor(x_static, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        out_scaled, _ = state.model(xs_t, xst_t)

    # 物理值反归一化
    scales_t = torch.tensor([4000.0, 10.0, 10.0], dtype=torch.float32).to(device)
    out_phys = out_scaled * scales_t.view(1, 1, 3)

    # 价格钳位
    pmin_t = torch.tensor(SEGMENT_PMIN, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    pmax_t = torch.tensor(SEGMENT_PMAX, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    out_clipped = out_phys.clone()
    out_clipped[..., 1:] = torch.max(
        torch.min(out_phys[..., 1:], pmax_t), pmin_t
    )
    out_np = out_clipped[0].cpu().numpy()

    # 构建输出
    segments    = ["T1", "T2", "T3", "T4", "T5"]
    metric_keys = ["调频容量需求", "边际排序价格", "市场出清价格_预测均价"]

    result = {"date": target_date_str, "segments": {}}
    for i, seg in enumerate(segments):
        result["segments"][seg] = {
            metric_keys[0]: round(float(out_np[i, 0]), 3),
            metric_keys[1]: round(float(out_np[i, 1]), 3),
            metric_keys[2]: round(float(out_np[i, 2]), 3),
        }

    return result


# ── 启动事件 ──────────────────────────────────────────────

@app.on_event("startup")
def startup_event():
    """服务启动时：后台构建 NPY + 加载模型"""
    def background_tasks():
        try:
            # 后台构建 NPY（增量）
            build_npy_background()
        except Exception as e:
            logger.warning(f"[NPY] 后台构建跳过: {e}")

        try:
            # 加载模型
            load_model()
        except Exception as e:
            state.startup_error = str(e)
            logger.error(f"[Model] 加载失败: {e}")
            return

        state.startup_done = True
        logger.info("[Startup] 服务就绪")

    t = threading.Thread(target=background_tasks, daemon=True)
    t.start()


# ── API 路由 ──────────────────────────────────────────────

@app.get("/health")
def health_check():
    """健康检查"""
    return {
        "status": "ok" if state.model_loaded else "loading",
        "model_loaded": state.model_loaded,
        "startup_done": state.startup_done,
        "startup_error": state.startup_error,
        "hidden_dim": state.hidden_dim,
        "npy_base": NPY_BASE,
        "results_dir": RESULTS_DIR,
    }


@app.get("/datasets/status")
def datasets_status():
    """NPY 数据集状态"""
    loader = state.data_loader
    if loader is None:
        from src.data_loader import NPYDataLoader
        loader = NPYDataLoader(NPY_BASE)

    health = loader.check_health()

    # 尝试列出已构建的日期范围
    ts_path = os.path.join(NPY_BASE, "01_日前各时段出清现货电量", "timestamps.npy")
    date_range = None
    if os.path.exists(ts_path):
        ts = np.load(ts_path, allow_pickle=True)
        if len(ts) > 0:
            date_range = {
                "first": str(ts[0])[:10],
                "last":  str(ts[-1])[:10],
                "total_rows": int(len(ts)),
            }

    return {
        "ok": health["ok"],
        "missing": health["missing"],
        "warnings": health.get("warnings", []),
        "date_range": date_range,
    }


@app.post("/datasets/rebuild")
def datasets_rebuild():
    """手动触发 NPY 重建（当挂载了新数据时）"""
    try:
        build_npy_on_demand()
        # 重建后重新加载模型
        if state.model_loaded:
            state.model_loaded = False
            load_model()
        return {"status": "ok", "message": "NPY 重建完成，模型已重新加载"}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/predict")
def predict(
    date: str = Query(..., description="预测日期 YYYY-MM-DD", example="2026-03-05"),
):
    """
    预测指定日期的二次调频指标

    返回 T1~T5 时段的:
      - 调频容量需求 (MW)
      - 边际排序价格 (元/MW)
      - 市场出清价格/预测均价 (元/MW)
    """
    if not state.startup_done and not state.model_loaded:
        raise HTTPException(
            503,
            detail={
                "message": "服务启动中，请稍后",
                "startup_error": state.startup_error,
            },
        )

    # 验证日期格式
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(400, f"日期格式错误，应为 YYYY-MM-DD，实际: {date}")

    try:
        result = do_predict(date)

        # 写入结果文件（覆盖同名日期）
        os.makedirs(RESULTS_DIR, exist_ok=True)
        json_path = os.path.join(RESULTS_DIR, f"prediction_{date}.json")
        result_for_file = {
            "date": result["date"],
            "segments": {
                seg: {
                    "调频容量需求": vals["调频容量需求"],
                    "边际排序价格": vals["边际排序价格"],
                    "市场出清价格(预测均价)": vals["市场出清价格_预测均价"],
                }
                for seg, vals in result["segments"].items()
            },
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result_for_file, f, ensure_ascii=False, indent=4)

        logger.info(f"[Predict] {date} → {json_path}")

        return PredictionResponse(
            date=result["date"],
            segments={
                seg: PredictionSegment(**vals)
                for seg, vals in result["segments"].items()
            },
            model_version=f"hidden_dim={state.hidden_dim}",
            generated_at=datetime.now().isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Predict] {date} 失败: {e}")
        raise HTTPException(500, str(e))


@app.get("/results")
def list_results():
    """列出已生成的预测结果文件"""
    if not os.path.exists(RESULTS_DIR):
        return {"files": []}

    files = sorted([
        f for f in os.listdir(RESULTS_DIR)
        if f.startswith("prediction_") and f.endswith(".json")
    ])
    return {"count": len(files), "files": files}


@app.get("/results/{filename}")
def get_result(filename: str):
    """读取指定预测结果文件"""
    if ".." in filename or filename.startswith("/"):
        raise HTTPException(400, "无效的文件名")

    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(404, f"文件不存在: {filename}")

    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── 入口 ──────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
