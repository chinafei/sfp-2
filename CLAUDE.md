# CLAUDE.md

## 项目概述

SFP-2 是电力二次调频预测服务，使用双塔深度学习模型 (DualTowerBiddingModel) 预测每日 5 个交易时段 (T1-T5) 的调频容量需求、边际排序价格、市场出清价格。通过 FastAPI 提供 REST API，支持 Docker 部署。

## 关键路径

- 服务入口: `app.py` (FastAPI, 端口 9527)
- 核心模型: `src/dual_tower_model.py` — DualTowerBiddingModel (双向GRU时序塔 + FC静态塔)
- 数据加载: `src/data_loader.py` — NPYDataLoader, 12维时序特征 + 4维静态特征
- 数据预处理: `src/data_preprocessing.py`
- NPY 构建: `scripts/build_npy_datasets.py` (原始CSV/Excel → NPY矩阵)
- 训练脚本: `scripts/train_and_tune.py` (Optuna 调参 + 训练)
- 预测脚本: `scripts/predict_only.py`
- 模型权重: `results/best_dual_tower_model.pt` 或 `checkpoints/best_dual_tower_model.pt`
- 模型参数: `results/params.json` (hidden_dim 等)
- 归一化统计: `results/normalizer.json`

## 技术栈

- Python 3.9, PyTorch 2.6, FastAPI, Optuna, NumPy, Pandas
- Docker (python:3.9-slim), Docker Compose
- 依赖清单: `requirements_docker.txt`

## 常用命令

```bash
# 构建 NPY 特征
python scripts/build_npy_datasets.py

# 训练模型
python scripts/train_and_tune.py

# 单日预测
python scripts/predict_only.py --date 2026-04-20

# 启动服务
uvicorn app:app --host 0.0.0.0 --port 9527

# Docker 部署
docker-compose up --build -d
```

## 核心业务逻辑

- 数据粒度: 96 点/天 (15分钟), 聚合为 5 个交易时段
- 时段划分: T1(0:00-4:48), T2(4:48-9:36), T3(9:36-14:24), T4(14:24-19:12), T5(19:12-24:00)
- 价格约束: T1/T2/T5 [5,10] 元/MW; T3/T4 [10,15] 元/MW
- 容量修正: 物理校正公式 = 平均负荷 × 0.122
- 损失函数: PenaltyBiddingLoss — MSE + 时段加权(T3/T4=1.5x) + 价格越界10x惩罚
- 归一化: Z-score (保存于 normalizer.json)

## 数据特征

- 时序输入 (96, 12): 日前出清电量(1d) + 非市场化出力(1d) + 检修容量(1d) + 负荷联络线(3d) + 输电通道(3d) + 新能源预测(3d)
- 静态输入 (4,): 日前正负备用需求(3d) + padding(1d)
- 标签 (5, 3): 5时段 × [容量需求, 边际价, 出清价]

## 代码规范

- 中文注释和文档
- 所有脚本使用 `conda activate tensorflow` 或直接 `python` 运行
- 预测结果 JSON 输出到 `results/prediction_{date}.json`
- 模型参数通过 `params.json` 管理，避免硬编码

## 注意事项

- `traindata/` 目录包含大文件 (>100MB CSV)，已从 git 历史移除，不要提交到 git
- `.gitignore` 应排除: traindata/, datasets/npy/, results/*.pt, checkpoints/
- Docker 构建上下文在上级目录 (`context: ../`)
- 服务启动时后台自动构建 NPY 并加载模型，约 30-60 秒就绪
