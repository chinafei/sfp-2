# SFP-2 二次调频预测服务

基于双塔深度学习模型的电力二次调频容量与价格预测系统，提供 REST API 服务，支持 Docker 容器化部署。

## 功能概述

系统针对每日 5 个交易时段 (T1-T5) 预测 3 项关键指标：

| 指标 | 单位 | 说明 |
|------|------|------|
| 调频容量需求 | MW | 该时段所需调频容量 |
| 边际排序价格 | 元/MW | 调频边际排序价格 |
| 市场出清价格 | 元/MW | 市场出清预测均价 |

**交易时段划分：**

| 时段 | 时间范围 | 价格区间 |
|------|----------|----------|
| T1 | 00:00 - 04:48 | 5 - 10 元/MW |
| T2 | 04:48 - 09:36 | 5 - 10 元/MW |
| T3 | 09:36 - 14:24 | 10 - 15 元/MW |
| T4 | 14:24 - 19:12 | 10 - 15 元/MW |
| T5 | 19:12 - 24:00 | 5 - 10 元/MW |

## 技术架构

```
原始台账 (traindata/)
    │
    ▼
NPY 特征矩阵 (datasets/npy/)   ← build_npy_datasets.py
    │
    ▼
双塔模型训练 (DualTowerBiddingModel)  ← train_and_tune.py + Optuna 自动调参
    │
    ▼
REST API 预测服务 (FastAPI)     ← app.py  端口 9527
    │
    ▼
JSON 预测结果 (results/)
```

**模型结构：**
- **Tower A (时序塔)**：双向 GRU 处理 96 点/天 (15分钟粒度) 的 12 维时序特征，带时间注意力机制 (偏重高负荷时段)
- **Tower B (静态塔)**：全连接网络处理 4 维日级静态特征
- **融合层**：拼接双塔输出 → FC → 输出 (5×3)

**技术栈：** PyTorch / FastAPI / Optuna / Docker

## 项目结构

```
sfp-2/
├── app.py                      # FastAPI 服务入口
├── src/
│   ├── dual_tower_model.py     # 双塔模型定义与损失函数
│   ├── data_loader.py          # NPY 数据加载与归一化
│   ├── data_preprocessing.py   # 数据对齐与特征工程
│   ├── model.py                # 备选 Transformer 模型
│   └── real_data_loader.py     # 实际数据加载工具
├── scripts/
│   ├── build_npy_datasets.py   # 原始数据 → NPY 特征矩阵
│   ├── train_and_tune.py       # Optuna 调参 + 模型训练
│   ├── predict_only.py         # 单日纯净预测
│   └── ...
├── datasets/npy/               # 构建后的 NPY 特征数据
├── results/                    # 模型权重与预测结果
├── checkpoints/                # 模型权重备份
├── Dockerfile                  # 容器镜像定义
├── docker-compose.yml          # 编排配置
├── API_DOC.md                  # API 接口文档
└── PROCESS_GUIDE.md            # 运维操作指南
```

## 快速开始

### 本地运行

```bash
# 安装依赖
pip install -r requirements_docker.txt

# 构建 NPY 特征矩阵 (首次或数据更新后)
python scripts/build_npy_datasets.py

# 训练模型 (Optuna 自动调参，50 epochs)
python scripts/train_and_tune.py

# 单日预测
python scripts/predict_only.py --date 2026-04-20

# 启动 API 服务
uvicorn app:app --host 0.0.0.0 --port 9527
```

### Docker 部署

```bash
# 构建并启动
docker-compose up --build -d

# 健康检查
curl http://localhost:9527/health

# 预测
curl -X POST "http://localhost:9527/predict?date=2026-04-20"

# 数据更新后重建 NPY
curl -X POST http://localhost:9527/datasets/rebuild
```

**Docker 卷挂载：**

| 容器路径 | 用途 | 权限 |
|----------|------|------|
| `/app/traindata` | 原始台账数据 | 只读 |
| `/app/sfp-2/checkpoints` | 模型权重 | 只读 |
| `/app/sfp-2/results` | 预测结果输出 | 读写 |
| `/app/sfp-2/datasets/npy` | NPY 缓存 (可选) | 读写 |

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/predict?date=YYYY-MM-DD` | 预测指定日期 |
| POST | `/datasets/rebuild` | 重建 NPY 数据集 |
| GET | `/datasets/status` | 数据集状态 |
| GET | `/results` | 结果列表 |
| GET | `/results/{filename}` | 获取指定结果 |

详见 [API_DOC.md](API_DOC.md)。

## 日常运维流程

1. 将最新台账数据放入 `traindata/` 目录
2. 执行 `python scripts/build_npy_datasets.py` (或调用 `POST /datasets/rebuild`)
3. 调用 `POST /predict?date=YYYY-MM-DD` 获取预测结果
4. 如模型漂移明显，执行 `python scripts/train_and_tune.py` 重新训练

详见 [PROCESS_GUIDE.md](PROCESS_GUIDE.md)。

## 环境要求

- Python 3.9+
- PyTorch 2.6+
- Docker & Docker Compose (容器化部署)
