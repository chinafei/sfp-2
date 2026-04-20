# SFP-2 二次调频预测服务 API 文档

## 服务信息

- **服务地址**: `http://<服务器IP>:9527`
- **协议**: HTTP/REST
- **数据格式**: JSON (UTF-8)

---

## 接口列表

### 1. 健康检查

检查服务运行状态和模型加载情况。

**请求**

```
GET /health
```

**响应示例**

```json
{
  "status": "ok",
  "model_loaded": true,
  "startup_done": true,
  "startup_error": null,
  "hidden_dim": 128,
  "npy_base": "/app/sfp-2/datasets/npy",
  "results_dir": "/app/sfp-2/results"
}
```

**字段说明**

| 字段 | 类型 | 说明 |
|------|------|------|
| status | string | `"ok"` 表示服务就绪，`"loading"` 表示模型加载中 |
| model_loaded | bool | 模型是否已加载完成 |
| startup_done | bool | 启动流程是否全部完成 |
| startup_error | string/null | 启动错误信息，null 表示正常 |

---

### 2. 预测指定日期

预测指定交易日的二次调频指标（5个时段 × 3个指标）。

**请求**

```
POST /predict?date=YYYY-MM-DD
```

**参数**

| 参数 | 位置 | 必填 | 说明 | 示例 |
|------|------|------|------|------|
| date | query | 是 | 预测目标日期，格式 YYYY-MM-DD | 2026-03-05 |

**响应示例**

```json
{
  "date": "2026-03-05",
  "segments": {
    "T1": {
      "调频容量需求": 3521.234,
      "边际排序价格": 7.856,
      "市场出清价格_预测均价": 8.123
    },
    "T2": {
      "调频容量需求": 3412.567,
      "边际排序价格": 6.789,
      "市场出清价格_预测均价": 7.234
    },
    "T3": {
      "调频容量需求": 3298.901,
      "边际排序价格": 12.345,
      "市场出清价格_预测均价": 11.678
    },
    "T4": {
      "调频容量需求": 3156.789,
      "边际排序价格": 13.456,
      "市场出清价格_预测均价": 12.890
    },
    "T5": {
      "调频容量需求": 3089.012,
      "边际排序价格": 8.901,
      "市场出清价格_预测均价": 9.012
    }
  },
  "model_version": "hidden_dim=128",
  "generated_at": "2026-03-04T18:30:00.123456"
}
```

**segments 字段说明**

| 时段 | 对应时间 |
|------|----------|
| T1 | 第1时段 (00:00-04:48) |
| T2 | 第2时段 (04:48-09:36) |
| T3 | 第3时段 (09:36-14:24) |
| T4 | 第4时段 (14:24-19:12) |
| T5 | 第5时段 (19:12-24:00) |

| 指标 | 单位 | 说明 |
|------|------|------|
| 调频容量需求 | MW | 该时段预测的调频容量需求 |
| 边际排序价格 | 元/MW | 该时段调频边际排序价格 |
| 市场出清价格_预测均价 | 元/MW | 该时段市场出清预测均价 |

**价格范围约束**

| 时段 | 价格下限 | 价格上限 |
|------|----------|----------|
| T1, T2, T5 | 5.0 元/MW | 10.0 元/MW |
| T3, T4 | 10.0 元/MW | 15.0 元/MW |

**错误响应**

- `400` — 日期格式错误

```json
{"detail": "日期格式错误，应为 YYYY-MM-DD，实际: abc"}
```

- `503` — 服务启动中

```json
{"detail": {"message": "服务启动中，请稍后", "startup_error": null}}
```

- `500` — 预测失败（内部错误）

```json
{"detail": "错误详情..."}
```

---

### 3. 重建 NPY 数据集

当 traindata 目录挂载了新数据后，调用此接口触发 NPY 数据集增量重建。

**请求**

```
POST /datasets/rebuild
```

**响应示例**

```json
{
  "status": "ok",
  "message": "NPY 重建完成，模型已重新加载"
}
```

**说明**

- 重建为同步操作，耗时视数据量可能 1-5 分钟
- 重建完成后模型会自动重新加载
- 建议在更新 traindata 数据后调用

---

### 4. 数据集状态

查看当前 NPY 数据集的构建状态和覆盖日期范围。

**请求**

```
GET /datasets/status
```

**响应示例**

```json
{
  "ok": true,
  "missing": [],
  "warnings": [],
  "date_range": {
    "first": "2025-01-01",
    "last": "2026-03-04",
    "total_rows": 428
  }
}
```

**字段说明**

| 字段 | 类型 | 说明 |
|------|------|------|
| ok | bool | 所有必需数据集是否完整 |
| missing | array | 缺失的数据集名称列表 |
| date_range.first | string | 数据覆盖的最早日期 |
| date_range.last | string | 数据覆盖的最晚日期 |
| date_range.total_rows | int | 数据总行数（天数） |

---

### 5. 结果列表

列出所有已生成的预测结果文件。

**请求**

```
GET /results
```

**响应示例**

```json
{
  "count": 3,
  "files": [
    "prediction_2026-03-03.json",
    "prediction_2026-03-04.json",
    "prediction_2026-03-05.json"
  ]
}
```

---

### 6. 获取指定结果

读取某次预测的详细结果。

**请求**

```
GET /results/{filename}
```

**参数**

| 参数 | 位置 | 必填 | 说明 |
|------|------|------|------|
| filename | path | 是 | 结果文件名，如 `prediction_2026-03-05.json` |

**响应示例**

```json
{
  "date": "2026-03-05",
  "segments": {
    "T1": {
      "调频容量需求": 3521.234,
      "边际排序价格": 7.856,
      "市场出清价格(预测均价)": 8.123
    },
    "T2": { "..." : "..." },
    "T3": { "..." : "..." },
    "T4": { "..." : "..." },
    "T5": { "..." : "..." }
  }
}
```

---

## 调用示例

### cURL

```bash
# 健康检查
curl http://112.126.80.142:9527/health

# 预测 2026-03-05
curl -X POST "http://112.126.80.142:9527/predict?date=2026-03-05"

# 更新数据后重建
curl -X POST http://112.126.80.142:9527/datasets/rebuild

# 查看数据集状态
curl http://112.126.80.142:9527/datasets/status

# 查看结果列表
curl http://112.126.80.142:9527/results

# 获取某天结果
curl http://112.126.80.142:9527/results/prediction_2026-03-05.json
```

### Python

```python
import requests

BASE_URL = "http://112.126.80.142:9527"

# 预测
resp = requests.post(f"{BASE_URL}/predict", params={"date": "2026-03-05"})
data = resp.json()

for seg, vals in data["segments"].items():
    print(f"{seg}: 容量={vals['调频容量需求']:.1f}MW, "
          f"边际价={vals['边际排序价格']:.2f}元/MW, "
          f"均价={vals['市场出清价格_预测均价']:.2f}元/MW")
```

---

## 典型使用流程

1. **首次部署后** — 等待服务启动（约30秒），通过 `/health` 确认 `status: "ok"`
2. **每日预测** — 调用 `POST /predict?date=YYYY-MM-DD` 获取次日预测结果
3. **数据更新时** — 更新 traindata 目录内容后，调用 `POST /datasets/rebuild` 触发重建
4. **查询历史** — 通过 `/results` 查看所有已生成的预测结果
