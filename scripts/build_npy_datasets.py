"""
数据处理主脚本 — 将 traindata 下 CSV 宽表按照「数据处理说明.xlsx」规则
转为结构化 npy 数据集，同时输出完整映射表。

输出目录: datasets/npy/
    ├── 01_日前各时段出清现货电量/        # 按序号+名称建目录
    │     ├── data.npy                    # float32 时序矩阵 (n_timestamps, n_features)
    │     └── meta.json                    # 数据标识 & 列映射
    ├── ...
    ├── station_mapping.csv               # 电站标准映射（带编号）
    ├── node_mapping.csv                  # 节点 / 母线映射
    ├── section_mapping.csv               # 断面映射
    ├── dataset_registry.json             # 全局数据集注册表（28类汇总）
    └── column_index_map.json             # 所有列名 → 全局 ID

运行:
    python scripts/data_processing/build_npy_datasets.py
"""

import os, sys, json, glob, re, warnings
from pathlib import Path
from collections import OrderedDict
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def infer_time_alignment_mode(page: str, time_attr: str, df: pd.DataFrame) -> str:
    text = f"{page}|{time_attr}"

    if "二次调频" in text:
        return "5point"
    if ("96" in text) or ("实时" in text and "时段" in text):
        return "96point"

    if "timestamp" in df.columns and len(df) > 0:
        dt = pd.to_datetime(df["timestamp"], errors="coerce")
        dt = dt.dropna()
        if len(dt) > 0:
            per_day = dt.dt.floor("D").value_counts()
            median_points = float(per_day.median()) if len(per_day) > 0 else 0.0
            if median_points >= 48:
                return "96point"
            if 4 <= median_points <= 12:
                return "5point"
            if median_points <= 3:
                return "daily"

    keywords_96 = [
        "节点边际电价", "联络线", "电网运行实际值", "机组实际发电曲线", "出清电价",
        "断面约束情况", "重要通道", "非市场化机组出力", "输电通道可用容量", "省内负荷",
    ]
    if any(k in text for k in keywords_96):
        return "96point"
    if ("日" in text) or ("每日" in text):
        return "daily"
    return "none"


def align_timeseries_for_npy(df: pd.DataFrame, feature_cols: List[str], mode: str) -> Tuple[pd.DataFrame, Dict]:
    info = {
        "mode": mode,
        "before_rows": int(len(df)),
        "after_rows": int(len(df)),
        "missing_before": 0,
        "missing_after": 0,
    }

    if "timestamp" not in df.columns or mode == "none":
        return df, info

    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work = work.dropna(subset=["timestamp"]).sort_values("timestamp")
    work = work.drop_duplicates(subset=["timestamp"], keep="last")

    if work.empty:
        info["after_rows"] = 0
        return work, info

    info["missing_before"] = int(work[feature_cols].isna().sum().sum()) if feature_cols else 0

    if mode == "daily":
        work["_date"] = work["timestamp"].dt.floor("D")
        daily = work.sort_values("timestamp").groupby("_date").tail(1).set_index("_date")
        full_dates = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
        daily = daily.reindex(full_dates)
        if feature_cols:
            daily[feature_cols] = daily[feature_cols].ffill().bfill()
        aligned = daily.reset_index().rename(columns={"index": "_anchor_date"})
        aligned["timestamp"] = pd.to_datetime(aligned["_anchor_date"]).dt.strftime("%Y-%m-%d 00:00:00")
        aligned = aligned.drop(columns=[c for c in ["_anchor_date"] if c in aligned.columns])
    elif mode == "96point":
        work["timestamp"] = work["timestamp"].dt.floor("15min")
        work = work.drop_duplicates(subset=["timestamp"], keep="last")
        work = work.set_index("timestamp").sort_index()

        # 构建完整的 15min 时间索引，只覆盖数据中实际出现的天
        existing_days = work.index.floor("D").unique().sort_values()
        full_index_parts = [
            pd.date_range(day, day + pd.Timedelta(hours=23, minutes=45), freq="15min")
            for day in existing_days
        ]
        full_index = full_index_parts[0].append(full_index_parts[1:]) if len(full_index_parts) > 1 else full_index_parts[0]
        work = work.reindex(full_index)

        # 向量化插值: 一次性对所有列做线性插值 + 前后填充
        if feature_cols:
            work[feature_cols] = work[feature_cols].interpolate(
                method="linear", limit_direction="both", axis=0
            )
            work[feature_cols] = work[feature_cols].ffill().bfill()

        aligned = work.reset_index().rename(columns={"index": "timestamp"})
        aligned["timestamp"] = aligned["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        aligned = work.copy()
        aligned["timestamp"] = pd.to_datetime(aligned["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    aligned = aligned[["timestamp"] + feature_cols]
    info["after_rows"] = int(len(aligned))
    info["missing_after"] = int(aligned[feature_cols].isna().sum().sum()) if feature_cols else 0
    return aligned, info

# ─────────────── 路径配置 ───────────────
SFP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SFP_DIR.parent
TRAIN_DIR = Path(os.environ.get("SFP_TRAINDATA_ROOT", str(PROJECT_ROOT / "traindata"))).resolve()
OUT_DIR = Path(os.environ.get("SFP_NPY_ROOT", str(SFP_DIR / "datasets" / "npy"))).resolve()
PERIOD_DIRS = sorted(
    [p for p in TRAIN_DIR.iterdir()
     if p.is_dir() and re.match(r"20\d{2}", p.name)]
)

# ─────────────── 数据处理说明 ───────────────
XLSX_PATH = TRAIN_DIR / "数据处理说明.xlsx"
STATION_CSV = TRAIN_DIR / "station_mapping_dataset.csv"

# ─────────────── 文件夹名与 xlsx「爬取页面」对应关系 ───────────────
# key = 子文件夹名(两个时段下通用),  value = xlsx 中「爬取页面」字段
FOLDER_TO_PAGE = {
    "日前各时段出清现货电量": "日前各时段出清现货电量",
    "日前节点边际电价": "日前节点边际电价",
    "二次调频出清结果": "二次调频报价",
    "断面约束": "断面约束",
    "非市场化机组出力": "非市场化机组出力",
    "检修总容量": "检修总容量",
    "开机不满72小时机组": "开机不满72小时机组",
    "日前备用总量": "日前备用总量",
    "日前必开必停机组": "日前必开必停机组",
    "日前机组开机安排": "日前机组开机安排",
    "日前联络线计划": "日前联络线计划",
    "日前正负备用需求": "日前正负备用需求",
    "省内负荷及联络线情况": "省内负荷及联络线情况",
    "输电通道可用容量": "输电通道可用容量",
    "新能源出力预测(场站申报）": "新能源出力预测(场站申报)",
    "96点电网运行实际值": "96点电网运行实际值",
    "出清电价": "出清电价",
    "断面约束情况及影子价格": "断面约束情况及影子价格",
    "机组实际发电曲线": "机组实际发电曲线",
    "实时备用总量": "实时备用总量",
    "实时各时段出清现货电量": "实时各时段出清现货电量",
    "实时节点边际电价": "实时节点边际电价",
    "实时联络线出力": "实时联络线出力",
    "实时输电断面约束及阻塞": "实时输电断面约束及阻塞",
    "输变电设备检修计划": "输变电设备检修计划",
    "重要通道实际输电情况": "重要通道实际输电情况",
    "抽蓄电站水位": "抽蓄电站水位",
    "节点分配因子": "节点分配因子",
}


# ═══════════════════════════════════════════════════════
#  1. 读取处理说明
# ═══════════════════════════════════════════════════════
def load_processing_rules() -> pd.DataFrame:
    """返回 DataFrame: 序号, 爬取页面, 数据属性, 时间属性, 处理机制, 应用"""
    df = pd.read_excel(str(XLSX_PATH), sheet_name="Sheet1")
    df["序号"] = df["序号"].astype(int)
    return df


# ═══════════════════════════════════════════════════════
#  2. 电站标准映射表
# ═══════════════════════════════════════════════════════
def build_station_mapping() -> pd.DataFrame:
    """读取 station_mapping_dataset.csv，构建标准映射表并输出。"""
    df = pd.read_csv(str(STATION_CSV))
    # 基础字段保留 & 增加数值编号
    df = df.sort_values("source_index").reset_index(drop=True)
    df["numeric_id"] = range(len(df))  # 0‑based 连续编号

    # 提取电站级聚合（同电站不同机组归并）
    station_group = (
        df.groupby("station_name")
        .agg(
            unit_count=("unit_name", "count"),
            energy_types=("energy_type", lambda x: ",".join(sorted(set(x)))),
            is_thermal=("is_thermal_unit", "any"),
            numeric_ids=("numeric_id", list),
        )
        .reset_index()
    )
    station_group["station_numeric_id"] = range(len(station_group))

    out_path = OUT_DIR / "station_mapping.csv"
    df.to_csv(str(out_path), index=False, encoding="utf-8-sig")
    print(f"  [✓] 电站映射表 → {out_path}  ({len(df)} 机组)")

    out_path2 = OUT_DIR / "station_group_mapping.csv"
    station_group.to_csv(str(out_path2), index=False, encoding="utf-8-sig")
    print(f"  [✓] 电站分组表 → {out_path2}  ({len(station_group)} 电站)")

    return df


# ═══════════════════════════════════════════════════════
#  3. 节点 / 断面 / 全局列映射
# ═══════════════════════════════════════════════════════
class ColumnIndexer:
    """全局列名 → 唯一 ID 注册器"""

    def __init__(self):
        self._map: OrderedDict = OrderedDict()
        self._counter = 0

    def register(self, col_name: str) -> int:
        if col_name not in self._map:
            self._map[col_name] = self._counter
            self._counter += 1
        return self._map[col_name]

    def register_many(self, cols):
        return [self.register(c) for c in cols]

    def save(self, path):
        with open(str(path), "w", encoding="utf-8") as f:
            json.dump(self._map, f, ensure_ascii=False, indent=2)
        print(f"  [✓] 全局列索引 → {path}  ({len(self._map)} 列)")


# ═══════════════════════════════════════════════════════
#  4. 读取 & 合并两个时段的 CSV
# ═══════════════════════════════════════════════════════
def _read_csv_if_has_timestamp(csv_path: str):
    """读取单个 CSV，若含 timestamp 列则返回 df，否则 None。"""
    try:
        df = pd.read_csv(csv_path)
        if "timestamp" in df.columns:
            return df
    except Exception:
        pass
    return None


def load_csvs_for_folder(folder_name: str) -> pd.DataFrame:
    """在 PERIOD_DIRS 里找同名子目录，读取所有 csv 合并。

    优化策略:
    - 对横向合并场景(大量 part 文件各有不同列): 采用流式策略,
      先扫描所有文件获取列名和时间索引, 预分配 numpy 数组,
      再逐 part 填充, 避免 DataFrame 级别的 merge/concat.
    - 使用多进程并行读取 CSV 文件加速 I/O.
    """
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    import gc

    # 收集所有待处理的 CSV 文件路径
    all_files = []
    for pd_dir in PERIOD_DIRS:
        sub = pd_dir / folder_name
        if not sub.exists():
            continue
        files = sorted(sub.glob("*.csv"))
        all_files.extend(files)

    if not all_files:
        return pd.DataFrame()

    # ── Phase 1: 并行读取所有 CSV (使用线程池)
    parts = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_read_csv_if_has_timestamp, str(c)) for c in all_files]
        for fut in futures:
            result = fut.result()
            if result is not None:
                parts.append(result)

    if not parts:
        return pd.DataFrame()

    # ── Phase 2: 判断合并策略
    ref_cols = set(parts[0].columns)
    all_same_cols = all(set(p.columns) == ref_cols for p in parts)

    if all_same_cols:
        # 纯纵向拼接 (同结构)
        result = pd.concat(parts, ignore_index=True)
        del parts
        gc.collect()
    else:
        # 横向合并场景 — 使用流式 numpy 填充
        # Step 1: 扫描所有列名和全局时间索引
        all_feature_cols = []  # 保持有序, 去重
        seen_feature_cols = set()
        all_timestamps = set()

        for p in parts:
            if "timestamp" in p.columns:
                all_timestamps.update(p["timestamp"].values)
            for c in p.columns:
                if c != "timestamp" and c not in seen_feature_cols:
                    all_feature_cols.append(c)
                    seen_feature_cols.add(c)

        # Step 2: 构建时间索引映射
        sorted_timestamps = sorted(all_timestamps)
        ts_to_idx = {ts: i for i, ts in enumerate(sorted_timestamps)}
        n_rows = len(sorted_timestamps)
        n_cols = len(all_feature_cols)
        col_to_idx = {c: i for i, c in enumerate(all_feature_cols)}

        # Step 3: 预分配 numpy 数组, 用 NaN 填充
        arr = np.full((n_rows, n_cols), np.nan, dtype=np.float32)

        # Step 4: 逐 part 填充数组后立即释放 DataFrame
        for p in parts:
            if "timestamp" not in p.columns:
                continue
            ts_vals = p["timestamp"].values
            feature_cols_in_part = [c for c in p.columns if c != "timestamp"]

            # 获取行索引映射
            row_indices = np.array([ts_to_idx.get(t, -1) for t in ts_vals], dtype=np.int64)
            valid_mask = row_indices >= 0

            # 获取列索引映射
            col_indices = [col_to_idx[c] for c in feature_cols_in_part if c in col_to_idx]
            part_col_names = [c for c in feature_cols_in_part if c in col_to_idx]

            if not col_indices or not valid_mask.any():
                continue

            # 批量填充
            valid_rows = row_indices[valid_mask]
            data_block = p[part_col_names].values[valid_mask].astype(np.float32)
            arr[np.ix_(valid_rows, col_indices)] = data_block

        del parts
        gc.collect()

        # Step 5: 转回 DataFrame (仅用于后续 align_timeseries_for_npy 兼容)
        result = pd.DataFrame(arr, columns=all_feature_cols)
        result.insert(0, "timestamp", sorted_timestamps)
        del arr
        gc.collect()

    # ── 去重
    if "timestamp" in result.columns:
        result = result.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    return result


# ═══════════════════════════════════════════════════════
#  5. 按类别构建 mapping data (非时序类)
# ═══════════════════════════════════════════════════════
def load_mapping_csvs(folder_name: str) -> pd.DataFrame:
    """读取映射类数据(断面约束, 必开必停, 开机安排)，多 csv 纵拼。"""
    frames = []
    for pd_dir in PERIOD_DIRS:
        sub = pd_dir / folder_name
        if not sub.exists():
            continue
        for c in sorted(sub.glob("*.csv")):
            try:
                part = pd.read_csv(str(c))
                part["_source_file"] = c.stem
                frames.append(part)
            except Exception:
                pass
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════
#  6. 主处理逻辑
# ═══════════════════════════════════════════════════════
def process_all(incremental=True):
    print("=" * 60)
    print(" 数据处理流水线 — 开始" +
          (" (增量模式)" if incremental else " (全量模式)"))
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 增量模式: 收集已有的数据集目录
    existing_ids = set()
    if incremental:
        for d in OUT_DIR.iterdir():
            if d.is_dir() and d.name[0].isdigit():
                try:
                    existing_ids.add(int(d.name.split("_", 1)[0]))
                except ValueError:
                    pass
        if existing_ids:
            print(f"\n  [增量] 已存在 {len(existing_ids)} 个数据集，"
                  f"将跳过已有目录\n")

    # 6.1 读取规则
    rules = load_processing_rules()
    print(f"\n[1] 处理规则: {len(rules)} 条")

    # 6.2 电站映射
    print("\n[2] 构建电站映射表")
    station_df = build_station_mapping()

    # 6.3 准备全局索引器
    col_indexer = ColumnIndexer()

    # 6.4  节点映射 & 断面映射 (从数据列名提取)
    node_names = set()   # 收集所有节点名(从节点边际电价、节点分配因子列名)
    section_names = set()  # 收集断面名

    # 6.5 逐类处理
    dataset_registry = []
    alignment_reports = []

    # ---- 映射类数据(非数值时序) ----
    MAPPING_FOLDERS = {"断面约束", "日前必开必停机组", "日前机组开机安排"}

    # ---- 特殊: 开机不满72小时 (文本列表) ----
    TEXT_FOLDERS = {"开机不满72小时机组"}

    print("\n[3] 逐类数据处理...")
    for _, row in rules.iterrows():
        seq = int(row["序号"])
        page = row["爬取页面"]
        data_attr = row["数据属性"]
        time_attr = row["时间属性"]
        mechanism = row["处理机制"]
        usage = row["应用"]

        # 查找对应文件夹名
        folder_name = None
        for k, v in FOLDER_TO_PAGE.items():
            if v == page:
                folder_name = k
                break
        if folder_name is None:
            print(f"  [!] 序号{seq} '{page}' — 未找到对应文件夹, 跳过")
            continue

        dir_label = f"{seq:02d}_{page}"
        out_sub = OUT_DIR / dir_label

        # 增量跳过: 如果目录已存在且有 data.npy，跳过处理
        if incremental and out_sub.exists() and (out_sub / "data.npy").exists():
            print(f"  [{seq:02d}] {page}  (跳过，已存在)")
            dataset_registry.append({
                "dataset_id": seq,
                "name": page,
                "dir": dir_label,
                "data_attr": data_attr,
                "time_attr": time_attr,
                "mechanism": mechanism,
                "usage": usage,
            })
            continue

        out_sub.mkdir(parents=True, exist_ok=True)

        # ---- 分支处理 ----
        if folder_name in MAPPING_FOLDERS:
            # 映射/分类数据 → 转 category codes
            df = load_mapping_csvs(folder_name)
            if df.empty:
                print(f"  [{seq:02d}] {page} — 无数据")
                continue

            # 构建值映射
            value_maps = {}
            feature_cols = [c for c in df.columns if c not in ("_source_file",)]
            for c in feature_cols:
                uniq = sorted(df[c].dropna().unique(), key=str)
                vmap = {str(v): i for i, v in enumerate(uniq)}
                value_maps[c] = vmap
                df[c + "_code"] = df[c].astype(str).map(vmap)

            code_cols = [c for c in df.columns if c.endswith("_code")]
            arr = df[code_cols].to_numpy(dtype=np.float32)
            np.save(str(out_sub / "data.npy"), arr)

            meta = {
                "dataset_id": seq,
                "name": page,
                "data_attr": data_attr,
                "time_attr": time_attr,
                "mechanism": mechanism,
                "usage": usage,
                "type": "mapping",
                "shape": list(arr.shape),
                "columns": feature_cols,
                "value_maps": value_maps,
            }
            with open(str(out_sub / "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            print(f"  [{seq:02d}] {page}  mapping  shape={arr.shape}")

        elif folder_name in TEXT_FOLDERS:
            # 开机不满72小时 → 文本标记
            df = load_csvs_for_folder(folder_name)
            if df.empty:
                print(f"  [{seq:02d}] {page} — 无数据")
                continue

            # 将机组名映射到 station_mapping 编号
            unit_set = sorted(df["机组名称"].dropna().unique())
            unit_map = {}
            for u in unit_set:
                match = station_df.loc[station_df["unit_name"] == u, "numeric_id"]
                unit_map[u] = int(match.iloc[0]) if len(match) > 0 else -1

            # 按 timestamp 编码为 one-hot style (timestamp × n_units)
            timestamps = sorted(df["timestamp"].unique()) if "timestamp" in df.columns else []
            if timestamps:
                ts_idx = {t: i for i, t in enumerate(timestamps)}
                arr = np.zeros((len(timestamps), len(unit_set)), dtype=np.float32)
                for _, r in df.iterrows():
                    ti = ts_idx.get(r.get("timestamp"))
                    ui = unit_set.index(r["机组名称"]) if r["机组名称"] in unit_set else None
                    if ti is not None and ui is not None:
                        arr[ti, ui] = 1.0
                np.save(str(out_sub / "data.npy"), arr)
            else:
                arr = np.array(list(unit_map.values()), dtype=np.int32)
                np.save(str(out_sub / "data.npy"), arr)

            meta = {
                "dataset_id": seq,
                "name": page,
                "data_attr": data_attr,
                "time_attr": time_attr,
                "mechanism": mechanism,
                "usage": usage,
                "type": "text_flag",
                "shape": list(arr.shape),
                "unit_list": unit_set,
                "unit_to_station_id": unit_map,
            }
            with open(str(out_sub / "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            print(f"  [{seq:02d}] {page}  text_flag  shape={arr.shape}")

        else:
            # ---- 常规时序数据 ----
            df = load_csvs_for_folder(folder_name)
            if df.empty:
                print(f"  [{seq:02d}] {page} — 无数据")
                continue

            # timestamp 列分离
            ts_col = None
            if "timestamp" in df.columns:
                ts_col = df["timestamp"].values
                feature_cols = [c for c in df.columns if c != "timestamp"]
            else:
                feature_cols = list(df.columns)

            # 特殊口径: 03_二次调频报价在模型中固定为 6 维
            # 线上新批次常带“时段/slot”辅助列，需在入模前剔除。
            if seq == 3:
                slot_like_cols = [
                    c for c in feature_cols
                    if ("时段" in str(c)) or ("slot" in str(c).lower())
                ]
                if slot_like_cols:
                    feature_cols = [c for c in feature_cols if c not in slot_like_cols]

                if len(feature_cols) > 6:
                    feature_cols = feature_cols[:6]
                elif len(feature_cols) < 6:
                    print(f"  [!] 03_二次调频报价 特征列不足6列，当前={len(feature_cols)}")

            # 注册到全局列索引
            col_indexer.register_many(feature_cols)

            alignment_info = None
            if "timestamp" in df.columns:
                mode = infer_time_alignment_mode(page, str(time_attr), df)
                df, alignment_info = align_timeseries_for_npy(df, feature_cols, mode)
                ts_col = df["timestamp"].values
                alignment_reports.append({
                    "dataset_id": seq,
                    "dataset": page,
                    "mode": alignment_info["mode"],
                    "before_rows": alignment_info["before_rows"],
                    "after_rows": alignment_info["after_rows"],
                    "missing_before": alignment_info["missing_before"],
                    "missing_after": alignment_info["missing_after"],
                })

            # 收集节点 / 断面名
            if page in ("日前节点边际电价", "实时节点边际电价", "节点分配因子"):
                node_names.update(feature_cols)
            if page in ("断面约束情况及影子价格", "实时输电断面约束及阻塞"):
                section_names.update(feature_cols)

            # 尝试匹配列名到 station_mapping
            col_station_map = {}
            for c in feature_cols:
                match = station_df.loc[station_df["unit_name"] == c, "numeric_id"]
                if len(match) > 0:
                    col_station_map[c] = int(match.iloc[0])

            # 转 numpy
            arr = df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
            np.save(str(out_sub / "data.npy"), arr)

            # 保存 timestamp 索引
            if ts_col is not None:
                np.save(str(out_sub / "timestamps.npy"), ts_col)

            meta = {
                "dataset_id": seq,
                "name": page,
                "data_attr": data_attr,
                "time_attr": time_attr,
                "mechanism": mechanism,
                "usage": usage,
                "type": "timeseries",
                "shape": list(arr.shape),
                "n_features": len(feature_cols),
                "columns": feature_cols,
                "col_to_station_id": col_station_map if col_station_map else None,
                "alignment": alignment_info,
            }
            with open(str(out_sub / "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            print(f"  [{seq:02d}] {page}  timeseries  shape={arr.shape}")

        dataset_registry.append({
            "dataset_id": seq,
            "name": page,
            "dir": dir_label,
            "data_attr": data_attr,
            "time_attr": time_attr,
            "mechanism": mechanism,
            "usage": usage,
        })

    # ═══════ 节点映射表 ═══════
    print("\n[4] 构建节点映射表")
    node_list = sorted(node_names)
    node_df = pd.DataFrame({
        "node_id": range(len(node_list)),
        "node_name": node_list,
    })
    node_df.to_csv(str(OUT_DIR / "node_mapping.csv"), index=False, encoding="utf-8-sig")
    print(f"  [✓] node_mapping.csv  ({len(node_df)} 节点)")

    # ═══════ 断面映射表 ═══════
    print("\n[5] 构建断面映射表")
    # 补充从「断面约束」文件夹读取断面描述
    sec_desc_frames = []
    for pd_dir in PERIOD_DIRS:
        sub = pd_dir / "断面约束"
        if not sub.exists():
            continue
        for c in sorted(sub.glob("*.csv")):
            try:
                sec_desc_frames.append(pd.read_csv(str(c)))
            except Exception:
                pass
    if sec_desc_frames:
        sec_desc = pd.concat(sec_desc_frames, ignore_index=True).drop_duplicates()
    else:
        sec_desc = pd.DataFrame(columns=["断面名称", "断面描述"])

    section_list = sorted(section_names)
    section_df = pd.DataFrame({
        "section_id": range(len(section_list)),
        "section_name": section_list,
    })
    # 尝试 join 描述
    if "断面名称" in sec_desc.columns:
        section_df = section_df.merge(
            sec_desc.rename(columns={"断面名称": "section_name"}),
            on="section_name", how="left"
        )
    section_df.to_csv(str(OUT_DIR / "section_mapping.csv"), index=False, encoding="utf-8-sig")
    print(f"  [✓] section_mapping.csv  ({len(section_df)} 断面)")

    # ═══════ 全局列索引 ═══════
    print("\n[6] 保存全局列索引")
    col_indexer.save(OUT_DIR / "column_index_map.json")

    # ═══════ 数据集注册表 ═══════
    print("\n[7] 保存数据集注册表")
    with open(str(OUT_DIR / "dataset_registry.json"), "w", encoding="utf-8") as f:
        json.dump(dataset_registry, f, ensure_ascii=False, indent=2)
    print(f"  [✓] dataset_registry.json  ({len(dataset_registry)} 条)")

    # ═══════ 时序对齐报告 ═══════
    if alignment_reports:
        with open(str(OUT_DIR / "timestamp_alignment_report.json"), "w", encoding="utf-8") as f:
            json.dump(alignment_reports, f, ensure_ascii=False, indent=2)
        print(f"  [✓] timestamp_alignment_report.json  ({len(alignment_reports)} 条)")

    # ═══════ 汇总 ═══════
    print("\n" + "=" * 60)
    print(" 处理完成! 输出目录:", OUT_DIR)
    print("=" * 60)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="将 traindata CSV 转为 NPY 数据集")
    parser.add_argument(
        "--no-incremental", dest="incremental", action="store_false", default=True,
        help="禁用增量模式，全量重跑所有数据集"
    )
    args = parser.parse_args()
    process_all(incremental=args.incremental)


if __name__ == "__main__":
    main()
