#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
历史强基线预测器。

该预测器只使用目标日前已发生的二次调频真实标签，不读取未来数据。
同一天的不同指标/时段允许独立回溯最近可用值，以充分利用部分缺失的真实标签。
它的定位不是替代最终深度模型，而是作为必须被打败的生产候选/强基线。
"""

import os
from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


TARGET_DATASET_DIR = "03_二次调频报价"
SEGMENTS = ["T1", "T2", "T3", "T4", "T5"]
TARGET_KEYS = ["capacity", "sort_price", "clear_price"]


class HistoryBaselinePredictor:
    def __init__(
        self,
        npy_base: str,
        strategies: Optional[Dict[str, str]] = None,
        mean_window: int = 7,
        weekday_window: int = 4,
    ):
        self.npy_base = npy_base
        self.strategies = strategies or {
            "capacity": "prev_day",
            "sort_price": "prev_day",
            "clear_price": "prev_day",
        }
        self.mean_window = mean_window
        self.weekday_window = weekday_window
        self.targets = self._load_targets()
        self.available_days = self._build_available_days()
        self.complete_days = self._build_complete_days()
        self.day_to_target = {
            day.strftime("%Y-%m-%d"): self.targets.loc[day.strftime("%Y-%m-%d")].values[:5, :3].astype(np.float32)
            for day in self.available_days
        }

    def _load_targets(self) -> pd.DataFrame:
        ts_path = os.path.join(self.npy_base, TARGET_DATASET_DIR, "timestamps.npy")
        data_path = os.path.join(self.npy_base, TARGET_DATASET_DIR, "data.npy")
        ts = np.load(ts_path, allow_pickle=True)
        data = np.load(data_path)
        return pd.DataFrame(data[:, :3], index=pd.to_datetime(ts), columns=TARGET_KEYS)

    def _build_available_days(self) -> List[date]:
        daily_counts = self.targets.groupby(self.targets.index.date).size()
        candidate_days = sorted([day for day, count in daily_counts.items() if count == 5])

        available_days = []
        for day in candidate_days:
            target = self.targets.loc[day.strftime("%Y-%m-%d")].values[:5, :3]
            if not np.isnan(target).all():
                available_days.append(day)
        return available_days

    def _build_complete_days(self) -> List[date]:
        complete_days = []
        for day in self.available_days:
            target = self.targets.loc[day.strftime("%Y-%m-%d")].values[:5, :3]
            if not np.isnan(target).any():
                complete_days.append(day)
        return complete_days

    @property
    def effective_days(self) -> List[date]:
        """兼容旧评测脚本；完整三目标评测使用 complete_days。"""
        return self.complete_days

    def _history_before(self, target_day: date) -> List[date]:
        return [day for day in self.available_days if day < target_day]

    def _matrix_for_days(self, days: List[date]) -> np.ndarray:
        if not days:
            return np.full((5, 3), np.nan, dtype=np.float32)
        matrices = [
            self.day_to_target[day.strftime("%Y-%m-%d")]
            for day in days
            if day.strftime("%Y-%m-%d") in self.day_to_target
        ]
        if not matrices:
            return np.full((5, 3), np.nan, dtype=np.float32)
        return np.nanmean(np.stack(matrices, axis=0), axis=0).astype(np.float32)

    def _prev_available_matrix(self, target_day: date) -> np.ndarray:
        history = self._history_before(target_day)
        output = np.full((5, 3), np.nan, dtype=np.float32)
        for history_day in reversed(history):
            matrix = self.day_to_target[history_day.strftime("%Y-%m-%d")]
            missing = np.isnan(output)
            output[missing] = matrix[missing]
            if not np.isnan(output).any():
                break
        return output

    def _recent_mean_matrix(self, history: List[date], limit: int) -> np.ndarray:
        output = np.full((5, 3), np.nan, dtype=np.float32)
        for segment_idx in range(5):
            for metric_idx in range(3):
                values = []
                for history_day in reversed(history):
                    matrix = self.day_to_target[history_day.strftime("%Y-%m-%d")]
                    value = matrix[segment_idx, metric_idx]
                    if not np.isnan(value):
                        values.append(float(value))
                    if len(values) >= limit:
                        break
                if values:
                    output[segment_idx, metric_idx] = float(np.mean(values))
        return output

    def _complete_fallback_matrix(self, target_day: date) -> np.ndarray:
        complete_history = [day for day in self.complete_days if day < target_day]
        return self._matrix_for_days(complete_history[-1:])

    def _fill_with_fallback(self, matrix: np.ndarray, target_day: date) -> np.ndarray:
        if not np.isnan(matrix).any():
            return matrix
        fallback = self._complete_fallback_matrix(target_day)
        output = matrix.copy()
        missing = np.isnan(output)
        output[missing] = fallback[missing]
        return output

    def _predict_metric(self, target_day: date, metric_idx: int, strategy: str) -> np.ndarray:
        history = self._history_before(target_day)
        if strategy == "prev_day":
            return self._fill_with_fallback(self._prev_available_matrix(target_day), target_day)
        if strategy == "mean_7d":
            return self._fill_with_fallback(
                self._recent_mean_matrix(history, self.mean_window),
                target_day,
            )
        if strategy == "same_weekday_recent4":
            weekday_history = [day for day in history if day.weekday() == target_day.weekday()]
            return self._fill_with_fallback(
                self._recent_mean_matrix(weekday_history, self.weekday_window),
                target_day,
            )
        raise ValueError(f"不支持的历史基线策略: {strategy}")

    def predict_matrix(self, target_date_str: str) -> np.ndarray:
        target_day = pd.to_datetime(target_date_str).date()
        output = np.full((5, 3), np.nan, dtype=np.float32)

        for metric_idx, metric_name in enumerate(TARGET_KEYS):
            strategy = self.strategies[metric_name]
            pred = self._predict_metric(target_day, metric_idx, strategy)
            output[:, metric_idx] = pred[:, metric_idx]

        return output

    def predict(self, target_date_str: str) -> Dict:
        matrix = self.predict_matrix(target_date_str)
        result = {"date": target_date_str, "segments": {}}

        for idx, segment in enumerate(SEGMENTS):
            result["segments"][segment] = {
                "调频容量需求": round(float(matrix[idx, 0]), 3),
                "边际排序价格": round(float(matrix[idx, 1]), 3),
                "市场出清价格_预测均价": round(float(matrix[idx, 2]), 3),
            }

        return result
