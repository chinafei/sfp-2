#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SFP-2 特征可用性与时间对齐审计脚本

目的:
  1. 识别当前模型真实使用的输入数据集
  2. 汇总每个数据集的时间口径、机制、使用范围与对齐方式
  3. 标记潜在风险: 训练/预测不一致、可能泄漏、日单点被扩成 96 点等
  4. 输出机器可读 JSON 和人工可读 Markdown

当前审计范围仅针对 sfp-2/，不会读取已废弃的 sfp/。
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SFP2_DIR = os.path.dirname(SCRIPT_DIR)
NPY_BASE = os.path.join(SFP2_DIR, "datasets", "npy")
RESULTS_DIR = os.path.join(SFP2_DIR, "results")
DOCS_DIR = os.path.join(SFP2_DIR, "docs")

sys.path.insert(0, SFP2_DIR)

from src.data_loader import SEQ_FOLDERS  # noqa: E402


STATIC_DATASET_DIR = "12_日前正负备用需求"
TARGET_DATASET_DIR = "03_二次调频报价"

USED_DATASETS = {
    "01_日前各时段出清现货电量": {
        "role": "sequence_feature",
        "used_by": ["train", "predict", "api"],
        "feature_dims_used": 1,
        "source": "src.data_loader.SEQ_FOLDERS",
    },
    "05_非市场化机组出力": {
        "role": "sequence_feature",
        "used_by": ["train", "predict", "api"],
        "feature_dims_used": 1,
        "source": "src.data_loader.SEQ_FOLDERS",
    },
    "06_检修总容量": {
        "role": "sequence_feature",
        "used_by": ["train", "predict", "api"],
        "feature_dims_used": 1,
        "source": "src.data_loader.SEQ_FOLDERS",
    },
    "12_日前正负备用需求": {
        "role": "static_feature",
        "used_by": ["train", "predict", "api"],
        "feature_dims_used": 3,
        "source": "src.data_loader._load_static_features",
    },
    "13_省内负荷及联络线情况": {
        "role": "sequence_feature",
        "used_by": ["train", "predict", "api"],
        "feature_dims_used": 3,
        "source": "src.data_loader.SEQ_FOLDERS",
    },
    "14_输电通道可用容量": {
        "role": "sequence_feature",
        "used_by": ["train", "predict", "api"],
        "feature_dims_used": 3,
        "source": "src.data_loader.SEQ_FOLDERS",
    },
    "15_新能源出力预测(场站申报)": {
        "role": "sequence_feature",
        "used_by": ["train", "predict", "api"],
        "feature_dims_used": 3,
        "source": "src.data_loader.SEQ_FOLDERS",
    },
    "03_二次调频报价": {
        "role": "target",
        "used_by": ["train", "evaluate"],
        "feature_dims_used": 3,
        "source": "src.data_loader.load_training_data",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="SFP-2 特征可用性与时间对齐审计")
    parser.add_argument(
        "--json-path",
        default="",
        help="JSON 输出路径；留空则写入 results/",
    )
    parser.add_argument(
        "--md-path",
        default="",
        help="Markdown 输出路径；留空则写入 docs/",
    )
    return parser.parse_args()


def load_meta(dataset_dir: str) -> Dict:
    meta_path = os.path.join(NPY_BASE, dataset_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.json 不存在: {meta_path}")
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def infer_risk_flags(meta: Dict, dataset_dir: str) -> List[str]:
    flags = []
    time_attr = str(meta.get("time_attr", ""))
    mechanism = str(meta.get("mechanism", ""))
    alignment = meta.get("alignment") or {}
    usage = str(meta.get("usage", ""))

    if ("昨日真实" in time_attr or "前日真实" in time_attr) and dataset_dir in USED_DATASETS:
        flags.append("requires_prediction_time_verification")

    if mechanism == "日单点" and alignment.get("mode") == "96point":
        flags.append("daily_feature_expanded_to_96point")

    if "训练/输入" in usage and dataset_dir not in USED_DATASETS:
        flags.append("available_but_not_consumed_by_current_model")

    if dataset_dir not in USED_DATASETS and "当日预测" in time_attr:
        flags.append("candidate_feature_not_used")

    if dataset_dir == "03_二次调频报价":
        flags.append("target_only_never_available_at_prediction_time")

    return flags


def severity_for_flags(flags: List[str]) -> str:
    if any(
        flag in flags
        for flag in (
            "requires_prediction_time_verification",
            "target_only_never_available_at_prediction_time",
        )
    ):
        return "high"
    if any(
        flag in flags
        for flag in (
            "daily_feature_expanded_to_96point",
            "available_but_not_consumed_by_current_model",
            "candidate_feature_not_used",
        )
    ):
        return "medium"
    return "low"


def describe_flag(flag: str) -> str:
    mapping = {
        "requires_prediction_time_verification": "时间口径含“昨日真实/前日真实”，需要确认 D 日预测时是否做了正确 shift，避免用到预测时拿不到的信息。",
        "daily_feature_expanded_to_96point": "原始机制标注为日单点，但当前对齐模式是 96point，需确认这种扩展是否符合业务含义。",
        "available_but_not_consumed_by_current_model": "数据集存在并标注为训练/输入，但当前模型未实际读取，可能存在可用但未利用的信息。",
        "candidate_feature_not_used": "数据集具备潜在预测价值，但当前模型未纳入。",
        "target_only_never_available_at_prediction_time": "该数据集只应作为训练/评估标签，不能在预测时直接使用。",
    }
    return mapping[flag]


def audit_datasets() -> Dict:
    dataset_dirs = sorted(
        d for d in os.listdir(NPY_BASE)
        if os.path.isdir(os.path.join(NPY_BASE, d)) and d[:2].isdigit()
    )

    used_dirs = set(USED_DATASETS.keys())
    records = []
    summary = {
        "used_dataset_count": 0,
        "available_dataset_count": len(dataset_dirs),
        "unused_training_candidate_count": 0,
        "high_risk_count": 0,
        "medium_risk_count": 0,
    }

    for dataset_dir in dataset_dirs:
        meta = load_meta(dataset_dir)
        used_cfg = USED_DATASETS.get(dataset_dir, {})
        flags = infer_risk_flags(meta, dataset_dir)
        severity = severity_for_flags(flags)
        if dataset_dir in used_dirs:
            summary["used_dataset_count"] += 1
        if "available_but_not_consumed_by_current_model" in flags or "candidate_feature_not_used" in flags:
            summary["unused_training_candidate_count"] += 1
        if severity == "high":
            summary["high_risk_count"] += 1
        elif severity == "medium":
            summary["medium_risk_count"] += 1

        record = {
            "dataset_dir": dataset_dir,
            "dataset_name": meta.get("name"),
            "role": used_cfg.get("role", "unused"),
            "used_by_current_model": dataset_dir in used_dirs,
            "used_by": used_cfg.get("used_by", []),
            "feature_dims_used": used_cfg.get("feature_dims_used"),
            "meta": {
                "data_attr": meta.get("data_attr"),
                "time_attr": meta.get("time_attr"),
                "mechanism": meta.get("mechanism"),
                "usage": meta.get("usage"),
                "n_features": meta.get("n_features"),
                "shape": meta.get("shape"),
                "alignment": meta.get("alignment"),
                "columns": meta.get("columns", []),
            },
            "source_binding": used_cfg.get("source"),
            "risk_severity": severity,
            "risk_flags": flags,
            "risk_details": [describe_flag(flag) for flag in flags],
        }
        records.append(record)

    return {
        "generated_at": datetime.now().isoformat(),
        "scope": "sfp-2",
        "sequence_folders_from_code": [
            {"dataset_dir": folder, "dims": dims}
            for folder, dims in SEQ_FOLDERS
        ],
        "static_dataset_dir": STATIC_DATASET_DIR,
        "target_dataset_dir": TARGET_DATASET_DIR,
        "summary": summary,
        "datasets": records,
    }


def build_markdown(report: Dict) -> str:
    lines = []
    lines.append("# SFP-2 Feature Availability Audit")
    lines.append("")
    lines.append(f"Generated at: `{report['generated_at']}`")
    lines.append("")
    summary = report["summary"]
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Available datasets: `{summary['available_dataset_count']}`")
    lines.append(f"- Datasets used by current model: `{summary['used_dataset_count']}`")
    lines.append(f"- Unused candidate datasets: `{summary['unused_training_candidate_count']}`")
    lines.append(f"- High risk items: `{summary['high_risk_count']}`")
    lines.append(f"- Medium risk items: `{summary['medium_risk_count']}`")
    lines.append("")
    lines.append("## Current Model Inputs")
    lines.append("")
    lines.append("| Dataset | Role | Time Attr | Mechanism | Used By | Risk |")
    lines.append("|---|---|---|---|---|---|")

    for dataset in report["datasets"]:
        if not dataset["used_by_current_model"]:
            continue
        lines.append(
            "| {dataset_dir} | {role} | {time_attr} | {mechanism} | {used_by} | {risk} |".format(
                dataset_dir=dataset["dataset_dir"],
                role=dataset["role"],
                time_attr=dataset["meta"]["time_attr"],
                mechanism=dataset["meta"]["mechanism"],
                used_by=",".join(dataset["used_by"]),
                risk=dataset["risk_severity"],
            )
        )

    lines.append("")
    lines.append("## Risk Notes")
    lines.append("")
    for dataset in report["datasets"]:
        if dataset["risk_severity"] == "low":
            continue
        lines.append(f"### {dataset['dataset_dir']}")
        lines.append("")
        lines.append(f"- Severity: `{dataset['risk_severity']}`")
        lines.append(f"- Time Attr: `{dataset['meta']['time_attr']}`")
        lines.append(f"- Mechanism: `{dataset['meta']['mechanism']}`")
        if dataset["risk_details"]:
            for detail in dataset["risk_details"]:
                lines.append(f"- {detail}")
        lines.append("")

    lines.append("## Unused Candidate Datasets")
    lines.append("")
    for dataset in report["datasets"]:
        if dataset["used_by_current_model"]:
            continue
        if dataset["risk_severity"] == "low":
            continue
        lines.append(
            f"- `{dataset['dataset_dir']}`: {dataset['meta']['time_attr']} / "
            f"{dataset['meta']['mechanism']} / {dataset['meta']['usage']}"
        )

    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)

    report = audit_datasets()

    if args.json_path:
        json_path = args.json_path
    else:
        json_path = os.path.join(RESULTS_DIR, "feature_availability_audit.json")

    if args.md_path:
        md_path = args.md_path
    else:
        md_path = os.path.join(DOCS_DIR, "FEATURE_AVAILABILITY.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    markdown = build_markdown(report)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n[✓] JSON audit written to: {json_path}")
    print(f"[✓] Markdown audit written to: {md_path}")


if __name__ == "__main__":
    main()
