"""数据质量门：检查 featured_data.xlsx 是否出现训练前无法接受的异常。

这是 Layer-1 的第一道关卡——任何自动重训 / 健康检查之前必须先过这里。
只读数据，不改动任何文件。

用法：
  python monitoring/check_data_quality.py
  python monitoring/check_data_quality.py --data path/to/featured_data.xlsx
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# 让脚本直接运行时能 import 到 Task1/ 根层模块
HERE = Path(__file__).resolve().parent
TASK1_ROOT = HERE.parent
if str(TASK1_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK1_ROOT))


# ---------------------------------------------------------------------------
# 阈值（这里写死在脚本里，便于随脚本版本追踪；未来复杂了再挪到 config）
# ---------------------------------------------------------------------------
GROUP_LABEL = "Top2.799_Bot2.799"
TOP_SP = 2.799
BOT_SP = 2.799

# 异常判断阈值
MAX_MISSING_RATIO = 0.05          # 关键字段缺失率 > 5% 报警
MAX_DELTA_OUTLIER_RATIO = 0.02    # |Δ| > 0.5 的样本占比 > 2% 报警
MIN_GROUP_SAMPLES = 200           # 样本少于 200 不建模
RECENT_WINDOW_DAYS = 30           # "近 N 天"用来判断是否有新数据进

# 需要完整的字段
REQUIRED_COLUMNS = [
    "Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Avg",
    "Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Avg",
    "Top_Delta", "Bot_Delta",
    "Top_Current_Sum", "Bot_Current_Sum",
    "Speed[m/min]_Process_Avg",
    "Dimension_[mm]_Width", "Dimension_[mm]_Thickness",
    "Steel_Grade_Encoded",
    "Top_Theoretical_Factor", "Bot_Theoretical_Factor",
]


def _infer_date_col(df: pd.DataFrame) -> str | None:
    for cand in ["DateTime", "datetime", "Date", "date", "Timestamp", "timestamp"]:
        if cand in df.columns:
            return cand
    return None


def run(data_path: str | Path, report_dir: str | None = None, silent: bool = False) -> dict:
    """执行数据质量检查，返回结构化结果（也写一份 JSON 报告）。

    返回 dict 里带一个整体 pass/fail 标志：{"pass": True/False, ...}
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"featured_data 不存在: {data_path}")

    if report_dir is None:
        report_dir = HERE / "reports"
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    df_all = pd.read_excel(data_path)

    # 过滤到目标规格组（Top & Bot 设定值都 round=2.799）
    top_sp_col = "Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_TOP_Avg"
    bot_sp_col = "Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_BOT_Avg"
    df_group = df_all[
        (df_all.get(top_sp_col, pd.Series(dtype=float)).round(3) == TOP_SP)
        & (df_all.get(bot_sp_col, pd.Series(dtype=float)).round(3) == BOT_SP)
    ].copy() if top_sp_col in df_all.columns and bot_sp_col in df_all.columns else df_all

    result: dict = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "data_path": str(data_path),
        "total_rows_in_file": int(len(df_all)),
        "group_label": GROUP_LABEL,
        "group_rows": int(len(df_group)),
        "checks": {},
        "pass": True,
    }

    def _fail(check_name: str, message: str, extra=None):
        result["checks"][check_name] = {"pass": False, "message": message, "extra": extra}
        result["pass"] = False

    def _ok(check_name: str, message: str, extra=None):
        result["checks"][check_name] = {"pass": True, "message": message, "extra": extra}

    # --------------------------------------------------------------
    # 检查 1：必需列是否存在
    # --------------------------------------------------------------
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df_group.columns]
    if missing_cols:
        _fail("required_columns_missing",
              f"缺少必需列: {missing_cols}")
    else:
        _ok("required_columns_missing",
            f"{len(REQUIRED_COLUMNS)} 列齐全")

    # --------------------------------------------------------------
    # 检查 2：规格组样本数是否足够
    # --------------------------------------------------------------
    if len(df_group) < MIN_GROUP_SAMPLES:
        _fail("group_sample_count",
              f"样本 {len(df_group)} < 阈值 {MIN_GROUP_SAMPLES}")
    else:
        _ok("group_sample_count",
            f"样本 {len(df_group)} ≥ 阈值 {MIN_GROUP_SAMPLES}")

    # --------------------------------------------------------------
    # 检查 3：关键字段缺失率
    # --------------------------------------------------------------
    numeric_check_cols = [c for c in REQUIRED_COLUMNS if c in df_group.columns]
    missing_ratio = df_group[numeric_check_cols].isna().mean().mean() if numeric_check_cols else 1.0
    if missing_ratio > MAX_MISSING_RATIO:
        _fail("missing_ratio",
              f"关键字段平均缺失率 {missing_ratio:.2%} > 阈值 {MAX_MISSING_RATIO:.0%}",
              extra={"per_col": df_group[numeric_check_cols].isna().mean().to_dict()})
    else:
        _ok("missing_ratio",
            f"关键字段平均缺失率 {missing_ratio:.2%} ≤ 阈值 {MAX_MISSING_RATIO:.0%}",
            extra={"per_col": df_group[numeric_check_cols].isna().mean().to_dict()})

    # --------------------------------------------------------------
    # 检查 4：Δ 分布是否出现极端值激增
    # --------------------------------------------------------------
    for prefix in ["Top", "Bot"]:
        col = f"{prefix}_Delta"
        if col in df_group.columns:
            series = df_group[col].dropna()
            extreme_ratio = (series.abs() > 0.5).mean()
            name = f"delta_outlier_{prefix}"
            if extreme_ratio > MAX_DELTA_OUTLIER_RATIO:
                _fail(name,
                      f"{prefix} |Δ|>0.5 占比 {extreme_ratio:.2%} > 阈值 {MAX_DELTA_OUTLIER_RATIO:.0%}")
            else:
                _ok(name,
                    f"{prefix} |Δ|>0.5 占比 {extreme_ratio:.2%}")

    # --------------------------------------------------------------
    # 检查 5：近 N 天是否有新数据进（判断"值得重训吗"的辅助信号）
    # --------------------------------------------------------------
    date_col = _infer_date_col(df_group)
    if date_col and pd.api.types.is_datetime64_any_dtype(df_group[date_col]):
        latest = df_group[date_col].max()
        days_since = (pd.Timestamp.now() - latest).days
        recent = df_group[date_col] >= (latest - pd.Timedelta(days=RECENT_WINDOW_DAYS))
        recent_n = int(recent.sum())
        _ok("recent_data_presence",
            f"最近数据距今 {days_since} 天，近 {RECENT_WINDOW_DAYS} 天内 {recent_n} 条",
            extra={"latest_timestamp": str(latest),
                   "days_since_latest": days_since,
                   "recent_rows": recent_n})
    else:
        _ok("recent_data_presence",
            "未识别到日期列或非 datetime，跳过近 N 天检查")

    # --------------------------------------------------------------
    # 写报告
    # --------------------------------------------------------------
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    report_path = report_dir / f"data_quality_{stamp}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    if not silent:
        print(f"[quality] {'✅ PASS' if result['pass'] else '❌ FAIL'}  —  {report_path}")
        for name, info in result["checks"].items():
            mark = "✅" if info["pass"] else "❌"
            print(f"  {mark} {name}: {info['message']}")

    return result


def main():
    parser = argparse.ArgumentParser(description="数据质量门（只读，不修改任何文件）")
    parser.add_argument("--data", type=str,
                        default=str(TASK1_ROOT / "result/data/feature_engineered_data/featured_data.xlsx"),
                        help="featured_data.xlsx 路径")
    parser.add_argument("--report-dir", type=str, default=None,
                        help="报告输出目录（默认 monitoring/reports/）")
    args = parser.parse_args()

    result = run(args.data, args.report_dir)
    sys.exit(0 if result["pass"] else 2)


if __name__ == "__main__":
    main()
