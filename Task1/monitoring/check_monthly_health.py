"""月度模型健康度检查：评估当前线上模型在最新数据上的效果。

只读现有 joblib 模型 + featured_data.xlsx，不改动任何旧文件。
输出：
  monitoring/reports/monthly_health_YYYY-MM-DD_HHMMSS.csv   # 按时间段明细
  monitoring/reports/monthly_health_YYYY-MM-DD_HHMMSS.json  # 汇总 + 异常判断

核心指标（按 surface × 时间段分别计算）：
  - Online 原始 MAE / RMSE（基线）
  - 模型校正后 MAE / RMSE
  - 提升幅度（%）
  - Δ 均值 / 标准差

异常判断（相对"全部历史段的均值 ± N*std"）：
  - 当前时间段的"模型 MAE 提升幅度" < 历史均值 - 2*std → 标记为 ⚠️ 退化

用法：
  python monitoring/check_monthly_health.py
  python monitoring/check_monthly_health.py --group Top2.799_Bot2.799
  python monitoring/check_monthly_health.py --window 30   # 只看最近 30 天（如果有日期列）
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TASK1_ROOT = HERE.parent
if str(TASK1_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK1_ROOT))

# 复用生产链路里的现成逻辑
from model_persistence import load_model  # noqa: E402


GROUP_LABEL = "Top2.799_Bot2.799"
TOP_SP = 2.799
BOT_SP = 2.799

JOBS_DIR = TASK1_ROOT / "result/trained_models/by_group" / GROUP_LABEL
DEFAULT_FEATURED = TASK1_ROOT / "result/data/feature_engineered_data/featured_data.xlsx"

# 异常阈值
REGRESSION_Z_THRESHOLD = -2.0      # z-score 低于此值判定退化
MIN_SEGMENT_SAMPLES = 30           # 单个时间段少于 30 条不单独判断


def _group_mask(df: pd.DataFrame) -> pd.Series:
    top_sp_col = "Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_TOP_Avg"
    bot_sp_col = "Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_BOT_Avg"
    if top_sp_col not in df.columns or bot_sp_col not in df.columns:
        return pd.Series(True, index=df.index)
    return (df[top_sp_col].round(3) == TOP_SP) & (df[bot_sp_col].round(3) == BOT_SP)


def _try_load_model(surface: str, group_label: str, train_df: pd.DataFrame | None = None,
                   hyperparams: dict | None = None):
    """加载 / 重建当前线上 ResidualCorrectionModel。

    优先按路径找 joblib；如果 numpy 版本漂移、pickle 反序列化失败，
    就用同样的超参重新训练一个等价模型（训练数据 / 超参必须与线上一致）。

    线上超参来源：group_params_2_799_only.json（alpha_smoothing + HGB 超参）。
    """
    import coating_model_by_group as cm
    import __main__
    __main__.ResidualCorrectionModel = cm.ResidualCorrectionModel

    primary = JOBS_DIR / f"{surface}_model.joblib"
    if primary.exists():
        try:
            import joblib
            return joblib.load(primary), True  # (model, loaded_from_joblib)
        except Exception as e:
            print(f"  [load_model] joblib 加载失败 ({e.__class__.__name__})，"
                  f"回退到用同超参重训一个临时模型", flush=True)

    # ---- 回退路径：用同超参重训 ----
    if train_df is None or hyperparams is None:
        raise RuntimeError(
            f"{surface} joblib 不存在且未提供重训所需的 train_df + hyperparams，无法回退。"
            f"请先跑一遍 coating_model_by_group.py 产出 joblib，或者在调用 run() 时传入 train_df"
        )
    return _retrain_equivalent(surface, train_df, hyperparams), False


def _retrain_equivalent(surface: str, df: pd.DataFrame, params: dict):
    """用与线上完全相同的特征 + 超参，在给定 df 上重新训练 ResidualCorrectionModel。

    复用 coating_model_by_group.fit_and_evaluate_surface 内的特征构造逻辑，
    只做 train_ratio=1.0 的"全量训练"。
    """
    import coating_model_by_group as cm
    prefix = surface
    speed_col = "Speed[m/min]_Process_Avg"
    online_col = f"Tin Weight_Actual[g/m2]_GALV_WEIGHT_{prefix.upper()}_Avg"
    current_col = f"{prefix}_Current_Sum"

    df = df.copy()
    df[f"{prefix}_Current_Per_Speed"] = df[current_col] / (df[speed_col] + 1e-5)

    feature_cols = [
        online_col, current_col,
        f"{prefix}_Current_Per_Speed",
        f"{prefix}_Theoretical_Factor",
        speed_col,
        "Dimension_[mm]_Width",
        "Dimension_[mm]_Thickness",
        "Steel_Grade_Encoded",
    ]
    X = df[feature_cols]
    y_delta = df[f"{prefix}_Delta"]
    corrector = cm._build_corrector_instance(params, cm.ResidualCorrectionModel)
    corrector.fit(X, y_delta)
    return corrector


def _build_features(df: pd.DataFrame, surface: str) -> pd.DataFrame:
    """与 coating_model_by_group.fit_and_evaluate_surface 中的特征逻辑保持一致。"""
    prefix = surface
    speed_col = "Speed[m/min]_Process_Avg"
    online_col = f"Tin Weight_Actual[g/m2]_GALV_WEIGHT_{prefix.upper()}_Avg"
    current_col = f"{prefix}_Current_Sum"

    df = df.copy()
    df[f"{prefix}_Current_Per_Speed"] = df[current_col] / (df[speed_col] + 1e-5)

    return df[[
        online_col, current_col,
        f"{prefix}_Current_Per_Speed",
        f"{prefix}_Theoretical_Factor",
        speed_col,
        "Dimension_[mm]_Width",
        "Dimension_[mm]_Thickness",
        "Steel_Grade_Encoded",
    ]]


def _metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    residual = actual - pred
    return {
        "n": int(len(actual)),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "bias": float(np.mean(residual)),
    }


def run(featured_path: str | Path, group_label: str = GROUP_LABEL,
        window_days: int | None = None,
        report_dir: str | None = None,
        silent: bool = False) -> dict:
    featured_path = Path(featured_path)
    if not featured_path.exists():
        raise FileNotFoundError(f"featured_data 不存在: {featured_path}")

    if report_dir is None:
        report_dir = HERE / "reports"
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    df_all = pd.read_excel(featured_path)
    df = df_all[_group_mask(df_all)].copy()

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    # --------------------------------------------------------------
    # 尝试识别日期列（用于按时间段切分报告）
    # --------------------------------------------------------------
    date_col = None
    for cand in ["DateTime", "datetime", "Date", "date", "Timestamp", "timestamp"]:
        if cand in df.columns:
            date_col = cand
            break

    segments = []  # list of (name, mask)
    if date_col and pd.api.types.is_datetime64_any_dtype(df[date_col]):
        ts = df[date_col]
        if window_days:
            cutoff = ts.max() - pd.Timedelta(days=window_days)
            segments.append((f"最近 {window_days} 天", ts >= cutoff))
            segments.append((f"之前（对比基线）", ts < cutoff))
        else:
            # 按月份分
            months = ts.dt.to_period("M")
            for name, mask in df.groupby(months).groups.items():
                segments.append((str(name), df.index.isin(mask)))
    else:
        # 没有日期列 — 只能整体 + 尾部 30% / 头部 30% 做对比
        n = len(df)
        segments.append(("整体", pd.Series(True, index=df.index)))
        segments.append(("尾部 30%", df.index >= int(0.7 * n)))
        segments.append(("头部 30%", df.index < int(0.3 * n)))

    # --------------------------------------------------------------
    # 读取线上超参（和 group_params_2_799_only.json 保持一致）
    # --------------------------------------------------------------
    _GROUP_PARAMS_PATH = TASK1_ROOT / "group_params_2_799_only.json"
    all_params_per_surface = {}
    if _GROUP_PARAMS_PATH.exists():
        import json as _json
        with open(_GROUP_PARAMS_PATH, "r", encoding="utf-8") as f:
            cfg = _json.load(f)
        override = cfg.get("group_params_override", {})
        default_params = cfg.get("default_params", {})
        for surface in ["Top", "Bot"]:
            key = f"{group_label}__{surface}"
            surface_params = dict(default_params)
            surface_params.update(override.get(key, {}))
            all_params_per_surface[surface] = surface_params

    # --------------------------------------------------------------
    # 分表面跑预测 + 算各段指标
    # --------------------------------------------------------------
    csv_rows = []
    summary = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "data_path": str(featured_path),
        "group_label": group_label,
        "group_rows": int(len(df)),
        "surfaces": {},
        "overall_verdict": "OK",
        "warnings": [],
    }

    for surface in ["Top", "Bot"]:
        params = all_params_per_surface.get(surface, {})
        corrector, loaded_from_joblib = _try_load_model(
            surface, group_label, train_df=df, hyperparams=params
        )

        prefix = surface
        online_col = f"Tin Weight_Actual[g/m2]_GALV_WEIGHT_{prefix.upper()}_Avg"
        lab_col_candidates = [
            f"{surface}表面镀层重量A(XA1_0)",
            f"{surface}表面镀层重量B(XA1_1)",
            f"{surface}表面镀层重量B(XA1_1)" if surface == "Top" else f"Bot表面镀层重量A(XA1_0)",
        ]
        lab_col = next((c for c in lab_col_candidates if c in df.columns), None)
        if lab_col is None:
            delta_col = f"{prefix}_Delta"
            if delta_col in df.columns:
                lab_col = "__derived__"
                df[f"__derived__"] = df[online_col] + df[delta_col]

        if lab_col is None:
            summary["surfaces"][surface] = {"error": "找不到 Lab 真实值列"}
            summary["warnings"].append(f"{surface}: 找不到 Lab 真实值列")
            summary["overall_verdict"] = "WARN"
            continue

        X = _build_features(df, surface)
        actual = df[online_col]
        y_true = df[lab_col] if lab_col != "__derived__" else df["__derived__"]

        try:
            pred_final, _ = corrector.predict_smooth(X, actual)
        except Exception as e:
            summary["surfaces"][surface] = {"error": f"predict_smooth 失败: {e}"}
            summary["warnings"].append(f"{surface}: predict_smooth 失败 — {e}")
            summary["overall_verdict"] = "WARN"
            continue

        seg_metrics = []
        for seg_name, seg_mask in segments:
            sub = pd.Series(seg_mask).values
            if sub.sum() < MIN_SEGMENT_SAMPLES:
                continue
            online_m = _metrics(y_true.values[sub], actual.values[sub])
            model_m = _metrics(y_true.values[sub], pred_final.values[sub])
            improve = {}
            for k in ["mae", "rmse"]:
                base = online_m[k]
                model = model_m[k]
                improve[k] = (model - base) / base if base > 0 else 0.0

            row = {
                "surface": surface,
                "segment": seg_name,
                "n": int(sub.sum()),
                "online_mae": online_m["mae"],
                "online_rmse": online_m["rmse"],
                "online_bias": online_m["bias"],
                "model_mae": model_m["mae"],
                "model_rmse": model_m["rmse"],
                "model_bias": model_m["bias"],
                "mae_improve_pct": improve["mae"],
                "rmse_improve_pct": improve["rmse"],
            }
            csv_rows.append(row)
            seg_metrics.append(row)

        # 异常判断：对"最近"段 vs "历史"段做 z-score
        verdict = "OK"
        if len(seg_metrics) >= 2:
            # 最后一段当作"最近"，其余当"历史"
            recent = seg_metrics[-1]
            historical = seg_metrics[:-1]
            hist_mean = np.mean([h["rmse_improve_pct"] for h in historical])
            hist_std = max(np.std([h["rmse_improve_pct"] for h in historical]), 1e-6)
            z = (recent["rmse_improve_pct"] - hist_mean) / hist_std
            recent["_z_score"] = float(z)
            if z < REGRESSION_Z_THRESHOLD:
                verdict = "REGRESSION"
                summary["warnings"].append(
                    f"{surface} 最近段 {recent['segment']} RMSE 提升幅度 z-score={z:.2f}，"
                    f"低于阈值 {REGRESSION_Z_THRESHOLD}，疑似退化")
                summary["overall_verdict"] = "REGRESSION"

        summary["surfaces"][surface] = {
            "verdict": verdict,
            "segments": seg_metrics,
        }

    # --------------------------------------------------------------
    # 写 CSV + JSON 报告
    # --------------------------------------------------------------
    csv_path = report_dir / f"monthly_health_{stamp}.csv"
    json_path = report_dir / f"monthly_health_{stamp}.json"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if not silent:
        print(f"[health] {summary['overall_verdict']}  —  {json_path}")
        for surface, info in summary["surfaces"].items():
            if "error" in info:
                print(f"  ❌ {surface}: {info['error']}")
                continue
            v = info.get("verdict", "?")
            mark = {"OK": "✅", "WARN": "⚠️", "REGRESSION": "❌"}.get(v, "?")
            for seg in info["segments"]:
                print(f"  {mark} {surface} [{seg['segment']:>10s}] "
                      f"n={seg['n']:4d}  "
                      f"Online RMSE={seg['online_rmse']:.4f} → Model RMSE={seg['model_rmse']:.4f}  "
                      f"(RMSE 提升 {seg['rmse_improve_pct']:+.2%})")

    return summary


def main():
    parser = argparse.ArgumentParser(description="月度模型健康度检查（只读）")
    parser.add_argument("--data", type=str, default=str(DEFAULT_FEATURED))
    parser.add_argument("--group", type=str, default=GROUP_LABEL)
    parser.add_argument("--window", type=int, default=None,
                        help="只看最近 N 天（需要日期列）；默认按月份切分")
    parser.add_argument("--report-dir", type=str, default=None)
    args = parser.parse_args()

    summary = run(args.data, args.group, args.window, args.report_dir)

    # 退出码：OK=0, WARN=1, REGRESSION=2
    code = {"OK": 0, "WARN": 1, "REGRESSION": 2}.get(summary["overall_verdict"], 1)
    sys.exit(code)


if __name__ == "__main__":
    main()
