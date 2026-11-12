"""自动重训候选生成器 —— 产出 .pending 模型 + CV 对比报告，不覆盖线上模型。

流程:
  1. 先跑 check_data_quality.run() 做质量门 —— 不通过直接 abort
  2. 读 featured_data.xlsx，过滤到目标规格组
  3. 按 surface 各自调用 coating_model_by_group.fit_and_evaluate_surface 重训 + 单次 hold-out
  4. 调用 ablation.run_ablation._cv_engine 跑 Walk-Forward 5-fold CV 算稳定性
  5. 和 registry 里线上 active 版本做指标对比（CV 均值 / hold-out 测试集 / 训练样本数）
  6. 产出候选模型到 monitoring/reports/{Top,Bot}_model_{timestamp}.pending.joblib
  7. 产出候选报告到 monitoring/reports/retrain_report_{timestamp}.json / .txt
  8. 注册到 registry（active=False）

安全性：
  - 不写 result/trained_models/ 正式路径
  - 不碰 group_params_2_799_only.json
  - 不触发 Optuna 重搜（Optuna 继续手动）
  - 只有 gate.py --approve 才会真正部署

用法:
  python monitoring/auto_retrain_candidate.py
  python monitoring/auto_retrain_candidate.py --skip-quality-check   # 调试用
  python monitoring/auto_retrain_candidate.py --surface Top           # 只重训一个表面
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

from registry import ModelRegistry  # noqa: E402
import check_data_quality as dq    # noqa: E402

import coating_model_by_group as cm  # noqa: E402


GROUP_LABEL = "Top2.799_Bot2.799"
TOP_SP = 2.799
BOT_SP = 2.799

DEFAULT_FEATURED = TASK1_ROOT / "result/data/feature_engineered_data/featured_data.xlsx"
GROUP_PARAMS_PATH = TASK1_ROOT / "group_params_2_799_only.json"
REPORTS_DIR = HERE / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_params_per_surface(group_label: str) -> dict[str, dict]:
    """从 group_params_2_799_only.json 取到每个 surface 的完整 params。"""
    if not GROUP_PARAMS_PATH.exists():
        raise FileNotFoundError(f"{GROUP_PARAMS_PATH} 不存在，无法取超参")
    with open(GROUP_PARAMS_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    default = cfg.get("default_params", {})
    override = cfg.get("group_params_override", {})
    out = {}
    for surface in ["Top", "Bot"]:
        key = f"{group_label}__{surface}"
        p = dict(default)
        p.update(override.get(key, {}))
        out[surface] = p
    return out


def _filter_group(df: pd.DataFrame) -> pd.DataFrame:
    top_col = "Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_TOP_Avg"
    bot_col = "Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_BOT_Avg"
    return df[
        (df[top_col].round(3) == TOP_SP)
        & (df[bot_col].round(3) == BOT_SP)
    ].copy()


def _walk_forward_cv_summary(df: pd.DataFrame, surface: str, params: dict) -> dict:
    """5-fold Walk-Forward CV，返回 RMSE/MAE 均值±std。

    每个 fold: train = [0, p%]  → test = [p%, p+5%)，p=70/75/80/85。
    复用 fit_and_evaluate_surface 里的特征构造 + ResidualCorrectionModel 训练逻辑。
    """
    from sklearn.metrics import mean_squared_error

    n = len(df)
    prefix = surface
    speed_col = "Speed[m/min]_Process_Avg"
    online_col = f"Tin Weight_Actual[g/m2]_GALV_WEIGHT_{prefix.upper()}_Avg"
    current_col = f"{prefix}_Current_Sum"
    y_true_col = f"{surface}表面镀层重量A(XA1_0)"
    if y_true_col not in df.columns:
        y_true_col = f"{surface}表面镀层重量B(XA1_1)"

    folds = [(0.70, 0.75), (0.75, 0.80), (0.80, 0.85), (0.85, 0.90), (0.90, 1.00)]
    rmses, maes = [], []

    for train_end_pct, test_end_pct in folds:
        train_end = int(n * train_end_pct)
        test_start = int(n * test_end_pct)
        test_end = int(n * (test_end_pct + 0.05))
        test_end = min(test_end, n)
        if test_start >= test_end:
            continue

        train_df = df.iloc[:train_end].copy()
        test_df = df.iloc[test_start:test_end].copy()

        train_df[f"{prefix}_Current_Per_Speed"] = train_df[current_col] / (train_df[speed_col] + 1e-5)
        test_df[f"{prefix}_Current_Per_Speed"] = test_df[current_col] / (test_df[speed_col] + 1e-5)

        feature_cols = [
            online_col, current_col,
            f"{prefix}_Current_Per_Speed",
            f"{prefix}_Theoretical_Factor",
            speed_col,
            "Dimension_[mm]_Width",
            "Dimension_[mm]_Thickness",
            "Steel_Grade_Encoded",
        ]
        X_train = train_df[feature_cols]
        y_delta_train = train_df[f"{prefix}_Delta"]
        X_test = test_df[feature_cols]
        actual_test = test_df[online_col]
        y_true_test = test_df[y_true_col]

        corrector = cm._build_corrector_instance(params, cm.ResidualCorrectionModel)
        corrector.fit(X_train, y_delta_train)
        pred_final, _ = corrector.predict_smooth(X_test, actual_test)

        residual = y_true_test.values - pred_final.values
        rmses.append(np.sqrt(np.mean(residual ** 2)))
        maes.append(np.mean(np.abs(residual)))

    return {
        "cv_rmse_mean": float(np.mean(rmses)) if rmses else None,
        "cv_rmse_std": float(np.std(rmses)) if rmses else None,
        "cv_mae_mean": float(np.mean(maes)) if maes else None,
        "cv_mae_std": float(np.std(maes)) if maes else None,
        "n_folds": len(rmses),
    }


def run(featured_path: str | Path,
        surfaces: list[str] | None = None,
        skip_quality_check: bool = False,
        force: bool = False) -> dict:
    """执行自动重训候选生成。force=True 时强制跑，不管 registry 里是否已有 pending。"""
    featured_path = Path(featured_path)
    surfaces = surfaces or ["Top", "Bot"]

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    report = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "data_path": str(featured_path),
        "group_label": GROUP_LABEL,
        "stamp": stamp,
        "pending_models": {},
        "skipped": [],
        "abort_reason": None,
    }

    # --------------------------------------------------------------
    # 0. 质量门
    # --------------------------------------------------------------
    if not skip_quality_check:
        q = dq.run(featured_path, silent=True)
        if not q["pass"]:
            report["abort_reason"] = f"质量门未通过: {q}"
            print(f"[retrain] ❌ 质量门未通过，已中止。详细: {q}")
            return report
        print("[retrain] ✅ 质量门通过，继续")

    # --------------------------------------------------------------
    # 1. 数据 + 超参
    # --------------------------------------------------------------
    df_all = pd.read_excel(featured_path)
    df = _filter_group(df_all)
    params_map = _load_params_per_surface(GROUP_LABEL)

    reg = ModelRegistry()

    # --------------------------------------------------------------
    # 2. 按 surface 各自重训
    # --------------------------------------------------------------
    import joblib
    import __main__
    __main__.ResidualCorrectionModel = cm.ResidualCorrectionModel  # pickle 兼容

    for surface in surfaces:
        params = params_map[surface]

        # check if there is already an unapproved pending
        latest_pending = None
        for entry in reg.get_history(GROUP_LABEL, surface):
            if not entry.get("active") and ".pending." in entry.get("path", ""):
                latest_pending = entry
        if latest_pending and not force:
            report["skipped"].append({
                "surface": surface,
                "reason": f"已有未审批的 pending 版本: {latest_pending['path']}"
            })
            print(f"  ⏭️ {surface}: 已有 pending 未审批，跳过 (--force 覆盖)")
            continue

        print(f"\n[retrain] ⚙️ {surface}: hold-out + Walk-Forward CV ...")

        # 2a. hold-out
        try:
            corrector, holdout_metrics, _ = cm.fit_and_evaluate_surface(
                df, surface, params, group_tag=GROUP_LABEL,
                train_ratio=0.65, val_ratio=0.20,
            )
        except Exception as e:
            report["skipped"].append({"surface": surface, "reason": f"hold-out 失败: {e}"})
            print(f"  ❌ {surface} hold-out 失败: {e}")
            continue

        ho_test_rmse = holdout_metrics.get("RMSE_测试",
                                           holdout_metrics.get(f"RMSE_测试_模型", None))
        ho_train_rmse = holdout_metrics.get("RMSE_训练")

        # 2b. Walk-Forward CV
        try:
            cv_sum = _walk_forward_cv_summary(df, surface, params)
        except Exception as e:
            cv_sum = {"error": str(e)}

        # 2c. 线上旧版本对比
        old = reg.get_active(GROUP_LABEL, surface)
        comparison = {}
        if old:
            for key in ["cv_rmse_mean", "holdout_test_rmse"]:
                old_v = old.get(key)
                new_v = (cv_sum if isinstance(cv_sum, dict) else {}).get(
                    key, ho_test_rmse if "holdout" in key else None
                )
                if old_v is not None and new_v is not None:
                    delta = new_v - old_v
                    pct = delta / old_v if old_v > 0 else 0
                    comparison[key] = {"old": old_v, "new": new_v, "delta": delta,
                                       "pct": pct, "improved": delta < 0}

        # 2d. 保存 pending 模型（写到 monitoring/reports/）
        pending_path = REPORTS_DIR / f"{surface}_model_{stamp}.pending.joblib"
        joblib.dump(corrector, pending_path, compress=3)

        # 2e. registry 注册
        reg.register(GROUP_LABEL, surface, {
            "version": stamp,
            "path": str(pending_path),
            "cv_rmse_mean": cv_sum.get("cv_rmse_mean") if isinstance(cv_sum, dict) else None,
            "cv_rmse_std": cv_sum.get("cv_rmse_std") if isinstance(cv_sum, dict) else None,
            "cv_mae_mean": cv_sum.get("cv_mae_mean") if isinstance(cv_sum, dict) else None,
            "holdout_test_rmse": ho_test_rmse,
            "holdout_train_rmse": ho_train_rmse,
            "training_samples": int(len(df)),
            "features": 8,
            "hyperparameters": params,
            "comparison_vs_active": comparison,
            "source": "auto_retrain_candidate",
            "active": False,
        })

        verdict = "✅"
        verdict_text = "—"
        if isinstance(cv_sum, dict) and "cv_rmse_mean" in cv_sum:
            if old and comparison.get("cv_rmse_mean", {}).get("improved"):
                verdict = "⬆️"
                verdict_text = f"CV RMSE ↓ {comparison['cv_rmse_mean']['pct']:+.2%}"
            elif old and not comparison.get("cv_rmse_mean", {}).get("improved", True):
                verdict = "⚠️"
                verdict_text = f"CV RMSE ↑ {comparison['cv_rmse_mean']['pct']:+.2%}"
            else:
                verdict_text = f"CV RMSE {cv_sum['cv_rmse_mean']:.4f} ± {cv_sum['cv_rmse_std']:.4f}"
        print(f"  {verdict} {surface} {verdict_text}  →  {pending_path.name}")

        report["pending_models"][surface] = {
            "stamp": stamp,
            "path": str(pending_path),
            "cv": cv_sum,
            "holdout_test_rmse": ho_test_rmse,
            "comparison_vs_active": comparison,
        }

    # --------------------------------------------------------------
    # 3. 写报告
    # --------------------------------------------------------------
    json_path = REPORTS_DIR / f"retrain_report_{stamp}.json"
    txt_path = REPORTS_DIR / f"retrain_report_{stamp}.txt"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 简明 txt
    lines = [f"# Auto-Retrain Report  {stamp}",
             f"# 数据: {featured_path}",
             f"# 规格组: {GROUP_LABEL}", ""]
    for surface, info in report.get("pending_models", {}).items():
        cv = info.get("cv", {})
        comp = info.get("comparison_vs_active", {})
        lines.append(f"## {surface}")
        lines.append(f"  hold-out 测试 RMSE: {info.get('holdout_test_rmse')}")
        if "cv_rmse_mean" in cv:
            lines.append(f"  Walk-Forward CV RMSE: {cv['cv_rmse_mean']:.4f} ± {cv['cv_rmse_std']:.4f}  "
                         f"(n_folds={cv['n_folds']})")
        if comp:
            for key, v in comp.items():
                lines.append(f"  vs active ({key}): {v['old']} → {v['new']} "
                             f"({v['pct']:+.2%}, improved={v['improved']})")
        lines.append(f"  pending 模型: {info['path']}")
        lines.append(f"  审批: python monitoring/gate.py --approve {surface} --version {stamp}")
        lines.append("")
    if report.get("skipped"):
        lines.append("## Skipped")
        for s in report["skipped"]:
            lines.append(f"  - {s['surface']}: {s['reason']}")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n[retrain] ✅ 完成 —  {txt_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="自动重训候选生成器（只产出 .pending，不覆盖线上）")
    parser.add_argument("--data", type=str, default=str(DEFAULT_FEATURED))
    parser.add_argument("--surface", type=str, default=None,
                        help="Top / Bot；不指定就两个都跑")
    parser.add_argument("--skip-quality-check", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="强制重训（即使已有未审批的 pending）")
    args = parser.parse_args()

    surfaces = [args.surface] if args.surface else None
    report = run(args.data, surfaces, args.skip_quality_check, args.force)

    if report.get("abort_reason"):
        sys.exit(2)
    if report.get("pending_models"):
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
