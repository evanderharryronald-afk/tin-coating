"""Walk-Forward 5-fold CV — 时序友好的交叉评估，不打乱顺序。

fold 划分:
  fold 1: train [0%, 70%)   → test [70%, 75%)
  fold 2: train [0%, 75%)   → test [75%, 80%)
  fold 3: train [0%, 80%)   → test [80%, 85%)
  fold 4: train [0%, 85%)   → test [85%, 90%)
  fold 5: train [0%, 90%)   → test [90%, 100%)

每次调用 fit_and_evaluate_surface 里的训练+预测逻辑，单独算 RMSE/MAE。
这是**评估模块**，不做任何部署决策——生产决策应基于 CV 均值 ± std。
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from ..modeling.residual_model import ResidualCorrectionModel
from ..modeling.params import build_corrector_instance


DEFAULT_FOLDS = [(0.70, 0.75), (0.75, 0.80), (0.80, 0.85),
                 (0.85, 0.90), (0.90, 1.00)]


def _prepare_group_df(df: pd.DataFrame, group_label: str,
                      top_col: str = 'Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_TOP_Min',
                      bot_col: str = 'Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_BOT_Min') -> pd.DataFrame:
    """从 featured_data 切出目标规格组，并提前算好 Current_Per_Speed / Theoretical 等特征。"""
    df = df.copy()
    if 'Setpoint_Group_Label' not in df.columns:
        df['Setpoint_Group_Label'] = df.apply(
            lambda r: f"Top{r[top_col]}_Bot{r[bot_col]}", axis=1
        )
    return df[df['Setpoint_Group_Label'] == group_label].copy()


def run_cv_for_surface(df_group: pd.DataFrame, surface: str, params: dict,
                       folds: List[Tuple[float, float]] | None = None,
                       label_col_name: str | None = None) -> dict:
    """对单个表面在给定规格组数据上跑 Walk-Forward CV。

    Returns:
        dict with keys:
          cv_rmse_mean / cv_rmse_std
          cv_mae_mean / cv_mae_std
          n_folds / folds_detail (每段 RMSE/MAE)
    """
    folds = folds or DEFAULT_FOLDS
    prefix = surface
    speed_col = 'Speed[m/min]_Process_Avg'
    current_col = f'{prefix}_Current_Sum'
    online_col = f'Tin Weight_Actual[g/m2]_GALV_WEIGHT_{prefix.upper()}_Avg'
    setpoint_col = f'Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_{prefix.upper()}_Avg'

    df = df_group.copy()
    df[f'{prefix}_Current_Per_Speed'] = df[current_col] / (df[speed_col] + 1e-5)
    df[f'{prefix}_Weight_Deviation'] = df[setpoint_col] - df[online_col]

    feature_cols = [
        online_col, current_col,
        f'{prefix}_Current_Per_Speed',
        f'{prefix}_Theoretical_Factor',
        speed_col,
        'Dimension_[mm]_Width',
        'Dimension_[mm]_Thickness',
        'Steel_Grade_Encoded',
    ]

    # Lab 真实值列（用"上/下"中文列名，和 group_train 保持一致）
    surface_cn = '上' if surface == 'Top' else '下'
    lab_col_candidates = [
        f'{surface_cn}表面镀层重量A(XA1_0)',
        f'{surface_cn}表面镀层重量B(XA1_0)',
    ]
    lab_col = label_col_name
    if lab_col and lab_col not in df.columns:
        lab_col = None
    if lab_col is None:
        lab_col = next((c for c in lab_col_candidates if c in df.columns), None)
    if lab_col is None:
        raise KeyError(f"找不到 Lab 真值列，尝试过 {lab_col_candidates}")

    X = df[feature_cols]
    y_delta = df[f'{prefix}_Delta']
    online_actual = df[online_col]
    y_true = df[lab_col]

    n = len(X)
    rmses, maes = [], []
    details = []

    for train_end_pct, test_end_pct in folds:
        train_end = int(n * train_end_pct)
        test_start = int(n * test_end_pct)
        test_end = int(n * (test_end_pct + 0.05))
        test_end = min(test_end, n)
        if test_start >= test_end or train_end < 30:
            continue

        X_train = X.iloc[:train_end]
        y_delta_train = y_delta.iloc[:train_end]
        X_test = X.iloc[test_start:test_end]
        actual_test = online_actual.iloc[test_start:test_end]
        y_true_test = y_true.iloc[test_start:test_end]

        corrector = build_corrector_instance(params, ResidualCorrectionModel)
        corrector.fit(X_train, y_delta_train)
        pred_final, _ = corrector.predict_smooth(X_test, actual_test)

        residual = y_true_test.values - pred_final.values
        rmse = float(np.sqrt(np.mean(residual ** 2)))
        mae = float(np.mean(np.abs(residual)))
        rmses.append(rmse)
        maes.append(mae)
        details.append({
            "fold": f"[{train_end_pct:.0%},{test_end_pct:.0%})→[{test_start/n:.0%},{test_end_pct+0.05:.0%})",
            "train_samples": train_end,
            "test_samples": test_end - test_start,
            "rmse": rmse,
            "mae": mae,
        })

    return {
        "cv_rmse_mean": float(np.mean(rmses)) if rmses else None,
        "cv_rmse_std": float(np.std(rmses)) if rmses else None,
        "cv_mae_mean": float(np.mean(maes)) if maes else None,
        "cv_mae_std": float(np.std(maes)) if maes else None,
        "n_folds": len(rmses),
        "folds_detail": details,
    }
