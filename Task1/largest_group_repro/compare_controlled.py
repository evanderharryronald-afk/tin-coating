#!/usr/bin/env python3
"""
reproduce_optimal vs Task1 —— 控制变量对比

核心问题: reproduce_optimal 的结果为什么比我们之前的好？
控制变量:
  - 数据路径相同
  - 时间切分相同 (前 80% train, 后 20% test)
  - 2.799/2.799 规格组相同
  - 评估函数相同

改变量:
  A. 训练目标: Delta (Lab_A - Online) vs Delta_mean (Lab 4点均值 - Online)
  B. 特征集:   无 lag 16 base vs 有 lag 25 feats
  C. HGB 参数: reproduce_optimal 版 vs Task1 对齐版
  D. EWMA 后处理: 加不加

这是一个 2×2×2×2 的对比，但我们只看关键组合
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "reproduce_optimal_soft_sensor"))
sys.path.insert(0, str(Path(__file__).parent))

import importlib.util
spec = importlib.util.spec_from_file_location("ros", Path(__file__).parent.parent / "reproduce_optimal_soft_sensor" / "reproduce_optimal_soft_sensor.py")
ros_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ros_module)

load_optimal = ros_module.load_and_fe
ewma_causal = ros_module.ewma_causal

# 直接内联 add_lag_features（避免 import 路径麻烦）
def add_lag_safe(train_df, test_df, col, lag=5):
    """对单个列构造 lag，numpy 数组避免 index 对齐问题"""
    tr_vals = train_df[col].values.astype(float)
    te_vals = test_df[col].values.astype(float)
    n_tr = len(tr_vals)
    for i in range(1, lag+1):
        # 构造全序列: [train | test]，然后 shift i，取 test 部分
        full = np.concatenate([tr_vals, te_vals])
        shifted = np.full(len(full), np.nan)
        shifted[i:] = full[:-i]
        tr_lag = shifted[:n_tr]
        te_lag = shifted[n_tr:]
        train_df[f"{col}_lag{i}"] = tr_lag
        test_df[f"{col}_lag{i}"] = te_lag

# evaluate 复用 ros_module
evaluate = ros_module.evaluate

DATA_PATH = Path(__file__).parent.parent / "result" / "data" / "cleaned_data" / "cleaned_data.xlsx"
print(f"数据路径: {DATA_PATH}")

# ===== 加载 + 切分 =====
df = load_optimal(DATA_PATH)
df = df.sort_values("Produce_Time").reset_index(drop=True)
g = df[(df["SP_TOP_r"] == 2.799) & (df["SP_BOT_r"] == 2.799)].copy().reset_index(drop=True)
print(f"SP=2.799 规格组: {len(g)} 条")

train, test = g.iloc[:int(len(g)*0.80)].copy(), g.iloc[int(len(g)*0.80):].copy()
print(f"Train: {len(train)}, Test: {len(test)}")

# ===== 两套 HGB 参数 =====
HGB_TASK1 = dict(
    max_iter=1000, learning_rate=0.04, max_depth=6, max_leaf_nodes=31,
    l2_regularization=0.1, early_stopping=True, n_iter_no_change=50,
    validation_fraction=0.15, random_state=42,
)

HGB_OPTIMAL = dict(
    max_iter=350, learning_rate=0.025, max_depth=5, min_samples_leaf=15,
    l2_regularization=0.15, random_state=42,
)

# ===== 特征集 =====
BASE_FEATS = [
    "Thickness", "Width", "Speed",
    "Top_Current_Sum", "Bot_Current_Sum",
    "Top_Current_Density", "Bot_Current_Density",
    "Online_SP_Diff_TOP", "Online_SP_Diff_BOT",
    "Speed_x_Thickness", "Top_Bot_Online_Diff", "Top_Bot_Current_Ratio",
    "Hour", "DayOfWeek", "Steel_Grade_enc",
    "SP_TOP", "SP_BOT",
]

# lag 构造（复用主脚本函数）
for col in ["Online_SP_Diff_TOP", "Online_SP_Diff_BOT"]:
    add_lag_safe(train, test, col, lag=5)

LAG_COLS = []
for col in ["Online_SP_Diff_TOP", "Online_SP_Diff_BOT"]:
    LAG_COLS.extend([f"{col}_lag{i}" for i in range(1, 6)])

# ===== 实验 =====
print("\n" + "="*70)
print("控制变量对比：reproduce_optimal 风格 vs Task1 风格")
print("="*70)

for surf in ["TOP", "BOT"]:
    online_col = f"Online_{surf}"
    lab_a_col = f"Lab_{surf}"            # Lab_A 单点（评估对它）
    lab_mean_col = f"Lab_{surf}_mean"   # Lab 4 点均值
    delta_a_col = f"Delta_{surf}"
    delta_mean_col = f"Delta_{surf}_mean"

    # 去掉 NaN 的公共子集
    need_common = BASE_FEATS + [online_col, lab_a_col, lab_mean_col, delta_a_col, delta_mean_col] + LAG_COLS
    tr = train.dropna(subset=need_common).copy()
    te = test.dropna(subset=need_common).copy()

    yte_lab_a = te[lab_a_col].values
    online_te = te[online_col].values

    results = []

    def run_comb(name, feats, delta_col, hgb_params, ewma_alpha=None):
        model = HistGradientBoostingRegressor(**hgb_params)
        model.fit(tr[feats].values, tr[delta_col].values)
        delta_te = model.predict(te[feats].values)
        pred_te_raw = online_te + delta_te
        m_raw = evaluate(yte_lab_a, pred_te_raw, online=online_te)
        row = {"name": name, "RMSE": m_raw["RMSE"], "MAE": m_raw["MAE"], "Bias": m_raw["Bias"],
               "Hit2%": m_raw.get("Hit2%", np.nan), "Hit3%": m_raw.get("Hit3%", np.nan)}
        if ewma_alpha is not None:
            # 先用 train 预热 init_bias
            delta_tr = model.predict(tr[feats].values)
            pred_tr_raw = tr[online_col].values + delta_tr
            _, init_bias = ewma_causal(pred_tr_raw, tr[lab_a_col].values, alpha=ewma_alpha, init_bias=0.0)
            pred_te_ewma, fb = ewma_causal(pred_te_raw, yte_lab_a, alpha=ewma_alpha, init_bias=init_bias)
            m_ewma = evaluate(yte_lab_a, pred_te_ewma, online=online_te)
            row_ewma = {"name": f"{name}+EWMA(α={ewma_alpha})", "RMSE": m_ewma["RMSE"], "MAE": m_ewma["MAE"],
                        "Bias": m_ewma["Bias"], "Hit2%": m_ewma.get("Hit2%", np.nan), "Hit3%": m_ewma.get("Hit3%", np.nan)}
            return [row, row_ewma]
        return [row]

    # 组合 1: reproduce_optimal 完整风格（无 lag + Δ_mean + optimal params + EWMA）
    # 组合 2: reproduce_optimal 无 EWMA
    results += run_comb("Optimal(无lag, Δ_mean, OptimalP)", BASE_FEATS + [online_col],
                        delta_mean_col, HGB_OPTIMAL, ewma_alpha=0.25)

    # 组合 3: Optimal params + lag + Δ_mean（其他不变，只加 lag）
    results += run_comb("Optimal(+lag, Δ_mean, OptimalP)", BASE_FEATS + LAG_COLS + [online_col],
                        delta_mean_col, HGB_OPTIMAL)

    # 组合 4: Optimal params + lag + Δ_A（训练目标换成 Lab_A 单点）
    results += run_comb("Optimal(+lag, Δ_A, OptimalP)", BASE_FEATS + LAG_COLS + [online_col],
                        delta_a_col, HGB_OPTIMAL)

    # 组合 5: Task1 完整风格（lag + Δ_A + Task1 对齐 params + 无 EWMA）
    results += run_comb("Task1(lag, Δ_A, Task1P)", BASE_FEATS + LAG_COLS + [online_col],
                        delta_a_col, HGB_TASK1)

    # 组合 6: Task1 params + Δ_mean + lag（Task1 风格只换训练目标）
    results += run_comb("Task1(lag, Δ_mean, Task1P)", BASE_FEATS + LAG_COLS + [online_col],
                        delta_mean_col, HGB_TASK1)

    # 组合 7: Raw Online baseline
    m0 = evaluate(yte_lab_a, online_te, online=online_te)
    results.append({"name": "RAW_Online", "RMSE": m0["RMSE"], "MAE": m0["MAE"], "Bias": m0["Bias"],
                    "Hit2%": m0.get("Hit2%", np.nan), "Hit3%": m0.get("Hit3%", np.nan)})

    # 打印
    print(f"\n  ===== {surf} 面 =====")
    print(f"  {'模型':<45s} {'RMSE':>9s} {'MAE':>9s} {'Hit2%':>9s} {'Hit3%':>9s}")
    print(f"  {'-'*82}")
    for r in sorted(results, key=lambda x: x["RMSE"]):
        print(f"  {r['name']:<45s} {r['RMSE']:9.4f} {r['MAE']:9.4f} {r['Hit2%']:9.4f} {r['Hit3%']:9.4f}")
