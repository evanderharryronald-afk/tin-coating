#!/usr/bin/env python3
"""
EWMA 真实生产场景模拟: 延迟 + 抽样率 + α 全扫

目的: 在 reproduce_optimal 的 HGB 模型预测基础上，
      模拟"部分抽样 + 化验延迟"的生产环境，看 EWMA 优势能保持到什么程度。

参数网格:
  抽样率:  [0.10, 0.20, 0.33, 0.50, 1.00]    ← 10%~100%
  延迟:    [0, 3, 5, 10, 20, 30]              ← 0~30 卷 (≈0~15h)
  α:       [0.10, 0.15, 0.20, 0.25, 0.30]    ← 慢→快更新
  每组合重复 10 次 (不同随机抽样 seed)

总组合: 5 × 6 × 5 × 2 (两面) × 10 = 3000 次评估
预计耗时: 几分钟 (HGB 训练只跑 1 次，剩余都是 O(N) 模拟)
"""

import sys
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
DATA_PATH = ROOT.parent / "result" / "data" / "cleaned_data" / "cleaned_data.xlsx"
print(f"数据: {DATA_PATH}")

# ===== 复用 reproduce_optimal 的数据加载 =====
spec = importlib.util.spec_from_file_location(
    "ros", ROOT.parent / "reproduce_optimal_soft_sensor" / "reproduce_optimal_soft_sensor.py")
ros = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ros)

df = ros.load_and_fe(DATA_PATH)
df = df.sort_values("Produce_Time").reset_index(drop=True)
g = df[(df["SP_TOP_r"] == 2.799) & (df["SP_BOT_r"] == 2.799)].copy().reset_index(drop=True)
print(f"SP=2.799: {len(g)} 条")

train, test = g.iloc[:int(len(g)*0.80)].copy(), g.iloc[int(len(g)*0.80):].copy()
print(f"Train: {len(train)}, Test: {len(test)}")

# ===== 特征（reproduce_optimal 无 lag 风格）=====
BASE_FEATS = [
    "Thickness", "Width", "Speed",
    "Top_Current_Sum", "Bot_Current_Sum",
    "Top_Current_Density", "Bot_Current_Density",
    "Online_SP_Diff_TOP", "Online_SP_Diff_BOT",
    "Speed_x_Thickness", "Top_Bot_Online_Diff", "Top_Bot_Current_Ratio",
    "Hour", "DayOfWeek", "Steel_Grade_enc",
    "SP_TOP", "SP_BOT",
]

HGB_OPTIMAL = dict(max_iter=350, learning_rate=0.025, max_depth=5, min_samples_leaf=15,
                   l2_regularization=0.15, random_state=42)

# ===== 网格 =====
SAMPLE_RATES = [0.10, 0.20, 0.33, 0.50, 1.00]
LAB_DELAYS   = [0, 3, 5, 10, 20, 30]
ALPHAS       = [0.10, 0.15, 0.20, 0.25, 0.30]
N_REPEATS    = 10   # 每种组合跑 10 次（随机抽样）

# ===== 核心函数：生产真实场景的 EWMA =====
def ewma_production(pred_lab, lab_true, alpha, sample_rate, lab_delay, seed):
    """
    生产环境 EWMA —— 接口和 reproduce_optimal.ewma_causal 完全一致
    
    pred_lab[i] = 模型对卷 i 的 Lab 预测 (Online + pred_delta)
    lab_true[i] = 卷 i 的真实 Lab 值
    更新公式: bias = α * (Lab - pred_lab) + (1-α) * bias
    输出: pred_lab + bias
    """
    n = len(pred_lab)
    rng = np.random.RandomState(seed)
    bias = 0.0
    pending = []
    out = np.empty(n)
    
    for i in range(n):
        out[i] = pred_lab[i] + bias
        if rng.random() < sample_rate:
            pending.append((i, lab_true[i], pred_lab[i]))
        while pending and pending[0][0] <= i - lab_delay:
            j, lab_j, pred_j = pending.pop(0)
            residual = lab_j - pred_j
            bias = alpha * residual + (1 - alpha) * bias
    return out


# ===== 主循环 =====
all_rows = []
print("\n" + "="*70)
print("EWMA 生产场景模拟：抽样率 × 延迟 × α 全扫")
print("="*70)

for surf in ["TOP", "BOT"]:
    print(f"\n--- 训练 HGB ({surf}) ---")
    online_col = f"Online_{surf}"
    lab_col = f"Lab_{surf}"
    feats = BASE_FEATS + [online_col]
    
    need = feats + [lab_col]
    tr = train.dropna(subset=need).copy()
    te = test.dropna(subset=need).copy()
    
    model = HistGradientBoostingRegressor(**HGB_OPTIMAL)
    model.fit(tr[feats].values, tr["Delta_TOP_mean" if surf == "TOP" else "Delta_BOT_mean"].values)
    
    pred_delta = model.predict(te[feats].values)
    online_te = te[online_col].values
    lab_te = te[lab_col].values
    
    # 纯模型（无 EWMA）的基线
    pred_raw_lab = online_te + pred_delta
    rmse_no_ewma = float(np.sqrt(mean_squared_error(lab_te, pred_raw_lab)))
    print(f"  纯模型基线 RMSE = {rmse_no_ewma:.4f}")
    
    # 理想 EWMA（100% 抽样，0 延迟，α=0.25）
    pred_ideal, _ = ros.ewma_causal(pred_raw_lab, lab_te, alpha=0.25, init_bias=0.0)
    rmse_ideal = float(np.sqrt(mean_squared_error(lab_te, pred_ideal)))
    print(f"  理想 EWMA (α=0.25, 100%, 延迟=0) RMSE = {rmse_ideal:.4f}")
    
    all_rows.append({"surface": surf, "sample_rate": "BASELINE", "lab_delay": 0, "alpha": 0.0,
                      "rmse_mean": rmse_no_ewma, "rmse_std": 0.0, "n_repeats": 1})
    all_rows.append({"surface": surf, "sample_rate": "IDEAL_0.25", "lab_delay": 0, "alpha": 0.25,
                      "rmse_mean": rmse_ideal, "rmse_std": 0.0, "n_repeats": 1})
    
    # 全扫
    for sr in SAMPLE_RATES:
        for delay in LAB_DELAYS:
            for alpha in ALPHAS:
                rmses = []
                for rep in range(N_REPEATS):
                    seed = 42 + rep  # 不同重复用不同 seed
                    pred_ewma = ewma_production(pred_raw_lab, lab_te,
                                                alpha=alpha, sample_rate=sr,
                                                lab_delay=delay, seed=seed)
                    rmse = float(np.sqrt(mean_squared_error(lab_te, pred_ewma)))
                    rmses.append(rmse)
                
                row = {
                    "surface": surf,
                    "sample_rate": sr,
                    "lab_delay": delay,
                    "alpha": alpha,
                    "rmse_mean": np.mean(rmses),
                    "rmse_std": np.std(rmses),
                    "rmse_best": np.min(rmses),
                    "improve_vs_baseline": rmse_no_ewma - np.mean(rmses),
                    "n_repeats": N_REPEATS,
                }
                all_rows.append(row)
                print(f"  {surf} sr={sr:.2f} delay={delay:2d} α={alpha:.2f} → "
                      f"RMSE={row['rmse_mean']:.4f}±{row['rmse_std']:.4f} "
                      f"(vs baseline -{row['improve_vs_baseline']:.4f})")

# ===== 汇总分析 =====
res = pd.DataFrame(all_rows)
out_csv = ROOT / "ewma_production_sweep.csv"
res.to_csv(out_csv, index=False)
print(f"\n结果已保存: {out_csv}")

print("\n" + "="*70)
print("汇总: 在不同场景下 EWMA 优势还剩多少？")
print("="*70)

# 每个 surface，按 (sample_rate, lab_delay) 分组，找最优 α
print("\n--- TOP 面: 每组 (抽样率, 延迟) 的最优 α 和 RMSE ---")
top = res[res["surface"] == "TOP"]
# 排除 BASELINE 和 IDEAL
top_grid = top[~top["sample_rate"].isin(["BASELINE", "IDEAL_0.25"])]
for sr in sorted(SAMPLE_RATES):
    for delay in sorted(LAB_DELAYS):
        sub = top_grid[(top_grid["sample_rate"] == sr) & (top_grid["lab_delay"] == delay)]
        if len(sub) == 0:
            continue
        best_row = sub.loc[sub["rmse_mean"].idxmin()]
        baseline_rmse = top[top["sample_rate"] == "BASELINE"]["rmse_mean"].values[0]
        print(f"  sr={sr:.0%}  delay={delay:2d}卷 → "
              f"α*={best_row['alpha']:.2f}, RMSE={best_row['rmse_mean']:.4f} "
              f"(比纯模型好 {best_row['improve_vs_baseline']:.4f})")

print("\n--- BOT 面: 每组 (抽样率, 延迟) 的最优 α 和 RMSE ---")
bot = res[res["surface"] == "BOT"]
bot_grid = bot[~bot["sample_rate"].isin(["BASELINE", "IDEAL_0.25"])]
for sr in sorted(SAMPLE_RATES):
    for delay in sorted(LAB_DELAYS):
        sub = bot_grid[(bot_grid["sample_rate"] == sr) & (bot_grid["lab_delay"] == delay)]
        if len(sub) == 0:
            continue
        best_row = sub.loc[sub["rmse_mean"].idxmin()]
        print(f"  sr={sr:.0%}  delay={delay:2d}卷 → "
              f"α*={best_row['alpha']:.2f}, RMSE={best_row['rmse_mean']:.4f} "
              f"(比纯模型好 {best_row['improve_vs_baseline']:.4f})")

# ===== 关键场景对比表 =====
print("\n" + "="*70)
print("关键场景：EWMA 还能不能帮上忙？")
print("="*70)
key_scenarios = [
    (1.00, 0,   "理想"),
    (0.50, 0,   "50% 抽样 0延迟"),
    (0.50, 5,   "50% 抽样 延迟5卷"),
    (0.33, 0,   "33% 抽样 0延迟"),
    (0.33, 5,   "33% 抽样 延迟5卷"),
    (0.20, 5,   "20% 抽样 延迟5卷"),
    (0.20, 10,  "20% 抽样 延迟10卷"),
    (0.10, 10,  "10% 抽样 延迟10卷"),
    (0.10, 30,  "10% 抽样 延迟30卷"),
]

for surf in ["TOP", "BOT"]:
    sub = res[(res["surface"] == surf)]
    baseline = sub[sub["sample_rate"] == "BASELINE"]["rmse_mean"].values[0]
    ideal = sub[sub["sample_rate"] == "IDEAL_0.25"]["rmse_mean"].values[0]
    print(f"\n  {surf} 面 (baseline={baseline:.4f}, ideal={ideal:.4f}):")
    print(f"  {'场景':<22s} {'最优α':>6s} {'RMSE':>8s} {'比纯模型':>8s} {'比理想':>8s}")
    print(f"  {'-'*56}")
    for sr, delay, label in key_scenarios:
        grid = sub[~sub["sample_rate"].isin(["BASELINE", "IDEAL_0.25"])]
        row = grid[(grid["sample_rate"] == sr) & (grid["lab_delay"] == delay)].loc[
            grid[(grid["sample_rate"] == sr) & (grid["lab_delay"] == delay)]["rmse_mean"].idxmin()]
        print(f"  {label:<22s} {row['alpha']:6.2f} {row['rmse_mean']:8.4f} "
              f"{row['improve_vs_baseline']:+8.4f} {row['rmse_mean']-ideal:+8.4f}")
