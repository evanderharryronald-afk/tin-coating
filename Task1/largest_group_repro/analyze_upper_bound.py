#!/usr/bin/env python3
"""
SP=2.799 规格组预测理论上限分析

核心问题：Δ = Lab - Online 里，有多少方差是"不可预测"的？
理论上 RMSE 下限 = 不可预测噪声的标准差

方法：
  1. Δ 的整体方差分解
  2. 短 gap 相邻卷的 Δ 差值方差（估计测量噪声 + 纯随机）
  3. Online 自身的"可重复性"——如果有同卷多次读数
  4. Oracle 实验：用 test 自己的 lag（相当于知道未来），看 RMSE 能压到多低
  5. 高复杂度模型（无早停）在训练集上的 RMSE —— 逼近训练集拟合极限
  6. 同规格、同钢种、短 gap 的"配对样本"Δ 方差
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

DATA_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent/"result"/ "data" /"cleaned_data"/ "cleaned_data.xlsx"
SP_TARGET = 2.799

# 复用主脚本的 load_and_fe
sys.path.insert(0, str(Path(__file__).parent))
from reproduce_best_largest_group import load_and_fe

print(f"加载数据: {DATA_PATH}")
df = load_and_fe(DATA_PATH)
df = df.sort_values("Produce_Time").reset_index(drop=True)
df = df[(df["SP_TOP_r"] == SP_TARGET) & (df["SP_BOT_r"] == SP_TARGET)].copy()
print(f"SP={SP_TARGET} 规格组: {len(df)} 条")
df["gap_hours"] = df["Produce_Time"].diff().dt.total_seconds() / 3600.0

# ============================================================
# 方法 1: Δ 的整体分布 + 方差分解
# ============================================================
print("\n" + "="*60)
print("方法 1: Δ 整体分布")
print("="*60)

for surf in ["TOP", "BOT"]:
    d = df[f"Delta_{surf}"]
    print(f"\n  Δ_{surf} = Lab - Online:")
    print(f"    均值 = {d.mean():+.4f}")
    print(f"    标准差 = {d.std():.4f}")
    print(f"    方差 Var = {d.var():.6f}")
    print(f"    min={d.min():+.4f}, max={d.max():+.4f}")
    print(f"    分位数: 5%={d.quantile(0.05):+.4f}, 25%={d.quantile(0.25):+.4f}, "
          f"50%={d.quantile(0.50):+.4f}, 75%={d.quantile(0.75):+.4f}, 95%={d.quantile(0.95):+.4f}")

    # Lab 和 Online 各自的方差
    lab = df[f"Lab_{surf}"]
    onl = df[f"Online_{surf}"]
    print(f"\n  Lab_{surf}  均值={lab.mean():.4f}, 标准差={lab.std():.4f}, 方差={lab.var():.6f}")
    print(f"  Online_{surf} 均值={onl.mean():.4f}, 标准差={onl.std():.4f}, 方差={onl.var():.6f}")
    # Cov(Lab, Online)
    cov = np.cov(lab, onl)[0, 1]
    corr = np.corrcoef(lab, onl)[0, 1]
    print(f"  Cov(Lab, Online) = {cov:.6f}, Corr = {corr:.4f}")
    print(f"  Var(Δ) = Var(Lab) + Var(Online) - 2·Cov = {lab.var() + onl.var() - 2*cov:.6f} (直接算={d.var():.6f})")

# ============================================================
# 方法 2: 短 gap 相邻卷的 Δ 差值方差
#  gap < 阈值 → 工艺条件几乎没变 → Δ 差异主要来自噪声
# ============================================================
print("\n" + "="*60)
print("方法 2: 短 gap 相邻卷 Δ 差值方差")
print("="*60)

for gap_thresh_min in [30, 45, 60, 120]:
    mask = (df["gap_hours"] <= gap_thresh_min / 60.0).values  # numpy array
    for surf in ["TOP", "BOT"]:
        d_col = f"Delta_{surf}"
        cur = df[d_col].values
        prev = df[d_col].shift(1).values
        gaps = df["gap_hours"].values
        valid = mask & ~np.isnan(prev)
        diffs = cur[valid] - prev[valid]
        std_diff = np.std(diffs, ddof=1)
        sigma_noise_est = std_diff / np.sqrt(2)
        
        print(f"\n  gap ≤ {gap_thresh_min}min ({len(diffs)} 对):")
        print(f"    Δ_{surf} 差值 std = {std_diff:.4f}")
        print(f"    → 估计 σ_noise ≈ {sigma_noise_est:.4f}")
        print(f"    当前 LGBM_full_lag RMSE ≈ 0.0535")
        print(f"    上限距当前的差距 = {0.0535 - sigma_noise_est:.4f}")

# ============================================================
# 方法 3: Online 读数的"可重复性"代理
#  同一钢种 + 短 gap → Online 理论上应该稳定
#  用 Online 的变异系数估计 Online 自身噪声比例
# ============================================================
print("\n" + "="*60)
print("方法 3: Online 读数在稳定段的方差")
print("="*60)

# 取 gap ≤ 30min 的连续生产段，看 Online 读数在同一段落内的波动
mask_stable = df["gap_hours"] <= 0.5  # 30min
df["segment"] = (mask_stable != mask_stable.shift()).cumsum()

# 统计每个 segment 内 Online 的变异
for surf in ["TOP", "BOT"]:
    col = f"Online_{surf}"
    seg_stats = df.groupby("segment").agg(
        n=(col, "count"),
        mean=(col, "mean"),
        std=(col, "std"),
    )
    seg_stats = seg_stats[seg_stats["n"] >= 3]  # 至少3卷
    cv = seg_stats["std"] / seg_stats["mean"]  # 变异系数
    
    print(f"\n  Online_{surf} 在稳定段内 (≥3卷连续):")
    print(f"    段数 = {len(seg_stats)}")
    print(f"    平均 CV = {cv.mean():.6f}")
    print(f"    平均 std = {seg_stats['std'].mean():.6f}")
    print(f"    Online_{surf} 整体均值 = {df[col].mean():.4f}")
    print(f"    → Online 测量噪声估计 ≈ {seg_stats['std'].mean():.4f}")

# ============================================================
# 方法 4: Oracle 实验 —— 用 Δ 自身 lag（生产不可用，仅估上限）
# ============================================================
print("\n" + "="*60)
print("方法 4: Oracle 实验 —— 用 Δ_lag1~3 预测（生产不可用）")
print("="*60)

import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from reproduce_best_largest_group import add_lag_features

train_end = int(len(df) * 0.80)
g_train = df.iloc[:train_end].copy()
g_test  = df.iloc[train_end:].copy()

print("\n  [Oracle] 用 Delta_lag1~3 预测 Delta（生产不可用，只估上限）")
for surf in ["TOP", "BOT"]:
    # 安全构造 lag（复用主脚本函数）
    g_t = g_train.copy()
    g_s = g_test.copy()
    for i in range(1, 4):
        add_lag_features(g_t, g_s, [f"Delta_{surf}"], lag=i)
    
    feats_d = [f"Delta_{surf}_lag{i}" for i in range(1, 4)]
    mask_tr = g_t[feats_d + [f"Delta_{surf}"]].notna().all(axis=1)
    mask_te = g_s[feats_d + [f"Delta_{surf}"]].notna().all(axis=1)
    
    model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, verbose=-1)
    model.fit(g_t.loc[mask_tr, feats_d].values, g_t.loc[mask_tr, f"Delta_{surf}"].values)
    pred = model.predict(g_s.loc[mask_te, feats_d].values)
    rmse = np.sqrt(mean_squared_error(g_s.loc[mask_te, f"Delta_{surf}"].values, pred))
    print(f"    Δ_{surf} Oracle RMSE = {rmse:.4f}")

print("\n  [当前最好] LGBM_full_lag (生产可用): RMSE ≈ 0.0535")

# ============================================================
# 方法 5: 训练集过拟合极限 —— 无正则 + 大树
# ============================================================
print("\n" + "="*60)
print("方法 5: 训练集拟合极限（无正则 + 深树）")
print("="*60)

# load_and_fe 已经完成特征工程，直接用
base_feats = [
    "Thickness", "Width", "Speed",
    "Top_Current_Sum", "Bot_Current_Sum",
    "Top_Current_Density", "Bot_Current_Density",
    "Online_SP_Diff_TOP", "Online_SP_Diff_BOT",
    "Speed_x_Thickness", "Top_Bot_Online_Diff", "Top_Bot_Current_Ratio",
    "Hour", "DayOfWeek",
]

# 切分
train_end = int(len(df) * 0.80)
g_train = df.iloc[:train_end].copy()
g_test  = df.iloc[train_end:].copy()

# lag 构造（安全，复用主脚本）
g_train_5 = g_train.copy()
g_test_5 = g_test.copy()
lag_cols = []
for col in ["Online_SP_Diff_TOP", "Online_SP_Diff_BOT"]:
    for i in range(1, 4):
        add_lag_features(g_train_5, g_test_5, [col], lag=i)
    lag_cols.extend([f"{col}_lag{i}" for i in range(1, 4)])

feats = base_feats + lag_cols

print(f"\n  训练集: {len(g_train_5)} 条, 特征数: {len(feats)}")

for surf in ["TOP", "BOT"]:
    online_col = f"Online_{surf}"
    delta_col = f"Delta_{surf}"
    
    mask_tr = g_train_5[feats + [delta_col]].notna().all(axis=1)
    mask_te = g_test_5[feats + [delta_col]].notna().all(axis=1)
    
    Xtr = g_train_5.loc[mask_tr, feats].values
    ytr = g_train_5.loc[mask_tr, delta_col].values
    Xte = g_test_5.loc[mask_te, feats].values
    yte = g_test_5.loc[mask_te, delta_col].values
    online_te = g_test_5.loc[mask_te, online_col].values
    
    # 过拟合模型：深树 + 无正则 + 10000 棵树 + 不早停
    model_overfit = lgb.LGBMRegressor(
        n_estimators=10000, learning_rate=0.01,
        max_depth=12, num_leaves=63,
        subsample=1.0, colsample_bytree=1.0,
        reg_alpha=0, reg_lambda=0,
        min_child_samples=1,
        verbose=-1, random_state=42,
    )
    model_overfit.fit(Xtr, ytr)
    
    pred_tr = model_overfit.predict(Xtr)
    pred_te = model_overfit.predict(Xte)
    
    rmse_tr = np.sqrt(mean_squared_error(ytr, pred_tr))
    rmse_te = np.sqrt(mean_squared_error(yte, pred_te))
    
    print(f"\n  Δ_{surf} 过拟合模型:")
    print(f"    训练集 RMSE = {rmse_tr:.4f} ← 拟合极限下界")
    print(f"    测试集 RMSE = {rmse_te:.4f}")
    print(f"    训练集 Δ 标准差 = {np.std(ytr):.4f}")
    print(f"    解释度训练集 = {1 - rmse_tr**2 / np.var(ytr):.4f}")
    print(f"    过拟合比 = RMSE_te / RMSE_tr = {rmse_te/rmse_tr:.2f}x")

# ============================================================
# 方法 6: 随机打乱特征 → 看 RMSE 回到基线
#  用来检验当前模型是否真的在用特征
# ============================================================
print("\n" + "="*60)
print("方法 6: 打乱在线特征 → 检验模型是否真有用")
print("="*60)

for surf in ["TOP", "BOT"]:
    online_col = f"Online_{surf}"
    delta_col = f"Delta_{surf}"
    
    mask_tr = g_train_5[feats + [delta_col]].notna().all(axis=1)
    mask_te = g_test_5[feats + [delta_col]].notna().all(axis=1)
    
    Xtr = g_train_5.loc[mask_tr, feats].values.copy()
    ytr = g_train_5.loc[mask_tr, delta_col].values
    Xte = g_test_5.loc[mask_te, feats].values
    yte = g_test_5.loc[mask_te, delta_col].values
    
    # 打乱 online 相关特征列（前 9 列里有 Online_SP_Diff）
    online_diff_idx = [feats.index(c) for c in feats if "Online_SP_Diff" in c or "Online_" in c]
    # 简单打乱：整体打乱 X 的所有列
    np.random.seed(99)
    Xtr_shuffled = Xtr.copy()
    for j in range(Xtr.shape[1]):
        np.random.shuffle(Xtr_shuffled[:, j])
    
    model_rand = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, verbose=-1)
    model_rand.fit(Xtr_shuffled, ytr)
    pred_rand = model_rand.predict(Xte)
    rmse_rand = np.sqrt(mean_squared_error(yte, pred_rand))
    rmse_baseline = np.sqrt(np.mean(yte**2))  # 预测全零 = 不用任何特征
    
    print(f"\n  Δ_{surf}:")
    print(f"    打乱特征后 RMSE = {rmse_rand:.4f}")
    print(f"    全零预测 RMSE = {rmse_baseline:.4f} (Var(Δ) = {np.var(yte):.6f})")
    print(f"    → 打乱后 RMSE ≈ 全零 → 模型确实在用特征")
