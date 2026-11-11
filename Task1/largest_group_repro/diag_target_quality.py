"""
目标值 (Lab) 本身的质量诊断
==================================
核心问题:
  1. Lab 测量噪声多大? (和 Online 噪声分开)
  2. Lab 有没有系统性偏移 / 慢漂移?
  3. Lab 值和 Online 值的时间对齐是不是错了?  (致命!)
     → 互信息应该在 lag=0 最大; 如果在 lag=±k 最大 → 时滞 k 卷

运行:
  python diag_target_quality.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).parent.parent / "result" / "data" / "cleaned_data" / "cleaned_data.xlsx"

RENAME = {
    "Steel Grade": "Steel_Grade",
    "Produce Time": "Produce_Time",
    "Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_TOP_Avg": "SP_TOP",
    "Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_BOT_Avg": "SP_BOT",
    "Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Avg": "Online_TOP",
    "Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Avg": "Online_BOT",
    "上表面镀层重量A(XA1_0)": "Lab_TOP",
    "下表面镀层重量A(XA1_0)": "Lab_BOT",
}

print("=" * 72)
print("目标值 (Lab) 质量诊断")
print("=" * 72)

df = pd.read_excel(DATA_PATH).rename(columns=RENAME)
df["Produce_Time"] = pd.to_datetime(df["Produce_Time"])
df["SP_TOP_r"] = df["SP_TOP"].round(3)
df["SP_BOT_r"] = df["SP_BOT"].round(3)
df = df.sort_values("Produce_Time").reset_index(drop=True)
df["Delta_TOP"] = df["Lab_TOP"] - df["Online_TOP"]
df["Delta_BOT"] = df["Lab_BOT"] - df["Online_BOT"]
df["Online_SP_Diff_TOP"] = df["Online_TOP"] - df["SP_TOP"]
df["Online_SP_Diff_BOT"] = df["Online_BOT"] - df["SP_BOT"]

print(f"\n全量 SP_TOP_r 唯一值 (前 10): {df['SP_TOP_r'].value_counts().head(10).to_dict()}")
print(f"全量 SP_BOT_r 唯一值 (前 10): {df['SP_BOT_r'].value_counts().head(10).to_dict()}")

# —— 取最大规格组 (按主脚本同款过滤) ——
g = df[(df["SP_TOP_r"] == 2.799) & (df["SP_BOT_r"] == 2.799)].copy()
g = g.dropna(subset=["Delta_TOP", "Delta_BOT", "Online_SP_Diff_TOP", "Online_SP_Diff_BOT"]).reset_index(drop=True)
print(f"\n最大规格组样本数 N = {len(g)}")

# ========================================================
# ① Lab vs Online 的基础统计 + 整体相关性
# ========================================================
print("\n" + "=" * 72)
print("① Lab vs Online 基础统计")
print("=" * 72)
for side in ["TOP", "BOT"]:
    lab = g[f"Lab_{side}"]
    onl = g[f"Online_{side}"]
    d = g[f"Delta_{side}"]
    corr = np.corrcoef(lab, onl)[0, 1]
    print(f"\n  [{side}]")
    print(f"    Lab  mean={lab.mean():.4f}  std={lab.std():.4f}  min={lab.min():.3f}  max={lab.max():.3f}")
    print(f"    Online mean={onl.mean():.4f}  std={onl.std():.4f}  min={onl.min():.3f}  max={onl.max():.3f}")
    print(f"    Delta mean={d.mean():+.4f}  std={d.std():.4f}")
    print(f"    Corr(Lab, Online) = {corr:.3f}")
    print(f"    Corr(Δ, Online_SP_Diff) = {np.corrcoef(d, g[f'Online_SP_Diff_{side}'])[0,1]:.3f}")

    # 关键: Lab 和 SP 的相关性 (Lab 应该以 SP 为中心波动)
    sp = g[f"SP_{side}"]
    print(f"    Corr(Lab, SP) = {np.corrcoef(lab, sp)[0,1]:.3f}  (预期≈1.0, 因同规格组 SP 几乎不变)")

# ========================================================
# ② 核心: 时间对齐诊断 —— MI(Δ_t, Online_SP_Diff_{t+k}) 扫 k
# ========================================================
print("\n" + "=" * 72)
print("② 时间对齐诊断 —— MI(Δ_t, Online_SP_Diff_{t+k}) 在 k=−10..+10 的分布")
print("  原理: 如果 Lab 时间对齐正确, MI 应该在 k=0 最大")
print("        如果最大在 k=+3 → Lab 滞后 3 卷才出来, 我们却当成同卷真值 (致命!)")
print("=" * 72)

from sklearn.feature_selection import mutual_info_regression

def mi_shift_check(delta_series, online_diff_series, k_range=10):
    """
    扫 k = -k_range .. +k_range, 对每个 k:
      当 k >= 0: MI( Δ_t, Online_{t-k} )   ← 当前偏差 vs k 卷前的 Online
      当 k < 0:  MI( Δ_t, Online_{t+|k|} ) ← 当前偏差 vs k 卷后的 Online
    """
    results = []
    xd = delta_series.values.astype(float)
    xo = online_diff_series.values.astype(float)
    n = len(xd)
    for k in range(-k_range, k_range + 1):
        if k >= 0:
            # 对齐: Δ_t 用 Online_{t-k}, t 从 k 到 n-1
            y = xd[k:]
            x = xo[:-k].reshape(-1, 1) if k > 0 else xo.reshape(-1, 1)
        else:
            # k=-3: Δ_t 用 Online_{t+3}, t 从 0 到 n-4
            shift = -k
            y = xd[:-shift]
            x = xo[shift:].reshape(-1, 1)
        if len(y) < 100:
            continue
        mi = mutual_info_regression(x, y, random_state=42, discrete_features=False)[0]
        pearson = np.corrcoef(x.ravel(), y)[0, 1]
        results.append({"k": k, "MI": mi, "Pearson": pearson})
    return pd.DataFrame(results)

for side in ["TOP", "BOT"]:
    print(f"\n  [{side}]")
    res = mi_shift_check(g[f"Delta_{side}"], g[f"Online_SP_Diff_{side}"], k_range=10)
    # 归一化 MI 到 [0,1] 以便画图式阅读
    mi_max = res["MI"].max()
    res["MI_norm"] = res["MI"] / mi_max

    # 找最优点
    best_k = res.loc[res["MI"].idxmax(), "k"]
    k0_mi = res.loc[res["k"] == 0, "MI"].values[0]
    print(f"    {'k':>4}  {'MI':>8}  {'MI_norm':>8}  {'Pearson':>10}")
    print(f"    {'-'*34}")
    for _, r in res.iterrows():
        marker = " ← BEST" if r["k"] == best_k else (" ← k=0 (对齐)" if r["k"] == 0 else "")
        bar = "█" * int(round(r["MI_norm"] * 20))
        print(f"    {int(r['k']):>4}  {r['MI']:>8.4f}  {r['MI_norm']:>8.3f}  {r['Pearson']:>+10.3f}  {bar}{marker}")

    if best_k != 0:
        print(f"\n  ⚠️⚠️⚠️  警告: MI 最大值在 k={best_k}, 不在 k=0 !")
        print(f"     这意味着 Lab 值和 Online 值的时间对齐可能有 {best_k} 卷的偏移")
        print(f"     如果 k>0 → Lab 滞后 (Lab[i] 实际对应的是 Online[i-k])")
        print(f"     如果 k<0 → Lab 超前 (Lab[i] 实际对应的是 Online[i+|k|])")
        print(f"     这会导致模型学到伪模式, 所有诊断结论都要重新审视!")
    else:
        print(f"\n  ✅ MI 最大值在 k=0, 时间对齐看起来正确 (Lab[i] 对应 Online[i])")

    # 比值: MI(k=0) / MI(k=1) 和 MI(k=0) / MI(k=-1)
    k1 = res[res["k"] == 1]["MI"].values[0] if (res["k"] == 1).any() else np.nan
    km1 = res[res["k"] == -1]["MI"].values[0] if (res["k"] == -1).any() else np.nan
    print(f"    MI(k=0)/MI(k=+1) = {k0_mi/k1:.2f}" if not np.isnan(k1) else f"    MI(k=+1) 无数据")
    print(f"    MI(k=0)/MI(k=-1) = {k0_mi/km1:.2f}" if not np.isnan(km1) else f"    MI(k=-1) 无数据")

# ========================================================
# ③ Lab vs Online 在稳定段 (gap <= 30min) 的"双噪声"估计
#   目标: 分开 Lab 噪声 vs Online 噪声
#   方法: 假设短 gap 内工艺不变 → Δ 变 = Lab 噪声 + Online 噪声 (近似独立)
#   同时: Online 在同一稳定段内的波动 ≈ Online 噪声 (不含 Lab)
#   所以: σ(Δ_diff)² ≈ σ_Lab² + σ_Online²  而 σ_Online 从稳定段单独估
# ========================================================
print("\n" + "=" * 72)
print("③ Lab 测量噪声 vs Online 测量噪声 分开估计")
print("  方法: gap<=30min 稳定段, Online 同段波动=Online噪声, Δ差波动=Lab+Online噪声")
print("=" * 72)

def stable_segment_noise(df_sub, gap_max_min=30):
    """找连续 gap <= gap_max_min 的段, 估 σ_Lab 和 σ_Online"""
    pt = df_sub["Produce_Time"].values
    onl_t = df_sub["Online_TOP"].values
    onl_b = df_sub["Online_BOT"].values
    lab_t = df_sub["Lab_TOP"].values
    lab_b = df_sub["Lab_BOT"].values
    d_t = df_sub["Delta_TOP"].values
    d_b = df_sub["Delta_BOT"].values

    # 找段
    n = len(df_sub)
    gap = np.zeros(n - 1)
    for i in range(n - 1):
        gap[i] = (pd.Timestamp(pt[i + 1]) - pd.Timestamp(pt[i])).total_seconds() / 60

    # 段标记
    segs = []
    seg_start = 0
    for i in range(n - 1):
        if gap[i] > gap_max_min:
            if i - seg_start + 1 >= 3:
                segs.append((seg_start, i + 1))
            seg_start = i + 1
    if n - seg_start >= 3:
        segs.append((seg_start, n))

    # 每段内部算: Online std + Δ std 差
    online_top_vars = []
    online_bot_vars = []
    delta_top_vars = []
    delta_bot_vars = []

    for s, e in segs:
        if e - s < 3:
            continue
        seg_onl_t = onl_t[s:e]
        seg_onl_b = onl_b[s:e]
        seg_d_t = d_t[s:e]
        seg_d_b = d_b[s:e]
        online_top_vars.append(np.var(seg_onl_t, ddof=1))
        online_bot_vars.append(np.var(seg_onl_b, ddof=1))
        # Δ 在段内的变化量 (相邻差的方差, 段长 >= 2)
        if e - s >= 2:
            delta_top_vars.append(np.var(np.diff(seg_d_t), ddof=1))
            delta_bot_vars.append(np.var(np.diff(seg_d_b), ddof=1))

    σ_onl_t = np.sqrt(np.mean(online_top_vars))
    σ_onl_b = np.sqrt(np.mean(online_bot_vars))
    # Δ 相邻差的方差 ≈ 2·(σ_Lab² + σ_Online²)  (独立噪声)
    # 但这里实际还包含真实工艺变化, 所以是上界
    σ_delta_diff_t = np.sqrt(np.mean(delta_top_vars))
    σ_delta_diff_b = np.sqrt(np.mean(delta_bot_vars))

    return {
        "n_segs": len(segs),
        "σ_Online_TOP": σ_onl_t,
        "σ_Online_BOT": σ_onl_b,
        "σ_Δ_diff_TOP": σ_delta_diff_t,
        "σ_Δ_diff_BOT": σ_delta_diff_b,
    }

noise = stable_segment_noise(g, gap_max_min=30)
print(f"\n  稳定段数 (gap<=30min, 长>=3): {noise['n_segs']}")
print(f"  {'':>12s}  {'TOP':>10s}  {'BOT':>10s}")
print(f"  {'-'*32}")
print(f"  σ_Online      {noise['σ_Online_TOP']:>10.4f}  {noise['σ_Online_BOT']:>10.4f}  ← 在线仪器短期波动")
print(f"  σ_Δ_diff      {noise['σ_Δ_diff_TOP']:>10.4f}  {noise['σ_Δ_diff_BOT']:>10.4f}  ← Δ 相邻卷差 (Lab+Online+工艺)")
print(f"  σ_Δ_diff/√2   {noise['σ_Δ_diff_TOP']/np.sqrt(2):>10.4f}  {noise['σ_Δ_diff_BOT']/np.sqrt(2):>10.4f}  ← 按相邻独立拆方差")
print(f"  Lab 噪声上界   ≤{(noise['σ_Δ_diff_TOP']/np.sqrt(2)):.4f}  ≤{(noise['σ_Δ_diff_BOT']/np.sqrt(2)):.4f}  ← 减去 Online 后")

# ========================================================
# ④ Lab 的时间漂移: 按时间窗口分组看 Lab 均值是否稳定
# ========================================================
print("\n" + "=" * 72)
print("④ Lab 均值是否有时间漂移? (按月分组)")
print("  固定偏移已经被 Bias_shifted 解决, 但如果 Lab 均值随时间慢漂移,")
print("  意味着化验仪器或试剂本身在变, Δ 目标值的分布就不稳定")
print("=" * 72)

g["month"] = g["Produce_Time"].dt.to_period("M")
for side in ["TOP", "BOT"]:
    monthly = g.groupby("month")[f"Lab_{side}"].agg(["mean", "std", "count"])
    print(f"\n  [{side}]")
    print(f"  {'月份':>10s}  {'N':>6s}  {'Lab_mean':>10s}  {'Lab_std':>10s}  {'Δ_mean':>10s}")
    print(f"  {'-'*50}")
    for m, row in monthly.iterrows():
        dm = g[g["month"] == m][f"Delta_{side}"].mean()
        print(f"  {str(m):>10s}  {int(row['count']):>6d}  {row['mean']:>10.4f}  {row['std']:>10.4f}  {dm:>+10.4f}")

    monthly_means = g.groupby("month")[f"Lab_{side}"].mean()
    drift_range = monthly_means.max() - monthly_means.min()
    print(f"\n  Lab 均值的月间漂移范围 = {drift_range:.4f}")
    print(f"  相对于 Lab std={g[f'Lab_{side}'].std():.4f}, 漂移占比 = {drift_range / g[f'Lab_{side}'].std() * 100:.1f}%")
    if drift_range > 0.03:
        print(f"  ⚠️ Lab 均值随时间漂移明显 (>0.03), 目标值分布不稳定")
    else:
        print(f"  ✅ Lab 均值相对稳定")

# ========================================================
# ⑤ 小结
# ========================================================
print("\n" + "=" * 72)
print("小结")
print("=" * 72)
print("""
  核心检查项 (按严重程度):
    🔴 时间对齐诊断 (MI 最大值在哪个 k) —— 致命, 如果不是 k=0
    🟡 Lab 测量噪声 —— 决定物理下界
    🟡 Lab 月间漂移 —— 决定是否需要目标对齐/分段建模
    🟢 固定偏移 —— Bias_shifted 已经解决
""")
