"""验证 Lab 漂移真实性: 规格组内时间是否连续 + Lab vs Online 月均值对比"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np

DATA = r"e:/SGAI_Project/2026/Tin coating/Task1/result/data/cleaned_data/cleaned_data.xlsx"
rename = {"Produce Time":"Produce_Time","Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_TOP_Avg":"SP_TOP","Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_BOT_Avg":"SP_BOT","Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Avg":"Online_TOP","Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Avg":"Online_BOT","上表面镀层重量A(XA1_0)":"Lab_TOP","下表面镀层重量A(XA1_0)":"Lab_BOT"}
df = pd.read_excel(DATA).rename(columns=rename)
df["Produce_Time"] = pd.to_datetime(df["Produce_Time"])
df["SP_TOP_r"] = df["SP_TOP"].round(3); df["SP_BOT_r"] = df["SP_BOT"].round(3)
df = df.sort_values("Produce_Time").reset_index(drop=True)
df["Delta_TOP"] = df["Lab_TOP"] - df["Online_TOP"]
df["Delta_BOT"] = df["Lab_BOT"] - df["Online_BOT"]
df["Online_SP_Diff_TOP"] = df["Online_TOP"] - df["SP_TOP"]
df["Online_SP_Diff_BOT"] = df["Online_BOT"] - df["SP_BOT"]
df["month"] = df["Produce_Time"].dt.to_period("M")
df["date"] = df["Produce_Time"].dt.date

# ========== 问题 1: 规格组内时间是否连续? ==========
print("=" * 72)
print("问题 1: 规格组 SP=2.799/2.799 在时间上是否连续?")
print("  连续 = 这个规格的卷一个接一个生产")
print("  不连续 = 混在多个规格之间, 中间隔了其他规格的卷")
print("=" * 72)

g = df[(df["SP_TOP_r"]==2.799)&(df["SP_BOT_r"]==2.799)].copy().reset_index(drop=True)
print(f"\n该规格组 N = {len(g)}")
print(f"时间范围: {g['Produce_Time'].min()} ~ {g['Produce_Time'].max()}")

# gap 分布
g["gap_next_min"] = g["Produce_Time"].shift(-1) - g["Produce_Time"]
g["gap_next_min"] = g["gap_next_min"].dt.total_seconds() / 60
print(f"\n规格组内相邻卷 gap (分钟):")
print(f"  mean={g['gap_next_min'].mean():.1f}, median={g['gap_next_min'].median():.1f}")
print(f"  p25={g['gap_next_min'].quantile(0.25):.1f}, p75={g['gap_next_min'].quantile(0.75):.1f}, p90={g['gap_next_min'].quantile(0.9):.1f}")
print(f"  > 30min 的比例: {(g['gap_next_min']>30).mean()*100:.1f}%")
print(f"  > 60min 的比例: {(g['gap_next_min']>60).mean()*100:.1f}%")
print(f"  > 120min 的比例: {(g['gap_next_min']>120).mean()*100:.1f}%")

# 更直接的问题: 这个规格是不是"一连续段", 还是分成了多段?
big_gaps = g[g["gap_next_min"] > 120]  # gap > 2h 算不连续段分界
print(f"\ngap > 2h 的断点数量 = {len(big_gaps)}")
if len(big_gaps) > 0:
    print(f"断点处时间:")
    for _, row in big_gaps.iterrows():
        idx = row.name
        print(f"  卷 #{idx}: {g.loc[idx, 'Produce_Time']} → {g.loc[idx+1, 'Produce_Time']}, gap={row['gap_next_min']:.0f}min")
    # 段长度
    seg_ids = (g["gap_next_min"].fillna(999) > 120).cumsum()
    seg_lens = g.groupby(seg_ids).size()
    print(f"\n分成了 {len(seg_lens)} 个连续段, 长度分布:")
    print(seg_lens.describe().to_string())
    print(f"最长段: {seg_lens.max()} 卷, 最短: {seg_lens.min()}")

# 另一个视角: 所有规格组的时间跨度
print("\n\n所有规格组的时间覆盖对比:")
groups = df.groupby(["SP_TOP_r", "SP_BOT_r"])
for (sp_t, sp_b), sub in groups:
    if len(sub) >= 500:
        span_days = (sub["Produce_Time"].max() - sub["Produce_Time"].min()).total_seconds() / 86400
        print(f"  SP={sp_t}/{sp_b}: N={len(sub):>5d}, 时间跨度={span_days:.0f}天, {sub['month'].nunique()}个月")

# ========== 问题 2: Lab 漂移 vs Online 漂移 ==========
print("\n" + "=" * 72)
print("问题 2: Lab 月均值漂移 vs Online 月均值漂移 —— 谁漂了?")
print("  如果 Lab 漂了但 Online 没漂 → 化验仪器/流程问题")
print("  如果两者都漂了 → 真实工艺漂移")
print("  如果 Online 漂了但 Lab 没漂 → 在线仪器问题")
print("=" * 72)

print("\n--- [TOP 面] ---")
monthly = g.groupby("month").agg(
    N=("Lab_TOP", "size"),
    Lab_mean=("Lab_TOP", "mean"), Lab_std=("Lab_TOP", "std"),
    Online_mean=("Online_TOP", "mean"), Online_std=("Online_TOP", "std"),
    Delta_mean=("Delta_TOP", "mean"),
    Onl_SP_mean=("Online_SP_Diff_TOP", "mean"),
).round(4)
monthly["Lab_Δ"] = monthly["Lab_mean"] - monthly["Lab_mean"].iloc[0]
monthly["Online_Δ"] = monthly["Online_mean"] - monthly["Online_mean"].iloc[0]
monthly["Delta_Δ"] = monthly["Delta_mean"] - monthly["Delta_mean"].iloc[0]
print(monthly.to_string())

print("\n--- [BOT 面] ---")
monthly_b = g.groupby("month").agg(
    N=("Lab_BOT", "size"),
    Lab_mean=("Lab_BOT", "mean"), Lab_std=("Lab_BOT", "std"),
    Online_mean=("Online_BOT", "mean"), Online_std=("Online_BOT", "std"),
    Delta_mean=("Delta_BOT", "mean"),
    Onl_SP_mean=("Online_SP_Diff_BOT", "mean"),
).round(4)
monthly_b["Lab_Δ"] = monthly_b["Lab_mean"] - monthly_b["Lab_mean"].iloc[0]
monthly_b["Online_Δ"] = monthly_b["Online_mean"] - monthly_b["Online_mean"].iloc[0]
monthly_b["Delta_Δ"] = monthly_b["Delta_mean"] - monthly_b["Delta_mean"].iloc[0]
print(monthly_b.to_string())

# 量化比较
for side, lab_c, onl_c, d_c in [("TOP", "Lab_TOP", "Online_TOP", "Delta_TOP"),
                                 ("BOT", "Lab_BOT", "Online_BOT", "Delta_BOT")]:
    m = g.groupby("month").agg(Lab=((lab_c, "mean")), Onl=((onl_c, "mean")), Delta=((d_c, "mean")))
    drifts = {col: m[col].max() - m[col].min() for col in ["Lab", "Onl", "Delta"]}
    stds = {col: g[col.replace("Onl", "Online") if col == "Onl" else col].std() for col in ["Lab", "Onl", "Delta"]}
    print(f"\n[{side}] 月间漂移范围 | 相对 std:")
    for col in ["Lab", "Onl", "Delta"]:
        print(f"   {col:>5s}: drift={drifts[col]:.4f} ({drifts[col]/(0.001 if drifts[col]==0 else drifts[col]) * 0:.1f}%)  vs  σ={stds[col]:.4f} → 漂移占 std 的 {(drifts[col]/stds[col]*100 if stds[col]>0 else 0):.0f}%")

print("\n" + "=" * 72)
print("初步判断准则:")
print("  如果 Lab 漂移 >> Online 漂移 → Lab 侧问题 (化验仪器/试剂/流程)")
print("  如果 Lab 漂移 ≈ Online 漂移 → 两者一起漂, 真实工艺变化")
print("  如果 Delta 漂移 ≈ Lab 漂移 - Online 漂移 → Δ 漂移方向由 Lab 和 Online 谁漂得多决定")
print("=" * 72)
