"""目标值质量诊断 Part 2: 噪声分离 + Lab 月漂移"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np

DATA = r"e:/SGAI_Project/2026/Tin coating/Task1/result/data/cleaned_data/cleaned_data.xlsx"
rename = {"Produce Time":"Produce_Time","Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_TOP_Avg":"SP_TOP","Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_BOT_Avg":"SP_BOT","Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Avg":"Online_TOP","Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Avg":"Online_BOT","上表面镀层重量A(XA1_0)":"Lab_TOP","下表面镀层重量A(XA1_0)":"Lab_BOT"}
df = pd.read_excel(DATA).rename(columns=rename)
df["Produce_Time"] = pd.to_datetime(df["Produce_Time"])
df["SP_TOP_r"] = df["SP_TOP"].round(3); df["SP_BOT_r"] = df["SP_BOT"].round(3)
df = df.sort_values("Produce_Time").reset_index(drop=True)
g = df[(df["SP_TOP_r"]==2.799)&(df["SP_BOT_r"]==2.799)].copy()
g["Delta_TOP"] = g["Lab_TOP"] - g["Online_TOP"]
g["Delta_BOT"] = g["Lab_BOT"] - g["Online_BOT"]

# ③ 稳定段噪声分离
print("="*60)
print("③ Lab vs Online 噪声分开估计")
print("="*60)
g["gap_min"] = g["Produce_Time"].diff().dt.total_seconds()/60
g["seg"] = (g["gap_min"].fillna(999) > 30).cumsum()
segs = g.groupby("seg").filter(lambda x: len(x) >= 3).groupby("seg")
agg = segs.agg(n=("seg","size"), OnlT_var=("Online_TOP","var"), OnlB_var=("Online_BOT","var"),
               DT_var=("Delta_TOP", lambda x: np.var(np.diff(x.values), ddof=1) if len(x)>=2 else np.nan),
               DB_var=("Delta_BOT", lambda x: np.var(np.diff(x.values), ddof=1) if len(x)>=2 else np.nan))
agg = agg.dropna()
print(f"稳定段数 (gap<=30min, 长>=3): {len(agg)}")
sig_onl_t = np.sqrt(agg["OnlT_var"].mean())
sig_onl_b = np.sqrt(agg["OnlB_var"].mean())
sig_dt = np.sqrt(agg["DT_var"].mean())/np.sqrt(2)
sig_db = np.sqrt(agg["DB_var"].mean())/np.sqrt(2)
print(f"{'':>20s}  {'TOP':>10s}  {'BOT':>10s}")
print(f"{'σ_Online (稳定段内 std)':>20s}  {sig_onl_t:>10.4f}  {sig_onl_b:>10.4f}")
print(f"{'σ_Δ (相邻卷差)/√2':>20s}  {sig_dt:>10.4f}  {sig_db:>10.4f}  (Lab+Online+工艺噪声上界)")
print(f"{'σ_Lab_noise 上界 (Δ-Online)':>20s}  {max(sig_dt**2 - sig_onl_t**2, 0)**0.5:>10.4f}  {max(sig_db**2 - sig_onl_b**2, 0)**0.5:>10.4f}")

# ④ Lab 月漂移
print("\n"+"="*60)
print("④ Lab 月间漂移")
print("="*60)
g["month"] = g["Produce_Time"].dt.to_period("M")
for side in ["TOP","BOT"]:
    m = g.groupby("month")[f"Lab_{side}"].agg(["mean","count"])
    print(f"\nLab_{side} 月均值:")
    print(m.to_string())
    drift = m["mean"].max() - m["mean"].min()
    s = g[f"Lab_{side}"].std()
    print(f"  月间漂移范围 = {drift:.4f}, 相对 Lab std={s:.4f} 的占比 = {drift/s*100:.1f}%")
    if drift > 0.03:
        print(f"  ⚠️ Lab 均值随时间漂移明显, 目标值分布不稳定")
    else:
        print(f"  ✅ Lab 均值相对稳定")

# ⑤ MI(k=0) 是不是远大于 MI(k=1)? 如果比值≈1 说明还有问题
print("\n"+"="*60)
print("⑤ 时间对齐质量量化 (来自 Part 1 输出)")
print("="*60)
print("""
  TOP: MI(k=0)=0.378 (BEST), MI(k=1)=0.305, MI(k=-1)=0.283
       MI(k=0)/MI(k=+1)=1.24, MI(k=0)/MI(k=-1)=1.33
  
  BOT: Part 1 脚本中断前打印了前半段, 待确认
  
  结论: 如果 MI(k=0) 明显大于 k=±1 (比值 > 1.1), 时间对齐可信
        如果 MI(k=0) ≈ MI(k=1) ≈ MI(k=-1), 说明要么对齐没信号, 要么真实对齐有 k 卷偏移
""")
