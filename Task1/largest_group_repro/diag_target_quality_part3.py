"""目标值质量诊断 Part 3: BOT 时间对齐扫描"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from sklearn.feature_selection import mutual_info_regression

DATA = r"e:/SGAI_Project/2026/Tin coating/Task1/result/data/cleaned_data/cleaned_data.xlsx"
rename = {"Produce Time":"Produce_Time","Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_TOP_Avg":"SP_TOP","Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_BOT_Avg":"SP_BOT","Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Avg":"Online_TOP","Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Avg":"Online_BOT","上表面镀层重量A(XA1_0)":"Lab_TOP","下表面镀层重量A(XA1_0)":"Lab_BOT"}
df = pd.read_excel(DATA).rename(columns=rename)
df["Produce_Time"] = pd.to_datetime(df["Produce_Time"])
df["SP_TOP_r"] = df["SP_TOP"].round(3); df["SP_BOT_r"] = df["SP_BOT"].round(3)
df = df.sort_values("Produce_Time").reset_index(drop=True)
g = df[(df["SP_TOP_r"]==2.799)&(df["SP_BOT_r"]==2.799)].copy()
g["Db"] = g["Lab_BOT"] - g["Online_BOT"]
g["Ob"] = g["Online_BOT"] - g["SP_BOT"]

xd = g["Db"].values.astype(float)
xo = g["Ob"].values.astype(float)
n = len(xd)
results = []
for k in range(-10, 11):
    if k >= 0:
        y = xd[k:]
        x = xo[:-k].reshape(-1,1) if k > 0 else xo.reshape(-1,1)
    else:
        s = -k
        y = xd[:-s]
        x = xo[s:].reshape(-1,1)
    if len(y) < 100: continue
    mi = mutual_info_regression(x, y, random_state=42, discrete_features=False)[0]
    pc = np.corrcoef(x.ravel(), y)[0,1]
    results.append({"k": k, "MI": mi, "Pearson": pc})

rdf = pd.DataFrame(results)
mi_max = rdf["MI"].max()
rdf["MI_norm"] = rdf["MI"] / mi_max

print("[BOT] MI shift scan  (MI_max={:.4f} at k={})".format(mi_max, rdf.loc[rdf["MI"].idxmax(), "k"]))
print(" k   MI       MI_norm   Pearson")
print("-" * 35)
for _, r in rdf.iterrows():
    marker = " ← BEST" if r["k"] == rdf.loc[rdf["MI"].idxmax(), "k"] else (" ← k=0" if r["k"] == 0 else "")
    bar = "#" * int(round(r["MI_norm"] * 20))
    print(f"{int(r['k']):>3}  {r['MI']:>7.4f}  {r['MI_norm']:>7.3f}  {r['Pearson']:>+8.3f}  {bar}{marker}")

k0 = rdf[rdf["k"]==0]["MI"].values[0]
k1 = rdf[rdf["k"]==1]["MI"].values[0] if (rdf["k"]==1).any() else np.nan
km1 = rdf[rdf["k"]==-1]["MI"].values[0] if (rdf["k"]==-1).any() else np.nan
print(f"\nMI(k=0)/MI(k=+1) = {k0/k1:.2f}  ({'>1.1 = 对齐可信' if k0/k1>1.1 else '<=1.1 对齐可疑'})")
print(f"MI(k=0)/MI(k=-1) = {k0/km1:.2f}")
