"""
Δ 序列的非线性自相关诊断
================================
对最大规格组 (SP=2.799) 的 Delta_TOP / Delta_BOT 序列做非线性检验:
  1. BDS 检验 (Brock-Dechert-Scheinkman) —— 时序非线性存在性的黄金检验
  2. 互信息 MI(Δ_t, Δ_{t-k}) —— 测任意形式依赖 (线性 + 非线性), 和 Pearson 对比
  3. 条件自相关 —— 按 Δ_{t-1} 的符号/幅度 分割后看 ACF 是否变化
  4. 延迟可获 Δ 的 MI —— 重点: 延迟 >= lab_delay=3 的 lag, 非线性强不强?
     (用户坚持不用 Δ_lag, 但非线性结构仍值得知道)

运行:
  python diag_nonlinear_delta.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# —— 和主脚本保持一致的数据路径 ——
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

print("=" * 70)
print("Δ 序列非线性自相关诊断")
print("=" * 70)

# —— 1. 加载 & 过滤 ——
df = pd.read_excel(DATA_PATH).rename(columns=RENAME)
df["Produce_Time"] = pd.to_datetime(df["Produce_Time"])
df["SP_TOP_r"] = df["SP_TOP"].round(3)
df["SP_BOT_r"] = df["SP_BOT"].round(3)
df = df.sort_values("Produce_Time").reset_index(drop=True)

g = df[(df["SP_TOP_r"] == 2.799) & (df["SP_BOT_r"] == 2.799)].copy()
g["Delta_TOP"] = g["Lab_TOP"] - g["Online_TOP"]
g["Delta_BOT"] = g["Lab_BOT"] - g["Online_BOT"]
g = g.dropna(subset=["Delta_TOP", "Delta_BOT"]).reset_index(drop=True)

print(f"\n样本数 N = {len(g)}")
print(f"Δ_TOP std = {g['Delta_TOP'].std():.4f}, mean = {g['Delta_TOP'].mean():.4f}")
print(f"Δ_BOT std = {g['Delta_BOT'].std():.4f}, mean = {g['Delta_BOT'].mean():.4f}")

# —— 2. BDS 检验 ——
print("\n" + "=" * 70)
print("方法 1: BDS 非线性检验")
print("  H0: 序列是独立同分布的 (无任何自相关, 包括非线性)")
print("  如果 p-value 极小 → 强烈拒绝 H0 → 存在自相关 (线性或非线性)")
print("  BDS 只能告诉你 '有非线性', 不能告诉你 '非线性有多强'")
print("=" * 70)

try:
    from statsmodels.stats.diagnostic import bds

    for col in ["Delta_TOP", "Delta_BOT"]:
        x = g[col].values
        print(f"\n  [{col}] N={len(x)}")
        print(f"    {'dim':>4}  {'eps':>8}  {'stat':>10}  {'pval':>12}  结论")
        print(f"    {'-'*58}")
        # m = embed dimension (2..5), eps = neighbor radius (std 的倍数)
        for m in [2, 3, 4, 5]:
            for eps_mul in [0.5, 1.0, 1.5]:
                try:
                    eps = np.std(x) * eps_mul
                    stat, pval = bds(x, max_dim=m, eps=eps)
                    # bds 返回 (stat, pval) 数组对应维度 2..max_dim
                    st = stat[-1]
                    pv = pval[-1]
                    sig = "*** 非线性" if pv < 0.001 else ("** " if pv < 0.01 else ("* " if pv < 0.05 else "线性 OK"))
                    print(f"    m={m} eps={eps_mul:.1f}σ {st:>10.3f}  {pv:>12.2e}  {sig}")
                except Exception as e:
                    print(f"    m={m} eps={eps_mul:.1f}σ  error: {e}")
except ImportError:
    print("  statsmodels BDS 不可用, 跳过 → 直接看互信息")

# —— 3. 互信息 Δ_t vs Δ_{t-k} ——
print("\n" + "=" * 70)
print("方法 2: 互信息 MI(Δ_t, Δ_{t-k}) vs Pearson |Δ_t, Δ_{t-k}|")
print("  MI 测任意形式依赖 (线性 + 非线性), 值越大越依赖")
print("  MI / |Pearson| >> 1  → 非线性主导")
print("  MI / |Pearson| ≈ 1   → 线性主导")
print("=" * 70)

from sklearn.feature_selection import mutual_info_regression

def mi_vs_pearson(series, max_lag=15):
    """对单序列算 MI 和 Pearson, 返回 DataFrame"""
    results = []
    x = series.values.astype(float)
    n = len(x)
    for k in range(1, max_lag + 1):
        y_target = x[k:]
        x_lag = x[:-k].reshape(-1, 1)
        # MI
        mi = mutual_info_regression(
            x_lag, y_target, random_state=42, discrete_features=False
        )[0]
        # Pearson (绝对值)
        pearson = np.abs(np.corrcoef(x_lag.ravel(), y_target)[0, 1])
        # 归一化 MI 到 [0,1] 便于跨 lag 比较
        # 用线性互信息 MI_linear = -0.5 log(1 - pearson^2) 作为理论线性上界
        if pearson < 0.999:
            mi_linear_upper = -0.5 * np.log(1 - pearson ** 2)
        else:
            mi_linear_upper = 5.0
        nonlinear_ratio = mi / mi_linear_upper if mi_linear_upper > 0.01 else None
        results.append({
            "lag": k, "Pearson": pearson, "MI": mi,
            "MI_over_linear": nonlinear_ratio,
        })
    return pd.DataFrame(results)

for col in ["Delta_TOP", "Delta_BOT"]:
    print(f"\n  [{col}]  (N={len(g)}, max_lag=15)")
    res = mi_vs_pearson(g[col], max_lag=15)
    print(res.to_string(index=False))

    # 标注目生产可用的 lag (假设 lab_delay=3, lag >= 3 理论上可拿)
    print(f"  → lab_delay=3 时 lag >= 3 的 lag 非线性强度:")
    d3 = res[res["lag"] >= 3].copy()
    print(f"    平均 MI_over_linear = {d3['MI_over_linear'].mean():.3f}")
    if d3["MI_over_linear"].mean() > 1.1:
        print(f"    ⚠️  lag >= 3 的 Δ 滞后项存在显著非线性依赖 (MI 是 Pearson 线性上限的 >1.1x)")
    else:
        print(f"    ✅ lag >= 3 的 Δ 滞后项主要是线性依赖 (MI/Pearson 比 ≈ 1)")

# —— 4. 条件自相关: 按 Δ_{t-1} 符号分割 ——
print("\n" + "=" * 70)
print("方法 3: 条件自相关 —— 按 Δ_{t-1} 幅度分割看 ACF")
print("  如果 ACF(正偏差) ≠ ACF(负偏差) → 存在符号不对称非线性")
print("  如果 ACF(大偏差) ≠ ACF(小偏差) → 存在阈值非线性")
print("=" * 70)

def conditional_acf(series, max_lag=10):
    """按 Δ_{t-1} 的不同条件算 Δ_t 对 Δ_{t-1} 的 Pearson 相关"""
    x = series.values.astype(float)
    results = {}
    x_lag1 = x[:-1]
    x_cur = x[1:]

    # 条件 1: Δ_{t-1} 正 vs 负
    mask_pos = x_lag1 > 0
    mask_neg = x_lag1 < 0
    results["lag1|Δ_{t-1}>0"] = np.corrcoef(x_lag1[mask_pos], x_cur[mask_pos])[0, 1] if mask_pos.sum() > 10 else np.nan
    results["lag1|Δ_{t-1}<0"] = np.corrcoef(x_lag1[mask_neg], x_cur[mask_neg])[0, 1] if mask_neg.sum() > 10 else np.nan

    # 条件 2: |Δ_{t-1}| 大 vs 小 (分位数分割)
    q67 = np.percentile(np.abs(x_lag1), 67)
    q33 = np.percentile(np.abs(x_lag1), 33)
    mask_big = np.abs(x_lag1) > q67
    mask_small = np.abs(x_lag1) < q33
    results["lag1|大偏差 (|Δ|>q67)"] = np.corrcoef(x_lag1[mask_big], x_cur[mask_big])[0, 1] if mask_big.sum() > 10 else np.nan
    results["lag1|小偏差 (|Δ|<q33)"] = np.corrcoef(x_lag1[mask_small], x_cur[mask_small])[0, 1] if mask_small.sum() > 10 else np.nan

    # lag 3 的条件 (lab_delay=3 生产可用)
    if len(x) > 4:
        x_lag3 = x[:-3]
        x_cur3 = x[3:]
        mask3_pos = x_lag3 > 0
        mask3_neg = x_lag3 < 0
        results["lag3|Δ_{t-3}>0"] = np.corrcoef(x_lag3[mask3_pos], x_cur3[mask3_pos])[0, 1] if mask3_pos.sum() > 10 else np.nan
        results["lag3|Δ_{t-3}<0"] = np.corrcoef(x_lag3[mask3_neg], x_cur3[mask3_neg])[0, 1] if mask3_neg.sum() > 10 else np.nan

    return results

for col in ["Delta_TOP", "Delta_BOT"]:
    print(f"\n  [{col}]")
    c = conditional_acf(g[col])
    for k, v in c.items():
        print(f"    {k:>28s} = {v:+.3f}")

    # 判断
    asym_lag1 = abs(c.get("lag1|Δ_{t-1}>0", 0) - c.get("lag1|Δ_{t-1}<0", 0))
    asym_lag3 = abs(c.get("lag3|Δ_{t-3}>0", 0) - c.get("lag3|Δ_{t-3}<0", 0))
    big_vs_small = abs(c.get("lag1|大偏差 (|Δ|>q67)", 0) - c.get("lag1|小偏差 (|Δ|<q33)", 0))

    print(f"\n  → 非线性信号:")
    print(f"    符号不对称 lag1 (|正-负|)   = {asym_lag1:.3f}  {'⚠️ 非线性显著' if asym_lag1 > 0.08 else '✅ 弱'}")
    print(f"    符号不对称 lag3 (|正-负|)   = {asym_lag3:.3f}  {'⚠️ 非线性显著' if asym_lag3 > 0.05 else '✅ 弱'}")
    print(f"    阈值不对称 lag1 (|大-小|)   = {big_vs_small:.3f}  {'⚠️ 非线性显著' if big_vs_small > 0.08 else '✅ 弱'}")

# —— 5. 小结 ——
print("\n" + "=" * 70)
print("小结: 我们的模型到底漏没漏 Δ 序列的非线性?")
print("=" * 70)
print("""
  关键前提: 生产中 Δ_{t-1} / Δ_{t-2} 拿不到 (化验延迟 3 卷)
  所以 lag1/lag2 即使有强非线性, 也没用

  生产中可用的是 lag >= 3 (lab_delay=3 时, Δ_{t-3} 在预测 t 时刚好回来)

  LightGBM full_lag 已经喂了 Online_SP_Diff lag1~5 ——
  虽然这不是 Δ lag, 但 Online_SP_Diff 和 Δ 的动态结构高度相关

  如果 lag >= 3 的 Δ 序列:
    (a) 非线性显著 → 值得把 Δ_lag{3,4,5} 重新加进模型 (生产可用)
    (b) 主要是线性 → 我们之前撤掉 Δ_lag 是对的, EWMA 已经抓了大部分
""")
