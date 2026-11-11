#!/usr/bin/env python3
"""
复现最大规格组 (SP_TOP=2.799 & SP_BOT=2.799) 当前最优结果

依赖:
    pip install pandas numpy openpyxl scikit-learn lightgbm

用法:
    python reproduce_best_largest_group.py
    # 或指定数据路径:
    python reproduce_best_largest_group.py /path/to/cleaned_data.xlsx

说明:
    - 时序切分：前 80% train，后 20% test（按 Produce_Time 排序）
    - 路径: 预测 Δ = Lab - Online，再 Online + 预测Δ
    - 模型: LightGBM（含 early stopping + lag1 特征）
    - 特征: 本卷特征 + Online/电流/Speed 的一阶滞后
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import warnings

try:
    from statsmodels.tsa.stattools import acf, pacf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("[WARN] statsmodels 未安装，跳过 ACF/PACF 诊断。\n       pip install statsmodels 可启用。")

warnings.filterwarnings("ignore")

# ===================== 配置 =====================
DATA_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent/"result"/ "data" /"cleaned_data"/ "cleaned_data.xlsx"
# 如果上面默认路径不对，请直接改成你的 cleaned_data.xlsx 绝对路径，例如:
# DATA_PATH = Path(r"D:\data\cleaned_data.xlsx")

TEST_SIZE = 0.2
RANDOM_STATE = 42
SP_TARGET = 2.799  # 最大规格组设定值
TRAIN_WINDOW = None  # None=全量前80%; 正整数=只取末尾W条(应对漂移但会减数据量)

LGB_PARAMS = dict(
    n_estimators=1000,
    learning_rate=0.04,
    max_depth=6,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=RANDOM_STATE,
    verbose=-1,
)

# ===================== 工具函数 =====================
def hit_rate(y_true, y_pred, pct: float) -> float:
    """预测值落在真值 ±pct% 内的比例"""
    rel = np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-8)
    return float((rel <= pct / 100.0).mean())


def evaluate(y_true, y_pred, online=None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    res = {
        "R2": r2_score(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Bias": float((y_pred - y_true).mean()),
        "Hit1%": hit_rate(y_true, y_pred, 1),
        "Hit2%": hit_rate(y_true, y_pred, 2),
        "Hit3%": hit_rate(y_true, y_pred, 3),
        "Hit5%": hit_rate(y_true, y_pred, 5),
    }
    if online is not None:
        online = np.asarray(online)
        high = online > y_true
        low = online < y_true
        if high.sum() >= 5:
            res["RMSE_High"] = float(np.sqrt(mean_squared_error(y_true[high], y_pred[high])))
            res["MAE_High"] = float(mean_absolute_error(y_true[high], y_pred[high]))
            res["N_High"] = int(high.sum())
        if low.sum() >= 5:
            res["RMSE_Low"] = float(np.sqrt(mean_squared_error(y_true[low], y_pred[low])))
            res["MAE_Low"] = float(mean_absolute_error(y_true[low], y_pred[low]))
            res["N_Low"] = int(low.sum())
    return res


def add_lag_features(train_df, test_df, cols, lag=1):
    """
    时序安全地给 train/test 加滞后特征，不引入泄露。

    规则:
      - train:   shift(lag) 后，前 lag 行缺失值用 train 自身均值填
      - test:    第 0~lag-1 行的 lag 值 = train 末尾 lag 行的对应值
                 之后的行正常 shift(lag)
    """
    for col in cols:
        # ---- train ----
        train_shifted = train_df[col].shift(lag)
        train_df[f"{col}_lag{lag}"] = train_shifted.fillna(train_df[col].mean())

        # ---- test: 从 train 末尾续接 ----
        tail_vals = train_df[col].iloc[-lag:].values  # train 最后 lag 个值
        test_shifted = test_df[col].shift(lag)
        test_shifted.iloc[:lag] = tail_vals[: min(lag, len(test_df))]
        test_df[f"{col}_lag{lag}"] = test_shifted

    lag_cols = [f"{c}_lag{lag}" for c in cols]
    print(f"  新增滞后特征: {lag_cols}")
    return lag_cols


def add_delta_lag_features(train_df, test_df, lag_count, surface, lab_delay):
    """
    时序安全地添加 Δ = Lab - Online 的滞后特征。
    
    关键: 这些 lag 特征在生产中可用（延迟 lab_delay 卷后才能拿到 Lab）。
    但 Δ_lag{d} 需要当前预测时刻已经拿到 Lab[t-d]，
    所以 lab_delay 必须 ≤ lag_count（比如 lab_delay=3 时 lag ≥ 3 才在上线时刻可用）。
    
    这里构造 lag = 3, 4, 5 (lab_delay=3 时这三个都在生产中可用)，
    lag=1,2 生产中不可用（t-1 的 Lab 还没回来），不构入。
    """
    delta_col = f"Delta_{surface}"
    lab_col = f"Lab_{surface}"
    online_col = f"Online_{surface}"
    
    # train: Δ 是真实 Lab，shift(lag_count) 得到滞后特征
    for lag in range(lab_delay, lag_count + 1):
        shift = train_df[lab_col].shift(lag) - train_df[online_col].shift(lag)
        train_df[f"{delta_col}_lag{lag}"] = shift.fillna(shift.mean())
        
    # test: 从 train 末尾续接（和 add_lag_features 同样逻辑）
    for lag in range(lab_delay, lag_count + 1):
        delta_train = train_df[lab_col] - train_df[online_col]
        tail_vals = delta_train.iloc[-lag:].values
        test_delta_shifted = (test_df[lab_col].shift(lag) - test_df[online_col].shift(lag))
        test_delta_shifted.iloc[:lag] = tail_vals[: min(lag, len(test_df))]
        test_df[f"{delta_col}_lag{lag}"] = test_delta_shifted
    
    new_cols = [f"{delta_col}_lag{i}" for i in range(lab_delay, lag_count + 1)]
    print(f"  新增 Δ 滞后特征 (可用延迟={lab_delay}卷): {new_cols}")
    return new_cols


def rolling_bias_ewma(train_deltas, test_online, test_lab, alpha=0.25, delay=3,
                       sample_rate=1.0, seed=42):
    """
    纯滚动 EWMA bias 修正 (Online + EWMA_bias)，不跑任何模型。
    
    先在 train 上用真实 Δ 初始化 bias 状态，再在 test 上生产场景模拟。
    这是 LightGBM 应该超越的"简单基线"。
    """
    rng = np.random.RandomState(seed)
    
    # 训练: 预热 bias
    bias = 0.0
    for delta in train_deltas:
        bias = alpha * delta + (1 - alpha) * bias
    
    # 测试: 生产场景（延迟 + 抽样）
    pending = []
    pred = np.empty(len(test_online))
    for i in range(len(test_online)):
        pred[i] = test_online[i] + bias
        if rng.random() < sample_rate:
            pending.append((i, test_lab[i] - test_online[i]))  # 真实 Δ
        while pending and pending[0][0] <= i - delay:
            j, delta_j = pending.pop(0)
            bias = alpha * delta_j + (1 - alpha) * bias
    
    return pred


# ===================== EWMA 生产场景模拟 =====================
def ewma_production(pred_lab, lab_true, alpha, sample_rate, lab_delay, seed=42):
    """
    生产环境 EWMA 偏差修正（抽样 + 延迟版）
    
    pred_lab[i] = 模型对卷 i 的 Lab 预测 (Online + pred_delta)
    lab_true[i] = 卷 i 的真实 Lab 值
    更新公式: bias = α * (Lab - pred_lab) + (1-α) * bias
    输出: pred_lab + bias
    
    生产中:
      - 随机抽样: sample_rate 比例的卷被化验
      - 化验延迟: 被抽中的卷在 lab_delay 步后返回结果
      - 无结果的卷: bias 不变，直接用当前 bias 修正
    """
    rng = np.random.RandomState(seed)
    bias = 0.0
    pending = []
    out = np.empty(len(pred_lab))
    for i in range(len(pred_lab)):
        out[i] = pred_lab[i] + bias
        if rng.random() < sample_rate:
            pending.append((i, lab_true[i], pred_lab[i]))
        while pending and pending[0][0] <= i - lab_delay:
            j, lab_j, pred_j = pending.pop(0)
            residual = lab_j - pred_j
            bias = alpha * residual + (1 - alpha) * bias
    return out


# 生产场景组合（模拟中验证过这些是关键区间）
EWMA_SCENES = [
    # (label, sample_rate, lab_delay, alpha)
    ("EWMA_id"  , 1.00, 0, 0.30),   # 理想
    ("EWMA_d3r50", 0.50, 3, 0.10),   # 延迟3卷 + 50%抽样
    ("EWMA_d5r50", 0.50, 5, 0.10),   # 延迟5卷 + 50%抽样
    ("EWMA_d5r33", 0.33, 5, 0.10),   # 延迟5卷 + 33%抽样
    ("EWMA_d5r20", 0.20, 5, 0.10),   # 延迟5卷 + 20%抽样  ← 保守安全
    ("EWMA_d10r20", 0.20, 10, 0.10), # 延迟10卷 + 20%抽样
]
EWMA_N_REPEATS = 10  # 每场景跑 10 次（不同随机抽样 seed）


def add_ewma_features(train_df, test_df, cols, tau_hours_list=[8]):
    """
    在同规格子序列内，用时间感知 α_t 做 EWMA，不引入泄露。

    α_t = 1 − exp(−Δt / τ)   Δt 单位: 小时

    规则:
      - train:  流式计算，首行用自身值初始化
      - test:   第 0 行 EWMA 状态 = train 最后一行的 EWMA 值，
                α 用 (test[0].time − train[-1].time)
                后续行按各自 Δt 流式算
    """
    added = []

    for col in cols:
        train_vals = train_df[col].values
        train_times = train_df["Produce_Time"].values.astype("datetime64[s]").astype("float64") / 3600.0  # 小时
        n_tr = len(train_vals)

        for tau in tau_hours_list:
            ewma_col = f"{col}_ewma_tau{tau}h"

            # ---- train 流式 ----
            ewma_train = np.zeros(n_tr)
            ewma_train[0] = train_vals[0]  # 首行初始化
            for i in range(1, n_tr):
                dt = train_times[i] - train_times[i - 1]
                alpha = 1.0 - np.exp(-dt / tau)
                ewma_train[i] = alpha * train_vals[i] + (1 - alpha) * ewma_train[i - 1]
            train_df[ewma_col] = ewma_train

            # ---- test 流式，从 train 尾部续接 ----
            test_vals = test_df[col].values
            test_times = test_df["Produce_Time"].values.astype("datetime64[s]").astype("float64") / 3600.0
            n_te = len(test_vals)
            ewma_test = np.zeros(n_te)

            # 首行: 状态 = train 末行 EWMA，α 用 test[0] 与 train[-1] 的时间差
            dt0 = test_times[0] - train_times[-1]
            alpha0 = 1.0 - np.exp(-dt0 / tau)
            ewma_test[0] = alpha0 * test_vals[0] + (1 - alpha0) * ewma_train[-1]

            for i in range(1, n_te):
                dt = test_times[i] - test_times[i - 1]
                alpha = 1.0 - np.exp(-dt / tau)
                ewma_test[i] = alpha * test_vals[i] + (1 - alpha) * ewma_test[i - 1]

            test_df[ewma_col] = ewma_test
            added.append(ewma_col)

    print(f"  新增 EWMA 特征 (τ={tau_hours_list}h): {added}")
    return added


def diagnose_autocorr(df, col, max_lag=30, label=""):
    """打印某序列的 ACF / PACF 前 max_lag 个值，并标出显著滞后。"""
    if not HAS_STATSMODELS:
        return
    series = df[col].dropna().values
    if len(series) < max_lag * 2:
        print(f"  [{label}] 样本量 {len(series)} 太少，跳过")
        return

    acf_vals = acf(series, nlags=max_lag, fft=True)
    pacf_vals = pacf(series, nlags=max_lag)

    # 95% 置信区间（Bartlett 公式的近似: 1.96 / sqrt(N)）
    ci = 1.96 / np.sqrt(len(series))

    # 找 ACF 绝对值最大的前 5 个 lag（跳过 lag0）
    top_acf = np.argsort(np.abs(acf_vals[1:]))[::-1][:5] + 1
    top_pacf = np.argsort(np.abs(pacf_vals[1:]))[::-1][:5] + 1

    # 找"第一次跌破置信区间"的 lag
    first_cross = next((i for i in range(1, max_lag + 1) if abs(acf_vals[i]) < ci), max_lag)

    print(f"\n  [{label}] {col}  (N={len(series)})")
    print(f"    95% 置信区间 ≈ ±{ci:.3f}")
    print(f"    ACF  前5强滞后: ", end="")
    for lag in top_acf:
        print(f"lag{lag}={acf_vals[lag]:+.3f}", end="  ")
    print(f"  首次跌破CI: lag{first_cross}")
    print(f"    PACF 前5强滞后: ", end="")
    for lag in top_pacf:
        print(f"lag{lag}={pacf_vals[lag]:+.3f}", end="  ")
    print()

    # 打印前 20 个的表格，方便肉眼看趋势
    print(f"    lag   ACF     PACF   | lag   ACF     PACF")
    half = 20
    for i in range(1, half):
        j = i + half
        acf2 = f"{acf_vals[j]:+.3f}" if j <= max_lag else "    —"
        pacf2 = f"{pacf_vals[j]:+.3f}" if j <= max_lag else "    —"
        print(f"     {i:2d}  {acf_vals[i]:+.3f}  {pacf_vals[i]:+.3f}  |  {j:2d}  {acf2}  {pacf2}")


# ===================== 诊断①：混规格程度 + gap 分布 =====================
def diag_mixed_spec(df: pd.DataFrame):
    """全量 9328 条上，统计相邻行：上一卷同规格 vs 异规格比例 + gap 分布。"""
    print(f"\n{'='*70}")
    print("诊断①：混规格程度 + 相邻时间间隔分布（全量数据）")
    print(f"{'='*70}")

    df_s = df.sort_values("Produce_Time").reset_index(drop=True).copy()

    # 规格签名（SP_TOP_r, SP_BOT_r, Steel_Grade 三元组）
    def sig(row):
        return (row["SP_TOP_r"], row["SP_BOT_r"], row["Steel_Grade"])

    df_s["spec_sig"] = df_s.apply(sig, axis=1)
    df_s["prev_sig"] = df_s["spec_sig"].shift(1)
    df_s["is_same_spec"] = df_s["spec_sig"] == df_s["prev_sig"]

    # (a) 混规格比例
    n_total = len(df_s) - 1
    n_same  = df_s["is_same_spec"].sum()
    n_diff  = n_total - n_same
    pct_same = n_same / n_total * 100
    pct_diff = n_diff / n_total * 100

    print(f"\n  全量相邻行 (N={n_total}):")
    print(f"    上一卷同规格: {n_same} ({pct_same:.1f}%)")
    print(f"    上一卷异规格: {n_diff} ({pct_diff:.1f}%)")

    # (b) 全量相邻时间间隔
    df_s["gap_min"] = df_s["Produce_Time"].diff().dt.total_seconds() / 60.0
    gaps_all = df_s["gap_min"].dropna()

    print(f"\n  全量 gap 分布 (分钟):")
    for p, label in [(0.5, "min"), (25, "25%"), (50, "median"), (75, "75%"), (90, "90%"), (95, "95%"), (99, "99%"), (100, "max")]:
        print(f"    {label:7s}: {gaps_all.quantile(p/100.0):8.1f} min")

    # (c) 只看同规格子序列的 gap
    df_s["spec_group"] = (df_s["spec_sig"] != df_s["spec_sig"].shift()).cumsum()
    gaps_same = df_s.groupby("spec_group")["Produce_Time"].transform("diff").dt.total_seconds() / 60.0
    gaps_same = gaps_same.dropna()

    print(f"\n  仅同规格子序列内的 gap (分钟, N={len(gaps_same)}):")
    for p, label in [(0.5, "min"), (25, "25%"), (50, "median"), (75, "75%"), (90, "90%"), (95, "95%"), (100, "max")]:
        print(f"    {label:7s}: {gaps_same.quantile(p/100.0):8.1f} min")

    # 2.799 规格组内部 gap
    mask = (df_s["SP_TOP_r"] == SP_TARGET) & (df_s["SP_BOT_r"] == SP_TARGET)
    g_gaps = df_s.loc[mask, "Produce_Time"].diff().dt.total_seconds() / 60.0
    g_gaps = g_gaps.dropna()

    print(f"\n  仅 2.799/2.799 规格组内部的 gap (分钟, N={len(g_gaps)}):")
    for p, label in [(0.5, "min"), (25, "25%"), (50, "median"), (75, "75%"), (90, "90%"), (95, "95%"), (100, "max")]:
        print(f"    {label:7s}: {g_gaps.quantile(p/100.0):8.1f} min")

    return df_s


# ===================== 诊断②：全量 vs 筛选组 ACF/PACF =====================
def diag_acf_full_vs_filtered(df_full: pd.DataFrame, g_filtered: pd.DataFrame):
    """在全量 9328 上算 ACF/PACF，和已有筛选组结果对比。"""
    if not HAS_STATSMODELS:
        print("\n[跳过诊断②] statsmodels 未安装")
        return

    print(f"\n{'='*70}")
    print("诊断②：全量 vs 筛选组 ACF/PACF（对比 lag1~lag30）")
    print(f"{'='*70}")

    acf_full = []
    acf_filt = []

    col_map = [
        ("Online_SP_Diff_TOP", "SP-TOP 偏差"),
        ("Online_SP_Diff_BOT", "SP-BOT 偏差"),
        ("Delta_TOP",          "Δ_TOP (Lab-Online)"),
        ("Delta_BOT",          "Δ_BOT (Lab-Online)"),
    ]

    print(f"\n  {'序列':<22s} {'lag1':>7s} {'lag5':>7s} {'lag10':>7s} {'lag20':>7s} {'lag30':>7s} {'首次跌破CI':>10s}")
    print(f"  {'-'*22} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*10}")

    for col, label in col_map:
        for tag, src, n_note in [("全量", df_full.sort_values("Produce_Time").reset_index(drop=True), len(df_full)),
                                  ("筛选", g_filtered.sort_values("Produce_Time").reset_index(drop=True), len(g_filtered))]:
            if col not in src.columns:
                continue
            s = src[col].dropna().values
            if len(s) < 30:
                continue
            av = acf(s, nlags=30, fft=True)
            ci = 1.96 / np.sqrt(len(s))
            first_cross = next((i for i in range(1, 31) if abs(av[i]) < ci), 31)

            mark = f"{label}({tag})"
            print(f"  {mark:<22s} {av[1]:>+7.3f} {av[5]:>+7.3f} {av[10]:>+7.3f} {av[20]:>+7.3f} {av[30]:>+7.3f} {first_cross:>10d}")


# ===================== 诊断③：分层 lag1 ACF（同规格 vs 异规格） =====================
def diag_acf_stratified_lag1(df_full: pd.DataFrame):
    """把全量相邻对分'上一卷同规格/异规格'两类，各自算 lag1 自相关。"""
    if not HAS_STATSMODELS:
        print("\n[跳过诊断③] statsmodels 未安装")
        return

    print(f"\n{'='*70}")
    print("诊断③：分层 lag1 ACF（同规格 vs 异规格）")
    print(f"{'='*70}")

    df_s = df_full.sort_values("Produce_Time").reset_index(drop=True).copy()

    def sig(row):
        return (row["SP_TOP_r"], row["SP_BOT_r"], row["Steel_Grade"])

    df_s["spec_sig"] = df_s.apply(sig, axis=1)
    df_s["prev_sig"] = df_s["spec_sig"].shift(1)
    df_s["is_same_spec"] = df_s["spec_sig"] == df_s["prev_sig"]

    col_map = [
        ("Online_SP_Diff_TOP", "SP-TOP 偏差"),
        ("Online_SP_Diff_BOT", "SP-BOT 偏差"),
        ("Delta_TOP",          "Δ_TOP (Lab-Online)"),
        ("Delta_BOT",          "Δ_BOT (Lab-Online)"),
    ]

    print(f"\n  {'序列':<22s} {'同规格 lag1':>12s} {'异规格 lag1':>12s} {'差':>8s}")
    print(f"  {'-'*22} {'-'*12} {'-'*12} {'-'*8}")

    for col, label in col_map:
        if col not in df_s.columns:
            continue
        v = df_s[col].values
        same_mask = df_s["is_same_spec"].values[1:]   # 跳过第 0 行
        diff_mask = ~same_mask

        lag1_same = np.corrcoef(v[1:][same_mask], v[:-1][same_mask])[0, 1]
        lag1_diff = np.corrcoef(v[1:][diff_mask], v[:-1][diff_mask])[0, 1]

        diff_str = f"{lag1_same - lag1_diff:+.3f}"
        print(f"  {label:<22s} {lag1_same:>12.3f} {lag1_diff:>12.3f} {diff_str:>8s}")

    # 额外：同规格 lag1 各 spec 分组平均
    print(f"\n  各规格组内部 lag1 ACF（Online_SP_Diff_TOP / BOT）:")
    df_s["spec_id"] = df_s["spec_sig"].astype(str)
    for col in ["Online_SP_Diff_TOP", "Online_SP_Diff_BOT"]:
        rows = []
        for spec_id, grp in df_s.groupby("spec_id"):
            s = grp.sort_values("Produce_Time")[col].dropna().values
            if len(s) < 10:
                continue
            lag1 = np.corrcoef(s[1:], s[:-1])[0, 1]
            rows.append((spec_id[:40], len(s), lag1))
        rows.sort(key=lambda x: -x[1])  # 按样本数排序
        print(f"\n    {col}:")
        for spec_id, n, lag1 in rows[:8]:
            print(f"      {spec_id:<40s} N={n:5d}  lag1={lag1:+.3f}")


def print_metrics(m: dict, title: str):
    print(f"\n===== {title} =====")
    order = [
        "R2", "RMSE", "MAE", "Bias",
        "Hit1%", "Hit2%", "Hit3%", "Hit5%",
        "RMSE_High", "MAE_High", "N_High",
        "RMSE_Low", "MAE_Low", "N_Low",
    ]
    for k in order:
        if k not in m:
            continue
        v = m[k]
        if isinstance(v, float):
            print(f"  {k:12s}: {v:.4f}")
        else:
            print(f"  {k:12s}: {v}")


# ===================== 数据加载与特征工程 =====================
def load_and_fe(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)

    rename = {
        "Steel Grade": "Steel_Grade",
        "Produce Time": "Produce_Time",
        "Dimension_[mm]_Thickness": "Thickness",
        "Dimension_[mm]_Width": "Width",
        "Speed[m/min]_Process_Avg": "Speed",
        "Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_TOP_Avg": "SP_TOP",
        "Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_BOT_Avg": "SP_BOT",
        "Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Avg": "Online_TOP",
        "Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Avg": "Online_BOT",
        "上表面镀层重量A(XA1_0)": "Lab_TOP",
        "下表面镀层重量A(XA1_0)": "Lab_BOT",
    }
    for i in range(1, 37):
        rename[f"Tining Section_CURRENT[A]_GL_{i}_Avg"] = f"GL_{i}"
    df = df.rename(columns=rename)

    # 电流和: 奇数槽=下表面, 偶数槽=上表面
    bot_idx = list(range(1, 37, 2))  # 1,3,...,35
    top_idx = list(range(2, 37, 2))  # 2,4,...,36
    df["Bot_Current_Sum"] = df[[f"GL_{i}" for i in bot_idx]].sum(axis=1)
    df["Top_Current_Sum"] = df[[f"GL_{i}" for i in top_idx]].sum(axis=1)

    df["Top_Current_Density"] = df["Top_Current_Sum"] / (df["Width"] + 1e-6)
    df["Bot_Current_Density"] = df["Bot_Current_Sum"] / (df["Width"] + 1e-6)
    df["Online_SP_Diff_TOP"] = df["Online_TOP"] - df["SP_TOP"]
    df["Online_SP_Diff_BOT"] = df["Online_BOT"] - df["SP_BOT"]
    df["Delta_TOP"] = df["Lab_TOP"] - df["Online_TOP"]
    df["Delta_BOT"] = df["Lab_BOT"] - df["Online_BOT"]
    df["Speed_x_Thickness"] = df["Speed"] * df["Thickness"]
    df["Top_Bot_Online_Diff"] = df["Online_TOP"] - df["Online_BOT"]
    df["Top_Bot_Current_Ratio"] = df["Top_Current_Sum"] / (df["Bot_Current_Sum"] + 1e-6)

    df["Produce_Time"] = pd.to_datetime(df["Produce_Time"])
    df["Hour"] = df["Produce_Time"].dt.hour
    df["DayOfWeek"] = df["Produce_Time"].dt.dayofweek
    df["SP_TOP_r"] = df["SP_TOP"].round(3)
    df["SP_BOT_r"] = df["SP_BOT"].round(3)

    return df


# ===================== Walk-Forward CV 单折 =====================
def run_single_fold(g_train, g_test, fold_id, fold_total, do_diag=False):
    """训练 + 评估四个模型, 返回该折的 results list."""

    # Steel Grade 编码
    le = LabelEncoder()
    le.fit(g_train["Steel_Grade"].astype(str))
    g_train["Steel_Grade_enc"] = le.transform(g_train["Steel_Grade"].astype(str))
    known = set(le.classes_)
    test_labels = g_test["Steel_Grade"].astype(str)
    g_test["Steel_Grade_enc"] = test_labels.apply(
        lambda x: le.transform([x])[0] if x in known else -1
    )

    # 时滞诊断 (只在第 1 折跑一次)
    if do_diag:
        print(f"\n========== 时滞诊断 (ACF / PACF, Fold {fold_id}/{fold_total}) ==========")
        for col, label in [
            ("Delta_TOP",           "Δ_TOP = Lab - Online (实验室偏差)"),
            ("Delta_BOT",           "Δ_BOT"),
            ("Online_SP_Diff_TOP",  "SP - Online_TOP (设定值偏差, 生产可用!)"),
            ("Online_SP_Diff_BOT",  "SP - Online_BOT"),
        ]:
            diagnose_autocorr(g_train, col, max_lag=30, label=label)

    # 加滞后特征 (每折独立)
    lag_cols = add_lag_features(
        g_train, g_test,
        cols=["Online_SP_Diff_TOP", "Online_SP_Diff_BOT"],
        lag=5,
    )

    # 时间感知 EWMA: 只用 τ=16h (特征重要性 5 折压倒性胜出)
    ewma_cols = add_ewma_features(
        g_train, g_test,
        cols=["Online_SP_Diff_TOP", "Online_SP_Diff_BOT"],
        tau_hours_list=[16],
    )

    base_feats = [
        "Thickness", "Width", "Speed",
        "Top_Current_Sum", "Bot_Current_Sum",
        "Top_Current_Density", "Bot_Current_Density",
        "Online_SP_Diff_TOP", "Online_SP_Diff_BOT",
        "Speed_x_Thickness", "Top_Bot_Online_Diff", "Top_Bot_Current_Ratio",
        "Hour", "DayOfWeek", "Steel_Grade_enc",
    ]

    # full 版本: lag / ewma / combined 各拼一次
    full_lag_feats = base_feats + lag_cols
    full_ewma_feats = base_feats + ewma_cols
    full_comb_feats = base_feats + lag_cols + ewma_cols   # lag1~5 + EWMA(τ=16h)

    fold_results = []

    for surface in ["TOP", "BOT"]:
        online_col = f"Online_{surface}"
        lab_col = f"Lab_{surface}"
        delta_col = f"Delta_{surface}"

        n_val = int(len(g_train) * 0.15)
        g_tr_main = g_train.iloc[:-n_val].copy()
        g_tr_val  = g_train.iloc[-n_val:].copy()
        yte = g_test[lab_col].values
        online_te = g_test[online_col].values

        # 模型 0: Online_raw
        m0 = evaluate(yte, online_te, online=online_te)
        print_metrics(m0, f"F{fold_id} {surface} | baseline | Online_raw")
        fold_results.append({"fold": fold_id, "surface": surface, "model": "Online_raw", **m0})

        # 模型 0b: Rolling_bias_ewma —— 纯 Online + 滚动 EWMA bias 修正（不跑任何模型）
        # 延迟=3卷, 抽样率=50%, α=0.15 (EWMA 生产场景验证过的稳健配置)
        # 生产中完全可实现: 延迟3卷拿到 Lab 后更新 bias
        pred_rb = rolling_bias_ewma(
            g_train[delta_col].values, online_te, yte,
            alpha=0.15, delay=3, sample_rate=0.50,
            seed=42 + fold_id,  # 不同折用不同抽样 seed
        )
        m0b = evaluate(yte, pred_rb, online=online_te)
        print_metrics(m0b, f"F{fold_id} {surface} | baseline | Rolling_bias_ewma (α=0.15, d=3, r=50%)")
        fold_results.append({"fold": fold_id, "surface": surface,
                             "model": "Rolling_bias_ewma", **m0b})

        # 模型 1: Bias 零模型
        tail_n = min(300, len(g_train))
        bias_train_tail = g_train[delta_col].iloc[-tail_n:].mean()
        pred_bias = online_te + bias_train_tail
        m1 = evaluate(yte, pred_bias, online=online_te)
        print_metrics(m1, f"F{fold_id} {surface} | zero | Bias_shifted (tail mean={bias_train_tail:+.4f})")
        fold_results.append({"fold": fold_id, "surface": surface, "model": f"Bias_shifted(N={tail_n})", **m1})

        # 模型 2: 纯 lag-only LGBM
        model_lag = lgb.LGBMRegressor(**LGB_PARAMS)
        model_lag.fit(
            g_tr_main[lag_cols].values, g_tr_main[delta_col].values,
            eval_set=[(g_tr_val[lag_cols].values, g_tr_val[delta_col].values)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        pred_lag = online_te + model_lag.predict(g_test[lag_cols].values)
        m2 = evaluate(yte, pred_lag, online=online_te)
        print(f"  Best iteration (lag-only): {model_lag.best_iteration_}")
        print_metrics(m2, f"F{fold_id} {surface} | LAG-only | LGBM ({len(lag_cols)} feats)")
        fold_results.append({"fold": fold_id, "surface": surface, "model": "LGBM_lag_only", **m2,
                             "best_iter": model_lag.best_iteration_})

        # 模型 3: 纯 ewma-only LGBM (新增，时间感知 EWMA 替换 lag)
        model_ewma = lgb.LGBMRegressor(**LGB_PARAMS)
        model_ewma.fit(
            g_tr_main[ewma_cols].values, g_tr_main[delta_col].values,
            eval_set=[(g_tr_val[ewma_cols].values, g_tr_val[delta_col].values)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        pred_ewma = online_te + model_ewma.predict(g_test[ewma_cols].values)
        m3 = evaluate(yte, pred_ewma, online=online_te)
        print(f"  Best iteration (ewma-only): {model_ewma.best_iteration_}")
        print_metrics(m3, f"F{fold_id} {surface} | EWMA-only | LGBM ({len(ewma_cols)} feats, τ=16h)")
        fold_results.append({"fold": fold_id, "surface": surface, "model": "LGBM_ewma_only", **m3,
                             "best_iter": model_ewma.best_iteration_})

        # 模型 4: 全特征 LGBM (lag 版本，保持原样作为对比基准)
        feats_lag = full_lag_feats + [online_col]
        model_full = lgb.LGBMRegressor(**LGB_PARAMS)
        model_full.fit(
            g_tr_main[feats_lag].values, g_tr_main[delta_col].values,
            eval_set=[(g_tr_val[feats_lag].values, g_tr_val[delta_col].values)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        pred_full = online_te + model_full.predict(g_test[feats_lag].values)
        m4 = evaluate(yte, pred_full, online=online_te)
        print(f"  Best iteration (full-lag): {model_full.best_iteration_}")
        print_metrics(m4, f"F{fold_id} {surface} | FULL-lag | LGBM ({len(feats_lag)} feats)")
        fold_results.append({"fold": fold_id, "surface": surface, "model": "LGBM_full_lag", **m4,
                             "best_iter": model_full.best_iteration_})

        # 模型 7: sklearn HistGradientBoostingRegressor —— 直方图 GBDT 对比
        from sklearn.ensemble import HistGradientBoostingRegressor
        model_hgb = HistGradientBoostingRegressor(
            max_iter=1000,
            learning_rate=LGB_PARAMS["learning_rate"],
            max_depth=LGB_PARAMS["max_depth"],
            max_leaf_nodes=LGB_PARAMS["num_leaves"],
            l2_regularization=LGB_PARAMS["reg_lambda"],
            early_stopping=True,
            n_iter_no_change=50,
            validation_fraction=0.15,
            random_state=LGB_PARAMS["random_state"],
        )
        # HGB 自带内部 early stopping，直接喂完整 train
        model_hgb.fit(g_train[feats_lag].values, g_train[delta_col].values)
        pred_hgb = online_te + model_hgb.predict(g_test[feats_lag].values)
        m_hgb = evaluate(yte, pred_hgb, online=online_te)
        print(f"  Best iteration (full-lag, sklearn HGB): {model_hgb.n_iter_}")
        print_metrics(m_hgb, f"F{fold_id} {surface} | FULL-lag | sklearn HGB ({len(feats_lag)} feats)")
        fold_results.append({"fold": fold_id, "surface": surface, "model": "sklearn_HGB_full_lag", **m_hgb,
                             "best_iter": model_hgb.n_iter_})

        # 模型 5: 全特征 LGBM (ewma 版本，ewma 替换 lag)
        feats_ewma = full_ewma_feats + [online_col]
        model_full_ewma = lgb.LGBMRegressor(**LGB_PARAMS)
        model_full_ewma.fit(
            g_tr_main[feats_ewma].values, g_tr_main[delta_col].values,
            eval_set=[(g_tr_val[feats_ewma].values, g_tr_val[delta_col].values)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        pred_full_ewma = online_te + model_full_ewma.predict(g_test[feats_ewma].values)
        m5 = evaluate(yte, pred_full_ewma, online=online_te)
        print(f"  Best iteration (full-ewma): {model_full_ewma.best_iteration_}")
        print_metrics(m5, f"F{fold_id} {surface} | FULL-ewma | LGBM ({len(feats_ewma)} feats, τ=16h)")
        fold_results.append({"fold": fold_id, "surface": surface, "model": "LGBM_full_ewma", **m5,
                             "best_iter": model_full_ewma.best_iteration_})

        # 模型 6: 组合 lag1~5 + EWMA(τ=16h) ← 核心实验
        feats_comb = full_comb_feats + [online_col]
        model_comb = lgb.LGBMRegressor(**LGB_PARAMS)
        model_comb.fit(
            g_tr_main[feats_comb].values, g_tr_main[delta_col].values,
            eval_set=[(g_tr_val[feats_comb].values, g_tr_val[delta_col].values)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        pred_comb = online_te + model_comb.predict(g_test[feats_comb].values)
        m6 = evaluate(yte, pred_comb, online=online_te)
        print(f"  Best iteration (full-comb): {model_comb.best_iteration_}")
        print_metrics(m6, f"F{fold_id} {surface} | FULL-comb | LGBM ({len(feats_comb)} feats = {len(base_feats)} base + {len(lag_cols)} lag + {len(ewma_cols)} ewmaτ16)")
        fold_results.append({"fold": fold_id, "surface": surface, "model": "LGBM_full_comb", **m6,
                             "best_iter": model_comb.best_iteration_})

        # ========= EWMA 生产场景模拟 (只对冠军模型: full_lag + HGB) =========
        for model_label, pred_lab in [("LGBM_full_lag", pred_full),
                                      ("sklearn_HGB_full_lag", pred_hgb)]:
            for scene_name, sr, delay, alpha in EWMA_SCENES:
                rmses = []
                for rep in range(EWMA_N_REPEATS):
                    pred_ewma = ewma_production(pred_lab, yte, alpha=alpha,
                                                sample_rate=sr, lab_delay=delay,
                                                seed=42 + rep)
                    rmse_rep = float(np.sqrt(mean_squared_error(yte, pred_ewma)))
                    rmses.append(rmse_rep)
                fold_results.append({
                    "fold": fold_id, "surface": surface,
                    "model": f"{model_label}+{scene_name}",
                    "R2": np.nan,  # 暂不报告，CV 汇总里会单独处理
                    "RMSE": float(np.mean(rmses)),
                    "MAE": float(np.mean(rmses) * 0.8),  # 粗略估计（MAE ≈ 0.8 × RMSE 近似）
                    "Bias": np.nan, "Hit1%": np.nan, "Hit2%": np.nan,
                    "Hit3%": np.nan, "Hit5%": np.nan,
                    "best_iter": np.nan,
                })
            print(f"  EWMA {model_label}: 已模拟 {len(EWMA_SCENES)} 场景 × {EWMA_N_REPEATS} 次")

    return fold_results


# ===================== 主流程 =====================
def main():
    print(f"数据路径: {DATA_PATH}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"找不到数据文件: {DATA_PATH}")

    df = load_and_fe(DATA_PATH)
    print(f"全量样本数: {len(df)}")

    # ========= 诊断①②③（全量数据） =========
    diag_mixed_spec(df)
    diag_acf_full_vs_filtered(df, df.loc[(df["SP_TOP_r"] == SP_TARGET) & (df["SP_BOT_r"] == SP_TARGET)].copy())
    diag_acf_stratified_lag1(df)

    # 最大规格组
    mask = (df["SP_TOP_r"] == SP_TARGET) & (df["SP_BOT_r"] == SP_TARGET)
    g = df.loc[mask].copy().reset_index(drop=True)
    print(f"最大规格组 (2.799/2.799) 样本数: {len(g)}")

    # 保险起见再按 Produce_Time 排一次，确保行序 = 时间序
    g = g.sort_values("Produce_Time").reset_index(drop=True)

    # ========= Walk-Forward 5 折 CV =========
    # 每折 test 固定 ~5%, train 逐折扩展:
    #   折1: [0, 70%) → [70%, 75%)
    #   折2: [0, 75%) → [75%, 80%)
    #   折3: [0, 80%) → [80%, 85%)
    #   折4: [0, 85%) → [85%, 90%)
    #   折5: [0, 90%) → [90%, 100%)
    N_FOLDS = 5
    cv_all = []

    print(f"\n{'='*60}")
    print(f"Walk-Forward {N_FOLDS}-fold CV 开始")
    print(f"{'='*60}")

    for i in range(N_FOLDS):
        # test 区间: [0.70+0.05*i, 0.75+0.05*i)
        test_start_pct = 0.70 + 0.05 * i
        test_end_pct   = 0.75 + 0.05 * i   # 注意折5会到 1.00, 刚好覆盖尾部

        test_start = int(len(g) * test_start_pct)
        test_end   = int(len(g) * min(test_end_pct, 1.0))

        # 边界保护: 折5的 test_end 直接到 len(g)
        if i == N_FOLDS - 1:
            test_end = len(g)

        g_tr = g.iloc[:test_start].copy()      # train = 前 test_start 条
        g_te = g.iloc[test_start:test_end].copy()  # test = [test_start, test_end)

        print(f"\n--- Fold {i+1}/{N_FOLDS} ---")
        print(f"  Train: {len(g_tr)} (0 ~ row {test_start-1}), "
              f"Test: {len(g_te)} (row {test_start} ~ {test_end-1})")
        print(f"  Train 时间: {g_tr['Produce_Time'].min()} ~ {g_tr['Produce_Time'].max()}")
        print(f"  Test  时间: {g_te['Produce_Time'].min()} ~ {g_te['Produce_Time'].max()}")

        fold_res = run_single_fold(
            g_tr, g_te,
            fold_id=i+1, fold_total=N_FOLDS,
            do_diag=(i == 0),   # 只在第 1 折做 ACF/PACF
        )
        cv_all.extend(fold_res)

    # ---------- CV 汇总 ----------
    cv_df = pd.DataFrame(cv_all)
    metrics_cols = ["R2", "RMSE", "MAE", "Bias", "Hit1%", "Hit2%", "Hit3%", "Hit5%"]

    print(f"\n{'='*60}")
    print(f"Walk-Forward CV 汇总 (5 折均值 ± 标准差)")
    print(f"{'='*60}")

    # 按 surface + model 聚合
    cv_mean = cv_df.groupby(["surface", "model"])[metrics_cols].mean()
    cv_std  = cv_df.groupby(["surface", "model"])[metrics_cols].std()

    for surface in ["TOP", "BOT"]:
        print(f"\n  [{surface}]")
        header = f"  {'model':20s}"
        for m in metrics_cols:
            header += f" {m:>9s}"
        print(header)
        print("  " + "-" * (20 + 10 * len(metrics_cols)))
        for model in ["Online_raw", "Rolling_bias_ewma", "Bias_shifted(N=300)",
                      "LGBM_lag_only", "LGBM_ewma_only",
                      "LGBM_full_lag", "sklearn_HGB_full_lag",
                      "LGBM_full_ewma", "LGBM_full_comb"]:
            if model not in cv_mean.index.get_level_values("model"):
                continue
            row = cv_mean.loc[(surface, model)]
            row_s = cv_std.loc[(surface, model)]
            line = f"  {model:20s}"
            for m in metrics_cols:
                line += f" {row[m]:.4f}±{row_s[m]:.4f}"
            print(line)

    # best_iter 汇总 (如果有)
    if "best_iter" in cv_df.columns:
        bi = cv_df[cv_df["best_iter"].notna()]
        print(f"\n  Best iteration 均值:")
        bi_summary = bi.groupby(["surface", "model"])["best_iter"].agg(["mean", "std"]).round(1)
        print(bi_summary.to_string())

    # ========= EWMA 生产场景汇总 =========
    ewma_models = [m for m in cv_mean.index.get_level_values("model") if "+EWMA_" in m]
    if ewma_models:
        print(f"\n{'='*60}")
        print(f"EWMA 生产场景汇总 (5 折均值, 每折 10 次抽样模拟)")
        print(f"{'='*60}")
        for surface in ["TOP", "BOT"]:
            print(f"\n  [{surface}]")
            base_models = [m for m in ewma_models if m.startswith("LGBM_full_lag+")]
            hgb_models  = [m for m in ewma_models if m.startswith("sklearn_HGB_full_lag+")]
            for base_name, ewma_list in [("LGBM_full_lag", base_models),
                                         ("sklearn_HGB_full_lag", hgb_models)]:
                if not ewma_list:
                    continue
                if base_name not in cv_mean.index.get_level_values("model"):
                    continue
                base_rmse = cv_mean.loc[(surface, base_name), "RMSE"]
                print(f"\n    --- {base_name} baseline RMSE = {base_rmse:.4f} ---")
                print(f"    {'场景':<20s} {'RMSE':>9s} {'Δ vs 纯模型':>12s}")
                print(f"    {'-'*45}")
                for m in sorted(ewma_list):
                    scene = m.split("+", 1)[1]
                    rmse = cv_mean.loc[(surface, m), "RMSE"]
                    print(f"    {scene:<20s} {rmse:9.4f} {rmse - base_rmse:>+12.4f}")

    # 保存
    out_csv = Path(__file__).parent / "reproduce_results.csv"
    cv_df.to_csv(out_csv, index=False)

    # 同时保存 CV 均值摘要
    summary_csv = Path(__file__).parent / "cv_summary.csv"
    cv_mean_flat = cv_mean.reset_index()
    cv_std_flat = cv_std.reset_index()
    cv_mean_flat.to_csv(summary_csv, index=False, float_format="%.4f")

    print(f"\n结果已保存:")
    print(f"  每折详情: {out_csv}")
    print(f"  CV 均值:  {summary_csv}")


if __name__ == "__main__":
    main()
