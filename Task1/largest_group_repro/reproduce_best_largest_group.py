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

    base_feats = [
        "Thickness", "Width", "Speed",
        "Top_Current_Sum", "Bot_Current_Sum",
        "Top_Current_Density", "Bot_Current_Density",
        "Online_SP_Diff_TOP", "Online_SP_Diff_BOT",
        "Speed_x_Thickness", "Top_Bot_Online_Diff", "Top_Bot_Current_Ratio",
        "Hour", "DayOfWeek", "Steel_Grade_enc",
    ] + lag_cols

    fold_results = []

    for surface in ["TOP", "BOT"]:
        online_col = f"Online_{surface}"
        lab_col = f"Lab_{surface}"
        delta_col = f"Delta_{surface}"
        feats = base_feats + [online_col]

        n_val = int(len(g_train) * 0.15)
        g_tr_main = g_train.iloc[:-n_val].copy()
        g_tr_val  = g_train.iloc[-n_val:].copy()
        yte = g_test[lab_col].values
        online_te = g_test[online_col].values

        # 模型 0: Online_raw
        m0 = evaluate(yte, online_te, online=online_te)
        print_metrics(m0, f"F{fold_id} {surface} | baseline | Online_raw")
        fold_results.append({"fold": fold_id, "surface": surface, "model": "Online_raw", **m0})

        # 模型 1: Bias 零模型
        tail_n = min(300, len(g_train))
        bias_train_tail = g_train[delta_col].iloc[-tail_n:].mean()
        pred_bias = online_te + bias_train_tail
        m1 = evaluate(yte, pred_bias, online=online_te)
        print_metrics(m1, f"F{fold_id} {surface} | zero | Bias_shifted (tail mean={bias_train_tail:+.4f})")
        fold_results.append({"fold": fold_id, "surface": surface, "model": f"Bias_shifted(N={tail_n})", **m1})

        # 模型 2: 纯 lag-only LGBM
        lag_only_feats = lag_cols
        model_lag = lgb.LGBMRegressor(**LGB_PARAMS)
        model_lag.fit(
            g_tr_main[lag_only_feats].values, g_tr_main[delta_col].values,
            eval_set=[(g_tr_val[lag_only_feats].values, g_tr_val[delta_col].values)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        pred_lag = online_te + model_lag.predict(g_test[lag_only_feats].values)
        m2 = evaluate(yte, pred_lag, online=online_te)
        print(f"  Best iteration (lag-only): {model_lag.best_iteration_}")
        print_metrics(m2, f"F{fold_id} {surface} | LAG-only | LGBM ({len(lag_only_feats)} feats)")
        fold_results.append({"fold": fold_id, "surface": surface, "model": "LGBM_lag_only", **m2,
                             "best_iter": model_lag.best_iteration_})

        # 模型 3: 全特征 LGBM
        model_full = lgb.LGBMRegressor(**LGB_PARAMS)
        model_full.fit(
            g_tr_main[feats].values, g_tr_main[delta_col].values,
            eval_set=[(g_tr_val[feats].values, g_tr_val[delta_col].values)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        pred_full = online_te + model_full.predict(g_test[feats].values)
        m3 = evaluate(yte, pred_full, online=online_te)
        print(f"  Best iteration (full): {model_full.best_iteration_}")
        print_metrics(m3, f"F{fold_id} {surface} | FULL | LGBM ({len(feats)} feats)")
        fold_results.append({"fold": fold_id, "surface": surface, "model": "LGBM_full", **m3,
                             "best_iter": model_full.best_iteration_})

    return fold_results


# ===================== 主流程 =====================
def main():
    print(f"数据路径: {DATA_PATH}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"找不到数据文件: {DATA_PATH}")

    df = load_and_fe(DATA_PATH)
    print(f"全量样本数: {len(df)}")

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
        for model in ["Online_raw", "Bias_shifted(N=300)", "LGBM_lag_only", "LGBM_full"]:
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
