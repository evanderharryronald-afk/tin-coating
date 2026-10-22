#!/usr/bin/env python3
"""
复现当前最优软测量结果（在线镀层测厚仪偏差校正）

依赖:
    pip install pandas numpy openpyxl scikit-learn

用法:
    python reproduce_optimal_soft_sensor.py
    python reproduce_optimal_soft_sensor.py /path/to/cleaned_data.xlsx

说明:
    - 严格按 Produce Time 时间顺序划分: 前 80% 训练, 后 20% 测试
    - 模型: HistGradientBoosting 预测 δ = Lab - Online
    - 后处理: 因果残差 EWMA
    - 分别报告 2.799/2.799 规格组 与 全量数据
    - 评估指标均在测试集上计算
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings("ignore")

# ===================== 配置 =====================
DATA_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent/"result"/ "data" /"cleaned_data"/ "cleaned_data.xlsx"


RANDOM_STATE = 42

# HistGBDT 超参（与实验一致）
HGB_PARAMS = dict(
    max_iter=350,
    learning_rate=0.025,
    max_depth=5,
    min_samples_leaf=15,
    l2_regularization=0.15,
    random_state=RANDOM_STATE,
)

# EWMA 平滑系数
ALPHA_2799 = 0.25
ALPHA_FULL = 0.20


# ===================== 工具函数 =====================
def hit_rate(y_true, y_pred, pct: float) -> float:
    rel = np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-8)
    return float((rel <= pct / 100.0).mean())


def evaluate(y_true, y_pred, online=None) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    res = {
        "R2": r2_score(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Bias": float((y_pred - y_true).mean()),
        "Hit2%": hit_rate(y_true, y_pred, 2),
        "Hit3%": hit_rate(y_true, y_pred, 3),
    }
    if online is not None:
        online = np.asarray(online, dtype=float)
        high = online > y_true  # 在线偏高
        low = online < y_true   # 在线偏低
        if high.sum() >= 5:
            res["RMSE_High"] = float(np.sqrt(mean_squared_error(y_true[high], y_pred[high])))
            res["MAE_High"] = float(mean_absolute_error(y_true[high], y_pred[high]))
            res["N_High"] = int(high.sum())
        if low.sum() >= 5:
            res["RMSE_Low"] = float(np.sqrt(mean_squared_error(y_true[low], y_pred[low])))
            res["MAE_Low"] = float(mean_absolute_error(y_true[low], y_pred[low]))
            res["N_Low"] = int(low.sum())
    return res


def print_metrics(m: dict, title: str):
    print(f"\n===== {title} =====")
    order = [
        "R2", "RMSE", "MAE", "Bias", "Hit2%", "Hit3%",
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


def ewma_causal(pred_uncorrected: np.ndarray, y_true: np.ndarray,
                alpha: float, init_bias: float = 0.0):
    """
    因果 EWMA 残差校正:
        输出 = 未校正预测 + bias
        然后用 (真值 - 未校正预测) 更新 bias
    上线时: 仅在有实验室结果时更新 bias; 无结果时保持 bias 不变。
    """
    bias = float(init_bias)
    out = np.empty_like(pred_uncorrected, dtype=float)
    for i in range(len(pred_uncorrected)):
        out[i] = pred_uncorrected[i] + bias
        r = y_true[i] - pred_uncorrected[i]
        bias = alpha * r + (1.0 - alpha) * bias
    return out, bias


# ===================== 数据与特征 =====================
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
        "Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_TOP_Min": "SP_TOP_Min",
        "Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_BOT_Min": "SP_BOT_Min",
        "Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Avg": "Online_TOP",
        "Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Avg": "Online_BOT",
        "上表面镀层重量A(XA1_0)": "Lab_TOP",
        "下表面镀层重量A(XA1_0)": "Lab_BOT",
        "上表面镀层重量C(XA1_0)": "Lab_TOP_C",
        "上表面镀层重量D(XA1_0)": "Lab_TOP_D",
        "上表面镀层重量W(XA1_0)": "Lab_TOP_W",
        "下表面镀层重量C(XA1_0)": "Lab_BOT_C",
        "下表面镀层重量D(XA1_0)": "Lab_BOT_D",
        "下表面镀层重量W(XA1_0)": "Lab_BOT_W",
    }
    for i in range(1, 37):
        rename[f"Tining Section_CURRENT[A]_GL_{i}_Avg"] = f"GL_{i}"
    df = df.rename(columns=rename)

    df["Produce_Time"] = pd.to_datetime(df["Produce_Time"])
    df = df.sort_values("Produce_Time").reset_index(drop=True)

    # 电流聚合: 奇数槽=下表面, 偶数槽=上表面
    bot_idx = list(range(1, 37, 2))
    top_idx = list(range(2, 37, 2))
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
    df["Hour"] = df["Produce_Time"].dt.hour
    df["DayOfWeek"] = df["Produce_Time"].dt.dayofweek
    df["SP_TOP_r"] = df["SP_TOP_Min"].round(3)
    df["SP_BOT_r"] = df["SP_BOT_Min"].round(3)

    # 多点平均（可选训练目标）
    df["Lab_TOP_mean"] = df[["Lab_TOP", "Lab_TOP_C", "Lab_TOP_D", "Lab_TOP_W"]].mean(axis=1)
    df["Lab_BOT_mean"] = df[["Lab_BOT", "Lab_BOT_C", "Lab_BOT_D", "Lab_BOT_W"]].mean(axis=1)
    df["Delta_TOP_mean"] = df["Lab_TOP_mean"] - df["Online_TOP"]
    df["Delta_BOT_mean"] = df["Lab_BOT_mean"] - df["Online_BOT"]

    le = LabelEncoder()
    df["Steel_Grade_enc"] = le.fit_transform(df["Steel_Grade"].astype(str))

    return df


def time_split(df: pd.DataFrame, test_ratio: float = 0.2):
    """前 (1-test_ratio) 训练, 后 test_ratio 测试（严格时间顺序）"""
    n = len(df)
    n_test = int(n * test_ratio)
    n_train = n - n_test
    return df.iloc[:n_train].copy(), df.iloc[n_train:].copy()


# ===================== 单表面训练与评估 =====================
def run_one_surface(
    train: pd.DataFrame,
    test: pd.DataFrame,
    surface: str,
    feat_cols: list,
    alpha: float,
    use_mean_target: bool = True,
):
    online_col = f"Online_{surface}"
    lab_col = f"Lab_{surface}"  # 评估始终对 Lab A

    if use_mean_target:
        delta_col = f"Delta_{surface}_mean"
        lab_train_col = f"Lab_{surface}_mean"
    else:
        delta_col = f"Delta_{surface}"
        lab_train_col = lab_col

    need = feat_cols + [delta_col, lab_col, lab_train_col, online_col]
    tr = train.dropna(subset=need).copy()
    te = test.dropna(subset=need).copy()

    Xtr = tr[feat_cols].values
    ytr = tr[delta_col].values
    Xte = te[feat_cols].values
    online_te = te[online_col].values
    online_tr = tr[online_col].values
    yte = te[lab_col].values
    ytr_lab = tr[lab_train_col].values

    model = HistGradientBoostingRegressor(**HGB_PARAMS)
    model.fit(Xtr, ytr)

    delta_tr = model.predict(Xtr)
    delta_te = model.predict(Xte)
    pred_tr_raw = online_tr + delta_tr
    pred_te_raw = online_te + delta_te

    # 用训练集残差预热 bias
    _, init_bias = ewma_causal(pred_tr_raw, ytr_lab, alpha=alpha, init_bias=0.0)

    # 测试集因果 EWMA（评估对 Lab A）
    pred_te, final_bias = ewma_causal(
        pred_te_raw, yte, alpha=alpha, init_bias=init_bias
    )

    m_raw_online = evaluate(yte, online_te, online=online_te)
    m_model = evaluate(yte, pred_te_raw, online=online_te)
    m_final = evaluate(yte, pred_te, online=online_te)

    return m_raw_online, m_model, m_final, final_bias, len(tr), len(te)


# ===================== 主流程 =====================
def main():
    print(f"数据路径: {DATA_PATH}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"找不到数据文件: {DATA_PATH}\n"
            "请把 cleaned_data.xlsx 路径作为命令行参数传入，或修改脚本顶部的 DATA_PATH。"
        )

    df = load_and_fe(DATA_PATH)
    print(f"全量样本数: {len(df)}")

    base_feats = [
        "Thickness", "Width", "Speed",
        "Top_Current_Sum", "Bot_Current_Sum",
        "Top_Current_Density", "Bot_Current_Density",
        "Online_SP_Diff_TOP", "Online_SP_Diff_BOT",
        "Speed_x_Thickness", "Top_Bot_Online_Diff", "Top_Bot_Current_Ratio",
        "Hour", "DayOfWeek", "Steel_Grade_enc",
        "SP_TOP", "SP_BOT",
    ]

    all_rows = []

    # ---------- A. 2.799 / 2.799 ----------
    print("\n" + "=" * 64)
    print("A. 最大规格组 2.799 / 2.799")
    print("=" * 64)
    mask = (df["SP_TOP_r"] == 2.799) & (df["SP_BOT_r"] == 2.799)
    g = df.loc[mask].copy().reset_index(drop=True)
    print(f"样本数: {len(g)}")

    train, test = time_split(g, test_ratio=0.2)
    print(f"Train: {len(train)}, Test: {len(test)}")
    print(f"Test 时间: {test['Produce_Time'].min()} -> {test['Produce_Time'].max()}")

    for surface in ["TOP", "BOT"]:
        feats = base_feats + [f"Online_{surface}"]
        m0, m1, m2, fb, ntr, nte = run_one_surface(
            train, test, surface, feats,
            alpha=ALPHA_2799, use_mean_target=True,
        )
        print_metrics(m0, f"{surface} | 原始在线")
        print_metrics(m1, f"{surface} | HGB 预测δ（无 EWMA）")
        print_metrics(m2, f"{surface} | HGB + EWMA(α={ALPHA_2799})  [final_bias={fb:.4f}]")
        all_rows.append({"group": "2.799", "surface": surface, "model": "RAW_Online", **m0})
        all_rows.append({"group": "2.799", "surface": surface, "model": "HGB", **m1})
        all_rows.append({"group": "2.799", "surface": surface, "model": "HGB_EWMA", **m2})

    # ---------- B. 全量 ----------
    print("\n" + "=" * 64)
    print("B. 全量数据")
    print("=" * 64)
    train, test = time_split(df, test_ratio=0.2)
    print(f"Train: {len(train)}, Test: {len(test)}")
    print(f"Test 时间: {test['Produce_Time'].min()} -> {test['Produce_Time'].max()}")

    for surface in ["TOP", "BOT"]:
        feats = base_feats + [f"Online_{surface}"]
        m0, m1, m2, fb, ntr, nte = run_one_surface(
            train, test, surface, feats,
            alpha=ALPHA_FULL, use_mean_target=True,
        )
        print_metrics(m0, f"{surface} | 原始在线")
        print_metrics(m1, f"{surface} | HGB 预测δ（无 EWMA）")
        print_metrics(m2, f"{surface} | HGB + EWMA(α={ALPHA_FULL})  [final_bias={fb:.4f}]")
        all_rows.append({"group": "FULL", "surface": surface, "model": "RAW_Online", **m0})
        all_rows.append({"group": "FULL", "surface": surface, "model": "HGB", **m1})
        all_rows.append({"group": "FULL", "surface": surface, "model": "HGB_EWMA", **m2})

    # ---------- 汇总 ----------
    res_df = pd.DataFrame(all_rows)
    print("\n\n========== 汇总（测试集） ==========")
    show = ["group", "surface", "model", "R2", "RMSE", "MAE", "Bias", "Hit2%", "Hit3%"]
    print(res_df[show].round(4).to_string(index=False))

    out_csv = Path(__file__).resolve().parent / "optimal_results.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"\n结果已保存: {out_csv}")


if __name__ == "__main__":
    main()
