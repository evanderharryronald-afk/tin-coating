"""
run_ablation.py — 探索性消融实验骨架
====================================

设计目标：
  1. **零污染生产代码**：所有探索逻辑独立放这里，ablation/ 目录可整块删除
  2. **最大程度复用**：模型类 ResidualCorrectionModel 从父目录 import，避免训练逻辑漂移
  3. **可组合实验**：特征集合 / EWMA α / 切分策略 三个维度独立可调

当前提供 3 个实验入口：
  - run_feature_ablation()     特征集合消融（恢复被剪枝的特征看收益）
  - run_ewma_alpha_sweep()     EWMA alpha 扫一遍（已跑过 α=0.779 vs 1.0，扫得更密）
  - run_walk_forward_cv()      Walk-Forward 5-fold CV，测当前 baseline 稳定性

用法：
  cd Task1
  python ablation/run_ablation.py --experiment feature_abl         # 特征消融
  python ablation/run_ablation.py --experiment ewma_sweep          # EWMA 扫
  python ablation/run_ablation.py --experiment cv                  # Walk-Forward CV
  python ablation/run_ablation.py --experiment all                 # 全部依次跑
"""

import os
import sys
import argparse
import warnings
from pathlib import Path

# -----------------------------
# 1. 路径 & import 复用
# -----------------------------
# 让本脚本可以 import 父目录（coating_model_by_group 等）
_HERE = Path(__file__).resolve().parent
_TASK1 = _HERE.parent
if str(_TASK1) not in sys.path:
    sys.path.insert(0, str(_TASK1))

# 复用生产代码里的模型类（保证训练逻辑 100% 一致）
from coating_model_by_group import (
    ResidualCorrectionModel,
)

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None

# -----------------------------
# 2. 路径 & 常量
# -----------------------------
# 所有路径相对 Task1/
DATA_PATH = _TASK1 / "result" / "data" / "feature_engineered_data" / "featured_data.xlsx"
RESULTS_DIR = _HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 目标规格组
SP_TARGET = 2.799

# 列名常量（避免硬编码散落各处）
COL_TOP_ONLINE = "Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Avg"
COL_BOT_ONLINE = "Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Avg"
COL_TOP_SP_MIN = "Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_TOP_Min"
COL_BOT_SP_MIN = "Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_BOT_Min"
COL_TOP_LAB = "上表面镀层重量A(XA1_0)"
COL_BOT_LAB = "下表面镀层重量A(XA1_0)"
COL_TOP_DELTA = "Top_Delta"
COL_BOT_DELTA = "Bot_Delta"

SURFACE_PREFIX = {"Top": "Top", "Bot": "Bot"}


# ============================================================
# 3. 数据加载 & 基础工具
# ============================================================

def load_target_group(path=DATA_PATH, sp_target=SP_TARGET) -> pd.DataFrame:
    """读 featured_data.xlsx，过滤 SP_TOP=SP_BOT=sp_target 这个规格组，按时间排序。"""
    df = pd.read_excel(path)
    df = df.dropna(subset=[COL_TOP_SP_MIN, COL_BOT_SP_MIN])
    mask = (df[COL_TOP_SP_MIN].round(3) == sp_target) & (df[COL_BOT_SP_MIN].round(3) == sp_target)
    g = df.loc[mask].copy()
    # 按生产时间排序（CV 和切分都依赖时间顺序）
    g["_sort_ts"] = pd.to_datetime(g["Produce Time"], errors="coerce")
    g = g.sort_values("_sort_ts").reset_index(drop=True)
    return g


def surface_feature_cols(surface: str, key: str) -> list:
    """
    根据 key 返回该表面的特征列列表。

    约定所有 FEATURE_SETS 里定义的 key 都能在这里查到。
    prefix = Top 或 Bot，替换进 "Top_Current_Sum"、"Top_Delta" 等列名。
    """
    prefix = SURFACE_PREFIX[surface]
    online_col = COL_TOP_ONLINE if surface == "Top" else COL_BOT_ONLINE

    # 基础列（所有集合都有）
    base = [online_col, f"{prefix}_Current_Sum", "Speed[m/min]_Process_Avg",
            "Dimension_[mm]_Width", "Dimension_[mm]_Thickness"]

    extra_map = {
        "current_ps":      f"{prefix}_Current_Per_Speed",
        "theoretical":     f"{prefix}_Theoretical_Factor",
        "steel_grade":     "Steel_Grade_Encoded",
    }

    sets = {
        "baseline_5":    base[:],                                     # 当前生产代码用的 5 个
        "+current_ps":   base + [extra_map["current_ps"]],            # 加电流/车速比 (= 6)
        "+theoretical":  base + [extra_map["theoretical"]],           # 加理论因子 (= 6)
        "+grade":        base + [extra_map["steel_grade"]],           # 加钢种编码 (= 6)
        "+ps_grade":     base + [extra_map["current_ps"], extra_map["steel_grade"]],   # = 7
        "+all_8":        base + [extra_map["current_ps"],             # 三者全开 (= 8)
                                 extra_map["theoretical"],
                                 extra_map["steel_grade"]],
        # 也可以在这里加更多集合（比如去掉 Current_Sum 只留 Current_Per_Speed 看谁更重要）
    }
    if key not in sets:
        raise KeyError(f"未知特征集合 key: {key}；可选: {list(sets.keys())}")
    return sets[key]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """简单的 RMSE / MAE / 偏差均值，返回 dict。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    bias = (y_pred - y_true).mean()
    return {"RMSE": rmse, "MAE": mae, "Bias": bias}


# ============================================================
# 4. 单次 hold-out 评估（与生产代码完全同口径）
# ============================================================

def run_single_fold(df: pd.DataFrame, surface: str, feature_cols: list,
                    alpha_smoothing: float, train_ratio: float = 0.65,
                    val_ratio: float = 0.20, params: dict = None) -> dict:
    """
    与 coating_model_by_group.fit_and_evaluate_surface 完全同口径：
      - 时间顺序切分 Train / Val / Test
      - ResidualCorrectionModel 训练（可配置 damping / pos_boost 等）
      - alpha_smoothing 控制预测后的 EWMA 平滑（=1.0 等于无 EWMA）
    返回包含 train/val/test 三份 metrics 以及 online baseline 的 dict。
    """
    prefix = SURFACE_PREFIX[surface]
    online_col = COL_TOP_ONLINE if surface == "Top" else COL_BOT_ONLINE
    delta_col = COL_TOP_DELTA if surface == "Top" else COL_BOT_DELTA
    lab_col = COL_TOP_LAB if surface == "Top" else COL_BOT_LAB

    X = df[feature_cols].copy()
    y_delta = df[delta_col].copy()
    online_actual = df[online_col].copy()
    y_true = df[lab_col].copy()

    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train, y_delta_train = X.iloc[:train_end], y_delta.iloc[:train_end]
    actual_train, y_true_train = online_actual.iloc[:train_end], y_true.iloc[:train_end]

    X_val = X.iloc[train_end:val_end]
    actual_val, y_true_val = online_actual.iloc[train_end:val_end], y_true.iloc[train_end:val_end]

    X_test = X.iloc[val_end:]
    actual_test, y_true_test = online_actual.iloc[val_end:], y_true.iloc[val_end:]

    # 默认超参就是当前生产的那套（从 group_params_2_799_only.json 里摘的）
    default_top = {"damping": 0.3046, "pos_boost": 1.6837, "max_iter": 250,
                   "learning_rate": 0.0144, "max_depth": 5}
    default_bot = {"damping": 0.2848, "pos_boost": 1.2582, "max_iter": 300,
                   "learning_rate": 0.0117, "max_depth": 4}
    base_params = default_top if surface == "Top" else default_bot
    if params is not None:
        base_params.update(params)
    base_params["alpha_smoothing"] = alpha_smoothing   # alpha 统一用实验值覆盖

    model = ResidualCorrectionModel(
        alpha_smoothing=base_params["alpha_smoothing"],
        pos_boost=base_params["pos_boost"],
        damping=base_params["damping"],
        max_iter=base_params["max_iter"],
        learning_rate=base_params["learning_rate"],
        max_depth=base_params["max_depth"],
    )

    # damping=0 等权（生产代码默认），ablation 里先不开方向权重复杂性
    model.fit(X_train, y_delta_train)

    # baseline：直接用在线值当预测
    res = {}
    for tag, X_, actual_, yt_ in [
        ("train", X_train, actual_train, y_true_train),
        ("val",   X_val,   actual_val,   y_true_val),
        ("test",  X_test,  actual_test,  y_true_test),
    ]:
        pred_model, _ = model.predict_smooth(X_, actual_)
        m_online = compute_metrics(yt_.values, actual_.values)
        m_model  = compute_metrics(yt_.values, pred_model.values)
        res[f"{tag}_online"] = m_online
        res[f"{tag}_model"]  = m_model
        res[f"{tag}_improve_pct"] = {
            "RMSE": (m_model["RMSE"] - m_online["RMSE"]) / m_online["RMSE"] * 100,
            "MAE":  (m_model["MAE"]  - m_online["MAE"])  / m_online["MAE"]  * 100,
        }
    res["n"] = {"total": n, "train": train_end, "val": val_end - train_end, "test": n - val_end}
    return res


# ============================================================
# 5. Walk-Forward 5-fold CV（重要：之前只做过 largest_group_repro，
#    现在搬到 coating_model_by_group 这套 5-特征 HGB 上来）
# ============================================================

def _cv_engine(df: pd.DataFrame, surface: str, feature_cols: list,
               alpha_smoothing: float, n_folds: int = 5,
               params: dict = None) -> pd.DataFrame:
    """
    Walk-Forward：
      fold 1: train [0,70%)      → test [70%,75%)
      fold 2: train [0,75%)      → test [75%,80%)
      ...
      fold 5: train [0,90%)      → test [90%,100%)
    每折独立训练，算 test RMSE/MAE，最后汇总均值±std。
    """
    prefix = SURFACE_PREFIX[surface]
    online_col = COL_TOP_ONLINE if surface == "Top" else COL_BOT_ONLINE
    delta_col = COL_TOP_DELTA if surface == "Top" else COL_BOT_DELTA
    lab_col = COL_TOP_LAB if surface == "Top" else COL_BOT_LAB

    X = df[feature_cols].copy()
    y_delta = df[delta_col].copy()
    online_actual = df[online_col].copy()
    y_true = df[lab_col].copy()
    n = len(X)

    # 切分锚点
    cut_pts = np.linspace(int(n * 0.70), int(n * 0.90), n_folds, dtype=int)
    folds = []
    for i, cut in enumerate(cut_pts):
        test_start = cut
        test_end = int(cut + n * 0.05)  # test 占 5%
        if test_end > n:
            test_end = n
        if test_start >= test_end:
            break
        folds.append((0, cut, test_start, test_end))

    # 复用默认超参（与 hold-out 保持一致）
    default_top = {"damping": 0.3046, "pos_boost": 1.6837, "max_iter": 250,
                   "learning_rate": 0.0144, "max_depth": 5}
    default_bot = {"damping": 0.2848, "pos_boost": 1.2582, "max_iter": 300,
                   "learning_rate": 0.0117, "max_depth": 4}
    base = default_top if surface == "Top" else default_bot
    if params is not None:
        base.update(params)
    base["alpha_smoothing"] = alpha_smoothing

    rows = []
    for i, (tr_start, tr_end, te_start, te_end) in enumerate(folds):
        X_tr, y_tr = X.iloc[tr_start:tr_end], y_delta.iloc[tr_start:tr_end]
        act_tr = online_actual.iloc[tr_start:tr_end]
        X_te = X.iloc[te_start:te_end]
        act_te, yt_te = online_actual.iloc[te_start:te_end], y_true.iloc[te_start:te_end]

        model = ResidualCorrectionModel(
            alpha_smoothing=base["alpha_smoothing"],
            pos_boost=base["pos_boost"], damping=base["damping"],
            max_iter=base["max_iter"], learning_rate=base["learning_rate"],
            max_depth=base["max_depth"],
        )
        model.fit(X_tr, y_tr)

        pred_model, _ = model.predict_smooth(X_te, act_te)
        m_online = compute_metrics(yt_te.values, act_te.values)
        m_model  = compute_metrics(yt_te.values, pred_model.values)

        rows.append({
            "fold": i + 1,
            "train_n": tr_end - tr_start,
            "test_n": te_end - te_start,
            "test_online_RMSE": m_online["RMSE"],
            "test_online_MAE":  m_online["MAE"],
            "test_model_RMSE":  m_model["RMSE"],
            "test_model_MAE":   m_model["MAE"],
            "improve_RMSE_pct": (m_model["RMSE"] - m_online["RMSE"]) / m_online["RMSE"] * 100,
        })

    df_folds = pd.DataFrame(rows)
    if len(df_folds) == 0:
        return df_folds
    summary = {
        "surface":         surface,
        "n_folds":         len(df_folds),
        "model_RMSE_mean": df_folds["test_model_RMSE"].mean(),
        "model_RMSE_std":  df_folds["test_model_RMSE"].std(),
        "model_MAE_mean":  df_folds["test_model_MAE"].mean(),
        "model_MAE_std":   df_folds["test_model_MAE"].std(),
        "improve_RMSE_pct_mean": df_folds["improve_RMSE_pct"].mean(),
        "improve_RMSE_pct_std":  df_folds["improve_RMSE_pct"].std(),
    }
    # 把 summary 附加一份到 folds 表里（长表方便画 bar/box）
    df_folds.attrs["summary"] = summary
    return df_folds


# ============================================================
# 6. 三个实验入口
# ============================================================

def run_feature_ablation():
    """特征集合消融：逐个加回 Current_Per_Speed / Theoretical_Factor / Steel_Grade。"""
    print("\n" + "=" * 70)
    print(" 实验 1 — 特征集合消融 (Top/Bot, hold-out)")
    print("=" * 70)

    df = load_target_group()
    print(f"样本量: {len(df)}")

    feature_keys = ["baseline_5", "+current_ps", "+theoretical", "+grade", "+ps_grade", "+all_8"]

    records = []
    for surface in ["Top", "Bot"]:
        for key in feature_keys:
            cols = surface_feature_cols(surface, key)
            res = run_single_fold(df, surface, cols, alpha_smoothing=0.779 if surface == "Top" else 0.727)

            row = {
                "surface":    surface,
                "feature_set": key,
                "n_features": len(cols),
                # test 集结果（主要看这个）
                "test_online_RMSE":  res["test_online"]["RMSE"],
                "test_online_MAE":   res["test_online"]["MAE"],
                "test_model_RMSE":   res["test_model"]["RMSE"],
                "test_model_MAE":    res["test_model"]["MAE"],
                "test_improve_RMSE_pct": res["test_improve_pct"]["RMSE"],
                "test_improve_MAE_pct":  res["test_improve_pct"]["MAE"],
                # val 集（防过拟合）
                "val_model_RMSE":    res["val_model"]["RMSE"],
                "val_model_MAE":     res["val_model"]["MAE"],
                "features":          " | ".join(cols),
            }
            records.append(row)
            print(f"  {surface:>3}  {key:<14}  feat={len(cols)}  "
                  f"test_RMSE={row['test_model_RMSE']:.4f}  "
                  f"test_MAE={row['test_model_MAE']:.4f}  "
                  f"RMSE提升={row['test_improve_RMSE_pct']:.1f}%")

    out = pd.DataFrame(records)
    save_path = RESULTS_DIR / "feature_ablation_summary.csv"
    out.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"\n[✓] 已保存: {save_path}")
    print("\n== Top 表面 ==")
    print(out[out["surface"] == "Top"][["feature_set", "n_features",
          "test_model_RMSE", "test_model_MAE", "test_improve_RMSE_pct"]].to_string(index=False))
    print("\n== Bot 表面 ==")
    print(out[out["surface"] == "Bot"][["feature_set", "n_features",
          "test_model_RMSE", "test_model_MAE", "test_improve_RMSE_pct"]].to_string(index=False))


def run_ewma_alpha_sweep():
    """EWMA alpha 扫一遍：0.0, 0.3, 0.5, 0.7, 0.727, 0.779, 1.0"""
    print("\n" + "=" * 70)
    print(" 实验 2 — EWMA alpha 扫 (Top/Bot, hold-out, baseline_5 特征)")
    print("=" * 70)

    df = load_target_group()
    feature_cols = surface_feature_cols("Top", "baseline_5")  # Top/Bot 特征列相同结构，key 通用

    # pandas ewm 要求 alpha ∈ (0, 1]，极接近 1.0 就是"几乎不滤波"
    alphas = [0.01, 0.3, 0.5, 0.7, 0.727, 0.779, 1.0]

    records = []
    for surface in ["Top", "Bot"]:
        cols = surface_feature_cols(surface, "baseline_5")
        for alpha in alphas:
            res = run_single_fold(df, surface, cols, alpha_smoothing=alpha)
            row = {
                "surface":  surface,
                "alpha":    alpha,
                "test_online_RMSE": res["test_online"]["RMSE"],
                "test_model_RMSE":  res["test_model"]["RMSE"],
                "test_model_MAE":   res["test_model"]["MAE"],
                "test_improve_RMSE_pct": res["test_improve_pct"]["RMSE"],
            }
            records.append(row)
            tag = " (生产默认)" if (surface == "Top" and abs(alpha - 0.779) < 0.001) or \
                                   (surface == "Bot" and abs(alpha - 0.727) < 0.001) else ""
            print(f"  {surface:>3}  α={alpha:<5}  test_RMSE={row['test_model_RMSE']:.4f}"
                  f"  MAE={row['test_model_MAE']:.4f}{tag}")

    out = pd.DataFrame(records)
    save_path = RESULTS_DIR / "ewma_alpha_sweep_summary.csv"
    out.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"\n[✓] 已保存: {save_path}")


# ============================================================
# 6.5 HGB 超参网格敏感性扫描（+all_8 特征下验证旧超参是否最优）
# ============================================================

HGB_GRID = {
    "Top": {
        "max_depth":     [4, 5, 6, 7],
        "learning_rate":  [0.008, 0.0144, 0.020, 0.030],
        "max_iter":      [150, 250, 400],
    },
    "Bot": {
        "max_depth":     [3, 4, 5, 6],
        "learning_rate":  [0.008, 0.0117, 0.018, 0.025],
        "max_iter":      [200, 300, 500],
    },
}

# 旧超参在网格里的位置（用来在输出里标记"生产当前值"）
CURRENT_MARK = {
    "Top": {"max_depth": 5, "learning_rate": 0.0144, "max_iter": 250},
    "Bot": {"max_depth": 4, "learning_rate": 0.0117, "max_iter": 300},
}


def run_hgb_grid_sweep():
    """
    在 +all_8 特征上，对 HGB 三个关键超参做网格扫描。
    目的：验证 Optuna 针对 baseline_5 搜出来的超参，在 +all_8 下是否仍接近最优。
    """
    from itertools import product

    print("\n" + "=" * 70)
    print(" 实验 4 — HGB 超参网格扫描 (+all_8 特征, hold-out)")
    print("=" * 70)

    df = load_target_group()
    records = []

    for surface in ["Top", "Bot"]:
        cols = surface_feature_cols(surface, "+all_8")
        alpha = 0.6   # 当前推荐的两表面统一 α

        grid = HGB_GRID[surface]
        keys = list(grid.keys())
        values = [grid[k] for k in keys]

        total = 1
        for v in values:
            total *= len(v)
        print(f"\n  {surface}: 特征={len(cols)}, α={alpha}, 网格组合数={total}")

        for combo in product(*values):
            hp = dict(zip(keys, combo))
            res = run_single_fold(df, surface, cols, alpha_smoothing=alpha,
                                  params=hp)

            row = {
                "surface":  surface,
                **{f"hp_{k}": v for k, v in hp.items()},
                "test_model_RMSE": res["test_model"]["RMSE"],
                "test_model_MAE":  res["test_model"]["MAE"],
                "val_model_RMSE":  res["val_model"]["RMSE"],
                "val_model_MAE":   res["val_model"]["MAE"],
                "train_model_RMSE": res["train_model"]["RMSE"],
                "improve_RMSE_pct": res["test_improve_pct"]["RMSE"],
            }
            row["is_current"] = all(
                abs(hp[k] - CURRENT_MARK[surface][k]) < 1e-9 for k in keys
            )
            records.append(row)

    out = pd.DataFrame(records)
    save_path = RESULTS_DIR / "hgb_grid_sweep_all_8.csv"
    out.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"\n[✓] 已保存: {save_path}")

    # ---- 打印每个表面的 Top-5 + 当前点 ----
    for surface in ["Top", "Bot"]:
        sub = out[out["surface"] == surface].copy()
        sub = sub.sort_values("test_model_RMSE")
        best_row = sub.iloc[0]
        cur_rows = sub[sub["is_current"]]
        cur_row = cur_rows.iloc[0] if len(cur_rows) else None

        print(f"\n{'='*60}")
        print(f"  {surface} 表面")
        print(f"{'='*60}")

        print(f"\n  🥇 网格最优:")
        print(f"    depth={best_row['hp_max_depth']}, "
              f"lr={best_row['hp_learning_rate']:.4f}, "
              f"iter={int(best_row['hp_max_iter'])}")
        print(f"    test_RMSE={best_row['test_model_RMSE']:.4f}, "
              f"test_MAE={best_row['test_model_MAE']:.4f}, "
              f"提升={best_row['improve_RMSE_pct']:.1f}%")
        print(f"    val_RMSE={best_row['val_model_RMSE']:.4f}, "
              f"train_RMSE={best_row['train_model_RMSE']:.4f}")

        if cur_row is not None:
            delta = best_row["test_model_RMSE"] - cur_row["test_model_RMSE"]
            pct = delta / cur_row["test_model_RMSE"] * 100
            print(f"\n  📌 当前生产超参 (baseline_5 Optuna 搜索结果, 在 +all_8 上):")
            print(f"    depth={cur_row['hp_max_depth']}, "
                  f"lr={cur_row['hp_learning_rate']:.4f}, "
                  f"iter={int(cur_row['hp_max_iter'])}")
            print(f"    test_RMSE={cur_row['test_model_RMSE']:.4f}, "
                  f"test_MAE={cur_row['test_model_MAE']:.4f}")
            if abs(pct) < 1.0:
                verdict = "✅ 几乎持平（影响 < 1%），旧超参仍可用"
            elif delta > 0:
                verdict = f"⚠️ 旧超参偏离最优 ({delta:+.4f}, {pct:+.1f}%)，值得更新"
            else:
                verdict = f"✅ 旧超参刚好最优 ({delta:+.4f})"
            print(f"    → {verdict}")

        print(f"\n  📊 Top-10 组合:")
        top10 = sub.head(10)
        display = top10[["hp_max_depth", "hp_learning_rate", "hp_max_iter",
                         "test_model_RMSE", "val_model_RMSE", "train_model_RMSE",
                         "improve_RMSE_pct"]].copy()
        display["hp_max_iter"] = display["hp_max_iter"].astype(int)
        print(display.to_string(index=False))

        # 诊断：过拟合严重吗？
        print(f"\n  🎯 最优组合过拟合检查: "
              f"(train→val→test gap)")
        b = best_row
        print(f"    train RMSE = {b['train_model_RMSE']:.4f}")
        print(f"    val   RMSE = {b['val_model_RMSE']:.4f}  "
              f"(+{(b['val_model_RMSE'] - b['train_model_RMSE']) / b['train_model_RMSE'] * 100:+.1f}%)")
        print(f"    test  RMSE = {b['test_model_RMSE']:.4f}  "
              f"(+{(b['test_model_RMSE'] - b['train_model_RMSE']) / b['train_model_RMSE'] * 100:+.1f}%)")


def run_walk_forward_cv():
    """Walk-Forward 5-fold CV，测 baseline_5 特征 + 生产默认 alpha 的稳定性。"""
    print("\n" + "=" * 70)
    print(" 实验 3 — Walk-Forward 5-fold CV (Top/Bot, baseline_5)")
    print("=" * 70)

    df = load_target_group()
    all_rows = []

    for surface in ["Top", "Bot"]:
        cols = surface_feature_cols(surface, "baseline_5")
        alpha = 0.779 if surface == "Top" else 0.727
        df_folds = _cv_engine(df, surface, cols, alpha_smoothing=alpha, n_folds=5)
        if len(df_folds) == 0:
            print(f"  {surface}: 切分失败，跳过")
            continue

        summary = df_folds.attrs["summary"]
        print(f"\n  — {surface}  —  ({summary['n_folds']} folds)")
        print(f"    model RMSE:  {summary['model_RMSE_mean']:.4f} ± {summary['model_RMSE_std']:.4f}")
        print(f"    model MAE:   {summary['model_MAE_mean']:.4f} ± {summary['model_MAE_std']:.4f}")
        print(f"    RMSE 提升:   {summary['improve_RMSE_pct_mean']:.1f}% ± {summary['improve_RMSE_pct_std']:.1f}%")
        print(df_folds[["fold", "train_n", "test_n", "test_model_RMSE",
                        "test_model_MAE", "improve_RMSE_pct"]].to_string(index=False))

        df_folds = df_folds.copy()
        df_folds["surface"] = surface
        df_folds["alpha"] = alpha
        df_folds["n_features"] = len(cols)
        all_rows.append(df_folds)

    if all_rows:
        out = pd.concat(all_rows, ignore_index=True)
        save_path = RESULTS_DIR / "walk_forward_cv_folds.csv"
        out.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"\n[✓] 已保存: {save_path}")


# ============================================================
# 7. 入口
# ============================================================

EXPERIMENTS = {
    "feature_abl":    run_feature_ablation,
    "ewma_sweep":     run_ewma_alpha_sweep,
    "cv":             run_walk_forward_cv,
    "hgb_grid":       run_hgb_grid_sweep,       # +all_8 上扫 HGB 三个关键超参
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ablation 实验骨架")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=list(EXPERIMENTS.keys()) + ["all"],
                        help="要跑的实验 (默认 all)")
    args = parser.parse_args()

    print("\n" + "#" * 70)
    print("#  Ablation 实验骨架 — Tin Coating Bias Prediction")
    print("#" * 70)
    print(f"数据: {DATA_PATH}")
    print(f"结果: {RESULTS_DIR}")
    print(f"目标规格组: Top{SP_TARGET}_Bot{SP_TARGET}")

    if args.experiment == "all":
        for name, fn in EXPERIMENTS.items():
            try:
                fn()
            except Exception as e:
                import traceback
                print(f"\n[✗] {name} 报错: {e}")
                traceback.print_exc()
    else:
        EXPERIMENTS[args.experiment]()

    print("\n" + "=" * 70)
    print("全部完成。结果文件在 ablation/results/ 下。")
    print("=" * 70)
