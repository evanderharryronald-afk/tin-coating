"""
lag 阶数对照实验: LGBM_full_lag 在 lag=5 / 10 / 15 下的 5 折 CV 对比
直接 import 主脚本的 load_and_fe, add_lag_features, add_ewma_features 确保口径一致
"""
import sys, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

# 直接 import 主脚本的函数
from reproduce_best_largest_group import (
    load_and_fe, add_lag_features, add_ewma_features, LGB_PARAMS,
)
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

DATA_PATH = str(__import__('pathlib').Path(__file__).parent.parent / "result" / "data" / "cleaned_data" / "cleaned_data.xlsx")

def cv_one_config(g, lag_max):
    """Walk-Forward 5 折 CV, 只用 LGBM_full_lag, lag 阶数 = lag_max"""
    results = []
    n = len(g)
    splits = [0.70, 0.75, 0.80, 0.85, 0.90]
    for fold_id, split_pct in enumerate(splits, 1):
        train_end = int(n * split_pct)
        g_tr = g.iloc[:train_end].copy()
        g_te = g.iloc[train_end:].copy()

        le = LabelEncoder()
        le.fit(g_tr["Steel_Grade"].astype(str))
        g_tr["Steel_Grade_enc"] = le.transform(g_tr["Steel_Grade"].astype(str))
        known = set(le.classes_)
        g_te["Steel_Grade_enc"] = g_te["Steel_Grade"].astype(str).apply(
            lambda x: le.transform([x])[0] if x in known else -1)

        # 关键修复: 用主脚本原版逻辑 (train 前 lag 行用 mean 填, 不用 dropna 丢行)
        lag_cols = []
        base_lag_cols = ["Online_SP_Diff_TOP", "Online_SP_Diff_BOT"]
        for col in base_lag_cols:
            for k in range(1, lag_max + 1):
                # train: 前 k 行 NaN 用 mean 填 (和主脚本 add_lag_features 一致)
                shifted = g_tr[col].shift(k)
                g_tr[f"{col}_lag{k}"] = shifted.fillna(g_tr[col].mean())
                # test: 从 train 末尾续接
                tail_vals = g_tr[col].iloc[-k:].values
                tshift = g_te[col].shift(k)
                tshift.iloc[:k] = tail_vals[:min(k, len(g_te))]
                g_te[f"{col}_lag{k}"] = tshift
                lag_cols.append(f"{col}_lag{k}")

        # train 可以全用 (前 lag 行已填 mean), test 也可以全用 (前 lag 行已填)
        # 只在 g_te 检查 lag 列是否有 NaN (理论上不会有)
        if g_te[lag_cols].isna().any().any():
            # 安全 dropna lag 列
            g_te = g_te.dropna(subset=lag_cols).reset_index(drop=True)

        if len(g_te) < 20:
            print(f"  fold {fold_id} lag={lag_max}: g_te = {len(g_te)} < 20, SKIP")
            continue

        base_feats = ["Thickness","Width","Speed","Top_Current_Sum","Bot_Current_Sum",
                      "Top_Current_Density","Bot_Current_Density",
                      "Online_SP_Diff_TOP","Online_SP_Diff_BOT",
                      "Speed_x_Thickness","Top_Bot_Online_Diff","Top_Bot_Current_Ratio",
                      "Hour","DayOfWeek","Steel_Grade_enc"]
        all_feats = base_feats + lag_cols

        n_val = int(len(g_tr) * 0.15)
        g_main = g_tr.iloc[:-n_val].copy(); g_val = g_tr.iloc[-n_val:].copy()

        for surface in ["TOP", "BOT"]:
            online_col = f"Online_{surface}"
            lab_col = f"Lab_{surface}"
            delta_col = f"Delta_{surface}"
            yte = g_te[lab_col].values
            online_te = g_te[online_col].values
            model = lgb.LGBMRegressor(**LGB_PARAMS)
            model.fit(g_main[all_feats].values, g_main[delta_col].values,
                      eval_set=[(g_val[all_feats].values, g_val[delta_col].values)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
            pred = online_te + model.predict(g_te[all_feats].values)
            rmse = np.sqrt(np.mean((pred - yte) ** 2))
            results.append({"fold": fold_id, "surface": surface,
                             "lag_max": lag_max, "RMSE": rmse,
                             "best_iter": model.best_iteration_,
                             "n_feats": len(all_feats)})
    return results

def add_lag(train_df, test_df, cols, lag_count):
    """自己实现版本 (避免重复导入), 和主脚本 add_lag_features 逻辑一致"""
    lag_cols = []
    for c in cols:
        for k in range(1, lag_count + 1):
            train_df[f"{c}_lag{k}"] = train_df[c].shift(k)
            test_df[f"{c}_lag{k}"] = test_df[c].shift(k)
            if k == 1:
                test_df.loc[test_df.index[0], f"{c}_lag{k}"] = train_df[c].iloc[-1]
            else:
                test_df.loc[test_df.index[0], f"{c}_lag{k}"] = train_df[f"{c}_lag{k-1}"].iloc[-1]
            lag_cols.append(f"{c}_lag{k}")
    return lag_cols

# ========== 跑 ==========
print("=" * 70)
print("lag 阶数对照实验: lag=5 / lag=10 / lag=15 (Walk-Forward 5 折 CV)")
print("=" * 70)
g = load_and_fe(__import__('pathlib').Path(DATA_PATH))
# 取最大规格组 SP=2.799
g = g[(g["SP_TOP_r"] == 2.799) & (g["SP_BOT_r"] == 2.799)].copy().reset_index(drop=True)
print(f"最大规格组样本数 N={len(g)}")

all_results = []
for lag_k in [5, 10, 15]:
    print(f"\n>>> lag={lag_k} ... ", end="", flush=True)
    res = cv_one_config(g, lag_k)
    print(f"跑了 {len(res)} 条结果 (TOP/BOT × 折数)")
    all_results.extend(res)

df_r = pd.DataFrame(all_results)
df_r.to_csv(__import__('pathlib').Path(__file__).parent / "lag_order_exp_results.csv", index=False)
print(f"\n总结果数 = {len(df_r)}")

if len(df_r) == 0:
    print("ERROR: 空结果, 调试一下")
    print("各 lag_max 数据点:")
    for lag_k in [5, 10, 15]:
        g_tr = g.iloc[:int(len(g)*0.70)].copy(); g_te = g.iloc[int(len(g)*0.70):].copy()
        _ = add_lag(g_tr, g_te, ["Online_SP_Diff_TOP","Online_SP_Diff_BOT"], lag_k)
        print(f"  lag={lag_k}: tr={len(g_tr)} dropna={len(g_tr.dropna())}, te={len(g_te)} dropna={len(g_te.dropna())}")
else:
    print("\n" + "=" * 70)
    print("结果汇总 (Walk-Forward 5 折 CV mean ± std)")
    print("=" * 70)
    summary = df_r.groupby(["surface", "lag_max"]).agg(
        RMSE_mean=("RMSE","mean"), RMSE_std=("RMSE","std"),
        best_iter_mean=("best_iter","mean"), n_feats=("n_feats","first")
    ).round(4).reset_index()

    for surface in ["TOP", "BOT"]:
        sub = summary[summary["surface"] == surface].sort_values("lag_max")
        print(f"\n  [{surface}]")
        print(f"  {'lag_max':>8s}  {'feats':>6s}  {'RMSE':>10s}  {'std':>10s}  {'best_iter':>12s}  {'vs lag=5':>14s}")
        print(f"  {'-'*66}")
        baseline = None
        for _, row in sub.iterrows():
            delta_str = ""
            if row["lag_max"] == 5:
                baseline = row["RMSE_mean"]; delta_str = "(基准)"
            else:
                d = row["RMSE_mean"] - baseline
                pct = d / baseline * 100
                delta_str = f"({d:+.4f}, {pct:+.1f}%)"
            print(f"  lag={int(row['lag_max']):>3d}  {int(row['n_feats']):>6d}  {row['RMSE_mean']:>10.4f}  {row['RMSE_std']:>10.4f}  {row['best_iter_mean']:>12.1f}  {delta_str}")

# 特征重要性 (lag=15, 折 1)
print("\n" + "=" * 70)
print("特征重要性快照 (lag=15, 折 1 train) —— lag>5 有没有被用到")
print("=" * 70)

lag_k = 15
train_end = int(len(g) * 0.70)
g_tr = g.iloc[:train_end].copy()
_ = add_lag(g_tr, g_tr.copy(), ["Online_SP_Diff_TOP","Online_SP_Diff_BOT"], lag_k)
g_tr = g_tr.dropna().reset_index(drop=True)
le = LabelEncoder(); le.fit(g_tr["Steel_Grade"].astype(str))
g_tr["Steel_Grade_enc"] = le.transform(g_tr["Steel_Grade"].astype(str))
base_feats = ["Thickness","Width","Speed","Top_Current_Sum","Bot_Current_Sum",
              "Top_Current_Density","Bot_Current_Density",
              "Online_SP_Diff_TOP","Online_SP_Diff_BOT",
              "Speed_x_Thickness","Top_Bot_Online_Diff","Top_Bot_Current_Ratio",
              "Hour","DayOfWeek","Steel_Grade_enc"]
lag_cols = [f"{c}_lag{k}" for c in ["Online_SP_Diff_TOP","Online_SP_Diff_BOT"] for k in range(1, lag_k+1)]
all_feats = base_feats + lag_cols

for surface in ["TOP", "BOT"]:
    delta_col = f"Delta_{surface}"
    model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42, verbose=-1)
    model.fit(g_tr[all_feats].values, g_tr[delta_col].values)
    imp = pd.Series(model.feature_importances_, index=all_feats).sort_values(ascending=False)
    lag_imp = imp[[i for i in imp.index if "lag" in i]]
    print(f"\n  [{surface}]")
    print(f"  base feats importance sum = {imp[[i for i in imp.index if 'lag' not in i]].sum():.0f} ({imp[[i for i in imp.index if 'lag' not in i]].sum()/imp.sum()*100:.1f}%)")
    print(f"  lag  feats importance sum = {lag_imp.sum():.0f} ({lag_imp.sum()/imp.sum()*100:.1f}%)")
    # 各 lag 阶数合计
    lag_by_k = {}
    for col in lag_cols:
        k = int(col.split("_lag")[1])
        lag_by_k[k] = lag_by_k.get(k, 0) + imp.get(col, 0)
    print(f"  lag 阶数分布:")
    for k in range(1, lag_k+1):
        v = lag_by_k.get(k, 0)
        pct = v / lag_imp.sum() * 100 if lag_imp.sum() > 0 else 0
        marker = " ← lag>5 被用到!" if k > 5 and pct > 3 else ""
        print(f"    lag{k:>2d}: importance={v:>6.0f}  占 lag 总重要性 {pct:>5.1f}%{marker}")
