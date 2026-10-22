#!/usr/bin/env python3
"""快速对比: sklearn HistGradientBoostingRegressor vs LightGBM"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from reproduce_best_largest_group import load_and_fe, add_lag_features, evaluate, print_metrics

DATA_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent/"result"/ "data" /"cleaned_data"/ "cleaned_data.xlsx"
SP_TARGET = 2.799

df = load_and_fe(DATA_PATH)
df = df.sort_values("Produce_Time").reset_index(drop=True)
df = df[(df["SP_TOP_r"] == SP_TARGET) & (df["SP_BOT_r"] == SP_TARGET)].copy()
print(f"SP={SP_TARGET}: {len(df)} 条")

# 时序切分 80/20
g_train = df.iloc[:int(len(df)*0.80)].copy()
g_test  = df.iloc[int(len(df)*0.80):].copy()

# Steel Grade
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(g_train["Steel_Grade"].astype(str))
g_train["Steel_Grade_enc"] = le.transform(g_train["Steel_Grade"].astype(str))
known = set(le.classes_)
g_test["Steel_Grade_enc"] = g_test["Steel_Grade"].astype(str).apply(
    lambda x: le.transform([x])[0] if x in known else -1
)

# Lag 构造
for col in ["Online_SP_Diff_TOP", "Online_SP_Diff_BOT"]:
    for i in range(1, 6):
        add_lag_features(g_train, g_test, [col], lag=i)

base_feats = [
    "Thickness", "Width", "Speed",
    "Top_Current_Sum", "Bot_Current_Sum",
    "Top_Current_Density", "Bot_Current_Density",
    "Online_SP_Diff_TOP", "Online_SP_Diff_BOT",
    "Speed_x_Thickness", "Top_Bot_Online_Diff", "Top_Bot_Current_Ratio",
    "Hour", "DayOfWeek", "Steel_Grade_enc",
]
lag_cols = []
for col in ["Online_SP_Diff_TOP", "Online_SP_Diff_BOT"]:
    lag_cols.extend([f"{col}_lag{i}" for i in range(1, 6)])
full_feats = base_feats + lag_cols  # 27 个
print(f"特征数: {len(full_feats)}")

# early stopping 的 val 切分 (15% of train tail)
n_val = int(len(g_train) * 0.15)
g_tr_main = g_train.iloc[:-n_val]
g_tr_val  = g_train.iloc[-n_val:]

# ===== LightGBM =====
LGB_PARAMS = dict(
    n_estimators=1000, learning_rate=0.04, max_depth=6, num_leaves=31,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
    random_state=42, verbose=-1,
)

# ===== sklearn HGB =====
# HistGradientBoostingRegressor 的 early stopping 用 validation_fraction
# 类似 LGBM: max_iter=1000, learning_rate=0.04, max_depth=6, l2=0.1

print("\n" + "="*60)
print("LightGBM vs sklearn HistGradientBoostingRegressor —— 单折对比")
print("="*60)

for surf in ["TOP", "BOT"]:
    online_col = f"Online_{surf}"
    lab_col = f"Lab_{surf}"
    delta_col = f"Delta_{surf}"
    feats = full_feats + [online_col]
    y_lab_te = g_test[lab_col].values
    
    # LGBM
    model_lgb = lgb.LGBMRegressor(**LGB_PARAMS)
    model_lgb.fit(
        g_tr_main[feats].values, g_tr_main[delta_col].values,
        eval_set=[(g_tr_val[feats].values, g_tr_val[delta_col].values)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    pred_lgb = g_test[online_col].values + model_lgb.predict(g_test[feats].values)
    m_lgb = evaluate(y_lab_te, pred_lgb, online=g_test[online_col].values)
    
    # sklearn HGB (无 early stopping 参数，max_iter + validation)
    # HistGradientBoostingRegressor 用 early_stopping=True + n_iter_no_change
    model_sk = HistGradientBoostingRegressor(
        max_iter=1000,
        learning_rate=0.04,
        max_depth=6,
        max_leaf_nodes=31,          # 对应 LGBM num_leaves
        l2_regularization=0.1,      # 对应 LGBM reg_lambda
        min_samples_leaf=20,        # 对应 LGBM min_child_samples（LightGBM 默认 20）
        early_stopping=True,
        n_iter_no_change=50,
        validation_fraction=0.15,
        random_state=42,
    )
    # HGB 的 early stopping 内部自己切分 train/val，直接 fit 全 train
    model_sk.fit(g_train[feats].values, g_train[delta_col].values)
    pred_sk = g_test[online_col].values + model_sk.predict(g_test[feats].values)
    m_sk = evaluate(y_lab_te, pred_sk, online=g_test[online_col].values)
    
    # XGBoost 也试试
    try:
        import xgboost as xgb
        model_xgb = xgb.XGBRegressor(
            n_estimators=1000, learning_rate=0.04, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
            early_stopping_rounds=50, random_state=42, verbosity=0,
        )
        model_xgb.fit(
            g_tr_main[feats].values, g_tr_main[delta_col].values,
            eval_set=[(g_tr_val[feats].values, g_tr_val[delta_col].values)],
            verbose=False,
        )
        pred_xgb = g_test[online_col].values + model_xgb.predict(g_test[feats].values)
        m_xgb = evaluate(y_lab_te, pred_xgb, online=g_test[online_col].values)
        has_xgb = True
    except ImportError:
        has_xgb = False

    print(f"\n  === Δ_{surf} ===")
    print(f"  {'模型':30s} {'RMSE':>8s} {'MAE':>8s} {'Hit1%':>8s} {'Hit5%':>8s} {'best_iter':>10s}")
    print(f"  {'-'*72}")
    
    lgb_iter = model_lgb.best_iteration_ or model_lgb.n_estimators
    sk_iter = model_sk.n_iter_
    xgb_iter = None
    
    print(f"  {'LightGBM':30s} {m_lgb['RMSE']:8.4f} {m_lgb['MAE']:8.4f} {m_lgb['Hit1%']:8.4f} {m_lgb['Hit5%']:8.4f} {lgb_iter:>10d}")
    print(f"  {'sklearn HGB':30s} {m_sk['RMSE']:8.4f} {m_sk['MAE']:8.4f} {m_sk['Hit1%']:8.4f} {m_sk['Hit5%']:8.4f} {sk_iter:>10d}")
    
    if has_xgb:
        print(f"  {'XGBoost':30s} {m_xgb['RMSE']:8.4f} {m_xgb['MAE']:8.4f} {m_xgb['Hit1%']:8.4f} {m_xgb['Hit5%']:8.4f} {xgb_iter or '':>10}")
    
    # 模型 0 baseline
    online = g_test[online_col].values
    base_pred = online
    m_base = evaluate(y_lab_te, base_pred, online=online)
    print(f"  {'Online_raw (baseline)':30s} {m_base['RMSE']:8.4f} {m_base['MAE']:8.4f} {m_base['Hit1%']:8.4f} {m_base['Hit5%']:8.4f}")

    # HGB vs LGBM 差值
    delta_rmse = m_sk['RMSE'] - m_lgb['RMSE']
    print(f"\n  sklearn HGB RMSE - LightGBM RMSE = {delta_rmse:+.4f} ({'HGB更好' if delta_rmse < 0 else 'LGBM更好'})")
