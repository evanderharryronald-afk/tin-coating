import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 复用你已有的预处理逻辑：直接从改进版脚本 import，
from analyse_data_improved import preprocess_and_filter_outliers

os.makedirs("result/grid_search", exist_ok=True)


# ==========================================
# 1. 带 damping 参数的方向权重函数
# ==========================================
def compute_direction_sample_weight(y_delta, pos_boost=1.0, damping=0.5):
    """
    damping: 0~1，越小权重越接近1.0（不加权，即原始等权重），
             越接近1越接近完全平衡（即上一版效果）。
    pos_boost: 对少数方向（通常是"在线偏低"，delta>0）的额外加权系数。
    """
    pos_mask = y_delta > 0
    neg_mask = y_delta < 0
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()
    n_total = n_pos + n_neg

    weights = pd.Series(1.0, index=y_delta.index)
    if n_pos > 0:
        full_balance_pos = n_total / (2.0 * n_pos)
        weights[pos_mask] = (1 - damping) * 1.0 + damping * full_balance_pos * pos_boost
    if n_neg > 0:
        full_balance_neg = n_total / (2.0 * n_neg)
        weights[neg_mask] = (1 - damping) * 1.0 + damping * full_balance_neg

    return weights


class ResidualCorrectionModel:
    def __init__(self, monotonic_feature_idx=None, alpha_smoothing=0.3,
                 pos_boost=1.0, damping=0.5):
        self.alpha_smoothing = alpha_smoothing
        self.pos_boost = pos_boost
        self.damping = damping
        self.monotonic_feature_idx = monotonic_feature_idx
        self.model = None

    def _build_model(self, n_features):
        monotonic_cst = None
        if self.monotonic_feature_idx is not None:
            monotonic_cst = [0] * n_features
            monotonic_cst[self.monotonic_feature_idx] = -1
        self.model = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_depth=4,
            loss='absolute_error',
            monotonic_cst=monotonic_cst,
            random_state=42
        )

    def fit(self, X, y_delta):
        self._build_model(n_features=X.shape[1])
        sample_weight = compute_direction_sample_weight(
            y_delta, pos_boost=self.pos_boost, damping=self.damping
        )
        self.model.fit(X, y_delta, sample_weight=sample_weight.values)

    def predict_smooth(self, X, online_actual):
        predicted_delta_raw = self.model.predict(X)
        delta_series = pd.Series(predicted_delta_raw, index=X.index)
        predicted_delta_smooth = delta_series.ewm(alpha=self.alpha_smoothing).mean()
        final_pred = online_actual + predicted_delta_smooth
        return final_pred, predicted_delta_smooth


# ==========================================
# 2. 单次评估函数：给定一组超参数，返回各项指标
# ==========================================
def evaluate_once(df, surface, damping, alpha_smoothing, pos_boost=1.0):
    prefix = 'Top' if surface == 'Top' else 'Bot'
    surface_cn = '上' if surface == 'Top' else '下'

    speed_col = 'Speed[m/min]_Process_Avg'
    current_col = f'{prefix}_Current_Sum'
    df[f'{prefix}_Current_Per_Speed'] = df[current_col] / (df[speed_col] + 1e-5)

    online_col = f'Tin Weight_Actual[g/m2]_GALV_WEIGHT_{prefix.upper()}_Avg'

    feature_cols = [
        online_col,
        current_col,
        f'{prefix}_Current_Per_Speed',
        f'{prefix}_Theoretical_Factor',
        speed_col,
        'Dimension_[mm]_Width',
        'Dimension_[mm]_Thickness',
        'Steel_Grade_Encoded'
    ]
    online_feature_idx = feature_cols.index(online_col)

    X = df[feature_cols]
    delta_col = f'{prefix}_Delta'
    y_delta = df[delta_col]
    online_actual = df[online_col]
    y_true_full = df[f'{surface_cn}表面镀层重量A(XA1_0)']

    X_train, X_test, y_delta_train, y_delta_test, actual_train, actual_test, y_true_train, y_true_test = \
        train_test_split(X, y_delta, online_actual, y_true_full, test_size=0.2, shuffle=False)

    corrector = ResidualCorrectionModel(
        monotonic_feature_idx=online_feature_idx,
        alpha_smoothing=alpha_smoothing,
        pos_boost=pos_boost,
        damping=damping
    )
    corrector.fit(X_train, y_delta_train)
    pred_series, _ = corrector.predict_smooth(X_test, actual_test)

    y_true_series = y_true_test
    online_series = actual_test

    raw_residuals = y_true_series - online_series
    model_residuals = y_true_series - pred_series

    mask_pos = (raw_residuals > 0)
    mask_neg = (raw_residuals < 0)

    mae_raw_pos = raw_residuals[mask_pos].abs().mean() if mask_pos.sum() > 0 else np.nan
    mae_model_pos = model_residuals[mask_pos].abs().mean() if mask_pos.sum() > 0 else np.nan
    mae_raw_neg = raw_residuals[mask_neg].abs().mean() if mask_neg.sum() > 0 else np.nan
    mae_model_neg = model_residuals[mask_neg].abs().mean() if mask_neg.sum() > 0 else np.nan

    r2_model = r2_score(y_true_series, pred_series)
    rmse_model = np.sqrt(mean_squared_error(y_true_series, pred_series))

    return {
        'surface': surface,
        'damping': damping,
        'alpha_smoothing': alpha_smoothing,
        'pos_boost': pos_boost,
        'mae_pos_raw': mae_raw_pos,
        'mae_pos_model': mae_model_pos,
        'mae_neg_raw': mae_raw_neg,
        'mae_neg_model': mae_model_neg,
        'r2_model': r2_model,
        'rmse_model': rmse_model,
        # 两个方向是否都不比原始差（越小越好，<=0 表示达标）
        'pos_delta': mae_model_pos - mae_raw_pos,
        'neg_delta': mae_model_neg - mae_raw_neg,
    }


# ==========================================
# 3. 网格搜索主流程
# ==========================================
def grid_search(df, surface, damping_list, alpha_list, pos_boost_list, save_path):
    results = []
    for damping in damping_list:
        for alpha in alpha_list:
            for pos_boost in pos_boost_list:
                res = evaluate_once(df, surface, damping, alpha, pos_boost)
                results.append(res)
                print(f"[{surface}] damping={damping}, alpha={alpha}, pos_boost={pos_boost} "
                      f"-> RMSE={res['rmse_model']:.4f}, R2={res['r2_model']:.4f}, "
                      f"偏低MAE={res['mae_pos_model']:.4f}(原始{res['mae_pos_raw']:.4f}), "
                      f"偏高MAE={res['mae_neg_model']:.4f}(原始{res['mae_neg_raw']:.4f})")

    result_df = pd.DataFrame(results)
    result_df.to_excel(save_path, index=False)
    print(f"\n[导出提示] {surface}表面网格搜索结果已保存至: {save_path}")

    # 筛选出"两个方向都不比原始差"的候选（真正意义上的双赢）
    both_better = result_df[(result_df['pos_delta'] <= 0) & (result_df['neg_delta'] <= 0)]
    if len(both_better) > 0:
        best_row = both_better.loc[both_better['rmse_model'].idxmin()]
        print(f"\n[{surface}表面] 两方向均改善的候选中，RMSE最优的一组参数:")
        print(best_row)
    else:
        # 如果没有双赢的组合，退而求其次，选RMSE最优（整体最优，但可能有一方向变差）
        best_row = result_df.loc[result_df['rmse_model'].idxmin()]
        print(f"\n[{surface}表面] 未找到两方向都改善的组合，按整体RMSE最优选择:")
        print(best_row)

    return result_df, best_row


# ==========================================
# 4. 主流程
# ==========================================
if __name__ == "__main__":
    raw_df = pd.read_excel("result/merged_data/merged_result_latest.xlsx")

    clean_df = preprocess_and_filter_outliers(
        raw_df,
        clean_save_path="result/cleaned_data/cleaned_data.xlsx",
        filtered_save_path="result/cleaned_data/filtered_outliers.xlsx"
    )

    # 网格范围：可以按需扩展或缩小
    damping_list = [0.0, 0.2, 0.3, 0.5, 0.7, 1.0]
    alpha_list = [0.3, 0.5, 0.7, 1.0]  # alpha=1.0 表示不做EWMA平滑（等价于直接用原始预测）
    pos_boost_list = [1.0]  # 如果双赢组合的偏低方向仍不理想，可以扩展为 [1.0, 1.2, 1.5]

    for surface in ['Top', 'Bot']:
        save_path = f"result/grid_search/grid_search_{surface}.xlsx"
        grid_search(clean_df, surface, damping_list, alpha_list, pos_boost_list, save_path)