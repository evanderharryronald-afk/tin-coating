import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from data_cleaner import SteelDataCleaner
from correlation_analyzer import SurfaceCorrelationAnalyzer

# 设置画图支持中文与负号
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建结果保存目录结构
os.makedirs("result/data/cleaned_data", exist_ok=True)
os.makedirs("result/correlation_result", exist_ok=True)
os.makedirs("result/fitting_result", exist_ok=True)


# ==========================================
# 3. 直接拟合核心类 (Direct Fitting)
# ==========================================
class DirectRegressionModel:
    """
    直接对镀层重量绝对值 y_true (实验室真实值) 建模。
    """

    def __init__(self, monotonic_feature_idx=None, alpha_smoothing=0.7,
                 learning_rate=0.05, max_iter=200, max_depth=4, **kwargs):
        self.alpha_smoothing = alpha_smoothing
        self.monotonic_feature_idx = monotonic_feature_idx

        # 树模型参数控制
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.kwargs = kwargs
        self.model = None

    def _build_model(self, n_features):
        monotonic_cst = None
        if self.monotonic_feature_idx is not None:
            monotonic_cst = [0] * n_features
            # 镀层绝对值拟合中，在线仪表测量值越大，真实镀层重量通常也越大，因此约束为正单调 (+1)
            monotonic_cst[self.monotonic_feature_idx] = 1

        self.model = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            loss='squared_error',  # 直接拟合绝对值时通常推荐 MSE / squared_error
            monotonic_cst=monotonic_cst,
            random_state=42,
            **self.kwargs
        )

    def fit(self, X, y_true):
        self._build_model(n_features=X.shape[1])
        # 直接拟合绝对值，不施加残差方向加权
        self.model.fit(X, y_true)

    def predict_smooth(self, X):
        predicted_raw = self.model.predict(X)
        pred_series = pd.Series(predicted_raw, index=X.index)

        # 对输出的绝对值预测结果施加 EWMA 平滑（抑制时序突变）
        if self.alpha_smoothing < 1.0:
            final_pred = pred_series.ewm(alpha=self.alpha_smoothing).mean()
        else:
            final_pred = pred_series

        return final_pred


# ==========================================
# 4. 表面建模与图形输出
# ==========================================
def run_surface_pipeline(df, surface='Top', params=None, **kwargs):
    prefix = 'Top' if surface == 'Top' else 'Bot'
    surface_cn = '上' if surface == 'Top' else '下'

    # 剔除针对残差加权的 damping 和 pos_boost
    default_config = {
        'alpha_smoothing': 0.7,
        'learning_rate': 0.05,
        'max_iter': 200,
        'max_depth': 4
    }
    if params is not None:
        # 自动过滤掉对直接拟合无效的参数
        valid_params = {k: v for k, v in params.items() if k not in ['damping', 'pos_boost']}
        default_config.update(valid_params)
    default_config.update(kwargs)

    print(f"\n==========================================")
    print(f"   开始运行【{surface_cn}表面】绝对值直接拟合与分析     ")
    print(f"        配置参数: {default_config}")
    print(f"==========================================")

    # 1. 相关性分析
    corr_analyzer = SurfaceCorrelationAnalyzer()
    corr_analyzer.analyze_surface(df, surface=surface, save_dir="result/correlation_result")

    # 2. 特征工程
    speed_col = 'Speed[m/min]_Process_Avg'
    current_col = f'{prefix}_Current_Sum'
    df[f'{prefix}_Current_Per_Speed'] = df[current_col] / (df[speed_col] + 1e-5)

    online_col = f'Tin Weight_Actual[g/m2]_GALV_WEIGHT_{prefix.upper()}_Avg'

    # 特征列表：在线测量值放在第0位
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
    online_actual = df[online_col]
    y_true_full = df[f'{surface_cn}表面镀层重量A(XA1_0)']  # 核心改变：拟合目标为绝对值

    # 3. 按时间划分
    X_train, X_test, y_true_train, y_true_test, actual_train, actual_test = \
        train_test_split(X, y_true_full, online_actual, test_size=0.2, shuffle=False)

    # 4. 直接拟合建模
    model = DirectRegressionModel(
        monotonic_feature_idx=online_feature_idx,
        **default_config
    )
    model.fit(X_train, y_true_train)

    # 预测绝对值
    pred_series = model.predict_smooth(X_test)

    y_true_series = y_true_test
    online_series = actual_test

    # 导出残差做后续诊断
    raw_residuals = y_true_series - online_series
    model_residuals = y_true_series - pred_series

    # ----------------------------------------------------
    # 导出逐行残差明细
    # ----------------------------------------------------
    result_detail = pd.DataFrame({
        '实验室真实值': y_true_series,
        '在线仪表值': online_series,
        '模型预测值': pred_series,
        '原始残差(真实-在线)': raw_residuals,
        '模型残差(真实-预测)': model_residuals,
        '模型残差绝对值': model_residuals.abs(),
    }, index=X_test.index)

    id_cols = [c for c in ['Coil ID', 'Steel Grade', 'Produce Time',
                           'Speed[m/min]_Process_Avg', 'Dimension_[mm]_Thickness',
                           'Dimension_[mm]_Width'] if c in df.columns]
    if id_cols:
        result_detail = df.loc[X_test.index, id_cols].join(result_detail)

    resid_std = model_residuals.std()
    resid_mean = model_residuals.mean()
    result_detail['是否残差离群(>3倍标准差)'] = (
            (model_residuals - resid_mean).abs() > 3 * resid_std
    )

    result_detail = result_detail.sort_values('模型残差绝对值', ascending=False)

    detail_save_path = f"result/fitting_result/direct_residual_detail_{surface}.xlsx"
    result_detail.to_excel(detail_save_path, index=True, index_label='原始数据行号')
    print(f"[导出提示] {surface_cn}表面逐行明细已保存至: {detail_save_path}")

    n_outliers = result_detail['是否残差离群(>3倍标准差)'].sum()
    print(f"[离群点提示] {surface_cn}表面共发现 {n_outliers} 个残差离群点")

    # ----------------------------------------------------
    # 残差诊断分析
    # ----------------------------------------------------
    print(f"\n-------- 【{surface_cn}表面 预测偏差诊断】 --------")
    mask_pos = (raw_residuals > 0)
    mask_neg = (raw_residuals < 0)

    if mask_pos.sum() > 0:
        mae_raw_pos = raw_residuals[mask_pos].abs().mean()
        mae_model_pos = model_residuals[mask_pos].abs().mean()
        print(
            f"当原始在线偏低 (样本数 {mask_pos.sum()}): 原始 MAE = {mae_raw_pos:.4f}  -->  模型预测 MAE = {mae_model_pos:.4f}")

    if mask_neg.sum() > 0:
        mae_raw_neg = raw_residuals[mask_neg].abs().mean()
        mae_model_neg = model_residuals[mask_neg].abs().mean()
        print(
            f"当原始在线偏高 (样本数 {mask_neg.sum()}): 原始 MAE = {mae_raw_neg:.4f}  -->  模型预测 MAE = {mae_model_neg:.4f}")
    print("------------------------------------------------------\n")

    # 指标计算
    r2_online = r2_score(y_true_series, online_series)
    r2_model = r2_score(y_true_series, pred_series)
    rmse_online = np.sqrt(mean_squared_error(y_true_series, online_series))
    rmse_model = np.sqrt(mean_squared_error(y_true_series, pred_series))

    print(f"======== 【{surface_cn}表面 拟合性能评估（测试集）】 ========")
    print(f"原始在线仪表与实验室真实值 -> R²: {r2_online:.4f}, RMSE: {rmse_online:.4f}")
    print(f"模型直接拟合与实验室真实值 -> R²: {r2_model:.4f}, RMSE: {rmse_model:.4f}")

    start_idx = X_test.index[0]
    end_idx = X_test.index[-1]

    # 5. 拟合对比图
    plt.figure(figsize=(12, 5))
    plt.plot(y_true_series, label='实验室真实测量值 (True Label)', color='black', linewidth=1.5)
    plt.plot(online_series, label='在线仪表原始测量值 (Online)', color='red', linestyle='--', alpha=0.7)
    plt.plot(pred_series, label='直接拟合模型预测值 (Direct Pred)', color='blue', linewidth=1.5, alpha=0.85)
    plt.title(f'{surface_cn}表面 镀层重量直接拟合对照图（原始数据行号: {start_idx} ~ {end_idx}）')
    plt.xlabel('原始数据行号 (Original Row Index)')
    plt.ylabel('镀层重量 (g/m2)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    fit_img_path = f"result/fitting_result/direct_fitting_result_{surface}.png"
    plt.savefig(fit_img_path, dpi=300)
    print(f"[图表保存] {surface_cn}表面拟合对照图已保存至: {fit_img_path}")
    plt.show()

    # 6. 残差对比图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})

    ax1.plot(raw_residuals, label='原始在线仪表残差 (True - Online)', color='red', alpha=0.5, linewidth=1)
    ax1.plot(model_residuals, label='模型直接拟合残差 (True - Model)', color='blue', alpha=0.8, linewidth=1.2)
    ax1.axhline(0, color='black', linestyle='--', linewidth=1)
    ax1.set_title(f'{surface_cn}表面 预测残差变化对比（原始数据行号: {start_idx} ~ {end_idx}）')
    ax1.set_xlabel('原始数据行号 (Original Row Index)')
    ax1.set_ylabel('残差/误差 (g/m2)')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    sns.histplot(raw_residuals, ax=ax2, color='red', label='原始残差分布', kde=True, stat="density", alpha=0.3)
    sns.histplot(model_residuals, ax=ax2, color='blue', label='模型拟合残差分布', kde=True, stat="density", alpha=0.5)
    ax2.axvline(0, color='black', linestyle='--', linewidth=1)
    ax2.set_title(f'{surface_cn}表面 残差概率密度分布（越集中在0且越窄越好）')
    ax2.set_xlabel('残差/误差 (g/m2)')
    ax2.set_ylabel('概率密度')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    res_img_path = f"result/fitting_result/direct_residual_analysis_{surface}.png"
    plt.savefig(res_img_path, dpi=300)
    print(f"[图表保存] {surface_cn}表面残差分析图已保存至: {res_img_path}")
    plt.show()

    return model


# ==========================================
# 5. 主流程
# ==========================================
if __name__ == "__main__":
    clean_df = pd.read_excel("result/cleaned_data/cleaned_data.xlsx")

    # 参数字典（去除了残差专用的 damping 和 pos_boost）
    top_params = {
        "alpha_smoothing": 0.93,
        "learning_rate": 0.072,
        "max_iter": 350,
        "max_depth": 8
    }

    bot_params = {
        "alpha_smoothing": 0.83,
        "learning_rate": 0.04,
        "max_iter": 200,
        "max_depth": 8
    }

    run_surface_pipeline(clean_df, surface='Top', params=top_params)
    run_surface_pipeline(clean_df, surface='Bot', params=bot_params)