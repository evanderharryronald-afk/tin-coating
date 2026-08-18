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
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.metrics import mean_absolute_error
from scipy import stats as scipy_stats

try:
    from model_interpreter import ModelInterpreter
    HAS_INTERPRETER = True
except ImportError:
    HAS_INTERPRETER = False
    print("[提示] 未找到 model_interpreter 模块，将跳过 SHAP 分析。")

# 设置画图支持中文与负号，消除特殊字符警告
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建结果保存目录结构
os.makedirs("result/cleaned_data", exist_ok=True)
os.makedirs("result/correlation_result", exist_ok=True)
os.makedirs("result/fitting_result", exist_ok=True)


# ==========================================
# 3. 残差建模核心类
# ==========================================
def compute_direction_sample_weight(y_delta, pos_boost=1.0, damping=0.0):
    """
    damping: 0~1，默认0（不加权，即完全等权重，对应网格搜索中RMSE最优的配置）。
             调大会向"两方向都不能变差"的保守解靠拢，但会牺牲整体RMSE，
             具体取舍参见网格搜索结果。
    pos_boost: 对少数方向（通常是"在线偏低"，delta>0）的额外加权系数，仅在 damping>0 时生效。
    """
    if damping <= 0:
        return pd.Series(1.0, index=y_delta.index)

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
    """
    直接对残差 Delta = 真实值 - 在线值 建模，而不是对绝对值建模。
    """

    def __init__(self, monotonic_feature_idx=None, alpha_smoothing=0.7,
                 pos_boost=1.0, damping=0.0, learning_rate=0.05,
                 max_iter=200, max_depth=4,loss='absolute_error', quantile=None, **kwargs):
        self.alpha_smoothing = alpha_smoothing
        self.pos_boost = pos_boost
        self.damping = damping
        self.monotonic_feature_idx = monotonic_feature_idx

        self.loss = loss
        self.quantile = quantile

        # 增加树模型参数控制
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.kwargs = kwargs  # 接收其它可能的额外参数
        self.model = None

    def _build_model(self, n_features):
        monotonic_cst = None
        if self.monotonic_feature_idx is not None:
            monotonic_cst = [0] * n_features
            monotonic_cst[self.monotonic_feature_idx] = -1

        model_kwargs = {
            'max_iter': self.max_iter,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'loss': self.loss,
            'monotonic_cst': monotonic_cst,
            'random_state': 42,
            **self.kwargs
        }
        if self.loss == 'quantile' and self.quantile is not None:
            model_kwargs['quantile'] = self.quantile

        self.model = HistGradientBoostingRegressor(**model_kwargs)

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
# 3.5 残差诊断：训练vs测试对比 + 异方差检验
# ==========================================
def diagnose_residuals(
    model_residuals_train, model_residuals_test,
    pred_train, pred_test,
    X_train, X_test,
    surface,surface_cn, save_dir, n_bins=8
):
    """
    对残差做三类诊断：
    1. 训练集 vs 测试集残差分布对比（判断过拟合程度）
    2. 残差 vs 预测值散点图（肉眼判断异方差，喇叭形/漏斗形即为异方差）
    3. 按预测值分位数分箱，比较各箱残差方差（Levene检验做正式判断）
    """
    os.makedirs(save_dir, exist_ok=True)

    # ---------- 1. 训练vs测试残差分布对比 ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(model_residuals_train, ax=axes[0], color='steelblue', label='训练集残差',
                 kde=True, stat="density", alpha=0.4)
    sns.histplot(model_residuals_test, ax=axes[0], color='orange', label='测试集残差',
                 kde=True, stat="density", alpha=0.4)
    axes[0].axvline(0, color='black', linestyle='--', linewidth=1)
    axes[0].set_title(f'{surface_cn}表面 训练集vs测试集残差分布对比')
    axes[0].set_xlabel('残差 (真实值-预测值)')
    axes[0].legend()
    axes[0].grid(True, linestyle=':', alpha=0.6)

    # 训练/测试关键统计量打印
    print(f"\n-------- 【{surface_cn}表面 训练vs测试 残差统计对比】 --------")
    print(f"{'':10s}{'均值':>10s}{'标准差':>10s}{'MAE':>10s}")
    print(f"{'训练集':10s}{model_residuals_train.mean():>10.4f}{model_residuals_train.std():>10.4f}{model_residuals_train.abs().mean():>10.4f}")
    print(f"{'测试集':10s}{model_residuals_test.mean():>10.4f}{model_residuals_test.std():>10.4f}{model_residuals_test.abs().mean():>10.4f}")
    overfit_ratio = model_residuals_test.abs().mean() / max(model_residuals_train.abs().mean(), 1e-6)
    print(f"过拟合比率 (测试MAE / 训练MAE) = {overfit_ratio:.2f}  (越接近1越好，>2 说明过拟合明显)")

    # ---------- 2. 残差 vs 预测值散点图（异方差肉眼判断） ----------
    axes[1].scatter(pred_train, model_residuals_train, s=8, alpha=0.3, color='steelblue', label='训练集')
    axes[1].scatter(pred_test, model_residuals_test, s=8, alpha=0.5, color='orange', label='测试集')
    axes[1].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[1].set_title(f'{surface_cn}表面 残差 vs 预测值（判断异方差用）')
    axes[1].set_xlabel('模型预测值')
    axes[1].set_ylabel('残差')
    axes[1].legend()
    axes[1].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    diag_path = f"{save_dir}/residual_train_test_heteroscedasticity_{surface}.png"
    plt.savefig(diag_path, dpi=300)
    plt.close()
    print(f"[图表保存] {surface_cn}表面 训练测试对比+异方差诊断图已保存至: {diag_path}")

    # ---------- 3. 按预测值分位数分箱，比较各箱残差方差 ----------
    test_df = pd.DataFrame({
        'pred': pred_test.values if hasattr(pred_test, 'values') else pred_test,
        'resid': model_residuals_test.values if hasattr(model_residuals_test, 'values') else model_residuals_test,
    })
    try:
        test_df['bin'] = pd.qcut(test_df['pred'], q=n_bins, duplicates='drop')
    except ValueError:
        test_df['bin'] = pd.cut(test_df['pred'], bins=n_bins)

    bin_stats = test_df.groupby('bin')['resid'].agg(['count', 'mean', 'std']).reset_index()
    print(f"\n-------- 【{surface_cn}表面 按预测值分位数分箱的残差统计（测试集）】 --------")
    print(bin_stats.to_string(index=False))

    # Levene 检验：各箱方差是否显著不同（原假设：方差齐性/同方差）
    groups = [g['resid'].values for _, g in test_df.groupby('bin') if len(g) >= 3]
    levene_stat, levene_p = (np.nan, np.nan)
    if len(groups) >= 2:
        levene_stat, levene_p = scipy_stats.levene(*groups)
        print(f"\nLevene 方差齐性检验: statistic={levene_stat:.4f}, p-value={levene_p:.4f}")
        if levene_p < 0.05:
            print("  -> p < 0.05，拒绝方差齐性假设，存在显著异方差性（不同预测值区间残差波动大小不同）")
        else:
            print("  -> p >= 0.05，未发现显著异方差性证据")
    print("------------------------------------------------------\n")

    # 【新增】汇总关键诊断指标，供后续统一落表
    summary_metrics = {
        '训练集残差均值': model_residuals_train.mean(),
        '训练集残差标准差': model_residuals_train.std(),
        '训练集MAE': model_residuals_train.abs().mean(),
        '测试集残差均值': model_residuals_test.mean(),
        '测试集残差标准差': model_residuals_test.std(),
        '测试集MAE': model_residuals_test.abs().mean(),
        '过拟合比率(测试MAE/训练MAE)': model_residuals_test.abs().mean() / max(model_residuals_train.abs().mean(), 1e-6),
        'Levene统计量': levene_stat,
        'Levene_p值': levene_p,
        '是否存在显著异方差': (levene_p < 0.05) if not np.isnan(levene_p) else None,
    }

    return bin_stats, summary_metrics


# ==========================================
# 4. 表面建模与图形输出
# ==========================================
def run_surface_pipeline(df, surface='Top', params=None, **kwargs):
    """
    params: 包含各种超参数的字典，如：
            {
                'damping': 0.6, 'pos_boost': 4.6, 'alpha_smoothing': 1.0,
                'learning_rate': 0.05, 'max_iter': 200, 'max_depth': 4
            }
    """
    prefix = 'Top' if surface == 'Top' else 'Bot'
    surface_cn = '上' if surface == 'Top' else '下'

    # 合并默认参数与自定义参数
    default_config = {
        'damping': 0.0,
        'alpha_smoothing': 0.7,
        'pos_boost': 1.0,
        'learning_rate': 0.05,
        'max_iter': 200,
        'max_depth': 4,
        'loss': 'absolute_error',  # 新增
        # 'quantile': 0.5,         # 只有 loss='quantile' 时才需要
    }
    if params is not None:
        default_config.update(params)
    default_config.update(kwargs)  # 允许直接用关键字参数覆盖

    print(f"\n==========================================")
    print(f"        开始运行【{surface_cn}表面】模型拟合与分析     ")
    print(f"        配置参数: {default_config}")
    print(f"==========================================")

    # # 1. 相关性分析（直接调用模块）
    # # 实例化相关性分析器模块
    # corr_analyzer = SurfaceCorrelationAnalyzer()
    # corr_analyzer.analyze_surface(df, surface=surface,save_dir="result/correlation_result")

    # 2. 特征工程
    speed_col = 'Speed[m/min]_Process_Avg'
    current_col = f'{prefix}_Current_Sum'
    df[f'{prefix}_Current_Per_Speed'] = df[current_col] / (df[speed_col] + 1e-5)

    online_col = f'Tin Weight_Actual[g/m2]_GALV_WEIGHT_{prefix.upper()}_Avg'

    # 特征列表：在线测量值放在第0位，方便对其施加单调约束
    feature_cols = [
        online_col,
        current_col,
        f'{prefix}_Current_Per_Speed',
        f'{prefix}_Theoretical_Factor',
        speed_col,
        'Dimension_[mm]_Width',
        'Dimension_[mm]_Thickness',
        # 'Dimension_[mm]_Length',
        'Steel_Grade_Encoded'
    ]
    online_feature_idx = feature_cols.index(online_col)
    # online_feature_idx = None  # 取消对在线测量值的单调约束，避免过度限制模型灵活性

    X = df[feature_cols]

    # delta_col = f'{prefix}_Delta_Centered'
    delta_col = f'{prefix}_Delta'
    y_delta = df[delta_col]
    online_actual = df[online_col]
    y_true_full = df[f'{surface_cn}表面镀层重量A(XA1_0)']

    # 3. 按时间划分 (保持原始索引 Index 不重置)
    X_train, X_test, y_delta_train, y_delta_test, actual_train, actual_test, y_true_train, y_true_test = \
        train_test_split(X, y_delta, online_actual, y_true_full, test_size=0.2, shuffle=False)

    # ---------- Naive baseline ----------
    mean_delta = y_delta_train.mean()
    median_delta = y_delta_train.median()

    pred_naive_mean = actual_test + mean_delta
    pred_naive_median = actual_test + median_delta

    # ---------- Linear baselines ----------
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_delta_train)
    pred_ridge = actual_test + ridge.predict(X_test)

    huber = HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=500)
    huber.fit(X_train, y_delta_train)
    pred_huber = actual_test + huber.predict(X_test)



    # 4. 残差建模 + 单调约束 + EWMA平滑
    # 将 default_config 解包传入 ResidualCorrectionModel
    corrector = ResidualCorrectionModel(
        monotonic_feature_idx=online_feature_idx,
        **default_config
    )
    corrector.fit(X_train, y_delta_train)

    pred_series, predicted_delta_smooth = corrector.predict_smooth(X_test, actual_test)

    # 【新增】训练集上也跑一次预测，用于判断过拟合程度和异方差性
    pred_train_series, _ = corrector.predict_smooth(X_train, actual_train)

    y_true_series = y_true_test
    online_series = actual_test

    raw_residuals = y_true_series - online_series
    model_residuals = y_true_series - pred_series

    # 【新增】训练集残差，用于对比诊断
    model_residuals_train = y_true_train - pred_train_series

    # 【新增】训练vs测试残差对比 + 异方差诊断（三合一函数，见上方定义）
    resid_bin_stats, resid_summary_metrics = diagnose_residuals(
        model_residuals_train=model_residuals_train,
        model_residuals_test=model_residuals,
        pred_train=pred_train_series,
        pred_test=pred_series,
        X_train=X_train,
        X_test=X_test,
        surface=surface,
        surface_cn=surface_cn,
        save_dir="result/fitting_result/residual_diagnosis",
    )

    # 【新增】SHAP 特征解释性分析（若 model_interpreter 模块可用）
    if HAS_INTERPRETER:
        interp_dir = f"result/fitting_result/interpretation/{surface}"
        interpreter = ModelInterpreter(
            model=corrector,  # 内部会取 .model
            X=X_train,
            feature_names=feature_cols,
            save_dir=interp_dir,
            max_samples_for_shap=500,
        )
        interpreter.full_analysis(
            y=None,
            run_permutation=False,
            run_shap=True,
            run_pdp=True,
            pdp_features=feature_cols[:5],
        )
        print(f"[SHAP分析] {surface_cn}表面 特征解释性分析结果已保存至: {interp_dir}")

    # ----------------------------------------------------
    # 【新增】导出逐行残差结果，便于定位图上那些异常大的离群点具体是哪条数据
    # ----------------------------------------------------
    result_detail = pd.DataFrame({
        '实验室真实值': y_true_series,
        '在线仪表值': online_series,
        '模型预测值': pred_series,
        '原始残差(真实-在线)': raw_residuals,
        '模型残差(真实-预测)': model_residuals,
        '模型残差绝对值': model_residuals.abs(),
    }, index=X_test.index)

    # 把测试集对应的原始行信息（Coil ID、钢种、速度等）拼接进来，方便对照工艺参数排查原因
    id_cols = [c for c in ['Coil ID', 'Steel Grade', 'Produce Time',
                            'Speed[m/min]_Process_Avg', 'Dimension_[mm]_Thickness',
                            'Dimension_[mm]_Width'] if c in df.columns]
    if id_cols:
        result_detail = df.loc[X_test.index, id_cols].join(result_detail)

    # 标记离群点：模型残差偏离均值超过3倍标准差的行，图上那几个突出的点基本会落在这里
    resid_std = model_residuals.std()
    resid_mean = model_residuals.mean()
    result_detail['是否残差离群(>3倍标准差)'] = (
        (model_residuals - resid_mean).abs() > 3 * resid_std
    )

    result_detail = result_detail.sort_values('模型残差绝对值', ascending=False)

    # detail_save_path = f"result/fitting_result/mae_lose/residual_detail_{surface}.xlsx"
    # detail_save_path = f"result/fitting_result/mse_lose/residual_detail_{surface}.xlsx"
    detail_save_path = f"result/fitting_result/quantile_lose/residual_detail_{surface}.xlsx"
    result_detail.to_excel(detail_save_path, index=True, index_label='原始数据行号')
    print(f"[导出提示] {surface_cn}表面逐行残差明细已保存至: {detail_save_path}")

    n_outliers = result_detail['是否残差离群(>3倍标准差)'].sum()
    print(f"[离群点提示] {surface_cn}表面共发现 {n_outliers} 个残差离群点（模型残差偏离均值超过3倍标准差）")
    if n_outliers > 0:
        print(result_detail[result_detail['是否残差离群(>3倍标准差)']].head(10).to_string())

    # ----------------------------------------------------
    # 残差诊断分析：验证正向与负向残差的矫正效果
    # ----------------------------------------------------
    print(f"\n-------- 【{surface_cn}表面 模型矫正前后残差诊断】 --------")
    mask_pos = (raw_residuals > 0)
    mask_neg = (raw_residuals < 0)

    if mask_pos.sum() > 0:
        mae_raw_pos = raw_residuals[mask_pos].abs().mean()
        mae_model_pos = model_residuals[mask_pos].abs().mean()
        print(
            f"当原始在线偏低 (残差 > 0, 样本数 {mask_pos.sum()}): 原始 MAE = {mae_raw_pos:.4f}  -->  模型矫正后 MAE = {mae_model_pos:.4f}")

    if mask_neg.sum() > 0:
        mae_raw_neg = raw_residuals[mask_neg].abs().mean()
        mae_model_neg = model_residuals[mask_neg].abs().mean()
        print(
            f"当原始在线偏高 (残差 < 0, 样本数 {mask_neg.sum()}): 原始 MAE = {mae_raw_neg:.4f}  -->  模型矫正后 MAE = {mae_model_neg:.4f}")
    print("------------------------------------------------------\n")

    # 指标计算
    r2_online = r2_score(y_true_series, online_series)
    r2_model = r2_score(y_true_series, pred_series)
    rmse_online = np.sqrt(mean_squared_error(y_true_series, online_series))
    rmse_model = np.sqrt(mean_squared_error(y_true_series, pred_series))

    print(f"======== 【{surface_cn}表面 拟合性能评估（测试集）】 ========")
    print(f"原始在线仪表与实验室真实值 -> R²: {r2_online:.4f}, RMSE: {rmse_online:.4f}")
    print(f"模型校正拟合后与实验室真实值 -> R²: {r2_model:.4f}, RMSE: {rmse_model:.4f}")

    start_idx = X_test.index[0]
    end_idx = X_test.index[-1]

    print(f"\n======== 【{surface_cn}表面 Baseline + 模型 对比（测试集）】 ========")

    # 【改动】把各模型对比结果同时存进列表，最后转成 DataFrame，而不只是打印
    baseline_rows = []

    # Online
    r2 = r2_score(y_true_test, actual_test)
    rmse = np.sqrt(mean_squared_error(y_true_test, actual_test))
    mae = mean_absolute_error(y_true_test, actual_test)
    print(f"{'Online (无校正)':20s} -> R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    baseline_rows.append({'模型': 'Online(无校正)', 'R2': r2, 'RMSE': rmse, 'MAE': mae})

    # Naive Mean
    r2 = r2_score(y_true_test, pred_naive_mean)
    rmse = np.sqrt(mean_squared_error(y_true_test, pred_naive_mean))
    mae = mean_absolute_error(y_true_test, pred_naive_mean)
    print(f"{'Naive Mean Δ':20s} -> R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    baseline_rows.append({'模型': 'Naive Mean', 'R2': r2, 'RMSE': rmse, 'MAE': mae})

    # Naive Median
    r2 = r2_score(y_true_test, pred_naive_median)
    rmse = np.sqrt(mean_squared_error(y_true_test, pred_naive_median))
    mae = mean_absolute_error(y_true_test, pred_naive_median)
    print(f"{'Naive Median Δ':20s} -> R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    baseline_rows.append({'模型': 'Naive Median', 'R2': r2, 'RMSE': rmse, 'MAE': mae})

    # Ridge
    r2 = r2_score(y_true_test, pred_ridge)
    rmse = np.sqrt(mean_squared_error(y_true_test, pred_ridge))
    mae = mean_absolute_error(y_true_test, pred_ridge)
    print(f"{'Ridge':20s} -> R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    baseline_rows.append({'模型': 'Ridge', 'R2': r2, 'RMSE': rmse, 'MAE': mae})

    # Huber
    r2 = r2_score(y_true_test, pred_huber)
    rmse = np.sqrt(mean_squared_error(y_true_test, pred_huber))
    mae = mean_absolute_error(y_true_test, pred_huber)
    print(f"{'Huber':20s} -> R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    baseline_rows.append({'模型': 'Huber', 'R2': r2, 'RMSE': rmse, 'MAE': mae})

    # GBDT
    r2 = r2_score(y_true_test, pred_series)
    rmse = np.sqrt(mean_squared_error(y_true_test, pred_series))
    mae = mean_absolute_error(y_true_test, pred_series)
    print(f"{'GBDT (当前模型)':20s} -> R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    baseline_rows.append({'模型': 'GBDT(当前模型)', 'R2': r2, 'RMSE': rmse, 'MAE': mae})

    baseline_comparison_df = pd.DataFrame(baseline_rows)

    # ---------- 双向 MAE 对比（表格形式） ----------
    raw_residuals = y_true_test - actual_test
    mask_pos = raw_residuals > 0
    mask_neg = raw_residuals < 0

    models = {
        "Online": actual_test,
        "Naive Mean": pred_naive_mean,
        "Naive Median": pred_naive_median,
        "Ridge": pred_ridge,
        "Huber": pred_huber,
        "GBDT": pred_series,
    }

    # 表头
    header = f"{'方向':<12}" + "".join(f"{name:>14}" for name in models.keys())
    print(f"\n======== 【{surface_cn}表面 双向 MAE 对比】 ========")
    print(header)
    print("-" * len(header))

    # 在线偏低
    row_pos = f"{'在线偏低时':<12}"
    for pred in models.values():
        mae = (y_true_test[mask_pos] - pred[mask_pos]).abs().mean() if mask_pos.sum() > 0 else float('nan')
        row_pos += f"{mae:>14.4f}"
    print(row_pos)

    # 在线偏高
    row_neg = f"{'在线偏高时':<12}"
    for pred in models.values():
        mae = (y_true_test[mask_neg] - pred[mask_neg]).abs().mean() if mask_neg.sum() > 0 else float('nan')
        row_neg += f"{mae:>14.4f}"
    print(row_neg)

    print("-" * len(header))

    # 【新增】把双向MAE对比也转成结构化 DataFrame
    bidirectional_rows = []
    for name, pred in models.items():
        mae_pos = (y_true_test[mask_pos] - pred[mask_pos]).abs().mean() if mask_pos.sum() > 0 else np.nan
        mae_neg = (y_true_test[mask_neg] - pred[mask_neg]).abs().mean() if mask_neg.sum() > 0 else np.nan
        bidirectional_rows.append({'模型': name, '在线偏低时MAE': mae_pos, '在线偏高时MAE': mae_neg})
    bidirectional_df = pd.DataFrame(bidirectional_rows)

    # 5. 拟合对比图
    plt.figure(figsize=(12, 5))
    plt.plot(y_true_series, label='实验室真实测量值 (True Label)', color='black', linewidth=1.5)
    plt.plot(online_series, label='在线仪表原始测量值 (Online)', color='red', linestyle='--', alpha=0.7)
    plt.plot(pred_series, label='模型残差校正值 (Model Pred)', color='green', linewidth=1.5, alpha=0.85)
    plt.title(f'{surface_cn}表面 镀层重量拟合对照图（原始数据行号: {start_idx} ~ {end_idx}）')
    plt.xlabel('原始数据行号 (Original Row Index)')
    plt.ylabel('镀层重量 (g/m2)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    # fit_img_path = f"result/fitting_result/mae_lose/fitting_result_{surface}.png"
    # fit_img_path = f"result/fitting_result/mse_lose/fitting_result_{surface}_.png"
    fit_img_path = f"result/fitting_result/quantile_lose/fitting_result_{surface}_.png"

    plt.savefig(fit_img_path, dpi=300)
    print(f"[图表保存] {surface_cn}表面拟合对照图已保存至: {fit_img_path}")
    # plt.show()

    # 6. 残差对比图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})

    ax1.plot(raw_residuals, label='原始在线仪表残差 (True - Online)', color='red', alpha=0.5, linewidth=1)
    ax1.plot(model_residuals, label='模型校正后残差 (True - Model)', color='green', alpha=0.8, linewidth=1.2)
    ax1.axhline(0, color='black', linestyle='--', linewidth=1)
    ax1.set_title(f'{surface_cn}表面 预测残差变化对比（原始数据行号: {start_idx} ~ {end_idx}）')
    ax1.set_xlabel('原始数据行号 (Original Row Index)')
    ax1.set_ylabel('残差/误差 (g/m2)')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    sns.histplot(raw_residuals, ax=ax2, color='red', label='原始残差分布', kde=True, stat="density", alpha=0.3)
    sns.histplot(model_residuals, ax=ax2, color='green', label='模型校正后残差分布', kde=True, stat="density",
                 alpha=0.5)
    ax2.axvline(0, color='black', linestyle='--', linewidth=1)
    ax2.set_title(f'{surface_cn}表面 残差概率密度分布（越集中在0且越窄越好）')
    ax2.set_xlabel('残差/误差 (g/m2)')
    ax2.set_ylabel('概率密度')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    # res_img_path = f"result/fitting_result/mae_lose/residual_analysis_{surface}.png"
    # res_img_path = f"result/fitting_result/mse_lose/residual_analysis_{surface}.png"
    res_img_path = f"result/fitting_result/quantile_lose/residual_analysis_{surface}.png"
    plt.savefig(res_img_path, dpi=300)
    print(f"[图表保存] {surface_cn}表面残差分析图已保存至: {res_img_path}")
    # plt.show()

    # 【新增】把本次运行所有关键结果打包成一个字典返回，供主流程统一落表
    overall_metrics_df = pd.DataFrame([
        {'指标': '原始在线R2', '数值': r2_online},
        {'指标': '原始在线RMSE', '数值': rmse_online},
        {'指标': '模型校正R2', '数值': r2_model},
        {'指标': '模型校正RMSE', '数值': rmse_model},
        {'指标': '离群点数量(>3倍std)', '数值': n_outliers},
        {'指标': '测试集样本数', '数值': len(X_test)},
        {'指标': '训练集样本数', '数值': len(X_train)},
    ])

    surface_report = {
        'surface': surface,
        'surface_cn': surface_cn,
        'params': default_config,
        'overall_metrics': overall_metrics_df,        # 整体R2/RMSE等
        'baseline_comparison': baseline_comparison_df,  # 各模型(Online/Ridge/Huber/GBDT等)对比
        'bidirectional_mae': bidirectional_df,          # 双向MAE对比
        'residual_diagnosis_summary': pd.DataFrame([resid_summary_metrics]),  # 训练/测试对比、过拟合比率、Levene检验
        'residual_bin_stats': resid_bin_stats,          # 分位数分箱残差方差表
        'outlier_detail': result_detail,                # 逐行离群点明细
    }

    return corrector, surface_report


# ==========================================
# 4.5 结果统一落表：多sheet Excel（横向对比布局） + 长格式CSV
# ==========================================
def export_reports_to_excel(reports, excel_path="result/fitting_result/summary_report.xlsx"):
    """
    把多个 surface_report（run_surface_pipeline 的第二个返回值）汇总导出：
    1. 一个多sheet Excel：关键对比类sheet做成"指标为行、表面/模型为列"的横向布局，
       方便左右对比Top/Bot或不同模型；明细类sheet（分箱、离群点）保持纵向记录形式。
    2. 一份长格式CSV（run_summary_long.csv）：一行一个(surface, 指标)，方便跨多次实验拼接对比
    """
    os.makedirs(os.path.dirname(excel_path), exist_ok=True)

    def _tag_surface(df, surface_cn):
        df = df.copy()
        df.insert(0, '表面', surface_cn)
        return df

    overall_all = pd.concat([_tag_surface(r['overall_metrics'], r['surface_cn']) for r in reports], ignore_index=True)
    baseline_all = pd.concat([_tag_surface(r['baseline_comparison'], r['surface_cn']) for r in reports], ignore_index=True)
    bidir_all = pd.concat([_tag_surface(r['bidirectional_mae'], r['surface_cn']) for r in reports], ignore_index=True)
    resid_summary_all = pd.concat([_tag_surface(r['residual_diagnosis_summary'], r['surface_cn']) for r in reports], ignore_index=True)
    resid_bins_all = pd.concat([_tag_surface(r['residual_bin_stats'], r['surface_cn']) for r in reports], ignore_index=True)
    outliers_all = pd.concat([_tag_surface(r['outlier_detail'].reset_index(), r['surface_cn']) for r in reports], ignore_index=True)

    # 参数配置表（每次跑用的超参数，方便追溯是哪次实验的结果）
    params_rows = []
    for r in reports:
        row = {'表面': r['surface_cn']}
        row.update(r['params'])
        params_rows.append(row)
    params_all = pd.DataFrame(params_rows)

    # ---------- 横向布局1：整体拟合指标（指标为行，表面为列） ----------
    overall_wide = overall_all.pivot(index='指标', columns='表面', values='数值').reset_index()

    # ---------- 横向布局2：多模型对比（模型为行，"表面_指标"为列） ----------
    baseline_long = baseline_all.melt(id_vars=['表面', '模型'], value_vars=['R2', 'RMSE', 'MAE'],
                                       var_name='指标', value_name='数值')
    baseline_wide = baseline_long.pivot_table(index='模型', columns=['表面', '指标'], values='数值')
    baseline_wide.columns = [f'{surface}_{metric}' for surface, metric in baseline_wide.columns]
    baseline_wide = baseline_wide.reset_index()

    # ---------- 横向布局3：双向MAE对比（模型为行，"表面_方向"为列） ----------
    bidir_pos = bidir_all.pivot_table(index='模型', columns='表面', values='在线偏低时MAE')
    bidir_pos.columns = [f'{c}_偏低MAE' for c in bidir_pos.columns]
    bidir_neg = bidir_all.pivot_table(index='模型', columns='表面', values='在线偏高时MAE')
    bidir_neg.columns = [f'{c}_偏高MAE' for c in bidir_neg.columns]
    bidir_wide = pd.concat([bidir_pos, bidir_neg], axis=1).reset_index()

    # ---------- 横向布局4：残差诊断（指标为行，表面为列，与整体指标风格统一） ----------
    resid_summary_wide = resid_summary_all.set_index('表面').T.reset_index().rename(columns={'index': '指标'})

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        params_all.to_excel(writer, sheet_name='运行参数配置', index=False)
        overall_wide.to_excel(writer, sheet_name='整体拟合指标', index=False)
        baseline_wide.to_excel(writer, sheet_name='多模型对比', index=False)
        bidir_wide.to_excel(writer, sheet_name='双向MAE对比', index=False)
        resid_summary_wide.to_excel(writer, sheet_name='残差过拟合与异方差诊断', index=False)
        # 明细类表格保持纵向记录形式，横向对比在这里没有意义
        resid_bins_all.to_excel(writer, sheet_name='残差分位数分箱明细', index=False)
        outliers_all.to_excel(writer, sheet_name='离群点明细', index=False)

    print(f"\n[汇总导出] 多sheet Excel 报告已保存至: {excel_path}")

    # ---------- 长格式CSV，方便未来多次实验对比追踪（结构不变，供程序读取而非人工浏览） ----------
    run_ts = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    long_rows = []
    for r in reports:
        for _, row in r['overall_metrics'].iterrows():
            long_rows.append({'运行时间': run_ts, '表面': r['surface_cn'], '来源': '整体拟合指标',
                               '指标': row['指标'], '数值': row['数值']})
        for _, row in r['residual_diagnosis_summary'].iterrows():
            for col in row.index:
                long_rows.append({'运行时间': run_ts, '表面': r['surface_cn'], '来源': '残差诊断',
                                   '指标': col, '数值': row[col]})
    long_df = pd.DataFrame(long_rows)

    csv_path = "result/fitting_result/run_summary_long.csv"
    write_header = not os.path.exists(csv_path)
    long_df.to_csv(csv_path, mode='a', header=write_header, index=False, encoding='utf-8-sig')
    print(f"[汇总导出] 长格式追加记录已写入: {csv_path}（可多次运行累积对比不同实验配置）")


# ==========================================
# 5. 主流程
# ==========================================
if __name__ == "__main__":
    # raw_df = pd.read_excel("result/merged_data/merged_result_latest.xlsx")
    #
    # cleaner = SteelDataCleaner(
    #     min_speed=20.0,
    #     max_range_abs=0.4,
    #     max_range_ratio=0.3,
    #     mad_factor=3.0
    # )
    #
    # clean_df = cleaner.process(
    #     raw_df,
    #     clean_save_path="result/cleaned_data/cleaned_data.xlsx",
    #     filtered_save_path="result/cleaned_data/filtered_outliers.xlsx"
    # )

    clean_df=pd.read_excel("result/cleaned_data/cleaned_data.xlsx")

    # -------------------------------------------------------------
    # 方式 A：定义参数字典（推荐，便于对接 JSON 配置文件或 Optuna 调参）
    # -------------------------------------------------------------
    top_params = {
        "damping": 0.28,
        "pos_boost": 1.33,
        "alpha_smoothing": 0.93,
        "learning_rate": 0.072,  # Optuna优化后的学习率
        "max_iter": 350,  # Optuna优化后的树数量
        "max_depth": 8,  # Optuna优化后的树最大深度
        # "loss": "absolute_error",
        # "loss": "squared_error",
        "loss": "quantile",
        "quantile": 0.9,
    }

    bot_params = {
        "damping": 0.67,
        "pos_boost": 7.88,
        "alpha_smoothing": 0.83,
        "learning_rate": 0.04,  # Optuna优化后的学习率
        "max_iter": 200,  # Optuna优化后的树数量
        "max_depth": 8,  # Optuna优化后的树最大深度
        # "loss": "absolute_error",
        # "loss": "squared_error",
        "loss": "quantile",
        "quantile": 0.3,
    }

    # 【改动】接收 run_surface_pipeline 的第二个返回值（surface_report）
    top_corrector, top_report = run_surface_pipeline(clean_df, surface='Top', params=top_params)
    bot_corrector, bot_report = run_surface_pipeline(clean_df, surface='Bot', params=bot_params)

    # 【新增】统一导出多sheet Excel + 长格式CSV
    export_reports_to_excel([top_report, bot_report])

    # # -------------------------------------------------------------
    # # 方式 B：直接用关键字参数修改
    # # -------------------------------------------------------------
    # run_surface_pipeline(
    #     clean_df,
    #     surface='Bot',
    #     damping=0.6,
    #     pos_boost=4.6,
    #     alpha_smoothing=1.0,
    #     learning_rate=0.02,
    #     max_iter=250,
    #     max_depth=4
    # )