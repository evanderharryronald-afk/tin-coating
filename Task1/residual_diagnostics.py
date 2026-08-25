"""
residual_diagnostics.py
========================
通用残差诊断模块。

设计目标：
- 与具体模型/业务解耦，只依赖 y_true / y_pred / X 三样东西，
  分组预测、其它 surface、其它项目都可以直接复用。
- 输出风格与项目现有习惯保持一致：图存文件，统计量汇总为 DataFrame，
  方便后续直接塞进现有的多 sheet Excel 导出流程。
- 参数尽量少、能自动推断的不要求用户传（数值/类别特征自动识别）。

包含的诊断子模块：
  1. distribution_diagnostics          残差分布 + 统计特性（均值/std/偏度/峰度）+ 正态性检验 + QQ图
  2. heteroscedasticity_diagnostics    训练vs测试对比 + 残差vs预测值 + 分箱方差齐性检验（Levene）
  3. feature_relationship_diagnostics  残差 vs 数值特征（散点+条件期望分箱）、残差 vs 类别特征（箱线图）
  4. temporal_diagnostics              残差自相关性（ACF + Ljung-Box），仅在数据有明确顺序时有意义
  5. explainability_score              对残差做二次建模，量化"残差中还剩多少可解释信息"

用法见文件末尾 __main__ 示例，或参考 analyse_data_final.py 中的接入方式。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats

try:
    from statsmodels.graphics.tsaplots import plot_acf
    from statsmodels.stats.diagnostic import acorr_ljungbox
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import cross_val_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 工具函数
# ==========================================
def _infer_feature_types(X: pd.DataFrame, categorical_features=None, max_unique_for_cat=15):
    """
    自动区分数值特征和类别特征。
    categorical_features: 用户可显式指定哪些列是类别列；未指定的列按
    dtype + 唯一值数量自动判断（object/category 类型，或数值但唯一值很少）。
    """
    if categorical_features is None:
        categorical_features = []

    numeric_cols, categorical_cols = [], []
    for col in X.columns:
        if col in categorical_features:
            categorical_cols.append(col)
            continue
        if pd.api.types.is_numeric_dtype(X[col]):
            if X[col].nunique(dropna=True) <= max_unique_for_cat:
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return numeric_cols, categorical_cols


def _safe_makedirs(path):
    os.makedirs(path, exist_ok=True)


# ==========================================
# 1. 分布诊断
# ==========================================
def distribution_diagnostics(residuals: pd.Series, label: str, save_dir: str, tag: str = ""):
    """
    残差分布统计特性 + 正态性检验 + 直方图/KDE + QQ图。
    label: 用于图标题的中文说明，如"测试集"
    tag: 文件名后缀，避免多次调用相互覆盖
    """
    _safe_makedirs(save_dir)
    residuals = pd.Series(residuals).dropna()

    skewness = scipy_stats.skew(residuals)
    kurtosis = scipy_stats.kurtosis(residuals)  # 超额峰度，正态分布为0

    # 正态性检验：样本量大时 Shapiro 效力下降甚至报警告，优先用 D'Agostino K^2；
    # 样本量较小(<5000)时两者都跑，互相印证。
    normality_stat, normality_p = scipy_stats.normaltest(residuals)
    shapiro_stat, shapiro_p = (np.nan, np.nan)
    if len(residuals) <= 5000:
        shapiro_stat, shapiro_p = scipy_stats.shapiro(residuals)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.histplot(residuals, kde=True, stat="density", ax=axes[0], color='steelblue')
    axes[0].axvline(0, color='black', linestyle='--', linewidth=1)
    axes[0].set_title(f'{label} 残差分布（偏度={skewness:.3f}, 峰度={kurtosis:.3f}）')
    axes[0].set_xlabel('残差')
    axes[0].grid(True, linestyle=':', alpha=0.6)

    scipy_stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title(f'{label} 残差 QQ图（正态性检验 p={normality_p:.4f}）')
    axes[1].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    img_path = f"{save_dir}/residual_distribution_{tag}.png"
    plt.savefig(img_path, dpi=300)
    plt.close()

    summary = {
        '残差均值': residuals.mean(),
        '残差标准差': residuals.std(),
        '残差MAE': residuals.abs().mean(),
        '残差偏度': skewness,
        '残差峰度(超额)': kurtosis,
        "D'Agostino正态性_p值": normality_p,
        'Shapiro正态性_p值': shapiro_p,
        '是否显著偏离正态(p<0.05)': normality_p < 0.05,
    }
    print(f"[分布诊断] {label} 图已保存至: {img_path}")
    return summary


# ==========================================
# 2. 异方差诊断（迁移自原 diagnose_residuals，接口不变，逻辑一致）
# ==========================================
def heteroscedasticity_diagnostics(
    residuals_train, residuals_test, pred_train, pred_test,
    label: str, save_dir: str, tag: str = "", n_bins=8
):
    _safe_makedirs(save_dir)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(residuals_train, ax=axes[0], color='steelblue', label='训练集残差',
                 kde=True, stat="density", alpha=0.4)
    sns.histplot(residuals_test, ax=axes[0], color='orange', label='测试集残差',
                 kde=True, stat="density", alpha=0.4)
    axes[0].axvline(0, color='black', linestyle='--', linewidth=1)
    axes[0].set_title(f'{label} 训练集vs测试集残差分布对比')
    axes[0].set_xlabel('残差')
    axes[0].legend()
    axes[0].grid(True, linestyle=':', alpha=0.6)

    axes[1].scatter(pred_train, residuals_train, s=8, alpha=0.3, color='steelblue', label='训练集')
    axes[1].scatter(pred_test, residuals_test, s=8, alpha=0.5, color='orange', label='测试集')
    axes[1].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[1].set_title(f'{label} 残差 vs 预测值（异方差判断）')
    axes[1].set_xlabel('预测值')
    axes[1].set_ylabel('残差')
    axes[1].legend()
    axes[1].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    img_path = f"{save_dir}/residual_heteroscedasticity_{tag}.png"
    plt.savefig(img_path, dpi=300)
    plt.close()

    overfit_ratio = residuals_test.abs().mean() / max(residuals_train.abs().mean(), 1e-6)

    test_df = pd.DataFrame({
        'pred': pred_test.values if hasattr(pred_test, 'values') else pred_test,
        'resid': residuals_test.values if hasattr(residuals_test, 'values') else residuals_test,
    })
    try:
        test_df['bin'] = pd.qcut(test_df['pred'], q=n_bins, duplicates='drop')
    except ValueError:
        test_df['bin'] = pd.cut(test_df['pred'], bins=n_bins)

    bin_stats = test_df.groupby('bin')['resid'].agg(['count', 'mean', 'std']).reset_index()

    groups = [g['resid'].values for _, g in test_df.groupby('bin') if len(g) >= 3]
    levene_stat, levene_p = (np.nan, np.nan)
    if len(groups) >= 2:
        levene_stat, levene_p = scipy_stats.levene(*groups)

    summary = {
        '训练集残差均值': residuals_train.mean(),
        '训练集残差标准差': residuals_train.std(),
        '训练集MAE': residuals_train.abs().mean(),
        '测试集残差均值': residuals_test.mean(),
        '测试集残差标准差': residuals_test.std(),
        '测试集MAE': residuals_test.abs().mean(),
        '过拟合比率(测试MAE/训练MAE)': overfit_ratio,
        'Levene统计量': levene_stat,
        'Levene_p值': levene_p,
        '是否存在显著异方差': (levene_p < 0.05) if not np.isnan(levene_p) else None,
    }
    print(f"[异方差诊断] {label} 图已保存至: {img_path}")
    return bin_stats, summary


# ==========================================
# 3. 残差 vs 特征关系诊断
# ==========================================
def feature_relationship_diagnostics(
    residuals: pd.Series, X: pd.DataFrame, label: str, save_dir: str,
    tag: str = "", n_bins=8, categorical_features=None, max_unique_for_cat=15,
    top_n_features=None
):
    """
    对每个特征分别画：
    - 数值特征：散点(残差 vs 特征原始值) + 分箱后条件期望曲线(E[residual|bin] ± 95%CI)
    - 类别特征：按类别分组的残差箱线图
    同时返回每个特征的"条件期望是否显著偏离0"的简单 F 检验结果，
    用于快速定位哪些特征上还有系统性没被模型吃掉的信息。
    """
    _safe_makedirs(save_dir)
    residuals = pd.Series(residuals).reindex(X.index)

    numeric_cols, categorical_cols = _infer_feature_types(
        X, categorical_features=categorical_features, max_unique_for_cat=max_unique_for_cat
    )
    if top_n_features is not None:
        numeric_cols = numeric_cols[:top_n_features]
        categorical_cols = categorical_cols[:top_n_features]

    feature_significance_rows = []

    # ---------- 数值特征 ----------
    for col in numeric_cols:
        feat = X[col]
        valid = feat.notna() & residuals.notna()
        if valid.sum() < n_bins * 3:
            continue

        df_tmp = pd.DataFrame({'feat': feat[valid], 'resid': residuals[valid]})
        try:
            df_tmp['bin'] = pd.qcut(df_tmp['feat'], q=n_bins, duplicates='drop')
        except ValueError:
            df_tmp['bin'] = pd.cut(df_tmp['feat'], bins=n_bins)

        bin_grp = df_tmp.groupby('bin')['resid']
        bin_mean = bin_grp.mean()
        bin_sem = bin_grp.sem()  # 标准误，用于画置信区间
        bin_center = df_tmp.groupby('bin')['feat'].mean()

        # 单因素方差分析：各 bin 的残差均值是否显著不同（即该特征上是否还有系统性偏差）
        groups = [g.values for _, g in df_tmp.groupby('bin')['resid'] if len(g) >= 3]
        f_stat, f_p = (np.nan, np.nan)
        if len(groups) >= 2:
            f_stat, f_p = scipy_stats.f_oneway(*groups)

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

        axes[0].scatter(feat[valid], residuals[valid], s=6, alpha=0.3, color='steelblue')
        axes[0].axhline(0, color='black', linestyle='--', linewidth=1)
        axes[0].set_title(f'{label} 残差 vs {col}')
        axes[0].set_xlabel(col)
        axes[0].set_ylabel('残差')
        axes[0].grid(True, linestyle=':', alpha=0.6)

        axes[1].errorbar(bin_center.values, bin_mean.values, yerr=1.96 * bin_sem.values,
                          fmt='o-', color='darkorange', ecolor='gray', capsize=3)
        axes[1].axhline(0, color='black', linestyle='--', linewidth=1)
        axes[1].set_title(f'{col} 分箱条件期望 E[残差|分箱]（F检验p={f_p:.4f}）')
        axes[1].set_xlabel(col)
        axes[1].set_ylabel('残差均值（95% CI）')
        axes[1].grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout()
        safe_col = col.replace('/', '_').replace(' ', '_')
        img_path = f"{save_dir}/residual_vs_feature_{tag}_{safe_col}.png"
        plt.savefig(img_path, dpi=300)
        plt.close()

        feature_significance_rows.append({
            '特征': col, '类型': '数值', 'F统计量': f_stat, 'p值': f_p,
            '是否显著相关(p<0.05)': (f_p < 0.05) if not np.isnan(f_p) else None,
        })

    # ---------- 类别特征 ----------
    for col in categorical_cols:
        feat = X[col]
        valid = feat.notna() & residuals.notna()
        if valid.sum() < 10:
            continue

        df_tmp = pd.DataFrame({'feat': feat[valid].astype(str), 'resid': residuals[valid]})
        groups = [g.values for _, g in df_tmp.groupby('feat')['resid'] if len(g) >= 3]
        f_stat, f_p = (np.nan, np.nan)
        if len(groups) >= 2:
            f_stat, f_p = scipy_stats.f_oneway(*groups)

        plt.figure(figsize=(max(6, df_tmp['feat'].nunique() * 0.6), 4.5))
        order = df_tmp.groupby('feat')['resid'].median().sort_values().index
        sns.boxplot(data=df_tmp, x='feat', y='resid', order=order, color='lightsteelblue')
        plt.axhline(0, color='black', linestyle='--', linewidth=1)
        plt.title(f'{label} 残差 vs {col}（类别，F检验p={f_p:.4f}）')
        plt.xlabel(col)
        plt.ylabel('残差')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()

        safe_col = col.replace('/', '_').replace(' ', '_')
        img_path = f"{save_dir}/residual_vs_feature_{tag}_{safe_col}.png"
        plt.savefig(img_path, dpi=300)
        plt.close()

        feature_significance_rows.append({
            '特征': col, '类型': '类别', 'F统计量': f_stat, 'p值': f_p,
            '是否显著相关(p<0.05)': (f_p < 0.05) if not np.isnan(f_p) else None,
        })

    print(f"[特征关系诊断] {label} 共分析 {len(feature_significance_rows)} 个特征，图保存于: {save_dir}")
    return pd.DataFrame(feature_significance_rows).sort_values('p值', na_position='last')


# ==========================================
# 4. 时序自相关诊断（数据需按时间/顺序排列）
# ==========================================
# ==========================================
# 4. 时序自相关诊断（数据需按时间/顺序排列）
# ==========================================
def temporal_diagnostics(
        residuals: pd.Series,
        label: str,
        save_dir: str,
        tag: str = "",
        n_lags=40,
        time_series: pd.Series = None,
        window_size: int = 160
):
    """
    残差时序诊断：
    - 残差 vs 时间/顺序散点图 + Rolling Mean 滑动平均线（捕捉低频漂移）
    - 残差自相关性 (ACF) + Ljung-Box 检验

    time_series: 可选，对应的时间戳序列（pd.Series）。如果不传，默认按索引顺序（Index/Sequence Order）绘图。
    window_size: 计算 Rolling Mean 的窗口大小，默认 160。
    """
    if not HAS_STATSMODELS:
        print("[时序诊断] 未安装 statsmodels，跳过自相关分析（pip install statsmodels 后可用）。")
        return None

    _safe_makedirs(save_dir)
    residuals = pd.Series(residuals).dropna()

    # 获取 X 轴坐标（优先使用传入的时间序列，没有则使用 index/顺序）
    if time_series is not None:
        x_axis = pd.to_datetime(time_series.reindex(residuals.index))
        x_label = "时间"
    else:
        x_axis = np.arange(len(residuals))
        x_label = "样本顺序 (Index)"

    # 计算滑动平均线（Rolling Mean）
    # min_periods=1 保证即使样本数不足 window_size 也能算出来
    roll_mean = residuals.rolling(window=window_size, min_periods=1).mean()

    # 绘制 2x1 子图：上图为残差随时间变化，下图为 ACF 自相关图
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # --- 1. 残差 vs 时间 / 顺序分布图 ---
    axes[0].scatter(x_axis, residuals, s=8, alpha=0.3, color='steelblue', label='残差样本点')
    axes[0].plot(x_axis, roll_mean, color='red', linewidth=2, label=f'滑动平均 (w={window_size})')
    axes[0].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[0].set_title(f'{label} 残差随{x_label}变化趋势')
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel('残差')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle=':', alpha=0.6)

    if time_series is not None:
        fig.autofmt_xdate()  # 如果是时间类型，自动旋转 X 轴日期标签

    # --- 2. ACF 自相关图 ---
    plot_acf(residuals.reset_index(drop=True), ax=axes[1], lags=min(n_lags, len(residuals) // 2 - 1))
    axes[1].set_title(f'{label} 残差自相关函数 (ACF)')
    axes[1].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    img_path = f"{save_dir}/residual_temporal_{tag}.png"
    plt.savefig(img_path, dpi=300)
    plt.close()

    # Ljung-Box 自相关检验
    lb_result = acorr_ljungbox(residuals, lags=[min(10, len(residuals) // 3)], return_df=True)
    lb_stat = lb_result['lb_stat'].iloc[0]
    lb_p = lb_result['lb_pvalue'].iloc[0]

    summary = {
        '残差时序漂移最大幅值(Rolling Max)': roll_mean.max(),
        '残差时序漂移最小幅值(Rolling Min)': roll_mean.min(),
        'Ljung-Box统计量': lb_stat,
        'Ljung-Box_p值': lb_p,
        '是否存在显著自相关(p<0.05)': lb_p < 0.05,
    }
    print(f"[时序诊断] {label} 时序趋势图与 ACF 图已保存至: {img_path}")
    if lb_p < 0.05:
        print("  -> p < 0.05，残差存在显著自相关，说明可能有时序结构（滞后特征/设备漂移等）未被模型捕捉")
    return summary


# ==========================================
# 5. 残差可解释性评分（二次建模）
# ==========================================
def explainability_score(residuals: pd.Series, X: pd.DataFrame, label: str,
                          categorical_features=None, cv=5, random_state=42):
    """
    用同样的特征对残差本身再拟合一个轻量模型，看能解释多少方差（R²）。
    这个 R² 越接近 0，说明残差里已经不剩什么系统性、可被现有特征解释的信息了
    （即模型已经把该学的都学走了）；越大于 0，说明还有信息残留，值得回去做特征工程。

    注意：这里用 cross_val_score 而不是同数据集拟合再评估，避免过拟合把分数虚高。
    """
    if not HAS_SKLEARN:
        print("[可解释性评分] 未安装 sklearn，跳过。")
        return None

    residuals = pd.Series(residuals).reindex(X.index)
    valid = residuals.notna()
    X_valid = X.loc[valid].copy()
    y_valid = residuals.loc[valid]

    # 类别特征简单编码为 category dtype，交给 HistGradientBoostingRegressor 原生处理
    _, categorical_cols = _infer_feature_types(X_valid, categorical_features=categorical_features)
    for col in categorical_cols:
        X_valid[col] = X_valid[col].astype('category')

    model = HistGradientBoostingRegressor(
        max_iter=150, max_depth=4, learning_rate=0.05,
        categorical_features=categorical_cols if categorical_cols else None,
        random_state=random_state,
    )
    scores = cross_val_score(model, X_valid, y_valid, cv=cv, scoring='r2')
    mean_r2 = scores.mean()

    print(f"[可解释性评分] {label} 残差二次建模 交叉验证R² = {mean_r2:.4f} "
          f"({'仍有明显可解释信息残留，建议回查特征工程' if mean_r2 > 0.1 else '残差已接近随机噪声，模型基本吃透现有特征'})")

    return {
        '残差二次建模CV_R2均值': mean_r2,
        '残差二次建模CV_R2标准差': scores.std(),
        '结论': '仍有可解释信息残留' if mean_r2 > 0.1 else '接近随机噪声',
    }


# ==========================================
# 汇总入口：一次性跑全部诊断
# ==========================================
def run_full_residual_diagnostics(
    y_true_train, y_true_test, pred_train, pred_test,
    X_train, X_test,
    label: str, save_dir: str, tag: str = "",
    n_bins=8, categorical_features=None,
    run_temporal=True, run_explainability=True,
    top_n_features=None,
    time_col=None,  # <--- 【新增参数】可以是列名字符串（如 "Produce Time"），也可以是独立的 pd.Series
    rolling_window=160, # <--- 【新增参数】控制 Rolling Mean 窗口大小
):
    """
    一次性跑完 1~5 全部诊断子模块，返回结构化结果字典，
    可以直接拼进现有的多 sheet Excel 导出流程。

    参数说明（保持在5个核心可调参数以内，其余自动推断）：
      n_bins: 分箱数量，用于异方差诊断和特征条件期望分析
      categorical_features: 显式指定类别特征列名列表，不传则自动推断
      run_temporal: 是否跑自相关诊断（数据非时间顺序排列时应设为 False）
      run_explainability: 是否跑二次建模评分（数据量较大时会慢一些）
      top_n_features: 只分析前 N 个特征（特征很多时用于控制画图数量）
    """

    y_true_train = pd.Series(y_true_train)
    y_true_test = pd.Series(y_true_test)
    pred_train = pd.Series(pred_train, index=y_true_train.index) if not isinstance(pred_train, pd.Series) else pred_train
    pred_test = pd.Series(pred_test, index=y_true_test.index) if not isinstance(pred_test, pd.Series) else pred_test

    residuals_train = y_true_train - pred_train
    residuals_test = y_true_test - pred_test

    print(f"\n========== 【{label} 残差诊断（完整版）】 ==========")

    dist_summary = distribution_diagnostics(
        residuals_test, label=f"{label}(测试集)",
        save_dir=f"{save_dir}/distribution", tag=tag
    )

    bin_stats, hetero_summary = heteroscedasticity_diagnostics(
        residuals_train, residuals_test, pred_train, pred_test,
        label=label, save_dir=f"{save_dir}/heteroscedasticity", tag=tag, n_bins=n_bins
    )

    feature_significance_df = feature_relationship_diagnostics(
        residuals_test, X_test, label=f"{label}(测试集)",
        save_dir=f"{save_dir}/feature_relationship", tag=tag,
        n_bins=n_bins, categorical_features=categorical_features,
        top_n_features=top_n_features,
    )
    # 提取时间序列 (如果是列名，从 X_test 中取；如果是 Series，直接使用)
    time_series_test = None
    if time_col is not None:
        if isinstance(time_col, str) and time_col in X_test.columns:
            time_series_test = X_test[time_col]
        elif isinstance(time_col, pd.Series):
            time_series_test = time_col

    temporal_summary = None
    if run_temporal:
        temporal_summary = temporal_diagnostics(
            residuals_test, label=f"{label}(测试集)",
            save_dir=f"{save_dir}/temporal", tag=tag,
            time_series=time_series_test,
            window_size=rolling_window
        )

    explain_summary = None
    if run_explainability:
        explain_summary = explainability_score(
            residuals_test, X_test, label=f"{label}(测试集)",
            categorical_features=categorical_features,
        )

    # 汇总成一张横向表，方便和现有 residual_diagnosis_summary 拼接
    summary_metrics = {}
    summary_metrics.update(dist_summary)
    summary_metrics.update(hetero_summary)
    if temporal_summary:
        summary_metrics.update(temporal_summary)
    if explain_summary:
        summary_metrics.update(explain_summary)

    print(f"========== 【{label} 残差诊断完成】 ==========\n")

    return {
        'summary_metrics': summary_metrics,          # dict，可直接转一行 DataFrame
        'heteroscedasticity_bin_stats': bin_stats,    # DataFrame
        'feature_significance': feature_significance_df,  # DataFrame，按 p 值排序
    }


if __name__ == "__main__":
    # 最小可运行示例，验证模块本身逻辑无误
    np.random.seed(0)
    n = 500
    X_demo = pd.DataFrame({
        'speed': np.random.uniform(20, 100, n),
        'thickness': np.random.uniform(0.3, 2.0, n),
        'grade': np.random.choice(['A', 'B', 'C'], n),
    })
    y_true_demo = 10 + 0.3 * X_demo['speed'] - 2 * X_demo['thickness'] + np.random.normal(0, 1, n)
    # 故意让预测值漏掉 thickness 的影响，制造"残差里还有可解释信息"的场景
    pred_demo = 10 + 0.3 * X_demo['speed'] + np.random.normal(0, 1.5, n)

    split = int(n * 0.8)
    result = run_full_residual_diagnostics(
        y_true_train=y_true_demo[:split], y_true_test=y_true_demo[split:],
        pred_train=pred_demo[:split], pred_test=pred_demo[split:],
        X_train=X_demo.iloc[:split], X_test=X_demo.iloc[split:],
        label="Demo表面", save_dir="demo_result/residual_diagnosis", tag="demo",
        run_temporal=False,  # 示例数据是随机生成，没有时序意义
    )
    print(result['feature_significance'])