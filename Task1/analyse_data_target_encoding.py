import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs("result/cleaned_data", exist_ok=True)
os.makedirs("result/correlation_result", exist_ok=True)
os.makedirs("result/fitting_result", exist_ok=True)


# ==========================================
# 1. 数据预处理、离群点诊断与 Excel 导出 (与原版一致，未改动)
# ==========================================
def preprocess_and_filter_outliers(df,
                                   clean_save_path="result/cleaned_data/cleaned_data.xlsx",
                                   filtered_save_path="result/cleaned_data/filtered_outliers.xlsx"):
    if 'Tining Section_CONCENT[NTU]_GL_1_Avg' in df.columns:
        df.rename(columns={'Tining Section_CONCENT[NTU]_GL_1_Avg': 'Tining Section_CURRENT[A]_GL_1_Avg'}, inplace=True)

    bot_curr_cols = [f'Tining Section_CURRENT[A]_GL_{i}_Avg' for i in range(1, 37, 2)]
    top_curr_cols = [f'Tining Section_CURRENT[A]_GL_{i}_Avg' for i in range(2, 37, 2)]

    df['Bot_Current_Sum'] = df[bot_curr_cols].sum(axis=1)
    df['Top_Current_Sum'] = df[top_curr_cols].sum(axis=1)

    df['Width_m'] = df['Dimension_[mm]_Width'] / 1000.0
    speed = df['Speed[m/min]_Process_Avg'].replace(0, np.nan)

    df['Top_Theoretical_Factor'] = df['Top_Current_Sum'] / (speed * df['Width_m'])
    df['Bot_Theoretical_Factor'] = df['Bot_Current_Sum'] / (speed * df['Width_m'])

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df['Top_Delta'] = df['上表面镀层重量A(XA1_0)'] - df['Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Avg']
    df['Bot_Delta'] = df['下表面镀层重量A(XA1_0)'] - df['Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Avg']

    if 'Steel Grade' in df.columns:
        grade_freq = df['Steel Grade'].value_counts(normalize=True).to_dict()
        df['Steel_Grade_Encoded'] = df['Steel Grade'].map(grade_freq).fillna(0)
    else:
        df['Steel_Grade_Encoded'] = 0

    initial_count = len(df)
    df['Filter_Reason'] = ""

    required_cols = [
        'Top_Current_Sum', 'Bot_Current_Sum',
        'Top_Theoretical_Factor', 'Bot_Theoretical_Factor',
        'Speed[m/min]_Process_Avg', 'Dimension_[mm]_Width', 'Dimension_[mm]_Thickness',
        'Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Avg', 'Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Avg',
        '上表面镀层重量A(XA1_0)', '下表面镀层重量A(XA1_0)',
        'Top_Delta', 'Bot_Delta'
    ]
    null_mask = df[required_cols].isnull().any(axis=1)
    df.loc[null_mask, 'Filter_Reason'] += "关键工艺/测量参数存在缺失值; "

    valid_df = df[~null_mask]
    top_delta_std = valid_df['Top_Delta'].std()
    top_delta_mean = valid_df['Top_Delta'].mean()
    bot_delta_std = valid_df['Bot_Delta'].std()
    bot_delta_mean = valid_df['Bot_Delta'].mean()

    top_threshold = 3.5 * top_delta_std
    bot_threshold = 3.5 * bot_delta_std

    steady_speed_mask = df['Speed[m/min]_Process_Avg'] > 80
    top_outlier_mask = (df['Top_Delta'] - top_delta_mean).abs() > top_threshold
    bot_outlier_mask = (df['Bot_Delta'] - bot_delta_mean).abs() > bot_threshold

    df.loc[
        steady_speed_mask & top_outlier_mask, 'Filter_Reason'] += f"上表面平稳工况下残差偏离过大(>{top_threshold:.2f}g/m2); "
    df.loc[
        steady_speed_mask & bot_outlier_mask, 'Filter_Reason'] += f"下表面平稳工况下残差偏离过大(>{bot_threshold:.2f}g/m2); "

    low_speed_mask = df['Speed[m/min]_Process_Avg'] <= 20
    df.loc[low_speed_mask, 'Filter_Reason'] += "极低速/停机过渡区数据; "

    filtered_df = df[df['Filter_Reason'] != ""].copy()
    clean_df = df[df['Filter_Reason'] == ""].copy()

    cols_to_export = ['Coil ID', 'Steel Grade', 'Speed[m/min]_Process_Avg',
                      'Top_Delta', 'Bot_Delta', 'Filter_Reason']
    cols_to_export = [c for c in cols_to_export if c in filtered_df.columns]

    filtered_df[cols_to_export].to_excel(filtered_save_path, index=False)
    clean_df.to_excel(clean_save_path, index=False)

    print("\n==========================================")
    print("        [数据清洗与异常诊断汇总]          ")
    print("==========================================")
    print(f"原始数据总行数: {initial_count}")
    print(f"被剔除异常点数: {len(filtered_df)} (占比: {len(filtered_df) / initial_count * 100:.2f}%)")
    print(f"保留干净样本数: {len(clean_df)}")
    print(f"[导出提示] 被剔除数据明细及原因已保存至: {filtered_save_path}")
    print(f"[导出提示] 训练用干净数据集已保存至: {clean_save_path}")
    print("==========================================\n")

    return clean_df


# ==========================================
# 2. 新增：钢种 Target Encoding（时序安全，无泄漏）
# ==========================================
def add_time_safe_target_encoding(df, surface, time_col='Produce Time',
                                   grade_col='Steel Grade', smoothing=10):
    """
    用"该钢种历史上的平均 Delta（实验室值-在线值）"来编码钢种，
    直接把"MR T-5 CA 历史上更容易偏低"这类信息喂给模型，
    比单纯的频率编码信息量大得多。

    关键点：必须按时间顺序，只用"过去"的数据计算编码值（expanding mean，
    并整体滞后一行），否则会把当前行自己的答案泄漏给自己，
    导致训练集指标虚高、测试集不可信。

    smoothing: 贝叶斯平滑强度。钢种样本数少的时候，编码值会被拉向全局均值，
               避免小样本钢种的编码值出现极端值。
    """
    prefix = 'Top' if surface == 'Top' else 'Bot'
    delta_col = f'{prefix}_Delta'
    encoded_col = f'{prefix}_Grade_TargetEnc'

    if time_col in df.columns:
        df = df.sort_values(time_col).copy()
    else:
        # 没有可用时间列时退化为按当前行序（即数据本身的顺序），
        # 假定数据已经是按生产时间排列的
        df = df.copy()

    global_mean = df[delta_col].mean()

    # 按钢种分组，计算截止到"上一行"为止的累计均值和累计计数（expanding，shift(1) 避免用到自己）
    grouped = df.groupby(grade_col)[delta_col]
    expanding_sum = grouped.cumsum().shift(1)
    expanding_count = grouped.cumcount()  # cumcount 本身就是"之前出现过几次"，不需要再shift

    # 贝叶斯平滑：小样本时向全局均值收缩
    smoothed_encoding = (expanding_sum.fillna(0) + smoothing * global_mean) / (expanding_count + smoothing)

    # 每个钢种第一次出现时 expanding_count=0，此时 smoothed_encoding 直接等于 global_mean，符合预期
    df[encoded_col] = smoothed_encoding.fillna(global_mean)

    return df, encoded_col, global_mean


def compute_full_history_grade_encoding(train_df, surface, grade_col='Steel Grade', smoothing=10):
    """
    用训练集的完整历史（不再是expanding，而是训练集内的全量均值）计算每个钢种的编码值，
    用于对测试集做映射（测试集本身不能用来计算自己的编码，只能查训练集学到的表）。
    """
    prefix = 'Top' if surface == 'Top' else 'Bot'
    delta_col = f'{prefix}_Delta'

    global_mean = train_df[delta_col].mean()
    grade_stats = train_df.groupby(grade_col)[delta_col].agg(['mean', 'count'])
    grade_stats['smoothed'] = (grade_stats['mean'] * grade_stats['count'] + smoothing * global_mean) / \
                               (grade_stats['count'] + smoothing)

    mapping = grade_stats['smoothed'].to_dict()
    return mapping, global_mean


# ==========================================
# 3. 新增：厚度非线性特征（针对 0.18~0.209mm 的非单调凸起区间）
# ==========================================
def add_thickness_nonlinear_features(df, low=0.18, high=0.209):
    """
    诊断显示厚度在 [0.18, 0.209] 区间内偏低发生率显著更高（非单调"中间凸起"模式），
    普通线性厚度特征无法表达这种关系，这里额外构造：
    1. Thickness_In_Hotzone: 是否落在该高发区间的二值特征；
    2. Thickness_Dist_To_Hotzone_Center: 与该区间中心点的距离，
       让模型能表达"越靠近中心风险越高、越远离风险越低"的连续关系。
    """
    center = (low + high) / 2.0
    df['Thickness_In_Hotzone'] = ((df['Dimension_[mm]_Thickness'] >= low) &
                                   (df['Dimension_[mm]_Thickness'] <= high)).astype(int)
    df['Thickness_Dist_To_Hotzone_Center'] = (df['Dimension_[mm]_Thickness'] - center).abs()
    return df


# ==========================================
# 4. 相关性分析 (未改动)
# ==========================================
def analyze_correlations(df, surface='Top'):
    prefix = 'Top' if surface == 'Top' else 'Bot'
    surface_cn = '上' if surface == 'Top' else '下'

    actual_col = f'Tin Weight_Actual[g/m2]_GALV_WEIGHT_{prefix.upper()}_Avg'
    lab_col = f'{surface_cn}表面镀层重量A(XA1_0)'

    cols_to_check = [
        lab_col,
        actual_col,
        f'{prefix}_Current_Sum',
        f'{prefix}_Theoretical_Factor',
        'Speed[m/min]_Process_Avg',
        'Dimension_[mm]_Thickness',
        'Dimension_[mm]_Width',
        'Steel_Grade_Encoded'
    ]

    corr_matrix = df[cols_to_check].corr()

    print(f"\n======== 【{surface_cn}表面 相关性矩阵】 ========")
    print(corr_matrix[lab_col].sort_values(ascending=False))

    plt.figure(figsize=(9, 7))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title(f'{surface_cn}表面参数与实验室测定值相关性热力图')
    plt.tight_layout()

    save_img_path = f"result/correlation_result/correlation_{surface}.png"
    plt.savefig(save_img_path, dpi=300)
    print(f"[图表保存] {surface_cn}表面相关性热力图已保存至: {save_img_path}")
    plt.show()


def check_residual_distribution(df):
    print("\n==========================================")
    print("      【数据集中原始残差正负分布诊断】       ")
    print("==========================================")
    for surface in ['Top', 'Bot']:
        surface_cn = '上' if surface == 'Top' else '下'
        delta_col = f'{surface}_Delta'
        if delta_col in df.columns:
            total = len(df[delta_col].dropna())
            pos = (df[delta_col] > 0).sum()
            neg = (df[delta_col] < 0).sum()
            mean_val = df[delta_col].mean()
            print(f"[{surface_cn}表面 Delta (实验室值 - 在线值)]")
            print(f"  - 总有效样本数: {total}")
            print(f"  - Delta > 0 (在线测量偏低): {pos} 条 (占比 {pos/total*100:.2f}%)")
            print(f"  - Delta < 0 (在线测量偏高): {neg} 条 (占比 {neg/total*100:.2f}%)")
            print(f"  - Delta 均值: {mean_val:.4f} g/m2")
    print("==========================================\n")


# ==========================================
# 5. 残差建模核心类 (未改动)
# ==========================================
def compute_direction_sample_weight(y_delta, pos_boost=1.0, damping=0.0):
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
    def __init__(self, monotonic_feature_idx=None, alpha_smoothing=0.7,
                 pos_boost=1.0, damping=0.0):
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
# 6. 表面建模与图形输出（核心改动：加入 target encoding + 厚度非线性特征）
# ==========================================
def run_surface_pipeline(df, surface='Top', damping=0.0, alpha_smoothing=0.7, pos_boost=1.0,
                          te_smoothing=10, thickness_hotzone=(0.18, 0.209)):
    prefix = 'Top' if surface == 'Top' else 'Bot'
    surface_cn = '上' if surface == 'Top' else '下'

    print(f"\n==========================================")
    print(f"        开始运行【{surface_cn}表面】模型拟合与分析     ")
    print(f"        (damping={damping}, alpha_smoothing={alpha_smoothing})")
    print(f"==========================================")

    analyze_correlations(df, surface=surface)

    # ---- 厚度非线性特征 ----
    df = add_thickness_nonlinear_features(df, low=thickness_hotzone[0], high=thickness_hotzone[1])

    # 电流衍生特征
    speed_col = 'Speed[m/min]_Process_Avg'
    current_col = f'{prefix}_Current_Sum'
    df[f'{prefix}_Current_Per_Speed'] = df[current_col] / (df[speed_col] + 1e-5)

    online_col = f'Tin Weight_Actual[g/m2]_GALV_WEIGHT_{prefix.upper()}_Avg'
    delta_col = f'{prefix}_Delta'

    # 3. 先按时间排序、划分训练/测试（时序安全的前提）
    if 'Produce Time' in df.columns:
        df = df.sort_values('Produce Time').reset_index(drop=False).rename(columns={'index': 'Original_Index'})
    else:
        df = df.reset_index(drop=False).rename(columns={'index': 'Original_Index'})

    split_point = int(len(df) * 0.8)
    train_df = df.iloc[:split_point].copy()
    test_df = df.iloc[split_point:].copy()

    # ---- 钢种 Target Encoding：只用训练集历史计算编码表，测试集查表映射（避免泄漏）----
    grade_map, global_mean = compute_full_history_grade_encoding(
        train_df, surface=surface, grade_col='Steel Grade', smoothing=te_smoothing
    )
    encoded_col = f'{prefix}_Grade_TargetEnc'

    # 训练集内部用 expanding（只用"过去"数据）计算编码，避免用到当前行自己的答案
    train_df, _, _ = add_time_safe_target_encoding(
        train_df, surface=surface, time_col='Produce Time' if 'Produce Time' in train_df.columns else None,
        grade_col='Steel Grade', smoothing=te_smoothing
    )
    # 测试集统一使用训练集学到的映射表；训练集中未出现过的新钢种用全局均值兜底
    test_df[encoded_col] = test_df['Steel Grade'].map(grade_map).fillna(global_mean)

    # 特征列表：在线测量值放在第0位，方便对其施加单调约束
    feature_cols = [
        online_col,
        current_col,
        f'{prefix}_Current_Per_Speed',
        f'{prefix}_Theoretical_Factor',
        speed_col,
        'Dimension_[mm]_Width',
        'Dimension_[mm]_Thickness',
        'Thickness_In_Hotzone',
        'Thickness_Dist_To_Hotzone_Center',
        encoded_col,          # 新增：钢种 target encoding，替代原来的频率编码
    ]
    online_feature_idx = feature_cols.index(online_col)

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_delta_train = train_df[delta_col]
    y_delta_test = test_df[delta_col]
    actual_train = train_df[online_col]
    actual_test = test_df[online_col]
    y_true_train = train_df[f'{surface_cn}表面镀层重量A(XA1_0)']
    y_true_test = test_df[f'{surface_cn}表面镀层重量A(XA1_0)']

    # 4. 残差建模 + 单调约束 + EWMA平滑
    corrector = ResidualCorrectionModel(
        monotonic_feature_idx=online_feature_idx,
        alpha_smoothing=alpha_smoothing,
        pos_boost=pos_boost,
        damping=damping
    )
    corrector.fit(X_train, y_delta_train)

    pred_series, predicted_delta_smooth = corrector.predict_smooth(X_test, actual_test)

    y_true_series = y_true_test
    online_series = actual_test

    raw_residuals = y_true_series - online_series
    model_residuals = y_true_series - pred_series

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

    r2_online = r2_score(y_true_series, online_series)
    r2_model = r2_score(y_true_series, pred_series)
    rmse_online = np.sqrt(mean_squared_error(y_true_series, online_series))
    rmse_model = np.sqrt(mean_squared_error(y_true_series, pred_series))

    print(f"======== 【{surface_cn}表面 拟合性能评估（测试集）】 ========")
    print(f"原始在线仪表与实验室真实值 -> R²: {r2_online:.4f}, RMSE: {rmse_online:.4f}")
    print(f"模型校正拟合后与实验室真实值 -> R²: {r2_model:.4f}, RMSE: {rmse_model:.4f}")

    # 特征重要性，方便确认 target encoding / 厚度非线性特征是否真的被模型用上了
    # 注意：HistGradientBoostingRegressor 不自带 feature_importances_，
    # 这里用 permutation_importance 在测试集上计算（打乱某列后模型误差上升越多，说明该特征越重要）
    perm_result = permutation_importance(
        corrector.model, X_test, y_delta_test,
        n_repeats=10, random_state=42, scoring='neg_mean_absolute_error'
    )
    importances = pd.Series(perm_result.importances_mean, index=feature_cols).sort_values(ascending=False)
    print(f"\n---- 特征重要性（{surface_cn}表面，permutation importance，基于测试集） ----")
    print(importances.to_string())

    start_idx = test_df['Original_Index'].iloc[0]
    end_idx = test_df['Original_Index'].iloc[-1]

    plt.figure(figsize=(12, 5))
    plt.plot(y_true_series.values, label='实验室真实测量值 (True Label)', color='black', linewidth=1.5)
    plt.plot(online_series.values, label='在线仪表原始测量值 (Online)', color='red', linestyle='--', alpha=0.7)
    plt.plot(pred_series.values, label='模型残差校正值 (Model Pred)', color='green', linewidth=1.5, alpha=0.85)
    plt.title(f'{surface_cn}表面 镀层重量拟合对照图（原始数据行号: {start_idx} ~ {end_idx}）')
    plt.xlabel('测试集样本序号')
    plt.ylabel('镀层重量 (g/m2)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    fit_img_path = f"result/fitting_result/fitting_result_{surface}.png"
    plt.savefig(fit_img_path, dpi=300)
    print(f"[图表保存] {surface_cn}表面拟合对照图已保存至: {fit_img_path}")
    plt.show()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})

    ax1.plot(raw_residuals.values, label='原始在线仪表残差 (True - Online)', color='red', alpha=0.5, linewidth=1)
    ax1.plot(model_residuals.values, label='模型校正后残差 (True - Model)', color='green', alpha=0.8, linewidth=1.2)
    ax1.axhline(0, color='black', linestyle='--', linewidth=1)
    ax1.set_title(f'{surface_cn}表面 预测残差变化对比（原始数据行号: {start_idx} ~ {end_idx}）')
    ax1.set_xlabel('测试集样本序号')
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
    res_img_path = f"result/fitting_result/residual_analysis_{surface}.png"
    plt.savefig(res_img_path, dpi=300)
    print(f"[图表保存] {surface_cn}表面残差分析图已保存至: {res_img_path}")
    plt.show()

    return corrector, importances


# ==========================================
# 7. 主流程
# ==========================================
if __name__ == "__main__":
    raw_df = pd.read_excel("result/merged_data/merged_result_latest.xlsx")

    clean_df = preprocess_and_filter_outliers(
        raw_df,
        clean_save_path="result/cleaned_data/cleaned_data.xlsx",
        filtered_save_path="result/cleaned_data/filtered_outliers.xlsx"
    )

    check_residual_distribution(clean_df)

    # 默认配置 damping=0.0（不加权）+ alpha_smoothing=0.7，是网格搜索得到的整体RMSE最优解。
    # 本版新增：钢种 target encoding（time-safe）+ 厚度非线性特征
    run_surface_pipeline(clean_df, surface='Top', damping=0.0, alpha_smoothing=0.7)
    run_surface_pipeline(clean_df, surface='Bot', damping=0.0, alpha_smoothing=0.7)