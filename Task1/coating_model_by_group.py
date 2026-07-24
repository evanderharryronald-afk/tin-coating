import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 设置画图支持中文与负号，消除特殊字符警告
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建结果保存目录结构
os.makedirs("result/cleaned_data", exist_ok=True)
os.makedirs("result/correlation_result", exist_ok=True)
os.makedirs("result/fitting_result", exist_ok=True)

# 每个规格组最少需要的样本数，低于此值只诊断不建模
MIN_GROUP_SAMPLES = 200

# Setpoint 分组用的两个字段（上下表面镀层重量下限设定值）
TOP_SETPOINT_COL = 'Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_TOP_Min'
BOT_SETPOINT_COL = 'Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_BOT_Min'


# ==========================================
# 1. 数据预处理、离群点诊断与 Excel 导出
# ==========================================
def preprocess_and_filter_outliers(df,
                                   clean_save_path="result/cleaned_data/cleaned_data.xlsx",
                                   filtered_save_path="result/cleaned_data/filtered_outliers.xlsx"):
    """
    数据预处理、诊断离群点并导出Excel记录剔除原因
    """
    # 1.0 纠正列名
    if 'Tining Section_CONCENT[NTU]_GL_1_Avg' in df.columns:
        df.rename(columns={'Tining Section_CONCENT[NTU]_GL_1_Avg': 'Tining Section_CURRENT[A]_GL_1_Avg'}, inplace=True)

    # 1.1 电流求和处理
    bot_curr_cols = [f'Tining Section_CURRENT[A]_GL_{i}_Avg' for i in range(1, 37, 2)]
    top_curr_cols = [f'Tining Section_CURRENT[A]_GL_{i}_Avg' for i in range(2, 37, 2)]

    df['Bot_Current_Sum'] = df[bot_curr_cols].sum(axis=1)
    df['Top_Current_Sum'] = df[top_curr_cols].sum(axis=1)

    # 1.2 构建理论因子
    df['Width_m'] = df['Dimension_[mm]_Width'] / 1000.0
    speed = df['Speed[m/min]_Process_Avg'].replace(0, np.nan)

    df['Top_Theoretical_Factor'] = df['Top_Current_Sum'] / (speed * df['Width_m'])
    df['Bot_Theoretical_Factor'] = df['Bot_Current_Sum'] / (speed * df['Width_m'])

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 1.3 计算目标残差 (Residual = 实验室真实值 - 在线测量值)
    df['Top_Delta'] = df['上表面镀层重量A(XA1_0)'] - df['Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Avg']
    df['Bot_Delta'] = df['下表面镀层重量A(XA1_0)'] - df['Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Avg']

    # 1.4 Steel Grade 钢种频率编码
    if 'Steel Grade' in df.columns:
        grade_freq = df['Steel Grade'].value_counts(normalize=True).to_dict()
        df['Steel_Grade_Encoded'] = df['Steel Grade'].map(grade_freq).fillna(0)
    else:
        df['Steel_Grade_Encoded'] = 0

    # ----------------------------------------------------
    # 离群点诊断与规则过滤逻辑
    # ----------------------------------------------------
    initial_count = len(df)
    df['Filter_Reason'] = ""

    # 规则 1: 基础关键字段存在缺失值
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

    # 计算残差分布统计量 (基于非空子集)
    valid_df = df[~null_mask]
    top_delta_std = valid_df['Top_Delta'].std()
    top_delta_mean = valid_df['Top_Delta'].mean()
    bot_delta_std = valid_df['Bot_Delta'].std()
    bot_delta_mean = valid_df['Bot_Delta'].mean()

    # 设置残差异常阈值 (例如超出 3.5 倍标准差)
    top_threshold = 3.5 * top_delta_std
    bot_threshold = 3.5 * bot_delta_std

    # 规则 2: 平稳工况下残差极大 (离群数据噪声)
    steady_speed_mask = df['Speed[m/min]_Process_Avg'] > 80
    top_outlier_mask = (df['Top_Delta'] - top_delta_mean).abs() > top_threshold
    bot_outlier_mask = (df['Bot_Delta'] - bot_delta_mean).abs() > bot_threshold

    df.loc[
        steady_speed_mask & top_outlier_mask, 'Filter_Reason'] += f"上表面平稳工况下残差偏离过大(>{top_threshold:.2f}g/m2); "
    df.loc[
        steady_speed_mask & bot_outlier_mask, 'Filter_Reason'] += f"下表面平稳工况下残差偏离过大(>{bot_threshold:.2f}g/m2); "

    # 规则 3: 停机/低速区残差异常 (工艺非正常状态)
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
# 1.5 按镀层规格（Setpoint 组合）分组
# ==========================================
def build_setpoint_group_key(df, top_col=TOP_SETPOINT_COL, bot_col=BOT_SETPOINT_COL):
    """
    用 (Top_Min, Bot_Min) 的精确组合作为镀层规格分组键。
    不做四舍五入，只按原始唯一值分组 —— 意味着同一规格在源数据里
    必须是完全一致的数值，如果上游系统对同一规格记录了轻微浮点误差
    （例如 1.1000000001 和 1.1），会被当成两个不同的组，需要提前确认
    源数据里这两列的写入方式是否足够干净。
    """
    if top_col not in df.columns or bot_col not in df.columns:
        raise KeyError(f"缺少分组所需字段: {top_col} 或 {bot_col}")

    df = df.copy()
    df['Setpoint_Group_Key'] = list(zip(df[top_col], df[bot_col]))
    df['Setpoint_Group_Label'] = df.apply(
        lambda r: f"Top{r[top_col]}_Bot{r[bot_col]}", axis=1
    )
    return df


def summarize_setpoint_groups(df):
    """打印每个镀层规格组的样本量，区分达标组与跳过组，表格化输出"""
    group_sizes = df.groupby('Setpoint_Group_Label').size().sort_values(ascending=False)

    valid_groups = group_sizes[group_sizes >= MIN_GROUP_SAMPLES]
    skipped_groups = group_sizes[group_sizes < MIN_GROUP_SAMPLES]

    print("\n==========================================")
    print("        [镀层规格分组样本量汇总]           ")
    print("==========================================")
    print(f"共 {len(group_sizes)} 个规格组，达标 {len(valid_groups)} 个，跳过 {len(skipped_groups)} 个")
    print(f"（达标阈值: >= {MIN_GROUP_SAMPLES} 条）\n")

    print(f"{'规格组':<30}{'样本数':>10}   状态")
    print("-" * 55)
    for label, size in valid_groups.items():
        print(f"{label:<30}{size:>10}   建模")
    for label, size in skipped_groups.items():
        print(f"{label:<30}{size:>10}   跳过")
    print("==========================================\n")

    return group_sizes


# ==========================================
# 2. 相关性分析
# ==========================================
def analyze_correlations(df, surface='Top', group_tag=""):
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

    tag_display = f"[{group_tag}] " if group_tag else ""
    print(f"\n======== 【{tag_display}{surface_cn}表面 相关性矩阵】 ========")
    print(corr_matrix[lab_col].sort_values(ascending=False))

    plt.figure(figsize=(9, 7))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title(f'{tag_display}{surface_cn}表面参数与实验室测定值相关性热力图')
    plt.tight_layout()

    safe_tag = f"_{group_tag}" if group_tag else ""
    save_img_path = f"result/grouped_by_coating_weight/correlation_result/correlation_{surface}{safe_tag}.png"
    plt.savefig(save_img_path, dpi=300)
    print(f"[图表保存] {tag_display}{surface_cn}表面相关性热力图已保存至: {save_img_path}")
    plt.close()


def check_residual_distribution(df, group_tag=""):
    """单独排查数据集残差分布状况的辅助函数"""
    tag_display = f"[{group_tag}] " if group_tag else ""
    print("\n==========================================")
    print(f"      【{tag_display}数据集中原始残差正负分布诊断】       ")
    print("==========================================")
    for surface in ['Top', 'Bot']:
        surface_cn = '上' if surface == 'Top' else '下'
        delta_col = f'{surface}_Delta'
        if delta_col in df.columns:
            total = len(df[delta_col].dropna())
            if total == 0:
                continue
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
# 4. 表面建模与图形输出（新增 group_tag 用于区分不同镀层规格组）
# ==========================================
def run_surface_pipeline(df, surface='Top', damping=0.0, alpha_smoothing=0.7,
                         pos_boost=1.0, group_tag=""):
    prefix = 'Top' if surface == 'Top' else 'Bot'
    surface_cn = '上' if surface == 'Top' else '下'
    tag_display = f"[{group_tag}] " if group_tag else ""
    safe_tag = f"_{group_tag}" if group_tag else ""

    print(f"\n==========================================")
    print(f"    开始运行【{tag_display}{surface_cn}表面】模型拟合与分析     ")
    print(f"        (damping={damping}, alpha_smoothing={alpha_smoothing})")
    print(f"==========================================")

    # 1. 相关性分析
    analyze_correlations(df, surface=surface, group_tag=group_tag)

    # 2. 特征工程
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

    # 3. 按时间划分 (保持原始索引 Index 不重置)
    X_train, X_test, y_delta_train, y_delta_test, actual_train, actual_test, y_true_train, y_true_test = \
        train_test_split(X, y_delta, online_actual, y_true_full, test_size=0.2, shuffle=False)

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

    print(f"\n-------- 【{tag_display}{surface_cn}表面 模型矫正前后残差诊断】 --------")
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

    print(f"======== 【{tag_display}{surface_cn}表面 拟合性能评估（测试集）】 ========")
    print(f"原始在线仪表与实验室真实值 -> R²: {r2_online:.4f}, RMSE: {rmse_online:.4f}")
    print(f"模型校正拟合后与实验室真实值 -> R²: {r2_model:.4f}, RMSE: {rmse_model:.4f}")

    start_idx = X_test.index[0]
    end_idx = X_test.index[-1]

    # 5. 拟合对比图
    plt.figure(figsize=(12, 5))
    plt.plot(y_true_series, label='实验室真实测量值 (True Label)', color='black', linewidth=1.5)
    plt.plot(online_series, label='在线仪表原始测量值 (Online)', color='red', linestyle='--', alpha=0.7)
    plt.plot(pred_series, label='模型残差校正值 (Model Pred)', color='green', linewidth=1.5, alpha=0.85)
    plt.title(f'{tag_display}{surface_cn}表面 镀层重量拟合对照图（原始数据行号: {start_idx} ~ {end_idx}）')
    plt.xlabel('原始数据行号 (Original Row Index)')
    plt.ylabel('镀层重量 (g/m2)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    fit_img_path = f"result/grouped_by_coating_weight/fitting_result/fitting_result/fitting_result_{surface}{safe_tag}.png"
    plt.savefig(fit_img_path, dpi=300)
    print(f"[图表保存] {tag_display}{surface_cn}表面拟合对照图已保存至: {fit_img_path}")
    plt.close()

    # 6. 残差对比图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})

    ax1.plot(raw_residuals, label='原始在线仪表残差 (True - Online)', color='red', alpha=0.5, linewidth=1)
    ax1.plot(model_residuals, label='模型校正后残差 (True - Model)', color='green', alpha=0.8, linewidth=1.2)
    ax1.axhline(0, color='black', linestyle='--', linewidth=1)
    ax1.set_title(f'{tag_display}{surface_cn}表面 预测残差变化对比（原始数据行号: {start_idx} ~ {end_idx}）')
    ax1.set_xlabel('原始数据行号 (Original Row Index)')
    ax1.set_ylabel('残差/误差 (g/m2)')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    sns.histplot(raw_residuals, ax=ax2, color='red', label='原始残差分布', kde=True, stat="density", alpha=0.3)
    sns.histplot(model_residuals, ax=ax2, color='green', label='模型校正后残差分布', kde=True, stat="density",
                 alpha=0.5)
    ax2.axvline(0, color='black', linestyle='--', linewidth=1)
    ax2.set_title(f'{tag_display}{surface_cn}表面 残差概率密度分布（越集中在0且越窄越好）')
    ax2.set_xlabel('残差/误差 (g/m2)')
    ax2.set_ylabel('概率密度')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    res_img_path = f"result/grouped_by_coating_weight/fitting_result/residual_analysis/residual_analysis_{surface}{safe_tag}.png"
    plt.savefig(res_img_path, dpi=300)
    print(f"[图表保存] {tag_display}{surface_cn}表面残差分析图已保存至: {res_img_path}")
    plt.close()

    metrics = {
        '规格组': group_tag,
        '表面': surface,
        '训练样本数': len(X_train),
        '测试样本数': len(X_test),
        'R2_在线': r2_online,
        'R2_模型': r2_model,
        'R2_提升': r2_model - r2_online,
        'RMSE_在线': rmse_online,
        'RMSE_模型': rmse_model,
        'RMSE_提升(%)': (rmse_online - rmse_model) / rmse_online * 100 if rmse_online != 0 else np.nan,
        'MAE_在线': raw_residuals.abs().mean(),
        'MAE_模型': model_residuals.abs().mean(),
        '偏差均值_在线': raw_residuals.mean(),
        '偏差均值_模型': model_residuals.mean(),
        '正偏差样本数': int(mask_pos.sum()),
        '正偏差MAE_在线': raw_residuals[mask_pos].abs().mean() if mask_pos.sum() > 0 else np.nan,
        '正偏差MAE_模型': model_residuals[mask_pos].abs().mean() if mask_pos.sum() > 0 else np.nan,
        '负偏差样本数': int(mask_neg.sum()),
        '负偏差MAE_在线': raw_residuals[mask_neg].abs().mean() if mask_neg.sum() > 0 else np.nan,
        '负偏差MAE_模型': model_residuals[mask_neg].abs().mean() if mask_neg.sum() > 0 else np.nan,
    }

    return corrector, metrics


# ==========================================
# 5. 主流程：按镀层规格分组，逐组训练 Top/Bot 模型
# ==========================================
if __name__ == "__main__":
    raw_df = pd.read_excel("result/merged_data/merged_result_latest.xlsx")

    # 步骤 1: 预处理、诊断离群点并自动导出 filtered_outliers.xlsx
    clean_df = preprocess_and_filter_outliers(
        raw_df,
        clean_save_path="result/cleaned_data/cleaned_data.xlsx",
        filtered_save_path="result/cleaned_data/filtered_outliers.xlsx"
    )

    # 步骤 2: 按 (Top_Min, Bot_Min) 精确组合构建镀层规格分组
    clean_df = build_setpoint_group_key(clean_df)
    group_sizes = summarize_setpoint_groups(clean_df)

    # 全量数据整体的残差分布诊断（作为对照基线，可选保留）
    check_residual_distribution(clean_df, group_tag="全部规格汇总")

    trained_models = {}
    all_metrics = []

    # 步骤 3: 逐个镀层规格组，分别训练 Top / Bot 模型
    for group_label, group_size in group_sizes.items():
        group_df = clean_df[clean_df['Setpoint_Group_Label'] == group_label].copy()

        if group_size < MIN_GROUP_SAMPLES:
            print(f"[跳过] 规格组 {group_label} 样本量 {group_size} < {MIN_GROUP_SAMPLES}，不做建模与诊断。")
            # print(f"[跳过建模] 规格组 {group_label} 样本量 {group_size} < {MIN_GROUP_SAMPLES}，"
            #       f"仅做残差分布诊断，不单独训练模型。")
            # check_residual_distribution(group_df, group_tag=group_label)
            continue

        print(f"\n########## 规格组 {group_label} (样本量 {group_size}) 开始建模 ##########")
        check_residual_distribution(group_df, group_tag=group_label)

        top_model, top_metrics = run_surface_pipeline(
            group_df, surface='Top', damping=0.0, alpha_smoothing=0.7, group_tag=group_label
        )
        bot_model, bot_metrics = run_surface_pipeline(
            group_df, surface='Bot', damping=0.0, alpha_smoothing=0.7, group_tag=group_label
        )
        trained_models[(group_label, 'Top')] = top_model
        trained_models[(group_label, 'Bot')] = bot_model
        all_metrics.append(top_metrics)
        all_metrics.append(bot_metrics)

    print("\n==========================================")
    print(f"全部规格组处理完毕，共成功训练 {len(trained_models)} 个模型（每组 Top/Bot 各一个）。")
    print("==========================================")
    # 导出样本量汇总表 + 建模效果汇总表到同一个 Excel 的不同 sheet
    sample_summary_df = group_sizes.reset_index()
    sample_summary_df.columns = ['规格组', '样本数']
    sample_summary_df['状态'] = sample_summary_df['样本数'].apply(
        lambda s: '建模' if s >= MIN_GROUP_SAMPLES else '跳过'
    )

    report_path = "result/grouped_by_coating_weight/summary_report.xlsx"
    with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
        sample_summary_df.to_excel(writer, sheet_name='样本量汇总', index=False)
        if all_metrics:
            metrics_df = pd.DataFrame(all_metrics)
            metrics_df.to_excel(writer, sheet_name='建模效果汇总', index=False)
        else:
            pd.DataFrame({'提示': ['没有达标组完成建模']}).to_excel(
                writer, sheet_name='建模效果汇总', index=False
            )

    print(f"[导出提示] 汇总报表已保存至: {report_path}")