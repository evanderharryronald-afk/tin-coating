# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy import stats
#
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
# plt.rcParams['axes.unicode_minus'] = False
#
# os.makedirs("result/low_reading_analysis", exist_ok=True)
#
#
# # ==========================================
# # 1. 抽取"在线偏低"子集 (Delta > 0)
# # ==========================================
# def extract_low_reading_subset(df, surface='Top'):
#     """
#     Delta = 实验室真实值 - 在线值。Delta > 0 表示在线测量偏低。
#     返回 (低值子集, 其余子集)，方便对照分析。
#     """
#     delta_col = f'{surface}_Delta'
#     low_mask = df[delta_col] > 0
#     low_df = df[low_mask].copy()
#     rest_df = df[~low_mask].copy()
#     return low_df, rest_df
#
#
# # ==========================================
# # 2. 数值特征对比：均值/中位数/分布差异 + 假设检验
# # ==========================================
# def compare_numeric_features(low_df, rest_df, surface='Top'):
#     prefix = 'Top' if surface == 'Top' else 'Bot'
#
#     candidate_cols = [
#         'Dimension_[mm]_Thickness', 'Dimension_[mm]_Width', 'Dimension_[mm]_Length',
#         'Speed[m/min]_Process_Avg',
#         f'{prefix}_Current_Sum', f'{prefix}_Theoretical_Factor',
#         'Steel_Grade_Encoded',
#         f'Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_{prefix.upper()}_Avg',
#         f'Tin Weight_Actual[g/m2]_GALV_WEIGHT_{prefix.upper()}_Avg',
#     ]
#     candidate_cols = [c for c in candidate_cols if c in low_df.columns]
#
#     rows = []
#     for col in candidate_cols:
#         low_vals = low_df[col].dropna()
#         rest_vals = rest_df[col].dropna()
#         if len(low_vals) < 5 or len(rest_vals) < 5:
#             continue
#
#         # Mann-Whitney U 检验：不要求正态分布，检验两组中位数是否有显著差异
#         try:
#             stat, p_value = stats.mannwhitneyu(low_vals, rest_vals, alternative='two-sided')
#         except ValueError:
#             p_value = np.nan
#
#         rows.append({
#             '特征': col,
#             '偏低组_均值': low_vals.mean(),
#             '偏低组_中位数': low_vals.median(),
#             '偏低组_标准差': low_vals.std(),
#             '其余组_均值': rest_vals.mean(),
#             '其余组_中位数': rest_vals.median(),
#             '其余组_标准差': rest_vals.std(),
#             '均值差(偏低-其余)': low_vals.mean() - rest_vals.mean(),
#             'p_value': p_value,
#             '显著性(p<0.05)': 'Yes' if (not np.isnan(p_value) and p_value < 0.05) else 'No'
#         })
#
#     result_df = pd.DataFrame(rows).sort_values('p_value')
#     return result_df
#
#
# # ==========================================
# # 3. 分类特征对比：钢种分布是否有明显偏移
# # ==========================================
# def compare_categorical_features(low_df, rest_df, col='Steel Grade'):
#     if col not in low_df.columns:
#         return None
#
#     low_ratio = low_df[col].value_counts(normalize=True)
#     rest_ratio = rest_df[col].value_counts(normalize=True)
#
#     combined = pd.DataFrame({
#         '偏低组占比': low_ratio,
#         '其余组占比': rest_ratio
#     }).fillna(0)
#     combined['占比差(偏低-其余)'] = combined['偏低组占比'] - combined['其余组占比']
#     combined = combined.sort_values('占比差(偏低-其余)', ascending=False)
#     return combined
#
#
# # ==========================================
# # 4. 分箱分析：速度、厚度、宽度区间上"偏低"的发生率是否有规律
# # ==========================================
# def analyze_bucket_incidence(df, surface='Top', bucket_col='Speed[m/min]_Process_Avg', n_bins=8):
#     delta_col = f'{surface}_Delta'
#     tmp = df[[bucket_col, delta_col]].dropna().copy()
#     tmp['is_low'] = tmp[delta_col] > 0
#
#     try:
#         tmp['bucket'] = pd.qcut(tmp[bucket_col], q=n_bins, duplicates='drop')
#     except ValueError:
#         tmp['bucket'] = pd.cut(tmp[bucket_col], bins=n_bins)
#
#     summary = tmp.groupby('bucket', observed=True).agg(
#         样本数=('is_low', 'size'),
#         偏低发生率=('is_low', 'mean')
#     ).reset_index()
#     summary['偏低发生率'] = (summary['偏低发生率'] * 100).round(2)
#     return summary
#
#
# # ==========================================
# # 5. 时间趋势：偏低样本是否集中在某些时间段（如仪表漂移/换液后）
# # ==========================================
# def analyze_time_trend(df, surface='Top', time_col='Produce Time'):
#     if time_col not in df.columns:
#         return None
#
#     delta_col = f'{surface}_Delta'
#     tmp = df[[time_col, delta_col]].dropna().copy()
#     tmp[time_col] = pd.to_datetime(tmp[time_col], errors='coerce')
#     tmp = tmp.dropna(subset=[time_col])
#     tmp['is_low'] = tmp[delta_col] > 0
#     tmp = tmp.sort_values(time_col)
#
#     # 按天聚合偏低发生率，观察是否有明显的时间段聚集
#     tmp['date'] = tmp[time_col].dt.date
#     daily = tmp.groupby('date').agg(
#         样本数=('is_low', 'size'),
#         偏低发生率=('is_low', 'mean')
#     ).reset_index()
#     daily['偏低发生率'] = (daily['偏低发生率'] * 100).round(2)
#     return daily
#
#
# # ==========================================
# # 6. 可视化
# # ==========================================
# def plot_low_reading_diagnostics(df, low_df, rest_df, surface='Top', bucket_summary=None, daily_trend=None):
#     prefix = 'Top' if surface == 'Top' else 'Bot'
#     surface_cn = '上' if surface == 'Top' else '下'
#
#     fig, axes = plt.subplots(2, 2, figsize=(14, 10))
#
#     # (1) 速度分布对比
#     ax = axes[0, 0]
#     sns.kdeplot(low_df['Speed[m/min]_Process_Avg'].dropna(), ax=ax, label='偏低组', color='red', fill=True, alpha=0.3)
#     sns.kdeplot(rest_df['Speed[m/min]_Process_Avg'].dropna(), ax=ax, label='其余组', color='blue', fill=True, alpha=0.3)
#     ax.set_title(f'{surface_cn}表面：速度分布对比')
#     ax.set_xlabel('Speed[m/min]')
#     ax.legend()
#
#     # (2) 电流分布对比
#     ax = axes[0, 1]
#     current_col = f'{prefix}_Current_Sum'
#     sns.kdeplot(low_df[current_col].dropna(), ax=ax, label='偏低组', color='red', fill=True, alpha=0.3)
#     sns.kdeplot(rest_df[current_col].dropna(), ax=ax, label='其余组', color='blue', fill=True, alpha=0.3)
#     ax.set_title(f'{surface_cn}表面：{prefix}_Current_Sum 分布对比')
#     ax.legend()
#
#     # (3) 速度分箱偏低发生率
#     ax = axes[1, 0]
#     if bucket_summary is not None and len(bucket_summary) > 0:
#         ax.bar(range(len(bucket_summary)), bucket_summary['偏低发生率'], color='orange')
#         ax.set_xticks(range(len(bucket_summary)))
#         ax.set_xticklabels([str(b) for b in bucket_summary['bucket']], rotation=45, ha='right', fontsize=8)
#         ax.set_ylabel('偏低发生率(%)')
#         ax.set_title(f'{surface_cn}表面：不同速度区间的偏低发生率')
#     else:
#         ax.text(0.5, 0.5, '无速度分箱数据', ha='center', va='center')
#
#     # (4) 时间趋势
#     ax = axes[1, 1]
#     if daily_trend is not None and len(daily_trend) > 0:
#         ax.plot(daily_trend['date'], daily_trend['偏低发生率'], marker='o', markersize=3, color='green')
#         ax.set_ylabel('当日偏低发生率(%)')
#         ax.set_title(f'{surface_cn}表面：偏低发生率的时间趋势')
#         ax.tick_params(axis='x', rotation=45)
#     else:
#         ax.text(0.5, 0.5, '无有效时间列', ha='center', va='center')
#
#     plt.tight_layout()
#     save_path = f"result/low_reading_analysis/low_reading_diagnostics_{surface}.png"
#     plt.savefig(save_path, dpi=300)
#     print(f"[图表保存] {surface_cn}表面偏低样本诊断图已保存至: {save_path}")
#     plt.show()
#
#
# # ==========================================
# # 7. 主流程：对 Top / Bot 分别执行完整分析
# # ==========================================
# def run_low_reading_analysis(df, surface='Top'):
#     surface_cn = '上' if surface == 'Top' else '下'
#     delta_col = f'{surface}_Delta'
#
#     low_df, rest_df = extract_low_reading_subset(df, surface=surface)
#
#     print(f"\n==========================================")
#     print(f"     【{surface_cn}表面 - 在线偏低样本诊断】")
#     print(f"==========================================")
#     print(f"偏低样本数: {len(low_df)} / 总样本数: {len(df)} (占比 {len(low_df)/len(df)*100:.2f}%)")
#
#     # 导出偏低样本明细，方便人工复核具体的 Coil ID
#     export_cols = [c for c in [
#         'Coil ID', 'Steel Grade', 'Produce Time',
#         'Dimension_[mm]_Thickness', 'Dimension_[mm]_Width',
#         'Speed[m/min]_Process_Avg',
#         f'{surface}_Current_Sum' if surface == 'Bot' else 'Top_Current_Sum',
#         f'{surface}_Theoretical_Factor' if surface == 'Bot' else 'Top_Theoretical_Factor',
#         f'Tin Weight_Actual[g/m2]_GALV_WEIGHT_{surface.upper()}_Avg',
#         f'{"上" if surface=="Top" else "下"}表面镀层重量A(XA1_0)',
#         delta_col
#     ] if c in low_df.columns]
#     low_export_path = f"result/low_reading_analysis/low_reading_samples_{surface}.xlsx"
#     low_df[export_cols].sort_values(delta_col, ascending=False).to_excel(low_export_path, index=False)
#     print(f"[导出提示] 偏低样本明细已保存至: {low_export_path}")
#
#     # 数值特征对比
#     numeric_compare = compare_numeric_features(low_df, rest_df, surface=surface)
#     print(f"\n---- 数值特征对比 (按 p_value 从小到大排序，越小说明该特征在两组间差异越显著) ----")
#     print(numeric_compare.to_string(index=False))
#     numeric_compare.to_excel(f"result/low_reading_analysis/numeric_compare_{surface}.xlsx", index=False)
#
#     # 钢种分布对比
#     grade_compare = compare_categorical_features(low_df, rest_df, col='Steel Grade')
#     if grade_compare is not None:
#         print(f"\n---- 钢种分布对比 (占比差 > 0 表示该钢种在偏低组里更常见) ----")
#         print(grade_compare.head(10).to_string())
#         grade_compare.to_excel(f"result/low_reading_analysis/grade_compare_{surface}.xlsx")
#
#     # 速度分箱发生率
#     bucket_summary = analyze_bucket_incidence(df, surface=surface, bucket_col='Speed[m/min]_Process_Avg')
#     print(f"\n---- 不同速度区间的偏低发生率 ----")
#     print(bucket_summary.to_string(index=False))
#
#     # 厚度、宽度分箱发生率（如果想看规格是否有关系）
#     for dim_col in ['Dimension_[mm]_Thickness', 'Dimension_[mm]_Width']:
#         if dim_col in df.columns:
#             dim_bucket = analyze_bucket_incidence(df, surface=surface, bucket_col=dim_col, n_bins=6)
#             print(f"\n---- 不同{dim_col}区间的偏低发生率 ----")
#             print(dim_bucket.to_string(index=False))
#
#     # 时间趋势
#     daily_trend = analyze_time_trend(df, surface=surface, time_col='Produce Time')
#     if daily_trend is not None:
#         print(f"\n---- 偏低发生率的每日趋势（前10行预览） ----")
#         print(daily_trend.head(10).to_string(index=False))
#         daily_trend.to_excel(f"result/low_reading_analysis/daily_trend_{surface}.xlsx", index=False)
#
#     # 可视化
#     plot_low_reading_diagnostics(df, low_df, rest_df, surface=surface,
#                                   bucket_summary=bucket_summary, daily_trend=daily_trend)
#
#     print(f"==========================================\n")
#
#     return low_df, rest_df, numeric_compare, grade_compare, bucket_summary, daily_trend
#
#
# if __name__ == "__main__":
#     # 直接读取已经清洗好的干净数据集
#     clean_df = pd.read_excel("result/cleaned_data/cleaned_data.xlsx")
#
#     run_low_reading_analysis(clean_df, surface='Top')
#     run_low_reading_analysis(clean_df, surface='Bot')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = "result/low_reading_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOP_SETPOINT_COL = 'Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_TOP_Min'
BOT_SETPOINT_COL = 'Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_BOT_Min'


# ==========================================
# 0. 规格分组键构建 (带浮点数容错)
# ==========================================
def build_setpoint_group_key(df, top_col=TOP_SETPOINT_COL, bot_col=BOT_SETPOINT_COL):
    """
    直接使用原数据中的原始规格数值，不做任何格式化或取整处理。
    """
    if top_col not in df.columns or bot_col not in df.columns:
        raise KeyError(f"缺少分组所需字段: {top_col} 或 {bot_col}")

    df = df.copy()
    # 直接拼接到字符串，完全保留原数据的实际数字
    df['Setpoint_Group_Label'] = df.apply(
        lambda r: f"Top_{r[top_col]}_Bot_{r[bot_col]}"
        if pd.notnull(r[top_col]) and pd.notnull(r[bot_col]) else "Unknown",
        axis=1
    )
    return df


# ==========================================
# 1. 抽取"在线偏低"子集 (Delta > 0)
# ==========================================
def extract_low_reading_subset(df, surface='Top'):
    delta_col = f'{surface}_Delta'
    low_mask = df[delta_col] > 0
    low_df = df[low_mask].copy()
    rest_df = df[~low_mask].copy()
    return low_df, rest_df


# ==========================================
# 2. 数值特征对比 (Mann-Whitney U)
# ==========================================
def compare_numeric_features(low_df, rest_df, surface='Top'):
    prefix = 'Top' if surface == 'Top' else 'Bot'

    candidate_cols = [
        'Dimension_[mm]_Thickness', 'Dimension_[mm]_Width', 'Dimension_[mm]_Length',
        'Speed[m/min]_Process_Avg',
        f'{prefix}_Current_Sum', f'{prefix}_Theoretical_Factor',
        'Steel_Grade_Encoded',
        f'Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_{prefix.upper()}_Avg',
        f'Tin Weight_Actual[g/m2]_GALV_WEIGHT_{prefix.upper()}_Avg',
    ]
    candidate_cols = [c for c in candidate_cols if c in low_df.columns]

    rows = []
    for col in candidate_cols:
        low_vals = low_df[col].dropna()
        rest_vals = rest_df[col].dropna()
        if len(low_vals) < 3 or len(rest_vals) < 3:
            continue

        try:
            stat, p_value = stats.mannwhitneyu(low_vals, rest_vals, alternative='two-sided')
        except ValueError:
            p_value = np.nan

        rows.append({
            '特征': col,
            '偏低组_均值': low_vals.mean(),
            '偏低组_中位数': low_vals.median(),
            '偏低组_标准差': low_vals.std(),
            '其余组_均值': rest_vals.mean(),
            '其余组_中位数': rest_vals.median(),
            '其余组_标准差': rest_vals.std(),
            '均值差(偏低-其余)': low_vals.mean() - rest_vals.mean(),
            'p_value': p_value,
            '显著性(p<0.05)': 'Yes' if (not np.isnan(p_value) and p_value < 0.05) else 'No'
        })

    result_df = pd.DataFrame(rows)
    if not result_df.empty and 'p_value' in result_df.columns:
        result_df = result_df.sort_values('p_value')
    return result_df


# ==========================================
# 3. 分箱分析
# ==========================================
def analyze_bucket_incidence(df, surface='Top', bucket_col='Speed[m/min]_Process_Avg', n_bins=5):
    delta_col = f'{surface}_Delta'
    tmp = df[[bucket_col, delta_col]].dropna().copy()
    if len(tmp) < n_bins * 2:
        return pd.DataFrame()

    tmp['is_low'] = tmp[delta_col] > 0

    try:
        tmp['bucket'] = pd.qcut(tmp[bucket_col], q=n_bins, duplicates='drop')
    except ValueError:
        tmp['bucket'] = pd.cut(tmp[bucket_col], bins=n_bins)

    summary = tmp.groupby('bucket', observed=True).agg(
        样本数=('is_low', 'size'),
        偏低发生率=('is_low', 'mean')
    ).reset_index()
    summary['偏低发生率'] = (summary['偏低发生率'] * 100).round(2)
    return summary


# ==========================================
# 4. 可视化诊断图
# ==========================================
def plot_low_reading_diagnostics(low_df, rest_df, surface='Top', bucket_summary=None,
                                 group_label='All', save_dir=OUTPUT_DIR):
    prefix = 'Top' if surface == 'Top' else 'Bot'
    surface_cn = '上' if surface == 'Top' else '下'

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"[{group_label}] {surface_cn}表面偏低诊断", fontsize=14)

    # (1) 速度分布对比
    ax = axes[0, 0]
    if len(low_df['Speed[m/min]_Process_Avg'].dropna()) > 1 and len(rest_df['Speed[m/min]_Process_Avg'].dropna()) > 1:
        sns.kdeplot(low_df['Speed[m/min]_Process_Avg'].dropna(), ax=ax, label='偏低组', color='red', fill=True,
                    alpha=0.3)
        sns.kdeplot(rest_df['Speed[m/min]_Process_Avg'].dropna(), ax=ax, label='其余组', color='blue', fill=True,
                    alpha=0.3)
        ax.set_title(f'{surface_cn}表面：速度分布对比')
        ax.set_xlabel('Speed[m/min]')
        ax.legend()
    else:
        ax.text(0.5, 0.5, '样本量过少，无法绘制KDE', ha='center', va='center')

    # (2) 电流分布对比
    ax = axes[0, 1]
    current_col = f'{prefix}_Current_Sum'
    if current_col in low_df.columns and len(low_df[current_col].dropna()) > 1 and len(
            rest_df[current_col].dropna()) > 1:
        sns.kdeplot(low_df[current_col].dropna(), ax=ax, label='偏低组', color='red', fill=True, alpha=0.3)
        sns.kdeplot(rest_df[current_col].dropna(), ax=ax, label='其余组', color='blue', fill=True, alpha=0.3)
        ax.set_title(f'{surface_cn}表面：{prefix}_Current_Sum 分布对比')
        ax.legend()
    else:
        ax.text(0.5, 0.5, '样本量过少或缺少电流数据', ha='center', va='center')

    # (3) 速度分箱发生率
    ax = axes[1, 0]
    if bucket_summary is not None and len(bucket_summary) > 0:
        ax.bar(range(len(bucket_summary)), bucket_summary['偏低发生率'], color='orange')
        ax.set_xticks(range(len(bucket_summary)))
        ax.set_xticklabels([str(b) for b in bucket_summary['bucket']], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('偏低发生率(%)')
        ax.set_title(f'{surface_cn}表面：不同速度区间的偏低发生率')
    else:
        ax.text(0.5, 0.5, '无速度分箱数据', ha='center', va='center')

    # (4) 厚度 vs 宽度 散点图
    ax = axes[1, 1]
    if 'Dimension_[mm]_Thickness' in low_df.columns and 'Dimension_[mm]_Width' in low_df.columns:
        ax.scatter(rest_df['Dimension_[mm]_Thickness'], rest_df['Dimension_[mm]_Width'],
                   c='blue', alpha=0.3, label='其余组', s=15)
        ax.scatter(low_df['Dimension_[mm]_Thickness'], low_df['Dimension_[mm]_Width'],
                   c='red', alpha=0.6, label='偏低组', s=25)
        ax.set_xlabel('Thickness [mm]')
        ax.set_ylabel('Width [mm]')
        ax.set_title(f'{surface_cn}表面：厚度与宽度散点分布')
        ax.legend()
    else:
        ax.text(0.5, 0.5, '缺少规格数据', ha='center', va='center')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"diagnostics_{surface}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()


# ==========================================
# 5. 全局跨规格发生率汇总对比
# ==========================================
def analyze_setpoint_groups_overview(df, surface='Top'):
    delta_col = f'{surface}_Delta'

    overview = df.groupby('Setpoint_Group_Label', observed=True).agg(
        总样本数=(delta_col, 'count'),
        偏低样本数=(delta_col, lambda x: (x > 0).sum()),
        偏低发生率=(delta_col, lambda x: ((x > 0).mean() * 100).round(2))
    ).reset_index().sort_values('总样本数', ascending=False)

    return overview


def plot_setpoint_groups_comparison(overview_top, overview_bot, save_dir=OUTPUT_DIR):
    """绘制跨厚度规格的偏低发生率对比柱状图"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Top
    ax = axes[0]
    top_plot = overview_top[overview_top['总样本数'] >= 10]  # 仅展示样本量>=10的规格
    ax.barh(top_plot['Setpoint_Group_Label'], top_plot['偏低发生率'], color='crimson', alpha=0.8)
    ax.set_xlabel('偏低发生率 (%)')
    ax.set_title('上表面：各规格偏低发生率对比 (样本数>=10)')
    ax.invert_yaxis()

    # Bot
    ax = axes[1]
    bot_plot = overview_bot[overview_bot['总样本数'] >= 10]
    ax.barh(bot_plot['Setpoint_Group_Label'], bot_plot['偏低发生率'], color='royalblue', alpha=0.8)
    ax.set_xlabel('偏低发生率 (%)')
    ax.set_title('下表面：各规格偏低发生率对比 (样本数>=10)')
    ax.invert_yaxis()

    plt.tight_layout()
    save_path = os.path.join(save_dir, "setpoint_groups_comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[图表保存] 跨规格对比图已保存至: {save_path}")


def plot_delta_by_index(df, surface='Top', group_label='All', save_dir=OUTPUT_DIR):
    """
    绘制残差(Delta = 真实值 - 在线值) 随样本编号变化的趋势图。
    红色点代表偏低样本 (Delta > 0)，蓝色点代表正常/偏高样本 (Delta <= 0)。
    """
    delta_col = f'{surface}_Delta'
    surface_cn = '上' if surface == 'Top' else '下'

    if delta_col not in df.columns:
        return

    tmp = df.copy()
    # 如果有 Coil ID 优先用 Coil ID 做横坐标，没有就用自增序号 (0, 1, 2...)
    x_col = 'Coil ID' if 'Coil ID' in tmp.columns else 'Index'
    if x_col == 'Index':
        tmp['Index'] = range(1, len(tmp) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))

    # 分别画出偏低组与非偏低组
    low_mask = tmp[delta_col] > 0

    # 绘制正常/偏高点 (Delta <= 0)
    ax.scatter(tmp.loc[~low_mask, x_col], tmp.loc[~low_mask, delta_col],
               color='blue', alpha=0.5, label='正常/偏高 (Delta <= 0)', s=20)

    # 绘制偏低点 (Delta > 0)
    ax.scatter(tmp.loc[low_mask, x_col], tmp.loc[low_mask, delta_col],
               color='red', alpha=0.8, label='偏低 (Delta > 0)', s=30)

    # 绘制 0 基准线
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)

    ax.set_title(f"[{group_label}] {surface_cn}表面：残差(Delta)随编号变化趋势", fontsize=12)
    ax.set_xlabel(x_col)
    ax.set_ylabel(f'{surface}_Delta (真实值 - 在线值)')
    ax.legend(loc='upper right')

    if x_col == 'Coil ID' and len(tmp) > 30:
        plt.xticks(rotation=90, fontsize=6)  # 如果 Coil ID 较多调整字号

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"delta_trend_{surface}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()




# ==========================================
# 6. 主流程：分组执行分析
# ==========================================
def run_grouped_low_reading_analysis(df, min_samples=10):
    # 1. 构建分组标签
    df_tagged = build_setpoint_group_key(df)

    # 2. 生成全数据集跨规格概览统计
    overview_top = analyze_setpoint_groups_overview(df_tagged, surface='Top')
    overview_bot = analyze_setpoint_groups_overview(df_tagged, surface='Bot')

    overview_top.to_excel(os.path.join(OUTPUT_DIR, "setpoint_overview_Top.xlsx"), index=False)
    overview_bot.to_excel(os.path.join(OUTPUT_DIR, "setpoint_overview_Bot.xlsx"), index=False)

    plot_setpoint_groups_comparison(overview_top, overview_bot, save_dir=OUTPUT_DIR)

    print("\n==========================================")
    print("      【规格总体偏低发生率概览 (Top)】")
    print("==========================================")
    print(overview_top.to_string(index=False))

    # 3. 按规格循环深入诊断
    groups = df_tagged['Setpoint_Group_Label'].unique()

    for group_label in groups:
        group_df = df_tagged[df_tagged['Setpoint_Group_Label'] == group_label]

        if len(group_df) < min_samples:
            print(f"\n[跳过规格] {group_label}: 样本数 {len(group_df)} < {min_samples}，不进行组内深入检验。")
            continue

        print(f"\n>>>> 正在分析规格组: {group_label} (总样本数: {len(group_df)}) <<<<")

        # 创建规格专属文件夹
        group_dir = os.path.join(OUTPUT_DIR, "by_setpoint", group_label)
        os.makedirs(group_dir, exist_ok=True)

        for surface in ['Top', 'Bot']:
            low_df, rest_df = extract_low_reading_subset(group_df, surface=surface)

            # ----------------------------------------------------
            # 【需求 1】将当前组中偏低的行单独抽出导出为 xlsx
            # ----------------------------------------------------
            if len(low_df) > 0:
                low_excel_path = os.path.join(group_dir, f"low_samples_{surface}.xlsx")
                # 按照 Delta 降序排列导出
                low_df.sort_values(f'{surface}_Delta', ascending=False).to_excel(low_excel_path, index=False)

            # ----------------------------------------------------
            # 【需求 2】绘制该组残差(Delta)随编号变化的图像
            # ----------------------------------------------------
            plot_delta_by_index(group_df, surface=surface, group_label=group_label, save_dir=group_dir)

            # 原有的样本数太少保护逻辑
            if len(low_df) < 3:
                print(f"  - [{surface}面] 偏低样本过少 ({len(low_df)}个)，跳过后续深入检验。")
                continue

            # 数值检验
            numeric_compare = compare_numeric_features(low_df, rest_df, surface=surface)
            if not numeric_compare.empty:
                numeric_compare.to_excel(os.path.join(group_dir, f"numeric_compare_{surface}.xlsx"), index=False)

            # 速度分箱
            bucket_summary = analyze_bucket_incidence(group_df, surface=surface, bucket_col='Speed[m/min]_Process_Avg')

            # 画图
            plot_low_reading_diagnostics(
                low_df, rest_df, surface=surface,
                bucket_summary=bucket_summary,
                group_label=group_label,
                save_dir=group_dir
            )

    print("\n分析完全结束！所有导出文件和图表已归档至:", OUTPUT_DIR)


if __name__ == "__main__":
    clean_df = pd.read_excel("result/cleaned_data/cleaned_data.xlsx")
    run_grouped_low_reading_analysis(clean_df, min_samples=200)