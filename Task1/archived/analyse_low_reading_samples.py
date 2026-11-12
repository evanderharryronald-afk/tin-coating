import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = "result/low_reading_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOP_SETPOINT_COL = 'Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_TOP_Min'
BOT_SETPOINT_COL = 'Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_BOT_Min'


# ==========================================
# 0. 自动匹配时间列 & 工具函数
# ==========================================
def find_and_parse_time_column(df):
    """自动寻找并解析时间列，防止写死列名导致的跳过绘图问题"""
    possible_time_cols = ['DateTime', 'datetime', 'Time', 'time', 'Timestamp', '生产时间', '日期时间', '日期','Produce Time']
    for col in possible_time_cols:
        if col in df.columns:
            print(f"[时间列匹配成功] 使用列名: '{col}'")
            parsed_series = pd.to_datetime(df[col], errors='coerce')
            if parsed_series.notnull().sum() > 0:
                df['Parsed_DateTime'] = parsed_series
                return df, 'Parsed_DateTime'
    print("[警告] 未匹配到标准时间列，将使用索引模拟时间轴。")
    return df, None


def cliffs_delta(x, y):
    x, y = np.asarray(x), np.asarray(y)
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return np.nan
    u, _ = stats.mannwhitneyu(x, y, alternative='two-sided')
    return (2 * u) / (nx * ny) - 1


def effect_size_label(delta):
    if np.isnan(delta): return "NA"
    ad = abs(delta)
    if ad < 0.147: return "可忽略"
    elif ad < 0.33: return "小"
    elif ad < 0.474: return "中"
    else: return "大"


def build_setpoint_group_key(df, top_col=TOP_SETPOINT_COL, bot_col=BOT_SETPOINT_COL):
    df = df.copy()
    valid_mask = df[top_col].notnull() & df[bot_col].notnull()
    df['Setpoint_Group_Label'] = "Unknown"
    df.loc[valid_mask, 'Setpoint_Group_Label'] = (
        "Top_" + df.loc[valid_mask, top_col].astype(str) +
        "_Bot_" + df.loc[valid_mask, bot_col].astype(str)
    )
    return df


def extract_low_reading_subset(df, surface='Top'):
    delta_col = f'{surface}_Delta'
    low_mask = df[delta_col] > 0
    return df[low_mask].copy(), df[~low_mask].copy()


# ==========================================
# 1. 重构直观图表：残差与偏低时序漂移图 (上下双子图)
# ==========================================
def plot_delta_time_series_intuitive(df, time_col, surface='Top', group_label='Global', save_dir=OUTPUT_DIR):
    delta_col = f'{surface}_Delta'
    surface_cn = '上' if surface == 'Top' else '下'

    if delta_col not in df.columns:
        print(f"  [跳过绘图] 缺少残差列: {delta_col}")
        return

    tmp = df.copy()
    if time_col and time_col in tmp.columns:
        tmp = tmp.dropna(subset=[time_col, delta_col]).sort_values(time_col)
        x_vals = tmp[time_col]
        x_label = "生产时间 (DateTime)"
    else:
        tmp = tmp.dropna(subset=[delta_col]).reset_index(drop=True)
        x_vals = tmp.index
        x_label = "生产流水序号 (Sequence Index)"

    if len(tmp) == 0:
        return

    low_mask = tmp[delta_col] > 0

    # 创建上下两个子图：上图看残差趋势，下图看偏低分布状态
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # --- 上图：残差趋势与零界线 ---
    ax1.plot(x_vals, tmp[delta_col], color='gray', linestyle='-', linewidth=0.8, alpha=0.5, label='残差波动')
    ax1.scatter(x_vals[~low_mask], tmp.loc[~low_mask, delta_col], color='dodgerblue', alpha=0.6, s=20, label='正常/偏高 (Delta ≤ 0)')
    ax1.scatter(x_vals[low_mask], tmp.loc[low_mask, delta_col], color='crimson', alpha=0.9, s=35, label='偏低 (Delta > 0)')
    ax1.axhline(0, color='black', linestyle='--', linewidth=1.2, label='基准线 (Delta=0)')

    ax1.set_title(f"[{group_label}] {surface_cn}表面：残差(Delta = 实际值 - 在线值) 时序漂移图", fontsize=13, fontweight='bold')
    ax1.set_ylabel(f'{surface}_Delta', fontsize=10)
    ax1.legend(loc='upper right', frameon=True)
    ax1.grid(True, linestyle=':', alpha=0.5)

    # --- 下图：状态指示条（红色=偏低，蓝色=正常） ---
    status_colors = ['crimson' if m else 'lightgray' for m in low_mask]
    ax2.bar(x_vals, [1]*len(tmp), color=status_colors, width=1.0 if not time_col else 0.01)
    ax2.set_yticks([])
    ax2.set_ylabel('偏低状态\n密集度', fontsize=9)
    ax2.set_xlabel(x_label, fontsize=10)

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"delta_time_trend_{surface}.png"), dpi=300)
    plt.close()


# ==========================================
# 2. 重构直观图表：按时间窗口（天/小时）统计偏低发生率双轴图
# ==========================================
def plot_time_window_incidence(df, time_col, surface='Top', group_label='Global', save_dir=OUTPUT_DIR):
    if not time_col or time_col not in df.columns:
        return

    delta_col = f'{surface}_Delta'
    surface_cn = '上' if surface == 'Top' else '下'

    tmp = df[[time_col, delta_col]].dropna().copy()
    tmp['is_low'] = (tmp[delta_col] > 0).astype(int)

    time_span = tmp[time_col].max() - tmp[time_col].min()
    rule = 'D' if time_span.days >= 2 else '2h'  # 跨度大按天，跨度小按2小时

    ts_summary = tmp.set_index(time_col).resample(rule).agg(
        总卷数=('is_low', 'count'),
        偏低卷数=('is_low', 'sum')
    ).reset_index()

    ts_summary = ts_summary[ts_summary['总卷数'] > 0]
    if len(ts_summary) < 2:
        return

    ts_summary['偏低发生率(%)'] = (ts_summary['偏低卷数'] / ts_summary['总卷数'] * 100).round(2)

    fig, ax1 = plt.subplots(figsize=(13, 5))

    # 柱状图：生产总卷数
    ax1.bar(ts_summary[time_col], ts_summary['总卷数'], alpha=0.3, color='slategray', label='生产总卷数', width=0.6 if rule=='D' else 0.05)
    ax1.set_ylabel('生产样本卷数', color='slategray', fontsize=10)
    ax1.tick_params(axis='y', labelcolor='slategray')

    # 折线图：偏低发生率
    ax2 = ax1.twinx()
    ax2.plot(ts_summary[time_col], ts_summary['偏低发生率(%)'], marker='o', color='crimson', linewidth=2, label='偏低发生率(%)')
    ax2.set_ylabel('偏低发生率 (%)', color='crimson', fontsize=10)
    ax2.tick_params(axis='y', labelcolor='crimson')
    ax2.set_ylim(-5, 105)

    plt.title(f"[{group_label}] {surface_cn}表面：偏低发生率随时间统计 (窗口粒度: {rule})", fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.4)
    fig.autofmt_xdate()
    fig.tight_layout()

    plt.savefig(os.path.join(save_dir, f"time_window_incidence_{surface}.png"), dpi=300)
    plt.close()


# ==========================================
# 3. 直观参数对比图：直方图与重叠密度分布 (比箱线图更直观)
# ==========================================
def plot_intuitive_feature_hist(low_df, rest_df, surface='Top', group_label='Global', save_dir=OUTPUT_DIR):
    prefix = 'Top' if surface == 'Top' else 'Bot'
    surface_cn = '上' if surface == 'Top' else '下'

    features = [
        ('Speed[m/min]_Process_Avg', '车速 [m/min]'),
        (f'{prefix}_Current_Sum', f'{surface_cn}面电流 Sum'),
        ('Dimension_[mm]_Thickness', '钢卷厚度 [mm]'),
        ('Dimension_[mm]_Width', '钢卷宽度 [mm]')
    ]
    valid_features = [(f, name) for f, name in features if f in low_df.columns]

    if not valid_features:
        return

    fig, axes = plt.subplots(1, len(valid_features), figsize=(4.5 * len(valid_features), 4))
    if len(valid_features) == 1: axes = [axes]

    for i, (col, col_cn) in enumerate(valid_features):
        ax = axes[i]
        sns.histplot(rest_df[col].dropna(), ax=ax, color='dodgerblue', label='正常组', kde=True, stat="density", common_norm=False, alpha=0.4)
        sns.histplot(low_df[col].dropna(), ax=ax, color='crimson', label='偏低组', kde=True, stat="density", common_norm=False, alpha=0.5)
        ax.set_title(col_cn, fontsize=11, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('密度分布')
        ax.legend()

    fig.suptitle(f"[{group_label}] {surface_cn}表面：关键工艺参数分布偏移对比", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"feature_distribution_{surface}.png"), dpi=300)
    plt.close()


# ==========================================
# 4. 执行流程控制
# ==========================================
def execute_single_diagnostic(df, time_col, surface='Top', group_label='Global', target_dir=OUTPUT_DIR):
    low_df, rest_df = extract_low_reading_subset(df, surface=surface)

    # 1. 导出Excel (强制按时间顺序)
    if len(low_df) > 0:
        low_excel_path = os.path.join(target_dir, f"low_samples_{surface}.xlsx")
        sort_cols = [time_col] if time_col and time_col in low_df.columns else []
        (low_df.sort_values(sort_cols) if sort_cols else low_df).to_excel(low_excel_path, index=False)

    # 2. 强行绘制【残差时序漂移图】（不受样本量门槛限制）
    plot_delta_time_series_intuitive(df, time_col, surface=surface, group_label=group_label, save_dir=target_dir)

    # 3. 强行绘制【时间窗口发生率图】
    plot_time_window_incidence(df, time_col, surface=surface, group_label=group_label, save_dir=target_dir)

    # 4. 绘制【直观特征分布对比图】
    if len(low_df) >= 3:
        plot_intuitive_feature_hist(low_df, rest_df, surface=surface, group_label=group_label, save_dir=target_dir)


def run_grouped_low_reading_analysis(df, min_samples_desc=10):
    # 自动检索并转换时间列
    df, time_col = find_and_parse_time_column(df)

    if time_col:
        df = df.sort_values(time_col).reset_index(drop=True)

    df_tagged = build_setpoint_group_key(df)

    # ------------------------------------
    # Step 1: 整体全局（不分组）诊断
    # ------------------------------------
    print("\n>>>> 正在生成【全线不分组总体诊断图】 <<<<")
    global_dir = os.path.join(OUTPUT_DIR, "global_overview")
    os.makedirs(global_dir, exist_ok=True)
    for surface in ['Top', 'Bot']:
        execute_single_diagnostic(df_tagged, time_col, surface=surface, group_label='Global_All', target_dir=global_dir)

    # ------------------------------------
    # Step 2: 按规格分组诊断
    # ------------------------------------
    groups = df_tagged['Setpoint_Group_Label'].unique()
    for group_label in groups:
        group_df = df_tagged[df_tagged['Setpoint_Group_Label'] == group_label]
        n = len(group_df)

        if n < min_samples_desc:
            continue

        print(f">>>> 正在生成规格组诊断图: {group_label} (总卷数: {n}) <<<<")
        group_dir = os.path.join(OUTPUT_DIR, "by_setpoint", group_label)
        os.makedirs(group_dir, exist_ok=True)

        for surface in ['Top', 'Bot']:
            execute_single_diagnostic(group_df, time_col, surface=surface, group_label=group_label, target_dir=group_dir)

    print("\n分析与可视化绘制完全结束！所有图表已输出至:", OUTPUT_DIR)


if __name__ == "__main__":
    featured_df = pd.read_excel("result/data/feature_engineered_data/featured_data.xlsx")
    run_grouped_low_reading_analysis(featured_df, min_samples_desc=200)