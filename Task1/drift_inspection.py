import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs("result/drift_inspection", exist_ok=True)

TOP_SETPOINT_COL = 'Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_TOP_Min'
BOT_SETPOINT_COL = 'Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_BOT_Min'

CLUSTER_GAP_THRESHOLD = 20  # 相邻"在线偏低"样本原始行号间隔小于此值，视为时间上聚集

def build_setpoint_group_key(df, top_col=TOP_SETPOINT_COL, bot_col=BOT_SETPOINT_COL):
    """与 coating_model_by_group.py 中的逻辑保持一致，用精确值组合生成分组标签"""
    df = df.copy()
    df['Setpoint_Group_Label'] = df.apply(
        lambda r: f"Top{r[top_col]}_Bot{r[bot_col]}", axis=1
    )
    return df


def plot_drift_for_group(df, group_label, surface='Top', rolling_window=20):
    """
    对指定规格组、指定表面，画原始残差随时间（原始行号）变化的诊断图，
    用于排查在线仪表是否存在系统性漂移。
    不做 train/test 切分，直接看该规格组从头到尾的全部数据。
    """
    prefix = 'Top' if surface == 'Top' else 'Bot'
    surface_cn = '上' if surface == 'Top' else '下'

    online_col = f'Tin Weight_Actual[g/m2]_GALV_WEIGHT_{prefix.upper()}_Avg'
    lab_col = f'{surface_cn}表面镀层重量A(XA1_0)'
    delta_col = f'{prefix}_Delta'  # 已在预处理阶段算好: lab - online

    group_df = df[df['Setpoint_Group_Label'] == group_label].copy()
    group_df = group_df.sort_index()  # 保持原始行号顺序，即按时间

    if group_df.empty:
        print(f"[跳过] 规格组 {group_label} 无数据")
        return

    if delta_col not in group_df.columns or online_col not in group_df.columns:
        print(f"[跳过] 缺少必要字段: {delta_col} / {online_col}")
        return

    n = len(group_df)
    rolling_mean = group_df[delta_col].rolling(window=min(rolling_window, max(n // 5, 1)),
                                                 min_periods=1).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={'height_ratios': [1.2, 1]})

    # 上图：在线值 vs 实验室真实值 随时间(原始行号)对比
    ax1.plot(group_df.index, group_df[lab_col], label='实验室真实值', color='black', linewidth=1.2)
    ax1.plot(group_df.index, group_df[online_col], label='在线仪表值', color='red',
              linestyle='--', alpha=0.7)
    ax1.set_title(f'[{group_label}] {surface_cn}表面 在线值 vs 真实值 随时间变化 (样本数={n})')
    ax1.set_xlabel('原始数据行号 (时间顺序)')
    ax1.set_ylabel('镀层重量 (g/m2)')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    # 下图：原始残差 (真实-在线) 及其滑动均值，重点看是否偏离0且不回归
    ax2.scatter(group_df.index, group_df[delta_col], color='gray', alpha=0.4, s=10, label='原始残差 (真实-在线)')
    ax2.plot(group_df.index, rolling_mean, color='blue', linewidth=1.8,
              label=f'残差滑动均值 (window={min(rolling_window, max(n // 5, 1))})')
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.set_title(f'[{group_label}] {surface_cn}表面 残差趋势（滑动均值持续偏离0即为系统性漂移信号）')
    ax2.set_xlabel('原始数据行号 (时间顺序)')
    ax2.set_ylabel('残差 (g/m2)')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    safe_label = group_label.replace('.', '_')
    save_path = f"result/drift_inspection/drift_{safe_label}_{surface}.png"
    plt.savefig(save_path, dpi=300)
    print(f"[图表保存] {save_path}")
    plt.close()


def inspect_groups(df, group_labels=None, surfaces=('Top', 'Bot'), rolling_window=20):
    """
    通用入口：
    - group_labels: 指定要排查的规格组列表；不传则默认排查数据中所有规格组
    - surfaces: 默认 Top、Bot 都画，可以只传一个
    """
    df = build_setpoint_group_key(df)

    if group_labels is None:
        group_labels = df['Setpoint_Group_Label'].unique().tolist()

    for label in group_labels:
        for surface in surfaces:
            plot_drift_for_group(df, group_label=label, surface=surface, rolling_window=rolling_window)


MIN_GROUP_SAMPLES_FOR_DRIFT_CHECK = 200
EARLY_RATIO = 0.7  # 前70%当作基线，后30%作为重点排查区间
DRIFT_SCORE_THRESHOLD = 1.0  # |后段均值-前段均值| / 前段标准差
DIRECTION_CONSISTENCY_THRESHOLD = 0.25  # 前后段"同向占比"之差，超过此值视为方向从不稳定变为高度一致


def compute_drift_metrics(df, group_label, surface):
    """
    对指定规格组、指定表面，计算系统性漂移相关指标。
    规则：把该组数据按时间顺序切成前70%（基线）/ 后30%（排查区间），
    从"偏移幅度"和"持续性"两个维度判断，避免单看均值被个别异常点带偏。
    """
    prefix = 'Top' if surface == 'Top' else 'Bot'
    delta_col = f'{prefix}_Delta'

    group_df = df[df['Setpoint_Group_Label'] == group_label].copy()
    group_df = group_df.sort_index()
    n = len(group_df)

    if n < MIN_GROUP_SAMPLES_FOR_DRIFT_CHECK or delta_col not in group_df.columns:
        return None

    split_idx = int(n * EARLY_RATIO)
    early = group_df[delta_col].iloc[:split_idx].dropna()
    late = group_df[delta_col].iloc[split_idx:].dropna()

    if len(early) < 10 or len(late) < 10:
        return None

    early_mean, early_std = early.mean(), early.std()
    late_mean = late.mean()

    if early_std == 0 or np.isnan(early_std):
        drift_score = np.nan
    else:
        drift_score = abs(late_mean - early_mean) / early_std

    late_sign = np.sign(late_mean)
    early_sign = np.sign(early_mean)

    # 后段同向占比：后段样本里，与"后段均值方向"一致的比例
    late_direction_consistency = (np.sign(late) == late_sign).mean() if late_sign != 0 else np.nan
    # 前段同向占比：前段样本里，与"后段均值方向"一致的比例（用同一个方向去衡量，才有可比性）
    early_direction_consistency = (np.sign(early) == late_sign).mean() if late_sign != 0 else np.nan

    if np.isnan(late_direction_consistency) or np.isnan(early_direction_consistency):
        direction_consistency_gap = np.nan
    else:
        direction_consistency_gap = late_direction_consistency - early_direction_consistency

    # 判定：偏移幅度要大，且方向一致性要比基线阶段明显收敛（而不是本来就一直偏一个方向）
    is_drift = (
            not np.isnan(drift_score) and drift_score > DRIFT_SCORE_THRESHOLD
            and not np.isnan(direction_consistency_gap) and direction_consistency_gap > DIRECTION_CONSISTENCY_THRESHOLD
    )

    return {
        '规格组': group_label,
        '表面': surface,
        '总样本数': n,
        '基线样本数(前70%)': len(early),
        '排查样本数(后30%)': len(late),
        '基线残差均值': early_mean,
        '基线残差标准差': early_std,
        '后段残差均值': late_mean,
        '漂移得分(drift_score)': drift_score,
        '前段同向占比': early_direction_consistency,
        '后段同向占比': late_direction_consistency,
        '同向占比差值': direction_consistency_gap,
        '系统性漂移嫌疑': '是' if is_drift else '否',
    }


def analyze_low_online_samples(df, group_label, surface):
    """
    对指定规格组、指定表面，专门分析 Delta>0（在线测量偏低）样本的特点，
    并与 Delta<0（在线偏高，占多数的正常方向）样本做对比。
    """
    prefix = 'Top' if surface == 'Top' else 'Bot'
    delta_col = f'{prefix}_Delta'

    group_df = df[df['Setpoint_Group_Label'] == group_label].copy().sort_index()
    if delta_col not in group_df.columns or group_df.empty:
        return None

    pos = group_df[group_df[delta_col] > 0]  # 在线偏低
    neg = group_df[group_df[delta_col] < 0]  # 在线偏高（多数方向）

    if len(pos) == 0:
        return None

    # 幅度对比：偏低样本的Delta绝对值 vs 偏高样本的Delta绝对值
    pos_abs_mean = pos[delta_col].abs().mean()
    pos_abs_median = pos[delta_col].abs().median()
    neg_abs_mean = neg[delta_col].abs().mean() if len(neg) > 0 else np.nan
    neg_abs_median = neg[delta_col].abs().median() if len(neg) > 0 else np.nan

    # 时间聚集程度：偏低样本按原始行号排序后，相邻间隔的中位数越小，越聚集
    pos_index_sorted = pos.index.to_series().sort_values()
    gaps = pos_index_sorted.diff().dropna()
    median_gap = gaps.median() if len(gaps) > 0 else np.nan
    cluster_ratio = (gaps < CLUSTER_GAP_THRESHOLD).mean() if len(gaps) > 0 else np.nan

    result = {
        '规格组': group_label,
        '表面': surface,
        '组内总样本数': len(group_df),
        '偏低样本数(Delta>0)': len(pos),
        '偏低样本占比': len(pos) / len(group_df),
        '偏低Delta绝对值均值': pos_abs_mean,
        '偏低Delta绝对值中位数': pos_abs_median,
        '偏高Delta绝对值均值(对照)': neg_abs_mean,
        '偏高Delta绝对值中位数(对照)': neg_abs_median,
        '幅度比值(偏低/偏高_中位数)': pos_abs_median / neg_abs_median if neg_abs_median else np.nan,
        '偏低样本行号间隔中位数': median_gap,
        '偏低样本聚集占比': cluster_ratio,
        '疑似聚集(非随机)': '是' if (not np.isnan(cluster_ratio) and cluster_ratio > 0.5) else '否',
    }

    # 关联维度：Steel Grade 在偏低样本里的分布，和整组分布做对比（找出是否有某钢种明显过度集中）
    if 'Steel Grade' in group_df.columns:
        overall_dist = group_df['Steel Grade'].value_counts(normalize=True)
        pos_dist = pos['Steel Grade'].value_counts(normalize=True)
        top_grade = pos_dist.index[0] if len(pos_dist) > 0 else None
        if top_grade is not None:
            result['偏低样本中占比最高的钢种'] = top_grade
            result['该钢种在偏低样本中占比'] = pos_dist.iloc[0]
            result['该钢种在整组中占比(对照)'] = overall_dist.get(top_grade, np.nan)

    return result, pos


def generate_low_online_report(df, surfaces=('Top', 'Bot'),
                               min_group_samples=MIN_GROUP_SAMPLES_FOR_DRIFT_CHECK,
                               save_path="result/drift_inspection/low_online_report.xlsx"):
    """
    批量扫描所有达标规格组，分析"在线偏低"样本特点，导出汇总表 + 明细表。
    """
    df = build_setpoint_group_key(df)
    group_sizes = df.groupby('Setpoint_Group_Label').size()
    valid_labels = group_sizes[group_sizes >= min_group_samples].index.tolist()

    summary_rows = []
    detail_frames = []

    for label in valid_labels:
        for surface in surfaces:
            out = analyze_low_online_samples(df, group_label=label, surface=surface)
            if out is None:
                continue
            result, pos_samples = out
            summary_rows.append(result)

            detail_cols = ['Setpoint_Group_Label']
            prefix = 'Top' if surface == 'Top' else 'Bot'
            for c in ['Coil ID', 'Steel Grade', f'{prefix}_Delta',
                      'Speed[m/min]_Process_Avg', 'Dimension_[mm]_Thickness']:
                if c in pos_samples.columns:
                    detail_cols.append(c)
            detail = pos_samples[detail_cols].copy()
            detail.insert(1, '表面', surface)
            detail_frames.append(detail)

    if not summary_rows:
        print("[提示] 没有找到任何在线偏低样本，未生成报告。")
        return None

    summary_df = pd.DataFrame(summary_rows).sort_values('偏低样本聚集占比', ascending=False)
    detail_df = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='偏低样本汇总', index=False)
        detail_df.to_excel(writer, sheet_name='偏低样本明细', index=False)

    print(f"[导出提示] 在线偏低样本分析报告已保存至: {save_path}")
    return summary_df, detail_df




def generate_drift_report(df, surfaces=('Top', 'Bot'),
                          save_path="result/drift_inspection/drift_report.xlsx",
                          rolling_window=20):
    """
    批量扫描所有样本数达标的规格组，计算漂移指标，导出 Excel 报告，
    并只对被标记为"系统性漂移嫌疑"的组自动画图（避免图片过多）。
    """
    df = build_setpoint_group_key(df)
    group_sizes = df.groupby('Setpoint_Group_Label').size()
    valid_labels = group_sizes[group_sizes >= MIN_GROUP_SAMPLES_FOR_DRIFT_CHECK].index.tolist()

    print(f"共 {len(valid_labels)} 个规格组样本数达标(>= {MIN_GROUP_SAMPLES_FOR_DRIFT_CHECK})，开始逐个扫描...")

    all_metrics = []
    for label in valid_labels:
        for surface in surfaces:
            m = compute_drift_metrics(df, group_label=label, surface=surface)
            if m is not None:
                all_metrics.append(m)

    if not all_metrics:
        print("[提示] 没有符合条件的规格组，未生成报告。")
        return None

    report_df = pd.DataFrame(all_metrics)
    report_df = report_df.sort_values('漂移得分(drift_score)', ascending=False)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
        report_df.to_excel(writer, sheet_name='漂移诊断汇总', index=False)

    print(f"[导出提示] 漂移诊断报告已保存至: {save_path}")

    flagged = report_df[report_df['系统性漂移嫌疑'] == '是']
    print(f"共标记出 {len(flagged)} 组(规格组×表面)存在系统性漂移嫌疑，自动生成对应诊断图...")
    for _, row in flagged.iterrows():
        plot_drift_for_group(df, group_label=row['规格组'], surface=row['表面'], rolling_window=rolling_window)

    return report_df


if __name__ == "__main__":
    clean_df = pd.read_excel("result/cleaned_data/cleaned_data.xlsx")

    # 批量扫描所有样本数>=200的规格组，自动判定漂移并导出报告+对应图表
    generate_drift_report(clean_df, surfaces=("Top", "Bot"))
    generate_low_online_report(clean_df, surfaces=("Top", "Bot"))
    # 如果只想手动看某个组，仍可以单独调用：
    # inspect_groups(clean_df, group_labels=["Top2.799_Bot1.1"], surfaces=("Top", "Bot"))