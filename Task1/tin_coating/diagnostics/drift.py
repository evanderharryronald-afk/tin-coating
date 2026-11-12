# residual_drift_diagnosis.py
# 残差时间漂移诊断（最小可行版本）
# 功能：
#   1. 整体 + 各规格组
#   2. Top / Bot 表面
#   3. 原始残差滑动均值 & 滑动标准差
#   4. 前80% vs 后20% 的 KS 检验 + 均值差
#   5. 出图 + 汇总表

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ====================== 配置区 ======================
CLEAN_DATA_PATH = "result/data/cleaned_data/cleaned_data.xlsx"
OUTPUT_DIR = "result/drift_diagnosis/grouped_drift_analysis"
MIN_SAMPLES = 200          # 组内至少这么多点才做分析
WINDOW_SIZE = 50           # 滑动窗口大小（可按数据量调整 50~120）
TRAIN_RATIO = 0.8          # 前 80% vs 后 20%

TOP_SETPOINT_COL = 'Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_TOP_Min'
BOT_SETPOINT_COL = 'Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_BOT_Min'

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# ====================================================


def build_setpoint_group_key(df):
    df = df.copy()
    df['Setpoint_Group_Key'] = list(zip(df[TOP_SETPOINT_COL], df[BOT_SETPOINT_COL]))
    df['Setpoint_Group_Label'] = df.apply(
        lambda r: f"Top{r[TOP_SETPOINT_COL]}_Bot{r[BOT_SETPOINT_COL]}", axis=1
    )
    return df


def get_delta_col(surface):
    return 'Top_Delta' if surface == 'Top' else 'Bot_Delta'


def sliding_stats(series, window):
    """返回滑动均值和滑动标准差（与原序列等长，边缘用 min_periods）"""
    s = pd.Series(series)
    mean = s.rolling(window=window, min_periods=max(10, window // 4)).mean()
    std = s.rolling(window=window, min_periods=max(10, window // 4)).std()
    return mean, std


def ks_and_mean_diff(y, ratio=0.8):
    """前 ratio 与后 1-ratio 的 KS 检验 + 均值差"""
    n = len(y)
    if n < 30:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    split = int(n * ratio)
    a = y[:split]
    b = y[split:]

    # 去掉 nan
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 10 or len(b) < 10:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    ks_stat, ks_p = stats.ks_2samp(a, b)
    mean_diff = np.mean(b) - np.mean(a)          # 后 - 前
    std_ratio = np.std(b) / (np.std(a) + 1e-8)   # 后 / 前

    return ks_stat, ks_p, mean_diff, std_ratio, len(b)


def analyze_one(df, surface, group_label, save_dir):
    """对单个（组 + 表面）做完整分析，返回一行指标，并保存图"""
    delta_col = get_delta_col(surface)
    if delta_col not in df.columns:
        return None

    y = df[delta_col].values.astype(float)
    valid_mask = ~np.isnan(y)
    y = y[valid_mask]
    n = len(y)

    if n < MIN_SAMPLES:
        return None

    # ----- 滑动统计 -----
    roll_mean, roll_std = sliding_stats(y, WINDOW_SIZE)

    # ----- 前80% vs 后20% -----
    ks_stat, ks_p, mean_diff, std_ratio, n_test = ks_and_mean_diff(y, TRAIN_RATIO)

    # ----- 画图 -----
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                             gridspec_kw={'height_ratios': [2, 1]})

    # 上图：原始残差 + 滑动均值
    axes[0].plot(y, color='gray', alpha=0.45, linewidth=0.8, label='原始残差')
    axes[0].plot(roll_mean, color='C0', linewidth=1.8, label=f'滑动均值 (window={WINDOW_SIZE})')
    axes[0].axhline(0, color='black', linestyle='--', linewidth=1)
    split_idx = int(n * TRAIN_RATIO)
    axes[0].axvline(split_idx, color='red', linestyle='--', alpha=0.7, label='80% 分割线')
    axes[0].set_ylabel('残差 (g/m2)')
    axes[0].set_title(f'[{group_label}] {surface}表面 原始残差随时间变化')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle=':', alpha=0.5)

    # 下图：滑动标准差
    axes[1].plot(roll_std, color='C1', linewidth=1.5, label='滑动标准差')
    axes[1].axvline(split_idx, color='red', linestyle='--', alpha=0.7)
    axes[1].set_xlabel('样本序号（按时间顺序）')
    axes[1].set_ylabel('滑动标准差')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    safe_name = group_label.replace('.', 'p').replace('/', '_')
    img_path = os.path.join(save_dir, f"drift_{safe_name}_{surface}.png")
    plt.savefig(img_path, dpi=200)
    plt.close()

    # ----- 返回指标 -----
    return {
        '规格组': group_label,
        '表面': surface,
        '样本数': n,
        '整体均值': float(np.nanmean(y)),
        '整体标准差': float(np.nanstd(y)),
        '前80%均值': float(np.nanmean(y[:split_idx])),
        '后20%均值': float(np.nanmean(y[split_idx:])),
        '均值差(后-前)': mean_diff,
        '标准差比(后/前)': std_ratio,
        'KS统计量': ks_stat,
        'KS_p值': ks_p,
        '漂移判断': (
            '明显漂移' if (ks_p is not None and ks_p < 0.01 and abs(mean_diff) > 0.05)
            else ('可疑漂移' if (ks_p is not None and ks_p < 0.05) else '较平稳')
        )
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_dir = os.path.join(OUTPUT_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    print("读取数据...")
    df = pd.read_excel(CLEAN_DATA_PATH)
    df = build_setpoint_group_key(df)

    # 确保有 Delta 列（如果清洗阶段已算好就直接用，否则现场算）
    for surface, prefix, cn in [('Top', 'Top', '上'), ('Bot', 'Bot', '下')]:
        delta_col = f'{prefix}_Delta'
        if delta_col not in df.columns:
            online = f'Tin Weight_Actual[g/m2]_GALV_WEIGHT_{prefix.upper()}_Avg'
            lab = f'{cn}表面镀层重量A(XA1_0)'
            if online in df.columns and lab in df.columns:
                df[delta_col] = df[lab] - df[online]
            else:
                print(f"[警告] 无法构建 {delta_col}，请检查列名")

    results = []

    # ---------- 1. 整体（不分规格） ----------
    print("\n===== 分析整体数据 =====")
    for surface in ['Top', 'Bot']:
        metrics = analyze_one(df, surface, group_label="整体", save_dir=plot_dir)
        if metrics:
            results.append(metrics)
            print(f"  {surface}: 均值差={metrics['均值差(后-前)']:.4f}, "
                  f"KS_p={metrics['KS_p值']:.4g}, 判断={metrics['漂移判断']}")

    # ---------- 2. 各规格组 ----------
    group_sizes = df.groupby('Setpoint_Group_Label').size().sort_values(ascending=False)
    print(f"\n===== 分析各规格组（样本量 >= {MIN_SAMPLES}）=====")

    for group_label, size in group_sizes.items():
        if size < MIN_SAMPLES:
            print(f"  [跳过] {group_label} 样本数 {size}")
            continue

        group_df = df[df['Setpoint_Group_Label'] == group_label].copy()
        # 保持原有时间顺序（不要重新排序）
        print(f"\n  >> {group_label} (n={size})")

        for surface in ['Top', 'Bot']:
            metrics = analyze_one(group_df, surface, group_label, save_dir=plot_dir)
            if metrics:
                results.append(metrics)
                print(f"     {surface}: 均值差={metrics['均值差(后-前)']:.4f}, "
                      f"KS_p={metrics['KS_p值']:.4g}, 判断={metrics['漂移判断']}")

    # ---------- 3. 导出汇总表 ----------
    if not results:
        print("没有足够样本的组可分析")
        return

    res_df = pd.DataFrame(results)
    # 按漂移严重程度大致排序（先按判断，再按 |均值差|）
    order_map = {'明显漂移': 0, '可疑漂移': 1, '较平稳': 2}
    res_df['_sort'] = res_df['漂移判断'].map(order_map)
    res_df = res_df.sort_values(['_sort', '均值差(后-前)'], key=lambda s: s.abs() if s.name == '均值差(后-前)' else s)
    res_df = res_df.drop(columns='_sort')

    excel_path = os.path.join(OUTPUT_DIR, "residual_drift_summary.xlsx")
    res_df.to_excel(excel_path, index=False)
    print(f"\n[完成] 汇总表已保存: {excel_path}")
    print(f"[完成] 图片已保存到: {plot_dir}")
    print("\n===== 漂移判断汇总 =====")
    print(res_df['漂移判断'].value_counts())
    print("\n前几行预览:")
    print(res_df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()