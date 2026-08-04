import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 设置画图支持中文与负号
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 2. 配置文件路径与保存目录
excel_path = "result_comparison.xlsx"  # 替换为你的 Excel 文件路径
save_dir = "plots"  # 保存图表的目录
os.makedirs(save_dir, exist_ok=True)


# 数值标注函数（在柱状图顶部标出数值）
def autolabel(ax, rects, precision=4):
    for rect in rects:
        height = rect.get_height()
        if not np.isnan(height):
            ax.annotate(f'{height:.{precision}f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 向上偏移 3 点
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=7)


# 封装通用的绘图函数
def plot_mae_comparison(df, title_name, save_filename):
    """
    根据传入的 DataFrame 绘制 MAE 模型对比柱状图
    包含：MAE_在线、MAE_模型(统一参数)、MAE_模型(分组调参)
    """
    # ---------- 关键：把 "—" 等非数值统一转成 NaN ----------
    mae_cols = ['MAE_在线', 'MAE_模型 (统一参数)', 'MAE_模型 (分组调参)']
    for col in mae_cols:
        if col in df.columns:
            df[col] = df[col].replace(['—', '–', '-', '－', '/', 'nan', 'NaN', 'None', ''], np.nan)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 提取并拼接 X 轴标签：规格组 + 表面
    group_col = '规格组' if '规格组' in df.columns else 'Group_Tag'
    surface_col = '表面' if '表面' in df.columns else 'Surface'

    x_labels = df[group_col].astype(str) + "_" + df[surface_col].astype(str)

    x = np.arange(len(x_labels))  # 柱状图位置
    width = 0.25  # 三组柱子，宽度稍窄

    fig, ax = plt.subplots(figsize=(16, 6))

    # 绘制三组柱子
    rects0 = ax.bar(x - width, df['MAE_在线'], width,
                    label='在线', color='#55A868', alpha=0.85)
    rects1 = ax.bar(x, df['MAE_模型 (统一参数)'], width,
                    label='模型 (统一参数)', color='#4C72B0', alpha=0.85)
    rects2 = ax.bar(x + width, df['MAE_模型 (分组调参)'], width,
                    label='模型 (分组调参)', color='#DD8452', alpha=0.85)

    # 细节修饰
    ax.set_ylabel('MAE (平均绝对误差)', fontsize=12)
    ax.set_title(f'各规格组 {title_name} 对比（在线 vs 统一参数 vs 分组调参）',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # 动态调整 Y 轴范围（忽略 NaN）
    max_mae = np.nanmax([
        df['MAE_在线'].max(),
        df['MAE_模型 (统一参数)'].max(),
        df['MAE_模型 (分组调参)'].max()
    ])
    if np.isnan(max_mae) or max_mae <= 0:
        max_mae = 1.0
    ax.set_ylim(0, max_mae * 1.15)

    # 添加数值标签
    autolabel(ax, rects0)
    autolabel(ax, rects1)
    autolabel(ax, rects2)

    plt.tight_layout()

    # 保存图片
    save_path = os.path.join(save_dir, save_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图表已成功保存至: {save_path}")


# ==========================================
# 图1：正偏差样本 MAE 对比柱状图 (读取 Sheet2 / index 1)
# ==========================================
df_pos = pd.read_excel(excel_path, sheet_name=1)
plot_mae_comparison(df_pos, title_name="正偏差样本 MAE", save_filename="mae_pos_comparison.png")

# ==========================================
# 图2：负偏差样本 MAE 对比柱状图 (读取 Sheet3 / index 2)
# ==========================================
df_neg = pd.read_excel(excel_path, sheet_name=2)
plot_mae_comparison(df_neg, title_name="负偏差样本 MAE", save_filename="mae_neg_comparison.png")