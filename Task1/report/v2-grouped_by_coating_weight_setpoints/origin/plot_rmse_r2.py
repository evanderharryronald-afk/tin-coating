import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

# ============================================================
# 1. 字体与负号全局设置（彻底消除 U+2212 警告）
# ============================================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False          # 强制使用 ASCII 减号 -
plt.rcParams['font.family'] = 'sans-serif'

# 额外保险：自定义 tick 格式化器，永远输出普通减号
def plain_minus_formatter(x, pos):
    return f'{x:g}'.replace('−', '-')   # 把可能出现的 Unicode 减号替换掉

# ============================================================
# 2. 配置文件路径
# ============================================================
excel_path = "result_comparison.xlsx"
save_dir = "plots"
os.makedirs(save_dir, exist_ok=True)

# 3. 读取数据
df = pd.read_excel(excel_path, sheet_name=0)

df['规格组'] = df['规格组'].astype(str).str.strip()
df['表面'] = df['表面'].astype(str).str.strip()
groups = df['规格组'] + "_" + df['表面']

# 数值转换 + 缺失符号处理
cols_to_numeric = [
    'RMSE_在线', 'RMSE_模型 (统一参数)', 'RMSE_模型 (分组调参)',
    'R2_在线', 'R2_模型 (统一参数)', 'R2_模型 (分组调参)'
]
for col in cols_to_numeric:
    if col in df.columns:
        df[col] = df[col].replace(['—', '–', '-', '－', '/', 'nan', 'NaN', 'None', ''], np.nan)
        df[col] = pd.to_numeric(df[col], errors='coerce')

x = np.arange(len(groups))
width = 0.25


def autolabel(ax, rects, precision=4):
    for rect in rects:
        height = rect.get_height()
        if np.isnan(height):
            continue
        ax.annotate(f'{height:.{precision}f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=7)


# ==========================================
# 图1：RMSE
# ==========================================
fig, ax = plt.subplots(figsize=(16, 6))

rects0 = ax.bar(x - width, df['RMSE_在线'], width, label='在线', color='#55A868', alpha=0.85)
rects1 = ax.bar(x, df['RMSE_模型 (统一参数)'], width, label='统一参数', color='#4C72B0', alpha=0.85)
rects2 = ax.bar(x + width, df['RMSE_模型 (分组调参)'], width, label='分组调参', color='#DD8452', alpha=0.85)

ax.set_ylabel('RMSE (越低越好)', fontsize=12)
ax.set_title('各规格组与表面 RMSE 对比（在线 vs 统一参数 vs 分组调参）',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(groups, rotation=30, ha='right', fontsize=9)
ax.legend(fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.5)

max_rmse = np.nanmax([df['RMSE_在线'].max(),
                      df['RMSE_模型 (统一参数)'].max(),
                      df['RMSE_模型 (分组调参)'].max()])
ax.set_ylim(0, (max_rmse if not np.isnan(max_rmse) else 1.0) * 1.15)

autolabel(ax, rects0)
autolabel(ax, rects1)
autolabel(ax, rects2)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "rmse_comparison.png"), dpi=300, bbox_inches='tight')
plt.close()
print(f"RMSE 对比柱状图已保存至: {os.path.join(save_dir, 'rmse_comparison.png')}")


# ==========================================
# 图2：R²（使用 symlog + 彻底消除负号警告）
# ==========================================
fig, ax = plt.subplots(figsize=(16, 7))

rects0 = ax.bar(x - width, df['R2_在线'], width, label='在线', color='#55A868', alpha=0.85)
rects1 = ax.bar(x, df['R2_模型 (统一参数)'], width, label='统一参数', color='#4C72B0', alpha=0.85)
rects2 = ax.bar(x + width, df['R2_模型 (分组调参)'], width, label='分组调参', color='#DD8452', alpha=0.85)

ax.set_ylabel('R2 (越高越好)', fontsize=12)
ax.set_title('各规格组与表面 R2 对比（在线 vs 统一参数 vs 分组调参）',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(groups, rotation=30, ha='right', fontsize=9)

# 对称对数坐标
ax.set_yscale('symlog', linthresh=0.05)
ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', zorder=0)

# 强制 tick 使用普通减号（消除警告的关键）
ax.yaxis.set_major_formatter(ticker.FuncFormatter(plain_minus_formatter))

ax.legend(fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.5)

autolabel(ax, rects0)
autolabel(ax, rects1)
autolabel(ax, rects2)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "r2_comparison.png"), dpi=300, bbox_inches='tight')
plt.close()
print(f"R² 对比柱状图已保存至: {os.path.join(save_dir, 'r2_comparison.png')}")