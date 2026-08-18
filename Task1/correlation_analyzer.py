import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dcor
from sklearn.feature_selection import mutual_info_regression

# ====================== 中文字体设置（解决 Glyph missing 警告） ======================
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

def _setup_chinese_font():
    """自动寻找系统中可用的中文字体并设置"""
    from matplotlib import font_manager

    # Windows 常见中文字体（按优先级）
    candidates = [
        'Microsoft YaHei',      # 微软雅黑
        'SimHei',               # 黑体
        'SimSun',               # 宋体
        'KaiTi',                # 楷体
        'FangSong',             # 仿宋
        'Microsoft JhengHei',   # 微软正黑体
        'PingFang SC',          # macOS
        'Heiti SC',
        'WenQuanYi Micro Hei',  # Linux
        'Noto Sans CJK SC',
        'Source Han Sans SC',
        'Arial Unicode MS',
    ]

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams['font.sans-serif'] = [name] + plt.rcParams.get('font.sans-serif', [])
            print(f"[字体] 已设置中文字体: {name}")
            return name

    # 都找不到时的兜底：尝试直接指定 Windows 字体文件路径
    win_font_paths = [
        r'C:\Windows\Fonts\msyh.ttc',      # 微软雅黑
        r'C:\Windows\Fonts\msyh.ttf',
        r'C:\Windows\Fonts\simhei.ttf',    # 黑体
        r'C:\Windows\Fonts\simsun.ttc',    # 宋体
    ]
    for path in win_font_paths:
        if os.path.exists(path):
            font_manager.fontManager.addfont(path)
            font_name = font_manager.FontProperties(fname=path).get_name()
            plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams.get('font.sans-serif', [])
            print(f"[字体] 通过路径加载中文字体: {font_name} ({path})")
            return font_name

    print("[字体] 警告: 未找到可用中文字体，图中中文可能显示为方块。建议安装微软雅黑或黑体。")
    return None

_setup_chinese_font()
# ====================================================================================


class SurfaceCorrelationAnalyzer:
    """模块化：钢板表面的相关性分析 + Spearman + Mutual Information"""

    def __init__(self, default_save_dir="result/correlation_result"):
        self.default_save_dir = default_save_dir

    def analyze_surface(
        self,
        df,
        surface='Top',
        extra_cols=None,
        save_dir=None,
        corr_method='pearson',      # 'pearson' | 'spearman' | 'both'
        compute_mi=True,            # 是否计算 Mutual Information
        compute_dcor=True,          # 是否计算距离相关性
        compute_mi_matrix=False,    # 是否计算互信息热力矩阵
        compute_dcor_matrix=False,  # 是否计算距离相关性热力矩阵
        mi_random_state=42
    ):
        """
        针对指定表面 (Top/Bot) 进行相关性矩阵计算、打印并绘制热力图，
        同时可选计算 Mutual Information 重要性。

        :param df: 数据 DataFrame
        :param surface: 'Top' 或 'Bot'
        :param extra_cols: 额外想要加入分析的列名列表 (可选)
        :param save_dir: 自定义结果保存目录
        :param corr_method: 相关性方法，支持 'pearson' / 'spearman' / 'both'
        :param compute_mi: 是否计算 Mutual Information（默认 True）
        :param mi_random_state: MI 计算的随机种子
        :return: dict，包含 corr_matrix(s) 和 mi_series（如果计算了）
        """
        out_dir = save_dir if save_dir is not None else self.default_save_dir
        os.makedirs(out_dir, exist_ok=True)

        prefix = 'Top' if surface == 'Top' else 'Bot'
        surface_cn = '上' if surface == 'Top' else '下'

        actual_col = f'Tin Weight_Actual[g/m2]_GALV_WEIGHT_{prefix.upper()}_Avg'
        lab_col = f'{surface_cn}表面镀层重量A(XA1_0)'
        residual_col=f'{prefix}_Residual'
        # setpoint_col=f'Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_{prefix.upper()}_Min'
        setpoint_col = f'Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_{prefix.upper()}_Avg'

        speed_col = 'Speed[m/min]_Process_Avg'
        current_col = f'{prefix}_Current_Sum'



        df[f'{prefix}_Current_Per_Speed'] = df[current_col] / (df[speed_col] + 1e-5)
        df[f'{prefix}_Residual'] = df[lab_col] - df[actual_col]  # 计算残差
        df[f'{prefix}_Deviation'] = df[setpoint_col] - df[actual_col] # 计算偏差

        df[f'{prefix}_Current_Per_Speed'] = df[current_col] / (df[speed_col] + 1e-5)

        cols_to_check = [
            residual_col,
            actual_col,
            f'{prefix}_Deviation',
            f'{prefix}_Current_Sum',
            f'{prefix}_Current_Per_Speed',
            f'{prefix}_Theoretical_Factor',
            'Speed[m/min]_Process_Avg',
            'Dimension_[mm]_Thickness',
            'Dimension_[mm]_Width',
            # 'Dimension_[mm]_Length',
            'Steel_Grade_Encoded'
        ]


        if extra_cols:
            for col in extra_cols:
                if col in df.columns and col not in cols_to_check:
                    cols_to_check.append(col)

        existing_cols = [c for c in cols_to_check if c in df.columns]
        if not existing_cols:
            raise ValueError("没有找到任何可用的列，请检查列名。")

        data = df[existing_cols].dropna()   # 统一去缺失，保证后续计算一致
        result = {}

        # ====================== 1. 相关性分析 ======================
        methods = []
        if corr_method == 'both':
            methods = ['pearson', 'spearman']
        else:
            methods = [corr_method]

        for method in methods:
            corr_matrix = data.corr(method=method)

            # ---------- 按与残差的相关性绝对值排序 ----------
            residual_col = f'{prefix}_Residual'
            if residual_col in corr_matrix.columns:
                # 按与残差的 |corr| 降序
                order = corr_matrix[residual_col].abs().sort_values(ascending=False).index.tolist()
                corr_matrix = corr_matrix.loc[order, order]
            # ---------------------------------------------

            result[f'corr_{method}'] = corr_matrix

            print(f"\n======== 【{surface_cn}表面 {method.upper()} 相关性矩阵（按残差|corr|排序）】 ========")
            print(corr_matrix[residual_col].sort_values(ascending=False))

            plt.figure(figsize=(10, 8))
            sns.heatmap(
                corr_matrix,
                annot=True,
                cmap='coolwarm',
                fmt=".2f",
                vmin=-1,
                vmax=1,
                square=True
            )
            plt.title(f'{surface_cn}表面参数与残差 {method.upper()} 相关性热力图（按|corr|排序）')
            plt.tight_layout()

            save_img_path = os.path.join(out_dir, f"correlation_{surface}_{method}.png")
            plt.savefig(save_img_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[图表保存] {save_img_path}")

        # ====================== 2. Mutual Information ======================
        residual_col = f'{prefix}_Residual'

        if compute_mi and residual_col in data.columns:
            feature_cols = [c for c in existing_cols if c != residual_col]
            if not feature_cols:
                print("没有可用于 Mutual Information 的特征列。")
            else:
                X = data[feature_cols]
                y = data[residual_col]  # 目标改为残差

                mi_scores = mutual_info_regression(
                    X, y,
                    random_state=mi_random_state
                )
                mi_series = pd.Series(mi_scores, index=feature_cols).sort_values(ascending=False)
                result['mi'] = mi_series

                print(f"\n======== 【{surface_cn}表面 Mutual Information（目标：残差）】 ========")
                print(mi_series)

                plt.figure(figsize=(8, max(4, len(mi_series) * 0.4)))
                mi_series.sort_values().plot(kind='barh', color='steelblue')
                plt.xlabel('Mutual Information')
                plt.title(f'{surface_cn}表面特征对残差的 Mutual Information')
                plt.tight_layout()

                save_mi_path = os.path.join(out_dir, f"mi_importance_{surface}.png")
                plt.savefig(save_mi_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"[图表保存] Mutual Information 重要性图已保存至: {save_mi_path}")

        # ====================== 全变量 Mutual Information 矩阵 ======================
        if compute_mi_matrix and residual_col in data.columns:
            cols = existing_cols
            n = len(cols)
            mi_matrix = pd.DataFrame(np.zeros((n, n)), index=cols, columns=cols)

            for i, col_i in enumerate(cols):
                # 对每一列，计算它与其他所有列的 MI
                X = data[cols].drop(columns=[col_i])
                y = data[col_i]
                mi_scores = mutual_info_regression(X, y, random_state=mi_random_state)
                mi_matrix.loc[col_i, X.columns] = mi_scores

            # 对称化（取平均）
            mi_matrix = (mi_matrix + mi_matrix.T) / 2
            # 对角线设为 0
            for col in mi_matrix.columns:
                mi_matrix.loc[col, col] = 0.0

            # ---------- 按与残差的 MI 值排序（残差固定第一位） ----------
            if residual_col in mi_matrix.columns:
                # 先取出与残差的 MI，排除自己
                mi_with_residual = mi_matrix[residual_col].drop(residual_col)
                # 按绝对值从大到小排序
                other_order = mi_with_residual.abs().sort_values(ascending=False).index.tolist()
                # 残差放第一位，后面跟排序后的其他变量
                order = [residual_col] + other_order
                mi_matrix = mi_matrix.loc[order, order]

            result['mi_matrix'] = mi_matrix

            plt.figure(figsize=(10, 8))
            sns.heatmap(mi_matrix, annot=True, cmap='YlGnBu', fmt=".2f", square=True)
            plt.title(f'{surface_cn}表面 全变量 Mutual Information 矩阵')
            plt.tight_layout()
            save_path = os.path.join(out_dir, f"mi_matrix_{surface}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[图表保存] {save_path}")

        # ====================== 3. 距离相关性 (Distance Correlation) ======================
        if compute_dcor and residual_col in data.columns:
            feature_cols = [c for c in existing_cols if c != residual_col]
            dcor_scores = {}
            y = data[residual_col].values.astype(float)

            for col in feature_cols:
                x = data[col].values.astype(float)
                # 处理可能的常数列（dcor 对常数会返回 nan）
                if np.std(x) < 1e-10:
                    dcor_scores[col] = 0.0
                else:
                    dcor_scores[col] = dcor.distance_correlation(x, y)

            dcor_series = pd.Series(dcor_scores).sort_values(ascending=False)
            result['dcor'] = dcor_series

            print(f"\n======== 【{surface_cn}表面 距离相关性（目标：残差）】 ========")
            print(dcor_series)

            plt.figure(figsize=(8, max(4, len(dcor_series) * 0.4)))
            dcor_series.sort_values().plot(kind='barh', color='darkorange')
            plt.xlabel('Distance Correlation')
            plt.title(f'{surface_cn}表面特征对残差的距离相关性')
            plt.tight_layout()

            save_dcor_path = os.path.join(out_dir, f"dcor_importance_{surface}.png")
            plt.savefig(save_dcor_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[图表保存] {save_dcor_path}")

        # ====================== 全变量距离相关性矩阵 ======================
        if compute_dcor_matrix and residual_col in data.columns:
            cols = existing_cols
            n = len(cols)
            dcor_matrix = pd.DataFrame(np.zeros((n, n)), index=cols, columns=cols)

            for i, col_i in enumerate(cols):
                for j, col_j in enumerate(cols):
                    if i <= j:  # 只算上三角，对称填充
                        if i == j:
                            dcor_matrix.iloc[i, j] = 1.0
                        else:
                            val = dcor.distance_correlation(
                                data[col_i].values.astype(float),
                                data[col_j].values.astype(float)
                            )
                            dcor_matrix.iloc[i, j] = val
                            dcor_matrix.iloc[j, i] = val

            #  dcor_matrix 按照对残差的距离相关性绝对值排序
            if residual_col in dcor_matrix.columns:
                order = dcor_matrix[residual_col].abs().sort_values(ascending=False).index.tolist()
                dcor_matrix = dcor_matrix.loc[order, order]

            result['dcor_matrix'] = dcor_matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(dcor_matrix, annot=True, cmap='YlOrRd', fmt=".2f", vmin=0, vmax=1, square=True)
            plt.title(f'{surface_cn}表面 全变量距离相关性矩阵')
            plt.tight_layout()
            save_path = os.path.join(out_dir, f"dcor_matrix_{surface}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[图表保存] {save_path}")

        return result

if __name__ == "__main__":
    # 读取清洗后的数据
    cleaned_df = pd.read_excel("result/cleaned_data/cleaned_data.xlsx")
    analyzer = SurfaceCorrelationAnalyzer(default_save_dir="result/correlation_result")

    # 分表面整体分析
    for surface in ['Top', 'Bot']:
        print("\n" + "=" * 60)
        print(f"开始分析 {surface} 表面")
        print("=" * 60)
        analyzer.analyze_surface(
            cleaned_df,
            surface=surface,
            extra_cols=None,
            save_dir="result/correlation_result",
            corr_method='both',      # 同时出 pearson + spearman
            compute_mi=True,         # 计算 Mutual Information
            compute_dcor=True,       # 计算距离相关性
            compute_mi_matrix=True,   # 计算全变量 MI 矩阵
            compute_dcor_matrix=True,  # 计算全变量距离相关性矩阵
            mi_random_state=42
        )

    print("\n全部完成。结果保存在: result/correlation_result/")