import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression


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
            result[f'corr_{method}'] = corr_matrix

            print(f"\n======== 【{surface_cn}表面 {method.upper()} 相关性矩阵】 ========")
            if lab_col in corr_matrix.columns:
                print(corr_matrix[lab_col].sort_values(ascending=False))
            else:
                print(corr_matrix)

            # 绘制热力图
            plt.figure(figsize=(9, 7))
            sns.heatmap(
                corr_matrix,
                annot=True,
                cmap='coolwarm',
                fmt=".2f",
                vmin=-1,
                vmax=1,
                square=True
            )
            plt.title(f'{surface_cn}表面参数与实验室测定值 {method.upper()} 相关性热力图')
            plt.tight_layout()

            save_img_path = os.path.join(out_dir, f"correlation_{surface}_{method}.png")
            plt.savefig(save_img_path, dpi=300)
            print(f"[图表保存] {surface_cn}表面 {method} 相关性热力图已保存至: {save_img_path}")
            plt.show()

        # ====================== 2. Mutual Information ======================
        if compute_mi and lab_col in data.columns:
            feature_cols = [c for c in existing_cols if c != lab_col]
            if not feature_cols:
                print("没有可用于 Mutual Information 的特征列。")
            else:
                X = data[feature_cols]
                y = data[lab_col]

                mi_scores = mutual_info_regression(
                    X, y,
                    random_state=mi_random_state
                )
                mi_series = pd.Series(mi_scores, index=feature_cols).sort_values(ascending=False)
                result['mi'] = mi_series

                print(f"\n======== 【{surface_cn}表面 Mutual Information（目标：{lab_col}）】 ========")
                print(mi_series)

                # 绘制 MI 条形图
                plt.figure(figsize=(8, max(4, len(mi_series) * 0.4)))
                mi_series.sort_values().plot(kind='barh', color='steelblue')
                plt.xlabel('Mutual Information')
                plt.title(f'{surface_cn}表面特征对实验室镀层重量的 Mutual Information')
                plt.tight_layout()

                save_mi_path = os.path.join(out_dir, f"mi_importance_{surface}.png")
                plt.savefig(save_mi_path, dpi=300)
                print(f"[图表保存] Mutual Information 重要性图已保存至: {save_mi_path}")
                plt.show()

        return result