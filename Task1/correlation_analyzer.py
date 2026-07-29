import os
import matplotlib.pyplot as plt
import seaborn as sns


class SurfaceCorrelationAnalyzer:
    """模块化：钢板表面的相关性分析与热力图绘制工具"""

    def __init__(self, default_save_dir="result/correlation_result"):
        self.default_save_dir = default_save_dir

    def analyze_surface(self, df, surface='Top', extra_cols=None, save_dir=None):
        """
        针对指定表面 (Top/Bot) 进行相关性矩阵计算、打印并绘制热力图

        :param df: 数据 DataFrame
        :param surface: 'Top' 或 'Bot'
        :param extra_cols: 额外想要加入相关性分析的列名列表 (可选)
        :param save_dir: 自定义结果保存目录，若不传则使用初始化时的 default_save_dir
        :return: corr_matrix (pd.DataFrame) 相关性矩阵
        """
        # 决定最终保存的文件夹
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

        # 支持传入自定义增加的特征列
        if extra_cols:
            for col in extra_cols:
                if col in df.columns and col not in cols_to_check:
                    cols_to_check.append(col)

        # 过滤只保留 df 中真实存在的列，防止 KeyError
        existing_cols = [c for c in cols_to_check if c in df.columns]

        corr_matrix = df[existing_cols].corr()

        print(f"\n======== 【{surface_cn}表面 相关性矩阵】 ========")
        if lab_col in corr_matrix.columns:
            print(corr_matrix[lab_col].sort_values(ascending=False))
        else:
            print(corr_matrix)

        # 绘图
        plt.figure(figsize=(9, 7))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
        plt.title(f'{surface_cn}表面参数与实验室测定值相关性热力图')
        plt.tight_layout()

        save_img_path = os.path.join(out_dir, f"correlation_{surface}.png")
        plt.savefig(save_img_path, dpi=300)
        print(f"[图表保存] {surface_cn}表面相关性热力图已保存至: {save_img_path}")
        plt.show()

        return corr_matrix