import os
import pandas as pd
import numpy as np


class FeatureEngineer:
    """
    电镀锡生产数据特征工程模块
    包含：电流求和、物理理论因子构建、数据去中心化残差计算、钢种频率编码等
    """

    def __init__(self, eps=1e-5):
        self.eps = eps

    def transform(self, df):
        """
        根据传入的 DataFrame 构建所有衍生特征
        """
        df = df.copy()

        # 1. 电流字段分组聚合
        bot_curr_cols = [f'Tining Section_CURRENT[A]_GL_{i}_Avg' for i in range(1, 37, 2) if f'Tining Section_CURRENT[A]_GL_{i}_Avg' in df.columns]
        top_curr_cols = [f'Tining Section_CURRENT[A]_GL_{i}_Avg' for i in range(2, 37, 2) if f'Tining Section_CURRENT[A]_GL_{i}_Avg' in df.columns]

        df['Bot_Current_Sum'] = df[bot_curr_cols].sum(axis=1) if bot_curr_cols else 0.0
        df['Top_Current_Sum'] = df[top_curr_cols].sum(axis=1) if top_curr_cols else 0.0

        # 2. 物理衍生特征（法拉第强相关特征）
        df['Width_m'] = df['Dimension_[mm]_Width'] / 1000.0
        speed = df['Speed[m/min]_Process_Avg'].replace(0, np.nan)

        # 单位速度电流比
        df['Top_Current_Per_Speed'] = df['Top_Current_Sum'] / (speed + self.eps)
        df['Bot_Current_Per_Speed'] = df['Bot_Current_Sum'] / (speed + self.eps)

        # 理论因子计算 (安培 / (速度*宽度))
        df['Top_Theoretical_Factor'] = df['Top_Current_Sum'] / (speed * df['Width_m'] + self.eps)
        df['Bot_Theoretical_Factor'] = df['Bot_Current_Sum'] / (speed * df['Width_m'] + self.eps)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # 3. 残差计算与去中心化 (Delta & Centered Delta)
        if '上表面镀层重量A(XA1_0)' in df.columns and 'Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Avg' in df.columns:
            df['Top_Delta'] = df['上表面镀层重量A(XA1_0)'] - df['Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Avg']
            df['Bot_Delta'] = df['下表面镀层重量A(XA1_0)'] - df['Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Avg']

            # 计算在干净集上的 Global Bias 并去除
            top_bias = df['Top_Delta'].mean()
            bot_bias = df['Bot_Delta'].mean()

            df['Top_Delta_Centered'] = df['Top_Delta'] - top_bias
            df['Bot_Delta_Centered'] = df['Bot_Delta'] - bot_bias

        # 4. 钢种频率编码
        if 'Steel Grade' in df.columns:
            grade_freq = df['Steel Grade'].value_counts(normalize=True).to_dict()
            df['Steel_Grade_Encoded'] = df['Steel Grade'].map(grade_freq).fillna(0)
        else:
            df['Steel_Grade_Encoded'] = 0

        return df


def main():
    """
    标准流水线：先读取数据/干净数据 -> 做特征工程 -> 保存结果
    """
    clean_data_path = "result/data/cleaned_data/cleaned_data.xlsx"
    featured_save_path = "result/data/feature_engineered_data/featured_data.xlsx"

    # 1. 检查清洗后的数据集是否存在，若不存在则尝试使用合并数据
    if os.path.exists(clean_data_path):
        print(f"[读取] 加载已清洗数据: {clean_data_path}")
        df_input = pd.read_excel(clean_data_path)
    else:
        print("[警告] 未找到清洗后的 cleaned_data.xlsx，尝试直接读取 merged_result_latest.xlsx")
        df_input = pd.read_excel("result/data/merged_data/merged_result_latest.xlsx")

    # 2. 执行特征工程
    fe = FeatureEngineer()
    featured_df = fe.transform(df_input)

    # 3. 保存特征工程后的数据集
    os.makedirs(os.path.dirname(featured_save_path), exist_ok=True)
    featured_df.to_excel(featured_save_path, index=False)

    print("\n==========================================")
    print("         [特征工程处理完成汇总]           ")
    print("==========================================")
    print(f"输入数据行数: {len(df_input)}")
    print(f"输出特征维度: {featured_df.shape[1]} 列")
    print(f"[导出提示] 特征工程数据集已保存至: {featured_save_path}")
    print("==========================================\n")


if __name__ == "__main__":
    main()