import os
import pandas as pd
import numpy as np
from scipy.stats import median_abs_deviation


class SteelDataCleaner:
    """
    电镀锡生产数据通用清洗与异常诊断模块
    """

    def __init__(self,
                 min_speed=20.0,
                 max_range_abs=0.5,  # Max与Min的最大绝对允许差值 (g/m2)
                 max_range_ratio=0.4,  # (Max - Min) / Avg 的最大允许波动比例
                 mad_factor=3.0):  # 残差离群点 MAD 倍数
        self.min_speed = min_speed
        self.max_range_abs = max_range_abs
        self.max_range_ratio = max_range_ratio
        self.mad_factor = mad_factor

    def process(self, df,
                clean_save_path="result/cleaned_data/cleaned_data.xlsx",
                filtered_save_path="result/cleaned_data/filtered_outliers.xlsx"):

        df = df.copy()

        # 1. 电流与理论因子构建
        bot_curr_cols = [f'Tining Section_CURRENT[A]_GL_{i}_Avg' for i in range(1, 37, 2)]
        top_curr_cols = [f'Tining Section_CURRENT[A]_GL_{i}_Avg' for i in range(2, 37, 2)]

        df['Bot_Current_Sum'] = df[bot_curr_cols].sum(axis=1)
        df['Top_Current_Sum'] = df[top_curr_cols].sum(axis=1)

        df['Width_m'] = df['Dimension_[mm]_Width'] / 1000.0
        speed = df['Speed[m/min]_Process_Avg'].replace(0, np.nan)

        df['Top_Theoretical_Factor'] = df['Top_Current_Sum'] / (speed * df['Width_m'])
        df['Bot_Theoretical_Factor'] = df['Bot_Current_Sum'] / (speed * df['Width_m'])

        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # 2. 计算残差
        df['Top_Delta'] = df['上表面镀层重量A(XA1_0)'] - df['Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Avg']
        df['Bot_Delta'] = df['下表面镀层重量A(XA1_0)'] - df['Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Avg']

        # 2. 去中心化（去除 Global Bias）
        top_bias = df['Top_Delta'].mean()
        bot_bias = df['Bot_Delta'].mean()

        df['Top_Delta_Centered'] = df['Top_Delta'] - top_bias
        df['Bot_Delta_Centered'] = df['Bot_Delta'] - bot_bias


        # 3. 钢种频率编码
        if 'Steel Grade' in df.columns:
            grade_freq = df['Steel Grade'].value_counts(normalize=True).to_dict()
            df['Steel_Grade_Encoded'] = df['Steel Grade'].map(grade_freq).fillna(0)
        else:
            df['Steel_Grade_Encoded'] = 0

        # 初始化剔除标记列
        df['Filter_Reason'] = ""
        initial_count = len(df)

        # ----------------------------------------------------
        # 扩展的异常过滤规则体系
        # ----------------------------------------------------

        # 规则 1: 关键字段缺失值检查
        required_cols = [
            'Top_Current_Sum', 'Bot_Current_Sum',
            'Top_Theoretical_Factor', 'Bot_Theoretical_Factor',
            'Speed[m/min]_Process_Avg', 'Dimension_[mm]_Width', 'Dimension_[mm]_Thickness',
            'Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Avg', 'Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Avg',
            '上表面镀层重量A(XA1_0)', '下表面镀层重量A(XA1_0)',
            'Top_Delta', 'Bot_Delta'
        ]
        null_mask = df[required_cols].isnull().any(axis=1)
        df.loc[null_mask, 'Filter_Reason'] += "关键工艺/测量参数缺失; "

        # 规则 2: 低速 / 停机过渡区
        low_speed_mask = df['Speed[m/min]_Process_Avg'] <= self.min_speed
        df.loc[low_speed_mask, 'Filter_Reason'] += f"车速低于{self.min_speed}m/min(停机或过渡区); "

        # 规则 3: 仪表零值/死值异常 (Min <= 0 或 Avg <= 0)
        top_zero_mask = (df['Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Min'] <= 0) | (
                    df['Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Avg'] <= 0)
        bot_zero_mask = (df['Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Min'] <= 0) | (
                    df['Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Avg'] <= 0)

        df.loc[top_zero_mask, 'Filter_Reason'] += "上表面在线仪表零值/死值异常; "
        df.loc[bot_zero_mask, 'Filter_Reason'] += "下表面在线仪表零值/死值异常; "

        # 规则 4: 仪表波动过大/非稳态 (分别判断 Top / Bot)
        top_range = df['Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Max'] - df[
            'Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Min']
        bot_range = df['Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Max'] - df[
            'Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Min']

        top_range_ratio = top_range / (df['Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Avg'] + 1e-5)
        bot_range_ratio = bot_range / (df['Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Avg'] + 1e-5)

        top_unstable_mask = (top_range > self.max_range_abs) | (top_range_ratio > self.max_range_ratio)
        bot_unstable_mask = (bot_range > self.max_range_abs) | (bot_range_ratio > self.max_range_ratio)

        df.loc[top_unstable_mask, 'Filter_Reason'] += "上表面在线仪表波动过大(Max/Min极差超标); "
        df.loc[bot_unstable_mask, 'Filter_Reason'] += "下表面在线仪表波动过大(Max/Min极差超标); "

        # 规则 5: 基于 MAD 的残差极值诊断 (仅在非缺失且非停机样本上统计)
        valid_mask = df['Filter_Reason'] == ""
        if valid_mask.sum() > 0:
            valid_df = df[valid_mask]

            top_delta_center = valid_df['Top_Delta'].median()
            top_delta_mad = median_abs_deviation(valid_df['Top_Delta'], scale='normal')
            bot_delta_center = valid_df['Bot_Delta'].median()
            bot_delta_mad = median_abs_deviation(valid_df['Bot_Delta'], scale='normal')

            top_threshold = self.mad_factor * top_delta_mad
            bot_threshold = self.mad_factor * bot_delta_mad

            top_outlier_mask = (df['Top_Delta'] - top_delta_center).abs() > top_threshold
            bot_outlier_mask = (df['Bot_Delta'] - bot_delta_center).abs() > bot_threshold

            df.loc[valid_mask & top_outlier_mask, 'Filter_Reason'] += f"上表面残差异常(>{top_threshold:.2f}g/m2); "
            df.loc[valid_mask & bot_outlier_mask, 'Filter_Reason'] += f"下表面残差异常(>{bot_threshold:.2f}g/m2); "

        # 拆分数据
        filtered_df = df[df['Filter_Reason'] != ""].copy()
        clean_df = df[df['Filter_Reason'] == ""].copy()

        # 导出 Excel 文件
        if clean_save_path or filtered_save_path:
            os.makedirs(os.path.dirname(clean_save_path), exist_ok=True)

            # 补全上下表面的所有极值与均值导出列
            cols_to_export = [
                'Coil ID', 'Steel Grade', 'Speed[m/min]_Process_Avg',
                # 上表面在线仪表数据
                'Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Avg',
                'Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Max',
                'Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Min',
                # 下表面在线仪表数据
                'Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Avg',
                'Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Max',
                'Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Min',
                # 残差与剔除原因
                'Top_Delta', 'Bot_Delta','Top_Delta_Centered','Bot_Delta_Centered','Filter_Reason'
            ]
            cols_to_export = [c for c in cols_to_export if c in filtered_df.columns]

            filtered_df[cols_to_export].to_excel(filtered_save_path, index=False)
            clean_df.to_excel(clean_save_path, index=False)

        print("\n==========================================")
        print("        [数据清洗与异常诊断汇总]          ")
        print("==========================================")
        print(f"原始数据总行数: {initial_count}")
        print(f"被剔除异常点数: {len(filtered_df)} (占比: {len(filtered_df) / initial_count * 100:.2f}%)")
        print(f"保留干净样本数: {len(clean_df)}")
        print(f"[导出提示] 被剔除数据明细及原因已保存至: {filtered_save_path}")
        print(f"[导出提示] 训练用干净数据集已保存至: {clean_save_path}")
        print("==========================================\n")

        return clean_df

if __name__ == "__main__":
    raw_df = pd.read_excel("result/merged_data/merged_result_latest.xlsx")

    # 步骤 1: 创建清洗器实例，进行预处理、诊断离群点并自动导出 filtered_outliers.xlsx
    cleaner = SteelDataCleaner()
    clean_df = cleaner.process(
        raw_df,
        clean_save_path="result/cleaned_data/cleaned_data.xlsx",
        filtered_save_path="result/cleaned_data/filtered_outliers.xlsx"
    )