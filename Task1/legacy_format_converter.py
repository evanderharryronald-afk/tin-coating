import os
import pandas as pd
import numpy as np

# 0. 动态获取当前脚本所在的绝对路径，确保相对路径拼接 100% 正确
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. 定义旧表规范的所有标准字段顺序
OLD_FORMAT_COLUMNS = [
    "Coil ID", "Steel Grade", "Produce Time", "Dimension_[mm]_Thickness",
    "Dimension_[mm]_Width", "Dimension_[mm]_Length", "Speed[m/min]_Process_Avg"
] + [f"Tining Section_CURRENT[A]_GL_{i}_Avg" for i in range(1, 37)] + [
    "Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_TOP_Avg",
    "Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_TOP_Max",
    "Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_TOP_Min",
    "Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_BOT_Avg",
    "Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_BOT_Max",
    "Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_BOT_Min",
    "Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Avg",
    "Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Max",
    "Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Min",
    "Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Avg",
    "Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Max",
    "Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Min",
    "上表面镀层重量A(XA1_0)", "上表面镀层重量C(XA1_0)", "上表面镀层重量D(XA1_0)", "上表面镀层重量W(XA1_0)",
    "下表面镀层重量A(XA1_0)", "下表面镀层重量C(XA1_0)", "下表面镀层重量D(XA1_0)", "下表面镀层重量W(XA1_0)",
    "上表面镀层标准重量(XA1_0)", "上表面镀层重量_最大(XA1_0)", "上表面镀层重量_最小(XA1_0)",
    "下表面镀层标准重量(XA1_0)", "下表面镀层重量_最大(XA1_0)", "下表面镀层重量_最小(XA1_0)"
]

# 2. 拼接绝对路径
input_file = os.path.join(BASE_DIR, "data", "new_data", "V_PCOILDATA_TIN 202607-202608.xlsx")
output_file = os.path.join(BASE_DIR, "data", "new_data", "converted_old_format.xlsx")

# 检查输入文件是否存在，若不存在给出明确提示
if not os.path.exists(input_file):
    raise FileNotFoundError(f"未找到输入文件，请检查文件是否存在于此路径下：\n{input_file}")

# 读取新表（header=1 表示将第 2 行英文列名作为表头）
df_new = pd.read_excel(input_file, header=1)

# 3. 构建新列名 -> 旧列名的“翻译”字典
column_mapping = {
    'MAT_IDENT': 'Coil ID',
    'PROD_STEELGRADE': 'Steel Grade',
    'PROD_TIME_END': 'Produce Time',
    'THICKNESS': 'Dimension_[mm]_Thickness',
    'WIDTH': 'Dimension_[mm]_Width',
    'LENGTH': 'Dimension_[mm]_Length',
    'V_CENTER_AVG': 'Speed[m/min]_Process_Avg',
    'GALV_WEIGHT_TOP': 'Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_TOP_Avg',
    'GALV_WEIGHT_TOP_MAX': 'Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_TOP_Max',
    'GALV_WEIGHT_TOP_MIN': 'Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_TOP_Min',
    'GALV_WEIGHT_BOT': 'Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_BOT_Avg',
    'GALV_WEIGHT_BOT_MAX': 'Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_BOT_Max',
    'GALV_WEIGHT_BOT_MIN': 'Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_BOT_Min',
    'GALV_WEIGHT_TOP_ACT': 'Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Avg',
    'GALV_WEIGHT_BOT_ACT': 'Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Avg',
}

# 批量添加 1~36 槽电流列名的翻译关系
for i in range(1, 37):
    column_mapping[f'GL_{i}_AVG'] = f'Tining Section_CURRENT[A]_GL_{i}_Avg'

# 4. 执行“翻译”（重命名列名）
df_translated = df_new.rename(columns=column_mapping)

# 5. 处理格式：先按旧表标准列顺序排列，再追加新数据中多余的列
extra_columns = [col for col in df_translated.columns if col not in OLD_FORMAT_COLUMNS]
final_columns = OLD_FORMAT_COLUMNS + extra_columns
df_final = df_translated.reindex(columns=final_columns)

# 6. 保存导出为旧格式 Excel
df_final.to_excel(output_file, index=False)
print("格式转换与字段翻译完成！已导出为：", output_file)