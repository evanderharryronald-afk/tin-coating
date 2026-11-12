"""新表 → 旧表格式转换器。

作为库导入时只暴露 OLD_FORMAT_COLUMNS / convert_new_to_old 函数；
作为脚本直接运行时才真正读写 Excel 文件。
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

# Task1/ 根目录 —— 无论本文件被 import 还是直接运行都指向同一个地方
TASK1_ROOT = Path(__file__).resolve().parents[2]

# 1. 旧表规范的所有标准字段顺序
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


def _build_mapping() -> dict:
    mapping = {
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
    for i in range(1, 37):
        mapping[f'GL_{i}_AVG'] = f'Tining Section_CURRENT[A]_GL_{i}_Avg'
    return mapping


def convert_new_to_old(df_new: pd.DataFrame) -> pd.DataFrame:
    """把新表（英文短列名）转成旧表规范列顺序。"""
    mapping = _build_mapping()
    df_translated = df_new.rename(columns=mapping)
    extra_columns = [c for c in df_translated.columns if c not in OLD_FORMAT_COLUMNS]
    final_columns = OLD_FORMAT_COLUMNS + extra_columns
    return df_translated.reindex(columns=final_columns)


def main():
    input_file = TASK1_ROOT / "data" / "new_data" / "V_PCOILDATA_TIN 202607-202608.xlsx"
    output_file = TASK1_ROOT / "data" / "new_data" / "converted_old_format.xlsx"

    if not input_file.exists():
        raise FileNotFoundError(f"未找到输入文件：{input_file}")

    df_new = pd.read_excel(input_file, header=1)
    df_final = convert_new_to_old(df_new)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_excel(output_file, index=False)
    print("格式转换与字段翻译完成！已导出为：", output_file)


if __name__ == "__main__":
    main()
