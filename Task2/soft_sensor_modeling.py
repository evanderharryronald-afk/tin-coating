import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

# 1. 读取 Excel 文件
file_path = "E:/SGAI_Project/2026/Tin coating/Task2/锡层三组数据整合数据.xlsx"
df = pd.read_excel(file_path, sheet_name=0, header=1, engine='openpyxl')

# 清洗列名
df.columns = df.columns.str.strip()

# 定义上下表面字段映射
top_cols = {
    'lab': '上表面镀层重量A(XA1_0)',  # 最终实验室数据
    'near_lab': '上表面镀锡量-均值',  # 机旁实验室数据
    'realtime': '上表面实时锡层重量_y',  # 实时测量数据
    'sn_num': '锡号-上'  # 锡号
}

bottom_cols = {
    'lab': '下表面镀层重量A(XA1_0)',
    'near_lab': '下表面镀锡量-均值',
    'realtime': '下表面实时锡层重量_y',
    'sn_num': '锡号-下'
}


def evaluate_and_correct(df_data, cols, surface_name):
    print(f"\n==================== {surface_name} 独立修正逻辑分析 ====================")

    # 剔除空值并确保数据为数值型
    sub_df = df_data[[cols['lab'], cols['near_lab'], cols['realtime'], cols['sn_num']]].dropna()
    for c in sub_df.columns:
        sub_df[c] = pd.to_numeric(sub_df[c], errors='coerce')
    sub_df = sub_df.dropna()

    # ---------------- 1. 相关性定量分析 ----------------
    corr_real_near = sub_df[cols['realtime']].corr(sub_df[cols['near_lab']])
    corr_real_lab = sub_df[cols['realtime']].corr(sub_df[cols['lab']])
    print(f"-> [相关性分析]")
    print(f"   实时测量数据 vs 机旁实验室数据 相关系数: {corr_real_near:.4f}")
    print(f"   实时测量数据 vs 最终实验室数据 相关系数: {corr_real_lab:.4f}")

    # ---------------- 2. 构建修正模型 ----------------
    # 我们的目标是只用 [实时数据, 锡号] 作为自变量 X
    X = sub_df[[cols['realtime'], cols['sn_num']]]

    # 路线 A：用实时数据 修正/取代 [机旁数据]
    y_near = sub_df[cols['near_lab']]
    X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(X, y_near, test_size=0.2, random_state=42)
    model_near = Ridge(alpha=1.0).fit(X_train_n, y_train_n)
    y_pred_n = model_near.predict(X_test_n)
    mae_n = mean_absolute_error(y_test_n, y_pred_n)
    r2_n = r2_score(y_test_n, y_pred_n)

    # 新增：计算路线 A 的 MAPE (平均绝对百分比误差)
    # 过滤掉真值为 0 的样本防止除以 0 报错
    valid_mask_n = y_test_n != 0
    mape_n = np.mean(np.abs((y_test_n[valid_mask_n] - y_pred_n[valid_mask_n]) / y_test_n[valid_mask_n])) * 100

    # 路线 B：用实时数据 修正/取代 [最终实验室数据]
    y_lab = sub_df[cols['lab']]
    X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X, y_lab, test_size=0.2, random_state=42)
    model_lab = Ridge(alpha=1.0).fit(X_train_l, y_train_l)
    y_pred_l = model_lab.predict(X_test_l)
    mae_l = mean_absolute_error(y_test_l, y_pred_l)
    r2_l = r2_score(y_test_l, y_pred_l)

    # 新增：计算路线 B 的 MAPE (平均绝对百分比误差)
    valid_mask_l = y_test_l != 0
    mape_l = np.mean(np.abs((y_test_l[valid_mask_l] - y_pred_l[valid_mask_l]) / y_test_l[valid_mask_l])) * 100

    # ---------------- 3. 输出修正公式与评估 ----------------
    print(f"\n-> 【修正路线一】：利用实时数据直接替代[机旁数据]")
    print(f"   替代后与机旁真值的平均绝对误差 (MAE): {mae_n:.4f} g/m²")
    print(f"   替代后与机旁真值的平均绝对百分比误差 (MAPE): {mape_n:.2f}%")  # 增加此行输出
    print(f"   模型拟合优度 (R2): {r2_n:.4f}")
    print(
        f"   [替代公式]: 机旁替代值 = {model_near.coef_[0]:.4f} * 实时测量值 + {model_near.coef_[1]:.4f} * 锡号 + ({model_near.intercept_:.4f})")

    print(f"\n-> 【修正路线二】：利用实时数据直接替代[最终实验室数据]")
    print(f"   替代后与实验室真值的平均绝对误差 (MAE): {mae_l:.4f} g/m²")
    print(f"   替代后与实验室真值的平均绝对百分比误差 (MAPE): {mape_l:.2f}%")  # 增加此行输出
    print(f"   模型拟合优度 (R2): {r2_l:.4f}")
    print(
        f"   [替代公式]: 实验室替代值 = {model_lab.coef_[0]:.4f} * 实时测量值 + {model_lab.coef_[1]:.4f} * 锡号 + ({model_lab.intercept_:.4f})")

    # 决策建议
    if mae_l < mae_n:
        print(
            f"\n💡 结论建议：{surface_name}实时数据对【最终实验室数据】的修正拟合度更高（误差更小），建议以此公式生成的修正值直接充当实验数据。")
    else:
        print(
            f"\n💡 结论建议：{surface_name}实时数据对【机旁实验室数据】的修正拟合度更高（误差更小），建议以此公式生成的修正值直接充当实验数据。")


# 运行分析
evaluate_and_correct(df, top_cols, "上表面")
evaluate_and_correct(df, bottom_cols, "下表面")