import pandas as pd
import numpy as np
from train_and_save_models import ResidualCorrectionModel

# ==========================================
# 通用推理脚本：
# 1. 读取新的原始数据（不需要包含实验室真实值列，只需要在线仪表值和工艺参数）
# 2. 复现训练时同样的特征工程（电流求和、理论因子、钢种编码）
# 3. 加载已训练好的 Top / Bot 模型，输出校正后的预测值
#
# 使用场景：新一批卷板生产完成、拿到在线仪表读数后，直接调用本脚本得到校正结果，
# 不需要重新训练。
# ==========================================

def build_inference_features(df):
    """
    复现训练时的特征工程步骤（电流求和、理论因子），
    这里不依赖 preprocess_and_filter_outliers 里的清洗/剔除逻辑，
    因为推理阶段通常是逐卷或逐批处理，不需要也不应该做基于全量统计的离群点剔除。
    """
    df = df.copy()

    if 'Tining Section_CONCENT[NTU]_GL_1_Avg' in df.columns:
        df.rename(columns={'Tining Section_CONCENT[NTU]_GL_1_Avg': 'Tining Section_CURRENT[A]_GL_1_Avg'},
                  inplace=True)

    bot_curr_cols = [f'Tining Section_CURRENT[A]_GL_{i}_Avg' for i in range(1, 37, 2)]
    top_curr_cols = [f'Tining Section_CURRENT[A]_GL_{i}_Avg' for i in range(2, 37, 2)]

    df['Bot_Current_Sum'] = df[bot_curr_cols].sum(axis=1)
    df['Top_Current_Sum'] = df[top_curr_cols].sum(axis=1)

    df['Width_m'] = df['Dimension_[mm]_Width'] / 1000.0
    speed = df['Speed[m/min]_Process_Avg'].replace(0, np.nan)

    df['Top_Theoretical_Factor'] = df['Top_Current_Sum'] / (speed * df['Width_m'])
    df['Bot_Theoretical_Factor'] = df['Bot_Current_Sum'] / (speed * df['Width_m'])

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df['Top_Current_Per_Speed'] = df['Top_Current_Sum'] / (speed + 1e-5)
    df['Bot_Current_Per_Speed'] = df['Bot_Current_Sum'] / (speed + 1e-5)

    return df


def predict_surface(df, model_path, warm_start=False):
    """
    加载指定表面的模型并对 df 做预测。
    df 需要已经过 build_inference_features 处理，且包含该模型 feature_cols 里列出的所有列。
    返回：预测值 Series（即校正后的镀层重量预测）、平滑残差 Series。
    """
    corrector = ResidualCorrectionModel.load(model_path)

    # 用模型自带的钢种频率表对新数据编码，保证和训练时使用同一份映射
    df = corrector.encode_steel_grade(df)

    # 校验特征是否齐全，提前报错比让 sklearn 抛出难懂的报错更友好
    missing_cols = [c for c in corrector.feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"输入数据缺少模型所需的特征列: {missing_cols}")

    pred, smoothed_delta = corrector.predict(df, warm_start=warm_start)
    return pred, smoothed_delta, corrector


def predict_both_surfaces(df, top_model_path="result/models/residual_model_Top.joblib",
                           bot_model_path="result/models/residual_model_Bot.joblib",
                           warm_start=False):
    """
    一次性对上下表面都做预测，返回一个附加了预测列的 DataFrame 副本。
    """
    df = build_inference_features(df)

    top_pred, top_smoothed_delta, _ = predict_surface(df, top_model_path, warm_start=warm_start)
    bot_pred, bot_smoothed_delta, _ = predict_surface(df, bot_model_path, warm_start=warm_start)

    result_df = df.copy()
    result_df['Top_Weight_Predicted'] = top_pred
    result_df['Top_Correction_Delta'] = top_smoothed_delta
    result_df['Bot_Weight_Predicted'] = bot_pred
    result_df['Bot_Correction_Delta'] = bot_smoothed_delta

    return result_df


# ==========================================
# 示例主流程：对一批新数据做推理并导出结果
# ==========================================
if __name__ == "__main__":
    # 新数据示例：可以是新一批生产数据，只需要包含原始工艺参数和在线仪表读数，
    # 不需要实验室真实值列（因为推理时我们本来就是要去预测/校正这个值）
    new_df = pd.read_excel("result/new_batch_data.xlsx")

    result_df = predict_both_surfaces(new_df, warm_start=False)

    output_cols = [c for c in [
        'Coil ID', 'Steel Grade', 'Produce Time',
        'Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Avg', 'Top_Weight_Predicted',
        'Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Avg', 'Bot_Weight_Predicted'
    ] if c in result_df.columns]

    save_path = "result/new_batch_predictions.xlsx"
    result_df[output_cols].to_excel(save_path, index=False)
    print(f"[导出提示] 预测结果已保存至: {save_path}")
    print(result_df[output_cols].head(10).to_string(index=False))