"""超参 / 配置 / 规格组工具函数。

从 coating_model_by_group.py 中抽离出与训练主流程解耦的"纯工具"部分。
"""
from __future__ import annotations

import json
import os


DEFAULT_PARAMS = {
    "damping": 0.6,
    "pos_boost": 4.6,
    "alpha_smoothing": 1.0,
    "max_iter": 200,
    "learning_rate": 0.05,
    "max_depth": 4,
}

MIN_GROUP_SAMPLES = 200

TOP_SETPOINT_COL = 'Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_TOP_Min'
BOT_SETPOINT_COL = 'Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_BOT_Min'


def load_pipeline_config(config_path: str):
    """读取 JSON 配置并提供合理默认兜底，config 不存在则返回内置默认。"""
    default_config = {
        "target_groups": None,
        "min_group_samples": MIN_GROUP_SAMPLES,
        "default_params": dict(DEFAULT_PARAMS),
        "group_params_override": {},
    }
    if not os.path.exists(config_path):
        return default_config
    with open(config_path, "r", encoding="utf-8") as f:
        user_config = json.load(f)
    default_config.update(user_config)
    return default_config


def get_params_for_group(config, group_label: str, surface: str) -> dict:
    """合并 default_params 与 group_params_override[组__面]。"""
    base = config.get("default_params", {}).copy()
    key = f"{group_label}__{surface}"
    base.update(config.get("group_params_override", {}).get(key, {}))
    return base


def build_corrector_instance(params: dict, ModelClass):
    """根据参数创建矫正模型实例。"""
    return ModelClass(
        alpha_smoothing=params["alpha_smoothing"],
        pos_boost=params["pos_boost"],
        damping=params["damping"],
        max_iter=params.get("max_iter", 200),
        learning_rate=params.get("learning_rate", 0.05),
        max_depth=params.get("max_depth", 4),
    )


def build_setpoint_group_key(df, top_col=TOP_SETPOINT_COL, bot_col=BOT_SETPOINT_COL):
    """用 (Top_Min, Bot_Min) 精确组合做规格分组键。"""
    if top_col not in df.columns or bot_col not in df.columns:
        raise KeyError(f"缺少分组所需字段: {top_col} 或 {bot_col}")
    df = df.copy()
    df['Setpoint_Group_Key'] = list(zip(df[top_col], df[bot_col]))
    df['Setpoint_Group_Label'] = df.apply(
        lambda r: f"Top{r[top_col]}_Bot{r[bot_col]}", axis=1
    )
    return df


def summarize_setpoint_groups(df):
    """打印规格组样本量，区分达标组 / 跳过组。"""
    group_sizes = df.groupby('Setpoint_Group_Label').size().sort_values(ascending=False)
    valid = group_sizes[group_sizes >= MIN_GROUP_SAMPLES]
    skipped = group_sizes[group_sizes < MIN_GROUP_SAMPLES]
    print("\n==========================================")
    print("        [镀层规格分组样本量汇总]           ")
    print("==========================================")
    print(f"共 {len(group_sizes)} 个规格组，达标 {len(valid)} 个，跳过 {len(skipped)} 个")
    print(f"（达标阈值: >= {MIN_GROUP_SAMPLES} 条）\n")
    print(f"{'规格组':<30}{'样本数':>10}   状态")
    print("-" * 55)
    for label, size in valid.items():
        print(f"{label:<30}{size:>10}   建模")
    for label, size in skipped.items():
        print(f"{label:<30}{size:>10}   跳过")
    print("==========================================\n")
    return group_sizes


def get_feature_cols(surface: str) -> list:
    """与 fit_and_evaluate_surface 中 feature_cols 完全一致（+all_8 组合）。"""
    prefix = "Top" if surface == "Top" else "Bot"
    return [
        f"Tin Weight_Actual[g/m2]_GALV_WEIGHT_{prefix.upper()}_Avg",
        f"{prefix}_Current_Sum",
        f"{prefix}_Current_Per_Speed",
        f"{prefix}_Theoretical_Factor",
        "Speed[m/min]_Process_Avg",
        "Dimension_[mm]_Width",
        "Dimension_[mm]_Thickness",
        "Steel_Grade_Encoded",
    ]
