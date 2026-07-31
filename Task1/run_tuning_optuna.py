import numpy as np
import os
import json
import argparse
import pandas as pd
import optuna
from optuna.visualization import (
    plot_pareto_front, plot_param_importances, plot_slice, plot_parallel_coordinate
)

from coating_model_by_group import (
    build_setpoint_group_key, fit_and_evaluate_surface, load_group_params,
    GROUP_PARAMS_PATH, DEFAULT_PARAMS
)
from data_cleaner import SteelDataCleaner

RESULT_DIR = "result/tuning"


def load_config(config_path):
    """加载 JSON 配置文件，支持默认兜底"""
    default_config = {
        "data_path": "result/cleaned_data/cleaned_data.xlsx",
        "min_samples": 200,
        "n_trials": 40,
        "groups": None
    }

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            user_config = json.load(f)
        default_config.update(user_config)
        print(f"[配置] 成功加载配置文件: {config_path}")
    else:
        print(f"[警告] 未找到配置文件 {config_path}，将使用内置默认配置")

    return default_config


def make_objective(group_df, surface, group_tag):
    def objective(trial):
        params = {
            "damping": trial.suggest_float("damping", 0.0, 1.0),
            "pos_boost": trial.suggest_float("pos_boost", 1.0, 8.0),
            "alpha_smoothing": trial.suggest_float("alpha_smoothing", 0.3, 1.0),
            "max_iter": trial.suggest_int("max_iter", 100, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
        }
        try:
            _, metrics, _ = fit_and_evaluate_surface(group_df, surface, params, group_tag=group_tag)
            rmse = metrics["RMSE_模型"]
            worst_mae = max(metrics["正偏差MAE_模型"], metrics["负偏差MAE_模型"])
        except Exception as e:
            return float("inf"), float("inf")

        return rmse, worst_mae

    return objective


def print_trial_result(study, trial):
    if trial.values is not None:
        rmse, worst_mae = trial.values
        print(f"[Trial {trial.number}] RMSE_模型={rmse:.4f}  最差方向MAE={worst_mae:.4f}  参数={trial.params}")
    else:
        print(f"[Trial {trial.number}] 失败/状态异常 (State: {trial.state})")


def tune_one(group_df, group_label, surface, n_trials):
    tag = f"{group_label}__{surface}"
    print(f"\n===== 开始调参: {tag}, trials={n_trials} =====")

    study = optuna.create_study(directions=["minimize", "minimize"], study_name=tag)
    study.optimize(
        make_objective(group_df, surface, group_label),
        n_trials=n_trials,
        show_progress_bar=True,
        callbacks=[print_trial_result],
    )

    out_dir = os.path.join(RESULT_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Pareto 前沿
    fig = plot_pareto_front(study, target_names=["RMSE_模型", "最差方向MAE"])
    fig.write_html(os.path.join(out_dir, "pareto_front.html"))

    # 2. 参数重要性
    fig_imp_rmse = plot_param_importances(study, target=lambda t: t.values[0], target_name="RMSE_模型")
    fig_imp_rmse.write_image(os.path.join(out_dir, "param_importance_rmse.png"))
    fig_imp_mae = plot_param_importances(study, target=lambda t: t.values[1], target_name="最差方向MAE")
    fig_imp_mae.write_image(os.path.join(out_dir, "param_importance_mae.png"))

    # 3. 单参数 slice 图
    fig_slice_rmse = plot_slice(study, target=lambda t: t.values[0], target_name="RMSE_模型")
    fig_slice_rmse.write_image(os.path.join(out_dir, "slice_rmse.png"))
    fig_slice_mae = plot_slice(study, target=lambda t: t.values[1], target_name="最差方向MAE")
    fig_slice_mae.write_image(os.path.join(out_dir, "slice_mae.png"))

    # 4. 参数耦合关系
    fig_parallel = plot_parallel_coordinate(study, target=lambda t: t.values[0], target_name="RMSE_模型")
    fig_parallel.write_image(os.path.join(out_dir, "parallel_coordinate.png"))

    # 5. 导出该规格组的 Pareto 前沿解
    best_trials_records = []
    for t in study.best_trials:
        rec = {
            "规格组": group_label,
            "表面": surface,
            "Tag": tag,
            "trial_number": t.number,
            "RMSE_模型": t.values[0],
            "最差方向MAE": t.values[1]
        }
        rec.update(t.params)
        best_trials_records.append(rec)

    best_trials_df = pd.DataFrame(best_trials_records).sort_values("RMSE_模型")
    best_trials_df.to_excel(os.path.join(out_dir, "best_trials.xlsx"), index=False)

    print(f"[完成] {tag} 调参结果已保存至: {out_dir}")
    print(best_trials_df.to_string(index=False))

    return study, out_dir, best_trials_df


def write_back_params(group_label, surface, trial_number, tag_dir):
    """人工看完图选好 trial_number 之后，调这个函数把参数写回 group_params.json"""
    best_trials_df = pd.read_excel(os.path.join(tag_dir, "best_trials.xlsx"))
    row = best_trials_df[best_trials_df["trial_number"] == trial_number]
    if row.empty:
        raise ValueError(f"trial_number {trial_number} 不在 best_trials 里，请检查 best_trials.xlsx")
    row = row.iloc[0]

    params = {k: row[k] for k in DEFAULT_PARAMS.keys()}
    for k in ["max_iter", "max_depth"]:
        params[k] = int(params[k])

    group_params = load_group_params()
    key = f"{group_label}__{surface}"
    group_params[key] = params

    with open(GROUP_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(group_params, f, ensure_ascii=False, indent=2)
    print(f"[写回完成] {key} -> {params}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="镀锌模型多规格组 Optuna 自动调参脚本")
    parser.add_argument("--config", type=str, default="optuna_tuning_config.json",
                        help="配置文件路径 (默认: optuna_tuning_config.json)")
    args = parser.parse_args()

    # 1. 加载 JSON 配置
    config = load_config(args.config)

    # 2. 读取清洗后的数据
    data_path = config["data_path"]
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}，请先运行数据清洗脚本！")

    clean_df = pd.read_excel(data_path)
    clean_df = build_setpoint_group_key(clean_df)

    # 3. 确定要调参的规格组列表
    group_sizes = clean_df.groupby('Setpoint_Group_Label').size()

    # 若 JSON 中 groups 为空/未传，则自动筛选满足 min_samples 的所有规格组
    target_labels = config["groups"] if config["groups"] is not None else \
        group_sizes[group_sizes >= config["min_samples"]].index.tolist()

    all_best_trials = []

    # 4. 遍历各个规格组进行调参
    for group_label in target_labels:
        if group_label not in clean_df['Setpoint_Group_Label'].values:
            print(f"[警告] 规格组 '{group_label}' 在数据集中未找到，跳过")
            continue

        group_df = clean_df[clean_df['Setpoint_Group_Label'] == group_label].copy()
        if len(group_df) < config["min_samples"]:
            print(f"[跳过] {group_label} 样本量 ({len(group_df)}) 不足 min_samples ({config['min_samples']})")
            continue

        for surface in ["Top", "Bot"]:
            _, _, best_trials_df = tune_one(group_df, group_label, surface, config["n_trials"])
            all_best_trials.append(best_trials_df)

    # 5. 汇总导出所有组的 Pareto 解集
    if all_best_trials:
        global_summary_df = pd.concat(all_best_trials, ignore_index=True)
        os.makedirs(RESULT_DIR, exist_ok=True)
        summary_path = os.path.join(RESULT_DIR, "all_best_trials_summary.xlsx")

        first_cols = ["规格组", "表面", "Tag", "trial_number", "RMSE_模型", "最差方向MAE"]
        other_cols = [c for c in global_summary_df.columns if c not in first_cols]
        global_summary_df = global_summary_df[first_cols + other_cols]

        global_summary_df.to_excel(summary_path, index=False)
        print(f"\n==========================================")
        print(f"[汇总成功] 所有规格组的 Pareto 解集已合并保存至:\n -> {summary_path}")
        print(f"==========================================\n")