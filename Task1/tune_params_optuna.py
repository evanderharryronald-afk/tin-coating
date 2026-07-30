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
        except Exception as e:
            # 出问题的参数组合直接判为最差，避免整个 study 崩掉
            return float("inf"), float("inf")

        rmse = metrics["RMSE_模型"]
        worst_mae = max(metrics["正偏差MAE_模型"] or float("inf"),
                         metrics["负偏差MAE_模型"] or float("inf"))
        return rmse, worst_mae
    return objective

def print_trial_result(study, trial):
    rmse, worst_mae = trial.values
    print(f"[Trial {trial.number}] RMSE_模型={rmse:.4f}  最差方向MAE={worst_mae:.4f}  参数={trial.params}")

def tune_one(group_df, group_label, surface, n_trials):
    tag = f"{group_label}__{surface}"
    print(f"\n===== 开始调参: {tag}, trials={n_trials} =====")

    study = optuna.create_study(directions=["minimize", "minimize"], study_name=tag)
    study.optimize(
        make_objective(group_df, surface, group_label),
        n_trials=n_trials,
        show_progress_bar=True,
        callbacks=[print_trial_result],  # 加这一行
    )

    out_dir = os.path.join(RESULT_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Pareto 前沿（交互式 html，重点看这张选参数）
    fig = plot_pareto_front(study, target_names=["RMSE_模型", "最差方向MAE"])
    fig.write_html(os.path.join(out_dir, "pareto_front.html"))

    # 2. 参数重要性（分别对两个目标）
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

    # 5. 导出 Pareto 前沿上所有非支配解的具体参数值，方便对着图查数值
    best_trials_records = []
    for t in study.best_trials:
        rec = {"trial_number": t.number, "RMSE_模型": t.values[0], "最差方向MAE": t.values[1]}
        rec.update(t.params)
        best_trials_records.append(rec)
    best_trials_df = pd.DataFrame(best_trials_records).sort_values("RMSE_模型")
    best_trials_df.to_excel(os.path.join(out_dir, "best_trials.xlsx"), index=False)

    print(f"[完成] {tag} 调参结果已保存至: {out_dir}")
    print(best_trials_df.to_string(index=False))
    return study, out_dir


def write_back_params(group_label, surface, trial_number, tag_dir):
    """人工看完图选好 trial_number 之后，调这个函数把参数写回 group_params.json"""
    best_trials_df = pd.read_excel(os.path.join(tag_dir, "best_trials.xlsx"))
    row = best_trials_df[best_trials_df["trial_number"] == trial_number]
    if row.empty:
        raise ValueError(f"trial_number {trial_number} 不在 best_trials 里，请检查 best_trials.xlsx")
    row = row.iloc[0]

    params = {k: row[k] for k in DEFAULT_PARAMS.keys()}
    # int 字段防止被 pandas 读成 float
    for k in ["max_iter", "max_depth"]:
        params[k] = int(params[k])

    group_params = load_group_params()
    key = f"{group_label}__{surface}"
    group_params[key] = params

    with open(GROUP_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(group_params, f, ensure_ascii=False, indent=2)
    print(f"[写回完成] {key} -> {params}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", nargs="*", default=None, help="要调参的规格组标签列表，不传则调全部达标组")
    parser.add_argument("--n_trials", type=int, default=80)
    parser.add_argument("--min_samples", type=int, default=200)
    args = parser.parse_args()

    # raw_df = pd.read_excel("result/merged_data/merged_result_latest.xlsx")
    # cleaner = SteelDataCleaner(min_speed=20.0, max_range_abs=0.4, max_range_ratio=0.3, mad_factor=3.0)
    # clean_df = cleaner.process(raw_df,
    #                             clean_save_path="result/cleaned_data/cleaned_data.xlsx",
    #                             filtered_save_path="result/cleaned_data/filtered_outliers.xlsx")
    clean_df= pd.read_excel("result/cleaned_data/cleaned_data.xlsx")
    clean_df = build_setpoint_group_key(clean_df)

    group_sizes = clean_df.groupby('Setpoint_Group_Label').size()
    target_labels = args.groups if args.groups else \
        group_sizes[group_sizes >= args.min_samples].index.tolist()

    for group_label in target_labels:
        group_df = clean_df[clean_df['Setpoint_Group_Label'] == group_label].copy()
        if len(group_df) < args.min_samples:
            print(f"[跳过] {group_label} 样本量不足 {args.min_samples}")
            continue
        for surface in ["Top", "Bot"]:
            tune_one(group_df, group_label, surface, args.n_trials)