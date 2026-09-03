import numpy as np
import os
import json
import argparse
import pandas as pd
import optuna
from optuna.samplers import NSGAIISampler
from optuna.visualization import (
    plot_pareto_front, plot_param_importances, plot_slice, plot_parallel_coordinate
)

from coating_model_by_group import (
    build_setpoint_group_key, fit_and_evaluate_surface
)

# 基础结果目录
BASE_RESULT_DIR = "result/tuning"


def load_config(config_path):
    """加载 JSON 配置文件，支持默认兜底"""
    default_config = {
        "data_path": "result/data/feature_engineered_data/featured_data.xlsx",
        "min_samples": 200,
        "n_trials": 40,
        "groups": None,
        "mode": "per_group",  # "per_group" 或 "global"
        # 搜索结束后，是否对 Pareto 前沿解额外评估一次测试集指标（仅记录，不参与选参）
        "eval_test_after_tuning": True,
        # 数据切分比例（与 coating_model_by_group 保持一致）
        "train_ratio": 0.65,
        "val_ratio": 0.20,
    }

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            user_config = json.load(f)
        default_config.update(user_config)
        print(f"[配置] 成功加载配置文件: {config_path}")
    else:
        print(f"[警告] 未找到配置文件 {config_path}，将使用内置默认配置")

    # 验证 mode 参数
    if default_config["mode"] not in ["per_group", "global"]:
        print(f"[警告] mode 参数 '{default_config['mode']}' 无效，已自动切换为 'per_group'")
        default_config["mode"] = "per_group"

    return default_config


def make_objective(group_df, surface, group_tag, train_ratio=0.65, val_ratio=0.20):
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
            _, metrics, _ = fit_and_evaluate_surface(
                group_df, surface, params, group_tag=group_tag,
                train_ratio=train_ratio, val_ratio=val_ratio
            )

            # 使用验证集指标（避免数据泄露）
            rmse = metrics["RMSE_模型_验证"]
            worst_mae = max(
                metrics["正偏差MAE_模型_验证"],
                metrics["负偏差MAE_模型_验证"]
            )

            # 处理可能的 NaN（某一方向样本为 0 时）
            if np.isnan(rmse):
                rmse = float("inf")
            if np.isnan(worst_mae):
                worst_mae = float("inf")

        except Exception as e:
            import traceback
            print(f"\n{'=' * 60}")
            print(f"[ERROR] 调参失败!")
            print(f"  规格组: {group_tag}")
            print(f"  表面: {surface}")
            print(f"  参数: {params}")
            print(f"  异常类型: {type(e).__name__}")
            print(f"  异常信息: {e}")
            traceback.print_exc()
            print(f"{'=' * 60}\n")
            return float("inf"), float("inf")

        return rmse, worst_mae

    return objective


def print_trial_result(study, trial):
    if trial.values is not None:
        rmse, worst_mae = trial.values
        print(
            f"[Trial {trial.number}] RMSE_模型_验证={rmse:.4f}  "
            f"最差方向MAE_验证={worst_mae:.4f}  参数={trial.params}"
        )
    else:
        print(f"[Trial {trial.number}] 失败/状态异常 (State: {trial.state})")


def evaluate_best_trials_on_test(study, group_df, group_label, surface,
                                 train_ratio=0.65, val_ratio=0.20):
    """
    搜索结束后，对 Pareto 前沿上的每个解额外评估一次测试集指标。
    仅用于观察泛化差距，结果不参与任何超参选择。
    """
    records = []
    print(f"\n----- 开始对 Pareto 前沿解评估测试集指标 ({group_label}__{surface}) -----")

    for t in study.best_trials:
        params = t.params
        try:
            _, metrics, _ = fit_and_evaluate_surface(
                group_df, surface, params, group_tag=group_label,
                train_ratio=train_ratio, val_ratio=val_ratio
            )

            test_rmse = metrics.get("RMSE_模型_测试", np.nan)
            test_worst_mae = max(
                metrics.get("正偏差MAE_模型_测试", np.nan),
                metrics.get("负偏差MAE_模型_测试", np.nan)
            )
            # 若某一方向无样本，max 可能仍是 NaN，保持 NaN 即可

            rec = {
                "规格组": group_label,
                "表面": surface,
                "Tag": f"{group_label}__{surface}",
                "trial_number": t.number,
                # 验证集（Optuna 优化目标）
                "RMSE_模型_验证": t.values[0],
                "最差方向MAE_验证": t.values[1],
                "正偏差MAE_模型_验证": metrics.get("正偏差MAE_模型_验证", np.nan),
                "负偏差MAE_模型_验证": metrics.get("负偏差MAE_模型_验证", np.nan),
                "MAE_模型_验证": metrics.get("MAE_模型_验证", np.nan),
                "R2_模型_验证": metrics.get("R2_模型_验证", np.nan),

                # 测试集（仅观察）
                "RMSE_模型_测试": test_rmse,
                "最差方向MAE_测试": test_worst_mae,
                "正偏差MAE_模型_测试": metrics.get("正偏差MAE_模型_测试", np.nan),
                "负偏差MAE_模型_测试": metrics.get("负偏差MAE_模型_测试", np.nan),
                "MAE_模型_测试": metrics.get("MAE_模型_测试", np.nan),
                "R2_模型_测试": metrics.get("R2_模型_测试", np.nan),
            }
            rec.update(params)
            records.append(rec)

            print(
                f"  Trial {t.number}: "
                f"Val RMSE={t.values[0]:.4f}, Val worstMAE={t.values[1]:.4f} | "
                f"Test RMSE={test_rmse:.4f}, Test worstMAE={test_worst_mae:.4f}"
            )
        except Exception as e:
            print(f"  [警告] Trial {t.number} 测试集评估失败: {e}")
            continue

    print("----- 测试集评估结束 -----\n")
    return records


def tune_one(group_df, group_label, surface, n_trials, result_subdir,
             eval_test_after_tuning=True, train_ratio=0.65, val_ratio=0.20):
    """
    执行单次调参

    Args:
        group_df: 数据
        group_label: 组标签
        surface: 表面 (Top/Bot)
        n_trials: 试验次数
        result_subdir: 结果子目录 (如 "global" 或 "per_group")
        eval_test_after_tuning: 搜索结束后是否对 Pareto 解评估测试集（默认 True）
        train_ratio / val_ratio: 与 fit_and_evaluate_surface 保持一致
    """
    tag = f"{group_label}__{surface}"
    print(f"\n===== 开始调参: {tag}, trials={n_trials} =====")
    print(f"  切分比例: train={train_ratio:.2f}, val={val_ratio:.2f}, "
          f"test={1.0 - train_ratio - val_ratio:.2f}")
    print(f"  搜索结束后评估测试集: {'是' if eval_test_after_tuning else '否'}")

    # 固定随机种子使得结果可复现
    study = optuna.create_study(
        directions=["minimize", "minimize"],
        study_name=tag,
        sampler=NSGAIISampler(seed=42)
    )

    study.optimize(
        make_objective(group_df, surface, group_label,
                       train_ratio=train_ratio, val_ratio=val_ratio),
        n_trials=n_trials,
        show_progress_bar=True,
        callbacks=[print_trial_result],
    )

    # 结果保存到对应的子目录
    out_dir = os.path.join(BASE_RESULT_DIR, result_subdir, tag)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Pareto 前沿（目标名称统一为验证集）
    fig = plot_pareto_front(
        study,
        target_names=["RMSE_模型_验证", "最差方向MAE_验证"]
    )
    fig.write_html(os.path.join(out_dir, "pareto_front.html"))

    # 2. 参数重要性
    fig_imp_rmse = plot_param_importances(
        study, target=lambda t: t.values[0], target_name="RMSE_模型_验证"
    )
    fig_imp_rmse.write_image(os.path.join(out_dir, "param_importance_rmse.png"))
    fig_imp_mae = plot_param_importances(
        study, target=lambda t: t.values[1], target_name="最差方向MAE_验证"
    )
    fig_imp_mae.write_image(os.path.join(out_dir, "param_importance_mae.png"))

    # 3. 单参数 slice 图
    fig_slice_rmse = plot_slice(
        study, target=lambda t: t.values[0], target_name="RMSE_模型_验证"
    )
    fig_slice_rmse.write_image(os.path.join(out_dir, "slice_rmse.png"))
    fig_slice_mae = plot_slice(
        study, target=lambda t: t.values[1], target_name="最差方向MAE_验证"
    )
    fig_slice_mae.write_image(os.path.join(out_dir, "slice_mae.png"))

    # 4. 参数耦合关系
    fig_parallel = plot_parallel_coordinate(
        study, target=lambda t: t.values[0], target_name="RMSE_模型_验证"
    )
    fig_parallel.write_image(os.path.join(out_dir, "parallel_coordinate.png"))

    # 5. 导出 Pareto 前沿解（验证集指标）
    best_trials_records = []
    for t in study.best_trials:
        rec = {
            "规格组": group_label,
            "表面": surface,
            "Tag": tag,
            "trial_number": t.number,
            "RMSE_模型_验证": t.values[0],
            "最差方向MAE_验证": t.values[1],
        }
        rec.update(t.params)
        best_trials_records.append(rec)

    best_trials_df = pd.DataFrame(best_trials_records).sort_values("RMSE_模型_验证")
    best_trials_df.to_excel(os.path.join(out_dir, "best_trials.xlsx"), index=False)

    # 6. （可选）搜索结束后评估测试集
    if eval_test_after_tuning and len(study.best_trials) > 0:
        test_records = evaluate_best_trials_on_test(
            study, group_df, group_label, surface,
            train_ratio=train_ratio, val_ratio=val_ratio
        )
        if test_records:
            test_df = pd.DataFrame(test_records).sort_values("RMSE_模型_验证")
            test_path = os.path.join(out_dir, "best_trials_with_test.xlsx")
            test_df.to_excel(test_path, index=False)
            print(f"[完成] 含测试集指标的 Pareto 结果已保存至: {test_path}")

            # 用带测试集的结果覆盖返回值，方便汇总
            best_trials_df = test_df

    print(f"[完成] {tag} 调参结果已保存至: {out_dir}")
    print(best_trials_df.to_string(index=False))

    return study, out_dir, best_trials_df


def save_summary(all_best_trials, mode):
    """
    保存汇总文件

    Args:
        all_best_trials: 所有 best_trials_df 的列表
        mode: 模式名称 ("global" 或 "per_group")
    """
    if not all_best_trials:
        return None

    global_summary_df = pd.concat(all_best_trials, ignore_index=True)

    # 保存到对应的模式目录
    summary_dir = os.path.join(BASE_RESULT_DIR, mode)
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, "all_best_trials_summary.xlsx")

    # 优先排列的列（兼容有/无测试集指标两种情况）
    preferred = [
        "规格组", "表面", "Tag", "trial_number",
        # 验证集
        "RMSE_模型_验证", "最差方向MAE_验证",
        "正偏差MAE_模型_验证", "负偏差MAE_模型_验证",
        "MAE_模型_验证", "R2_模型_验证",
        # 测试集
        "RMSE_模型_测试", "最差方向MAE_测试",
        "正偏差MAE_模型_测试", "负偏差MAE_模型_测试",
        "MAE_模型_测试", "R2_模型_测试",
    ]
    first_cols = [c for c in preferred if c in global_summary_df.columns]
    other_cols = [c for c in global_summary_df.columns if c not in first_cols]
    global_summary_df = global_summary_df[first_cols + other_cols]

    global_summary_df.to_excel(summary_path, index=False)

    print(f"\n{'=' * 50}")
    print(f"[汇总成功] {mode} 模式调参结果已保存至:\n -> {summary_path}")
    print(f"{'=' * 50}\n")

    return summary_path


def run_global_tuning(data_path, config):
    """对所有样本进行统一调参（不分规格组）"""
    clean_df = pd.read_excel(data_path)

    print("\n" + "=" * 60)
    print("【全局调参模式】使用所有样本数据训练一个通用模型")
    print(f"总样本数: {len(clean_df)}")
    print("=" * 60 + "\n")

    GLOBAL_LABEL = "ALL_DATA"
    all_best_trials = []

    for surface in ["Top", "Bot"]:
        _, _, best_trials_df = tune_one(
            clean_df, GLOBAL_LABEL, surface,
            config["n_trials"],
            result_subdir="global",
            eval_test_after_tuning=config.get("eval_test_after_tuning", True),
            train_ratio=config.get("train_ratio", 0.65),
            val_ratio=config.get("val_ratio", 0.20),
        )
        all_best_trials.append(best_trials_df)

    save_summary(all_best_trials, mode="global")


def run_per_group_tuning(data_path, config):
    """按规格组分别调参（原有逻辑）"""
    clean_df = pd.read_excel(data_path)
    clean_df = build_setpoint_group_key(clean_df)

    if 'Setpoint_Group_Label' not in clean_df.columns:
        raise ValueError("数据中缺少 'Setpoint_Group_Label' 列，请检查 build_setpoint_group_key 函数")

    print("\n" + "=" * 60)
    print("【分组调参模式】按规格组分别训练模型")
    print("=" * 60 + "\n")

    group_sizes = clean_df.groupby('Setpoint_Group_Label').size()

    target_labels = config["groups"] if config["groups"] is not None else \
        group_sizes[group_sizes >= config["min_samples"]].index.tolist()

    if not target_labels:
        print(f"[警告] 没有满足条件的规格组 (min_samples={config['min_samples']})")
        return

    print(f"目标规格组 ({len(target_labels)} 个): {target_labels}")

    all_best_trials = []

    for group_label in target_labels:
        if group_label not in clean_df['Setpoint_Group_Label'].values:
            print(f"[警告] 规格组 '{group_label}' 在数据集中未找到，跳过")
            continue

        group_df = clean_df[clean_df['Setpoint_Group_Label'] == group_label].copy()
        if len(group_df) < config["min_samples"]:
            print(f"[跳过] {group_label} 样本量 ({len(group_df)}) 不足 min_samples ({config['min_samples']})")
            continue

        for surface in ["Top", "Bot"]:
            _, _, best_trials_df = tune_one(
                group_df, group_label, surface,
                config["n_trials"],
                result_subdir="per_group",
                eval_test_after_tuning=config.get("eval_test_after_tuning", True),
                train_ratio=config.get("train_ratio", 0.65),
                val_ratio=config.get("val_ratio", 0.20),
            )
            all_best_trials.append(best_trials_df)

    save_summary(all_best_trials, mode="per_group")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="镀锌模型多规格组 Optuna 自动调参脚本")
    parser.add_argument(
        "--config", type=str, default="optuna_tuning_config_global.json",
        help="配置文件路径 (默认: optuna_tuning_config_global.json)"
    )
    # parser.add_argument(
    #     "--config", type=str, default="optuna_tuning_config_grouped.json",
    #     help="配置文件路径 (默认: optuna_tuning_config_grouped.json)"
    # )
    args = parser.parse_args()

    # 1. 加载 JSON 配置
    config = load_config(args.config)
    print(f"[配置] 当前模式: {config['mode']}")
    print(f"[配置] 搜索后评估测试集: {config.get('eval_test_after_tuning', True)}")

    # 2. 验证数据文件存在
    data_path = config["data_path"]
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}，请先运行数据清洗和特征工程脚本！")

    # 3. 根据模式执行调参
    if config["mode"] == "global":
        run_global_tuning(data_path, config)
    else:  # per_group
        run_per_group_tuning(data_path, config)
