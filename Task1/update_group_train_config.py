import pandas as pd
import json
import os

# 定义路径
EXCEL_PATH = "result/tuning/all_best_trials_summary.xlsx"  # 你的 CSV/Excel 导出文件路径
JSON_PATH = "group_params_optimum_for_each.json"  # 待更新的 JSON 配置文件路径
OUTPUT_JSON_PATH = "group_params_optimum_for_each.json"  # 输出的 JSON 配置文件路径


def update_json_parameters(excel_path, json_path, output_path):
    # 1. 读取 Excel 文件
    if not os.path.exists(excel_path):
        print(f"错误: 找不到数据文件 {excel_path}")
        return

    df = pd.read_excel(excel_path)

    # 2. 按 Tag 分组，并筛选出 '最差方向MAE' 最小的记录
    # 先按照 最差方向MAE 升序排序，然后每个 Tag 取第一行 (head(1))
    best_trials_df = (
        df.sort_values(by="最差方向MAE", ascending=True)
        .groupby("Tag", as_index=False)
        .first()
    )

    print(f"成功筛选出 {len(best_trials_df)} 个 Tag 的最佳参数。")

    # 3. 读取原始 JSON 配置文件
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        print(f"警告: 找不到 JSON 文件 {json_path}，将创建新的配置结构。")
        config = {"group_params_override": {}}

    if "group_params_override" not in config:
        config["group_params_override"] = {}

    # 需要提取并更新的超参数字段列表
    param_fields = ["damping", "pos_boost", "alpha_smoothing", "max_iter", "learning_rate", "max_depth"]

    # 4. 遍历筛选后的最佳参数并更新到 config 字典中
    updated_count = 0
    for _, row in best_trials_df.iterrows():
        tag = row["Tag"]

        # 构建参数字典并转换为合适的数据类型
        params_dict = {
            "damping": round(float(row["damping"]), 4),
            "pos_boost": round(float(row["pos_boost"]), 4),
            "alpha_smoothing": round(float(row["alpha_smoothing"]), 4),
            "max_iter": int(row["max_iter"]),
            "learning_rate": round(float(row["learning_rate"]), 4),
            "max_depth": int(row["max_depth"])
        }

        # 更新 JSON 结构
        config["group_params_override"][tag] = params_dict
        updated_count += 1

    # 5. 保存回 JSON 文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"成功将 {updated_count} 个 Tag 的最优参数写入 {output_path}！")


if __name__ == "__main__":
    update_json_parameters(EXCEL_PATH, JSON_PATH, OUTPUT_JSON_PATH)