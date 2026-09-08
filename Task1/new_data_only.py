import os
import pandas as pd
from correlation_analyzer import SurfaceCorrelationAnalyzer
from eda_analyzer import SurfaceEDAAnalyzer  # 引入 EDA 分析模块


def main():
    # 1. 相对路径定义
    excel_path = os.path.join("data", "new_data", "V_PCOILDATA_TIN 202607-202608.xlsx")
    output_base_dir = os.path.join("result", "new_data", "correlation_result")
    eda_output_base_dir = os.path.join("result", "new_data", "eda_result")

    # 2. 确保输出目录及其子目录存在
    top_save_dir = os.path.join(output_base_dir, "Top")
    bot_save_dir = os.path.join(output_base_dir, "Bot")
    os.makedirs(top_save_dir, exist_ok=True)
    os.makedirs(bot_save_dir, exist_ok=True)

    top_eda_dir = os.path.join(eda_output_base_dir, "Top")
    bot_eda_dir = os.path.join(eda_output_base_dir, "Bot")

    # 3. 读取 Excel 数据（header=1 表示读取第2行英文列名，跳过第1行中文列名）
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"找不到指定的数据文件：{excel_path}")

    df = pd.read_excel(excel_path, header=1)
    df.columns = df.columns.astype(str).str.strip()  # 去除列名前后多余空格

    # 4. 实例化分析类
    analyzer = SurfaceCorrelationAnalyzer(default_save_dir=output_base_dir)
    eda_analyzer = SurfaceEDAAnalyzer(default_save_dir=eda_output_base_dir)  # 实例化 EDA 分析器

    # 5. 上表面分析 (Top)
    print("\n" + "=" * 50)
    print("开始上表面分析（相关性 + EDA）")
    print("=" * 50)

    # 5.1 相关性分析
    analyzer.analyze_custom_features(
        df=df,
        target_col='GALV_WEIGHT_TOP_ACT',  # 上表面镀层实际克数
        feature_cols=['GALV_EFF_TOP', 'GALV_MAX_CUR_DEN_TOP'],  # 电镀效率 & 电流密度
        title_prefix="上表面",
        save_dir=top_save_dir,
        corr_method='both',
        compute_mi=True,
        compute_dcor=True
    )

    # 5.2 EDA 分析
    # 注意：需确保 df 中包含用于计算偏差的列，或直接将 target_col 作为 delta_col 进行分布分析
    eda_analyzer.analyze(
        df=df,
        delta_col='GALV_WEIGHT_TOP_ACT',  # 这里将目标特征作为要观察分布的主要变量，也可换成残差列/测量偏差列
        feature_cols=['GALV_EFF_TOP', 'GALV_MAX_CUR_DEN_TOP'],  # 需要分析分布与关系的特征列表
        save_dir=top_eda_dir,
        plot_univariate=True,  # 画单变量分布图[cite: 1]
        plot_vs_delta=True,  # 画特征 vs 目标/偏差散点趋势图[cite: 1]
        enable_by_group=False  # 若新数据没有规格分组列（group_col），可关闭分组分析[cite: 1]
    )

    # 6. 下表面分析 (Bot)
    print("\n" + "=" * 50)
    print("开始下表面分析（相关性 + EDA）")
    print("=" * 50)

    # 6.1 相关性分析
    analyzer.analyze_custom_features(
        df=df,
        target_col='GALV_WEIGHT_BOT_ACT',  # 下表面镀层实际克数
        feature_cols=['GALV_EFF_BOT', 'GALV_MAX_CUR_DEN_BOT'],  # 电镀效率 & 电流密度
        title_prefix="下表面",
        save_dir=bot_save_dir,
        corr_method='both',
        compute_mi=True,
        compute_dcor=True
    )

    # 6.2 EDA 分析[cite: 1]
    eda_analyzer.analyze(
        df=df,
        delta_col='GALV_WEIGHT_BOT_ACT',
        feature_cols=['GALV_EFF_BOT', 'GALV_MAX_CUR_DEN_BOT'],
        save_dir=bot_eda_dir,
        plot_univariate=True,
        plot_vs_delta=True,
        enable_by_group=False
    )

    print(f"\n全部分析完成！相关性结果保存至：{output_base_dir}，EDA结果保存至：{eda_output_base_dir}")


if __name__ == '__main__':
    main()