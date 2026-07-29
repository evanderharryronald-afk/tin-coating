import pandas as pd
from param_tuner import ParameterSensitivityTuner

if __name__ == "__main__":
    # 1. 加载数据
    clean_df = pd.read_excel("result/cleaned_data/cleaned_data.xlsx")

    # 2. 实例化调参工具
    tuner = ParameterSensitivityTuner(save_dir="result/param_tuning")

    # 3. 稳健网格参数候选（根据需要调节）
    damping_candidates = [0.4, 0.5,0.6, 0.8]
    pos_boost_candidates = [3.0, 3.5, 4.5,5.0, 5.5, 6.0]
    alpha_candidates = [0.8, 1.0]

    surfaces = ['Top', 'Bot']

    for surface in surfaces:
        surface_cn = '上' if surface == 'Top' else '下'
        print(f"\n==================== 开始评估【{surface_cn}表面】 ====================")

        # 运行包含正负残差拆分评估的安全网格搜索
        res_df = tuner.evaluate_grid_safe(
            clean_df,
            surface=surface,
            damping_list=damping_candidates,
            pos_boost_list=pos_boost_candidates,
            alpha_list=alpha_candidates
        )

        # 1. 绘制平行坐标全景图（同时穿过 MAE_pos 和 MAE_neg，查看参数平衡性）
        tuner.plot_parallel_coordinates(res_df, surface=surface, top_k_percent=0.25)

        # 2. 【核心修改】：绘制 MAE 与 RMSE 的 Pos/Neg 上下配对对比大图（一张图看清两侧演变）
        tuner.plot_paired_heatmaps(res_df, surface=surface, metric_prefix='MAE')
        tuner.plot_paired_heatmaps(res_df, surface=surface, metric_prefix='RMSE')
        tuner.plot_paired_heatmaps(res_df, surface=surface, metric_prefix='R2')

        # # 3. （可选）绘制整体拟合度指标热力图（如整体 MAE_total 或 R2_total）
        # tuner.plot_facet_heatmaps(res_df, surface=surface, metric='MAE_total')
        # tuner.plot_facet_heatmaps(res_df, surface=surface, metric='R2_total')

    print("\n完成！针对样本不均衡的正负子集配对对比图表已全部生成。")