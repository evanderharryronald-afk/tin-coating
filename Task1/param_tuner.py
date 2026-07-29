import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 设置画图支持中文与负号
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def compute_direction_sample_weight_fast(y_delta_vals, pos_boost=1.0, damping=0.0):
    """矢量化样本权重计算"""
    if damping <= 0:
        return np.ones(len(y_delta_vals), dtype=np.float64)

    pos_mask = y_delta_vals > 0
    neg_mask = y_delta_vals < 0
    n_pos = np.count_nonzero(pos_mask)
    n_neg = np.count_nonzero(neg_mask)
    n_total = n_pos + n_neg

    weights = np.ones(len(y_delta_vals), dtype=np.float64)
    if n_pos > 0:
        full_balance_pos = n_total / (2.0 * n_pos)
        weights[pos_mask] = (1 - damping) * 1.0 + damping * full_balance_pos * pos_boost
    if n_neg > 0:
        full_balance_neg = n_total / (2.0 * n_neg)
        weights[neg_mask] = (1 - damping) * 1.0 + damping * full_balance_neg

    return weights


class ResidualCorrectionModel:
    """轻量化残差矫正模型"""

    def __init__(self, monotonic_feature_idx=None, alpha_smoothing=0.7,
                 pos_boost=1.0, damping=0.0):
        self.alpha_smoothing = alpha_smoothing
        self.pos_boost = pos_boost
        self.damping = damping
        self.monotonic_feature_idx = monotonic_feature_idx
        self.model = None

    def _build_model(self, n_features):
        monotonic_cst = None
        if self.monotonic_feature_idx is not None:
            monotonic_cst = [0] * n_features
            monotonic_cst[self.monotonic_feature_idx] = -1

        self.model = HistGradientBoostingRegressor(
            max_iter=60,
            learning_rate=0.08,
            max_depth=4,
            loss='absolute_error',
            monotonic_cst=monotonic_cst,
            random_state=42
        )

    def fit(self, X_values, y_delta_values):
        self._build_model(n_features=X_values.shape[1])
        sample_weight = compute_direction_sample_weight_fast(
            y_delta_values, pos_boost=self.pos_boost, damping=self.damping
        )
        self.model.fit(X_values, y_delta_values, sample_weight=sample_weight)

    def predict_smooth(self, X_df, online_actual):
        predicted_delta_raw = self.model.predict(X_df.values)
        delta_series = pd.Series(predicted_delta_raw, index=X_df.index)
        predicted_delta_smooth = delta_series.ewm(alpha=self.alpha_smoothing).mean()
        final_pred = online_actual + predicted_delta_smooth
        return final_pred, predicted_delta_smooth


class ParameterSensitivityTuner:
    """支持正负残差子集分步拆分评估的细粒度调参工具"""

    def __init__(self, save_dir="result/param_tuning"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def evaluate_grid_safe(self, df, surface='Top',
                           damping_list=[0.0, 0.2, 0.4, 0.6, 0.8],
                           pos_boost_list=[1.0, 2.0, 3.0, 4.5, 6.0],
                           alpha_list=[0.2, 0.4, 0.6, 0.8, 1.0]):
        prefix = 'Top' if surface == 'Top' else 'Bot'
        surface_cn = '上' if surface == 'Top' else '下'

        # 特征工程
        speed_col = 'Speed[m/min]_Process_Avg'
        current_col = f'{prefix}_Current_Sum'
        df_copy = df.copy()
        df_copy[f'{prefix}_Current_Per_Speed'] = df_copy[current_col] / (df_copy[speed_col] + 1e-5)

        online_col = f'Tin Weight_Actual[g/m2]_GALV_WEIGHT_{prefix.upper()}_Avg'
        feature_cols = [
            online_col,
            current_col,
            f'{prefix}_Current_Per_Speed',
            f'{prefix}_Theoretical_Factor',
            speed_col,
            'Dimension_[mm]_Width',
            'Dimension_[mm]_Thickness',
            'Steel_Grade_Encoded'
        ]
        online_feature_idx = feature_cols.index(online_col)

        X = df_copy[feature_cols]
        delta_col = f'{prefix}_Delta'
        y_delta = df_copy[delta_col]
        online_actual = df_copy[online_col]
        y_true = df_copy[f'{surface_cn}表面镀层重量A(XA1_0)']

        # 时序划分
        X_train, X_test, y_delta_train, y_delta_test, actual_train, actual_test, y_true_train, y_true_test = \
            train_test_split(X, y_delta, online_actual, y_true, test_size=0.2, shuffle=False)

        X_train_vals = X_train.values
        y_delta_train_vals = y_delta_train.values

        y_true_vals = y_true_test.values
        actual_vals = actual_test.values
        raw_res = y_true_vals - actual_vals

        # 拆分掩码：在线偏低 (Delta > 0) 与 在线偏高 (Delta < 0)
        mask_pos = (raw_res > 0)
        mask_neg = (raw_res < 0)

        # 计算原始在线仪表的基准指标（用于衡量改进幅度）
        raw_mae_pos = np.mean(np.abs(raw_res[mask_pos])) if np.any(mask_pos) else np.nan
        raw_mae_neg = np.mean(np.abs(raw_res[mask_neg])) if np.any(mask_neg) else np.nan
        raw_rmse_pos = np.sqrt(np.mean(raw_res[mask_pos] ** 2)) if np.any(mask_pos) else np.nan
        raw_rmse_neg = np.sqrt(np.mean(raw_res[mask_neg] ** 2)) if np.any(mask_neg) else np.nan

        total_combos = len(damping_list) * len(pos_boost_list) * len(alpha_list)
        print(f"\n>>>> 正在安全搜索【{surface_cn}表面】，共计 {total_combos} 组组合 (支持正负残差分立评估)...")

        results = []
        count = 0

        for d in damping_list:
            for pb in pos_boost_list:
                for alpha in alpha_list:
                    count += 1
                    if count % 25 == 0 or count == total_combos:
                        print(f"   进度: {count}/{total_combos} ({(count / total_combos) * 100:.0f}%)...")

                    model = ResidualCorrectionModel(
                        monotonic_feature_idx=online_feature_idx,
                        alpha_smoothing=alpha,
                        pos_boost=pb,
                        damping=d
                    )
                    model.fit(X_train_vals, y_delta_train_vals)
                    pred_series, _ = model.predict_smooth(X_test, actual_test)
                    pred_vals = pred_series.values

                    # 1. 整体指标
                    r2_total = r2_score(y_true_vals, pred_vals)
                    rmse_total = np.sqrt(mean_squared_error(y_true_vals, pred_vals))
                    mae_total = mean_absolute_error(y_true_vals, pred_vals)

                    # 2. 正向/负向残差子集独立评估
                    model_res = y_true_vals - pred_vals

                    # 【在线偏低 (pos)】子集
                    res_pos = model_res[mask_pos]
                    y_true_pos = y_true_vals[mask_pos]
                    pred_pos = pred_vals[mask_pos]
                    mae_pos = np.mean(np.abs(res_pos)) if len(res_pos) > 0 else np.nan
                    rmse_pos = np.sqrt(np.mean(res_pos ** 2)) if len(res_pos) > 0 else np.nan
                    r2_pos = r2_score(y_true_pos, pred_pos) if len(res_pos) > 1 else np.nan

                    # 【在线偏高 (neg)】子集
                    res_neg = model_res[mask_neg]
                    y_true_neg = y_true_vals[mask_neg]
                    pred_neg = pred_vals[mask_neg]
                    mae_neg = np.mean(np.abs(res_neg)) if len(res_neg) > 0 else np.nan
                    rmse_neg = np.sqrt(np.mean(res_neg ** 2)) if len(res_neg) > 0 else np.nan
                    r2_neg = r2_score(y_true_neg, pred_neg) if len(res_neg) > 1 else np.nan

                    # 3. 衡量指标：是否实现正负双向同时优化
                    is_both_improved = (mae_pos < raw_mae_pos) and (mae_neg < raw_mae_neg)

                    # 综合不平衡度损耗（平衡得分）：两边 MAE 的极差/最大值，数值越小越平衡
                    balance_gap = abs(mae_pos - mae_neg)

                    results.append({
                        'surface': surface,
                        'damping': d,
                        'pos_boost': pb,
                        'alpha_smoothing': alpha,

                        # 整体指标
                        'MAE_total': mae_total,
                        'RMSE_total': rmse_total,
                        'R2_total': r2_total,

                        # 在线偏低 (pos, Delta>0) 指标
                        'MAE_pos': mae_pos,
                        'RMSE_pos': rmse_pos,
                        'R2_pos': r2_pos,

                        # 在线偏高 (neg, Delta<0) 指标
                        'MAE_neg': mae_neg,
                        'RMSE_neg': rmse_neg,
                        'R2_neg': r2_neg,

                        # 判定逻辑
                        'Is_Both_Improved': is_both_improved,
                        'Balance_Gap': balance_gap
                    })

        res_df = pd.DataFrame(results)

        # 排序策略：优先挑选双向都优化的组合，在此基础上以整体 MAE 排序
        res_df = res_df.sort_values(
            by=['Is_Both_Improved', 'MAE_total', 'Balance_Gap'],
            ascending=[False, True, True]
        ).reset_index(drop=True)

        csv_path = os.path.join(self.save_dir, f"grid_search_{surface}.csv")
        res_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        best_row = res_df.iloc[0]
        print(f"✅ 【{surface_cn}表面】搜索完成！最优参数组合：")
        print(
            f"   - 参数: damping={best_row['damping']}, pos_boost={best_row['pos_boost']}, alpha={best_row['alpha_smoothing']}")
        print(f"   - 整体 MAE: {best_row['MAE_total']:.4f} (R²: {best_row['R2_total']:.4f})")
        print(f"   - 偏低样本 MAE: {best_row['MAE_pos']:.4f} | 偏高样本 MAE: {best_row['MAE_neg']:.4f}")
        print(f"   - 是否实现正负双向同时改进: {'是' if best_row['Is_Both_Improved'] else '否'}")

        return res_df

    def plot_parallel_coordinates(self, res_df, surface='Top', top_k_percent=0.3):
        """支持同时展示正负样本 MAE 影响的平行坐标全景图"""
        surface_cn = '上' if surface == 'Top' else '下'

        # 筛选实现双向优化的前 Top-K 组合
        sub_df = res_df[res_df['Is_Both_Improved'] == True].copy()
        if len(sub_df) < 5:
            sub_df = res_df.copy()

        cutoff = max(int(len(sub_df) * top_k_percent), 5)
        sub_df = sub_df.nsmallest(cutoff, 'MAE_total').copy()

        sub_df['MAE_Group'] = pd.qcut(sub_df['MAE_total'], q=min(3, len(sub_df)),
                                      labels=['优秀(综合低误差)', '良好', '一般'][:min(3, len(sub_df))])

        # 增加展示 MAE_pos 和 MAE_neg 两个分支指标
        cols_to_plot = ['damping', 'pos_boost', 'alpha_smoothing', 'MAE_pos', 'MAE_neg', 'MAE_total', 'MAE_Group']
        plot_data = sub_df[cols_to_plot]

        plt.figure(figsize=(12, 6))
        parallel_coordinates(plot_data, class_column='MAE_Group', colormap='Set1_r', alpha=0.75, linewidth=2)

        plt.title(f'【{surface_cn}表面】双向残差均衡参数敏感性全景图 (展示 MAE_pos 与 MAE_neg 演变)', fontsize=12)
        plt.ylabel('参数取值 / 评估误差数值')
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.legend(title="综合 MAE 绩效区间", loc='upper right')
        plt.tight_layout()

        save_path = os.path.join(self.save_dir, f"parallel_coords_{surface}.png")
        plt.savefig(save_path, dpi=300)
        print(f"[图表保存] 细粒度平行坐标全景图已保存至: {save_path}")
        plt.show()

    def plot_facet_heatmaps(self, res_df, surface='Top', metric='MAE_total'):
        """支持绘制 MAE_pos, MAE_neg, RMSE_pos, RMSE_neg 等拆分指标的热力图"""
        surface_cn = '上' if surface == 'Top' else '下'
        alphas = sorted(res_df['alpha_smoothing'].unique())
        n_alphas = len(alphas)

        fig, axes = plt.subplots(1, n_alphas, figsize=(3.8 * n_alphas, 4), sharey=True)
        if n_alphas == 1:
            axes = [axes]

        min_val = res_df[metric].min()
        max_val = res_df[metric].max()

        cmap_choice = "viridis" if "R2" in metric else "viridis_r"

        for idx, alpha in enumerate(alphas):
            sub_df = res_df[(res_df['surface'] == surface) & (res_df['alpha_smoothing'] == alpha)]
            pivot_tbl = sub_df.pivot(index='damping', columns='pos_boost', values=metric)

            ax = axes[idx]
            sns.heatmap(
                pivot_tbl, annot=True, fmt=".4f" if "R2" in metric else ".3f",
                cmap=cmap_choice, cbar=(idx == n_alphas - 1), ax=ax,
                vmin=min_val, vmax=max_val
            )
            ax.set_title(f'alpha_smooth = {alpha}')
            ax.set_xlabel('pos_boost')
            if idx == 0:
                ax.set_ylabel('damping')
            else:
                ax.set_ylabel('')

        plt.suptitle(f'【{surface_cn}表面】不同 alpha 平滑度下 {metric} 演变矩阵', fontsize=13, y=0.98)
        plt.tight_layout()

        save_path = os.path.join(self.save_dir, f"facet_heatmaps_{surface}_{metric}.png")
        plt.savefig(save_path, dpi=300)
        print(f"[图表保存] {metric} 切片热力图矩阵已保存至: {save_path}")
        plt.show()


    def plot_paired_heatmaps(self, res_df, surface='Top', metric_prefix='MAE'):
        """
        将同一个指标的 pos (在线偏低) 和 neg (在线偏高) 矩阵放在一张图左右/上下对比
        支持 metric_prefix 为 'MAE' 或 'RMSE'
        """
        surface_cn = '上' if surface == 'Top' else '下'
        alphas = sorted(res_df['alpha_smoothing'].unique())
        n_alphas = len(alphas)

        pos_metric = f"{metric_prefix}_pos"
        neg_metric = f"{metric_prefix}_neg"

        # 统一两边的 Colorbar 色阶范围，确保对比公平
        v_min = min(res_df[pos_metric].min(), res_df[neg_metric].min())
        v_max = max(res_df[pos_metric].max(), res_df[neg_metric].max())

        # 创建 2 行 N 列的子图结构 (第一行为 Pos，第二行为 Neg)
        fig, axes = plt.subplots(
            nrows=2, ncols=n_alphas,
            figsize=(3.8 * n_alphas, 7.5),
            sharex=True, sharey=True
        )

        # 防止只有一个 alpha 时维度报错
        if n_alphas == 1:
            axes = np.array([[axes[0]], [axes[1]]])

        cmap_choice = "viridis_r"  # MAE/RMSE 越低颜色越亮/越好

        for col_idx, alpha in enumerate(alphas):
            sub_df = res_df[(res_df['surface'] == surface) & (res_df['alpha_smoothing'] == alpha)]

            # 1. 绘制 Pos (在线偏低) - 上半排
            pivot_pos = sub_df.pivot(index='damping', columns='pos_boost', values=pos_metric)
            ax_pos = axes[0, col_idx]
            sns.heatmap(
                pivot_pos, annot=True, fmt=".3f", cmap=cmap_choice,
                cbar=(col_idx == n_alphas - 1), ax=ax_pos,
                vmin=v_min, vmax=v_max
            )
            ax_pos.set_title(f'α = {alpha} | 偏低样本 ({pos_metric})', fontsize=11, fontweight='bold', color='darkblue')
            ax_pos.set_xlabel('')
            if col_idx == 0:
                ax_pos.set_ylabel('damping (阻尼系数)', fontsize=10)
            else:
                ax_pos.set_ylabel('')

            # 2. 绘制 Neg (在线偏高) - 下半排
            pivot_neg = sub_df.pivot(index='damping', columns='pos_boost', values=neg_metric)
            ax_neg = axes[1, col_idx]
            sns.heatmap(
                pivot_neg, annot=True, fmt=".3f", cmap=cmap_choice,
                cbar=(col_idx == n_alphas - 1), ax=ax_neg,
                vmin=v_min, vmax=v_max
            )
            ax_neg.set_title(f'α = {alpha} | 偏高样本 ({neg_metric})', fontsize=11, fontweight='bold', color='darkred')
            ax_neg.set_xlabel('pos_boost (增益)', fontsize=10)
            if col_idx == 0:
                ax_neg.set_ylabel('damping (阻尼系数)', fontsize=10)
            else:
                ax_neg.set_ylabel('')

        plt.suptitle(
            f'【{surface_cn}表面】正负残差子集 ({metric_prefix}) 协同演变对比阵列',
            fontsize=14, fontweight='bold', y=0.98
        )
        plt.tight_layout()

        save_path = os.path.join(self.save_dir, f"paired_heatmap_{surface}_{metric_prefix}.png")
        plt.savefig(save_path, dpi=300)
        print(f"[图表保存] 配对对比热力图已保存至: {save_path}")
        plt.show()