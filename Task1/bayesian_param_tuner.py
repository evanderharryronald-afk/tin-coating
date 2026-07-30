import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna

# 屏蔽 optuna 详细的 info 级别日志，保持控制台整洁（若想看每轮输出可设为 optuna.logging.INFO）
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

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
    """扩展版残差矫正模型，支持传入超参数"""

    def __init__(self, monotonic_feature_idx=None, alpha_smoothing=0.7,
                 pos_boost=1.0, damping=0.0, max_iter=60, learning_rate=0.08,
                 max_depth=4, l2_regularization=0.0):
        self.alpha_smoothing = alpha_smoothing
        self.pos_boost = pos_boost
        self.damping = damping
        self.monotonic_feature_idx = monotonic_feature_idx
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.l2_regularization = l2_regularization
        self.model = None

    def _build_model(self, n_features):
        monotonic_cst = None
        if self.monotonic_feature_idx is not None:
            monotonic_cst = [0] * n_features
            monotonic_cst[self.monotonic_feature_idx] = -1

        self.model = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            l2_regularization=self.l2_regularization,
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


class BayesianResidualTuner:
    """基于 Optuna 贝叶斯优化的残差矫正模型调参工具"""

    def __init__(self, save_dir="result/bayesian_param_tuning"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def optimize_surface(self, df, surface='Top', n_trials=60, penalty_factor=5.0, balance_weight=0.5):
        """
        对指定表面（Top/Bot）执行贝叶斯搜索
        """
        prefix = 'Top' if surface == 'Top' else 'Bot'
        surface_cn = '上' if surface == 'Top' else '下'

        # 1. 特征工程
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

        # 2. 时序划分
        X_train, X_test, y_delta_train, y_delta_test, actual_train, actual_test, y_true_train, y_true_test = \
            train_test_split(X, y_delta, online_actual, y_true, test_size=0.2, shuffle=False)

        X_train_vals = X_train.values
        y_delta_train_vals = y_delta_train.values
        y_true_vals = y_true_test.values
        actual_vals = actual_test.values
        raw_res = y_true_vals - actual_vals

        # 在线仪表基准拆分掩码
        mask_pos = (raw_res > 0)
        mask_neg = (raw_res < 0)

        raw_mae_pos = np.mean(np.abs(raw_res[mask_pos])) if np.any(mask_pos) else np.nan
        raw_mae_neg = np.mean(np.abs(raw_res[mask_neg])) if np.any(mask_neg) else np.nan

        print(f"\n>>>> 🚀 开始贝叶斯优化【{surface_cn}表面】(计划试验次数: {n_trials} 轮)...")

        trials_records = []

        # 3. 定义 Optuna 目标函数
        def objective(trial):
            # 搜索空间定义
            damping = trial.suggest_float('damping', 0.0, 1.0, step=0.1)
            pos_boost = trial.suggest_float('pos_boost', 1.0, 8.0, step=0.5)
            alpha_smoothing = trial.suggest_float('alpha_smoothing', 0.1, 1.0, step=0.1)

            max_depth = trial.suggest_int('max_depth', 2, 6)
            max_iter = trial.suggest_int('max_iter', 30, 150, step=10)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.15, log=True)
            l2_regularization = trial.suggest_float('l2_regularization', 1e-3, 10.0, log=True)

            model = ResidualCorrectionModel(
                monotonic_feature_idx=online_feature_idx,
                alpha_smoothing=alpha_smoothing,
                pos_boost=pos_boost,
                damping=damping,
                max_iter=max_iter,
                learning_rate=learning_rate,
                max_depth=max_depth,
                l2_regularization=l2_regularization
            )

            model.fit(X_train_vals, y_delta_train_vals)
            pred_series, _ = model.predict_smooth(X_test, actual_test)
            pred_vals = pred_series.values

            # 计算整体指标
            mae_total = mean_absolute_error(y_true_vals, pred_vals)
            rmse_total = np.sqrt(mean_squared_error(y_true_vals, pred_vals))
            r2_total = r2_score(y_true_vals, pred_vals)

            # 拆分残差评估
            model_res = y_true_vals - pred_vals

            res_pos = model_res[mask_pos]
            mae_pos = np.mean(np.abs(res_pos)) if len(res_pos) > 0 else mae_total

            res_neg = model_res[mask_neg]
            mae_neg = np.mean(np.abs(res_neg)) if len(res_neg) > 0 else mae_total

            is_both_improved = (mae_pos < raw_mae_pos) and (mae_neg < raw_mae_neg)
            balance_gap = abs(mae_pos - mae_neg)

            # 记录 Trial 数据
            record = {
                'trial_id': trial.number,
                'surface': surface,
                'damping': damping,
                'pos_boost': pos_boost,
                'alpha_smoothing': alpha_smoothing,
                'max_depth': max_depth,
                'max_iter': max_iter,
                'learning_rate': learning_rate,
                'l2_regularization': l2_regularization,
                'MAE_total': mae_total,
                'RMSE_total': rmse_total,
                'R2_total': r2_total,
                'MAE_pos': mae_pos,
                'MAE_neg': mae_neg,
                'Is_Both_Improved': is_both_improved,
                'Balance_Gap': balance_gap
            }
            trials_records.append(record)

            # 惩罚函数构建：不仅关注整体 MAE，且若双向改善失败则施加惩罚，同时引导极差缩小
            custom_loss = mae_total + balance_weight * balance_gap
            if not is_both_improved:
                custom_loss += penalty_factor

            return custom_loss

        # 4. 创建 Study 并启动优化
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials)

        res_df = pd.DataFrame(trials_records)
        res_df = res_df.sort_values(
            by=['Is_Both_Improved', 'MAE_total', 'Balance_Gap'],
            ascending=[False, True, True]
        ).reset_index(drop=True)

        # 保存搜索 CSV
        csv_path = os.path.join(self.save_dir, f"bayesian_search_{surface}.csv")
        res_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        best_row = res_df.iloc[0]
        print(f"✅ 【{surface_cn}表面】贝叶斯优化完成！最佳组合 (Trial #{best_row['trial_id']})：")
        print(
            f"   - 核心权重: damping={best_row['damping']}, pos_boost={best_row['pos_boost']}, alpha={best_row['alpha_smoothing']}")
        print(
            f"   - 算法结构: depth={best_row['max_depth']}, iter={best_row['max_iter']}, lr={best_row['learning_rate']:.4f}, l2={best_row['l2_regularization']:.4f}")
        print(f"   - 整体 MAE: {best_row['MAE_total']:.4f} (R²: {best_row['R2_total']:.4f})")
        print(
            f"   - 偏低 MAE: {best_row['MAE_pos']:.4f} (基准: {raw_mae_pos:.4f}) | 偏高 MAE: {best_row['MAE_neg']:.4f} (基准: {raw_mae_neg:.4f})")
        print(f"   - 是否实现正负双向改善: {'是' if best_row['Is_Both_Improved'] else '否'}")

        return res_df, study

    def plot_parallel_coordinates(self, res_df, surface='Top', top_k_percent=0.3):
        """展示贝叶斯搜索出的优秀参数平行坐标全景图"""
        surface_cn = '上' if surface == 'Top' else '下'

        sub_df = res_df[res_df['Is_Both_Improved'] == True].copy()
        if len(sub_df) < 5:
            sub_df = res_df.copy()

        cutoff = max(int(len(sub_df) * top_k_percent), 5)
        sub_df = sub_df.nsmallest(cutoff, 'MAE_total').copy()

        sub_df['MAE_Group'] = pd.qcut(sub_df['MAE_total'], q=min(3, len(sub_df)),
                                      labels=['优秀(综合低误差)', '良好', '一般'][:min(3, len(sub_df))])

        cols_to_plot = ['damping', 'pos_boost', 'alpha_smoothing', 'max_depth', 'MAE_pos', 'MAE_neg', 'MAE_total',
                        'MAE_Group']
        plot_data = sub_df[cols_to_plot]

        plt.figure(figsize=(13, 6))
        parallel_coordinates(plot_data, class_column='MAE_Group', colormap='Set1_r', alpha=0.75, linewidth=2)

        plt.title(f'【{surface_cn}表面】贝叶斯搜索最优区域参数敏感性全景图', fontsize=12)
        plt.ylabel('参数/性能指标取值')
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.legend(title="MAE 绩效区间", loc='upper right')
        plt.tight_layout()

        save_path = os.path.join(self.save_dir, f"bayesian_parallel_coords_{surface}.png")
        plt.savefig(save_path, dpi=300)
        print(f"[图表保存] 平行坐标全景图已保存至: {save_path}")
        plt.close()

    def plot_optimization_history(self, study, surface='Top'):
        """绘制贝叶斯搜索收敛历程图"""
        surface_cn = '上' if surface == 'Top' else '下'
        trials = study.trials
        vals = [t.value for t in trials if t.value is not None]
        best_vals = np.minimum.accumulate(vals)

        plt.figure(figsize=(9, 4.5))
        plt.plot(vals, label='每轮试验 Loss', color='gray', alpha=0.5, marker='o', markersize=3)
        plt.plot(best_vals, label='当前历史最优 Loss', color='red', linewidth=2)
        plt.title(f'【{surface_cn}表面】Optuna 贝叶斯优化收敛轨迹', fontsize=12)
        plt.xlabel('Trial (试验轮数)')
        plt.ylabel('Objective Loss (综合目标函数损耗)')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
        plt.tight_layout()

        save_path = os.path.join(self.save_dir, f"optuna_history_{surface}.png")
        plt.savefig(save_path, dpi=300)
        print(f"[图表保存] 优化轨迹收敛图已保存至: {save_path}")
        plt.close()


# ==========================================
# 主流程调用示例
# ==========================================
if __name__ == "__main__":
    # 演示数据加载逻辑（请替换为实际已清洗的生产数据 cleaned_data.xlsx 或 DataFrame）
    cleaned_excel_path = "result/cleaned_data/cleaned_data.xlsx"

    if os.path.exists(cleaned_excel_path):
        df_all = pd.read_excel(cleaned_excel_path)

        tuner = BayesianResidualTuner(save_dir="result/bayesian_param_tuning")

        # 遍历 上/下 两个表面独立执行贝叶斯优化
        for surf in ['Top', 'Bot']:
            res_df, study = tuner.optimize_surface(
                df=df_all,
                surface=surf,
                n_trials=50,  # 可根据性能灵活设置，例如 50 ~ 100 轮
                penalty_factor=2.0  # 对未实现双向改善组合的惩罚权重
            )

            # 生成诊断图表
            tuner.plot_parallel_coordinates(res_df, surface=surf)
            tuner.plot_optimization_history(study, surface=surf)
    else:
        print(f"未找到输入数据文件: {cleaned_excel_path}，请修改数据加载路径。")