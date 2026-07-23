import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs("result/cleaned_data", exist_ok=True)
os.makedirs("result/correlation_result", exist_ok=True)
os.makedirs("result/fitting_result", exist_ok=True)
os.makedirs("result/models", exist_ok=True)


# ==========================================
# 1. 数据预处理、离群点诊断与 Excel 导出 (未改动)
# ==========================================
def preprocess_and_filter_outliers(df,
                                   clean_save_path="result/cleaned_data/cleaned_data.xlsx",
                                   filtered_save_path="result/cleaned_data/filtered_outliers.xlsx"):
    if 'Tining Section_CONCENT[NTU]_GL_1_Avg' in df.columns:
        df.rename(columns={'Tining Section_CONCENT[NTU]_GL_1_Avg': 'Tining Section_CURRENT[A]_GL_1_Avg'}, inplace=True)

    bot_curr_cols = [f'Tining Section_CURRENT[A]_GL_{i}_Avg' for i in range(1, 37, 2)]
    top_curr_cols = [f'Tining Section_CURRENT[A]_GL_{i}_Avg' for i in range(2, 37, 2)]

    df['Bot_Current_Sum'] = df[bot_curr_cols].sum(axis=1)
    df['Top_Current_Sum'] = df[top_curr_cols].sum(axis=1)

    df['Width_m'] = df['Dimension_[mm]_Width'] / 1000.0
    speed = df['Speed[m/min]_Process_Avg'].replace(0, np.nan)

    df['Top_Theoretical_Factor'] = df['Top_Current_Sum'] / (speed * df['Width_m'])
    df['Bot_Theoretical_Factor'] = df['Bot_Current_Sum'] / (speed * df['Width_m'])

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df['Top_Delta'] = df['上表面镀层重量A(XA1_0)'] - df['Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Avg']
    df['Bot_Delta'] = df['下表面镀层重量A(XA1_0)'] - df['Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Avg']

    if 'Steel Grade' in df.columns:
        grade_freq = df['Steel Grade'].value_counts(normalize=True).to_dict()
        df['Steel_Grade_Encoded'] = df['Steel Grade'].map(grade_freq).fillna(0)
    else:
        df['Steel_Grade_Encoded'] = 0
        grade_freq = {}

    initial_count = len(df)
    df['Filter_Reason'] = ""

    required_cols = [
        'Top_Current_Sum', 'Bot_Current_Sum',
        'Top_Theoretical_Factor', 'Bot_Theoretical_Factor',
        'Speed[m/min]_Process_Avg', 'Dimension_[mm]_Width', 'Dimension_[mm]_Thickness',
        'Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Avg', 'Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Avg',
        '上表面镀层重量A(XA1_0)', '下表面镀层重量A(XA1_0)',
        'Top_Delta', 'Bot_Delta'
    ]
    null_mask = df[required_cols].isnull().any(axis=1)
    df.loc[null_mask, 'Filter_Reason'] += "关键工艺/测量参数存在缺失值; "

    valid_df = df[~null_mask]
    top_delta_std = valid_df['Top_Delta'].std()
    top_delta_mean = valid_df['Top_Delta'].mean()
    bot_delta_std = valid_df['Bot_Delta'].std()
    bot_delta_mean = valid_df['Bot_Delta'].mean()

    top_threshold = 3.5 * top_delta_std
    bot_threshold = 3.5 * bot_delta_std

    steady_speed_mask = df['Speed[m/min]_Process_Avg'] > 80
    top_outlier_mask = (df['Top_Delta'] - top_delta_mean).abs() > top_threshold
    bot_outlier_mask = (df['Bot_Delta'] - bot_delta_mean).abs() > bot_threshold

    df.loc[
        steady_speed_mask & top_outlier_mask, 'Filter_Reason'] += f"上表面平稳工况下残差偏离过大(>{top_threshold:.2f}g/m2); "
    df.loc[
        steady_speed_mask & bot_outlier_mask, 'Filter_Reason'] += f"下表面平稳工况下残差偏离过大(>{bot_threshold:.2f}g/m2); "

    low_speed_mask = df['Speed[m/min]_Process_Avg'] <= 20
    df.loc[low_speed_mask, 'Filter_Reason'] += "极低速/停机过渡区数据; "

    filtered_df = df[df['Filter_Reason'] != ""].copy()
    clean_df = df[df['Filter_Reason'] == ""].copy()

    cols_to_export = ['Coil ID', 'Steel Grade', 'Speed[m/min]_Process_Avg',
                      'Top_Delta', 'Bot_Delta', 'Filter_Reason']
    cols_to_export = [c for c in cols_to_export if c in filtered_df.columns]

    filtered_df[cols_to_export].to_excel(filtered_save_path, index=False)
    clean_df.to_excel(clean_save_path, index=False)

    print("\n==========================================")
    print("        [数据清洗与异常诊断汇总]          ")
    print("==========================================")
    print(f"原始数据总行数: {initial_count}")
    print(f"被剔除异常点数: {len(filtered_df)} (占比: {len(filtered_df) / initial_count * 100:.2f}%)")
    print(f"保留干净样本数: {len(clean_df)}")
    print(f"[导出提示] 被剔除数据明细及原因已保存至: {filtered_save_path}")
    print(f"[导出提示] 训练用干净数据集已保存至: {clean_save_path}")
    print("==========================================\n")

    # 把钢种频率表也返回，方便训练/推理复用同一份编码，而不是分别在训练/推理时各自计算
    return clean_df, grade_freq


# ==========================================
# 2. 相关性分析、残差分布诊断 (未改动)
# ==========================================
def analyze_correlations(df, surface='Top'):
    prefix = 'Top' if surface == 'Top' else 'Bot'
    surface_cn = '上' if surface == 'Top' else '下'

    actual_col = f'Tin Weight_Actual[g/m2]_GALV_WEIGHT_{prefix.upper()}_Avg'
    lab_col = f'{surface_cn}表面镀层重量A(XA1_0)'

    cols_to_check = [
        lab_col, actual_col,
        f'{prefix}_Current_Sum', f'{prefix}_Theoretical_Factor',
        'Speed[m/min]_Process_Avg', 'Dimension_[mm]_Thickness',
        'Dimension_[mm]_Width', 'Steel_Grade_Encoded'
    ]

    corr_matrix = df[cols_to_check].corr()

    print(f"\n======== 【{surface_cn}表面 相关性矩阵】 ========")
    print(corr_matrix[lab_col].sort_values(ascending=False))

    plt.figure(figsize=(9, 7))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title(f'{surface_cn}表面参数与实验室测定值相关性热力图')
    plt.tight_layout()
    save_img_path = f"result/correlation_result/correlation_{surface}.png"
    plt.savefig(save_img_path, dpi=300)
    print(f"[图表保存] {surface_cn}表面相关性热力图已保存至: {save_img_path}")
    plt.show()


def check_residual_distribution(df):
    print("\n==========================================")
    print("      【数据集中原始残差正负分布诊断】       ")
    print("==========================================")
    for surface in ['Top', 'Bot']:
        surface_cn = '上' if surface == 'Top' else '下'
        delta_col = f'{surface}_Delta'
        if delta_col in df.columns:
            total = len(df[delta_col].dropna())
            pos = (df[delta_col] > 0).sum()
            neg = (df[delta_col] < 0).sum()
            mean_val = df[delta_col].mean()
            print(f"[{surface_cn}表面 Delta (实验室值 - 在线值)]")
            print(f"  - 总有效样本数: {total}")
            print(f"  - Delta > 0 (在线测量偏低): {pos} 条 (占比 {pos/total*100:.2f}%)")
            print(f"  - Delta < 0 (在线测量偏高): {neg} 条 (占比 {neg/total*100:.2f}%)")
            print(f"  - Delta 均值: {mean_val:.4f} g/m2")
    print("==========================================\n")


# ==========================================
# 3. 残差建模核心类：加入保存/加载能力，并把训练时依赖的
#    全部状态（特征列表、钢种频率表等）打包在一起，做到"一个对象即可复现推理"
# ==========================================
def compute_direction_sample_weight(y_delta, pos_boost=1.0, damping=0.0):
    if damping <= 0:
        return pd.Series(1.0, index=y_delta.index)

    pos_mask = y_delta > 0
    neg_mask = y_delta < 0
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()
    n_total = n_pos + n_neg

    weights = pd.Series(1.0, index=y_delta.index)
    if n_pos > 0:
        full_balance_pos = n_total / (2.0 * n_pos)
        weights[pos_mask] = (1 - damping) * 1.0 + damping * full_balance_pos * pos_boost
    if n_neg > 0:
        full_balance_neg = n_total / (2.0 * n_neg)
        weights[neg_mask] = (1 - damping) * 1.0 + damping * full_balance_neg

    return weights


class ResidualCorrectionModel:
    """
    通用残差修正模型：对任意一个表面(surface)都适用，只要传入正确的
    feature_cols / online_col / delta_col。

    当前最优配置（网格搜索得出）：
        damping=0.0（不做方向加权）, alpha_smoothing=0.7, loss='absolute_error'
        对'在线测量值'特征施加单调约束(monotonic_cst=-1)

    该类同时负责：
    1. 训练 (fit)
    2. 预测 + EWMA平滑 (predict_smooth)
    3. 保存/加载全部推理所需状态 (save / load)，确保训练和推理时特征处理完全一致
    """

    def __init__(self, feature_cols, online_col, delta_col, lab_col,
                 monotonic_feature_name=None, alpha_smoothing=0.7,
                 pos_boost=1.0, damping=0.0, grade_freq_map=None):
        self.feature_cols = feature_cols            # 特征列表，顺序即模型输入顺序
        self.online_col = online_col                # 在线仪表测量值列名
        self.delta_col = delta_col                  # 拟合目标（残差）列名
        self.lab_col = lab_col                      # 实验室真实值列名（仅评估用）
        self.monotonic_feature_name = monotonic_feature_name
        self.alpha_smoothing = alpha_smoothing
        self.pos_boost = pos_boost
        self.damping = damping
        self.grade_freq_map = grade_freq_map or {}   # 训练集算出的钢种频率表，推理时复用
        self.model = None
        # 记录最后一次预测的平滑残差值，方便下一批新数据做EWMA"热启动"，
        # 避免每次推理都从头平滑导致的起始段失真
        self._last_smoothed_delta = None

    def _build_model(self, n_features):
        monotonic_cst = None
        if self.monotonic_feature_name is not None:
            monotonic_cst = [0] * n_features
            idx = self.feature_cols.index(self.monotonic_feature_name)
            monotonic_cst[idx] = -1
        self.model = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_depth=4,
            loss='absolute_error',
            monotonic_cst=monotonic_cst,
            random_state=42
        )

    def fit(self, df):
        """直接传入包含所有 feature_cols + delta_col 的 DataFrame。"""
        X = df[self.feature_cols]
        y_delta = df[self.delta_col]
        self._build_model(n_features=X.shape[1])
        sample_weight = compute_direction_sample_weight(
            y_delta, pos_boost=self.pos_boost, damping=self.damping
        )
        self.model.fit(X, y_delta, sample_weight=sample_weight.values)
        return self

    def predict(self, df, warm_start=False):
        """
        对新数据做预测，返回 (最终预测值, 平滑后的残差)。
        df 需要包含 feature_cols 和 online_col。
        warm_start=True 时，会把上一次调用结尾的平滑残差作为本次EWMA的起点，
        适合"新数据是紧接着上一批之后的连续生产"这种场景；
        如果新数据和上次调用无时间连续性（比如重新分析历史某一段），建议 warm_start=False。
        """
        X = df[self.feature_cols]
        online_actual = df[self.online_col]

        predicted_delta_raw = self.model.predict(X)
        delta_series = pd.Series(predicted_delta_raw, index=X.index)

        if warm_start and self._last_smoothed_delta is not None:
            # 把上次的平滑值作为虚拟的"第0个观测"接到序列最前面，再整体做EWMA，
            # 结束后去掉这个虚拟头部，实现平滑状态的跨批次延续
            prepended = pd.concat([
                pd.Series([self._last_smoothed_delta], index=[-1]),
                delta_series
            ])
            smoothed_full = prepended.ewm(alpha=self.alpha_smoothing).mean()
            predicted_delta_smooth = smoothed_full.iloc[1:]
            predicted_delta_smooth.index = delta_series.index
        else:
            predicted_delta_smooth = delta_series.ewm(alpha=self.alpha_smoothing).mean()

        self._last_smoothed_delta = predicted_delta_smooth.iloc[-1]
        final_pred = online_actual + predicted_delta_smooth
        return final_pred, predicted_delta_smooth

    def encode_steel_grade(self, df, grade_col='Steel Grade', out_col='Steel_Grade_Encoded'):
        """
        用训练时保存的钢种频率表对新数据编码，保证训练/推理使用同一份映射，
        新数据中未出现过的钢种编码为0（表示训练集里从未见过的新钢种）。
        """
        df = df.copy()
        df[out_col] = df[grade_col].map(self.grade_freq_map).fillna(0) if grade_col in df.columns else 0
        return df

    def save(self, path):
        """
        保存模型本身及全部推理所需的状态（特征列表、钢种频率表、超参数等），
        单个文件即可完整复现推理，不依赖训练脚本里的其他变量。
        """
        payload = {
            'model': self.model,
            'feature_cols': self.feature_cols,
            'online_col': self.online_col,
            'delta_col': self.delta_col,
            'lab_col': self.lab_col,
            'monotonic_feature_name': self.monotonic_feature_name,
            'alpha_smoothing': self.alpha_smoothing,
            'pos_boost': self.pos_boost,
            'damping': self.damping,
            'grade_freq_map': self.grade_freq_map,
            'last_smoothed_delta': self._last_smoothed_delta,
        }
        joblib.dump(payload, path)
        print(f"[模型保存] 已保存至: {path}")

    @classmethod
    def load(cls, path):
        """从磁盘加载一个完整的 ResidualCorrectionModel 实例，可直接调用 .predict()。"""
        payload = joblib.load(path)
        instance = cls(
            feature_cols=payload['feature_cols'],
            online_col=payload['online_col'],
            delta_col=payload['delta_col'],
            lab_col=payload['lab_col'],
            monotonic_feature_name=payload['monotonic_feature_name'],
            alpha_smoothing=payload['alpha_smoothing'],
            pos_boost=payload['pos_boost'],
            damping=payload['damping'],
            grade_freq_map=payload['grade_freq_map'],
        )
        instance.model = payload['model']
        instance._last_smoothed_delta = payload.get('last_smoothed_delta')
        print(f"[模型加载] 已从 {path} 加载完成")
        return instance


# ==========================================
# 4. 表面建模、评估、画图、保存 —— 通用化为一个函数，Top/Bot 都调用同一套逻辑
# ==========================================
def run_surface_pipeline(df, surface='Top', grade_freq_map=None,
                          damping=0.0, alpha_smoothing=0.7, pos_boost=1.0,
                          model_save_path=None):
    prefix = 'Top' if surface == 'Top' else 'Bot'
    surface_cn = '上' if surface == 'Top' else '下'

    print(f"\n==========================================")
    print(f"        开始运行【{surface_cn}表面】模型拟合与分析     ")
    print(f"        (damping={damping}, alpha_smoothing={alpha_smoothing})")
    print(f"==========================================")

    analyze_correlations(df, surface=surface)

    speed_col = 'Speed[m/min]_Process_Avg'
    current_col = f'{prefix}_Current_Sum'
    df = df.copy()
    df[f'{prefix}_Current_Per_Speed'] = df[current_col] / (df[speed_col] + 1e-5)

    online_col = f'Tin Weight_Actual[g/m2]_GALV_WEIGHT_{prefix.upper()}_Avg'
    delta_col = f'{prefix}_Delta'
    lab_col = f'{surface_cn}表面镀层重量A(XA1_0)'

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

    # 按时间划分（若有 Produce Time 列则先按时间排序，保持因果顺序）
    if 'Produce Time' in df.columns:
        df = df.sort_values('Produce Time').reset_index(drop=True)

    # 注意：online_col 本身已经在 feature_cols 里，这里去重，
    # 避免 train_df/test_df 出现两列同名，导致后续 df[online_col] 取出 DataFrame 而非 Series
    extra_cols = [c for c in [delta_col, online_col, lab_col] if c not in feature_cols]
    X_all_cols = feature_cols + extra_cols
    train_df, test_df = train_test_split(df[X_all_cols], test_size=0.2, shuffle=False)

    corrector = ResidualCorrectionModel(
        feature_cols=feature_cols,
        online_col=online_col,
        delta_col=delta_col,
        lab_col=lab_col,
        monotonic_feature_name=online_col,
        alpha_smoothing=alpha_smoothing,
        pos_boost=pos_boost,
        damping=damping,
        grade_freq_map=grade_freq_map
    )
    corrector.fit(train_df)

    pred_series, predicted_delta_smooth = corrector.predict(test_df, warm_start=False)

    y_true_series = test_df[lab_col]
    online_series = test_df[online_col]

    raw_residuals = y_true_series - online_series
    model_residuals = y_true_series - pred_series

    print(f"\n-------- 【{surface_cn}表面 模型矫正前后残差诊断】 --------")
    mask_pos = (raw_residuals > 0)
    mask_neg = (raw_residuals < 0)

    if mask_pos.sum() > 0:
        mae_raw_pos = raw_residuals[mask_pos].abs().mean()
        mae_model_pos = model_residuals[mask_pos].abs().mean()
        print(
            f"当原始在线偏低 (残差 > 0, 样本数 {mask_pos.sum()}): 原始 MAE = {mae_raw_pos:.4f}  -->  模型矫正后 MAE = {mae_model_pos:.4f}")

    if mask_neg.sum() > 0:
        mae_raw_neg = raw_residuals[mask_neg].abs().mean()
        mae_model_neg = model_residuals[mask_neg].abs().mean()
        print(
            f"当原始在线偏高 (残差 < 0, 样本数 {mask_neg.sum()}): 原始 MAE = {mae_raw_neg:.4f}  -->  模型矫正后 MAE = {mae_model_neg:.4f}")
    print("------------------------------------------------------\n")

    r2_online = r2_score(y_true_series, online_series)
    r2_model = r2_score(y_true_series, pred_series)
    rmse_online = np.sqrt(mean_squared_error(y_true_series, online_series))
    rmse_model = np.sqrt(mean_squared_error(y_true_series, pred_series))

    print(f"======== 【{surface_cn}表面 拟合性能评估（测试集）】 ========")
    print(f"原始在线仪表与实验室真实值 -> R²: {r2_online:.4f}, RMSE: {rmse_online:.4f}")
    print(f"模型校正拟合后与实验室真实值 -> R²: {r2_model:.4f}, RMSE: {rmse_model:.4f}")

    start_idx = test_df.index[0]
    end_idx = test_df.index[-1]

    plt.figure(figsize=(12, 5))
    plt.plot(y_true_series.values, label='实验室真实测量值 (True Label)', color='black', linewidth=1.5)
    plt.plot(online_series.values, label='在线仪表原始测量值 (Online)', color='red', linestyle='--', alpha=0.7)
    plt.plot(pred_series.values, label='模型残差校正值 (Model Pred)', color='green', linewidth=1.5, alpha=0.85)
    plt.title(f'{surface_cn}表面 镀层重量拟合对照图（原始数据行号: {start_idx} ~ {end_idx}）')
    plt.xlabel('测试集样本序号')
    plt.ylabel('镀层重量 (g/m2)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    fit_img_path = f"result/fitting_result/fitting_result_{surface}.png"
    plt.savefig(fit_img_path, dpi=300)
    print(f"[图表保存] {surface_cn}表面拟合对照图已保存至: {fit_img_path}")
    plt.show()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})
    ax1.plot(raw_residuals.values, label='原始在线仪表残差 (True - Online)', color='red', alpha=0.5, linewidth=1)
    ax1.plot(model_residuals.values, label='模型校正后残差 (True - Model)', color='green', alpha=0.8, linewidth=1.2)
    ax1.axhline(0, color='black', linestyle='--', linewidth=1)
    ax1.set_title(f'{surface_cn}表面 预测残差变化对比（原始数据行号: {start_idx} ~ {end_idx}）')
    ax1.set_xlabel('测试集样本序号')
    ax1.set_ylabel('残差/误差 (g/m2)')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    sns.histplot(raw_residuals, ax=ax2, color='red', label='原始残差分布', kde=True, stat="density", alpha=0.3)
    sns.histplot(model_residuals, ax=ax2, color='green', label='模型校正后残差分布', kde=True, stat="density",
                 alpha=0.5)
    ax2.axvline(0, color='black', linestyle='--', linewidth=1)
    ax2.set_title(f'{surface_cn}表面 残差概率密度分布（越集中在0且越窄越好）')
    ax2.set_xlabel('残差/误差 (g/m2)')
    ax2.set_ylabel('概率密度')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    res_img_path = f"result/fitting_result/residual_analysis_{surface}.png"
    plt.savefig(res_img_path, dpi=300)
    print(f"[图表保存] {surface_cn}表面残差分析图已保存至: {res_img_path}")
    plt.show()

    # ---- 保存模型 ----
    if model_save_path is None:
        model_save_path = f"result/models/residual_model_{surface}.joblib"
    corrector.save(model_save_path)

    return corrector


# ==========================================
# 5. 主流程
# ==========================================
if __name__ == "__main__":
    raw_df = pd.read_excel("result/merged_data/merged_result_latest.xlsx")

    clean_df, grade_freq_map = preprocess_and_filter_outliers(
        raw_df,
        clean_save_path="result/cleaned_data/cleaned_data.xlsx",
        filtered_save_path="result/cleaned_data/filtered_outliers.xlsx"
    )

    check_residual_distribution(clean_df)

    # 默认配置：damping=0.0（不加权）+ alpha_smoothing=0.7，网格搜索得到的整体最优解
    run_surface_pipeline(clean_df, surface='Top', grade_freq_map=grade_freq_map,
                          damping=0.0, alpha_smoothing=0.7,
                          model_save_path="result/models/residual_model_Top.joblib")

    run_surface_pipeline(clean_df, surface='Bot', grade_freq_map=grade_freq_map,
                          damping=0.0, alpha_smoothing=0.7,
                          model_save_path="result/models/residual_model_Bot.joblib")