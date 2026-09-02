import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from data_cleaner import SteelDataCleaner
import argparse
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from correlation_analyzer import SurfaceCorrelationAnalyzer
from model_interpreter import ModelInterpreter   # ===== 模型解释性分析 =====
from eda_analyzer import SurfaceEDAAnalyzer  # eda 数据分析， 模型训练前后

# 设置画图支持中文与负号，消除特殊字符警告
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建结果保存目录结构
os.makedirs("result/data/cleaned_data", exist_ok=True)
os.makedirs("result/correlation_result", exist_ok=True)
os.makedirs("result/fitting_result", exist_ok=True)

# 每个规格组最少需要的样本数，低于此值只诊断不建模
MIN_GROUP_SAMPLES = 200

# Setpoint 分组用的两个字段（上下表面镀层重量下限设定值）
TOP_SETPOINT_COL = 'Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_TOP_Min'
BOT_SETPOINT_COL = 'Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_BOT_Min'


# ==========================================
# 0. 分组参数表（按 组+表面 覆盖训练参数）
# ==========================================
# GROUP_PARAMS_PATH = "group_params_all_the_same.json"
GROUP_PARAMS_PATH = "group_params_optimum_for_each.json"

DEFAULT_PARAMS = {
    "damping": 0.6,
    "pos_boost": 4.6,
    "alpha_smoothing": 1.0,
    "max_iter": 200,
    "learning_rate": 0.05,
    "max_depth": 4,
}


def load_pipeline_config(config_path="config.json"):
    """读取统一的 JSON 配置文件，并提供合理的默认兜底值"""
    default_config = {
        "target_groups": None,  # None 表示自动跑全部达标组
        "min_group_samples": 200,
        "default_params": {
            "damping": 0.6,
            "pos_boost": 4.6,
            "alpha_smoothing": 1.0,
            "max_iter": 200,
            "learning_rate": 0.05,
            "max_depth": 4,
        },
        "group_params_override": {}
    }

    if not os.path.exists(config_path):
        print(f"[提示] 配置文件 {config_path} 不存在，使用内置默认参数。")
        return default_config

    with open(config_path, "r", encoding="utf-8") as f:
        user_config = json.load(f)

    # 深度合并/更新配置
    default_config.update(user_config)
    return default_config

def get_params_for_group(config, group_label, surface):
    """
    根据组名和表面获取最终模型参数
    优先级：group_params_override[组名__表面] > default_params
    
    Args:
        config: 配置字典，包含 default_params 和 group_params_override
        group_label: 规格组标签（如 'Top2.799_Bot2.799'）
        surface: 表面类型 ('Top' 或 'Bot')
    
    Returns:
        dict: 合并后的参数字典
    """
    # 1. 基础默认参数
    base_params = config.get("default_params", {}).copy()

    # 2. 查找是否有特定组+表面的覆盖配置
    key = f"{group_label}__{surface}"
    override_params = config.get("group_params_override", {}).get(key, {})

    # 3. 合并字典（override_params 会覆盖 base_params 中的同名键）
    base_params.update(override_params)

    return base_params



# ==========================================
# 1.5 按镀层规格（Setpoint 组合）分组
# ==========================================
def build_setpoint_group_key(df, top_col=TOP_SETPOINT_COL, bot_col=BOT_SETPOINT_COL):
    """
    用 (Top_Min, Bot_Min) 的精确组合作为镀层规格分组键。
    不做四舍五入，只按原始唯一值分组 —— 意味着同一规格在源数据里
    必须是完全一致的数值，如果上游系统对同一规格记录了轻微浮点误差
    （例如 1.1000000001 和 1.1），会被当成两个不同的组，需要提前确认
    源数据里这两列的写入方式是否足够干净。
    """
    if top_col not in df.columns or bot_col not in df.columns:
        raise KeyError(f"缺少分组所需字段: {top_col} 或 {bot_col}")

    df = df.copy()
    df['Setpoint_Group_Key'] = list(zip(df[top_col], df[bot_col]))
    df['Setpoint_Group_Label'] = df.apply(
        lambda r: f"Top{r[top_col]}_Bot{r[bot_col]}", axis=1
    )
    return df


def summarize_setpoint_groups(df):
    """打印每个镀层规格组的样本量，区分达标组与跳过组，表格化输出"""
    group_sizes = df.groupby('Setpoint_Group_Label').size().sort_values(ascending=False)

    valid_groups = group_sizes[group_sizes >= MIN_GROUP_SAMPLES]
    skipped_groups = group_sizes[group_sizes < MIN_GROUP_SAMPLES]

    print("\n==========================================")
    print("        [镀层规格分组样本量汇总]           ")
    print("==========================================")
    print(f"共 {len(group_sizes)} 个规格组，达标 {len(valid_groups)} 个，跳过 {len(skipped_groups)} 个")
    print(f"（达标阈值: >= {MIN_GROUP_SAMPLES} 条）\n")

    print(f"{'规格组':<30}{'样本数':>10}   状态")
    print("-" * 55)
    for label, size in valid_groups.items():
        print(f"{label:<30}{size:>10}   建模")
    for label, size in skipped_groups.items():
        print(f"{label:<30}{size:>10}   跳过")
    print("==========================================\n")

    return group_sizes


def check_residual_distribution(df, group_tag=""):
    """单独排查数据集残差分布状况的辅助函数"""
    tag_display = f"[{group_tag}] " if group_tag else ""
    print("\n==========================================")
    print(f"      【{tag_display}数据集中原始残差正负分布诊断】       ")
    print("==========================================")
    for surface in ['Top', 'Bot']:
        surface_cn = '上' if surface == 'Top' else '下'
        delta_col = f'{surface}_Delta'
        if delta_col in df.columns:
            total = len(df[delta_col].dropna())
            if total == 0:
                continue
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
# 3. 残差建模核心类
# ==========================================

def _build_corrector_instance(params, ModelClass):
    """
    根据参数创建矫正模型实例
    
    Args:
        params: 参数字典，包含 alpha_smoothing, pos_boost, damping, max_iter, learning_rate, max_depth
        ModelClass: 模型类 (ResidualCorrectionModel 或 LinearResidualCorrectionModel)
    
    Returns:
        实例化的矫正模型对象
    """
    corrector = ModelClass(
        alpha_smoothing=params["alpha_smoothing"],
        pos_boost=params["pos_boost"],
        damping=params["damping"],
        # 如果用树模型会读取 max_iter/learning_rate，如果用线性模型额外参数会被 **kwargs 吸收，完全兼容
        max_iter=params.get("max_iter", 200),
        learning_rate=params.get("learning_rate", 0.05),
        max_depth=params.get("max_depth", 4),
    )
    return corrector


def compute_direction_sample_weight(y_delta, pos_boost=1.0, damping=0.0):
    """
    damping: 0~1，默认0（不加权，即完全等权重，对应网格搜索中RMSE最优的配置）。
             调大会向"两方向都不能变差"的保守解靠拢，但会牺牲整体RMSE，
             具体取舍参见网格搜索结果。
    pos_boost: 对少数方向（通常是"在线偏低"，delta>0）的额外加权系数，仅在 damping>0 时生效。
    """
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
    基于 HistGradientBoostingRegressor 的残差矫正模型
    直接对残差 Delta = 真实值 - 在线值 建模，而不是对绝对值建模。
    """

    def __init__(self, monotonic_feature_idx=None, alpha_smoothing=0.7,
                 pos_boost=1.0, damping=0.0,
                 max_iter=200, learning_rate=0.05, max_depth=4):
        self.alpha_smoothing = alpha_smoothing
        self.pos_boost = pos_boost
        self.damping = damping
        self.monotonic_feature_idx = monotonic_feature_idx
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_depth = max_depth
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
            loss='absolute_error',
            # loss='squared_error',
            # loss='quantile',  # 使用分位数损失函数，减少异常值影响
            monotonic_cst=monotonic_cst,
            random_state=42
        )

    def fit(self, X, y_delta):
        self._build_model(n_features=X.shape[1])
        sample_weight = compute_direction_sample_weight(
            y_delta, pos_boost=self.pos_boost, damping=self.damping
        )
        self.model.fit(X, y_delta, sample_weight=sample_weight.values)

    def predict_smooth(self, X, online_actual):
        predicted_delta_raw = self.model.predict(X)
        delta_series = pd.Series(predicted_delta_raw, index=X.index)
        predicted_delta_smooth = delta_series.ewm(alpha=self.alpha_smoothing).mean()
        final_pred = online_actual + predicted_delta_smooth
        return final_pred, predicted_delta_smooth

class LinearResidualCorrectionModel:
    """
    【新增】基于 Huber 线性回归 + L2 正则的残差矫正模型
    接口与 ResidualCorrectionModel 保持一致，方便无缝切换。
    """
    def __init__(self, alpha_smoothing=0.7, pos_boost=1.0, damping=0.0,
                 alpha=1.0, epsilon=1.35, **kwargs):
        self.alpha_smoothing = alpha_smoothing
        self.pos_boost = pos_boost
        self.damping = damping
        self.alpha = alpha       # L2 正则化强度
        self.epsilon = epsilon   # Huber 损失对异常值的敏感阈值 (1.35 是标准默认值)
        self.model = None

    def _build_model(self):
        # 线性模型对量纲敏感，因此封装 StandardScaler 做 Pipeline
        self.model = make_pipeline(
            StandardScaler(),
            HuberRegressor(alpha=self.alpha, epsilon=self.epsilon, max_iter=1000)
        )

    def fit(self, X, y_delta):
        self._build_model()
        sample_weight = compute_direction_sample_weight(
            y_delta, pos_boost=self.pos_boost, damping=self.damping
        )
        # 将样本权重传递给 Pipeline 内的 HuberRegressor
        self.model.fit(X, y_delta, huberregressor__sample_weight=sample_weight.values)

    def predict_smooth(self, X, online_actual):
        predicted_delta_raw = self.model.predict(X)
        delta_series = pd.Series(predicted_delta_raw, index=X.index)
        predicted_delta_smooth = delta_series.ewm(alpha=self.alpha_smoothing).mean()
        final_pred = online_actual + predicted_delta_smooth
        return final_pred, predicted_delta_smooth


def _compute_directional_metrics(residuals):
    """
    计算残差的正/负偏差相关指标
    
    Args:
        residuals: pandas Series，残差数组
    
    Returns:
        dict: 包含 mask_pos, mask_neg, pos_count, neg_count, pos_mae, neg_mae 的字典
    """
    mask_pos = (residuals > 0)
    mask_neg = (residuals < 0)
    
    pos_count = mask_pos.sum()
    neg_count = mask_neg.sum()
    
    pos_mae = residuals[mask_pos].abs().mean() if pos_count > 0 else 0.0
    neg_mae = residuals[mask_neg].abs().mean() if neg_count > 0 else 0.0

    pos_rmse = np.sqrt((residuals[mask_pos] ** 2).mean()) if pos_count > 0 else 0.0
    neg_rmse = np.sqrt((residuals[mask_neg] ** 2).mean()) if neg_count > 0 else 0.0

    
    return {
        'mask_pos': mask_pos,
        'mask_neg': mask_neg,
        'pos_count': pos_count,
        'neg_count': neg_count,
        'pos_mae': pos_mae,
        'neg_mae': neg_mae,
        'pos_rmse': pos_rmse,
        'neg_rmse': neg_rmse,
    }


def _plot_residual_distribution(ax, residuals_train, residuals_test, 
                                 color_train, color_test, 
                                 label_train, label_test, title, xlabel):
    """
    绘制残差分布直方图（训练集 vs 测试集对比）
    
    Args:
        ax: matplotlib 子图对象
        residuals_train: 训练集残差 Series
        residuals_test: 测试集残差 Series
        color_train: 训练集颜色
        color_test: 测试集颜色
        label_train: 训练集标签
        label_test: 测试集标签
        title: 子图标题
        xlabel: x轴标签
    """
    sns.histplot(residuals_train, ax=ax, color=color_train, label=label_train,
                 kde=True, stat="density", alpha=0.4)
    sns.histplot(residuals_test, ax=ax, color=color_test, label=label_test,
                 kde=True, stat="density", alpha=0.4)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('概率密度')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)







# ==========================================
# 4. 纯计算：切分数据 + 训练 + 算 metrics（不画图，供调参脚本和正式训练共用）
# ==========================================
def fit_and_evaluate_surface(df, surface, params, group_tag="",
                             use_expanding_window=False,
                             train_ratio=0.7,
                             n_splits=3):
    """
    纯计算部分：切分数据、训练模型、算 metrics，不画图不存图。

    新增参数:
        use_expanding_window: True 时使用扩展窗口，False 时保持原来的固定 8:2 时间切分
        train_ratio: 扩展窗口初始训练集比例（默认 0.7）
        n_splits: 扩展窗口评估次数（默认 3，最后一次窗口的结果作为最终报告）

    返回:
        corrector: 训练好的 ResidualCorrectionModel（最后一个窗口训练的）
        metrics: dict，各项评估指标（固定切分 或 最后一次窗口）
        aux: dict，画图需要的中间变量
    """
    prefix = 'Top' if surface == 'Top' else 'Bot'
    surface_cn = '上' if surface == 'Top' else '下'

    df = df.copy()
    speed_col = 'Speed[m/min]_Process_Avg'
    current_col = f'{prefix}_Current_Sum'
    online_col = f'Tin Weight_Actual[g/m2]_GALV_WEIGHT_{prefix.upper()}_Avg'
    setpoint_weight = f'Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_{prefix.upper()}_Avg'

    df[f'{prefix}_Current_Per_Speed'] = df[current_col] / (df[speed_col] + 1e-5)
    df[f'{prefix}_Weight_Deviation'] = df[setpoint_weight] - df[online_col]

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

    X = df[feature_cols]
    delta_col = f'{prefix}_Delta'
    y_delta = df[delta_col]
    online_actual = df[online_col]
    y_true_full = df[f'{surface_cn}表面镀层重量A(XA1_0)']

    n_samples = len(X)

    USE_LINEAR_MODEL = False  # 想切回树模型时改这里
    ModelClass = LinearResidualCorrectionModel if USE_LINEAR_MODEL else ResidualCorrectionModel

    if n_samples < 50:
        use_expanding_window = False

    # ------------------------------------------------------------------
    # 分支 1：原来的固定 8:2 时间切分
    # ------------------------------------------------------------------
    if not use_expanding_window:
        X_train, X_test, y_delta_train, y_delta_test, actual_train, actual_test, y_true_train, y_true_test = \
            train_test_split(X, y_delta, online_actual, y_true_full, test_size=0.2, shuffle=False)

        corrector = _build_corrector_instance(params, ModelClass)
        corrector.fit(X_train, y_delta_train)

        # 测试集预测
        pred_series, _ = corrector.predict_smooth(X_test, actual_test)
        y_true_series = y_true_test
        online_series = actual_test

        raw_residuals = y_true_series - online_series
        model_residuals = y_true_series - pred_series

        # 训练集预测
        pred_train, _ = corrector.predict_smooth(X_train, actual_train)
        raw_residuals_train = y_true_train - actual_train
        model_residuals_train = y_true_train - pred_train

    # ------------------------------------------------------------------
    # 分支 2：扩展窗口（Expanding Window）
    # ------------------------------------------------------------------
    else:
        indices = np.arange(n_samples)
        min_train_size = int(n_samples * train_ratio)
        remaining = n_samples - min_train_size
        step = max(remaining // n_splits, 1)

        window_metrics_list = []
        last_corrector = None
        last_aux = None

        for i in range(n_splits):
            train_end = min_train_size + i * step
            if i == n_splits - 1:
                train_end = n_samples - max(int(n_samples * 0.15), 20)
                test_end = n_samples
            else:
                test_end = min(train_end + step, n_samples)

            if train_end >= test_end or train_end < 30:
                continue

            train_idx = indices[:train_end]
            test_idx = indices[train_end:test_end]

            X_train = X.iloc[train_idx]
            y_delta_train = y_delta.iloc[train_idx]
            actual_train = online_actual.iloc[train_idx]
            y_true_train = y_true_full.iloc[train_idx]

            X_test = X.iloc[test_idx]
            actual_test = online_actual.iloc[test_idx]
            y_true_test = y_true_full.iloc[test_idx]

            corrector = _build_corrector_instance(params, ModelClass)
            corrector.fit(X_train, y_delta_train)

            pred_series, _ = corrector.predict_smooth(X_test, actual_test)
            raw_residuals = y_true_test - actual_test
            model_residuals = y_true_test - pred_series

            pred_train, _ = corrector.predict_smooth(X_train, actual_train)
            raw_residuals_train = y_true_train - actual_train
            model_residuals_train = y_true_train - pred_train

            window_mae = model_residuals.abs().mean()
            window_metrics_list.append({
                'window': i + 1,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'mae_model': window_mae,
                'mae_online': raw_residuals.abs().mean(),
            })

            last_corrector = corrector
            last_aux = dict(
                X_test=X_test,
                y_true_series=y_true_test,
                online_series=actual_test,
                pred_series=pred_series,
                raw_residuals=raw_residuals,
                model_residuals=model_residuals,
                raw_residuals_train=raw_residuals_train,
                model_residuals_train=model_residuals_train,
                X_train=X_train,
                y_true_train=y_true_train,
                actual_train=actual_train,
            )

        if last_corrector is None:
            return fit_and_evaluate_surface(df, surface, params, group_tag=group_tag,
                                            use_expanding_window=False)

        corrector = last_corrector
        aux_from_window = last_aux
        X_test = aux_from_window['X_test']
        y_true_series = aux_from_window['y_true_series']
        online_series = aux_from_window['online_series']
        pred_series = aux_from_window['pred_series']
        raw_residuals = aux_from_window['raw_residuals']
        model_residuals = aux_from_window['model_residuals']
        raw_residuals_train = aux_from_window['raw_residuals_train']
        model_residuals_train = aux_from_window['model_residuals_train']
        X_train = aux_from_window['X_train']
        y_true_train = aux_from_window['y_true_train']
        actual_train = aux_from_window['actual_train']

        if window_metrics_list:
            print(f"  [扩展窗口] {group_tag}-{surface} 各窗口 MAE: "
                  + ", ".join([f"W{w['window']}={w['mae_model']:.4f}" for w in window_metrics_list]))

    # ------------------------------------------------------------------
    # 统一计算指标（按原始残差的掩码，统一正/负偏差的评价基准）
    # ------------------------------------------------------------------
    overall_mae_model = model_residuals.abs().mean()
    overall_mae_online = raw_residuals.abs().mean()
    # 1. 全量数据原始残差正负样本数
    raw_residuals_full = y_true_full - online_actual
    total_pos_count = int((raw_residuals_full < 0).sum())
    total_neg_count = int((raw_residuals_full > 0).sum())

    # 2. 训练集原始残差正负样本数
    train_pos_count = int((raw_residuals_train < 0).sum())
    train_neg_count = int((raw_residuals_train > 0).sum())


    # 1. 统一以【原始在线残差】划分正/负偏差样本掩码
    mask_pos = (raw_residuals < 0)
    mask_neg = (raw_residuals > 0)

    pos_count = int(mask_pos.sum())
    neg_count = int(mask_neg.sum())

    # 2. 在【相同掩码】下计算在线与模型的 MAE / RMSE
    pos_mae_online = raw_residuals[mask_pos].abs().mean() if pos_count > 0 else np.nan
    pos_mae_model = model_residuals[mask_pos].abs().mean() if pos_count > 0 else np.nan
    pos_rmse_online = np.sqrt((raw_residuals[mask_pos] ** 2).mean()) if pos_count > 0 else np.nan
    pos_rmse_model = np.sqrt((model_residuals[mask_pos] ** 2).mean()) if pos_count > 0 else np.nan

    neg_mae_online = raw_residuals[mask_neg].abs().mean() if neg_count > 0 else np.nan
    neg_mae_model = model_residuals[mask_neg].abs().mean() if neg_count > 0 else np.nan
    neg_rmse_online = np.sqrt((raw_residuals[mask_neg] ** 2).mean()) if neg_count > 0 else np.nan
    neg_rmse_model = np.sqrt((model_residuals[mask_neg] ** 2).mean()) if neg_count > 0 else np.nan

    # 3. 计算提升量（无样本时差值自动为 NaN）
    pos_mae_diff = pos_mae_model - pos_mae_online if pos_count > 0 else np.nan
    pos_rmse_diff = pos_rmse_model - pos_rmse_online if pos_count > 0 else np.nan
    neg_mae_diff = neg_mae_model - neg_mae_online if neg_count > 0 else np.nan
    neg_rmse_diff = neg_rmse_model - neg_rmse_online if neg_count > 0 else np.nan

    # 4. 重新封装适配 aux 的字典，防止报错，并保证兼容性
    test_directional_metrics_raw = {
        'pos_count': pos_count,
        'neg_count': neg_count,
        'pos_mae': pos_mae_online,
        'neg_mae': neg_mae_online,
        'pos_rmse': pos_rmse_online,
        'neg_rmse': neg_rmse_online,
    }
    test_directional_metrics_model = {
        'pos_count': pos_count,
        'neg_count': neg_count,
        'pos_mae': pos_mae_model,
        'neg_mae': neg_mae_model,
        'pos_rmse': pos_rmse_model,
        'neg_rmse': neg_rmse_model,
    }

    r2_online = r2_score(y_true_series, online_series)
    r2_model = r2_score(y_true_series, pred_series)
    rmse_online = np.sqrt(mean_squared_error(y_true_series, online_series))
    rmse_model = np.sqrt(mean_squared_error(y_true_series, pred_series))

    r2_online_train = r2_score(y_true_train, actual_train)
    r2_model_train = r2_score(y_true_train, pred_train)
    rmse_online_train = np.sqrt(mean_squared_error(y_true_train, actual_train))
    rmse_model_train = np.sqrt(mean_squared_error(y_true_train, pred_train))
    mae_model_train = model_residuals_train.abs().mean()
    mae_online_train = raw_residuals_train.abs().mean()

    overfitting_r2 = r2_model_train - r2_model
    overfitting_mae = mae_model_train - overall_mae_model
    overfitting_rmse = rmse_model_train - rmse_model
    mae_ratio = overall_mae_model / (mae_model_train + 1e-8)
    rmse_ratio = rmse_model / (rmse_model_train + 1e-8)

    metrics = {
        '规格组': group_tag,
        '表面': surface,
        '训练样本数': len(X_train),
        '测试样本数': len(X_test),
        '正偏差样本数_总': total_pos_count,
        '负偏差样本数_总': total_neg_count,
        '使用扩展窗口': use_expanding_window,

        # ---- 训练集 ----
        '正偏差样本数_训练': train_pos_count,
        '负偏差样本数_训练': train_neg_count,
        'R2_在线_训练': r2_online_train,
        'R2_模型_训练': r2_model_train,
        'R2_提升_训练': r2_model_train - r2_online_train,
        'RMSE_在线_训练': rmse_online_train,
        'RMSE_模型_训练': rmse_model_train,
        'RMSE_提升_训练': rmse_model_train - rmse_online_train,
        'MAE_在线_训练': mae_online_train,
        'MAE_模型_训练': mae_model_train,
        'MAE_提升_训练': mae_model_train - mae_online_train,

        # ---- 测试集 ----
        'R2_在线_测试': r2_online,
        'R2_模型_测试': r2_model,
        'R2_提升_测试': r2_model - r2_online,
        'RMSE_在线_测试': rmse_online,
        'RMSE_模型_测试': rmse_model,
        'RMSE_提升_测试(%)': (rmse_online - rmse_model) / rmse_online * 100 if rmse_online != 0 else 0.0,
        'MAE_在线_测试': overall_mae_online,
        'MAE_模型_测试': overall_mae_model,
        'MAE_提升_测试': overall_mae_model - overall_mae_online,
        '偏差均值_在线_测试': raw_residuals.mean(),
        '偏差均值_模型_测试': model_residuals.mean(),
        '偏差均值_提升_测试': model_residuals.mean() - raw_residuals.mean(),
        '正偏差样本数_测试': pos_count,
        '正偏差MAE_在线_测试': pos_mae_online,
        '正偏差MAE_模型_测试': pos_mae_model,
        '正偏差MAE_提升_测试': pos_mae_diff,
        '正偏差RMSE_在线_测试': pos_rmse_online,
        '正偏差RMSE_模型_测试': pos_rmse_model,
        '正偏差RMSE_提升_测试': pos_rmse_diff,
        '负偏差样本数_测试': neg_count,
        '负偏差MAE_在线_测试': neg_mae_online,
        '负偏差MAE_模型_测试': neg_mae_model,
        '负偏差MAE_提升_测试': neg_mae_diff,
        '负偏差RMSE_在线_测试': neg_rmse_online,
        '负偏差RMSE_模型_测试': neg_rmse_model,
        '负偏差RMSE_提升_测试': neg_rmse_diff,

        # ---- 过拟合程度 ----
        '过拟合程度_R2': overfitting_r2,
        '过拟合程度_MAE': overfitting_mae,
        '过拟合程度_RMSE': overfitting_rmse,
        'MAE比值_测试/训练': mae_ratio,
        'RMSE比值_测试/训练': rmse_ratio,
    }

    aux = dict(
        X_test=X_test,
        y_true_series=y_true_series,
        online_series=online_series,
        pred_series=pred_series,
        raw_residuals=raw_residuals,
        model_residuals=model_residuals,
        raw_residuals_train=raw_residuals_train,
        model_residuals_train=model_residuals_train,
        test_directional_metrics_raw=test_directional_metrics_raw,
        test_directional_metrics_model=test_directional_metrics_model,
    )
    return corrector, metrics, aux

# ==========================================
# 5. 表面建模与图形输出（薄壳：相关性分析 + 调用纯计算 + 画图）
# ==========================================
def run_surface_pipeline(df, surface='Top', group_tag="", group_params=None,
                         use_expanding_window=False, train_ratio=0.7, n_splits=3):
    surface_cn = '上' if surface == 'Top' else '下'
    tag_display = f"[{group_tag}] " if group_tag else ""
    safe_tag = f"_{group_tag}" if group_tag else ""

    params = group_params or DEFAULT_PARAMS

    print(f"\n==========================================")
    print(f"    开始运行【{tag_display}{surface_cn}表面】模型拟合与分析     ")
    print(f"        使用参数: {params}")
    print(f"==========================================")


    # # 1. 相关性分析
    # save_dir = f"result/grouped_by_coating_weight/correlation_result/correlation_result{safe_tag}"
    # analyzer = SurfaceCorrelationAnalyzer(default_save_dir=save_dir)
    # analyzer.analyze_surface(
    #     df,
    #     surface=surface,
    #     extra_cols=None,
    #     save_dir=save_dir,
    #     corr_method='both',  # 同时输出 Pearson 和 Spearman
    #     compute_mi=True      # 计算 Mutual Information
    # )

    # 2. 纯计算：切分、训练、算指标
    corrector, metrics, aux = fit_and_evaluate_surface(
        df, surface, params, group_tag=group_tag,
        use_expanding_window=use_expanding_window,
        train_ratio=train_ratio,
        n_splits=n_splits
    )

    X_test = aux['X_test']
    y_true_series = aux['y_true_series']
    online_series = aux['online_series']
    pred_series = aux['pred_series']
    raw_residuals = aux['raw_residuals']
    model_residuals = aux['model_residuals']
    # 新增
    raw_residuals_train = aux['raw_residuals_train']
    model_residuals_train = aux['model_residuals_train']
    test_directional_metrics = aux['test_directional_metrics_raw']
    test_directional_metrics_model = aux['test_directional_metrics_model']

    # # ========== 训练后 EDA（模型残差） ==========
    # eda_post_dir = f"result/grouped_by_coating_weight/eda_post/{group_tag}_{surface}"
    # eda_post = SurfaceEDAAnalyzer(default_save_dir=eda_post_dir)
    #
    # # 方式 A：直接传 residual Series + 特征（推荐，最干净）
    # # 测试集模型残差
    # eda_post.analyze_residual_series(
    #     residual=model_residuals,  # True - Model
    #     features=X_test,  # 与训练特征一致
    #     # time_series=df.loc[X_test.index, "Produce Time"] if "Produce Time" in df.columns else None,
    #     train_idx=None,  # 如需 train/test 对比可再传
    #     test_idx=None,
    #     save_dir=os.path.join(eda_post_dir, "model_residual_test"),
    #     title_prefix=f"{tag_display}{surface_cn}模型残差(测试)",
    # )
    #
    # # 可选：原始在线残差也跑一遍，方便对比
    # eda_post.analyze_residual_series(
    #     residual=raw_residuals,  # True - Online
    #     features=X_test,
    #     save_dir=os.path.join(eda_post_dir, "raw_residual_test"),
    #     title_prefix=f"{tag_display}{surface_cn}原始在线残差(测试)",
    # )

    # 方式 B：用 analyze + y_true/y_pred（等价）
    # eda_post.analyze(
    #     df=df.loc[X_test.index],
    #     y_true=y_true_series,
    #     y_pred=pred_series,
    #     feature_cols=feature_cols,
    #     surface=surface,
    #     save_dir=eda_post_dir,
    #     plot_train_test=False,
    #     plot_model_residual=False,
    # )


    print(f"\n-------- 【{tag_display}{surface_cn}表面 模型矫正前后残差诊断】 --------")
    # 使用已计算的 mask_pos 和 mask_neg
    if test_directional_metrics['pos_count'] > 0:
        mae_raw_pos = test_directional_metrics['pos_mae']
        mae_model_pos = test_directional_metrics_model['pos_mae']
        print(
            f"当原始在线偏低 (残差 > 0, 样本数 {test_directional_metrics['pos_count']}): 原始 MAE = {mae_raw_pos:.4f}  -->  模型矫正后 MAE = {mae_model_pos:.4f}")
    if test_directional_metrics['neg_count'] > 0:
        mae_raw_neg = test_directional_metrics['neg_mae']
        mae_model_neg = test_directional_metrics_model['neg_mae']
        print(
            f"当原始在线偏高 (残差 < 0, 样本数 {test_directional_metrics['neg_count']}): 原始 MAE = {mae_raw_neg:.4f}  -->  模型矫正后 MAE = {mae_model_neg:.4f}")
    print("------------------------------------------------------\n")

    print(f"======== 【{tag_display}{surface_cn}表面 拟合性能评估】 ========")
    print(f"【训练集】")
    print( f"  原始在线 -> R²: {metrics['R2_在线_训练']:.4f}, RMSE: {metrics['RMSE_在线_训练']:.4f}, MAE: {metrics['MAE_在线_训练']:.4f}")
    print(f"  模型校正 -> R²: {metrics['R2_模型_训练']:.4f}, RMSE: {metrics['RMSE_模型_训练']:.4f}, MAE: {metrics['MAE_模型_训练']:.4f}")
    print(f"【测试集】")
    print(f"  原始在线 -> R²: {metrics['R2_在线_测试']:.4f}, RMSE: {metrics['RMSE_在线_测试']:.4f}, MAE: {metrics['MAE_在线_测试']:.4f}")
    print(f"  模型校正 -> R²: {metrics['R2_模型_测试']:.4f}, RMSE: {metrics['RMSE_模型_测试']:.4f}, MAE: {metrics['MAE_模型_测试']:.4f}")


    start_idx = X_test.index[0]
    end_idx = X_test.index[-1]

    # 3. 拟合对比图
    plt.figure(figsize=(12, 5))
    plt.plot(y_true_series, label='实验室真实测量值 (True Label)', color='black', linewidth=1.5)
    plt.plot(online_series, label='在线仪表原始测量值 (Online)', color='red', linestyle='--', alpha=0.7)
    plt.plot(pred_series, label='模型残差校正值 (Model Pred)', color='green', linewidth=1.5, alpha=0.85)
    plt.title(f'{tag_display}{surface_cn}表面 镀层重量拟合对照图（原始数据行号: {start_idx} ~ {end_idx}）')
    plt.xlabel('原始数据行号 (Original Row Index)')
    plt.ylabel('镀层重量 (g/m2)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    fit_img_path = f"result/grouped_by_coating_weight/fitting_result/fitting_result/fitting_result_{surface}{safe_tag}.png"
    os.makedirs(os.path.dirname(fit_img_path), exist_ok=True)
    plt.savefig(fit_img_path, dpi=300)
    print(f"[图表保存] {tag_display}{surface_cn}表面拟合对照图已保存至: {fit_img_path}")
    plt.close()

    # 4. 残差对比图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})

    ax1.plot(raw_residuals, label='原始在线仪表残差 (True - Online)', color='red', alpha=0.5, linewidth=1)
    ax1.plot(model_residuals, label='模型校正后残差 (True - Model)', color='green', alpha=0.8, linewidth=1.2)
    ax1.axhline(0, color='black', linestyle='--', linewidth=1)
    ax1.set_title(f'{tag_display}{surface_cn}表面 预测残差变化对比（原始数据行号: {start_idx} ~ {end_idx}）')
    ax1.set_xlabel('原始数据行号 (Original Row Index)')
    ax1.set_ylabel('残差/误差 (g/m2)')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    sns.histplot(raw_residuals, ax=ax2, color='red', label='原始残差分布', kde=True, stat="density", alpha=0.3)
    sns.histplot(model_residuals, ax=ax2, color='green', label='模型校正后残差分布', kde=True, stat="density",
                 alpha=0.5)
    ax2.axvline(0, color='black', linestyle='--', linewidth=1)
    ax2.set_title(f'{tag_display}{surface_cn}表面 残差概率密度分布（越集中在0且越窄越好）')
    ax2.set_xlabel('残差/误差 (g/m2)')
    ax2.set_ylabel('概率密度')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    res_img_path = f"result/grouped_by_coating_weight/fitting_result/residual_analysis/residual_analysis_{surface}{safe_tag}.png"
    os.makedirs(os.path.dirname(res_img_path), exist_ok=True)
    plt.savefig(res_img_path, dpi=300)
    print(f"[图表保存] {tag_display}{surface_cn}表面残差分析图已保存至: {res_img_path}")
    plt.close()

    # ========== 5. 新增：训练集 vs 测试集 残差分布对比图 ==========
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：原始在线残差（True - Online）
    _plot_residual_distribution(
        axes[0], raw_residuals_train, raw_residuals,
        'orange', 'red',
        '训练集 原始残差', '测试集 原始残差',
        f'{tag_display}{surface_cn}表面 原始在线残差分布\n(训练集 vs 测试集)',
        '残差 (True - Online) g/m2'
    )

    # 右图：模型校正后残差（True - Model）
    _plot_residual_distribution(
        axes[1], model_residuals_train, model_residuals,
        'cyan', 'green',
        '训练集 模型残差', '测试集 模型残差',
        f'{tag_display}{surface_cn}表面 模型校正后残差分布\n(训练集 vs 测试集)',
        '残差 (True - Model) g/m2'
    )

    plt.tight_layout()
    dist_img_path = (f"result/grouped_by_coating_weight/fitting_result/"
                     f"residual_train_vs_test/residual_train_vs_test_{surface}{safe_tag}.png")
    os.makedirs(os.path.dirname(dist_img_path), exist_ok=True)
    plt.savefig(dist_img_path, dpi=300)
    print(f"[图表保存] {tag_display}{surface_cn}表面 训练集vs测试集残差分布对比图已保存至: {dist_img_path}")
    plt.close()

    # ===== 模型解释性分析 =====
    # 特征列使用统一的 get_feature_cols 函数，保持与 fit_and_evaluate_surface 一致
    feature_cols = get_feature_cols(surface)

    interp_dir = f"result/grouped_by_coating_weight/interpretation/{group_tag}_{surface}"
    interpreter = ModelInterpreter(
        model=corrector,  # 直接传包装类即可，内部会取 .model
        X=aux.get('X_train', X_test),  # 优先用训练集
        feature_names=feature_cols,
        save_dir=interp_dir,
        max_samples_for_shap=500,
    )

    # 一键跑完（也可单独调用某个方法）
    interpreter.full_analysis(
        y=None,  # 如果想做 permutation，需要把 y_delta_train 也从 aux 里带出来
        run_permutation=False,  # 暂时可先关掉，等 aux 补全 y 再开
        run_shap=True,
        run_pdp=True,
        pdp_features=feature_cols[:5],
    )


    return corrector, metrics

def get_feature_cols(surface: str) -> list:
    prefix = "Top" if surface == "Top" else "Bot"
    return [
        f"Tin Weight_Actual[g/m2]_GALV_WEIGHT_{prefix.upper()}_Avg",
        f'{prefix}_Weight_Deviation',
        f"{prefix}_Current_Sum",
        f"{prefix}_Current_Per_Speed",
        f"{prefix}_Theoretical_Factor",
        "Speed[m/min]_Process_Avg",
        "Dimension_[mm]_Width",
        "Dimension_[mm]_Thickness",
        "Steel_Grade_Encoded",
    ]




# ==========================================
# 6. 主流程：按镀层规格分组，逐组训练 Top/Bot 模型
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="镀层重量分规格组模型训练脚本")
    # parser.add_argument(                                           ## 所有组都使用同样的超参数，即对Top2.799_Bot2.799最优的超参数
    #     "--config", type=str, default="group_params_all_the_same.json",
    #     help="配置文件 JSON 路径 (默认: group_params_all_the_same.json)"
    # )
    parser.add_argument(                                           ## 所有组都使用同样的超参数，即对Top2.799_Bot2.799最优的超参数
        "--config", type=str, default="group_params_all_the_same_2.json",
        help="配置文件 JSON 路径 (默认: group_params_all_the_same_2.json)"
    )
    # parser.add_argument(
    #     "--config", type=str, default="group_params_optimum_for_each.json",  ## 使用Optuna对各组搜索出来的最优的超参数
    #     help="配置文件 JSON 路径 (默认: group_params_optimum_for_each.json)"
    # )
    # parser.add_argument(
    #     "--config", type=str, default="group_params_optimum_for_each_optimal.json",  ## 使用Optuna对各组搜索出来的最优的超参数
    #     help="配置文件 JSON 路径 (默认: group_params_optimum_for_each_optimal.json)"
    # )
    # parser.add_argument(
    #     "--config", type=str, default="group_params_optimum_for_each_reduce_overfitting.json",  ## 降低模型过拟合的超参数配置
    #     help="配置文件 JSON 路径 (默认: group_params_optimum_for_each_reduce_overfitting.json)"
    # )


    args = parser.parse_args()

    # 1. 加载统一配置
    config = load_pipeline_config(args.config)
    MIN_GROUP_SAMPLES = config.get("min_group_samples", 200)

    # 2. 读取数据并分组汇总
    clean_df = pd.read_excel(config.get("data_paths", {}).get("clean_data", "result/data/feature_engineered_data/featured_data.xlsx"))
    clean_df = build_setpoint_group_key(clean_df)
    group_sizes = summarize_setpoint_groups(clean_df)

    # # 全量训练前 EDA
    # valid_labels = group_sizes[group_sizes >= MIN_GROUP_SAMPLES].index.tolist()

    # for surface in ["Top", "Bot"]:
    #     # 确保衍生列存在（与训练时一致）
    #     prefix = "Top" if surface == "Top" else "Bot"
    #     speed_col = "Speed[m/min]_Process_Avg"
    #     current_col = f"{prefix}_Current_Sum"
    #     per_speed = f"{prefix}_Current_Per_Speed"
    #     weight_deviation = f"{prefix}_Weight_Deviation"
    #     online_col = f'Tin Weight_Actual[g/m2]_GALV_WEIGHT_{prefix.upper()}_Avg'
    #     setpoint_weight = f'Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_{prefix.upper()}_Avg'
    #
    #     clean_df = clean_df.copy()
    #
    #     # 衍生 Current_Per_Speed
    #     if current_col in clean_df.columns and speed_col in clean_df.columns:
    #         if per_speed not in clean_df.columns:
    #             clean_df[per_speed] = clean_df[current_col] / (clean_df[speed_col] + 1e-5)
    #
    #     # 衍生 Weight_Deviation
    #     if online_col in clean_df.columns and setpoint_weight in clean_df.columns:
    #         if weight_deviation not in clean_df.columns:
    #             clean_df[weight_deviation] = clean_df[setpoint_weight] - clean_df[online_col]
    #
    #     eda_global = SurfaceEDAAnalyzer(
    #         default_save_dir=f"result/grouped_by_coating_weight/eda_pre_global/{surface}"
    #     )
    #     eda_global.analyze(
    #         df=clean_df,
    #         surface=surface,
    #         feature_cols=get_feature_cols(surface),  # ← 与训练完全一致
    #         group_col="Setpoint_Group_Label",
    #         target_groups=valid_labels,
    #         time_col="Produce Time",
    #         plot_train_test=False,
    #         plot_model_residual=False,
    #         max_groups=20,
    #     )


    # 3. 过滤要运行的目标规格组
    target_groups = config.get("target_groups")
    if target_groups:
        target_set = set(target_groups)
        # 仅保留 JSON 里面配置且在数据中存在的组
        group_sizes = group_sizes[group_sizes.index.isin(target_set)]
        print(f"[配置生效] 根据 JSON 配置，仅运行指定的 {len(group_sizes)} 个规格组。")
    else:
        print("[配置生效] 未在 JSON 中指定 target_groups，将运行所有样本量达标的规格组。")

    # 4. 逐组训练
    trained_models = {}
    all_metrics = []

    for group_label, group_size in group_sizes.items():
        if group_size < MIN_GROUP_SAMPLES:
            print(f"[跳过] 规格组 {group_label} 样本量 {group_size} < {MIN_GROUP_SAMPLES}")
            continue

        group_df = clean_df[clean_df['Setpoint_Group_Label'] == group_label].copy()
        
        # 确保衍生特征存在（与 fit_and_evaluate_surface 中的衍生一致）
        for surface in ['Top', 'Bot']:
            prefix = "Top" if surface == "Top" else "Bot"
            speed_col = "Speed[m/min]_Process_Avg"
            current_col = f"{prefix}_Current_Sum"
            per_speed = f"{prefix}_Current_Per_Speed"
            weight_deviation = f"{prefix}_Weight_Deviation"
            online_col = f'Tin Weight_Actual[g/m2]_GALV_WEIGHT_{prefix.upper()}_Avg'
            setpoint_weight = f'Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_{prefix.upper()}_Avg'
            
            if current_col in group_df.columns and speed_col in group_df.columns:
                if per_speed not in group_df.columns:
                    group_df[per_speed] = group_df[current_col] / (group_df[speed_col] + 1e-5)
            
            if online_col in group_df.columns and setpoint_weight in group_df.columns:
                if weight_deviation not in group_df.columns:
                    group_df[weight_deviation] = group_df[setpoint_weight] - group_df[online_col]

        # 分别获取 Top / Bot 的参数（自动处理了默认值 + 覆盖值）
        top_params = get_params_for_group(config, group_label, 'Top')
        bot_params = get_params_for_group(config, group_label, 'Bot')

        # # 高漂移组名单（根据样本数据漂移诊断调整）
        # HIGH_DRIFT_GROUPS = {
        #     "Top2.2_Bot2.2",
        #     "Top2.2_Bot1.7",
        #     "Top1.1_Bot2.2",
        #     "Top2.0_Bot5.0",
        #     "Top1.1_Bot2.799",
        # }
        #
        # use_window = group_label in HIGH_DRIFT_GROUPS
        #
        # top_model, top_metrics = run_surface_pipeline(
        #     group_df, surface='Top', group_tag=group_label, group_params=top_params,
        #     use_expanding_window=use_window, train_ratio=0.7, n_splits=3
        # )
        # bot_model, bot_metrics = run_surface_pipeline(
        #     group_df, surface='Bot', group_tag=group_label, group_params=bot_params,
        #     use_expanding_window=use_window, train_ratio=0.7, n_splits=3
        # )
        top_model, top_metrics = run_surface_pipeline(
            group_df, surface='Top', group_tag=group_label, group_params=top_params,
            use_expanding_window=False, train_ratio=0.7, n_splits=3
        )
        bot_model, bot_metrics = run_surface_pipeline(
            group_df, surface='Bot', group_tag=group_label, group_params=bot_params,
            use_expanding_window=False, train_ratio=0.7, n_splits=3
        )

        trained_models[(group_label, 'Top')] = top_model
        trained_models[(group_label, 'Bot')] = bot_model
        all_metrics.append(top_metrics)
        all_metrics.append(bot_metrics)

    # 5. 导出结果报表...
    print("\n==========================================")
    print(f"全部规格组处理完毕，共成功训练 {len(trained_models)} 个模型（每组 Top/Bot 各一个）。")
    print("==========================================")

    # 导出样本量汇总表 + 建模效果汇总表到同一个 Excel 的不同 sheet
    sample_summary_df = group_sizes.reset_index()
    sample_summary_df.columns = ['规格组', '样本数']
    sample_summary_df['状态'] = sample_summary_df['样本数'].apply(
        lambda s: '建模' if s >= MIN_GROUP_SAMPLES else '跳过'
    )

    report_path = "result/grouped_by_coating_weight/summary_report_group_optimum_all_the_same_2.xlsx"
    # report_path = "result/grouped_by_coating_weight/summary_report_group_optimum_for_each.xlsx"
    # report_path = "result/grouped_by_coating_weight/summary_report_group_optimum_for_each_optimal.xlsx"
    # report_path = "result/grouped_by_coating_weight/summary_report_group_optimum_for_each_reducing_overfitting.xlsx"  # 降低过拟合
    # report_path = "result/grouped_by_coating_weight/summary_report_group_optimum_for_each_expanding_window.xlsx"
    # report_path = "result/grouped_by_coating_weight/summary_report_group_optimum_for_each_linear_for_overfitting.xlsx"
    # report_path = "result/grouped_by_coating_weight/summary_report_group_optimum_for_each_linear_model.xlsx"


    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
        sample_summary_df.to_excel(writer, sheet_name='样本量汇总', index=False)
        if all_metrics:
            metrics_df = pd.DataFrame(all_metrics)
            # ==========================================
            # 针对 Excel 报表的针对性置空清洗
            # 当正/负偏差样本数为 0 时，将其 MAE 设为 np.nan (导出 Excel 即为空白)
            # ==========================================
            mask_pos_zero = (metrics_df['正偏差样本数_测试'] == 0)
            metrics_df.loc[mask_pos_zero, '正偏差MAE_在线_测试'] = np.nan
            metrics_df.loc[mask_pos_zero, '正偏差MAE_模型_测试'] = np.nan

            mask_neg_zero = (metrics_df['负偏差样本数_测试'] == 0)
            metrics_df.loc[mask_neg_zero, '负偏差MAE_在线_测试'] = np.nan
            metrics_df.loc[mask_neg_zero, '负偏差MAE_模型_测试'] = np.nan

            mask_pos_zero = (metrics_df['正偏差样本数_测试'] == 0)
            metrics_df.loc[mask_pos_zero, ['正偏差MAE_在线_测试', '正偏差MAE_模型_测试',
                                           '正偏差RMSE_在线_测试', '正偏差RMSE_模型_测试']] = np.nan

            mask_neg_zero = (metrics_df['负偏差样本数_测试'] == 0)
            metrics_df.loc[mask_neg_zero, ['负偏差MAE_在线_测试', '负偏差MAE_模型_测试',
                                           '负偏差RMSE_在线_测试', '负偏差RMSE_模型_测试']] = np.nan
            # ==========================================
            metrics_df.to_excel(writer, sheet_name='建模效果汇总', index=False)
        else:
            pd.DataFrame({'提示': ['没有达标组完成建模']}).to_excel(
                writer, sheet_name='建模效果汇总', index=False
            )

    print(f"[导出提示] 汇总报表已保存至: {report_path}")