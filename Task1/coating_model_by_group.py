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
    print(f"      【{tag_display}数据集中原始残差方向分布诊断】       ")
    print("==========================================")
    for surface in ['Top', 'Bot']:
        surface_cn = '上' if surface == 'Top' else '下'
        delta_col = f'{surface}_Delta'
        if delta_col in df.columns:
            total = len(df[delta_col].dropna())
            if total == 0:
                continue
            # Delta > 0 → 在线偏低；Delta < 0 → 在线偏高
            low_count = (df[delta_col] > 0).sum()   # 在线偏低
            high_count = (df[delta_col] < 0).sum()  # 在线偏高
            mean_val = df[delta_col].mean()
            print(f"[{surface_cn}表面 Delta (实验室值 - 在线值)]")
            print(f"  - 总有效样本数: {total}")
            print(f"  - Delta > 0 (在线测量偏低): {low_count} 条 (占比 {low_count/total*100:.2f}%)")
            print(f"  - Delta < 0 (在线测量偏高): {high_count} 条 (占比 {high_count/total*100:.2f}%)")
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

    pos_mask = y_delta > 0  # 在线偏低
    neg_mask = y_delta < 0  # 在线偏高
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
    计算残差的在线偏高 / 在线偏低 相关指标

    residual = y_true - online
      residual < 0 → 在线测量偏高  → high
      residual > 0 → 在线测量偏低  → low

    Returns:
        dict: mask_high, mask_low, high_count, low_count, high_mae, low_mae, high_rmse, low_rmse
    """
    mask_high = (residuals < 0)  # 在线偏高
    mask_low = (residuals > 0)   # 在线偏低

    high_count = int(mask_high.sum())
    low_count = int(mask_low.sum())

    high_mae = residuals[mask_high].abs().mean() if high_count > 0 else np.nan
    low_mae = residuals[mask_low].abs().mean() if low_count > 0 else np.nan

    high_rmse = np.sqrt((residuals[mask_high] ** 2).mean()) if high_count > 0 else np.nan
    low_rmse = np.sqrt((residuals[mask_low] ** 2).mean()) if low_count > 0 else np.nan

    return {
        'mask_high': mask_high,
        'mask_low': mask_low,
        'high_count': high_count,
        'low_count': low_count,
        'high_mae': high_mae,
        'low_mae': low_mae,
        'high_rmse': high_rmse,
        'low_rmse': low_rmse,
    }


def _plot_residual_distribution(ax, residuals_train, residuals_test,
                                 color_train, color_test,
                                 label_train, label_test, title, xlabel):
    """
    绘制残差分布直方图（训练集 vs 测试集对比）
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


def _calc_set_metrics(y_true, online, pred, raw_residuals, model_residuals, suffix=""):
    """
    计算某一数据划分（train / val / test）上的完整指标字典。
    suffix 用于区分，例如 "_训练" / "_验证" / "_测试"

    方向命名约定：
      在线偏高 = residual < 0
      在线偏低 = residual > 0
    """
    dir_raw = _compute_directional_metrics(raw_residuals)
    dir_model = _compute_directional_metrics(model_residuals)

    # 掩码基于原始在线残差（与历史逻辑一致）
    mask_high = dir_raw['mask_high']
    mask_low = dir_raw['mask_low']
    high_count = dir_raw['high_count']
    low_count = dir_raw['low_count']

    high_mae_online = dir_raw['high_mae']
    low_mae_online = dir_raw['low_mae']
    high_rmse_online = dir_raw['high_rmse']
    low_rmse_online = dir_raw['low_rmse']

    # 模型在相同掩码下的指标
    high_mae_model = model_residuals[mask_high].abs().mean() if high_count > 0 else np.nan
    low_mae_model = model_residuals[mask_low].abs().mean() if low_count > 0 else np.nan
    high_rmse_model = np.sqrt((model_residuals[mask_high] ** 2).mean()) if high_count > 0 else np.nan
    low_rmse_model = np.sqrt((model_residuals[mask_low] ** 2).mean()) if low_count > 0 else np.nan

    overall_mae_online = raw_residuals.abs().mean()
    overall_mae_model = model_residuals.abs().mean()

    r2_online = r2_score(y_true, online) if len(y_true) > 1 else np.nan
    r2_model = r2_score(y_true, pred) if len(y_true) > 1 else np.nan
    rmse_online = np.sqrt(mean_squared_error(y_true, online)) if len(y_true) > 0 else np.nan
    rmse_model = np.sqrt(mean_squared_error(y_true, pred)) if len(y_true) > 0 else np.nan

    metrics = {
        f'在线偏高样本数{suffix}': high_count,
        f'在线偏低样本数{suffix}': low_count,

        f'R2_在线{suffix}': r2_online,
        f'R2_模型{suffix}': r2_model,
        f'R2_提升{suffix}': r2_model - r2_online if not (np.isnan(r2_model) or np.isnan(r2_online)) else np.nan,

        f'RMSE_在线{suffix}': rmse_online,
        f'RMSE_模型{suffix}': rmse_model,
        f'RMSE_提升{suffix}': rmse_model - rmse_online if not (np.isnan(rmse_model) or np.isnan(rmse_online)) else np.nan,

        f'MAE_在线{suffix}': overall_mae_online,
        f'MAE_模型{suffix}': overall_mae_model,
        f'MAE_提升{suffix}': overall_mae_model - overall_mae_online if not (np.isnan(overall_mae_model) or np.isnan(overall_mae_online)) else np.nan,

        f'偏差均值_在线{suffix}': raw_residuals.mean(),
        f'偏差均值_模型{suffix}': model_residuals.mean(),
        f'偏差均值_提升{suffix}': model_residuals.mean() - raw_residuals.mean(),

        f'在线偏高MAE_在线{suffix}': high_mae_online,
        f'在线偏高MAE_模型{suffix}': high_mae_model,
        f'在线偏高MAE_提升{suffix}': high_mae_model - high_mae_online if high_count > 0 else np.nan,
        f'在线偏高RMSE_在线{suffix}': high_rmse_online,
        f'在线偏高RMSE_模型{suffix}': high_rmse_model,
        f'在线偏高RMSE_提升{suffix}': high_rmse_model - high_rmse_online if high_count > 0 else np.nan,

        f'在线偏低MAE_在线{suffix}': low_mae_online,
        f'在线偏低MAE_模型{suffix}': low_mae_model,
        f'在线偏低MAE_提升{suffix}': low_mae_model - low_mae_online if low_count > 0 else np.nan,
        f'在线偏低RMSE_在线{suffix}': low_rmse_online,
        f'在线偏低RMSE_模型{suffix}': low_rmse_model,
        f'在线偏低RMSE_提升{suffix}': low_rmse_model - low_rmse_online if low_count > 0 else np.nan,
    }
    return metrics, dir_raw, dir_model


# ==========================================
# 4. 纯计算：切分数据 + 训练 + 算 metrics（不画图，供调参脚本和正式训练共用）
# ==========================================
def fit_and_evaluate_surface(df, surface, params, group_tag="",
                             train_ratio=0.65, val_ratio=0.20):
    """
    纯计算部分：按时间顺序切分为 Train / Validation / Test，
    只在 Train 上训练模型，分别在 Val 和 Test 上计算完整指标。

    参数:
        train_ratio: 训练集比例（默认 0.65）
        val_ratio  : 验证集比例（默认 0.20），剩余部分为测试集

    返回:
        corrector: 训练好的 ResidualCorrectionModel
        metrics  : dict，同时包含 _训练 / _验证 / _测试 指标
        aux      : dict，画图需要的中间变量（以测试集为主）
    """
    prefix = 'Top' if surface == 'Top' else 'Bot'
    surface_cn = '上' if surface == 'Top' else '下'

    df = df.copy()
    speed_col = 'Speed[m/min]_Process_Avg'
    current_col = f'{prefix}_Current_Sum'
    online_col = f'Tin Weight_Actual[g/m2]_GALV_WEIGHT_{prefix.upper()}_Avg'
    setpoint_weight = f'Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_{prefix.upper()}_Avg'

    # 衍生特征
    df[f'{prefix}_Current_Per_Speed'] = df[current_col] / (df[speed_col] + 1e-5)
    df[f'{prefix}_Weight_Deviation'] = df[setpoint_weight] - df[online_col]

    # 与 get_feature_cols 保持完全一致（当前训练未使用 Weight_Deviation）
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

    # ------------------------------------------------------------------
    # 按时间顺序固定切分：Train / Val / Test
    # ------------------------------------------------------------------
    if n_samples < 50:
        # 样本极少时退化为简单 80/20（无独立验证集）
        train_end = int(n_samples * 0.8)
        val_end = train_end
        test_end = n_samples
    else:
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        test_end = n_samples

        # 保护：确保每个集合至少有一定样本
        min_size = max(10, int(n_samples * 0.05))
        if train_end < min_size:
            train_end = min_size
        if val_end - train_end < min_size:
            val_end = min(train_end + min_size, n_samples - min_size)
        if test_end - val_end < min_size:
            val_end = max(train_end, n_samples - min_size)

    train_idx = slice(0, train_end)
    val_idx = slice(train_end, val_end)
    test_idx = slice(val_end, test_end)

    X_train = X.iloc[train_idx]
    y_delta_train = y_delta.iloc[train_idx]
    actual_train = online_actual.iloc[train_idx]
    y_true_train = y_true_full.iloc[train_idx]

    X_val = X.iloc[val_idx]
    actual_val = online_actual.iloc[val_idx]
    y_true_val = y_true_full.iloc[val_idx]

    X_test = X.iloc[test_idx]
    actual_test = online_actual.iloc[test_idx]
    y_true_test = y_true_full.iloc[test_idx]

    # ------------------------------------------------------------------
    # 训练（只用 Train）
    # ------------------------------------------------------------------
    corrector = _build_corrector_instance(params, ModelClass)
    corrector.fit(X_train, y_delta_train)

    # ------------------------------------------------------------------
    # 预测
    # ------------------------------------------------------------------
    pred_train, _ = corrector.predict_smooth(X_train, actual_train)
    pred_val, _ = corrector.predict_smooth(X_val, actual_val) if len(X_val) > 0 else (pd.Series(dtype=float), None)
    pred_test, _ = corrector.predict_smooth(X_test, actual_test) if len(X_test) > 0 else (pd.Series(dtype=float), None)

    # 残差
    raw_residuals_train = y_true_train - actual_train
    model_residuals_train = y_true_train - pred_train

    if len(X_val) > 0:
        raw_residuals_val = y_true_val - actual_val
        model_residuals_val = y_true_val - pred_val
    else:
        raw_residuals_val = pd.Series(dtype=float)
        model_residuals_val = pd.Series(dtype=float)

    if len(X_test) > 0:
        raw_residuals_test = y_true_test - actual_test
        model_residuals_test = y_true_test - pred_test
    else:
        raw_residuals_test = pd.Series(dtype=float)
        model_residuals_test = pd.Series(dtype=float)

    # ------------------------------------------------------------------
    # 计算三套指标
    # ------------------------------------------------------------------
    metrics_train, _, _ = _calc_set_metrics(
        y_true_train, actual_train, pred_train,
        raw_residuals_train, model_residuals_train, suffix="_训练"
    )

    if len(X_val) > 0:
        metrics_val, dir_raw_val, dir_model_val = _calc_set_metrics(
            y_true_val, actual_val, pred_val,
            raw_residuals_val, model_residuals_val, suffix="_验证"
        )
    else:
        # 无验证集时用训练集指标兜底（极少发生）
        metrics_val = {k.replace("_训练", "_验证"): v for k, v in metrics_train.items()}
        dir_raw_val = dir_model_val = {}

    if len(X_test) > 0:
        metrics_test, dir_raw_test, dir_model_test = _calc_set_metrics(
            y_true_test, actual_test, pred_test,
            raw_residuals_test, model_residuals_test, suffix="_测试"
        )
    else:
        metrics_test = {k.replace("_训练", "_测试"): v for k, v in metrics_train.items()}
        dir_raw_test = dir_model_test = {}

    # 全量数据原始残差方向样本数（仅统计用）
    # residual < 0 → 在线偏高；residual > 0 → 在线偏低
    raw_residuals_full = y_true_full - online_actual
    total_high_count = int((raw_residuals_full < 0).sum())
    total_low_count = int((raw_residuals_full > 0).sum())

    # ---- 过拟合：训练 vs 测试 ----
    mae_model_train = metrics_train.get('MAE_模型_训练', np.nan)
    mae_model_test = metrics_test.get('MAE_模型_测试', np.nan)
    rmse_model_train = metrics_train.get('RMSE_模型_训练', np.nan)
    rmse_model_test = metrics_test.get('RMSE_模型_测试', np.nan)
    r2_model_train = metrics_train.get('R2_模型_训练', np.nan)
    r2_model_test = metrics_test.get('R2_模型_测试', np.nan)

    overfitting_r2 = r2_model_train - r2_model_test if not (np.isnan(r2_model_train) or np.isnan(r2_model_test)) else np.nan
    overfitting_mae = mae_model_train - mae_model_test if not (np.isnan(mae_model_train) or np.isnan(mae_model_test)) else np.nan
    overfitting_rmse = rmse_model_train - rmse_model_test if not (np.isnan(rmse_model_train) or np.isnan(rmse_model_test)) else np.nan
    mae_ratio = mae_model_test / (mae_model_train + 1e-8) if not np.isnan(mae_model_train) else np.nan
    rmse_ratio = rmse_model_test / (rmse_model_train + 1e-8) if not np.isnan(rmse_model_train) else np.nan

    # ---- 过拟合：训练 vs 验证（调参/防过拟合更应看这一组）----
    mae_model_val = metrics_val.get('MAE_模型_验证', np.nan)
    rmse_model_val = metrics_val.get('RMSE_模型_验证', np.nan)
    r2_model_val = metrics_val.get('R2_模型_验证', np.nan)

    overfitting_r2_tv = r2_model_train - r2_model_val if not (np.isnan(r2_model_train) or np.isnan(r2_model_val)) else np.nan
    overfitting_mae_tv = mae_model_train - mae_model_val if not (np.isnan(mae_model_train) or np.isnan(mae_model_val)) else np.nan
    overfitting_rmse_tv = rmse_model_train - rmse_model_val if not (np.isnan(rmse_model_train) or np.isnan(rmse_model_val)) else np.nan
    mae_ratio_tv = mae_model_val / (mae_model_train + 1e-8) if not np.isnan(mae_model_train) else np.nan
    rmse_ratio_tv = rmse_model_val / (rmse_model_train + 1e-8) if not np.isnan(rmse_model_train) else np.nan

    # RMSE 提升百分比（测试集）：正值表示模型更好
    rmse_online_test = metrics_test.get('RMSE_在线_测试', np.nan)
    rmse_model_test_val = metrics_test.get('RMSE_模型_测试', np.nan)
    rmse_improve_pct = (
        (rmse_online_test - rmse_model_test_val) / rmse_online_test * 100
        if rmse_online_test and rmse_online_test != 0 and not np.isnan(rmse_online_test)
        else 0.0
    )

    # 汇总 metrics
    metrics = {
        '规格组': group_tag,
        '表面': surface,
        '训练样本数': len(X_train),
        '验证样本数': len(X_val),
        '测试样本数': len(X_test),
        '在线偏高样本数_总': total_high_count,
        '在线偏低样本数_总': total_low_count,
        '使用扩展窗口': False,  # 已移除，固定为 False 保持字段兼容

        # ---- 训练集 ----
        **metrics_train,

        # ---- 验证集（Optuna 应使用这套指标）----
        **metrics_val,

        # ---- 测试集（最终评估用）----
        **metrics_test,
        'RMSE_提升_测试(%)': rmse_improve_pct,

        # ---- 过拟合程度（训练 vs 测试）----
        '过拟合程度_R2_训练测试': overfitting_r2,
        '过拟合程度_MAE_训练测试': overfitting_mae,
        '过拟合程度_RMSE_训练测试': overfitting_rmse,
        'MAE比值_测试/训练': mae_ratio,
        'RMSE比值_测试/训练': rmse_ratio,

        # ---- 过拟合程度（训练 vs 验证）----
        '过拟合程度_R2_训练验证': overfitting_r2_tv,
        '过拟合程度_MAE_训练验证': overfitting_mae_tv,
        '过拟合程度_RMSE_训练验证': overfitting_rmse_tv,
        'MAE比值_验证/训练': mae_ratio_tv,
        'RMSE比值_验证/训练': rmse_ratio_tv,
    }

    # aux 以测试集为主（兼容原有画图逻辑），同时带上验证集信息
    aux = dict(
        X_train=X_train,
        y_true_train=y_true_train,
        actual_train=actual_train,
        pred_train=pred_train,
        raw_residuals_train=raw_residuals_train,
        model_residuals_train=model_residuals_train,

        X_val=X_val,
        y_true_val=y_true_val,
        actual_val=actual_val,
        pred_val=pred_val if len(X_val) > 0 else None,
        raw_residuals_val=raw_residuals_val,
        model_residuals_val=model_residuals_val,

        X_test=X_test,
        y_true_series=y_true_test,          # 兼容原字段名
        online_series=actual_test,          # 兼容原字段名
        pred_series=pred_test,              # 兼容原字段名
        raw_residuals=raw_residuals_test,    # 兼容原字段名
        model_residuals=model_residuals_test,
        test_directional_metrics_raw=dir_raw_test if dir_raw_test else {},
        test_directional_metrics_model=dir_model_test if dir_model_test else {},
    )
    return corrector, metrics, aux


# ==========================================
# 5. 表面建模与图形输出（薄壳：相关性分析 + 调用纯计算 + 画图）
# ==========================================
def run_surface_pipeline(df, surface='Top', group_tag="", group_params=None,
                         train_ratio=0.65, val_ratio=0.20):
    surface_cn = '上' if surface == 'Top' else '下'
    tag_display = f"[{group_tag}] " if group_tag else ""
    safe_tag = f"_{group_tag}" if group_tag else ""

    params = group_params or DEFAULT_PARAMS

    print(f"\n==========================================")
    print(f"    开始运行【{tag_display}{surface_cn}表面】模型拟合与分析     ")
    print(f"        使用参数: {params}")
    print(f"==========================================")

    # 2. 纯计算：切分、训练、算指标
    corrector, metrics, aux = fit_and_evaluate_surface(
        df, surface, params, group_tag=group_tag,
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )

    X_test = aux['X_test']
    y_true_series = aux['y_true_series']
    online_series = aux['online_series']
    pred_series = aux['pred_series']
    raw_residuals = aux['raw_residuals']
    model_residuals = aux['model_residuals']
    raw_residuals_train = aux['raw_residuals_train']
    model_residuals_train = aux['model_residuals_train']
    test_directional_metrics = aux['test_directional_metrics_raw']
    test_directional_metrics_model = aux['test_directional_metrics_model']

    print(f"\n-------- 【{tag_display}{surface_cn}表面 测量误差Delta与残差Residual诊断（测试集）】 --------")
    if test_directional_metrics.get('high_count', 0) > 0:
        mae_raw_high = test_directional_metrics['high_mae']
        mae_model_high = test_directional_metrics_model.get('high_mae', np.nan)
        print(
            f"当测量误差Delta < 0 (在线偏高, 样本数 {test_directional_metrics['high_count']}): "
            f"测量误差Delta MAE = {mae_raw_high:.4f}  -->  残差Residual MAE = {mae_model_high:.4f}"
        )
    if test_directional_metrics.get('low_count', 0) > 0:
        mae_raw_low = test_directional_metrics['low_mae']
        mae_model_low = test_directional_metrics_model.get('low_mae', np.nan)
        print(
            f"当测量误差Delta > 0 (在线偏低, 样本数 {test_directional_metrics['low_count']}): "
            f"测量误差Delta MAE = {mae_raw_low:.4f}  -->  残差Residual MAE = {mae_model_low:.4f}"
        )
    print("------------------------------------------------------\n")

    print(f"======== 【{tag_display}{surface_cn}表面 拟合性能评估】 ========")
    print(f"【训练集】 n={metrics['训练样本数']}")
    print(f"  原始在线 -> R²: {metrics.get('R2_在线_训练', np.nan):.4f}, "
          f"RMSE: {metrics.get('RMSE_在线_训练', np.nan):.4f}, "
          f"MAE: {metrics.get('MAE_在线_训练', np.nan):.4f}")
    print(f"  模型校正 -> R²: {metrics.get('R2_模型_训练', np.nan):.4f}, "
          f"RMSE: {metrics.get('RMSE_模型_训练', np.nan):.4f}, "
          f"MAE: {metrics.get('MAE_模型_训练', np.nan):.4f}")

    print(f"【验证集】 n={metrics['验证样本数']}")
    print(f"  原始在线 -> R²: {metrics.get('R2_在线_验证', np.nan):.4f}, "
          f"RMSE: {metrics.get('RMSE_在线_验证', np.nan):.4f}, "
          f"MAE: {metrics.get('MAE_在线_验证', np.nan):.4f}")
    print(f"  模型校正 -> R²: {metrics.get('R2_模型_验证', np.nan):.4f}, "
          f"RMSE: {metrics.get('RMSE_模型_验证', np.nan):.4f}, "
          f"MAE: {metrics.get('MAE_模型_验证', np.nan):.4f}")

    print(f"【测试集】 n={metrics['测试样本数']}")
    print(f"  原始在线 -> R²: {metrics.get('R2_在线_测试', np.nan):.4f}, "
          f"RMSE: {metrics.get('RMSE_在线_测试', np.nan):.4f}, "
          f"MAE: {metrics.get('MAE_在线_测试', np.nan):.4f}")
    print(f"  模型校正 -> R²: {metrics.get('R2_模型_测试', np.nan):.4f}, "
          f"RMSE: {metrics.get('RMSE_模型_测试', np.nan):.4f}, "
          f"MAE: {metrics.get('MAE_模型_测试', np.nan):.4f}")

    if len(X_test) == 0:
        print("[警告] 测试集为空，跳过画图")
        return corrector, metrics

    start_idx = X_test.index[0]
    end_idx = X_test.index[-1]

    # 3. 拟合对比图
    plt.figure(figsize=(12, 5))
    plt.plot(y_true_series, label='实验室真实测量值 (True Label)', color='black', linewidth=1.5)
    plt.plot(online_series, label='在线仪表原始测量值 (Online)', color='red', linestyle='--', alpha=0.7)
    plt.plot(pred_series, label='模型残差校正值 (Model Pred)', color='green', linewidth=1.5, alpha=0.85)
    plt.title(f'{tag_display}{surface_cn}表面 镀层重量拟合对照图（测试集，原始数据行号: {start_idx} ~ {end_idx}）')
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

    ax1.plot(raw_residuals, label='测量误差Delta (True - Online)', color='red', alpha=0.5, linewidth=1)
    ax1.plot(model_residuals, label='残差Residual (True - Model)', color='green', alpha=0.8, linewidth=1.2)
    ax1.axhline(0, color='black', linestyle='--', linewidth=1)
    ax1.set_title(f'{tag_display}{surface_cn}表面 测量误差Delta与残差Residual对比（测试集，原始数据行号: {start_idx} ~ {end_idx}）')
    ax1.set_xlabel('原始数据行号 (Original Row Index)')
    ax1.set_ylabel('误差 (g/m2)')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    sns.histplot(raw_residuals, ax=ax2, color='red', label='测量误差Delta分布', kde=True, stat="density", alpha=0.3)
    sns.histplot(model_residuals, ax=ax2, color='green', label='残差Residual分布', kde=True, stat="density",
                 alpha=0.5)
    ax2.axvline(0, color='black', linestyle='--', linewidth=1)
    ax2.set_title(f'{tag_display}{surface_cn}表面 测量误差Delta与残差Residual概率密度（测试集，越集中在0且越窄越好）')
    ax2.set_xlabel('误差 (g/m2)')
    ax2.set_ylabel('概率密度')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    res_img_path = f"result/grouped_by_coating_weight/fitting_result/residual_analysis/residual_analysis_{surface}{safe_tag}.png"
    os.makedirs(os.path.dirname(res_img_path), exist_ok=True)
    plt.savefig(res_img_path, dpi=300)
    print(f"[图表保存] {tag_display}{surface_cn}表面残差分析图已保存至: {res_img_path}")
    plt.close()

    # ========== 5. 训练集 vs 测试集 残差分布对比图 ==========
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：测量误差Delta（True - Online）
    _plot_residual_distribution(
        axes[0], raw_residuals_train, raw_residuals,
        'orange', 'red',
        '训练集 测量误差Delta', '测试集 测量误差Delta',
        f'{tag_display}{surface_cn}表面 测量误差Delta分布\n(训练集 vs 测试集)',
        '测量误差Delta (True - Online) g/m2'
    )

    # 右图：残差Residual（True - Model）
    _plot_residual_distribution(
        axes[1], model_residuals_train, model_residuals,
        'cyan', 'green',
        '训练集 残差Residual', '测试集 残差Residual',
        f'{tag_display}{surface_cn}表面 残差Residual分布\n(训练集 vs 测试集)',
        '残差Residual (True - Model) g/m2'
    )

    plt.tight_layout()
    dist_img_path = (f"result/grouped_by_coating_weight/fitting_result/"
                     f"residual_train_vs_test/residual_train_vs_test_{surface}{safe_tag}.png")
    os.makedirs(os.path.dirname(dist_img_path), exist_ok=True)
    plt.savefig(dist_img_path, dpi=300)
    print(f"[图表保存] {tag_display}{surface_cn}表面 训练集vs测试集残差分布对比图已保存至: {dist_img_path}")
    plt.close()

    # ===== 模型解释性分析 =====
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
    """与 fit_and_evaluate_surface 中 feature_cols 完全一致"""
    prefix = "Top" if surface == "Top" else "Bot"
    return [
        f"Tin Weight_Actual[g/m2]_GALV_WEIGHT_{prefix.upper()}_Avg",
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
    # parser.add_argument(                                           ## 所有组都使用同样的超参数，即对Top2.799_Bot2.799最优的超参数
    #     "--config", type=str, default="group_params_all_the_same_2.json",
    #     help="配置文件 JSON 路径 (默认: group_params_all_the_same_2.json)"
    # )
    parser.add_argument(                                           ## 所有组都使用同样的超参数，即对Top2.799_Bot2.799最优的超参数
        "--config", type=str, default="group_params_2_799_only.json",
        help="配置文件 JSON 路径 (默认: group_params_2_799_only.json)"
    )
    # parser.add_argument(                                           ## 所有组都使用同样的超参数，即对Top2.799_Bot2.799最优的超参数
    #     "--config", type=str, default="group_params_all_the_same_3.json",
    #     help="配置文件 JSON 路径 (默认: group_params_all_the_same_3.json)"
    # )
    # parser.add_argument(
    #     "--config", type=str, default="group_params_optimum_for_each.json",  ## 使用Optuna对各组搜索出来的最优的超参数
    #     help="配置文件 JSON 路径 (默认: group_params_optimum_for_each.json)"
    # )
    # parser.add_argument(
    #     "--config", type=str, default="group_params_optimum_for_each_optimal.json",  ## 使用Optuna对各组搜索出来的最优的超参数
    #     help="配置文件 JSON 路径 (默认: group_params_optimum_for_each_optimal.json)"
    # )

    args = parser.parse_args()

    # 1. 加载统一配置
    config = load_pipeline_config(args.config)
    MIN_GROUP_SAMPLES = config.get("min_group_samples", 200)

    # 2. 读取数据并分组汇总
    clean_df = pd.read_excel(config.get("data_paths", {}).get("clean_data", "result/data/feature_engineered_data/featured_data.xlsx"))
    clean_df = build_setpoint_group_key(clean_df)
    group_sizes = summarize_setpoint_groups(clean_df)

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

        top_model, top_metrics = run_surface_pipeline(
            group_df, surface='Top', group_tag=group_label, group_params=top_params,
            train_ratio=0.65, val_ratio=0.20
        )
        bot_model, bot_metrics = run_surface_pipeline(
            group_df, surface='Bot', group_tag=group_label, group_params=bot_params,
            train_ratio=0.65, val_ratio=0.20
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

    report_path = "result/grouped_by_coating_weight/summary_report_group_params_2_799_only.xlsx"
    # report_path = "result/grouped_by_coating_weight/summary_report_group_optimum_all_the_same_3.xlsx"
    # report_path = "result/grouped_by_coating_weight/summary_report_group_optimum_all_the_same.xlsx"
    # report_path = "result/grouped_by_coating_weight/summary_report_group_optimum_for_each.xlsx"
    # report_path = "result/grouped_by_coating_weight/summary_report_group_optimum_for_each_2.xlsx"
    # report_path = "result/grouped_by_coating_weight/summary_report_group_optimum_for_each_3.xlsx"

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
        sample_summary_df.to_excel(writer, sheet_name='样本量汇总', index=False)
        if all_metrics:
            metrics_df = pd.DataFrame(all_metrics)
            # ==========================================
            # 当某一方向样本数为 0 时，将该方向 MAE/RMSE 置为 NaN（Excel 显示空白）
            # ==========================================
            for suffix in ["_测试", "_验证", "_训练"]:
                high_col = f'在线偏高样本数{suffix}'
                low_col = f'在线偏低样本数{suffix}'
                if high_col in metrics_df.columns:
                    mask_high_zero = (metrics_df[high_col] == 0)
                    metrics_df.loc[mask_high_zero, [
                        f'在线偏高MAE_在线{suffix}', f'在线偏高MAE_模型{suffix}',
                        f'在线偏高RMSE_在线{suffix}', f'在线偏高RMSE_模型{suffix}'
                    ]] = np.nan
                if low_col in metrics_df.columns:
                    mask_low_zero = (metrics_df[low_col] == 0)
                    metrics_df.loc[mask_low_zero, [
                        f'在线偏低MAE_在线{suffix}', f'在线偏低MAE_模型{suffix}',
                        f'在线偏低RMSE_在线{suffix}', f'在线偏低RMSE_模型{suffix}'
                    ]] = np.nan
            # ==========================================
            metrics_df.to_excel(writer, sheet_name='建模效果汇总', index=False)
        else:
            pd.DataFrame({'提示': ['没有达标组完成建模']}).to_excel(
                writer, sheet_name='建模效果汇总', index=False
            )

    print(f"[导出提示] 汇总报表已保存至: {report_path}")
