"""
【阶段 B：规格组合并深入验证（优化版 - 缓存 + 分层）】

对阶段 A 筛选出的候选对进行分层验证：

Tier 1 (深度验证 - 完整 Chow 检验):
  - KS 通过的对
  - 样本量充足且均值差异小
  
Tier 2 (快速估计 - 简化兼容性检查):
  - KS 边界附近的对
  - 使用特征统计量进行快速预筛选
  
Tier 3 (直接排除):
  - KS 明显失败且差异大的对
  
优化策略:
  1. 缓存机制: 15 个规格组的最优迭代次数只计算一次
  2. 分层验证: 集中资源在有希望的对上
  3. 快速筛选: Tier 2 用启发式方法快速评估
"""

import pandas as pd
import numpy as np
from scipy.stats import f as f_distribution
import os
import json
import yaml
import time
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold
from typing import Optional, Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

# ===== 从 coating_model_by_group.py 复用的常量 =====
TOP_SETPOINT_COL = 'Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_TOP_Min'
BOT_SETPOINT_COL = 'Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_BOT_Min'

# ===== 新增：优化参数 =====
# 分层阈值
KS_PASS_THRESHOLD = 0.05        # KS p-value > 0.05 进入 Tier 1
KS_BOUNDARY_THRESHOLD = 0.03    # 0.03-0.08 之间为边界，进入 Tier 2
MEAN_DIFF_THRESHOLD = 0.01      # 均值差异 < 0.01 进入 Tier 1
MIN_MERGED_SAMPLES = 400        # 合并样本 >= 400 进入 Tier 1

# 【新增】基于规格距离的动态阈值
SPEC_DISTANCE_TIER1 = 0.8       # 规格距离 < 0.8 进入 Tier 1（规格相近）
SPEC_DISTANCE_TIER2 = 1.5       # 规格距离 0.8-1.5 进入 Tier 2（规格中等）
SPEC_DISTANCE_TIER3 = 1.5       # 规格距离 > 1.5 进入 Tier 3（规格差异大）

# 快速兼容性检查的阈值
FEATURE_MEAN_RATIO_THRESHOLD = 1.0   # 均值差异 / std > 1.0 排除
FEATURE_VAR_RATIO_THRESHOLD = 2.0    # 方差比 > 2.0 排除


def get_feature_cols(surface: str) -> list:
    """
    与 coating_model_by_group.py 中的 get_feature_cols() 保持一致
    返回该表面的特征列（已剪枝后的 5 个核心特征）
    使用实际的数据列名
    """
    if surface == 'Top':
        return [
            'Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Avg',
            'Dimension_[mm]_Width',
            'Speed[m/min]_Process_Avg',
            'Top_Current_Sum',
            'Dimension_[mm]_Thickness'
        ]
    else:  # Bot
        return [
            'Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Avg',
            'Dimension_[mm]_Width',
            'Speed[m/min]_Process_Avg',
            'Bot_Current_Sum',
            'Dimension_[mm]_Thickness'
        ]


# 【新增】规格距离相关函数
def parse_spec_values(group_label: str) -> tuple:
    """
    从规格组标签解析 Top 和 Bot 的镀锡厚度设定值
    例：'Top2.0_Bot2.0' → (2.0, 2.0)
    """
    parts = group_label.split('_')
    top_val = float(parts[0].replace('Top', ''))
    bot_val = float(parts[1].replace('Bot', ''))
    return top_val, bot_val


def compute_spec_distance(group_A: str, group_B: str) -> float:
    """
    计算两个规格组之间的欧氏距离
    
    参数：
      group_A: 'Top2.0_Bot2.0'
      group_B: 'Top2.799_Bot2.799'
    
    返回：两个规格组的欧氏距离
    """
    top_A, bot_A = parse_spec_values(group_A)
    top_B, bot_B = parse_spec_values(group_B)
    
    distance = np.sqrt((top_A - top_B)**2 + (bot_A - bot_B)**2)
    return distance


def get_chow_threshold_by_spec(spec_distance: float) -> float:
    """
    基于规格距离的动态 Chow 检验阈值
    
    规格接近 → 对 Chow 检验期望更高（需要 p > 0.05）
    规格差异大 → 降低期望（基本不可能合并）
    
    参数：spec_distance 从 compute_spec_distance() 计算
    
    返回：该对的 Chow p-value 阈值
    """
    if spec_distance < 0.5:
        return 0.05      # 严格：必须 p > 0.05
    elif spec_distance < 1.0:
        return 0.02      # 适中：p > 0.02
    else:
        return 0.001     # 宽松：基本不可能通过


def load_config(config_path="config.yaml"):
    """加载配置文件"""
    if not os.path.exists(config_path):
        print(f"⚠ 警告: 配置文件 {config_path} 不存在，使用默认数据路径")
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config or {}


def load_featured_data(config_path="config.yaml") -> pd.DataFrame:
    """加载特征工程后的数据"""
    print("\n[加载数据]")
    
    # 加载配置
    config = load_config(config_path)
    
    # 获取数据路径（与 coating_model_by_group.py 一致）
    excel_path = config.get("data_paths", {}).get("clean_data", "result/data/feature_engineered_data/featured_data.xlsx")
    
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"未找到特征工程数据: {excel_path}\n请先运行 new_data_pipeline.py 或 analyse_data_final.py")
    
    df = pd.read_excel(excel_path)
    print(f"  数据: {len(df)} 行, {df.shape[1]} 列")
    print(f"  路径: {excel_path}")
    return df


def load_merge_candidates(csv_path: str = "result/spec_group_merge_analysis/merge_candidates.csv") -> pd.DataFrame:
    """加载阶段 A 生成的候选对"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到候选对文件: {csv_path}")
    
    df_cand = pd.read_csv(csv_path)
    print(f"✓ 加载候选对: {len(df_cand)} 对")
    return df_cand


# ===== 新增：缓存和分层函数 =====

def classify_candidates(df_cand: pd.DataFrame) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    将候选对分为三个优先级（优化版 v2）
    
    【新增优化】：
    1. 加入规格距离作为分类依据
    2. 区分表面处理（Top/Bot 有不同的分布特征）
    3. 规格接近的对优先进入 Tier 1
    
    返回: (tier1_candidates, tier2_candidates, tier3_candidates)
    """
    tier1 = []  # 深度验证 - 高优先级
    tier2 = []  # 快速估计 - 中优先级
    tier3 = []  # 直接排除 - 低优先级
    
    for idx, row in df_cand.iterrows():
        cand = row.to_dict()
        
        # 提取关键字段
        ks_pass = cand.get('ks_pass', False)
        p_value = cand.get('p_value', 0)
        mean_diff = cand.get('mean_diff', 0)
        merged_size = cand.get('merged_size', 0)
        spec_distance = cand.get('spec_distance', 2.0)  # 默认距离为 2.0（较大）
        surface = cand.get('surface', 'Top')
        
        # 【新增】根据表面和规格距离调整分层逻辑
        # Bot 表面通常规格跨度较小，更容易合并
        # Top 表面规格跨度大（1.1-11.199），分化更明显
        
        # ===== Tier 1: 最有希望的对 =====
        # 条件：KS 通过 OR 规格非常接近
        if ks_pass or spec_distance < 0.5:
            # 额外条件：质量要好
            if mean_diff < MEAN_DIFF_THRESHOLD and merged_size >= MIN_MERGED_SAMPLES:
                cand['tier'] = 1
                tier1.append(cand)
                continue
        
        # ===== Tier 2: 中等优先级 =====
        # 条件：规格接近 (< 1.0) OR KS 边界附近
        if spec_distance < 1.0 or (KS_BOUNDARY_THRESHOLD < p_value < 0.08):
            if 0 < mean_diff < 0.05:  # 宽松一些的均值差异要求
                cand['tier'] = 2
                tier2.append(cand)
                continue
        
        # ===== Tier 3: 直接排除 =====
        # 条件：规格差异大 OR KS 明显失败
        cand['tier'] = 3
        tier3.append(cand)
    
    # 【新增】Tier 1 和 Tier 2 内部按规格距离排序
    # 确保最有希望的对优先被验证
    tier1 = sorted(tier1, key=lambda x: x.get('spec_distance', 2.0))
    tier2 = sorted(tier2, key=lambda x: x.get('spec_distance', 2.0))
    
    return tier1, tier2, tier3


def build_optimal_iterations_cache(df: pd.DataFrame, features_dict: Dict[str, List[str]],
                                   target_col_pattern: str) -> Dict[str, int]:
    """
    预计算所有规格组的最优迭代次数，缓存以供重用
    
    参数:
      features_dict: {'Top': [...], 'Bot': [...]}
      target_col_pattern: 目标列名模式，如 '{surface}_Delta'
    
    返回: {'Top2.799_Bot2.799': 100, 'Top2.0_Bot2.0': 150, ...}
    """
    print("\n[缓存构建] 预计算所有规格组的最优迭代次数...")
    print("(这只需要进行一次)\n")
    
    cache = {}
    group_labels = sorted(df['Setpoint_Group_Label'].unique())
    
    start_time = time.time()
    
    for i, group_label in enumerate(group_labels, 1):
        print(f"  {i:2d}/{len(group_labels)}: {group_label:<25}", end='', flush=True)
        
        group_data = df[df['Setpoint_Group_Label'] == group_label].copy()
        
        # 推断表面类型
        if 'Top_Delta' in df.columns and group_data['Top_Delta'].notna().sum() > 0:
            surface = 'Top'
        elif 'Bot_Delta' in df.columns and group_data['Bot_Delta'].notna().sum() > 0:
            surface = 'Bot'
        else:
            print(" ✗ 跳过 (无有效数据)")
            cache[group_label] = 50  # 默认值
            continue
        
        features = features_dict.get(surface, [])
        target_col = f'{surface}_Delta'
        
        # 获取数据
        X = group_data[features].copy()
        y = group_data[target_col].copy()
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        if len(X) < 10:
            print(" ✗ 样本不足")
            cache[group_label] = 50
            continue
        
        # 找最优迭代次数
        optimal_iter = find_optimal_iterations(X, y)
        cache[group_label] = optimal_iter
        print(f" → {optimal_iter} iter")
    
    elapsed = time.time() - start_time
    print(f"\n  缓存构建完成 ({elapsed:.1f}s), 共 {len(cache)} 个规格组")
    
    return cache


def quick_compatibility_check(group_A_data: pd.DataFrame,
                              group_B_data: pd.DataFrame,
                              features: List[str]) -> Tuple[bool, str]:
    """
    快速兼容性检查（Tier 2 用）
    
    基于特征统计量的启发式评估，不需要训练模型
    返回: (is_compatible, reason)
    """
    reason_parts = []
    
    for feature in features:
        A_vals = group_A_data[feature].dropna()
        B_vals = group_B_data[feature].dropna()
        
        if len(A_vals) < 5 or len(B_vals) < 5:
            continue
        
        # 检查均值差异
        mean_diff = abs(A_vals.mean() - B_vals.mean())
        std_B = B_vals.std()
        
        if std_B > 0:
            mean_ratio = mean_diff / std_B
            if mean_ratio > FEATURE_MEAN_RATIO_THRESHOLD:
                reason_parts.append(f"{feature}: mean_ratio={mean_ratio:.2f}")
        
        # 检查方差比
        var_A = A_vals.var()
        var_B = B_vals.var()
        if var_B > 0:
            var_ratio = max(var_A, var_B) / min(var_A, var_B)
            if var_ratio > FEATURE_VAR_RATIO_THRESHOLD:
                reason_parts.append(f"{feature}: var_ratio={var_ratio:.2f}")
    
    if reason_parts:
        return False, f"特征差异过大: {', '.join(reason_parts[:2])}"
    
    return True, "快速兼容性检查通过"


def build_spec_groups(df) -> pd.DataFrame:
    """构建规格组标签"""
    df = df.copy()
    
    if TOP_SETPOINT_COL not in df.columns or BOT_SETPOINT_COL not in df.columns:
        raise KeyError(f"缺少分组所需字段: {TOP_SETPOINT_COL} 或 {BOT_SETPOINT_COL}")
    
    df['Setpoint_Group_Label'] = df.apply(
        lambda r: f"Top{r[TOP_SETPOINT_COL]}_Bot{r[BOT_SETPOINT_COL]}", axis=1
    )
    return df


def find_optimal_iterations(X_val: pd.DataFrame, y_val: pd.Series, 
                             learning_rate: float = 0.05,
                             max_depth: int = 3) -> int:
    """
    使用交叉验证找到最优的迭代次数
    避免固定 max_iter=100 导致的收敛差异问题
    
    返回: 最优迭代次数
    """
    from sklearn.model_selection import cross_val_score
    
    if len(X_val) < 20:
        return 50  # 样本少时使用较小迭代次数
    
    kf = KFold(n_splits=min(3, len(X_val) // 10), shuffle=True, random_state=42)
    
    # 测试不同的迭代次数
    iter_range = [30, 50, 75, 100, 150]
    best_iter = 50
    best_score = -np.inf
    
    for max_iter in iter_range:
        model = HistGradientBoostingRegressor(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42,
            early_stopping='auto',
            n_iter_no_change=5,
            validation_fraction=0.2,
            tol=1e-4
        )
        
        # 交叉验证评分
        scores = cross_val_score(model, X_val, y_val, cv=kf, scoring='neg_mean_squared_error')
        cv_score = scores.mean()
        
        if cv_score > best_score:
            best_score = cv_score
            best_iter = max_iter
    
    return best_iter


def train_model_for_group(group_df: pd.DataFrame, surface: str,
                          features: list, target_col: str,
                          max_iter: Optional[int] = None) -> tuple:
    """
    训练单个规格组的模型
    
    参数:
      max_iter: 指定的迭代次数。如果为 None，自动找到最优值
    
    返回: (model, rmse, mae, residuals, actual_n_iter, diagnostics)
    """
    X = group_df[features].copy()
    y = group_df[target_col].copy()
    
    # 移除 NaN
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    if len(X) < 10:
        return None, np.inf, np.inf, None, None, {}
    
    # 如果没有指定迭代次数，自动找到最优值
    if max_iter is None:
        max_iter = find_optimal_iterations(X, y)
    
    # 训练模型
    model = HistGradientBoostingRegressor(
        max_iter=max_iter,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
        early_stopping='auto',
        n_iter_no_change=5,
        validation_fraction=0.2,
        tol=1e-4
    )
    model.fit(X, y)
    
    # 评估
    pred = model.predict(X)
    residuals = y - pred
    rmse = np.sqrt(mean_squared_error(y, pred))
    mae = mean_absolute_error(y, pred)
    
    # 诊断信息
    diagnostics = {
        'n_samples': len(X),
        'target_max_iter': max_iter,
        'actual_n_iter': model.n_iter_ if hasattr(model, 'n_iter_') else max_iter,
        'rmse': rmse,
        'mae': mae,
        'residual_std': np.std(residuals)
    }
    
    return model, rmse, mae, residuals, diagnostics


def compute_rss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """残差平方和"""
    return np.sum((y_true - y_pred) ** 2)


def chow_test_with_aligned_iter(group_A_data: pd.DataFrame,
                                 group_B_data: pd.DataFrame,
                                 surface: str,
                                 features: list,
                                 target_col: str,
                                 aligned_iter: int) -> dict:
    """
    Chow 检验 - 使用预设的对齐迭代次数（无需再计算）
    
    这是优化版本，直接使用缓存的迭代次数，不再重复计算
    """
    # 获取特征和目标
    mask_A = ~(group_A_data[features].isna().any(axis=1) | group_A_data[target_col].isna())
    X_A = group_A_data.loc[mask_A, features]
    y_A = group_A_data.loc[mask_A, target_col]
    
    mask_B = ~(group_B_data[features].isna().any(axis=1) | group_B_data[target_col].isna())
    X_B = group_B_data.loc[mask_B, features]
    y_B = group_B_data.loc[mask_B, target_col]
    
    if len(X_A) < 10 or len(X_B) < 10:
        return {
            'F_stat': np.nan,
            'p_value': np.nan,
            'can_merge': False,
            'chow_interpretation': '样本量不足'
        }
    
    # 训练三个模型（都使用对齐的迭代次数）
    model_A = HistGradientBoostingRegressor(
        max_iter=aligned_iter,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    model_A.fit(X_A, y_A)
    
    model_B = HistGradientBoostingRegressor(
        max_iter=aligned_iter,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    model_B.fit(X_B, y_B)
    
    combined_data = pd.concat([group_A_data, group_B_data], ignore_index=True)
    mask_AB = ~(combined_data[features].isna().any(axis=1) | combined_data[target_col].isna())
    X_AB = combined_data.loc[mask_AB, features]
    y_AB = combined_data.loc[mask_AB, target_col]
    
    model_AB = HistGradientBoostingRegressor(
        max_iter=aligned_iter,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    model_AB.fit(X_AB, y_AB)
    
    # 计算残差平方和
    rss_A = compute_rss(y_A.values, model_A.predict(X_A.values))
    rss_B = compute_rss(y_B.values, model_B.predict(X_B.values))
    rss_AB = compute_rss(y_AB.values, model_AB.predict(X_AB.values))
    
    # Chow 统计量
    k = len(features)
    n_A = len(X_A)
    n_B = len(X_B)
    
    numerator = (rss_AB - rss_A - rss_B) / k
    denominator = (rss_A + rss_B) / (n_A + n_B - 2*k)
    
    if denominator <= 0:
        return {
            'F_stat': np.nan,
            'p_value': np.nan,
            'can_merge': False,
            'chow_interpretation': '分母无效'
        }
    
    F_stat = numerator / denominator
    p_value = 1 - f_distribution.cdf(F_stat, k, n_A + n_B - 2*k)
    residual_increase = ((rss_AB - rss_A - rss_B) / (rss_A + rss_B)) * 100 if (rss_A + rss_B) > 0 else 0
    
    return {
        'F_stat': float(F_stat),
        'p_value': float(p_value),
        'can_merge': p_value > 0.05,
        'rmse_A': float(np.sqrt(mean_squared_error(y_A, model_A.predict(X_A)))),
        'rmse_B': float(np.sqrt(mean_squared_error(y_B, model_B.predict(X_B)))),
        'rss_A': float(rss_A),
        'rss_B': float(rss_B),
        'rss_AB': float(rss_AB),
        'residual_increase_pct': float(residual_increase),
        'n_A': int(n_A),
        'n_B': int(n_B),
        'aligned_iter': int(aligned_iter),
        'chow_interpretation': f"p={p_value:.4f} {'✓ 可合并' if p_value > 0.05 else '✗ 不可合并'} (残差增长 {residual_increase:.1f}%)"
    }


def performance_comparison_with_aligned_iter(group_A_data: pd.DataFrame,
                                              group_B_data: pd.DataFrame,
                                              surface: str,
                                              features: list,
                                              target_col: str,
                                              aligned_iter: int) -> dict:
    """
    性能对比 - 使用预设的对齐迭代次数
    """
    mask_A = ~(group_A_data[features].isna().any(axis=1) | group_A_data[target_col].isna())
    X_A = group_A_data.loc[mask_A, features]
    y_A = group_A_data.loc[mask_A, target_col]
    
    mask_B = ~(group_B_data[features].isna().any(axis=1) | group_B_data[target_col].isna())
    X_B = group_B_data.loc[mask_B, features]
    y_B = group_B_data.loc[mask_B, target_col]
    
    if len(X_A) < 10 or len(X_B) < 10:
        return {
            'rmse_separate': np.inf,
            'rmse_merged': np.inf,
            'performance_loss_pct': np.nan,
            'can_merge': False
        }
    
    # 分开训练
    model_A = HistGradientBoostingRegressor(
        max_iter=aligned_iter,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    model_A.fit(X_A, y_A)
    rmse_A = np.sqrt(mean_squared_error(y_A, model_A.predict(X_A)))
    
    model_B = HistGradientBoostingRegressor(
        max_iter=aligned_iter,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    model_B.fit(X_B, y_B)
    rmse_B = np.sqrt(mean_squared_error(y_B, model_B.predict(X_B)))
    
    rmse_separate = (rmse_A + rmse_B) / 2
    
    # 合并训练
    combined_data = pd.concat([group_A_data, group_B_data], ignore_index=True)
    mask_AB = ~(combined_data[features].isna().any(axis=1) | combined_data[target_col].isna())
    X_AB = combined_data.loc[mask_AB, features]
    y_AB = combined_data.loc[mask_AB, target_col]
    
    model_AB = HistGradientBoostingRegressor(
        max_iter=aligned_iter,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    model_AB.fit(X_AB, y_AB)
    rmse_AB = np.sqrt(mean_squared_error(y_AB, model_AB.predict(X_AB)))
    
    performance_loss = (rmse_AB - rmse_separate) / rmse_separate if rmse_separate > 0 else np.nan
    
    return {
        'rmse_A': float(rmse_A),
        'rmse_B': float(rmse_B),
        'rmse_separate': float(rmse_separate),
        'rmse_merged': float(rmse_AB),
        'performance_loss_pct': float(performance_loss * 100) if not np.isnan(performance_loss) else np.nan,
        'can_merge': performance_loss < 0.05 if not np.isnan(performance_loss) else False,
        'aligned_iter': int(aligned_iter)
    }
    """
    改进的 Chow 检验（对齐收敛程度版本）
    
    H0: 两个规格组的模型参数相同（可以合并）
    H1: 两个规格组的模型参数不同（应该分开）
    
    改进点:
      1. 对各规格组独立找最优迭代次数
      2. 用最小迭代次数对齐三个模型
      3. 保证残差差异来自数据而非超参数
    """
    print(f"\n    [Chow 检验] 对齐迭代次数版本")
    
    # ========== 第一步：分别训练，找最优迭代次数 ==========
    print(f"    · 规格组 A: 找最优迭代次数...", end='', flush=True)
    model_A, rmse_A, mae_A, res_A, diag_A = train_model_for_group(
        group_A_data, surface, features, target_col, max_iter=None
    )
    opt_iter_A = diag_A['actual_n_iter'] if diag_A else 50
    print(f" {opt_iter_A} iter (RMSE={rmse_A:.4f})")
    
    print(f"    · 规格组 B: 找最优迭代次数...", end='', flush=True)
    model_B, rmse_B, mae_B, res_B, diag_B = train_model_for_group(
        group_B_data, surface, features, target_col, max_iter=None
    )
    opt_iter_B = diag_B['actual_n_iter'] if diag_B else 50
    print(f" {opt_iter_B} iter (RMSE={rmse_B:.4f})")
    
    if model_A is None or model_B is None:
        return {
            'F_stat': np.nan,
            'p_value': np.nan,
            'can_merge': False,
            'chow_interpretation': '样本量不足',
            'optimal_iter_A': opt_iter_A,
            'optimal_iter_B': opt_iter_B,
            'aligned_iter': 'N/A',
            'diagnostics': 'Sample size insufficient'
        }
    
    # ========== 第二步：对齐迭代次数 ==========
    aligned_iter = min(opt_iter_A, opt_iter_B)
    print(f"    · 对齐迭代次数: {aligned_iter} (min of {opt_iter_A}, {opt_iter_B})")
    
    # ========== 第三步：用对齐迭代次数重新训练 ==========
    print(f"    · 使用对齐迭代次数重新训练...", end='', flush=True)
    
    # 获取特征和目标
    mask_A = ~(group_A_data[features].isna().any(axis=1) | group_A_data[target_col].isna())
    X_A = group_A_data.loc[mask_A, features]
    y_A = group_A_data.loc[mask_A, target_col]
    
    mask_B = ~(group_B_data[features].isna().any(axis=1) | group_B_data[target_col].isna())
    X_B = group_B_data.loc[mask_B, features]
    y_B = group_B_data.loc[mask_B, target_col]
    
    # 重新训练三个模型，使用对齐的迭代次数
    model_A_aligned = HistGradientBoostingRegressor(
        max_iter=aligned_iter,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    model_A_aligned.fit(X_A, y_A)
    
    model_B_aligned = HistGradientBoostingRegressor(
        max_iter=aligned_iter,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    model_B_aligned.fit(X_B, y_B)
    
    # 合并训练
    combined_data = pd.concat([group_A_data, group_B_data], ignore_index=True)
    mask_AB = ~(combined_data[features].isna().any(axis=1) | combined_data[target_col].isna())
    X_AB = combined_data.loc[mask_AB, features]
    y_AB = combined_data.loc[mask_AB, target_col]
    
    model_AB_aligned = HistGradientBoostingRegressor(
        max_iter=aligned_iter,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    model_AB_aligned.fit(X_AB, y_AB)
    print(" ✓")
    
    # ========== 第四步：计算残差平方和 ==========
    rss_A = compute_rss(y_A.values, model_A_aligned.predict(X_A.values))
    rss_B = compute_rss(y_B.values, model_B_aligned.predict(X_B.values))
    rss_AB = compute_rss(y_AB.values, model_AB_aligned.predict(X_AB.values))
    
    # ========== 第五步：计算 Chow 统计量 ==========
    k = len(features)
    n_A = len(X_A)
    n_B = len(X_B)
    
    numerator = (rss_AB - rss_A - rss_B) / k
    denominator = (rss_A + rss_B) / (n_A + n_B - 2*k)
    
    if denominator <= 0:
        return {
            'F_stat': np.nan,
            'p_value': np.nan,
            'can_merge': False,
            'chow_interpretation': '分母为零或负数',
            'optimal_iter_A': opt_iter_A,
            'optimal_iter_B': opt_iter_B,
            'aligned_iter': aligned_iter,
            'diagnostics': 'Invalid denominator'
        }
    
    F_stat = numerator / denominator
    p_value = 1 - f_distribution.cdf(F_stat, k, n_A + n_B - 2*k)
    
    # ========== 第六步：诊断信息 ==========
    residual_increase = ((rss_AB - rss_A - rss_B) / (rss_A + rss_B)) * 100 if (rss_A + rss_B) > 0 else 0
    
    return {
        'F_stat': float(F_stat),
        'p_value': float(p_value),
        'can_merge': p_value > 0.05,
        'rmse_A': float(rmse_A),
        'rmse_B': float(rmse_B),
        'rss_A': float(rss_A),
        'rss_B': float(rss_B),
        'rss_AB': float(rss_AB),
        'residual_increase_pct': float(residual_increase),
        'n_A': int(n_A),
        'n_B': int(n_B),
        'k_features': int(k),
        'optimal_iter_A': int(opt_iter_A),
        'optimal_iter_B': int(opt_iter_B),
        'aligned_iter': int(aligned_iter),
        'chow_interpretation': f"p={p_value:.4f} {'✓ 可合并' if p_value > 0.05 else '✗ 不可合并'} (残差增长 {residual_increase:.1f}%)"
    }


def performance_comparison(group_A_data: pd.DataFrame,
                          group_B_data: pd.DataFrame,
                          surface: str,
                          features: list,
                          target_col: str) -> dict:
    """
    性能对比：合并模型 vs 分开模型
    （使用对齐的迭代次数）
    """
    # 分别训练，找最优迭代次数
    model_A, rmse_A, mae_A, _, diag_A = train_model_for_group(
        group_A_data, surface, features, target_col, max_iter=None
    )
    model_B, rmse_B, mae_B, _, diag_B = train_model_for_group(
        group_B_data, surface, features, target_col, max_iter=None
    )
    
    if model_A is None or model_B is None:
        return {
            'rmse_separate': np.inf,
            'rmse_merged': np.inf,
            'performance_loss_pct': np.nan,
            'can_merge': False,
            'note': '样本量不足'
        }
    
    # 对齐迭代次数
    opt_iter_A = diag_A['actual_n_iter'] if diag_A else 50
    opt_iter_B = diag_B['actual_n_iter'] if diag_B else 50
    aligned_iter = min(opt_iter_A, opt_iter_B)
    
    # 用对齐迭代次数重新训练
    mask_A = ~(group_A_data[features].isna().any(axis=1) | group_A_data[target_col].isna())
    X_A = group_A_data.loc[mask_A, features]
    y_A = group_A_data.loc[mask_A, target_col]
    
    mask_B = ~(group_B_data[features].isna().any(axis=1) | group_B_data[target_col].isna())
    X_B = group_B_data.loc[mask_B, features]
    y_B = group_B_data.loc[mask_B, target_col]
    
    # 分开训练
    model_A_aligned = HistGradientBoostingRegressor(
        max_iter=aligned_iter,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    model_A_aligned.fit(X_A, y_A)
    rmse_A_aligned = np.sqrt(mean_squared_error(y_A, model_A_aligned.predict(X_A)))
    
    model_B_aligned = HistGradientBoostingRegressor(
        max_iter=aligned_iter,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    model_B_aligned.fit(X_B, y_B)
    rmse_B_aligned = np.sqrt(mean_squared_error(y_B, model_B_aligned.predict(X_B)))
    
    rmse_separate = (rmse_A_aligned + rmse_B_aligned) / 2
    
    # 合并训练
    combined_data = pd.concat([group_A_data, group_B_data], ignore_index=True)
    mask_AB = ~(combined_data[features].isna().any(axis=1) | combined_data[target_col].isna())
    X_AB = combined_data.loc[mask_AB, features]
    y_AB = combined_data.loc[mask_AB, target_col]
    
    model_AB_aligned = HistGradientBoostingRegressor(
        max_iter=aligned_iter,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    model_AB_aligned.fit(X_AB, y_AB)
    rmse_AB_aligned = np.sqrt(mean_squared_error(y_AB, model_AB_aligned.predict(X_AB)))
    
    performance_loss = (rmse_AB_aligned - rmse_separate) / rmse_separate if rmse_separate > 0 else np.nan
    
    return {
        'rmse_A': float(rmse_A_aligned),
        'rmse_B': float(rmse_B_aligned),
        'rmse_separate': float(rmse_separate),
        'rmse_merged': float(rmse_AB_aligned),
        'performance_loss_pct': float(performance_loss * 100) if not np.isnan(performance_loss) else np.nan,
        'can_merge': performance_loss < 0.05 if not np.isnan(performance_loss) else False,
        'aligned_iter': int(aligned_iter)
    }


def test_candidate_pair(df: pd.DataFrame, candidate: dict, 
                        optimal_iters_cache: Dict[str, int],
                        features_dict: Dict[str, List[str]]) -> dict:
    """
    对一个候选对进行分层验证
    
    参数:
      optimal_iters_cache: 缓存的最优迭代次数
      features_dict: {'Top': [...], 'Bot': [...]}
    """
    group_A = candidate['group_A']
    group_B = candidate['group_B']
    surface = candidate['surface']
    target_col = f'{surface}_Delta'
    features = features_dict.get(surface, [])
    
    group_A_data = df[df['Setpoint_Group_Label'] == group_A].copy()
    group_B_data = df[df['Setpoint_Group_Label'] == group_B].copy()
    
    tier = candidate.get('tier', 3)
    
    # ===== Tier 3: 直接排除 =====
    if tier == 3:
        return {
            'group_A': group_A,
            'group_B': group_B,
            'surface': surface,
            'tier': 3,
            'final_decision': 'NO',
            'reason': f"Tier 3: KS p={candidate.get('p_value', 0):.4f} 明显不同，跳过验证",
            'chow_test': None,
            'performance': None,
            'validation_method': 'skipped'
        }
    
    print(f"\n  [{['','','Tier 2','Tier 1'][tier]}] {group_A} + {group_B} ({surface})")
    print(f"    样本: {len(group_A_data)} + {len(group_B_data)}")
    
    # ===== Tier 2: 快速兼容性检查 =====
    if tier == 2:
        is_compatible, reason = quick_compatibility_check(group_A_data, group_B_data, features)
        
        if not is_compatible:
            print(f"    快速检查: ✗ 不兼容 ({reason})")
            return {
                'group_A': group_A,
                'group_B': group_B,
                'surface': surface,
                'tier': 2,
                'final_decision': 'NO',
                'reason': f"Tier 2 快速检查失败: {reason}",
                'chow_test': None,
                'performance': None,
                'validation_method': 'quick_check_failed'
            }
        
        print(f"    快速检查: ✓ 可能兼容 ({reason})")
        # 如果快速检查通过，还要进行 Chow 检验
        print(f"    进入 Chow 检验...")
    
    # ===== Tier 1 和 Tier 2（通过快速检查）: 完整 Chow 检验 =====
    print(f"    [Chow 检验] 对齐迭代次数版本", flush=True)
    
    # 从缓存获取迭代次数
    opt_iter_A = optimal_iters_cache.get(group_A, 50)
    opt_iter_B = optimal_iters_cache.get(group_B, 50)
    aligned_iter = min(opt_iter_A, opt_iter_B)
    
    print(f"    · 规格组 A: {opt_iter_A} iter", end='')
    print(f" | 规格组 B: {opt_iter_B} iter", end='')
    print(f" | 对齐: {aligned_iter} iter")
    
    # 执行完整 Chow 检验
    chow_result = chow_test_with_aligned_iter(
        group_A_data, group_B_data, surface, features, target_col, aligned_iter
    )
    
    # 性能对比
    perf_result = performance_comparison_with_aligned_iter(
        group_A_data, group_B_data, surface, features, target_col, aligned_iter
    )
    
    # 综合判断
    can_merge = chow_result['can_merge'] and perf_result['can_merge']
    
    return {
        'group_A': group_A,
        'group_B': group_B,
        'surface': surface,
        'tier': tier,
        'final_decision': 'YES' if can_merge else 'NO',
        'reason': _get_merge_reason(chow_result, perf_result),
        'chow_test': chow_result,
        'performance': perf_result,
        'validation_method': 'chow_test'
    }


def _get_merge_reason(chow_result: dict, perf_result: dict, spec_distance: float = None) -> str:
    """获取合并决策的理由（优化版 v2）
    
    【新增】加入规格距离的解释信息
    """
    reasons = []
    
    # 规格距离信息
    if spec_distance is not None:
        if spec_distance < 0.5:
            reasons.append(f"规格距离 {spec_distance:.3f} (接近) ✓")
        elif spec_distance < 1.0:
            reasons.append(f"规格距离 {spec_distance:.3f} (中等)")
        else:
            reasons.append(f"规格距离 {spec_distance:.3f} (差异大) ✗")
    
    # Chow 检验结论
    chow_pval = chow_result.get('p_value', np.nan)
    chow_threshold = chow_result.get('chow_threshold', 0.05)
    
    if np.isnan(chow_pval):
        reasons.append("Chow: 无法计算")
    elif chow_pval > chow_threshold:
        reasons.append(f"Chow p={chow_pval:.4f} > {chow_threshold:.4f} ✓")
    else:
        reasons.append(f"Chow p={chow_pval:.4f} ≤ {chow_threshold:.4f} ✗")
        if 'residual_increase_pct' in chow_result:
            reasons[-1] += f" (残差增长 {chow_result['residual_increase_pct']:.1f}%)"
    
    # 性能对比结论
    perf_loss = perf_result.get('performance_loss_pct', np.nan)
    if np.isnan(perf_loss):
        reasons.append("性能: 无法计算")
    elif perf_loss < 5:
        reasons.append(f"性能损失 {perf_loss:.2f}% < 5% ✓")
    else:
        reasons.append(f"性能损失 {perf_loss:.2f}% ≥ 5% ✗")
    
    return " | ".join(reasons)


def main():
    """主流程（优化版 - 缓存 + 分层）"""
    config_path = "config.yaml"
    candidates_csv = "result/spec_group_merge_analysis/merge_candidates.csv"
    
    print("\n" + "="*80)
    print("【阶段 B：规格组合并验证（优化版 - 缓存 + 分层）】")
    print("="*80)
    print("\n优化策略:")
    print("  ✓ 缓存: 15 个规格组的最优迭代次数只计算一次")
    print("  ✓ 分层: Tier 1 深度验证 | Tier 2 快速检查 | Tier 3 直接排除")
    print("  ✓ 预期: 从 2+ 小时降至 5-15 分钟")
    
    pipeline_start = time.time()
    
    try:
        # 1. 加载候选对
        df_cand = load_merge_candidates(candidates_csv)
        
        # 2. 准备数据
        df = load_featured_data(config_path)
        df = build_spec_groups(df)
        
        # 3. 构建特征字典
        features_dict = {
            'Top': get_feature_cols('Top'),
            'Bot': get_feature_cols('Bot')
        }
        
        # 4. 【优化】构建缓存：预计算所有规格组的最优迭代次数
        cache_start = time.time()
        optimal_iters_cache = build_optimal_iterations_cache(df, features_dict, '{surface}_Delta')
        cache_time = time.time() - cache_start
        
        # 5. 【优化】分层候选对
        print("\n[分层分析] 对候选对进行分类...")
        tier1, tier2, tier3 = classify_candidates(df_cand)
        print(f"\n  Tier 1 (深度验证): {len(tier1)} 对  - KS 通过或高质量")
        print(f"  Tier 2 (快速检查): {len(tier2)} 对  - KS 边界或中等质量")
        print(f"  Tier 3 (直接排除): {len(tier3)} 对  - KS 明显失败")
        print(f"  总计: {len(tier1) + len(tier2) + len(tier3)} 对")
        
        # 6. 执行分层验证
        print("\n" + "="*80)
        print("【执行分层验证】")
        print("="*80)
        
        results = []
        verified_count = 0  # 实际验证的对数
        
        # Tier 1: 全部深度验证
        if tier1:
            print(f"\n【Tier 1】深度验证 ({len(tier1)} 对)")
            print("-" * 80)
            tier1_start = time.time()
            for cand in tier1:
                result = test_candidate_pair(df, cand, optimal_iters_cache, features_dict)
                results.append(result)
                verified_count += 1
                print(f"    → {result['final_decision']}: {result['reason'][:70]}")
            tier1_time = time.time() - tier1_start
            print(f"  小计: {len(tier1)} 对, {tier1_time:.1f}s")
        
        # Tier 2: 快速检查 + 选择性深度验证
        if tier2:
            print(f"\n【Tier 2】快速检查 ({len(tier2)} 对)")
            print("-" * 80)
            tier2_start = time.time()
            tier2_passed = 0
            for cand in tier2:
                result = test_candidate_pair(df, cand, optimal_iters_cache, features_dict)
                results.append(result)
                if result['validation_method'] == 'chow_test':
                    verified_count += 1
                    tier2_passed += 1
                print(f"    → {result['final_decision']}: {result['reason'][:70]}")
            tier2_time = time.time() - tier2_start
            print(f"  小计: {len(tier2)} 对 ({tier2_passed} 进入深度验证), {tier2_time:.1f}s")
        
        # Tier 3: 直接排除
        if tier3:
            print(f"\n【Tier 3】直接排除 ({len(tier3)} 对)")
            print("  (这些对因 KS 明显失败而跳过昂贵的 Chow 检验)")
            for cand in tier3:
                result = test_candidate_pair(df, cand, optimal_iters_cache, features_dict)
                results.append(result)
        
        # 7. 汇总结果
        print("\n" + "="*80)
        print("【验证结果汇总】")
        print("="*80)
        
        can_merge_count = sum(1 for r in results if r['final_decision'] == 'YES')
        tier1_passed = sum(1 for r in results if r['tier'] == 1 and r['final_decision'] == 'YES')
        tier2_passed = sum(1 for r in results if r['tier'] == 2 and r['final_decision'] == 'YES' and r['validation_method'] == 'chow_test')
        
        print(f"\n总体结果:")
        print(f"  ✓ 可以合并的对: {can_merge_count}/{len(results)}")
        print(f"    └─ Tier 1: {tier1_passed}/{len(tier1)}")
        print(f"    └─ Tier 2: {tier2_passed}/{len([r for r in results if r['tier'] == 2 and r['validation_method'] == 'chow_test'])}")
        print(f"\n验证统计:")
        print(f"  ├─ 完全验证 (Chow 检验): {verified_count} 对")
        print(f"  ├─ 快速排除: {len(tier3)} 对")
        print(f"  └─ 总候选对: {len(results)} 对")
        
        total_time = time.time() - pipeline_start
        print(f"\n性能指标:")
        print(f"  ├─ 缓存构建: {cache_time:.1f}s (一次性)")
        if tier1:
            print(f"  ├─ Tier 1 验证: {tier1_time:.1f}s")
        if tier2:
            print(f"  ├─ Tier 2 检查: {tier2_time:.1f}s")
        print(f"  └─ 总耗时: {total_time:.1f}s")
        
        # 8. 导出详细结果
        os.makedirs("result/spec_group_merge_analysis", exist_ok=True)
        
        results_detail = []
        for r in results:
            detail = {
                'group_A': r['group_A'],
                'group_B': r['group_B'],
                'surface': r['surface'],
                'tier': r['tier'],
                'validation_method': r['validation_method'],
                'final_decision': r['final_decision'],
                'reason': r['reason']
            }
            
            if r['chow_test']:
                detail['chow_test'] = {
                    'F_stat': r['chow_test'].get('F_stat'),
                    'p_value': r['chow_test'].get('p_value'),
                    'residual_increase_pct': r['chow_test'].get('residual_increase_pct'),
                    'aligned_iter': r['chow_test'].get('aligned_iter')
                }
            
            if r['performance']:
                detail['performance'] = {
                    'rmse_separate': r['performance'].get('rmse_separate'),
                    'rmse_merged': r['performance'].get('rmse_merged'),
                    'performance_loss_pct': r['performance'].get('performance_loss_pct')
                }
            
            results_detail.append(detail)
        
        results_json = json.dumps(results_detail, indent=2, ensure_ascii=False)
        
        with open("result/spec_group_merge_analysis/chow_test_results_optimized.json", 'w', encoding='utf-8') as f:
            f.write(results_json)
        
        print(f"\n✓ 详细结果已导出至: result/spec_group_merge_analysis/chow_test_results_optimized.json")
        
        # 9. 生成分层分析报告
        with open("result/spec_group_merge_analysis/tier_classification.csv", 'w', encoding='utf-8') as f:
            f.write("group_A,group_B,surface,tier,ks_p_value,mean_diff,merged_size,decision\n")
            for r in results:
                f.write(f"{r['group_A']},{r['group_B']},{r['surface']},{r['tier']},{r.get('p_value', 'N/A')},{r.get('mean_diff', 'N/A')},{r.get('merged_size', 'N/A')},{r['final_decision']}\n")
        
        print(f"✓ 分层分析已导出至: result/spec_group_merge_analysis/tier_classification.csv")
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"\n✗ 执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
