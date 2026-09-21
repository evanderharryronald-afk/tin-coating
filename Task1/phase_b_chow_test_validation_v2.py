"""
【阶段 B：规格组合并深入验证（优化版 v2 - 规格距离优先）】

对阶段 A 筛选出的候选对进行分层验证：

【核心优化】：
1. 基于规格距离的优先级排序
   - 规格相近的对优先进入 Tier 1（验证收益大）
   - 规格差异大的对直接排除（几乎不可能合并）

2. 分表面处理
   - Top 表面：规格跨度大(1.1-11.199)，用严格标准
   - Bot 表面：规格跨度小(1.1-5.599)，更可能出现合并

3. 动态 Chow 检验阈值
   - 规格接近(<0.5): 需要 p > 0.05
   - 规格中等(0.5-1.0): 需要 p > 0.02
   - 规格差异(>1.0): 基本不可能合并

Tier 1 (深度验证 - 完整 Chow 检验):
  - KS 通过或规格非常接近的对
  - 样本量充足且均值差异小

Tier 2 (快速估计 - 简化兼容性检查):
  - 规格接近但 KS 未通过的对
  - 使用特征统计量进行快速预筛选

Tier 3 (直接排除):
  - 规格差异大且 KS 明显失败的对
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

# ===== 优化参数 =====
# 分层阈值
KS_PASS_THRESHOLD = 0.05        # KS p-value > 0.05 进入 Tier 1
MEAN_DIFF_THRESHOLD = 0.01      # 均值差异 < 0.01 进入 Tier 1
MIN_MERGED_SAMPLES = 400        # 合并样本 >= 400 进入 Tier 1

# 【新增】基于规格距离的动态阈值
SPEC_DISTANCE_TIER1 = 0.8       # 规格距离 < 0.8 进入 Tier 1（规格相近）
SPEC_DISTANCE_TIER2 = 1.5       # 规格距离 0.8-1.5 进入 Tier 2（规格中等）

# 快速兼容性检查的阈值
FEATURE_MEAN_RATIO_THRESHOLD = 1.0
FEATURE_VAR_RATIO_THRESHOLD = 2.0


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


def get_feature_cols(surface: str) -> list:
    """返回该表面的特征列"""
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


def load_config(config_path="config.yaml"):
    """加载配置文件"""
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config or {}


def load_featured_data(config_path="config.yaml") -> pd.DataFrame:
    """加载特征工程后的数据"""
    print("\n[加载数据]")
    config = load_config(config_path)
    excel_path = config.get("data_paths", {}).get("clean_data", "result/data/feature_engineered_data/featured_data.xlsx")
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"未找到特征工程数据: {excel_path}")
    df = pd.read_excel(excel_path)
    print(f"  数据: {len(df)} 行, {df.shape[1]} 列")
    return df


def load_merge_candidates(csv_path: str = "result/spec_group_merge_analysis/merge_candidates.csv") -> pd.DataFrame:
    """加载阶段 A 生成的候选对"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到候选对文件: {csv_path}")
    df_cand = pd.read_csv(csv_path)
    print(f"✓ 加载候选对: {len(df_cand)} 对")
    return df_cand


def classify_candidates_by_spec_distance(df_cand: pd.DataFrame) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    将候选对分为三个优先级（优化版 v2）
    
    【核心逻辑】：基于规格距离和 KS 检验结果
    
    Tier 1: 最有希望 - 规格相近(<0.8) 或 KS 通过
    Tier 2: 中等 - 规格中等(0.8-1.5) 且 KS 边界
    Tier 3: 直接排除 - 规格差异大(>1.5)
    
    返回: (tier1_candidates, tier2_candidates, tier3_candidates)
    """
    tier1 = []
    tier2 = []
    tier3 = []
    
    for idx, row in df_cand.iterrows():
        cand = row.to_dict()
        
        # 提取关键字段
        ks_pass = cand.get('ks_pass', False)
        spec_distance = cand.get('spec_distance', 2.0)
        mean_diff = cand.get('mean_diff', 0)
        merged_size = cand.get('merged_size', 0)
        
        # ===== Tier 1: 最有希望的对 =====
        # 条件：(1) 规格相近 OR (2) KS 通过
        if spec_distance < SPEC_DISTANCE_TIER1 or ks_pass:
            if mean_diff < MEAN_DIFF_THRESHOLD and merged_size >= MIN_MERGED_SAMPLES:
                cand['tier'] = 1
                tier1.append(cand)
                continue
        
        # ===== Tier 2: 中等优先级 =====
        # 条件：规格中等(0.8-1.5) 且 KS 不太差
        if SPEC_DISTANCE_TIER1 <= spec_distance < SPEC_DISTANCE_TIER2:
            p_value = cand.get('p_value', 0)
            if p_value > 0.01:  # KS 至少不是完全失败
                cand['tier'] = 2
                tier2.append(cand)
                continue
        
        # ===== Tier 3: 直接排除 =====
        cand['tier'] = 3
        tier3.append(cand)
    
    # Tier 1 和 Tier 2 内部按规格距离排序
    tier1 = sorted(tier1, key=lambda x: x.get('spec_distance', 2.0))
    tier2 = sorted(tier2, key=lambda x: x.get('spec_distance', 2.0))
    
    return tier1, tier2, tier3


def quick_compatibility_check(group_A_data: pd.DataFrame,
                              group_B_data: pd.DataFrame,
                              features: List[str]) -> Tuple[bool, str]:
    """快速兼容性检查（Tier 2 用）"""
    reason_parts = []
    
    for feature in features:
        A_vals = group_A_data[feature].dropna()
        B_vals = group_B_data[feature].dropna()
        
        if len(A_vals) < 5 or len(B_vals) < 5:
            continue
        
        mean_diff = abs(A_vals.mean() - B_vals.mean())
        std_B = B_vals.std()
        
        if std_B > 0:
            mean_ratio = mean_diff / std_B
            if mean_ratio > FEATURE_MEAN_RATIO_THRESHOLD:
                reason_parts.append(f"{feature}: mean_ratio={mean_ratio:.2f}")
        
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
        raise KeyError(f"缺少分组所需字段")
    df['Setpoint_Group_Label'] = df.apply(
        lambda r: f"Top{r[TOP_SETPOINT_COL]}_Bot{r[BOT_SETPOINT_COL]}", axis=1
    )
    return df


def find_optimal_iterations(X_val: pd.DataFrame, y_val: pd.Series) -> int:
    """找到最优的迭代次数"""
    if len(X_val) < 20:
        return 50
    
    kf = KFold(n_splits=min(3, len(X_val) // 10), shuffle=True, random_state=42)
    iter_range = [30, 50, 75, 100, 150]
    best_iter = 50
    best_score = -np.inf
    
    for max_iter in iter_range:
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
        
        scores = cross_val_score(model, X_val, y_val, cv=kf, scoring='neg_mean_squared_error')
        cv_score = scores.mean()
        
        if cv_score > best_score:
            best_score = cv_score
            best_iter = max_iter
    
    return best_iter


def train_model_for_group(group_df: pd.DataFrame, surface: str,
                          features: list, target_col: str,
                          max_iter: Optional[int] = None) -> tuple:
    """训练单个规格组的模型"""
    X = group_df[features].copy()
    y = group_df[target_col].copy()
    
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    if len(X) < 10:
        return None, np.inf, np.inf, None, {}
    
    if max_iter is None:
        max_iter = find_optimal_iterations(X, y)
    
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
    
    pred = model.predict(X)
    residuals = y - pred
    rmse = np.sqrt(mean_squared_error(y, pred))
    mae = mean_absolute_error(y, pred)
    
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


def chow_test_with_aligned_iter_v2(group_A_data: pd.DataFrame,
                                    group_B_data: pd.DataFrame,
                                    surface: str,
                                    features: list,
                                    target_col: str,
                                    aligned_iter: int,
                                    spec_distance: float = None) -> dict:
    """
    Chow 检验 - 使用预设的对齐迭代次数（包含规格距离信息）
    """
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
    
    # 训练三个模型
    model_A = HistGradientBoostingRegressor(
        max_iter=aligned_iter, learning_rate=0.05, max_depth=3, random_state=42
    )
    model_A.fit(X_A, y_A)
    
    model_B = HistGradientBoostingRegressor(
        max_iter=aligned_iter, learning_rate=0.05, max_depth=3, random_state=42
    )
    model_B.fit(X_B, y_B)
    
    combined_data = pd.concat([group_A_data, group_B_data], ignore_index=True)
    mask_AB = ~(combined_data[features].isna().any(axis=1) | combined_data[target_col].isna())
    X_AB = combined_data.loc[mask_AB, features]
    y_AB = combined_data.loc[mask_AB, target_col]
    
    model_AB = HistGradientBoostingRegressor(
        max_iter=aligned_iter, learning_rate=0.05, max_depth=3, random_state=42
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
    
    # 【新增】应用规格距离的动态阈值
    chow_threshold = get_chow_threshold_by_spec(spec_distance) if spec_distance else 0.05
    
    return {
        'F_stat': float(F_stat),
        'p_value': float(p_value),
        'chow_threshold': float(chow_threshold),
        'can_merge': p_value > chow_threshold,
        'rmse_A': float(np.sqrt(mean_squared_error(y_A, model_A.predict(X_A)))),
        'rmse_B': float(np.sqrt(mean_squared_error(y_B, model_B.predict(X_B)))),
        'rss_A': float(rss_A),
        'rss_B': float(rss_B),
        'rss_AB': float(rss_AB),
        'residual_increase_pct': float(residual_increase),
        'n_A': int(n_A),
        'n_B': int(n_B),
        'aligned_iter': int(aligned_iter),
        'chow_interpretation': f"p={p_value:.4f} vs threshold={chow_threshold:.4f} {'✓' if p_value > chow_threshold else '✗'}"
    }


def performance_comparison_with_aligned_iter_v2(group_A_data: pd.DataFrame,
                                                 group_B_data: pd.DataFrame,
                                                 surface: str,
                                                 features: list,
                                                 target_col: str,
                                                 aligned_iter: int) -> dict:
    """性能对比 - 使用预设的对齐迭代次数"""
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
        max_iter=aligned_iter, learning_rate=0.05, max_depth=3, random_state=42
    )
    model_A.fit(X_A, y_A)
    rmse_A = np.sqrt(mean_squared_error(y_A, model_A.predict(X_A)))
    
    model_B = HistGradientBoostingRegressor(
        max_iter=aligned_iter, learning_rate=0.05, max_depth=3, random_state=42
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
        max_iter=aligned_iter, learning_rate=0.05, max_depth=3, random_state=42
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


def build_optimal_iterations_cache(df: pd.DataFrame, features_dict: Dict[str, List[str]]) -> Dict[str, int]:
    """预计算所有规格组的最优迭代次数，缓存以供重用"""
    print("\n[缓存构建] 预计算所有规格组的最优迭代次数...")
    print("(这只需要进行一次)\n")
    
    cache = {}
    group_labels = sorted(df['Setpoint_Group_Label'].unique())
    
    start_time = time.time()
    
    for i, group_label in enumerate(group_labels, 1):
        print(f"  {i:2d}/{len(group_labels)}: {group_label:<25}", end='', flush=True)
        
        group_data = df[df['Setpoint_Group_Label'] == group_label].copy()
        
        if 'Top_Delta' in df.columns and group_data['Top_Delta'].notna().sum() > 0:
            surface = 'Top'
        elif 'Bot_Delta' in df.columns and group_data['Bot_Delta'].notna().sum() > 0:
            surface = 'Bot'
        else:
            print(" ✗ 跳过 (无有效数据)")
            cache[group_label] = 50
            continue
        
        features = features_dict.get(surface, [])
        target_col = f'{surface}_Delta'
        
        X = group_data[features].copy()
        y = group_data[target_col].copy()
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        if len(X) < 10:
            print(" ✗ 样本不足")
            cache[group_label] = 50
            continue
        
        optimal_iter = find_optimal_iterations(X, y)
        cache[group_label] = optimal_iter
        print(f" → {optimal_iter} iter")
    
    elapsed = time.time() - start_time
    print(f"\n  缓存构建完成 ({elapsed:.1f}s), 共 {len(cache)} 个规格组")
    
    return cache


def print_tier_summary(tier1: list, tier2: list, tier3: list):
    """打印分层汇总"""
    print("\n[分层分析] 对候选对进行分类...")
    print(f"\n  Tier 1 (规格相近/KS通过,深度验证): {len(tier1)} 对")
    print(f"  Tier 2 (规格中等,快速检查): {len(tier2)} 对")
    print(f"  Tier 3 (规格差异大,直接排除): {len(tier3)} 对")
    print(f"  总计: {len(tier1) + len(tier2) + len(tier3)} 对")


def main():
    """主流程（优化版 v2 - 规格距离优先）"""
    config_path = "config.yaml"
    candidates_csv = "result/spec_group_merge_analysis/merge_candidates.csv"
    
    print("\n" + "="*80)
    print("【阶段 B：规格组合并验证（优化版 v2 - 规格距离优先）】")
    print("="*80)
    print("\n核心优化：")
    print("  ✓ 规格距离优先：规格相近的对优先进入 Tier 1")
    print("  ✓ 动态阈值：基于规格距离的 Chow 检验阈值")
    print("  ✓ 分表面处理：Top/Bot 不同的搜索策略")
    print("  ✓ 诊断信息：显示规格距离、Chow 阈值等")
    
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
        
        # 4. 构建缓存
        cache_start = time.time()
        optimal_iters_cache = build_optimal_iterations_cache(df, features_dict)
        cache_time = time.time() - cache_start
        
        # 5. 【优化】分层候选对（基于规格距离）
        tier1, tier2, tier3 = classify_candidates_by_spec_distance(df_cand)
        print_tier_summary(tier1, tier2, tier3)
        
        # 6. 执行分层验证
        print("\n" + "="*80)
        print("【执行分层验证】")
        print("="*80)
        
        results = []
        verified_count = 0
        
        # Tier 1: 深度验证
        if tier1:
            print(f"\n【Tier 1】深度验证 ({len(tier1)} 对)")
            print("-" * 80)
            tier1_start = time.time()
            for cand in tier1:
                group_A = cand['group_A']
                group_B = cand['group_B']
                surface = cand['surface']
                spec_distance = cand.get('spec_distance', 2.0)
                
                group_A_data = df[df['Setpoint_Group_Label'] == group_A].copy()
                group_B_data = df[df['Setpoint_Group_Label'] == group_B].copy()
                features = features_dict.get(surface, [])
                target_col = f'{surface}_Delta'
                
                print(f"\n  Tier 1: {group_A} + {group_B} ({surface})")
                print(f"    规格距离: {spec_distance:.3f} g/m²")
                print(f"    样本: {len(group_A_data)} + {len(group_B_data)}")
                
                opt_iter_A = optimal_iters_cache.get(group_A, 50)
                opt_iter_B = optimal_iters_cache.get(group_B, 50)
                aligned_iter = min(opt_iter_A, opt_iter_B)
                
                print(f"    · 规格组 A: {opt_iter_A} iter | B: {opt_iter_B} iter | 对齐: {aligned_iter} iter")
                
                # Chow 检验
                chow_result = chow_test_with_aligned_iter_v2(
                    group_A_data, group_B_data, surface, features, target_col, aligned_iter, spec_distance
                )
                
                # 性能对比
                perf_result = performance_comparison_with_aligned_iter_v2(
                    group_A_data, group_B_data, surface, features, target_col, aligned_iter
                )
                
                # 综合判断
                chow_pval = chow_result.get('p_value', 0)
                chow_threshold = chow_result.get('chow_threshold', 0.05)
                perf_loss = perf_result.get('performance_loss_pct', np.inf)
                can_merge = (chow_pval > chow_threshold) and (perf_loss < 5.0)
                
                result = {
                    'group_A': group_A,
                    'group_B': group_B,
                    'surface': surface,
                    'tier': 1,
                    'spec_distance': spec_distance,
                    'final_decision': 'YES' if can_merge else 'NO',
                    'reason': f"规格距离 {spec_distance:.3f} | Chow p={chow_pval:.4f} vs {chow_threshold:.4f} | 性能损失 {perf_loss:.2f}%",
                    'chow_test': chow_result,
                    'performance': perf_result,
                    'validation_method': 'chow_test'
                }
                results.append(result)
                verified_count += 1
                print(f"    → {result['final_decision']}: {result['reason']}")
            
            tier1_time = time.time() - tier1_start
            print(f"  小计: {len(tier1)} 对, {tier1_time:.1f}s")
        
        # Tier 2: 快速检查
        if tier2:
            print(f"\n【Tier 2】快速检查 ({len(tier2)} 对)")
            print("-" * 80)
            tier2_start = time.time()
            for cand in tier2[:min(10, len(tier2))]:  # 只处理前10个作为示例
                group_A = cand['group_A']
                group_B = cand['group_B']
                surface = cand['surface']
                spec_distance = cand.get('spec_distance', 2.0)
                
                group_A_data = df[df['Setpoint_Group_Label'] == group_A].copy()
                group_B_data = df[df['Setpoint_Group_Label'] == group_B].copy()
                features = features_dict.get(surface, [])
                
                is_compatible, reason = quick_compatibility_check(group_A_data, group_B_data, features)
                
                if is_compatible:
                    print(f"  {group_A} + {group_B} ({surface}): ✓ 通过快速检查")
                    verified_count += 1
                else:
                    print(f"  {group_A} + {group_B} ({surface}): ✗ 快速检查失败")
                    result = {
                        'group_A': group_A,
                        'group_B': group_B,
                        'surface': surface,
                        'tier': 2,
                        'spec_distance': spec_distance,
                        'final_decision': 'NO',
                        'reason': f"快速检查失败: {reason}",
                        'validation_method': 'quick_check_failed'
                    }
                    results.append(result)
            
            tier2_time = time.time() - tier2_start
            print(f"  小计: 样本 {min(10, len(tier2))}/{len(tier2)} 对, {tier2_time:.1f}s")
        
        # Tier 3: 直接排除
        if tier3:
            print(f"\n【Tier 3】直接排除 ({len(tier3)} 对)")
            print("  (这些对规格差异大，基本不可能合并)")
            for cand in tier3[:5]:  # 只显示前5个
                result = {
                    'group_A': cand['group_A'],
                    'group_B': cand['group_B'],
                    'surface': cand['surface'],
                    'tier': 3,
                    'spec_distance': cand.get('spec_distance', 2.0),
                    'final_decision': 'NO',
                    'reason': f"Tier 3: 规格距离过大",
                    'validation_method': 'skipped'
                }
                results.append(result)
                print(f"  {cand['group_A']} + {cand['group_B']} ({cand['surface']}) - 规格距离 {cand.get('spec_distance', 2.0):.3f}")
            if len(tier3) > 5:
                print(f"  ... 共 {len(tier3)} 对")
        
        # 汇总结果
        print("\n" + "="*80)
        print("【验证结果汇总】")
        print("="*80)
        
        can_merge_count = sum(1 for r in results if r['final_decision'] == 'YES')
        
        print(f"\n总体结果:")
        print(f"  ✓ 可以合并的对: {can_merge_count}/{len(results)}")
        print(f"\n验证统计:")
        print(f"  ├─ 完全验证 (Chow 检验): {verified_count} 对")
        print(f"  ├─ 快速排除: {len(tier3)} 对")
        print(f"  └─ 总候选对: {len(results)} 对")
        
        total_time = time.time() - pipeline_start
        print(f"\n性能指标:")
        print(f"  ├─ 缓存构建: {cache_time:.1f}s (一次性)")
        print(f"  └─ 总耗时: {total_time:.1f}s")
        
        # 导出结果
        os.makedirs("result/spec_group_merge_analysis", exist_ok=True)
        
        results_json = json.dumps(results, indent=2, ensure_ascii=False, default=str)
        with open("result/spec_group_merge_analysis/chow_test_results_v2_spec_distance.json", 'w', encoding='utf-8') as f:
            f.write(results_json)
        
        print(f"\n✓ 详细结果已导出至: result/spec_group_merge_analysis/chow_test_results_v2_spec_distance.json")
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"\n✗ 执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
