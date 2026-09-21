"""
【阶段 B：规格组合并验证（优化版 v3 - 改进决策逻辑）】

【v3 核心改进】：
1. 决策条件改为 OR 逻辑：Chow 通过 OR 性能提升
2. 样本量不平衡检查：比例 > 5:1 时降低期望
3. 支持可配置的性能损失容限
4. 增强诊断信息：标记样本不平衡、统计不可靠等

决策矩阵（新）：
  ✓ YES: (Chow p > 阈值) OR (性能提升 > 3%)
  ✗ NO:  (Chow p ≤ 阈值) AND (性能损失 ≥ 3%)
  ⚠️ MAYBE: 边界情况，标记为可信度低
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

TOP_SETPOINT_COL = 'Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_TOP_Min'
BOT_SETPOINT_COL = 'Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_BOT_Min'

# ===== 改进的参数 =====
SPEC_DISTANCE_HARD_LIMIT = 1.0      # 硬性规格距离约束：> 1.0 直接排除
SAMPLE_IMBALANCE_RATIO = 5.0        # 样本比例 > 5:1 时标记为不平衡
PERF_GAIN_THRESHOLD = 0.03          # 性能提升 > 3% 时可接受
PERF_LOSS_THRESHOLD = 0.03          # 性能损失 > 3% 时拒绝
CHOW_PASS_THRESHOLD = 0.05          # Chow p > 0.05 判定通过
MIN_SAMPLE_SIZE = 20                # 最小样本量要求


def parse_spec_values(group_label: str) -> tuple:
    parts = group_label.split('_')
    top_val = float(parts[0].replace('Top', ''))
    bot_val = float(parts[1].replace('Bot', ''))
    return top_val, bot_val


def compute_spec_distance(group_A: str, group_B: str) -> float:
    top_A, bot_A = parse_spec_values(group_A)
    top_B, bot_B = parse_spec_values(group_B)
    distance = np.sqrt((top_A - top_B)**2 + (bot_A - bot_B)**2)
    return distance


def get_chow_threshold_by_spec(spec_distance: float) -> float:
    """基于规格距离的动态阈值"""
    if spec_distance < 0.5:
        return 0.05
    elif spec_distance < 1.0:
        return 0.02
    else:
        return 0.001


def get_feature_cols(surface: str) -> list:
    if surface == 'Top':
        return [
            'Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Avg',
            'Dimension_[mm]_Width',
            'Speed[m/min]_Process_Avg',
            'Top_Current_Sum',
            'Dimension_[mm]_Thickness'
        ]
    else:
        return [
            'Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Avg',
            'Dimension_[mm]_Width',
            'Speed[m/min]_Process_Avg',
            'Bot_Current_Sum',
            'Dimension_[mm]_Thickness'
        ]


def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config or {}


def load_featured_data(config_path="config.yaml") -> pd.DataFrame:
    print("\n[加载数据]")
    config = load_config(config_path)
    excel_path = config.get("data_paths", {}).get("clean_data", "result/data/feature_engineered_data/featured_data.xlsx")
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"未找到特征工程数据: {excel_path}")
    df = pd.read_excel(excel_path)
    print(f"  数据: {len(df)} 行, {df.shape[1]} 列")
    return df


def load_merge_candidates(csv_path: str = "result/spec_group_merge_analysis/merge_candidates.csv") -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到候选对文件: {csv_path}")
    df_cand = pd.read_csv(csv_path)
    print(f"✓ 加载候选对: {len(df_cand)} 对")
    return df_cand


def check_sample_imbalance(n_A: int, n_B: int) -> Tuple[bool, float, str]:
    """
    检查样本不平衡
    返回: (is_imbalanced, ratio, advice)
    """
    ratio = max(n_A, n_B) / min(n_A, n_B)
    is_imbalanced = ratio > SAMPLE_IMBALANCE_RATIO
    
    if is_imbalanced:
        advice = f"样本比 {ratio:.1f}:1，统计可靠性降低"
    else:
        advice = f"样本平衡 ({ratio:.1f}:1)"
    
    return is_imbalanced, ratio, advice


def build_spec_groups(df) -> pd.DataFrame:
    df = df.copy()
    if TOP_SETPOINT_COL not in df.columns or BOT_SETPOINT_COL not in df.columns:
        raise KeyError(f"缺少分组所需字段")
    df['Setpoint_Group_Label'] = df.apply(
        lambda r: f"Top{r[TOP_SETPOINT_COL]}_Bot{r[BOT_SETPOINT_COL]}", axis=1
    )
    return df


def find_optimal_iterations(X_val: pd.DataFrame, y_val: pd.Series) -> int:
    if len(X_val) < 20:
        return 50
    kf = KFold(n_splits=min(3, len(X_val) // 10), shuffle=True, random_state=42)
    iter_range = [30, 50, 75, 100, 150]
    best_iter = 50
    best_score = -np.inf
    
    for max_iter in iter_range:
        model = HistGradientBoostingRegressor(
            max_iter=max_iter, learning_rate=0.05, max_depth=3, random_state=42,
            early_stopping='auto', n_iter_no_change=5, validation_fraction=0.2, tol=1e-4
        )
        scores = cross_val_score(model, X_val, y_val, cv=kf, scoring='neg_mean_squared_error')
        cv_score = scores.mean()
        if cv_score > best_score:
            best_score = cv_score
            best_iter = max_iter
    return best_iter


def compute_rss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sum((y_true - y_pred) ** 2)


def chow_test_with_aligned_iter_v3(group_A_data: pd.DataFrame, group_B_data: pd.DataFrame,
                                     surface: str, features: list, target_col: str,
                                     aligned_iter: int, spec_distance: float = None) -> dict:
    """Chow 检验 - v3 改进版本"""
    
    mask_A = ~(group_A_data[features].isna().any(axis=1) | group_A_data[target_col].isna())
    X_A = group_A_data.loc[mask_A, features]
    y_A = group_A_data.loc[mask_A, target_col]
    
    mask_B = ~(group_B_data[features].isna().any(axis=1) | group_B_data[target_col].isna())
    X_B = group_B_data.loc[mask_B, features]
    y_B = group_B_data.loc[mask_B, target_col]
    
    n_A, n_B = len(X_A), len(X_B)
    
    if n_A < MIN_SAMPLE_SIZE or n_B < MIN_SAMPLE_SIZE:
        return {
            'F_stat': np.nan, 'p_value': np.nan, 'can_merge': False,
            'chow_interpretation': f'样本不足: A={n_A}, B={n_B}'
        }
    
    # 【新增】样本不平衡检查
    is_imbalanced, ratio, imbalance_msg = check_sample_imbalance(n_A, n_B)
    
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
    
    rss_A = compute_rss(y_A.values, model_A.predict(X_A.values))
    rss_B = compute_rss(y_B.values, model_B.predict(X_B.values))
    rss_AB = compute_rss(y_AB.values, model_AB.predict(X_AB.values))
    
    k = len(features)
    numerator = (rss_AB - rss_A - rss_B) / k
    denominator = (rss_A + rss_B) / (n_A + n_B - 2*k)
    
    if denominator <= 0:
        return {
            'F_stat': np.nan, 'p_value': np.nan, 'can_merge': False,
            'chow_interpretation': '分母无效'
        }
    
    F_stat = numerator / denominator
    p_value = 1 - f_distribution.cdf(F_stat, k, n_A + n_B - 2*k)
    residual_increase = ((rss_AB - rss_A - rss_B) / (rss_A + rss_B)) * 100 if (rss_A + rss_B) > 0 else 0
    
    chow_threshold = get_chow_threshold_by_spec(spec_distance) if spec_distance else CHOW_PASS_THRESHOLD
    chow_passed = p_value > chow_threshold
    
    return {
        'F_stat': float(F_stat),
        'p_value': float(p_value),
        'chow_threshold': float(chow_threshold),
        'chow_passed': chow_passed,
        'rmse_A': float(np.sqrt(mean_squared_error(y_A, model_A.predict(X_A)))),
        'rmse_B': float(np.sqrt(mean_squared_error(y_B, model_B.predict(X_B)))),
        'rss_A': float(rss_A),
        'rss_B': float(rss_B),
        'rss_AB': float(rss_AB),
        'residual_increase_pct': float(residual_increase),
        'n_A': int(n_A),
        'n_B': int(n_B),
        'sample_ratio': float(ratio),
        'is_sample_imbalanced': is_imbalanced,
        'sample_imbalance_msg': imbalance_msg,
        'aligned_iter': int(aligned_iter),
        'chow_interpretation': f"p={p_value:.4f} vs {chow_threshold:.4f} {'✓' if chow_passed else '✗'} [{imbalance_msg}]"
    }


def performance_comparison_v3(group_A_data: pd.DataFrame, group_B_data: pd.DataFrame,
                               surface: str, features: list, target_col: str,
                               aligned_iter: int) -> dict:
    """性能对比 - v3 支持性能提升"""
    
    mask_A = ~(group_A_data[features].isna().any(axis=1) | group_A_data[target_col].isna())
    X_A = group_A_data.loc[mask_A, features]
    y_A = group_A_data.loc[mask_A, target_col]
    
    mask_B = ~(group_B_data[features].isna().any(axis=1) | group_B_data[target_col].isna())
    X_B = group_B_data.loc[mask_B, features]
    y_B = group_B_data.loc[mask_B, target_col]
    
    if len(X_A) < MIN_SAMPLE_SIZE or len(X_B) < MIN_SAMPLE_SIZE:
        return {
            'rmse_separate': np.inf,
            'rmse_merged': np.inf,
            'performance_change_pct': np.nan,
            'has_performance_gain': False
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
    
    # 改为"变化百分比"而非"损失"（支持正负）
    performance_change = (rmse_AB - rmse_separate) / rmse_separate if rmse_separate > 0 else np.nan
    has_performance_gain = performance_change < -PERF_GAIN_THRESHOLD  # 提升 > 3%
    
    return {
        'rmse_A': float(rmse_A),
        'rmse_B': float(rmse_B),
        'rmse_separate': float(rmse_separate),
        'rmse_merged': float(rmse_AB),
        'performance_change_pct': float(performance_change * 100) if not np.isnan(performance_change) else np.nan,
        'has_performance_gain': has_performance_gain,
        'aligned_iter': int(aligned_iter)
    }


def decide_merge_v3(chow_result: dict, perf_result: dict, spec_distance: float) -> Tuple[str, str]:
    """
    v3 改进的合并决策逻辑（OR 逻辑）
    
    返回: (决策, 理由)
    """
    chow_pval = chow_result.get('p_value', np.nan)
    chow_threshold = chow_result.get('chow_threshold', 0.05)
    chow_passed = chow_pval > chow_threshold
    
    perf_change = perf_result.get('performance_change_pct', np.nan)
    has_gain = perf_result.get('has_performance_gain', False)
    
    is_imbalanced = chow_result.get('is_sample_imbalanced', False)
    ratio = chow_result.get('sample_ratio', 1.0)
    
    # 【新增】硬性规格距离约束
    if spec_distance > SPEC_DISTANCE_HARD_LIMIT:
        return 'NO', f"规格距离 {spec_distance:.3f} > {SPEC_DISTANCE_HARD_LIMIT}（硬性限制）"
    
    # 【改进】OR 逻辑
    if chow_passed:
        reason = f"Chow p={chow_pval:.4f} > {chow_threshold:.4f} ✓"
    elif has_gain:
        reason = f"性能提升 {abs(perf_change):.2f}% > {PERF_GAIN_THRESHOLD*100}% ✓"
    else:
        reason = f"Chow p={chow_pval:.4f} ≤ {chow_threshold:.4f} ✗ | 性能变化 {perf_change:.2f}%"
        if not chow_passed and perf_change is not None and perf_change > PERF_LOSS_THRESHOLD:
            reason += " ✗（性能恶化）"
        return 'NO', reason
    
    # 警告：样本不平衡
    if is_imbalanced:
        reason += f" | ⚠️ 样本比 {ratio:.1f}:1，可信度降低"
    
    return 'YES', reason


def main():
    """主流程 - v3"""
    print("\n" + "="*80)
    print("【阶段 B v3：改进决策逻辑和样本量处理】")
    print("="*80)
    print("\n【v3 改进】：")
    print("  ✓ 决策 OR 逻辑：Chow 通过 OR 性能提升")
    print("  ✓ 样本不平衡检查：ratio > 5:1 时标记")
    print("  ✓ 规格距离硬性约束：> 1.0 g/m² 直接排除")
    print("  ✓ 性能提升支持：-3% 改进时允许合并")
    
    config_path = "config.yaml"
    candidates_csv = "result/spec_group_merge_analysis/merge_candidates.csv"
    
    try:
        # 加载数据
        df_cand = load_merge_candidates(candidates_csv)
        df = load_featured_data(config_path)
        df = build_spec_groups(df)
        
        features_dict = {
            'Top': get_feature_cols('Top'),
            'Bot': get_feature_cols('Bot')
        }
        
        # 构建缓存
        print("\n[缓存构建] 预计算最优迭代次数...")
        cache = {}
        for group_label in sorted(df['Setpoint_Group_Label'].unique()):
            group_data = df[df['Setpoint_Group_Label'] == group_label]
            surface = 'Top' if 'Top_Delta' in df.columns and group_data['Top_Delta'].notna().sum() > 0 else 'Bot'
            features = features_dict.get(surface, [])
            target_col = f'{surface}_Delta'
            
            X = group_data[features].dropna()
            y = group_data[target_col].dropna()
            if len(X) >= 20:
                cache[group_label] = find_optimal_iterations(X[features], y)
            else:
                cache[group_label] = 50
        
        print(f"✓ 缓存完成: {len(cache)} 个规格组")
        
        # 执行验证
        print("\n" + "="*80)
        print("【执行 Chow 检验】")
        print("="*80)
        
        results = []
        merged_count = 0
        imbalanced_count = 0
        
        for idx, cand in df_cand.iterrows():
            group_A = cand['group_A']
            group_B = cand['group_B']
            surface = cand['surface']
            spec_distance = cand.get('spec_distance', 2.0)
            
            # 【新增】硬性规格距离约束
            if spec_distance > SPEC_DISTANCE_HARD_LIMIT:
                results.append({
                    'group_A': group_A,
                    'group_B': group_B,
                    'surface': surface,
                    'spec_distance': spec_distance,
                    'final_decision': 'NO',
                    'reason': f"规格距离 {spec_distance:.3f} > {SPEC_DISTANCE_HARD_LIMIT}",
                    'validation_method': 'hard_limit',
                    'skip_reason': 'spec_distance_exceeded'
                })
                continue
            
            group_A_data = df[df['Setpoint_Group_Label'] == group_A].copy()
            group_B_data = df[df['Setpoint_Group_Label'] == group_B].copy()
            features = features_dict.get(surface, [])
            target_col = f'{surface}_Delta'
            
            opt_iter_A = cache.get(group_A, 50)
            opt_iter_B = cache.get(group_B, 50)
            aligned_iter = min(opt_iter_A, opt_iter_B)
            
            # Chow 检验
            chow_result = chow_test_with_aligned_iter_v3(
                group_A_data, group_B_data, surface, features, target_col, aligned_iter, spec_distance
            )
            
            # 性能对比
            perf_result = performance_comparison_v3(
                group_A_data, group_B_data, surface, features, target_col, aligned_iter
            )
            
            # 决策
            decision, reason = decide_merge_v3(chow_result, perf_result, spec_distance)
            
            result = {
                'group_A': group_A,
                'group_B': group_B,
                'surface': surface,
                'spec_distance': spec_distance,
                'final_decision': decision,
                'reason': reason,
                'chow_test': chow_result,
                'performance': perf_result,
                'validation_method': 'chow_test_v3'
            }
            results.append(result)
            
            if decision == 'YES':
                merged_count += 1
                if chow_result.get('is_sample_imbalanced'):
                    imbalanced_count += 1
        
        # 输出结果
        print(f"\n✓ 验证完成: {len(results)} 对")
        print(f"  · 可合并: {merged_count} 对")
        print(f"  · 其中样本不平衡: {imbalanced_count} 对")
        
        # 保存结果
        os.makedirs("result/spec_group_merge_analysis", exist_ok=True)
        with open("result/spec_group_merge_analysis/chow_test_results_v3.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✓ 结果保存至: result/spec_group_merge_analysis/chow_test_results_v3.json")
        
    except Exception as e:
        print(f"\n✗ 执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
