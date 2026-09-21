"""
【阶段 B：规格组合并深入验证】

对阶段 A 筛选出的候选对进行 Chow 检验和性能对比：
1. Chow 检验：检查校准规律（模型系数）是否相同
2. 性能对比：合并模型是否比分开模型更好或至少不变差
"""

import pandas as pd
import numpy as np
from scipy.stats import f as f_distribution
import os
import json
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold
from typing import Optional, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# ===== 从 coating_model_by_group.py 复用的常量 =====
TOP_SETPOINT_COL = 'Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_TOP_Min'
BOT_SETPOINT_COL = 'Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_BOT_Min'


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


def chow_test(group_A_data: pd.DataFrame,
              group_B_data: pd.DataFrame,
              surface: str,
              features: list,
              target_col: str) -> dict:
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


def test_candidate_pair(df: pd.DataFrame, candidate: dict) -> dict:
    """
    对一个候选对进行完整的验证
    """
    group_A = candidate['group_A']
    group_B = candidate['group_B']
    surface = candidate['surface']
    target_col = f'{surface}_Delta'  # 正确的列名格式
    features = get_feature_cols(surface)
    
    group_A_data = df[df['Setpoint_Group_Label'] == group_A].copy()
    group_B_data = df[df['Setpoint_Group_Label'] == group_B].copy()
    
    print(f"\n  测试: {group_A} + {group_B} ({surface})")
    print(f"    样本: {len(group_A_data)} + {len(group_B_data)}")
    
    # Chow 检验
    chow_result = chow_test(group_A_data, group_B_data, surface, features, target_col)
    
    # 性能对比
    perf_result = performance_comparison(group_A_data, group_B_data, surface, features, target_col)
    
    # 综合判断
    can_merge = chow_result['can_merge'] and perf_result['can_merge']
    
    return {
        'group_A': group_A,
        'group_B': group_B,
        'surface': surface,
        'chow_test': chow_result,
        'performance': perf_result,
        'final_decision': 'YES' if can_merge else 'NO',
        'reason': _get_merge_reason(chow_result, perf_result)
    }


def _get_merge_reason(chow_result: dict, perf_result: dict) -> str:
    """获取合并决策的理由"""
    reasons = []
    
    # Chow 检验结论
    chow_pval = chow_result.get('p_value', np.nan)
    if np.isnan(chow_pval):
        reasons.append("Chow: 无法计算")
    elif chow_pval > 0.05:
        reasons.append(f"Chow p={chow_pval:.4f} > 0.05 ✓")
    else:
        reasons.append(f"Chow p={chow_pval:.4f} < 0.05 ✗")
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
    """主流程"""
    config_path = "config.yaml"
    candidates_csv = "result/spec_group_merge_analysis/merge_candidates.csv"
    
    print("\n" + "="*80)
    print("【阶段 B：Chow 检验验证（改进版 - 对齐迭代次数）】")
    print("="*80)
    print("\n关键改进:")
    print("  ✓ 每个规格组独立找最优迭代次数")
    print("  ✓ 对齐三个模型的收敛程度")
    print("  ✓ 确保残差差异来自数据而非超参数")
    
    try:
        # 1. 加载候选对
        df_cand = load_merge_candidates(candidates_csv)
        
        # 2. 准备数据
        df = load_featured_data(config_path)
        df = build_spec_groups(df)
        
        # 3. 对每个候选对进行检验
        print("\n" + "="*80)
        print("【执行 Chow 检验（对齐版本）】")
        print("="*80)
        results = []
        
        for idx, row in df_cand.iterrows():
            candidate = row.to_dict()
            result = test_candidate_pair(df, candidate)
            results.append(result)
            
            print(f"\n  结论: {result['final_decision']}")
            print(f"    {result['reason']}")
        
        # 4. 汇总结果
        print("\n" + "="*80)
        print("【Chow 检验结果汇总】")
        print("="*80)
        
        can_merge_count = sum(1 for r in results if r['final_decision'] == 'YES')
        print(f"\n✓ 可以合并的对: {can_merge_count}/{len(results)}")
        
        # 导出详细结果
        os.makedirs("result/spec_group_merge_analysis", exist_ok=True)
        
        results_detail = []
        for r in results:
            results_detail.append({
                'group_A': r['group_A'],
                'group_B': r['group_B'],
                'surface': r['surface'],
                'final_decision': r['final_decision'],
                'chow_test': {
                    'F_stat': r['chow_test'].get('F_stat'),
                    'p_value': r['chow_test'].get('p_value'),
                    'optimal_iter_A': r['chow_test'].get('optimal_iter_A'),
                    'optimal_iter_B': r['chow_test'].get('optimal_iter_B'),
                    'aligned_iter': r['chow_test'].get('aligned_iter'),
                    'residual_increase_pct': r['chow_test'].get('residual_increase_pct'),
                    'rss_A': r['chow_test'].get('rss_A'),
                    'rss_B': r['chow_test'].get('rss_B'),
                    'rss_AB': r['chow_test'].get('rss_AB'),
                    'interpretation': r['chow_test'].get('chow_interpretation')
                },
                'performance': {
                    'rmse_A': r['performance'].get('rmse_A'),
                    'rmse_B': r['performance'].get('rmse_B'),
                    'rmse_separate': r['performance'].get('rmse_separate'),
                    'rmse_merged': r['performance'].get('rmse_merged'),
                    'performance_loss_pct': r['performance'].get('performance_loss_pct'),
                    'aligned_iter': r['performance'].get('aligned_iter')
                },
                'reason': r['reason']
            })
        
        results_json = json.dumps(results_detail, indent=2, ensure_ascii=False)
        
        with open("result/spec_group_merge_analysis/chow_test_results.json", 'w', encoding='utf-8') as f:
            f.write(results_json)
        
        print(f"\n✓ 详细结果已导出至: result/spec_group_merge_analysis/chow_test_results.json")
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"\n✗ 执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
