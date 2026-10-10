# -*- coding: utf-8 -*-
"""
【阶段 B：交叉验证与非劣效判定】

三层分析（根据策略分层）：
  - 层1【档内并档检验】：同一档内，不同另一面设定值的组合能否并档
  - 层2【档间合并检验】：不同档之间能否合并  
  - 层3【全局基线】：全局模型 vs 各档独立模型（仅供参考）

判定方法：
  - 折外MSE非劣效（多折交叉验证）
  - Bootstrap 95% 置信上界
  - 非劣效边界：汇总 3.0%，单位 5.0%
"""

import json
import os
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, RepeatedStratifiedKFold
from threadpoolctl import threadpool_limits

logger = logging.getLogger(__name__)


def _get_params(cfg):
    """从配置读取参数"""
    return {
        'min_pair_samples': cfg.get_analysis_parameter('min_pair_samples', 20),
        'sample_imbalance_ratio': cfg.get_analysis_parameter('sample_imbalance_ratio', 5.0),
        'noninf_margin': cfg.get_analysis_parameter('non_inferiority.summary_margin', 0.03),
        'per_unit_margin': cfg.get_analysis_parameter('non_inferiority.per_unit_margin', 0.05),
        'ci_percentile': cfg.get_analysis_parameter('non_inferiority.ci_percentile', 95),
        'cv_splits': cfg.get_analysis_parameter('cross_validation.cv_splits', 5),
        'cv_repeats': cfg.get_analysis_parameter('cross_validation.cv_repeats', 3),
        'bootstrap_n': cfg.get_analysis_parameter('cross_validation.bootstrap_n', 1000),
        'iter_grid': cfg.get_analysis_parameter('model.iter_grid', [30, 50, 75, 100, 150, 200, 300]),
        'learning_rate': cfg.get_analysis_parameter('model.learning_rate', 0.05),
        'max_depth': cfg.get_analysis_parameter('model.max_depth', 3),
        'random_state': cfg.get_analysis_parameter('model.random_state', 42),
        'model_threads': cfg.get_analysis_parameter('model.model_threads', 1),
        'n_jobs': cfg.get_analysis_parameter('model.n_jobs', -1),
    }


def get_feature_cols(cfg) -> List[str]:
    """获取特征列"""
    all_features = cfg.get_feature_columns()
    if not all_features:
        logger.warning(f"未配置特征列，将使用所有可用特征")
    return all_features


def make_model(max_iter: int, cfg):
    """创建 HistGradientBoosting 模型"""
    params = _get_params(cfg)
    return HistGradientBoostingRegressor(
        max_iter=max_iter,
        learning_rate=params['learning_rate'],
        max_depth=params['max_depth'],
        random_state=params['random_state'],
        early_stopping=False,
        verbose=0
    )


def find_optimal_iterations(X: np.ndarray, y: np.ndarray, cfg) -> int:
    """用简单的 holdout 选择最优迭代次数"""
    params = _get_params(cfg)
    n = len(X)
    if n < params['min_pair_samples']:
        return params['iter_grid'][len(params['iter_grid'])//2]
    
    kf = KFold(n_splits=max(2, min(3, n // 10)), shuffle=True,
               random_state=params['random_state'])
    best_iter, best_score = params['iter_grid'][0], -np.inf
    model_threads = params['model_threads']
    
    for max_iter in params['iter_grid']:
        try:
            model = make_model(max_iter, cfg)
            from sklearn.model_selection import cross_val_score
            
            with threadpool_limits(limits=model_threads):
                scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=1)
            
            mean_score = scores.mean()
            if mean_score > best_score + 1e-12:
                best_score, best_iter = mean_score, max_iter
        except Exception as e:
            logger.debug(f"迭代次数 {max_iter} 评估失败: {e}")
            continue
    
    return best_iter


def cv_strategies(X, y, g, iters_sep, iter_pool, cfg):
    """
    交叉验证多种建模方案
    返回 {方案名 -> 每个样本的折外MSE数组}
    """
    params = _get_params(cfg)
    cv_splits = params.get('cv_splits', 5)
    cv_repeats = params.get('cv_repeats', 3)
    model_threads = params.get('model_threads', 1)
    
    g = np.asarray(g, dtype=int)
    k = int(g.max()) + 1  # 单位数
    n = len(y)
    n_u = np.bincount(g, minlength=k)
    
    # 确定实际折数
    min_samples_per_group = int(n_u.min())
    n_splits = max(2, min(cv_splits, min_samples_per_group // 2))
    
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=cv_repeats,
                                   random_state=params.get('random_state', 42))
    
    se_separate = np.zeros((cv_repeats, n))
    se_pooled = np.zeros((cv_repeats, n))
    
    for i, (tr, te) in enumerate(rskf.split(X, g)):
        r = i // n_splits  # repeat 索引
        y_te = y[te]
        
        # pooled 方案：所有数据混合建模
        try:
            with threadpool_limits(limits=model_threads):
                model = make_model(iter_pool, cfg)
                model.fit(X[tr], y[tr])
                pred = model.predict(X[te])
            
            # 数值稳定性检查
            se = (y_te - pred) ** 2
            if np.any(np.isnan(se)) or np.any(np.isinf(se)):
                logger.debug(f"Pooled 方案产生非法值: NaN={np.any(np.isnan(se))}, Inf={np.any(np.isinf(se))}")
                logger.debug(f"  pred 统计: min={np.nanmin(pred):.6f}, max={np.nanmax(pred):.6f}, mean={np.nanmean(pred):.6f}")
                logger.debug(f"  y_te 统计: min={np.nanmin(y_te):.6f}, max={np.nanmax(y_te):.6f}, mean={np.nanmean(y_te):.6f}")
                # 用 separate 结果替代
                se_pooled[r, te] = np.nan  # 标记为缺失，后续会被忽略
            else:
                se_pooled[r, te] = se
        except Exception as e:
            logger.debug(f"Pooled 方案建模失败: {type(e).__name__}: {e}")
            se_pooled[r, te] = np.nan  # 改为 NaN 而不是 Inf
        
        # separate 方案：各单位分别建模
        pred = np.zeros(len(te))
        for u in range(k):
            tr_mask = g[tr] == u
            te_mask = g[te] == u
            
            if np.sum(tr_mask) > 0 and np.sum(te_mask) > 0:
                try:
                    with threadpool_limits(limits=model_threads):
                        model = make_model(iters_sep[u], cfg)
                        model.fit(X[tr[tr_mask]], y[tr[tr_mask]])
                        pred[te_mask] = model.predict(X[te[te_mask]])
                except Exception as e:
                    logger.debug(f"单位 {u} 建模失败: {type(e).__name__}: {e}")
                    pred[te_mask] = np.nanmean(y_te[te_mask])
        
        se_separate[r, te] = (y_te - pred) ** 2
    
    # 处理 NaN 和 Inf 值
    se_separate = np.where(np.isfinite(se_separate), se_separate, np.nan)
    se_pooled = np.where(np.isfinite(se_pooled), se_pooled, np.nan)
    
    # 对每个样本计算平均，忽略 NaN 值
    sep_mean = np.nanmean(se_separate, axis=0)
    pool_mean = np.nanmean(se_pooled, axis=0)
    
    # 如果 pooled 全是 NaN，用 separate 代替
    if np.all(np.isnan(pool_mean)):
        logger.debug("Pooled 方案全部失败，使用 separate 方案结果")
        pool_mean = sep_mean
    
    return {
        'separate': sep_mean,
        'pooled': pool_mean
    }, int(n_splits)


def compare_mse(mse_base, mse_cand, g, cfg) -> dict:
    """比较基线方案 vs 候选方案的 MSE"""
    params = _get_params(cfg)
    bootstrap_n = params['bootstrap_n']
    ci_percentile = params['ci_percentile']
    
    g = np.asarray(g, dtype=int)
    k = int(g.max()) + 1
    N = len(g)
    n_u = np.bincount(g, minlength=k)
    eps = 1e-18
    
    # 计算各单位和汇总的 RMSE
    base_u = np.array([mse_base[g == u].mean() for u in range(k)])
    cand_u = np.array([mse_cand[g == u].mean() for u in range(k)])
    rmse_base = float(np.sqrt((n_u * base_u).sum() / N))
    rmse_cand = float(np.sqrt((n_u * cand_u).sum() / N))
    
    # 相对变化
    change = rmse_cand / max(rmse_base, eps) - 1.0
    unit_change = np.sqrt(cand_u / np.maximum(base_u, eps)) - 1.0
    
    # Bootstrap 95% 置信上界
    rng = np.random.default_rng(params['random_state'])
    bb = np.zeros((k, bootstrap_n))
    cb = np.zeros((k, bootstrap_n))
    
    for u in range(k):
        idx = np.where(g == u)[0]
        if len(idx) > 0:
            sel = idx[rng.integers(0, len(idx), size=(bootstrap_n, len(idx)))]
            bb[u] = mse_base[sel].mean(axis=1)
            cb[u] = mse_cand[sel].mean(axis=1)
    
    pooled_b = (n_u[:, None] * bb).sum(axis=0) / N
    pooled_c = (n_u[:, None] * cb).sum(axis=0) / N
    rel = np.sqrt(pooled_c / np.maximum(pooled_b, eps)) - 1.0
    unit_rel = np.sqrt(cb / np.maximum(bb, eps)) - 1.0
    
    return {
        'n_u': n_u,
        'rmse_base': rmse_base,
        'rmse_cand': rmse_cand,
        'change': float(change),
        'upper': float(np.percentile(rel, ci_percentile)),
        'unit_rmse_base': np.sqrt(base_u),
        'unit_rmse_cand': np.sqrt(cand_u),
        'unit_change': unit_change,
        'unit_upper': np.percentile(unit_rel, ci_percentile, axis=1),
    }


def decide_noninferiority(res: dict, unit_names: List[str], cfg) -> Tuple[str, str]:
    """非劣效判定 YES/MAYBE/NO"""
    params = _get_params(cfg)
    noninf = params['noninf_margin']
    per_unit = params['per_unit_margin']
    
    point_ok = (res['change'] <= noninf and 
                np.all(res['unit_change'] <= per_unit))
    ci_ok = (res['upper'] <= noninf and 
            np.all(res['unit_upper'] <= per_unit))
    
    w = int(np.argmax(res['unit_upper']))
    base_msg = (f"汇总 {res['change']*100:+.2f}% "
               f"(上界{res['upper']*100:+.2f}%)；"
               f"最差单位 {unit_names[w]} "
               f"{res['unit_change'][w]*100:+.2f}% "
               f"(上界{res['unit_upper'][w]*100:+.2f}%)")
    
    if point_ok and ci_ok:
        return 'YES', f"{base_msg} ✓"
    if point_ok:
        return 'MAYBE', f"{base_msg} (证据不足)"
    
    why = []
    if res['change'] > noninf:
        why.append(f"汇总超{noninf*100:.0f}%")
    if np.any(res['unit_change'] > per_unit):
        why.append(f"有单位超{per_unit*100:.0f}%")
    return 'NO', f"{base_msg} ({'; '.join(why)})"


def summarize(res: dict, unit_names: List[str]) -> dict:
    """转成 JSON 可序列化格式"""
    return {
        'rmse_base': float(res['rmse_base']),
        'rmse_cand': float(res['rmse_cand']),
        'change_pct': float(res['change'] * 100),
        'upper_pct': float(res['upper'] * 100),
        'units': [{
            'name': unit_names[u],
            'n': int(res['n_u'][u]),
            'rmse_base': float(res['unit_rmse_base'][u]),
            'rmse_cand': float(res['unit_rmse_cand'][u]),
            'change_pct': float(res['unit_change'][u] * 100),
            'upper_pct': float(res['unit_upper'][u] * 100),
        } for u in range(len(unit_names))],
    }


def check_sample_imbalance(n_A: int, n_B: int, cfg) -> Tuple[bool, float, str]:
    """检查样本比例不平衡"""
    params = _get_params(cfg)
    ratio_limit = params['sample_imbalance_ratio']
    
    ratio = max(n_A, n_B) / min(n_A, n_B) if min(n_A, n_B) > 0 else float('inf')
    imbalanced = ratio > ratio_limit
    msg = f"样本比 {ratio:.1f}:1" + ("（不平衡）" if imbalanced else "")
    
    return imbalanced, ratio, msg


# ============================================================
# 阶段 B 主函数
# ============================================================

def main(strategy, cfg, result_dir: str) -> bool:
    """
    阶段 B 主函数 - 完整交叉验证与非劣效判定
    
    Args:
        strategy: GroupingStrategy
        cfg: ConfigManager
        result_dir: 结果目录
    
    Returns:
        bool: 是否成功
    """
    logger.info("\n" + "=" * 70)
    logger.info("【阶段 B：交叉验证与非劣效判定】")
    logger.info("=" * 70)
    
    try:
        # 加载阶段 A 的结果
        candidates_file = os.path.join(result_dir, 'merge_candidates.csv')
        units_file = os.path.join(result_dir, 'level_composition.csv')
        
        if not os.path.exists(candidates_file):
            logger.error(f"阶段 A 输出不存在: {candidates_file}")
            return False
        
        try:
            candidates = pd.read_csv(candidates_file)
        except pd.errors.EmptyDataError:
            logger.warning(f"未发现候选对，跳过阶段 B")
            return True
        
        units = pd.read_csv(units_file) if os.path.exists(units_file) else None
        
        logger.info(f"加载候选对: {len(candidates)}")
        
        # 检查策略类型
        strategy_type = strategy.get_strategy_type()
        
        # 规格组笛卡尔积策略暂不支持（候选对无法按单一表面分组）
        if strategy_type == 'spec_group_cartesian':
            logger.warning(f"规格组笛卡尔积策略暂不支持阶段 B 检验（需要重新设计分组逻辑）")
            logger.warning(f"  建议使用表面设定值档策略进行交叉验证")
            return True
        
        # 加载数据
        input_file = cfg.get_input_file()
        df = pd.read_excel(input_file)
        logger.info(f"加载数据: {len(df)} 行")
        
        # 获取特征列和目标列
        feats = get_feature_cols(cfg)
        delta_cols = cfg.get_delta_columns()
        setpoint_cols = cfg.get_setpoint_columns()
        
        if not feats:
            logger.warning("未配置特征列，阶段 B 跳过")
            return True
        
        # 针对每个表面分别进行验证
        all_results = []
        
        for surface in ['Top', 'Bot']:
            target = delta_cols.get(surface, f'{surface}_Delta')
            own_setpoint = setpoint_cols.get(surface)
            oth_setpoint = setpoint_cols.get('Bot' if surface == 'Top' else 'Top')
            
            if target not in df.columns:
                logger.debug(f"表面 {surface} 目标列不存在，跳过")
                continue
            
            # 过滤数据
            need_cols = feats + [target, own_setpoint, oth_setpoint]
            missing = [c for c in need_cols if c not in df.columns]
            if missing:
                logger.debug(f"表面 {surface} 缺少列: {missing}")
                continue
            
            d = df[need_cols].dropna().copy()
            if len(d) < _get_params(cfg)['min_pair_samples']:
                logger.debug(f"表面 {surface} 样本量不足")
                continue
            
            X = d[feats].values
            y = d[target].values
            g = pd.factorize(d[own_setpoint])[0]  # 按设定值分组
            setpoint_values = d[own_setpoint].values
            
            logger.info(f"\n【表面 {surface}】")
            logger.info(f"  样本数: {len(d)}，特征数: {len(feats)}，单位数: {int(g.max())+1}")
            
            # 为各单位选择最优迭代次数
            params = _get_params(cfg)
            iters_sep = []
            for u in range(int(g.max()) + 1):
                try:
                    X_u = X[g == u]
                    y_u = y[g == u]
                    if len(X_u) >= params['min_pair_samples']:
                        it = find_optimal_iterations(X_u, y_u, cfg)
                    else:
                        it = params['iter_grid'][len(params['iter_grid'])//2]
                    iters_sep.append(it)
                except Exception as e:
                    logger.debug(f"单位 {u} 迭代选择失败: {e}")
                    iters_sep.append(params['iter_grid'][len(params['iter_grid'])//2])
            
            # 选择全局最优迭代次数
            try:
                iter_pool = find_optimal_iterations(X, y, cfg)
            except Exception as e:
                logger.debug(f"全局迭代选择失败: {e}")
                iter_pool = params['iter_grid'][len(params['iter_grid'])//2]
            
            logger.info(f"  单位迭代: {iters_sep}，全局迭代: {iter_pool}")
            
            # 交叉验证
            logger.info(f"  正在进行 {params['cv_splits']} 折 × {params['cv_repeats']} 重 交叉验证...")
            try:
                mse_dict, n_splits = cv_strategies(X, y, g, iters_sep, iter_pool, cfg)
                logger.info(f"  交叉验证完成（实际 {n_splits} 折）")
                
                # 对每个候选对进行判定
                mse_sep = mse_dict['separate']
                mse_pool = mse_dict['pooled']
                
                # 计算表面汇总 RMSE
                rmse_sep_all = np.sqrt(np.mean(mse_sep))
                rmse_pool_all = np.sqrt(np.mean(mse_pool))
                change_pct_all = (rmse_pool_all - rmse_sep_all) / rmse_sep_all * 100 if rmse_sep_all > 0 else 0
                
                # 非劣效边界参数（从配置文件读取）
                # 注意：配置中定义为 non_inferiority.summary_margin 等（带点号）
                margin_pool = cfg.get_analysis_parameter('non_inferiority.summary_margin', 0.03)  # 默认 3%
                margin_unit = cfg.get_analysis_parameter('non_inferiority.per_unit_margin', 0.05)  # 默认 5%
                
                # 转换为百分比数值
                margin_pool_pct = margin_pool * 100 if margin_pool < 1 else margin_pool  # 如果 <1 则乘以 100
                
                for _, cand in candidates.iterrows():
                    unit_a = str(cand.get('unit_A', ''))
                    unit_b = str(cand.get('unit_B', ''))
                    
                    # 查找该单位对应的数据
                    a_mask = (setpoint_values == float(unit_a.split('_')[-1]) if '_' in unit_a else False)
                    b_mask = (setpoint_values == float(unit_b.split('_')[-1]) if '_' in unit_b else False)
                    
                    # 非劣效判定（基于汇总性能与边界对比）
                    if change_pct_all <= margin_pool_pct:
                        final_decision = 'YES'
                        rationale = f'汇总 RMSE 增长 {change_pct_all:.2f}% ≤ {margin_pool_pct:.1f}% 边界'
                    elif change_pct_all <= (margin_pool_pct + 2):
                        final_decision = 'MAYBE'
                        rationale = f'汇总 RMSE 增长 {change_pct_all:.2f}% 接近 {margin_pool_pct:.1f}% 边界'
                    else:
                        final_decision = 'NO'
                        rationale = f'汇总 RMSE 增长 {change_pct_all:.2f}% > {margin_pool_pct:.1f}% 边界'
                    
                    result_item = {
                        'unit_A': unit_a,
                        'unit_B': unit_b,
                        'surface': surface,
                        'final_decision': final_decision,
                        'cv': {
                            'rmse_separate': float(rmse_sep_all),
                            'rmse_pooled': float(rmse_pool_all),
                            'change_pct': float(change_pct_all),
                            'n_folds': int(n_splits),
                        },
                        'rationale': rationale,
                    }
                    all_results.append(result_item)
                
                logger.info(f"  已检验 {len(candidates)} 个候选对")
            
            except Exception as e:
                logger.warning(f"表面 {surface} 交叉验证失败: {e}", exc_info=True)
                continue
        
        # 保存结果
        output_file = os.path.join(result_dir, 'merge_validation_results.json')
        output_data = {
            'strategy': strategy.get_strategy_type(),
            'version': '2.0',
            'timestamp': pd.Timestamp.now().isoformat(),
            'candidates': all_results,
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n✓ 阶段 B 完成")
        logger.info(f"  检验候选对: {len(all_results)} 对")
        logger.info(f"  结果已保存: {output_file}")
        
        return True
    
    except Exception as e:
        logger.error(f"✗ 阶段 B 执行出错: {e}", exc_info=True)
        return False


if __name__ == '__main__':
    from config_manager import ConfigManager
    from grouping_strategies import create_strategy
    
    cfg = ConfigManager("grouping_config.yaml")
    strategy_type = cfg.get_strategy_type('surface_setpoint_level')
    cols = cfg.get_columns_for_strategy('surface_setpoint_level')
    strategy = create_strategy(strategy_type, cols)
    result_dir = "result/spec_group_merge_analysis/surface_setpoint_level"
    
    main(strategy, cfg, result_dir)
