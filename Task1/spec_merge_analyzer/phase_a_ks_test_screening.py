"""
【阶段 A：按分组策略的合并初筛】

通用逻辑：
  1. 使用 GroupingStrategy 构建分析单位表
  2. 对分析单位两两做 KS 检验（比较目标变量的分布）
  3. 计算分析单位间的距离（取决于策略定义）
  4. 输出合并候选对

支持多种分组策略：
  - surface_setpoint_level: 按表面设定值档分组
  - spec_group_cartesian: Top×Bot 规格组笛卡尔积
  - 其他自定义策略
"""

import itertools
import os
import logging

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)


def load_featured_data(cfg) -> pd.DataFrame:
    """从配置加载特征工程数据"""
    logger.info("[步骤 1] 加载特征工程数据...")
    
    input_file = cfg.get_input_file()
    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"未找到特征工程数据: {input_file}\n"
            f"请先运行 new_data_pipeline.py 或 analyse_data_final.py"
        )
    
    df = pd.read_excel(input_file)
    logger.info(f"✓ 数据加载: {len(df)} 行, {df.shape[1]} 列")
    logger.info(f"  路径: {input_file}")
    return df


def check_columns(df, cfg) -> None:
    """检查必要的列是否存在"""
    setpoint_cols = cfg.get_setpoint_columns()
    delta_cols = cfg.get_delta_columns()
    
    need = list(setpoint_cols.values()) + list(delta_cols.values())
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"缺少必要字段: {missing}")


def build_units_with_strategy(df, strategy, cfg) -> pd.DataFrame:
    """使用策略构建分析单位表"""
    logger.info("[步骤 2] 构建分析单位...")
    
    units = strategy.build_analysis_units(df)
    
    # 记录统计信息
    logger.info(f"✓ 共 {len(units)} 个分析单位")
    total_samples = units['n_samples'].sum()
    logger.info(f"  覆盖 {total_samples}/{len(df)} 行（{total_samples/len(df)*100:.1f}%）")
    
    # 按是否达到最小样本量统计
    min_samples = cfg.get_analysis_parameter('min_level_samples')
    eligible = units[units['n_samples'] >= min_samples]
    logger.info(f"  达标（≥{min_samples}）: {len(eligible)} 个")
    
    return units


def spec_distances(value_A: float, value_B: float, cfg) -> tuple:
    """计算两个值之间的绝对差和相对差"""
    abs_limit = cfg.get_analysis_parameter('spec_distance.absolute_limit')
    rel_limit = cfg.get_analysis_parameter('spec_distance.relative_limit')
    
    abs_d = round(abs(value_A - value_B), 6)
    rel_d = round(abs_d / min(value_A, value_B), 6) if min(value_A, value_B) > 0 else float('inf')
    
    within = bool(abs_d <= abs_limit and rel_d <= rel_limit)
    
    return abs_d, rel_d, within, abs_limit, rel_limit


def perform_ks_tests(df, units: pd.DataFrame, strategy, cfg) -> list:
    """
    对分析单位两两做 KS 检验。
    
    对于表面设定值档策略：
      - 按表面分别进行，比较同一表面不同档的 Delta 分布
    对于规格组笛卡尔积策略：
      - 比较不同规格组的 Top/Bot Delta 分布（或混合）
    """
    logger.info("[步骤 3] KS 检验单位间分布差异...")
    
    min_samples = cfg.get_analysis_parameter('min_level_samples')
    ks_threshold = cfg.get_analysis_parameter('ks_test.p_threshold')
    setpoint_cols = strategy.get_setpoint_columns()
    delta_cols = cfg.get_delta_columns()
    
    # 筛选达标的单位
    eligible_units = units[units['n_samples'] >= min_samples].copy()
    
    candidates = []
    
    # 按策略类型分别处理
    strategy_type = strategy.get_strategy_type()
    
    if strategy_type == 'surface_setpoint_level':
        # 表面设定值档策略：按表面分别处理
        for surface in eligible_units['surface'].unique():
            surface_units = eligible_units[eligible_units['surface'] == surface].sort_values('unit_key')
            logger.debug(f"  处理 {surface} 表面: {len(surface_units)} 个单位")
            
            setpoint_col = setpoint_cols.get(surface)
            delta_col = delta_cols.get(surface)
            
            if setpoint_col is None or delta_col is None:
                logger.warning(f"  跳过 {surface} 表面（缺少列定义）")
                continue
            
            # 预加载每个单位的 Delta 向量
            deltas = {}
            for _, unit in surface_units.iterrows():
                # 对于表面设定值档策略，unit_key 形如 "Top2.799"
                if surface in unit['unit_key']:
                    # 提取设定值数值部分
                    value = unit.get('setpoint_value')
                    if pd.notna(value):
                        delta_vec = df.loc[df[setpoint_col] == value, delta_col].dropna()
                        deltas[unit['unit_key']] = delta_vec
            
            # 两两 KS 检验
            for (_, uA), (_, uB) in itertools.combinations(surface_units.iterrows(), 2):
                dA = deltas.get(uA['unit_key'])
                dB = deltas.get(uB['unit_key'])
                
                if dA is None or dB is None or len(dA) < 10 or len(dB) < 10:
                    continue
                
                stat, p = ks_2samp(dA, dB)
                
                # 计算距离
                val_A = uA.get('setpoint_value')
                val_B = uB.get('setpoint_value')
                abs_d, rel_d, within, abs_limit, rel_limit = spec_distances(val_A, val_B, cfg)
                
                candidates.append({
                    'surface': surface,
                    'unit_A': uA['unit_key'],
                    'unit_B': uB['unit_key'],
                    'value_A': val_A,
                    'value_B': val_B,
                    'p_value': p,
                    'stat': stat,
                    'ks_pass': bool(p > ks_threshold),
                    'size_A': int(len(dA)),
                    'size_B': int(len(dB)),
                    'merged_size': int(len(dA) + len(dB)),
                    'mean_A': dA.mean(),
                    'mean_B': dB.mean(),
                    'mean_diff': abs(dA.mean() - dB.mean()),
                    'std_A': dA.std(),
                    'std_B': dB.std(),
                    'spec_distance': abs_d,
                    'rel_distance': rel_d,
                    'abs_limit': abs_limit,
                    'rel_limit': rel_limit,
                    'within_limit': within,
                })
    else:
        # 规格组笛卡尔积策略：直接两两比较
        # 对于每对单位，比较两个 Top×Bot 规格组合的合并 Delta 分布
        logger.debug(f"  处理规格组策略: {len(eligible_units)} 个单位")
        
        top_setpoint_col = setpoint_cols.get('Top')
        bot_setpoint_col = setpoint_cols.get('Bot')
        delta_col = delta_cols.get('Top')  # 使用 Top Delta 作为代表
        
        if top_setpoint_col is None or bot_setpoint_col is None or delta_col is None:
            logger.warning("  规格组策略缺少列定义，跳过 KS 检验")
        else:
            # 预加载每个单位的 Delta 向量（按该单位的 Top/Bot 值过滤）
            deltas = {}
            for _, unit in eligible_units.iterrows():
                top_val = unit.get('top_setpoint')
                bot_val = unit.get('bot_setpoint')
                
                if pd.notna(top_val) and pd.notna(bot_val):
                    # 只取该规格组合的样本
                    mask = (df[top_setpoint_col] == top_val) & (df[bot_setpoint_col] == bot_val)
                    delta_vec = df.loc[mask, delta_col].dropna()
                    if len(delta_vec) > 0:
                        deltas[unit['unit_key']] = delta_vec
            
            # 两两 KS 检验
            for (_, uA), (_, uB) in itertools.combinations(eligible_units.iterrows(), 2):
                dA = deltas.get(uA['unit_key'])
                dB = deltas.get(uB['unit_key'])
                
                if dA is None or dB is None or len(dA) < 10 or len(dB) < 10:
                    continue
                
                stat, p = ks_2samp(dA, dB)
                
                # 计算规格距离（同时考虑 Top 和 Bot 的差异）
                top_diff = abs(uA.get('top_setpoint') - uB.get('top_setpoint'))
                bot_diff = abs(uA.get('bot_setpoint') - uB.get('bot_setpoint'))
                abs_d = max(top_diff, bot_diff)  # 取最大差异
                
                # 相对差异基于较小值
                min_top = min(uA.get('top_setpoint', 1), uB.get('top_setpoint', 1))
                rel_d = abs_d / min_top if min_top > 0 else float('inf')
                
                abs_limit = cfg.get_analysis_parameter('spec_distance.absolute_limit')
                rel_limit = cfg.get_analysis_parameter('spec_distance.relative_limit')
                within = bool(abs_d <= abs_limit and rel_d <= rel_limit)
                
                candidates.append({
                    'surface': 'Both',
                    'unit_A': uA['unit_key'],
                    'unit_B': uB['unit_key'],
                    'value_A': f"Top{uA.get('top_setpoint'):.3f}_Bot{uA.get('bot_setpoint'):.3f}",
                    'value_B': f"Top{uB.get('top_setpoint'):.3f}_Bot{uB.get('bot_setpoint'):.3f}",
                    'p_value': p,
                    'stat': stat,
                    'ks_pass': bool(p > ks_threshold),
                    'size_A': int(len(dA)),
                    'size_B': int(len(dB)),
                    'merged_size': int(len(dA) + len(dB)),
                    'mean_A': dA.mean(),
                    'mean_B': dB.mean(),
                    'mean_diff': abs(dA.mean() - dB.mean()),
                    'std_A': dA.std(),
                    'std_B': dB.std(),
                    'spec_distance': abs_d,
                    'rel_distance': rel_d,
                    'abs_limit': abs_limit,
                    'rel_limit': rel_limit,
                    'within_limit': within,
                })
    
    candidates.sort(key=lambda x: (not x['within_limit'], not x['ks_pass'], x['spec_distance']))
    logger.info(f"✓ 共生成 {len(candidates)} 个候选对")
    
    return candidates


def print_ks_results(candidates: list, cfg):
    """打印 KS 检验结果摘要"""
    abs_limit = cfg.get_analysis_parameter('spec_distance.absolute_limit')
    rel_limit = cfg.get_analysis_parameter('spec_distance.relative_limit')
    
    logger.info("\n" + "=" * 100)
    logger.info("【KS 检验结果 - 单位间合并候选对】")
    logger.info("=" * 100)
    
    if not candidates:
        logger.info("\n✗ 未发现候选对")
        return
    
    logger.info(f"\n总计 {len(candidates)} 个候选对")
    logger.info(f"  规格距离限制: 绝对差 ≤ {abs_limit} g/m² 且 相对差 ≤ {rel_limit:.0%}")
    
    # 按表面统计
    for surface in set(c.get('surface') for c in candidates if c.get('surface')):
        s = [c for c in candidates if c.get('surface') == surface]
        w = [c for c in s if c['within_limit']]
        kp = [c for c in w if c['ks_pass']]
        logger.info(f"  · {surface}: 共 {len(s)} 对，限内 {len(w)} 对"
                   f"（其中 KS 通过 {len(kp)} 对），超限 {len(s) - len(w)} 对")
    
    within = [c for c in candidates if c['within_limit']]
    logger.info("\n" + "-" * 100)
    logger.info(f"【限内候选对（将进入阶段 B 检验）】共 {len(within)} 对")
    logger.info("-" * 100)
    
    for i, c in enumerate(within, 1):
        mark = "KS ✓" if c['ks_pass'] else "KS ✗"
        logger.info(f"{i:>3}. {c.get('surface', 'N/A')}: {c['unit_A']:<15} + {c['unit_B']:<15} "
                   f"绝对差 {c['spec_distance']:.3f} / 相对差 {c['rel_distance']:.0%}  "
                   f"样本 {c['size_A']}+{c['size_B']}  p={c['p_value']:.4f} {mark}")


def export_outputs(candidates: list, units: pd.DataFrame, result_dir: str):
    """导出候选对和单位构成表"""
    os.makedirs(result_dir, exist_ok=True)
    
    candidates_file = os.path.join(result_dir, 'merge_candidates.csv')
    units_file = os.path.join(result_dir, 'level_composition.csv')
    
    pd.DataFrame(candidates).to_csv(candidates_file, index=False, encoding='utf-8')
    units.to_csv(units_file, index=False, encoding='utf-8-sig')
    
    logger.info(f"✓ 候选对已导出: {candidates_file}")
    logger.info(f"✓ 分析单位已导出: {units_file}")


def main(strategy, cfg, result_dir: str):
    """
    阶段 A 主函数
    
    Args:
        strategy: GroupingStrategy 实例
        cfg: ConfigManager 实例
        result_dir: 结果输出目录
    """
    logger.info("\n" + "=" * 70)
    logger.info("【阶段 A：分析单位筛选与 KS 检验】")
    logger.info("=" * 70)
    
    try:
        # 加载数据
        df = load_featured_data(cfg)
        check_columns(df, cfg)
        
        # 构建分析单位
        units = build_units_with_strategy(df, strategy, cfg)
        
        # KS 检验
        candidates = perform_ks_tests(df, units, strategy, cfg)
        
        # 打印结果
        print_ks_results(candidates, cfg)
        
        # 导出输出
        export_outputs(candidates, units, result_dir)
        
        logger.info("\n" + "=" * 70)
        logger.info("【阶段 A 完成】")
        logger.info("=" * 70)
        logger.info("\n后续步骤: 阶段 B（交叉验证与非劣效判定）\n")
        
        return True
    
    except Exception as e:
        logger.error(f"\n✗ 阶段 A 执行出错: {e}", exc_info=True)
        return False


if __name__ == '__main__':
    # 独立运行测试（用于调试）
    from config_manager import ConfigManager
    from grouping_strategies import create_strategy
    
    cfg = ConfigManager("grouping_config.yaml")
    strategy_type = cfg.get_strategy_type('surface_setpoint_level')
    cols = cfg.get_columns_for_strategy('surface_setpoint_level')
    strategy = create_strategy(strategy_type, cols['setpoint'])
    result_dir = "result/spec_group_merge_analysis"
    
    main(strategy, cfg, result_dir)
