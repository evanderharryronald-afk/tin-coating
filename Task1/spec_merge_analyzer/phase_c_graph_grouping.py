"""
【阶段 C：分析单位分组方案（图聚类）】

输入：
  - level_composition.csv：分析单位信息
  - merge_candidates.csv：候选对（阶段 A）
  - merge_validation_results.json：交叉验证结果（阶段 B）

逻辑：
  1. 从阶段 B 结果构建图：节点=分析单位，边=YES判定的单位对
  2. 图分解：找所有极大团（聚类内两两通过检验）
  3. 贪心团划分：生成非重叠聚类覆盖所有节点
  4. 生成分组方案及建模指导
"""

import json
import os
import logging
from typing import Dict, List, Set, Optional, Tuple
import time

import numpy as np
import pandas as pd

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None

logger = logging.getLogger(__name__)


# ============================================================
# 数据加载函数
# ============================================================

def load_json(path: str) -> Optional[dict]:
    """加载 JSON 文件"""
    if not os.path.exists(path):
        logger.debug(f"JSON 文件不存在: {path}")
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"加载 JSON 失败 {path}: {e}")
        return None


def build_units_dict(units_df: pd.DataFrame) -> Dict[str, dict]:
    """构建单位字典: unit_key -> {单位信息}"""
    units_dict = {}
    if units_df is not None and len(units_df) > 0:
        for _, row in units_df.iterrows():
            unit_key = str(row['unit_key'])
            surface = row.get('surface', 'Both')
            n_samples = int(row.get('n_samples', 0))
            
            # 尝试从多个列读取设定值
            setpoint_val = None
            for col in ['setpoint_value', 'top_setpoint', 'bot_setpoint']:
                if col in row.index and pd.notna(row[col]):
                    setpoint_val = row[col]
                    break
            
            units_dict[unit_key] = {
                'unit_key': unit_key,
                'n_samples': n_samples,
                'surface': surface,
                'setpoint_value': setpoint_val,
            }
    return units_dict


def extract_surface_from_unit_key(unit_key: str) -> str:
    """从单位 key 提取表面名称（'Top' 或 'Bot'）"""
    unit_key_str = str(unit_key).upper()
    if unit_key_str.startswith('TOP'):
        return 'Top'
    elif unit_key_str.startswith('BOT'):
        return 'Bot'
    return 'Both'


def extract_yes_pairs_from_results(merge_results_json: Optional[dict],
                                    surface: Optional[str]) -> List[Tuple[str, str]]:
    """
    从 Phase B 结果提取所有 YES 判定的单位对
    
    Args:
        merge_results_json: Phase B 输出的 JSON
        surface: 表面名称（'Top'/'Bot'/None）
    
    Returns:
        [(unit_A, unit_B), ...] 的列表
    """
    yes_pairs = []
    if merge_results_json is None:
        return yes_pairs
    
    # 尝试不同的 JSON 格式
    candidates_list = []
    
    if isinstance(merge_results_json, list):
        candidates_list = merge_results_json
    elif isinstance(merge_results_json, dict):
        # 主要格式：'candidates' 字段包含所有候选对
        if 'candidates' in merge_results_json:
            candidates_list = merge_results_json['candidates']
        # 备选格式 1: 顶层 'surfaces' 下按表面组织
        elif 'surfaces' in merge_results_json:
            surface_data = merge_results_json['surfaces'].get(surface, [])
            if isinstance(surface_data, list):
                candidates_list = surface_data
        # 备选格式 2: 直接 'pairs' 字段
        elif 'pairs' in merge_results_json:
            candidates_list = merge_results_json['pairs']
    
    # 筛选 YES 的对
    for item in candidates_list:
        if isinstance(item, dict):
            final_decision = item.get('final_decision', item.get('decision', ''))
            if final_decision.upper() == 'YES':
                unit_a = str(item.get('unit_A', item.get('level_A', '')))
                unit_b = str(item.get('unit_B', item.get('level_B', '')))
                if unit_a and unit_b:
                    yes_pairs.append((unit_a, unit_b))
    
    return yes_pairs


# ============================================================
# 图与聚类函数
# ============================================================

def build_graph_from_units_and_pairs(units_dict: Dict[str, dict], 
                                      yes_pairs: List[Tuple[str, str]],
                                      surface: Optional[str] = None) -> Optional[nx.Graph]:
    """
    从单位列表和 YES 对构建图
    
    节点：符合表面条件的单位
    边：YES 对
    """
    if not NETWORKX_AVAILABLE:
        logger.warning("networkx 未安装，图聚类跳过")
        return None
    
    G = nx.Graph()
    
    # 添加节点
    for unit_key, info in units_dict.items():
        if surface is None or info['surface'] == surface or info['surface'] == 'Both':
            G.add_node(unit_key, **info)
    
    # 添加边
    for unit_a, unit_b in yes_pairs:
        if unit_a in G.nodes() and unit_b in G.nodes():
            G.add_edge(unit_a, unit_b)
    
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    logger.info(f"  构建图: {n_nodes} 个节点, {n_edges} 条边")
    
    return G


def find_maximal_cliques(G: nx.Graph) -> List[Set[str]]:
    """找所有极大团（maximal cliques）"""
    if G is None or G.number_of_nodes() == 0:
        return []
    
    try:
        cliques = list(nx.find_cliques(G))
        return [set(c) for c in cliques]
    except Exception as e:
        logger.warning(f"计算最大团失败: {e}")
        return []


def greedy_clique_cover(G: nx.Graph) -> List[Set[str]]:
    """
    贪心团覆盖：找一组不相交的团覆盖所有节点
    
    策略：重复找最大团，移除其节点，直到无节点剩余
    """
    if G is None or G.number_of_nodes() == 0:
        return []
    
    remaining = set(G.nodes())
    clusters = []
    iterations = 0
    max_iter = G.number_of_nodes()  # 防止无限循环
    
    while remaining and iterations < max_iter:
        iterations += 1
        sub_G = G.subgraph(remaining)
        
        # 找该子图的所有最大团
        cliques = find_maximal_cliques(sub_G)
        
        if not cliques:
            # 无团存在，孤立节点自成一类
            clusters.extend([{n} for n in remaining])
            break
        
        # 选择最大的团
        best_clique = max(cliques, key=len)
        clusters.append(best_clique)
        remaining -= best_clique
    
    logger.info(f"  贪心团覆盖: {len(clusters)} 个聚类")
    return clusters


def describe_cluster(members: Set[str], units_dict: Dict[str, dict], 
                    cfg, surface: str) -> Dict:
    """描述一个聚类"""
    members_sorted = sorted(members, 
                           key=lambda m: float(units_dict.get(m, {}).get('setpoint_value', 0)))
    
    # 计算总样本量
    total_samples = sum(units_dict.get(m, {}).get('n_samples', 0) for m in members_sorted)
    
    # 检查是否达到正式样本量阈值
    formal_min = cfg.get_analysis_parameter('min_formal_samples', 200)
    meets_formal = total_samples >= formal_min
    
    return {
        'members': list(members_sorted),
        'count': len(members_sorted),
        'total_samples': int(total_samples),
        'meets_formal_threshold': bool(meets_formal),
        'sample_density': total_samples / len(members_sorted) if members_sorted else 0,
    }


def generate_grouping_recommendation(clusters_by_surface: Dict[str, List[Set[str]]],
                                     units_dict: Dict[str, dict],
                                     cfg) -> Dict:
    """生成最终的分组建议"""
    recommendation = {
        'version': '3.0',
        'methodology': '非劣效判定聚类（图团划分）',
        'surfaces': {},
        'summary': {
            'total_surfaces': len(clusters_by_surface),
            'total_models': 0,
            'merged_clusters': 0,
            'independent_units': 0,
        }
    }
    
    for surface, clusters in clusters_by_surface.items():
        merged_clusters = []
        independent_units = []
        
        for cluster in clusters:
            if len(cluster) > 1:
                desc = describe_cluster(cluster, units_dict, cfg, surface)
                merged_clusters.append(desc)
            else:
                desc = describe_cluster(cluster, units_dict, cfg, surface)
                independent_units.append(desc)
        
        recommendation['surfaces'][surface] = {
            'surface': surface,
            'merged_clusters': merged_clusters,
            'independent_units': independent_units,
            'total_models': len(merged_clusters) + len(independent_units),
        }
        
        recommendation['summary']['total_models'] += len(merged_clusters) + len(independent_units)
        recommendation['summary']['merged_clusters'] += len(merged_clusters)
        recommendation['summary']['independent_units'] += len(independent_units)
    
    return recommendation


# ============================================================
# 主函数
# ============================================================

def main(strategy, cfg, result_dir: str) -> bool:
    """
    阶段 C 主函数：图聚类与分组建议
    
    Args:
        strategy: GroupingStrategy
        cfg: ConfigManager
        result_dir: 结果目录
    
    Returns:
        bool: 是否成功
    """
    logger.info("\n" + "=" * 70)
    logger.info("【阶段 C：分析单位分组方案】")
    logger.info("=" * 70)
    
    try:
        # ============ 第一步：加载输入数据 ============
        units_file = os.path.join(result_dir, 'level_composition.csv')
        candidates_file = os.path.join(result_dir, 'merge_candidates.csv')
        merge_results_file = os.path.join(result_dir, 'merge_validation_results.json')
        
        # 加载单位信息
        if not os.path.exists(units_file):
            logger.error(f"单位文件不存在: {units_file}")
            return False
        
        units_df = pd.read_csv(units_file)
        logger.info(f"✓ 加载分析单位: {len(units_df)} 个")
        
        # 加载候选对信息
        if os.path.exists(candidates_file):
            try:
                candidates_df = pd.read_csv(candidates_file)
                n_candidates = len(candidates_df) if len(candidates_df) > 0 else 0
            except pd.errors.EmptyDataError:
                candidates_df = pd.DataFrame()
                n_candidates = 0
        else:
            candidates_df = pd.DataFrame()
            n_candidates = 0
        logger.info(f"✓ 加载候选对: {n_candidates} 对")
        
        # 加载 Phase B 的检验结果
        merge_results_json = load_json(merge_results_file)
        if merge_results_json is None:
            logger.warning("未发现 Phase B 结果，使用默认分组（每个单位独立）")
        
        # ============ 第二步：构建单位字典 ============
        units_dict = build_units_dict(units_df)
        
        # ============ 第三步：按表面分别进行聚类 ============
        clusters_by_surface = {}
        
        # 从单位 key 提取表面列表
        surfaces_in_units = set()
        for unit_key in units_dict.keys():
            surf = extract_surface_from_unit_key(unit_key)
            surfaces_in_units.add(surf)
        
        surfaces = sorted(surfaces_in_units)
        logger.info(f"检测到表面: {surfaces}")
        
        for surface in surfaces:
            logger.info(f"\n【{surface} 表面】")
            
            # 筛选该表面的单位（按单位 key 前缀）
            surface_units = {k: v for k, v in units_dict.items() 
                            if extract_surface_from_unit_key(k) == surface}
            logger.info(f"  单位数: {len(surface_units)}")
            
            if len(surface_units) == 0:
                logger.info(f"  无单位")
                clusters_by_surface[surface] = []
                continue
            
            # 提取 YES 对
            yes_pairs = extract_yes_pairs_from_results(merge_results_json, surface)
            logger.info(f"  YES 对: {len(yes_pairs)}")
            
            # 构建图并进行团划分
            if NETWORKX_AVAILABLE and len(yes_pairs) > 0:
                G = build_graph_from_units_and_pairs(surface_units, yes_pairs, surface)
                
                if G and G.number_of_nodes() > 0:
                    # 团覆盖
                    clusters = greedy_clique_cover(G)
                    logger.info(f"  聚类方案: {len(clusters)} 个")
                    clusters_by_surface[surface] = clusters
                else:
                    logger.info(f"  图为空，单位独立建模")
                    clusters_by_surface[surface] = [{k} for k in surface_units.keys()]
            else:
                logger.info(f"  无 YES 对或 networkx 不可用，单位独立建模")
                clusters_by_surface[surface] = [{k} for k in surface_units.keys()]
        
        # ============ 第四步：生成分组建议 ============
        recommendation = generate_grouping_recommendation(clusters_by_surface, units_dict, cfg)
        
        # ============ 第五步：导出结果 ============
        output_file = os.path.join(result_dir, 'recommended_grouping.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(recommendation, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n✓ 分组方案已导出: {output_file}")
        
        # ============ 第六步：打印摘要 ============
        logger.info("\n" + "=" * 70)
        logger.info("【阶段 C 完成】")
        logger.info("=" * 70)
        
        summary = recommendation['summary']
        logger.info(f"总建模单位: {summary['total_models']}")
        logger.info(f"  · 合并聚类: {summary['merged_clusters']}")
        logger.info(f"  · 独立单位: {summary['independent_units']}")
        
        logger.info("\n【分组细节】")
        for surface, surf_info in recommendation['surfaces'].items():
            logger.info(f"\n  {surface} 表面:")
            logger.info(f"    · 合并聚类: {len(surf_info['merged_clusters'])}")
            for i, cluster in enumerate(surf_info['merged_clusters'], 1):
                logger.info(f"      {i}. {cluster['members']} ({cluster['total_samples']} 样本)")
            
            if surf_info['independent_units']:
                logger.info(f"    · 独立单位: {len(surf_info['independent_units'])}")
                for unit_info in surf_info['independent_units']:
                    logger.info(f"      - {unit_info['members'][0]} ({unit_info['total_samples']} 样本)")
        
        logger.info("\n" + "=" * 70 + "\n")
        
        return True
    
    except Exception as e:
        logger.error(f"✗ 阶段 C 执行出错: {e}", exc_info=True)
        return False


if __name__ == '__main__':
    """
    CLI 测试入口（可选）
    """
    logging.basicConfig(level=logging.INFO)
    
    # 需要手动配置以下参数
    from config_manager import ConfigManager
    from grouping_strategies import create_strategy
    
    cfg = ConfigManager("grouping_config.yaml")
    strategy_type = cfg.get_strategy_type('surface_setpoint_level')
    cols = cfg.get_columns_for_strategy('surface_setpoint_level')
    strategy = create_strategy(strategy_type, cols['setpoint'])
    
    result_dir = "result/spec_group_merge_analysis/surface_setpoint_level"
    
    success = main(strategy, cfg, result_dir)
    print(f"\nPhase C {'成功' if success else '失败'}")
