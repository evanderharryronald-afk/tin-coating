"""
【阶段 C：规格组重新分组（最终版本 - 正确的独立表面聚类）】

【核心改进】：
1. 分表面聚类：Top 和 Bot 完全独立处理
2. 规格距离硬性约束：聚类内只连接距离 < 1.0 的对
3. 独立输出：Top 聚类和 Bot 聚类分开，互不影响
4. 清晰的建模指导：每个表面一份清晰的聚类清单

【设计理念】：
建模时对 Top 和 Bot 分别建模，所以合并决策也应该分别进行。
例如：Top 表面可以是"A+C+D 共享一个模型"，
      Bot 表面可以是"A+B+E 共享一个模型"，
这两个决策完全独立。
"""

import json
import pandas as pd
import networkx as nx
from typing import List, Set, Dict, Tuple
import os


def load_chow_results(results_json: str = "result/spec_group_merge_analysis/chow_test_results_v3.json") -> list:
    """加载 Phase B 的检验结果"""
    if not os.path.exists(results_json):
        raise FileNotFoundError(f"未找到检验结果文件: {results_json}")
    
    with open(results_json, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print(f"\n✓ 加载 Chow 检验结果: {len(results)} 对")
    return results


def build_merge_graph(results: list, surface: str, max_spec_distance: float = 1.0) -> Tuple[nx.Graph, Dict]:
    """
    构建合并关系图
    
    规则：
    1. 只选择该表面的可合并对
    2. 规格距离 < 1.0 的对才能形成边
    3. 返回图和诊断信息
    """
    G = nx.Graph()
    diagnostics = {
        'surface': surface,
        'can_merge_pairs': 0,
        'valid_pairs': 0,
        'excluded_by_distance': 0,
        'pairs_detail': []
    }
    
    # 筛选该表面的可合并对
    can_merge_pairs = [r for r in results if r['final_decision'] == 'YES' and r['surface'] == surface]
    diagnostics['can_merge_pairs'] = len(can_merge_pairs)
    
    # 规格距离约束
    valid_pairs = [p for p in can_merge_pairs if p.get('spec_distance', 2.0) < max_spec_distance]
    diagnostics['valid_pairs'] = len(valid_pairs)
    diagnostics['excluded_by_distance'] = len(can_merge_pairs) - len(valid_pairs)
    
    for pair in valid_pairs:
        group_A = pair['group_A']
        group_B = pair['group_B']
        chow_p = pair.get('chow_test', {}).get('p_value', 0)
        
        G.add_edge(group_A, group_B, weight=chow_p)
        diagnostics['pairs_detail'].append({
            'group_A': group_A,
            'group_B': group_B,
            'spec_distance': pair.get('spec_distance', 0),
            'chow_p_value': chow_p
        })
    
    return G, diagnostics


def find_connected_components(G: nx.Graph) -> Tuple[List[Set], Dict]:
    """
    找到连通分量
    
    返回：(聚类列表, 诊断信息)
    """
    components = list(nx.connected_components(G))
    
    diagnostics = {
        'total_components': len(components),
        'merged_clusters': [],      # 两个或以上规格组的聚类
        'single_independent': 0      # 单独的规格组
    }
    
    for component in components:
        groups = sorted(list(component))
        if len(groups) > 1:
            diagnostics['merged_clusters'].append({
                'groups': groups,
                'size': len(groups)
            })
        else:
            diagnostics['single_independent'] += 1
    
    return components, diagnostics


def collect_independent_groups(all_groups: Set, merged_components: List[Set]) -> Set:
    """
    收集无法合并的独立规格组
    """
    merged_groups = set()
    for comp in merged_components:
        merged_groups.update(comp)
    
    return all_groups - merged_groups


def print_grouping_results(all_groups: Set, top_components: List[Set], top_diag: Dict,
                               bot_components: List[Set], bot_diag: Dict):
    """打印分组结果（两个表面独立显示）"""
    print("\n" + "="*80)
    print("【规格组重新分组方案 - 独立表面聚类】")
    print("="*80)
    
    print("\n【Top 表面聚类结果】")
    print(f"  总聚类数: {len(top_components)}")
    print(f"  - 合并聚类（≥2个规格组）: {len(top_diag['merged_clusters'])}")
    print(f"  - 独立规格组: {top_diag['single_independent']}")
    
    if top_diag['merged_clusters']:
        print("\n  合并聚类：")
        for i, cluster_info in enumerate(top_diag['merged_clusters'], 1):
            print(f"    聚类 {i}（{cluster_info['size']} 个规格组）: {cluster_info['groups']}")
    
    top_independent = collect_independent_groups(all_groups, top_components)
    if top_independent:
        print(f"\n  独立建模（Top）: {sorted(list(top_independent))}")
    
    print("\n" + "-"*80)
    print("\n【Bot 表面聚类结果】")
    print(f"  总聚类数: {len(bot_components)}")
    print(f"  - 合并聚类（≥2个规格组）: {len(bot_diag['merged_clusters'])}")
    print(f"  - 独立规格组: {bot_diag['single_independent']}")
    
    if bot_diag['merged_clusters']:
        print("\n  合并聚类：")
        for i, cluster_info in enumerate(bot_diag['merged_clusters'], 1):
            print(f"    聚类 {i}（{cluster_info['size']} 个规格组）: {cluster_info['groups']}")
    
    bot_independent = collect_independent_groups(all_groups, bot_components)
    if bot_independent:
        print(f"\n  独立建模（Bot）: {sorted(list(bot_independent))}")
    
    print("\n" + "="*80)
    print("【建模指导】")
    print("="*80)
    print("\n说明：Top 和 Bot 表面的聚类结果完全独立。")
    print("一个规格组可能在 Top 表面与其他组合并，但在 Bot 表面独立建模。")
    print("\n示例：")
    print("  - 某个规格组在 Top 表面可能属于聚类 1（与其他2-3个规格组合并）")
    print("  - 同一个规格组在 Bot 表面可能独立建模（无法与其他组合并）")
    print("  这两个决策互不影响。")


def export_grouping_config(all_groups: Set, top_components: List[Set], bot_components: List[Set],
                               output_path: str = "result/spec_group_merge_analysis/recommended_grouping.json"):
    """导出分组配置（两个表面独立）"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 提取 Top 聚类（只输出合并的，独立规格组隐含）
    top_clusters = []
    for comp in top_components:
        if len(comp) > 1:
            top_clusters.append(sorted(list(comp)))
    
    # 提取 Bot 聚类（只输出合并的，独立规格组隐含）
    bot_clusters = []
    for comp in bot_components:
        if len(comp) > 1:
            bot_clusters.append(sorted(list(comp)))
    
    # 收集两个表面的独立规格组
    top_merged_groups = set()
    for comp in top_components:
        if len(comp) > 1:
            top_merged_groups.update(comp)
    top_independent = sorted(list(all_groups - top_merged_groups))
    
    bot_merged_groups = set()
    for comp in bot_components:
        if len(comp) > 1:
            bot_merged_groups.update(comp)
    bot_independent = sorted(list(all_groups - bot_merged_groups))
    
    config = {
        'version': '1.0',
        'description': '基于独立表面聚类的规格组合并方案',
        'methodology': '分表面聚类 + 规格距离约束（<1.0）+ 独立决策',
        
        'top_surface': {
            'description': 'Top表面的聚类结果',
            'merged_clusters': top_clusters,
            'independent_groups': top_independent,
            'total_groups': len(top_clusters) + len(top_independent)
        },
        
        'bot_surface': {
            'description': 'Bot表面的聚类结果',
            'merged_clusters': bot_clusters,
            'independent_groups': bot_independent,
            'total_groups': len(bot_clusters) + len(bot_independent)
        },
        
        'modeling_guidance': {
            'explanation': 'Top 和 Bot 表面的聚类完全独立，在建立模型时分别使用这两个聚类方案。',
            'example': '某规格组可能在Top表面与A、C合并建立一个模型，同时在Bot表面独立建模。'
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 分组方案已保存至: {output_path}")


def main():
    """主流程"""
    print("\n" + "="*80)
    print("【阶段 C：正确的独立表面聚类】")
    print("="*80)
    print("\n【核心改进】：")
    print("  ✓ 分表面聚类：Top 和 Bot 完全独立处理")
    print("  ✓ 规格距离约束：聚类内只连接距离 < 1.0 的对")
    print("  ✓ 独立输出：两个表面各自一份聚类清单")
    print("  ✓ 清晰的建模指导：说明两表面决策的独立性")
    
    try:
        # 加载 Phase B 的结果
        results = load_chow_results()
        
        # 获取所有规格组
        all_groups_set = set()
        for r in results:
            all_groups_set.add(r['group_A'])
            all_groups_set.add(r['group_B'])
        
        # 分表面构建图
        print("\n[构建合并关系图]")
        top_graph, top_diag = build_merge_graph(results, 'Top', max_spec_distance=1.0)
        bot_graph, bot_diag = build_merge_graph(results, 'Bot', max_spec_distance=1.0)
        
        print(f"\n  Top 表面:")
        print(f"    可合并对: {top_diag['can_merge_pairs']} 对")
        print(f"    规格距离 < 1.0 的对: {top_diag['valid_pairs']} 对")
        if top_diag['excluded_by_distance'] > 0:
            print(f"    被规格距离约束排除: {top_diag['excluded_by_distance']} 对")
        
        print(f"\n  Bot 表面:")
        print(f"    可合并对: {bot_diag['can_merge_pairs']} 对")
        print(f"    规格距离 < 1.0 的对: {bot_diag['valid_pairs']} 对")
        if bot_diag['excluded_by_distance'] > 0:
            print(f"    被规格距离约束排除: {bot_diag['excluded_by_distance']} 对")
        
        # 找连通分量
        print("\n[寻找连通分量]")
        top_components, top_comp_diag = find_connected_components(top_graph)
        bot_components, bot_comp_diag = find_connected_components(bot_graph)
        
        print(f"\n  Top 表面：{len(top_components)} 个连通分量")
        print(f"    - 合并聚类: {len(top_comp_diag['merged_clusters'])}")
        print(f"    - 独立规格组: {top_comp_diag['single_independent']}")
        
        print(f"\n  Bot 表面：{len(bot_components)} 个连通分量")
        print(f"    - 合并聚类: {len(bot_comp_diag['merged_clusters'])}")
        print(f"    - 独立规格组: {bot_comp_diag['single_independent']}")
        
        # 打印结果
        print_grouping_results(all_groups_set, top_components, top_comp_diag, 
                                  bot_components, bot_comp_diag)
        
        # 导出配置
        export_grouping_config(all_groups_set, top_components, bot_components)
        
        print("\n" + "="*80)
        print("【阶段 C 完成】")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\n✗ 执行出错: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    main()
