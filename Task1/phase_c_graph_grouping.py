"""
【阶段 C：基于检验结果的规格组重新分组】

使用图论方法将可合并的规格组自动分组：
- 将可合并的规格组对视为图的边
- 使用连通分量算法找到自然分组
- 输出最终的规格组重新分组方案
"""

import json
import pandas as pd
import networkx as nx
from typing import List, Set, Dict
import os


def load_chow_results(results_json: str = "result/spec_group_merge_analysis/chow_test_results.json") -> list:
    """加载阶段 B 的检验结果"""
    if not os.path.exists(results_json):
        raise FileNotFoundError(f"未找到检验结果文件: {results_json}")
    
    with open(results_json, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print(f"✓ 加载 Chow 检验结果: {len(results)} 对")
    return results


def build_merge_graph(results: list) -> nx.Graph:
    """
    构建合并关系图
    可以合并的规格组对作为边
    """
    G = nx.Graph()
    
    can_merge_pairs = [r for r in results if r['final_decision'] == 'YES']
    
    print(f"\n可以合并的对: {len(can_merge_pairs)} 对")
    
    for pair in can_merge_pairs:
        group_A = pair['group_A']
        group_B = pair['group_B']
        surface = pair['surface']
        
        # 节点格式: "group_label_surface"
        node_A = f"{group_A}_{surface}"
        node_B = f"{group_B}_{surface}"
        
        G.add_edge(node_A, node_B, weight=pair.get('chow_p_value', 0))
        print(f"  添加边: {group_A} -- {group_B} ({surface})")
    
    return G


def find_connected_components(G: nx.Graph) -> List[Set]:
    """找到图的所有连通分量"""
    components = list(nx.connected_components(G))
    return components


def extract_group_names_from_components(components: List[Set]) -> List[Set]:
    """
    从节点中提取规格组名称
    节点格式是 "group_label_surface"，我们需要提取 "group_label"
    """
    group_components = []
    
    for component in components:
        groups = set()
        for node in component:
            # 从 "Top1.1_Bot1.1_Top" 中提取 "Top1.1_Bot1.1"
            parts = node.rsplit('_', 1)  # 从右边分割一次，保留表面标记
            group_label = parts[0]
            groups.add(group_label)
        
        group_components.append(groups)
    
    return group_components


def print_grouping_results(group_components: List[Set]):
    """打印重新分组的结果"""
    print("\n" + "="*70)
    print("【规格组重新分组方案】")
    print("="*70)
    
    # 分类
    merged_groups = [g for g in group_components if len(g) > 1]
    single_groups = [g for g in group_components if len(g) == 1]
    
    print(f"\n✓ 总共 {len(group_components)} 个分组")
    print(f"  - 合并组: {len(merged_groups)} 个")
    print(f"  - 独立组: {len(single_groups)} 个")
    
    print("\n【合并组】")
    for i, group_set in enumerate(merged_groups, 1):
        groups = sorted(list(group_set))
        print(f"\n新分组 {i}:")
        for g in groups:
            print(f"  - {g}")
    
    print("\n【保持独立建模的规格组】")
    for g_set in single_groups:
        g = list(g_set)[0]
        print(f"  - {g}")


def export_grouping_config(group_components: List[Set], output_path: str = "result/spec_group_merge_analysis/recommended_grouping.json"):
    """导出重新分组方案到 JSON 配置"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    config = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'description': '基于 KS 检验 + Chow 检验的规格组推荐合并方案',
        'methodology': '阶段 A（KS 筛选）→ 阶段 B（Chow 验证）→ 阶段 C（图论分组）',
        'grouping_strategy': {
            'phase_a': 'KS 检验筛选 Delta 分布相同的规格组对',
            'phase_b': 'Chow 检验验证校准规律是否相同，性能对比',
            'phase_c': '使用连通分量算法自动分组'
        },
        'merged_groups': [sorted(list(g_set)) for g_set in group_components if len(g_set) > 1],
        'independent_groups': [list(g_set)[0] for g_set in group_components if len(g_set) == 1]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 重新分组方案已保存至: {output_path}")
    return config


def generate_migration_script(group_components: List[Set], output_path: str = "result/spec_group_merge_analysis/migration_guide.md"):
    """生成规格组合并的迁移指南"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    merged_groups = [g for g in group_components if len(g) > 1]
    
    content = """# 规格组合并迁移指南

## 概述
基于三阶段统计检验，以下规格组可以进行合并建模。

## 推荐的合并方案

"""
    
    for i, group_set in enumerate(merged_groups, 1):
        groups = sorted(list(group_set))
        content += f"### 合并 {i}: {' + '.join(groups)}\n\n"
        content += f"可以使用统一的校准模型对所有这些规格组进行预测。\n\n"
    
    content += """## 实施步骤

1. 更新 optuna_tuning_config_grouped.json 中的 target_spec_groups
2. 根据新分组重新组织数据
3. 训练新的统一模型
4. 对比性能并验证

"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ 迁移指南已保存至: {output_path}")


def main():
    """主流程"""
    print("\n" + "="*70)
    print("【阶段 C：规格组重新分组】")
    print("="*70)
    
    try:
        # 1. 加载 Chow 检验结果
        results = load_chow_results()
        
        # 2. 构建合并关系图
        print("\n[构建合并关系图]")
        G = build_merge_graph(results)
        print(f"  图中节点数: {G.number_of_nodes()}")
        print(f"  图中边数: {G.number_of_edges()}")
        
        # 3. 找到连通分量
        print("\n[寻找自然分组]")
        components = find_connected_components(G)
        print(f"  连通分量数: {len(components)}")
        
        group_components = extract_group_names_from_components(components)
        
        # 4. 打印结果
        print_grouping_results(group_components)
        
        # 5. 导出配置
        config = export_grouping_config(group_components)
        
        # 6. 生成迁移指南
        generate_migration_script(group_components)
        
        print("\n" + "="*70)
        print("【阶段 C 完成】")
        print("="*70)
        print()
        
    except Exception as e:
        print(f"\n✗ 执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
