"""
Spec Merge Analyzer - 规格组合并可行性分析包

这个包提供了一个三阶段流水线来评估不同镀锡规格组是否可以合并建立统一的校准模型。

核心模块：
- phase_a_ks_test_screening: KS 分布检验筛选
- phase_b_chow_test_validation: Chow 校准规律验证
- phase_c_graph_grouping: 图论分组聚类

典型用法：
    from spec_merge_analyzer import run_analysis
    
    # 运行完整流水线
    success = run_analysis(phase='all')
    
    # 或单独运行某个阶段
    success = run_analysis(phase='a')
    success = run_analysis(phase='b')
    success = run_analysis(phase='c')
"""

__version__ = '1.0.0'
__author__ = 'Tin Coating Analysis Team'

import sys
import os

# 导入核心模块
from . import phase_a_ks_test_screening as phase_a
from . import phase_b_chow_test_validation as phase_b
from . import phase_c_graph_grouping as phase_c


def run_phase_a():
    """执行阶段 A: KS 检验筛选"""
    print("\n" + "="*70)
    print("【执行阶段 A: KS 检验筛选（包含规格距离）】")
    print("="*70)
    
    try:
        phase_a.main()
        print("\n✓ 阶段 A 完成")
        return True
    except Exception as e:
        print(f"\n✗ 阶段 A 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_phase_b():
    """执行阶段 B: Chow 检验验证"""
    if not os.path.exists("result/spec_group_merge_analysis/merge_candidates.csv"):
        print("\n✗ 阶段 B 依赖阶段 A 的输出 (merge_candidates.csv)")
        return False
    
    print("\n" + "="*70)
    print("【执行阶段 B: Chow 检验验证】")
    print("="*70)
    
    try:
        phase_b.main()
        print("\n✓ 阶段 B 完成")
        return True
    except Exception as e:
        print(f"\n✗ 阶段 B 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_phase_c():
    """执行阶段 C: 图论分组"""
    if not os.path.exists("result/spec_group_merge_analysis/chow_test_results_v3.json"):
        print("\n✗ 阶段 C 依赖阶段 B 的输出 (chow_test_results_v3.json)")
        print("  请先执行阶段 B")
        return False
    
    print("\n" + "="*70)
    print("【执行阶段 C: 图论分组（独立表面聚类）】")
    print("="*70)
    
    try:
        result = phase_c.main()
        if result:
            print("\n✓ 阶段 C 完成")
            return True
        else:
            return False
    except Exception as e:
        print(f"\n✗ 阶段 C 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_analysis(phase='all', skip_c=False):
    """
    运行规格组合并分析流水线
    
    参数：
        phase (str): 执行的阶段
                    'a': 只执行阶段 A (KS 检验)
                    'b': 只执行阶段 B (Chow 检验)
                    'c': 只执行阶段 C (图论分组)
                    'all': 执行所有阶段 (默认)
        skip_c (bool): 是否跳过阶段 C
    
    返回：
        bool: 流水线是否成功执行
    """
    if phase not in ['a', 'b', 'c', 'all']:
        print(f"✗ 不支持的阶段: {phase}")
        return False
    
    # 创建结果目录
    os.makedirs("result/spec_group_merge_analysis", exist_ok=True)
    
    if phase == 'a' or phase == 'all':
        success_a = run_phase_a()
        if not success_a and phase == 'a':
            return False
    
    if phase == 'b' or phase == 'all':
        success_b = run_phase_b()
        if not success_b and phase == 'b':
            return False
    
    if (phase == 'c' or (phase == 'all' and not skip_c)):
        success_c = run_phase_c()
        if not success_c and phase == 'c':
            return False
    
    return True


def print_summary():
    """打印流水线摘要"""
    print("\n" + "="*80)
    print("【规格组合并分析流水线 - 完整版本】")
    print("="*80)
    print()
    print("【三阶段流水线】：")
    print("  ✓ 阶段 A: KS 检验筛选")
    print("    - 快速筛选 Delta 分布相同的规格组对")
    print("    - 计算规格距离（欧氏距离，单位 g/m²）")
    print("    - 生成 merge_candidates.csv")
    print()
    print("  ✓ 阶段 B: Chow 检验验证")
    print("    - 对候选对进行深入验证")
    print("    - 决策逻辑：Chow 通过 OR 性能提升")
    print("    - 样本不平衡检查和规格距离约束")
    print()
    print("  ✓ 阶段 C: 图论分组")
    print("    - 基于检验结果自动分组")
    print("    - Top 和 Bot 表面独立决策")
    print("    - 输出最终分组方案")
    print()
    print("【输出文件】：")
    print("  - merge_candidates.csv: 阶段 A 候选对")
    print("  - chow_test_results_v3.json: 阶段 B 检验结果")
    print("  - recommended_grouping.json: 阶段 C 最终分组")
    print()
    print("="*80)


__all__ = [
    'phase_a',
    'phase_b', 
    'phase_c',
    'run_phase_a',
    'run_phase_b',
    'run_phase_c',
    'run_analysis',
    'print_summary'
]
