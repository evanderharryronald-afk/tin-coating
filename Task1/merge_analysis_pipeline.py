"""
【完整规格组合并分析流水线】

整合三个阶段：
  阶段 A: KS 检验 - 筛选 Delta 分布相同的规格组对
  阶段 B: Chow 检验 + 性能对比 - 验证校准规律和实际性能
  阶段 C: 图论分组 - 基于检验结果自动分组

用户可以逐阶段执行或一键运行全流程。
"""

import sys
import os
import argparse

# 导入三个阶段的模块
try:
    import phase_a_ks_test_screening as phase_a
    import phase_b_chow_test_validation as phase_b
    import phase_c_graph_grouping as phase_c
except ImportError as e:
    print(f"✗ 导入模块失败: {e}")
    print("  请确保 phase_a/b/c 的脚本与此脚本在同一目录")
    sys.exit(1)


def print_banner():
    """打印欢迎横幅"""
    print("\n" + "="*80)
    print("  规格组合并可行性分析 - 完整流水线（改进版）")
    print("  (Phase A → Phase B → Phase C)")
    print("="*80)
    print()
    print("核心改进：")
    print("  ✓ Phase A: KS 检验现在保留所有候选对（而非硬性过滤）")
    print("  ✓ Phase B: 使用对齐迭代次数的 Chow 检验（确保公平比较）")
    print("  ✓ 诊断信息: 详细输出模型收敛、残差增长等诊断数据")
    print()
    print("阶段说明:")
    print("  A. KS 检验（必须）")
    print("     - 快速筛选 Delta 分布相同的规格组对（参考性）")
    print("     - 保留所有候选对，优先级: KS通过 > KS未通过")
    print("     - 生成 merge_candidates.csv（210 个候选对）")
    print()
    print("  B. Chow 检验（改进版 - 对齐迭代次数）")
    print("     - 对所有候选对进行深入验证")
    print("     - 每个规格组独立找最优迭代次数")
    print("     - 对齐三个模型的收敛程度，确保残差差异来自数据")
    print("     - 验证校准规律（模型系数）是否相同")
    print("     - 输出: 最优迭代次数、残差增长%、详细诊断信息")
    print()
    print("  C. 图论分组（可选，依赖 B）")
    print("     - 基于 Chow 检验结果自动构建最终分组方案")
    print("     - 生成 recommended_grouping.json")
    print()


def run_phase_a():
    """执行阶段 A"""
    print("\n" + "="*70)
    print("【执行阶段 A: KS 检验筛选】")
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
    """执行阶段 B"""
    # 检查阶段 A 的输出
    if not os.path.exists("result/spec_group_merge_analysis/merge_candidates.csv"):
        print("\n✗ 阶段 B 依赖阶段 A 的输出 (merge_candidates.csv)")
        print("  请先执行阶段 A")
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
    """执行阶段 C"""
    # 检查阶段 B 的输出
    if not os.path.exists("result/spec_group_merge_analysis/chow_test_results.json"):
        print("\n✗ 阶段 C 依赖阶段 B 的输出 (chow_test_results.json)")
        print("  请先执行阶段 A 和 B")
        return False
    
    print("\n" + "="*70)
    print("【执行阶段 C: 图论分组】")
    print("="*70)
    
    try:
        phase_c.main()
        print("\n✓ 阶段 C 完成")
        return True
    except Exception as e:
        print(f"\n✗ 阶段 C 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主程序"""
    parser = argparse.ArgumentParser(
        description="规格组合并可行性分析流水线"
    )
    
    parser.add_argument(
        '--phase',
        choices=['a', 'b', 'c', 'all'],
        default='all',
        help="执行的阶段（默认: all - 执行所有阶段）"
    )
    
    parser.add_argument(
        '--skip-b',
        action='store_true',
        help="跳过阶段 B（仅执行 A 和 C）"
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    # 创建结果目录
    os.makedirs("result/spec_group_merge_analysis", exist_ok=True)
    
    if args.phase == 'a' or args.phase == 'all':
        success_a = run_phase_a()
        if not success_a and args.phase == 'a':
            print("\n✗ 阶段 A 失败，退出")
            sys.exit(1)
    
    if args.phase == 'b' or (args.phase == 'all' and not args.skip_b):
        success_b = run_phase_b()
        if not success_b and args.phase == 'b':
            print("\n✗ 阶段 B 失败，退出")
            sys.exit(1)
    
    if args.phase == 'c' or (args.phase == 'all' and not args.skip_b):
        success_c = run_phase_c()
        if not success_c and args.phase == 'c':
            print("\n✗ 阶段 C 失败，退出")
            sys.exit(1)
    
    # 最终总结
    print("\n" + "="*80)
    print("【流水线执行完成】")
    print("="*80)
    print("\n生成的输出文件:")
    
    result_files = [
        ("result/spec_group_merge_analysis/merge_candidates.csv", "阶段 A: 所有候选合并对"),
        ("result/spec_group_merge_analysis/chow_test_results.json", "阶段 B: Chow 检验结果（对齐迭代次数版）"),
        ("result/spec_group_merge_analysis/recommended_grouping.json", "阶段 C: 最终分组方案"),
        ("result/spec_group_merge_analysis/migration_guide.md", "阶段 C: 迁移指南")
    ]
    
    for file_path, description in result_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"  ✓ {file_path}")
            print(f"    └─ {description} ({file_size} bytes)")
        else:
            print(f"  ✗ {file_path} (未生成)")
    
    print("\n" + "="*80)
    print("【关键改进说明】")
    print("="*80)
    print("""
1. KS 检验的改进（Phase A）
   - 之前: 仅保留 KS p > 0.05 的对（约 4 个）
   - 现在: 保留所有候选对（约 210 个），按优先级分类
   - 原因: KS 只检查边际分布，不检查条件分布
         即使分布略异，如果校准规律相同（Chow），仍可合并
   
2. Chow 检验的改进（Phase B）
   - 之前: 使用固定 max_iter=100，样本量大的规格组欠拟合
   - 现在: 每个规格组找最优迭代次数，对齐后再比较
   - 原因: 样本量不同导致 100 次迭代收敛程度不同
         残差差异可能来自超参数而非数据本身
   - 输出: 新增了最优迭代次数、残差增长%等诊断信息
   
3. 诊断信息的增强
   - 显示每个规格组的最优迭代次数
   - 显示三个模型的对齐迭代次数
   - 显示合并后的残差增长百分比
   - 帮助理解 Chow 检验的原因
""")
    print("="*80)
    print("\n下一步建议:")
    print("  1. 查看 merge_candidates.csv 的 ks_pass 列，区分通过/未通过的对")
    print("  2. 查看 chow_test_results.json 的诊断信息，理解为何不能合并")
    print("  3. 根据实际业务需求，考虑是否降低合并阈值或调整规格分组策略")
    print()


if __name__ == '__main__':
    main()
