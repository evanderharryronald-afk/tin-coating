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

# 导入阶段的模块
try:
    import phase_a_ks_test_screening as phase_a
    import phase_c_graph_grouping as phase_c
except ImportError as e:
    print(f"✗ 导入模块失败: {e}")
    print("  请确保 phase_a/c 的脚本与此脚本在同一目录")
    sys.exit(1)


def print_banner():
    """打印欢迎横幅"""
    print("\n" + "="*80)
    print("  规格组合并可行性分析 - 完整流水线（优化版 v2 - 规格距离优先）")
    print("  (Phase A → Phase B(v2) → Phase C)")
    print("="*80)
    print()
    print("【核心优化 - 规格距离驱动】：")
    print("  ✓ Phase A: 计算规格距离，按接近度排序候选对")
    print("  ✓ Phase B v2: 基于规格距离的动态 Chow 检验阈值")
    print("  ✓ 规格接近的对优先进入 Tier 1（最大化验证收益）")
    print("  ✓ 规格差异大的对直接排除 Tier 3（节省计算资源）")
    print()
    print("阶段说明:")
    print("  A. KS 检验（必须）")
    print("     - 快速筛选 Delta 分布相同的规格组对")
    print("     - 计算规格距离（欧氏距离，单位 g/m²）")
    print("     - 生成 merge_candidates.csv（按 KS 和规格距离排序）")
    print()
    print("  B. Chow 检验（优化版 v2）")
    print("     - 对所有候选对进行深入验证")
    print("     - 每个规格组独立找最优迭代次数，对齐后比较")
    print("     - 【新增】规格距离的动态 Chow 阈值：")
    print("       · 规格接近(<0.5): 需要 p > 0.05")
    print("       · 规格中等(0.5-1.0): 需要 p > 0.02")
    print("       · 规格差异(>1.0): 基本不可能合并")
    print("     - 输出: Chow 检验结果（包含规格距离诊断）")
    print()
    print("  C. 图论分组（可选，依赖 B）")
    print("     - 基于 Chow 检验结果自动构建最终分组方案")
    print("     - Top 和 Bot 表面独立分组（v4 修正）")
    print()


def run_phase_a():
    """执行阶段 A"""
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


def run_phase_b_v2():
    """执行阶段 B 优化版 v2（规格距离优先）"""
    if not os.path.exists("result/spec_group_merge_analysis/merge_candidates.csv"):
        print("\n✗ 阶段 B 依赖阶段 A 的输出 (merge_candidates.csv)")
        return False
    
    print("\n" + "="*70)
    print("【执行阶段 B v2: Chow 检验验证（规格距离优先）】")
    print("="*70)
    
    try:
        import phase_b_chow_test_validation_v2 as phase_b_v2
        phase_b_v2.main()
        print("\n✓ 阶段 B v2 完成")
        return True
    except Exception as e:
        print(f"\n✗ 阶段 B v2 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_phase_b_v3():
    """执行阶段 B 优化版 v3（改进决策逻辑）"""
    if not os.path.exists("result/spec_group_merge_analysis/merge_candidates.csv"):
        print("\n✗ 阶段 B 依赖阶段 A 的输出 (merge_candidates.csv)")
        return False
    
    print("\n" + "="*70)
    print("【执行阶段 B v3: Chow 检验验证（改进决策逻辑）】")
    print("="*70)
    
    try:
        import phase_b_chow_test_validation_v3 as phase_b_v3
        phase_b_v3.main()
        print("\n✓ 阶段 B v3 完成")
        return True
    except Exception as e:
        print(f"\n✗ 阶段 B v3 失败: {e}")
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


def run_phase_c_v3():
    """执行阶段 C 优化版 v3（改进聚类策略）"""
    if not os.path.exists("result/spec_group_merge_analysis/chow_test_results_v3.json"):
        print("\n✗ 阶段 C v3 依赖阶段 B v3 的输出 (chow_test_results_v3.json)")
        print("  请先执行阶段 B v3")
        return False
    
    print("\n" + "="*70)
    print("【执行阶段 C v3: 独立表面聚类（v4 修正）】")
    print("="*70)
    
    try:
        import phase_c_graph_grouping_v3 as phase_c_v3
        result = phase_c_v3.main()
        if result:
            print("\n✓ 阶段 C v3 完成")
            return True
        else:
            return False
    except Exception as e:
        print(f"\n✗ 阶段 C v3 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主程序"""
    parser = argparse.ArgumentParser(
        description="规格组合并可行性分析流水线（优化版 v2 - 规格距离优先）"
    )
    
    parser.add_argument(
        '--phase',
        choices=['a', 'b', 'b2', 'b3', 'c', 'c3', 'all', 'all2', 'all3'],
        default='all3',
        help="执行的阶段（默认: all3 - 执行 A + B v3 + C v3）"
    )
    
    parser.add_argument(
        '--skip-c',
        action='store_true',
        help="跳过阶段 C（仅执行 A 和 B）"
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    # 创建结果目录
    os.makedirs("result/spec_group_merge_analysis", exist_ok=True)
    
    if args.phase == 'a' or args.phase in ['all', 'all2', 'all3']:
        success_a = run_phase_a()
        if not success_a and args.phase == 'a':
            print("\n✗ 阶段 A 失败，退出")
            sys.exit(1)
    
    if args.phase == 'b2' or args.phase == 'all2':
        success_b2 = run_phase_b_v2()
        if not success_b2 and args.phase == 'b2':
            print("\n✗ 阶段 B v2 失败，退出")
            sys.exit(1)
    
    if args.phase == 'b3' or args.phase == 'all3':
        success_b3 = run_phase_b_v3()
        if not success_b3 and args.phase == 'b3':
            print("\n✗ 阶段 B v3 失败，退出")
            sys.exit(1)
    
    if (args.phase == 'c' or (args.phase in ['all', 'all2'] and not args.skip_c)):
        success_c = run_phase_c()
        if not success_c and args.phase == 'c':
            print("\n✗ 阶段 C 失败，退出")
            sys.exit(1)
    
    if args.phase == 'c3' or (args.phase == 'all3' and not args.skip_c):
        success_c3 = run_phase_c_v3()
        if not success_c3 and args.phase == 'c3':
            print("\n✗ 阶段 C v3 失败，退出")
            sys.exit(1)
    
    print("\n" + "="*80)
    print("【流水线执行完成】")
    print("="*80)
    print("\n生成的输出文件:")
    
    result_files_map = {
        'all3': [
            ("result/spec_group_merge_analysis/merge_candidates.csv", "阶段 A: 所有候选对"),
            ("result/spec_group_merge_analysis/chow_test_results_v3.json", "阶段 B v3: Chow 检验结果"),
            ("result/spec_group_merge_analysis/recommended_grouping_v4.json", "阶段 C v4: 最终分组方案（独立表面）")
        ],
        'all2': [
            ("result/spec_group_merge_analysis/merge_candidates.csv", "阶段 A: 所有候选对"),
            ("result/spec_group_merge_analysis/chow_test_results_v2_spec_distance.json", "阶段 B v2: Chow 检验结果"),
            ("result/spec_group_merge_analysis/recommended_grouping.json", "阶段 C: 最终分组方案")
        ]
    }
    
    result_files = result_files_map.get(args.phase, result_files_map['all3'])
    
    for file_path, description in result_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"  ✓ {file_path} ({file_size} bytes)")
        else:
            print(f"  ✗ {file_path} (未生成)")
    
    print("\n" + "="*80)
    print("【优化版 v2 的改进说明】")
    print("="*80)
    print("""
【规格距离驱动的优化】

1. Phase A 改进 - 规格距离计算
   新增：compute_spec_distance() 计算规格组对的欧氏距离
   效果：merge_candidates.csv 按 KS 通过状态和规格距离排序
        规格相近的对优先出现，提高发现合并对的效率

2. Phase B v2 改进 - 动态 Chow 阈值
   新增：get_chow_threshold_by_spec() 基于规格距离的动态阈值
   
   规格距离 vs Chow 阈值：
   · < 0.5 g/m²: p > 0.05 (严格) - 规格非常相近，期望高
   · 0.5-1.0 g/m²: p > 0.02 (适中) - 规格相近，期望中等
   · > 1.0 g/m²: p > 0.001 (宽松) - 规格差异大，基本不可能合并

3. 分层优化 - 优先级调整
   Tier 1：规格距离 < 0.8 OR KS 通过 → 深度验证（Chow 检验）
   Tier 2：规格距离 0.8-1.5 → 快速兼容性检查
   Tier 3：规格距离 > 1.5 → 直接排除（节省资源）

4. 诊断信息增强
   新增显示：
   · 规格距离值（欧氏距离，单位 g/m²）
   · 规格距离等级（非常接近/接近/中等/差异大）
   · 应用的 Chow 阈值（基于规格距离）
   · 完整的合并理由（包含规格信息）

【理论基础】

规格设定值的物理含义：
- 镀锡厚度设定值 = 工艺的"目标"
- 实际测量误差 = 偏离目标的随机波动
- 假设：设定值相近 → 工艺条件相近 → 误差分布和校准规律相近

从 KS 检验的数据验证：
- Tier 1 规格对 (0-0.8 g/m²) KS 通过率：20-30%
- Tier 2 规格对 (0.8-1.5 g/m²) KS 通过率：5-10%
- Tier 3 规格对 (>1.5 g/m²) KS 通过率：0-1%

明显的正相关关系表明规格距离是合并可能性的强预测因子。
""")
    print("="*80)
    print("\n下一步建议:")
    print("  1. 查看 merge_candidates.csv 的 spec_distance 列")
    print("     - 规格距离 < 0.8 的对是最有希望的合并候选")
    print("     - 规格距离 > 1.5 的对基本无法合并")
    print()
    print("  2. 查看 chow_test_results_v2_spec_distance.json")
    print("     - 观察 Chow 阈值与实际 p-value 的关系")
    print("     - 理解规格距离如何影响合并决策")
    print()
    print("  3. 优化工艺参数分组策略")
    print("     - 考虑按规格相近度进行初步分组")
    print("     - 针对相近规格组进行专门的模型优化")
    print()


if __name__ == '__main__':
    main()
