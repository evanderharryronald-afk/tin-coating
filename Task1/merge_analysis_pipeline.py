"""
【完整规格组合并分析流水线】

使用 spec_merge_analyzer 包执行规格组合并分析。

整合三个阶段：
  阶段 A: KS 检验 - 筛选 Delta 分布相同的规格组对
  阶段 B: Chow 检验 + 性能对比 - 验证校准规律和实际性能
  阶段 C: 图论分组 - 基于检验结果自动分组

用户可以逐阶段执行或一键运行全流程。

使用方式：
  python merge_analysis_pipeline.py --phase all
  python merge_analysis_pipeline.py --phase a
  python merge_analysis_pipeline.py --phase b
  python merge_analysis_pipeline.py --phase c
"""

import sys
import os
import argparse

# 导入包
try:
    import spec_merge_analyzer as analyzer
except ImportError as e:
    print(f"✗ 导入 spec_merge_analyzer 包失败: {e}")
    print("  请确保 spec_merge_analyzer 包在当前目录或 Python 路径中")
    sys.exit(1)


def main():
    """主程序"""
    parser = argparse.ArgumentParser(
        description="规格组合并可行性分析流水线（使用 spec_merge_analyzer 包）"
    )
    
    parser.add_argument(
        '--phase',
        choices=['a', 'b', 'c', 'all'],
        default='all',
        help="执行的阶段（默认: all - 执行 A + B + C）"
    )
    
    parser.add_argument(
        '--skip-c',
        action='store_true',
        help="跳过阶段 C（仅执行 A 和 B）"
    )
    
    args = parser.parse_args()
    
    analyzer.print_summary()
    
    # 执行分析流水线
    success = analyzer.run_analysis(phase=args.phase, skip_c=args.skip_c)
    
    if success:
        print("\n" + "="*80)
        print("【流水线执行完成】")
        print("="*80)
        print("\n生成的输出文件:")
        
        output_files = [
            ("result/spec_group_merge_analysis/merge_candidates.csv", "阶段 A: 所有候选对"),
            ("result/spec_group_merge_analysis/chow_test_results_v3.json", "阶段 B: Chow 检验结果"),
            ("result/spec_group_merge_analysis/recommended_grouping.json", "阶段 C: 最终分组方案（独立表面）")
        ]
        
        for file_path, description in output_files:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"  ✓ {file_path} ({file_size} bytes)")
            else:
                print(f"  ✗ {file_path} (未生成)")
        
        print("\n" + "="*80)
    else:
        print("\n✗ 流水线执行失败")
        sys.exit(1)


if __name__ == '__main__':
    main()
