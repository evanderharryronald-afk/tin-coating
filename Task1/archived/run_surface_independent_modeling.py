"""
表面独立分组建模 - 一键运行脚本

支持多种运行模式和分组过滤
"""

from surface_independent_modeling import SurfaceIndependentModelingPipeline

# ============================================
# 配置：选择运行模式
# ============================================

# 运行模式选择（修改这里即可切换）
RUN_MODE = 'quick_test'  # 可选值: 'full', 'analysis_only', 'modeling_only', 'quick_test'

# 超参数调优的试验次数
N_TRIALS = 3

# 输出目录（留空使用默认）
OUTPUT_DIR = None

# 指定要分析的分组（留空分析所有分组）
# 例如: ['Top_2.799', 'Bot_1.1', 'Top_2.0']
# 留空则分析所有样本数>=500的分组
GROUP_FILTER = ['Top_2.799']

# ============================================
# 各种运行模式的配置
# ============================================

MODES = {
    'full': {
        'description': '完整管道：分析 + 调优 + 训练 + 评估 + 可解释性',
        'config': {
            'run_analysis': True,
            'run_tuning': True,
            'run_training': True,
            'run_evaluation': True,
            'run_explainability': True,
            'n_trials': N_TRIALS,
        }
    },
    'quick_test': {
        'description': '快速测试：分析 + 训练（跳过调优，用默认参数） + 评估 + 可解释性',
        'config': {
            'run_analysis': True,
            'run_tuning': False,
            'run_training': True,
            'run_evaluation': True,
            'run_explainability': True,
            'n_trials': 0,
        }
    },
    'analysis_only': {
        'description': '仅分析：相关性分析 + EDA',
        'config': {
            'run_analysis': True,
            'run_tuning': False,
            'run_training': False,
            'run_evaluation': False,
            'run_explainability': False,
            'n_trials': 0,
        }
    },
    'modeling_only': {
        'description': '仅建模：调优 + 训练 + 评估',
        'config': {
            'run_analysis': False,
            'run_tuning': True,
            'run_training': True,
            'run_evaluation': True,
            'run_explainability': False,
            'n_trials': N_TRIALS,
        }
    },
    'explainability_only': {
        'description': '仅可解释性：假设模型已训练，进行特征重要性和残差分析',
        'config': {
            'run_analysis': False,
            'run_tuning': False,
            'run_training': True,  # 需要训练模型
            'run_evaluation': False,
            'run_explainability': True,
            'n_trials': 0,
        }
    },
}


def print_modes():
    """打印所有可用的运行模式"""
    print("\n" + "=" * 80)
    print("【可用的运行模式】")
    print("=" * 80)
    
    for mode_name, mode_info in MODES.items():
        print(f"\n{mode_name}:")
        print(f"  {mode_info['description']}")
    
    print("\n使用方法：修改脚本顶部的 RUN_MODE = '{mode_name}' 即可切换")
    print("=" * 80 + "\n")


def main():
    """主函数"""
    
    # 检查模式是否有效
    if RUN_MODE not in MODES:
        print(f"[ERROR] 无效的运行模式 '{RUN_MODE}'")
        print_modes()
        return 1
    
    # 获取模式配置
    mode_config = MODES[RUN_MODE]['config']
    mode_desc = MODES[RUN_MODE]['description']
    
    print("\n" + "=" * 80)
    print(f"【启动运行】模式: {RUN_MODE}")
    print(f"说明: {mode_desc}")
    if GROUP_FILTER:
        print(f"分组过滤: {GROUP_FILTER}")
    print("=" * 80 + "\n")
    
    try:
        # 创建管道
        pipeline = SurfaceIndependentModelingPipeline(
            output_dir=OUTPUT_DIR,
            verbose=True,
        )
        
        # 运行管道
        pipeline.run_full_pipeline(
            run_analysis=mode_config['run_analysis'],
            run_tuning=mode_config['run_tuning'],
            run_training=mode_config['run_training'],
            run_evaluation=mode_config['run_evaluation'],
            run_explainability=mode_config['run_explainability'],
            n_trials=mode_config['n_trials'],
            group_filter=GROUP_FILTER,
        )
        
        print("\n" + "=" * 80)
        print("[OK] 执行成功")
        print("=" * 80)
        print(f"结果已保存到: {pipeline.output_dir}")
        print("=" * 80 + "\n")
        
        return 0
    
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"[ERROR] 执行失败: {e}")
        print("=" * 80 + "\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    import sys
    
    # 如果命令行带 --show-modes，显示所有模式
    if len(sys.argv) > 1 and sys.argv[1] == '--show-modes':
        print_modes()
        sys.exit(0)
    
    exit_code = main()
    sys.exit(exit_code)
