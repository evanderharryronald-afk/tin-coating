"""
命令行入口
"""
import argparse
import sys
import logging

from pipeline import SurfaceIndependentModelingPipeline

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主程序"""
    parser = argparse.ArgumentParser(
        description='表面独立分组建模管道',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 运行完整管道
  python main.py --stage all --n_trials 100

  # 只运行分析
  python main.py --stage analysis

  # 只运行建模 (包含调优和训练)
  python main.py --stage modeling --n_trials 50

  # 只运行评估和可解释性
  python main.py --stage evaluation

  # 自定义输出目录
  python main.py --stage all --output_dir result/custom_output
        """
    )
    
    parser.add_argument(
        '--stage',
        choices=['all', 'analysis', 'tuning', 'training', 'evaluation', 'explainability', 'modeling'],
        default='all',
        help='执行的阶段 (默认: all)'
    )
    
    parser.add_argument(
        '--n_trials',
        type=int,
        default=100,
        help='超参数调优的试验次数 (默认: 100)'
    )
    
    parser.add_argument(
        '--n_splits',
        type=int,
        default=3,
        help='交叉验证的折数 (默认: 3)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='输出目录 (默认: result/surface_independent_modeling)'
    )
    
    parser.add_argument(
        '--skip_analysis',
        action='store_true',
        help='跳过分析阶段'
    )
    
    parser.add_argument(
        '--skip_tuning',
        action='store_true',
        help='跳过超参数调优，使用默认参数'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='详细输出 (默认: True)'
    )
    
    args = parser.parse_args()
    
    # 创建管道
    pipeline = SurfaceIndependentModelingPipeline(
        output_dir=args.output_dir,
        verbose=args.verbose,
    )
    
    try:
        if args.stage == 'all':
            # 运行完整管道
            pipeline.run_full_pipeline(
                run_analysis=not args.skip_analysis,
                run_tuning=not args.skip_tuning,
                run_training=True,
                run_evaluation=True,
                run_explainability=True,
                n_trials=args.n_trials,
            )
        
        elif args.stage == 'analysis':
            # 只运行分析
            pipeline.load_data()
            pipeline.build_groups()
            pipeline.run_analysis()
        
        elif args.stage == 'tuning':
            # 只运行调优
            pipeline.load_data()
            pipeline.build_groups()
            pipeline.tune_hyperparameters(
                n_trials=args.n_trials,
                n_splits=args.n_splits,
            )
        
        elif args.stage == 'training':
            # 运行调优和训练
            pipeline.load_data()
            pipeline.build_groups()
            best_params = pipeline.tune_hyperparameters(
                n_trials=args.n_trials,
                n_splits=args.n_splits,
            ) if not args.skip_tuning else None
            pipeline.train_models(params_dict=best_params)
        
        elif args.stage == 'modeling':
            # 运行调优和训练
            pipeline.load_data()
            pipeline.build_groups()
            best_params = pipeline.tune_hyperparameters(
                n_trials=args.n_trials,
                n_splits=args.n_splits,
            ) if not args.skip_tuning else None
            pipeline.train_models(params_dict=best_params)
        
        elif args.stage == 'evaluation':
            # 运行评估和可解释性
            pipeline.load_data()
            pipeline.build_groups()
            pipeline.train_models()  # 需要先训练模型
            pipeline.evaluate_models()
            pipeline.run_explainability()
        
        elif args.stage == 'explainability':
            # 只运行可解释性分析
            pipeline.load_data()
            pipeline.build_groups()
            pipeline.train_models()  # 需要先训练模型
            pipeline.run_explainability()
        
        pipeline.generate_summary_report()
        
        logger.info("=" * 80)
        logger.info("【执行完成】")
        logger.info("=" * 80)
        return 0
    
    except Exception as e:
        logger.error(f"执行失败: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
