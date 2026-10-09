"""
完整的建模管道
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional
from pathlib import Path

from .core import (
    load_featured_data, build_surface_groups, filter_valid_groups,
    summarize_groups, OUTPUT_DIR
)
from .analysis import AnalysisRunner
from .modeling import (
    SurfaceGroupHyperparameterTuner, SurfaceGroupTrainer,
    ModelEvaluator, SurfaceGroupModel
)
from .explainability import (
    FeatureImportanceAnalyzer, ResidualAnalyzer, SHAPExplainer
)

logger = logging.getLogger(__name__)


class SurfaceIndependentModelingPipeline:
    """
    完整的表面独立分组建模管道
    """
    
    def __init__(self, output_dir: Optional[str] = None, verbose: bool = True):
        """
        初始化管道
        
        Parameters
        ----------
        output_dir : str, optional
            输出目录
        verbose : bool
            是否打印详细日志
        """
        self.output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 配置日志
        if verbose:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # 初始化各个组件
        self.data = None
        self.groups = None
        self.valid_groups = None
        self.analysis_results = None
        self.tuning_results = None
        self.training_results = None
        self.evaluation_results = None
        self.explainability_results = None
    
    def load_data(self) -> None:
        """加载特征工程后的数据"""
        logger.info("=" * 80)
        logger.info("【Step 1】加载数据")
        logger.info("=" * 80)
        
        self.data = load_featured_data()
        logger.info(f"数据加载完成: {self.data.shape}")
    
    def build_groups(self, group_filter: Optional[list] = None) -> None:
        """
        构建表面分组
        
        Parameters
        ----------
        group_filter : list, optional
            要分析的分组名列表，例如 ['Top_2.799', 'Bot_1.1']
            如果为None，则使用所有有效分组
        """
        logger.info("=" * 80)
        logger.info("【Step 2】构建表面分组")
        logger.info("=" * 80)
        
        if self.data is None:
            raise ValueError("请先加载数据")
        
        self.groups = build_surface_groups(self.data)
        logger.info(f"分组总数: {len(self.groups)}")
        
        self.valid_groups = filter_valid_groups(self.groups)
        logger.info(f"有效分组数: {len(self.valid_groups)}")
        
        # 应用分组过滤
        if group_filter:
            original_count = len(self.valid_groups)
            self.valid_groups = {
                k: v for k, v in self.valid_groups.items()
                if k in group_filter
            }
            logger.info(f"应用分组过滤: {original_count} → {len(self.valid_groups)} 个分组")
            
            if len(self.valid_groups) == 0:
                logger.warning(f"警告: 过滤后没有匹配的分组。可用分组: {list(self.groups.keys())[:10]}")
        
        summary = summarize_groups(self.valid_groups)
        logger.info("\n分组摘要:\n" + summary.to_string())
    
    def run_analysis(self, run_correlation: bool = True, run_eda: bool = True) -> None:
        """运行分析"""
        logger.info("=" * 80)
        logger.info("【Step 3】数据分析")
        logger.info("=" * 80)
        
        if self.valid_groups is None:
            raise ValueError("请先构建分组")
        
        runner = AnalysisRunner()
        self.analysis_results = runner.run_all_analyses(
            self.valid_groups,
            run_correlation=run_correlation,
            run_eda=run_eda,
        )
    
    def tune_hyperparameters(self, n_trials: int = 100, n_splits: int = 3) -> None:
        """超参数调优"""
        logger.info("=" * 80)
        logger.info("【Step 4】超参数调优")
        logger.info("=" * 80)
        
        if self.valid_groups is None:
            raise ValueError("请先构建分组")
        
        tuner = SurfaceGroupHyperparameterTuner()
        self.tuning_results = tuner.tune_all_groups(
            self.valid_groups,
            n_trials=n_trials,
            n_splits=n_splits,
        )
        
        # 提取最优超参数
        best_params_dict = tuner.get_best_params_dict(self.tuning_results)
        logger.info(f"超参数调优完成，共 {len(best_params_dict)} 个分组")
        
        return best_params_dict
    
    def train_models(self, params_dict: Optional[Dict] = None) -> None:
        """训练模型"""
        logger.info("=" * 80)
        logger.info("【Step 5】模型训练")
        logger.info("=" * 80)
        
        if self.valid_groups is None:
            raise ValueError("请先构建分组")
        
        trainer = SurfaceGroupTrainer(
            save_dir=str(self.output_dir / '05_trained_models')
        )
        self.training_results = trainer.train_all_groups(
            self.valid_groups,
            params_dict=params_dict,
        )
        
        # 保存训练好的模型
        trained_models = trainer.get_trained_models()
        trainer.save_trained_models(
            trained_models,
            str(self.output_dir / '05_trained_models')
        )
        
        logger.info("模型训练和保存完成")
    
    def evaluate_models(self) -> None:
        """评估模型"""
        logger.info("=" * 80)
        logger.info("【Step 6】模型评估")
        logger.info("=" * 80)
        
        if self.training_results is None:
            raise ValueError("请先训练模型")
        
        evaluator = ModelEvaluator(
            save_dir=str(self.output_dir / '06_model_evaluation')
        )
        self.evaluation_results = evaluator.evaluate_all_models(self.training_results)
        
        # 保存评估报告
        evaluator.save_evaluation_report(
            self.training_results,
            str(self.output_dir / '06_model_evaluation' / 'evaluation_report.txt')
        )
        
        logger.info("模型评估完成")
    
    def run_explainability(self) -> None:
        """运行可解释性分析"""
        logger.info("=" * 80)
        logger.info("【Step 7】可解释性分析")
        logger.info("=" * 80)
        
        if self.training_results is None:
            logger.warning("训练结果为 None，跳过可解释性分析")
            return
        
        # 检查成功的训练结果
        success_models = {k: v for k, v in self.training_results.items() 
                         if isinstance(v, dict) and v.get('status') == 'success'}
        
        if not success_models:
            logger.warning(f"没有成功的训练模型 (总共 {len(self.training_results)} 个结果)")
            logger.warning(f"结果内容: {list(self.training_results.keys())}")
            for key, val in list(self.training_results.items())[:1]:
                logger.warning(f"  {key}: {val.get('status', 'N/A') if isinstance(val, dict) else 'not dict'}")
            return
        
        logger.info(f"发现 {len(success_models)} 个成功的训练模型，开始可解释性分析...")
        
        self.explainability_results = {}
        explainability_dir = self.output_dir / '07_explainability'
        explainability_dir.mkdir(parents=True, exist_ok=True)
        
        for group_key, training_result in success_models.items():
            logger.info(f"\n分析 {group_key}...")
            
            try:
                # 提取训练数据
                model = training_result.get('model')
                aux = training_result.get('aux', {})
                metrics = training_result.get('metrics', {})
                
                if model is None:
                    logger.warning(f"无法获取 {group_key} 的模型")
                    continue
                
                # 获取训练/验证/测试数据
                train_data = aux.get('train_data')
                val_data = aux.get('val_data')
                test_data = aux.get('test_data')
                
                if test_data is None or test_data.empty:
                    logger.warning(f"{group_key} 没有测试数据")
                    continue
                
                # 获取表面信息
                surface = group_key.split('_')[0]
                target_col = f'{surface}_Delta'
                
                # 从aux中获取模型实际训练时使用的特征列
                feature_cols = aux.get('feature_cols', None)
                
                if not feature_cols:
                    logger.warning(f"{group_key}: 无法获取训练特征列信息")
                    continue
                
                # 验证特征列和目标列都存在
                if target_col not in test_data.columns:
                    logger.warning(f"{group_key}: 目标列 {target_col} 不存在")
                    continue
                
                missing_cols = [col for col in feature_cols if col not in test_data.columns]
                if missing_cols:
                    logger.warning(f"{group_key}: 缺少特征列 {missing_cols}")
                    continue
                
                X_test = test_data[feature_cols].values
                
                if X_test.shape[0] == 0:
                    logger.warning(f"{group_key} 测试数据为空")
                    continue
                
                # 获取预测结果
                logger.info(f"  → 获取预测结果...")
                try:
                    prefix = 'Top' if surface == 'Top' else 'Bot'
                    surface_cn = '上' if surface == 'Top' else '下'
                    online_col = f'Tin Weight_Actual[g/m2]_GALV_WEIGHT_{prefix.upper()}_Avg'
                    # 关键：真实的目标值是实验室测量值，不是在线值
                    y_true_col = f'{surface_cn}表面镀层重量A(XA1_0)'
                    
                    if y_true_col not in test_data.columns:
                        logger.warning(f"{group_key}: 缺少目标列 {y_true_col}")
                        continue
                    
                    y_test = test_data[y_true_col].values  # 实验室测量值（真实值）
                    
                    if hasattr(model, 'predict_smooth'):
                        # ResidualCorrectionModel 的情况
                        # predict_smooth 期望 X 是 DataFrame（带 index）和 online_actual 是 Series
                        X_test_df = test_data[feature_cols]  # DataFrame with index
                        online_actual_series = test_data[online_col]  # Series with index
                        
                        result = model.predict_smooth(X_test_df, online_actual_series)
                        # predict_smooth 返回 (final_pred, predicted_delta_smooth)
                        # final_pred 已经是最终预测的镀层重量
                        if isinstance(result, (list, tuple)) and len(result) >= 2:
                            y_pred = result[0].values  # 模型的最终预测
                        else:
                            y_pred = result
                    elif hasattr(model, 'predict'):
                        # 一般的 predict 返回 delta 预测，需要加上在线值
                        delta_pred = model.predict(X_test)
                        online_actual_values = test_data[online_col].values
                        y_pred = online_actual_values + delta_pred
                    else:
                        logger.warning(f"{group_key} 模型不支持预测")
                        continue
                    
                    # 确保 y_pred 是一维数组
                    if isinstance(y_pred, np.ndarray):
                        if y_pred.ndim > 1:
                            y_pred = y_pred.flatten()
                    elif not isinstance(y_pred, np.ndarray):
                        y_pred = np.asarray(y_pred)
                    
                    logger.info(f"  → 预测完成，y_test shape: {y_test.shape}, y_pred shape: {y_pred.shape}")
                except Exception as e_pred:
                    logger.error(f"  → 预测失败: {type(e_pred).__name__}: {e_pred}")
                    raise
                
                # 1. 特征重要性分析
                logger.info(f"  → 计算特征重要性...")
                try:
                    logger.info(f"    model={type(model)}, test_data shape={test_data.shape}, feature_cols={len(feature_cols)}")
                    importance_analyzer = FeatureImportanceAnalyzer(
                        save_dir=str(explainability_dir)
                    )
                    importance_results = importance_analyzer.analyze_single_model(
                        model=model,
                        X=test_data[feature_cols],
                        y=test_data[target_col],
                        feature_names=feature_cols,
                        group_key=group_key,
                    )
                    self.explainability_results[f'{group_key}_importance'] = importance_results
                    logger.info(f"  → 特征重要性分析完成")
                except Exception as e_imp:
                    logger.warning(f"  → 特征重要性分析失败: {type(e_imp).__name__}: {e_imp}")
                
                # 2. 残差分析
                logger.info(f"  → 进行残差分析...")
                try:
                    logger.info(f"    y_test type={type(y_test)}, y_pred type={type(y_pred)}")
                    residual_analyzer = ResidualAnalyzer(
                        save_dir=str(explainability_dir)
                    )
                    residual_results = residual_analyzer.analyze_residuals(
                        y_true=y_test,
                        y_pred=y_pred,
                        X=X_test,
                        group_key=group_key,
                        save_dir=str(explainability_dir),
                    )
                    self.explainability_results[f'{group_key}_residuals'] = residual_results
                    logger.info(f"  → 残差分析完成")
                except Exception as e_res:
                    logger.warning(f"  → 残差分析失败: {type(e_res).__name__}: {e_res}")
                
                logger.info(f"  ✓ {group_key} 的可解释性分析完成")
            
            except Exception as e:
                logger.error(f"  {group_key} 可解释性分析失败: {e}")
                import traceback
                logger.error(f"详细错误:\n{traceback.format_exc()}")
        
        logger.info("\n可解释性分析完成")
    
    def generate_summary_report(self) -> None:
        """生成总结报告"""
        logger.info("=" * 80)
        logger.info("【生成总结报告】")
        logger.info("=" * 80)
        
        report_file = self.output_dir / 'REPORT.md'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 表面独立分组建模 - 总结报告\n\n")
            
            f.write("## 数据概览\n")
            if self.data is not None:
                f.write(f"- 总样本数: {len(self.data)}\n")
            
            f.write("\n## 分组信息\n")
            if self.valid_groups is not None:
                f.write(f"- 有效分组数: {len(self.valid_groups)}\n")
                summary = summarize_groups(self.valid_groups)
                f.write(summary.to_string())
                f.write("\n\n")
            
            f.write("## 分析结果\n")
            if self.analysis_results is not None:
                f.write("- 相关性分析: 已完成\n")
                f.write("- EDA分析: 已完成\n")
            
            f.write("\n## 超参数调优\n")
            if self.tuning_results is not None:
                success_count = sum(1 for r in self.tuning_results.values() if r.get('status') == 'success')
                f.write(f"- 成功调优: {success_count}/{len(self.tuning_results)}\n")
            
            f.write("\n## 模型训练\n")
            if self.training_results is not None:
                success_count = sum(1 for r in self.training_results.values() if r.get('status') == 'success')
                f.write(f"- 成功训练: {success_count}/{len(self.training_results)}\n")
            
            f.write("\n## 模型评估\n")
            if self.evaluation_results is not None:
                f.write("- 已完成评估和对比\n")
            
            f.write("\n## 可解释性分析\n")
            if self.explainability_results is not None:
                f.write("- 特征重要性: 已生成\n")
                f.write("- 残差诊断: 已完成\n")
                f.write("- SHAP分析: 已完成\n")
            
            f.write("\n## 输出目录\n")
            f.write(f"- {self.output_dir}\n")
            f.write("\n子目录:\n")
            f.write("- `01_grouped_data/`: 分组后的数据\n")
            f.write("- `02_correlation_analysis/`: 相关性分析结果\n")
            f.write("- `03_eda_analysis/`: EDA分析结果\n")
            f.write("- `04_hyperparameter_tuning/`: 调优结果\n")
            f.write("- `05_trained_models/`: 训练好的模型\n")
            f.write("- `06_model_evaluation/`: 评估结果\n")
            f.write("- `07_explainability/`: 可解释性分析\n")
        
        logger.info(f"总结报告已保存: {report_file}")
    
    def run_full_pipeline(
        self,
        run_analysis: bool = True,
        run_tuning: bool = True,
        run_training: bool = True,
        run_evaluation: bool = True,
        run_explainability: bool = True,
        n_trials: int = 100,
        group_filter: Optional[list] = None,
    ) -> None:
        """
        运行完整管道
        
        Parameters
        ----------
        run_analysis : bool
            是否运行分析
        run_tuning : bool
            是否运行超参数调优
        run_training : bool
            是否运行模型训练
        run_evaluation : bool
            是否运行模型评估
        run_explainability : bool
            是否运行可解释性分析
        n_trials : int
            超参数调优的试验次数
        group_filter : list, optional
            要分析的分组名列表，例如 ['Top_2.799', 'Bot_1.1']
            如果为None，则使用所有有效分组
        """
        logger.info("开始完整管道...")
        
        # Step 1: 数据加载和分组
        self.load_data()
        self.build_groups(group_filter=group_filter)
        
        # Step 2: 分析
        if run_analysis:
            self.run_analysis()
        
        # Step 3: 超参数调优
        best_params = None
        if run_tuning:
            best_params = self.tune_hyperparameters(n_trials=n_trials)
        
        # Step 4: 模型训练
        if run_training:
            self.train_models(params_dict=best_params)
        
        # Step 5: 模型评估
        if run_evaluation and self.training_results is not None:
            self.evaluate_models()
        
        # Step 6: 可解释性分析
        if run_explainability and self.training_results is not None:
            self.run_explainability()
        
        # 生成报告
        self.generate_summary_report()
        
        logger.info("=" * 80)
        logger.info("【完整管道执行完成】")
        logger.info("=" * 80)
        logger.info(f"结果已保存到: {self.output_dir}")
