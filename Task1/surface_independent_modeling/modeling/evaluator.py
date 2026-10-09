"""
模型评估接口
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional
from pathlib import Path

from ..core import EVALUATION_DIR

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    模型评估和对比
    """
    
    def __init__(self, save_dir: Optional[str] = None):
        """
        初始化评估器
        
        Parameters
        ----------
        save_dir : str, optional
            结果保存目录
        """
        if save_dir is None:
            save_dir = str(EVALUATION_DIR)
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_all_models(self, training_results: Dict) -> Dict:
        """
        评估所有模型
        
        Parameters
        ----------
        training_results : dict
            训练结果 {group_key: {model, metrics, aux, status}}
        
        Returns
        -------
        evaluation_report : dict
            评估报告
        """
        evaluation_data = []
        
        for group_key, result in training_results.items():
            if result.get('status') == 'success' and result.get('metrics'):
                metrics = result['metrics']
                
                eval_row = {
                    'Group_Key': group_key,
                    'Train_Samples': metrics.get('训练样本数', 0),
                    'Val_Samples': metrics.get('验证样本数', 0),
                    'Test_Samples': metrics.get('测试样本数', 0),
                    'RMSE_Model_Test': metrics.get('RMSE_模型_测试', np.nan),
                    'MAE_Model_Test': metrics.get('MAE_模型_测试', np.nan),
                    'R2_Model_Test': metrics.get('R2_模型_测试', np.nan),
                    'RMSE_Improve_Pct': metrics.get('RMSE_提升_测试(%)', np.nan),
                    'Overfitting_RMSE': metrics.get('过拟合程度_RMSE_训练测试', np.nan),
                }
                evaluation_data.append(eval_row)
            else:
                logger.warning(f"跳过失败的模型: {group_key}")
        
        evaluation_df = pd.DataFrame(evaluation_data)
        
        # 保存到文件
        eval_file = self.save_dir / 'evaluation_summary.csv'
        evaluation_df.to_csv(eval_file, index=False, encoding='utf-8-sig')
        logger.info(f"评估汇总已保存: {eval_file}")
        
        return {
            'summary_table': evaluation_df,
            'save_path': str(eval_file),
        }
    
    def generate_comparison_table(self, training_results: Dict) -> pd.DataFrame:
        """
        生成模型对比表
        
        Parameters
        ----------
        training_results : dict
            训练结果
        
        Returns
        -------
        comparison_table : DataFrame
            对比表
        """
        evaluation = self.evaluate_all_models(training_results)
        return evaluation['summary_table']
    
    def get_best_models(
        self,
        training_results: Dict,
        metric: str = 'R2_Model_Test',
        top_n: int = 5,
    ) -> pd.DataFrame:
        """
        获取表现最好的N个模型
        
        Parameters
        ----------
        training_results : dict
            训练结果
        metric : str
            评估指标
        top_n : int
            返回的模型数量
        
        Returns
        -------
        best_models : DataFrame
            最好的模型表
        """
        comparison_table = self.generate_comparison_table(training_results)
        
        if metric not in comparison_table.columns:
            logger.warning(f"指标 {metric} 不存在，使用默认指标 RMSE_Model_Test")
            metric = 'RMSE_Model_Test'
            ascending = True
        else:
            # R2 越大越好，RMSE/MAE 越小越好
            ascending = metric.startswith('RMSE') or metric.startswith('MAE')
        
        best_models = comparison_table.nlargest(
            top_n, metric, keep='first'
        ) if not ascending else comparison_table.nsmallest(top_n, metric, keep='first')
        
        logger.info(f"返回表现最好的 {len(best_models)} 个模型 (按 {metric})")
        
        return best_models
    
    def save_evaluation_report(self, training_results: Dict, report_file: Optional[str] = None) -> str:
        """
        保存完整的评估报告
        
        Parameters
        ----------
        training_results : dict
            训练结果
        report_file : str, optional
            报告文件路径
        
        Returns
        -------
        report_file : str
            报告文件路径
        """
        if report_file is None:
            report_file = self.save_dir / 'evaluation_report.txt'
        else:
            report_file = Path(report_file)
        
        evaluation = self.evaluate_all_models(training_results)
        summary_table = evaluation['summary_table']
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("模型评估汇总报告\n")
            f.write("=" * 100 + "\n\n")
            
            f.write(summary_table.to_string(index=False))
            f.write("\n\n")
            
            # 统计信息
            f.write("=" * 100 + "\n")
            f.write("统计摘要\n")
            f.write("=" * 100 + "\n")
            f.write(f"总模型数: {len(summary_table)}\n")
            
            # 只有当有数据时才计算统计
            if len(summary_table) > 0:
                f.write(f"成功训练: {len(summary_table)}\n")
                if 'RMSE_Model_Test' in summary_table.columns:
                    f.write(f"平均测试RMSE: {summary_table['RMSE_Model_Test'].mean():.4f}\n")
                if 'MAE_Model_Test' in summary_table.columns:
                    f.write(f"平均测试MAE: {summary_table['MAE_Model_Test'].mean():.4f}\n")
                if 'R2_Model_Test' in summary_table.columns:
                    f.write(f"平均测试R2: {summary_table['R2_Model_Test'].mean():.4f}\n")
            else:
                f.write("没有成功训练的模型\n")
        
        logger.info(f"评估报告已保存: {report_file}")
        
        return str(report_file)
