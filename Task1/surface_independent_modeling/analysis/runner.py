"""
分析流程协调模块
"""
import logging
from typing import Dict, Optional

from .correlation import SurfaceGroupCorrelationAnalyzer
from .eda import SurfaceGroupEDAAnalyzer
from ..core import CORRELATION_DIR, EDA_DIR

logger = logging.getLogger(__name__)


class AnalysisRunner:
    """
    协调所有分析的运行
    """
    
    def __init__(
        self,
        correlation_dir: Optional[str] = None,
        eda_dir: Optional[str] = None,
    ):
        """
        初始化分析运行器
        
        Parameters
        ----------
        correlation_dir : str, optional
            相关性分析保存目录
        eda_dir : str, optional
            EDA分析保存目录
        """
        self.correlation_analyzer = SurfaceGroupCorrelationAnalyzer(
            save_dir=correlation_dir or str(CORRELATION_DIR)
        )
        self.eda_analyzer = SurfaceGroupEDAAnalyzer(
            save_dir=eda_dir or str(EDA_DIR)
        )
    
    def run_all_analyses(
        self,
        groups: Dict,
        run_correlation: bool = True,
        run_eda: bool = True,
    ) -> Dict:
        """
        运行所有分析
        
        Parameters
        ----------
        groups : dict
            分组信息
        run_correlation : bool
            是否运行相关性分析
        run_eda : bool
            是否运行EDA分析
        
        Returns
        -------
        all_results : dict
            所有分析结果
        """
        all_results = {}
        
        if run_correlation:
            logger.info("=" * 80)
            logger.info("开始相关性分析...")
            logger.info("=" * 80)
            correlation_results = self.correlation_analyzer.analyze_all_groups(groups)
            all_results['correlation'] = correlation_results
        
        if run_eda:
            logger.info("=" * 80)
            logger.info("开始EDA分析...")
            logger.info("=" * 80)
            eda_results = self.eda_analyzer.analyze_all_groups(groups, advanced=True)
            all_results['eda'] = eda_results
            
            # 生成EDA汇总报告
            eda_report = self.eda_analyzer.generate_eda_report(eda_results)
            all_results['eda_report'] = eda_report
        
        logger.info("=" * 80)
        logger.info("所有分析完成！")
        logger.info("=" * 80)
        
        return all_results
    
    def run_correlation_analysis(self, groups: Dict) -> Dict:
        """
        仅运行相关性分析
        
        Parameters
        ----------
        groups : dict
            分组信息
        
        Returns
        -------
        results : dict
            相关性分析结果
        """
        logger.info("=" * 80)
        logger.info("开始相关性分析...")
        logger.info("=" * 80)
        
        results = self.correlation_analyzer.analyze_all_groups(groups)
        
        logger.info("相关性分析完成！")
        
        return results
    
    def run_eda_analysis(self, groups: Dict, advanced: bool = True) -> Dict:
        """
        仅运行EDA分析
        
        Parameters
        ----------
        groups : dict
            分组信息
        advanced : bool
            是否进行高级分析
        
        Returns
        -------
        results : dict
            EDA分析结果
        """
        logger.info("=" * 80)
        logger.info("开始EDA分析...")
        logger.info("=" * 80)
        
        eda_results = self.eda_analyzer.analyze_all_groups(groups, advanced=advanced)
        eda_report = self.eda_analyzer.generate_eda_report(eda_results)
        
        results = {
            'eda': eda_results,
            'eda_report': eda_report,
        }
        
        logger.info("EDA分析完成！")
        
        return results
