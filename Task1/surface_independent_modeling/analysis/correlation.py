"""
相关性分析模块 - 复用 correlation_analyzer.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Optional

from ..core import (
    CORRELATION_DIR, get_target_column, CORRELATION_METHODS,
    COMPUTE_MI, COMPUTE_DCOR
)

logger = logging.getLogger(__name__)


class SurfaceGroupCorrelationAnalyzer:
    """
    按表面分组进行相关性分析
    复用 correlation_analyzer.py 的 SurfaceCorrelationAnalyzer
    """
    
    def __init__(self, save_dir: Optional[str] = None):
        """
        初始化分析器
        
        Parameters
        ----------
        save_dir : str, optional
            结果保存目录
        """
        if save_dir is None:
            save_dir = str(CORRELATION_DIR)
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 延迟导入，避免循环依赖
        self.analyzer_ready = False
        self._correlation_analyzer = None
    
    def _get_correlation_analyzer(self):
        """懒加载 correlation_analyzer"""
        if not self.analyzer_ready:
            try:
                from correlation_analyzer import SurfaceCorrelationAnalyzer
                self._correlation_analyzer = SurfaceCorrelationAnalyzer(
                    default_save_dir=str(self.save_dir)
                )
                self.analyzer_ready = True
                logger.info("已加载 correlation_analyzer")
            except ImportError as e:
                logger.error(f"无法导入 correlation_analyzer: {e}")
                raise
        
        return self._correlation_analyzer
    
    def analyze_single_group(
        self,
        df: pd.DataFrame,
        surface: str,
        value: float,
        corr_method: str = 'both',
        compute_mi: bool = COMPUTE_MI,
        compute_dcor: bool = COMPUTE_DCOR,
        save_dir: Optional[str] = None
    ) -> Dict:
        """
        分析单个表面分组的相关性
        
        Parameters
        ----------
        df : DataFrame
            分组数据
        surface : str
            'Top' 或 'Bot'
        value : float
            设定厚度值
        corr_method : str
            相关性方法
        compute_mi : bool
            是否计算互信息
        compute_dcor : bool
            是否计算距离相关性
        save_dir : str, optional
            自定义保存目录
        
        Returns
        -------
        results : dict
            分析结果
        """
        if save_dir is None:
            save_dir = str(self.save_dir / f'{surface}_{value}')
        
        analyzer = self._get_correlation_analyzer()
        
        try:
            results = analyzer.analyze_surface(
                df=df,
                surface=surface,
                save_dir=save_dir,
                corr_method=corr_method,
                compute_mi=compute_mi,
                compute_dcor=compute_dcor,
            )
            
            logger.info(f"完成相关性分析: {surface}_{value}")
            return results
        
        except Exception as e:
            logger.error(f"相关性分析失败: {surface}_{value}, 错误: {e}")
            return {}
    
    def analyze_all_groups(
        self,
        groups: Dict,
        corr_method: str = 'both',
        compute_mi: bool = COMPUTE_MI,
        compute_dcor: bool = COMPUTE_DCOR,
    ) -> Dict:
        """
        分析所有表面分组的相关性
        
        Parameters
        ----------
        groups : dict
            分组信息 {group_key: (surface, value, group_df, n_samples)}
        corr_method : str
            相关性方法
        compute_mi : bool
            是否计算互信息
        compute_dcor : bool
            是否计算距离相关性
        
        Returns
        -------
        all_results : dict
            所有分组的分析结果
        """
        all_results = {}
        
        for group_key, (surface, value, group_df, n_samples) in groups.items():
            logger.info(f"正在分析 {group_key} ({n_samples} 样本)...")
            
            results = self.analyze_single_group(
                df=group_df,
                surface=surface,
                value=value,
                corr_method=corr_method,
                compute_mi=compute_mi,
                compute_dcor=compute_dcor,
            )
            
            all_results[group_key] = results
        
        logger.info(f"完成所有分组的相关性分析，共 {len(all_results)} 个分组")
        
        return all_results
