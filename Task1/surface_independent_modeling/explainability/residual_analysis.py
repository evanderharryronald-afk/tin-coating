"""
残差诊断分析模块
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from ..core import EXPLAINABILITY_DIR

logger = logging.getLogger(__name__)


class ResidualAnalyzer:
    """
    残差诊断分析
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
            save_dir = str(EXPLAINABILITY_DIR)
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict:
        """
        计算残差及其诊断统计
        
        Parameters
        ----------
        y_true : array
            真实值
        y_pred : array
            预测值
        
        Returns
        -------
        residual_stats : dict
            残差统计信息
        """
        residuals = y_true - y_pred
        
        stats_dict = {
            'residuals': residuals,
            'mean': float(np.mean(residuals)),
            'std': float(np.std(residuals)),
            'min': float(np.min(residuals)),
            'max': float(np.max(residuals)),
            'median': float(np.median(residuals)),
            'q1': float(np.percentile(residuals, 25)),
            'q3': float(np.percentile(residuals, 75)),
            'skewness': float(stats.skew(residuals)),
            'kurtosis': float(stats.kurtosis(residuals)),
        }
        
        return stats_dict
    
    def detect_patterns(self, residuals: np.ndarray) -> Dict:
        """
        检测残差中的模式
        
        Parameters
        ----------
        residuals : array
            残差
        
        Returns
        -------
        patterns : dict
            检测到的模式
        """
        patterns = {
            'heteroscedasticity': self._check_heteroscedasticity(residuals),
            'autocorrelation': self._check_autocorrelation(residuals),
            'normality': self._check_normality(residuals),
        }
        
        return patterns
    
    def _check_heteroscedasticity(self, residuals: np.ndarray) -> Dict:
        """检查异方差性"""
        try:
            from scipy.stats import levene
            
            # 分成两组
            mid = len(residuals) // 2
            group1 = residuals[:mid]
            group2 = residuals[mid:]
            
            stat, p_value = levene(group1, group2)
            
            return {
                'test': 'Levene',
                'statistic': float(stat),
                'p_value': float(p_value),
                'conclusion': 'present' if p_value < 0.05 else 'absent',
            }
        except Exception as e:
            logger.warning(f"异方差性检验失败: {e}")
            return {}
    
    def _check_autocorrelation(self, residuals: np.ndarray, lag: int = 1) -> Dict:
        """检查自相关性"""
        try:
            from scipy.stats import pearsonr
            
            if len(residuals) < lag + 1:
                return {}
            
            r, p_value = pearsonr(residuals[:-lag], residuals[lag:])
            
            return {
                'lag': lag,
                'correlation': float(r),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
            }
        except Exception as e:
            logger.warning(f"自相关性检验失败: {e}")
            return {}
    
    def _check_normality(self, residuals: np.ndarray) -> Dict:
        """检查正态性"""
        try:
            from scipy.stats import shapiro, normaltest
            
            # Shapiro-Wilk 测试
            if len(residuals) < 5000:
                stat, p_value = shapiro(residuals)
                test_name = 'Shapiro-Wilk'
            else:
                stat, p_value = normaltest(residuals)
                test_name = 'D\'Agostino'
            
            return {
                'test': test_name,
                'statistic': float(stat),
                'p_value': float(p_value),
                'normal': p_value > 0.05,
            }
        except Exception as e:
            logger.warning(f"正态性检验失败: {e}")
            return {}
    
    def visualize_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        X: Optional[np.ndarray] = None,
        save_dir: Optional[str] = None,
    ) -> Dict:
        """
        绘制残差诊断图
        
        Parameters
        ----------
        y_true : array
            真实值
        y_pred : array
            预测值
        X : array, optional
            特征矩阵
        save_dir : str, optional
            保存目录
        
        Returns
        -------
        plot_files : dict
            保存的图文件路径
        """
        if save_dir is None:
            save_dir = self.save_dir
        else:
            save_dir = Path(save_dir)
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        residuals = y_true - y_pred
        plot_files = {}
        
        # 1. Residuals vs Fitted Values
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_pred, residuals, alpha=0.5, s=20)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel('Fitted Values')
        ax.set_ylabel('Residuals')
        ax.set_title('Residuals vs Fitted Values')
        plt.tight_layout()
        
        file_path = save_dir / 'residuals_vs_fitted.png'
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files['residuals_vs_fitted'] = str(file_path)
        
        # 2. Q-Q Plot
        fig, ax = plt.subplots(figsize=(8, 8))
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title('Normal Q-Q Plot')
        plt.tight_layout()
        
        file_path = save_dir / 'qq_plot.png'
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files['qq_plot'] = str(file_path)
        
        # 3. Residuals Distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Residuals')
        plt.tight_layout()
        
        file_path = save_dir / 'residuals_distribution.png'
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files['residuals_distribution'] = str(file_path)
        
        # 4. Residuals over Time/Index
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(residuals, alpha=0.7)
        ax.axhline(y=0, color='r', linestyle='--', lw=1)
        ax.set_xlabel('Index')
        ax.set_ylabel('Residuals')
        ax.set_title('Residuals over Time/Index')
        plt.tight_layout()
        
        file_path = save_dir / 'residuals_over_time.png'
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files['residuals_over_time'] = str(file_path)
        
        logger.info(f"残差诊断图已保存到 {save_dir}")
        
        return plot_files
    
    def analyze_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        X: Optional[np.ndarray] = None,
        group_key: str = 'default',
        save_dir: Optional[str] = None,
    ) -> Dict:
        """
        进行完整的残差分析
        
        Parameters
        ----------
        y_true : array
            真实值
        y_pred : array
            预测值
        X : array, optional
            特征矩阵
        group_key : str
            分组键
        save_dir : str, optional
            保存目录
        
        Returns
        -------
        analysis_results : dict
            分析结果
        """
        if save_dir is None:
            save_dir = self.save_dir / group_key
        else:
            save_dir = Path(save_dir) / group_key
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 计算残差统计
        residual_stats = self.compute_residuals(y_true, y_pred)
        
        # 检测模式
        patterns = self.detect_patterns(residual_stats['residuals'])
        
        # 绘制诊断图
        plot_files = self.visualize_residuals(y_true, y_pred, X, save_dir)
        
        # 保存统计结果
        stats_file = save_dir / 'residual_stats.txt'
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("残差诊断分析报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("【残差统计】\n")
            for key, value in residual_stats.items():
                if key != 'residuals':
                    f.write(f"{key}: {value}\n")
            
            f.write("\n【模式检测】\n")
            f.write(str(patterns))
        
        logger.info(f"残差分析已保存到 {save_dir}")
        
        return {
            'group_key': group_key,
            'residual_stats': residual_stats,
            'patterns': patterns,
            'plot_files': plot_files,
            'save_dir': str(save_dir),
            'status': 'success',
        }
