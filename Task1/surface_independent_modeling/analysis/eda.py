"""
EDA分析模块 - 复用 eda_analyzer.py 中的 SurfaceEDAAnalyzer
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
from typing import Dict, Optional, List

from ..core import EDA_DIR, get_target_column, get_feature_columns

logger = logging.getLogger(__name__)


class SurfaceGroupEDAAnalyzer:
    """
    按表面分组进行EDA分析
    复用现有的 SurfaceEDAAnalyzer 从 eda_analyzer.py
    """
    
    def __init__(self, save_dir: Optional[str] = None):
        """
        初始化EDA分析器
        
        Parameters
        ----------
        save_dir : str, optional
            结果保存目录
        """
        if save_dir is None:
            save_dir = str(EDA_DIR)
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._surface_eda_analyzer = None
        self._eda_analyzer_loaded = False
    
    def _get_surface_eda_analyzer(self):
        """懒加载 SurfaceEDAAnalyzer"""
        if not self._eda_analyzer_loaded:
            try:
                # 从 Task1 目录导入 eda_analyzer
                import sys
                from pathlib import Path
                
                # 获取当前文件所在目录，向上两级到 Task1
                task1_dir = Path(__file__).parent.parent.parent
                if str(task1_dir) not in sys.path:
                    sys.path.insert(0, str(task1_dir))
                
                from eda_analyzer import SurfaceEDAAnalyzer
                self._surface_eda_analyzer = SurfaceEDAAnalyzer()
                self._eda_analyzer_loaded = True
                logger.info("✓ 已加载 SurfaceEDAAnalyzer")
            except ImportError as e:
                logger.error(f"✗ 无法导入 SurfaceEDAAnalyzer: {e}")
                self._eda_analyzer_loaded = True
                return None
        
        return self._surface_eda_analyzer
    
    def analyze_single_group_basic(
        self,
        df: pd.DataFrame,
        surface: str,
        value: float,
        save_dir: Optional[str] = None
    ) -> Dict:
        """
        进行基础EDA分析（单变量分布、时间趋势等）
        
        Parameters
        ----------
        df : DataFrame
            分组数据
        surface : str
            'Top' 或 'Bot'
        value : float
            设定厚度值
        save_dir : str, optional
            自定义保存目录
        
        Returns
        -------
        results : dict
            分析结果
        """
        if save_dir is None:
            save_dir = str(self.save_dir / f'{surface}_{value}')
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        results = {
            'group_key': f'{surface}_{value}',
            'surface': surface,
            'value': value,
            'n_samples': len(df),
            'data_quality': self._check_data_quality(df),
            'plots_generated': False,
        }
        
        # 保存数据质量统计
        quality_file = save_path / 'data_quality.txt'
        with open(quality_file, 'w', encoding='utf-8') as f:
            f.write(f"数据质量报告: {surface}_{value}\n")
            f.write("=" * 60 + "\n")
            for key, val in results['data_quality'].items():
                f.write(f"{key}: {val}\n")
        
        logger.info(f"完成基础EDA: {surface}_{value}, 已保存到 {save_path}")
        
        return results
    
    def _check_data_quality(self, df: pd.DataFrame) -> Dict:
        """
        检查数据质量
        
        Parameters
        ----------
        df : DataFrame
            数据
        
        Returns
        -------
        quality_report : dict
            质量报告
        """
        quality = {
            'total_samples': len(df),
            'missing_values': df.isna().sum().sum(),
            'missing_ratio': (df.isna().sum().sum()) / (len(df) * len(df.columns)) if len(df) * len(df.columns) > 0 else 0,
            'numeric_cols': df.select_dtypes(include=[np.number]).shape[1],
            'columns': df.shape[1],
        }
        
        return quality
    
    def analyze_single_group_advanced(
        self,
        df: pd.DataFrame,
        surface: str,
        value: float,
        save_dir: Optional[str] = None
    ) -> Dict:
        """
        进行高级EDA分析（生成分布图、散点图等）
        复用 SurfaceEDAAnalyzer.analyze() 方法
        
        Parameters
        ----------
        df : DataFrame
            分组数据
        surface : str
            'Top' 或 'Bot'
        value : float
            设定厚度值
        save_dir : str, optional
            自定义保存目录
        
        Returns
        -------
        results : dict
            分析结果
        """
        if save_dir is None:
            save_dir = str(self.save_dir / f'{surface}_{value}')
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        target_col = get_target_column(surface)
        feature_cols = get_feature_columns(surface)
        
        # 基础信息
        results = {
            'group_key': f'{surface}_{value}',
            'surface': surface,
            'value': value,
            'n_samples': len(df),
            'data_quality': self._check_data_quality(df),
            'plots_generated': False,
        }
        
        # 保存数据质量统计
        quality_file = save_path / 'data_quality.txt'
        with open(quality_file, 'w', encoding='utf-8') as f:
            f.write(f"数据质量报告: {surface}_{value}\n")
            f.write("=" * 60 + "\n")
            for key, val in results['data_quality'].items():
                f.write(f"{key}: {val}\n")
        
        # 目标变量统计
        if target_col in df.columns:
            target_data = df[target_col].dropna()
            if len(target_data) > 0:
                results['target_stats'] = {
                    'mean': float(target_data.mean()),
                    'std': float(target_data.std()),
                    'min': float(target_data.min()),
                    'max': float(target_data.max()),
                    'median': float(target_data.median()),
                    'count': len(target_data),
                }
                
                # 保存目标变量统计
                target_file = save_path / 'target_stats.txt'
                with open(target_file, 'w', encoding='utf-8') as f:
                    f.write(f"目标变量统计: {surface}_{value}\n")
                    f.write("=" * 60 + "\n")
                    for key, val in results['target_stats'].items():
                        f.write(f"{key}: {val:.6f}\n")
        
        # 调用 SurfaceEDAAnalyzer 生成图表
        analyzer = self._get_surface_eda_analyzer()
        if analyzer is not None and len(df) > 0:
            try:
                # 准备输出目录
                plot_dir = str(save_path / 'overall')
                Path(plot_dir).mkdir(parents=True, exist_ok=True)
                
                logger.info(f"  → 使用 SurfaceEDAAnalyzer 生成可视化图表...")
                
                # 调用 analyze 方法生成分布图、散点图等
                eda_result = analyzer.analyze(
                    df=df,
                    surface=surface,  # 'Top' 或 'Bot'
                    delta_col=target_col,  # 目标列
                    feature_cols=feature_cols,  # 特征列表
                    save_dir=plot_dir,
                    plot_univariate=True,  # 生成分布图
                    plot_vs_delta=True,  # 生成散点图
                    plot_time=True,  # 生成时间序列图
                    enable_overall=True,  # 生成整体分析
                    enable_by_group=False,  # 不进行再次分组
                )
                
                results['plots_generated'] = True
                results['eda_analysis_detail'] = eda_result
                logger.info(f"  → 可视化图表已生成，保存到 {plot_dir}")
                
            except Exception as e:
                logger.warning(f"  → 生成可视化图表时出错: {e}")
                # 不中断流程，继续进行
        
        logger.info(f"完成高级EDA: {surface}_{value}, 已保存到 {save_path}")
        
        return results
    
    def analyze_all_groups(
        self,
        groups: Dict,
        advanced: bool = True,
    ) -> Dict:
        """
        分析所有表面分组
        
        Parameters
        ----------
        groups : dict
            分组信息 {group_key: (surface, value, group_df, n_samples)}
        advanced : bool
            是否进行高级分析（包括图表生成）
        
        Returns
        -------
        all_results : dict
            所有分组的分析结果
        """
        all_results = {}
        success_count = 0
        error_count = 0
        
        for group_key, (surface, value, group_df, n_samples) in groups.items():
            logger.info(f"正在进行EDA分析 {group_key} ({n_samples} 样本)...")
            
            try:
                if advanced:
                    results = self.analyze_single_group_advanced(
                        df=group_df,
                        surface=surface,
                        value=value,
                    )
                else:
                    results = self.analyze_single_group_basic(
                        df=group_df,
                        surface=surface,
                        value=value,
                    )
                
                all_results[group_key] = results
                if results.get('plots_generated', False):
                    success_count += 1
                
            except Exception as e:
                logger.error(f"分析 {group_key} 时出错: {e}")
                error_count += 1
                # 记录失败但继续处理其他分组
                all_results[group_key] = {
                    'group_key': group_key,
                    'surface': surface,
                    'value': value,
                    'error': str(e),
                }
        
        logger.info(f"\n{'='*60}")
        logger.info(f"EDA分析完成!")
        logger.info(f"  - 总分组数: {len(all_results)}")
        logger.info(f"  - 成功生成图表: {success_count}")
        logger.info(f"  - 遇到错误: {error_count}")
        logger.info(f"{'='*60}")
        
        return all_results
    
    def generate_eda_report(self, all_results: Dict) -> pd.DataFrame:
        """
        生成EDA汇总报告
        
        Parameters
        ----------
        all_results : dict
            所有分组的EDA结果
        
        Returns
        -------
        report : DataFrame
            汇总报告表
        """
        rows = []
        for group_key, result in all_results.items():
            row = {
                'Group_Key': group_key,
                'Surface': result.get('surface', ''),
                'Setpoint_Value': result.get('value', np.nan),
                'Sample_Count': result.get('n_samples', 0),
                'Missing_Ratio': result.get('data_quality', {}).get('missing_ratio', 0),
            }
            
            if 'target_stats' in result:
                target_stats = result['target_stats']
                row['Target_Mean'] = target_stats.get('mean', np.nan)
                row['Target_Std'] = target_stats.get('std', np.nan)
                row['Target_Min'] = target_stats.get('min', np.nan)
                row['Target_Max'] = target_stats.get('max', np.nan)
            
            rows.append(row)
        
        report_df = pd.DataFrame(rows)
        
        # 保存到文件
        report_path = self.save_dir / 'eda_summary_report.csv'
        report_df.to_csv(report_path, index=False, encoding='utf-8-sig')
        logger.info(f"EDA汇总报告已保存: {report_path}")
        
        return report_df
