"""
特征重要性分析模块
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, List
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from ..core import EXPLAINABILITY_DIR

logger = logging.getLogger(__name__)


class FeatureImportanceAnalyzer:
    """
    特征重要性分析
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
    
    def compute_feature_importance(
        self,
        model,
        X,
        y,
        method: str = 'permutation',
    ) -> Dict:
        """
        计算特征重要性
        
        Parameters
        ----------
        model : object
            已训练的模型
        X : DataFrame
            特征矩阵
        y : Series
            目标变量
        method : str
            方法: 'permutation' 或 'gain'
        
        Returns
        -------
        importance : dict
            特征重要性结果
        """
        importance = {}
        
        try:
            if method == 'permutation':
                from sklearn.inspection import permutation_importance
                
                # 获取模型的预测方法
                if hasattr(model, 'predict_smooth'):
                    # ResidualCorrectionModel 的情况
                    def predict_func(X_input):
                        if hasattr(X_input, 'shape'):
                            if len(X_input.shape) == 1:
                                X_input = X_input.reshape(1, -1)
                        return model.predict_smooth(X_input, X_input.iloc[:, 0])[0] if hasattr(X_input, 'iloc') else model.predict_smooth(X_input, X_input[:, 0])[0]
                    
                    # 获取模型的底层模型
                    if hasattr(model, 'model'):
                        base_model = model.model
                    else:
                        logger.warning("无法获取底层模型，跳过特征重要性计算")
                        return {}
                    
                    result = permutation_importance(
                        base_model, X, y,
                        n_repeats=10,
                        random_state=42,
                    )
                    
                    importance['feature_importance'] = result.importances_mean
                    importance['feature_std'] = result.importances_std
                    importance['method'] = 'permutation'
                else:
                    logger.warning("模型不支持特征重要性计算")
                    return {}
            
            elif method == 'gain':
                # 对于梯度提升模型，使用 feature_importances_
                if hasattr(model, 'feature_importances_'):
                    importance['feature_importance'] = model.feature_importances_
                    importance['method'] = 'gain'
                else:
                    logger.warning("模型不支持gain特征重要性")
                    return {}
            
            return importance
        
        except Exception as e:
            logger.error(f"计算特征重要性失败: {e}")
            return {}
    
    def rank_features(
        self,
        feature_importance_dict: Dict,
        feature_names: List[str],
    ) -> pd.DataFrame:
        """
        排序特征重要性
        
        Parameters
        ----------
        feature_importance_dict : dict
            特征重要性字典
        feature_names : list
            特征名列表
        
        Returns
        -------
        ranked_df : DataFrame
            排序后的特征重要性表
        """
        if not feature_importance_dict or 'feature_importance' not in feature_importance_dict:
            logger.warning("无有效的特征重要性数据")
            return pd.DataFrame()
        
        importance_values = feature_importance_dict['feature_importance']
        
        # 创建DataFrame
        ranked_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_values,
        })
        
        if 'feature_std' in feature_importance_dict:
            ranked_df['Std'] = feature_importance_dict['feature_std']
        
        ranked_df = ranked_df.sort_values('Importance', ascending=False).reset_index(drop=True)
        ranked_df['Rank'] = range(1, len(ranked_df) + 1)
        
        return ranked_df
    
    def visualize_importance(
        self,
        ranked_df: pd.DataFrame,
        top_n: int = 15,
        save_file: Optional[str] = None,
    ) -> str:
        """
        可视化特征重要性
        
        Parameters
        ----------
        ranked_df : DataFrame
            排序后的特征重要性表
        top_n : int
            显示的特征数量
        save_file : str, optional
            保存文件路径
        
        Returns
        -------
        save_path : str
            保存文件路径
        """
        if ranked_df.empty:
            logger.warning("无特征重要性数据可视化")
            return ""
        
        # 取top_n
        plot_df = ranked_df.head(top_n).copy()
        
        # 绘图
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
        
        bars = ax.barh(range(len(plot_df)), plot_df['Importance'], color='steelblue')
        ax.set_yticks(range(len(plot_df)))
        ax.set_yticklabels(plot_df['Feature'])
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {min(top_n, len(ranked_df))} Feature Importance')
        ax.invert_yaxis()
        
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, plot_df['Importance'])):
            ax.text(val, i, f' {val:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_file is None:
            save_file = self.save_dir / 'feature_importance.png'
        else:
            save_file = Path(save_file)
        
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"特征重要性图已保存: {save_file}")
        
        return str(save_file)
    
    def analyze_single_model(
        self,
        model,
        X,
        y,
        feature_names: List[str],
        group_key: str,
        save_dir: Optional[str] = None,
    ) -> Dict:
        """
        分析单个模型的特征重要性
        
        Parameters
        ----------
        model : object
            已训练的模型
        X : DataFrame
            特征矩阵
        y : Series
            目标变量
        feature_names : list
            特征名列表
        group_key : str
            分组键
        save_dir : str, optional
            保存目录
        
        Returns
        -------
        results : dict
            分析结果
        """
        if save_dir is None:
            save_dir = self.save_dir / group_key
        else:
            save_dir = Path(save_dir) / group_key
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 计算特征重要性
        importance_dict = self.compute_feature_importance(model, X, y)
        
        if not importance_dict:
            logger.warning(f"无法计算 {group_key} 的特征重要性")
            return {'status': 'failed'}
        
        # 排序
        ranked_df = self.rank_features(importance_dict, feature_names)
        
        # 保存排序结果
        ranked_file = save_dir / 'feature_importance.csv'
        ranked_df.to_csv(ranked_file, index=False, encoding='utf-8-sig')
        logger.info(f"特征重要性排序已保存: {ranked_file}")
        
        # 可视化
        plot_file = save_dir / 'feature_importance.png'
        self.visualize_importance(ranked_df, top_n=15, save_file=str(plot_file))
        
        return {
            'group_key': group_key,
            'ranked_df': ranked_df,
            'importance_dict': importance_dict,
            'save_dir': str(save_dir),
            'status': 'success',
        }
