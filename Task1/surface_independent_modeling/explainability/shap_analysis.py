"""
SHAP可解释性分析模块
"""
import logging
from typing import Dict, Optional, List
from pathlib import Path
import numpy as np
import pandas as pd

from ..core import EXPLAINABILITY_DIR

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    SHAP可解释性分析
    """
    
    def __init__(self, save_dir: Optional[str] = None):
        """
        初始化SHAP分析器
        
        Parameters
        ----------
        save_dir : str, optional
            结果保存目录
        """
        if save_dir is None:
            save_dir = str(EXPLAINABILITY_DIR)
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._shap_ready = False
    
    def _check_shap_availability(self) -> bool:
        """检查SHAP库是否可用"""
        if self._shap_ready:
            return True
        
        try:
            import shap
            self._shap_ready = True
            logger.info("SHAP库可用")
            return True
        except ImportError:
            logger.warning("SHAP库未安装，跳过SHAP分析")
            logger.info("安装SHAP: pip install shap")
            return False
    
    def analyze_single_model(
        self,
        model,
        X_train,
        X_test,
        y_test,
        feature_names: List[str],
        group_key: str,
        save_dir: Optional[str] = None,
        n_samples: int = 100,
    ) -> Dict:
        """
        分析单个模型的SHAP值
        
        Parameters
        ----------
        model : object
            已训练的模型
        X_train : DataFrame
            训练数据特征
        X_test : DataFrame
            测试数据特征
        y_test : Series
            测试数据标签
        feature_names : list
            特征名列表
        group_key : str
            分组键
        save_dir : str, optional
            保存目录
        n_samples : int
            用于背景数据的样本数
        
        Returns
        -------
        results : dict
            分析结果
        """
        if not self._check_shap_availability():
            logger.warning(f"跳过SHAP分析: {group_key}")
            return {'status': 'skipped', 'reason': 'SHAP not available'}
        
        if save_dir is None:
            save_dir = self.save_dir / group_key
        else:
            save_dir = Path(save_dir) / group_key
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            import shap
            import matplotlib.pyplot as plt
            
            # 处理样本数
            n_background = min(n_samples, len(X_train))
            X_background = X_train.sample(n=n_background, random_state=42)
            
            # 创建explainer (使用TreeExplainer或KernelExplainer)
            try:
                # 首先尝试TreeExplainer (对XGBoost/LightGBM/CatBoost等)
                if hasattr(model, 'predict'):
                    explainer = shap.KernelExplainer(
                        model=self._make_predictor(model),
                        data=shap.sample(X_background, 50),
                    )
                else:
                    logger.warning(f"模型不支持SHAP分析: {group_key}")
                    return {'status': 'skipped', 'reason': 'model not compatible'}
            except Exception as e:
                logger.warning(f"无法创建SHAP explainer: {e}")
                return {'status': 'skipped', 'reason': str(e)}
            
            # 计算SHAP值
            logger.info(f"计算SHAP值: {group_key}")
            shap_values = explainer.shap_values(X_test)
            
            # 如果返回的是多个数组（多输出），取第一个
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # 绘制摘要图
            try:
                fig, ax = plt.subplots(figsize=(10, 8))
                shap.summary_plot(
                    shap_values, X_test,
                    feature_names=feature_names,
                    show=False,
                    plot_type='bar',
                )
                plt.title(f'SHAP Summary Plot - {group_key}')
                plt.tight_layout()
                
                summary_file = save_dir / 'shap_summary_plot.png'
                plt.savefig(summary_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"SHAP摘要图已保存: {summary_file}")
            except Exception as e:
                logger.warning(f"绘制SHAP摘要图失败: {e}")
            
            return {
                'group_key': group_key,
                'shap_values': shap_values,
                'explainer': explainer,
                'save_dir': str(save_dir),
                'status': 'success',
            }
        
        except Exception as e:
            logger.error(f"SHAP分析失败: {group_key}, 错误: {e}")
            return {
                'status': 'failed',
                'error': str(e),
            }
    
    def _make_predictor(self, model):
        """
        为模型创建预测函数
        
        Parameters
        ----------
        model : object
            模型对象
        
        Returns
        -------
        predictor : callable
            预测函数
        """
        def predictor(X):
            if hasattr(model, 'predict_smooth'):
                # ResidualCorrectionModel 的情况
                try:
                    results = []
                    for row in X.values if hasattr(X, 'values') else X:
                        pred, _ = model.predict_smooth(
                            pd.DataFrame([row], columns=X.columns if hasattr(X, 'columns') else None),
                            pd.Series([row[0]] * len(row)) if isinstance(row, np.ndarray) else None,
                        )
                        results.append(pred.values[0] if hasattr(pred, 'values') else pred[0])
                    return np.array(results)
                except Exception as e:
                    logger.warning(f"predict_smooth失败，尝试predict: {e}")
                    return model.predict(X) if hasattr(model, 'predict') else np.zeros(len(X))
            elif hasattr(model, 'predict'):
                return model.predict(X)
            else:
                logger.error("模型没有predict方法")
                return np.zeros(len(X))
        
        return predictor
