"""
统一的训练接口 - 复用 fit_and_evaluate_surface
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Optional, Any
from pathlib import Path

from ..core import prepare_xy, get_feature_columns, get_target_column
from .model import SurfaceGroupModel

logger = logging.getLogger(__name__)


class SurfaceGroupTrainer:
    """
    按表面分组进行模型训练
    复用 coating_model_by_group.py 的 fit_and_evaluate_surface
    """
    
    def __init__(self, save_dir: Optional[str] = None):
        """
        初始化训练器
        
        Parameters
        ----------
        save_dir : str, optional
            模型保存目录
        """
        self.save_dir = Path(save_dir) if save_dir else Path('result/models')
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.trained_models = SurfaceGroupModel()
    
    def train_single_group(
        self,
        df: pd.DataFrame,
        surface: str,
        value: float,
        params: Optional[Dict] = None,
        train_ratio: float = 0.65,
        val_ratio: float = 0.20,
    ) -> Tuple[Any, Dict, Dict]:
        """
        训练单个表面分组的模型
        
        Parameters
        ----------
        df : DataFrame
            分组数据
        surface : str
            'Top' 或 'Bot'
        value : float
            设定厚度值
        params : dict, optional
            超参数
        train_ratio : float
            训练集比例
        val_ratio : float
            验证集比例
        
        Returns
        -------
        model, metrics, aux : tuple
            模型、指标、辅助信息
        """
        logger.info(f"开始训练 {surface}_{value} 的模型...")
        
        try:
            # 导入现有的训练函数
            from coating_model_by_group import fit_and_evaluate_surface
            
            group_tag = f"{surface}_{value}"
            
            # 如果没有提供超参数，使用默认参数
            if params is None or not params:
                params = {
                    'damping': 0.0,
                    'pos_boost': 1.0,
                    'alpha_smoothing': 0.7,
                    'max_iter': 200,
                    'learning_rate': 0.05,
                    'max_depth': 4,
                }
                logger.info(f"  → 使用默认参数: {params}")
            
            # 调用现有的训练函数
            model, metrics, aux = fit_and_evaluate_surface(
                df=df,
                surface=surface,
                params=params,
                group_tag=group_tag,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
            )
            
            logger.info(f"完成训练 {surface}_{value}")
            
            # 从原始数据中提取训练/验证/测试集数据用于可解释性分析
            # fit_and_evaluate_surface 已经按时间顺序切分了数据
            n_samples = len(df)
            train_end = int(n_samples * train_ratio)
            val_end = int(n_samples * (train_ratio + val_ratio))
            test_end = n_samples
            
            # 保护：确保每个集合至少有一定样本
            min_size = max(10, int(n_samples * 0.05))
            if train_end < min_size:
                train_end = min_size
            if val_end - train_end < min_size:
                val_end = min(train_end + min_size, n_samples - min_size)
            if test_end - val_end < min_size:
                val_end = max(train_end, n_samples - min_size)
            
            # 按照时间顺序切分数据
            train_data = df.iloc[:train_end].copy()
            val_data = df.iloc[train_end:val_end].copy()
            test_data = df.iloc[val_end:].copy()
            
            # 添加到aux中用于可解释性分析
            aux['train_data'] = train_data
            aux['val_data'] = val_data
            aux['test_data'] = test_data
            
            # 添加模型训练时使用的特征列信息
            # 根据 fit_and_evaluate_surface 中的特征定义
            prefix = 'Top' if surface == 'Top' else 'Bot'
            speed_col = 'Speed[m/min]_Process_Avg'
            current_col = f'{prefix}_Current_Sum'
            online_col = f'Tin Weight_Actual[g/m2]_GALV_WEIGHT_{prefix.upper()}_Avg'
            
            feature_cols = [
                online_col,
                current_col,
                speed_col,
                'Dimension_[mm]_Width',
                'Dimension_[mm]_Thickness',
            ]
            aux['feature_cols'] = feature_cols
            
            # 保存模型到字典
            metadata = {
                'surface': surface,
                'setpoint_value': value,
                'group_tag': group_tag,
                'train_samples': len(train_data),
                'val_samples': len(val_data),
                'test_samples': len(test_data),
            }
            
            group_key = f'{surface}_{value}'
            self.trained_models.add_model(group_key, model, metadata)
            
            return model, metrics, aux
        
        except ImportError as e:
            logger.error(f"无法导入训练函数: {e}")
            raise
        except Exception as e:
            logger.error(f"训练失败 {surface}_{value}: {e}")
            raise
    
    def train_all_groups(
        self,
        groups: Dict,
        params_dict: Optional[Dict] = None,
        train_ratio: float = 0.65,
        val_ratio: float = 0.20,
    ) -> Dict:
        """
        训练所有表面分组的模型
        
        Parameters
        ----------
        groups : dict
            分组信息 {group_key: (surface, value, group_df, n_samples)}
        params_dict : dict, optional
            超参数字典 {group_key: params}，如果为空或None则使用默认参数
        train_ratio : float
            训练集比例
        val_ratio : float
            验证集比例
        
        Returns
        -------
        all_training_results : dict
            所有训练结果 {group_key: (model, metrics, aux)}
        """
        all_training_results = {}
        
        # 如果 params_dict 为空，使用默认参数
        if not params_dict:
            logger.info("未提供超参数，将使用默认参数进行训练")
            params_dict = {}
        
        for group_key, (surface, value, group_df, n_samples) in groups.items():
            logger.info(f"训练 {group_key} ({n_samples} 样本)...")
            
            # 获取该分组的超参数（如果有的话）
            params = params_dict.get(group_key, None)
            
            if params is None:
                logger.info(f"  → 使用默认参数训练 {group_key}")
            else:
                logger.info(f"  → 使用优化参数训练 {group_key}")
            
            try:
                model, metrics, aux = self.train_single_group(
                    df=group_df,
                    surface=surface,
                    value=value,
                    params=params,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                )
                
                all_training_results[group_key] = {
                    'model': model,
                    'metrics': metrics,
                    'aux': aux,
                    'status': 'success',
                }
            
            except Exception as e:
                logger.error(f"训练失败 {group_key}: {e}")
                import traceback
                traceback.print_exc()
                all_training_results[group_key] = {
                    'model': None,
                    'metrics': None,
                    'aux': None,
                    'status': 'failed',
                    'error': str(e),
                }
        
        logger.info(f"完成所有模型训练，共 {len(all_training_results)} 个分组")
        
        return all_training_results
    
    def get_trained_models(self) -> SurfaceGroupModel:
        """
        获取训练好的模型集合
        
        Returns
        -------
        models : SurfaceGroupModel
            训练好的模型集合
        """
        return self.trained_models
    
    def save_trained_models(self, models: SurfaceGroupModel, output_dir: str) -> None:
        """
        保存训练好的模型
        
        Parameters
        ----------
        models : SurfaceGroupModel
            模型集合
        output_dir : str
            输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 使用 model_persistence 来保存模型
        try:
            from model_persistence import ModelSaver
            
            saver = ModelSaver(base_dir=str(output_path))
            
            for group_key in models.get_all_group_keys():
                model, metadata = models.get_model(group_key)
                
                # 解析 group_key 获取 surface 和 value
                parts = group_key.split('_')
                surface = parts[0]
                value = float('_'.join(parts[1:]))
                
                saver.save_model(
                    model=model,
                    metadata=metadata,
                    surface=surface,
                    group_label=f'{surface}_{value}'
                )
                
                logger.info(f"保存模型: {group_key}")
        
        except ImportError:
            logger.warning("未找到 model_persistence，使用默认保存方法")
            # 使用 joblib 直接保存
            import joblib
            for group_key in models.get_all_group_keys():
                model, _ = models.get_model(group_key)
                model_file = output_path / f'{group_key}_model.joblib'
                joblib.dump(model, model_file)
                logger.info(f"保存模型: {model_file}")
        
        # 保存元数据
        models.save_metadata(str(output_path))
