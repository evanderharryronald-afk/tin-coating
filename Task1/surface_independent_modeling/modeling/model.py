"""
模型包装和统一接口 - 复用 ResidualCorrectionModel
"""
import logging
from typing import Dict, Tuple, Optional, Any
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class SurfaceGroupModel:
    """
    表面分组模型的统一接口
    复用 ResidualCorrectionModel
    """
    
    def __init__(self, models_dict: Optional[Dict] = None):
        """
        初始化模型集合
        
        Parameters
        ----------
        models_dict : dict, optional
            模型字典 {group_key: (model_object, metadata)}
        """
        self.models = models_dict or {}
    
    def add_model(
        self,
        group_key: str,
        model: Any,
        metadata: Dict,
    ) -> None:
        """
        添加一个模型
        
        Parameters
        ----------
        group_key : str
            模型键，格式: 'Top_value' 或 'Bot_value'
        model : object
            模型对象 (ResidualCorrectionModel)
        metadata : dict
            模型元数据
        """
        self.models[group_key] = (model, metadata)
        logger.info(f"添加模型: {group_key}")
    
    def get_model(self, group_key: str) -> Tuple[Any, Dict]:
        """
        获取指定的模型
        
        Parameters
        ----------
        group_key : str
            模型键
        
        Returns
        -------
        model, metadata : tuple
            模型对象和元数据
        """
        if group_key not in self.models:
            raise KeyError(f"模型不存在: {group_key}")
        
        return self.models[group_key]
    
    def predict(
        self,
        X,
        surface: str,
        value: float,
        online_actual=None,
    ):
        """
        使用指定的表面分组模型进行预测
        
        Parameters
        ----------
        X : DataFrame or array
            特征矩阵
        surface : str
            表面 ('Top' 或 'Bot')
        value : float
            设定厚度值
        online_actual : Series or array, optional
            在线测量值
        
        Returns
        -------
        predictions : array
            预测结果
        """
        group_key = f'{surface}_{value}'
        model, metadata = self.get_model(group_key)
        
        if online_actual is not None:
            predictions, _ = model.predict_smooth(X, online_actual)
        else:
            predictions = model.predict(X)
        
        return predictions
    
    def get_all_group_keys(self):
        """
        获取所有模型的键
        
        Returns
        -------
        keys : list
            所有模型键
        """
        return list(self.models.keys())
    
    def get_models_by_surface(self, surface: str):
        """
        获取某个表面的所有模型
        
        Parameters
        ----------
        surface : str
            表面 ('Top' 或 'Bot')
        
        Returns
        -------
        surface_models : dict
            该表面的所有模型
        """
        surface_models = {}
        for key, (model, metadata) in self.models.items():
            if key.startswith(surface):
                surface_models[key] = (model, metadata)
        
        return surface_models
    
    def get_metadata(self, group_key: str) -> Dict:
        """
        获取模型元数据
        
        Parameters
        ----------
        group_key : str
            模型键
        
        Returns
        -------
        metadata : dict
            元数据
        """
        if group_key not in self.models:
            raise KeyError(f"模型不存在: {group_key}")
        
        _, metadata = self.models[group_key]
        return metadata
    
    def save_metadata(self, save_dir: str) -> None:
        """
        保存所有模型的元数据
        
        Parameters
        ----------
        save_dir : str
            保存目录
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for group_key, (model, metadata) in self.models.items():
            metadata_file = save_path / f'{group_key}_metadata.json'
            
            # 元数据可能包含非JSON序列化的对象，需要处理
            try:
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
                logger.info(f"保存元数据: {metadata_file}")
            except Exception as e:
                logger.error(f"保存元数据失败: {group_key}, 错误: {e}")
