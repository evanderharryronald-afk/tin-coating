"""
模型持久化模块：统一处理模型的保存和加载

特性：
1. 支持整体模型（overall）和分规格组模型（by_group）
2. 自动生成目录结构
3. 每个模型配套 metadata.json
4. 统一的保存和加载接口
"""

import os
import json
import joblib
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional


class ModelMetadata:
    """模型元数据管理"""
    
    def __init__(self, 
                 model_type: str,
                 surface: str,
                 feature_names: list,
                 hyperparameters: dict,
                 group_label: Optional[str] = None,
                 training_samples: int = 0,
                 training_metrics: Optional[dict] = None,
                 source_script: str = "unknown"):
        """
        Args:
            model_type: 模型类名，如 'ResidualCorrectionModel'
            surface: 'Top' 或 'Bot'
            feature_names: 特征列表
            hyperparameters: 模型超参数字典
            group_label: 规格组标签（整体模型时为 None）
            training_samples: 训练样本数
            training_metrics: 训练指标字典
            source_script: 源脚本名称
        """
        self.model_type = model_type
        self.surface = surface
        self.group_label = group_label
        self.feature_names = feature_names
        self.hyperparameters = hyperparameters
        self.training_samples = training_samples
        self.training_metrics = training_metrics or {}
        self.source_script = source_script
        self.saved_at = datetime.now().isoformat()
    
    def _serialize_value(self, value):
        """将各种类型的值转换为 JSON 可序列化的形式"""
        import pandas as pd
        import numpy as np
        
        if value is None:
            return None
        elif isinstance(value, (str, int, float, bool)):
            return value
        elif isinstance(value, (np.integer, np.floating)):
            return float(value) if isinstance(value, np.floating) else int(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, pd.DataFrame):
            # DataFrame 转换为字典格式
            return value.to_dict(orient='records')[:5]  # 只保存前5行以减少文件大小
        elif isinstance(value, pd.Series):
            return value.to_dict()
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        else:
            # 其他类型尝试转换为字符串
            return str(value)
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'model_type': self.model_type,
            'surface': self.surface,
            'group_label': self.group_label,
            'feature_names': self.feature_names,
            'hyperparameters': self._serialize_value(self.hyperparameters),
            'training_samples': self.training_samples,
            'training_metrics': self._serialize_value(self.training_metrics),
            'source_script': self.source_script,
            'saved_at': self.saved_at,
        }
    
    @staticmethod
    def from_dict(data: dict) -> 'ModelMetadata':
        """从字典创建元数据"""
        return ModelMetadata(
            model_type=data.get('model_type', 'unknown'),
            surface=data.get('surface', 'Top'),
            feature_names=data.get('feature_names', []),
            hyperparameters=data.get('hyperparameters', {}),
            group_label=data.get('group_label'),
            training_samples=data.get('training_samples', 0),
            training_metrics=data.get('training_metrics', {}),
            source_script=data.get('source_script', 'unknown'),
        )


class ModelSaver:
    """统一的模型保存器"""
    
    def __init__(self, base_dir: str = "result/trained_models", 
                 compress: bool = True):
        """
        Args:
            base_dir: 模型保存的基础目录
            compress: 是否压缩模型文件
        """
        self.base_dir = Path(base_dir)
        self.compress = compress
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def _serialize_for_json(self, obj):
        """将对象转换为 JSON 可序列化的形式"""
        import pandas as pd
        import numpy as np
        
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            # DataFrame 转换为字典格式（只保存前5行）
            return {
                'type': 'DataFrame',
                'shape': list(obj.shape),
                'data': obj.head(5).to_dict(orient='records') if len(obj) > 0 else []
            }
        elif isinstance(obj, pd.Series):
            return {
                'type': 'Series',
                'data': obj.to_dict()
            }
        elif isinstance(obj, dict):
            return {k: self._serialize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._serialize_for_json(v) for v in obj]
        else:
            # 其他类型转换为字符串
            return str(obj)
    
    def _get_model_dir(self, surface: str, group_label: Optional[str] = None) -> Path:
        """获取模型保存目录"""
        if group_label is None:
            # 整体模型
            model_dir = self.base_dir / "overall"
        else:
            # 分规格组模型
            model_dir = self.base_dir / "by_group" / group_label
        
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir
    
    def save_model(self, 
                   model: Any,
                   metadata: ModelMetadata,
                   surface: Optional[str] = None,
                   group_label: Optional[str] = None) -> Dict[str, str]:
        """
        保存单个模型及其元数据
        
        Args:
            model: 训练好的模型对象
            metadata: 模型元数据
            surface: 表面类型（可从 metadata 中获取，此参数为冗余确认）
            group_label: 规格组标签（可从 metadata 中获取，此参数为冗余确认）
        
        Returns:
            {'model_path': ..., 'metadata_path': ...}
        """
        # 使用 metadata 中的值，允许参数覆盖
        if surface is None:
            surface = metadata.surface
        if group_label is None:
            group_label = metadata.group_label
        
        # 验证表面值
        if surface not in ['Top', 'Bot']:
            raise ValueError(f"Invalid surface: {surface}. Must be 'Top' or 'Bot'")
        
        model_dir = self._get_model_dir(surface, group_label)
        
        # 生成文件名
        model_filename = f"{surface}_model.joblib"
        metadata_filename = f"{surface}_metadata.json"
        
        model_path = model_dir / model_filename
        metadata_path = model_dir / metadata_filename
        
        try:
            # 保存模型
            compress_level = 3 if self.compress else 0
            joblib.dump(model, str(model_path), compress=compress_level)
            print(f"[✓] 模型已保存: {model_path}")
            
            # 保存元数据（使用自定义序列化）
            metadata_dict = metadata.to_dict()
            # 再次清理序列化后的字典中的非 JSON 可序列化对象
            metadata_dict = self._serialize_for_json(metadata_dict)
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
            print(f"[✓] 元数据已保存: {metadata_path}")
            
            return {
                'model_path': str(model_path),
                'metadata_path': str(metadata_path)
            }
        
        except Exception as e:
            print(f"[✗] 保存失败: {e}")
            raise
    
    def save_batch(self, 
                   models_dict: Dict[Tuple[str, str], Tuple[Any, ModelMetadata]]) -> Dict:
        """
        批量保存多个模型
        
        Args:
            models_dict: {(group_label, surface): (model, metadata), ...}
                        或 {('overall', surface): (model, metadata), ...}
        
        Returns:
            保存结果汇总
        """
        results = {}
        
        for (group_label, surface), (model, metadata) in models_dict.items():
            try:
                save_result = self.save_model(
                    model=model,
                    metadata=metadata,
                    surface=surface,
                    group_label=group_label if group_label != 'overall' else None
                )
                results[f"{group_label}_{surface}"] = {
                    'status': 'success',
                    'paths': save_result
                }
            except Exception as e:
                results[f"{group_label}_{surface}"] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return results


class ModelLoader:
    """统一的模型加载器"""
    
    def __init__(self, base_dir: str = "result/trained_models"):
        """
        Args:
            base_dir: 模型保存的基础目录
        """
        self.base_dir = Path(base_dir)
    
    def _get_model_dir(self, surface: str, group_label: Optional[str] = None) -> Path:
        """获取模型保存目录"""
        if group_label is None:
            model_dir = self.base_dir / "overall"
        else:
            model_dir = self.base_dir / "by_group" / group_label
        
        return model_dir
    
    def load_model(self, 
                   surface: str,
                   group_label: Optional[str] = None,
                   fallback_to_overall: bool = True) -> Tuple[Any, ModelMetadata]:
        """
        加载单个模型
        
        Args:
            surface: 'Top' 或 'Bot'
            group_label: 规格组标签（None 表示加载整体模型）
            fallback_to_overall: 当分组模型不存在时，是否降级加载整体模型
        
        Returns:
            (model, metadata)
        
        Raises:
            FileNotFoundError: 模型文件不存在
        """
        if surface not in ['Top', 'Bot']:
            raise ValueError(f"Invalid surface: {surface}. Must be 'Top' or 'Bot'")
        
        model_dir = self._get_model_dir(surface, group_label)
        model_path = model_dir / f"{surface}_model.joblib"
        metadata_path = model_dir / f"{surface}_metadata.json"
        
        # 如果分组模型不存在，尝试降级到整体模型
        if not model_path.exists() and group_label is not None and fallback_to_overall:
            print("[!] 分组模型不存在，降级使用整体模型")
            return self.load_model(surface, group_label=None, fallback_to_overall=False)
        
        # 检查文件是否存在
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"元数据文件不存在: {metadata_path}")
        
        try:
            # 加载模型
            model = joblib.load(str(model_path))
            print(f"[OK] 模型已加载: {model_path}")
            
            # 加载元数据
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata_dict = json.load(f)
            metadata = ModelMetadata.from_dict(metadata_dict)
            print(f"[OK] 元数据已加载: {metadata_path}")
            
            return model, metadata
        
        except Exception as e:
            print(f"[ERROR] 加载失败: {e}")
            raise
    
    def list_available_models(self) -> Dict[str, Any]:
        """
        列出所有可用的模型
        
        Returns:
            {
                'overall': {'Top': metadata, 'Bot': metadata},
                'by_group': {
                    'group_label1': {'Top': metadata, 'Bot': metadata},
                    'group_label2': {'Top': metadata, 'Bot': metadata},
                }
            }
        """
        available = {
            'overall': {},
            'by_group': {}
        }
        
        overall_dir = self.base_dir / "overall"
        if overall_dir.exists():
            for surface in ['Top', 'Bot']:
                metadata_path = overall_dir / f"{surface}_metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            available['overall'][surface] = json.load(f)
                    except:
                        pass
        
        by_group_dir = self.base_dir / "by_group"
        if by_group_dir.exists():
            for group_dir in by_group_dir.iterdir():
                if group_dir.is_dir():
                    group_label = group_dir.name
                    available['by_group'][group_label] = {}
                    
                    for surface in ['Top', 'Bot']:
                        metadata_path = group_dir / f"{surface}_metadata.json"
                        if metadata_path.exists():
                            try:
                                with open(metadata_path, 'r', encoding='utf-8') as f:
                                    available['by_group'][group_label][surface] = json.load(f)
                            except:
                                pass
        
        return available


# 便捷函数
def save_model(model, metadata, surface=None, group_label=None):
    """快速保存单个模型"""
    saver = ModelSaver()
    return saver.save_model(model, metadata, surface, group_label)


def load_model(surface, group_label=None, fallback_to_overall=True):
    """快速加载单个模型"""
    loader = ModelLoader()
    return loader.load_model(surface, group_label, fallback_to_overall)


def list_models():
    """快速列出所有可用模型"""
    loader = ModelLoader()
    return loader.list_available_models()


if __name__ == "__main__":
    # 测试：列出可用模型
    print("可用的模型:")
    models = list_models()
    print(json.dumps(models, indent=2, ensure_ascii=False))
