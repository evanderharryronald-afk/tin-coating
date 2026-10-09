"""
表面独立分组建模包

使用表面厚度独立分组代替规格组合分组，进行建模分析。
"""

__version__ = "1.0.0"
__author__ = "Kiro AI"

from .pipeline import SurfaceIndependentModelingPipeline
from .core import (
    load_featured_data,
    build_surface_groups,
    filter_valid_groups,
    summarize_groups,
)
from .analysis import AnalysisRunner
from .modeling import (
    SurfaceGroupModel,
    SurfaceGroupTrainer,
    ModelEvaluator,
    SurfaceGroupHyperparameterTuner,
)
from .explainability import (
    FeatureImportanceAnalyzer,
    ResidualAnalyzer,
    SHAPExplainer,
)

__all__ = [
    'SurfaceIndependentModelingPipeline',
    'load_featured_data',
    'build_surface_groups',
    'filter_valid_groups',
    'summarize_groups',
    'AnalysisRunner',
    'SurfaceGroupModel',
    'SurfaceGroupTrainer',
    'ModelEvaluator',
    'SurfaceGroupHyperparameterTuner',
    'FeatureImportanceAnalyzer',
    'ResidualAnalyzer',
    'SHAPExplainer',
]
