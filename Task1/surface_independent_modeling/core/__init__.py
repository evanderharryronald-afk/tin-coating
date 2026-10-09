"""
核心模块包
"""
from .constants import *
from .grouping import (
    build_surface_groups,
    filter_valid_groups,
    summarize_groups,
    get_specs_in_group,
)
from .data_loader import (
    load_featured_data,
    filter_by_surface_group,
    prepare_train_val_test_split,
    get_feature_columns,
    get_target_column,
    prepare_xy,
    validate_data_quality,
)

__all__ = [
    # constants
    'TOP_SETPOINT_COL', 'BOT_SETPOINT_COL',
    'OUTPUT_DIR', 'GROUPED_DATA_DIR', 'CORRELATION_DIR', 'EDA_DIR',
    'TUNING_DIR', 'MODELS_DIR', 'EVALUATION_DIR', 'EXPLAINABILITY_DIR',
    # grouping
    'build_surface_groups', 'filter_valid_groups', 'summarize_groups', 'get_specs_in_group',
    # data_loader
    'load_featured_data', 'filter_by_surface_group', 'prepare_train_val_test_split',
    'get_feature_columns', 'get_target_column', 'prepare_xy', 'validate_data_quality',
]
