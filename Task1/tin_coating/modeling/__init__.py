"""tin_coating.modeling — 训练、评估、持久化、CV 子包。

顶层导出（pip install 后可以直接 from tin_coating.modeling import ...）:

  from tin_coating.modeling import (
      ResidualCorrectionModel,          # HGB 残差校正模型
      LinearResidualCorrectionModel,    # Huber 线性版本（对照）
      compute_direction_sample_weight,
      DEFAULT_PARAMS,                   # 默认超参字典
      MIN_GROUP_SAMPLES,
      load_pipeline_config,             # 读 JSON 配置
      get_params_for_group,             # 合并 default + override
      build_corrector_instance,
      build_setpoint_group_key,         # Setpoint 分组
      summarize_setpoint_groups,
      get_feature_cols,                 # 8 特征列名
      run_cv_for_surface,               # Walk-Forward 5-fold CV
  )
"""
from .residual_model import (
    ResidualCorrectionModel,
    LinearResidualCorrectionModel,
    compute_direction_sample_weight,
)
from .params import (
    DEFAULT_PARAMS,
    MIN_GROUP_SAMPLES,
    load_pipeline_config,
    get_params_for_group,
    build_corrector_instance,
    build_setpoint_group_key,
    summarize_setpoint_groups,
    get_feature_cols,
)
from .walk_forward_cv import run_cv_for_surface

__all__ = [
    "ResidualCorrectionModel", "LinearResidualCorrectionModel",
    "compute_direction_sample_weight",
    "DEFAULT_PARAMS", "MIN_GROUP_SAMPLES",
    "load_pipeline_config", "get_params_for_group", "build_corrector_instance",
    "build_setpoint_group_key", "summarize_setpoint_groups", "get_feature_cols",
    "run_cv_for_surface",
]
