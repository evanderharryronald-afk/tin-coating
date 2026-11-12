"""tin_coating — 连续热镀锌线镀层重量偏差校正包。

顶层公共 API（pip install -e . 后可以直接 from tin_coating import ...）:

  from tin_coating import (
      ResidualCorrectionModel,           # HGB 残差校正模型
      run_cv_for_surface,                # Walk-Forward CV
      load_pipeline_config,              # 读 JSON 超参配置
      get_params_for_group,
      build_setpoint_group_key,
      summarize_setpoint_groups,
      get_feature_cols,                  # Top/Bot 各 8 特征列名
      PredictionPipeline,                # 推理管道
      SteelDataCleaner,                  # 数据清洗
  )

子包结构:
  tin_coating.data         # 清洗 / 特征工程 / 格式转换 / 合并
  tin_coating.modeling     # 核心模型 / CV / 持久化 / Optuna 调参
  tin_coating.inference    # PredictionPipeline 批量推理
  tin_coating.eda          # 探索性分析 / 相关性 / 分组 / 解释
  tin_coating.diagnostics  # 残差 / 漂移诊断
  tin_coating.ops          # 7x24 运维：日志 / 指标 / 锁 / 告警 / 自检 / 定时任务
  tin_coating.service      # 在线推理服务（/healthz /readyz /predict /metrics）
"""
from __future__ import annotations

from .modeling import (
    ResidualCorrectionModel,
    run_cv_for_surface,
    load_pipeline_config,
    get_params_for_group,
    build_setpoint_group_key,
    summarize_setpoint_groups,
    get_feature_cols,
    DEFAULT_PARAMS,
    MIN_GROUP_SAMPLES,
)
from .data.clean import SteelDataCleaner
from .inference.pipeline import PredictionPipeline

__all__ = [
    "ResidualCorrectionModel",
    "run_cv_for_surface",
    "load_pipeline_config",
    "get_params_for_group",
    "build_setpoint_group_key",
    "summarize_setpoint_groups",
    "get_feature_cols",
    "DEFAULT_PARAMS",
    "MIN_GROUP_SAMPLES",
    "SteelDataCleaner",
    "PredictionPipeline",
]
