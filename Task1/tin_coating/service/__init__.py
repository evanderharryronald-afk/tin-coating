"""在线推理服务：std model HTTP server + 契约校验 + 模型热加载。"""
from .contract import raw_required_cols, validate_rows  # noqa: F401
from .models import ModelStore  # noqa: F401
