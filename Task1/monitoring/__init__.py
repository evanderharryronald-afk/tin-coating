"""Task1 monitoring sub-package.

所有新增功能全部放在本目录，零侵入 Task1/ 原模块。
依赖关系：monitoring → Task1（coating_model_by_group / model_persistence 等），
Task1 永远不会 import monitoring。
"""

from .registry import ModelRegistry

__all__ = ["ModelRegistry"]
