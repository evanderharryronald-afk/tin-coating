"""Task1 monitoring sub-package.

所有监控/重训/版本管理相关脚本都在本目录。
依赖关系：monitoring → tin_coating 子包（单向）。
"""

from .registry import ModelRegistry

__all__ = ["ModelRegistry"]
