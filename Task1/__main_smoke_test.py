"""全局自检入口 — 直接运行即等价于 `pytest tests/test_smoke.py -v`。

跑法二选一:
  python __main_smoke_test.py
  pytest tests/test_smoke.py -v

两个入口都执行同一套 9 项 smoke tests。
"""
from __future__ import annotations
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

# 检查 pytest 是否可用
try:
    import pytest  # noqa: F401
except ImportError:
    sys.stderr.write(
        "pytest 未安装，请先: pip install pytest\n"
    )
    sys.exit(1)

rc = pytest.main(["-v", "tests/test_smoke.py"])
sys.exit(0 if rc == 0 else 1)
