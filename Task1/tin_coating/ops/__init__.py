"""运维支撑模块：结构化日志、指标、文件锁、告警、启动自检。

这些模块只依赖标准库 + 已有第三方包，不引入新依赖，保证 7x24 部署环境简单。
"""
from .log import get_logger, new_request_id  # noqa: F401
from .metrics import METRICS, Metrics  # noqa: F401
from .lock import FileLock  # noqa: F401
from .notify import Notifier  # noqa: F401
from .selfcheck import run_selfcheck, assert_ready  # noqa: F401
