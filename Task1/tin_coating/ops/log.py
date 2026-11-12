"""结构化 JSON 日志。

7x24 无人值守场景下，日志必须可被采集器直接解析：一行一个 JSON，
带时间戳、级别、事件名和上下文字段；同时按大小轮转，避免磁盘被写满。
"""
from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONFIGURED: set[str] = set()


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc)
            .astimezone().isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        ctx = getattr(record, "ctx", None)
        if isinstance(ctx, dict):
            payload.update(ctx)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def get_logger(name: str = "tin_coating",
               log_dir: str | Path | None = None,
               level: str | None = None,
               also_stdout: bool = True) -> logging.Logger:
    """获取一个 JSON logger（同名只会配置一次）。"""
    logger = logging.getLogger(name)
    key = f"{name}|{log_dir}"
    if key in _CONFIGURED:
        return logger

    logger.setLevel((level or os.environ.get("TIN_LOG_LEVEL", "INFO")).upper())
    logger.propagate = False
    fmt = JsonFormatter()

    if log_dir:
        p = Path(log_dir)
        p.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(p / f"{name}.jsonl",
                                 maxBytes=10 * 1024 * 1024,
                                 backupCount=5, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    if also_stdout:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    _CONFIGURED.add(key)
    return logger


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def log_event(logger: logging.Logger, level: str, event: str, **fields):
    """一行一条业务事件：log_event(log, "INFO", "predict_ok", surface="Top", n=10)。"""
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    logger.log(lvl, event, extra={"ctx": {"event": event, **fields}})
