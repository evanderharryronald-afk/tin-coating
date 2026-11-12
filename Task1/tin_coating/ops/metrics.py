"""极简指标注册表，输出 Prometheus 文本格式。

只做三件事：counter / gauge / histogram。无人值守时靠 /metrics 观察
请求量、错误率、P95 延迟、模型版本，不需要额外引入 prometheus_client。
"""
from __future__ import annotations

import math
import threading
import time
from collections import defaultdict

_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


def _key(name: str, labels: dict | None) -> str:
    if not labels:
        return name
    inner = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    return f"{name}{{{inner}}}"


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._hist: dict[str, dict] = {}
        self._started = time.time()

    # ---- 写入 -------------------------------------------------------
    def inc(self, name: str, value: float = 1.0, labels: dict | None = None):
        with self._lock:
            self._counters[_key(name, labels)] += value

    def set_gauge(self, name: str, value: float, labels: dict | None = None):
        with self._lock:
            self._gauges[_key(name, labels)] = float(value)

    def observe(self, name: str, seconds: float, labels: dict | None = None):
        k = _key(name, labels)
        with self._lock:
            h = self._hist.setdefault(k, {
                "count": 0, "sum": 0.0,
                "buckets": {b: 0 for b in _BUCKETS},
            })
            h["count"] += 1
            h["sum"] += float(seconds)
            for b in _BUCKETS:
                if seconds <= b:
                    h["buckets"][b] += 1

    # ---- 输出 -------------------------------------------------------
    def render(self) -> str:
        lines: list[str] = []
        with self._lock:
            uptime = time.time() - self._started
            lines.append("# HELP tin_coating_uptime_seconds 进程运行时长")
            lines.append("# TYPE tin_coating_uptime_seconds gauge")
            lines.append(f"tin_coating_uptime_seconds {uptime:.3f}")

            for k, v in sorted(self._counters.items()):
                base = k.split("{", 1)[0]
                lines.append(f"# TYPE {base} counter")
                lines.append(f"{k} {v:g}")

            for k, v in sorted(self._gauges.items()):
                base = k.split("{", 1)[0]
                lines.append(f"# TYPE {base} gauge")
                lines.append(f"{k} {v:g}")

            for k, h in sorted(self._hist.items()):
                base = k.split("{", 1)[0]
                lbl = ""
                if "{" in k:
                    lbl = "," + k.split("{", 1)[1][:-1]
                lines.append(f"# TYPE {base} histogram")
                for b in _BUCKETS:
                    lines.append(f'{base}_bucket{{le="{b}"{lbl}}} {h["buckets"][b]}')
                lines.append(f'{base}_bucket{{le="+Inf"{lbl}}} {h["count"]}')
                lines.append(f"{base}_sum{lbl and '{' + lbl[1:] + '}' or ''} {h['sum']:.6f}")
                lines.append(f"{base}_count{lbl and '{' + lbl[1:] + '}' or ''} {h['count']}")
        return "\n".join(lines) + "\n"

    def reset(self):
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._hist.clear()


METRICS = Metrics()
