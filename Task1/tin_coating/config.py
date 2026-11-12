"""统一配置加载器 — 所有路径、阈值、默认规格组从这里读。

优先级（高 → 低）:
  1. 环境变量覆盖（TIN_COATING_ROOT / MONITORING_REGRESSION_Z_THRESHOLD / ...）
  2. configs/app.yaml 里的值（environment.dev / prod 可切换）
  3. fallback 默认值（写在代码里，用作最后兜底）

用法:
  from tin_coating.config import Config, load_config
  cfg = load_config()                           # 默认 dev 环境
  cfg = load_config(env="prod")                 # 切 prod
  cfg.paths.feature_data                        # → Path
  cfg.app.default_group                         # → str
  cfg.monitoring.regression_z_threshold         # → float
  cfg.resolve("paths.online_jobs")              # → Path (展开 ${app.default_group})

  作为独立脚本（monitoring/*.py）使用:
  import sys; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
  from tin_coating.config import load_config
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# 默认 Task1/ 根目录探测
# ---------------------------------------------------------------------------
def _detect_task1_root() -> Path:
    """优先读 env var TIN_COATING_ROOT，否则向上搜 pyproject.toml 或 configs/app.yaml。"""
    env = os.environ.get("TIN_COATING_ROOT", "").strip()
    if env:
        p = Path(env).resolve()
        if p.exists():
            return p

    # 向上搜：从当前工作目录
    for parent in [Path.cwd(), Path(__file__).resolve().parent.parent]:
        cur = parent
        while True:
            if (cur / "pyproject.toml").exists() or (cur / "configs" / "app.yaml").exists():
                return cur
            up = cur.parent
            if up == cur:
                break
            cur = up
    # 最后兜底：本文件在 tin_coating/config.py，往上两级就是 Task1/
    return Path(__file__).resolve().parent.parent


ROOT = _detect_task1_root()
APP_YAML = ROOT / "configs" / "app.yaml"


# ---------------------------------------------------------------------------
# ${var} 插值解析
# ---------------------------------------------------------------------------
_INTERPOLATION_RE = re.compile(r"\$\{([^}]+)\}")

def _resolve_dotted(context: dict, dotted: str):
    cur: Any = context
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _interpolate(value: Any, context: dict) -> Any:
    if isinstance(value, str):
        def _sub(m):
            dotted = m.group(1)
            resolved = _resolve_dotted(context, dotted)
            return str(resolved) if resolved is not None else m.group(0)
        return _INTERPOLATION_RE.sub(_sub, value)
    if isinstance(value, list):
        return [_interpolate(v, context) for v in value]
    if isinstance(value, dict):
        return {k: _interpolate(v, context) for k, v in value.items()}
    return value


# ---------------------------------------------------------------------------
# Config 类 — 点号访问 + path 自动 resolve
# ---------------------------------------------------------------------------
def _make_accessor(data: dict):
    """把 dict 递归包装成支持属性访问的对象。"""
    class _Accessor:
        def __init__(self, mapping):
            self._data = dict(mapping)
            self._keys = sorted(self._data.keys())
        def __getattr__(self, name):
            if name.startswith("_"):
                raise AttributeError(name)
            if name not in self._data:
                raise AttributeError(f"config has no key '{name}'")
            val = self._data[name]
            if isinstance(val, dict):
                return _make_accessor(val)
            return val
        def __getitem__(self, name):
            return self._data[name]
        def __contains__(self, name):
            return name in self._data
        def get(self, name, default=None):
            return self._data.get(name, default)
        def as_dict(self):
            return dict(self._data)
        def __repr__(self):
            return f"ConfigAccessor({self._keys})"
    return _Accessor(data)


class Config:
    """统一配置入口。"""

    def __init__(self, data: dict, root: Path):
        self._root = root
        self._data = data
        self.app = _make_accessor(data.get("app", {}))
        self.model = _make_accessor(data.get("model", {}))
        self.monitoring = _make_accessor(data.get("monitoring", {}))
        self.paths = _make_accessor(data.get("paths", {}))
        self.service = _make_accessor(data.get("service", {}))
        self.ops = _make_accessor(data.get("ops", {}))
        self.root = root

    def resolve(self, dotted_key: str) -> Path:
        """点号路径查找 + 转成绝对 Path（相对于 root）。

        例子: cfg.resolve("paths.online_jobs") → Path(".../result/trained_models/by_group/Top2.799_Bot2.799")
        """
        cur: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(cur, dict):
                raise KeyError(dotted_key)
            cur = cur[part]
        return self.root / Path(cur)

    def env_override(self, env_prefix: str = "TIN_COATING_"):
        """从环境变量覆盖配置，格式 TIN_COATING__<DOTTED_KEY>=value。

        例: TIN_COATING__MONITORING__REGRESSION_Z_THRESHOLD=-1.5
        """
        changed = []
        for k, v in os.environ.items():
            if not k.startswith(env_prefix):
                continue
            rest = k[len(env_prefix):]
            # 只接受 TIN_COATING__<DOTTED> 形式；TIN_COATING_ROOT / TIN_COATING_ENV 不参与覆盖
            if not rest.startswith("_"):
                continue
            dotted = rest[1:].lower().replace("__", ".")
            if not dotted:
                continue
            # 解析 dotted path
            cur = self._data
            parts = dotted.split(".")
            for p in parts[:-1]:
                cur = cur.setdefault(p, {})
            old = cur.get(parts[-1])
            cur[parts[-1]] = _coerce(v, old)
            changed.append(dotted)
        # 重建 accessor
        self.app = _make_accessor(self._data.get("app", {}))
        self.model = _make_accessor(self._data.get("model", {}))
        self.monitoring = _make_accessor(self._data.get("monitoring", {}))
        self.paths = _make_accessor(self._data.get("paths", {}))
        self.service = _make_accessor(self._data.get("service", {}))
        self.ops = _make_accessor(self._data.get("ops", {}))
        return changed


def _coerce(raw: str, old: Any) -> Any:
    """把 env var 字符串转成合理的类型。"""
    if isinstance(old, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(old, int):
        try: return int(raw)
        except ValueError: return old
    if isinstance(old, float):
        try: return float(raw)
        except ValueError: return old
    return raw


def load_config(env: str | None = None, config_path: Path | str | None = None) -> Config:
    """加载配置。env=None 时读 env var 里的 TIN_COATING_ENV，默认 dev。"""
    path = Path(config_path) if config_path else APP_YAML
    if not path.exists():
        # fallback — 返回硬编码默认值，不崩溃（方便单元测试）
        return Config(_FALLBACK_DATA, ROOT)

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # 合并 environment 段 (dev / prod)
    env_key = env or os.environ.get("TIN_COATING_ENV", "dev")
    env_section = raw.get("environment", {}).get(env_key, {})
    merged = _deep_merge(raw, env_section)
    merged.pop("environment", None)

    # 插值 + env var 覆盖
    merged = _interpolate(merged, merged)
    cfg = Config(merged, ROOT)
    cfg.env_override()
    return cfg


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# 硬编码 fallback（yaml 不存在时的兜底值，保证 import 安全）
# ---------------------------------------------------------------------------
_FALLBACK_DATA = {
    "app": {"default_group": "Top2.799_Bot2.799", "default_surface": "Top"},
    "paths": {
        "feature_data": "result/data/feature_engineered_data/featured_data.xlsx",
        "trained_models": "result/trained_models/by_group",
        "online_jobs": "result/trained_models/by_group/Top2.799_Bot2.799",
        "registry": "monitoring/registry.json",
        "reports_dir": "monitoring/reports",
        "prod_config": "group_params_2_799_only.json",
        "logs_dir": "monitoring/logs",
        "alerts_file": "monitoring/logs/alerts.jsonl",
        "locks_dir": "monitoring/locks",
    },
    "model": {
        "min_group_samples": 200,
        "default_train_ratio": 0.65,
        "default_val_ratio": 0.20,
        "cv_folds": 5,
        "random_state": 42,
        "ewma_alpha": 0.6,
    },
    "monitoring": {
        "regression_z_threshold": 2.0,
        "regression_min_drop": 0.05,
        "quality_max_missing_ratio": 0.05,
    },
    "service": {
        "host": "0.0.0.0",
        "port": 8080,
        "max_batch_rows": 5000,
        "max_concurrent_requests": 4,
        "request_timeout_seconds": 30,
        "api_key_env": "TIN_API_KEY",
    },
    "ops": {
        "lock_timeout_seconds": 5,
        "alert_webhook_env": "TIN_ALERT_WEBHOOK",
        "keep_logs_days": 30,
    },
}
