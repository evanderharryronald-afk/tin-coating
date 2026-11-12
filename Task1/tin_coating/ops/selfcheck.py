"""启动自检 —— fail fast。

无人值守系统最怕"进程起来了但模型是坏的"。启动时把所有前置条件检查一遍，
有问题直接抛错退出，让进程管理器（systemd / docker restart）感知到异常，
而不是带着坏状态接流量。
"""
from __future__ import annotations

from pathlib import Path

from tin_coating.config import load_config


def _check(name: str, fn, fatal: bool = True) -> dict:
    try:
        detail = fn()
        return {"check": name, "ok": True, "detail": detail, "fatal": fatal}
    except Exception as e:
        return {"check": name, "ok": False, "detail": f"{type(e).__name__}: {e}", "fatal": fatal}


def run_selfcheck(group_label: str | None = None) -> list[dict]:
    cfg = load_config()
    group = group_label or cfg.app.default_group
    jobs_dir = cfg.resolve("paths.trained_models") / group

    def _prod_config():
        p = cfg.resolve("paths.prod_config")
        if not p.exists():
            raise FileNotFoundError(p)
        return str(p)

    def _registry():
        import json
        p = cfg.resolve("paths.registry")
        if not p.exists():
            return "registry 尚未创建（首次运行会自动生成）"
        json.loads(p.read_text(encoding="utf-8"))   # 解析失败即视为损坏
        return str(p)

    def _models():
        missing = []
        for surface in ("Top", "Bot"):
            f = jobs_dir / f"{surface}_model.joblib"
            if not f.exists() or f.stat().st_size == 0:
                missing.append(str(f))
        if missing:
            raise FileNotFoundError(f"缺少模型文件: {missing}")
        return str(jobs_dir)

    def _writable():
        d = cfg.resolve("paths.logs_dir")
        d.mkdir(parents=True, exist_ok=True)
        probe = d / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return str(d)

    def _feature_data():
        p = cfg.resolve("paths.feature_data")
        if not p.exists():
            raise FileNotFoundError(str(p))
        return str(p)

    checks = [
        _check("prod_config", _prod_config),
        _check("registry", _registry),
        _check("model_artifacts", _models),
        _check("log_dir_writable", _writable),
        _check("feature_data", _feature_data, fatal=False),
    ]
    return checks


def assert_ready(group_label: str | None = None):
    """有致命项失败就抛 RuntimeError（服务不应启动）。"""
    checks = run_selfcheck(group_label)
    failed = [c for c in checks if not c["ok"] and c["fatal"]]
    if failed:
        detail = "; ".join(f"{c['check']}: {c['detail']}" for c in failed)
        raise RuntimeError(f"启动自检失败: {detail}")
    return checks
