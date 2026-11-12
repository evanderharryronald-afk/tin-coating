"""运维底座测试：日志、指标、文件锁、告警、registry 并发、启动自检。"""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "monitoring") not in sys.path:
    sys.path.insert(0, str(ROOT / "monitoring"))

from tin_coating.ops.lock import FileLock, LockTimeout
from tin_coating.ops.log import get_logger
from tin_coating.ops.metrics import Metrics
from tin_coating.ops.notify import Notifier
from tin_coating.ops.selfcheck import run_selfcheck


def test_json_log_line(tmp_path):
    logger = get_logger("t_log", tmp_path, also_stdout=False)
    logger.info("hello", extra={"ctx": {"event": "unit", "n": 3}})
    line = (tmp_path / "t_log.jsonl").read_text(encoding="utf-8").strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload["msg"] == "hello"
    assert payload["event"] == "unit"
    assert payload["n"] == 3
    assert "ts" in payload and payload["level"] == "INFO"


def test_env_override_dotted_keys(monkeypatch):
    """TIN_COATING__A__B 必须能覆盖配置；TIN_COATING_ROOT/ENV 不能污染配置。"""
    from tin_coating.config import load_config

    monkeypatch.setenv("TIN_COATING__SERVICE__PORT", "9099")
    monkeypatch.setenv("TIN_COATING__OPS__LOCK_TIMEOUT_SECONDS", "7")
    monkeypatch.setenv("TIN_COATING_ROOT", str(ROOT))
    cfg = load_config()
    assert cfg.service.port == 9099
    assert cfg.ops.lock_timeout_seconds == 7.0
    assert "root" not in cfg._data


def test_metrics_render():
    m = Metrics()
    m.inc("req_total", labels={"path": "/predict"})
    m.observe("lat", 0.02, labels={"path": "/predict"})
    m.set_gauge("ready", 1)
    text = m.render()
    assert 'req_total{path="/predict"} 1' in text
    assert 'lat_bucket{le="0.025",path="/predict"} 1' in text
    assert "ready 1" in text
    assert "_sum" in text and "_count" in text


def test_file_lock_mutual_exclusion(tmp_path):
    p = tmp_path / "x.lock"
    with FileLock(p, timeout=1.0):
        with pytest.raises(LockTimeout):
            FileLock(p, timeout=0.2).acquire()
    # 释放后应能再次获取
    with FileLock(p, timeout=1.0):
        pass


def test_notifier_file_channel(tmp_path):
    f = tmp_path / "alerts.jsonl"
    n = Notifier(file_path=f)
    assert n.send("ERROR", "title", "detail", job="health") is True
    payload = json.loads(f.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert payload["level"] == "ERROR"
    assert payload["title"] == "title"
    assert payload["job"] == "health"


def test_registry_concurrent_writes(tmp_path):
    from registry import ModelRegistry

    reg_path = tmp_path / "registry.json"
    reg = ModelRegistry(reg_path)

    def worker(tag):
        for i in range(5):
            reg.register("G", "Top", {"version": f"{tag}-{i}", "path": f"/x/{tag}-{i}.joblib"})

    threads = [threading.Thread(target=worker, args=(f"t{k}",)) for k in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    data = json.loads(reg_path.read_text(encoding="utf-8"))   # 必须是合法 JSON
    versions = {e["version"] for e in data["G"]["Top"]}
    assert len(versions) == 30, f"并发写丢失: {len(versions)}"


def test_selfcheck_returns_all_checks():
    checks = run_selfcheck()
    names = {c["check"] for c in checks}
    assert {"prod_config", "registry", "model_artifacts", "log_dir_writable"} <= names
    for c in checks:
        assert "ok" in c and "fatal" in c and "detail" in c
