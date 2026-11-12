"""在线服务集成测试：真实起 HTTP 服务 + 真实请求 + 故障注入。"""
from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tin_coating.config import load_config
from tin_coating.service.contract import ContractError, raw_required_cols, validate_rows
from tin_coating.service.server import create_server


# ---------------------------------------------------------------------------
# 契约单测
# ---------------------------------------------------------------------------
def test_contract_rejects_missing_column():
    with pytest.raises(ContractError):
        validate_rows([{"foo": 1}], "Top", 100)


def test_contract_rejects_empty_and_oversize():
    with pytest.raises(ContractError):
        validate_rows([], "Top", 100)
    with pytest.raises(ContractError):
        validate_rows([{}] * 5, "Top", 2)


def test_contract_rejects_non_numeric():
    cols = raw_required_cols("Top")
    row = {c: 1.0 for c in cols}
    row[cols[0]] = "abc"
    with pytest.raises(ContractError):
        validate_rows([row], "Top", 100)


# ---------------------------------------------------------------------------
# 服务集成
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def live_server():
    httpd, ctx = create_server(host="127.0.0.1", port=0)
    t = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    t.start()
    host, port = httpd.server_address[0], httpd.server_address[1]
    yield f"http://{host}:{port}", ctx
    httpd.shutdown()
    httpd.server_close()


def _get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.status, r.read().decode("utf-8")


def _post(url, payload, headers=None):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")


def _sample_rows(surface: str, n: int = 5):
    cfg = load_config()
    df = pd.read_excel(cfg.resolve("paths.feature_data"), nrows=n)
    cols = [c for c in raw_required_cols(surface) if c in df.columns]
    assert len(cols) == len(raw_required_cols(surface)), "featured_data 缺列"
    return df[cols].to_dict(orient="records")


def test_health_and_ready(live_server):
    base, _ = live_server
    status, body = _get(f"{base}/healthz")
    assert status == 200 and json.loads(body)["status"] == "ok"
    status, body = _get(f"{base}/readyz")
    assert status == 200
    assert json.loads(body)["status"] == "ready"


def test_predict_happy_path(live_server):
    base, _ = live_server
    rows = _sample_rows("Top")
    status, body = _post(f"{base}/predict", {"surface": "Top", "rows": rows})
    assert status == 200, body
    payload = json.loads(body)
    assert payload["n"] == len(rows)
    assert len(payload["predictions"]["Top"]) == len(rows)
    assert payload["predictions"]["Top"][0]["corrected"] > 0


def test_predict_both_surfaces(live_server):
    base, _ = live_server
    status, body = _post(f"{base}/predict", {"surface": "both", "rows": _sample_rows("Top")})
    assert status == 400 or status == 200
    if status == 200:
        assert set(json.loads(body)["predictions"]) == {"Top", "Bot"}


def test_predict_bad_payload_returns_400(live_server):
    base, _ = live_server
    status, body = _post(f"{base}/predict", {"surface": "Top", "rows": [{"x": 1}]})
    assert status == 400
    assert json.loads(body)["error"] == "bad_request"


def test_predict_invalid_json_returns_400(live_server):
    base, _ = live_server
    req = urllib.request.Request(f"{base}/predict", data=b"{not json",
                                 method="POST", headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=10)
        assert False, "应当返回 400"
    except urllib.error.HTTPError as e:
        assert e.code == 400


def test_unknown_path_404(live_server):
    base, _ = live_server
    try:
        _get(f"{base}/nope")
        assert False, "应当 404"
    except urllib.error.HTTPError as e:
        assert e.code == 404


def test_metrics_exposes_http_counters(live_server):
    base, _ = live_server
    _get(f"{base}/healthz")
    status, body = _get(f"{base}/metrics")
    assert status == 200
    assert "tin_coating_http_requests_total" in body
    assert "tin_coating_uptime_seconds" in body


def test_api_key_enforced(live_server, monkeypatch):
    base, _ = live_server
    monkeypatch.setenv("TIN_API_KEY", "secret-123")
    status, _ = _post(f"{base}/predict", {"surface": "Top", "rows": _sample_rows("Top")})
    assert status == 401
    status, _ = _post(f"{base}/predict",
                      {"surface": "Top", "rows": _sample_rows("Top")},
                      headers={"X-API-Key": "secret-123"})
    assert status == 200


def test_concurrent_predict(live_server):
    """并发请求都能正常返回（验证 semaphore / ThreadingHTTPServer 不串味）。"""
    from concurrent.futures import ThreadPoolExecutor

    base, _ = live_server
    rows = _sample_rows("Top", n=3)

    def _call(_):
        return _post(f"{base}/predict", {"surface": "Top", "rows": rows})

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_call, range(16)))
    assert all(s == 200 for s, _ in results), [b for s, b in results if s != 200][:2]
    assert all(json.loads(b)["n"] == len(rows) for s, b in results)


def test_model_store_hot_reload_on_mtime_change():
    """换模型/覆盖产物后无需重启：mtime 变化即触发重载。"""
    import os

    from tin_coating.service.models import ModelStore

    store = ModelStore(cfg=load_config())
    assert store.load_all() == {"Top": True, "Bot": True}
    assert store.ensure_fresh() == []          # 刚加载过，指纹未变

    p = store.artifact_path("Top")
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))

    assert "Top" in store.ensure_fresh()
    assert store.ready() is True


def test_model_store_keeps_old_model_on_corrupt_artifact(tmp_path):
    """坏模型文件不应让服务停摆：加载失败时退回 live 目录产物，ready 仍为 True。"""
    import joblib

    from tin_coating.service.models import ModelStore

    store = ModelStore(cfg=load_config())
    live_dir = tmp_path / "live"
    live_dir.mkdir()
    joblib.dump({"dummy": "model"}, live_dir / "Bot_model.joblib")
    store.dir = live_dir

    bad = tmp_path / "bad_bot.joblib"
    bad.write_bytes(b"not a joblib file")
    reg = tmp_path / "registry.json"
    reg.write_text(json.dumps({store.group: {"Bot": [
        {"version": "broken", "active": True, "path": str(bad)}]}}), encoding="utf-8")
    store.registry_path = reg

    assert store.load("Bot") is True
    assert store.get("Bot") is not None
    assert store.status()["models"]["Bot"]["fallback"] is True


def test_model_store_not_ready_when_all_artifacts_unusable(tmp_path):
    """registry 与 live 目录都没有可用模型时，ready 必须为 False（探针能发现）。"""
    from tin_coating.service.models import ModelStore

    store = ModelStore(cfg=load_config())
    store.dir = tmp_path / "empty_group"
    bad = tmp_path / "bad.joblib"
    bad.write_bytes(b"not a joblib file")
    reg = tmp_path / "registry.json"
    reg.write_text(json.dumps({store.group: {
        "Top": [{"version": "broken", "active": True, "path": str(bad)}],
        "Bot": [{"version": "broken", "active": True, "path": str(bad)}],
    }}), encoding="utf-8")
    store.registry_path = reg

    assert store.load_all() == {"Top": False, "Bot": False}
    assert store.ready() is False

