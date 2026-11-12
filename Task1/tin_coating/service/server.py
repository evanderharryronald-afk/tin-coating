"""7x24 在线推理服务（标准库 HTTP，无额外依赖）。

端点：
  GET  /healthz    存活探针（进程能响应就 200）
  GET  /readyz     就绪探针（模型已加载才 200，否则 503）
  GET  /metrics    Prometheus 文本指标
  GET  /version    服务与模型版本
  POST /predict    批量推理 {"surface":"Top|Bot|both","rows":[{...}]}
  POST /reload     强制重载模型（可选 X-API-Key 保护）

设计要点：
  - 模型热加载：每次请求前检查 active 版本/文件指纹，变更即重载，不中断服务
  - 加载失败保留旧模型继续服务，只告警，不 500 全线崩
  - 并发上限 + 请求超时 + 请求体大小限制，防止被打爆
"""
from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from tin_coating.config import load_config
from tin_coating.modeling.params import get_feature_cols
from tin_coating.ops.log import get_logger, log_event, new_request_id
from tin_coating.ops.metrics import METRICS
from tin_coating.ops.notify import Notifier
from tin_coating.service.contract import ContractError, validate_rows
from tin_coating.service.models import ModelStore

MAX_BODY_BYTES = 16 * 1024 * 1024


class Context:
    def __init__(self, cfg, logger, store: ModelStore, notifier: Notifier, sem: threading.Semaphore):
        self.cfg = cfg
        self.logger = logger
        self.store = store
        self.notifier = notifier
        self.sem = sem


class Handler(BaseHTTPRequestHandler):
    server_version = "TinCoating/1.0"
    ctx: Context = None  # 由 create_server 注入

    # ---- 基础工具 -----------------------------------------------------
    def log_message(self, fmt, *args):  # 覆盖默认 stderr 输出
        pass

    def _json(self, code: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Request-Id", getattr(self, "_rid", ""))
        self.end_headers()
        self.wfile.write(body)

    def _text(self, code: int, text: str, ctype: str = "text/plain; charset=utf-8"):
        body = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _auth_ok(self) -> bool:
        env_name = self.ctx.cfg.service.api_key_env
        expected = os.environ.get(env_name, "").strip()
        if not expected:
            return True
        return self.headers.get("X-API-Key", "") == expected

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            raise ContractError("请求体为空")
        if length > MAX_BODY_BYTES:
            raise ContractError(f"请求体超过 {MAX_BODY_BYTES} 字节")
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            raise ContractError("请求体不是合法 JSON")

    # ---- 请求入口 -----------------------------------------------------
    def do_GET(self):
        self._dispatch("GET", {
            "/healthz": self._healthz,
            "/readyz": self._readyz,
            "/metrics": self._metrics,
            "/version": self._version,
        })

    def do_POST(self):
        self._dispatch("POST", {
            "/predict": self._predict,
            "/reload": self._reload,
        })

    def _dispatch(self, method: str, routes: dict):
        self._rid = self.headers.get("X-Request-Id") or new_request_id()
        path = self.path.split("?", 1)[0]
        start = time.time()
        status = 200
        try:
            fn = routes.get(path)
            if fn is None:
                status = 404
                self._json(404, {"error": "not_found", "path": path, "request_id": self._rid})
            else:
                status = fn()
        except ContractError as e:
            status = 400
            self._json(400, {"error": "bad_request", "detail": str(e), "request_id": self._rid})
        except Exception as e:  # 兜底：任何异常都要返回 JSON，不能挂连接
            status = 500
            self.ctx.logger.error("unhandled_error path=%s err=%s\n%s", path, e, traceback.format_exc())
            self._json(500, {"error": "internal_error", "detail": str(e), "request_id": self._rid})
        finally:
            elapsed = time.time() - start
            METRICS.inc("tin_coating_http_requests_total", labels={
                "method": method, "path": path, "status": str(status)})
            METRICS.observe("tin_coating_http_request_duration_seconds", elapsed,
                            labels={"path": path})
            log_event(self.ctx.logger, "INFO", "http_request",
                      request_id=self._rid, method=method, path=path,
                      status=status, latency_ms=round(elapsed * 1000, 2),
                      remote=self.client_address[0])

    # ---- 端点实现 -----------------------------------------------------
    def _healthz(self) -> int:
        self._json(200, {"status": "ok", "request_id": self._rid})
        return 200

    def _readyz(self) -> int:
        ctx = self.ctx
        if ctx.store.ensure_fresh():
            ctx.logger.info("models_reloaded_by_readyz")
        ready = ctx.store.ready()
        METRICS.set_gauge("tin_coating_ready", 1.0 if ready else 0.0)
        payload = {"status": "ready" if ready else "not_ready",
                   "request_id": self._rid, **ctx.store.status()}
        self._json(200 if ready else 503, payload)
        return 200 if ready else 503

    def _metrics(self) -> int:
        self._text(200, METRICS.render(), "text/plain; version=0.0.4; charset=utf-8")
        return 200

    def _version(self) -> int:
        self._json(200, {"service": "tin-coating", "version": "1.0.0",
                         "request_id": self._rid, **self.ctx.store.status()})
        return 200

    def _reload(self) -> int:
        if not self._auth_ok():
            self._json(401, {"error": "unauthorized", "request_id": self._rid})
            return 401
        changed = self.ctx.store.ensure_fresh()
        ok = self.ctx.store.ready()
        self._json(200 if ok else 503, {
            "status": "reloaded" if ok else "not_ready",
            "changed": changed, "request_id": self._rid,
            **self.ctx.store.status()})
        return 200 if ok else 503

    def _predict(self) -> int:
        ctx = self.ctx
        start = time.time()

        if not ctx.sem.acquire(timeout=float(ctx.cfg.service.request_timeout_seconds)):
            self._json(503, {"error": "busy", "detail": "并发请求已达上限",
                             "request_id": self._rid})
            METRICS.inc("tin_coating_predict_rejected_total", labels={"reason": "busy"})
            return 503
        try:
            if not self._auth_ok():
                self._json(401, {"error": "unauthorized", "request_id": self._rid})
                return 401

            if ctx.store.ensure_fresh():
                ctx.logger.info("models_reloaded_on_predict")

            payload = self._read_json()
            surface = payload.get("surface", "Top")
            rows = payload.get("rows")
            surfaces = ["Top", "Bot"] if surface == "both" else [surface]
            if surface not in ("Top", "Bot", "both"):
                raise ContractError("surface 必须是 Top / Bot / both")

            max_rows = int(ctx.cfg.service.max_batch_rows)
            result: dict = {}
            versions: dict = {}
            for s in surfaces:
                model = ctx.store.get(s)
                if model is None:
                    METRICS.inc("tin_coating_predict_errors_total", labels={"reason": "model_not_ready"})
                    self._json(503, {"error": "model_not_ready", "surface": s,
                                     "request_id": self._rid})
                    return 503

                df = validate_rows(rows, s, max_rows)
                df[f"{s}_Current_Per_Speed"] = (
                    df[f"{s}_Current_Sum"] / (df["Speed[m/min]_Process_Avg"] + 1e-5))
                online_col = f"Tin Weight_Actual[g/m2]_GALV_WEIGHT_{s.upper()}_Avg"
                X = df[get_feature_cols(s)]
                final, delta = model.predict_smooth(X, df[online_col])

                result[s] = [
                    {"corrected": round(float(v), 6), "delta": round(float(d), 6)}
                    for v, d in zip(final.values, delta.values)
                ]
                versions[s] = ctx.store.status()["models"][s]["version"]
                METRICS.inc("tin_coating_predict_rows_total", len(result[s]), labels={"surface": s})

            elapsed_ms = round((time.time() - start) * 1000, 2)
            self._json(200, {"status": "ok", "n": len(rows), "predictions": result,
                             "model_versions": versions, "elapsed_ms": elapsed_ms,
                             "request_id": self._rid})
            return 200
        finally:
            ctx.sem.release()


def create_server(host: str | None = None, port: int | None = None,
                  group_label: str | None = None) -> tuple[ThreadingHTTPServer, Context]:
    cfg = load_config()
    logger = get_logger("service", cfg.resolve("paths.logs_dir"), also_stdout=True)
    store = ModelStore(group_label, cfg=cfg, logger=logger)
    loaded = store.load_all()
    if not all(loaded.values()):
        logger.warning("models_not_all_loaded %s", loaded)

    webhook_env = cfg.ops.alert_webhook_env
    notifier = Notifier(
        webhook=os.environ.get(webhook_env) if webhook_env else None,
        file_path=cfg.resolve("paths.alerts_file"),
        logger=logger,
    )
    sem = threading.Semaphore(int(cfg.service.max_concurrent_requests))
    ctx = Context(cfg, logger, store, notifier, sem)

    handler_cls = type("BoundHandler", (Handler,), {"ctx": ctx})
    handler_cls.timeout = int(cfg.service.request_timeout_seconds)

    bind_host = host if host is not None else str(cfg.service.host)
    bind_port = int(port if port is not None else cfg.service.port)
    httpd = ThreadingHTTPServer((bind_host, bind_port), handler_cls)
    httpd.daemon_threads = True

    METRICS.set_gauge("tin_coating_ready", 1.0 if store.ready() else 0.0)
    return httpd, ctx


def serve(host: str | None = None, port: int | None = None, group_label: str | None = None) -> int:
    from tin_coating.ops.selfcheck import assert_ready

    cfg = load_config()
    try:
        checks = assert_ready(group_label)
    except RuntimeError as e:
        print(f"[selfcheck] FATAL {e}", file=sys.stderr)
        return 2
    for c in checks:
        print(f"[selfcheck] {'OK ' if c['ok'] else 'FAIL'} {c['check']}: {c['detail']}")

    httpd, ctx = create_server(host, port, group_label)
    ctx.logger.info("service_started host=%s port=%s group=%s",
                    httpd.server_address[0], httpd.server_address[1], ctx.store.group)

    def _graceful(signum, _frame):
        ctx.logger.info("shutdown_signal received=%s", signum)
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    try:
        signal.signal(signal.SIGINT, _graceful)
        signal.signal(signal.SIGTERM, _graceful)
    except (ValueError, AttributeError):
        pass  # 非主线程（测试环境）无法注册信号

    try:
        httpd.serve_forever(poll_interval=0.5)
    finally:
        httpd.server_close()
        ctx.logger.info("service_stopped")
    return 0
