"""告警出口：落盘 + 可选 webhook。

原则：告警通道本身绝不能把主流程拖垮。所有异常都在内部吞掉并记日志，
失败最多重试 3 次（指数退避），调用方只会拿到 True/False。
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


class Notifier:
    def __init__(self, webhook: str | None = None, file_path: str | Path | None = None,
                 timeout: float = 5.0, retries: int = 3, logger=None):
        self.webhook = (webhook or "").strip() or None
        self.file_path = Path(file_path) if file_path else None
        self.timeout = timeout
        self.retries = max(1, int(retries))
        self.logger = logger
        if self.file_path:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def _write_file(self, payload: dict):
        if not self.file_path:
            return
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _post(self, payload: dict) -> bool:
        if not self.webhook:
            return False
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(
                    self.webhook, data=body,
                    headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    if 200 <= resp.status < 300:
                        return True
            except Exception:
                pass
            time.sleep(0.5 * (2 ** attempt))
        return False

    def send(self, level: str, title: str, detail: str = "", **fields) -> bool:
        payload = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "level": str(level).upper(),
            "title": title,
            "detail": detail,
            **fields,
        }
        ok_file = True
        try:
            self._write_file(payload)
        except Exception as e:  # 磁盘满等极端情况也不能抛
            ok_file = False
            if self.logger:
                self.logger.error("alert_file_failed: %s", e)

        ok_web = self._post(payload) if self.webhook else True
        if self.logger:
            self.logger.info("alert_sent level=%s title=%s file=%s webhook=%s",
                             payload["level"], title, ok_file, ok_web)
        return ok_file and ok_web
