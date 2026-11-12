"""模型加载与热更新。

无人值守时不能因为"换了一版模型"就重启服务：
每次请求顺带检查 active 版本 / 文件 mtime / 大小，变了就在后台重载，
加载失败则保留旧模型继续服务（并告警），避免一次坏部署导致全线停摆。
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import joblib

from tin_coating.config import load_config

SURFACES = ("Top", "Bot")


class ModelStore:
    def __init__(self, group_label: str | None = None, cfg=None, logger=None):
        self.cfg = cfg or load_config()
        self.group = group_label or self.cfg.app.default_group
        self.dir = self.cfg.resolve("paths.trained_models") / self.group
        self.registry_path = self.cfg.resolve("paths.registry")
        self.logger = logger
        self._models: dict[str, object] = {}
        self._meta: dict[str, dict] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # 解析当前应使用的产物
    # ------------------------------------------------------------------
    def _registry_active(self, surface: str) -> dict:
        try:
            data = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        for entry in data.get(self.group, {}).get(surface, []):
            if entry.get("active"):
                return entry
        return {}

    def _live_path(self, surface: str) -> Path:
        return self.dir / f"{surface}_model.joblib"

    def _candidates(self, surface: str) -> list[Path]:
        """按优先级返回可尝试的产物：registry 记录 → live 目录。

        live 目录只是兜底：例如 registry 里的归档模型是旧版 sklearn 训练的，
        当前环境反序列化不了（__pyx_unpickle_* 缺失），此时用 live 目录里
        最近一次训练/部署的文件，避免服务整体不可用。
        """
        entry = self._registry_active(surface)
        out: list[Path] = []
        for key in ("deployed_path", "path"):
            p = entry.get(key)
            if p and Path(p).exists() and Path(p) not in out:
                out.append(Path(p))
        live = self._live_path(surface)
        if live.exists() and live not in out:
            out.append(live)
        return out

    def artifact_path(self, surface: str) -> Path:
        cands = self._candidates(surface)
        return cands[0] if cands else self._live_path(surface)

    def _fingerprint(self, surface: str) -> tuple:
        entry = self._registry_active(surface)
        parts = [("version", str(entry.get("version", "")))]
        for p in self._candidates(surface):
            st = p.stat()
            parts.append((str(p), st.st_mtime_ns, st.st_size))
        return tuple(parts)

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------
    def load(self, surface: str) -> bool:
        import tin_coating.modeling.residual_model  # noqa: F401  (pickle 反序列化依赖)

        cands = self._candidates(surface)
        if not cands:
            if self.logger:
                self.logger.error("model_missing surface=%s path=%s",
                                  surface, self._live_path(surface))
            return False

        primary = cands[0]
        model = None
        used = None
        errors = []
        for path in cands:
            try:
                model = joblib.load(path)
                used = path
                break
            except Exception as e:
                errors.append(f"{path.name}: {type(e).__name__}: {e}")
                if self.logger:
                    self.logger.error("model_load_failed surface=%s path=%s err=%s",
                                      surface, path, e)

        if model is None:
            if self.logger:
                self.logger.error("model_unusable surface=%s errors=%s", surface, errors)
            return False

        fallback = used != primary
        with self._lock:
            self._models[surface] = model
            entry = self._registry_active(surface)
            self._meta[surface] = {
                "surface": surface,
                "path": str(used),
                "version": entry.get("version", "unknown"),
                "fallback": fallback,
                "fingerprint": self._fingerprint(surface),
            }
        if self.logger:
            if fallback:
                self.logger.warning(
                    "model_fallback surface=%s version=%s primary=%s used=%s",
                    surface, self._meta[surface]["version"], primary, used)
            else:
                self.logger.info("model_loaded surface=%s version=%s path=%s",
                                 surface, self._meta[surface]["version"], used)
        return True

    def load_all(self) -> dict:
        return {s: self.load(s) for s in SURFACES}

    def ensure_fresh(self) -> list[str]:
        """文件或 active 版本变化时重载；返回发生变化的 surface 列表。"""
        changed = []
        for surface in SURFACES:
            fp = self._fingerprint(surface)
            with self._lock:
                meta = self._meta.get(surface)
            if meta is None or meta.get("fingerprint") != fp:
                if self.load(surface):
                    changed.append(surface)
        return changed

    # ------------------------------------------------------------------
    # 读
    # ------------------------------------------------------------------
    def get(self, surface: str):
        with self._lock:
            return self._models.get(surface)

    def status(self) -> dict:
        with self._lock:
            return {
                "group": self.group,
                "models": {
                    s: {
                        "loaded": s in self._models,
                        "version": self._meta.get(s, {}).get("version"),
                        "path": self._meta.get(s, {}).get("path"),
                        "fallback": self._meta.get(s, {}).get("fallback", False),
                    } for s in SURFACES
                },
            }

    def ready(self) -> bool:
        with self._lock:
            return all(s in self._models for s in SURFACES)
