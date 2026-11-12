"""Model version registry — 追踪各规格组+表面的模型版本历史。

registry.json 结构示意:
{
  "Top2.799_Bot2.799": {
    "Top": [
      {
        "version": "2026-11-12",
        "path": "result/trained_models/.../Top_model.joblib",
        "cv_rmse_mean": 0.0520,
        "cv_rmse_std": 0.0073,
        "holdout_test_rmse": 0.0514,
        "training_samples": 2813,
        "features": 8,
        "active": true,
        "deployed_at": "2026-11-12T10:00:00"
      }
    ],
    "Bot": [...]
  }
}

每个 surface 下最后一条 active=true 的记录为当前在线版本。
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

# 默认 registry 路径（相对于 Task1/ 根目录）
DEFAULT_REGISTRY_PATH = Path(__file__).parent / "registry.json"


class ModelRegistry:
    """轻量 JSON 注册表 —— 记录模型版本 + 指标 + 部署状态。"""

    def __init__(self, registry_path=None):
        self.path = Path(registry_path) if registry_path else DEFAULT_REGISTRY_PATH
        self._ensure_exists()

    # ------------------------------------------------------------------
    # 基础 I/O
    # ------------------------------------------------------------------
    def _ensure_exists(self):
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._write({})

    def _read(self):
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write(self, data):
        """原子写 + numpy 类型清洗。

        旧实现直接 json.dump 到正式文件：一旦序列化中途抛异常（如 numpy.bool_），
        registry.json 会被截断成非法 JSON，后续所有读写全部崩掉。
        """
        import numpy as _np

        def _native(o):
            if isinstance(o, dict):
                return {k: _native(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [_native(v) for v in o]
            if isinstance(o, _np.bool_):
                return bool(o)
            if isinstance(o, _np.integer):
                return int(o)
            if isinstance(o, _np.floating):
                return float(o)
            if isinstance(o, _np.ndarray):
                return o.tolist()
            return o

        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_native(data), f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def _guard(self):
        """跨进程互斥锁：定时任务与人工操作同时改 registry 时不会互相覆盖。"""
        from tin_coating.ops.lock import FileLock
        timeout = 10.0
        try:
            from tin_coating.config import load_config
            timeout = max(1.0, float(load_config().ops.lock_timeout_seconds) * 2)
        except Exception:
            pass
        return FileLock(self.path.with_suffix(".lock"), timeout=timeout)

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def get_active(self, group_label: str, surface: str) -> dict | None:
        """返回指定规格组+表面当前在线版本；没有则返回 None。"""
        data = self._read()
        history = data.get(group_label, {}).get(surface, [])
        for entry in reversed(history):
            if entry.get("active"):
                return entry
        # 兼容：如果没有 active 标记，取最后一条
        return history[-1] if history else None

    def get_history(self, group_label: str, surface: str) -> list:
        data = self._read()
        return data.get(group_label, {}).get(surface, [])

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def register(self, group_label: str, surface: str, entry: dict,
                 deploy_reason: str | None = None):
        """注册一个新版本（默认 active=False，等 gate 审批）。

        entry 必须包含至少：version, path；建议包含 cv_rmse_mean, holdout_test_rmse。
        deploy_reason: 上线理由（如 "walk-forward CV 改进", "修复低读数样本"），
                      会同时写到 entry["deploy_reason"] 和 deploy_reason 字段。
        """
        with self._guard():
            data = self._read()
            data.setdefault(group_label, {}).setdefault(surface, [])

            entry.setdefault("version", datetime.now().strftime("%Y-%m-%d_%H%M%S"))
            entry.setdefault("active", False)
            entry.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
            if deploy_reason:
                entry.setdefault("deploy_reason", deploy_reason)

            data[group_label][surface].append(entry)
            self._write(data)

    def activate(self, group_label: str, surface: str, version: str | None = None,
                 deploy_reason: str | None = None, deployed_path: str | None = None):
        """把指定版本标记为 active，并把同组同面的其他版本 inactive。

        version=None 时激活最后一条。
        deploy_reason 可以覆盖 / 追加该版本的上线理由。
        deployed_path 记录真正上线的那份文件（线上模型路径），回滚时优先用它。
        """
        with self._guard():
            data = self._read()
            history = data.setdefault(group_label, {}).setdefault(surface, [])
            if not history:
                raise ValueError(f"{group_label} / {surface} 历史为空")

            target_idx = None
            for i, e in enumerate(history):
                if version is None and i == len(history) - 1:
                    target_idx = i
                    break
                if e.get("version") == version:
                    target_idx = i
                    break

            if target_idx is None:
                raise ValueError(f"找不到 version={version}")

            prev_active = next((e.get("version") for e in history if e.get("active")), None)

            for i, e in enumerate(history):
                e["active"] = (i == target_idx)
            history[target_idx]["deployed_at"] = datetime.now().isoformat(timespec="seconds")
            # 记录"本版本上线前线上跑的是哪个版本"，供 rollback 精确回退
            if prev_active and prev_active != history[target_idx].get("version"):
                history[target_idx]["previous_active"] = prev_active
            if deploy_reason:
                history[target_idx]["deploy_reason"] = deploy_reason
            if deployed_path:
                history[target_idx]["deployed_path"] = str(deployed_path)
            self._write(data)

    def set_deployed_path(self, group_label: str, surface: str, version: str, path: str):
        """记录某个历史版本真正上线时留下的产物路径（不清改 active 标记）。"""
        with self._guard():
            data = self._read()
            for e in data.get(group_label, {}).get(surface, []):
                if e.get("version") == version:
                    e["deployed_path"] = str(path)
                    self._write(data)
                    return

    def preview_rollback(self, group_label: str, surface: str) -> dict:
        """解析"回滚会回到哪个版本"，返回该条记录（不修改 registry）。

        规则：优先用 active 条目记录的 previous_active（最近一次部署前的线上版本）；
        老数据没有该字段时，退化为按 created_at 找更早的最近版本。
        """
        data = self._read()
        history = data.get(group_label, {}).get(surface, [])
        if len(history) < 2:
            raise ValueError("历史版本不足，无法回滚")

        def _sort_key(e):
            return e.get("created_at") or e.get("version", "")

        active_entry = next((e for e in history if e.get("active")), None)
        if active_entry is None:
            active_entry = max(history, key=_sort_key)

        prev_version = active_entry.get("previous_active")
        if prev_version:
            target = next((e for e in history if e.get("version") == prev_version), None)
            if target is not None:
                return target

        active_key = _sort_key(active_entry)
        earlier = [e for e in history if _sort_key(e) < active_key]
        if not earlier:
            raise ValueError("没有更早版本可回滚")
        return max(earlier, key=_sort_key)

    def rollback(self, group_label: str, surface: str, version: str | None = None) -> str:
        """激活上一个"真正上线过"的版本（回滚），返回被激活的 version。"""
        with self._guard():
            data = self._read()
            history = data.get(group_label, {}).get(surface, [])
            active_entry = next((e for e in history if e.get("active")), None)
            active_version = active_entry.get("version") if active_entry else None

            target = self.preview_rollback(group_label, surface)
            target_version = target.get("version")

            for e in history:
                e["active"] = (e.get("version") == target_version)
            for e in history:
                if e.get("active"):
                    e["deployed_at"] = datetime.now().isoformat(timespec="seconds")
                    e.pop("previous_active", None)   # 回滚后不再保留旧的"前一版"指向
                    e["deploy_reason"] = e.get("deploy_reason", "") + f" [auto-rollback from {active_version or '?'}]"
                    break
            self._write(data)
            return target_version

    # ------------------------------------------------------------------
    # 辅助：首次启动时把线上已有模型注册进去
    # ------------------------------------------------------------------
    def seed_from_artifact(self, group_label: str, surface: str,
                           artifact_path: str, training_samples: int,
                           features: int, cv_rmse_mean=None,
                           cv_rmse_std=None, holdout_test_rmse=None,
                           hyperparameters=None):
        """首次建立 registry 时，把线上已在跑的模型作为初始 active 版本。"""
        if self.get_active(group_label, surface) is not None:
            return  # 已经有了，不覆盖

        self.register(group_label, surface, {
            "version": datetime.now().strftime("%Y-%m-%d_init"),
            "path": str(artifact_path),
            "cv_rmse_mean": cv_rmse_mean,
            "cv_rmse_std": cv_rmse_std,
            "holdout_test_rmse": holdout_test_rmse,
            "training_samples": training_samples,
            "features": features,
            "hyperparameters": hyperparameters,
            "source": "seed_from_artifact",
            "active": True,
        })
