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
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

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
    def register(self, group_label: str, surface: str, entry: dict):
        """注册一个新版本（默认 active=False，等 gate 审批）。

        entry 必须包含至少：version, path；建议包含 cv_rmse_mean, holdout_test_rmse。
        """
        data = self._read()
        data.setdefault(group_label, {}).setdefault(surface, [])

        entry.setdefault("version", datetime.now().strftime("%Y-%m-%d_%H%M%S"))
        entry.setdefault("active", False)
        entry.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))

        data[group_label][surface].append(entry)
        self._write(data)

    def activate(self, group_label: str, surface: str, version: str | None = None):
        """把指定版本标记为 active，并把同组同面的其他版本 inactive。

        version=None 时激活最后一条。
        """
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

        for i, e in enumerate(history):
            e["active"] = (i == target_idx)
        history[target_idx]["deployed_at"] = datetime.now().isoformat(timespec="seconds")
        self._write(data)

    def rollback(self, group_label: str, surface: str, version: str | None = None):
        """激活上一个版本（回滚）。

        定义：找当前 active 版本，然后找 history 里"比它更早的那条最近的版本"激活。
        如果 history 不足 2 条，或者找不到更早版本，报错。
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

        active_key = _sort_key(active_entry)

        # 候选：所有比 active_key 更早的版本
        earlier = [e for e in history if _sort_key(e) < active_key]
        if not earlier:
            raise ValueError("没有更早版本可回滚")

        target = max(earlier, key=_sort_key)
        target_version = target.get("version")

        for e in history:
            e["active"] = (e.get("version") == target_version)
        for e in history:
            if e.get("active"):
                e["deployed_at"] = datetime.now().isoformat(timespec="seconds")
                break
        self._write(data)

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
