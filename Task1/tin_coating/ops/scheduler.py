"""无人值守定时任务执行器。

给 OS 调度器（cron / systemd timer / Windows 任务计划）调用：
  python cli/main.py ops run --job quality
  python cli/main.py ops run --job health
  python cli/main.py ops run --job retrain

保证：
  - 同一 job 不会并发跑（文件锁）
  - 失败一定告警（落盘 + 可选 webhook）
  - 退出码可被调度器判断：0 正常 / 1 警告 / 2 失败 / 3 跳过（锁占用）
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from tin_coating.config import load_config
from tin_coating.ops.lock import FileLock, LockTimeout
from tin_coating.ops.log import get_logger, log_event
from tin_coating.ops.metrics import METRICS
from tin_coating.ops.notify import Notifier

JOBS = ("quality", "health", "retrain")


def _prune_old_files(cfg, logger) -> None:
    """按 ops.keep_logs_days 清理过期日志/报告；永远不动模型文件与锁文件。

    无人值守场景下磁盘是会被慢慢吃满的：日志虽有轮转，但报告 JSON/TXT 只增不减。
    这里在每次定时任务结束时顺手清理，失败不影响任务结果。
    """
    try:
        days = int(getattr(cfg.ops, "keep_logs_days", 30) or 30)
        cutoff = time.time() - days * 86400
        removed = 0
        for key in ("paths.logs_dir", "paths.reports_dir"):
            try:
                d = cfg.resolve(key)
            except Exception:
                continue
            if not d.exists():
                continue
            for p in d.iterdir():
                if not p.is_file() or p.suffix in (".joblib", ".lock"):
                    continue
                try:
                    if p.stat().st_mtime < cutoff:
                        p.unlink()
                        removed += 1
                except OSError:
                    pass
        if removed:
            logger.info("pruned_files n=%s keep_days=%s", removed, days)
    except Exception:
        pass



def _monitor_dir(cfg) -> Path:
    return cfg.root / "monitoring"


def _ensure_importable(cfg):
    d = str(_monitor_dir(cfg))
    if d not in sys.path:
        sys.path.insert(0, d)


def _make_notifier(cfg, logger) -> Notifier:
    env_name = getattr(cfg.ops, "alert_webhook_env", "TIN_ALERT_WEBHOOK")
    webhook = os.environ.get(env_name, "") if env_name else ""
    return Notifier(webhook=webhook or None,
                    file_path=cfg.resolve("paths.alerts_file"),
                    logger=logger)


def run_job(job: str, data: str | None = None, force: bool = False,
            group_label: str | None = None) -> dict:
    if job not in JOBS:
        raise ValueError(f"未知 job: {job}，可选 {JOBS}")

    cfg = load_config()
    logger = get_logger("ops", cfg.resolve("paths.logs_dir"), also_stdout=False)
    notifier = _make_notifier(cfg, logger)
    feature_data = data or str(cfg.resolve("paths.feature_data"))
    started = time.time()

    lock_path = cfg.resolve("paths.locks_dir") / f"job_{job}.lock"
    try:
        lock = FileLock(lock_path, timeout=float(cfg.ops.lock_timeout_seconds))
        lock.acquire()
    except LockTimeout:
        msg = f"{job} 上一次执行尚未结束，本次跳过"
        logger.warning("job_skipped job=%s reason=locked", job)
        notifier.send("WARN", f"定时任务跳过: {job}", msg, job=job)
        return {"job": job, "status": "skipped", "reason": "locked",
                "detail": msg, "exit_code": 3}

    status, detail, exit_code = "fail", "", 2
    try:
        _ensure_importable(cfg)
        if job == "quality":
            import check_data_quality as q
            r = q.run(feature_data, silent=False)
            ok = bool(r.get("pass"))
            status, exit_code = ("ok", 0) if ok else ("fail", 2)
            detail = f"pass={ok}"
        elif job == "health":
            import check_monthly_health as h
            r = h.run(feature_data, silent=False)
            verdict = r.get("overall_verdict", "WARN")
            status = {"OK": "ok", "WARN": "warn"}.get(verdict, "fail")
            exit_code = {"ok": 0, "warn": 1}.get(status, 2)
            detail = f"verdict={verdict}"
        else:  # retrain
            import auto_retrain_candidate as r
            rep = r.run(feature_data, force=force, skip_quality_check=False)
            skipped = rep.get("skipped") or []
            if rep.get("pending_models"):
                status, exit_code = "ok", 0
                detail = f"pending={len(rep['pending_models'])}"
            elif rep.get("abort_reason"):
                status, exit_code = "fail", 2
                detail = f"abort={rep['abort_reason']}"
            elif skipped and all("已有未审批" in str(s.get("reason", "")) for s in skipped):
                # 已有候选等待人工审批：这是正常状态，不应每天重复告警
                status, exit_code = "ok", 0
                detail = f"awaiting_approval={len(skipped)}"
            else:
                status, exit_code = "warn", 1
                detail = "no_pending"
    except Exception as e:
        status, exit_code = "fail", 2
        detail = f"{type(e).__name__}: {e}"
        logger.error("job_exception job=%s err=%s", job, detail)
    finally:
        lock.release()

    elapsed = round(time.time() - started, 2)
    logger.info("job_finished job=%s status=%s detail=%s elapsed_s=%s", job, status, detail, elapsed)
    METRICS.inc("tin_coating_ops_jobs_total", labels={"job": job, "status": status})

    if status == "fail":
        notifier.send("ERROR", f"定时任务失败: {job}", detail, job=job, elapsed_s=elapsed)
    elif status == "warn":
        notifier.send("WARN", f"定时任务警告: {job}", detail, job=job, elapsed_s=elapsed)

    _prune_old_files(cfg, logger)

    return {"job": job, "status": status, "detail": detail,
            "exit_code": exit_code, "elapsed_s": elapsed}
