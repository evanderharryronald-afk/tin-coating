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

JOBS = ("quality", "health", "retrain", "drift", "report")


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
        elif job == "retrain":
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
        elif job == "drift":
            import pandas as pd
            import joblib
            import numpy as np
            from tin_coating.modeling.residual_model import ResidualCorrectionModel  # noqa: F401
            from tin_coating.modeling.params import (
                build_setpoint_group_key, get_feature_cols,
            )
            from tin_coating.diagnostics.drift import ks_test_drift

            g = group_label or str(cfg.app.default_group)
            df = pd.read_excel(feature_data)
            df = build_setpoint_group_key(df)
            df = df[df["Setpoint_Group_Label"] == g].copy()
            for s in ["Top", "Bot"]:
                cs, sp = f"{s}_Current_Sum", "Speed[m/min]_Process_Avg"
                if cs in df.columns and sp in df.columns:
                    df[f"{s}_Current_Per_Speed"] = df[cs] / (df[sp] + 1e-5)

            model_dir = cfg.resolve("paths.online_jobs")
            verdicts = []
            for surface in ("Top", "Bot"):
                feat_cols = get_feature_cols(surface)
                online_col = f"Tin Weight_Actual[g/m2]_GALV_WEIGHT_{surface.upper()}_Avg"
                model_path = model_dir / f"{surface}_model.joblib"
                sub = df.dropna(subset=feat_cols + [online_col])
                if (not model_path.exists()) or len(sub) < 60:
                    verdicts.append((surface, "OK", "skip"))
                    continue
                try:
                    m = joblib.load(model_path)
                except Exception as e:
                    verdicts.append((surface, "WARN", f"model_load_failed:{type(e).__name__}"))
                    continue
                pred, _ = m.predict_smooth(sub[feat_cols], sub[online_col])
                resids = pred.values - sub[online_col].values
                ks, p, md, v = ks_test_drift(resids, ratio=0.8)
                verdicts.append((surface, v, f"ks={ks:.3f} p={p:.3g} mean_diff={md:.3f}"))

            worst = max(("OK", "WARN", "DRIFT").index(v) for _, v, _ in verdicts)
            status = ("ok", "warn", "fail")[worst]
            exit_code = (0, 1, 2)[worst]
            detail = "; ".join(f"{s}:{v}({d})" for s, v, d in verdicts)

            # 落盘一份 JSON 摘要（给 Grafana / 自己查都方便）
            try:
                import json
                (cfg.resolve("paths.reports_dir")).mkdir(parents=True, exist_ok=True)
                out = cfg.resolve("paths.reports_dir") / f"drift_{g}.json"
                out.write_text(json.dumps({
                    "group": g, "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "verdicts": [{"surface": s, "verdict": v, "detail": d}
                                 for s, v, d in verdicts],
                    "worst_verdict": ("OK", "WARN", "DRIFT")[worst],
                }, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

        elif job == "report":
            # 聚合 quality / health / drift 三份摘要，写一份总览 JSON
            import json
            import time as _t
            try:
                import check_data_quality as q
                import check_monthly_health as h
                r = {}
                g = group_label or str(cfg.app.default_group)
                r["generated_at"] = _t.strftime("%Y-%m-%d %H:%M:%S")
                r["group"] = g
                r["quality"] = q.run(feature_data, silent=True)
                r["health"] = h.run(feature_data, silent=True)
                # drift 复跑一遍；不想重复算可以在 quality/health 报告里先写临时文件
                import pandas as pd
                import joblib
                from tin_coating.modeling.residual_model import ResidualCorrectionModel  # noqa: F401
                from tin_coating.modeling.params import build_setpoint_group_key, get_feature_cols
                from tin_coating.diagnostics.drift import ks_test_drift
                df = pd.read_excel(feature_data)
                df = build_setpoint_group_key(df)
                df = df[df["Setpoint_Group_Label"] == g].copy()
                for s in ["Top", "Bot"]:
                    cs, sp = f"{s}_Current_Sum", "Speed[m/min]_Process_Avg"
                    if cs in df.columns and sp in df.columns:
                        df[f"{s}_Current_Per_Speed"] = df[cs] / (df[sp] + 1e-5)
                model_dir = cfg.resolve("paths.online_jobs")
                drift_verdicts = []
                for surface in ("Top", "Bot"):
                    feat_cols = get_feature_cols(surface)
                    online_col = f"Tin Weight_Actual[g/m2]_GALV_WEIGHT_{surface.upper()}_Avg"
                    model_path = model_dir / f"{surface}_model.joblib"
                    sub = df.dropna(subset=feat_cols + [online_col])
                    if model_path.exists() and len(sub) >= 60:
                        try:
                            m = joblib.load(model_path)
                            pred, _ = m.predict_smooth(sub[feat_cols], sub[online_col])
                            ks, p, md, v = ks_test_drift(pred.values - sub[online_col].values)
                            drift_verdicts.append({"surface": surface, "verdict": v,
                                                   "ks": ks, "p": p, "mean_diff": md})
                        except Exception as e:
                            drift_verdicts.append({"surface": surface, "verdict": f"SKIP:{type(e).__name__}"})
                    else:
                        drift_verdicts.append({"surface": surface, "verdict": "SKIP"})
                r["drift"] = drift_verdicts

                out_dir = cfg.resolve("paths.reports_dir")
                out_dir.mkdir(parents=True, exist_ok=True)
                out = out_dir / f"weekly_overview_{g}.json"
                out.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")

                q_pass = bool(r["quality"].get("pass"))
                h_verdict = r["health"].get("overall_verdict", "WARN")
                d_worst = "OK"
                for dv in drift_verdicts:
                    if dv.get("verdict") == "DRIFT":
                        d_worst = "DRIFT"
                        break
                    if dv.get("verdict") == "WARN":
                        d_worst = "WARN"
                combined = {"OK": "OK", "WARN": "WARN", "REGRESSION": "FAIL",
                            "DRIFT": "FAIL"}.get(h_verdict, "FAIL")
                if d_worst == "DRIFT":
                    combined = "FAIL"
                elif d_worst == "WARN" and combined != "FAIL":
                    combined = "WARN"
                if not q_pass:
                    combined = "FAIL"
                status_map = {"OK": ("ok", 0), "WARN": ("warn", 1), "FAIL": ("fail", 2)}
                status, exit_code = status_map[combined]
                detail = f"q_pass={q_pass} h_verdict={h_verdict} d_worst={d_worst}"
            except Exception as e:
                status, exit_code = "fail", 2
                detail = f"{type(e).__name__}: {e}"
        else:
            raise ValueError(f"未知 job: {job}，可选 {JOBS}")
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
