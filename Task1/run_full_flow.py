"""一键跑通全流程（本地自检用，不部署、不改线上模型）。

用法（在 Task1/ 目录下）:
  python run_full_flow.py                 # 完整：自检 → 训练 → CV → 质量 → 健康 → 批量推理
  python run_full_flow.py --quick         # 快速：跳过训练/CV，只跑自检 + 质量 + 健康 + 200 行推理
  python run_full_flow.py --skip-train    # 复用已有模型，其余照跑
  python run_full_flow.py --with-retrain  # 额外跑一次重训候选（生成 pending，不部署）

说明:
  - 训练写的是 result/trained_models/by_group/<group>/ 下的产物，不会自动上线；
    推理阶段和线上服务一致，走 registry 里 active 版本，因此要不要部署由 gate 决定。
  - 退出码：0 全部正常；1 有非致命项（质量/健康告警）；2 有致命失败。
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "monitoring") not in sys.path:
    sys.path.insert(0, str(ROOT / "monitoring"))

_STAGES: list[dict] = []


def _cfg():
    """延迟导入配置：保证顶层没有任何 tin_coating 副作用 import。"""
    from tin_coating.config import load_config
    return load_config()


def _stage(name: str):
    """装饰器：统一记录每个阶段的耗时与结果。"""
    def deco(fn):
        def wrapper(*args, **kwargs):
            print(f"\n{'=' * 70}\n[stage] {name}\n{'=' * 70}")
            t0 = time.time()
            ok, detail = True, ""
            try:
                detail = fn(*args, **kwargs) or ""
            except Exception as e:
                ok = False
                detail = f"{type(e).__name__}: {e}"
                traceback.print_exc()
            elapsed = round(time.time() - t0, 2)
            _STAGES.append({"name": name, "ok": ok, "detail": detail, "elapsed": elapsed})
            print(f"[stage] {name} {'OK' if ok else 'FAIL'} ({elapsed}s) {detail}")
            return ok
        return wrapper
    return deco


# ---------------------------------------------------------------------------
# 各阶段
# ---------------------------------------------------------------------------
@_stage("启动自检")
def stage_selfcheck(group: str) -> str:
    from tin_coating.ops.selfcheck import run_selfcheck

    checks = run_selfcheck(group)
    fatal = [c for c in checks if not c["ok"] and c["fatal"]]
    for c in checks:
        mark = "OK  " if c["ok"] else ("FAIL" if c["fatal"] else "WARN")
        print(f"  [{mark}] {c['check']}: {c['detail']}")
    if fatal:
        raise RuntimeError("自检存在致命项: " + ", ".join(c["check"] for c in fatal))
    return f"{len(checks)} 项检查通过"


@_stage("正式训练")
def stage_train(group: str) -> str:
    import subprocess

    cfg = _cfg()
    cmd = [sys.executable, str(ROOT / "coating_model_by_group.py"),
           "--config", str(cfg.resolve("paths.prod_config"))]
    print("  " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        raise RuntimeError(f"训练脚本退出码 {r.returncode}")
    model_dir = cfg.resolve("paths.online_jobs")
    return f"模型目录 {model_dir}"


@_stage("Walk-Forward CV")
def stage_cv(group: str) -> str:
    import pandas as pd
    from tin_coating.modeling import (
        build_setpoint_group_key, get_params_for_group,
        load_pipeline_config, run_cv_for_surface,
    )

    cfg = _cfg()
    cfg_pipe = load_pipeline_config(str(cfg.resolve("paths.prod_config")))
    df = pd.read_excel(cfg.resolve("paths.feature_data"))
    df = build_setpoint_group_key(df)
    gdf = df[df["Setpoint_Group_Label"] == group].copy()
    if gdf.empty:
        raise ValueError(f"规格组 {group} 没有数据")

    parts = []
    for surface in ("Top", "Bot"):
        params = get_params_for_group(cfg_pipe, group, surface)
        cv = run_cv_for_surface(gdf, surface, params)
        parts.append(f"{surface} RMSE={cv['cv_rmse_mean']:.4f}±{cv['cv_rmse_std']:.4f}")
        print(f"  {surface}: CV RMSE={cv['cv_rmse_mean']:.4f} ± {cv['cv_rmse_std']:.4f} "
              f"MAE={cv['cv_mae_mean']:.4f} folds={cv['n_folds']}")
    return " | ".join(parts)


@_stage("数据质量门")
def stage_quality(group: str) -> str:
    import check_data_quality as q

    r = q.run(str(_cfg().resolve("paths.feature_data")), silent=False)
    if not r.get("pass"):
        raise RuntimeError("质量门未通过，详见 monitoring/reports/")
    return "PASS"


@_stage("月度健康检查")
def stage_health(group: str) -> str:
    import check_monthly_health as h

    r = h.run(str(_cfg().resolve("paths.feature_data")), silent=False)
    verdict = r.get("overall_verdict", "WARN")
    if verdict == "REGRESSION":
        raise RuntimeError("健康检查判定 REGRESSION")
    if verdict != "OK":
        print(f"  ⚠️ verdict={verdict}（非致命，详见 monitoring/reports/）")
    return f"verdict={verdict}"


@_stage("批量推理")
def stage_predict(group: str, rows: int, output: Path) -> str:
    import pandas as pd

    import tin_coating.modeling.residual_model  # noqa: F401  (joblib 反序列化依赖)
    from tin_coating.modeling.params import get_feature_cols
    from tin_coating.service.models import ModelStore

    cfg = _cfg()
    store = ModelStore(group, cfg=cfg)
    loaded = store.load_all()
    missing = [s for s, ok in loaded.items() if not ok]
    if missing:
        raise RuntimeError(f"模型未就绪: {missing}")

    df = pd.read_excel(cfg.resolve("paths.feature_data"), nrows=rows)
    out = df.copy()
    for surface in ("Top", "Bot"):
        speed_col = "Speed[m/min]_Process_Avg"
        current_col = f"{surface}_Current_Sum"
        online_col = f"Tin Weight_Actual[g/m2]_GALV_WEIGHT_{surface.upper()}_Avg"
        out[f"{surface}_Current_Per_Speed"] = out[current_col] / (out[speed_col] + 1e-5)

        model = store.get(surface)
        feat = get_feature_cols(surface)
        pred, delta = model.predict_smooth(out[feat], out[online_col])
        out[f"{surface}_Online_plus_correction"] = pred.values
        out[f"{surface}_Correction_Delta"] = delta.values
        print(f"  {surface}: {len(out)} 行  平均修正量={delta.values.mean():+.4f} g/m2")

    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_excel(output, index=False)
    versions = store.status()["models"]

    def _tag(surface):
        m = versions[surface]
        return f"{m['version']}{'(fallback)' if m['fallback'] else ''}"

    return (f"→ {output}  模型版本 "
            f"Top={_tag('Top')} Bot={_tag('Bot')}")


@_stage("重训候选")
def stage_retrain(group: str) -> str:
    import auto_retrain_candidate as r

    rep = r.run(str(_cfg().resolve("paths.feature_data")), force=False)
    if rep.get("abort_reason"):
        raise RuntimeError(rep["abort_reason"])
    if rep.get("pending_models"):
        return f"生成 pending {len(rep['pending_models'])} 个（需 gate 审批）"
    return f"跳过 {len(rep.get('skipped') or [])} 项（已有待审批候选或有失败）"


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> int:
    cfg = _cfg()
    p = argparse.ArgumentParser(description="tin-coating 全流程一键运行")
    p.add_argument("--group", default=cfg.app.default_group)
    p.add_argument("--quick", action="store_true", help="跳过训练/CV，只跑自检+质量+健康+少量推理")
    p.add_argument("--skip-train", action="store_true", help="复用已有模型，不重新训练")
    p.add_argument("--skip-cv", action="store_true", help="跳过 Walk-Forward CV")
    p.add_argument("--with-retrain", action="store_true", help="额外生成重训候选（不部署）")
    p.add_argument("--rows", type=int, default=200, help="推理样本行数（默认 200）")
    p.add_argument("--output", default=None, help="推理输出 Excel 路径")
    args = p.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    output = Path(args.output) if args.output else (
        cfg.root / "result" / "predictions" / f"full_flow_{stamp}.xlsx")

    t0 = time.time()
    print(f"[flow] group={args.group} quick={args.quick} "
          f"train={'no' if (args.quick or args.skip_train) else 'yes'}")

    if not stage_selfcheck(args.group):
        return 2                                   # 自检失败直接停

    if not (args.quick or args.skip_train):
        if not stage_train(args.group):
            return 2                               # 训练失败直接停

    if not (args.quick or args.skip_cv):
        stage_cv(args.group)

    stage_quality(args.group)
    stage_health(args.group)

    if not stage_predict(args.group, args.rows, output):
        return 2                                   # 推理失败直接停

    if args.with_retrain:
        stage_retrain(args.group)

    # ---- 汇总 ----
    print(f"\n{'=' * 70}\n[flow] 汇总（总耗时 {round(time.time() - t0, 2)}s）\n{'=' * 70}")
    for s in _STAGES:
        print(f"  {'✅' if s['ok'] else '❌'} {s['name']:<10} {s['elapsed']:>7.2f}s  {s['detail']}")

    failed = [s for s in _STAGES if not s["ok"]]
    if failed:
        print(f"\n[flow] 有 {len(failed)} 个阶段失败: "
              f"{', '.join(s['name'] for s in failed)}")
        return 2

    if any("(fallback)" in s["detail"] for s in _STAGES):
        print("\n⚠️  当前环境与 registry 归档模型不兼容（版本不一致），推理已用 live 目录产物兜底。")
        print("   建议统一环境：在 .venv 执行 pip install -r requirements.txt"
              "（锁 sklearn<1.5 / pandas<2.2），或在同一环境重训后走 gate 审批。")

    print("\n[flow] 全流程完成。部署模型请走: python monitoring/gate.py --list → --approve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
