"""CLI 统一入口 — argparse 子命令风格，零外部依赖（不用 typer/click）。

子命令:
  train          按 config 跑正式训练（调用 tin_coating.modeling.group_train）
  cv             Walk-Forward 5-fold CV（调用 tin_coating.modeling.walk_forward_cv）
  predict        批量推理（调用 tin_coating.inference.pipeline）
  monitor        子命令：quality | health | retrain | approve | rollback

每个子命令都能独立 import，不互相耦合。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 保证从任何 cwd 调用都能找到 tin_coating 包
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tin_coating.config import load_config as _load_cfg
_CFG = _load_cfg()


def cmd_train(args):
    """正式训练。"""
    import subprocess
    cfg = args.config or str(_CFG.resolve("paths.prod_config"))
    print(f"[train] config={cfg}")
    subprocess.run([sys.executable, str(_ROOT / "coating_model_by_group.py"),
                    "--config", cfg], check=False)


def cmd_cv(args):
    """Walk-Forward 5-fold CV — 用子包独立模块。"""
    import pandas as pd
    from tin_coating.modeling import (
        load_pipeline_config, get_params_for_group,
        build_setpoint_group_key, run_cv_for_surface,
    )

    cfg_path = args.config or str(_CFG.resolve("paths.prod_config"))
    data_path = args.data or str(_CFG.resolve("paths.feature_data"))
    cfg = load_pipeline_config(cfg_path)

    target = args.group or _CFG.app.default_group
    surfaces = [args.surface] if args.surface else ["Top", "Bot"]

    df = pd.read_excel(data_path)
    df = build_setpoint_group_key(df)
    group_df = df[df['Setpoint_Group_Label'] == target].copy()
    print(f"[cv] group={target}  n={len(group_df)}  surfaces={surfaces}")

    for surface in surfaces:
        params = get_params_for_group(cfg, target, surface)
        cv = run_cv_for_surface(group_df, surface, params)
        rmse = cv["cv_rmse_mean"]
        std = cv["cv_rmse_std"]
        mae = cv["cv_mae_mean"]
        print(f"  {surface}  CV RMSE={rmse:.4f} ± {std:.4f}  MAE={mae:.4f}  folds={cv['n_folds']}")
        for d in cv["folds_detail"]:
            print(f"    {d['fold']}: RMSE={d['rmse']:.4f}  n_test={d['test_samples']}")


def cmd_predict(args):
    """批量推理。"""
    import pandas as pd
    import joblib
    from tin_coating.modeling.params import get_feature_cols

    model_dir = args.model_dir or str(_CFG.resolve("paths.online_jobs"))
    data_path = args.input
    output_path = args.output

    df = pd.read_excel(data_path)
    out_df = df.copy()

    for surface in ["Top", "Bot"]:
        model_path = Path(model_dir) / f"{surface}_model.joblib"
        if not model_path.exists():
            print(f"[predict] {surface}: model not found at {model_path}, skip")
            continue
        import joblib
        from tin_coating.modeling.residual_model import ResidualCorrectionModel  # noqa: F401  # (joblib 反序列化需要这个 import 让类注册到子包)

        corrector = joblib.load(model_path)
        feat_cols = get_feature_cols(surface)
        # 预先构造 Current_Per_Speed 等衍生列
        prefix = surface
        speed_col = "Speed[m/min]_Process_Avg"
        current_col = f"{prefix}_Current_Sum"
        online_col = f"Tin Weight_Actual[g/m2]_GALV_WEIGHT_{prefix.upper()}_Avg"

        # Schema 校验：必需列缺失就提前报错，不要静默 skip
        required = list(feat_cols) + [speed_col, current_col, online_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"[predict] {surface}: 输入 Excel 缺少必需列 {missing}，"
                f"请检查表头。必需列={required}"
            )

        df[f"{prefix}_Current_Per_Speed"] = df[current_col] / (df[speed_col] + 1e-5)

        pred, _ = corrector.predict_smooth(df[feat_cols], df[online_col])
        out_df[f"{surface}_Online_plus_correction"] = pred.values
        print(f"[predict] {surface}: {len(df)} rows  avg_correction={(pred.values - df[online_col].values).mean():.4f}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_excel(output_path, index=False)
    print(f"[predict] saved → {output_path}")


def _run_monitor(subcmd: str, args):
    """monitor 子命令路由 — 转发到 monitoring 目录下的对应脚本。"""
    import sys as _sys
    monitor_dir = _ROOT / "monitoring"
    _sys.path.insert(0, str(monitor_dir))
    default_featured = str(_CFG.resolve("paths.feature_data"))

    if subcmd == "quality":
        import check_data_quality as q
        r = q.run(args.data or default_featured, silent=False)
        return 0 if r["pass"] else 2
    elif subcmd == "health":
        import check_monthly_health as h
        r = h.run(args.data or default_featured, silent=False)
        return {"OK": 0, "WARN": 1, "REGRESSION": 2}.get(r["overall_verdict"], 1)
    elif subcmd == "retrain":
        import auto_retrain_candidate as r
        rep = r.run(args.data or default_featured,
                    force=args.force, skip_quality_check=args.skip_quality)
        if rep.get("pending_models"):
            return 0
        if rep.get("abort_reason"):
            return 2
        return 1
    elif subcmd == "list":
        import gate as g
        g.main([sys.argv[0], "--list", "--surface", args.surface or ""])
        return 0
    elif subcmd == "approve":
        import gate as g
        argv = ["--approve", "--surface", args.surface]
        if args.version:
            argv += ["--version", args.version]
        if args.yes:
            argv += ["--yes"]
        g.main([sys.argv[0]] + argv)
        return 0
    elif subcmd == "rollback":
        import gate as g
        argv = ["--rollback", "--surface", args.surface]
        if args.yes:
            argv += ["--yes"]
        g.main([sys.argv[0]] + argv)
        return 0
    else:
        print(f"unknown monitor sub-command: {subcmd}")
        return 1


def cmd_serve(args):
    """启动 7x24 在线推理服务。"""
    from tin_coating.service.server import serve
    return serve(host=args.host, port=args.port, group_label=args.group)


def cmd_ops(args):
    """无人值守运维任务：run（给调度器调用）/ selfcheck。"""
    from tin_coating.ops.scheduler import run_job
    from tin_coating.ops.selfcheck import run_selfcheck

    if args.op == "selfcheck":
        checks = run_selfcheck(args.group)
        failed = 0
        for c in checks:
            mark = "OK  " if c["ok"] else ("FAIL" if c["fatal"] else "WARN")
            print(f"[{mark}] {c['check']}: {c['detail']}")
            if not c["ok"] and c["fatal"]:
                failed += 1
        return 0 if failed == 0 else 2

    result = run_job(args.job, data=args.data, force=args.force, group_label=args.group)
    print(f"[ops] job={result['job']} status={result['status']} detail={result.get('detail','')}")
    return result["exit_code"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tin-coating",
                                description="镀层重量偏差校正 pipeline 统一 CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    # --- train ---
    t = sub.add_parser("train", help="按 config 跑正式训练")
    t.add_argument("--config", default=None, help="JSON 配置路径 (默认 group_params_2_799_only.json)")
    t.set_defaults(func=cmd_train)

    # --- cv ---
    c = sub.add_parser("cv", help="Walk-Forward 5-fold CV 单规格组")
    c.add_argument("--config", default=None)
    c.add_argument("--data", default=None)
    c.add_argument("--group", default=_CFG.app.default_group)
    c.add_argument("--surface", default=None, help="Top / Bot，不指定就两个都跑")
    c.set_defaults(func=cmd_cv)

    # --- predict ---
    pr = sub.add_parser("predict", help="批量推理")
    pr.add_argument("--input", required=True, help="Excel 输入路径")
    pr.add_argument("--output", required=True, help="Excel 输出路径")
    pr.add_argument("--model-dir", default=None)
    pr.set_defaults(func=cmd_predict)

    # --- serve ---
    s = sub.add_parser("serve", help="启动 7x24 在线推理服务（/healthz /readyz /predict /metrics）")
    s.add_argument("--host", default=None)
    s.add_argument("--port", type=int, default=None)
    s.add_argument("--group", default=None)
    s.set_defaults(func=cmd_serve)

    # --- ops ---
    o = sub.add_parser("ops", help="无人值守运维：run / selfcheck")
    o.add_argument("op", choices=["run", "selfcheck"])
    o.add_argument("--job", default=None, choices=["quality", "health", "retrain"])
    o.add_argument("--data", default=None)
    o.add_argument("--group", default=None)
    o.add_argument("--force", action="store_true")
    o.set_defaults(func=cmd_ops)

    # --- monitor ---
    m = sub.add_parser("monitor", help="监控子命令: quality|health|retrain|list|approve|rollback")
    m.add_argument("subcmd", choices=["quality", "health", "retrain", "list", "approve", "rollback"])
    m.add_argument("--data", default=None)
    m.add_argument("--surface", default="Top")
    m.add_argument("--version", default=None)
    m.add_argument("--force", action="store_true")
    m.add_argument("--skip-quality", action="store_true")
    m.add_argument("--yes", action="store_true", help="approve / rollback 需要")
    m.set_defaults(func=None)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "monitor":
        return _run_monitor(args.subcmd, args)

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
