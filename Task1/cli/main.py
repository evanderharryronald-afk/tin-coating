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
    """正式训练 — 调用 tin_coating.modeling.group_train.run_surface_pipeline。

    产物落盘：result/trained_models/by_group/<group>/{Top,Bot}_model.joblib + metadata.json。
    """
    import pandas as pd
    from tin_coating.modeling import load_pipeline_config, get_params_for_group
    from tin_coating.modeling.group_train import run_surface_pipeline
    from tin_coating.modeling.params import build_setpoint_group_key, get_feature_cols
    from tin_coating.modeling.persistence import ModelSaver, ModelMetadata

    cfg_path = args.config or str(_CFG.resolve("paths.prod_config"))
    data_path = str(_CFG.resolve("paths.feature_data"))
    target = _CFG.app.default_group
    print(f"[train] config={cfg_path}  group={target}  data={data_path}")

    pipeline_cfg = load_pipeline_config(cfg_path)
    df = pd.read_excel(data_path)
    df = build_setpoint_group_key(df)
    group_df = df[df["Setpoint_Group_Label"] == target].copy()
    if len(group_df) == 0:
        print(f"[train] 过滤到 group={target} 后为空 — 检查数据")
        return 2

    # 确保衍生特征存在
    for surface in ['Top', 'Bot']:
        prefix = surface
        speed_col = "Speed[m/min]_Process_Avg"
        current_col = f"{prefix}_Current_Sum"
        if current_col in group_df.columns and speed_col in group_df.columns:
            if f"{prefix}_Current_Per_Speed" not in group_df.columns:
                group_df[f"{prefix}_Current_Per_Speed"] = (
                    group_df[current_col] / (group_df[speed_col] + 1e-5)
                )

    saver = ModelSaver(base_dir="result/trained_models", compress=True)
    rc = 0
    for surface in ["Top", "Bot"]:
        try:
            params = get_params_for_group(pipeline_cfg, target, surface)
            model, metrics = run_surface_pipeline(
                group_df, surface=surface, group_tag=target, group_params=params,
                train_ratio=0.65, val_ratio=0.20,
            )
            rmse = metrics.get("RMSE_模型_测试")
            n_train = int(metrics.get("训练样本数", 0))
            feat_cols = get_feature_cols(surface)
            meta = ModelMetadata(
                model_type=model.__class__.__name__,
                surface=surface,
                feature_names=feat_cols,
                hyperparameters=params,
                group_label=target,
                training_samples=n_train,
                training_metrics=metrics,
                source_script="cli.main:train",
            )
            info = saver.save_model(model, meta, surface=surface, group_label=target)
            print(f"  ✅ {surface}: Test RMSE={rmse:.4f}  → {info['model_path']}")
        except Exception as e:
            print(f"  ❌ {surface}: 训练/保存失败 — {e}")
            rc = 2
    return rc


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


def cmd_eda(args):
    """数据分析子命令：dist / corr / mi / surface（相关性 + 互信息 + 分组 EDA）。

    --group 传 'all' 时不过滤规格组，对全体数据分析；传具体规格组名时精确定位。
    """
    import pandas as pd
    from pathlib import Path

    data_path = args.data or str(_CFG.resolve("paths.feature_data"))
    group = args.group or _CFG.app.default_group
    surface = args.surface
    df = pd.read_excel(data_path)

    # 过滤到指定规格组（group="all" → 不过滤）
    if group and group.lower() != "all":
        from tin_coating.modeling.params import build_setpoint_group_key
        df = build_setpoint_group_key(df)
        df = df[df["Setpoint_Group_Label"] == group].copy()
        print(f"[eda] group={group} n={len(df)}")
    else:
        print(f"[eda] group=all n={len(df)}")

    # 确保有 _Current_Per_Speed 衍生列
    for s in ["Top", "Bot"]:
        cs, sp = f"{s}_Current_Sum", "Speed[m/min]_Process_Avg"
        if cs in df.columns and sp in df.columns and f"{s}_Current_Per_Speed" not in df.columns:
            df[f"{s}_Current_Per_Speed"] = df[cs] / (df[sp] + 1e-5)

    out_dir = Path(args.output) if args.output else Path("result/eda") / (group or "all")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.sub == "corr":
        from tin_coating.eda.analyzer import SurfaceCorrelationAnalyzer
        ana = SurfaceCorrelationAnalyzer(default_save_dir=str(out_dir))
        surfaces = [surface] if surface else ["Top", "Bot"]
        for s in surfaces:
            print(f"\n== corr {s} ==")
            ana.analyze_surface(df, surface=s, save_dir=str(out_dir / s),
                               corr_method="both", compute_mi=True, compute_dcor=False,
                               compute_mi_matrix=False, compute_dcor_matrix=False)
    elif args.sub == "surface":
        from tin_coating.eda.grouping import GroupEDADiagnoser
        surfaces = [surface] if surface else ["Top", "Bot"]
        d = GroupEDADiagnoser(surfaces=surfaces,
                              pre_diagnosis_dir=str(out_dir / "pre"),
                              post_diagnosis_dir=str(out_dir / "post"))
        d.pre_modeling_diagnosis(df, group_label=group)
    else:  # dist 默认
        from tin_coating.modeling.params import get_feature_cols as _get_feat
        surfaces = [surface] if surface else ["Top", "Bot"]
        for s in surfaces:
            cols = _get_feat(s)
            exist = [c for c in cols if c in df.columns]
            if not exist:
                continue
            print(f"\n== dist {s} ==")
            stats = df[exist].describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
            stats.to_csv(out_dir / f"dist_{s}.csv", encoding="utf-8-sig")
            print(stats.round(3).to_string())
    print(f"[eda] saved → {out_dir}")
    return 0


def cmd_diagnose(args):
    """诊断子命令：residual（残差 + QQ + ACF）/ drift（漂移 KS）/ interpret（SHAP/PDP）。

    --group 传 'all' 时遍历所有规格组（n>=200 的那些，小样本跳过），
    逐个加载对应组的模型，逐个跑诊断，落盘到 result/diagnostics/all/<group>/。
    对全体做 EDA 不需要模型，但 residual/drift/interpret 都是模型残差层面
    的诊断，必须 per-group 逐一对齐模型产物。
    """
    import pandas as pd
    from pathlib import Path

    data_path = args.data or str(_CFG.resolve("paths.feature_data"))
    group = args.group or _CFG.app.default_group
    surface = args.surface
    df = pd.read_excel(data_path)

    from tin_coating.modeling.params import build_setpoint_group_key, get_feature_cols
    df = build_setpoint_group_key(df)

    # 确定要处理的规格组列表
    if group and group.lower() != "all":
        groups = [group]
    else:
        counts = df["Setpoint_Group_Label"].value_counts()
        groups = counts[counts >= 200].index.tolist()
        print(f"[diagnose] group=all → {len(groups)} groups with n>=200 "
              f"(skipped {len(counts) - len(groups)} small groups)")

    # 构造衍生列（一次性对全体做，比分组里重复做快）
    for s in ["Top", "Bot"]:
        cs, sp = f"{s}_Current_Sum", "Speed[m/min]_Process_Avg"
        if cs in df.columns and sp in df.columns and f"{s}_Current_Per_Speed" not in df.columns:
            df[f"{s}_Current_Per_Speed"] = df[cs] / (df[sp] + 1e-5)

    out_root = Path(args.output) if args.output else Path("result/diagnostics") / (group or "all")
    out_root.mkdir(parents=True, exist_ok=True)

    for g in groups:
        sub = df[df["Setpoint_Group_Label"] == g].copy()
        if len(sub) < 60:
            continue
        sub_out = out_root / g
        sub_out.mkdir(parents=True, exist_ok=True)

        if args.sub in ("residual", "drift"):
            _diag_model_based(sub, g, surface, sub_out, args.sub)
        elif args.sub == "interpret":
            _diag_interpret(sub, g, surface, sub_out)

    print(f"[diagnose] saved → {out_root}")
    return 0


def _diag_model_based(df, group, surface, out_dir, sub):
    """diagnose residual / drift 共享内核：load model → predict → residual → analyse。"""
    import joblib
    from tin_coating.modeling.residual_model import ResidualCorrectionModel  # noqa
    from tin_coating.modeling.params import get_feature_cols
    from tin_coating.config import load_config

    online_dir = load_config().resolve("paths.online_jobs")
    surfaces = [surface] if surface else ["Top", "Bot"]

    for s in surfaces:
        feat_cols = get_feature_cols(s)
        # paths.online_jobs 已经是带组名后缀的目录，固定读它（2.799 组线上模型）
        # 若以后多规格组部署了各自模型，这里按 group 扩展路径即可
        model_path = online_dir / f"{s}_model.joblib"
        online_col = f"Tin Weight_Actual[g/m2]_GALV_WEIGHT_{s.upper()}_Avg"
        if not model_path.exists() or online_col not in df.columns:
            print(f"[{sub}] {group}/{s}: model or online_col missing, skip")
            continue
        try:
            m = joblib.load(model_path)
        except Exception as e:
            print(f"[{sub}] {group}/{s}: model load failed ({type(e).__name__}), skip")
            continue

        sub_df = df.dropna(subset=feat_cols + [online_col]).copy()
        if len(sub_df) < 60:
            print(f"[{sub}] {group}/{s}: n={len(sub_df)} too small, skip")
            continue

        pred, _ = m.predict_smooth(sub_df[feat_cols], sub_df[online_col])
        residuals = pred.values - sub_df[online_col].values

        if sub == "drift":
            from tin_coating.diagnostics.drift import ks_test_drift
            ks, p, md, verdict = ks_test_drift(residuals)
            print(f"[drift] {group}/{s}: KS={ks:.4f} p={p:.4g} mean_diff={md:.4f} verdict={verdict}")
            (out_dir / s).mkdir(parents=True, exist_ok=True)
            (out_dir / s / "drift_summary.txt").write_text(
                f"group={group}\nsurface={s}\nn={len(residuals)}\n"
                f"ks_stat={ks}\nks_p={p}\nmean_diff={md}\nverdict={verdict}\n",
                encoding="utf-8"
            )
        else:  # residual
            from tin_coating.diagnostics.residual import run_full_residual_diagnostics
            n = len(sub_df)
            split = int(n * 0.8)
            online = sub_df[online_col].values
            diag = run_full_residual_diagnostics(
                y_true_train=online[:split], y_true_test=online[split:],
                pred_train=pred.values[:split], pred_test=pred.values[split:],
                X_train=sub_df[feat_cols].iloc[:split],
                X_test=sub_df[feat_cols].iloc[split:],
                label=f"{group}/{s}表面",
                save_dir=str(out_dir / s / "residual"),
                tag=f"{group}_{s}", run_temporal=True, run_explainability=True,
            )
            top = diag['feature_significance'].head(8)
            print(f"[residual] {group}/{s}: feature_significance top=\n{top.to_string(index=False)}")


def _diag_interpret(df, group, surface, out_dir):
    """diagnose interpret（Permutation + SHAP + PDP，SHAP 可选）。"""
    import joblib
    from tin_coating.modeling.residual_model import ResidualCorrectionModel  # noqa
    from tin_coating.modeling.params import get_feature_cols
    from tin_coating.config import load_config

    online_dir = load_config().resolve("paths.online_jobs")
    surfaces = [surface] if surface else ["Top", "Bot"]

    for s in surfaces:
        feat_cols = get_feature_cols(s)
        model_path = online_dir / f"{s}_model.joblib"
        online_col = f"Tin Weight_Actual[g/m2]_GALV_WEIGHT_{s.upper()}_Avg"
        cn = "上" if s == "Top" else "下"
        lab_col = f"{cn}表面镀层重量A(XA1_0)"
        required = feat_cols + [online_col, lab_col]
        missing = [c for c in required if c not in df.columns]
        if missing or not model_path.exists():
            print(f"[interpret] {group}/{s}: missing={missing} or model absent, skip")
            continue
        try:
            model = joblib.load(model_path)
        except Exception as e:
            print(f"[interpret] {group}/{s}: model load failed ({type(e).__name__}), skip")
            continue
        sub_df = df.dropna(subset=required)
        if len(sub_df) < 80:
            print(f"[interpret] {group}/{s}: n={len(sub_df)} too small, skip")
            continue
        X = sub_df[feat_cols]
        y_delta = sub_df[lab_col] - sub_df[online_col]
        try:
            from tin_coating.eda.interpreter import ModelInterpreter
            interpreter = ModelInterpreter(
                model=model, X=X, feature_names=feat_cols,
                save_dir=str(out_dir / s / "interpretation"),
            )
            interpreter.full_analysis(y=y_delta, run_permutation=True,
                                      run_shap=True, run_pdp=True)
        except ImportError as e:
            print(f"[interpret] {group}/{s}: {e}（缺 shap/statsmodels？）→ 仅跑 permutation")
            try:
                from tin_coating.eda.interpreter import ModelInterpreter
                interpreter = ModelInterpreter(
                    model=model, X=X, feature_names=feat_cols,
                    save_dir=str(out_dir / s / "interpretation"),
                )
                interpreter.permutation_importance(y=y_delta)
            except Exception as e2:
                print(f"[interpret] {group}/{s}: still failed → {e2}")
        except Exception as e:
            print(f"[interpret] {group}/{s}: {type(e).__name__}: {e}")


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

    # --- eda ---
    ed = sub.add_parser("eda", help="EDA 数据分析：dist|corr|surface")
    ed.add_argument("sub", choices=["dist", "corr", "surface"],
                    help="dist=描述统计; corr=相关性+MI+DCOR; surface=分组 EDA 诊断")
    ed.add_argument("--data", default=None)
    ed.add_argument("--group", default=_CFG.app.default_group)
    ed.add_argument("--surface", default=None)
    ed.add_argument("--output", default=None, help="输出目录（默认 result/eda/<group>）")
    ed.set_defaults(func=cmd_eda)

    # --- diagnose ---
    dg = sub.add_parser("diagnose", help="模型诊断：residual|drift|interpret")
    dg.add_argument("sub", choices=["residual", "drift", "interpret"],
                    help="residual=分布+QQ+ACF; drift=KS 漂移; interpret=SHAP+Permutation+PDP")
    dg.add_argument("--data", default=None)
    dg.add_argument("--group", default=_CFG.app.default_group)
    dg.add_argument("--surface", default=None)
    dg.add_argument("--output", default=None, help="输出目录（默认 result/diagnostics/<group>）")
    dg.set_defaults(func=cmd_diagnose)

    # --- serve ---
    s = sub.add_parser("serve", help="启动 7x24 在线推理服务（/healthz /readyz /predict /metrics）")
    s.add_argument("--host", default=None)
    s.add_argument("--port", type=int, default=None)
    s.add_argument("--group", default=None)
    s.set_defaults(func=cmd_serve)

    # --- ops ---
    o = sub.add_parser("ops", help="无人值守运维：run / selfcheck")
    o.add_argument("op", choices=["run", "selfcheck"])
    o.add_argument("--job", default=None, choices=["quality", "health", "retrain", "drift", "report"])
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
