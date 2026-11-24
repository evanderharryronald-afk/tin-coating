"""一键跑通全流程（本地自检用，不部署、不改线上模型）。

用法（在 Task1/ 目录下）:
  python run_full_flow.py                 # 完整：自检 → 训练 → CV → EDA+诊断 → 质量 → 健康 → 推理
  python run_full_flow.py --quick         # 快速：跳过训练/CV/EDA+诊断，只跑自检 + 质量 + 健康 + 200 行推理
  python run_full_flow.py --skip-train    # 复用已有模型，其余照跑
  python run_full_flow.py --skip-analytics  # 跳过 EDA + 残差/漂移/解释性分析（省时间）
  python run_full_flow.py --with-retrain  # 额外跑一次重训候选（生成 pending，不部署）
  python run_full_flow.py --version 20261124_133722  # 手动指定 version 目录（默认当前时间戳）

说明:
  - 训练写的是 result/trained_models/by_group/<group>/ 下的产物，不会自动上线；
    推理阶段和线上服务一致，走 registry 里 active 版本，因此要不要部署由 gate 决定。
  - 每次运行生成一个唯一 version=YYYYMMDD_HHMMSS，所有分析产物落到
    result/<module>/<group>/<version>/ 下，同时写 LATEST 指针 + _run_meta.json
    记录本次运行的参数、耗时、exit code。
  - 退出码：0 全部正常；1 有非致命项（质量/健康告警）；2 有致命失败。
"""
from __future__ import annotations

import argparse
import json as _json
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
_VERSION: str = ""  # 本次运行唯一 version，main() 里赋值后全程共用


def _vroot(*parts: str) -> Path:
    """统一带 version 的产物根目录。
    例: _vroot('eda', 'Top2.799_Bot2.799')
     → result/eda/Top2.799_Bot2.799/20261124_133722/
    """
    return ROOT / "result" / parts[0] / "/".join(parts[1:]) / _VERSION


def _mark_latest(version_dir: Path, module: str) -> None:
    """写 LATEST 指针文件，内容是 version 目录的 basename。
    例: result/eda/<group>/_LATEST.txt  →  20261124_133722
    """
    version_dir.parent.mkdir(parents=True, exist_ok=True)
    (version_dir.parent / "_LATEST.txt").write_text(version_dir.name, encoding="utf-8")


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
    import pandas as pd

    cfg = _cfg()
    cfg_path = str(cfg.resolve("paths.prod_config"))
    data_path = str(cfg.resolve("paths.feature_data"))
    print(f"  config={cfg_path}")

    from tin_coating.modeling import load_pipeline_config, get_params_for_group
    from tin_coating.modeling.group_train import run_surface_pipeline
    from tin_coating.modeling.params import build_setpoint_group_key, get_feature_cols
    from tin_coating.modeling.persistence import ModelSaver, ModelMetadata

    pipeline_cfg = load_pipeline_config(cfg_path)
    df = pd.read_excel(data_path)
    df = build_setpoint_group_key(df)
    group_df = df[df["Setpoint_Group_Label"] == group].copy()
    if len(group_df) == 0:
        raise RuntimeError(f"group={group} 过滤后为空")

    saver = ModelSaver(base_dir="result/trained_models", compress=True)
    for surface in ["Top", "Bot"]:
        params = get_params_for_group(pipeline_cfg, group, surface)
        model, metrics = run_surface_pipeline(
            group_df, surface=surface, group_tag=group, group_params=params,
            train_ratio=0.65, val_ratio=0.20,
        )
        meta = ModelMetadata(
            model_type=model.__class__.__name__, surface=surface,
            feature_names=get_feature_cols(surface), hyperparameters=params,
            group_label=group, training_samples=int(metrics.get("训练样本数", 0)),
            training_metrics=metrics, source_script="run_full_flow",
        )
        saver.save_model(model, meta, surface=surface, group_label=group)

    model_dir = cfg.resolve("paths.online_jobs")
    return f"模型目录 {model_dir}"


@_stage("Walk-Forward CV")
def stage_cv(group: str) -> str:
    import pandas as pd
    import json as _json
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
    summary = {"group": group, "n": len(gdf), "surfaces": {}}
    for surface in ("Top", "Bot"):
        params = get_params_for_group(cfg_pipe, group, surface)
        cv = run_cv_for_surface(gdf, surface, params)
        parts.append(f"{surface} RMSE={cv['cv_rmse_mean']:.4f}±{cv['cv_rmse_std']:.4f}")
        print(f"  {surface}: CV RMSE={cv['cv_rmse_mean']:.4f} ± {cv['cv_rmse_std']:.4f} "
              f"MAE={cv['cv_mae_mean']:.4f} folds={cv['n_folds']}")
        summary["surfaces"][surface] = {
            "cv_rmse_mean": float(cv["cv_rmse_mean"]),
            "cv_rmse_std": float(cv["cv_rmse_std"]),
            "cv_mae_mean": float(cv["cv_mae_mean"]),
            "n_folds": cv["n_folds"],
            "folds": cv["folds_detail"],
        }

    # 落盘 JSON（带 version 目录）
    cv_json = _vroot("cv", group) / "cv_summary.json"
    cv_json.parent.mkdir(parents=True, exist_ok=True)
    cv_json.write_text(_json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _mark_latest(_vroot("cv", group), "cv")
    return " | ".join(parts) + f"  → {cv_json}"


# ----- EDA / Diagnose / Drift / Interp：非致命，fail 不中断主流程 -----
def _eda_and_diagnostics(group: str) -> str:
    """训练后 EDA + 诊断套件。每个子步骤 fail 单独 try/except，不中断主流程。
    所有产物落到 result/<module>/<group>/<version>/ 下，同时写 LATEST 指针。"""
    import pandas as pd, joblib, traceback as tb
    from tin_coating.config import load_config
    from tin_coating.modeling.params import (
        build_setpoint_group_key, get_feature_cols,
    )
    from tin_coating.modeling.residual_model import ResidualCorrectionModel  # side-effect for pickle

    cfg = load_config()
    df = pd.read_excel(str(cfg.resolve("paths.feature_data")))
    df = build_setpoint_group_key(df)
    gdf = df[df["Setpoint_Group_Label"] == group].copy()
    for s in ('Top', 'Bot'):
        cs, sp = f'{s}_Current_Sum', 'Speed[m/min]_Process_Avg'
        if cs in gdf.columns and sp in gdf.columns:
            gdf[f'{s}_Current_Per_Speed'] = gdf[cs] / (gdf[sp] + 1e-5)

    lab_map = {'Top': '上表面镀层重量A(XA1_0)', 'Bot': '下表面镀层重量A(XA1_0)'}
    online_map = {'Top': 'Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Avg',
                  'Bot': 'Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Avg'}

    results: list[str] = []
    eda_root = _vroot("eda", group)
    diag_root = _vroot("diagnostics", group)

    def _safe(name: str, fn):
        try:
            fn()
            print(f"  ✅ {name}")
            results.append(name)
        except Exception as e:
            print(f"  ⚠️ {name} 跳过（非致命）— {type(e).__name__}: {e}")
            tb.print_exc()
            results.append(f"{name}(skip)")

    # 1. EDA corr — 完整 pearson/spearman + MI + DCOR（每面 12 png）
    def _corr():
        from tin_coating.eda.correlation import SurfaceCorrelationAnalyzer
        ana = SurfaceCorrelationAnalyzer(default_save_dir=str(eda_root))
        for s in ('Top', 'Bot'):
            sub = gdf.dropna(subset=get_feature_cols(s) + [lab_map[s]])
            ana.analyze_surface(
                sub, surface=s,
                save_dir=str(eda_root / s),
                corr_method='both',
                compute_mi=True, compute_dcor=True,
                compute_mi_matrix=True, compute_dcor_matrix=True,
            )
        _mark_latest(eda_root, "eda")
    _safe('eda corr (Top+Bot × 12 png)', _corr)

    # 2. EDA surface pre-diagnosis
    def _surface():
        from tin_coating.eda.grouping import GroupEDADiagnoser
        d = GroupEDADiagnoser(surfaces=['Top', 'Bot'],
                              pre_diagnosis_dir=str(eda_root / "pre"),
                              post_diagnosis_dir=str(eda_root / "post"))
        d.pre_modeling_diagnosis(gdf, group_label=group)
    _safe('eda surface pre-diagnosis', _surface)

    # 3. 残差诊断 + 漂移 + 解释性（每面各一份）
    for s in ('Top', 'Bot'):
        feat = get_feature_cols(s)
        sub = gdf.dropna(subset=feat + [lab_map[s]]).copy()
        if sub.empty:
            print(f"  ⚠️ {s} 表面无有效数据，跳过诊断")
            continue
        model_path = ROOT / "result" / "trained_models" / "by_group" / group / f"{s}_model.joblib"
        if not model_path.exists():
            print(f"  ⚠️ {s}_model.joblib 不存在，跳过诊断")
            continue
        m = joblib.load(model_path)
        online = sub[online_map[s]].values
        lab = sub[lab_map[s]].values
        sub[f'{s}_Delta'] = lab - online
        model_out_dir = diag_root / s

        def _residual():
            from tin_coating.diagnostics.residual import run_full_residual_diagnostics
            n = len(sub); split = int(n * 0.8)
            pred, _ = m.predict_smooth(sub[feat], online)
            # 关键：显式把 residuals 和 X 的 index 对齐。如果不传 index，
            # run_full_residual_diagnostics 内部 residuals 是纯 Series
            # （index=RangeIndex 0..split-1），而 X_test.index 是原始 DataFrame 的
            # （可能是 1829..2812），feature_relationship_diagnostics 里
            # residuals.reindex(X.index) 就会全 NaN → 推断出 0 个特征。
            yt_train = pd.Series(online[:split].astype(float), index=sub.index[:split])
            yt_test = pd.Series(online[split:].astype(float), index=sub.index[split:])
            p_train = pd.Series(pred.values[:split], index=sub.index[:split])
            p_test = pd.Series(pred.values[split:], index=sub.index[split:])
            run_full_residual_diagnostics(
                y_true_train=yt_train, y_true_test=yt_test,
                pred_train=p_train, pred_test=p_test,
                X_train=sub[feat].iloc[:split], X_test=sub[feat].iloc[split:],
                label=f"{group}/{s}表面",
                save_dir=str(model_out_dir / "residual"),
                tag=f"{group}_{s}", run_temporal=True, run_explainability=True,
                top_n_features=5,  # 防止 8 特征 × 8 bins = 512 张图
            )
        _safe(f"diagnose residual ({s})", _residual)

        def _drift():
            from tin_coating.diagnostics.drift import ks_test_drift
            residuals = lab - m.predict_smooth(sub[feat], online)[0].values
            ks, p, md, verdict = ks_test_drift(residuals, ratio=0.8)
            (model_out_dir / "drift").mkdir(parents=True, exist_ok=True)
            (model_out_dir / "drift" / "drift_summary.txt").write_text(
                f"group={group}\nsurface={s}\nn={len(residuals)}\n"
                f"ks_stat={ks}\nks_p={p}\nmean_diff={md}\nverdict={verdict}\n",
                encoding="utf-8",
            )
            print(f"    {s}: KS={ks:.4f} p={p:.4g} mean_diff={md:.4f} verdict={verdict}")
        _safe(f"diagnose drift ({s})", _drift)

        def _interp():
            from tin_coating.eda.interpreter import ModelInterpreter
            interp = ModelInterpreter(
                model=m, X=sub[feat], feature_names=feat,
                save_dir=str(model_out_dir / "interpretation"),
                max_samples_for_shap=500,
            )
            interp.full_analysis(
                y=sub[f'{s}_Delta'].values.astype(float),
                run_permutation=True, run_shap=True, run_pdp=True,
                pdp_features=feat[:5],
            )
        _safe(f"diagnose interpret ({s})", _interp)

    _mark_latest(diag_root, "diagnostics")
    return f"完成 {len(results)} 个子模块（失败不中断主流程）"


@_stage("EDA + 诊断套件")
def stage_analytics(group: str) -> str:
    return _eda_and_diagnostics(group)


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
    global _VERSION
    cfg = _cfg()
    p = argparse.ArgumentParser(description="tin-coating 全流程一键运行")
    p.add_argument("--group", default=cfg.app.default_group)
    p.add_argument("--quick", action="store_true", help="跳过训练/CV/分析套件，只跑自检+质量+健康+少量推理")
    p.add_argument("--skip-train", action="store_true", help="复用已有模型，不重新训练")
    p.add_argument("--skip-cv", action="store_true", help="跳过 Walk-Forward CV")
    p.add_argument("--skip-analytics", action="store_true", help="跳过 EDA + 残差/漂移/解释性分析（省时间）")
    p.add_argument("--with-retrain", action="store_true", help="额外生成重训候选（不部署）")
    p.add_argument("--rows", type=int, default=200, help="推理样本行数（默认 200）")
    p.add_argument("--output", default=None, help="推理输出 Excel 路径")
    p.add_argument("--version", default=None, help="手动指定 version 目录（默认当前时间戳 YYYYMMDD_HHMMSS）")
    args = p.parse_args()

    _VERSION = args.version or time.strftime("%Y%m%d_%H%M%S")
    stamp = _VERSION
    output = Path(args.output) if args.output else (
        cfg.root / "result" / "predictions" / f"full_flow_{stamp}.xlsx")

    t0 = time.time()
    print(f"[flow] version={_VERSION}  group={args.group}  quick={args.quick}  "
          f"train={'no' if (args.quick or args.skip_train) else 'yes'}  "
          f"analytics={'no' if (args.quick or args.skip_analytics) else 'yes'}")

    if not stage_selfcheck(args.group):
        return 2                                   # 自检失败直接停

    if not (args.quick or args.skip_train):
        if not stage_train(args.group):
            return 2                               # 训练失败直接停

    if not (args.quick or args.skip_cv):
        stage_cv(args.group)

    if not (args.quick or args.skip_analytics):
        stage_analytics(args.group)                # EDA + 残差/漂移/解释性（非致命）

    stage_quality(args.group)
    stage_health(args.group)

    exit_code = 0

    if not stage_predict(args.group, args.rows, output):
        exit_code = 2                              # 推理失败继续往下汇总

    if args.with_retrain:
        stage_retrain(args.group)

    # ---- 汇总 ----
    total_elapsed = round(time.time() - t0, 2)
    print(f"\n{'=' * 70}\n[flow] 汇总（version={_VERSION}，总耗时 {total_elapsed}s）\n{'=' * 70}")
    for s in _STAGES:
        print(f"  {'✅' if s['ok'] else '❌'} {s['name']:<12} {s['elapsed']:>7.2f}s  {s['detail']}")

    failed = [s for s in _STAGES if not s["ok"]]
    if failed and exit_code != 2:
        print(f"\n[flow] 有 {len(failed)} 个阶段失败: "
              f"{', '.join(s['name'] for s in failed)}")
        exit_code = 2

    if any("(fallback)" in s["detail"] for s in _STAGES):
        print("\n⚠️  当前环境与 registry 归档模型不兼容（版本不一致），推理已用 live 目录产物兜底。")
        print("   建议统一环境：在 .venv 执行 pip install -r requirements.txt"
              "（锁 sklearn<1.5 / pandas<2.2），或在同一环境重训后走 gate 审批。")

    # ---- 写 _run_meta.json（MLflow params.txt 等价物）----
    meta_root = ROOT / "result" / "_flows"
    meta_root.mkdir(parents=True, exist_ok=True)
    meta = {
        "version": _VERSION,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_elapsed_s": total_elapsed,
        "exit_code": exit_code,
        "group": args.group,
        "args": {k: v for k, v in vars(args).items() if k != "output"},
        "stage_results": [{"name": s["name"], "ok": s["ok"],
                           "elapsed_s": s["elapsed"], "detail": s["detail"]}
                          for s in _STAGES],
        "artifacts": {
            "models": f"result/trained_models/by_group/{args.group}/",
            "predictions": str(output) if output else None,
            "eda": str(_vroot("eda", args.group)) if not (args.quick or args.skip_analytics) else None,
            "diagnostics": str(_vroot("diagnostics", args.group)) if not (args.quick or args.skip_analytics) else None,
            "cv": str(_vroot("cv", args.group)) if not (args.quick or args.skip_cv) else None,
        },
    }
    (meta_root / f"{_VERSION}.json").write_text(
        _json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (meta_root / "_LATEST.json").write_text(
        _json.dumps({"version": _VERSION, "exit_code": exit_code,
                     "start_time": meta["start_time"],
                     "artifacts": meta["artifacts"]},
                    indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[flow] 全流程完成（exit_code={exit_code}）。"
          f"部署模型请走: python monitoring/gate.py --list → --approve")
    print(f"[flow] run_meta → result/_flows/{_VERSION}.json")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
