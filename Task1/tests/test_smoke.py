"""pytest smoke tests — 与 __main_smoke_test.py 内容等价，但每条是独立 test case。

跑法:
  cd Task1
  pytest tests/test_smoke.py -v
  python __main_smoke_test.py   # wrapper 也会跑 pytest.main
"""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
import pandas as pd

TASK1_ROOT = Path(__file__).resolve().parent.parent
if str(TASK1_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK1_ROOT))


# ---------------------------------------------------------------------------
# 1) tin_coating 子包全部模块可 import
# ---------------------------------------------------------------------------
MODULES = [
    "tin_coating",
    "tin_coating.modeling.residual_model",
    "tin_coating.modeling.params",
    "tin_coating.modeling.walk_forward_cv",
    "tin_coating.modeling.group_train",
    "tin_coating.modeling.persistence",
    "tin_coating.modeling.optuna_tuning",
    "tin_coating.data.clean",
    "tin_coating.data.merge",
    "tin_coating.data.feature_engineering",
    "tin_coating.data.format_converter",
    "tin_coating.inference.pipeline",
    "tin_coating.eda.analyzer",
    "tin_coating.eda.correlation",
    "tin_coating.eda.grouping",
    "tin_coating.eda.interpreter",
    "tin_coating.diagnostics.residual",
    "tin_coating.diagnostics.drift",
]


def test_tin_coating_all_modules_importable():
    for m in MODULES:
        importlib.import_module(m)


# ---------------------------------------------------------------------------
# 2) 原 coating_model_by_group.py 零副作用 — 还能独立 import / 导出完整
# ---------------------------------------------------------------------------
def test_original_entrypoint_zero_side_effects():
    import coating_model_by_group as cm
    for name in ["fit_and_evaluate_surface", "run_surface_pipeline",
                 "ResidualCorrectionModel", "load_pipeline_config",
                 "MIN_GROUP_SAMPLES"]:
        assert name in dir(cm), f"{name} 不在 coating_model_by_group 导出里"


# ---------------------------------------------------------------------------
# 3) tin_coating 顶层公共 API 暴露正常
# ---------------------------------------------------------------------------
def test_top_level_public_api():
    from tin_coating import (
        ResidualCorrectionModel, run_cv_for_surface, load_pipeline_config,
        PredictionPipeline, get_feature_cols, SteelDataCleaner,
        build_setpoint_group_key, DEFAULT_PARAMS, MIN_GROUP_SAMPLES,
    )
    assert ResidualCorrectionModel is not None
    assert callable(run_cv_for_surface)
    assert len(get_feature_cols("Top")) == 8
    assert len(get_feature_cols("Bot")) == 8


# ---------------------------------------------------------------------------
# 4) Walk-Forward CV Top RMSE 在合理范围（ablation 已锁 0.045~0.060）
# ---------------------------------------------------------------------------
def test_walk_forward_cv_top_rmse_range():
    df = pd.read_excel(TASK1_ROOT / "result/data/feature_engineered_data/featured_data.xlsx")
    from tin_coating.modeling import (
        load_pipeline_config, get_params_for_group,
        build_setpoint_group_key, run_cv_for_surface,
    )
    df = build_setpoint_group_key(df)
    group_df = df[df['Setpoint_Group_Label'] == 'Top2.799_Bot2.799'].copy()
    cfg = load_pipeline_config(str(TASK1_ROOT / "group_params_2_799_only.json"))
    params = get_params_for_group(cfg, "Top2.799_Bot2.799", "Top")
    cv = run_cv_for_surface(group_df, "Top", params)
    assert cv["cv_rmse_mean"] is not None
    assert 0.045 < cv["cv_rmse_mean"] < 0.060, f"RMSE 越界: {cv['cv_rmse_mean']}"


# ---------------------------------------------------------------------------
# 5) CLI cv 子命令
# ---------------------------------------------------------------------------
def test_cli_cv_subcommand():
    r = subprocess.run(
        [sys.executable, str(TASK1_ROOT / "cli/main.py"), "cv",
         "--group", "Top2.799_Bot2.799", "--surface", "Top"],
        capture_output=True, text=True, timeout=180,
    )
    assert r.returncode == 0, f"exit={r.returncode}\nSTDERR={r.stderr[:300]}"
    assert "CV RMSE=" in r.stdout


# ---------------------------------------------------------------------------
# 6) CLI monitor quality 子命令
# ---------------------------------------------------------------------------
def test_cli_monitor_quality_subcommand():
    r = subprocess.run(
        [sys.executable, str(TASK1_ROOT / "cli/main.py"), "monitor", "quality"],
        capture_output=True, text=True, timeout=120,
    )
    # quality 子命令返回 0=PASS, 2=FAIL（都算"正常跑完"）
    assert r.returncode in (0, 2), f"exit={r.returncode}\nSTDERR={r.stderr[:300]}"


# ---------------------------------------------------------------------------
# 7) 目录结构 — 关键文件/目录齐全
# ---------------------------------------------------------------------------
def test_directory_layout_complete():
    p = TASK1_ROOT
    for rel in [
        "tin_coating/modeling/group_train.py",
        "tin_coating/modeling/residual_model.py",
        "tin_coating/modeling/walk_forward_cv.py",
        "tin_coating/data/clean.py",
        "tin_coating/eda/interpreter.py",
        "monitoring/check_data_quality.py",
        "monitoring/gate.py",
        "cli/main.py",
        "requirements.txt",
        "pyproject.toml",
        "README.md",
    ]:
        assert (p / rel).exists(), f"缺失: {rel}"
    assert (p / "archived").is_dir(), "缺失 archived/"
    assert (p / "configs" / "model").is_dir(), "缺失 configs/model/"


# ---------------------------------------------------------------------------
# 8) 顶层 .py 单向依赖闭环 — 无 tin_coating/cli/monitoring import
# ---------------------------------------------------------------------------
def test_top_level_no_leak_imports():
    root = TASK1_ROOT
    for p in root.glob("*.py"):
        # 这两个是总编排脚本，允许 import 子包；其余顶层脚本必须保持单向依赖
        if p.name in ("__main_smoke_test.py", "run_full_flow.py"):
            continue
        txt = p.read_text(encoding="utf-8")
        for bad in ["import tin_coating", "from tin_coating",
                    "import cli.main", "from cli.main",
                    "import monitoring", "from monitoring"]:
            assert bad not in txt, f"{p.name} 里有残留 {bad}"


# ---------------------------------------------------------------------------
# 9) inference/pipeline.py 不再重复定义 ResidualCorrectionModel（P0 去重后）
# ---------------------------------------------------------------------------
def test_inference_pipeline_no_duplicate_model():
    src = (TASK1_ROOT / "tin_coating/inference/pipeline.py").read_text(encoding="utf-8")
    assert "class ResidualCorrectionModel:" not in src, (
        "inference/pipeline.py 里还保留着 ResidualCorrectionModel 类定义，"
        "应该改成 from tin_coating.modeling.residual_model import ResidualCorrectionModel"
    )
