# Tin Coating — Code Map

一个给新同事 / 后续接手者用的地图。**先看第一章和第二章**就知道哪些能动、哪些别动、动了会有什么影响。

这是 2026-11 重构后的**单一来源（single-source）**架构：所有生产逻辑都在 `tin_coating/` 子包下；根目录不再有 `coating_model_by_group.py`、`legacy_*.py`、`archived/`、`surface_independent_modeling/` 等旧原型。任何新代码都只 import `tin_coating.*` / `monitoring.*` / `cli.main`。

---

## 1. 单一来源（Single Source）边界

| 类别 | 目录 / 文件 | 定义 |
|---|---|---|
| **🚨 受保护 (PROTECTED)** | `tin_coating/modeling/residual_model.py`（类级 `__module__` 强制）、`monitoring/registry.py` 的版本历史、`config.yaml`（merge.py 自引用）、`rename_patch.yaml`（merge.py 自引用） | 直接影响线上模型产物、回滚正确性、或数据处理脚本的硬依赖；改动前必须跑完整 `pytest tests/` + `python run_full_flow.py` |
| **🛠️ 可演进 (EVOLVABLE)** | `tin_coating/eda/`、`tin_coating/diagnostics/`、`tin_coating/ops/`、`cli/main.py` 的子命令、`configs/app.yaml`、`monitoring/gate.py`、`monitoring/auto_retrain_candidate.py` | 分析脚本、诊断工具、CLI 入口、运维调度、监控闭环；单独改、单独测，不影响训练核心 |
| **🗃️ 运行时产物（可删可改，别 git）** | `result/`、`monitoring/logs/`、`monitoring/reports/`、`monitoring/locks/`、`monitoring/registry.json` | 训练产物、日志、报告、锁、版本历史；gitignore 已覆盖；其中只有 `registry.json` 需要**备份** |

**核心纪律**：新代码只能 import 前两类。`__main__` 兼容注册（`residual_model.py` 末尾）是唯一一处往 `sys.modules["__main__"]` 挂类的 hack，为了兼容 **2026-11 之前**用 `__main__.__module__` 训练的老 joblib。以后新训练的 joblib 类 `__module__` 强制为 `tin_coating.modeling.residual_model`，不再有这个问题。

---

## 2. 关键约束（动了会出事的）

| # | 约束 | 在哪 | 动了会怎样 |
|---|---|---|---|
| A | `ResidualCorrectionModel.__module__ = "tin_coating.modeling.residual_model"`（类级强制）| `tin_coating/modeling/residual_model.py` L43 | 改了之后用新 joblib 反序列化会 KeyError；动之前先确认需要全量重训 |
| B | Lab 真值列是**中文**：`上表面镀层重量A(XA1_0)` / `下表面镀层重量A(XA1_0)` | Excel 表头 | 写英文列名会 KeyError；CV/重训/推理脚本里所有 `lab_col` 都得保持 `{"Top":"上","Bot":"下"}` 映射 |
| C | 线上规格组默认 `Top2.799_Bot2.799`；`Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_TOP_Min.round(3) == 2.799` | `app.yaml` / 过滤代码 | 过滤条件改了会污染训练集；新规格组要先在 configs/model 里注册，再在 CLI 里加 |
| D | 8 个特征、α=0.6、random_state=42 | `params.get_feature_cols` / `group_params_2_799_only.json` | 改特征集合等于重训；α 改了要同步改模型的 `predict_smooth` 调用 |
| E | `registry.json` 的 active 标记 = 线上生效模型；**训练 ≠ 部署** | `monitoring/registry.py` | `train` 只写 joblib 产物目录，不碰 active 标记；必须 `gate.py --approve` 才上线 |
| F | 数据清洗的主副表合并硬依赖根目录 `config.yaml` + `rename_patch.yaml` + `master_schema_cache.json`（merge.py 自引用）| `tin_coating/data/merge.py` | 不能删这三个文件；改了会导致数据管道断 |

---

## 3. 目录结构

```
Task1/
│
├─ tin_coating/ — 所有生产逻辑（单一来源）
│  ├── config.py                    统一配置加载器（点号访问 + 插值 + env 覆盖）
│  │
│  ├── modeling/ — PROTECTED 算法核心
│  │  ├── residual_model.py          ResidualCorrectionModel；类级 __module__ + sys.modules["__main__"] 兼容注册（为旧 joblib 反序列化）
│  │  ├── group_train.py             按规格组训练 + 可视化 + SHAP 解释；CLI train 入口调用 run_surface_pipeline
│  │  ├── walk_forward_cv.py         Walk-Forward CV（中文 Lab 列）
│  │  ├── params.py                  get_feature_cols / get_params_for_group / build_setpoint_group_key / DEFAULT_PARAMS
│  │  ├── persistence.py             ModelSaver / ModelLoader / ModelMetadata（统一落盘）
│  │  ├── optuna_tuning.py           调参脚本（可选，需手动跑）
│  │  └── __init__.py                公共 API 汇总（fit_and_evaluate_surface / run_surface_pipeline / ResidualCorrectionModel / …）
│  │
│  ├── data/ — 🛠️ EVOLVABLE
│  │  ├── clean.py                   数据清洗
│  │  ├── feature_engineering.py     构造衍生列 + Steel_Grade 频率编码（新钢种 fallback 到均值）
│  │  ├── format_converter.py        格式转换
│  │  └── merge.py                   主副表合并（硬依赖根 config.yaml + rename_patch.yaml）
│  │
│  ├── inference/pipeline.py         批量推理（PredictionPipeline）
│  │
│  ├── eda/ — 🛠️ EVOLVABLE（EDA 工具）
│  │  ├── analyzer.py                SurfaceCorrelationAnalyzer
│  │  ├── correlation.py             相关性 + MI + DCOR
│  │  ├── grouping.py                GroupEDADiagnoser
│  │  └── interpreter.py             ModelInterpreter（Permutation + SHAP + PDP，SHAP 可选）
│  │
│  ├── diagnostics/ — 🛠️ EVOLVABLE（诊断工具）
│  │  ├── residual.py                run_full_residual_diagnostics — 已修：空特征列表不崩、自适应 cv 折数
│  │  └── drift.py                   ks_test_drift 纯函数 + analyze_one 完整分析
│  │
│  ├── ops/ — 🛠️ EVOLVABLE（7×24 运维底座）
│  │  ├── log.py                     结构化 JSON 日志 + 轮转
│  │  ├── metrics.py                 Prometheus 文本 counter/gauge/histogram
│  │  ├── lock.py                    跨平台 FileLock（msvcrt/fcntl）
│  │  ├── notify.py                  Notifier 文件 + webhook 指数退避
│  │  ├── selfcheck.py               run_selfcheck / assert_ready（serve 启动前 fail-fast）
│  │  └── scheduler.py               run_job：quality/health/retrain/drift/report 5 个 job
│  │
│  └── service/ — 🛠️ EVOLVABLE（7×24 在线服务）
│     ├── contract.py                 请求校验
│     ├── models.py                   ModelStore（registry 驱动 + deployed_path→path→live 候选链 + 热加载 + 坏产物 fallback）
│     └── server.py                   ThreadingHTTPServer（healthz/readyz/metrics/version/predict/reload + API Key + SIGTERM）
│
├─ monitoring/ — 监控闭环脚本（🛠️ EVOLVABLE，单向依赖 tin_coating.*）
│  ├─ registry.py                     版本注册表（原子写+跨进程锁+精确回滚指针）
│  ├─ gate.py                         审批/回滚 CLI
│  ├─ auto_retrain_candidate.py       自动重训候选生成（产 .pending，不部署）
│  ├─ check_monthly_health.py         月度健康检查（z-score 方向修过，阈值 2.0+regression_min_drop=0.05）
│  ├─ check_data_quality.py           数据质量门
│  └─ registry.json                   ← 持久状态，要备份
│
├─ cli/main.py                         🛠️ EVOLVABLE（统一入口，12 个子命令）
│
├─ 运行时产物（持久状态 / 可删临时）
│  ├─ result/trained_models/by_group/<group>/{Top,Bot}_model.joblib
│  ├─ result/trained_models/by_group/<group>/_archive/{surface}_{version}.joblib    审批归档
│  ├─ result/trained_models/by_group/<group>/.pending/*.joblib                    重训候选（待 gate）
│  ├─ monitoring/registry.json                                                      持久状态，要备份
│  ├─ monitoring/reports/*.json / *.txt / *.csv / .pending.joblib                  keep_logs_days 自动清理
│  ├─ monitoring/locks/*.lock                                                       文件锁
│  ├─ monitoring/logs/*.jsonl                                                         日志（10MB×5 轮转）+ alerts.jsonl
│  └─ result/predictions/*.xlsx                                                      批量推理输出
│
├─ 构建 / 测试 / CI / 一键
│  ├─ run_full_flow.py                    全流程编排：selfcheck→train→cv→quality→health→predict
│  ├─ __main_smoke_test.py
│  ├─ tests/
│  │  ├─ conftest.py, pytest.ini
│  │  ├─ test_smoke.py                   9 项：导入、建模子包完整导出、Walk-Forward CV RMSE、CLI cv/monitor、目录布局、import 边界、inference 去重
│  │  ├─ test_ops.py                     6 项：结构化日志、Prometheus 指标渲染、FileLock 互斥、Notifier 文件通道、registry 6 线程并发、selfcheck 完整
│  │  └─ test_service.py                 15 项：契约 3 反例 + live_server fixture + 8 个 HTTP 用例 + 并发 16 + 热加载 + fallback + 全不可用时 ready=False + API Key
│  ├─ Dockerfile / docker-compose.yml / .dockerignore
│  ├─ .github/workflows/ci.yml
│  ├─ pyproject.toml / requirements.txt
│  ├─ configs/app.yaml / configs/default.yaml / configs/model/*.json / configs/optuna/*.json
│  └─ README.md / docs/CODEMAP.md
│
└─ 数据处理辅助脚本（git 管、不 import tin_coating）
   ├─ config.yaml                  tin_coating/data/merge.py 主副表合并主配置（硬依赖）
   ├─ rename_patch.yaml            merge.py 列名更正补丁
   └─ master_schema_cache.json     merge.py 列名缓存
```

---

## 4. 生产闭环端到端依赖链

```
Excel ──▶ tin_coating.modeling.group_train.run_surface_pipeline
         │                              (ModelSaver.save_model)
         ▼
result/trained_models/.../{Top,Bot}_model.joblib
         │  ←  不碰 registry
         ▼
  monitoring/registry.json
         │
 ┌────────┼──────────────────────┐
 ▼        ▼                      ▼
check_  check_data_         auto_retrain_
monthly health.py           candidate.py
   health.py                  │
   │                          │
   └─────────┬────────────────┘
             ▼
      gate.py --approve / rollback  (唯一能动 active 标记的地方)
             │
             ▼
registry.json active 切换
             │
   ┌─────────┴──────────────┐
   ▼                        ▼
tin_coating.service.server  cli/main.py predict
(7x24 HTTP 服务, 热加载)    (离线批量推理)
```

**关键纪律**：`train` 不碰 registry；只有 `gate.py --approve` 才能动 active 标记。服务每请求按 registry 指纹热加载，**gate approve 后不用重启**。

---

## 5. 新功能加到哪

| 你想加 | 放到哪 | 要不要加 CLI 入口 |
|---|---|---|
| 新的分析方法（比如 KS-test 之外的漂移探针） | `tin_coating/diagnostics/` 新建 `.py` | 加：`cli/main.py diagnose drift --method xxx` |
| 新的数据预处理步骤 | `tin_coating/data/clean.py` 或新建 `.py` | 加：`cli/main.py` 新子命令或集成到 `feature_engineering` |
| 新的模型评估指标 | `tin_coating/modeling/walk_forward_cv.py` | 不用单独 CLI，它被 CV 和 retrain 都调用 |
| 新的定时任务（比如每周一次的 PCA） | `tin_coating/ops/scheduler.py` 加新 `elif job == "xxx":`，`JOBS` 元组里加 `"xxx"`，`build_parser` 的 `ops run --job` choices 里加 | 自动有了，因为 parser choices 跟 JOBS 同步维护 |
| 新的在线服务端点 | `tin_coating/service/server.py` 的 `Handler` 里加新方法 | 自动暴露到 HTTP |
| 新的配置项 | `configs/app.yaml` 加字段 + `tin_coating/config.py` 的 `_FALLBACK_DATA` 补默认值 | 自动 `cfg.new_section.new_key` 可访问；env override 自动生效 |
| 新的监控脚本 | `monitoring/` 里新建 `.py` | CLI `monitor` 子命令加选项；scheduler `run_job` 加分支 |

---

## 6. 跑全套验证的最短路径

```powershell
# 轻量：pytest（~3 分钟）
python -m pytest tests/ -v

# 真跑全链（~90 秒，含训练）
python run_full_flow.py

# 只跑生产链：质量门 + 健康 + 漂移
python cli/main.py ops run --job quality
python cli/main.py ops run --job health
python cli/main.py ops run --job drift
python cli/main.py ops run --job report

# 单跑数据分析
python cli/main.py eda dist --group Top2.799_Bot2.799
python cli/main.py diagnose drift --group Top2.799_Bot2.799
python cli/main.py diagnose residual --group Top2.799_Bot2.799

# 起服务（10 秒）
python cli/main.py serve --host 0.0.0.0 --port 8080
curl http://localhost:8080/readyz
```
