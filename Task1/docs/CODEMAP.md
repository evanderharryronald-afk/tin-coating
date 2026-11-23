# Tin Coating — Code Map

一个给新同事 / 后续接手者用的地图。**先看第一章和第二章**就知道哪些能动、哪些别动、动了会有什么影响。

---

## 1. 三类代码的边界

| 类别 | 目录 / 文件 | 定义 | 改动代价 |
|---|---|---|---|
| **🚨 受保护（PROTECTED）** | `coating_model_by_group.py`、`tin_coating/modeling/residual_model.py` 里的算法逻辑、`monitoring/registry.py` 的版本历史 | 直接影响线上模型产物或回滚正确性 | 改之前必须跑完整 `run_full_flow.py` + `pytest tests/` + 手动 `gate --approve` 把新产物登记；改了 `residual_model` 的 predict_smooth 签名等于**全量重新训练** |
| **🛠️ 可演进（EVOLVABLE）** | `tin_coating/eda/`、`tin_coating/diagnostics/`、`tin_coating/data/`、`cli/main.py` 的子命令、`configs/app.yaml` | 分析脚本、诊断工具、新 CLI 入口、默认参数 | 单独改、单独测；不影响生产闭环 |
| **🗃️ 遗留（LEGACY）** | 根目录 `legacy_*.py`、`archived/` 里所有、`largest_group_repro/`、`spec_merge_analyzer/`、`surface_independent_modeling/`、`ablation/` | 旧原型代码，一行未动 | 只读，可删但别让 CI 崩 |

**核心纪律**：新代码只 import 前两类，永远不 import 第三类。新功能往 **EVOLVABLE** 里加，加完再接进 `cli/main.py` 做入口。

---

## 2. 关键约束（动了会出事的）

| # | 约束 | 在哪 | 动了会怎样 |
|---|---|---|---|
| A | `coating_model_by_group.py` 零副作用、一行未改 | 根目录 | 承诺保持；新代码全部单向依赖它。你如果改了它，所有之前的 joblib artifact 反序列化都会失败（旧 joblib 是按 `__main__` 注册的类 pickle 的，我们靠模块末尾注册兜底）|
| B | Lab 真值列是**中文**：`上表面镀层重量A(XA1_0)` / `下表面镀层重量A(XA1_0)` | Excel 表头 | 写英文列名会 KeyError；CV/重训/推理脚本里所有 `lab_col` 都得保持 `{"Top":"上","Bot":"下"}` 映射 |
| C | 线上规格组默认 `Top2.799_Bot2.799`；`Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_TOP_Min.round(3) == 2.799` | prod_config JSON / 过滤代码 | 过滤条件改了会污染训练集；新规格组要先在 prod_config 里注册，再在 CLI 里加 |
| D | 8 个特征、α=0.6、random_state=42 | `params.get_feature_cols` / `group_params_2_799_only.json` | 改特征集合等于重训；α 改了要同步改模型的 `predict_smooth` 调用 |
| E | `registry.json` 的 active 标记 = 线上生效模型；**训练 ≠ 部署** | `monitoring/registry.py` | `train` 只写 joblib 产物目录，不碰 active 标记；必须 `gate --approve` 才上线 |

---

## 3. 目录结构（分类标注）

```
Task1/
│
├─ PROTECTED — 别动
│  ├─ coating_model_by_group.py          原生产训练入口；唯一承诺"零副作用一行未改"
│  ├─ configs/group_params_2_799_only.json   训练超参来源（Top/Bot damping/pos_boost/lr/depth/max_iter）
│  └─ monitoring/
│     ├─ registry.py                     版本注册表（原子写+跨进程锁+精确回滚指针）
│     ├─ gate.py                         审批/回滚 CLI
│     ├─ auto_retrain_candidate.py       自动重训候选生成
│     ├─ check_monthly_health.py         月度健康检查（z-score 方向修过，阈值 2.0+regression_min_drop=0.05）
│     ├─ check_data_quality.py           数据质量门
│     └─ registry.json                   ← 持久状态，要备份
│
├─ tin_coating/ — 新子包（全部 NEW，生产闭环主干）
│  ├─ config.py                          统一配置加载器（点号访问 + 插值 + env 覆盖）
│  ├─ modeling/                          PROTECTED 算法核心
│  │  ├─ residual_model.py               ResidualCorrectionModel；类级 __module__ + sys.modules["__main__"] 注册（为旧 joblib 反序列化）
│  │  ├─ group_train.py                  按规格组训练
│  │  ├─ walk_forward_cv.py              Walk-Forward CV
│  │  ├─ params.py                       get_feature_cols / get_params_for_group / build_setpoint_group_key
│  │  ├─ persistence.py                 save/load + metadata
│  │  └─ optuna_tuning.py
│  ├─ data/                              🛠️ EVOLVABLE
│  │  ├─ feature_engineering.py          构造衍生列 + Steel_Grade 频率编码（新钢种 fallback 到均值）
│  │  ├─ clean.py
│  │  ├─ format_converter.py
│  │  └─ merge.py
│  ├─ inference/pipeline.py              批量推理
│  ├─ eda/                               🛠️ EVOLVABLE（EDA 工具）
│  │  ├─ analyzer.py                     SurfaceCorrelationAnalyzer
│  │  ├─ correlation.py
│  │  ├─ grouping.py                     GroupEDADiagnoser
│  │  └─ interpreter.py                  ModelInterpreter（Permutation + SHAP + PDP，SHAP 可选）
│  ├─ diagnostics/                       🛠️ EVOLVABLE（诊断工具）
│  │  ├─ residual.py                     run_full_residual_diagnostics — 已修：空特征列表不崩、自适应 cv 折数
│  │  └─ drift.py                        ks_test_drift 纯函数 + analyze_one 完整分析
│  ├─ ops/                               🛠️ EVOLVABLE（7×24 运维底座）
│  │  ├─ log.py                          结构化 JSON 日志 + 轮转
│  │  ├─ metrics.py                      Prometheus 文本 counter/gauge/histogram
│  │  ├─ lock.py                         跨平台 FileLock（msvcrt/fcntl）
│  │  ├─ notify.py                       Notifier 文件 + webhook 指数退避
│  │  ├─ selfcheck.py                    run_selfcheck / assert_ready（serve 启动前 fail-fast）
│  │  └─ scheduler.py                     run_job：quality/health/retrain/drift/report 5 个 job
│  └─ service/                           🛠️ EVOLVABLE（7×24 在线服务）
│     ├─ contract.py                     请求校验
│     ├─ models.py                       ModelStore（registry 驱动 + deployed_path→path→live 候选链 + 热加载 + 坏产物 fallback）
│     └─ server.py                       ThreadingHTTPServer（healthz/readyz/metrics/version/predict/reload + API Key + SIGTERM）
│
├─ cli/main.py                            🛠️ EVOLVABLE（统一入口，12 个子命令）
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
├─ LEGACY — 只读，一行未动
│  ├─ 根目录：legacy_feature_engineering.py / legacy_model_inference.py / legacy_predict_by_group.py / legacy_residual_diagnostics.py / legacy_residual_drift_diagnosis.py / legacy_format_converter.py / legacy_run_merge.py / legacy_run_tuning_optuna.py
│  ├─ 根目录：eda_analyzer.py / correlation_analyzer.py / model_interpreter.py / group_eda_manager.py / data_cleaner.py / model_persistence.py
│  ├─ archived/：analyse_data_final.py / analyse_direct.py / analyse_low_reading_samples.py / check_specs.py / merge_analysis_pipeline.py / new_data_pipeline.py / param_tuner.py / run_surface_independent_modeling.py / update_group_train_config.py / core/analysis.py
│  ├─ largest_group_repro/                                              2.799 组 11 个复现实验
│  ├─ spec_merge_analyzer/                                              规格合并分析三段
│  ├─ surface_independent_modeling/                                     面无关建模独立子系统
│  ├─ ablation/run_ablation.py                                          消融实验
│  └─ report/                                                            画图脚本
│
├─ 构建 / 测试 / CI / 一键
│  ├─ run_full_flow.py                    全流程编排：selfcheck→train→cv→quality→health→predict
│  ├─ __main_smoke_test.py
│  ├─ tests/
│  │  ├─ conftest.py, pytest.ini
│  │  ├─ test_smoke.py                   9 项：导入、入口零副作用、Walk-Forward CV RMSE、CLI cv/monitor、目录布局、无 leak import、pipeline 不重复模型
│  │  ├─ test_ops.py                     6 项：结构化日志、Prometheus 指标渲染、FileLock 互斥、Notifier 文件通道、registry 6 线程并发、selfcheck 完整
│  │  └─ test_service.py                 15 项：契约 3 反例 + live_server fixture + 8 个 HTTP 用例 + 并发 16 + 热加载 + fallback + 全不可用时 ready=False + API Key
│  ├─ Dockerfile / docker-compose.yml / .dockerignore
│  ├─ .github/workflows/ci.yml
│  ├─ pyproject.toml / requirements.txt
│  ├─ configs/app.yaml
│  └─ README.md / docs/CODEMAP.md
```

---

## 4. 生产闭环端到端依赖链

```
Excel ──▶ tin_coating.modeling.group_train ──▶ result/trained_models/.../{Top,Bot}_model.joblib
                       │  ←  不碰 registry
                       ▼
                monitoring/registry.json
                       │
       ┌──────────────┼──────────────────┐
       ▼              ▼                  ▼
 check_monthly   check_data_quality  auto_retrain_candidate
    health.py        .py                  .py
                       │                  │
                       ▼                  ▼
                gate.py --approve / rollback  (唯一能动 active 标记的地方)
                       │
                       ▼
            registry.json active 切换
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
  tin_coating.service.server    cli/main.py predict
  (7x24 HTTP 服务, 热加载)      (离线批量推理)
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
# 轻量：pytest（2 分钟内）
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
