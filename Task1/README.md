# tin-coating

连续热镀锌线镀层重量（Top / Bot 表面）偏差校正模型。基于 HistGradientBoostingResidualCorrection 的残差校正 + EWMA 平滑，按 `Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_TOP_Min` / `..._BOT_Min` 分组训练，Walk-Forward CV 验证。

当前生产配置（Top2.799_Bot2.799 规格组）：
- Top HGB: damping=0.3046, pos_boost=1.6837, lr=0.0144, depth=5, max_iter=250
- Bot HGB: damping=0.2848, pos_boost=1.2582, lr=0.0117, depth=4, max_iter=300
- 统一 α=0.6（EWMA smoothing），random_state=42
- Top/Bot 各 8 特征，Walk-Forward CV Top RMSE ≈ 0.0506 ± 0.0060

---

## 目录结构

```
Task1/
├── tin_coating/               # ← 可 pip install 的子包
│   ├── data/                  #   清洗 / 特征工程 / 格式转换 / 合并
│   ├── modeling/              #   核心模型 / CV / 持久化 / 调参
│   ├── inference/             #   PredictionPipeline 批量推理
│   ├── eda/                   #   探索性分析
│   └── diagnostics/           #   残差 / 漂移诊断
├── cli/main.py                # ← 统一 CLI 入口（argparse，零外部依赖）
├── run_full_flow.py           # ← 一键全流程（自检→训练→CV→质量→健康→推理）
├── monitoring/                #   线上监控闭环（quality → health → retrain → gate approve/rollback）
├── configs/                   #   JSON/YAML 配置镜像（生产配置仍以根目录 group_params_2_799_only.json 为准）
├── archived/                  #   历史脚本归档
├── coating_model_by_group.py  #   原生产训练入口（零副作用，保持独立可跑）
├── group_params_2_799_only.json  #   生产超参（不动）
├── pyproject.toml             #   pip install -e . 配置
└── requirements.txt
```

> 根目录下的 `predict_by_group.py` / `model_inference.py` / `feature_engineering.py` 等旧脚本保留作参考，日常使用请走 `tin_coating/*` 子包或 `cli/main.py`。

---

## 安装

```bash
cd Task1
pip install -e .
```

依赖：numpy / pandas / scikit-learn / matplotlib / seaborn / openpyxl / optuna / joblib / pyyaml。Python 3.9–3.11。

---

## 快速开始

### 一键全流程（推荐先跑这个）

```bash
cd Task1
python run_full_flow.py                # 自检 → 训练 → CV → 质量门 → 健康检查 → 批量推理
python run_full_flow.py --quick        # 跳过训练/CV，约 30 秒跑完检查 + 200 行推理
python run_full_flow.py --skip-train   # 复用现有模型
python run_full_flow.py --with-retrain # 额外生成重训候选（只生成 pending，不部署）
```

每个阶段打印耗时与结果，最后输出汇总表；推理结果写到 `result/predictions/full_flow_<时间戳>.xlsx`。
退出码：`0` 正常 / `1` 有非致命告警 / `2` 有致命失败。

> 训练只更新 `result/trained_models/` 下的产物，不会自动上线；线上服务始终按 registry 的 active 版本推理，
> 要部署必须走 `monitoring/gate.py --approve`（见下文）。

### 常用命令

```bash
# Walk-Forward CV（验证当前生产模型在规格组上的稳健性）
python cli/main.py cv --group Top2.799_Bot2.799 --surface Top

# 线上数据质量检查
python cli/main.py monitor quality
python cli/main.py monitor health

# 批量推理（对一批新料做 Top / Bot 校正）
python cli/main.py predict --input new_coils.xlsx --output corrected.xlsx

# 7×24 在线服务（换模型自动热加载，无需重启）
python cli/main.py serve --host 0.0.0.0 --port 8080

# 完整 CLI 帮助
python cli/main.py --help
python cli/main.py ops --help
```

### 直接用 tin_coating 子包

```python
from tin_coating import (
    ResidualCorrectionModel,
    run_cv_for_surface,
    load_pipeline_config,
    get_params_for_group,
    build_setpoint_group_key,
    PredictionPipeline,
    get_feature_cols,
)
import pandas as pd

df = pd.read_excel("result/data/feature_engineered_data/featured_data.xlsx")
df = build_setpoint_group_key(df)
group_df = df[df["Setpoint_Group_Label"] == "Top2.799_Bot2.799"].copy()

cfg = load_pipeline_config("group_params_2_799_only.json")
params = get_params_for_group(cfg, "Top2.799_Bot2.799", "Top")

# CV
cv = run_cv_for_surface(group_df, "Top", params)
print(f"Top RMSE = {cv['cv_rmse_mean']:.4f} ± {cv['cv_rmse_std']:.4f}")

# 推理
pipeline = PredictionPipeline("result/trained_models/by_group/Top2.799_Bot2.799")
result = pipeline.predict_both_surfaces(group_df.head(50), group_label="Top2.799_Bot2.799")
```

---

## 线上模型管理（monitoring/）

```bash
# 1. 质量门：数据分布 + 缺失 + 规格占比
python monitoring/check_data_quality.py
# 2. 健康检查：训练 vs 月度 RMSE / MAE / 漂移
python monitoring/check_monthly_health.py
# 3. 自动重训候选：触发 Walk-Forward CV，产出 .pending 模型
python monitoring/auto_retrain_candidate.py --force
# 4. 审批 Gate：唯一有权覆盖线上 joblib 的脚本
python monitoring/gate.py --list                                        # 列出版本历史
python monitoring/gate.py --approve Top2.799_Bot2.799 Top \
    --reason "walk-forward CV 改进" --yes                              # 部署
python monitoring/gate.py --rollback Top2.799_Bot2.799 Top --yes        # 回滚
```

所有版本历史写在 `monitoring/registry.json`，每条记录带 `cv_rmse_mean / cv_rmse_std / holdout_test_rmse / deploy_reason`，随时可 `--rollback`。

---

## 7×24 无人值守

### 1. 在线推理服务

```bash
python cli/main.py serve --host 0.0.0.0 --port 8080
```

启动时先跑自检（配置 / registry / 模型产物 / 日志目录），任一致命项失败直接退出（fail-fast），不会带病上线。

| 端点 | 方法 | 用途 |
|---|---|---|
| `/healthz` | GET | 存活探针：进程能响应即 200 |
| `/readyz` | GET | 就绪探针：Top/Bot 模型都已加载才 200，否则 503 |
| `/metrics` | GET | Prometheus 文本指标（请求数 / 延迟直方图 / 预测行数 / 就绪状态）|
| `/version` | GET | 服务版本 + 当前模型版本 |
| `/predict` | POST | 批量推理 `{"surface":"Top|Bot|both","rows":[...]}` |
| `/reload` | POST | 强制重载模型 |

- **模型热加载**：每次请求顺带检查 registry 里 active 版本与文件指纹，换模型不用重启；新模型加载失败时保留旧模型继续服务并告警。
- **并发保护**：信号量限制并发（`service.max_concurrent_requests`），排队超时返回 503；请求体上限 16MB，批大小上限 `service.max_batch_rows`。
- **鉴权**：设置环境变量 `TIN_API_KEY` 后，`/predict`、`/reload` 需带 `X-API-Key`。
- 日志：`monitoring/logs/service.jsonl`（结构化 JSON，10MB 轮转 5 份）；告警：`monitoring/logs/alerts.jsonl`。

`POST /predict` 示例：

```json
{"surface": "Top", "rows": [{"Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Avg": 2.81, "...": "..."}]}
```

### 2. 无人值守任务

```bash
python cli/main.py ops selfcheck              # 上线前/巡检：配置、registry、模型、日志目录
python cli/main.py ops run --job quality      # 数据质量门
python cli/main.py ops run --job health       # 月度健康检查
python cli/main.py ops run --job retrain      # 自动重训候选（生成 pending，等 gate 审批）
```

退出码：`0` 正常 / `1` 警告 / `2` 失败 / `3` 跳过（同一 job 上一轮未结束）。失败或警告会自动写 `monitoring/logs/alerts.jsonl`；配置了 `TIN_ALERT_WEBHOOK` 时同时 POST 到 webhook（JSON body 含 `level/title/detail`）。

同一 job 用文件锁 `monitoring/locks/job_<name>.lock` 防重叠执行；registry 的读改写也走同一把跨进程锁，定时任务和人工操作不会互相覆盖。

### 3. OS 调度示例

**Linux / cron**（每天 02:00 质量门、每月 1 号健康检查）：

```cron
0 2 * * *    cd /opt/tin-coating/Task1 && python cli/main.py ops run --job quality  >> monitoring/logs/cron.log 2>&1
0 3 1 * *    cd /opt/tin-coating/Task1 && python cli/main.py ops run --job health   >> monitoring/logs/cron.log 2>&1
0 4 * * 0    cd /opt/tin-coating/Task1 && python cli/main.py ops run --job retrain  >> monitoring/logs/cron.log 2>&1
```

**Linux / systemd**（服务常驻 + 崩溃自动重启）：

```ini
# /etc/systemd/system/tin-coating.service
[Unit]
Description=Tin Coating Inference Service
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/tin-coating/Task1
Environment=TIN_COATING_ROOT=/opt/tin-coating/Task1
ExecStart=/usr/bin/python3 cli/main.py serve --host 0.0.0.0 --port 8080
Restart=always
RestartSec=5
KillSignal=SIGTERM

[Install]
WantedBy=multi-user.target
```

**Windows / 任务计划**（开机常驻；`python` 建议写绝对路径，SYSTEM 账户 PATH 可能不同）：

```powershell
schtasks /Create /TN "TinCoating-Serve" /SC ONSTART /RU SYSTEM /RL HIGHEST /F /TR "cmd /c cd /d \"E:\SGAI_Project\2026\Tin coating\Task1\" && python cli/main.py serve"
schtasks /Create /TN "TinCoating-Quality" /SC DAILY /ST 02:00 /F /TR "cmd /c cd /d \"E:\SGAI_Project\2026\Tin coating\Task1\" && python cli/main.py ops run --job quality"
```

> 进程管理器（systemd / NSSM / 任务计划）只需保证：进程异常退出后自动拉起；拉起后 `/readyz` 返回 200 即恢复服务。

### 4. 运维参数

`configs/app.yaml`：

```yaml
service:
  host: 0.0.0.0
  port: 8080
  max_batch_rows: 5000
  max_concurrent_requests: 4
  request_timeout_seconds: 30
  api_key_env: TIN_API_KEY
ops:
  lock_timeout_seconds: 5
  alert_webhook_env: TIN_ALERT_WEBHOOK
```

可用 `TIN_COATING__SERVICE__PORT=9090` 这类环境变量临时覆盖，无需改文件。

---

## 全局自检

```bash
python __main_smoke_test.py
```

跑 8 项：tin_coating 全部模块 import / 原 coating_model_by_group 零副作用 / Walk-Forward CV RMSE 范围 / CLI cv / CLI monitor quality / 目录结构 / 单向依赖闭环。任何一项 FAIL 会 exit 1 并打印明细。

---

## 关键约束（不要动）

1. **生产入口零副作用** —— `coating_model_by_group.py` 可以独立 import 和 `python coating_model_by_group.py` 运行，导出的函数/类/常量签名不变。
2. **单向依赖** —— 新代码 `tin_coating/*` / `cli/*` / `monitoring/*` 可以 import 顶层旧代码，但旧代码永不 import 新代码。`__main_smoke_test.py` 的"原顶层 .py 无 tin_coating/cli/monitoring import"检查保证这一点。
3. **规格组过滤** —— `Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_TOP_Min` `.round(3) == 2.799`（Bot 同理）。
4. **Lab 列名是中文** —— `上表面镀层重量A(XA1_0)` / `下表面镀层重量A(XA1_0)`，不是英文 Top/`Bot...`。
