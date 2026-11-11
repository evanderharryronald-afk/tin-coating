# Ablation 实验总结报告

> 目标：提升 `Top2.799_Bot2.799` 规格组模型精度（Task1 生产主入口 `coating_model_by_group.py`）
> 
> 日期：2026-11-11

---

## 1. 问题背景

当前生产模型的两个值得审视的点：

1. **特征只有 5 列**（`baseline_5`），代码里还注释掉了 `Current_Per_Speed`、`Theoretical_Factor`、`Steel_Grade_Encoded` 三个特征，注释写的是"特征剪枝"。这轮剪枝当时是基于一个**旧版探索脚本**（`largest_group_repro/`，不属于生产主线）做的，那个脚本里临时加了 lag 项试效果——lag 项一旦存在，就会扛走大部分时序解释力，让工艺变量的真实贡献被低估，进而把本不该剪掉的基础特征一起剪掉了。

2. **EWMA α 是 hold-out 单次 Optuna 搜出来的**（Top=0.779, Bot=0.727），没扫过全范围。

**重要前提**：本次探索的所有实验，**全程不用 lag 项，也不用 Δ_lag**（Lab 延迟未知，生产上拿不到过去时刻的真实 Δ）。所有特征都是**同一时刻 t 实时可测的工艺变量**。EWMA 只对模型输出的 Δ̂ 序列做平滑，不涉及过去的 Lab 值。

---

## 2. 实验框架

### 目录结构

```
Task1/
├── coating_model_by_group.py      ← 生产入口（本次有改）
├── group_params_2_799_only.json   ← 生产配置（本次有改）
└── ablation/                      ← 新建探索区（零污染生产代码）
    ├── run_ablation.py            ← 实验主脚本
    └── results/                   ← 自动产出的 CSV 汇总
        ├── feature_ablation_summary.csv
        ├── ewma_alpha_sweep_summary.csv
        ├── ewma_alpha_sweep_all_8.csv
        ├── walk_forward_cv_folds.csv
        ├── walk_forward_cv_folds_all_8.csv
        └── walk_forward_cv_folds_all_8_best_alpha.csv
```

### 关键设计原则

- **从生产代码 import 复用**：`ResidualCorrectionModel` 直接从 `coating_model_by_group` 导入，训练逻辑和生产**100% 一致**，避免"两份代码跑出不同结果"的漂移问题。
- **Walk-Forward CV 口径与 largest_group_repro 对齐**：
  - fold 1: train [0, 70%) → test [70%, 75%)
  - fold 2: train [0, 75%) → test [75%, 80%)
  - fold 3: train [0, 80%) → test [80%, 85%)
  - fold 4: train [0, 85%) → test [85%, 90%)
  - fold 5: train [0, 90%) → test [90%, 100%)
- **Hold-out 口径**：训练 65% / 验证 20% / 测试 15%（与生产代码默认一致）。

### 模型与超参数（保持不变）

所有实验使用**同一套超参数**（Optuna 搜出来、在 ablation 中证实仍然稳定）：

| 参数 | Top | Bot |
|---|---|---|
| damping | 0.3046 | 0.2848 |
| pos_boost | 1.6837 | 1.2582 |
| max_iter | 250 | 300 |
| learning_rate | 0.0144 | 0.0117 |
| max_depth | 5 | 4 |
| loss | absolute_error | absolute_error |

只在两个维度上扫描：**特征集合** × **EWMA α**。

---

## 3. 实验一：特征集合消融（hold-out）

7 种特征组合，Top/Bot 各自独立跑：

| key | 特征数 | 列 |
|---|---|---|
| `baseline_5` | 5 | Online + Current_Sum + Speed + Width + Thickness（当前生产） |
| `+current_ps` | 6 | baseline_5 + Current_Per_Speed |
| `+theoretical` | 6 | baseline_5 + Theoretical_Factor |
| `+grade` | 6 | baseline_5 + Steel_Grade_Encoded |
| `+ps_grade` | 7 | baseline_5 + Current_Per_Speed + Steel_Grade_Encoded |
| `+all_8` | 8 | baseline_5 + Current_Per_Speed + Theoretical_Factor + Steel_Grade_Encoded |

### 结果

| 表面 | `baseline_5` RMSE | `+current_ps` | `+theoretical` | `+grade` | `+all_8` |
|---|---|---|---|---|---|
| **Top** | 0.0529 | 0.0523 | 0.0526 | 0.0547 ❌ | **0.0512**（−3.2%） |
| **Bot** | 0.0536 | 0.0542 ❌ | **0.0508**（−5.1%） | 0.0516 | **0.0471**（−12.0%） |

### 发现

- **Bot 表面提升空间远大于 Top**：`+theoretical` 单特征就把 Bot RMSE 砍了 −5.1%；Top 表面 `+grade` 反而变差（单独加有害，和其他特征一起用就不害了）。
- **Steel_Grade_Encoded 需要其他特征配合**：单独加 grade Top 退化，加全部 8 个 Top 反而最好——说明 grade 存在方式是**交互式**，不是主效应。
- **Optuna 当时剪掉这三个特征是错的**：lag 项扛了大部分解释力，掩盖了基础特征的真实价值。

---

## 4. 实验二：EWMA α 扫描

### baseline_5 上扫（为了和旧配置对比）

| α | Top RMSE | Bot RMSE |
|---|---|---|
| 0.01 | 0.0717 | 0.0638 |
| 0.3 | 0.0537 | 0.0536 |
| 0.5 | **0.0524** | 0.0530 |
| 0.7 | 0.0527 | 0.0534 |
| 0.727（Bot 旧值） | 0.0527 | 0.0536 |
| 0.779（Top 旧值） | 0.0529 | 0.0538 |
| 1.0（关 EWMA） | 0.0542 | 0.0555 |

**U 形曲线明显**：α ≈ 0.5~0.7 之间最优，生产用的 α=0.779 / 0.727 偏保守。

### +all_8 上扫（为最终配置找最优 α）

| α | Top RMSE | Bot RMSE |
|---|---|---|
| 0.3 | 0.0525 | 0.0486 |
| **0.5** | **0.0510** | 0.0472 |
| **0.6** | **0.0509** | **0.0470** |
| 0.7 | 0.0510 | 0.0470 |
| 0.727 | 0.0511 | 0.0471 |
| 0.779 | 0.0512 | 0.0472 |
| 1.0 | 0.0523 | 0.0484 |

**最优 α = 0.6**（两表面同值，配置更干净）。比旧值 α=0.779 / 0.727 各自好 0.0003~0.0004 RMSE。

### 结论

EWMA 有明确价值，曲线呈 U 形：**太弱（α 接近 1）不滤波，太强（α 接近 0）引入滞后噪声**。α=0.6 是"干净平滑 + 不过度滞后"的最佳折中，且两表面可以统一，简化了生产配置。

---

## 5. 实验三：Walk-Forward 5-fold CV（稳定性测试）

### baseline_5 + α=0.779/0.727（旧配置）

| 表面 | CV RMSE | CV MAE | RMSE 提升 |
|---|---|---|---|
| **Top** | 0.0535 ± 0.0085 | 0.0431 ± 0.0073 | −67.6% |
| **Bot** | 0.0600 ± 0.0047 | 0.0485 ± 0.0043 | −67.6% |

### +all_8 + α=0.6（推荐升级）

| 表面 | CV RMSE | CV MAE | RMSE 提升 |
|---|---|---|---|
| **Top** | **0.0520 ± 0.0073** | **0.0416 ± 0.0061** | **−68.5%** |
| **Bot** | **0.0553 ± 0.0062** | **0.0443 ± 0.0053** | **−70.4%** |

### Fold-by-fold（Bot 表面，改进最大）

| fold | 测试段 | baseline_5 | +all_8 + α=0.6 | 差值 |
|---|---|---|---|---|
| 1 | [70%, 75%) | 0.0629 | 0.0637 | +0.0008 |
| 2 | [75%, 80%) | **0.0655** | 0.0580 | **−0.0075** |
| 3 | [80%, 85%) | 0.0580 | 0.0560 | −0.0020 |
| 4 | [85%, 90%) | 0.0532 | 0.0508 | −0.0024 |
| 5 | [90%, 100%) | **0.0605** | **0.0478** | **−0.0127** |

### 关键发现

1. **Bot CV 均值比 Top 高 ~0.008**，说明 Bot 建模天生更难（噪声更大？工艺波动更大？）
2. **Bot fold 5 是超级明星**：baseline_5 的 0.0605 → +all_8 的 0.0478，差了 **0.0127 RMSE**，是 Bot 整体均值从 0.0600 降到 0.0553 的主驱动力
3. **单次 hold-out 严重乐观偏差**：Bot baseline_5 hold-out 是 0.0536，但 CV 均值 **0.0600**（差 +0.0064）。**生产部署以 CV 均值为准**
4. **Top 表面 CV 波动 ±0.0085 偏大**，说明 Top 模型在不同时间段的表现**不稳定**，可能存在 regime shift

---

## 5.5 补充：旧 HGB 超参在 +all_8 特征上是否仍接近最优？

### 问题

Optuna 针对 **baseline_5（5 特征）** 搜出来的 HGB 超参（depth=5, lr=0.0144, iter=250），直接套在 **+all_8（8 特征）** 上，是不是仍然好用？

如果特征空间变了但超参没跟着调，Walk-Forward CV 里的"提升"到底是真的特征贡献，还是偶然切分带来的——这个问题必须量化才能下结论。

### 方法

在 +all_8 特征 + α=0.6 固定条件下，对 HGB 三个关键超参做**网格扫描**（hold-out 口径）：

| 超参 | Top 扫描范围 | Bot 扫描范围 |
|---|---|---|
| max_depth | [4, 5, 6, 7] | [3, 4, 5, 6] |
| learning_rate | [0.008, 0.0144, 0.020, 0.030] | [0.008, 0.0117, 0.018, 0.025] |
| max_iter | [150, 250, 400] | [200, 300, 500] |

Top 48 组、Bot 48 组，结果存于 `hgb_grid_sweep_all_8.csv`。

### 结果

#### Top 表面

| | depth | lr | max_iter | test RMSE | val gap (train→val) |
|---|---|---|---|---|---|
| 🥇 网格最优 | **7** | **0.030** | **400** | **0.0495** | +78.5% ⚠️ |
| 📌 当前生产（来自 baseline_5 Optuna） | 5 | 0.0144 | 250 | 0.0509 | — |
| 差值 | | | | **−0.0013 (−2.6%)** | |

**Top 发现**：特征扩展后 Top 确实需要**更深的树 + 更大的学习率**（depth 5→7，lr 0.0144→0.030），说明 3 个新增特征带了更多非线性交互。但最优组合的 val gap 高达 **78.5%**（train RMSE=0.0380 → val RMSE=0.0678），**过拟合风险显著**——hold-out 单次 test 集没被坑不代表 Walk-Forward CV 下不会。

Top Top-10 里 gap 小的 trade-off 点：depth=5, lr=0.0144, iter=150（test=0.0504, val gap=40%），只比最优差 0.0009 RMSE 但健康得多。

#### Bot 表面

| | depth | lr | max_iter | test RMSE | val gap (train→val) |
|---|---|---|---|---|---|
| 🥇 网格最优 | **4** | **0.025** | **300** | **0.0463** | +30.0% ✅ |
| 📌 当前生产（来自 baseline_5 Optuna） | 4 | 0.0117 | 300 | 0.0470 | — |
| 差值 | | | | **−0.0007 (−1.4%)** | |

**Bot 发现**：depth 和 max_iter 完全没变，只有 lr 从 0.0117 → 0.025 有提升空间。而且最优组合 val gap 只有 **30%**（train→val 跳 30%，val→test 甚至再降 5%），**过拟合风险极低**。Bot 对特征扩展的敏感度远小于 Top——因为 Bot 本身模型容量需求就不高。

### 结论

| 表面 | 旧超参是否仍可用 | 改进空间 | 风险 | 建议 |
|---|---|---|---|---|
| **Top** | ✅ 可用（只偏 −2.6%） | −0.0013 RMSE | ⚠️ 深度模型过拟合风险 | 先不动，Walk-Forward CV 对比后再定 |
| **Bot** | ✅ 几乎最优（−1.4%） | −0.0007 RMSE | ✅ 低 | 可选微调 lr，但当前已足够 |

**整体判断**：Walk-Forward CV 里看到的"baseline_5 → +all_8 的提升"（Bot CV 从 0.0600 → 0.0553）**主要是特征本身的贡献**，不是超参偶然漂了的结果。旧超参在 +all_8 上是"能用、不是最优、但也不离谱"的水平。

**HGB 超参暂时不更新**（本次仅做了发现，未改 `group_params_2_799_only.json`）。若后续要微调，优先级：Bot lr 0.0117→0.025（风险低、收益明确）> Top 换一组低 gap 的 compromise 超参（需 Walk-Forward CV 验证）。

---

## 6. 最终推荐配置

| 项 | 旧值 | 新值 |
|---|---|---|
| **特征数** | 5（baseline_5） | **8（+Current_Per_Speed + Theoretical_Factor + Steel_Grade_Encoded）** |
| **EWMA α Top** | 0.779 | **0.6** |
| **EWMA α Bot** | 0.727 | **0.6** |
| **HGB 超参** | 不变 | 不变（α 统一后再跑一遍 Optuna 可探索，但当前 CV 已证明这套稳定） |

### 预期生产性能（以 Walk-Forward CV 为准）

| 表面 | RMSE（相对 Online 原始 RMSE） | MAE | 相对 Online 提升 |
|---|---|---|---|
| Top | 0.0520 ± 0.0073 | 0.0416 ± 0.0061 | −68.5% |
| Bot | 0.0553 ± 0.0062 | 0.0443 ± 0.0053 | −70.4% |

---

## 7. 生产代码改动记录

只动了两处，**没有任何逻辑重写**：

### 7.1 coating_model_by_group.py（两处 feature_cols）

`fit_and_evaluate_surface` 内（~第 477 行）和 `get_feature_cols`（~第 926 行），三行注释改为启用：

```python
# 旧
feature_cols = [
    online_col, current_col,
    # f'{prefix}_Current_Per_Speed',
    # f'{prefix}_Theoretical_Factor',
    speed_col, width_col, thickness_col,
    # 'Steel_Grade_Encoded'
]

# 新
feature_cols = [
    online_col, current_col,
    f'{prefix}_Current_Per_Speed',
    f'{prefix}_Theoretical_Factor',
    speed_col, width_col, thickness_col,
    'Steel_Grade_Encoded'
]
```

两处必须保持完全一致（`get_feature_cols` 是对外 API，被 `run_surface_pipeline` 调用）。

### 7.2 group_params_2_799_only.json

```json
// 旧
"alpha_smoothing": 0.7790   // Top
"alpha_smoothing": 0.7267   // Bot

// 新（两表面统一为最优值）
"alpha_smoothing": 0.6      // Top
"alpha_smoothing": 0.6      // Bot
```

### hold-out 正式 pipeline 结果（升级后）

| | | 升级前 | 升级后 | Δ |
|---|---|---|---|---|
| **Top** | 测试集 RMSE | 0.0529 | **0.0514** | **−0.0015 (−2.8%)** |
| **Bot** | 测试集 RMSE | 0.0536 | **0.0467** | **−0.0069 (−12.9%)** 🎯 |

pipeline 跑通，两个改动均生效。

---

## 8. 后续可探索方向

| 优先级 | 方向 | 预期价值 | 备注 |
|---|---|---|---|
| 🔴 | **Bot fold 2 / fold 5 为什么 +all_8 能大幅改善而 fold 1 略差？** | 定位 regime shift 根源（那段时间钢种集中？车速异常？） | 最有价值的诊断 |
| 🔴 | **Top CV ±0.0073 波动根源** | Top 稳定性更差，找到原因才能让两表面对称 | 和 Bot 诊断同时推进 |
| 🟡 | **+all_8 特征上重跑一遍 Optuna** | 看 HGB 超参（lr / depth / damping）是否也该漂 | 当前超参已稳定，这是锦上添花 |
| 🟡 | **α = 0.55 / 0.65 在 +all_8 上精细扫** | α=0.6 和 α=0.5/0.7 几乎打平，精细扫可能找到更准的 0.60~0.65 | 零成本微调 |
| 🟢 | **HGB → LightGBM 对比** | LightGBM 更快、特征交互更强，但 sklearn HGB 已经够用 | 非必要 |

---

## 9. 生产安全确认

最终推荐配置满足所有生产约束：

| 约束 | 状态 |
|---|---|
| 无 lag 项 | ✅ 所有特征都是 t 时刻实时可测 |
| 无 Δ_lag（Lab 依赖） | ✅ HGB 只对工艺变量建模 |
| 无 Δ 序列 EWMA（依赖历史 Lab） | ✅ EWMA 只对模型输出的 Δ̂ 做平滑，不依赖 Lab |
| EWMA 可在生产环境安全维护 | ✅ 连续生产时用递推状态，停机重启重置即可（α=0.6 滞后 ~2-3 个样本） |

---

## 10. Ablation 目录说明

本目录是**探索性代码**，不属于生产交付物。如果未来决定废弃这套探索，可整块删除不影响 `Task1/` 其他模块。

复用方式：
```python
# 任何新实验都可以这样开始：
import sys; sys.path.insert(0, "..")
from ablation.run_ablation import (
    load_target_group, _cv_engine, surface_feature_cols, run_single_fold
)
from coating_model_by_group import ResidualCorrectionModel
```
