# 最大规格组 (SP=2.799) 镀层偏差预测 —— 探索历程与最终结论

> 数据来源：`Task1/result/data/cleaned_data/cleaned_data.xlsx`
> 有效样本：全量 9328 条，筛选后最大规格组 **2813 条**（约占 30%）
> 时间跨度：2025-07 ~ 2026-06（约 11 个月）
> 目标：**用在线过程数据预测实验室化验值偏差 Δ = Lab − Online**，再以 `Online + 预测Δ` 得到校正后的镀层厚度

---

## 一、预测思路

- **残差建模**：不去直接预测 Lab 值，而是预测 `Delta = Lab − Online`（Online 偏差），最终 `pred = Online + 预测Δ`
- 两层目标：
  - **科研目标**：理解偏差来自哪里，评估模型可预测性上限
  - **生产目标**：LGBM_full 特征集中的特征是否全部实时可得？（已确认：全是实时特征）

---

## 二、最终稳定结果（Walk-Forward 5 折 CV）

### 模型对比 —— TOP 面

| 模型 | RMSE | CV% | Hit1% | Hit5% | 特征数 | 说明 |
|---|---|---|---|---|---|---|
| **Online_raw** (基线) | 0.169 ± 0.027 | 16.1% | 3.3% | 39.1% | 0 | 直接用在线读数 |
| **Bias_shifted** (零模型) | 0.079 ± 0.016 | 20.4% | 27.9% | 88.5% | 0 | 仅做固定平移修正 |
| **LGBM_lag_only** | 0.068 ± 0.012 | 17.3% | 31.7% | 93.1% | 10 | 仅用 Online_SP_Diff lag1~lag5 |
| **LGBM_full** | **0.054 ± 0.005** | **9.7%** ✅ | **33.4%** | **98.9%** | 27 | 本卷过程变量 + lag1~lag5 |

### 模型对比 —— BOT 面

| 模型 | RMSE | CV% | Hit1% | Hit5% | 说明 |
|---|---|---|---|---|---|
| **Online_raw** (基线) | 0.194 ± 0.037 | 18.8% | 1.6% | 22.1% | |
| **Bias_shifted** (零模型) | 0.082 ± 0.010 | 12.0% | 28.0% | 86.6% | |
| **LGBM_lag_only** | 0.065 ± 0.006 | 8.8% | 33.9% | 94.9% | |
| **LGBM_full** | **0.054 ± 0.004** | **8.0%** ✅ | **44.0%** | **98.4%** | |

**稳定性评定**：LGBM_full 两面上 CV% 均 < 10%，是所有模型中最稳的。CV% 越低越好（反映模型对切分点的敏感度）。

### 可预测性分解（LGBM_full 相对基线）

```
                    TOP 面         BOT 面
基线 RMSE           0.169          0.194
  ↓ 降 53-58%
Bias_shifted        0.079          0.082       ← 53% 来自"可一次性修正的系统偏移"
  ↓ 再降 12-14%
LGBM_lag_only       0.068          0.065       ← 12% 来自"AR(5) 时滞惯性"
  ↓ 再降 21-25%
LGBM_full           0.054          0.054       ← 24% 来自"本卷工艺过程变量"

Hit1%  基线 → full：  3.3% → 33.4% (TOP),   1.6% → 44.0% (BOT)   ← 升 10~27 倍
Hit5%  基线 → full： 39.1% → 98.9% (TOP),  22.1% → 98.4% (BOT)   ← 覆盖近 100%
```

---

## 三、特征工程详解

LGBM_full 共 **27 个特征**，按来源分三类：

### 🟢 原始字段（9 个，直接来自 Excel 重命名）

| 脚本名 | Excel 原始列名 | 含义 | 入模？ |
|---|---|---|---|
| `Thickness` | `Dimension_[mm]_Thickness` | 在线测量厚度 | ✅ |
| `Width` | `Dimension_[mm]_Width` | 带钢宽度 | ✅ |
| `Speed` | `Speed[m/min]_Process_Avg` | 产线速度 | ✅ |
| `Online_TOP` | `Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Avg` | TOP 面在线镀层重量 | ✅ |
| `Online_BOT` | `Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Avg` | BOT 面在线镀层重量 | ✅ |
| `SP_TOP` | `Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_TOP_Avg` | TOP 面设定值 | ❌ 被 `Online_SP_Diff_TOP` 吸收 |
| `SP_BOT` | `Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_BOT_Avg` | BOT 面设定值 | ❌ 被 `Online_SP_Diff_BOT` 吸收 |
| `Steel_Grade` | `Steel Grade` | 钢种（类别） | ✅ → 编码后入模 |
| `Produce_Time` | `Produce Time` | 生产时间 | ❌ 派生出 Hour/DayOfWeek |

> 另外原始有 36 个 `GL_1 ~ GL_36`（每个镀锡槽的电流），**没有直接入模**——按工艺逻辑"奇数槽=下表面、偶数槽=上表面"聚合后生成电流和。

### 🟡 构造特征（在 `load_and_fe` 里生成的，共 18 个）

#### 第 1 类：36 槽电流聚合（行 204-208）

```python
# 奇数槽 (1,3,...,35) = 下表面; 偶数槽 (2,4,...,36) = 上表面
Bot_Current_Sum = GL_1 + GL_3 + ... + GL_35
Top_Current_Sum = GL_2 + GL_4 + ... + GL_36
```

**合理聚合**——36 个槽各自电流没必要单独喂（维度高、很多冗余），按物理意义求和保留总量。

#### 第 2 类：交叉比值 / 差值（行 210-218）

```python
Top_Current_Density    = Top_Current_Sum / Width          # 电流 / 宽度 = 电流密度
Bot_Current_Density    = Bot_Current_Sum / Width
Online_SP_Diff_TOP     = Online_TOP - SP_TOP             # 设定值偏差（生产实时可用）
Online_SP_Diff_BOT     = Online_BOT - SP_BOT
Delta_TOP              = Lab_TOP - Online_TOP             # ← 目标变量本身，不入模
Delta_BOT              = Lab_BOT - Online_BOT             # ← 目标变量本身，不入模
Speed_x_Thickness      = Speed * Thickness                # 速度×厚度 ≈ 质量流量代理
Top_Bot_Online_Diff    = Online_TOP - Online_BOT          # 上下表面在线读数差
Top_Bot_Current_Ratio  = Top_Current_Sum / Bot_Current_Sum  # 上下表面电流比
```

这些是**工业过程数据的常规操作**——比率、差值、乘积能表达原始单变量表达不出来的相互关系。注意 `Delta_TOP/BOT` 是目标变量本身（`Lab − Online`），**不入模**，只在训练时当 y 用。

#### 第 3 类：时间派生（行 220-222）

```python
Hour      = Produce_Time.dt.hour          # 小时 (0-23)
DayOfWeek = Produce_Time.dt.dayofweek     # 星期几 (0-6)
```

#### 第 4 类：滞后特征（`add_lag_features`，10 个）

```python
Online_SP_Diff_TOP_lag1 ~ lag5    # 前 5 卷的 TOP 设定值偏差
Online_SP_Diff_BOT_lag1 ~ lag5    # 前 5 卷的 BOT 设定值偏差
```

**为什么选 lag1~lag5 而不是 lag1~lag10？** ACF/PACF 诊断告诉我们 `Online_SP_Diff` 的 **PACF 在 lag5 截尾**（lag6+ 之后偏自相关完全进入 CI 置信区间），加再多就是噪声。

**为什么不用 Delta 的 lag？** 实验室化验值非实时，预测当前卷时前一卷的 Lab 值还没出，生产中用不了。

#### 第 5 类：Steel_Grade_enc（LabelEncoder）

每折独立在 train 上 fit，test 上 transform（未知类别兜底 -1）。

### 📋 最终 27 个入模特征按来源汇总

| 来源 | 特征 | 数量 |
|---|---|---|
| 原始直接用 | `Thickness`, `Width`, `Speed`, `Online_TOP`, `Online_BOT` | **5** |
| 36 槽聚合 | `Top_Current_Sum`, `Bot_Current_Sum` | **2** |
| 交叉/比值 | `Top_Current_Density`, `Bot_Current_Density`, `Online_SP_Diff_TOP`, `Online_SP_Diff_BOT`, `Speed_x_Thickness`, `Top_Bot_Online_Diff`, `Top_Bot_Current_Ratio` | **7** |
| 时间派生 | `Hour`, `DayOfWeek` | **2** |
| 编码 | `Steel_Grade_enc` | **1** |
| 滞后 | `Online_SP_Diff_TOP_lag1~5`, `Online_SP_Diff_BOT_lag1~5` | **10** |
| **合计** | | **27** |

### 🤔 有没有冗余特征？

老实说有两个可以考虑精简：
- `Steel_Grade_enc`：最大规格组里钢种可能只有 1-2 个，信息量小
- `Top_Bot_Current_Ratio` 和 `Top_Current_Sum` + `Bot_Current_Sum` 之间有强冗余（ratio = sum_top / sum_bot），LightGBM 自己能算出 ratio

但 LightGBM 通过树的分裂频率会自动筛掉没用的特征（重要性降为 0），**不精简也没坏处**。

---

## 四、探索历程（关键转折点）

### ① 起点：发现路径 bug → 修正后跑出 R2=0.58

原始脚本用 `Path(__file__)/"result"/...` 拼路径，但 `Path(__file__)` 是**文件路径**（`reproduce_best_largest_group.py`），不能当目录用。改成 `Path(__file__).parent/"result"/...` 后跑通，报告 R2=0.58/0.50，RMSE≈0.04。

**问题**：这个 R2 是被高估还是真实？

---

### ② 随机 train_test_split → 时序切分，R2 崩了

关键修正：用户指出"数据行序就是严格的时间顺序"。

原代码：
```python
train_test_split(idx, test_size=0.2, random_state=42)   # 随机打散！
```

修正后：
```python
g = g.sort_values("Produce_Time").reset_index(drop=True)
train_end = len(g) - int(len(g) * TEST_SIZE)
g_train = g.iloc[:train_end].copy()
g_test  = g.iloc[train_end:].copy()    # 后 20% = 测试期
```

结果对比：

| 实验 | 分割方式 | TOP R2 | TOP RMSE | TOP Hit1% |
|---|---|---|---|---|
| ① | 随机（有时间穿越） | **0.5831** | 0.0406 | 57.2% |
| ② | 时序（无穿越） | **-0.4061** | 0.0527 | 39.9% |

**结论**：随机切分让测试集混入了时间上早于训练集的样本，模型"开了天眼"。修正后 R2 跌 1 个点是消除 look-ahead bias 后的真实水平。

---

### ③ early stopping：best_iter 只有 26/37

时序切分后 LightGBM 在 TOP 面 best_iter=26，BOT 面 37 就停了。600 棵树硬训 vs 加了 `early_stopping(50)` 后只训 26 棵。

**根因**：
- train 内部本身就有严重漂移（train 覆盖 9 个月），模型在前 20-30 棵树学到早期规律后，再训就是对早期特定模式过拟合
- validation loss 立刻反弹，early stopping 触发

**修复**：`n_estimators` 调到 1000（给早停留上限），保留 early_stopping(50)。

---

### ④ LabelEncoder：从全量 fit → 仅 train fit

原代码：
```python
le.fit_transform(g["Steel_Grade"])   # 在 train_test_split 之前！
```

修正后（在 split 之后）：
```python
le.fit(g_train["Steel_Grade"].astype(str))
g_train["Steel_Grade_enc"] = le.transform(...)
g_test["Steel_Grade_enc"] = test_labels.apply(
    lambda x: le.transform([x])[0] if x in known else -1
)
```

---

### ⑤ Lag 特征：ACF/PACF 告诉我们是 AR(5) 过程

跑了 ACF/PACF 诊断后才知道：
- `Online_SP_Diff` 的 **PACF 在 lag5 截尾** → 是 AR(5) 过程
- lag1 PACF 高达 0.87-0.90，lag2-5 仍显著，lag6 之后全跌进 CI（±0.037）
- **lag-only vs Bias_shifted 对比**：lag-only RMSE=0.068 vs Bias_shifted=0.079，滞后特征只比固定平移好 14%——AR 结构的增量有限
- **Lab_lag 排除**：实验室化验值非实时，生产中预测当前卷时前一卷的 Lab 值还没出来

最终 lag 策略：只用 **Online_SP_Diff_TOP_lag1~5 + Online_SP_Diff_BOT_lag1~5**（10 个实时可得的特征）。

---

### ⑥ 滚动窗口 W=500：缩反了

把训练集限制为"train 区最后 500 条"，结果更差（R2 -0.68 比全量 -0.34 更差）。**模型需要数据量来覆盖工艺变化的多样性**，单纯减小窗口不能解决分布漂移问题。恢复全量训练。

---

### ⑦ Walk-Forward 5 折 CV：确认稳定性

最终方案，每折独立做 LabelEncoder + lag 构造 + 训练/评估。切分方式：

```
折1: train = 前 70% (1969条),   test = 70-75% (141条)
折2: train = 前 75% (2110条),   test = 75-80% (141条)
折3: train = 前 80% (2251条),   test = 80-85% (141条)
折4: train = 前 85% (2393条),   test = 85-90% (141条)
折5: train = 前 90% (2534条),   test = 90-100% (279条) ← 覆盖尾部
```

---

## 五、为什么 R2 为负但模型有用？

同规格组 2813 个样本，Lab 值全部挤在 **2.75~2.85** 极窄区间内（SP=2.799），总方差极小。R2 = 1 − SS_res/SS_tot，分母 SS_tot 太小导致：

- baseline R2 = -14：RMSE 0.17 远大于 Lab 方差
- Bias_shifted R2 = -2：RMSE 0.08 仍比 Lab 方差大
- LGBM_full R2 = -0.29~-0.15：RMSE 0.05 刚好接近但还没压过 Lab 方差

**不要纠结 R2 转正**，在同规格组这种特殊场景下 RMSE、Hit1%、Hit5% 才是可靠指标。

---

## 六、best_iter 逐折变化

| Fold | train 条数 | TOP full | TOP lag | BOT full | BOT lag |
|---|---|---|---|---|---|
| 1 | 1969 | 31 | 18 | 21 | 13 |
| 2 | 2110 | 53 | 26 | 33 | 24 |
| 3 | 2251 | 31 | 25 | 43 | 20 |
| 4 | 2393 | 189 | 14 | 34 | 16 |
| 5 | 2534 | 117 | 34 | 46 | 35 |
| **均值** | | **84.2±68.4** | **23.4±7.7** | **35.4±9.8** | **21.6±8.6** |

- lag-only 的 best_iter 极其稳定（13-35 棵）：纯 AR(5) 结构的有效信息量就是固定的
- TOP full 在 F4/F5 暴涨（189/117 棵）：训练数据多了 + 内部漂移小了 → early stopping 不再早触发
- BOT full 更稳定（21-46 棵）：BOT 面增量上限更低

---

## 七、生产可用结论

| 层次 | 改进 | 生产中能否实现？ |
|---|---|---|
| **① 系统偏移**（53%） | 知道偏差规律后做一次性平移 | ✅ **最容易**，定期校准传感器 |
| **② 时滞惯性**（12%） | Online_SP_Diff lag1~lag5 | ✅ 实时可得，当前卷之前的 SP 和 Online 值全部存档 |
| **③ 本卷工艺变量**（24%） | Speed、Width、电流、Hour 等 | ✅ 全部实时可测 |

LGBM_full 27 个特征中**没有任何一个是非实时的**（Delta 没入模，只用了 Online_SP_Diff 的 lag），在产线上部署完全可行。

**注意**：其他规格组（非 SP=2.799）需要单独建模，因为不同规格组的工艺窗口可能不同。

---

## 八、脚本文件说明

### 入口脚本
- `reproduce_best_largest_group.py` — 包含：
  - `load_and_fe()`：数据加载与特征工程（电流和、密度、差值、比值、时间特征）
  - `add_lag_features()`：时序安全的 lag 特征构造（train 和 test 独立 lag，test 首行从 train 末行续接）
  - `diagnose_autocorr()`：ACF/PACF 诊断
  - `evaluate()` / `print_metrics()`：评估工具
  - `run_single_fold()`：单折训练（4 个模型）
  - `main()`：Walk-Forward 5 折 CV 主流程

### 输出文件
- `reproduce_results.csv` — 每折 × 每模型的完整指标（20 行）
- `cv_summary.csv` — CV 均值汇总（8 行）

### 依赖
```
pip install pandas numpy openpyxl scikit-learn lightgbm statsmodels
```

### 运行
```powershell
& ".venv/Scripts/python.exe" "largest_group_repro/reproduce_best_largest_group.py"
```

---

## 九、后续方向

- **推广到其他规格组**：当前只做了 SP=2.799/2.799，其他规格组（如 2.6/2.6、2.8/2.8）需要同样流程验证
- **生产部署**：在产线上实时更新偏差预测模型，每生产一卷就用最新数据增量训练
- **Bias 随时间漂移的诊断**：如果能拿到传感器校准/换辊/换料记录，可以定位 bias 漂移的触发事件
- **Lab 值滞后场景**：如果未来 Lab 化验变准实时，可以把 `Delta_lag1~lag5` 加进模型，预期 RMSE 再降 10-15%

---

## 附录：四模型完整指标（CV 均值）

### TOP 面

| 模型 | R2 | RMSE | MAE | Hit1% | Hit2% | Hit3% | Hit5% |
|---|---|---|---|---|---|---|---|
| Online_raw | -12.69 | 0.1693 | 0.1553 | 0.0328 | 0.0934 | 0.1375 | 0.3913 |
| Bias_shifted(N=300) | -1.94 | 0.0792 | 0.0641 | 0.2787 | 0.4642 | 0.6533 | 0.8854 |
| LGBM_lag_only | -1.10 | 0.0678 | 0.0539 | 0.3170 | 0.5737 | 0.7515 | 0.9309 |
| LGBM_full | **-0.29** | **0.0535** | **0.0436** | **0.3342** | **0.6684** | **0.8562** | **0.9893** |

### BOT 面

| 模型 | R2 | RMSE | MAE | Hit1% | Hit2% | Hit3% | Hit5% |
|---|---|---|---|---|---|---|---|
| Online_raw | -15.25 | 0.1944 | 0.1821 | 0.0157 | 0.0527 | 0.0855 | 0.2214 |
| Bias_shifted(N=300) | -1.73 | 0.0818 | 0.0655 | 0.2802 | 0.4665 | 0.6309 | 0.8663 |
| LGBM_lag_only | -0.71 | 0.0649 | 0.0504 | 0.3391 | 0.6250 | 0.7630 | 0.9488 |
| LGBM_full | **-0.15** | **0.0536** | **0.0406** | **0.4396** | **0.6941** | **0.8491** | **0.9836** |
