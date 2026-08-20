# 步骤一：数据匹配与聚合 (Data Matching & Aggregation)

## 1. 业务与技术目标
按钢卷唯一标识（`Coil ID`）将在线生产过程仪表数据（主表）与实验室分析中心检测真值（副表）进行精确对齐与多源数据融合，构建包含物理过程特征与真实质量标记的全局建模基础大表。

## 2. 数据匹配流向与统计
本次处理共涉及 **10 个月份/批次的在线主表文件** 与 **1 个实验室检测副表文件**：

### (1) 主表（在线生产数据）纵向拼接
跨越 2025年7月 至 2026年6月，共读取 10 个 Excel 文件（单表 55 列）：
- 各月份数据量：2,719 + 2,496 + 2,093 + 2,406 + 2,637 + 2,936 + 2,648 + 2,158 + 2,158 + 2,601 行。
- 主表纵向合并后累积基础数据总量：**24,852 行 × 55 列**。

### (2) 副表（实验室检测值）前置剪枝与聚合匹配
副表原始数据规模庞大且存在冗余，通过两阶段前置剪枝提升效率与数据质量：

| 处理阶段 | 样本行数 | 数据变动说明 / 过滤规则 |
| :--- | :--- | :--- |
| **副表原始输入** | 44,839 行 | `25.6-26.6分析中心检测值.csv`（37列） |
| **规则过滤 1：必填项校验** | 18,258 行 | 剔除关键字段缺失/无效检测记录 |
| **规则过滤 2：主表对齐** | 11,149 行 | 仅保留在主表（在线数据）中真实存在的钢卷 Coil ID |
| **最终聚合对齐** | **9,901 行** | 执行底层向量化分组聚合，完成主副表横向拼接 |

## 3. 产出物与处理性能
- **执行耗时**：81.0 秒
- **最终生成大表**：`merged_result_latest.xlsx`
- **数据矩阵规格**：**9,901 行 × 69 列**（包含在线工艺参数与实验室 A/C/D/W 四点测定真值）

## 4. 核心工程优化亮点
1. **结构缓存机制**：利用主表结构缓存实现秒级加载，避免重复解析 Excel 样式与元数据。
2. **两阶段前置剪枝 (Pre-pruning)**：在横向 Join 前先按必填列与 Coil ID 集合过滤副表，大幅降低了内存占用与 Join 计算开销。
3. **向量化分组聚合**：采用 Pandas/Numpy 底层向量化运算替代循环匹配，保证了多源数据的高效吞吐。


# 步骤二：数据清洗与异常诊断 (Data Cleaning & Outlier Diagnosis)

## 1. 业务与技术目标
在完成主副表匹配后的全局大表基础上，系统性剔除影响模型训练质量的异常样本，包括：
- 关键工艺/测量参数缺失
- 低速/停机过渡工况
- 在线仪表零值、死值或非稳态波动
- 基于残差（实验室真值 - 在线测量值）的统计离群点

最终输出干净、可直接用于后续特征工程与建模的训练数据集，同时完整记录被剔除样本及其原因，便于追溯与业务复核。

## 2. 清洗规则体系（按执行顺序）

清洗器采用**多规则叠加 + 原因累加**机制，任意一条规则触发即标记剔除，并记录具体原因字符串。

| 规则编号 | 规则名称 | 判定逻辑 | 参数设置 |
|:---|:---|:---|:---|
| 规则1 | 关键字段缺失检查 | 以下任一字段存在缺失即剔除：电流总和、理论因子、速度、宽度、厚度、在线镀层均值、实验室真值、残差 | — |
| 规则2 | 低速/停机过渡区 | 过程平均速度 ≤ 20 m/min | `min_speed=20.0` |
| 规则3 | 在线仪表零值/死值 | 上表面或下表面：Min ≤ 0 或 Avg ≤ 0 | — |
| 规则4 | 仪表非稳态波动 | 上表面或下表面满足任一条件：<br>• Max - Min > 0.5 g/m²<br>• (Max - Min) / Avg > 0.4 | `max_range_abs=0.5`<br>`max_range_ratio=0.4` |
| 规则5 | 残差MAD离群点 | 仅在通过前4条规则的样本上计算中位数与MAD，剔除 \|残差 - 中位数\| > 3×MAD 的样本（上下表面独立判断） | `mad_factor=3.0` |

> 注：规则5的中心与MAD统计仅在“当前仍干净”的样本上计算，避免被前序异常污染。

## 3. 清洗统计结果

| 指标 | 数值 |
|:---|:---|
| 原始匹配后总行数 | 9,901 |
| 被剔除异常样本数 | 573 |
| 剔除占比 | **5.79%** |
| 保留干净样本数 | **9,328** |

剔除原因明细已完整导出至 `result/cleaned_data/filtered_outliers.xlsx`（含 Coil ID、关键测量值、残差及完整 `Filter_Reason` 字符串），可支持后续按原因分类统计与业务复核。

## 4. 产出物

- **干净训练数据**：`result/cleaned_data/cleaned_data.xlsx`（9,328 行）
- **被剔除明细**：`result/cleaned_data/filtered_outliers.xlsx`（含剔除原因）
- 清洗过程中同步计算并保留的字段（供后续使用）：
  - `Top_Current_Sum` / `Bot_Current_Sum`
  - `Top_Theoretical_Factor` / `Bot_Theoretical_Factor`
  - `Top_Delta` / `Bot_Delta`（原始残差）
  - `Top_Delta_Centered` / `Bot_Delta_Centered`（去全局偏差后的残差）
  - `Steel_Grade_Encoded`（钢种频率编码）

## 5. 工程实现与流程说明

1. **规则叠加设计**：采用字符串累加记录所有触发原因，便于一次排查多重异常。
2. **MAD计算保护**：离群点检测仅在已通过基础物理/工况过滤的样本上进行，提升稳健性。
3. **特征工程混入说明**：  
   当前清洗代码中同步完成了电流求和、理论镀层因子、残差计算及钢种编码等特征工程操作。  
   **建议后续流程优化**：将纯清洗逻辑与特征工程彻底解耦——清洗阶段仅负责样本过滤与原因记录，特征工程作为独立步骤在干净数据上执行，以提升代码可维护性与可复用性。

## 6. 小结
经过上述五条规则严格过滤后，数据质量显著提升，异常样本占比控制在 5.79%，保留了 9,328 条高质量样本，为后续目标分布分析、相关性分析及模型训练提供了可靠基础。


# 步骤三：预测目标分布与相关性分析 (Target Distribution & Correlation Analysis)

## 1. 业务与技术目标
在干净数据集上，系统评估预测目标（残差 = 实验室真值 - 在线测量值）与各工艺/测量特征之间的线性与非线性关联强度，识别对残差最具解释力的变量，为后续特征选择与模型建模提供数据驱动依据。

分析维度覆盖：
- **Pearson 相关系数**（线性关系）
- **Spearman 秩相关系数**（单调关系）
- **Mutual Information（互信息）**（非线性依赖）
- **Distance Correlation（距离相关性）**（更广义的依赖关系）

上下表面（Top / Bot）独立分析。

## 2. 分析范围与特征清单
分析对象为清洗后保留的 9,328 条样本。主要考察特征包括：
- 在线测量值：`Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP/BOT_Avg`
- 理论因子：`Top/Bot_Theoretical_Factor`
- 电流相关：`Top/Bot_Current_Sum`、`Top/Bot_Current_Per_Speed`
- 尺寸与速度：`Dimension_[mm]_Width`、`Dimension_[mm]_Thickness`、`Speed[m/min]_Process_Avg`
- 钢种编码：`Steel_Grade_Encoded`
- 其他偏差特征：`Top/Bot_Deviation`

目标变量：`Top_Residual` / `Bot_Residual`

## 3. 上表面（Top）相关性结果摘要

### (1) Pearson 线性相关性（按 |corr| 排序）
| 特征 | Pearson corr |
|:---|---:|
| Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Avg | -0.291 |
| Top_Theoretical_Factor | -0.281 |
| Top_Current_Sum | -0.249 |
| Dimension_[mm]_Width | +0.243 |
| Top_Current_Per_Speed | -0.225 |
| Dimension_[mm]_Thickness | -0.140 |
| Top_Deviation | +0.142 |
| Steel_Grade_Encoded | +0.136 |
| Speed[m/min]_Process_Avg | +0.111 |

### (2) Spearman 单调相关性（按 |corr| 排序）
| 特征 | Spearman corr |
|:---|---:|
| Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Avg | **-0.535** |
| Top_Theoretical_Factor | -0.424 |
| Top_Current_Per_Speed | -0.327 |
| Top_Current_Sum | -0.280 |
| Dimension_[mm]_Width | +0.224 |
| Steel_Grade_Encoded | +0.117 |
| Dimension_[mm]_Thickness | -0.109 |
| Top_Deviation | +0.080 |
| Speed[m/min]_Process_Avg | +0.083 |

### (3) Mutual Information（非线性依赖强度）
| 特征 | MI Score |
|:---|---:|
| Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Avg | **0.549** |
| Dimension_[mm]_Width | 0.500 |
| Top_Deviation | 0.388 |
| Top_Theoretical_Factor | 0.363 |
| Speed[m/min]_Process_Avg | 0.322 |
| Dimension_[mm]_Thickness | 0.287 |
| Top_Current_Per_Speed | 0.280 |
| Top_Current_Sum | 0.262 |
| Steel_Grade_Encoded | 0.213 |

### (4) Distance Correlation（广义依赖）
| 特征 | Distance Corr |
|:---|---:|
| Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Avg | **0.461** |
| Top_Theoretical_Factor | 0.404 |
| Top_Current_Per_Speed | 0.313 |
| Dimension_[mm]_Width | 0.289 |
| Top_Current_Sum | 0.286 |
| Top_Deviation | 0.267 |
| Steel_Grade_Encoded | 0.210 |
| Dimension_[mm]_Thickness | 0.204 |
| Speed[m/min]_Process_Avg | 0.152 |

## 4. 下表面（Bot）相关性结果摘要

### (1) Pearson 线性相关性（按 |corr| 排序）
| 特征 | Pearson corr |
|:---|---:|
| Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Avg | **-0.464** |
| Bot_Theoretical_Factor | -0.306 |
| Bot_Current_Sum | -0.277 |
| Bot_Current_Per_Speed | -0.269 |
| Steel_Grade_Encoded | +0.206 |
| Dimension_[mm]_Width | +0.169 |
| Dimension_[mm]_Thickness | -0.151 |
| Speed[m/min]_Process_Avg | +0.150 |
| Bot_Deviation | +0.135 |

### (2) Spearman 单调相关性（按 |corr| 排序）
| 特征 | Spearman corr |
|:---|---:|
| Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Avg | **-0.628** |
| Bot_Theoretical_Factor | -0.434 |
| Bot_Current_Per_Speed | -0.359 |
| Bot_Current_Sum | -0.301 |
| Steel_Grade_Encoded | +0.192 |
| Dimension_[mm]_Width | +0.165 |
| Speed[m/min]_Process_Avg | +0.128 |
| Bot_Deviation | +0.108 |
| Dimension_[mm]_Thickness | -0.088 |

### (3) Mutual Information（非线性依赖强度）
| 特征 | MI Score |
|:---|---:|
| Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Avg | **0.598** |
| Dimension_[mm]_Width | 0.489 |
| Bot_Theoretical_Factor | 0.372 |
| Bot_Deviation | 0.351 |
| Speed[m/min]_Process_Avg | 0.314 |
| Bot_Current_Per_Speed | 0.296 |
| Dimension_[mm]_Thickness | 0.291 |
| Bot_Current_Sum | 0.255 |
| Steel_Grade_Encoded | 0.216 |

### (4) Distance Correlation（广义依赖）
| 特征 | Distance Corr |
|:---|---:|
| Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Avg | **0.556** |
| Bot_Theoretical_Factor | 0.399 |
| Bot_Current_Per_Speed | 0.328 |
| Bot_Current_Sum | 0.299 |
| Bot_Deviation | 0.255 |
| Steel_Grade_Encoded | 0.251 |
| Dimension_[mm]_Thickness | 0.249 |
| Dimension_[mm]_Width | 0.220 |
| Speed[m/min]_Process_Avg | 0.150 |

## 5. 关键发现与业务解读

1. **在线测量值本身是残差最强的关联特征**  
   无论是线性（Pearson）、单调（Spearman）还是非线性（MI / Distance Correlation），`Tin Weight_Actual[...]_Avg` 均稳居第一。Spearman 相关系数达到 -0.53（Top）和 -0.63（Bot），说明在线值越高，残差越倾向于负向（即在线测量值偏高）。

2. **理论因子与电流相关特征次之**  
   `Theoretical_Factor`、`Current_Per_Speed`、`Current_Sum` 在四种度量下均表现稳定，验证了“电流–速度–宽度”物理关系对镀层偏差的解释能力。

3. **宽度与钢种具有一定正向关联**  
   宽度越大、特定钢种频率越高，残差倾向于正向（在线测量值偏低）。这一现象在上下表面均存在。

4. **非线性度量进一步确认了关键特征排序**  
   Mutual Information 与 Distance Correlation 的排序与 Spearman 高度一致，说明主要依赖关系并非单纯线性，树模型（后续步骤）具备捕捉这些关系的优势。

5. **上下表面模式高度相似**  
   特征重要性排序和相关方向在 Top / Bot 之间基本一致，后续可考虑共享特征工程或分别建模后对比。

## 6. 产出物
所有相关性矩阵热图、重要性条形图已保存至：
- `result/correlation_result/correlation_Top_pearson.png`
- `result/correlation_result/correlation_Top_spearman.png`
- `result/correlation_result/mi_importance_Top.png`
- `result/correlation_result/dcor_importance_Top.png`
- 以及对应的 Bot 表面图表

完整结果目录：`result/correlation_result/`

## 7. 小结
相关性分析清晰揭示了残差与在线测量值、理论镀层因子、电流及宽度等变量的强关联。在线测量值本身对残差具有最强解释力，这为后续“用模型预测残差 → 修正在线值”的建模策略提供了直接数据支撑。非线性度量进一步确认了树模型的适用性。