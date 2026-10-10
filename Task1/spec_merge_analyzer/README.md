# spec_merge_analyzer - 镀锡规格合并可行性分析框架

## 问题定义

### 实际背景

镀锡机生产有很多不同的**镀锡规格**（Top 和 Bot 两个表面的设定值组合），每个规格需要一套**校准模型**来预测实际镀锡重量。

**当前状况**：
- 有 39 个不同的规格组合/表面档次
- 需要维护 39 套独立的校准模型
- 数据分散，某些规格样本量太少（< 200）
- 模型维护和更新成本高

### 核心目标

**判断：哪些规格的数据可以混在一起建模，用一个统一模型代替多个独立模型，而不显著降低预测精度？**

### 两种考虑方式

#### 方式 1：表面内设定值合并策略 (surface_setpoint_level)

**问题**：同一表面的**相同设定值**是否能共用一个模型？

**关键点**：**设定值相同，不管另一个表面的设定值**

```
例：Top 表面
  问：所有 Top 设定值为 2.0 的数据能否用一个模型？
      不管这些数据对应的 Bot 设定值是 1.1 还是 2.0 还是 2.2
  
  再问：Top 设定值为 2.0 和 Top 设定值为 2.2 的数据能混在一起吗？

当前：按表面设定值分档
  Top1.1（所有Bot组合）→ 1 个模型
  Top2.0（所有Bot组合）→ 1 个模型
  Top2.2（所有Bot组合）→ 1 个模型
  ... = 19 个模型

合并后：可以合并相邻的设定值档
  Top1.1（所有Bot） + Top2.0（所有Bot） → 共用 1 个模型
  Top2.2（所有Bot） + Top2.799（所有Bot） → 共用 1 个模型
  ... = 8-12 个模型
```

**特点**：
- **分析单位**：同表面 + 同设定值 = 一个单位
  - 单位内部忽略另一个表面的设定值差异
  - 例：Top2.0 单位包含所有 (Top2.0, Bot任意值) 的数据
- **比较范围**：同表面、不同设定值的单位间能否共用模型
- **样本量**：充足（每个单位通常 1000+ 样本）
- **模型数**：中等（通常 14-20 个）

#### 方式 2：完整规格组合并策略 (spec_group_cartesian)

**问题**：完整规格组合 (Top设定值, Bot设定值) 之间能否共用模型？

```
例：完整规格组
  问：(Top2.0, Bot2.0) 和 (Top2.0, Bot2.2) 能否用一个模型？
      (Top2.0, Bot2.0) 和 (Top2.2, Bot2.0) 能否用一个模型？
  
当前：每个规格组独立建模
  (Top1.1, Bot1.1) → 1 个模型
  (Top1.1, Bot2.0) → 1 个模型
  (Top2.0, Bot1.1) → 1 个模型
  ... = 51 个模型

合并后：规格组聚类建模
  {(Top1.1, Bot1.1), (Top1.1, Bot2.0)} → 共用 1 个模型
  {(Top2.0, Bot1.1), (Top2.0, Bot2.0)} → 共用 1 个模型
  ... = 20-30 个模型
```

**特点**：
- **分析单位**：完整规格组 (Top设定值 × Bot设定值)
- **比较范围**：精确到规格组合级别
- **样本量**：分散（从几十到几千）
- **模型数**：较多（通常 25-40 个）

### 成功标准

合并后的统一模型与独立模型相比，**预测误差（RMSE）增加不超过 50%**

```
RMSE变化率 = (RMSE_合并 - RMSE_分离) / RMSE_分离

判定：
  ≤ 50%   → YES，可以合并
  50-52%  → MAYBE，临界情况
  > 52%   → NO，不可合并
```

---

## 两种策略对比

| 维度 | 表面内设定值合并 | 规格组合并 |
|------|-----------------|----------|
| **分析单位** | 同表面+同设定值 | Top设定值×Bot设定值 |
| **单位例子** | Top2.0(所有Bot组合), Top2.2(所有Bot组合) | Top2.0_Bot2.0, Top2.0_Bot2.2 |
| **单位数** | 少 (~20) | 多 (~50) |
| **样本/单位** | 大 (1000+) | 分散 (50-5000) |
| **忽略什么** | 另一个表面的设定值差异 | 无（精确到规格组） |
| **聚类粒度** | 中粒度（表面级） | 细粒度（规格组级） |
| **模型减少** | 中等 (-30%) | 低 (-10%) |
| **计算复杂度** | 低 | 高 |
| **适用** | 表面独立配置 | 需要规格组精准判定 |

---

## 三阶段分析流程

```
【输入】9328 行生产数据，39 个规格单位
  ↓
【阶段 A - 初筛】KS 分布检验
  问: 哪些规格的数据分布相似？
  方法: Kolmogorov-Smirnov 检验 + 规格距离过滤
  输出: 合并候选对列表
  ↓
【阶段 B - 验证】交叉验证 + 非劣效判定
  问: 混合建模会不会损失太多精度？
  方法: 5折交叉验证，Separate vs Pooled 对比
  输出: YES/NO/MAYBE 决策
  ↓
【阶段 C - 聚类】图团划分
  问: 综合所有 YES 对，找最优分组方案？
  方法: 贪心团覆盖（最大团划分）
  输出: 推荐的分组聚类
  ↓
【输出】最终方案：X 个合并聚类 + Y 个独立单位 = Z 个模型
```

---

## 阶段 A：初筛（KS 检验）

### 逻辑

对任意两个规格单位，检查两个条件是否都满足：

1. **分布相似性**：用 KS 检验比较两个单位的 Delta（实际-设定）分布
2. **规格距离**：设定值差异是否在允许范围内

### 关键指标

#### Kolmogorov-Smirnov (KS) 检验

```
原理：比较两个样本的累积分布曲线，计算最大偏离距离

H0: 两个样本来自相同分布
H1: 两个样本来自不同分布

p-value > 0.05 ⟹ 无法拒绝 H0 ⟹ 分布相似 ✓
p-value ≤ 0.05 ⟹ 拒绝 H0 ⟹ 分布显著不同 ✗
```

#### 规格距离限制

```
绝对差: |设定值_A - 设定值_B| ≤ 2.0 g/m²
相对差: |设定值_A - 设定值_B| / min(设定值_A, 设定值_B) ≤ 50%

两个都满足 ⟹ 规格"距离限内" ✓
```

### 输出

- `merge_candidates.csv` - 候选对列表
  - 示例：36 对对（按表面），51 对对（按规格组）

---

## 阶段 B：验证（交叉验证 + 非劣效判定）

### 核心逻辑

对每个候选对（规格 A 和 B），用 **5 折交叉验证** 对比两种建模方案：

#### 方案对比

```python
分离建模 (Separate):
  ├─ 数据：A 的训练集 → 建模型 A
  ├─ 数据：B 的训练集 → 建模型 B
  ├─ 预测：测试集 A 用模型 A，测试集 B 用模型 B
  └─ 结果：RMSE_separate

合并建模 (Pooled):
  ├─ 数据：(A + B) 混合训练集 → 建一个统一模型
  ├─ 预测：所有测试集都用这个统一模型
  └─ 结果：RMSE_pooled

判定：
  change% = (RMSE_pooled - RMSE_separate) / RMSE_separate
  
  if change% ≤ 50%  → YES (精度损失在容限内)
  if 50% < change% ≤ 52%  → MAYBE
  if change% > 52%  → NO (精度下降太多)
```

### 交叉验证过程

```
Fold 1:  训练集={1,2,3,4}, 测试集={5}
Fold 2:  训练集={1,2,3,5}, 测试集={4}
...
Fold 5:  训练集={2,3,4,5}, 测试集={1}

对每个 fold：
  分离方案：各建各的模型
  合并方案：混合数据建一个模型
  
最后：平均 5 个 fold 的 RMSE
```

### 关键指标

#### RMSE 变化率

```
change_pct = (RMSE_pooled - RMSE_separate) / RMSE_separate × 100%

例：
  RMSE_separate = 0.050
  RMSE_pooled = 0.053
  change_pct = (0.053 - 0.050) / 0.050 × 100% = 6%
  
  判定：6% ≤ 50% ⟹ YES ✓
```

#### 非劣效边界

配置参数：`non_inferiority.summary_margin`（默认 0.50 = 50%）

```
判定规则：
  change_pct ≤ 50%     → YES
  50% < change_pct ≤ 52%  → MAYBE
  change_pct > 52%     → NO
```

### 输出

`merge_validation_results.json` 示例：

```json
{
  "unit_A": "Top2.0",
  "unit_B": "Top2.2",
  "surface": "Top",
  "final_decision": "YES",
  "cv": {
    "rmse_separate": 0.0525,
    "rmse_pooled": 0.0611,
    "change_pct": 16.19,
    "n_folds": 2
  },
  "rationale": "汇总 RMSE 增长 16.19% ≤ 50.0% 边界"
}
```

**含义**：Top2.0 和 Top2.2 混合建模，RMSE 只增加 16%，远低于 50% 容限 → YES

---

## 阶段 C：聚类（图团划分）

### 逻辑

整合所有 YES 对，找**最优的分组方案**。

#### 第一步：构建图

```
节点：所有规格单位
边：Phase B 判定为 YES 的单位对

例：按表面策略
  节点：Top1.1, Top2.0, Top2.2, Top2.799, ...
  边：(Top1.1, Top2.0)=YES, (Top2.0, Top2.2)=YES, ...
```

#### 第二步：团划分

**团** = 完全连通子图 = 任意两个节点都有 YES 对相连

```
如果节点 {A, B, C} 形成一个团，意味着：
  (A, B) = YES ✓
  (B, C) = YES ✓
  (A, C) = YES ✓
  
那么这 3 个单位可以用一个统一模型
```

#### 第三步：贪心覆盖

算法：反复找最大团，移除其节点，直到无节点剩余

```
while 还有节点:
  找剩余节点中最大的团
  把这个团作为一个聚类
  移除这个团的所有节点
```

### 示例

**输入**（Phase B 的 YES 对）：
```
(A, B)=YES, (B, C)=YES, (A, C)=YES
(D, E)=YES
(F) 孤立（没有 YES 对）
```

**构建的图**：
```
    A ─── B
     ╲   ╱
       C

D ─── E

F (孤立)
```

**团划分结果**：
```
聚类 1: {A, B, C} → 用 1 个模型
聚类 2: {D, E}   → 用 1 个模型
聚类 3: {F}      → 独立模型

总模型数：3（vs 6 个独立单位）
```

### 连通性验证

**聚类内的任意两个单位都必须有 YES 对**（完全图性质）

```
验证聚类 {A, B, C}：
  对(A, B)  Phase B 结果  YES ✓
  对(B, C)  Phase B 结果  YES ✓
  对(A, C)  Phase B 结果  YES ✓
  
所有对都是 YES ⟹ 三个单位都能共用一个模型 ✓
```

### 输出

`recommended_grouping.json` 结构：

```json
{
  "surfaces": {
    "Top": {
      "merged_clusters": [
        {
          "members": ["Top1.1", "Top2.0", "Top2.2", "Top2.799"],
          "total_samples": 8995
        }
      ],
      "independent_units": [
        {"members": ["Top5.0"], "total_samples": 66}
      ],
      "total_models": 6
    },
    "Bot": {
      "merged_clusters": [...],
      "independent_units": [...],
      "total_models": 5
    }
  },
  "summary": {
    "total_models": 11,
    "merged_clusters": 2,
    "independent_units": 1
  }
}
```

---

## 使用方法

### 1. 配置选择

编辑 `grouping_config.yaml`：

```yaml
grouping_strategies:
  # 选择要分析的策略（可同时启用两个）
  surface_setpoint_level:
    enabled: true          # 分析同表面不同设定值能否合并
  
  spec_group_cartesian:
    enabled: true          # 分析完整规格组能否合并

# 非劣效边界（核心参数）
non_inferiority:
  summary_margin: 0.50     # 50% 的 RMSE 增长容限
```

### 2. 运行

```bash
python merge_analysis_pipeline.py
```

### 3. 查看结果

```
result/spec_group_merge_analysis/
├── surface_setpoint_level/
│   ├── merge_candidates.csv              # 候选对
│   ├── merge_validation_results.json     # 验证结果
│   └── recommended_grouping.json         # 最终推荐
│
└── spec_group_cartesian/
    ├── merge_candidates.csv
    ├── merge_validation_results.json
    └── recommended_grouping.json
```

---

## 结果解读示例

### 按表面策略的结果

```json
{
  "Top": {
    "merged_clusters": [
      {
        "members": ["Top1.1", "Top2.0", "Top2.2", "Top2.799"],
        "count": 4,
        "total_samples": 8995,
        "meets_formal_threshold": true,
        "sample_density": 2248.75
      }
    ],
    "independent_units": [
      {"members": ["Top5.0"], "total_samples": 66}
    ],
    "total_models": 6
  }
}
```

**解读**：

- ✅ Top 表面可以分成：
  - **1 个合并聚类**：Top {1.1, 2.0, 2.2, 2.799} 用 **1 个模型**
    - 这 4 个设定值的 YES 对形成完全连通
    - 共包含 8995 个样本 → 样本充足
  - **1 个独立单位**：Top 5.0 用 **1 个模型**
    - 只有 66 个样本 < 200 门槛，无法合并

- 结论：Top 表面从 19 个单位的 19 个模型 → 2 个聚类的 6 个模型？

### 按规格组策略的结果

```json
{
  "merged_clusters": [
    {"members": ["Top2.0_Bot2.0", "Top2.0_Bot2.2"], "total_samples": 5000}
  ],
  "independent_units": [
    {"members": ["Top1.1_Bot1.1"], "total_samples": 150}
  ]
}
```

**解读**：

- ✅ (Top2.0, Bot2.0) 和 (Top2.0, Bot2.2) 可以用 **1 个统一模型**
- ⚠️ (Top1.1, Bot1.1) 等其他规格组需要独立模型

---

## 关键参数

### 核心参数（grouping_config.yaml）

```yaml
# 规格距离限制
spec_distance:
  absolute_limit: 2.0           # 绝对差 ≤ 2.0 g/m²
  relative_limit: 0.5           # 相对差 ≤ 50%

# 非劣效判定
non_inferiority:
  summary_margin: 0.50          # 50% RMSE 增长容限

# 交叉验证
cross_validation:
  cv_splits: 3                  # 3 折
  cv_repeats: 2                 # 重复 2 次

# 模型
model:
  type: "HistGradientBoosting"
  max_depth: 3
  learning_rate: 0.05
```

### 调参建议

| 参数 | 作用 | 调大 | 调小 |
|------|------|------|------|
| `spec_distance.absolute_limit` | 允许的设定值差异 | 更激进的合并 | 更保守 |
| `non_inferiority.summary_margin` | RMSE 增长容限 | 容许更多精度损失 | 要求更高精度 |
| `min_formal_samples` | 正式建模的样本量门槛 | 更多单位参与 | 更严格筛选 |

---

## 常见问题

**Q: 为什么有些单位无法合并？**

A: 可能原因：
1. KS 检验未通过 → 分布差异大
2. 规格距离超限 → 设定值差太大
3. RMSE 增长超过 50% → 精度下降太多

**Q: 如何理解 "合并成 6 个模型"？**

A: 表示最终需要 6 个模型来覆盖该表面的所有规格：
- 1 个是合并聚类（代表多个规格）
- 5 个是独立单位（各自一个模型）

**Q: MAYBE 怎么处理？**

A: MAYBE 表示在临界边界（50-52% 增长），可以：
- 调整 `summary_margin` 参数重新判定
- 或保守地视为 NO，不合并

**Q: 两种策略的结果会不同吗？**

A: 是的，很可能不同：
- 按表面：关注表面内的聚类
- 按规格组：关注完整规格的聚类
- 选择取决于业务需求

---

## 文件导航

```
spec_merge_analyzer/
├── README.md                          # 本文档
├── __init__.py                        # 主入口，管道编排
├── config_manager.py                  # 配置管理
├── grouping_strategies.py             # 分组策略定义
├── phase_a_ks_test_screening.py       # 阶段 A：初筛
├── phase_b_chow_test_validation.py    # 阶段 B：验证
└── phase_c_graph_grouping.py          # 阶段 C：聚类
```

---

**版本**: 3.0  
**更新**: 2026-10-10  
**问题定位**: 镀锡规格合并可行性判定
