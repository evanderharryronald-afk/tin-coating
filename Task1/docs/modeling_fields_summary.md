# 镀锡建模项目字段汇总

本项目旨在通过机器学习修正镀锡板在线测量仪表的系统性偏差。建模过程涉及原始工艺数据及基于物理机理构造的衍生特征。

## 1. 原始数据字段 (Raw Fields)
直接从传感器、生产系统或实验室获取的原始参数。

| 字段名称 | 说明 | 用途 |
| :--- | :--- | :--- |
| `Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP/BOT_Avg` | 在线测厚仪实际读数（上/下表面） | 模型输入特征 |
| `Speed[m/min]_Process_Avg` | 生产线运行速度 | 模型输入特征 |
| `Dimension_[mm]_Width` | 钢带宽度 | 模型输入特征 |
| `Dimension_[mm]_Thickness` | 钢带厚度 | 模型输入特征 |
| `Tining Section_CURRENT[A]_GL_1~36_Avg` | 36个电镀槽的原始电流值 | 构造特征的基石 |
| `上/下表面镀层重量A(XA1_0)` | 实验室化验真值 | 计算预测目标 (Delta) |
| `Steel Grade` | 钢种 | 分类特征（经编码后使用） |
| `GALV_EFF_TOP/BOT` | 电镀效率 | 物理机理辅助分析 |
| `Produce Time` / `Coil ID` | 生产时间与卷号 | 数据对齐与时序分析 |

## 2. 特征工程构造字段 (Engineered Features)
基于物理机理（如法拉第定律）或数据处理手段生成的衍生特征。

| 衍生字段名称 | 构造规则 / 公式 | 物理意义 |
| :--- | :--- | :--- |
| `Top_Current_Sum` | 偶数槽电流之和 (GL_2, 4, ..., 36) | 上表面投入的总电量 |
| `Bot_Current_Sum` | 奇数槽电流之和 (GL_1, 3, ..., 35) | 下表面投入的总电量 |
| `Top/Bot_Theoretical_Factor` | `Current_Sum / (Speed * Width_m)` | 单位面积理论附着因子 |
| `Top/Bot_Current_Per_Speed` | `Current_Sum / Speed` | 单位速度下的电流贡献 |
| `Steel_Grade_Encoded` | 对 `Steel Grade` 进行频率编码或标签编码 | 钢种的数值化表示 |
| `Top/Bot_Delta` | `实验室真值 - 在线测量值` | **模型预测的目标变量** |
| `Top/Bot_Predicted_Weight_By_Theory` | `Theoretical_Factor * Efficiency` | 含效率修正的理论沉积厚度 |

## 3. 实际建模特征集 (Model Input Set)
根据代码配置，模型训练时最终选用的核心特征包括：
- **在线测量值** (`Actual`)
- **电流总和** (`Current_Sum`)
- **速度** (`Speed`)
- **宽度** (`Width`)
- **厚度** (`Thickness`)
- **理论因子** (`Theoretical_Factor`)

---
*注：字段定义参考 [constants.py](file:///e:/SGAI_Project/2026/Tin coating/task1/surface_independent_modeling/core/constants.py) 与 [feature_engineering.py](file:///e:/SGAI_Project/2026/Tin coating/task1/feature_engineering.py)。*
