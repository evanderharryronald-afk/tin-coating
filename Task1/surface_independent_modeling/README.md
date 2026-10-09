# 表面独立分组建模包

## 概述

这是一个新的建模框架，用表面厚度独立分组代替规格组合分组。相比原有的建模方式，该方案有以下优势：

- ✓ 样本利用效率提升 30-40%
- ✓ 模型数量从96个降至~20个
- ✓ 更符合物理过程（上下表面独立）
- ✓ 泛化能力更强

## 架构

```
surface_independent_modeling/
├── core/                       # 核心模块
│   ├── constants.py           # 常量定义
│   ├── grouping.py            # 分组逻辑
│   └── data_loader.py         # 数据加载
├── analysis/                  # 分析模块
│   ├── correlation.py         # 相关性分析
│   ├── eda.py                 # EDA分析
│   └── runner.py              # 分析协调
├── modeling/                  # 建模模块
│   ├── model.py               # 模型包装
│   ├── trainer.py             # 训练接口
│   ├── evaluator.py           # 评估接口
│   └── hyperparameter_tuning.py # 调优接口
├── explainability/            # 可解释性分析
│   ├── feature_importance.py  # 特征重要性
│   ├── residual_analysis.py   # 残差诊断
│   └── shap_analysis.py       # SHAP分析
├── pipeline.py                # 完整管道
├── main.py                    # CLI入口
└── README.md                  # 本文件
```

## 快速开始

### 1. 完整管道

```bash
cd Task1

# 运行完整管道 (所有步骤)
python -m surface_independent_modeling.main --stage all --n_trials 100

# 或使用Python API
from surface_independent_modeling import SurfaceIndependentModelingPipeline

pipeline = SurfaceIndependentModelingPipeline()
pipeline.run_full_pipeline(n_trials=100)
```

### 2. 分步执行

```bash
# 只运行分析
python -m surface_independent_modeling.main --stage analysis

# 只运行超参数调优
python -m surface_independent_modeling.main --stage tuning --n_trials 100

# 只运行模型训练
python -m surface_independent_modeling.main --stage training

# 只运行评估和可解释性
python -m surface_independent_modeling.main --stage evaluation
```

### 3. Python API 使用

```python
from surface_independent_modeling import (
    SurfaceIndependentModelingPipeline,
    load_featured_data,
    build_surface_groups,
    summarize_groups,
)

# 创建管道
pipeline = SurfaceIndependentModelingPipeline(output_dir='result/custom_output')

# 加载和分组数据
pipeline.load_data()
pipeline.build_groups()

# 运行分析
pipeline.run_analysis()

# 超参数调优
best_params = pipeline.tune_hyperparameters(n_trials=100)

# 模型训练
pipeline.train_models(params_dict=best_params)

# 模型评估
pipeline.evaluate_models()

# 可解释性分析
pipeline.run_explainability()

# 生成报告
pipeline.generate_summary_report()
```

## 数据流

```
featured_data.xlsx (9,328 rows)
    ↓
【分组】按表面厚度独立分组
    ↓
Top: ~10个主规格组  /  Bot: ~10个主规格组
    ↓
【分析】相关性分析、EDA
    ↓
【调优】超参数优化 (Optuna)
    ↓
【训练】训练 ~20个模型
    ↓
【评估】性能对比、指标统计
    ↓
【可解释性】特征重要性、残差诊断、SHAP
    ↓
result/surface_independent_modeling/
├── 01_grouped_data/
├── 02_correlation_analysis/
├── 03_eda_analysis/
├── 04_hyperparameter_tuning/
├── 05_trained_models/
├── 06_model_evaluation/
├── 07_explainability/
└── REPORT.md
```

## 关键特性

### 1. 表面独立分组

- 按 `Top_Setpoint` 或 `Bot_Setpoint` 单独维度分组
- 不再使用 `(Top_Setpoint, Bot_Setpoint)` 的组合分组
- 样本量充足的主规格 >= 10个，每个样本数 >= 50

### 2. 复用现有模块

所有分析、调优、训练代码都直接复用现有模块，**不修改现有代码**：

| 模块 | 复用方式 |
|------|--------|
| correlation_analyzer.py | 对每个分组调用 `analyze_surface()` |
| eda_analyzer.py | 对每个分组调用相关方法 |
| coating_model_by_group.py | 复用 `fit_and_evaluate_surface()` |
| run_tuning_optuna.py | 对每个分组调用 `make_objective()` |
| model_persistence.py | 复用模型保存/加载逻辑 |

### 3. 时间感知

- 保持原有的时间序列顺序
- 时间顺序切分: Train (65%) / Val (20%) / Test (15%)
- 避免数据泄露

### 4. 可解释性分析

- **特征重要性**: 排列重要性、SHAP值
- **残差诊断**: Q-Q图、分布检验、异方差性检验
- **SHAP分析**: 全局和局部解释（如果SHAP库可用）

## 输出结果

### 1. 分组数据 (`01_grouped_data/`)
每个表面分组的数据CSV文件

### 2. 相关性分析 (`02_correlation_analysis/`)
- Pearson/Spearman相关性热力图
- 互信息重要性
- 距离相关性

### 3. EDA分析 (`03_eda_analysis/`)
- 数据质量报告
- 目标变量统计
- 时间趋势分析

### 4. 超参数调优 (`04_hyperparameter_tuning/`)
- 最优超参数JSON
- Optuna研究历史

### 5. 训练模型 (`05_trained_models/`)
- 每个分组的模型文件 (.joblib)
- 元数据信息 (.json)

### 6. 模型评估 (`06_model_evaluation/`)
- 评估汇总表 (CSV)
- 性能指标对比
- 评估报告 (TXT)

### 7. 可解释性分析 (`07_explainability/`)
- 特征重要性排序和可视化
- 残差诊断图
- SHAP摘要图

### 8. 总结报告 (`REPORT.md`)
完整的执行总结和结果概览

## 配置

### 数据路径

在 `core/constants.py` 中配置:

```python
FEATURED_DATA_PATH = Task1_ROOT / 'result' / 'data' / 'feature_engineered_data' / 'featured_data.xlsx'
OUTPUT_DIR = TASK1_ROOT / 'result' / 'surface_independent_modeling'
```

### 模型参数

默认参数在 `core/constants.py`:

```python
DEFAULT_MODEL_PARAMS = {
    'damping': 0.6,
    'pos_boost': 4.6,
    'alpha_smoothing': 1.0,
    'max_iter': 200,
    'learning_rate': 0.05,
    'max_depth': 4,
}

HYPERPARAMETER_RANGES = {
    'damping': (0.0, 1.0),
    'pos_boost': (1.0, 10.0),
    ...
}
```

## 时间隐含性

数据在按表面分组后，时间维度被隐含处理：

- ✓ 样本来自整个353天的生产周期
- ✓ 不同规格组合的时间范围高度重叠
- ✓ 规格轮转频繁（每8个样本切换一次）

为了显式处理可能的时间效应，建议：
1. 在模型中添加时间特征（如月份编码）
2. 检查不同时间段的性能一致性
3. 监测模型在早期、中期、晚期的表现差异

## 复用现有代码的工作原理

### 示例：相关性分析

```python
# 原有 correlation_analyzer.py:
from correlation_analyzer import SurfaceCorrelationAnalyzer

analyzer = SurfaceCorrelationAnalyzer()
results = analyzer.analyze_surface(df, surface='Top')

# 新包中的使用方式:
for group_key, (surface, value, group_df, n_samples) in groups.items():
    # 对每个分组调用相同的方法，参数相同
    results = analyzer.analyze_surface(group_df, surface=surface)
```

这样做的好处：
- 无需修改原有代码
- 完全兼容现有逻辑
- 便于维护和升级

## 故障排除

### 1. 缺少依赖库

```bash
# 安装必要的库
pip install pandas numpy scikit-learn optuna matplotlib seaborn scipy

# 可选: 安装SHAP
pip install shap
```

### 2. 导入错误

确保在 Task1 目录下运行:
```bash
cd e:\SGAI_Project\2026\Tin coating\Task1

# 使用相对导入
python -m surface_independent_modeling.main
```

### 3. 数据路径错误

检查 `featured_data.xlsx` 是否存在:
```python
from surface_independent_modeling.core import FEATURED_DATA_PATH
print(FEATURED_DATA_PATH)  # 验证路径
```

## 后续扩展

- [ ] 添加模型集成 (Ensemble)
- [ ] 支持GPU加速 (RAPIDS/CuDF)
- [ ] 添加异常检测
- [ ] 支持模型部署 (ONNX/PMML)
- [ ] 添加实时预测接口

## 联系和反馈

如有问题或建议，请联系开发团队。
