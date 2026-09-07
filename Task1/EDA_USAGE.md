# GroupEDADiagnoser 使用说明

## 快速开始

### 1. 只分析整体数据（推荐）
```python
from group_eda_manager import create_eda_diagnoser_from_config
import json

# 使用 eda_config.json（enable_overall=true, enable_by_group=false）
with open('eda_config.json', 'r', encoding='utf-8') as f:
    eda_config = json.load(f)

eda_mgr = create_eda_diagnoser_from_config(eda_config)

# 建模前诊断：只分析全量数据
pre_results = eda_mgr.pre_modeling_diagnosis(group_df, group_label='analysis')

# 训练模型...
models = {'Top': model_top, 'Bot': model_bot}

# 建模后诊断：只分析全量数据
post_results = eda_mgr.post_modeling_diagnosis(group_df, models, group_label='analysis')
```

### 2. 只分析分规格数据（所有规格）
```python
# 使用 eda_config_with_group.json
with open('eda_config_with_group.json', 'r', encoding='utf-8') as f:
    eda_config = json.load(f)

eda_mgr = create_eda_diagnoser_from_config(eda_config)

# 诊断会对每个规格组分别分析
pre_results = eda_mgr.pre_modeling_diagnosis(group_df, group_label='analysis')
```

### 3. 只分析特定规格
```python
# 使用 eda_config_specific_group.json
with open('eda_config_specific_group.json', 'r', encoding='utf-8') as f:
    eda_config = json.load(f)

eda_mgr = create_eda_diagnoser_from_config(eda_config)

# 只会分析 Top2.799_Bot2.799 这个规格
pre_results = eda_mgr.pre_modeling_diagnosis(group_df, group_label='analysis')
```

### 4. 同时分析整体和分规格
```python
from group_eda_manager import GroupEDADiagnoser, SurfaceEDAConfig

config = {
    'Top': SurfaceEDAConfig(
        'Top',
        enable_overall=True,           # 分析全量
        enable_by_group=True,          # 也分析分规格
        group_col='Setpoint_Group_Label',
        target_groups=None  # 分析所有规格
    ),
    'Bot': SurfaceEDAConfig(
        'Bot',
        enable_overall=True,
        enable_by_group=True,
        group_col='Setpoint_Group_Label',
        target_groups=None
    )
}

eda_mgr = GroupEDADiagnoser(surfaces=['Top', 'Bot'], surface_configs=config)
pre_results = eda_mgr.pre_modeling_diagnosis(group_df, 'analysis')
```

### 5. 运行时覆盖配置
```python
eda_mgr = create_eda_diagnoser_from_config(eda_config)

# 运行时改为只分析特定规格
pre_results = eda_mgr.pre_modeling_diagnosis(
    group_df,
    group_label='analysis',
    enable_overall=False,
    enable_by_group=True,
    group_col='Setpoint_Group_Label',
    target_groups=['Top2.799_Bot2.799', 'Top2.8_Bot2.8']
)
```

## 配置文件说明

### eda_config.json（只分析整体）
```json
{
  "pre_diagnosis_dir": "result/eda/pre_modeling",
  "post_diagnosis_dir": "result/eda/post_modeling",
  "eda_top": {
    "enabled": true,
    "enable_overall": true,
    "enable_by_group": false,
    "group_col": null,
    "target_groups": null
  }
}
```

### eda_config_with_group.json（分析所有规格）
```json
{
  "eda_top": {
    "enabled": true,
    "enable_overall": false,
    "enable_by_group": true,
    "group_col": "Setpoint_Group_Label",
    "target_groups": null
  }
}
```

### eda_config_specific_group.json（只分析指定规格）
```json
{
  "eda_top": {
    "enabled": true,
    "enable_overall": false,
    "enable_by_group": true,
    "group_col": "Setpoint_Group_Label",
    "target_groups": ["Top2.799_Bot2.799"]
  }
}
```

### eda_config_fast_tuning.json（快速调参）
```json
{
  "eda_top": {
    "enabled": true,
    "enable_overall": true,
    "enable_by_group": false,
    "compute_stats_only_pre": true,
    "compute_stats_only_post": true
  }
}
```

### 配置参数说明

| 参数 | 说明 |
|------|------|
| enabled | 是否启用该表面的诊断 |
| enable_overall | 是否执行整体分析（对全量数据）|
| enable_by_group | 是否执行分规格分析 |
| group_col | 分规格列名（仅当enable_by_group=true时有效） |
| target_groups | 规格白名单（仅当enable_by_group=true时有效，null表示所有规格） |
| compute_stats_only_pre | 建模前是否仅计算统计（True=快速，False=画图详细） |
| compute_stats_only_post | 建模后是否仅计算统计 |
| max_groups | 分规格分析时最多分析的组数 |
| sample_for_scatter | 散点图采样数（null=不采样） |

## 主要方法

### pre_modeling_diagnosis
建模前诊断：数据分布、特征质量、异常检测
```python
results = eda_mgr.pre_modeling_diagnosis(
    df=group_df,
    group_label='analysis',
    surfaces=['Top', 'Bot'],
    enable_overall=True,                # 整体分析开关
    enable_by_group=True,               # 分规格分析开关
    group_col='Setpoint_Group_Label',   # 分规格列名
    target_groups=['Top2.799_Bot2.799']  # 规格白名单
)
```

**参数优先级：** 运行时参数 > 配置参数

### post_modeling_diagnosis
建模后诊断：残差质量、方向性偏差、校正效果
```python
results = eda_mgr.post_modeling_diagnosis(
    df=group_df,
    models={'Top': model_top, 'Bot': model_bot},
    group_label='analysis',
    surfaces=['Top', 'Bot'],
    enable_overall=True,
    enable_by_group=True,
    group_col='Setpoint_Group_Label',
    target_groups=None  # 分析所有规格
)
```

### 动态控制

```python
# 查看启用的表面
active = eda_mgr.get_active_surfaces()

# 禁用Bot
eda_mgr.disable_surface('Bot')

# 启用Top
eda_mgr.enable_surface('Top')

# 更新配置
new_config = SurfaceEDAConfig(
    'Top',
    enable_overall=False,
    enable_by_group=True
)
eda_mgr.set_surface_config('Top', new_config)
```

## 保存目录结构

### 纯整体分析（Overall Only）
```
pre_diagnosis_dir/
  overall/
    Top/
      eda_summary.xlsx
      delta_dist.png
      ... (全量Top表面的整体分析结果)
    Bot/
      eda_summary.xlsx
      ... (全量Bot表面的整体分析结果)
```

### 纯分规格分析（By Group Only）
```
pre_diagnosis_dir/
  by_group/
    {group_label}/
      Top/
        Top2.799_Bot2.799/
          eda_summary.xlsx
          delta_dist.png
          ...
        Top2.8_Bot2.8/
          ...
      Bot/
        Top2.799_Bot2.799/
          ...
        ...
```

### 同时分析（Overall + By Group）
```
pre_diagnosis_dir/
  overall/
    Top/
      ... (整体分析)
    Bot/
      ...
  by_group/
    {group_label}/
      Top/
        规格1/
          ...
        规格2/
          ...
      Bot/
        ...
```

**目录层级关键点：**
- `overall/` — 对全量数据的整体分析，不分规格组，直接在表面下放结果
- `by_group/` — 对分规格数据的分析，按group_label → surface → 各规格组织
- 两个模式互不影响，可独立启用或同时启用
- 只有enable_by_group=true且target_groups指定的规格，才会在by_group下创建对应目录

## 常见场景

### 场景1：只分析全量数据（整体模式）
```python
config = {
    'Top': SurfaceEDAConfig('Top', enable_overall=True, enable_by_group=False),
    'Bot': SurfaceEDAConfig('Bot', enable_overall=True, enable_by_group=False)
}
eda_mgr = GroupEDADiagnoser(surfaces=['Top', 'Bot'], surface_configs=config)

pre_results = eda_mgr.pre_modeling_diagnosis(group_df, 'analysis')
# 只会生成 result/eda/pre_modeling/overall/Top/ 等
```

### 场景2：只分析分规格数据（分组模式）
```python
config = {
    'Top': SurfaceEDAConfig(
        'Top',
        enable_overall=False,
        enable_by_group=True,
        group_col='Setpoint_Group_Label'
    ),
    'Bot': SurfaceEDAConfig(
        'Bot',
        enable_overall=False,
        enable_by_group=True,
        group_col='Setpoint_Group_Label'
    )
}
eda_mgr = GroupEDADiagnoser(surfaces=['Top', 'Bot'], surface_configs=config)

pre_results = eda_mgr.pre_modeling_diagnosis(group_df, 'analysis')
# 只会生成 result/eda/pre_modeling/by_group/analysis/Top/规格1/ 等
```

### 场景3：同时分析整体和分规格
```python
config = {
    'Top': SurfaceEDAConfig(
        'Top',
        enable_overall=True,
        enable_by_group=True,
        group_col='Setpoint_Group_Label'
    )
}
eda_mgr = GroupEDADiagnoser(surfaces=['Top'], surface_configs=config)

pre_results = eda_mgr.pre_modeling_diagnosis(group_df, 'analysis')
# 会同时生成 overall/ 和 by_group/ 两个目录树
```

### 场景4：关闭所有诊断（快速训练）
```python
eda_mgr.disable_surface('Top')
eda_mgr.disable_surface('Bot')
# 调用诊断时会自动跳过
```

### 场景5：提取summary用于报表
```python
pre_results = eda_mgr.pre_modeling_diagnosis(group_df, 'analysis')
summaries = eda_mgr.get_summary_dataframes(pre_results)
# summaries['Top'] 可直接导出Excel
```
