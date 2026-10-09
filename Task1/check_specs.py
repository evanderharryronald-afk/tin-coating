import pandas as pd

# 读取特征工程后的数据
df = pd.read_excel('result/data/feature_engineered_data/featured_data.xlsx')

# 获取上下表面设定厚度列
top_col = 'Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_TOP_Avg'
bot_col = 'Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_BOT_Avg'

# 统计规格分布
specs = df.groupby([top_col, bot_col]).size().reset_index(name='count')
specs = specs.sort_values('count', ascending=False)

print('='*70)
print(f'总样本数: {len(df)}')
print(f'独立规格组数: {len(specs)}')
print()

print('='*70)
print('按上表面厚度分组 - 展示每个上表面厚度包含的所有规格:')
print('='*70)
for top_val in sorted(df[top_col].unique()):
    subset = specs[specs[top_col] == top_val].sort_values('count', ascending=False)
    total_count = subset['count'].sum()
    print(f'\n【上表面厚度 = {top_val}】总样本: {total_count}')
    for idx, row in subset.iterrows():
        bot_val = row[bot_col]
        count = row['count']
        print(f'    + Top{top_val}_Bot{bot_val:<6} -> {count:>6} 样本')

print('\n')
print('='*70)
print('按下表面厚度分组 - 展示每个下表面厚度包含的所有规格:')
print('='*70)
for bot_val in sorted(df[bot_col].unique()):
    subset = specs[specs[bot_col] == bot_val].sort_values('count', ascending=False)
    total_count = subset['count'].sum()
    print(f'\n【下表面厚度 = {bot_val}】总样本: {total_count}')
    for idx, row in subset.iterrows():
        top_val = row[top_col]
        count = row['count']
        print(f'    + Top{top_val:<6}_Bot{bot_val} -> {count:>6} 样本')
