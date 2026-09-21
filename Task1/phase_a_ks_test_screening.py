"""
【阶段 A：规格组合并可行性初步筛选】

使用 KS 检验快速判断规格组间的测量误差分布是否相同。
这是判断两个规格组是否能合并建模的必要条件。

步骤：
1. 加载特征工程后的数据
2. 按规格组分类
3. 计算每个规格组的 Delta（实际 - 在线）分布
4. 对所有规格组对进行 KS 检验
5. 输出可能合并的候选对（按优先级排序）
"""

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import itertools
import os
import yaml

# ===== 从 coating_model_by_group.py 复用的常量 =====
TOP_SETPOINT_COL = 'Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_TOP_Min'
BOT_SETPOINT_COL = 'Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_BOT_Min'

# 配置
MIN_GROUP_SAMPLES = 100  # 规格组最小样本量
KS_P_THRESHOLD = 0.05   # KS 检验 p-value 阈值（> 0.05 说明分布相同）


def load_config(config_path="config.yaml"):
    """加载配置文件"""
    if not os.path.exists(config_path):
        print(f"⚠ 警告: 配置文件 {config_path} 不存在，使用默认数据路径")
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config or {}


def load_featured_data(config_path="config.yaml") -> pd.DataFrame:
    """加载特征工程后的数据"""
    print("\n" + "="*70)
    print("【阶段 A：规格组合并初筛】")
    print("="*70)
    
    print("\n[步骤 1] 加载特征工程数据...")
    
    # 加载配置
    config = load_config(config_path)
    
    # 获取数据路径（与 coating_model_by_group.py 一致）
    excel_path = config.get("data_paths", {}).get("clean_data", "result/data/feature_engineered_data/featured_data.xlsx")
    
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"未找到特征工程数据: {excel_path}\n请先运行 new_data_pipeline.py 或 analyse_data_final.py")
    
    df = pd.read_excel(excel_path)
    print(f"✓ 数据加载: {len(df)} 行, {df.shape[1]} 列")
    print(f"  路径: {excel_path}")
    
    return df


def build_spec_groups(df) -> pd.DataFrame:
    """构建规格组标签（复用 coating_model_by_group.py 的逻辑）"""
    df = df.copy()
    
    if TOP_SETPOINT_COL not in df.columns or BOT_SETPOINT_COL not in df.columns:
        raise KeyError(f"缺少分组所需字段: {TOP_SETPOINT_COL} 或 {BOT_SETPOINT_COL}")
    
    df['Setpoint_Group_Key'] = list(zip(df[TOP_SETPOINT_COL], df[BOT_SETPOINT_COL]))
    df['Setpoint_Group_Label'] = df.apply(
        lambda r: f"Top{r[TOP_SETPOINT_COL]}_Bot{r[BOT_SETPOINT_COL]}", axis=1
    )
    return df


def compute_delta_columns(df) -> pd.DataFrame:
    """
    检查 Delta 列是否存在（应该在特征工程时已生成）
    Delta 列名: Top_Delta, Bot_Delta（不是 Delta_Top, Delta_Bot）
    """
    df = df.copy()
    
    for surface in ['Top', 'Bot']:
        delta_col = f'{surface}_Delta'  # 正确的列名格式
        
        if delta_col not in df.columns:
            print(f"⚠ 警告: 缺少 {surface} 表面的列 {delta_col}")
            print(f"  这个列应该在特征工程时生成")
    
    return df


def summarize_groups(df) -> pd.DataFrame:
    """打印规格组统计信息"""
    print("\n[步骤 4] 规格组统计...")
    
    group_sizes = df.groupby('Setpoint_Group_Label').size().sort_values(ascending=False)
    
    valid_groups = group_sizes[group_sizes >= MIN_GROUP_SAMPLES]
    skipped_groups = group_sizes[group_sizes < MIN_GROUP_SAMPLES]
    
    print(f"\n  共 {len(group_sizes)} 个规格组")
    print(f"  ✓ 达标（≥{MIN_GROUP_SAMPLES}）: {len(valid_groups)} 个")
    print(f"  ✗ 跳过（<{MIN_GROUP_SAMPLES}）: {len(skipped_groups)} 个\n")
    
    print(f"  {'规格组':<25} {'样本数':>8} 状态")
    print("  " + "-"*50)
    for label, size in valid_groups.items():
        print(f"  {label:<25} {size:>8} ✓ 建模")
    for label, size in skipped_groups.items():
        print(f"  {label:<25} {size:>8} ✗ 跳过")
    
    return df[df['Setpoint_Group_Label'].isin(valid_groups.index)].copy()


def perform_ks_tests(df) -> list:
    """
    对所有规格组对进行 KS 检验（改进版）
    
    改进点：
    1. KS 检验作为参考而非硬性过滤
    2. 保留所有候选对（包括边界情况）
    3. 添加详细的分布诊断信息
    4. 记录不通过理由（用于后续分析）
    
    返回候选合并对列表
    """
    print("\n[步骤 5] KS 检验测量误差分布...")
    
    group_labels = sorted(df['Setpoint_Group_Label'].unique())
    candidates = []
    
    # 对每个表面（Top/Bot）分别进行检验
    for surface in ['Top', 'Bot']:
        delta_col = f'{surface}_Delta'  # 正确的列名格式
        
        if delta_col not in df.columns:
            print(f"⚠ 跳过 {surface} 表面（缺少 {delta_col} 列）")
            continue
        
        print(f"\n  ──────── {surface} 表面 ────────")
        
        for group_A, group_B in itertools.combinations(group_labels, 2):
            delta_A = df[df['Setpoint_Group_Label'] == group_A][delta_col].dropna()
            delta_B = df[df['Setpoint_Group_Label'] == group_B][delta_col].dropna()
            
            if len(delta_A) < 10 or len(delta_B) < 10:
                continue  # 样本太少，跳过
            
            # 执行 KS 检验
            stat, p_value = ks_2samp(delta_A, delta_B)
            
            # 判断 KS 检验是否通过（p > 0.05）
            ks_pass = p_value > KS_P_THRESHOLD
            
            # 计算分布相似度指标
            mean_diff = abs(delta_A.mean() - delta_B.mean())
            std_diff = abs(delta_A.std() - delta_B.std())
            
            # 构建候选对记录
            candidate = {
                'surface': surface,
                'group_A': group_A,
                'group_B': group_B,
                'p_value': p_value,
                'stat': stat,
                'ks_pass': ks_pass,
                'size_A': len(delta_A),
                'size_B': len(delta_B),
                'mean_A': delta_A.mean(),
                'mean_B': delta_B.mean(),
                'mean_diff': mean_diff,
                'std_A': delta_A.std(),
                'std_B': delta_B.std(),
                'std_diff': std_diff,
                'merged_size': len(delta_A) + len(delta_B),
                'ks_interpretation': f"{'✓ 分布相同' if ks_pass else '✗ 分布不同'} (p={p_value:.4f})"
            }
            
            # 改进规则：保留所有候选对，而不是只保留通过的
            # 这样可以在 Phase B 中更灵活地处理
            candidates.append(candidate)
    
    # 按优先级排序：
    # 1. 优先保留 KS 通过的对
    # 2. 其次按小样本数量排序（小样本更容易受噪声影响，合并的收益更大）
    candidates = sorted(
        candidates, 
        key=lambda x: (not x['ks_pass'], -min(x['size_A'], x['size_B']))
    )
    
    return candidates


def print_ks_results(candidates: list):
    """
    打印 KS 检验结果（改进版）
    
    说明：
    - ✓ 标记: KS 通过（分布相同），优先级高
    - ○ 标记: KS 边界或略微不同，可作为备选
    """
    print("\n" + "="*100)
    print("【KS 检验结果 - 规格组合并候选对分析】")
    print("="*100)
    
    if not candidates:
        print("\n✗ 未发现可能的合并候选对")
        return
    
    # 分类显示
    ks_pass = [c for c in candidates if c['ks_pass']]
    ks_fail = [c for c in candidates if not c['ks_pass']]
    
    print(f"\n总计 {len(candidates)} 个候选对")
    print(f"  ✓ KS 通过（分布相同）: {len(ks_pass)} 对")
    print(f"  ○ KS 未通过（分布略异）: {len(ks_fail)} 对")
    
    # 显示 KS 通过的对
    if ks_pass:
        print("\n" + "-"*100)
        print("【第一优先级：KS 分布检验通过】")
        print("-"*100)
        for i, cand in enumerate(ks_pass, 1):
            print(f"\n{i}. {cand['surface']} 表面: {cand['group_A']} + {cand['group_B']}")
            print(f"   样本数: {cand['size_A']:>4} + {cand['size_B']:>4} = {cand['merged_size']}")
            print(f"   均值: {cand['mean_A']:>8.4f} vs {cand['mean_B']:>8.4f}  (差异: {cand['mean_diff']:.4f})")
            print(f"   标准差: {cand['std_A']:>8.4f} vs {cand['std_B']:>8.4f}  (差异: {cand['std_diff']:.4f})")
            print(f"   KS 检验: p={cand['p_value']:.6f} ✓ ({cand['ks_interpretation']})")
            print(f"   含义: 两个规格组的测量误差分布统计上相同（可能可以合并）")
    
    # 显示 KS 未通过的对（如果有）
    if ks_fail:
        print("\n" + "-"*100)
        print("【第二优先级：KS 分布检验未通过（分布略有不同）】")
        print("-"*100)
        print("\n说明: 这些对虽然 KS 检验未通过，但仍可作为备选。")
        print("原因：KS 检验只检查边际分布，不检查条件分布。即使分布略异，")
        print("如果校准规律相同（Chow 检验），仍可考虑合并。")
        print()
        
        for i, cand in enumerate(ks_fail, 1):
            print(f"{i}. {cand['surface']} 表面: {cand['group_A']} + {cand['group_B']}")
            print(f"   样本数: {cand['size_A']:>4} + {cand['size_B']:>4} = {cand['merged_size']}")
            print(f"   均值: {cand['mean_A']:>8.4f} vs {cand['mean_B']:>8.4f}  (差异: {cand['mean_diff']:.4f})")
            print(f"   标准差: {cand['std_A']:>8.4f} vs {cand['std_B']:>8.4f}  (差异: {cand['std_diff']:.4f})")
            print(f"   KS 检验: p={cand['p_value']:.6f} ✗ ({cand['ks_interpretation']})")
    
    print()


def export_candidates_to_csv(candidates: list, output_path: str = "result/spec_group_merge_analysis/merge_candidates.csv"):
    """导出候选对到 CSV"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df_cand = pd.DataFrame(candidates)
    df_cand = df_cand.sort_values('merged_size', ascending=False)
    df_cand.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"\n✓ 候选合并对已导出至: {output_path}")
    return df_cand


def main():
    """主流程"""
    config_path = "config.yaml"
    
    try:
        # 1. 加载特征工程后的数据
        df = load_featured_data(config_path)
        
        # 2. 构建规格组
        df = build_spec_groups(df)
        
        # 3. 计算 Delta
        df = compute_delta_columns(df)
        
        # 4. 规格组统计
        df_valid = summarize_groups(df)
        
        # 5. KS 检验
        candidates = perform_ks_tests(df_valid)
        
        # 6. 打印结果
        print_ks_results(candidates)
        
        # 7. 导出到 CSV
        if candidates:
            export_candidates_to_csv(candidates)
        
        print("\n" + "="*70)
        print("【阶段 A 完成】")
        print("="*70)
        print("\n阶段 A 的作用:")
        print("  ✓ 快速筛选 Delta 分布相同的规格组对")
        print("  ✓ KS 通过的对优先级高，但不是硬性要求")
        print("  ✓ KS 未通过的对仍可进入 Phase B 进一步检验")
        print("\n后续步骤:")
        print("  1. 查看 merge_candidates.csv 了解所有候选对")
        print("  2. 执行阶段 B（Chow 检验 + 性能对比）进行深入验证")
        print("  3. Chow 检验会识别出模型系数是否真正相同")
        print()
        
    except Exception as e:
        print(f"\n✗ 执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
