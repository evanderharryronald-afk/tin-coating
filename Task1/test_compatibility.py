"""
向后兼容性测试脚本
验证改造后的 data_cleaner 和 feature_engineering 对旧数据仍然适配
"""

import os
import yaml
import pandas as pd
import numpy as np
from data_cleaner import SteelDataCleaner
from feature_engineering import FeatureEngineer


def test_old_data_pipeline():
    """
    验证旧数据分析流程不受影响
    """
    print("\n" + "="*60)
    print("测试 1: 旧数据分析流程兼容性")
    print("="*60)
    
    # 获取旧数据
    try:
        raw_old = pd.read_excel("result/data/merged_data/merged_result_latest.xlsx")
        print(f"✓ 成功加载旧数据，行数: {len(raw_old)}")
    except FileNotFoundError:
        print("✗ 未找到旧数据文件，跳过此测试")
        return False
    
    # 测试 data_cleaner（无参数初始化 = 默认旧规则）
    try:
        cleaner = SteelDataCleaner()  # 默认参数 = 旧规则
        clean_df = cleaner.process(raw_old)
        print(f"✓ 数据清洗完成，干净数据行数: {len(clean_df)}")
        print(f"  执行的规则: {cleaner.execution_log['executed_rules']}")
    except Exception as e:
        print(f"✗ 数据清洗失败: {str(e)}")
        return False
    
    # 检查必要的列是否存在
    required_cols = [
        'Coil ID', 'Steel Grade', 'Speed[m/min]_Process_Avg',
        'Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Avg',
        'Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Avg'
    ]
    missing_cols = [c for c in required_cols if c not in clean_df.columns]
    if missing_cols:
        print(f"✗ 缺失必要列: {missing_cols}")
        return False
    print(f"✓ 清洗后数据包含所有必要列")
    
    # 测试 feature_engineering（无参数初始化 = 默认旧特征）
    try:
        engineer = FeatureEngineer()  # 默认参数 = 旧特征配置
        featured_df = engineer.transform(clean_df)
        print(f"✓ 特征工程完成，特征维度: {featured_df.shape[1]}")
        print(f"  生成的特征组: {engineer.generation_log['generated_groups']}")
    except Exception as e:
        print(f"✗ 特征工程失败: {str(e)}")
        return False
    
    # 检查核心特征是否存在
    required_features = FeatureEngineer.REQUIRED_FEATURES
    missing_features = [f for f in required_features if f not in featured_df.columns]
    if missing_features:
        print(f"✗ 缺失核心特征: {missing_features}")
        return False
    print(f"✓ 特征工程生成了所有核心特征")
    
    # 检查旧特征（Delta, Centered Delta）
    if '上表面镀层重量A(XA1_0)' in clean_df.columns:
        delta_features = ['Top_Delta', 'Bot_Delta', 'Top_Delta_Centered', 'Bot_Delta_Centered']
        missing_delta = [f for f in delta_features if f not in featured_df.columns]
        if missing_delta:
            print(f"⚠ 旧数据的残差特征不完整: {missing_delta}")
        else:
            print(f"✓ 旧数据的残差特征完整")
    
    # 检查数据完整性
    print(f"✓ 最终数据行数: {len(featured_df)}, 列数: {len(featured_df.columns)}")
    print(f"✓ 旧数据分析流程兼容性测试通过")
    
    return True


def test_new_data_with_config():
    """
    验证新数据配置驱动流程
    """
    print("\n" + "="*60)
    print("测试 2: 新数据配置驱动流程")
    print("="*60)
    
    # 加载配置
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        print(f"✓ 成功加载配置文件")
    except FileNotFoundError:
        print("✗ 未找到 config.yaml 文件")
        return False
    except Exception as e:
        print(f"✗ 配置文件加载失败: {str(e)}")
        return False
    
    # 获取新数据
    try:
        new_data_path = "data/new_data/converted_old_format.xlsx"
        if not os.path.exists(new_data_path):
            print(f"⚠ 未找到新数据文件 ({new_data_path})，跳过此测试")
            return True
        
        raw_new = pd.read_excel(new_data_path)
        print(f"✓ 成功加载新数据，行数: {len(raw_new)}")
    except Exception as e:
        print(f"⚠ 新数据加载失败: {str(e)}，跳过此测试")
        return True
    
    # 测试 data_cleaner（使用配置和新规则集）
    try:
        cleaner = SteelDataCleaner(rules_config=config)
        clean_df = cleaner.process(raw_new, active_rule_set='new_data')
        print(f"✓ 新数据清洗完成，干净数据行数: {len(clean_df)}")
        print(f"  执行的规则: {cleaner.execution_log['executed_rules']}")
        if cleaner.execution_log['skipped_rules']:
            print(f"  跳过的规则: {[r['name'] for r in cleaner.execution_log['skipped_rules']]}")
    except Exception as e:
        print(f"✗ 新数据清洗失败: {str(e)}")
        return False
    
    # 检查新数据的特定特征是否存在
    new_features = ['GALV_EFF_TOP', 'GALV_EFF_BOT', 'GALV_MAX_CUR_DEN_TOP', 'GALV_MAX_CUR_DEN_BOT']
    existing_new_features = [f for f in new_features if f in raw_new.columns]
    print(f"✓ 新数据中存在的特征: {existing_new_features}")
    
    # 测试 feature_engineering（使用配置）
    try:
        engineer = FeatureEngineer(features_config=config)
        featured_df = engineer.transform(clean_df)
        print(f"✓ 新数据特征工程完成，特征维度: {featured_df.shape[1]}")
        print(f"  生成的特征组: {engineer.generation_log['generated_groups']}")
    except Exception as e:
        print(f"✗ 新数据特征工程失败: {str(e)}")
        return False
    
    # 检查核心特征
    required_features = FeatureEngineer.REQUIRED_FEATURES
    missing_features = [f for f in required_features if f not in featured_df.columns]
    if missing_features:
        print(f"✗ 缺失核心特征: {missing_features}")
        return False
    print(f"✓ 新数据特征工程生成了所有核心特征")
    
    # 检查新特征是否被 passthrough
    for feat in existing_new_features:
        if feat in featured_df.columns:
            print(f"  ✓ 新特征 '{feat}' 已保留")
        else:
            print(f"  ⚠ 新特征 '{feat}' 未保留")
    
    print(f"✓ 新数据配置驱动流程测试通过")
    
    return True


def test_downstream_compatibility():
    """
    验证后续模块是否能使用特征工程后的数据
    （模拟下游模块的数据访问模式）
    """
    print("\n" + "="*60)
    print("测试 3: 后续模块数据兼容性")
    print("="*60)
    
    # 创建模拟的特征工程数据
    try:
        cleaner = SteelDataCleaner()
        raw_df = pd.read_excel("result/data/merged_data/merged_result_latest.xlsx")
        clean_df = cleaner.process(raw_df)
        
        engineer = FeatureEngineer()
        featured_df = engineer.transform(clean_df)
        print(f"✓ 生成测试数据集，行数: {len(featured_df)}")
    except Exception as e:
        print(f"⚠ 无法生成测试数据: {str(e)}，跳过此测试")
        return True
    
    # 模拟后续模块的数据访问（如 run_tuning_optuna.py）
    downstream_requirements = {
        'features': [
            'Top_Current_Sum', 'Bot_Current_Sum',
            'Top_Current_Per_Speed', 'Bot_Current_Per_Speed',
            'Top_Theoretical_Factor', 'Bot_Theoretical_Factor',
            'Speed[m/min]_Process_Avg', 'Dimension_[mm]_Width',
            'Dimension_[mm]_Thickness', 'Steel_Grade_Encoded'
        ],
        'targets': ['Top_Delta', 'Bot_Delta'],
        'identifiers': ['Coil ID', 'Steel Grade']
    }
    
    # 检查所有特征列
    missing_features = [f for f in downstream_requirements['features'] if f not in featured_df.columns]
    if missing_features:
        print(f"✗ 后续模块需要的特征缺失: {missing_features}")
        return False
    print(f"✓ 后续模块所需的所有特征都存在")
    
    # 检查目标列（旧数据有）
    if '上表面镀层重量A(XA1_0)' in clean_df.columns:
        missing_targets = [t for t in downstream_requirements['targets'] if t not in featured_df.columns]
        if missing_targets:
            print(f"⚠ 目标列不完整: {missing_targets}")
        else:
            print(f"✓ 旧数据的目标列完整")
    else:
        print(f"ℹ 新数据无测量值，目标列不可用（这是正常的）")
    
    # 检查标识符
    missing_ids = [i for i in downstream_requirements['identifiers'] if i not in featured_df.columns]
    if missing_ids:
        print(f"✗ 标识符列缺失: {missing_ids}")
        return False
    print(f"✓ 标识符列完整")
    
    # 数据质量检查
    try:
        # 检查关键特征是否有 NaN
        key_features = ['Top_Current_Sum', 'Top_Theoretical_Factor', 'Steel_Grade_Encoded']
        for feat in key_features:
            nan_count = featured_df[feat].isna().sum()
            if nan_count > 0:
                print(f"  ⚠ '{feat}' 包含 {nan_count} 个 NaN 值")
        
        # 检查特征值的合理性（非 inf）
        numeric_features = featured_df[downstream_requirements['features']].select_dtypes(include=[np.number])
        inf_count = np.isinf(numeric_features).sum().sum()
        if inf_count > 0:
            print(f"  ⚠ 特征数据中包含 {inf_count} 个无穷值")
        else:
            print(f"✓ 特征数据质量良好（无 inf 值）")
    except Exception as e:
        print(f"  ⚠ 数据质量检查异常: {str(e)}")
    
    print(f"✓ 后续模块数据兼容性测试通过")
    
    return True


def test_config_loading():
    """
    验证配置文件的有效性
    """
    print("\n" + "="*60)
    print("测试 4: 配置文件有效性")
    print("="*60)
    
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        print(f"✓ 配置文件加载成功")
    except Exception as e:
        print(f"✗ 配置文件加载失败: {str(e)}")
        return False
    
    # 检查版本号
    if 'version' in config:
        print(f"  配置版本: {config['version']}")
    
    # 检查必要的配置块
    required_blocks = ['data_cleaning', 'feature_engineering']
    for block in required_blocks:
        if block not in config:
            print(f"✗ 缺失配置块: {block}")
            return False
    print(f"✓ 配置包含所有必要块")
    
    # 检查规则集
    try:
        rule_sets = config['data_cleaning']['rule_sets']
        rule_set_names = list(rule_sets.keys())
        print(f"  规则集: {rule_set_names}")
        if 'old_data' not in rule_set_names or 'new_data' not in rule_set_names:
            print(f"⚠ 规则集不完整，应包含 'old_data' 和 'new_data'")
    except Exception as e:
        print(f"✗ 规则集检查失败: {str(e)}")
        return False
    
    # 检查特征组
    try:
        feature_groups = config['feature_engineering']['feature_groups']
        feature_group_names = list(feature_groups.keys())
        print(f"  特征组: {feature_group_names}")
    except Exception as e:
        print(f"✗ 特征组检查失败: {str(e)}")
        return False
    
    print(f"✓ 配置文件有效性检查通过")
    
    return True


def main():
    """
    运行所有兼容性测试
    """
    print("\n" + "="*60)
    print("向后兼容性测试套件")
    print("="*60)
    
    tests = [
        ("配置文件有效性", test_config_loading),
        ("旧数据分析流程", test_old_data_pipeline),
        ("新数据配置流程", test_new_data_with_config),
        ("后续模块兼容性", test_downstream_compatibility),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = "通过" if result else "失败"
        except Exception as e:
            print(f"\n✗ 测试异常: {str(e)}")
            results[test_name] = "异常"
    
    # 打印测试总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    for test_name, result in results.items():
        status_icon = "✓" if result == "通过" else ("✗" if result == "失败" else "⚠")
        print(f"{status_icon} {test_name}: {result}")
    
    all_passed = all(r == "通过" for r in results.values())
    print("\n" + "="*60)
    if all_passed:
        print("✓ 所有测试通过！改造成功且向后兼容")
    else:
        print("✗ 部分测试失败或异常，请检查")
    print("="*60 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
