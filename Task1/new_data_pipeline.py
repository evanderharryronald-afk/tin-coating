"""
新数据处理流水线
使用配置驱动的清洗和特征工程处理新数据（202607-202608）
并对新特征（GALV_EFF_TOP/BOT、GALV_MAX_CUR_DEN_TOP/BOT）进行相关性和EDA分析
"""

import os
import yaml
import pandas as pd
from data_cleaner import SteelDataCleaner
from feature_engineering import FeatureEngineer
from correlation_analyzer import SurfaceCorrelationAnalyzer
from eda_analyzer import SurfaceEDAAnalyzer


def load_config(config_path: str = 'config.yaml') -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def process_new_data(raw_data_path: str, config: dict) -> tuple:
    """
    处理新数据：清洗 + 特征工程
    
    returns:
    - cleaned_df: 清洗后的数据
    - featured_df: 特征工程后的数据
    """
    print("\n" + "="*70)
    print("【新数据处理流水线】")
    print("="*70)
    
    # 1. 读取原始数据
    print("\n[步骤 1] 加载原始数据...")
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"未找到数据文件: {raw_data_path}")
    
    raw_df = pd.read_excel(raw_data_path)
    print(f"✓ 原始数据加载完成，行数: {len(raw_df)}, 列数: {len(raw_df.columns)}")
    
    # 2. 数据清洗（使用新规则集）
    print("\n[步骤 2] 执行数据清洗（新数据规则集）...")
    cleaner = SteelDataCleaner(rules_config=config)
    cleaned_df = cleaner.process(
        raw_df,
        active_rule_set='new_data',
        clean_save_path="result/new_data/cleaned_data/cleaned_data.xlsx",
        filtered_save_path="result/new_data/cleaned_data/filtered_outliers.xlsx"
    )
    print(f"✓ 数据清洗完成，干净数据行数: {len(cleaned_df)}")
    
    # 3. 特征工程
    print("\n[步骤 3] 执行特征工程...")
    engineer = FeatureEngineer(features_config=config)
    featured_df = engineer.transform(cleaned_df)
    print(f"✓ 特征工程完成，特征维度: {featured_df.shape[1]} 列")
    
    # 保存特征工程后的数据
    output_dir = "result/new_data/feature_engineered_data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "featured_data.xlsx")
    featured_df.to_excel(output_path, index=False)
    print(f"✓ 特征工程数据已保存至: {output_path}")
    
    return cleaned_df, featured_df


def analyze_new_features(featured_df: pd.DataFrame) -> dict:
    """
    对新特征进行相关性和EDA分析
    
    params:
    - featured_df: 特征工程后的数据
    
    returns:
    - analysis_results: 分析结果概览
    """
    print("\n" + "="*70)
    print("【新特征分析】")
    print("="*70)
    
    analysis_results = {
        'new_features_found': [],
        'new_features_missing': [],
        'correlation_analyses': [],
        'eda_analyses': []
    }
    
    # 新特征列表
    new_features = ['GALV_EFF_TOP', 'GALV_EFF_BOT', 'GALV_MAX_CUR_DEN_TOP', 'GALV_MAX_CUR_DEN_BOT']
    
    # 检查新特征是否存在
    for feat in new_features:
        if feat in featured_df.columns:
            analysis_results['new_features_found'].append(feat)
            print(f"✓ 发现新特征: {feat}")
        else:
            analysis_results['new_features_missing'].append(feat)
            print(f"✗ 缺失新特征: {feat}")
    
    if not analysis_results['new_features_found']:
        print("\n⚠ 未发现任何新特征，跳过相关性和EDA分析")
        return analysis_results
    
    # 初始化分析器
    print("\n[步骤 1] 初始化分析工具...")
    correlation_analyzer = SurfaceCorrelationAnalyzer(
        default_save_dir="result/new_data/correlation_result"
    )
    eda_analyzer = SurfaceEDAAnalyzer(
        default_save_dir="result/new_data/eda_result"
    )
    print("✓ 分析工具初始化完成")
    
    # 对每个新特征进行相关性分析
    print("\n[步骤 2] 执行新特征相关性分析...")
    
    # 分析上表面相关特征
    top_features = [f for f in analysis_results['new_features_found'] if 'TOP' in f]
    if top_features:
        try:
            print(f"\n  分析上表面特征: {top_features}")
            correlation_analyzer.analyze_custom_features(
                df=featured_df,
                target_col='Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Avg',
                feature_cols=top_features,
                title_prefix="上表面_新特征",
                save_dir="result/new_data/correlation_result/Top",
                corr_method='both',
                compute_mi=True,
                compute_dcor=True
            )
            analysis_results['correlation_analyses'].append({
                'surface': 'Top',
                'features': top_features,
                'status': 'completed'
            })
            print(f"  ✓ 上表面新特征相关性分析完成")
        except Exception as e:
            print(f"  ✗ 上表面新特征相关性分析失败: {str(e)}")
            analysis_results['correlation_analyses'].append({
                'surface': 'Top',
                'features': top_features,
                'status': 'failed',
                'error': str(e)
            })
    
    # 分析下表面相关特征
    bot_features = [f for f in analysis_results['new_features_found'] if 'BOT' in f]
    if bot_features:
        try:
            print(f"\n  分析下表面特征: {bot_features}")
            correlation_analyzer.analyze_custom_features(
                df=featured_df,
                target_col='Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Avg',
                feature_cols=bot_features,
                title_prefix="下表面_新特征",
                save_dir="result/new_data/correlation_result/Bot",
                corr_method='both',
                compute_mi=True,
                compute_dcor=True
            )
            analysis_results['correlation_analyses'].append({
                'surface': 'Bot',
                'features': bot_features,
                'status': 'completed'
            })
            print(f"  ✓ 下表面新特征相关性分析完成")
        except Exception as e:
            print(f"  ✗ 下表面新特征相关性分析失败: {str(e)}")
            analysis_results['correlation_analyses'].append({
                'surface': 'Bot',
                'features': bot_features,
                'status': 'failed',
                'error': str(e)
            })
    
    # 对每个新特征进行EDA分析
    print("\n[步骤 3] 执行新特征EDA分析...")
    
    if top_features:
        try:
            print(f"\n  EDA分析上表面特征: {top_features}")
            eda_analyzer.analyze(
                df=featured_df,
                delta_col='Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Avg',
                feature_cols=top_features,
                save_dir="result/new_data/eda_result/Top",
                plot_univariate=True,
                plot_vs_delta=True,
                enable_by_group=False
            )
            analysis_results['eda_analyses'].append({
                'surface': 'Top',
                'features': top_features,
                'status': 'completed'
            })
            print(f"  ✓ 上表面新特征EDA分析完成")
        except Exception as e:
            print(f"  ✗ 上表面新特征EDA分析失败: {str(e)}")
            analysis_results['eda_analyses'].append({
                'surface': 'Top',
                'features': top_features,
                'status': 'failed',
                'error': str(e)
            })
    
    if bot_features:
        try:
            print(f"\n  EDA分析下表面特征: {bot_features}")
            eda_analyzer.analyze(
                df=featured_df,
                delta_col='Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Avg',
                feature_cols=bot_features,
                save_dir="result/new_data/eda_result/Bot",
                plot_univariate=True,
                plot_vs_delta=True,
                enable_by_group=False
            )
            analysis_results['eda_analyses'].append({
                'surface': 'Bot',
                'features': bot_features,
                'status': 'completed'
            })
            print(f"  ✓ 下表面新特征EDA分析完成")
        except Exception as e:
            print(f"  ✗ 下表面新特征EDA分析失败: {str(e)}")
            analysis_results['eda_analyses'].append({
                'surface': 'Bot',
                'features': bot_features,
                'status': 'failed',
                'error': str(e)
            })
    
    return analysis_results


def print_summary(analysis_results: dict):
    """打印分析总结"""
    print("\n" + "="*70)
    print("【处理总结】")
    print("="*70)
    
    print(f"\n新特征统计:")
    print(f"  ✓ 发现的新特征: {len(analysis_results['new_features_found'])}")
    for feat in analysis_results['new_features_found']:
        print(f"    - {feat}")
    
    if analysis_results['new_features_missing']:
        print(f"  ✗ 缺失的新特征: {len(analysis_results['new_features_missing'])}")
        for feat in analysis_results['new_features_missing']:
            print(f"    - {feat}")
    
    print(f"\n相关性分析结果:")
    for analysis in analysis_results['correlation_analyses']:
        status_icon = "✓" if analysis['status'] == 'completed' else "✗"
        print(f"  {status_icon} {analysis['surface']}: {analysis['status']}")
    
    print(f"\nEDA分析结果:")
    for analysis in analysis_results['eda_analyses']:
        status_icon = "✓" if analysis['status'] == 'completed' else "✗"
        print(f"  {status_icon} {analysis['surface']}: {analysis['status']}")
    
    print("\n" + "="*70)
    print("✓ 新数据处理流水线完成")
    print("="*70 + "\n")


def main():
    """主函数"""
    try:
        # 加载配置
        print("加载配置文件...")
        config = load_config('config.yaml')
        print("✓ 配置文件加载成功")
        
        # 处理新数据
        raw_data_path = "data/new_data/converted_old_format.xlsx"
        cleaned_df, featured_df = process_new_data(raw_data_path, config)
        
        # 分析新特征
        analysis_results = analyze_new_features(featured_df)
        
        # 打印总结
        print_summary(analysis_results)
        
    except Exception as e:
        print(f"\n✗ 处理过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
