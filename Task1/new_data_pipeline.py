"""
新数据处理流水线
使用配置驱动的清洗和特征工程处理新数据（202607-202608）
并对旧特征、新特征、组合特征进行相关性和EDA分析
"""

import os
import yaml
import pandas as pd
from data_cleaner import SteelDataCleaner
from feature_engineering import FeatureEngineer
from correlation_analyzer import SurfaceCorrelationAnalyzer
from eda_analyzer import SurfaceEDAAnalyzer


# ============================================
# 特征定义（直接定义，避免配置文件臃肿）
# ============================================

# 旧特征（经过验证的标准特征）
LEGACY_FEATURES_TOP = [
    'Top_Current_Sum',
    'Top_Current_Per_Speed',
    'Top_Theoretical_Factor',
    'Speed[m/min]_Process_Avg',
    'Dimension_[mm]_Thickness',
    'Dimension_[mm]_Width',
    'Steel_Grade_Encoded',
]

LEGACY_FEATURES_BOT = [
    'Bot_Current_Sum',
    'Bot_Current_Per_Speed',
    'Bot_Theoretical_Factor',
    'Speed[m/min]_Process_Avg',
    'Dimension_[mm]_Thickness',
    'Dimension_[mm]_Width',
    'Steel_Grade_Encoded',
]

# 新特征
NEW_FEATURES_TOP = ['GALV_EFF_TOP', 'GALV_MAX_CUR_DEN_TOP']
NEW_FEATURES_BOT = ['GALV_EFF_BOT', 'GALV_MAX_CUR_DEN_BOT']

# EDA 分析特征（在线值前置）
EDA_LEGACY_FEATURES_TOP = [
    'Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Avg',
] + LEGACY_FEATURES_TOP

EDA_LEGACY_FEATURES_BOT = [
    'Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Avg',
] + LEGACY_FEATURES_BOT


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
    对新数据的特征进行全面分析：旧特征、新特征、组合特征
    
    params:
    - featured_df: 特征工程后的数据
    
    returns:
    - analysis_results: 分析结果概览
    """
    print("\n" + "="*70)
    print("【特征分析：旧特征 / 新特征 / 组合】")
    print("="*70)
    
    analysis_results = {
        'new_features_found': [],
        'new_features_missing': [],
        'correlation_analyses': [],
        'eda_analyses': []
    }
    
    # 检查新特征是否存在
    new_features_top = [f for f in NEW_FEATURES_TOP if f in featured_df.columns]
    new_features_bot = [f for f in NEW_FEATURES_BOT if f in featured_df.columns]
    
    for feat in new_features_top + new_features_bot:
        analysis_results['new_features_found'].append(feat)
        print(f"✓ 发现新特征: {feat}")
    
    missing_new = [f for f in NEW_FEATURES_TOP + NEW_FEATURES_BOT if f not in featured_df.columns]
    if missing_new:
        for feat in missing_new:
            analysis_results['new_features_missing'].append(feat)
            print(f"✗ 缺失新特征: {feat}")
    
    # 初始化分析器
    print("\n[初始化] 分析工具...")
    correlation_analyzer = SurfaceCorrelationAnalyzer(
        default_save_dir="result/new_data/correlation_result"
    )
    eda_analyzer = SurfaceEDAAnalyzer(
        default_save_dir="result/new_data/eda_result"
    )
    print("✓ 分析工具初始化完成")
    
    # ============================================
    # 上表面分析
    # ============================================
    print("\n[步骤 1] 上表面相关性分析...")
    target_top = 'Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Avg'
    
    # 1.1 旧特征相关性
    try:
        print(f"  分析旧特征相关性 ({len(LEGACY_FEATURES_TOP)} 个特征)...")
        correlation_analyzer.analyze_custom_features(
            df=featured_df,
            target_col=target_top,
            feature_cols=LEGACY_FEATURES_TOP,
            title_prefix="上表面_旧特征",
            save_dir="result/new_data/correlation_result/Top/Legacy",
            corr_method='both',
            compute_mi=True,
            compute_dcor=True
        )
        analysis_results['correlation_analyses'].append({
            'surface': 'Top', 'type': 'legacy', 'status': 'completed'
        })
        print(f"  ✓ 旧特征相关性分析完成")
    except Exception as e:
        print(f"  ✗ 旧特征相关性分析失败: {str(e)}")
        analysis_results['correlation_analyses'].append({
            'surface': 'Top', 'type': 'legacy', 'status': 'failed', 'error': str(e)
        })
    
    # 1.2 新特征相关性
    if new_features_top:
        try:
            print(f"  分析新特征相关性 ({len(new_features_top)} 个特征)...")
            correlation_analyzer.analyze_custom_features(
                df=featured_df,
                target_col=target_top,
                feature_cols=new_features_top,
                title_prefix="上表面_新特征",
                save_dir="result/new_data/correlation_result/Top/New",
                corr_method='both',
                compute_mi=True,
                compute_dcor=True
            )
            analysis_results['correlation_analyses'].append({
                'surface': 'Top', 'type': 'new', 'status': 'completed'
            })
            print(f"  ✓ 新特征相关性分析完成")
        except Exception as e:
            print(f"  ✗ 新特征相关性分析失败: {str(e)}")
            analysis_results['correlation_analyses'].append({
                'surface': 'Top', 'type': 'new', 'status': 'failed', 'error': str(e)
            })
        
        # 1.3 组合特征相关性
        try:
            combined_top = LEGACY_FEATURES_TOP + new_features_top
            print(f"  分析组合特征相关性 ({len(combined_top)} 个特征)...")
            correlation_analyzer.analyze_custom_features(
                df=featured_df,
                target_col=target_top,
                feature_cols=combined_top,
                title_prefix="上表面_组合特征",
                save_dir="result/new_data/correlation_result/Top/Combined",
                corr_method='both',
                compute_mi=True,
                compute_dcor=True
            )
            analysis_results['correlation_analyses'].append({
                'surface': 'Top', 'type': 'combined', 'status': 'completed'
            })
            print(f"  ✓ 组合特征相关性分析完成")
        except Exception as e:
            print(f"  ✗ 组合特征相关性分析失败: {str(e)}")
            analysis_results['correlation_analyses'].append({
                'surface': 'Top', 'type': 'combined', 'status': 'failed', 'error': str(e)
            })
    
    # ============================================
    # 下表面分析
    # ============================================
    print("\n[步骤 2] 下表面相关性分析...")
    target_bot = 'Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Avg'
    
    # 2.1 旧特征相关性
    try:
        print(f"  分析旧特征相关性 ({len(LEGACY_FEATURES_BOT)} 个特征)...")
        correlation_analyzer.analyze_custom_features(
            df=featured_df,
            target_col=target_bot,
            feature_cols=LEGACY_FEATURES_BOT,
            title_prefix="下表面_旧特征",
            save_dir="result/new_data/correlation_result/Bot/Legacy",
            corr_method='both',
            compute_mi=True,
            compute_dcor=True
        )
        analysis_results['correlation_analyses'].append({
            'surface': 'Bot', 'type': 'legacy', 'status': 'completed'
        })
        print(f"  ✓ 旧特征相关性分析完成")
    except Exception as e:
        print(f"  ✗ 旧特征相关性分析失败: {str(e)}")
        analysis_results['correlation_analyses'].append({
            'surface': 'Bot', 'type': 'legacy', 'status': 'failed', 'error': str(e)
        })
    
    # 2.2 新特征相关性
    if new_features_bot:
        try:
            print(f"  分析新特征相关性 ({len(new_features_bot)} 个特征)...")
            correlation_analyzer.analyze_custom_features(
                df=featured_df,
                target_col=target_bot,
                feature_cols=new_features_bot,
                title_prefix="下表面_新特征",
                save_dir="result/new_data/correlation_result/Bot/New",
                corr_method='both',
                compute_mi=True,
                compute_dcor=True
            )
            analysis_results['correlation_analyses'].append({
                'surface': 'Bot', 'type': 'new', 'status': 'completed'
            })
            print(f"  ✓ 新特征相关性分析完成")
        except Exception as e:
            print(f"  ✗ 新特征相关性分析失败: {str(e)}")
            analysis_results['correlation_analyses'].append({
                'surface': 'Bot', 'type': 'new', 'status': 'failed', 'error': str(e)
            })
        
        # 2.3 组合特征相关性
        try:
            combined_bot = LEGACY_FEATURES_BOT + new_features_bot
            print(f"  分析组合特征相关性 ({len(combined_bot)} 个特征)...")
            correlation_analyzer.analyze_custom_features(
                df=featured_df,
                target_col=target_bot,
                feature_cols=combined_bot,
                title_prefix="下表面_组合特征",
                save_dir="result/new_data/correlation_result/Bot/Combined",
                corr_method='both',
                compute_mi=True,
                compute_dcor=True
            )
            analysis_results['correlation_analyses'].append({
                'surface': 'Bot', 'type': 'combined', 'status': 'completed'
            })
            print(f"  ✓ 组合特征相关性分析完成")
        except Exception as e:
            print(f"  ✗ 组合特征相关性分析失败: {str(e)}")
            analysis_results['correlation_analyses'].append({
                'surface': 'Bot', 'type': 'combined', 'status': 'failed', 'error': str(e)
            })
    
    # ============================================
    # EDA 分析
    # ============================================
    print("\n[步骤 3] 上表面 EDA 分析...")
    
    # 3.1 旧特征 EDA
    try:
        print(f"  分析旧特征 ({len(EDA_LEGACY_FEATURES_TOP)} 个特征)...")
        eda_analyzer.analyze(
            df=featured_df,
            delta_col=target_top,
            feature_cols=EDA_LEGACY_FEATURES_TOP,
            save_dir="result/new_data/eda_result/Top/Legacy",
            plot_univariate=True,
            plot_vs_delta=True,
            enable_by_group=False
        )
        analysis_results['eda_analyses'].append({
            'surface': 'Top', 'type': 'legacy', 'status': 'completed'
        })
        print(f"  ✓ 旧特征 EDA 完成")
    except Exception as e:
        print(f"  ✗ 旧特征 EDA 失败: {str(e)}")
        analysis_results['eda_analyses'].append({
            'surface': 'Top', 'type': 'legacy', 'status': 'failed', 'error': str(e)
        })
    
    # 3.2 新特征 EDA
    if new_features_top:
        try:
            print(f"  分析新特征 ({len(new_features_top)} 个特征)...")
            eda_analyzer.analyze(
                df=featured_df,
                delta_col=target_top,
                feature_cols=new_features_top,
                save_dir="result/new_data/eda_result/Top/New",
                plot_univariate=True,
                plot_vs_delta=True,
                enable_by_group=False
            )
            analysis_results['eda_analyses'].append({
                'surface': 'Top', 'type': 'new', 'status': 'completed'
            })
            print(f"  ✓ 新特征 EDA 完成")
        except Exception as e:
            print(f"  ✗ 新特征 EDA 失败: {str(e)}")
            analysis_results['eda_analyses'].append({
                'surface': 'Top', 'type': 'new', 'status': 'failed', 'error': str(e)
            })
        
        # 3.3 组合特征 EDA
        try:
            combined_eda_top = EDA_LEGACY_FEATURES_TOP + new_features_top
            print(f"  分析组合特征 ({len(combined_eda_top)} 个特征)...")
            eda_analyzer.analyze(
                df=featured_df,
                delta_col=target_top,
                feature_cols=combined_eda_top,
                save_dir="result/new_data/eda_result/Top/Combined",
                plot_univariate=True,
                plot_vs_delta=True,
                enable_by_group=False
            )
            analysis_results['eda_analyses'].append({
                'surface': 'Top', 'type': 'combined', 'status': 'completed'
            })
            print(f"  ✓ 组合特征 EDA 完成")
        except Exception as e:
            print(f"  ✗ 组合特征 EDA 失败: {str(e)}")
            analysis_results['eda_analyses'].append({
                'surface': 'Top', 'type': 'combined', 'status': 'failed', 'error': str(e)
            })
    
    # ============================================
    # 下表面 EDA
    # ============================================
    print("\n[步骤 4] 下表面 EDA 分析...")
    
    # 4.1 旧特征 EDA
    try:
        print(f"  分析旧特征 ({len(EDA_LEGACY_FEATURES_BOT)} 个特征)...")
        eda_analyzer.analyze(
            df=featured_df,
            delta_col=target_bot,
            feature_cols=EDA_LEGACY_FEATURES_BOT,
            save_dir="result/new_data/eda_result/Bot/Legacy",
            plot_univariate=True,
            plot_vs_delta=True,
            enable_by_group=False
        )
        analysis_results['eda_analyses'].append({
            'surface': 'Bot', 'type': 'legacy', 'status': 'completed'
        })
        print(f"  ✓ 旧特征 EDA 完成")
    except Exception as e:
        print(f"  ✗ 旧特征 EDA 失败: {str(e)}")
        analysis_results['eda_analyses'].append({
            'surface': 'Bot', 'type': 'legacy', 'status': 'failed', 'error': str(e)
        })
    
    # 4.2 新特征 EDA
    if new_features_bot:
        try:
            print(f"  分析新特征 ({len(new_features_bot)} 个特征)...")
            eda_analyzer.analyze(
                df=featured_df,
                delta_col=target_bot,
                feature_cols=new_features_bot,
                save_dir="result/new_data/eda_result/Bot/New",
                plot_univariate=True,
                plot_vs_delta=True,
                enable_by_group=False
            )
            analysis_results['eda_analyses'].append({
                'surface': 'Bot', 'type': 'new', 'status': 'completed'
            })
            print(f"  ✓ 新特征 EDA 完成")
        except Exception as e:
            print(f"  ✗ 新特征 EDA 失败: {str(e)}")
            analysis_results['eda_analyses'].append({
                'surface': 'Bot', 'type': 'new', 'status': 'failed', 'error': str(e)
            })
        
        # 4.3 组合特征 EDA
        try:
            combined_eda_bot = EDA_LEGACY_FEATURES_BOT + new_features_bot
            print(f"  分析组合特征 ({len(combined_eda_bot)} 个特征)...")
            eda_analyzer.analyze(
                df=featured_df,
                delta_col=target_bot,
                feature_cols=combined_eda_bot,
                save_dir="result/new_data/eda_result/Bot/Combined",
                plot_univariate=True,
                plot_vs_delta=True,
                enable_by_group=False
            )
            analysis_results['eda_analyses'].append({
                'surface': 'Bot', 'type': 'combined', 'status': 'completed'
            })
            print(f"  ✓ 组合特征 EDA 完成")
        except Exception as e:
            print(f"  ✗ 组合特征 EDA 失败: {str(e)}")
            analysis_results['eda_analyses'].append({
                'surface': 'Bot', 'type': 'combined', 'status': 'failed', 'error': str(e)
            })
    
    return analysis_results


def print_summary(analysis_results: dict):
    """打印分析总结"""
    print("\n" + "="*70)
    print("【处理总结】")
    print("="*70)
    
    print(f"\n新特征统计:")
    print(f"  发现: {len(analysis_results['new_features_found'])} 个")
    if analysis_results['new_features_missing']:
        print(f"  缺失: {len(analysis_results['new_features_missing'])} 个")
    
    # 统计成功失败
    corr_success = sum(1 for a in analysis_results['correlation_analyses'] if a['status'] == 'completed')
    corr_failed = len(analysis_results['correlation_analyses']) - corr_success
    
    eda_success = sum(1 for a in analysis_results['eda_analyses'] if a['status'] == 'completed')
    eda_failed = len(analysis_results['eda_analyses']) - eda_success
    
    print(f"\n相关性分析: {corr_success}/{len(analysis_results['correlation_analyses'])} 成功")
    print(f"EDA分析: {eda_success}/{len(analysis_results['eda_analyses'])} 成功")
    
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
        
        # 分析特征（旧+新+组合）
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
