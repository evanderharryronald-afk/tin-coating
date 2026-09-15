import os
import yaml
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class FeatureEngineer:
    """
    电镀锡生产数据特征工程模块
    包含：电流求和、物理理论因子构建、数据去中心化残差计算、钢种频率编码等
    
    支持配置驱动的特征框架，兼容旧数据和新数据
    """
    
    # 核心必需特征（任何情况下都必须生成，否则后续模块崩溃）
    REQUIRED_FEATURES = [
        'Top_Current_Sum', 'Bot_Current_Sum',
        'Top_Current_Per_Speed', 'Bot_Current_Per_Speed',
        'Top_Theoretical_Factor', 'Bot_Theoretical_Factor',
        'Steel_Grade_Encoded'
    ]

    def __init__(self, features_config: Optional[Dict] = None, eps: float = 1e-5):
        """
        初始化特征工程器
        
        params:
        - features_config: 配置字典（如为 None，则使用默认配置）
        - eps: 数值稳定性参数
        """
        self.features_config = features_config
        self.eps = eps
        self.generation_log = {}

    def transform(self, df, enabled_groups: Optional[List[str]] = None):
        """
        根据配置构建所有衍生特征
        
        params:
        - df: 清洗后的 DataFrame
        - enabled_groups: 启用的特征组（如为 None，使用配置中的 enabled_groups）
        """
        df = df.copy()
        self.generation_log = {
            'generated_groups': [],
            'failed_groups': [],
            'generation_errors': []
        }
        
        # 加载配置
        if self.features_config is not None:
            config = self.features_config
        else:
            config = self._get_default_config()
        
        # 确定启用的特征组
        if enabled_groups is None:
            enabled_groups = config['feature_engineering']['enabled_groups']
        
        # 按顺序执行特征组
        for group_name in enabled_groups:
            if group_name not in config['feature_engineering']['feature_groups']:
                self.generation_log['failed_groups'].append({
                    'group': group_name,
                    'error': 'Feature group not found in config'
                })
                continue
            
            group_config = config['feature_engineering']['feature_groups'][group_name]
            
            if not group_config.get('enabled', True):
                continue
            
            try:
                df = self._add_feature_group(df, group_name, group_config)
                self.generation_log['generated_groups'].append(group_name)
            except Exception as e:
                self.generation_log['failed_groups'].append({
                    'group': group_name,
                    'error': str(e)
                })
                self.generation_log['generation_errors'].append({
                    'group': group_name,
                    'error': str(e),
                    'type': type(e).__name__
                })

        # 验证核心特征
        missing_required = [f for f in self.REQUIRED_FEATURES if f not in df.columns]
        if missing_required:
            raise ValueError(f"缺少核心特征: {missing_required}")
        
        # 打印生成日志
        self._print_generation_log()

        return df

    def _add_feature_group(self, df: pd.DataFrame, group_name: str, group_config: Dict) -> pd.DataFrame:
        """根据组类型调用相应的特征生成函数"""
        if group_name == 'electrical':
            return self._add_electrical_features(df, group_config)
        elif group_name == 'physical':
            return self._add_physical_features(df, group_config)
        elif group_name == 'residuals':
            return self._add_residual_features(df, group_config)
        elif group_name == 'encoding':
            return self._add_encoding_features(df, group_config)
        elif group_name == 'new_metrics':
            return self._add_new_metric_features(df, group_config)
        else:
            raise ValueError(f"Unknown feature group: {group_name}")

    def _add_electrical_features(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """电流聚合特征"""
        bot_indices = config['bot_current_indices']
        top_indices = config['top_current_indices']
        
        bot_cols = [f'Tining Section_CURRENT[A]_GL_{i}_Avg' for i in bot_indices]
        top_cols = [f'Tining Section_CURRENT[A]_GL_{i}_Avg' for i in top_indices]
        
        # 检查列存在
        missing_bot = [c for c in bot_cols if c not in df.columns]
        missing_top = [c for c in top_cols if c not in df.columns]
        
        if missing_bot or missing_top:
            raise KeyError(f"缺少电流列: {missing_bot + missing_top}")
        
        df['Bot_Current_Sum'] = df[bot_cols].sum(axis=1)
        df['Top_Current_Sum'] = df[top_cols].sum(axis=1)
        
        return df

    def _add_physical_features(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """物理因子特征"""
        speed_col = config['speed_col']
        width_col = config['width_col']
        eps = config.get('eps', self.eps)
        
        if speed_col not in df.columns or width_col not in df.columns:
            raise KeyError(f"缺少物理参数列: {speed_col}, {width_col}")
        
        speed = df[speed_col].replace(0, np.nan)
        width_m = df[width_col] / 1000.0
        
        df['Top_Current_Per_Speed'] = df['Top_Current_Sum'] / (speed + eps)
        df['Bot_Current_Per_Speed'] = df['Bot_Current_Sum'] / (speed + eps)
        
        df['Top_Theoretical_Factor'] = df['Top_Current_Sum'] / (speed * width_m + eps)
        df['Bot_Theoretical_Factor'] = df['Bot_Current_Sum'] / (speed * width_m + eps)
        
        df['Width_m'] = width_m
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        return df

    def _add_residual_features(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """残差特征（仅当测量值存在时）"""
        skip_if_missing = config.get('skip_if_missing', True)
        pairs = config['pairs']
        
        for pair in pairs:
            measurement_col = pair['measurement_col']
            online_col = pair['online_col']
            delta_output = pair['delta_output']
            centered_output = pair['centered_output']
            
            if measurement_col not in df.columns or online_col not in df.columns:
                if skip_if_missing:
                    continue
                else:
                    raise KeyError(f"缺少残差计算字段: {measurement_col}, {online_col}")
            
            df[delta_output] = df[measurement_col] - df[online_col]
            
            # 计算在干净集上的 Global Bias 并去除
            top_bias = df[delta_output].mean()
            df[centered_output] = df[delta_output] - top_bias
        
        return df

    def _add_encoding_features(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """分类特征编码"""
        steel_grade_col = config['steel_grade_col']
        encoding_type = config.get('encoding_type', 'frequency')
        
        if steel_grade_col not in df.columns:
            raise KeyError(f"缺少钢种列: {steel_grade_col}")
        
        if encoding_type == 'frequency':
            grade_freq = df[steel_grade_col].value_counts(normalize=True).to_dict()
            df['Steel_Grade_Encoded'] = df[steel_grade_col].map(grade_freq).fillna(0)
        elif encoding_type == 'label':
            df['Steel_Grade_Encoded'] = pd.factorize(df[steel_grade_col])[0]
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")
        
        return df

    def _add_new_metric_features(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """新指标特征（passthrough）"""
        metrics = config['metrics']
        
        for metric in metrics:
            source_col = metric['source_col']
            metric_name = metric['name']
            required = metric.get('required', False)
            metric_type = metric.get('type', 'passthrough')
            
            if source_col not in df.columns:
                if required:
                    raise KeyError(f"缺少必需指标列: {source_col}")
                else:
                    continue
            
            # passthrough：直接复制
            if metric_type == 'passthrough':
                df[metric_name] = df[source_col]
            else:
                raise ValueError(f"Unknown metric type: {metric_type}")
        
        return df

    def _get_default_config(self) -> Dict:
        """
        返回默认配置（向后兼容）
        这是原有逻辑的配置表示
        """
        return {
            'feature_engineering': {
                'feature_groups': {
                    'electrical': {
                        'enabled': True,
                        'bot_current_indices': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35],
                        'top_current_indices': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36],
                        'output_features': ['Bot_Current_Sum', 'Top_Current_Sum']
                    },
                    'physical': {
                        'enabled': True,
                        'speed_col': 'Speed[m/min]_Process_Avg',
                        'width_col': 'Dimension_[mm]_Width',
                        'thickness_col': 'Dimension_[mm]_Thickness',
                        'eps': self.eps,
                        'output_features': [
                            'Top_Current_Per_Speed', 'Bot_Current_Per_Speed',
                            'Top_Theoretical_Factor', 'Bot_Theoretical_Factor', 'Width_m'
                        ]
                    },
                    'residuals': {
                        'enabled': True,
                        'pairs': [
                            {
                                'measurement_col': '上表面镀层重量A(XA1_0)',
                                'online_col': 'Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Avg',
                                'delta_output': 'Top_Delta',
                                'centered_output': 'Top_Delta_Centered',
                                'surface': 'Top'
                            },
                            {
                                'measurement_col': '下表面镀层重量A(XA1_0)',
                                'online_col': 'Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Avg',
                                'delta_output': 'Bot_Delta',
                                'centered_output': 'Bot_Delta_Centered',
                                'surface': 'Bot'
                            }
                        ],
                        'skip_if_missing': True,
                        'output_features': ['Top_Delta', 'Bot_Delta', 'Top_Delta_Centered', 'Bot_Delta_Centered']
                    },
                    'encoding': {
                        'enabled': True,
                        'steel_grade_col': 'Steel Grade',
                        'encoding_type': 'frequency',
                        'output_features': ['Steel_Grade_Encoded']
                    },
                    'new_metrics': {
                        'enabled': True,
                        'metrics': [
                            {'name': 'GALV_EFF_TOP', 'source_col': 'GALV_EFF_TOP', 'type': 'passthrough', 'required': False},
                            {'name': 'GALV_EFF_BOT', 'source_col': 'GALV_EFF_BOT', 'type': 'passthrough', 'required': False},
                            {'name': 'GALV_MAX_CUR_DEN_TOP', 'source_col': 'GALV_MAX_CUR_DEN_TOP', 'type': 'passthrough', 'required': False},
                            {'name': 'GALV_MAX_CUR_DEN_BOT', 'source_col': 'GALV_MAX_CUR_DEN_BOT', 'type': 'passthrough', 'required': False}
                        ]
                    }
                },
                'required_features': self.REQUIRED_FEATURES,
                'enabled_groups': ['electrical', 'physical', 'encoding', 'residuals', 'new_metrics'],
                'output_behavior': {
                    'drop_intermediate': False,
                    'na_strategy': 'keep',
                    'duplicate_cols': 'keep_first'
                }
            }
        }

    def _print_generation_log(self):
        """打印特征生成日志"""
        print("\n==========================================")
        print("         [特征工程处理完成汇总]           ")
        print("==========================================")
        print(f"[生成的特征组] {self.generation_log['generated_groups']}")
        if self.generation_log['failed_groups']:
            print(f"[失败的特征组]")
            for failed in self.generation_log['failed_groups']:
                print(f"  - {failed['group']}: {failed['error']}")
        if self.generation_log['generation_errors']:
            print(f"[生成错误详情]")
            for error in self.generation_log['generation_errors']:
                print(f"  - {error['group']} ({error['type']}): {error['error']}")
        print("==========================================\n")



def main():
    """
    标准流水线：先读取数据/干净数据 -> 做特征工程 -> 保存结果
    """
    clean_data_path = "result/data/cleaned_data/cleaned_data.xlsx"
    featured_save_path = "result/data/feature_engineered_data/featured_data.xlsx"

    # 1. 检查清洗后的数据集是否存在，若不存在则尝试使用合并数据
    if os.path.exists(clean_data_path):
        print(f"[读取] 加载已清洗数据: {clean_data_path}")
        df_input = pd.read_excel(clean_data_path)
    else:
        print("[警告] 未找到清洗后的 cleaned_data.xlsx，尝试直接读取 merged_result_latest.xlsx")
        df_input = pd.read_excel("result/data/merged_data/merged_result_latest.xlsx")

    # 2. 执行特征工程
    fe = FeatureEngineer()
    featured_df = fe.transform(df_input)

    # 3. 保存特征工程后的数据集
    os.makedirs(os.path.dirname(featured_save_path), exist_ok=True)
    featured_df.to_excel(featured_save_path, index=False)

    print("\n==========================================")
    print("         [特征工程处理完成汇总]           ")
    print("==========================================")
    print(f"输入数据行数: {len(df_input)}")
    print(f"输出特征维度: {featured_df.shape[1]} 列")
    print(f"[导出提示] 特征工程数据集已保存至: {featured_save_path}")
    print("==========================================\n")


if __name__ == "__main__":
    main()