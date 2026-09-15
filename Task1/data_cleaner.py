import os
import yaml
import pandas as pd
import numpy as np
from scipy.stats import median_abs_deviation
from typing import Dict, List, Optional


class SteelDataCleaner:
    """
    电镀锡生产数据通用清洗与异常诊断模块
    只负责数据的质量诊断、过滤与异常标记，不包含任何特征工程衍生逻辑
    
    支持配置驱动的规则引擎，兼容旧数据和新数据
    """

    def __init__(self, rules_config: Optional[Dict] = None,
                 min_speed=20.0,
                 max_range_abs=0.5,     # Max与Min的最大绝对允许差值 (g/m2)
                 max_range_ratio=0.4,   # (Max - Min) / Avg 的最大允许波动比例
                 mad_factor=3.0):       # 残差离群点 MAD 倍数
        """
        初始化数据清洗器
        
        params:
        - rules_config: 配置字典（如为 None，则使用默认旧规则）
        - min_speed, max_range_abs, max_range_ratio, mad_factor: 向后兼容参数
        """
        self.rules_config = rules_config
        self.min_speed = min_speed
        self.max_range_abs = max_range_abs
        self.max_range_ratio = max_range_ratio
        self.mad_factor = mad_factor
        self.execution_log = {}

    def process(self, df,
                active_rule_set: str = 'old_data',
                clean_save_path="result/cleaned_data/cleaned_data.xlsx",
                filtered_save_path="result/cleaned_data/filtered_outliers.xlsx"):
        """
        执行数据清洗
        
        params:
        - df: 原始数据
        - active_rule_set: 激活的规则集（'old_data' 或 'new_data'）
        - clean_save_path: 干净数据保存路径
        - filtered_save_path: 被剔除数据保存路径
        """
        df = df.copy()
        self.execution_log = {}
        
        # 获取规则集
        if self.rules_config is not None:
            # 使用配置中的规则
            rules = self.rules_config['data_cleaning']['rule_sets'][active_rule_set]['rules']
        else:
            # 向后兼容：使用默认旧规则（原有逻辑）
            rules = self._get_default_old_rules()
        
        # 初始化剔除标记列
        df['Filter_Reason'] = ""
        initial_count = len(df)

        # 执行每条规则
        self.execution_log['executed_rules'] = []
        self.execution_log['skipped_rules'] = []

        for rule in rules:
            if not rule.get('active', True):
                continue
            
            try:
                df = self._apply_rule(df, rule)
                self.execution_log['executed_rules'].append(rule['name'])
            except KeyError as e:
                # 字段缺失时记录而非崩溃
                self.execution_log['skipped_rules'].append({
                    'name': rule['name'],
                    'reason': f'Field not found: {str(e)}'
                })
            except Exception as e:
                self.execution_log['skipped_rules'].append({
                    'name': rule['name'],
                    'reason': f'Error: {str(e)}'
                })

        # 拆分数据
        filtered_df = df[df['Filter_Reason'] != ""].copy()
        clean_df = df[df['Filter_Reason'] == ""].copy()

        # 导出 Excel 文件
        if clean_save_path or filtered_save_path:
            os.makedirs(os.path.dirname(clean_save_path), exist_ok=True)

            cols_to_export = [
                'Coil ID', 'Steel Grade', 'Speed[m/min]_Process_Avg',
                'Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Avg',
                'Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Max',
                'Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Min',
                'Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Avg',
                'Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Max',
                'Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Min',
                'Filter_Reason'
            ]
            cols_to_export = [c for c in cols_to_export if c in filtered_df.columns]

            filtered_df[cols_to_export].to_excel(filtered_save_path, index=False)
            clean_df.to_excel(clean_save_path, index=False)

        self._print_summary(initial_count, len(filtered_df), len(clean_df))
        self._print_execution_log()

        return clean_df

    def _apply_rule(self, df: pd.DataFrame, rule: Dict) -> pd.DataFrame:
        """根据规则类型应用对应逻辑"""
        rule_type = rule['type']
        
        if rule_type == 'required_fields':
            return self._check_required_fields(df, rule)
        elif rule_type == 'threshold':
            return self._check_threshold(df, rule)
        elif rule_type == 'zero_value':
            return self._check_zero_value(df, rule)
        elif rule_type == 'instability':
            return self._check_instability(df, rule)
        elif rule_type == 'mad_based':
            return self._check_mad_residuals(df, rule)
        elif rule_type == 'range':
            return self._check_range(df, rule)
        else:
            raise ValueError(f"Unknown rule type: {rule_type}")

    def _check_required_fields(self, df: pd.DataFrame, rule: Dict) -> pd.DataFrame:
        """检查必需字段"""
        required_fields = rule['fields']
        skip_if_missing = rule.get('skip_if_missing', [])
        
        # 可用字段 = 所有字段 - 允许缺失的字段
        check_fields = [f for f in required_fields if f not in skip_if_missing and f in df.columns]
        
        if check_fields:
            null_mask = df[check_fields].isnull().any(axis=1)
            df.loc[null_mask, 'Filter_Reason'] += f"{rule['reason']}; "
        
        return df

    def _check_threshold(self, df: pd.DataFrame, rule: Dict) -> pd.DataFrame:
        """阈值检查（单个字段）"""
        field = rule['field']
        if field not in df.columns:
            raise KeyError(field)
        
        operator = rule['operator']
        threshold = rule['threshold']
        
        if operator == '<=':
            mask = df[field] <= threshold
        elif operator == '<':
            mask = df[field] < threshold
        elif operator == '>=':
            mask = df[field] >= threshold
        elif operator == '>':
            mask = df[field] > threshold
        elif operator == '==':
            mask = df[field] == threshold
        elif operator == '!=':
            mask = df[field] != threshold
        else:
            raise ValueError(f"Unknown operator: {operator}")
        
        df.loc[mask, 'Filter_Reason'] += f"{rule['reason']}; "
        return df

    def _check_zero_value(self, df: pd.DataFrame, rule: Dict) -> pd.DataFrame:
        """零值检查"""
        check_pairs = rule['check_pairs']
        
        for pair in check_pairs:
            field = pair['field']
            surface = pair.get('surface', '')
            
            if field not in df.columns:
                continue
            
            zero_mask = df[field] <= 0
            df.loc[zero_mask, 'Filter_Reason'] += f"{surface}在线仪表零值/死值异常; "
        
        return df

    def _check_instability(self, df: pd.DataFrame, rule: Dict) -> pd.DataFrame:
        """仪表波动检查"""
        pairs = rule['pairs']
        max_range_abs = rule.get('max_range_abs', self.max_range_abs)
        max_range_ratio = rule.get('max_range_ratio', self.max_range_ratio)
        
        for pair in pairs:
            avg_col = pair['avg_col']
            min_col = pair['min_col']
            max_col = pair['max_col']
            surface = pair.get('surface', '')
            
            if not all(c in df.columns for c in [avg_col, min_col, max_col]):
                continue
            
            range_val = df[max_col] - df[min_col]
            range_ratio = range_val / (df[avg_col] + 1e-5)
            
            unstable_mask = (range_val > max_range_abs) | (range_ratio > max_range_ratio)
            df.loc[unstable_mask, 'Filter_Reason'] += f"{surface}在线仪表波动过大(Max/Min极差超标); "
        
        return df

    def _check_range(self, df: pd.DataFrame, rule: Dict) -> pd.DataFrame:
        """范围检查（新数据特有指标）"""
        field = rule['field']
        if field not in df.columns:
            return df  # 字段不存在，跳过
        
        min_val = rule.get('min')
        max_val = rule.get('max')
        operator = rule['operator']
        
        if operator == 'outside':
            mask = (df[field] < min_val) | (df[field] > max_val)
        elif operator == 'inside':
            mask = (df[field] >= min_val) & (df[field] <= max_val)
        else:
            raise ValueError(f"Unknown operator: {operator}")
        
        df.loc[mask, 'Filter_Reason'] += f"{rule['reason']}; "
        return df

    def _check_mad_residuals(self, df: pd.DataFrame, rule: Dict) -> pd.DataFrame:
        """基于 MAD 的残差极值诊断"""
        residual_pairs = rule['residual_pairs']
        mad_factor = rule.get('mad_factor', self.mad_factor)
        
        # 仅在有效样本上统计
        valid_mask = df['Filter_Reason'] == ""
        if valid_mask.sum() == 0:
            return df
        
        for pair in residual_pairs:
            measurement_col = pair['measurement']
            online_col = pair['online']
            surface = pair.get('surface', '')
            
            if measurement_col not in df.columns or online_col not in df.columns:
                continue
            
            delta = df[measurement_col] - df[online_col]
            delta_valid = delta[valid_mask]
            
            if delta_valid.isnull().any():
                delta_valid = delta_valid.dropna()
            
            if len(delta_valid) < 3:  # 需要至少 3 个有效值
                continue
            
            delta_center = delta_valid.median()
            delta_mad = median_abs_deviation(delta_valid, scale='normal')
            threshold = mad_factor * delta_mad
            
            outlier_mask = valid_mask & ((delta - delta_center).abs() > threshold)
            df.loc[outlier_mask, 'Filter_Reason'] += f"{surface}残差异常(>{threshold:.2f}g/m2); "
        
        return df

    def _get_default_old_rules(self) -> List[Dict]:
        """
        返回默认的旧规则集（向后兼容）
        这是原有逻辑的规则表示
        """
        return [
            {
                'name': 'required_fields_check',
                'type': 'required_fields',
                'fields': [
                    'Speed[m/min]_Process_Avg', 'Dimension_[mm]_Width', 'Dimension_[mm]_Thickness',
                    'Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Avg', 'Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Avg',
                    '上表面镀层重量A(XA1_0)', '下表面镀层重量A(XA1_0)'
                ],
                'active': True,
                'reason': '关键工艺/测量参数缺失',
                'skip_if_missing': []
            },
            {
                'name': 'low_speed_check',
                'type': 'threshold',
                'field': 'Speed[m/min]_Process_Avg',
                'operator': '<=',
                'threshold': self.min_speed,
                'active': True,
                'reason': f'车速低于{self.min_speed}m/min(停机或过渡区)'
            },
            {
                'name': 'zero_value_check',
                'type': 'zero_value',
                'check_pairs': [
                    {'field': 'Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Min', 'surface': '上表面'},
                    {'field': 'Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Min', 'surface': '下表面'}
                ],
                'active': True,
                'reason': '仪表零值/死值异常'
            },
            {
                'name': 'instability_check',
                'type': 'instability',
                'pairs': [
                    {
                        'avg_col': 'Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Avg',
                        'min_col': 'Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Min',
                        'max_col': 'Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Max',
                        'surface': '上表面'
                    },
                    {
                        'avg_col': 'Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Avg',
                        'min_col': 'Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Min',
                        'max_col': 'Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Max',
                        'surface': '下表面'
                    }
                ],
                'max_range_abs': self.max_range_abs,
                'max_range_ratio': self.max_range_ratio,
                'active': True,
                'reason': '在线仪表波动过大(Max/Min极差超标)'
            },
            {
                'name': 'residual_mad_check',
                'type': 'mad_based',
                'residual_pairs': [
                    {
                        'measurement': '上表面镀层重量A(XA1_0)',
                        'online': 'Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Avg',
                        'surface': '上表面'
                    },
                    {
                        'measurement': '下表面镀层重量A(XA1_0)',
                        'online': 'Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Avg',
                        'surface': '下表面'
                    }
                ],
                'mad_factor': self.mad_factor,
                'active': True,
                'reason': '残差异常(MAD-based离群检测)'
            }
        ]

    def _print_summary(self, initial_count: int, filtered_count: int, clean_count: int):
        """打印清洗汇总"""
        print("\n==========================================")
        print("        [数据清洗与异常诊断汇总]          ")
        print("==========================================")
        print(f"原始数据总行数: {initial_count}")
        print(f"被剔除异常点数: {filtered_count} (占比: {filtered_count / initial_count * 100:.2f}%)")
        print(f"保留干净样本数: {clean_count}")
        print("==========================================\n")

    def _print_execution_log(self):
        """打印执行日志"""
        print(f"[执行的规则] {self.execution_log.get('executed_rules', [])}")
        if self.execution_log.get('skipped_rules'):
            print(f"[跳过的规则]")
            for skipped in self.execution_log['skipped_rules']:
                print(f"  - {skipped['name']}: {skipped['reason']}")



if __name__ == "__main__":
    raw_df = pd.read_excel("result/data/merged_data/merged_result_latest.xlsx")
    cleaner = SteelDataCleaner()
    clean_df = cleaner.process(
        raw_df,
        clean_save_path="result/data/cleaned_data/cleaned_data.xlsx",
        filtered_save_path="result/data/cleaned_data/filtered_outliers.xlsx"
    )