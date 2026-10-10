"""
【配置管理模块】

负责加载、验证和管理 grouping_config.yaml 配置文件。
提供统一的配置查询接口，隔离对原始配置的直接访问。
"""

import os
import yaml
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """配置管理器：加载、验证、查询 YAML 配置"""

    def __init__(self, config_path: str = "grouping_config.yaml"):
        """
        初始化配置管理器。
        
        Args:
            config_path: 配置文件路径（相对或绝对）
        
        Raises:
            FileNotFoundError: 配置文件不存在
            yaml.YAMLError: 配置文件格式错误
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()

    def _load_config(self) -> Dict:
        """加载 YAML 配置文件"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"配置文件不存在: {self.config_path}\n"
                f"请确保 grouping_config.yaml 在当前目录或指定的路径中"
            )
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"✓ 配置文件已加载: {self.config_path}")
            return config
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"配置文件格式错误: {e}")

    def _validate_config(self) -> None:
        """验证配置的完整性"""
        required_sections = [
            'project', 'data', 'columns', 'grouping_strategies', 
            'analysis_parameters', 'output', 'runtime'
        ]
        for section in required_sections:
            if section not in self.config:
                logger.warning(f"⚠ 配置缺少必要部分: {section}")

    # ============================================================
    # 数据路径查询
    # ============================================================
    def get_input_file(self) -> str:
        """获取输入数据文件路径"""
        return self.config.get('data', {}).get('input_file', 
                                              'result/data/feature_engineered_data/featured_data.xlsx')

    def get_output_dir(self) -> str:
        """获取输出目录路径"""
        return self.config.get('data', {}).get('output_dir', 
                                              'result/spec_group_merge_analysis')

    # ============================================================
    # 列名查询
    # ============================================================
    def get_column_name(self, section: str, key: str, surface: Optional[str] = None) -> str:
        """
        获取指定列的名称。
        
        Args:
            section: "setpoint" | "actual_weight" | "delta" 等
            key: 键名（通常为子键如 "top"、"bot"）
            surface: 可选，如果 section 支持 surface 维度
        
        Returns:
            列名字符串
        
        Raises:
            KeyError: 列名不存在
        """
        cols = self.config.get('columns', {})
        
        if section not in cols:
            raise KeyError(f"列配置中不存在部分: {section}")
        
        section_config = cols[section]
        
        if isinstance(section_config, dict):
            if key not in section_config:
                raise KeyError(f"列配置中不存在: columns.{section}.{key}")
            return section_config[key]
        else:
            raise KeyError(f"columns.{section} 的格式不正确")

    def get_setpoint_columns(self) -> Dict[str, str]:
        """获取 Top 和 Bot 的设定值列名映射"""
        return {
            'Top': self.get_column_name('setpoint', 'top'),
            'Bot': self.get_column_name('setpoint', 'bot'),
        }

    def get_delta_columns(self) -> Dict[str, str]:
        """获取 Top 和 Bot 的误差列名映射"""
        return {
            'Top': self.get_column_name('delta', 'top'),
            'Bot': self.get_column_name('delta', 'bot'),
        }

    def get_feature_columns(self) -> List[str]:
        """获取特征列列表"""
        return self.config.get('columns', {}).get('features', [])

    # ============================================================
    # 分组策略查询
    # ============================================================
    def get_enabled_strategies(self) -> List[str]:
        """获取启用的分组策略列表"""
        strategies = self.config.get('grouping_strategies', {})
        return [name for name, cfg in strategies.items() if cfg.get('enabled', False)]

    def get_strategy_config(self, strategy_name: str) -> Dict:
        """获取指定策略的配置"""
        strategies = self.config.get('grouping_strategies', {})
        if strategy_name not in strategies:
            raise KeyError(f"策略不存在: {strategy_name}")
        return strategies[strategy_name]

    def get_strategy_type(self, strategy_name: str) -> str:
        """获取策略的类型"""
        strategy_config = self.get_strategy_config(strategy_name)
        return strategy_config.get('type', 'unknown')

    def get_all_strategy_configs(self) -> Dict[str, Dict]:
        """获取所有启用的策略配置"""
        strategies = self.config.get('grouping_strategies', {})
        return {name: cfg for name, cfg in strategies.items() if cfg.get('enabled', False)}

    # ============================================================
    # 分析参数查询
    # ============================================================
    def get_analysis_parameter(self, key: str, default: Any = None) -> Any:
        """
        获取分析参数。
        
        Args:
            key: 参数键，支持嵌套（如 "spec_distance.absolute_limit"）
            default: 默认值
        
        Returns:
            参数值或默认值
        """
        params = self.config.get('analysis_parameters', {})
        
        if '.' in key:
            # 支持嵌套键查询
            keys = key.split('.')
            value = params
            for k in keys:
                if isinstance(value, dict):
                    value = value.get(k)
                else:
                    return default
            return value if value is not None else default
        else:
            return params.get(key, default)

    def get_all_analysis_parameters(self) -> Dict:
        """获取所有分析参数"""
        return self.config.get('analysis_parameters', {})

    # ============================================================
    # 输出配置查询
    # ============================================================
    def should_organize_by_strategy(self) -> bool:
        """是否按策略组织输出目录"""
        return self.config.get('output', {}).get('organize_by_strategy', True)

    def should_generate_comparison_report(self) -> bool:
        """是否生成对比分析报告"""
        return self.config.get('output', {}).get('generate_comparison_report', False)

    def get_comparison_report_file(self) -> str:
        """获取对比报告的文件名"""
        return self.config.get('output', {}).get('comparison_report_file', 
                                               'comparison_report.md')

    # ============================================================
    # 运行配置查询
    # ============================================================
    def is_verbose(self) -> bool:
        """是否启用详细日志"""
        return self.config.get('runtime', {}).get('verbose', True)

    def get_log_level(self) -> str:
        """获取日志级别"""
        return self.config.get('runtime', {}).get('log_level', 'INFO')

    def get_n_jobs(self) -> int:
        """获取并行进程数"""
        return self.config.get('runtime', {}).get('n_jobs_strategy', -1)

    # ============================================================
    # 便捷方法
    # ============================================================
    def get_columns_for_strategy(self, strategy_name: str) -> Dict:
        """
        获取指定策略需要的所有列名。
        
        Returns:
            {'setpoint': {...}, 'delta': {...}, 'features': [...]}
        """
        return {
            'setpoint': self.get_setpoint_columns(),
            'delta': self.get_delta_columns(),
            'features': self.get_feature_columns(),
        }

    def print_config_summary(self) -> None:
        """打印配置摘要"""
        print("\n" + "=" * 80)
        print("【配置文件摘要】")
        print("=" * 80)
        
        project = self.config.get('project', {})
        print(f"\n项目: {project.get('name', 'N/A')}")
        print(f"版本: {project.get('version', 'N/A')}")
        
        print(f"\n数据文件: {self.get_input_file()}")
        print(f"输出目录: {self.get_output_dir()}")
        
        enabled = self.get_enabled_strategies()
        print(f"\n启用的分组策略 ({len(enabled)}):")
        for strategy in enabled:
            cfg = self.get_strategy_config(strategy)
            print(f"  - {strategy}: {cfg.get('description', 'N/A')}")
        
        print(f"\n分析参数摘要:")
        print(f"  - 样本量门槛（正式）: {self.get_analysis_parameter('min_formal_samples')} 行")
        print(f"  - 设定值绝对差限: {self.get_analysis_parameter('spec_distance.absolute_limit')} g/m2")
        print(f"  - 非劣效边界（汇总）: {self.get_analysis_parameter('non_inferiority.summary_margin')*100:.1f}%")
        print(f"  - 非劣效边界（单位）: {self.get_analysis_parameter('non_inferiority.per_unit_margin')*100:.1f}%")
        
        print(f"\n运行配置:")
        print(f"  - 日志级别: {self.get_log_level()}")
        print(f"  - 并行进程: {self.get_n_jobs()}")
        
        print("\n" + "=" * 80 + "\n")
