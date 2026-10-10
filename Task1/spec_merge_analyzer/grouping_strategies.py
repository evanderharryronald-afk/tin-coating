"""
【分组策略接口与实现】

定义了"分析单位"的抽象，支持多种分组维度（表面设定值档、规格组笛卡尔积、自定义规则）。
所有分组策略实现同一接口，确保 phase_a/b/c 的核心逻辑通用。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


class GroupingStrategy(ABC):
    """
    分组策略的抽象基类。
    
    所有具体策略必须实现：
    1. build_analysis_units() - 构建分析单位表
    2. describe() - 描述该策略
    3. get_setpoint_columns() - 返回设定值列名映射（用于 KS 检验）
    """

    @abstractmethod
    def build_analysis_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        从原始数据构建分析单位表。
        
        返回 DataFrame，列包括：
        - unit_key: str，唯一标识符（如 "Top2.799" 或 "Top2.799_Bot1.1"）
        - unit_label: str，人类可读标签
        - surface: str，所属表面（"Top"、"Bot" 或 "Both"）
        - n_samples: int，该单位包含的样本数
        - 其他维度列：如 setpoint_value、top_setpoint、bot_setpoint 等
        
        按 surface 和 unit_key 排序。
        """
        pass

    @abstractmethod
    def describe(self) -> str:
        """返回策略的描述文本"""
        pass

    @abstractmethod
    def get_setpoint_columns(self) -> Dict[str, str]:
        """
        返回 {surface: column_name} 映射，用于 KS 检验和距离计算。
        
        e.g., {"Top": "Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_TOP_Min", 
               "Bot": "Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_BOT_Min"}
        """
        pass

    @abstractmethod
    def get_other_surface(self) -> Dict[str, str]:
        """
        返回 {surface: other_surface} 映射，用于二维判定逻辑。
        
        e.g., {"Top": "Bot", "Bot": "Top"}
        对于非"两表面"的策略，可返回 {}
        """
        pass

    @abstractmethod
    def get_strategy_type(self) -> str:
        """返回策略类型标识"""
        pass

    @abstractmethod
    def get_analysis_layers(self) -> List[str]:
        """
        返回该策略支持的分析层级列表。
        
        e.g., ["layer1", "layer2", "layer3"] - 表面设定值档
        e.g., ["layer1"] - 规格组笛卡尔积（无层级，直接两两检验）
        """
        pass


class SurfaceSetpointLevelStrategy(GroupingStrategy):
    """
    【表面设定值档策略】v3
    
    按表面独立分档：Top 按 Top 设定值分档，Bot 按 Bot 设定值分档。
    档内不区分另一面设定值。
    
    分析单位：Top2.799、Top3.5、Bot2.0、Bot2.799 等（~20-30 个）
    分析层级：
      - layer1：档内组合能否并档（不同另一面设定值的组合）
      - layer2：档间能否合并
      - layer3：全局基线（参考用）
    """

    def __init__(self, config: Dict):
        """
        初始化策略。
        
        Args:
            config: Dict，包含列名定义，可以是:
                - 完整的列名字典 {'setpoint': {'top': ..., 'bot': ...}, 'delta': {...}}
                - 大小写 Top/Bot 键的字典 {'Top': ..., 'Bot': ...}
                - 小写 top/bot 键的字典 {'top': ..., 'bot': ...}
        """
        self.config = config
        
        # 调试输出
        # import logging
        # logger = logging.getLogger(__name__)
        # logger.info(f"[DEBUG] SurfaceSetpointLevelStrategy.__init__ config keys: {config.keys()}")
        # logger.info(f"[DEBUG] config: {config}")
        
        # 处理多种配置格式
        if 'setpoint' in config:
            # 完整格式 - setpoint 中可能使用 'Top'/'Bot'（大写）或 'top'/'bot'（小写）
            setpoint_cfg = config.get("setpoint", {})
            self.top_setpoint_col = setpoint_cfg.get("top") or setpoint_cfg.get("Top")
            self.bot_setpoint_col = setpoint_cfg.get("bot") or setpoint_cfg.get("Bot")
            
            delta_cfg = config.get("delta", {})
            self.top_delta_col = delta_cfg.get("top") or delta_cfg.get("Top") or "Top_Delta"
            self.bot_delta_col = delta_cfg.get("bot") or delta_cfg.get("Bot") or "Bot_Delta"
        elif 'Top' in config:
            # ConfigManager 返回的格式（大写 Top/Bot）
            self.top_setpoint_col = config.get("Top")
            self.bot_setpoint_col = config.get("Bot")
            self.top_delta_col = "Top_Delta"
            self.bot_delta_col = "Bot_Delta"
        elif 'top' in config:
            # 小写格式
            self.top_setpoint_col = config.get("top")
            self.bot_setpoint_col = config.get("bot")
            self.top_delta_col = "Top_Delta"
            self.bot_delta_col = "Bot_Delta"
        else:
            # 默认值
            self.top_setpoint_col = 'Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_TOP_Min'
            self.bot_setpoint_col = 'Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_BOT_Min'
            self.top_delta_col = "Top_Delta"
            self.bot_delta_col = "Bot_Delta"

    def build_analysis_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """按表面设定值分档"""
        rows = []
        for surface in ('Top', 'Bot'):
            own_col = self.top_setpoint_col if surface == 'Top' else self.bot_setpoint_col
            oth_col = self.bot_setpoint_col if surface == 'Top' else self.top_setpoint_col
            oth_surface = 'Bot' if surface == 'Top' else 'Top'
            
            sub = df[[own_col, oth_col]].dropna()
            
            for value, g in sub.groupby(own_col):
                comp = g[oth_col].value_counts().sort_values(ascending=False)
                comp_str = "; ".join(f"{oth_surface}{v}:{n}" for v, n in comp.items())
                
                unit_key = f"{surface}{value}"
                rows.append({
                    'unit_key': unit_key,
                    'unit_label': unit_key,
                    'surface': surface,
                    'setpoint_value': float(value),
                    'n_samples': int(len(g)),
                    'n_subgroups': int(len(comp)),
                    'composition': comp_str,  # 档内按另一面的构成
                })
        
        out = pd.DataFrame(rows).sort_values(['surface', 'setpoint_value'], ascending=[True, False])
        return out.reset_index(drop=True)

    def describe(self) -> str:
        return (
            "【表面设定值档策略 (v3)】\n"
            "  按表面独立分档：Top 按 Top 设定值分档，Bot 按 Bot 设定值分档\n"
            "  档内样本不区分另一面设定值\n"
            "  分析层级：layer1（档内并档）→ layer2（档间合并）→ layer3（全局基线参考）\n"
            "  预期单位数：~20-30 个（两个表面合计）\n"
            "  优点：样本量大，模型数少；缺点：丢失规格组级别的精度"
        )

    def get_setpoint_columns(self) -> Dict[str, str]:
        return {
            'Top': self.top_setpoint_col,
            'Bot': self.bot_setpoint_col,
        }

    def get_other_surface(self) -> Dict[str, str]:
        return {'Top': 'Bot', 'Bot': 'Top'}

    def get_strategy_type(self) -> str:
        return "surface_setpoint_level"

    def get_analysis_layers(self) -> List[str]:
        return ["layer1", "layer2", "layer3"]


class SpecGroupCartesianStrategy(GroupingStrategy):
    """
    【规格组笛卡尔积策略】v2
    
    分析单位：Top×Bot 的完整组合（如 Top2.799_Bot1.1、Top2.799_Bot2.799 等）
    样本总数：51 个（或根据数据中实际存在的组合数）
    
    分析层级：
      - layer1：直接两两检验（或按表面拆分后分别检验）
      无层1"档内并档"和层2"档间合并"的概念，只做一层规格组间对比
    """

    def __init__(self, config: Dict, split_by_surface: bool = True):
        """
        初始化策略。
        
        Args:
            config: Dict，包含列名定义，可以是:
                - 完整的列名字典 {'setpoint': {'top': ..., 'bot': ...}, 'delta': {...}}
                - 大小写 Top/Bot 键的字典 {'Top': ..., 'Bot': ...}
                - 小写 top/bot 键的字典 {'top': ..., 'bot': ...}
            split_by_surface: bool，是否分表面判定（Top 和 Bot 独立）
        """
        self.config = config
        
        # 处理多种配置格式
        if 'setpoint' in config:
            # 完整格式 - setpoint 中可能使用 'Top'/'Bot'（大写）或 'top'/'bot'（小写）
            setpoint_cfg = config.get("setpoint", {})
            self.top_setpoint_col = setpoint_cfg.get("top") or setpoint_cfg.get("Top")
            self.bot_setpoint_col = setpoint_cfg.get("bot") or setpoint_cfg.get("Bot")
            
            delta_cfg = config.get("delta", {})
            self.top_delta_col = delta_cfg.get("top") or delta_cfg.get("Top") or "Top_Delta"
            self.bot_delta_col = delta_cfg.get("bot") or delta_cfg.get("Bot") or "Bot_Delta"
        elif 'Top' in config:
            # ConfigManager 返回的格式（大写 Top/Bot）
            self.top_setpoint_col = config.get("Top")
            self.bot_setpoint_col = config.get("Bot")
            self.top_delta_col = "Top_Delta"
            self.bot_delta_col = "Bot_Delta"
        elif 'top' in config:
            # 小写格式
            self.top_setpoint_col = config.get("top")
            self.bot_setpoint_col = config.get("bot")
            self.top_delta_col = "Top_Delta"
            self.bot_delta_col = "Bot_Delta"
        else:
            # 默认值
            self.top_setpoint_col = 'Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_TOP_Min'
            self.bot_setpoint_col = 'Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_BOT_Min'
            self.top_delta_col = "Top_Delta"
            self.bot_delta_col = "Bot_Delta"
        
        self.split_by_surface = split_by_surface

    def build_analysis_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """按 Top×Bot 笛卡尔积构建分析单位"""
        rows = []
        
        sub = df[[self.top_setpoint_col, self.bot_setpoint_col]].dropna()
        for (top_val, bot_val), g in sub.groupby([self.top_setpoint_col, self.bot_setpoint_col]):
            unit_key = f"Top{top_val}_Bot{bot_val}"
            rows.append({
                'unit_key': unit_key,
                'unit_label': unit_key,
                'surface': 'Both',  # 规格组涉及两个表面
                'top_setpoint': float(top_val),
                'bot_setpoint': float(bot_val),
                'n_samples': int(len(g)),
            })
        
        out = pd.DataFrame(rows).sort_values(['top_setpoint', 'bot_setpoint'])
        return out.reset_index(drop=True)

    def describe(self) -> str:
        split_note = "（分表面判定）" if self.split_by_surface else "（共用判定）"
        return (
            "【规格组笛卡尔积策略 (v2)】\n"
            f"  分析单位：Top×Bot 完整组合{split_note}\n"
            "  预期单位数：~51 个（或根据数据中实际存在的组合）\n"
            "  分析层级：layer1（规格组两两检验）\n"
            "  优点：精细判定，规格组级别清晰；缺点：单位多，样本量分散"
        )

    def get_setpoint_columns(self) -> Dict[str, str]:
        return {
            'Top': self.top_setpoint_col,
            'Bot': self.bot_setpoint_col,
        }

    def get_other_surface(self) -> Dict[str, str]:
        return {'Top': 'Bot', 'Bot': 'Top'}

    def get_strategy_type(self) -> str:
        return "spec_group_cartesian"

    def get_analysis_layers(self) -> List[str]:
        return ["layer1"]  # 单层，无递进关系


class CustomStrategy(GroupingStrategy):
    """
    【自定义分组策略】
    
    用户可以自定义分组规则，通过传入一个回调函数。
    用于支持其他业务规则（如按最大设定值分档、按功能分组等）。
    """

    def __init__(self, 
                 name: str,
                 description: str,
                 build_func,
                 setpoint_cols: Dict[str, str],
                 other_surface_map: Dict[str, str] = None,
                 analysis_layers: List[str] = None):
        """
        初始化自定义策略。
        
        Args:
            name: 策略名称
            description: 策略描述
            build_func: 回调函数，签名为 (df) -> pd.DataFrame
                       返回包含 unit_key, unit_label, surface, n_samples 的 DataFrame
            setpoint_cols: 设定值列名映射
            other_surface_map: 表面映射（可选）
            analysis_layers: 分析层级列表（可选，默认 ["layer1"]）
        """
        self.name = name
        self._description = description
        self.build_func = build_func
        self._setpoint_cols = setpoint_cols
        self._other_surface_map = other_surface_map or {}
        self._analysis_layers = analysis_layers or ["layer1"]

    def build_analysis_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """调用用户提供的回调函数"""
        return self.build_func(df)

    def describe(self) -> str:
        return self._description

    def get_setpoint_columns(self) -> Dict[str, str]:
        return self._setpoint_cols

    def get_other_surface(self) -> Dict[str, str]:
        return self._other_surface_map

    def get_strategy_type(self) -> str:
        return "custom"

    def get_analysis_layers(self) -> List[str]:
        return self._analysis_layers


def create_strategy(strategy_type: str, config: Dict) -> GroupingStrategy:
    """
    工厂函数：根据类型创建分组策略实例。
    
    Args:
        strategy_type: "surface_setpoint_level" | "spec_group_cartesian" | "custom"
        config: 配置字典（通常来自 grouping_config.yaml）
    
    Returns:
        GroupingStrategy 的具体实现
    
    Raises:
        ValueError: 不支持的策略类型
    """
    if strategy_type == "surface_setpoint_level":
        return SurfaceSetpointLevelStrategy(config)
    elif strategy_type == "spec_group_cartesian":
        split_by_surface = config.get("split_by_surface", True)
        return SpecGroupCartesianStrategy(config, split_by_surface=split_by_surface)
    else:
        raise ValueError(f"不支持的分组策略类型: {strategy_type}")
