"""
Spec Merge Analyzer - 规格组合并可行性分析包 (v3.0)

评估不同镀锡规格能否合并建立统一的校准模型。
支持多种分组维度：表面设定值档、规格组笛卡尔积、自定义规则。

【架构】
- grouping_strategies: 分组策略接口与实现（表面设定值档、规格组笛卡尔积、自定义）
- config_manager: 配置加载与管理（从 grouping_config.yaml 读取）
- phase_a_ks_test_screening: 设定值分档统计 + KS 检验（通用逻辑）
- phase_b_chow_test_validation: 交叉验证 + 非劣效判定（通用逻辑）
- phase_c_graph_grouping: 团划分分组（通用逻辑）

【典型用法】
    from spec_merge_analyzer import run_analysis_with_strategy
    
    # 使用表面设定值档策略
    success = run_analysis_with_strategy(
        strategy_name='surface_setpoint_level',
        phase='all'
    )
    
    # 使用规格组笛卡尔积策略
    success = run_analysis_with_strategy(
        strategy_name='spec_group_cartesian',
        phase='all'
    )
    
    # 或直接运行所有启用的策略
    success = run_all_enabled_strategies(phase='all')
"""

__version__ = '3.0.0'
__author__ = 'Tin Coating Analysis Team'

import os
import time
import logging
from typing import Dict, Optional
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# 导入核心模块
from . import phase_a_ks_test_screening as phase_a
from . import phase_b_chow_test_validation as phase_b
from . import phase_c_graph_grouping as phase_c
from .config_manager import ConfigManager
from .grouping_strategies import create_strategy, GroupingStrategy

# ============================================================
# 全局配置管理
# ============================================================

config = None
logger = logging.getLogger(__name__)
_log_handler_initialized = False


def init_config(config_path: str = "grouping_config.yaml") -> ConfigManager:
    """初始化全局配置"""
    global config
    config = ConfigManager(config_path)
    return config


def get_config() -> ConfigManager:
    """获取全局配置（确保已初始化）"""
    global config
    if config is None:
        config = ConfigManager()
    return config


def _init_logging():
    """初始化日志"""
    global _log_handler_initialized
    if _log_handler_initialized:
        return
    
    cfg = get_config()
    log_level = getattr(logging, cfg.get_log_level(), logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)-8s %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    logger.setLevel(log_level)
    _log_handler_initialized = True


def get_result_dir(strategy_name: Optional[str] = None) -> str:
    """
    获取结果输出目录。
    
    Args:
        strategy_name: 策略名称（可选）。如果指定且 organize_by_strategy=True，
                      返回策略专用的子目录
    
    Returns:
        输出目录路径
    """
    cfg = get_config()
    base_dir = cfg.get_output_dir()
    
    if strategy_name and cfg.should_organize_by_strategy():
        return os.path.join(base_dir, strategy_name)
    
    return base_dir


def ensure_result_dir(strategy_name: Optional[str] = None) -> str:
    """确保结果目录存在"""
    result_dir = get_result_dir(strategy_name)
    os.makedirs(result_dir, exist_ok=True)
    return result_dir


def _get_phase_inputs(result_dir: str) -> tuple:
    """获取阶段间依赖的输入文件列表"""
    phase_b_inputs = [
        f"{result_dir}/merge_candidates.csv",
        f"{result_dir}/level_composition.csv",
    ]
    # Phase C 依赖：merge_validation_results.json（Phase B 生成）
    # 不再需要 level_pooling_results.json（已移除）
    phase_c_inputs = [
        f"{result_dir}/merge_validation_results.json",
        f"{result_dir}/level_composition.csv",
    ]
    return phase_b_inputs, phase_c_inputs


def _missing(paths):
    """检查路径列表中哪些不存在"""
    return [p for p in paths if not os.path.exists(p)]


def _not_refreshed(paths, since):
    """返回在 since 时刻之后没有被重新写入的文件"""
    return [p for p in paths if (not os.path.exists(p)) or os.path.getmtime(p) < since - 1]


# ============================================================
# 支持多策略的流水线
# ============================================================

def run_phase_with_strategy(phase: str, strategy, strategy_name: str) -> bool:
    """
    使用指定策略运行某个分析阶段。
    
    Args:
        phase: 'a' | 'b' | 'c' | 'all'
        strategy: GroupingStrategy 实例
        strategy_name: 策略名称（用于日志和输出目录）
    
    Returns:
        bool: 该策略的流水线是否成功
    """
    _init_logging()
    cfg = get_config()
    result_dir = ensure_result_dir(strategy_name)
    phase_b_inputs, phase_c_inputs = _get_phase_inputs(result_dir)
    
    try:
        if phase in ('a', 'all'):
            logger.info(f"【{strategy_name}】执行阶段 A...")
            success_a = phase_a.main(strategy, cfg, result_dir)
            if not success_a:
                logger.error(f"【{strategy_name}】阶段 A 失败")
                return False
            logger.info(f"✓ 阶段 A 完成")
        
        if phase in ('b', 'all'):
            missing = _missing(phase_b_inputs)
            if missing:
                logger.warning(f"【{strategy_name}】阶段 B 依赖缺失，跳过")
                logger.warning(f"  缺失: {missing}")
                # Phase B/C 不是必须的 - 只要 Phase A 成功就够了
                if phase == 'all':
                    return True
                return False
            
            logger.info(f"【{strategy_name}】执行阶段 B...")
            t_start = time.time()
            success_b = phase_b.main(strategy, cfg, result_dir)
            if not success_b:
                logger.error(f"【{strategy_name}】阶段 B 失败")
                if phase == 'all':
                    return True  # Phase A 成功，B失败不影响整体
                return False
            
            if _not_refreshed(phase_c_inputs, t_start):
                logger.warning(f"【{strategy_name}】阶段 B 未生成预期输出")
            logger.info(f"✓ 阶段 B 完成")
        
        if phase in ('c', 'all'):
            missing = _missing(phase_c_inputs)
            if missing:
                logger.warning(f"【{strategy_name}】阶段 C 依赖缺失，跳过")
                logger.warning(f"  缺失: {missing}")
                return True  # Phase A/B 成功即可
            
            logger.info(f"【{strategy_name}】执行阶段 C...")
            success_c = phase_c.main(strategy, cfg, result_dir)
            if not success_c:
                logger.error(f"【{strategy_name}】阶段 C 失败")
                return True  # Phase A/B 成功即可
            logger.info(f"✓ 阶段 C 完成")
        
        return True
    
    except Exception as e:
        logger.error(f"【{strategy_name}】执行失败: {e}", exc_info=True)
        return False


def run_analysis_with_strategy(strategy_name: str, phase: str = 'all', 
                               config_path: str = "grouping_config.yaml") -> bool:
    """
    使用指定的分组策略运行完整分析流水线。
    
    Args:
        strategy_name: 策略名称（如 'surface_setpoint_level' 或 'spec_group_cartesian'）
        phase: 执行的阶段（'a' | 'b' | 'c' | 'all'）
        config_path: 配置文件路径
    
    Returns:
        bool: 流水线是否成功
    
    Raises:
        KeyError: 策略不存在或不启用
    """
    # 初始化配置
    global config
    if config is None:
        init_config(config_path)
    
    _init_logging()
    
    try:
        # 检查策略是否存在且启用
        strategy_config = get_config().get_strategy_config(strategy_name)
        if not strategy_config.get('enabled', False):
            logger.warning(f"策略 '{strategy_name}' 未启用，请在 grouping_config.yaml 中检查配置")
            return False
        
        # 创建策略实例
        cols_config = get_config().get_columns_for_strategy(strategy_name)
        strategy_type = get_config().get_strategy_type(strategy_name)
        
        strategy = create_strategy(strategy_type, cols_config)
        
        # 打印策略信息
        logger.info("\n" + "=" * 80)
        logger.info(f"【策略】{strategy_name}")
        logger.info("=" * 80)
        logger.info(strategy.describe())
        
        # 运行流水线
        success = run_phase_with_strategy(phase, strategy, strategy_name)
        
        if success:
            logger.info("\n" + "=" * 80)
            logger.info(f"✓ 【{strategy_name}】流水线执行完成")
            logger.info("=" * 80)
        else:
            logger.error(f"\n✗ 【{strategy_name}】流水线执行失败")
        
        return success
    
    except Exception as e:
        logger.error(f"策略 '{strategy_name}' 初始化失败: {e}", exc_info=True)
        return False


def run_all_enabled_strategies(phase: str = 'all', 
                                config_path: str = "grouping_config.yaml") -> Dict[str, bool]:
    """
    运行所有启用的分组策略。
    
    Args:
        phase: 执行的阶段（'a' | 'b' | 'c' | 'all'）
        config_path: 配置文件路径
    
    Returns:
        Dict[strategy_name -> bool]：每个策略的执行结果
    """
    # 初始化配置
    global config
    if config is None:
        init_config(config_path)
    
    _init_logging()
    
    enabled_strategies = get_config().get_enabled_strategies()
    logger.info(f"\n共 {len(enabled_strategies)} 个启用的分组策略：{', '.join(enabled_strategies)}\n")
    
    results = {}
    for strategy_name in enabled_strategies:
        logger.info("")
        success = run_analysis_with_strategy(strategy_name, phase, config_path)
        results[strategy_name] = success
    
    # 总结
    logger.info("\n" + "=" * 80)
    logger.info("【所有策略执行完成】")
    logger.info("=" * 80)
    for strategy_name, success in results.items():
        status = "✓ 成功" if success else "✗ 失败"
        logger.info(f"  {strategy_name}: {status}")
    logger.info("=" * 80)
    
    return results


# ============================================================
# 向后兼容：保留原有接口
# ============================================================

def run_analysis(phase='all', skip_c=False):
    """
    【向后兼容】运行规格组合并分析流水线（使用默认配置）
    
    注意：此接口仅用于向后兼容。建议使用 run_analysis_with_strategy() 或 run_all_enabled_strategies()。
    
    参数：
        phase (str): 执行的阶段（'a' | 'b' | 'c' | 'all'）
        skip_c (bool): 是否跳过阶段 C
    
    返回：
        bool: 流水线是否成功执行
    """
    if phase not in ['a', 'b', 'c', 'all']:
        logger.error(f"✗ 不支持的阶段: {phase}")
        return False
    
    if skip_c and phase == 'all':
        phase_to_run = 'b'
    else:
        phase_to_run = phase
    
    # 使用表面设定值档策略作为默认
    return run_analysis_with_strategy('surface_setpoint_level', phase_to_run)


def print_summary():
    """打印全局配置摘要"""
    _init_logging()
    cfg = get_config()
    cfg.print_config_summary()


__all__ = [
    'init_config',
    'get_config',
    'run_analysis_with_strategy',
    'run_all_enabled_strategies',
    'run_analysis',
    'print_summary',
]
