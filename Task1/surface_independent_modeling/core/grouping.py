"""
按表面厚度分组的核心模块
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

from .constants import (
    TOP_SETPOINT_COL, BOT_SETPOINT_COL, TIME_COL,
    MIN_GROUP_SAMPLES
)

logger = logging.getLogger(__name__)


def build_surface_groups(df: pd.DataFrame) -> Dict[str, Tuple[str, float, pd.DataFrame]]:
    """
    按表面厚度独立分组
    
    Parameters
    ----------
    df : DataFrame
        特征工程后的数据
    
    Returns
    -------
    groups : dict
        键: 'Top_{value}' 或 'Bot_{value}'
        值: (surface, value, group_df)
    """
    groups = {}
    
    # 按上表面分组
    top_values = sorted(df[TOP_SETPOINT_COL].unique())
    for top_val in top_values:
        top_group_df = df[df[TOP_SETPOINT_COL] == top_val].copy()
        n_samples = len(top_group_df)
        
        # 保持时间顺序
        if TIME_COL in top_group_df.columns:
            top_group_df = top_group_df.sort_values(TIME_COL).reset_index(drop=True)
        else:
            top_group_df = top_group_df.reset_index(drop=True)
        
        group_key = f'Top_{top_val}'
        groups[group_key] = ('Top', top_val, top_group_df, n_samples)
    
    # 按下表面分组
    bot_values = sorted(df[BOT_SETPOINT_COL].unique())
    for bot_val in bot_values:
        bot_group_df = df[df[BOT_SETPOINT_COL] == bot_val].copy()
        n_samples = len(bot_group_df)
        
        # 保持时间顺序
        if TIME_COL in bot_group_df.columns:
            bot_group_df = bot_group_df.sort_values(TIME_COL).reset_index(drop=True)
        else:
            bot_group_df = bot_group_df.reset_index(drop=True)
        
        group_key = f'Bot_{bot_val}'
        groups[group_key] = ('Bot', bot_val, bot_group_df, n_samples)
    
    return groups


def filter_valid_groups(groups: Dict) -> Dict:
    """
    过滤样本数不足的分组
    
    Parameters
    ----------
    groups : dict
        所有分组
    
    Returns
    -------
    valid_groups : dict
        样本数 >= MIN_GROUP_SAMPLES 的分组
    """
    valid_groups = {}
    for key, (surface, value, group_df, n_samples) in groups.items():
        if n_samples >= MIN_GROUP_SAMPLES:
            valid_groups[key] = (surface, value, group_df, n_samples)
        else:
            logger.warning(f"分组 {key} 样本数过少 ({n_samples} < {MIN_GROUP_SAMPLES})，已过滤")
    
    return valid_groups


def summarize_groups(groups: Dict) -> pd.DataFrame:
    """
    生成分组统计摘要
    
    Parameters
    ----------
    groups : dict
        分组信息
    
    Returns
    -------
    summary : DataFrame
        分组统计表
    """
    summary_data = []
    for key, (surface, value, group_df, n_samples) in groups.items():
        summary_data.append({
            'Group_Key': key,
            'Surface': surface,
            'Setpoint_Value': value,
            'Sample_Count': n_samples,
        })
    
    summary_df = pd.DataFrame(summary_data)
    return summary_df.sort_values('Sample_Count', ascending=False).reset_index(drop=True)


def get_specs_in_group(df: pd.DataFrame, surface: str, value: float) -> pd.DataFrame:
    """
    获取某个表面分组内包含的所有规格组合及其样本数
    
    Parameters
    ----------
    df : DataFrame
        某个表面的分组数据
    surface : str
        'Top' 或 'Bot'
    value : float
        设定厚度值
    
    Returns
    -------
    specs : DataFrame
        规格组合及样本数
    """
    if surface == 'Top':
        specs = df.groupby([TOP_SETPOINT_COL, BOT_SETPOINT_COL]).size().reset_index(name='count')
    else:
        specs = df.groupby([TOP_SETPOINT_COL, BOT_SETPOINT_COL]).size().reset_index(name='count')
    
    specs = specs.sort_values('count', ascending=False).reset_index(drop=True)
    return specs
