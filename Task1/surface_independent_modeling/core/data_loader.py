"""
数据加载和准备模块
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
import logging

from .constants import (
    FEATURED_DATA_PATH, TOP_SETPOINT_COL, BOT_SETPOINT_COL,
    TIME_COL, TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    TOP_DELTA_COL, BOT_DELTA_COL
)

logger = logging.getLogger(__name__)


def load_featured_data(path: Optional[str] = None) -> pd.DataFrame:
    """
    加载特征工程后的数据
    
    Parameters
    ----------
    path : str, optional
        数据文件路径，默认使用 FEATURED_DATA_PATH
    
    Returns
    -------
    df : DataFrame
        加载的数据
    """
    if path is None:
        path = FEATURED_DATA_PATH
    
    logger.info(f"加载数据: {path}")
    df = pd.read_excel(path)
    logger.info(f"数据形状: {df.shape}")
    
    return df


def filter_by_surface_group(
    df: pd.DataFrame,
    surface: str,
    value: float
) -> pd.DataFrame:
    """
    按表面和厚度值过滤数据
    
    Parameters
    ----------
    df : DataFrame
        原始数据
    surface : str
        'Top' 或 'Bot'
    value : float
        设定厚度值
    
    Returns
    -------
    filtered_df : DataFrame
        过滤后的数据
    """
    if surface == 'Top':
        filtered_df = df[df[TOP_SETPOINT_COL] == value].copy()
    elif surface == 'Bot':
        filtered_df = df[df[BOT_SETPOINT_COL] == value].copy()
    else:
        raise ValueError(f"Invalid surface: {surface}. Must be 'Top' or 'Bot'")
    
    # 按时间排序，保证顺序一致
    if TIME_COL in filtered_df.columns:
        filtered_df = filtered_df.sort_values(TIME_COL).reset_index(drop=True)
    else:
        filtered_df = filtered_df.reset_index(drop=True)
    
    logger.info(f"过滤数据: surface={surface}, value={value}, samples={len(filtered_df)}")
    
    return filtered_df


def prepare_train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: Optional[float] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    按时间顺序分割训练/验证/测试集
    
    Parameters
    ----------
    df : DataFrame
        数据
    train_ratio : float
        训练集比例
    val_ratio : float
        验证集比例
    test_ratio : float, optional
        测试集比例，默认 = 1 - train_ratio - val_ratio
    
    Returns
    -------
    train_df, val_df, test_df : tuple of DataFrame
        分割后的数据集
    """
    n = len(df)
    
    if test_ratio is None:
        test_ratio = 1.0 - train_ratio - val_ratio
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    # 确保每个集合至少有一定样本
    min_size = max(10, int(n * 0.05))
    if train_end < min_size:
        train_end = min_size
    if val_end - train_end < min_size:
        val_end = min(train_end + min_size, n - min_size)
    if n - val_end < min_size:
        val_end = max(train_end, n - min_size)
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    logger.info(f"分割数据: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    return train_df, val_df, test_df


def get_feature_columns(surface: str) -> List[str]:
    """
    获取某个表面的特征列
    
    Parameters
    ----------
    surface : str
        'Top' 或 'Bot'
    
    Returns
    -------
    feature_cols : list
        特征列名
    """
    if surface == 'Top':
        from .constants import FEATURE_COLUMNS_TOP
        return FEATURE_COLUMNS_TOP
    elif surface == 'Bot':
        from .constants import FEATURE_COLUMNS_BOT
        return FEATURE_COLUMNS_BOT
    else:
        raise ValueError(f"Invalid surface: {surface}")


def get_target_column(surface: str) -> str:
    """
    获取某个表面的目标列 (Delta)
    
    Parameters
    ----------
    surface : str
        'Top' 或 'Bot'
    
    Returns
    -------
    target_col : str
        目标列名
    """
    if surface == 'Top':
        return TOP_DELTA_COL
    elif surface == 'Bot':
        return BOT_DELTA_COL
    else:
        raise ValueError(f"Invalid surface: {surface}")


def prepare_xy(
    df: pd.DataFrame,
    surface: str,
    feature_cols: Optional[List[str]] = None,
    drop_na: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    准备特征矩阵和目标向量
    
    Parameters
    ----------
    df : DataFrame
        数据
    surface : str
        'Top' 或 'Bot'
    feature_cols : list, optional
        特征列，默认自动选择
    drop_na : bool
        是否删除缺失值
    
    Returns
    -------
    X : DataFrame
        特征矩阵
    y : Series
        目标向量
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(surface)
    
    target_col = get_target_column(surface)
    
    # 检查列是否存在
    missing_cols = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing_cols:
        logger.warning(f"缺失列: {missing_cols}")
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    if drop_na:
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx]
        y = y[valid_idx]
        logger.info(f"删除缺失值后: samples={len(X)}")
    
    return X, y


def validate_data_quality(df: pd.DataFrame, min_samples: int = 50) -> bool:
    """
    验证数据质量
    
    Parameters
    ----------
    df : DataFrame
        数据
    min_samples : int
        最小样本数
    
    Returns
    -------
    is_valid : bool
        数据是否有效
    """
    if len(df) < min_samples:
        logger.warning(f"样本数不足: {len(df)} < {min_samples}")
        return False
    
    return True
