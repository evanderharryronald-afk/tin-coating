"""请求契约：服务对外承诺的输入格式。

原则：宁可 400 拒绝，也不要带着缺列/NaN 的数据静默算出一堆错误结果。
"""
from __future__ import annotations

import pandas as pd

from tin_coating.modeling.params import get_feature_cols

SURFACES = ("Top", "Bot")


class ContractError(ValueError):
    pass


def _derived_cols(surface: str) -> set[str]:
    return {f"{surface}_Current_Per_Speed"}


def raw_required_cols(surface: str) -> list[str]:
    """调用方必须提供的原始列（衍生列由服务自己算）。"""
    if surface not in SURFACES:
        raise ContractError(f"surface 必须是 {SURFACES} 之一，收到: {surface}")
    prefix = surface
    base = [
        "Speed[m/min]_Process_Avg",
        f"{prefix}_Current_Sum",
        f"Tin Weight_Actual[g/m2]_GALV_WEIGHT_{prefix.upper()}_Avg",
    ]
    derived = _derived_cols(surface)
    cols = base + [c for c in get_feature_cols(surface) if c not in derived]
    return list(dict.fromkeys(cols))


def validate_rows(rows, surface: str, max_rows: int) -> pd.DataFrame:
    """校验并转换成 DataFrame；不合法直接抛 ContractError。"""
    if surface not in SURFACES:
        raise ContractError(f"surface 必须是 {SURFACES} 之一，收到: {surface}")
    if not isinstance(rows, list) or not rows:
        raise ContractError("rows 必须是非空数组")
    if len(rows) > max_rows:
        raise ContractError(f"rows 数量 {len(rows)} 超过上限 {max_rows}")

    required = raw_required_cols(surface)
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ContractError(f"rows[{i}] 必须是对象")
        missing = [c for c in required if c not in row]
        if missing:
            raise ContractError(f"rows[{i}] 缺少字段 {missing}")

    df = pd.DataFrame(rows)
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    bad = df[required].isna().any()
    if bool(bad.any()):
        cols = [c for c in required if bad[c]]
        raise ContractError(f"字段存在空值或非数值: {cols}")

    return df
