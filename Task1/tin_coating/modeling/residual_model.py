"""残差校正模型核心类。

与 coating_model_by_group.py 中原代码完全一致，只做物理搬运，
零逻辑改动。LinearResidualCorrectionModel 保留作对照。
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def compute_direction_sample_weight(y_delta, pos_boost=1.0, damping=0.0):
    """damping=0 等权；调大会向"两方向都不能变差"的保守解靠拢。"""
    if damping <= 0:
        return pd.Series(1.0, index=y_delta.index)
    pos_mask = y_delta > 0
    neg_mask = y_delta < 0
    n_pos, n_neg = pos_mask.sum(), neg_mask.sum()
    n_total = n_pos + n_neg
    weights = pd.Series(1.0, index=y_delta.index)
    if n_pos > 0:
        full_balance_pos = n_total / (2.0 * n_pos)
        weights[pos_mask] = (1 - damping) * 1.0 + damping * full_balance_pos * pos_boost
    if n_neg > 0:
        full_balance_neg = n_total / (2.0 * n_neg)
        weights[neg_mask] = (1 - damping) * 1.0 + damping * full_balance_neg
    return weights


class ResidualCorrectionModel:
    """基于 HistGradientBoostingRegressor 的残差矫正模型。

    类级别强制 __module__ —— 无论在哪个 __main__ / CLI / 子包里实例化，
    joblib pickle 反序列化时都会从 tin_coating.modeling.residual_model 导入，
    彻底消除 monitoring/cli 里的 `import __main__` hack。
    """

    __module__ = "tin_coating.modeling.residual_model"

    def __init__(self, monotonic_feature_idx=None, alpha_smoothing=0.7,
                 pos_boost=1.0, damping=0.0,
                 max_iter=200, learning_rate=0.05, max_depth=4):
        self.alpha_smoothing = alpha_smoothing
        self.pos_boost = pos_boost
        self.damping = damping
        self.monotonic_feature_idx = monotonic_feature_idx
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.model = None

    def _build_model(self, n_features):
        monotonic_cst = None
        if self.monotonic_feature_idx is not None:
            monotonic_cst = [0] * n_features
            monotonic_cst[self.monotonic_feature_idx] = -1
        self.model = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            loss='absolute_error',
            monotonic_cst=monotonic_cst,
            random_state=42,
        )

    def fit(self, X, y_delta):
        self._build_model(n_features=X.shape[1])
        sample_weight = compute_direction_sample_weight(
            y_delta, pos_boost=self.pos_boost, damping=self.damping
        )
        self.model.fit(X, y_delta, sample_weight=sample_weight.values)

    def predict_smooth(self, X, online_actual):
        predicted_delta_raw = self.model.predict(X)
        delta_series = pd.Series(predicted_delta_raw, index=X.index)
        predicted_delta_smooth = delta_series.ewm(alpha=self.alpha_smoothing).mean()
        final_pred = online_actual + predicted_delta_smooth
        return final_pred, predicted_delta_smooth


class LinearResidualCorrectionModel:
    """基于 Huber 线性回归 + L2 正则的残差矫正模型，接口与 HGB 版本一致。"""

    def __init__(self, alpha_smoothing=0.7, pos_boost=1.0, damping=0.0,
                 alpha=1.0, epsilon=1.35, **kwargs):
        self.alpha_smoothing = alpha_smoothing
        self.pos_boost = pos_boost
        self.damping = damping
        self.alpha = alpha
        self.epsilon = epsilon
        self.model = None

    def _build_model(self):
        self.model = make_pipeline(
            StandardScaler(),
            HuberRegressor(alpha=self.alpha, epsilon=self.epsilon, max_iter=1000)
        )

    def fit(self, X, y_delta):
        self._build_model()
        sample_weight = compute_direction_sample_weight(
            y_delta, pos_boost=self.pos_boost, damping=self.damping
        )
        self.model.fit(X, y_delta, huberregressor__sample_weight=sample_weight.values)

    def predict_smooth(self, X, online_actual):
        predicted_delta_raw = self.model.predict(X)
        delta_series = pd.Series(predicted_delta_raw, index=X.index)
        predicted_delta_smooth = delta_series.ewm(alpha=self.alpha_smoothing).mean()
        final_pred = online_actual + predicted_delta_smooth
        return final_pred, predicted_delta_smooth


# ---------------------------------------------------------------------------
# 向后兼容层
# ---------------------------------------------------------------------------
# 把 ResidualCorrectionModel 也挂到 sys.modules['__main__'] 下。
# 这样历史上在 coating_model_by_group.py 直接 run 训练的 joblib（类 __module__=='__main__'）
# 反序列化时也能找到这个类。新训练的 joblib（类 __module__ 强制设成子包路径）不受影响。
import sys as _sys  # noqa: E402
try:
    if "__main__" in _sys.modules:
        _main = _sys.modules["__main__"]
        if not hasattr(_main, "ResidualCorrectionModel"):
            _main.ResidualCorrectionModel = ResidualCorrectionModel
except Exception:
    pass
