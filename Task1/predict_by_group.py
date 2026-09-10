"""
分组模型预测 - 快捷脚本（代理到 model_inference.py）

用法示例:
  python predict_by_group.py                  # 用 Top2.799_Bot2.799 规格组预测
  python predict_by_group.py input.xlsx       # 指定输入文件
  python predict_by_group.py input.xlsx Top3.5_Bot3.5  # 指定规格组
"""

import sys
import pandas as pd
import os
from sklearn.ensemble import HistGradientBoostingRegressor

from model_inference import PredictionPipeline


# 定义模型类以支持 joblib 反序列化
class ResidualCorrectionModel:
    """残差校正模型 - 用于 joblib 反序列化"""
    def __init__(self, monotonic_feature_idx=None, alpha_smoothing=0.7,
                 pos_boost=1.0, damping=0.0, learning_rate=0.05,
                 max_iter=200, max_depth=4, loss='absolute_error', quantile=None, **kwargs):
        self.alpha_smoothing = alpha_smoothing
        self.pos_boost = pos_boost
        self.damping = damping
        self.monotonic_feature_idx = monotonic_feature_idx
        self.loss = loss
        self.quantile = quantile
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.kwargs = kwargs
        self.model = None

    def _build_model(self, n_features):
        monotonic_cst = None
        if self.monotonic_feature_idx is not None:
            monotonic_cst = [0] * n_features
            monotonic_cst[self.monotonic_feature_idx] = -1
        
        model_kwargs = {
            'max_iter': self.max_iter,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'loss': self.loss,
            'monotonic_cst': monotonic_cst,
            'random_state': 42,
            **self.kwargs
        }
        if self.loss == 'quantile' and self.quantile is not None:
            model_kwargs['quantile'] = self.quantile
        
        self.model = HistGradientBoostingRegressor(**model_kwargs)

    def fit(self, X, y_delta):
        self._build_model(n_features=X.shape[1])
        self.model.fit(X, y_delta)

    def predict_smooth(self, X, online_actual):
        """预测并平滑处理"""
        predicted_delta_raw = self.model.predict(X)
        delta_series = pd.Series(predicted_delta_raw, index=X.index)
        predicted_delta_smooth = delta_series.ewm(alpha=self.alpha_smoothing).mean()
        final_pred = online_actual + predicted_delta_smooth
        return final_pred, predicted_delta_smooth


if __name__ == "__main__":
    # 默认规格组
    default_group = "Top2.799_Bot2.799"
    
    # 解析参数
    input_file = sys.argv[1] if len(sys.argv) > 1 else "result/data/feature_engineered_data/featured_data.xlsx"
    group_label = sys.argv[2] if len(sys.argv) > 2 else default_group
    output_file = sys.argv[3] if len(sys.argv) > 3 else f"result/predictions/predictions_{group_label}.xlsx"
    
    print("\n" + "="*60)
    print("分组模型预测")
    print("="*60)
    
    print(f"\n[参数]")
    print(f"  输入文件: {input_file}")
    print(f"  规格组: {group_label}")
    print(f"  输出文件: {output_file}\n")
    
    # 直接调用 model_inference 的功能
    if not os.path.exists(input_file):
        print(f"[ERROR] 输入文件不存在: {input_file}")
        sys.exit(1)
    
    print(f"[加载] 读取数据...")
    if input_file.endswith('.xlsx'):
        data = pd.read_excel(input_file)
    else:
        data = pd.read_csv(input_file)
    
    print(f"  - 数据行数: {len(data)}")
    
    # 使用 PredictionPipeline 进行预测
    pipeline = PredictionPipeline()
    result = data.copy()
    
    print(f"\n[预测] 使用规格组 {group_label}")
    
    for surface in ['Top', 'Bot']:
        try:
            print(f"  [{surface}] 处理中...")
            result = pipeline.predict_single_surface(
                result,
                surface=surface,
                group_label=group_label
            )
        except Exception as e:
            print(f"  [{surface}] 失败: {e}")
    
    print(f"\n[导出]")
    pipeline.export_predictions(result, output_file)
    
    print("\n" + "="*60 + "\n")
