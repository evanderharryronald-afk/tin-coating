"""
通用模型推理脚本：支持整体模型和分规格组模型预测

功能：
1. 加载保存的模型
2. 验证输入数据特征
3. 支持单条和批量预测
4. 输出预测结果到 Excel
"""

import os
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor

from model_persistence import ModelLoader, ModelMetadata


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


class PredictionPipeline:
    """通用预测管道"""
    
    def __init__(self, base_model_dir: str = "result/trained_models"):
        """
        Args:
            base_model_dir: 模型保存目录
        """
        self.loader = ModelLoader(base_dir=base_model_dir)
        self.loaded_models = {}  # 缓存已加载的模型
    
    def load_and_cache(self, 
                       surface: str, 
                       group_label: Optional[str] = None) -> Tuple[Any, ModelMetadata]:
        """
        加载模型并缓存
        
        Args:
            surface: 'Top' 或 'Bot'
            group_label: 规格组标签（None 表示整体模型）
        
        Returns:
            (model, metadata)
        """
        cache_key = (surface, group_label)
        
        if cache_key not in self.loaded_models:
            model, metadata = self.loader.load_model(
                surface=surface,
                group_label=group_label,
                fallback_to_overall=True
            )
            self.loaded_models[cache_key] = (model, metadata)
        
        return self.loaded_models[cache_key]
    
    def validate_features(self,
                         data: pd.DataFrame,
                         required_features: list,
                         surface: str) -> Tuple[bool, str]:
        """
        验证输入数据是否包含所需特征
        
        Args:
            data: 输入 DataFrame
            required_features: 所需特征列表
            surface: 表面类型（用于错误提示）
        
        Returns:
            (is_valid, message)
        """
        missing = set(required_features) - set(data.columns)
        
        if missing:
            return False, f"[错误] {surface} 表面缺少特征: {missing}"
        
        # 检查是否有 NaN
        nan_cols = data[required_features].columns[data[required_features].isna().any()].tolist()
        if nan_cols:
            return False, f"[警告] {surface} 表面特征包含 NaN: {nan_cols}"
        
        return True, "验证通过"
    
    def predict_single_surface(self,
                               data: pd.DataFrame,
                               surface: str,
                               group_label: Optional[str] = None,
                               output_col_name: Optional[str] = None) -> pd.DataFrame:
        """
        对单个表面进行预测
        
        Args:
            data: 输入 DataFrame
            surface: 'Top' 或 'Bot'
            group_label: 规格组标签（None 表示使用整体模型或降级）
            output_col_name: 输出列名（默认为 '{surface}_Predicted_Weight'）
        
        Returns:
            包含预测结果的 DataFrame
        """
        if output_col_name is None:
            output_col_name = f"{surface}_Predicted_Weight"
        
        # 加载模型
        model, metadata = self.load_and_cache(surface, group_label)
        
        # 获取所需特征
        required_features = metadata.feature_names
        
        # 验证特征
        is_valid, message = self.validate_features(data, required_features, surface)
        if not is_valid:
            print(message)
            if "缺少特征" in message:
                raise ValueError(message)
        else:
            print(f"[✓] {surface} 表面特征验证通过")
        
        # 提取特征
        X = data[required_features].copy()
        
        # 获取在线值（用于 predict_smooth）
        prefix = surface.upper()
        online_col = f'Tin Weight_Actual[g/m2]_GALV_WEIGHT_{prefix}_Avg'
        
        if online_col not in data.columns:
            raise ValueError(f"缺少在线值列: {online_col}")
        
        online_actual = data[online_col]
        
        # 进行预测
        try:
            predicted_weight, predicted_delta = model.predict_smooth(X, online_actual)
            
            # 创建结果 DataFrame
            result = data.copy()
            result[output_col_name] = predicted_weight.values
            result[f"{surface}_Predicted_Delta"] = predicted_delta.values
            
            print(f"[✓] {surface} 表面预测完成，预测样本数: {len(result)}")
            return result
        
        except Exception as e:
            print(f"[✗] {surface} 表面预测失败: {e}")
            raise
    
    def predict_both_surfaces(self,
                             data: pd.DataFrame,
                             group_label: Optional[str] = None) -> pd.DataFrame:
        """
        对两个表面同时进行预测
        
        Args:
            data: 输入 DataFrame
            group_label: 规格组标签
        
        Returns:
            包含两个表面预测结果的 DataFrame
        """
        result = data.copy()
        
        for surface in ['Top', 'Bot']:
            try:
                result = self.predict_single_surface(
                    result,
                    surface=surface,
                    group_label=group_label
                )
            except Exception as e:
                print(f"[✗] 跳过 {surface} 表面: {e}")
        
        return result
    
    def predict_batch(self,
                     data: pd.DataFrame,
                     group_label_col: Optional[str] = None) -> pd.DataFrame:
        """
        批量预测（支持按规格组分组预测）
        
        Args:
            data: 输入 DataFrame
            group_label_col: 规格组列名（None 表示使用整体模型）
        
        Returns:
            包含预测结果的 DataFrame
        """
        result = data.copy()
        
        if group_label_col is None:
            # 使用整体模型
            print("\n[预测] 使用整体模型进行预测...")
            result = self.predict_both_surfaces(result, group_label=None)
        else:
            # 按规格组分组预测
            if group_label_col not in data.columns:
                raise ValueError(f"规格组列不存在: {group_label_col}")
            
            group_labels = data[group_label_col].unique()
            print(f"\n[预测] 检测到 {len(group_labels)} 个规格组，开始分组预测...")
            
            for group_label in group_labels:
                mask = data[group_label_col] == group_label
                group_indices = result[mask].index
                
                print(f"\n[预测] 规格组: {group_label} (样本数: {mask.sum()})")
                
                try:
                    group_result = self.predict_both_surfaces(
                        result.loc[group_indices],
                        group_label=group_label
                    )
                    result.loc[group_indices] = group_result
                except Exception as e:
                    print(f"[✗] 规格组 {group_label} 预测失败: {e}")
        
        return result
    
    def export_predictions(self,
                          result_df: pd.DataFrame,
                          output_path: str = "result/predictions.xlsx") -> None:
        """
        将预测结果导出到 Excel
        
        Args:
            result_df: 预测结果 DataFrame
            output_path: 输出文件路径
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            result_df.to_excel(output_path, index=False, engine='openpyxl')
            print(f"[✓] 预测结果已导出: {output_path}")
        except Exception as e:
            print(f"[✗] 导出失败: {e}")
            raise


def quick_predict(input_file: str,
                  output_file: Optional[str] = None,
                  group_label_col: Optional[str] = None,
                  model_dir: str = "result/trained_models") -> pd.DataFrame:
    """
    快速预测函数
    
    Args:
        input_file: 输入数据文件（Excel 或 CSV）
        output_file: 输出文件路径（默认为 result/predictions.xlsx）
        group_label_col: 规格组列名（可选）
        model_dir: 模型目录
    
    Returns:
        预测结果 DataFrame
    """
    # 读取输入数据
    if input_file.endswith('.xlsx'):
        data = pd.read_excel(input_file)
    elif input_file.endswith('.csv'):
        data = pd.read_csv(input_file)
    else:
        raise ValueError("输入文件必须是 Excel 或 CSV 格式")
    
    print(f"[加载] 数据已读取，样本数: {len(data)}")
    
    # 创建预测管道
    pipeline = PredictionPipeline(base_model_dir=model_dir)
    
    # 执行预测
    result = pipeline.predict_batch(data, group_label_col=group_label_col)
    
    # 导出结果
    if output_file is None:
        output_file = "result/predictions.xlsx"
    
    pipeline.export_predictions(result, output_file)
    
    return result


if __name__ == "__main__":
    """直接运行时执行快速测试或预测"""
    import sys
    
    print("\n" + "="*60)
    print("镀层重量预测 - 模型推理")
    print("="*60)
    
    # 解析命令行参数
    input_file = None
    output_file = None
    group_label = None
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        # 判断第二个参数是否是文件路径或规格组标签
        if sys.argv[2].endswith(('.xlsx', '.csv')):
            output_file = sys.argv[2]
            if len(sys.argv) > 3:
                group_label = sys.argv[3]
        else:
            # 第二个参数是规格组标签
            group_label = sys.argv[2]
            if len(sys.argv) > 3:
                output_file = sys.argv[3]
    
    # 默认值
    if input_file is None:
        input_file = "result/data/feature_engineered_data/featured_data.xlsx"
    if output_file is None:
        if group_label:
            output_file = f"result/predictions/predictions_{group_label}.xlsx"
        else:
            output_file = "result/predictions/predictions_overall.xlsx"
    
    print(f"\n[参数配置]")
    print(f"  输入文件: {input_file}")
    if group_label:
        print(f"  规格组: {group_label}")
    else:
        print(f"  模式: 整体模型")
    print(f"  输出文件: {output_file}")
    
    # 验证输入文件
    if not os.path.exists(input_file):
        print(f"\n[ERROR] 输入文件不存在: {input_file}")
        print("\n使用方法:")
        print("  python model_inference.py                           # 整体模型预测（默认数据）")
        print("  python model_inference.py input.xlsx                # 整体模型预测（自定义数据）")
        print("  python model_inference.py input.xlsx output.xlsx    # 整体模型预测（自定义输出）")
        print("  python model_inference.py input.xlsx group_label output.xlsx  # 分组预测")
        sys.exit(1)
    
    print(f"\n[加载] 读取数据: {input_file}")
    if input_file.endswith('.xlsx'):
        data = pd.read_excel(input_file)
    elif input_file.endswith('.csv'):
        data = pd.read_csv(input_file)
    else:
        print("[ERROR] 文件格式不支持，需要 .xlsx 或 .csv")
        sys.exit(1)
    
    print(f"  - 数据行数: {len(data)}")
    print(f"  - 数据列数: {len(data.columns)}")
    
    # 执行预测
    try:
        pipeline = PredictionPipeline()
        
        if group_label:
            # 分组模型预测
            print(f"\n[预测] 使用规格组模型预测: {group_label}")
            result = data.copy()
            
            for surface in ['Top', 'Bot']:
                try:
                    print(f"  [{surface}] 处理中...")
                    result = pipeline.predict_single_surface(
                        result,
                        surface=surface,
                        group_label=group_label
                    )
                except Exception as e:
                    print(f"  [{surface}] 预测失败: {e}")
        else:
            # 整体模型预测
            print(f"\n[预测] 使用整体模型预测")
            result = pipeline.predict_batch(data, group_label_col=None)
        
        # 导出结果
        print(f"\n[导出] 保存结果...")
        pipeline.export_predictions(result, output_file)
        
        # 统计
        print(f"\n[统计] 预测结果:")
        for col in ['Top_Predicted_Weight', 'Bot_Predicted_Weight']:
            if col in result.columns:
                valid = result[col].notna().sum()
                print(f"  - {col}: {valid}/{len(result)} 有效值")
        
        print(f"\n[OK] 预测完成")
        print(f"[OK] 结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"\n[ERROR] 预测失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "="*60 + "\n")
