"""
model_interpreter.py
通用模型解释性分析模块（SHAP + Permutation Importance + PDP/ICE）

设计目标：
1. 可独立运行（带 if __name__ == "__main__"）
2. 可被其他脚本轻松 import 并接入
3. 对树模型（HistGradientBoosting 等）和 sklearn Pipeline（线性模型）都兼容
4. 结果自动保存到指定目录，风格与现有项目一致
"""

import os
import warnings
from typing import Optional, List, Union, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.pipeline import Pipeline

# SHAP 是可选依赖，没装时给出友好提示
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    warnings.warn("未安装 shap，SHAP 相关功能将不可用。请执行: pip install shap")

# 画图中文支持（与主项目保持一致）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ModelInterpreter:
    """
    通用模型解释器

    使用示例（接入现有代码）：
        interpreter = ModelInterpreter(
            model=corrector.model,          # 或 corrector（如果是包装类）
            X=X_train,                      # 建议用训练集做解释
            feature_names=feature_cols,
            save_dir=f"result/.../interpretation_{group_tag}_{surface}"
        )
        interpreter.full_analysis(y=y_delta_train)
    """

    def __init__(
        self,
        model,
        X: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
        save_dir: str = "result/interpretation",
        random_state: int = 42,
        max_samples_for_shap: int = 800,
    ):
        """
        :param model: 已训练好的模型。可以是：
                      - sklearn 估计器
                      - Pipeline
                      - 你项目里的 ResidualCorrectionModel / LinearResidualCorrectionModel
                        （会自动取 .model 属性）
        :param X: 用于解释的特征数据（建议用训练集）
        :param feature_names: 特征名列表。若 X 是 DataFrame 可自动获取
        :param save_dir: 图片与结果保存根目录
        :param random_state: 随机种子
        :param max_samples_for_shap: SHAP 计算时最多使用的样本数（控制计算量）
        """
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.random_state = random_state
        self.max_samples_for_shap = max_samples_for_shap

        # 兼容包装类
        if hasattr(model, "model") and model.model is not None:
            self.estimator = model.model
        else:
            self.estimator = model

        # 处理特征名与数据
        if isinstance(X, pd.DataFrame):
            self.feature_names = feature_names or list(X.columns)
            self.X = X.copy()
        else:
            if feature_names is None:
                raise ValueError("当 X 不是 DataFrame 时，必须显式传入 feature_names")
            self.feature_names = feature_names
            self.X = pd.DataFrame(X, columns=self.feature_names)

        self.X_np = self.X.values  # 部分接口需要 numpy

        # 内部缓存
        self._shap_values = None
        self._shap_explainer = None
        self._perm_result = None

    # ------------------------------------------------------------------
    # 1. Permutation Importance
    # ------------------------------------------------------------------
    def permutation_importance(
        self,
        y: Union[pd.Series, np.ndarray],
        n_repeats: int = 15,
        scoring: str = "neg_mean_absolute_error",
        plot: bool = True,
        top_n: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        计算并可视化排列重要性
        """
        print("\n[ModelInterpreter] 正在计算 Permutation Importance ...")
        result = permutation_importance(
            self.estimator,
            self.X,
            y,
            n_repeats=n_repeats,
            scoring=scoring,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._perm_result = result

        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

        print(importance_df.to_string(index=False))

        if plot:
            plot_df = importance_df if top_n is None else importance_df.head(top_n)
            plt.figure(figsize=(9, max(4, len(plot_df) * 0.35)))
            plt.barh(
                plot_df["feature"][::-1],
                plot_df["importance_mean"][::-1],
                xerr=plot_df["importance_std"][::-1],
                color="steelblue",
                alpha=0.85,
            )
            plt.xlabel("Permutation Importance (mean ± std)")
            plt.title("特征排列重要性 (Permutation Importance)")
            plt.tight_layout()
            path = os.path.join(self.save_dir, "permutation_importance.png")
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"[图表保存] Permutation Importance → {path}")

        # 同时保存 csv
        csv_path = os.path.join(self.save_dir, "permutation_importance.csv")
        importance_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        return importance_df

    # ------------------------------------------------------------------
    # 2. SHAP 分析
    # ------------------------------------------------------------------
    def _get_shap_explainer_and_values(self, background_samples: int = 100):
        """内部：构建 explainer 并计算 shap_values（带缓存）"""
        if not HAS_SHAP:
            raise ImportError("请先安装 shap: pip install shap")

        if self._shap_values is not None:
            return self._shap_explainer, self._shap_values

        # 采样控制计算量
        if len(self.X) > self.max_samples_for_shap:
            X_sample = self.X.sample(self.max_samples_for_shap, random_state=self.random_state)
        else:
            X_sample = self.X

        # 背景数据
        if len(X_sample) > background_samples:
            background = shap.sample(X_sample, background_samples, random_state=self.random_state)
        else:
            background = X_sample

        # 取出真正的估计器（兼容 Pipeline）
        model_for_shap = self.estimator
        if isinstance(self.estimator, Pipeline):
            model_for_shap = self.estimator.steps[-1][1]

        # ---------- 核心修复：优先 TreeExplainer + 关闭 additivity 检查 ----------
        try:
            explainer = shap.TreeExplainer(model_for_shap, data=background)
            # 关键：关闭 check_additivity，避免数值精度导致的报错
            shap_values = explainer.shap_values(X_sample, check_additivity=False)
            print("[SHAP] 使用 TreeExplainer（已关闭 additivity check）")
        except Exception as e:
            print(f"[SHAP] TreeExplainer 失败，切换到通用 Explainer。原因: {e}")
            explainer = shap.Explainer(model_for_shap, background)
            # 通用 Explainer 调用时也关闭检查
            shap_values = explainer(X_sample, check_additivity=False)

            # 统一转成 numpy 数组，方便后续 summary_plot / dependence_plot 使用
            if hasattr(shap_values, "values"):
                shap_values = shap_values.values
            print("[SHAP] 使用通用 Explainer")

        self._shap_explainer = explainer
        self._shap_values = shap_values
        self._X_shap = X_sample
        return explainer, shap_values

    def shap_summary(
        self,
        plot_type: str = "dot",          # 'dot' | 'bar' | 'violin'
        max_display: int = 15,
        show: bool = False,
    ):
        """SHAP 全局摘要图"""
        if not HAS_SHAP:
            print("[警告] 未安装 shap，跳过 SHAP 分析")
            return None

        print("\n[ModelInterpreter] 正在计算 SHAP 值 ...")
        explainer, shap_values = self._get_shap_explainer_and_values()

        plt.figure()
        shap.summary_plot(
            shap_values,
            self._X_shap,
            feature_names=self.feature_names,
            plot_type=plot_type,
            max_display=max_display,
            show=False,
        )
        plt.tight_layout()
        path = os.path.join(self.save_dir, f"shap_summary_{plot_type}.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[图表保存] SHAP Summary ({plot_type}) → {path}")
        return shap_values

    def shap_dependence(
        self,
        features: Optional[List[str]] = None,
        interaction_idx: str = "auto",
    ):
        """SHAP 依赖图（可看交互）"""
        if not HAS_SHAP:
            return

        explainer, shap_values = self._get_shap_explainer_and_values()

        if features is None:
            # 默认画重要性最高的前几个
            mean_abs = np.abs(shap_values).mean(axis=0)
            top_idx = np.argsort(mean_abs)[::-1][:min(4, len(self.feature_names))]
            features = [self.feature_names[i] for i in top_idx]

        for feat in features:
            if feat not in self.feature_names:
                continue
            plt.figure()
            shap.dependence_plot(
                feat,
                shap_values,
                self._X_shap,
                feature_names=self.feature_names,
                interaction_index=interaction_idx,
                show=False,
            )
            plt.tight_layout()
            safe_name = feat.replace("/", "_").replace("[", "").replace("]", "")
            path = os.path.join(self.save_dir, f"shap_dependence_{safe_name}.png")
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"[图表保存] SHAP Dependence ({feat}) → {path}")

    def shap_bar(self, max_display: int = 15):
        """SHAP 平均绝对贡献条形图（更简洁的全局重要性）"""
        return self.shap_summary(plot_type="bar", max_display=max_display)

    # ------------------------------------------------------------------
    # 3. Partial Dependence + ICE
    # ------------------------------------------------------------------
    def partial_dependence(
        self,
        features: Optional[List[Union[str, int]]] = None,
        kind: str = "average",          # 'average' | 'individual' | 'both'
        grid_resolution: int = 50,
        n_cols: int = 2,
    ):
        """
        绘制 PDP / ICE 图
        kind='both' 时同时显示平均 PDP 和个体 ICE 曲线
        """
        print("\n[ModelInterpreter] 正在计算 Partial Dependence ...")

        if features is None:
            # 默认取前几个数值特征
            features = list(range(min(4, len(self.feature_names))))

        # 转换为索引
        feature_indices = []
        for f in features:
            if isinstance(f, str):
                feature_indices.append(self.feature_names.index(f))
            else:
                feature_indices.append(f)

        n_feats = len(feature_indices)
        n_rows = (n_feats + n_cols - 1) // n_cols

        fig, ax = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_feats == 1:
            ax = np.array([ax])
        ax = ax.flatten()

        display = PartialDependenceDisplay.from_estimator(
            self.estimator,
            self.X,
            features=feature_indices,
            kind=kind,
            grid_resolution=grid_resolution,
            ax=ax[:n_feats],
            n_cols=n_cols,
        )

        for i in range(n_feats, len(ax)):
            ax[i].set_visible(False)

        fig.suptitle("Partial Dependence / ICE", fontsize=14)
        plt.tight_layout()
        path = os.path.join(self.save_dir, f"partial_dependence_{kind}.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[图表保存] Partial Dependence ({kind}) → {path}")
        return display

    # ------------------------------------------------------------------
    # 4. 一键完整分析
    # ------------------------------------------------------------------
    def full_analysis(
        self,
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        run_permutation: bool = True,
        run_shap: bool = True,
        run_pdp: bool = True,
        pdp_features: Optional[List[str]] = None,
        shap_dependence_features: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        一键运行常用解释分析，返回结果字典
        """
        results = {}

        if run_permutation:
            if y is None:
                print("[警告] 未提供 y，跳过 Permutation Importance")
            else:
                results["permutation"] = self.permutation_importance(y)

        if run_shap and HAS_SHAP:
            results["shap_values"] = self.shap_summary(plot_type="dot")
            self.shap_bar()
            self.shap_dependence(features=shap_dependence_features)

        if run_pdp:
            results["pdp"] = self.partial_dependence(
                features=pdp_features,
                kind="both"          # 同时看平均效应和个体曲线
            )

        print(f"\n[ModelInterpreter] 全部分析完成，结果已保存至: {self.save_dir}")
        return results


# # ======================================================================
# # 独立运行入口（方便单独调试或快速分析）
# # ======================================================================
# if __name__ == "__main__":
#     import argparse
#     from sklearn.ensemble import HistGradientBoostingRegressor
#     from sklearn.model_selection import train_test_split
#
#     parser = argparse.ArgumentParser(description="模型解释性分析独立运行脚本")
#     parser.add_argument("--data", type=str, default="result/cleaned_data/cleaned_data.xlsx",
#                         help="干净数据路径")
#     parser.add_argument("--surface", type=str, default="Top", choices=["Top", "Bot"])
#     parser.add_argument("--group", type=str, default=None,
#                         help="可选：指定规格组，例如 Top2.799_Bot2.799")
#     parser.add_argument("--save_dir", type=str, default="result/interpretation_demo")
#     args = parser.parse_args()
#
#     print("=" * 60)
#     print("ModelInterpreter 独立运行模式")
#     print("=" * 60)
#
#     # ---------- 1. 准备一份简单的残差数据（示例） ----------
#     df = pd.read_excel(args.data)
#
#     prefix = args.surface
#     surface_cn = "上" if prefix == "Top" else "下"
#     online_col = f"Tin Weight_Actual[g/m2]_GALV_WEIGHT_{prefix.upper()}_Avg"
#     lab_col = f"{surface_cn}表面镀层重量A(XA1_0)"
#     current_col = f"{prefix}_Current_Sum"
#     speed_col = "Speed[m/min]_Process_Avg"
#
#     # 构造与主流程一致的特征
#     df[f"{prefix}_Current_Per_Speed"] = df[current_col] / (df[speed_col] + 1e-5)
#     df[f"{prefix}_Delta"] = df[lab_col] - df[online_col]
#
#     feature_cols = [
#         online_col,
#         current_col,
#         f"{prefix}_Current_Per_Speed",
#         f"{prefix}_Theoretical_Factor",
#         speed_col,
#         "Dimension_[mm]_Width",
#         "Dimension_[mm]_Thickness",
#         "Steel_Grade_Encoded",
#     ]
#
#     # 可选：按规格组过滤
#     if args.group and "Setpoint_Group_Label" in df.columns:
#         df = df[df["Setpoint_Group_Label"] == args.group].copy()
#         print(f"已过滤规格组: {args.group}，剩余样本 {len(df)}")
#
#     df = df.dropna(subset=feature_cols + [f"{prefix}_Delta"])
#     X = df[feature_cols]
#     y = df[f"{prefix}_Delta"]
#
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, shuffle=False
#     )
#
#     # ---------- 2. 训练一个简单模型 ----------
#     model = HistGradientBoostingRegressor(
#         max_iter=150,
#         learning_rate=0.06,
#         max_depth=4,
#         loss="absolute_error",
#         random_state=42,
#     )
#     model.fit(X_train, y_train)
#     print(f"示例模型训练完成，测试集 MAE = {np.abs(model.predict(X_test) - y_test).mean():.4f}")
#
#
#     # ---------- 3. 解释 ----------
#     interpreter = ModelInterpreter(
#         model=model,
#         X=X_train,
#         feature_names=feature_cols,
#         save_dir=args.save_dir,
#         max_samples_for_shap=600,
#     )
#
#     interpreter.full_analysis(
#         y=y_train,
#         run_permutation=True,
#         run_shap=True,
#         run_pdp=True,
#         pdp_features=feature_cols[:4],          # 只画前4个，避免图太多
#     )
#
#     print("\n独立运行结束。")