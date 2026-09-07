"""
GroupEDADiagnoser: 为分规格组建模流程设计的EDA诊断管理器

支持Top/Bot表面独立配置、灵活启用/禁用、参数差异化
"""

from typing import Optional, List, Dict, Any
import os
import pandas as pd
from eda_analyzer import SurfaceEDAAnalyzer


class SurfaceEDAConfig:
    """单个表面的EDA诊断配置"""
    
    def __init__(self, 
                 surface: str,
                 enabled: bool = True,
                 compute_stats_only_pre: bool = True,
                 compute_stats_only_post: bool = False,
                 max_groups: int = 12,
                 sample_for_scatter: Optional[int] = 8000,
                 enable_overall: bool = True,
                 enable_by_group: bool = True,
                 group_col: Optional[str] = None,
                 target_groups: Optional[List[str]] = None):
        """
        Parameters
        ----------
        surface : str
            表面标识，如 'Top' 或 'Bot'
        enabled : bool
            是否启用该表面的诊断
        compute_stats_only_pre : bool
            建模前是否仅计算统计（不画图，快速模式）
        compute_stats_only_post : bool
            建模后是否仅计算统计（不画图）
        max_groups : int
            分规格分析时最多分析的组数
        sample_for_scatter : Optional[int]
            散点图采样数，None时不采样
        enable_overall : bool
            是否执行整体分析（全量数据）
        enable_by_group : bool
            是否执行分规格分析
        group_col : Optional[str]
            分规格列名，例如 'Setpoint_Group_Label'
            仅当 enable_by_group=True 时有效
        target_groups : Optional[List[str]]
            规格白名单，仅分析这些规格
            仅当 enable_by_group=True 时有效
            例如 ['Top2.799_Bot2.799', 'Top2.8_Bot2.8']
        """
        self.surface = surface
        self.enabled = enabled
        self.compute_stats_only_pre = compute_stats_only_pre
        self.compute_stats_only_post = compute_stats_only_post
        self.max_groups = max_groups
        self.sample_for_scatter = sample_for_scatter
        self.enable_overall = enable_overall
        self.enable_by_group = enable_by_group
        self.group_col = group_col
        self.target_groups = target_groups
    
    def __repr__(self):
        return (f"SurfaceEDAConfig({self.surface}, enabled={self.enabled}, "
                f"pre_stats_only={self.compute_stats_only_pre}, "
                f"post_stats_only={self.compute_stats_only_post})")


class GroupEDADiagnoser:
    """支持Top/Bot灵活配置的EDA诊断管理器"""
    
    def __init__(self, 
                 surfaces: Optional[List[str]] = None,
                 surface_configs: Optional[Dict[str, SurfaceEDAConfig]] = None,
                 pre_diagnosis_dir: str = 'result/eda/pre_modeling',
                 post_diagnosis_dir: str = 'result/eda/post_modeling'):
        """
        Parameters
        ----------
        surfaces : Optional[List[str]]
            要诊断的表面列表，如 ['Top', 'Bot'] 或 ['Top']
            默认为 ['Top', 'Bot']
        
        surface_configs : Optional[Dict[str, SurfaceEDAConfig]]
            按表面名称配置诊断参数
            例：{'Top': SurfaceEDAConfig('Top', ...), 'Bot': SurfaceEDAConfig('Bot', ...)}
            未指定的表面使用默认配置
        
        pre_diagnosis_dir : str
            建模前诊断结果的根目录
        
        post_diagnosis_dir : str
            建模后诊断结果的根目录
        """
        self.surfaces = surfaces or ['Top', 'Bot']
        self.analyzer = SurfaceEDAAnalyzer()
        self.pre_diagnosis_dir = pre_diagnosis_dir
        self.post_diagnosis_dir = post_diagnosis_dir
        
        # 构建每个表面的配置字典
        self.configs = {}
        for surface in self.surfaces:
            if surface_configs and surface in surface_configs:
                self.configs[surface] = surface_configs[surface]
            else:
                self.configs[surface] = SurfaceEDAConfig(surface)
    
    def pre_modeling_diagnosis(self, 
                               df: pd.DataFrame, 
                               group_label: str,
                               surfaces: Optional[List[str]] = None,
                               enable_overall: Optional[bool] = None,
                               enable_by_group: Optional[bool] = None,
                               group_col: Optional[str] = None,
                               target_groups: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        建模前诊断：数据分布、特征质量、异常检测
        
        Parameters
        ----------
        df : pd.DataFrame
            包含所有表面数据的DataFrame
        group_label : str
            规格组标签，仅用于by_group模式的目录分类（如 'analysis'）
        surfaces : Optional[List[str]]
            要诊断的表面列表，不指定则用初始化时的surfaces
        enable_overall : Optional[bool]
            是否执行整体分析，运行时指定则覆盖config
        enable_by_group : Optional[bool]
            是否执行分规格分析，运行时指定则覆盖config
        group_col : Optional[str]
            分规格列名，运行时指定则覆盖config中的设置
        target_groups : Optional[List[str]]
            规格白名单，运行时指定则覆盖config中的设置
        
        Returns
        -------
        Dict[str, Dict]
            按表面名称组织的诊断结果
            {'Top': {...结果...}, 'Bot': {...结果...}}
        """
        surfaces = surfaces or self.surfaces
        results = {}
        
        for surface in surfaces:
            if surface not in self.configs:
                print(f"[跳过] 表面 {surface} 未在配置中")
                continue
            
            config = self.configs[surface]
            if not config.enabled:
                print(f"[跳过] 表面 {surface} 已禁用")
                continue
            
            print(f"\n{'='*70}")
            print(f"[建模前诊断] {surface}表面")
            print(f"{'='*70}")
            
            try:
                # 参数优先级：运行时参数 > 配置参数
                _enable_overall = enable_overall if enable_overall is not None else config.enable_overall
                _enable_by_group = enable_by_group if enable_by_group is not None else config.enable_by_group
                _group_col = group_col or config.group_col
                _target_groups = target_groups or config.target_groups
                
                # 根据模式确定保存目录
                # Overall: pre_diagnosis_dir/overall/{surface}/
                # By Group: pre_diagnosis_dir/by_group/{group_label}/{surface}/
                if _enable_overall and not _enable_by_group:
                    # 纯overall模式
                    save_dir = f'{self.pre_diagnosis_dir}/overall/{surface}'
                    os.makedirs(save_dir, exist_ok=True)
                    
                    result = self.analyzer.analyze(
                        df=df.copy(),
                        surface=surface,
                        delta_col=f'{surface}_Delta',
                        feature_cols=self._get_feature_cols(surface),
                        compute_stats_only=config.compute_stats_only_pre,
                        max_groups=config.max_groups,
                        sample_for_scatter=config.sample_for_scatter,
                        save_dir=save_dir,
                        group_col=None,
                        target_groups=None,
                        enable_overall=True,
                        enable_by_group=False,
                    )
                    results[surface] = result
                    
                elif _enable_by_group and not _enable_overall:
                    # 纯by_group模式
                    # 先检查是否有有效的规格（样本数>=30）
                    work = df.copy()
                    delta_col_check = f'{surface}_Delta'
                    if delta_col_check not in work.columns:
                        print(f"[警告] {surface}_Delta 列不存在，跳过 {surface} by_group 分析")
                        results[surface] = None
                        continue
                    
                    # 检查有多少有效规格
                    data_check = work[[_group_col, delta_col_check]].dropna(subset=[delta_col_check])
                    if _target_groups:
                        data_check = data_check[data_check[_group_col].isin(_target_groups)]
                    
                    valid_specs = data_check[_group_col].value_counts()
                    valid_specs = valid_specs[valid_specs >= 30]  # 只计算样本数>=30的规格
                    
                    if len(valid_specs) == 0:
                        print(f"[跳过] {surface}表面没有有效的规格（样本数>=30）")
                        results[surface] = None
                        continue
                    
                    print(f"[检查] {surface}表面共 {len(valid_specs)} 个有效规格")
                    
                    # save_dir = by_group/{group_label}/{surface}
                    # analyzer会在这个目录下直接创建规格子目录
                    save_dir = f'{self.pre_diagnosis_dir}/by_group/{group_label}/{surface}'
                    os.makedirs(save_dir, exist_ok=True)
                    
                    result = self.analyzer.analyze(
                        df=df.copy(),
                        surface=surface,
                        delta_col=f'{surface}_Delta',
                        feature_cols=self._get_feature_cols(surface),
                        compute_stats_only=config.compute_stats_only_pre,
                        max_groups=config.max_groups,
                        sample_for_scatter=config.sample_for_scatter,
                        save_dir=save_dir,
                        group_col=_group_col,
                        target_groups=_target_groups,
                        enable_overall=False,
                        enable_by_group=True,
                    )
                    results[surface] = result
                    
                else:
                    # 同时启用both模式：分别调用两次
                    overall_result = None
                    by_group_result = None
                    
                    if _enable_overall:
                        overall_save_dir = f'{self.pre_diagnosis_dir}/overall/{surface}'
                        os.makedirs(overall_save_dir, exist_ok=True)
                        
                        overall_result = self.analyzer.analyze(
                            df=df.copy(),
                            surface=surface,
                            delta_col=f'{surface}_Delta',
                            feature_cols=self._get_feature_cols(surface),
                            compute_stats_only=config.compute_stats_only_pre,
                            max_groups=config.max_groups,
                            sample_for_scatter=config.sample_for_scatter,
                            save_dir=overall_save_dir,
                            group_col=None,
                            target_groups=None,
                            enable_overall=True,
                            enable_by_group=False,
                        )
                    
                    if _enable_by_group:
                        # 先检查是否有有效的规格（样本数>=30）
                        work_check = df.copy()
                        delta_col_check = f'{surface}_Delta'
                        if delta_col_check in work_check.columns:
                            data_check = work_check[[_group_col, delta_col_check]].dropna(subset=[delta_col_check])
                            if _target_groups:
                                data_check = data_check[data_check[_group_col].isin(_target_groups)]
                            
                            valid_specs = data_check[_group_col].value_counts()
                            valid_specs = valid_specs[valid_specs >= 30]
                            
                            if len(valid_specs) > 0:
                                print(f"[检查] {surface}表面共 {len(valid_specs)} 个有效规格")
                                by_group_save_dir = f'{self.pre_diagnosis_dir}/by_group/{group_label}/{surface}'
                                os.makedirs(by_group_save_dir, exist_ok=True)
                                
                                by_group_result = self.analyzer.analyze(
                                    df=df.copy(),
                                    surface=surface,
                                    delta_col=f'{surface}_Delta',
                                    feature_cols=self._get_feature_cols(surface),
                                    compute_stats_only=config.compute_stats_only_pre,
                                    max_groups=config.max_groups,
                                    sample_for_scatter=config.sample_for_scatter,
                                    save_dir=by_group_save_dir,
                                    group_col=_group_col,
                                    target_groups=_target_groups,
                                    enable_overall=False,
                                    enable_by_group=True,
                                )
                            else:
                                print(f"[跳过] {surface}表面没有有效的规格（样本数>=30）")
                    
                    # 合并结果
                    results[surface] = {
                        'overall': overall_result,
                        'by_group': by_group_result
                    }
                
                print(f"[完成] {surface}表面建模前诊断")
            except Exception as e:
                print(f"[错误] {surface}表面建模前诊断失败: {e}")
                results[surface] = None
        
        return results
    
    def post_modeling_diagnosis(self,
                                df: pd.DataFrame,
                                models: Dict[str, Any],
                                group_label: str,
                                surfaces: Optional[List[str]] = None,
                                enable_overall: Optional[bool] = None,
                                enable_by_group: Optional[bool] = None,
                                group_col: Optional[str] = None,
                                target_groups: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        建模后诊断：残差质量、方向性偏差、模型校正效果
        
        Parameters
        ----------
        df : pd.DataFrame
            包含原始数据和预测结果的DataFrame
        models : Dict[str, Any]
            训练好的模型字典，如 {'Top': model_obj, 'Bot': model_obj}
        group_label : str
            规格组标签，仅用于by_group模式的目录分类
        surfaces : Optional[List[str]]
            要诊断的表面列表，不指定则用初始化时的surfaces
        enable_overall : Optional[bool]
            是否执行整体分析，运行时指定则覆盖config
        enable_by_group : Optional[bool]
            是否执行分规格分析，运行时指定则覆盖config
        group_col : Optional[str]
            分规格列名，运行时指定则覆盖config中的设置
        target_groups : Optional[List[str]]
            规格白名单，运行时指定则覆盖config中的设置
        
        Returns
        -------
        Dict[str, Dict]
            按表面名称组织的诊断结果
            {'Top': {...结果...}, 'Bot': {...结果...}}
        """
        surfaces = surfaces or self.surfaces
        results = {}
        
        for surface in surfaces:
            if surface not in self.configs:
                print(f"[跳过] 表面 {surface} 未在配置中")
                continue
            
            config = self.configs[surface]
            if not config.enabled:
                print(f"[跳过] 表面 {surface} 已禁用")
                continue
            
            if surface not in models or models[surface] is None:
                print(f"[警告] 表面 {surface} 无训练模型，跳过建模后诊断")
                continue
            
            print(f"\n{'='*70}")
            print(f"[建模后诊断] {surface}表面")
            print(f"{'='*70}")
            
            try:
                # 参数优先级：运行时参数 > 配置参数
                _enable_overall = enable_overall if enable_overall is not None else config.enable_overall
                _enable_by_group = enable_by_group if enable_by_group is not None else config.enable_by_group
                _group_col = group_col or config.group_col
                _target_groups = target_groups or config.target_groups
                
                # 根据模式确定保存目录
                # Overall: post_diagnosis_dir/overall/{surface}/
                # By Group: post_diagnosis_dir/by_group/{group_label}/{surface}/
                if _enable_overall and not _enable_by_group:
                    # 纯overall模式
                    save_dir = f'{self.post_diagnosis_dir}/overall/{surface}'
                    os.makedirs(save_dir, exist_ok=True)
                    
                    result = self.analyzer.analyze(
                        df=df.copy(),
                        surface=surface,
                        delta_col=f'{surface}_Delta',
                        feature_cols=self._get_feature_cols(surface),
                        model_residual_col=f'{surface}_Model_Residual',
                        compute_stats_only=config.compute_stats_only_post,
                        max_groups=config.max_groups,
                        sample_for_scatter=config.sample_for_scatter,
                        save_dir=save_dir,
                        group_col=None,
                        target_groups=None,
                        enable_overall=True,
                        enable_by_group=False,
                    )
                    results[surface] = result
                    
                elif _enable_by_group and not _enable_overall:
                    # 纯by_group模式
                    # 先检查是否有有效的规格（样本数>=30）
                    work = df.copy()
                    delta_col_check = f'{surface}_Delta'
                    if delta_col_check not in work.columns:
                        print(f"[警告] {surface}_Delta 列不存在，跳过 {surface} by_group 分析")
                        results[surface] = None
                        continue
                    
                    # 检查有多少有效规格
                    data_check = work[[_group_col, delta_col_check]].dropna(subset=[delta_col_check])
                    if _target_groups:
                        data_check = data_check[data_check[_group_col].isin(_target_groups)]
                    
                    valid_specs = data_check[_group_col].value_counts()
                    valid_specs = valid_specs[valid_specs >= 30]  # 只计算样本数>=30的规格
                    
                    if len(valid_specs) == 0:
                        print(f"[跳过] {surface}表面没有有效的规格（样本数>=30）")
                        results[surface] = None
                        continue
                    
                    print(f"[检查] {surface}表面共 {len(valid_specs)} 个有效规格")
                    
                    # save_dir = by_group/{group_label}/{surface}
                    # analyzer会在这个目录下直接创建规格子目录
                    save_dir = f'{self.post_diagnosis_dir}/by_group/{group_label}/{surface}'
                    os.makedirs(save_dir, exist_ok=True)
                    
                    result = self.analyzer.analyze(
                        df=df.copy(),
                        surface=surface,
                        delta_col=f'{surface}_Delta',
                        feature_cols=self._get_feature_cols(surface),
                        model_residual_col=f'{surface}_Model_Residual',
                        compute_stats_only=config.compute_stats_only_post,
                        max_groups=config.max_groups,
                        sample_for_scatter=config.sample_for_scatter,
                        save_dir=save_dir,
                        group_col=_group_col,
                        target_groups=_target_groups,
                        enable_overall=False,
                        enable_by_group=True,
                    )
                    results[surface] = result
                    
                else:
                    # 同时启用both模式：分别调用两次
                    overall_result = None
                    by_group_result = None
                    
                    if _enable_overall:
                        overall_save_dir = f'{self.post_diagnosis_dir}/overall/{surface}'
                        os.makedirs(overall_save_dir, exist_ok=True)
                        
                        overall_result = self.analyzer.analyze(
                            df=df.copy(),
                            surface=surface,
                            delta_col=f'{surface}_Delta',
                            feature_cols=self._get_feature_cols(surface),
                            model_residual_col=f'{surface}_Model_Residual',
                            compute_stats_only=config.compute_stats_only_post,
                            max_groups=config.max_groups,
                            sample_for_scatter=config.sample_for_scatter,
                            save_dir=overall_save_dir,
                            group_col=None,
                            target_groups=None,
                            enable_overall=True,
                            enable_by_group=False,
                        )
                    
                    if _enable_by_group:
                        # 先检查是否有有效的规格（样本数>=30）
                        work_check = df.copy()
                        delta_col_check = f'{surface}_Delta'
                        if delta_col_check in work_check.columns:
                            data_check = work_check[[_group_col, delta_col_check]].dropna(subset=[delta_col_check])
                            if _target_groups:
                                data_check = data_check[data_check[_group_col].isin(_target_groups)]
                            
                            valid_specs = data_check[_group_col].value_counts()
                            valid_specs = valid_specs[valid_specs >= 30]
                            
                            if len(valid_specs) > 0:
                                print(f"[检查] {surface}表面共 {len(valid_specs)} 个有效规格")
                                by_group_save_dir = f'{self.post_diagnosis_dir}/by_group/{group_label}/{surface}'
                                os.makedirs(by_group_save_dir, exist_ok=True)
                                
                                by_group_result = self.analyzer.analyze(
                                    df=df.copy(),
                                    surface=surface,
                                    delta_col=f'{surface}_Delta',
                                    feature_cols=self._get_feature_cols(surface),
                                    model_residual_col=f'{surface}_Model_Residual',
                                    compute_stats_only=config.compute_stats_only_post,
                                    max_groups=config.max_groups,
                                    sample_for_scatter=config.sample_for_scatter,
                                    save_dir=by_group_save_dir,
                                    group_col=_group_col,
                                    target_groups=_target_groups,
                                    enable_overall=False,
                                    enable_by_group=True,
                                )
                            else:
                                print(f"[跳过] {surface}表面没有有效的规格（样本数>=30）")
                    
                    # 合并结果
                    results[surface] = {
                        'overall': overall_result,
                        'by_group': by_group_result
                    }
                
                print(f"[完成] {surface}表面建模后诊断")
            except Exception as e:
                print(f"[错误] {surface}表面建模后诊断失败: {e}")
                results[surface] = None
        
        return results
    
    @staticmethod
    def _get_feature_cols(surface: str) -> List[str]:
        """
        获取特定表面的特征列列表（与建模流程保持一致）
        
        Parameters
        ----------
        surface : str
            表面标识 ('Top' 或 'Bot')
        
        Returns
        -------
        List[str]
            该表面对应的特征列列表
        """
        prefix = "Top" if surface == "Top" else "Bot"
        return [
            f"Tin Weight_Actual[g/m2]_GALV_WEIGHT_{prefix.upper()}_Avg",
            f"{prefix}_Current_Sum",
            f"{prefix}_Current_Per_Speed",
            f"{prefix}_Theoretical_Factor",
            "Speed[m/min]_Process_Avg",
            "Dimension_[mm]_Width",
            "Dimension_[mm]_Thickness",
            "Steel_Grade_Encoded",
        ]
    
    def get_summary_dataframes(self, 
                               diagnosis_results: Dict[str, Dict]) -> Dict[str, pd.DataFrame]:
        """
        从诊断结果中提取summary_df（方便导出Excel）
        
        Parameters
        ----------
        diagnosis_results : Dict[str, Dict]
            诊断结果，来自 pre_modeling_diagnosis 或 post_modeling_diagnosis
        
        Returns
        -------
        Dict[str, pd.DataFrame]
            按表面组织的summary数据框
            {'Top': DataFrame(...), 'Bot': DataFrame(...)}
        """
        summaries = {}
        for surface, result in diagnosis_results.items():
            if result is not None and 'summary_df' in result:
                summaries[surface] = result['summary_df']
        return summaries
    
    def set_surface_config(self, surface: str, config: SurfaceEDAConfig) -> None:
        """
        动态更新某个表面的诊断配置
        
        Parameters
        ----------
        surface : str
            表面标识
        config : SurfaceEDAConfig
            新的配置对象
        """
        self.configs[surface] = config
        print(f"[更新] 表面 {surface} 的EDA诊断配置已更新")
    
    def enable_surface(self, surface: str) -> None:
        """启用某个表面的诊断"""
        if surface in self.configs:
            self.configs[surface].enabled = True
            print(f"[启用] 表面 {surface}")
    
    def disable_surface(self, surface: str) -> None:
        """禁用某个表面的诊断"""
        if surface in self.configs:
            self.configs[surface].enabled = False
            print(f"[禁用] 表面 {surface}")
    
    def get_active_surfaces(self) -> List[str]:
        """获取当前启用的表面列表"""
        return [s for s in self.surfaces if self.configs[s].enabled]


# ==========================================
# 工厂函数：便捷初始化
# ==========================================

def create_eda_diagnoser_from_config(config_dict: Dict[str, Any]) -> GroupEDADiagnoser:
    """
    从配置字典创建诊断管理器
    
    Parameters
    ----------
    config_dict : Dict[str, Any]
        配置字典，例：
        {
            'pre_diagnosis_dir': 'result/eda/pre_modeling',
            'post_diagnosis_dir': 'result/eda/post_modeling',
            'eda_top': {
                'enabled': True,
                'compute_stats_only_pre': True,
                'compute_stats_only_post': False,
                'group_col': 'Setpoint_Group_Label',
                'target_groups': ['Top2.799_Bot2.799'],
                ...
            },
            'eda_bot': {
                'enabled': True,
                ...
            }
        }
    
    Returns
    -------
    GroupEDADiagnoser
        初始化后的诊断管理器
    """
    surfaces = []
    surface_configs = {}
    
    # 提取目录配置
    pre_dir = config_dict.get('pre_diagnosis_dir', 'result/eda/pre_modeling')
    post_dir = config_dict.get('post_diagnosis_dir', 'result/eda/post_modeling')
    
    # 按约定的键名解析配置
    for key in ['eda_top', 'eda_bot']:
        if key in config_dict:
            surface = 'Top' if key == 'eda_top' else 'Bot'
            surfaces.append(surface)
            
            cfg_params = config_dict[key].copy() if isinstance(config_dict[key], dict) else {}
            # 移除 'surface' 字段，因为我们已经知道
            cfg_params.pop('surface', None)
            cfg_params['surface'] = surface
            
            surface_configs[surface] = SurfaceEDAConfig(**cfg_params)
    
    return GroupEDADiagnoser(
        surfaces=surfaces or ['Top', 'Bot'],
        surface_configs=surface_configs or None,
        pre_diagnosis_dir=pre_dir,
        post_diagnosis_dir=post_dir
    )


if __name__ == "__main__":
    # 使用示例
    print("GroupEDADiagnoser 已加载，可直接导入使用。")
