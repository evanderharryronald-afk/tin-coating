"""
超参数调优接口 - 复用 run_tuning_optuna.py
"""
import logging
from typing import Dict, Optional, Tuple
from pathlib import Path
import pickle

from ..core import TUNING_DIR

logger = logging.getLogger(__name__)


class SurfaceGroupHyperparameterTuner:
    """
    表面分组超参数调优
    复用 run_tuning_optuna.py
    """
    
    def __init__(self, save_dir: Optional[str] = None):
        """
        初始化调优器
        
        Parameters
        ----------
        save_dir : str, optional
            结果保存目录
        """
        if save_dir is None:
            save_dir = str(TUNING_DIR)
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def tune_single_group(
        self,
        df,
        surface: str,
        value: float,
        n_trials: int = 100,
        n_splits: int = 3,
        timeout: Optional[float] = None,
    ) -> Dict:
        """
        对单个表面分组进行超参数调优
        
        Parameters
        ----------
        df : DataFrame
            分组数据
        surface : str
            'Top' 或 'Bot'
        value : float
            设定厚度值
        n_trials : int
            试验次数
        n_splits : int
            交叉验证折数
        timeout : float, optional
            超时时间（秒）
        
        Returns
        -------
        best_params : dict
            最优超参数
        """
        logger.info(f"开始调优 {surface}_{value} ({n_trials} trials)...")
        
        try:
            import optuna
            from run_tuning_optuna import make_objective_with_timeseries_cv
            
            group_tag = f"{surface}_{value}"
            
            # 创建优化研究（多目标：最小化 RMSE 和 最差方向MAE）
            study = optuna.create_study(
                directions=['minimize', 'minimize'],  # 多目标：同时最小化 RMSE 和 最差方向MAE
                sampler=optuna.samplers.TPESampler(seed=42),
            )
            
            # 创建目标函数
            objective = make_objective_with_timeseries_cv(
                group_df=df,
                surface=surface,
                group_tag=group_tag,
                n_splits=n_splits,
            )
            
            # 优化
            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=False,
            )
            
            # 多目标优化：获取 Pareto 前沿上的最优试验
            # 使用 best_trials 获取 Pareto 前沿上所有试验，选择 RMSE 最小的一个
            best_trials = study.best_trials
            if best_trials:
                # 按 RMSE（第一个目标）排序，取最小的
                best_trial = min(best_trials, key=lambda t: t.values[0] if t.values else float('inf'))
                best_params = best_trial.params
                best_values = best_trial.values  # [best_rmse, best_worst_mae]
                
                logger.info(f"完成调优 {surface}_{value}")
                logger.info(f"  - 最优RMSE: {best_values[0]:.6f}")
                logger.info(f"  - 最优最差方向MAE: {best_values[1]:.6f}")
                logger.info(f"最优超参数: {best_params}")
            else:
                raise ValueError(f"未找到有效的调优试验 {surface}_{value}")
            
            # 保存调优结果
            self._save_study_results(study, group_tag)
            
            return {
                'group_key': f'{surface}_{value}',
                'best_params': best_params,
                'best_values': best_values,  # [best_rmse, best_worst_mae]
                'n_trials': len(study.trials),
                'study': study,
            }
        
        except ImportError as e:
            logger.error(f"无法导入调优函数: {e}")
            raise
        except Exception as e:
            logger.error(f"调优失败 {surface}_{value}: {e}")
            raise
    
    def tune_all_groups(
        self,
        groups: Dict,
        n_trials: int = 100,
        n_splits: int = 3,
        timeout: Optional[float] = None,
    ) -> Dict:
        """
        对所有表面分组进行超参数调优
        
        Parameters
        ----------
        groups : dict
            分组信息 {group_key: (surface, value, group_df, n_samples)}
        n_trials : int
            每个分组的试验次数
        n_splits : int
            交叉验证折数
        timeout : float, optional
            超时时间（秒）
        
        Returns
        -------
        all_tuning_results : dict
            所有调优结果 {group_key: best_params_dict}
        """
        all_tuning_results = {}
        
        for group_key, (surface, value, group_df, n_samples) in groups.items():
            logger.info(f"调优 {group_key} ({n_samples} 样本)...")
            
            try:
                tuning_result = self.tune_single_group(
                    df=group_df,
                    surface=surface,
                    value=value,
                    n_trials=n_trials,
                    n_splits=n_splits,
                    timeout=timeout,
                )
                
                all_tuning_results[group_key] = {
                    'best_params': tuning_result['best_params'],
                    'best_values': tuning_result['best_values'],  # [best_rmse, best_worst_mae]
                    'n_trials': tuning_result['n_trials'],
                    'status': 'success',
                }
            
            except Exception as e:
                logger.error(f"调优失败 {group_key}: {e}")
                all_tuning_results[group_key] = {
                    'best_params': None,
                    'best_values': None,
                    'status': 'failed',
                    'error': str(e),
                }
        
        logger.info(f"完成所有调优，共 {len(all_tuning_results)} 个分组")
        
        return all_tuning_results
    
    def _save_study_results(self, study, group_tag: str) -> None:
        """
        保存Optuna研究结果
        
        Parameters
        ----------
        study : optuna.Study
            调优研究
        group_tag : str
            分组标签
        """
        try:
            # 保存study对象
            study_file = self.save_dir / f'{group_tag}_study.pkl'
            with open(study_file, 'wb') as f:
                pickle.dump(study, f)
            logger.info(f"保存调优研究: {study_file}")
        
        except Exception as e:
            logger.warning(f"保存研究失败: {e}")
    
    def get_best_params_dict(self, all_tuning_results: Dict) -> Dict:
        """
        从调优结果中提取最优超参数字典
        
        Parameters
        ----------
        all_tuning_results : dict
            所有调优结果
        
        Returns
        -------
        params_dict : dict
            超参数字典 {group_key: best_params}
        """
        params_dict = {}
        
        for group_key, result in all_tuning_results.items():
            if result.get('status') == 'success':
                params_dict[group_key] = result['best_params']
        
        return params_dict
