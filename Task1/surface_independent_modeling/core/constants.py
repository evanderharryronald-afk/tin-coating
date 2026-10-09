"""
常量和配置定义
"""
import os
from pathlib import Path

# ============================================
# 数据列名常量
# ============================================
TOP_SETPOINT_COL = 'Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_TOP_Avg'
BOT_SETPOINT_COL = 'Tin Weight_Setpoints[g/m2]_GALV_WEIGHT_BOT_Avg'

TOP_ACTUAL_COL = 'Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Avg'
BOT_ACTUAL_COL = 'Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Avg'

TOP_LAB_COL = '上表面镀层重量A(XA1_0)'
BOT_LAB_COL = '下表面镀层重量A(XA1_0)'

TOP_DELTA_COL = 'Top_Delta'
BOT_DELTA_COL = 'Bot_Delta'

TOP_CURRENT_SUM_COL = 'Top_Current_Sum'
BOT_CURRENT_SUM_COL = 'Bot_Current_Sum'

SPEED_COL = 'Speed[m/min]_Process_Avg'
WIDTH_COL = 'Dimension_[mm]_Width'
THICKNESS_COL = 'Dimension_[mm]_Thickness'
STEEL_GRADE_COL = 'Steel_Grade_Encoded'

TOP_CURRENT_PER_SPEED = 'Top_Current_Per_Speed'
BOT_CURRENT_PER_SPEED = 'Bot_Current_Per_Speed'

TOP_THEORETICAL_FACTOR = 'Top_Theoretical_Factor'
BOT_THEORETICAL_FACTOR = 'Bot_Theoretical_Factor'

TIME_COL = 'Produce Time'
COIL_ID_COL = 'Coil ID'

# ============================================
# 特征列表 (用于建模的特征)
# ============================================
FEATURE_COLUMNS_TOP = [
    TOP_ACTUAL_COL,
    TOP_CURRENT_SUM_COL,
    SPEED_COL,
    WIDTH_COL,
    THICKNESS_COL,
    TOP_THEORETICAL_FACTOR,
]

FEATURE_COLUMNS_BOT = [
    BOT_ACTUAL_COL,
    BOT_CURRENT_SUM_COL,
    SPEED_COL,
    WIDTH_COL,
    THICKNESS_COL,
    BOT_THEORETICAL_FACTOR,
]

# ============================================
# 文件路径
# ============================================
TASK1_ROOT = Path(__file__).parent.parent.parent
FEATURED_DATA_PATH = TASK1_ROOT / 'result' / 'data' / 'feature_engineered_data' / 'featured_data.xlsx'

# 输出目录
OUTPUT_DIR = TASK1_ROOT / 'result' / 'surface_independent_modeling'
GROUPED_DATA_DIR = OUTPUT_DIR / '01_grouped_data'
CORRELATION_DIR = OUTPUT_DIR / '02_correlation_analysis'
EDA_DIR = OUTPUT_DIR / '03_eda_analysis'
TUNING_DIR = OUTPUT_DIR / '04_hyperparameter_tuning'
MODELS_DIR = OUTPUT_DIR / '05_trained_models'
EVALUATION_DIR = OUTPUT_DIR / '06_model_evaluation'
EXPLAINABILITY_DIR = OUTPUT_DIR / '07_explainability'

# ============================================
# 数据处理配置
# ============================================
MIN_GROUP_SAMPLES = 500  # 最小样本数

# 训练-验证-测试比例
TRAIN_RATIO = 0.65
VAL_RATIO = 0.20
TEST_RATIO = 0.15

# ============================================
# 建模配置
# ============================================
DEFAULT_MODEL_PARAMS = {
    'damping': 0.6,
    'pos_boost': 4.6,
    'alpha_smoothing': 1.0,
    'max_iter': 200,
    'learning_rate': 0.05,
    'max_depth': 4,
}

# 超参数调优范围
HYPERPARAMETER_RANGES = {
    'damping': (0.0, 1.0),
    'pos_boost': (1.0, 10.0),
    'alpha_smoothing': (0.5, 2.0),
    'max_iter': (50, 300),
    'learning_rate': (0.01, 0.2),
    'max_depth': (2, 8),
}

# ============================================
# 分析配置
# ============================================
CORRELATION_METHODS = ['pearson', 'spearman']  # 相关性计算方法
COMPUTE_MI = True  # 是否计算互信息
COMPUTE_DCOR = True  # 是否计算距离相关性

# ============================================
# 日志配置
# ============================================
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'

# ============================================
# 绘图配置
# ============================================
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'
HEATMAP_CMAP = 'coolwarm'
