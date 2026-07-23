"""
ModelPipeline: Model training and evaluation

Core functionality:
  - Feature engineering for modeling
  - Residual correction model training
  - Model evaluation and diagnostics
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def compute_direction_sample_weight(y_delta, pos_boost=1.0, damping=0.0):
    """
    Compute sample weights based on residual direction
    
    Args:
        y_delta: Residual series (lab - online)
        pos_boost: Boost factor for positive residuals (when damping > 0)
        damping: Balance factor (0 = no weighting, higher = more balance)
    
    Returns:
        Sample weight series
    """
    if damping <= 0:
        return pd.Series(1.0, index=y_delta.index)

    pos_mask = y_delta > 0
    neg_mask = y_delta < 0
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()
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
    """
    Direct residual modeling (delta = lab - online)
    
    Features:
      - Model residuals instead of absolute values
      - Optional monotonic constraint on online measurement
      - EWMA smoothing to reduce prediction noise
      - Optional direction-aware sample weighting
    """

    def __init__(self, monotonic_feature_idx=None, alpha_smoothing=0.7,
                 pos_boost=1.0, damping=0.0):
        """
        Initialize model
        
        Args:
            monotonic_feature_idx: Index of feature with monotonic constraint
            alpha_smoothing: EWMA alpha for smoothing
            pos_boost: Boost factor for minority direction
            damping: Direction balance factor
        """
        self.alpha_smoothing = alpha_smoothing
        self.pos_boost = pos_boost
        self.damping = damping
        self.monotonic_feature_idx = monotonic_feature_idx
        self.model = None

    def _build_model(self, n_features):
        """Build gradient boosting model with optional monotonic constraint"""
        monotonic_cst = None
        if self.monotonic_feature_idx is not None:
            monotonic_cst = [0] * n_features
            monotonic_cst[self.monotonic_feature_idx] = -1
        
        self.model = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_depth=4,
            loss='absolute_error',
            monotonic_cst=monotonic_cst,
            random_state=42
        )

    def fit(self, X, y_delta):
        """
        Train model on residuals
        
        Args:
            X: Feature matrix
            y_delta: Target residuals
        """
        self._build_model(n_features=X.shape[1])
        sample_weight = compute_direction_sample_weight(
            y_delta, pos_boost=self.pos_boost, damping=self.damping
        )
        self.model.fit(X, y_delta, sample_weight=sample_weight.values)

    def predict_smooth(self, X, online_actual):
        """
        Predict with EWMA smoothing
        
        Args:
            X: Feature matrix
            online_actual: Online measurement values
        
        Returns:
            (final_predictions, smoothed_residuals)
        """
        predicted_delta_raw = self.model.predict(X)
        delta_series = pd.Series(predicted_delta_raw, index=X.index)
        predicted_delta_smooth = delta_series.ewm(alpha=self.alpha_smoothing).mean()
        final_pred = online_actual + predicted_delta_smooth
        return final_pred, predicted_delta_smooth


class ModelPipeline:
    """Model training pipeline for residual correction"""

    def __init__(self, df: pd.DataFrame, output_dir: str = "result"):
        """
        Initialize model pipeline
        
        Args:
            df: Clean DataFrame (from DataPreprocessor)
            output_dir: Result output directory
        """
        self.df = df.copy()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "fitting_result").mkdir(exist_ok=True)
        
        # Set up matplotlib
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        print("[ModelPipeline] Initialized")

    def _prepare_features(self, surface='Top'):
        """
        Prepare feature columns for modeling
        
        Returns:
            (X, feature_cols, online_feature_idx, y_true, online_actual, y_delta)
        """
        prefix = 'Top' if surface == 'Top' else 'Bot'
        
        # Online measurement column
        online_col = f'Tin Weight_Actual[g/m2]_GALV_WEIGHT_{prefix.upper()}_Avg'
        
        # Create speed-based features if not exist
        speed_col = 'Speed[m/min]_Process_Avg'
        current_col = f'{prefix}_Current_Sum'
        
        if f'{prefix}_Current_Per_Speed' not in self.df.columns:
            self.df[f'{prefix}_Current_Per_Speed'] = self.df[current_col] / (self.df[speed_col] + 1e-5)
        
        # Feature columns (online measurement first for monotonic constraint)
        feature_cols = [
            online_col,
            current_col,
            f'{prefix}_Current_Per_Speed',
            f'{prefix}_Theoretical_Factor',
            speed_col,
            'Dimension_[mm]_Width',
            'Dimension_[mm]_Thickness',
            'Steel_Grade_Encoded'
        ]
        
        # Filter to existing columns
        feature_cols = [c for c in feature_cols if c in self.df.columns]
        online_feature_idx = feature_cols.index(online_col)
        
        X = self.df[feature_cols]
        
        # Target variables
        delta_col = f'{surface}_Delta'
        y_delta = self.df[delta_col]
        online_actual = self.df[online_col]
        
        # Lab measurement column
        lab_col = '上表面镀层重量A(XA1_0)' if surface == 'Top' else '下表面镀层重量A(XA1_0)'
        y_true = self.df[lab_col]
        
        return X, feature_cols, online_feature_idx, y_true, online_actual, y_delta

    def run_surface(self, surface='Top', damping=0.0, alpha_smoothing=0.7, pos_boost=1.0):
        """
        Train and evaluate model for one surface
        
        Args:
            surface: 'Top' or 'Bot'
            damping: Direction balance factor (0-1)
            alpha_smoothing: EWMA smoothing factor (0-1)
            pos_boost: Boost factor for minority direction
        """
        prefix = 'Top' if surface == 'Top' else 'Bot'
        
        print(f"\n{'='*60}")
        print(f"[{surface} Surface] Model Training and Evaluation")
        print(f"  damping={damping}, alpha_smoothing={alpha_smoothing}")
        print(f"{'='*60}")
        
        # Prepare features
        X, feature_cols, online_feature_idx, y_true, online_actual, y_delta = \
            self._prepare_features(surface=surface)
        
        # Split data (time-based, no shuffle)
        X_train, X_test, y_delta_train, y_delta_test, \
        actual_train, actual_test, y_true_train, y_true_test = \
            train_test_split(X, y_delta, online_actual, y_true, 
                           test_size=0.2, shuffle=False)
        
        # Train model
        corrector = ResidualCorrectionModel(
            monotonic_feature_idx=online_feature_idx,
            alpha_smoothing=alpha_smoothing,
            pos_boost=pos_boost,
            damping=damping
        )
        corrector.fit(X_train, y_delta_train)
        
        # Predict
        pred_series, predicted_delta_smooth = corrector.predict_smooth(X_test, actual_test)
        
        y_true_series = y_true_test
        online_series = actual_test
        
        raw_residuals = y_true_series - online_series
        model_residuals = y_true_series - pred_series
        
        # -- Direction diagnostics --
        print(f"\n[Direction Diagnostics]")
        mask_pos = (raw_residuals > 0)
        mask_neg = (raw_residuals < 0)
        
        if mask_pos.sum() > 0:
            mae_raw_pos = raw_residuals[mask_pos].abs().mean()
            mae_model_pos = model_residuals[mask_pos].abs().mean()
            print(f"  Online low ({mask_pos.sum()} samples): {mae_raw_pos:.4f} -> {mae_model_pos:.4f}")
        
        if mask_neg.sum() > 0:
            mae_raw_neg = raw_residuals[mask_neg].abs().mean()
            mae_model_neg = model_residuals[mask_neg].abs().mean()
            print(f"  Online high ({mask_neg.sum()} samples): {mae_raw_neg:.4f} -> {mae_model_neg:.4f}")
        
        # -- Performance metrics --
        r2_online = r2_score(y_true_series, online_series)
        r2_model = r2_score(y_true_series, pred_series)
        rmse_online = np.sqrt(mean_squared_error(y_true_series, online_series))
        rmse_model = np.sqrt(mean_squared_error(y_true_series, pred_series))
        
        print(f"\n[Performance Metrics]")
        print(f"  Online instrument: R2={r2_online:.4f}, RMSE={rmse_online:.4f}")
        print(f"  Model correction:  R2={r2_model:.4f}, RMSE={rmse_model:.4f}")
        
        # -- Visualizations --
        self._plot_fitting_result(y_true_series, online_series, pred_series, surface, 
                                 X_test.index[0], X_test.index[-1])
        self._plot_residual_analysis(raw_residuals, model_residuals, surface, 
                                    X_test.index[0], X_test.index[-1])
        
        return corrector

    def _plot_fitting_result(self, y_true, online, pred, surface, start_idx, end_idx):
        """Plot fitting comparison"""
        plt.figure(figsize=(12, 5))
        plt.plot(y_true, label='Lab Measurement (True)', color='black', linewidth=1.5)
        plt.plot(online, label='Online Instrument (Raw)', color='red', linestyle='--', alpha=0.7)
        plt.plot(pred, label='Model Corrected', color='green', linewidth=1.5, alpha=0.85)
        plt.title(f'{surface} Surface Fitting Comparison (Row {start_idx} to {end_idx})')
        plt.xlabel('Original Row Index')
        plt.ylabel('Coating Weight (g/m2)')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        
        save_path = self.output_dir / "fitting_result" / f"fitting_result_{surface}.png"
        plt.savefig(save_path, dpi=300)
        print(f"[SAVE] Fitting plot: {save_path}")
        plt.close()

    def _plot_residual_analysis(self, raw_residuals, model_residuals, surface, start_idx, end_idx):
        """Plot residual analysis"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})
        
        ax1.plot(raw_residuals, label='Online Residuals', color='red', alpha=0.5, linewidth=1)
        ax1.plot(model_residuals, label='Model Residuals', color='green', alpha=0.8, linewidth=1.2)
        ax1.axhline(0, color='black', linestyle='--', linewidth=1)
        ax1.set_title(f'{surface} Surface Residual Comparison (Row {start_idx} to {end_idx})')
        ax1.set_xlabel('Original Row Index')
        ax1.set_ylabel('Residual (g/m2)')
        ax1.legend()
        ax1.grid(True, linestyle=':', alpha=0.6)
        
        sns.histplot(raw_residuals, ax=ax2, color='red', label='Online Residuals', 
                    kde=True, stat="density", alpha=0.3)
        sns.histplot(model_residuals, ax=ax2, color='green', label='Model Residuals', 
                    kde=True, stat="density", alpha=0.5)
        ax2.axvline(0, color='black', linestyle='--', linewidth=1)
        ax2.set_title(f'{surface} Surface Residual Distribution')
        ax2.set_xlabel('Residual (g/m2)')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        save_path = self.output_dir / "fitting_result" / f"residual_analysis_{surface}.png"
        plt.savefig(save_path, dpi=300)
        print(f"[SAVE] Residual analysis: {save_path}")
        plt.close()

    def run(self):
        """Execute full modeling pipeline"""
        print("\n[START] Modeling pipeline")
        
        # Train models for both surfaces with default configuration
        # damping=0.0, alpha_smoothing=0.7 is from grid search as optimal
        self.run_surface('Top', damping=0.0, alpha_smoothing=0.7)
        self.run_surface('Bot', damping=0.0, alpha_smoothing=0.7)
        
        print("\n[END] Modeling pipeline\n")
