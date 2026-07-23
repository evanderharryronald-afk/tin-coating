"""
DataPreprocessor: Data preprocessing and feature engineering

Core functionality:
  - Column name correction
  - Feature engineering (current sum, theoretical factor, steel grade encoding)
  - Residual calculation
  - Outlier detection and filtering
  - Export cleaned data and diagnostic reports
"""

import pandas as pd
import numpy as np
from pathlib import Path


class DataPreprocessor:
    """Data preprocessing and feature engineering"""

    def __init__(self, df: pd.DataFrame, output_dir: str = "result"):
        """
        Initialize preprocessor
        
        Args:
            df: Raw DataFrame
            output_dir: Result output directory
        """
        self.df = df.copy()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "cleaned_data").mkdir(exist_ok=True)
        
        print("[DataPreprocessor] Initialized")

    # ── feature engineering ──
    def create_current_features(self):
        """Create current sum features"""
        bot_curr_cols = [f'Tining Section_CURRENT[A]_GL_{i}_Avg' for i in range(1, 37, 2)]
        top_curr_cols = [f'Tining Section_CURRENT[A]_GL_{i}_Avg' for i in range(2, 37, 2)]

        bot_exist = [c for c in bot_curr_cols if c in self.df.columns]
        top_exist = [c for c in top_curr_cols if c in self.df.columns]

        if bot_exist:
            self.df['Bot_Current_Sum'] = self.df[bot_exist].sum(axis=1)
        if top_exist:
            self.df['Top_Current_Sum'] = self.df[top_exist].sum(axis=1)

        print("[OK] Current features created")
        return self

    def create_theoretical_factor(self):
        """Create theoretical factor features"""
        if 'Dimension_[mm]_Width' not in self.df.columns:
            print("[WARN] Dimension_[mm]_Width not found")
            return self

        self.df['Width_m'] = self.df['Dimension_[mm]_Width'] / 1000.0
        speed = self.df['Speed[m/min]_Process_Avg'].replace(0, np.nan)

        if 'Top_Current_Sum' in self.df.columns:
            self.df['Top_Theoretical_Factor'] = self.df['Top_Current_Sum'] / (speed * self.df['Width_m'])
        if 'Bot_Current_Sum' in self.df.columns:
            self.df['Bot_Theoretical_Factor'] = self.df['Bot_Current_Sum'] / (speed * self.df['Width_m'])

        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)

        print("[OK] Theoretical factor features created")
        return self

    def create_steel_grade_encoding(self):
        """Create steel grade frequency encoding feature"""
        if 'Steel Grade' not in self.df.columns:
            print("[WARN] Steel Grade not found")
            self.df['Steel_Grade_Encoded'] = 0
            return self

        grade_freq = self.df['Steel Grade'].value_counts(normalize=True).to_dict()
        self.df['Steel_Grade_Encoded'] = self.df['Steel Grade'].map(grade_freq).fillna(0)
        print(f"[OK] Steel grade encoding created: {len(grade_freq)} types")
        return self

    def create_residuals(self):
        """Calculate residuals: lab value - online value"""
        surfaces = [
            ('Top', 'Top表面镀层重量A(XA1_0)', 'Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Avg'),
            ('Bot', 'Bot表面镀层重量A(XA1_0)', 'Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Avg'),
        ]

        # Use flexible column matching
        lab_cols = ['上表面镀层重量A(XA1_0)', '下表面镀层重量A(XA1_0)']
        online_cols = ['Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Avg', 'Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Avg']
        
        # Top surface
        lab_col_top = next((c for c in lab_cols if '上' in c and c in self.df.columns), None)
        online_col_top = next((c for c in online_cols if 'TOP' in c and c in self.df.columns), None)
        if lab_col_top and online_col_top:
            self.df['Top_Delta'] = self.df[lab_col_top] - self.df[online_col_top]
        
        # Bot surface
        lab_col_bot = next((c for c in lab_cols if '下' in c and c in self.df.columns), None)
        online_col_bot = next((c for c in online_cols if 'BOT' in c and c in self.df.columns), None)
        if lab_col_bot and online_col_bot:
            self.df['Bot_Delta'] = self.df[lab_col_bot] - self.df[online_col_bot]

        print("[OK] Residuals created")
        return self

    # ── Outlier detection ──
    def diagnose_and_filter_outliers(self,
                                     clean_save_path: str = None,
                                     filtered_save_path: str = None):
        """
        Detect and filter outliers
        
        Rules:
          1. Missing required fields
          2. Large residuals in steady state
          3. Very low speed / shutdown data
        """
        if clean_save_path is None:
            clean_save_path = self.output_dir / "cleaned_data" / "cleaned_data.xlsx"
        if filtered_save_path is None:
            filtered_save_path = self.output_dir / "cleaned_data" / "filtered_outliers.xlsx"

        initial_count = len(self.df)
        self.df['Filter_Reason'] = ""

        # -- Rule 1: Missing required fields --
        required_cols = [
            'Top_Current_Sum', 'Bot_Current_Sum',
            'Top_Theoretical_Factor', 'Bot_Theoretical_Factor',
            'Speed[m/min]_Process_Avg', 'Dimension_[mm]_Width', 'Dimension_[mm]_Thickness',
            'Tin Weight_Actual[g/m2]_GALV_WEIGHT_TOP_Avg', 'Tin Weight_Actual[g/m2]_GALV_WEIGHT_BOT_Avg',
            'Top_Delta', 'Bot_Delta'
        ]
        valid_required_cols = [c for c in required_cols if c in self.df.columns]
        null_mask = self.df[valid_required_cols].isnull().any(axis=1)
        self.df.loc[null_mask, 'Filter_Reason'] += "Missing required fields; "

        # -- Calculate residual thresholds --
        valid_df = self.df[~null_mask]
        
        top_delta_std = valid_df['Top_Delta'].std() if 'Top_Delta' in valid_df.columns else np.nan
        top_delta_mean = valid_df['Top_Delta'].mean() if 'Top_Delta' in valid_df.columns else np.nan
        bot_delta_std = valid_df['Bot_Delta'].std() if 'Bot_Delta' in valid_df.columns else np.nan
        bot_delta_mean = valid_df['Bot_Delta'].mean() if 'Bot_Delta' in valid_df.columns else np.nan

        top_threshold = 3.5 * top_delta_std if not np.isnan(top_delta_std) else np.inf
        bot_threshold = 3.5 * bot_delta_std if not np.isnan(bot_delta_std) else np.inf

        # -- Rule 2: Large residuals in steady state --
        if 'Speed[m/min]_Process_Avg' in self.df.columns and 'Top_Delta' in self.df.columns:
            steady_speed_mask = self.df['Speed[m/min]_Process_Avg'] > 80
            top_outlier_mask = (self.df['Top_Delta'] - top_delta_mean).abs() > top_threshold
            self.df.loc[
                steady_speed_mask & top_outlier_mask, 'Filter_Reason'
            ] += f"Top surface large residual in steady state; "

        if 'Speed[m/min]_Process_Avg' in self.df.columns and 'Bot_Delta' in self.df.columns:
            steady_speed_mask = self.df['Speed[m/min]_Process_Avg'] > 80
            bot_outlier_mask = (self.df['Bot_Delta'] - bot_delta_mean).abs() > bot_threshold
            self.df.loc[
                steady_speed_mask & bot_outlier_mask, 'Filter_Reason'
            ] += f"Bot surface large residual in steady state; "

        # -- Rule 3: Very low speed / shutdown --
        if 'Speed[m/min]_Process_Avg' in self.df.columns:
            low_speed_mask = self.df['Speed[m/min]_Process_Avg'] <= 20
            self.df.loc[low_speed_mask, 'Filter_Reason'] += "Very low speed / shutdown; "

        # -- Separate clean and outlier data --
        filtered_df = self.df[self.df['Filter_Reason'] != ""].copy()
        clean_df = self.df[self.df['Filter_Reason'] == ""].copy()

        # -- Export --
        cols_to_export = ['Coil ID', 'Steel Grade', 'Speed[m/min]_Process_Avg',
                          'Top_Delta', 'Bot_Delta', 'Filter_Reason']
        cols_to_export = [c for c in cols_to_export if c in filtered_df.columns]
        
        Path(filtered_save_path).parent.mkdir(parents=True, exist_ok=True)
        if len(filtered_df) > 0:
            filtered_df[cols_to_export].to_excel(filtered_save_path, index=False)
        else:
            print("[INFO] No outliers detected")

        Path(clean_save_path).parent.mkdir(parents=True, exist_ok=True)
        clean_df.to_excel(clean_save_path, index=False)

        # -- Print summary --
        print("\n" + "=" * 50)
        print("Data Cleaning and Outlier Detection Summary")
        print("=" * 50)
        print(f"Original rows: {initial_count}")
        print(f"Filtered outliers: {len(filtered_df)} ({len(filtered_df) / initial_count * 100:.2f}%)")
        print(f"Clean samples: {len(clean_df)}")
        print(f"Outliers saved: {filtered_save_path}")
        print(f"Clean data saved: {clean_save_path}")
        print("=" * 50 + "\n")

        # Update self.df to clean dataset
        self.df = clean_df.copy()
        return self

    # ── Main entry point ──
    def run(self) -> pd.DataFrame:
        """Execute full preprocessing pipeline"""
        print("\n[START] Preprocessing pipeline")
        self.create_current_features() \
            .create_theoretical_factor() \
            .create_steel_grade_encoding() \
            .create_residuals() \
            .diagnose_and_filter_outliers()
        
        print("[END] Preprocessing pipeline\n")
        return self.df
