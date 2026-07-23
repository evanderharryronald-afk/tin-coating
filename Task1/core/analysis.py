"""
DataAnalyzer: Data analysis and diagnostics

Core functionality:
  - Correlation analysis
  - Residual distribution diagnostics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path



class DataAnalyzer:
    """Data analysis and diagnostics"""

    def __init__(self, df: pd.DataFrame, output_dir: str = "result"):
        """
        Initialize analyzer
        
        Args:
            df: Clean DataFrame to analyze
            output_dir: Result output directory
        """
        self.df = df.copy()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "correlation_result").mkdir(exist_ok=True)
        
        # Set up matplotlib
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        print("[DataAnalyzer] Initialized")

    def analyze_correlations(self, surface='Top'):
        """
        Analyze and visualize correlations
        
        Args:
            surface: 'Top' or 'Bot'
        """
        prefix = 'Top' if surface == 'Top' else 'Bot'
        surface_cn = 'Top' if surface == 'Top' else 'Bot'

        # Determine column names flexibly
        actual_col = None
        lab_col = None
        
        for col in self.df.columns:
            if f'{prefix.upper()}_Avg' in col and 'GALV_WEIGHT' in col and 'Actual' in col:
                actual_col = col
            if '上' in col if surface == 'Top' else '下' in col:
                if '镀层重量' in col and 'A(' in col:
                    lab_col = col
        
        # Fallback to known column names
        if actual_col is None:
            actual_col = f'Tin Weight_Actual[g/m2]_GALV_WEIGHT_{prefix.upper()}_Avg'
        if lab_col is None:
            lab_col = '上表面镀层重量A(XA1_0)' if surface == 'Top' else '下表面镀层重量A(XA1_0)'

        cols_to_check = [
            lab_col,
            actual_col,
            f'{prefix}_Current_Sum',
            f'{prefix}_Theoretical_Factor',
            'Speed[m/min]_Process_Avg',
            'Dimension_[mm]_Thickness',
            'Dimension_[mm]_Width',
            'Steel_Grade_Encoded'
        ]

        # Filter to only existing columns
        cols_to_check = [c for c in cols_to_check if c in self.df.columns]
        
        if len(cols_to_check) < 2:
            print(f"[WARN] Not enough columns for {surface} surface correlation analysis")
            return

        corr_matrix = self.df[cols_to_check].corr()

        print(f"\n========== {surface} Surface Correlation Matrix ==========")
        if lab_col in corr_matrix.columns:
            print(corr_matrix[lab_col].sort_values(ascending=False))
        print("=" * 55 + "\n")

        # Plot
        plt.figure(figsize=(9, 7))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
        plt.title(f'{surface} Surface Parameters vs Lab Measurement Correlation')
        plt.tight_layout()

        save_img_path = self.output_dir / "correlation_result" / f"correlation_{surface}.png"
        plt.savefig(save_img_path, dpi=300)
        print(f"[SAVE] Correlation heatmap: {save_img_path}")
        plt.close()

    def check_residual_distribution(self):
        """Check residual distribution across data"""
        print("\n" + "=" * 50)
        print("Residual Distribution Diagnosis")
        print("=" * 50)
        
        for surface in ['Top', 'Bot']:
            delta_col = f'{surface}_Delta'
            if delta_col not in self.df.columns:
                print(f"[SKIP] {surface}_Delta not found")
                continue
            
            total = len(self.df[delta_col].dropna())
            pos = (self.df[delta_col] > 0).sum()
            neg = (self.df[delta_col] < 0).sum()
            mean_val = self.df[delta_col].mean()
            std_val = self.df[delta_col].std()
            
            print(f"\n[{surface} Surface Delta (Lab - Online)]")
            print(f"  Total valid samples: {total}")
            print(f"  Delta > 0 (online low): {pos} ({pos/total*100:.2f}%)")
            print(f"  Delta < 0 (online high): {neg} ({neg/total*100:.2f}%)")
            print(f"  Mean: {mean_val:.4f} g/m2")
            print(f"  Std: {std_val:.4f} g/m2")
        
        print("=" * 50 + "\n")

    def run(self):
        """Execute full analysis pipeline"""
        print("\n[START] Analysis pipeline")
        
        self.analyze_correlations(surface='Top')
        self.analyze_correlations(surface='Bot')
        self.check_residual_distribution()
        
        print("[END] Analysis pipeline\n")
