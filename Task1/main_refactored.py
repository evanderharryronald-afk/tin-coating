"""
main_refactored.py

Refactored main script using modular data processing pipeline

Pipeline:
  1. Table merge (MergePipeline)
  2. Data preprocessing (DataPreprocessor)
  3. Data analysis (DataAnalyzer)
  4. Model training (ModelPipeline)
"""

import sys
from pathlib import Path

# Add core modules to path
sys.path.insert(0, str(Path(__file__).parent))

from core.merge import MergePipeline
from core.preprocess import DataPreprocessor
from core.analysis import DataAnalyzer
from core.modeling import ModelPipeline


def main():
    print("\n" + "=" * 70)
    print("Refactored Data Processing Pipeline")
    print("=" * 70 + "\n")

    # ══════════════════════════════════════════════════════════════════════
    # Step 1: Merge master and sub tables
    # ══════════════════════════════════════════════════════════════════════
    print("[Step 1/4] Merging master and sub tables...")
    config_path = Path(__file__).parent / "config.yaml"
    
    merger = MergePipeline(config_path=str(config_path))
    merged_df = merger.run()

    # ══════════════════════════════════════════════════════════════════════
    # Step 2: Data preprocessing
    # ══════════════════════════════════════════════════════════════════════
    print("[Step 2/4] Data preprocessing...")
    preprocessor = DataPreprocessor(merged_df, output_dir="result")
    clean_df = preprocessor.run()

    # ══════════════════════════════════════════════════════════════════════
    # Step 3: Data analysis
    # ══════════════════════════════════════════════════════════════════════
    print("[Step 3/4] Data analysis...")
    analyzer = DataAnalyzer(clean_df, output_dir="result")
    analyzer.run()

    # ══════════════════════════════════════════════════════════════════════
    # Step 4: Model training and evaluation
    # ══════════════════════════════════════════════════════════════════════
    print("[Step 4/4] Model training and evaluation...")
    modeler = ModelPipeline(clean_df, output_dir="result")
    modeler.run()

    # ══════════════════════════════════════════════════════════════════════
    # Done
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Pipeline Completed Successfully!")
    print("=" * 70 + "\n")

    return clean_df


if __name__ == "__main__":
    clean_df = main()
