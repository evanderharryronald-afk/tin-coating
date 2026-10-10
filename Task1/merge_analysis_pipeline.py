# -*- coding: utf-8 -*-
"""
spec_merge_analyzer - Framework for merging feasibility analysis
v3.0 - Configuration-driven, multi-strategy support

Run directly in PyCharm - no CLI arguments needed
"""

import sys
import os
import logging

# Import package
try:
    import spec_merge_analyzer as analyzer
except ImportError as e:
    print(f"[E] Failed to import spec_merge_analyzer: {e}")
    print("    Please ensure spec_merge_analyzer package is in current directory or Python path")
    sys.exit(1)


def main():
    """
    Main program - select one of the execution methods below
    """
    
    # ========== [ACTIVE - Run all enabled strategies] ==========
    run_all_strategies()
    
    
    # ========== [Other methods - uncomment to use] ==========
    
    # # Method 2: Run only surface_setpoint_level strategy
    # run_single_strategy('surface_setpoint_level', 'all')
    
    # # Method 3: Run only spec_group_cartesian strategy
    # run_single_strategy('spec_group_cartesian', 'all')
    
    # # Method 4: Run surface_setpoint_level strategy in phases
    # run_single_strategy('surface_setpoint_level', 'a')   # Phase A only
    # run_single_strategy('surface_setpoint_level', 'b')   # Phase B only
    # run_single_strategy('surface_setpoint_level', 'c')   # Phase C only
    
    # # Method 5: Run phase A for all enabled strategies (quick screening)
    # run_all_strategies_phase('a')
    
    # # Method 6: Custom config file
    # run_all_strategies(config_path='custom_config.yaml')


def run_all_strategies(config_path: str = 'grouping_config.yaml', phase: str = 'all'):
    """
    Run all enabled grouping strategies
    
    Args:
        config_path: path to config file
        phase: analysis phase ('a', 'b', 'c', 'all')
    """
    try:
        # Initialize config
        analyzer.init_config(config_path)
        
        # Print config summary
        analyzer.print_summary()
        
        # Run all enabled strategies
        print("\n" + "=" * 80)
        print("[INFO] Running all enabled grouping strategies")
        print("=" * 80)
        
        results = analyzer.run_all_enabled_strategies(phase=phase)
        
        # Check results
        all_success = all(results.values())
        
        print("\n" + "=" * 80)
        print("[SUMMARY] Execution results")
        print("=" * 80)
        for strategy_name, success in results.items():
            status = "[OK]" if success else "[FAIL]"
            print(f"  {strategy_name}: {status}")
        print("=" * 80)
        
        if not all_success:
            failed = [name for name, success in results.items() if not success]
            print(f"\n[WARN] Failed strategies: {', '.join(failed)}")
            return False
        
        # Output file summary
        show_output_files()
        
        return True
    
    except Exception as e:
        print(f"\n[ERROR] Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_single_strategy(strategy_name: str, phase: str = 'all', 
                       config_path: str = 'grouping_config.yaml'):
    """
    Run a single grouping strategy
    
    Args:
        strategy_name: strategy name ('surface_setpoint_level' or 'spec_group_cartesian')
        phase: analysis phase ('a', 'b', 'c', 'all')
        config_path: path to config file
    """
    try:
        # Initialize config
        analyzer.init_config(config_path)
        
        # Print config summary
        analyzer.print_summary()
        
        # Run specified strategy
        print("\n" + "=" * 80)
        print(f"[INFO] Running strategy: {strategy_name}")
        print("=" * 80)
        
        success = analyzer.run_analysis_with_strategy(strategy_name, phase=phase)
        
        print("\n" + "=" * 80)
        if success:
            print(f"[OK] Strategy '{strategy_name}' completed successfully")
            show_output_files(strategy_name)
        else:
            print(f"[FAIL] Strategy '{strategy_name}' failed")
        print("=" * 80)
        
        return success
    
    except Exception as e:
        print(f"\n[ERROR] Strategy '{strategy_name}' execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_strategies_phase(phase: str = 'a', config_path: str = 'grouping_config.yaml'):
    """
    Run all enabled strategies for a specific phase
    
    Args:
        phase: analysis phase ('a', 'b', 'c')
        config_path: path to config file
    """
    try:
        # Initialize config
        analyzer.init_config(config_path)
        
        # Print config summary
        analyzer.print_summary()
        
        # Run all enabled strategies for specified phase
        print("\n" + "=" * 80)
        print(f"[INFO] Running all enabled strategies - Phase {phase.upper()}")
        print("=" * 80)
        
        results = analyzer.run_all_enabled_strategies(phase=phase)
        
        # Check results
        all_success = all(results.values())
        
        print("\n" + "=" * 80)
        print("[SUMMARY] Execution results")
        print("=" * 80)
        for strategy_name, success in results.items():
            status = "[OK]" if success else "[FAIL]"
            print(f"  {strategy_name}: {status}")
        print("=" * 80)
        
        if not all_success:
            failed = [name for name, success in results.items() if not success]
            print(f"\n[WARN] Failed strategies: {', '.join(failed)}")
            return False
        
        return True
    
    except Exception as e:
        print(f"\n[ERROR] Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_output_files(strategy_name: str = None):
    """Show generated output files"""
    try:
        cfg = analyzer.get_config()
        
        if strategy_name:
            result_dirs = [analyzer.get_result_dir(strategy_name)]
        else:
            result_dirs = [analyzer.get_result_dir(s) for s in cfg.get_enabled_strategies()]
        
        print("\n[INFO] Generated output files:")
        for result_dir in result_dirs:
            if os.path.exists(result_dir):
                files = sorted(os.listdir(result_dir))
                print(f"\n  Directory: {result_dir}/")
                for f in files:
                    file_path = os.path.join(result_dir, f)
                    if os.path.isfile(file_path):
                        file_size = os.path.getsize(file_path)
                        print(f"     [OK] {f} ({file_size:,} bytes)")
            else:
                print(f"\n  [WARN] Directory does not exist: {result_dir}")
        
        print()
    
    except Exception as e:
        print(f"[WARN] Failed to display output files: {e}")


if __name__ == '__main__':
    main()
