#!/usr/bin/env python3
"""
Example Usage Script for Position SR Trainer
===========================================

Demonstrates how to use the new position-based symbolic regression trainer
with various configurations and warm starting capabilities.
"""

import os
import sys
from pathlib import Path

def run_example(example_name: str, command: str):
    """Run an example with clear formatting."""
    print(f"\n{'='*60}")
    print(f"📘 EXAMPLE: {example_name}")
    print(f"{'='*60}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    # Don't actually run the command, just show it
    print(f"💡 To run this example, execute:")
    print(f"   {command}")
    print()

def main():
    """Show example usage scenarios."""
    print("🚀 Position-based Symbolic Regression Trainer Examples")
    print("=" * 80)
    
    # Check if we're in the right directory
    scripts_dir = Path("scripts")
    if not scripts_dir.exists():
        print("⚠️ Please run this from the Stockfish root directory")
        return 1
    
    trainer_script = "python scripts/position_sr_trainer.py"
    
    # Example 1: Basic training
    run_example(
        "Basic Training",
        f"{trainer_script} position_dataset.npz --output outputs/basic_training"
    )
    
    # Example 2: Quick test with small parameters
    run_example(
        "Quick Test (5 minutes)",
        f"{trainer_script} position_dataset.npz --output outputs/quick_test "
        f"--complexity 20 --iterations 500 --timeout 0.08 --populations 5"
    )
    
    # Example 3: High-accuracy training with max_error loss
    run_example(
        "High-Accuracy Training (Max Error Loss)",
        f"{trainer_script} position_dataset.npz --output outputs/max_error_training "
        f"--loss max_error --complexity 40 --iterations 5000 --timeout 2.0"
    )
    
    # Example 4: Using configuration file
    run_example(
        "Using Configuration File",
        f"{trainer_script} position_dataset.npz --output outputs/config_training "
        f"--config scripts/config_aggressive.json"
    )
    
    # Example 5: Warm start from previous training
    run_example(
        "Warm Start (Continue Previous Training)",
        f"{trainer_script} position_dataset.npz --output outputs/continued_training "
        f"--warm-start outputs/basic_training --iterations 3000"
    )
    
    # Example 6: Robust training with frequent saves
    run_example(
        "Robust Training (Frequent Saves)",
        f"{trainer_script} position_dataset.npz --output outputs/robust_training "
        f"--save-interval 2 --backup-count 10 --warm-start-mode strict"
    )
    
    # Example 7: Progress tracking and comparison
    run_example(
        "Progress Tracking Enabled",
        f"{trainer_script} position_dataset.npz --output outputs/tracked_training "
        f"--track-progress --progress-log tracked_progress.csv --iterations 2000"
    )
    
    run_example(
        "Multiple Runs for Comparison",
        f"# Run different loss functions with progress tracking\n"
        f"   {trainer_script} position_dataset.npz --output outputs/mse_run --loss mse --progress-log mse_progress.csv\n"
        f"   {trainer_script} position_dataset.npz --output outputs/max_error_run --loss max_error --progress-log max_error_progress.csv\n"
        f"   {trainer_script} position_dataset.npz --output outputs/huber_run --loss huber --progress-log huber_progress.csv"
    )
    
    # Example 8: Different loss functions comparison
    for loss_func in ['mse', 'max_error', 'percentile', 'huber', 'mae']:
        run_example(
            f"Loss Function: {loss_func.upper()}",
            f"{trainer_script} position_dataset.npz --output outputs/loss_{loss_func} "
            f"--loss {loss_func} --complexity 30 --iterations 2000 --timeout 1.0 "
            f"--progress-log {loss_func}_progress.csv"
        )
    
    # Example 9: Performance optimization
    run_example(
        "Performance Optimized",
        f"{trainer_script} position_dataset.npz --output outputs/performance_optimized "
        f"--procs 16 --batch-size 200 --populations 30 --population-size 150"
    )
    
    # Example 10: Analysis of training progress
    run_example(
        "Analyze Training Progress",
        "python scripts/analyze_progress.py outputs/*/training_progress.csv --output analysis_results --plots --report"
    )
    
    # Show the complete workflow
    print(f"\n📋 COMPLETE WORKFLOW EXAMPLE:")
    print("=" * 80)
    
    workflow_commands = [
        "# 1. Generate position dataset",
        "python scripts/position_extractor.py dataset_evaluations/evaluations_500.json",
        "",
        "# 2. Test the trainer quickly",
        f"{trainer_script} dataset_pos_features/evaluations_500_features.npz --output outputs/test --complexity 15 --timeout 0.05",
        "",
        "# 3. Run multiple training experiments with progress tracking",
        f"{trainer_script} dataset_pos_features/evaluations_500_features.npz --output outputs/mse_training --loss mse --progress-log mse_progress.csv",
        f"{trainer_script} dataset_pos_features/evaluations_500_features.npz --output outputs/max_error_training --loss max_error --progress-log max_error_progress.csv",
        "",
        "# 4. Continue training from previous results",
        f"{trainer_script} dataset_pos_features/evaluations_500_features.npz --output outputs/continued --warm-start outputs/mse_training --iterations 5000",
        "",
        "# 5. Analyze and compare results",
        "python scripts/analyze_progress.py outputs/*/training_progress.csv --output comparison_analysis --plots --report",
        "# Check comparison_analysis/comparison_report.md for detailed analysis",
    ]
    
    for cmd in workflow_commands:
        print(cmd)
    
    print(f"\n🎯 KEY FEATURES:")
    print("=" * 80)
    print("✅ Configurable loss functions: mse, max_error, percentile, huber, mae")
    print("✅ Robust periodic saving with backup rotation")
    print("✅ Proper warm starting from previous models")
    print("✅ Progress tracking with loss/iteration/complexity logging")
    print("✅ Comparison analysis across multiple training runs")
    print("✅ 73-feature position representation (vs 782 bitboard features)")
    print("✅ Clear progress tracking and result analysis")
    print("✅ JSON configuration files for reproducible experiments")
    print("✅ Comprehensive metadata and summary generation")
    
    print(f"\n📁 OUTPUT STRUCTURE:")
    print("=" * 80)
    print("outputs/training_name/")
    print("├── hall_of_fame.csv          # Final equations")
    print("├── training_config.json      # Configuration used")
    print("├── training_metadata.json    # Training metadata")
    print("├── training_summary.md       # Human-readable summary")
    print("├── training_progress.csv     # Progress log (loss/iteration/complexity)")
    print("├── checkpoint_equations_*.csv # Periodic backups")
    print("├── latest_equations.csv      # Link to latest checkpoint")
    print("└── model files               # Complete PySR model")
    
    print(f"\n📊 PROGRESS TRACKING:")
    print("=" * 80)
    print("The training_progress.csv file contains:")
    print("- run_name: Unique identifier for each training run")
    print("- timestamp: When each measurement was taken")
    print("- elapsed_minutes: Training time elapsed")
    print("- iteration: Progress tracking iteration number")
    print("- best_loss: Current best loss achieved")
    print("- best_complexity: Complexity of current best equation")
    print("- best_score: R² score of current best equation")
    print("- best_equation: Current best equation (truncated if long)")
    print("- total_equations: Total equations discovered so far")
    print("- loss_function: Loss function being used")
    print("- max_complexity_limit: Maximum complexity setting")
    
    print(f"\n💡 TIPS:")
    print("=" * 80)
    print("🔧 Start with quick tests using small complexity and short timeout")
    print("📊 Use max_error loss for minimizing worst predictions")
    print("💾 Enable frequent saves for long training runs")
    print("🔥 Use warm start to continue interrupted training")
    print("⚙️ Create custom config files for repeated experiments")
    print("📈 Monitor outputs/training_name/training_progress.csv for real-time progress")
    print("📊 Use analyze_progress.py to compare multiple training runs")
    print("🎯 Track progress logs to identify optimal stopping points")
    
    return 0

if __name__ == "__main__":
    exit(main())