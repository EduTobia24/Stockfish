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
    
    # Example 7: Different loss functions comparison
    for loss_func in ['mse', 'max_error', 'percentile', 'huber', 'mae']:
        run_example(
            f"Loss Function: {loss_func.upper()}",
            f"{trainer_script} position_dataset.npz --output outputs/loss_{loss_func} "
            f"--loss {loss_func} --complexity 30 --iterations 2000 --timeout 1.0"
        )
    
    # Example 8: Performance optimization
    run_example(
        "Performance Optimized",
        f"{trainer_script} position_dataset.npz --output outputs/performance_optimized "
        f"--procs 16 --batch-size 200 --populations 30 --population-size 150"
    )
    
    # Show the complete workflow
    print(f"\n📋 COMPLETE WORKFLOW EXAMPLE:")
    print("=" * 80)
    
    workflow_commands = [
        "# 1. Generate position dataset",
        "python scripts/position_extractor.py positions.fen --output position_dataset.npz",
        "",
        "# 2. Test the trainer quickly",
        f"{trainer_script} position_dataset.npz --output outputs/test --complexity 15 --timeout 0.05",
        "",
        "# 3. Run full training with max_error loss",
        f"{trainer_script} position_dataset.npz --output outputs/full_training --loss max_error --config scripts/config_aggressive.json",
        "",
        "# 4. Continue training from previous results",
        f"{trainer_script} position_dataset.npz --output outputs/continued --warm-start outputs/full_training --iterations 5000",
        "",
        "# 5. Analyze results",
        "# Check outputs/continued/training_summary.md",
        "# Use outputs/continued/hall_of_fame.csv for integration",
    ]
    
    for cmd in workflow_commands:
        print(cmd)
    
    print(f"\n🎯 KEY FEATURES:")
    print("=" * 80)
    print("✅ Configurable loss functions: mse, max_error, percentile, huber, mae")
    print("✅ Robust periodic saving with backup rotation")
    print("✅ Proper warm starting from previous models")
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
    print("├── checkpoint_equations_*.csv # Periodic backups")
    print("├── latest_equations.csv      # Link to latest checkpoint")
    print("└── model files               # Complete PySR model")
    
    print(f"\n💡 TIPS:")
    print("=" * 80)
    print("🔧 Start with quick tests using small complexity and short timeout")
    print("📊 Use max_error loss for minimizing worst predictions")
    print("💾 Enable frequent saves for long training runs")
    print("🔥 Use warm start to continue interrupted training")
    print("⚙️ Create custom config files for repeated experiments")
    print("📈 Monitor outputs/training_name/latest_equations.csv for progress")
    
    return 0

if __name__ == "__main__":
    exit(main())