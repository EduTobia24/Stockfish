#!/usr/bin/env python3
"""
Example: New Progress Tracking Structure
========================================

Shows the improved progress tracking organization and format.
"""

def show_new_structure():
    """Show the new file organization and CSV format."""
    
    print("🗂️  NEW PROGRESS TRACKING STRUCTURE")
    print("=" * 60)
    
    print("\n📁 File Organization:")
    print("""
    outputs/
    ├── 20251005_143022_A1B2C3/     # Timestamped run folder
    │   ├── training_progress.csv    # Simplified progress log
    │   ├── checkpoint_marker_*.txt  # Checkpoint markers
    │   ├── hall_of_fame.csv        # Final results
    │   └── training_config.json    # Configuration
    ├── 20251005_144510_D4E5F6/     # Another run
    │   ├── training_progress.csv
    │   └── ...
    └── 20251005_150045_G7H8I9/     # Yet another run
        └── ...
    """)
    
    print("\n📊 Simplified CSV Format:")
    print("elapsed_minutes,best_loss,best_complexity,timestamp")
    print("0.50,,,'2025-10-05T14:30:30.123456'")
    print("1.00,,,'2025-10-05T14:31:00.234567'")
    print("1.50,0.234567,8,'2025-10-05T14:31:30.345678'")
    print("2.00,0.198432,12,'2025-10-05T14:32:00.456789'")
    print("2.50,0.187654,15,'2025-10-05T14:32:30.567890'")
    
    print("\n⚙️  Configurable Options:")
    print("--progress-interval 15    # Snapshot every 15 seconds")
    print("--progress-interval 60    # Snapshot every 1 minute")
    print("--progress-interval 300   # Snapshot every 5 minutes")
    
    print("\n🎯 Benefits:")
    print("✅ Each run isolated in timestamped folder")
    print("✅ Simplified CSV with only essential data")
    print("✅ Configurable snapshot frequency")
    print("✅ Easy to compare multiple runs")
    print("✅ No data pollution between runs")

def show_usage_examples():
    """Show usage examples with new options."""
    
    print("\n🚀 USAGE EXAMPLES")
    print("=" * 60)
    
    print("\n💨 Fast snapshots (every 10 seconds):")
    print("""python scripts/position_sr_trainer.py \\
    dataset.npz --output outputs/ \\
    --progress-interval 10""")
    
    print("\n⏰ Standard snapshots (every 30 seconds - default):")
    print("""python scripts/position_sr_trainer.py \\
    dataset.npz --output outputs/""")
    
    print("\n🐌 Slow snapshots (every 5 minutes):")
    print("""python scripts/position_sr_trainer.py \\
    dataset.npz --output outputs/ \\
    --progress-interval 300""")
    
    print("\n📈 Expected output structure:")
    print("""outputs/
└── 20251005_150045_G7H8I9/
    ├── training_progress.csv    # Clean, focused data
    ├── checkpoint_marker_*.txt  # Proof of activity
    └── hall_of_fame.csv        # Final results""")

if __name__ == "__main__":
    show_new_structure()
    show_usage_examples()