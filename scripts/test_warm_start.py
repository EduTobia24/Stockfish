#!/usr/bin/env python3
"""
Test Warm Start Example
======================

Example showing how to use the fixed warm start functionality.
"""

import sys
from pathlib import Path

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))

def test_warm_start():
    """Test the warm start functionality."""
    
    # Path to previous training results
    previous_run = Path("c:/Users/eduar/Stockfish/outputs/20251003_163637_AVaB19")
    dataset_path = Path("c:/Users/eduar/Stockfish/dataset_pos_features/evaluations_500_features.npz")
    
    if not previous_run.exists():
        print("❌ Previous run directory not found")
        return
        
    if not dataset_path.exists():
        print("❌ Dataset not found")
        return
    
    print("🧪 Testing Warm Start Configuration")
    print("=" * 50)
    print(f"📂 Previous run: {previous_run}")
    print(f"📊 Dataset: {dataset_path}")
    
    # Command to run warm start
    cmd = f'''python scripts/position_sr_trainer.py \\
    "{dataset_path}" \\
    --output "outputs/warm_start_test" \\
    --warm-start "{previous_run}" \\
    --complexity 30 \\
    --iterations 100 \\
    --timeout 0.1 \\
    --save-interval 0.1'''
    
    print(f"\n🚀 Recommended command:")
    print(cmd)
    
    print(f"\n💡 Expected behavior:")
    print(f"   ✅ Should load previous 66 equations")
    print(f"   ✅ Should continue training from there")
    print(f"   ✅ Progress tracking should work")
    print(f"   ✅ No PySR parameter errors")

if __name__ == "__main__":
    test_warm_start()