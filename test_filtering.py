#!/usr/bin/env python3
"""
Test Evaluation Filtering
=========================

Test the extreme evaluation filtering functionality.
"""

import numpy as np
import json
import os
from bitboard_extractor import ChessBitboardExtractor

def create_test_dataset():
    """Create a test dataset with some extreme evaluations."""
    dataset = []
    
    # Normal positions
    dataset.extend([
        {"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "evaluation": 0.2},
        {"fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "evaluation": 0.3},
        {"fen": "rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2", "evaluation": 0.1},
        {"fen": "rnbqkb1r/pppp1ppp/4pn2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3", "evaluation": -0.1},
        {"fen": "rnbqkb1r/pppp1ppp/4pn2/8/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 3", "evaluation": 0.2},
    ])
    
    # Extreme positive evaluations (should be filtered)
    dataset.extend([
        {"fen": "rnbqkb1r/pppp1ppp/4pn2/8/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 3", "evaluation": 25.0},
        {"fen": "rnbqkb1r/pppp1ppp/4pn2/8/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 1 3", "evaluation": 30.5},
        {"fen": "rnbqkb1r/pppp1ppp/4pn2/8/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 3", "evaluation": 22.3},
    ])
    
    # Extreme negative evaluations (should be filtered)
    dataset.extend([
        {"fen": "rnbqkb1r/pppp1ppp/4pn2/8/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 1 3", "evaluation": -25.7},
        {"fen": "rnbqkb1r/pppp1ppp/4pn2/8/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 3", "evaluation": -21.2},
        {"fen": "rnbqkb1r/pppp1ppp/4pn2/8/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 1 3", "evaluation": -35.0},
    ])
    
    # Edge cases (exactly at threshold)
    dataset.extend([
        {"fen": "rnbqkb1r/pppp1ppp/4pn2/8/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 3", "evaluation": 20.0},
        {"fen": "rnbqkb1r/pppp1ppp/4pn2/8/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 1 3", "evaluation": -20.0},
    ])
    
    return dataset

def test_filtering():
    """Test the evaluation filtering functionality."""
    
    print("🧪 Testing Evaluation Filtering")
    print("="*50)
    
    # Create test dataset
    test_dataset = create_test_dataset()
    
    print(f"📊 Created test dataset with {len(test_dataset)} positions")
    
    # Count evaluations by range
    evals = [pos['evaluation'] for pos in test_dataset]
    normal_count = sum(1 for e in evals if -20 <= e <= 20)
    extreme_positive = sum(1 for e in evals if e > 20)
    extreme_negative = sum(1 for e in evals if e < -20)
    
    print(f"   Normal evaluations (±20): {normal_count}")
    print(f"   Extreme positive (>20): {extreme_positive}")
    print(f"   Extreme negative (<-20): {extreme_negative}")
    
    # Save test dataset
    test_file = "test_dataset_with_extremes.json"
    with open(test_file, 'w') as f:
        json.dump(test_dataset, f, indent=2)
    
    print(f"💾 Saved test dataset to: {test_file}")
    
    # Test without filtering
    print(f"\n🔄 Testing WITHOUT filtering...")
    extractor = ChessBitboardExtractor()
    features1, evaluations1 = extractor.extract_dataset_features(test_file, filter_extreme=False)
    
    print(f"✅ Without filtering: {len(features1)} positions")
    print(f"   Evaluation range: {np.min(evaluations1):.1f} to {np.max(evaluations1):.1f}")
    
    # Test with filtering
    print(f"\n🔍 Testing WITH filtering (±20)...")
    features2, evaluations2 = extractor.extract_dataset_features(test_file, filter_extreme=True, eval_threshold=20.0)
    
    print(f"✅ With filtering: {len(features2)} positions")
    print(f"   Evaluation range: {np.min(evaluations2):.1f} to {np.max(evaluations2):.1f}")
    
    # Verify filtering worked correctly
    expected_filtered = normal_count
    actual_filtered = len(features2)
    
    print(f"\n📊 FILTERING RESULTS:")
    print(f"   Expected positions after filtering: {expected_filtered}")
    print(f"   Actual positions after filtering: {actual_filtered}")
    
    if actual_filtered == expected_filtered:
        print(f"✅ FILTERING WORKING CORRECTLY!")
    else:
        print(f"❌ FILTERING NOT WORKING AS EXPECTED")
    
    # Check no extreme evaluations remain
    has_extremes = any(abs(e) > 20 for e in evaluations2)
    if not has_extremes:
        print(f"✅ No extreme evaluations remain in filtered dataset")
    else:
        print(f"❌ Some extreme evaluations still present!")
    
    # Test with different threshold
    print(f"\n🔍 Testing with custom threshold (±15)...")
    features3, evaluations3 = extractor.extract_dataset_features(test_file, filter_extreme=True, eval_threshold=15.0)
    
    print(f"✅ With threshold ±15: {len(features3)} positions")
    print(f"   Evaluation range: {np.min(evaluations3):.1f} to {np.max(evaluations3):.1f}")
    
    # Cleanup
    os.remove(test_file)
    print(f"\n🧹 Cleaned up test files")
    
    print(f"\n🎉 Filtering test complete!")

if __name__ == "__main__":
    test_filtering()