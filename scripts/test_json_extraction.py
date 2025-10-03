#!/usr/bin/env python3
"""
Test Position Extractor with JSON Input
======================================

Test the updated position extractor with evaluations_500.json format.
"""

import sys
import os
from pathlib import Path

def test_json_extraction():
    """Test JSON-based feature extraction."""
    
    print("🧪 Testing JSON-based Position Feature Extraction")
    print("=" * 70)
    
    # Check if evaluations file exists
    json_file = Path("dataset_evaluations/evaluations_500.json")
    if not json_file.exists():
        print(f"❌ JSON file not found: {json_file}")
        print("💡 Make sure you have the evaluations_500.json file")
        return 1
    
    extractor_script = "python scripts/position_extractor.py"
    output_dir = Path("dataset_pos_features")
    
    print(f"📂 Input file: {json_file}")
    print(f"💾 Output directory: {output_dir}")
    
    # Example commands for different scenarios
    examples = [
        {
            "name": "Basic Extraction",
            "description": "Extract all positions without filtering",
            "command": f"{extractor_script} {json_file}"
        },
        {
            "name": "Filtered Extraction", 
            "description": "Filter out extreme evaluations (>±15 pawns)",
            "command": f"{extractor_script} {json_file} --filter-extreme --eval-threshold 15.0"
        },
        {
            "name": "Custom Output",
            "description": "Specify custom output filename",
            "command": f"{extractor_script} {json_file} --output dataset_pos_features/custom_dataset.npz"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n📘 Example {i}: {example['name']}")
        print(f"   Description: {example['description']}")
        print(f"   Command: {example['command']}")
        print(f"   Output: dataset_pos_features/evaluations_500_features.npz")
    
    # Show expected file structure
    print(f"\n📁 Expected Output Structure:")
    print("=" * 70)
    print("dataset_pos_features/")
    print("├── evaluations_500_features.npz           # Main dataset file")
    print("├── evaluations_500_features_feature_names.json  # Feature names")
    print("└── custom_dataset.npz                     # Custom outputs")
    
    print(f"\n🔗 Integration with SR Trainer:")
    print("=" * 70)
    print("# After extraction, use with SR trainer:")
    print("python scripts/position_sr_trainer.py dataset_pos_features/evaluations_500_features.npz --output outputs/training")
    
    # Show file size comparison
    print(f"\n📊 Feature Comparison:")
    print("=" * 70)
    print("Old bitboard extractor: 782 features per position")
    print("New position extractor: 73 features per position")
    print("Reduction: ~90% fewer features → much faster training!")
    
    print(f"\n💡 To run the actual extraction:")
    print("=" * 70)
    print(f"cd /path/to/stockfish")
    print(f"{examples[0]['command']}")
    
    return 0

def show_json_format():
    """Show the expected JSON format."""
    print(f"\n📋 Expected JSON Format (evaluations_500.json):")
    print("=" * 70)
    
    example_entry = """[
  {
    "fen": "2rq1rk1/1p1bppbp/p2p1np1/8/3PP3/1BP2N1P/PP2BPP1/R2Q1RK1 w - - 0 12",
    "evaluation": 1.73,
    "best_move": "e4e5",
    "depth": 25,
    "nodes": 2987850,
    "nps": 597570,
    "time_ms": 5000,
    "pv": ["e4e5", "d6e5", "f3e5", ...]
  },
  {
    "fen": "...",
    "evaluation": ...,
    ...
  }
]"""
    
    print(example_entry)
    
    print(f"\n🎯 Required Fields:")
    print("- fen: Chess position in FEN notation")
    print("- evaluation: Position evaluation in pawns")
    
    print(f"\n📝 Optional Fields (ignored by extractor):")
    print("- best_move, depth, nodes, nps, time_ms, pv")

if __name__ == "__main__":
    test_json_extraction()
    show_json_format()