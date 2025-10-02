#!/usr/bin/env python3
"""
Quick Full Feature Test
======================

Fast test of symbolic regression with all features to verify it works.
"""

import numpy as np
import json
from pathlib import Path

def quick_full_feature_test():
    """Quick test with minimal iterations."""
    
    # Load data
    print("📊 Loading bitboard dataset...")
    data = np.load('bitboard_dataset_300.npz')
    features = data['features']
    evaluations = data['evaluations']
    
    print(f"✅ Loaded {len(features)} positions with {features.shape[1]} features")
    
    # Generate feature names
    feature_names = []
    
    # Piece bitboard names (768 features)
    piece_names = ['WP', 'WR', 'WN', 'WB', 'WQ', 'WK', 'BP', 'BR', 'BN', 'BB', 'BQ', 'BK']
    for piece_idx, piece in enumerate(piece_names):
        for square_idx in range(64):
            square = chr(ord('a') + square_idx % 8) + str(square_idx // 8 + 1)
            feature_names.append(f"{piece}_{square}")
    
    # Game state names (14 features)
    game_state_names = [
        'white_to_move', 'castle_K', 'castle_Q', 'castle_k', 'castle_q',
        'ep_a', 'ep_b', 'ep_c', 'ep_d', 'ep_e', 'ep_f', 'ep_g', 'ep_h', 'reserved'
    ]
    feature_names.extend(game_state_names)
    
    print(f"✅ Generated {len(feature_names)} feature names")
    
    # Quick feature analysis
    feature_activity = np.mean(features, axis=0)
    active_features = np.sum(feature_activity > 0.01)
    
    print(f"🔍 Active features (>1%): {active_features}/{features.shape[1]}")
    
    # Import PySR
    try:
        from pysr import PySRRegressor
        print("✅ PySR imported successfully")
    except ImportError:
        print("❌ PySR not available")
        return False
    
    # Create simple model for quick test
    print(f"\n🚀 Starting QUICK test with ALL {len(feature_names)} features...")
    
    model = PySRRegressor(
        niterations=50,              # Just 50 iterations for quick test
        populations=10,              # Fewer populations
        population_size=20,          # Smaller population
        binary_operators=["+", "-", "*"],
        unary_operators=["abs"],
        maxsize=15,                  # Reasonable size limit
        parsimony=0.1,               # Prefer simple expressions
        verbosity=1,
        progress=True,
        random_state=42,
        timeout_in_seconds=300,      # 5 minutes max
    )
    
    print("⏱️  Training for 5 minutes max...")
    
    try:
        # Train
        model.fit(features, evaluations)
        
        # Get best result
        best_eq = model.get_best()
        score = model.score(features, evaluations)
        
        print(f"\n🎉 SUCCESS!")
        print(f"📐 Best formula: {best_eq}")
        print(f"📊 R² score: {score:.4f}")
        
        # Check which features were used
        equation_str = str(best_eq)
        used_features = [name for name in feature_names if name in equation_str]
        
        print(f"🔍 Features used: {len(used_features)}")
        for feature in used_features[:10]:  # Show first 10
            print(f"   {feature}")
        if len(used_features) > 10:
            print(f"   ... and {len(used_features) - 10} more")
        
        # Save quick result
        result = {
            'formula': str(best_eq),
            'r2_score': float(score),
            'features_used': used_features,
            'total_features': len(feature_names),
            'test_success': True
        }
        
        with open('quick_test_result.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n✅ Quick test successful!")
        print(f"💾 Results saved to quick_test_result.json")
        print(f"🚀 Ready for full training with longer iterations!")
        
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_full_feature_test()
    if success:
        print(f"\n🎯 Next: Run full training with more iterations!")
        print(f"   python3 full_feature_sr_trainer.py --dataset bitboard_dataset_300.npz")
    else:
        print(f"\n🔧 Fix issues before full training")