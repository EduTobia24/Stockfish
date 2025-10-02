#!/usr/bin/env python3
"""
High-Complexity Symbolic Regression Trainer
==========================================

Trains symbolic regression with ALL 782 features and higher complexity limits
for larger datasets (500+ positions).
"""

import numpy as np
import json
import re
import os
from pathlib import Path

def clear_terminal():
    """Clear the terminal screen for cleaner output."""
    try:
        # Try multiple methods for better compatibility
        if os.name == 'nt':  # Windows
            os.system('cls')
        else:  # Unix/Linux/Mac
            os.system('clear')
        
        # Also try ANSI escape sequences as backup
        print('\033[2J\033[H', end='', flush=True)
    except:
        # Fallback: print newlines
        print('\n' * 50)

def analyze_discovered_features(equation_str, feature_names):
    """Analyze which chess features were discovered in the equation."""
    
    # Extract x_numbers from equation (e.g., x212, x24, x610, etc.)
    feature_indices = re.findall(r'x(\d+)', equation_str)
    feature_indices = [int(idx) for idx in set(feature_indices)]
    
    discovered_features = []
    for idx in feature_indices:
        if idx < len(feature_names):
            feature_name = feature_names[idx]
            discovered_features.append((feature_name, idx))
    
    return discovered_features

def decode_chess_features(discovered_features):
    """Decode the chess meaning of discovered features."""
    
    piece_features = []
    game_state_features = []
    
    for name, idx in discovered_features:
        if idx < 768:  # Piece bitboard features
            # Parse piece type and square
            if '_' in name:
                piece, square = name.split('_', 1)
                piece_features.append({
                    'name': name,
                    'index': idx,
                    'piece': piece,
                    'square': square,
                    'piece_type': piece[1] if len(piece) == 2 else piece,
                    'color': 'White' if piece[0] == 'W' else 'Black'
                })
        else:  # Game state features
            game_state_features.append({
                'name': name,
                'index': idx,
                'type': 'game_state'
            })
    
    return piece_features, game_state_features

def load_and_analyze_data(npz_path: str):
    """Load and analyze the bitboard features."""
    print(f"📊 Loading data from {npz_path}")
    data = np.load(npz_path)
    features = data['features']
    evaluations = data['evaluations']
    
    print(f"✅ Loaded {len(features)} positions with {features.shape[1]} features")
    print(f"📈 Evaluation range: {np.min(evaluations):+.2f} to {np.max(evaluations):+.2f}")
    print(f"✨ Feature sparsity: {1.0 - np.count_nonzero(features)/features.size:.1%}")
    
    return features, evaluations

def get_all_feature_names():
    """Generate names for all 782 bitboard features."""
    feature_names = []
    
    # Piece bitboard names (768 features: 0-767)
    piece_names = ['WP', 'WR', 'WN', 'WB', 'WQ', 'WK', 'BP', 'BR', 'BN', 'BB', 'BQ', 'BK']
    for piece_idx, piece in enumerate(piece_names):
        for square_idx in range(64):
            square = chr(ord('a') + square_idx % 8) + str(square_idx // 8 + 1)
            feature_names.append(f"{piece}_{square}")
    
    # Game state names (14 features: 768-781)
    game_state_names = [
        'white_to_move',
        'castle_K', 'castle_Q', 'castle_k', 'castle_q',
        'ep_a', 'ep_b', 'ep_c', 'ep_d', 'ep_e', 'ep_f', 'ep_g', 'ep_h',
        'reserved'
    ]
    feature_names.extend(game_state_names)
    
    return feature_names

def train_high_complexity_sr_model(features, evaluations, feature_names, max_complexity=100):
    """Train symbolic regression with high complexity for large datasets."""
    try:
        from pysr import PySRRegressor
    except ImportError:
        print("❌ PySR not available. Please install with: pip install pysr")
        return None
    
    n_positions = len(features)
    n_features = len(features[0])
    
    print(f"\n🚀 Training High-Complexity Symbolic Regression Model")
    print(f"📊 Training data: {n_positions} positions × {n_features} features")
    print(f"🎯 Using ALL {n_features} bitboard features (no selection)")
    print(f"🔥 Max complexity: {max_complexity} (high for large datasets)")
    
    # Clear before showing configuration
    clear_terminal()
    
    # Configure PySR for high complexity and large datasets
    model = PySRRegressor(
        # Evolution parameters (scaled for larger datasets)
        niterations=500,              # More iterations for complex search
        populations=30,               # More populations for exploration
        population_size=50,           # Larger population for diversity
        
        # Operators (expanded set for complex patterns)
        binary_operators=["+", "-", "*", "/"],  # Include division
        unary_operators=["abs", "sqrt", "square"],  # More unary functions
        
        # Complexity control (HIGH for large datasets)
        maxsize=max_complexity,       # High complexity limit
        parsimony=0.05,              # Lower parsimony for complex expressions
        
        # Performance (optimized for large datasets)
        procs=4,                     # Use multiple cores
        multithreading=True,
        batching=True,               # Essential for large datasets
        batch_size=100,              # Larger batches
        
        # Model selection
        model_selection="best",      # Choose best accuracy
        
        # Advanced optimization
        weight_optimize=0.01,        # More weight optimization
        weight_mutate_constant=0.1,
        weight_mutate_operator=0.15,
        
        # Output control
        verbosity=1,
        progress=True,
        
        # Random seed
        random_state=42,
        
        # Extended timeout for complex search
        timeout_in_seconds=3600,     # 1 hour max
    )
    
    print(f"⚙️  PySR High-Complexity Configuration:")
    print(f"   Iterations: {model.niterations}")
    print(f"   Populations: {model.populations}")
    print(f"   Max size: {model.maxsize}")
    print(f"   Operators: {model.binary_operators + model.unary_operators}")
    print(f"   Timeout: {model.timeout_in_seconds/60:.0f} minutes")
    print(f"   Batch size: {model.batch_size}")
    
    print(f"\n🔥 Starting high-complexity symbolic regression...")
    print(f"⏱️  This may take 30-60 minutes with {n_features} features and complexity {max_complexity}...")
    print("\n" + "="*80)
    print("🚀 TRAINING IN PROGRESS - PySR OUTPUT:")
    print("="*80)
    
    # Force clear and flush
    import sys
    sys.stdout.flush()
    
    try:
        # Train the model
        model.fit(features, evaluations)
        
        print(f"\n✅ Training completed!")
        
        # Strong clear before results
        import time
        time.sleep(1)  # Small delay
        clear_terminal()
        time.sleep(0.5)
        
        print("\n" + "🎉" * 20 + " TRAINING COMPLETE " + "🎉" * 20)
        print()
        
        # Get results
        best_equation = model.get_best()
        training_score = model.score(features, evaluations)
        
        print(f"\n🏆 DISCOVERED HIGH-COMPLEXITY CHESS EVALUATION FORMULA:")
        print("=" * 80)
        print(f"📐 Formula: {best_equation}")
        print(f"📊 Training R²: {training_score:.4f}")
        print(f"🧠 Size: {getattr(best_equation, 'complexity', getattr(best_equation, 'size', 'N/A'))}")
        print("=" * 80)
        
        # Performance metrics
        predictions = model.predict(features)
        rmse = np.sqrt(np.mean((predictions - evaluations) ** 2))
        mae = np.mean(np.abs(predictions - evaluations))
        
        print(f"\n📈 Performance Metrics:")
        print(f"   RMSE: {rmse:.3f} pawns")
        print(f"   MAE:  {mae:.3f} pawns")
        print(f"   R²:   {training_score:.4f}")
        
        # Analyze discovered features
        equation_str = str(best_equation)
        discovered_features = analyze_discovered_features(equation_str, feature_names)
        piece_features, game_state_features = decode_chess_features(discovered_features)
        
        print(f"\n🔍 Chess Features Discovered in Formula:")
        print(f"   Total features used: {len(discovered_features)}/{len(feature_names)}")
        
        if piece_features:
            print(f"\n♟️  Piece Features ({len(piece_features)}):")
            for pf in piece_features:
                activity = np.mean(features[:, pf['index']])
                print(f"   {pf['name']:12s}: {pf['color']:5s} {pf['piece_type']} on {pf['square']}, activity={activity:.1%}")
        
        if game_state_features:
            print(f"\n🎮 Game State Features ({len(game_state_features)}):")
            for gf in game_state_features:
                activity = np.mean(features[:, gf['index']])
                print(f"   {gf['name']:15s}: activity={activity:.1%}")
        
        # Sample predictions
        print(f"\n🔍 Sample Predictions:")
        for i in range(min(15, len(evaluations))):
            error = predictions[i] - evaluations[i]
            accuracy = "✅" if abs(error) < 0.5 else "⚠️" if abs(error) < 1.0 else "❌"
            print(f"   {accuracy} Pos {i+1:2d}: Actual={evaluations[i]:+.2f}, Predicted={predictions[i]:+.2f}, Error={error:+.3f}")
        
        # Save detailed results
        results = {
            'best_formula': str(best_equation),
            'training_r2': float(training_score),
            'rmse': float(rmse),
            'mae': float(mae),
            'total_features': int(len(feature_names)),
            'features_used': int(len(discovered_features)),
            'used_feature_names': [name for name, _ in discovered_features],
            'piece_features': [{
                'name': pf['name'],
                'piece': pf['piece'],
                'square': pf['square'],
                'color': pf['color'],
                'activity': float(np.mean(features[:, pf['index']]))
            } for pf in piece_features],
            'game_state_features': [{
                'name': gf['name'],
                'activity': float(np.mean(features[:, gf['index']]))
            } for gf in game_state_features],
            'n_positions': int(len(features)),
            'feature_usage_efficiency': float(len(discovered_features) / len(feature_names)),
            'model_complexity': int(getattr(best_equation, 'complexity', getattr(best_equation, 'size', 0))),
            'max_complexity_used': int(max_complexity)
        }
        
        with open('high_complexity_sr_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Detailed results saved to high_complexity_sr_results.json")
        
        # Chess insights
        efficiency = len(discovered_features) / len(feature_names) * 100
        print(f"\n🎯 Chess AI Insights:")
        print(f"   Feature efficiency: {efficiency:.2f}%")
        print(f"   Formula complexity: {getattr(best_equation, 'complexity', getattr(best_equation, 'size', 'N/A'))}")
        print(f"   Pieces involved: {len(piece_features)}")
        print(f"   Game states used: {len(game_state_features)}")
        
        if len(piece_features) > 0:
            colors = [pf['color'] for pf in piece_features]
            white_pieces = colors.count('White')
            black_pieces = colors.count('Black')
            print(f"   Color balance: {white_pieces} White, {black_pieces} Black features")
        
        return model
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main training pipeline with high complexity."""
    import argparse
    
    # Clear terminal for clean start
    clear_terminal()
    
    parser = argparse.ArgumentParser(description='Train high-complexity SR with ALL features')
    parser.add_argument('--dataset', default='bitboard_dataset_500.npz',
                       help='Path to NPZ features file')
    parser.add_argument('--complexity', type=int, default=100,
                       help='Maximum complexity (default: 100)')
    parser.add_argument('--timeout', type=int, default=3600,
                       help='Training timeout in seconds (default: 1 hour)')
    
    args = parser.parse_args()
    
    # Clear terminal early
    clear_terminal()
    
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print(f"💡 Available datasets:")
        for npz_file in Path('.').glob('*.npz'):
            print(f"   {npz_file}")
        return 1
    
    print("🚀 High-Complexity Symbolic Regression Trainer")
    print("=" * 70)
    print(f"📂 Dataset: {dataset_path}")
    print(f"🔢 Using ALL features (no selection)")
    print(f"🔥 Max complexity: {args.complexity}")
    print(f"⏱️  Timeout: {args.timeout/60:.0f} minutes")
    print("=" * 70)
    
    # Load data
    features, evaluations = load_and_analyze_data(str(dataset_path))
    
    # Generate feature names
    feature_names = get_all_feature_names()
    print(f"✅ Generated {len(feature_names)} feature names")
    
    # Clear terminal before training starts
    print("\n" + "🔄 Preparing for training..." + "\n")
    clear_terminal()
    
    print("🚀 High-Complexity Symbolic Regression Trainer")
    print("=" * 70)
    print(f"📂 Dataset: {dataset_path}")
    print(f"🔢 Using ALL {len(feature_names)} features (no selection)")
    print(f"🔥 Max complexity: {args.complexity}")
    print(f"⏱️  Timeout: {args.timeout/60:.0f} minutes")
    print(f"📊 Loaded {len(features)} positions")
    print("=" * 70)
    
    # Train model with high complexity
    model = train_high_complexity_sr_model(features, evaluations, feature_names, args.complexity)
    
    if model:
        print(f"\n🎉 SUCCESS!")
        print(f"✅ High-complexity symbolic regression completed")
        print(f"🧬 Discovered complex chess evaluation formula")
        print(f"📈 Ready for advanced chess engine integration!")
        return 0
    else:
        print(f"\n❌ Training failed")
        return 1

if __name__ == "__main__":
    exit(main())