#!/usr/bin/env python3
"""
Full Feature Symbolic Regression Trainer
========================================

Trains symbolic regression using ALL 782 bitboard features.
No feature selection - let PySR discover the most important patterns itself.
"""

import numpy as np
import json
from pathlib import Path

def load_and_analyze_data(npz_path: str):
    """Load and analyze the bitboard features."""
    print(f"📊 Loading data from {npz_path}")
    data = np.load(npz_path)
    features = data['features']
    evaluations = data['evaluations']
    
    print(f"✅ Loaded {len(features)} positions with {features.shape[1]} features")
    print(f"📈 Evaluation range: {np.min(evaluations):+.2f} to {np.max(evaluations):+.2f}")
    print(f"✨ Feature sparsity: {1.0 - np.count_nonzero(features)/features.size:.1%}")
    
    # Quick feature analysis
    feature_activity = np.mean(features, axis=0)
    active_features = np.sum(feature_activity > 0.01)  # Active in >1% of positions
    
    print(f"🔍 Active features (>1%): {active_features}/{features.shape[1]}")
    print(f"📊 Most active feature: {np.max(feature_activity):.1%}")
    
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

def train_full_feature_sr_model(features, evaluations, feature_names):
    """Train symbolic regression using ALL features."""
    try:
        from pysr import PySRRegressor
    except ImportError:
        print("❌ PySR not available. Please install with: pip install pysr")
        return None
    
    print(f"\n🚀 Training Full-Feature Symbolic Regression Model")
    print(f"📊 Training data: {len(features)} positions × {len(features[0])} features")
    print(f"🎯 Using ALL {len(features[0])} bitboard features (no selection)")
    
    # Configure PySR for handling many features
    model = PySRRegressor(
        # Evolution parameters (more conservative with many features)
        niterations=300,              # More iterations for complex search
        populations=25,               # More populations for exploration
        population_size=40,           # Moderate population size
        
        # Operators (keep simple for interpretability)
        binary_operators=["+", "-", "*"],  # Core operators
        unary_operators=["abs", "sqrt"],   # Safe unary functions
        
        # Complexity control (CRITICAL with many features)
        maxsize=20,                  # Limit expression tree size
        parsimony=0.1,               # Strong simplicity preference
        
        # Feature selection (let PySR choose)
        select_k_features=None,      # Don't pre-select features
        
        # Performance
        procs=4,                     # Use multiple cores
        multithreading=True,
        batching=True,               # Enable batching for efficiency
        batch_size=50,               # Process in batches
        
        # Model selection
        model_selection="best",      # Choose best accuracy
        
        # Regularization (important with many features)
        weight_optimize=0.001,       # Slight weight optimization
        weight_mutate_constant=0.076,
        weight_mutate_operator=0.1,
        
        # Output control
        verbosity=1,
        progress=True,
        
        # Random seed
        random_state=42,
        
        # Timeout (don't run forever)
        timeout_in_seconds=1800,    # 30 minutes max
    )
    
    print(f"⚙️  PySR Configuration:")
    print(f"   Iterations: {model.niterations}")
    print(f"   Populations: {model.populations}")
    print(f"   Max size: {model.maxsize}")
    print(f"   Operators: {model.binary_operators + model.unary_operators}")
    print(f"   Timeout: {model.timeout_in_seconds/60:.0f} minutes")
    
    print(f"\n🔥 Starting full-feature symbolic regression training...")
    print(f"⏱️  This may take 15-30 minutes with {len(features[0])} features...")
    
    try:
        # Train the model
        model.fit(features, evaluations)
        
        print(f"\n✅ Training completed!")
        
        # Get results
        best_equation = model.get_best()
        training_score = model.score(features, evaluations)
        
        print(f"\n🏆 DISCOVERED CHESS EVALUATION FORMULA:")
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
        
        # Analyze which features were actually used
        equation_str = str(best_equation)
        used_features = []
        for i, name in enumerate(feature_names):
            if name in equation_str:
                used_features.append((name, i))
        
        print(f"\n🔍 Features Used in Final Formula:")
        print(f"   Total features used: {len(used_features)}/{len(feature_names)}")
        if len(used_features) <= 20:  # Show all if not too many
            for name, idx in used_features:
                activity = np.mean(features[:, idx])
                print(f"   {name}: activity={activity:.1%}")
        else:
            print(f"   Top 10 most active used features:")
            used_activities = [(name, idx, np.mean(features[:, idx])) for name, idx in used_features]
            used_activities.sort(key=lambda x: x[2], reverse=True)
            for name, idx, activity in used_activities[:10]:
                print(f"   {name}: activity={activity:.1%}")
        
        # Sample predictions
        print(f"\n🔍 Sample Predictions:")
        for i in range(min(10, len(evaluations))):
            error = predictions[i] - evaluations[i]
            print(f"   Pos {i+1}: Actual={evaluations[i]:+.2f}, Predicted={predictions[i]:+.2f}, Error={error:+.3f}")
        
        # Save detailed results
        results = {
            'best_formula': str(best_equation),
            'training_r2': float(training_score),
            'rmse': float(rmse),
            'mae': float(mae),
            'total_features': len(feature_names),
            'features_used': len(used_features),
            'used_feature_names': [name for name, _ in used_features],
            'n_positions': len(features),
            'feature_usage_efficiency': len(used_features) / len(feature_names),
            'model_complexity': getattr(best_equation, 'complexity', getattr(best_equation, 'size', None))
        }
        
        with open('full_feature_sr_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Detailed results saved to full_feature_sr_results.json")
        
        # Show feature efficiency
        efficiency = len(used_features) / len(feature_names) * 100
        print(f"\n📊 Feature Usage Efficiency: {efficiency:.1f}%")
        print(f"   PySR selected {len(used_features)} out of {len(feature_names)} available features")
        
        return model
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_feature_importance(features, evaluations, feature_names):
    """Quick analysis of feature importance before training."""
    print(f"\n🔍 Pre-Training Feature Analysis")
    print("=" * 60)
    
    n_positions, n_features = features.shape
    
    # Calculate basic statistics
    feature_activity = np.mean(features, axis=0)
    feature_variance = np.var(features, axis=0)
    
    # Calculate correlations (for non-zero variance features)
    correlations = np.zeros(n_features)
    for i in range(n_features):
        if feature_variance[i] > 1e-10:
            correlations[i] = abs(np.corrcoef(features[:, i], evaluations)[0, 1])
    
    # Find most promising features
    active_mask = feature_activity > 0.01  # Active in >1% of positions
    high_corr_mask = correlations > 0.1    # Correlation > 0.1
    promising_mask = active_mask & high_corr_mask
    
    promising_count = np.sum(promising_mask)
    promising_indices = np.where(promising_mask)[0]
    
    print(f"📊 Feature Statistics:")
    print(f"   Total features: {n_features}")
    print(f"   Active features (>1%): {np.sum(active_mask)}")
    print(f"   High correlation (>0.1): {np.sum(high_corr_mask)}")
    print(f"   Promising features: {promising_count}")
    
    if promising_count > 0:
        print(f"\n🌟 Most Promising Features:")
        # Sort by correlation × activity
        scores = correlations[promising_indices] * feature_activity[promising_indices]
        sorted_idx = np.argsort(scores)[-10:]  # Top 10
        
        for i, idx in enumerate(sorted_idx[::-1]):
            real_idx = promising_indices[idx]
            name = feature_names[real_idx]
            activity = feature_activity[real_idx]
            corr = correlations[real_idx]
            print(f"   {i+1:2d}. {name:12s}: activity={activity:.1%}, corr={corr:.3f}")
    
    print("=" * 60)

def main():
    """Main training pipeline with all features."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train SR with ALL bitboard features')
    parser.add_argument('--dataset', default='bitboard_dataset_300.npz',
                       help='Path to NPZ features file')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze features, don\'t train')
    parser.add_argument('--timeout', type=int, default=1800,
                       help='Training timeout in seconds (default: 30 min)')
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print(f"💡 Available datasets:")
        for npz_file in Path('.').glob('*.npz'):
            print(f"   {npz_file}")
        return 1
    
    print("🚀 Full-Feature Symbolic Regression Trainer")
    print("=" * 60)
    print(f"📂 Dataset: {dataset_path}")
    print(f"🔢 Using ALL features (no selection)")
    print(f"⏱️  Timeout: {args.timeout/60:.0f} minutes")
    print("=" * 60)
    
    # Load data
    features, evaluations = load_and_analyze_data(str(dataset_path))
    
    # Generate feature names
    feature_names = get_all_feature_names()
    print(f"✅ Generated {len(feature_names)} feature names")
    
    # Pre-training analysis
    analyze_feature_importance(features, evaluations, feature_names)
    
    if args.analyze_only:
        print("✅ Analysis complete (--analyze-only mode)")
        return 0
    
    # Train model with all features
    model = train_full_feature_sr_model(features, evaluations, feature_names)
    
    if model:
        print(f"\n🎉 SUCCESS!")
        print(f"✅ Full-feature symbolic regression completed")
        print(f"🧬 PySR automatically selected the most important features")
        print(f"📈 Formula discovered using real chess patterns")
        print(f"🔗 Ready for integration into Stockfish!")
        return 0
    else:
        print(f"\n❌ Training failed")
        return 1

if __name__ == "__main__":
    exit(main())