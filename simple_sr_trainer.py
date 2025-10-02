#!/usr/bin/env python3
"""
Simple Chess SR Trainer
=======================

Simplified symbolic regression trainer for our initial 100-position dataset.
Focuses on discovering basic chess evaluation patterns.
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
    
    return features, evaluations

def select_meaningful_features(features, evaluations, n_select=30):
    """Select the most meaningful features for SR training."""
    print(f"\n🔍 Selecting top {n_select} features...")
    
    n_positions, n_features = features.shape
    
    # Calculate feature importance metrics
    feature_activity = np.mean(features, axis=0)  # How often each feature is active
    feature_variance = np.var(features, axis=0)   # Variance in feature values
    
    # Calculate correlation with evaluations (handle zero variance)
    correlations = np.zeros(n_features)
    for i in range(n_features):
        if feature_variance[i] > 1e-10:  # Only for features with some variance
            correlations[i] = abs(np.corrcoef(features[:, i], evaluations)[0, 1])
    
    # Combined importance score
    # Boost features that are active in 5-50% of positions (not too rare, not too common)
    activity_bonus = np.where((feature_activity > 0.05) & (feature_activity < 0.5), 2.0, 1.0)
    importance_scores = correlations * feature_variance * activity_bonus
    
    # Select top features
    top_indices = np.argsort(importance_scores)[-n_select:]
    selected_features = features[:, top_indices]
    
    # Generate feature names
    def get_feature_name(idx):
        if idx < 768:  # Piece features
            piece_idx = idx // 64
            square_idx = idx % 64
            pieces = ['WP', 'WR', 'WN', 'WB', 'WQ', 'WK', 'BP', 'BR', 'BN', 'BB', 'BQ', 'BK']
            square = chr(ord('a') + square_idx % 8) + str(square_idx // 8 + 1)
            return f"{pieces[piece_idx]}_{square}"
        else:  # Game state features
            game_features = ['white_to_move', 'castle_K', 'castle_Q', 'castle_k', 'castle_q',
                           'ep_a', 'ep_b', 'ep_c', 'ep_d', 'ep_e', 'ep_f', 'ep_g', 'ep_h', 'reserved']
            return game_features[idx - 768]
    
    feature_names = [get_feature_name(idx) for idx in top_indices]
    
    print(f"✅ Selected {len(selected_features[0])} features")
    print(f"🏆 Top 5 most important features:")
    for i in range(min(5, len(top_indices))):
        idx = top_indices[-(i+1)]
        score = importance_scores[idx]
        activity = feature_activity[idx]
        corr = correlations[idx]
        print(f"   {i+1}. {feature_names[-(i+1)]}: score={score:.4f}, activity={activity:.1%}, corr={corr:.3f}")
    
    return selected_features, feature_names

def train_simple_sr_model(features, evaluations, feature_names):
    """Train a simple symbolic regression model."""
    try:
        from pysr import PySRRegressor
    except ImportError:
        print("❌ PySR not available. Please install with: pip install pysr")
        return None
    
    print(f"\n🚀 Training Symbolic Regression Model")
    print(f"📊 Training data: {len(features)} positions × {len(features[0])} features")
    
    # Configure PySR for simple, interpretable results
    model = PySRRegressor(
        # Evolution parameters
        niterations=200,              # Fewer iterations for quick results
        populations=15,               # Moderate population size
        population_size=30,           # Smaller population for speed
        
        # Operators (keep simple for interpretability)
        binary_operators=["+", "-", "*"],  # No division to avoid instability
        unary_operators=["abs"],           # Just absolute value
        
        # Complexity control (keep formulas simple)
        max_complexity=10,            # Simple formulas
        maxsize=15,                  # Small expression trees
        parsimony=0.05,              # Strong preference for simplicity
        
        # Performance
        procs=2,                     # Use 2 cores
        multithreading=True,
        
        # Model selection
        model_selection="best",      # Choose best accuracy
        
        # Feature names
        feature_names_in=feature_names,
        
        # Output
        verbosity=1,
        progress=True,
        random_state=42
    )
    
    print(f"⚙️  PySR Configuration: {model.niterations} iterations, max complexity {model.max_complexity}")
    print(f"🔥 Starting training...")
    
    try:
        # Train the model
        model.fit(features, evaluations)
        
        print(f"\n✅ Training completed!")
        
        # Get best equation
        best_equation = model.get_best()
        training_score = model.score(features, evaluations)
        
        print(f"\n🏆 DISCOVERED CHESS EVALUATION FORMULA:")
        print("=" * 60)
        print(f"📐 Formula: {best_equation}")
        print(f"📊 Training R²: {training_score:.4f}")
        print(f"🧠 Complexity: {getattr(best_equation, 'complexity', 'N/A')}")
        print("=" * 60)
        
        # Test predictions
        predictions = model.predict(features)
        rmse = np.sqrt(np.mean((predictions - evaluations) ** 2))
        mae = np.mean(np.abs(predictions - evaluations))
        
        print(f"\n📈 Performance Metrics:")
        print(f"   RMSE: {rmse:.3f} pawns")
        print(f"   MAE:  {mae:.3f} pawns")
        print(f"   R²:   {training_score:.4f}")
        
        # Show some predictions vs actual
        print(f"\n🔍 Sample Predictions:")
        for i in range(min(8, len(evaluations))):
            error = predictions[i] - evaluations[i]
            print(f"   Pos {i+1}: Actual={evaluations[i]:+.2f}, Predicted={predictions[i]:+.2f}, Error={error:+.3f}")
        
        # Save results
        results = {
            'best_formula': str(best_equation),
            'training_r2': float(training_score),
            'rmse': float(rmse),
            'mae': float(mae),
            'feature_names': feature_names,
            'n_positions': len(features),
            'n_features': len(features[0])
        }
        
        with open('sr_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to sr_results.json")
        return model
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None

def main():
    print("🚀 Simple Chess Symbolic Regression Trainer")
    print("=" * 60)
    
    # Load data
    features, evaluations = load_and_analyze_data('bitboard_dataset_100.npz')
    
    # Select meaningful features
    selected_features, feature_names = select_meaningful_features(features, evaluations, n_select=25)
    
    # Train model
    model = train_simple_sr_model(selected_features, evaluations, feature_names)
    
    if model:
        print(f"\n🎉 SUCCESS! Discovered a symbolic formula for chess evaluation!")
        print(f"📖 The formula combines {len(feature_names)} chess features")
        print(f"🔗 Next step: Integrate this formula into Stockfish")
    else:
        print(f"\n❌ Training failed")

if __name__ == "__main__":
    main()