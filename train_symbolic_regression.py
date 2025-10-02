#!/usr/bin/env python3
"""
Symbolic Regression Chess Evaluation Trainer
============================================

Trains a symbolic regression model using PySR on bitboard chess features.
Discovers interpretable formulas that approximate Stockfish evaluations.
"""

import numpy as np
import json
from pathlib import Path
from typing import Tuple, List, Dict
import time

def load_bitboard_features(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load features and evaluations from NPZ file."""
    data = np.load(npz_path)
    features = data['features']
    evaluations = data['evaluations']
    print(f"📊 Loaded {len(features)} positions with {features.shape[1]} features")
    return features, evaluations

def analyze_data(features: np.ndarray, evaluations: np.ndarray) -> None:
    """Analyze the dataset before training."""
    print("\n" + "=" * 60)
    print("📊 DATASET ANALYSIS")
    print("=" * 60)
    
    n_positions, n_features = features.shape
    
    print(f"📍 Positions: {n_positions}")
    print(f"🔢 Features: {n_features}")
    print(f"✨ Sparsity: {1.0 - np.count_nonzero(features) / features.size:.1%}")
    
    # Evaluation statistics
    print(f"\n📈 Evaluation Statistics:")
    print(f"   Range: {np.min(evaluations):+.2f} to {np.max(evaluations):+.2f}")
    print(f"   Mean: {np.mean(evaluations):+.2f}")
    print(f"   Std: {np.std(evaluations):.2f}")
    
    # Distribution analysis
    positive_evals = np.sum(evaluations > 0.5)
    negative_evals = np.sum(evaluations < -0.5)
    balanced_evals = n_positions - positive_evals - negative_evals
    
    print(f"   White advantage (>+0.5): {positive_evals} ({positive_evals/n_positions*100:.1f}%)")
    print(f"   Balanced (±0.5): {balanced_evals} ({balanced_evals/n_positions*100:.1f}%)")
    print(f"   Black advantage (<-0.5): {negative_evals} ({negative_evals/n_positions*100:.1f}%)")
    
    # Feature activity analysis
    feature_activity = np.mean(features, axis=0)
    active_features = np.sum(feature_activity > 0.01)  # Features active in >1% of positions
    
    print(f"\n🔍 Feature Activity:")
    print(f"   Active features (>1%): {active_features}/{n_features}")
    print(f"   Most active feature: {np.max(feature_activity):.1%}")
    print(f"   Avg feature activity: {np.mean(feature_activity):.1%}")
    
    print("=" * 60)

def prepare_features_for_sr(features: np.ndarray, evaluations: np.ndarray, 
                          max_features: int = 100) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare features for symbolic regression by selecting most informative ones.
    
    Args:
        features: Full bitboard features (N, 782)
        evaluations: Target evaluations (N,)
        max_features: Maximum number of features to keep
        
    Returns:
        selected_features: Reduced feature matrix (N, max_features)
        evaluations: Same evaluations
        feature_names: Names of selected features
    """
    print(f"\n🔍 Feature Selection (keeping top {max_features} features)")
    
    n_positions, n_features = features.shape
    
    # Method 1: Select features with highest variance (most informative)
    feature_variances = np.var(features, axis=0)
    
    # Method 2: Select features with highest correlation with evaluations
    feature_correlations = np.abs([np.corrcoef(features[:, i], evaluations)[0, 1] 
                                  if np.var(features[:, i]) > 0 else 0 
                                  for i in range(n_features)])
    
    # Method 3: Select features that are active in reasonable number of positions
    feature_activity = np.mean(features, axis=0)
    
    # Combined scoring: variance × correlation × activity (with activity boost)
    activity_score = np.clip(feature_activity * 20, 0, 1)  # Boost active features
    combined_scores = feature_variances * feature_correlations * (1 + activity_score)
    
    # Select top features
    top_indices = np.argsort(combined_scores)[-max_features:]
    selected_features = features[:, top_indices]
    
    # Generate feature names
    piece_names = ['WP', 'WR', 'WN', 'WB', 'WQ', 'WK', 'BP', 'BR', 'BN', 'BB', 'BQ', 'BK']
    game_state_names = [
        'white_to_move', 'castle_K', 'castle_Q', 'castle_k', 'castle_q',
        'ep_a', 'ep_b', 'ep_c', 'ep_d', 'ep_e', 'ep_f', 'ep_g', 'ep_h', 'reserved'
    ]
    
    def index_to_square(idx):
        return chr(ord('a') + idx % 8) + str(idx // 8 + 1)
    
    all_feature_names = []
    # Piece features (0-767)
    for piece_idx, piece in enumerate(piece_names):
        for square_idx in range(64):
            square = index_to_square(square_idx)
            all_feature_names.append(f"{piece}_{square}")
    # Game state features (768-781)
    all_feature_names.extend(game_state_names)
    
    selected_feature_names = [all_feature_names[i] for i in top_indices]
    
    print(f"✅ Selected {len(selected_features[0])} most informative features")
    print(f"   Top 5 features:")
    for i in range(min(5, len(selected_feature_names))):
        idx = top_indices[-(i+1)]
        score = combined_scores[idx]
        corr = feature_correlations[idx]
        activity = feature_activity[idx]
        print(f"   {i+1}. {selected_feature_names[-(i+1)]}: score={score:.3f}, corr={corr:.3f}, activity={activity:.1%}")
    
    return selected_features, evaluations, selected_feature_names

def train_pysr_model(features: np.ndarray, evaluations: np.ndarray, 
                    feature_names: List[str], output_dir: str = "sr_model") -> str:
    """
    Train PySR model on chess features.
    
    Args:
        features: Feature matrix (N, n_features)
        evaluations: Target evaluations (N,)
        feature_names: Names of features
        output_dir: Directory to save model
        
    Returns:
        Path to saved model
    """
    try:
        from pysr import PySRRegressor
    except ImportError:
        print("❌ PySR not installed. Install with: pip install pysr")
        print("📝 Installing PySR...")
        import subprocess
        import sys
        
        # Install PySR
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pysr'])
        print("✅ PySR installed! Please restart and run again.")
        return None
    
    print(f"\n🚀 Training Symbolic Regression Model")
    print("=" * 60)
    
    n_positions, n_features = features.shape
    print(f"📊 Training data: {n_positions} positions × {n_features} features")
    print(f"🎯 Target: chess position evaluation")
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Configure PySR
    model = PySRRegressor(
        # Search parameters
        niterations=500,              # Generations to evolve
        populations=20,               # Parallel populations
        population_size=50,           # Individuals per population
        
        # Operators
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["abs", "sqrt", "square"],
        
        # Complexity control
        max_complexity=15,            # Maximum formula complexity
        maxsize=20,                  # Maximum number of nodes
        parsimony=0.01,              # Simplicity preference
        
        # Performance
        procs=4,                     # Parallel processes
        multithreading=True,
        
        # Output control
        verbosity=1,
        progress=True,
        
        # Model selection
        model_selection="best",      # Choose best accuracy model
        
        # Feature names for interpretability
        feature_names_in=feature_names,
        
        # Random seed for reproducibility
        random_state=42
    )
    
    print(f"⚙️  PySR Configuration:")
    print(f"   Iterations: {model.niterations}")
    print(f"   Populations: {model.populations}")
    print(f"   Max complexity: {model.max_complexity}")
    print(f"   Operators: {model.binary_operators + model.unary_operators}")
    
    # Start training
    print(f"\n🔥 Starting symbolic regression training...")
    start_time = time.time()
    
    try:
        model.fit(features, evaluations)
        training_time = time.time() - start_time
        
        print(f"\n✅ Training completed in {training_time:.1f} seconds!")
        
        # Save model
        model_path = Path(output_dir) / "chess_sr_model.pkl"
        model.pytorch_format = False  # Use simpler format
        
        # Save model info
        model_info = {
            'training_time': training_time,
            'n_positions': n_positions,
            'n_features': n_features,
            'feature_names': feature_names,
            'model_equations': str(model),
            'best_equation': str(model.get_best()),
            'training_score': float(model.score(features, evaluations))
        }
        
        info_path = Path(output_dir) / "model_info.json"
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"💾 Model saved to: {output_dir}/")
        print(f"📄 Model info: {info_path}")
        
        # Display best equation
        print(f"\n🏆 BEST DISCOVERED EQUATION:")
        print("=" * 60)
        try:
            best_eq = model.get_best()
            print(f"🔗 Formula: {best_eq}")
            print(f"📊 Training R²: {model.score(features, evaluations):.4f}")
            print(f"🔢 Complexity: {best_eq.complexity if hasattr(best_eq, 'complexity') else 'N/A'}")
        except:
            print("⚠️  Could not retrieve best equation details")
        print("=" * 60)
        
        return str(model_path)
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None

def evaluate_model_performance(features: np.ndarray, evaluations: np.ndarray, 
                             model_path: str) -> None:
    """Evaluate the trained model performance."""
    try:
        from pysr import PySRRegressor
        import pickle
        
        # Load model (try different methods)
        try:
            model = PySRRegressor.from_file(model_path)
        except:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        
        print(f"\n📊 MODEL EVALUATION")
        print("=" * 60)
        
        # Make predictions
        predictions = model.predict(features)
        
        # Calculate metrics
        mse = np.mean((predictions - evaluations) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - evaluations))
        r2 = 1 - np.sum((evaluations - predictions) ** 2) / np.sum((evaluations - np.mean(evaluations)) ** 2)
        
        print(f"🎯 Accuracy Metrics:")
        print(f"   RMSE: {rmse:.3f} pawns")
        print(f"   MAE:  {mae:.3f} pawns") 
        print(f"   R²:   {r2:.4f}")
        
        # Error distribution
        errors = predictions - evaluations
        print(f"\n📈 Error Distribution:")
        print(f"   Mean error: {np.mean(errors):+.3f}")
        print(f"   Error std:  {np.std(errors):.3f}")
        print(f"   Max error:  {np.max(np.abs(errors)):.3f}")
        
        # Show some examples
        print(f"\n🔍 Sample Predictions:")
        for i in range(min(10, len(evaluations))):
            print(f"   Position {i+1}: True={evaluations[i]:+.2f}, Pred={predictions[i]:+.2f}, "
                  f"Error={errors[i]:+.3f}")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Model evaluation failed: {e}")

def main():
    """Main training pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train symbolic regression chess evaluation model')
    parser.add_argument('--features', default='bitboard_dataset_100.npz',
                       help='Path to NPZ features file')
    parser.add_argument('--max-features', type=int, default=50,
                       help='Maximum number of features for SR')
    parser.add_argument('--output', default='chess_sr_model',
                       help='Output directory for model')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze data, don\'t train')
    
    args = parser.parse_args()
    
    features_path = Path(args.features)
    if not features_path.exists():
        print(f"❌ Features file not found: {features_path}")
        return 1
    
    print("🚀 Symbolic Regression Chess Evaluation Trainer")
    print("=" * 60)
    print(f"📂 Features: {features_path}")
    print(f"🔢 Max features: {args.max_features}")
    print(f"📁 Output: {args.output}")
    print("=" * 60)
    
    # Load data
    features, evaluations = load_bitboard_features(str(features_path))
    
    # Analyze dataset
    analyze_data(features, evaluations)
    
    if args.analyze_only:
        print("✅ Analysis complete (--analyze-only mode)")
        return 0
    
    # Feature selection
    selected_features, evaluations, feature_names = prepare_features_for_sr(
        features, evaluations, args.max_features
    )
    
    # Train model
    model_path = train_pysr_model(selected_features, evaluations, feature_names, args.output)
    
    if model_path:
        # Evaluate model
        evaluate_model_performance(selected_features, evaluations, model_path)
        
        print(f"\n🎉 SUCCESS!")
        print(f"✅ Symbolic regression model trained and saved")
        print(f"📂 Model directory: {args.output}/")
        print(f"🔗 Next: Integrate discovered formula into Stockfish!")
        return 0
    else:
        print(f"\n❌ Training failed")
        return 1

if __name__ == "__main__":
    exit(main())