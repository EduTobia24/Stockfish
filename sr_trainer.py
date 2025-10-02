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
import pandas as pd
import shutil
from datetime import datetime

def custom_loss_function(y_true, y_pred):
    """
    Custom loss function that minimizes the maximum error (worst case).
    This focuses on reducing the biggest prediction errors rather than average error.
    """
    import numpy as np
    
    # Calculate absolute errors for each prediction
    errors = np.abs(y_true - y_pred)
    
    # Return the maximum error (worst case)
    # This forces the model to minimize the biggest mistakes
    max_error = np.max(errors)
    
    # Add a small penalty for average error to maintain overall accuracy
    mean_error = np.mean(errors)
    
    # Weighted combination: 80% max error + 20% mean error
    return 0.8 * max_error + 0.2 * mean_error

def percentile_loss_function(y_true, y_pred, percentile=95):
    """
    Loss function that minimizes the 95th percentile error.
    This focuses on reducing the worst 5% of predictions.
    """
    import numpy as np
    
    errors = np.abs(y_true - y_pred)
    percentile_error = np.percentile(errors, percentile)
    mean_error = np.mean(errors)
    
    # Weighted combination: 70% percentile error + 30% mean error
    return 0.7 * percentile_error + 0.3 * mean_error

def huber_loss_function(y_true, y_pred, delta=1.0):
    """
    Huber loss function - less sensitive to outliers than MSE.
    Combines L1 and L2 loss for robust regression.
    """
    import numpy as np
    
    errors = y_true - y_pred
    abs_errors = np.abs(errors)
    
    # Use L2 loss for small errors, L1 loss for large errors
    quadratic = np.minimum(abs_errors, delta)
    linear = abs_errors - quadratic
    
    loss = 0.5 * quadratic**2 + delta * linear
    return np.mean(loss)

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

def load_previous_hall_of_fame(previous_path):
    """Cargar ecuaciones del hall_of_fame anterior para entrenamiento incremental."""
    if not os.path.exists(previous_path):
        print(f"❌ Archivo hall_of_fame anterior no encontrado: {previous_path}")
        return None
    
    try:
        df = pd.read_csv(previous_path)
        print(f"✅ Cargadas {len(df)} ecuaciones del entrenamiento anterior")
        print(f"   Mejor pérdida anterior: {df['Loss'].min():.6f}")
        print(f"   Mejor complejidad anterior: {df.loc[df['Loss'].idxmin(), 'Complexity']}")
        
        # Mostrar progreso del entrenamiento anterior
        if len(df) > 1:
            print(f"   Rango de complejidad: {df['Complexity'].min()}-{df['Complexity'].max()}")
            print(f"   Rango de pérdida: {df['Loss'].min():.6f}-{df['Loss'].max():.6f}")
        
        return df
    except Exception as e:
        print(f"❌ Error cargando hall_of_fame anterior: {e}")
        return None

def setup_incremental_training(previous_hall_of_fame, output_dir):
    """Configurar entrenamiento incremental copiando ecuaciones anteriores."""
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    # Copiar hall_of_fame anterior como punto de partida
    previous_equations_path = os.path.join(output_dir, "previous_equations.csv")
    shutil.copy2(previous_hall_of_fame, previous_equations_path)
    
    print(f"📂 Configuración de entrenamiento incremental:")
    print(f"   Ecuaciones anteriores copiadas a: {previous_equations_path}")
    print(f"   Directorio de salida: {output_dir}")
    
    return previous_equations_path

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

def train_high_complexity_sr_model(features, evaluations, feature_names, max_complexity=100, previous_equations_path=None, output_dir="outputs", loss_function="mse"):
    """Train symbolic regression with high complexity for large datasets and custom loss functions.
    
    Args:
        features: Feature matrix
        evaluations: Target evaluations  
        feature_names: Names of features
        max_complexity: Maximum equation complexity
        previous_equations_path: Path to previous equations for warm start
        output_dir: Output directory
        loss_function: Loss function type - "mse", "max_error", "percentile", or "huber"
    """
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
    
    # Configurar archivo de ecuaciones para entrenamiento incremental
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    equation_file = os.path.join(output_dir, f"hall_of_fame_{timestamp}.csv")
    
    # Configuración base de PySR
    pysr_config = {
        # Evolution parameters (scaled for larger datasets)
        'niterations': 1000,              # More iterations for complex search
        'populations': 30,               # More populations for exploration
        'population_size': 50,           # Larger population for diversity
        
        # Operators (expanded set for complex patterns)
        'binary_operators': ["+", "-", "*", "/"],  # Include division
        'unary_operators': ["abs", "sqrt", "square"],  # More unary functions
        
        # Complexity control (HIGH for large datasets)
        'maxsize': max_complexity,       # High complexity limit
        'parsimony': 0.05,              # Lower parsimony for complex expressions
        
        # Performance (optimized for large datasets)
        'procs': 4,                     # Use multiple cores
        'multithreading': True,
        'batching': True,               # Essential for large datasets
        'batch_size': 100,              # Larger batches
        
        # Model selection
        'model_selection': "best",      # Choose best accuracy
        
        # Advanced optimization
        'weight_optimize': 0.01,        # More weight optimization
        'weight_mutate_constant': 0.1,
        'weight_mutate_operator': 0.15,
        
        # Output control
        'verbosity': 1,
        'progress': True,
        
        # Random seed
        'random_state': 42,
        
        # Extended timeout for complex search
        'timeout_in_seconds': 3600,     # 1 hour max
        
        # Remove temp_equation_file since it doesn't work for continuous saving
        # We'll implement manual saving instead
    }
    
    # Add custom loss function based on user choice
    if loss_function == "max_error":
        pysr_config['loss_function'] = custom_loss_function
        print(f"🎯 Using MAXIMUM ERROR loss function (minimizes worst predictions)")
    elif loss_function == "percentile":
        pysr_config['loss_function'] = lambda y_true, y_pred: percentile_loss_function(y_true, y_pred, 95)
        print(f"🎯 Using 95th PERCENTILE loss function (minimizes worst 5% of predictions)")
    elif loss_function == "huber":
        pysr_config['loss_function'] = lambda y_true, y_pred: huber_loss_function(y_true, y_pred, 1.0)
        print(f"🎯 Using HUBER loss function (robust to outliers)")
    else:
        print(f"🎯 Using default MSE loss function")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    hall_of_fame_path = os.path.join(output_dir, "hall_of_fame.csv")
    
    # Proper warm start implementation using PySRRegressor.from_file()
    if previous_equations_path and os.path.exists(previous_equations_path):
        try:
            # Extract the model directory from the equations path
            # previous_equations_path should be like "outputs/20251002_184214_OEeD6B/hall_of_fame.csv"
            model_dir = os.path.dirname(previous_equations_path)
            
            print(f"🔥 ENTRENAMIENTO INCREMENTAL ACTIVADO")
            print(f"   📂 Cargando modelo desde: {model_dir}")
            
            # Load the previous model using PySRRegressor.from_file()
            model = PySRRegressor.from_file(model_dir)
            
            # Update model parameters for continued training
            model.niterations = pysr_config['niterations']
            model.timeout_in_seconds = pysr_config['timeout_in_seconds']
            
            # Read previous performance
            if os.path.exists(previous_equations_path):
                prev_df = pd.read_csv(previous_equations_path)
                print(f"   📋 Ecuaciones previas: {len(prev_df)}")
                print(f"   📊 Mejor pérdida previa: {prev_df['Loss'].min():.6f}")
                if 'Score' in prev_df.columns:
                    print(f"   � Mejor R² previo: {prev_df['Score'].max():.4f}")
            
            print(f"   💾 Continuando entrenamiento...")
            
        except Exception as e:
            print(f"⚠️ Error cargando modelo previo: {e}")
            print(f"   Creando modelo nuevo...")
            model = PySRRegressor(**pysr_config)
    else:
        print(f"🆕 ENTRENAMIENTO DESDE CERO")
        print(f"   💾 Hall of fame: {hall_of_fame_path}")
        
        # Configure PySR for high complexity and large datasets
        model = PySRRegressor(**pysr_config)
    print(f"⚙️  PySR High-Complexity Configuration:")
    print(f"   Iterations: {model.niterations}")
    print(f"   Populations: {model.populations}")
    print(f"   Max size: {model.maxsize}")
    print(f"   Operators: {model.binary_operators + model.unary_operators}")
    print(f"   Timeout: {model.timeout_in_seconds/60:.0f} minutes")
    print(f"   Batch size: {model.batch_size}")
    print(f"   Warm start: {'✅ SÍ' if previous_equations_path else '❌ NO'}")
    print(f"   Modo: {'INCREMENTAL' if previous_equations_path else 'DESDE CERO'}")
    
    print(f"\n🔥 Starting high-complexity symbolic regression...")
    print(f"⏱️  This may take 30-60 minutes with {n_features} features and complexity {max_complexity}...")
    print("\n" + "="*80)
    print("🚀 TRAINING IN PROGRESS - PySR OUTPUT:")
    print("="*80)
    
    # Force clear and flush
    import sys
    sys.stdout.flush()
    
    # Create a custom callback to save hall of fame during training
    import threading
    import time
    
    def periodic_save():
        """Save hall of fame every 5 minutes during training."""
        while getattr(periodic_save, 'training_active', True):
            time.sleep(300)  # 5 minutes
            if hasattr(model, 'equations_') and model.equations_ is not None:
                try:
                    temp_path = hall_of_fame_path.replace('.csv', '_temp.csv')
                    model.equations_.to_csv(temp_path, index=False)
                    # Atomic move to prevent corruption
                    import shutil
                    shutil.move(temp_path, hall_of_fame_path)
                    print(f"\n💾 Hall of fame auto-saved: {len(model.equations_)} equations")
                except Exception as e:
                    print(f"\n⚠️ Auto-save failed: {e}")
    
    # Start periodic saving thread
    periodic_save.training_active = True
    save_thread = threading.Thread(target=periodic_save, daemon=True)
    save_thread.start()
    
    try:
        # Train the model
        model.fit(features, evaluations)
        
        print(f"\n✅ Training completed!")
        
        # IMPORTANT: Save hall of fame manually after training
        hall_of_fame_path = os.path.join(output_dir, "hall_of_fame.csv")
        try:
            # Save the equations DataFrame to CSV
            if hasattr(model, 'equations_'):
                print(f"💾 Saving hall of fame to: {hall_of_fame_path}")
                model.equations_.to_csv(hall_of_fame_path, index=False)
                print(f"✅ Hall of fame saved with {len(model.equations_)} equations")
            else:
                print(f"⚠️ No equations_ attribute found in model")
        except Exception as save_error:
            print(f"❌ Error saving hall of fame: {save_error}")
            # Try alternative save method
            try:
                model.to_file(output_dir)
                print(f"✅ Model saved to directory: {output_dir}")
            except Exception as model_save_error:
                print(f"❌ Error saving model: {model_save_error}")
        
        # Strong clear before results
        import time
        time.sleep(1)  # Small delay
        
        # Stop periodic saving
        periodic_save.training_active = False
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
        # Stop periodic saving in case of error
        periodic_save.training_active = False
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
    parser.add_argument('--previous', type=str, default=None,
                       help='Path to previous hall_of_fame.csv for incremental training')
    parser.add_argument('--loss', type=str, default='mse', 
                       choices=['mse', 'max_error', 'percentile', 'huber'],
                       help='Loss function: mse (default), max_error (minimize worst predictions), percentile (minimize worst 5%), huber (robust)')
    
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
    
    # Configurar directorio de salida
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.previous:
        output_dir = f"outputs/incremental_{timestamp}"
        mode = "INCREMENTAL"
    else:
        output_dir = f"outputs/fresh_{timestamp}"
        mode = "DESDE CERO"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("🚀 High-Complexity Symbolic Regression Trainer")
    print("=" * 70)
    print(f"📂 Dataset: {dataset_path}")
    print(f"🔢 Using ALL features (no selection)")
    print(f"🔥 Max complexity: {args.complexity}")
    print(f"⏱️  Timeout: {args.timeout/60:.0f} minutes")
    print(f"🎯 Loss function: {args.loss}")
    print(f"🎯 Modo: {mode}")
    if args.previous:
        print(f"📋 Ecuaciones anteriores: {args.previous}")
    print(f"💾 Salida: {output_dir}")
    print("=" * 70)
    
    # Load data
    features, evaluations = load_and_analyze_data(str(dataset_path))
    
    # Generate feature names
    feature_names = get_all_feature_names()
    print(f"✅ Generated {len(feature_names)} feature names")
    
    # Configurar entrenamiento incremental si se especifica
    previous_equations_path = None
    if args.previous:
        # Validar archivo anterior
        if not os.path.exists(args.previous):
            print(f"❌ Archivo hall_of_fame anterior no encontrado: {args.previous}")
            return 1
        
        # Cargar y analizar ecuaciones anteriores
        previous_df = load_previous_hall_of_fame(args.previous)
        if previous_df is None:
            return 1
        
        # Configurar entrenamiento incremental
        previous_equations_path = setup_incremental_training(args.previous, output_dir)
    
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
    model = train_high_complexity_sr_model(
        features, evaluations, feature_names, 
        args.complexity, previous_equations_path, output_dir, args.loss
    )
    
    if model:
        print(f"\n🎉 ¡ÉXITO!")
        print(f"✅ Entrenamiento de regresión simbólica completado")
        if args.previous:
            print(f"🔄 Entrenamiento incremental desde ecuaciones anteriores")
        print(f"🧬 Fórmula de evaluación de ajedrez descubierta")
        print(f"� Resultados guardados en: {output_dir}")
        print(f"📈 ¡Listo para integración avanzada en motor de ajedrez!")
        return 0
    else:
        print(f"\n❌ Entrenamiento falló")
        print(f"💾 Resultados parciales en: {output_dir}")
        return 1

if __name__ == "__main__":
    exit(main())