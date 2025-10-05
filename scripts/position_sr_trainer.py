#!/usr/bin/env python3
"""
Position-based Symbolic Regression Trainer
==========================================

Advanced symbolic regression trainer for position-based chess features with:
1. Configurable training parameters and loss functions
2. Robust periodic saving with backup strategies
3. Proper warm starting from previous models

Feature format: 73 features (64 piece positions + 9 game state)
"""

import numpy as np
import pandas as pd
import json
import os
import shutil
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import argparse

class SRTrainerConfig:
    """Configuration class for SR trainer parameters."""
    
    def __init__(self):
        # Training parameters
        self.max_complexity = 25
        self.niterations = 5000
        self.populations = 20
        self.population_size = 100
        self.timeout_hours = 8.0
        
        # Loss function configuration
        self.loss_function = "mse"  # mse, max_error, percentile, huber, mae
        self.loss_percentile = 95   # for percentile loss
        self.huber_delta = 1.0      # for huber loss
        
        # Operators
        self.binary_operators = ["+", "-", "*", "/"]
        self.unary_operators = ["abs", "sqrt", "square"]
        
        # Constraints (safer bounds to prevent Julia errors)
        self.constraints = {
            'square': 3,         # Limit square operations  
            'sqrt': 3,           # Limit sqrt operations
            'abs': 2,            # Limit abs operations
        }
        self.nested_constraints = {     # Prevent deeply nested expressions
            'square': {'square': 1},     # Limit nested square operations
            'sqrt': {'sqrt': 1},         # Limit nested sqrt operations
        }
        
        # Performance settings
        self.procs = 8
        self.parallelism = "multiprocessing"  # multiprocessing for speed, serial for deterministic
        self.batching = True
        self.batch_size = 100
        self.turbo = True
        
        # Deterministic settings
        self.deterministic = False  # Disable for better performance
        
        # Julia optimization settings
        self.optimizer_iterations = 8
        self.fast_cycle = False
        
        # Model selection
        self.model_selection = "best"
        self.parsimony = 0.01
        self.weight_optimize = 0.001
        
        # Output and saving
        self.verbosity = 1
        self.progress = True
        self.save_interval_minutes = 0.5
        self.backup_count = 5
        
        # Random seed
        self.random_state = 42
        
        # Warm start
        self.warm_start = False  # Enable warm start functionality
        self.warm_start_path = None
        self.warm_start_mode = "auto"  # auto, strict, lenient
        self.warm_start_increase_complexity = True  # Allow complexity to be increased on warm start
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Load config from dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def save(self, filepath: str) -> None:
        """Save config to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def load(self, filepath: str) -> None:
        """Load config from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        self.from_dict(config_dict)

class PeriodicSaver:
    """Handles periodic saving of training results with backup management."""
    
    def __init__(self, output_dir: str, save_interval_minutes: float, backup_count: int):
        self.output_dir = Path(output_dir)
        self.save_interval = save_interval_minutes * 60  # Convert to seconds
        self.backup_count = backup_count
        self.is_active = False
        self.thread = None
        self.model = None
        self.last_save_time = time.time()
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def start(self, model) -> None:
        """Start periodic saving thread."""
        self.model = model
        self.is_active = True
        self.thread = threading.Thread(target=self._save_loop, daemon=True)
        self.thread.start()
        print(f"💾 Periodic saving started (every {self.save_interval/60:.1f} minutes)")
        
    def stop(self) -> None:
        """Stop periodic saving thread."""
        self.is_active = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        print(f"💾 Periodic saving stopped")
        
    def _save_loop(self) -> None:
        """Main saving loop running in separate thread."""
        while self.is_active:
            time.sleep(10)  # Check every 10 seconds
            
            if time.time() - self.last_save_time >= self.save_interval:
                try:
                    self._save_checkpoint()
                    self.last_save_time = time.time()
                except Exception as e:
                    print(f"⚠️ Periodic save failed: {e}")
                    
    def _save_checkpoint(self) -> None:
        """Save current model state with backup rotation."""
        if not self.model:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Always create a checkpoint marker to show saving is working
            checkpoint_marker = self.output_dir / f"checkpoint_marker_{timestamp}.txt"
            
            with open(checkpoint_marker, 'w') as f:
                f.write(f"Checkpoint at {timestamp}\n")
                f.write(f"Model state: {type(self.model).__name__}\n")
                f.write(f"Has equations: {hasattr(self.model, 'equations_')}\n")
                if hasattr(self.model, 'equations_'):
                    f.write(f"Equations available: {self.model.equations_ is not None}\n")
                    if self.model.equations_ is not None:
                        f.write(f"Number of equations: {len(self.model.equations_)}\n")
            
            # Save equations if available
            if hasattr(self.model, 'equations_') and self.model.equations_ is not None and len(self.model.equations_) > 0:
                equations_file = self.output_dir / f"checkpoint_equations_{timestamp}.csv"
                self.model.equations_.to_csv(equations_file, index=False)
                
                # Create symlink to latest
                latest_link = self.output_dir / "latest_equations.csv"
                if latest_link.exists() or latest_link.is_symlink():
                    latest_link.unlink()
                
                # Create relative symlink
                try:
                    latest_link.symlink_to(equations_file.name)
                except OSError:
                    # Fallback: copy file if symlink fails (Windows compatibility)
                    shutil.copy2(equations_file, latest_link)
                
                print(f"💾 Checkpoint saved: {len(self.model.equations_)} equations")
                
                # Manage backup rotation
                self._rotate_backups("checkpoint_equations_*.csv")
            else:
                # Just log that checkpoint marker was created
                print(f"🔍 Checkpoint marker saved (equations not yet available)")
                
            # Clean up old checkpoint markers (keep only 3 most recent)
            self._rotate_backups("checkpoint_marker_*.txt", max_files=3)
                
        except Exception as e:
            print(f"❌ Checkpoint save error: {e}")
            import traceback
            print(f"📋 Traceback: {traceback.format_exc()}")
            
    def _rotate_backups(self, pattern: str, max_files: int = None) -> None:
        """Rotate backup files, keeping only the most recent ones."""
        try:
            backup_files = list(self.output_dir.glob(pattern))
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Use provided max_files or default backup_count
            limit = max_files if max_files is not None else self.backup_count
            
            # Remove old backups beyond the limit
            for old_file in backup_files[limit:]:
                old_file.unlink()
                
        except Exception as e:
            print(f"⚠️ Backup rotation error: {e}")
            
    def save_final(self, model, config: SRTrainerConfig, metadata: Dict[str, Any]) -> str:
        """Save final results with comprehensive metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save equations
        equations_file = self.output_dir / "hall_of_fame.csv"
        if hasattr(model, 'equations_') and model.equations_ is not None:
            model.equations_.to_csv(equations_file, index=False)
            print(f"✅ Final equations saved: {len(model.equations_)} entries")
        
        # Save complete model
        try:
            model.to_file(str(self.output_dir))
            print(f"✅ Complete model saved to: {self.output_dir}")
        except Exception as e:
            print(f"⚠️ Model save warning: {e}")
        
        # Save configuration
        config_file = self.output_dir / "training_config.json"
        config.save(str(config_file))
        
        # Save metadata
        metadata_file = self.output_dir / "training_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Create training summary
        summary_file = self.output_dir / "training_summary.md"
        self._create_summary(summary_file, model, config, metadata)
        
        return str(self.output_dir)
        
    def _create_summary(self, summary_file: Path, model, config: SRTrainerConfig, metadata: Dict[str, Any]) -> None:
        """Create a markdown training summary."""
        with open(summary_file, 'w') as f:
            f.write(f"# Training Summary\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"## Configuration\n")
            f.write(f"- **Loss Function:** {config.loss_function}\n")
            f.write(f"- **Max Complexity:** {config.max_complexity}\n")
            f.write(f"- **Iterations:** {config.niterations}\n")
            f.write(f"- **Timeout:** {config.timeout_hours} hours\n")
            f.write(f"- **Populations:** {config.populations}\n")
            f.write(f"- **Population Size:** {config.population_size}\n\n")
            
            f.write(f"## Dataset\n")
            f.write(f"- **Positions:** {metadata.get('n_positions', 'N/A')}\n")
            f.write(f"- **Features:** {metadata.get('n_features', 'N/A')}\n")
            f.write(f"- **Evaluation Range:** {metadata.get('eval_range', 'N/A')}\n\n")
            
            if hasattr(model, 'equations_') and model.equations_ is not None:
                best_eq = model.equations_.iloc[0] if len(model.equations_) > 0 else None
                if best_eq is not None:
                    f.write(f"## Best Result\n")
                    
                    # Robust column access for summary
                    equation_text = "N/A"
                    loss_value = "N/A"
                    complexity_value = "N/A"
                    score_value = "N/A"
                    
                    # Try different equation column names
                    for eq_col in ['Equation', 'equation', 'sympy_format']:
                        if eq_col in best_eq.index:
                            equation_text = str(best_eq[eq_col])
                            break
                    
                    # Try different loss column names
                    for loss_col in ['Loss', 'loss', 'mse']:
                        if loss_col in best_eq.index:
                            loss_value = f"{best_eq[loss_col]:.6f}"
                            break
                    
                    # Try different complexity column names
                    for comp_col in ['Complexity', 'complexity', 'size']:
                        if comp_col in best_eq.index:
                            complexity_value = str(best_eq[comp_col])
                            break
                    
                    # Try different score column names
                    for score_col in ['Score', 'score', 'r2', 'R2']:
                        if score_col in best_eq.index:
                            score_value = f"{best_eq[score_col]:.6f}"
                            break
                    
                    f.write(f"- **Equation:** `{equation_text}`\n")
                    f.write(f"- **Loss:** {loss_value}\n")
                    f.write(f"- **Complexity:** {complexity_value}\n")
                    if score_value != "N/A":
                        f.write(f"- **R² Score:** {score_value}\n")
                    f.write(f"\n")

class PositionSRTrainer:
    """Main trainer class for position-based symbolic regression."""
    
    def __init__(self, config: SRTrainerConfig):
        self.config = config
        self.model = None
        self.saver = None
        self.feature_names = None
        
    def load_data(self, dataset_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load position dataset from NPZ file."""
        print(f"📊 Loading dataset from {dataset_path}")
        
        data = np.load(dataset_path)
        features = data['features']
        evaluations = data['evaluations']
        
        print(f"✅ Loaded {len(features)} positions with {features.shape[1]} features")
        print(f"📈 Evaluation range: {evaluations.min():+.2f} to {evaluations.max():+.2f}")
        print(f"✨ Feature density: {np.count_nonzero(features)/features.size:.1%}")
        
        return features, evaluations
    
    def load_feature_names(self, dataset_path: str) -> List[str]:
        """Load feature names from JSON file or generate defaults."""
        names_file = dataset_path.replace('.npz', '_feature_names.json')
        
        if os.path.exists(names_file):
            with open(names_file, 'r') as f:
                feature_names = json.load(f)
            print(f"✅ Loaded {len(feature_names)} feature names")
        else:
            # Generate default names
            feature_names = [f"feature_{i}" for i in range(73)]
            print(f"⚠️ Feature names file not found, using defaults")
            
        return feature_names
    
    def setup_warm_start(self, output_dir: str) -> Optional[object]:
        """Setup warm start from previous training if configured."""
        # Enable warm start if path is provided
        if self.config.warm_start_path:
            self.config.warm_start = True
        
        if not self.config.warm_start or not self.config.warm_start_path:
            return None
            
        warm_start_path = Path(self.config.warm_start_path)
        
        if not warm_start_path.exists():
            print(f"⚠️ Warm start path not found: {warm_start_path}")
            if self.config.warm_start_mode == "strict":
                raise FileNotFoundError(f"Strict warm start failed: {warm_start_path}")
            return None
        
        try:
            from pysr import PySRRegressor
            
            print(f"🔥 Setting up warm start from: {warm_start_path}")
            
            # Try to load previous model with better error handling
            try:
                if warm_start_path.is_dir():
                    # Load from directory - PySR expects the directory itself
                    model = PySRRegressor.from_file(run_directory=str(warm_start_path))
                elif warm_start_path.suffix == '.csv':
                    # Load from hall of fame file - use parent directory
                    model_dir = warm_start_path.parent
                    model = PySRRegressor.from_file(run_directory=str(model_dir))
                else:
                    raise ValueError(f"Unsupported warm start path: {warm_start_path}")
            except TypeError as e:
                # Handle older PySR versions that don't use run_directory
                print(f"⚠️ New PySR API failed, trying legacy method: {e}")
                if warm_start_path.is_dir():
                    model = PySRRegressor.from_file(str(warm_start_path))
                elif warm_start_path.suffix == '.csv':
                    model_dir = warm_start_path.parent
                    model = PySRRegressor.from_file(str(model_dir))
                else:
                    raise ValueError(f"Unsupported warm start path: {warm_start_path}")
            
            # Update model parameters for continued training with new complexity
            old_complexity = getattr(model, 'maxsize', None) or getattr(model, 'max_size', 'unknown')
            model.niterations = self.config.niterations
            model.timeout_in_seconds = self.config.timeout_hours * 3600
            
            if self.config.warm_start_increase_complexity:
                # Allow complexity to be increased
                model.maxsize = self.config.max_complexity  # Use maxsize instead of max_size
                model.max_size = self.config.max_complexity  # Also set max_size for compatibility
                
                # Override other complexity-related parameters to allow growth
                model.parsimony = self.config.parsimony
                model.weight_optimize = self.config.weight_optimize
                
                # Ensure complexity can be increased
                if hasattr(model, 'options') and model.options:
                    model.options['maxsize'] = self.config.max_complexity
                    model.options['parsimony'] = self.config.parsimony
                
                print(f"🔥 Complexity progression: {old_complexity} → {self.config.max_complexity}")
            else:
                # Keep original complexity
                print(f"🔒 Keeping original complexity: {old_complexity}")
            
            print(f"🔥 Warm start configured successfully")
            
            # Load previous results for analysis
            equations_file = None
            if warm_start_path.is_dir():
                equations_file = warm_start_path / "hall_of_fame.csv"
            elif warm_start_path.suffix == '.csv':
                equations_file = warm_start_path
                
            if equations_file and equations_file.exists():
                prev_equations = pd.read_csv(equations_file)
                print(f"📋 Previous training had {len(prev_equations)} equations")
                
                # Robust loss column access
                best_loss = "N/A"
                best_score = "N/A"
                
                for loss_col in ['Loss', 'loss', 'mse']:
                    if loss_col in prev_equations.columns:
                        best_loss = f"{prev_equations[loss_col].min():.6f}"
                        break
                
                for score_col in ['Score', 'score', 'r2', 'R2']:
                    if score_col in prev_equations.columns:
                        best_score = f"{prev_equations[score_col].max():.6f}"
                        break
                
                print(f"📊 Best previous loss: {best_loss}")
                if best_score != "N/A":
                    print(f"🎯 Best previous R²: {best_score}")
            
            print(f"✅ Warm start configured successfully")
            return model
            
        except Exception as e:
            print(f"❌ Warm start failed: {e}")
            if self.config.warm_start_mode == "strict":
                raise
            elif self.config.warm_start_mode == "lenient":
                print(f"🔄 Continuing with fresh start")
                return None
            else:  # auto mode
                print(f"🔄 Auto-fallback to fresh start")
                return None
    
    def create_pysr_config(self, is_warm_start: bool = False) -> Dict[str, Any]:
        """Create PySR configuration dictionary."""
        pysr_config = {
            # Evolution parameters
            'niterations': self.config.niterations,
            'populations': self.config.populations,
            'population_size': self.config.population_size,
            
            # Operators and constraints
            'binary_operators': self.config.binary_operators,
            'unary_operators': self.config.unary_operators,
            'constraints': self.config.constraints,
            'nested_constraints': self.config.nested_constraints,
            
            # Complexity control
            'maxsize': self.config.max_complexity,
            'parsimony': self.config.parsimony,
            
            # Performance - optimized for speed
            'procs': self.config.procs,
            'parallelism': self.config.parallelism,
            'batching': self.config.batching,
            'batch_size': self.config.batch_size,
            'turbo': self.config.turbo,
            
            # Model selection
            'model_selection': self.config.model_selection,
            'weight_optimize': self.config.weight_optimize,
            
            # Output control
            'verbosity': self.config.verbosity,
            'progress': self.config.progress,
            
            # Timeout
            'timeout_in_seconds': self.config.timeout_hours * 3600,
            
            # Julia optimization
            'optimizer_iterations': self.config.optimizer_iterations,
            'fast_cycle': self.config.fast_cycle,
        }
        
        # Only set warm_start=True when actually doing a warm start
        if is_warm_start:
            pysr_config['warm_start'] = True
        
        # Handle deterministic vs non-deterministic mode
        if self.config.deterministic:
            # Deterministic mode: set random_state and force serial
            pysr_config.update({
                'random_state': self.config.random_state,
                'deterministic': True,
                'parallelism': 'serial'
            })
        else:
            # Non-deterministic mode for speed (no random_state to avoid warnings)
            pysr_config.update({
                'deterministic': False,
            })
        
        # Loss function specific adjustments
        if self.config.loss_function == "max_error":
            pysr_config.update({
                'parsimony': 0.1,
                'weight_optimize': 0.05,
                'alpha': 100.0,
            })
        elif self.config.loss_function == "percentile":
            pysr_config.update({
                'parsimony': 0.08,
                'weight_optimize': 0.1,
                'alpha': 150.0,
            })
        elif self.config.loss_function == "huber":
            pysr_config.update({
                'parsimony': 0.15,
                'weight_optimize': 0.02,
            })
        elif self.config.loss_function == "mae":
            pysr_config.update({
                'parsimony': 0.03,
                'weight_optimize': 0.005,
            })
        
        return pysr_config
    
    def train(self, features: np.ndarray, evaluations: np.ndarray, output_dir: str) -> object:
        """Main training function."""
        try:
            from pysr import PySRRegressor
        except ImportError:
            print("❌ PySR not available. Install with: pip install pysr")
            return None
        
        # Setup output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(output_dir) / timestamp
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Training output directory: {output_path}")
        
        # Setup periodic saver
        self.saver = PeriodicSaver(
            str(output_path), 
            self.config.save_interval_minutes, 
            self.config.backup_count
        )
        
        # Setup warm start or create new model
        model = self.setup_warm_start(str(output_path))
        
        if model is None:
            print(f"🆕 Creating fresh model")
            pysr_config = self.create_pysr_config(is_warm_start=False)
            model = PySRRegressor(**pysr_config)
        else:
            print(f"🔥 Using warm start model")
            # For warm start models, we need to update the config to continue from existing expressions
            pysr_config = self.create_pysr_config(is_warm_start=True)
            # Update the loaded model with new config but keep warm_start=True
            for key, value in pysr_config.items():
                if hasattr(model, key):
                    setattr(model, key, value)
        
        self.model = model
        
        # Display configuration
        print(f"\n⚙️ Training Configuration:")
        print(f"   Loss function: {self.config.loss_function}")
        print(f"   Max complexity: {self.config.max_complexity}")
        print(f"   Iterations: {self.config.niterations}")
        print(f"   Timeout: {self.config.timeout_hours} hours")
        print(f"   Populations: {self.config.populations}")
        print(f"   Population size: {self.config.population_size}")
        print(f"   Periodic saves: every {self.config.save_interval_minutes} minutes")
        print(f"   Output: {output_path}")
        
        # Start periodic saving
        self.saver.start(model)
        
        try:
            print(f"\n🚀 Starting symbolic regression training...")
            print(f"💡 Training {len(features)} positions with {features.shape[1]} features")
            print("=" * 80)
            
            # Train the model
            start_time = time.time()
            model.fit(features, evaluations)
            training_time = time.time() - start_time
            
            print(f"\n✅ Training completed in {training_time/3600:.2f} hours!")
            
            # Stop periodic saving
            self.saver.stop()
            
            # Analyze results
            self._analyze_results(model, features, evaluations)
            
            # Save final results
            metadata = {
                'training_time_seconds': training_time,
                'n_positions': len(features),
                'n_features': features.shape[1],
                'eval_range': f"{evaluations.min():+.2f} to {evaluations.max():+.2f}",
                'loss_function': self.config.loss_function,
                'warm_start_used': self.config.warm_start_path is not None,
            }
            
            final_path = self.saver.save_final(model, self.config, metadata)
            print(f"💾 Results saved to: {final_path}")
            
            return model
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            self.saver.stop()
            raise
        
    def _analyze_results(self, model, features: np.ndarray, evaluations: np.ndarray) -> None:
        """Analyze and display training results."""
        if not hasattr(model, 'equations_') or model.equations_ is None:
            print("⚠️ No equations found in model")
            return
        
        # Get best equation
        best_equation = model.get_best()
        training_score = model.score(features, evaluations)
        predictions = model.predict(features)
        
        # Calculate metrics
        rmse = np.sqrt(np.mean((predictions - evaluations) ** 2))
        mae = np.mean(np.abs(predictions - evaluations))
        max_error = np.max(np.abs(predictions - evaluations))
        
        print(f"\n🏆 TRAINING RESULTS:")
        print("=" * 80)
        print(f"📐 Best equation: {best_equation}")
        print(f"📊 Training R²: {training_score:.6f}")
        print(f"📈 RMSE: {rmse:.3f} pawns")
        print(f"📈 MAE: {mae:.3f} pawns")
        print(f"📈 Max error: {max_error:.3f} pawns")
        print(f"🧠 Equation complexity: {getattr(best_equation, 'complexity', 'N/A')}")
        print(f"📋 Total equations found: {len(model.equations_)}")
        print("=" * 80)
        
        # Sample predictions
        print(f"\n🔍 Sample Predictions:")
        n_samples = min(10, len(evaluations))
        for i in range(n_samples):
            error = predictions[i] - evaluations[i]
            accuracy = "✅" if abs(error) < 0.5 else "⚠️" if abs(error) < 1.0 else "❌"
            print(f"   {accuracy} Pos {i+1:2d}: Actual={evaluations[i]:+.2f}, "
                  f"Predicted={predictions[i]:+.2f}, Error={error:+.3f}")

def create_default_config() -> SRTrainerConfig:
    """Create default configuration."""
    return SRTrainerConfig()

def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Position-based Symbolic Regression Trainer')
    
    # Required arguments
    parser.add_argument('dataset', help='Path to position dataset NPZ file')
    parser.add_argument('--output', '-o', required=True, help='Output directory for results')
    
    # Training parameters
    parser.add_argument('--complexity', type=int, default=25, help='Maximum equation complexity')
    parser.add_argument('--iterations', type=int, default=5000, help='Number of iterations')
    parser.add_argument('--timeout', type=float, default=8.0, help='Timeout in hours')
    parser.add_argument('--populations', type=int, default=20, help='Number of populations')
    parser.add_argument('--population-size', type=int, default=100, help='Population size')
    
    # Loss function
    parser.add_argument('--loss', choices=['mse', 'max_error', 'percentile', 'huber', 'mae'],
                       default='mse', help='Loss function type')
    parser.add_argument('--loss-percentile', type=int, default=95, help='Percentile for percentile loss')
    parser.add_argument('--huber-delta', type=float, default=1.0, help='Delta parameter for Huber loss')
    
    # Saving options
    parser.add_argument('--save-interval', type=float, default=0.5, help='Save interval in minutes')
    parser.add_argument('--backup-count', type=int, default=5, help='Number of backup files to keep')
    

    
    # Warm start
    parser.add_argument('--warm-start', help='Path to previous model for warm start')
    parser.add_argument('--warm-start-mode', choices=['auto', 'strict', 'lenient'], 
                       default='auto', help='Warm start behavior')
    parser.add_argument('--keep-complexity', action='store_true', default=False, 
                       help='Keep original complexity when warm starting (default: allow increase)')
    
    # Performance
    parser.add_argument('--procs', type=int, default=8, help='Number of processes')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size')
    parser.add_argument('--parallelism', choices=['multiprocessing', 'multithreading', 'serial'],
                       default='multiprocessing', help='Parallelism mode (multiprocessing for speed)')
    parser.add_argument('--deterministic', action='store_true', default=False, help='Enable deterministic search (slower, requires serial parallelism)')
    parser.add_argument('--fast-mode', action='store_true', default=True, help='Use multiprocessing for speed (default, non-deterministic)')
    
    # Configuration file
    parser.add_argument('--config', help='Load configuration from JSON file')
    parser.add_argument('--save-config', help='Save configuration to JSON file and exit')
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_default_config()
    
    # Load from file if specified
    if args.config:
        config.load(args.config)
        print(f"✅ Loaded configuration from {args.config}")
    
    # Override with command line arguments
    config.max_complexity = args.complexity
    config.niterations = args.iterations
    config.timeout_hours = args.timeout
    config.populations = args.populations
    config.population_size = args.population_size
    config.loss_function = args.loss
    config.loss_percentile = args.loss_percentile
    config.huber_delta = args.huber_delta
    config.save_interval_minutes = args.save_interval
    config.backup_count = args.backup_count
    config.warm_start_path = args.warm_start
    config.warm_start_mode = args.warm_start_mode
    config.warm_start_increase_complexity = not args.keep_complexity
    config.procs = args.procs
    config.batch_size = args.batch_size
    config.parallelism = args.parallelism
    config.deterministic = args.deterministic
    
    # Handle deterministic mode override (forces serial)
    if args.deterministic:
        print("� Deterministic mode enabled: using serial parallelism (slower but reproducible)")
        config.parallelism = 'serial'
        config.deterministic = True
    

    
    # Save config and exit if requested
    if args.save_config:
        config.save(args.save_config)
        print(f"💾 Configuration saved to {args.save_config}")
        return 0
    
    # Validate inputs
    if not Path(args.dataset).exists():
        print(f"❌ Dataset not found: {args.dataset}")
        return 1
    
    print("🚀 Position-based Symbolic Regression Trainer")
    print("=" * 80)
    print(f"📂 Dataset: {args.dataset}")
    print(f"💾 Output: {args.output}")
    print(f"🎯 Loss function: {config.loss_function}")
    print(f"🔥 Max complexity: {config.max_complexity}")
    print(f"⏱️ Timeout: {config.timeout_hours} hours")
    print("=" * 80)
    
    # Create trainer and run
    trainer = PositionSRTrainer(config)
    
    try:
        # Load data
        features, evaluations = trainer.load_data(args.dataset)
        feature_names = trainer.load_feature_names(args.dataset)
        trainer.feature_names = feature_names
        
        # Train model
        model = trainer.train(features, evaluations, args.output)
        
        if model:
            print(f"\n🎉 Training completed successfully!")
            print(f"📈 Ready for chess engine integration!")
            return 0
        else:
            print(f"\n❌ Training failed!")
            return 1
            
    except KeyboardInterrupt:
        print(f"\n⏹️ Training interrupted by user")
        if trainer.saver:
            trainer.saver.stop()
        return 1
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())