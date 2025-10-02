#!/usr/bin/env python3
"""
Test Different Loss Functions
============================

Test how different loss functions affect symbolic regression results.
"""

import os
import numpy as np
from sr_trainer import train_high_complexity_sr_model, get_all_feature_names

def test_loss_functions():
    """Test different loss functions on the same data."""
    
    print("🧪 Testing Different Loss Functions")
    print("="*60)
    
    # Create test data with outliers to see loss function differences
    n_positions = 100
    n_features = 782
    
    np.random.seed(42)
    features = np.random.rand(n_positions, n_features) * 0.1  # Sparse features
    
    # Create evaluations with some outliers
    base_eval = np.sum(features[:, :10], axis=1)
    noise = np.random.normal(0, 0.1, n_positions)
    
    # Add some outliers (10% of data)
    outlier_indices = np.random.choice(n_positions, size=n_positions//10, replace=False)
    outliers = np.random.normal(0, 2.0, len(outlier_indices))  # Large outliers
    
    evaluations = base_eval + noise
    evaluations[outlier_indices] += outliers
    
    feature_names = get_all_feature_names()
    
    print(f"📊 Test data: {n_positions} positions × {n_features} features")
    print(f"📈 Evaluation range: {np.min(evaluations):.3f} to {np.max(evaluations):.3f}")
    print(f"⚠️  Added {len(outlier_indices)} outliers for testing")
    
    # Test different loss functions
    loss_functions = [
        ("mse", "Mean Squared Error (L2)"),
        ("mae", "Mean Absolute Error (L1)"), 
        ("max_error", "Maximum Error (L∞)"),
        ("huber", "Huber Loss (Robust)")
    ]
    
    results = {}
    
    for loss_name, loss_desc in loss_functions:
        print(f"\n🔬 Testing {loss_desc}")
        print("-" * 40)
        
        try:
            model = train_high_complexity_sr_model(
                features, 
                evaluations, 
                feature_names,
                max_complexity=8,  # Small for quick test
                previous_equations_path=None,
                output_dir=f"outputs/loss_test_{loss_name}",
                loss_function=loss_name
            )
            
            if model:
                # Evaluate performance
                predictions = model.predict(features)
                mse = np.mean((predictions - evaluations) ** 2)
                mae = np.mean(np.abs(predictions - evaluations))
                max_error = np.max(np.abs(predictions - evaluations))
                
                results[loss_name] = {
                    'mse': mse,
                    'mae': mae, 
                    'max_error': max_error,
                    'formula': str(model.get_best())
                }
                
                print(f"✅ {loss_desc} Results:")
                print(f"   MSE: {mse:.4f}")
                print(f"   MAE: {mae:.4f}")
                print(f"   Max Error: {max_error:.4f}")
                print(f"   Formula: {str(model.get_best())[:60]}...")
            else:
                print(f"❌ {loss_desc} failed")
                
        except Exception as e:
            print(f"❌ Error with {loss_desc}: {e}")
    
    # Compare results
    if len(results) > 1:
        print(f"\n📊 LOSS FUNCTION COMPARISON")
        print("="*60)
        print(f"{'Loss Function':<15} {'MSE':<8} {'MAE':<8} {'Max Error':<10}")
        print("-" * 60)
        
        for loss_name, metrics in results.items():
            print(f"{loss_name:<15} {metrics['mse']:<8.4f} {metrics['mae']:<8.4f} {metrics['max_error']:<10.4f}")
        
        print(f"\n🎯 INSIGHTS:")
        
        # Find best for each metric
        best_mse = min(results.keys(), key=lambda k: results[k]['mse'])
        best_mae = min(results.keys(), key=lambda k: results[k]['mae'])
        best_max = min(results.keys(), key=lambda k: results[k]['max_error'])
        
        print(f"   Best MSE: {best_mse}")
        print(f"   Best MAE: {best_mae}")
        print(f"   Best Max Error: {best_max}")
        
        if best_max == 'max_error':
            print(f"✅ L∞ norm successfully minimized worst-case errors!")
        if best_mae == 'mae':
            print(f"✅ L1 norm successfully minimized average absolute errors!")

if __name__ == "__main__":
    test_loss_functions()