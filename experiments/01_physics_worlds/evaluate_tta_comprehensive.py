"""Comprehensive TTA evaluation on true OOD physics with all methods."""

import numpy as np
import pickle
from pathlib import Path
import time
import json
import os
import sys
from datetime import datetime

# Add project to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Set backend
os.environ['KERAS_BACKEND'] = 'jax'

def create_physics_model(input_steps=1, output_steps=49):
    """Create a physics prediction model."""
    import keras
    
    model = keras.Sequential([
        keras.layers.Input(shape=(input_steps, 8)),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(8 * output_steps),
        keras.layers.Reshape((output_steps, 8))
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def prepare_data(trajectories, input_steps=1, output_steps=49):
    """Prepare data for training/evaluation."""
    X, y = [], []
    
    for traj in trajectories:
        for i in range(len(traj) - input_steps - output_steps + 1):
            X.append(traj[i:i+input_steps])
            y.append(traj[i+input_steps:i+input_steps+output_steps])
    
    return np.array(X), np.array(y)

def evaluate_with_tta(model, test_trajectories, tta_config=None, n_samples=20):
    """Evaluate model with optional TTA."""
    from models.test_time_adaptation.tta_wrappers import TTAWrapper
    
    errors = []
    times = []
    physics_violations = []
    
    # Create TTA wrapper if config provided
    if tta_config:
        tta_model = TTAWrapper(model, task_type='regression', **tta_config)
    else:
        tta_model = model
    
    # Evaluate on subset of trajectories
    print(f"    Evaluating on {min(n_samples, len(test_trajectories))} trajectories...")
    for i, traj in enumerate(test_trajectories[:n_samples]):
        start_time = time.time()
        
        # Prepare input
        X = traj[0:1].reshape(1, 1, 8)
        y_true = traj[1:50]  # Next 49 steps
        
        # Skip if not enough data
        if len(y_true) < 1:
            continue
        
        # Predict
        if tta_config:
            y_pred = tta_model.predict(X, adapt=True)
            tta_model.reset()
        else:
            y_pred = model.predict(X, verbose=0)
        
        elapsed = time.time() - start_time
        times.append(elapsed)
        
        # Compute error
        y_pred = y_pred[0] if len(y_pred.shape) == 3 else y_pred
        min_len = min(len(y_true), len(y_pred))
        
        # MSE
        mse = np.mean((y_true[:min_len] - y_pred[:min_len]) ** 2)
        errors.append(mse)
        
        # Physics violation (energy conservation)
        # Simplified: check if total displacement is reasonable
        pred_displacement = np.sum(np.abs(y_pred[:, :2] - y_pred[0, :2]))
        true_displacement = np.sum(np.abs(y_true[:, :2] - y_true[0, :2]))
        violation = abs(pred_displacement - true_displacement) / true_displacement
        physics_violations.append(violation)
    
    # Handle empty results
    if not errors:
        return {
            'mse': float('inf'),
            'mse_std': 0.0,
            'mae': float('inf'),
            'time_per_sample': 0.0,
            'physics_violation': 0.0,
            'n_samples': 0
        }
    
    return {
        'mse': np.mean(errors) if errors else float('inf'),
        'mse_std': np.std(errors) if errors else 0.0,
        'mae': np.sqrt(np.mean(errors)) if errors else float('inf'),
        'time_per_sample': np.mean(times) if times else 0.0,
        'physics_violation': np.mean(physics_violations) if physics_violations else 0.0,
        'n_samples': len(errors)
    }

def main():
    try:
        import keras
        from keras import ops
        
        print("Comprehensive TTA Evaluation on True OOD Physics")
        print("="*70)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Backend: {keras.backend.backend()}")
        
        # Load data
        data_dir = Path(__file__).parent.parent.parent / "data" / "true_ood_physics"
        
        # Load datasets
        print("\nLoading data...")
        const_files = sorted(data_dir.glob("constant_gravity_*.pkl"))
        varying_files = sorted(data_dir.glob("time_varying_gravity_*.pkl"))
        
        with open(const_files[-1], 'rb') as f:
            const_data = pickle.load(f)
        with open(varying_files[-1], 'rb') as f:
            varying_data = pickle.load(f)
            
        print(f"Constant gravity: {const_data['trajectories'].shape}")
        print(f"Time-varying gravity: {varying_data['trajectories'].shape}")
        
        # Also try to load other OOD types if available
        ood_datasets = {}
        for ood_type in ['rotating_frame', 'spring_coupled', 'height_dependent']:
            files = sorted(data_dir.glob(f"{ood_type}_physics_*.pkl"))
            if files:
                with open(files[-1], 'rb') as f:
                    ood_datasets[ood_type] = pickle.load(f)
                print(f"{ood_type}: {ood_datasets[ood_type]['trajectories'].shape}")
        
        # Create and train model
        print("\nCreating and training model...")
        model = create_physics_model()
        
        # Prepare training data
        X_train, y_train = prepare_data(const_data['trajectories'][:80])
        X_val, y_val = prepare_data(const_data['trajectories'][80:100])
        
        print(f"Training data: X={X_train.shape}, y={y_train.shape}")
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=30,
            batch_size=32,
            verbose=1,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5)
            ]
        )
        
        print(f"\nFinal training loss: {history.history['loss'][-1]:.4f}")
        print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
        
        # Define TTA configurations
        tta_configs = {
            'No TTA': None,
            'TENT': {
                'tta_method': 'tent',
                'adaptation_steps': 5,
                'learning_rate': 1e-4
            },
            'PhysicsTENT': {
                'tta_method': 'physics_tent',
                'adaptation_steps': 5,
                'learning_rate': 1e-4,
                'physics_loss_weight': 0.1
            },
            'TTT': {
                'tta_method': 'ttt',
                'adaptation_steps': 10,
                'learning_rate': 5e-5,
                'auxiliary_weight': 0.3
            }
        }
        
        # Evaluate on all datasets
        results = {}
        
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        
        # Baseline: constant gravity
        print("\nEvaluating on constant gravity (in-distribution)...")
        const_test = const_data['trajectories'][100:]
        baseline_results = evaluate_with_tta(model, const_test, None)
        print(f"Baseline MSE: {baseline_results['mse']:.4f}")
        
        # Time-varying gravity with different TTA methods
        print("\nEvaluating on time-varying gravity...")
        varying_results = {}
        
        for method_name, config in tta_configs.items():
            print(f"\n  Testing {method_name}...")
            result = evaluate_with_tta(model, varying_data['trajectories'], config)
            varying_results[method_name] = result
            
            print(f"    MSE: {result['mse']:.4f} (±{result['mse_std']:.4f})")
            print(f"    Time: {result['time_per_sample']:.3f}s")
            
            if method_name != 'No TTA':
                improvement = (varying_results['No TTA']['mse'] - result['mse']) / varying_results['No TTA']['mse'] * 100
                print(f"    Improvement: {improvement:+.1f}%")
        
        results['constant_gravity'] = baseline_results
        results['time_varying_gravity'] = varying_results
        
        # Evaluate on other OOD types if available
        for ood_type, ood_data in ood_datasets.items():
            print(f"\nEvaluating on {ood_type}...")
            ood_results = {}
            
            for method_name, config in tta_configs.items():
                if method_name in ['No TTA', 'PhysicsTENT']:  # Test subset for speed
                    print(f"\n  Testing {method_name}...")
                    result = evaluate_with_tta(model, ood_data['trajectories'], config, n_samples=10)
                    ood_results[method_name] = result
                    print(f"    MSE: {result['mse']:.4f}")
            
            results[ood_type] = ood_results
        
        # Generate summary table
        print("\n" + "="*70)
        print("SUMMARY TABLE")
        print("="*70)
        print(f"{'Method':<15} {'Const Grav':<12} {'Time-Vary':<12} {'Improvement':<12}")
        print("-" * 51)
        
        # Baseline
        print(f"{'Baseline':<15} {baseline_results['mse']:>10.4f}  {'-':<12} {'-':<12}")
        
        # TTA methods on time-varying
        for method in ['No TTA', 'TENT', 'PhysicsTENT', 'TTT']:
            if method in varying_results:
                mse = varying_results[method]['mse']
                degradation = (mse - baseline_results['mse']) / baseline_results['mse'] * 100
                
                if method != 'No TTA':
                    improvement = (varying_results['No TTA']['mse'] - mse) / varying_results['No TTA']['mse'] * 100
                    imp_str = f"{improvement:+.1f}%"
                else:
                    imp_str = "-"
                
                print(f"{method:<15} {'-':<12} {mse:>10.4f}  {imp_str:<12}")
        
        # Best method
        best_method = min(varying_results.items(), key=lambda x: x[1]['mse'])
        print(f"\nBest method: {best_method[0]}")
        
        # Save results
        output_dir = Path(__file__).parent / "outputs" / "tta_evaluation"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # Save detailed results
        serializable_results = convert_to_serializable(results)
        with open(output_dir / f"tta_results_{timestamp}.json", 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Save summary
        summary = {
            'timestamp': timestamp,
            'baseline_mse': baseline_results['mse'],
            'ood_degradation': (varying_results['No TTA']['mse'] - baseline_results['mse']) / baseline_results['mse'] * 100,
            'best_tta_method': best_method[0],
            'best_tta_improvement': (varying_results['No TTA']['mse'] - best_method[1]['mse']) / varying_results['No TTA']['mse'] * 100,
            'methods_tested': list(tta_configs.keys())
        }
        
        with open(output_dir / f"tta_summary_{timestamp}.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nResults saved to: {output_dir}")
        
        # Final assessment
        print("\n" + "="*70)
        if any(varying_results[m]['mse'] < varying_results['No TTA']['mse'] for m in tta_configs if m != 'No TTA'):
            print("✓ SUCCESS: TTA improves performance on true OOD scenarios!")
            print(f"  Best improvement: {summary['best_tta_improvement']:.1f}% with {best_method[0]}")
        else:
            print("✗ TTA did not improve performance in this evaluation")
            
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()