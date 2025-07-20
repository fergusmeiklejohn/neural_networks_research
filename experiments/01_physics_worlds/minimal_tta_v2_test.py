"""Minimal TTA V2 test to quickly identify if any configuration works."""

import numpy as np
import pickle
from pathlib import Path
import sys
import json
from datetime import datetime
import keras

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.imports import setup_project_paths
setup_project_paths()

from utils.config import setup_environment
from utils.paths import get_data_path, get_output_path
from models.test_time_adaptation.tta_wrappers import TTAWrapper


def create_physics_model(input_steps=1, output_steps=10):
    """Create a physics prediction model."""
    model = keras.Sequential([
        keras.layers.Input(shape=(input_steps, 8)),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
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


def prepare_data(trajectories, input_steps=1, output_steps=10):
    """Prepare trajectory data for training/evaluation."""
    X, y = [], []
    for traj in trajectories:
        for i in range(len(traj) - input_steps - output_steps + 1):
            X.append(traj[i:i+input_steps])
            y.append(traj[i+input_steps:i+input_steps+output_steps])
    return np.array(X), np.array(y)


def test_tta_config(model, trajectories, config_name, **tta_kwargs):
    """Test a single TTA configuration."""
    print(f"\n--- Testing: {config_name} ---")
    
    # Create TTA wrapper
    tta_model = TTAWrapper(model, **tta_kwargs)
    
    # Test on 5 trajectories
    errors = []
    for i, traj in enumerate(trajectories[:5]):
        X = traj[0:1].reshape(1, 1, 8)
        y_true = traj[1:11]
        
        # Adapt and predict
        y_pred = tta_model.predict(X, adapt=True)
        if len(y_pred.shape) == 3:
            y_pred = y_pred[0]
        
        # Compute error
        mse = np.mean((y_true[:len(y_pred)] - y_pred)**2)
        errors.append(mse)
        
        if i == 0:
            print(f"  First trajectory MSE: {mse:.4f}")
        
        # Reset model
        tta_model.reset()
    
    mean_mse = np.mean(errors)
    std_mse = np.std(errors)
    print(f"  Mean MSE: {mean_mse:.4f} ± {std_mse:.4f}")
    
    return mean_mse, std_mse


def main():
    """Run minimal TTA V2 test."""
    print("Minimal TTA V2 Test")
    print("="*70)
    
    # Setup
    config = setup_environment()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load data
    data_dir = get_data_path() / "true_ood_physics"
    
    print("Loading data...")
    const_files = sorted(data_dir.glob("constant_gravity_*.pkl"))
    with open(const_files[-1], 'rb') as f:
        const_data = pickle.load(f)
    
    varying_files = sorted(data_dir.glob("time_varying_gravity_*.pkl"))
    with open(varying_files[-1], 'rb') as f:
        ood_data = pickle.load(f)
    
    print(f"Loaded {len(const_data['trajectories'])} constant gravity trajectories")
    print(f"Loaded {len(ood_data['trajectories'])} time-varying gravity trajectories")
    
    # Train minimal model
    print("\nTraining base model (minimal)...")
    model = create_physics_model()
    
    X_train, y_train = prepare_data(const_data['trajectories'][:30])
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
    
    # Baseline
    print("\nBaseline evaluation...")
    X_test, y_test = prepare_data(ood_data['trajectories'][:5])
    baseline_mse = model.evaluate(X_test, y_test, verbose=0)[0]
    print(f"Baseline OOD MSE (no TTA): {baseline_mse:.4f}")
    
    # Test key configurations
    results = {}
    
    # 1. Minimal adaptation (1 step, very low lr)
    mse, std = test_tta_config(
        model, ood_data['trajectories'],
        "Minimal adaptation (1 step, lr=1e-8)",
        tta_method='regression_v2',
        adaptation_steps=1,
        learning_rate=1e-8,
        consistency_loss_weight=0.0,
        smoothness_loss_weight=0.0
    )
    results['minimal_adapt'] = {'mse': mse, 'std': std}
    
    # 2. Very low learning rate with BN only
    mse, std = test_tta_config(
        model, ood_data['trajectories'],
        "BN-only, lr=1e-7, steps=10",
        tta_method='regression_v2',
        adaptation_steps=10,
        learning_rate=1e-7,
        bn_only_mode=True,
        consistency_loss_weight=0.0,
        smoothness_loss_weight=0.0
    )
    results['bn_only_minimal'] = {'mse': mse, 'std': std}
    
    # 3. Extremely low learning rate
    mse, std = test_tta_config(
        model, ood_data['trajectories'],
        "All params, lr=1e-8, steps=5",
        tta_method='regression_v2',
        adaptation_steps=5,
        learning_rate=1e-8,
        bn_only_mode=False,
        consistency_loss_weight=0.1,
        smoothness_loss_weight=0.05
    )
    results['ultra_low_lr'] = {'mse': mse, 'std': std}
    
    # 4. Single step adaptation
    mse, std = test_tta_config(
        model, ood_data['trajectories'],
        "All params, lr=1e-6, single step",
        tta_method='regression_v2',
        adaptation_steps=1,
        learning_rate=1e-6,
        bn_only_mode=False,
        consistency_loss_weight=0.0,
        smoothness_loss_weight=0.0
    )
    results['single_step'] = {'mse': mse, 'std': std}
    
    # 5. Physics-informed with minimal weights
    mse, std = test_tta_config(
        model, ood_data['trajectories'],
        "Physics V2, lr=1e-7, minimal weights",
        tta_method='physics_regression_v2',
        adaptation_steps=10,
        learning_rate=1e-7,
        bn_only_mode=False,
        consistency_loss_weight=0.01,
        smoothness_loss_weight=0.01,
        energy_loss_weight=0.01,
        momentum_loss_weight=0.01
    )
    results['physics_minimal'] = {'mse': mse, 'std': std}
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Baseline OOD MSE: {baseline_mse:.4f}")
    
    best_config = min(results.items(), key=lambda x: x[1]['mse'])
    print(f"\nBest configuration: {best_config[0]}")
    print(f"  MSE: {best_config[1]['mse']:.4f}")
    print(f"  Improvement: {(1 - best_config[1]['mse']/baseline_mse)*100:.1f}%")
    
    # Save results
    output_dir = get_output_path() / "tta_tuning"
    output_dir.mkdir(exist_ok=True)
    results_file = output_dir / f"minimal_tta_v2_test_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'baseline_mse': float(baseline_mse),
            'results': {k: {'mse': float(v['mse']), 'std': float(v['std'])} for k, v in results.items()},
            'best_config': best_config[0],
            'best_mse': float(best_config[1]['mse'])
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Analysis
    if best_config[1]['mse'] < baseline_mse:
        print("\n✓ Found configuration with positive improvement!")
    else:
        print("\n✗ No configuration improved over baseline")
        print("\nPossible issues:")
        print("- TTA losses may be inappropriate for this task")
        print("- Model may not have learned adaptable features")
        print("- True OOD may be too different for adaptation")
        print("- Need to investigate the actual adaptation behavior")


if __name__ == "__main__":
    main()