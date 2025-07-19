"""Test regression-specific TTA methods on physics data."""

import numpy as np
import pickle
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.imports import setup_project_paths
setup_project_paths()

from utils.config import setup_environment
from utils.paths import get_data_path
import keras
from models.test_time_adaptation.tta_wrappers import TTAWrapper


def create_physics_model():
    """Create a physics prediction model."""
    model = keras.Sequential([
        keras.layers.Input(shape=(8,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(8)
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )
    return model


def main():
    """Test regression-specific TTA."""
    print("Testing Regression-Specific TTA Methods")
    print("="*60)
    
    config = setup_environment()
    
    # Load data
    data_dir = get_data_path() / "true_ood_physics"
    
    # Load constant gravity
    const_files = sorted(data_dir.glob("constant_gravity_*.pkl"))
    with open(const_files[-1], 'rb') as f:
        const_data = pickle.load(f)
    
    # Load time-varying gravity  
    varying_files = sorted(data_dir.glob("time_varying_gravity_*.pkl"))
    with open(varying_files[-1], 'rb') as f:
        ood_data = pickle.load(f)
    
    print(f"Loaded {len(const_data['trajectories'])} constant gravity trajectories")
    print(f"Loaded {len(ood_data['trajectories'])} time-varying gravity trajectories")
    
    # Prepare training data
    X_train = []
    y_train = []
    for traj in const_data['trajectories'][:60]:
        for i in range(len(traj) - 1):
            X_train.append(traj[i])
            y_train.append(traj[i+1])
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Train model
    print("\nTraining model...")
    model = create_physics_model()
    model.fit(X_train, y_train, epochs=15, batch_size=64, verbose=0)
    
    # Baseline evaluation
    print("\nBaseline evaluation (no TTA):")
    
    # Constant gravity
    X_test_const = []
    y_test_const = []
    for traj in const_data['trajectories'][80:85]:
        for i in range(len(traj) - 1):
            X_test_const.append(traj[i])
            y_test_const.append(traj[i+1])
    X_test_const = np.array(X_test_const)
    y_test_const = np.array(y_test_const)
    
    mse_const = model.evaluate(X_test_const, y_test_const, verbose=0)[0]
    print(f"  Constant gravity MSE: {mse_const:.4f}")
    
    # Time-varying gravity
    X_test_ood = []
    y_test_ood = []
    for traj in ood_data['trajectories'][:5]:
        for i in range(len(traj) - 1):
            X_test_ood.append(traj[i])
            y_test_ood.append(traj[i+1])
    X_test_ood = np.array(X_test_ood)
    y_test_ood = np.array(y_test_ood)
    
    mse_ood_baseline = model.evaluate(X_test_ood, y_test_ood, verbose=0)[0]
    print(f"  Time-varying gravity MSE: {mse_ood_baseline:.4f}")
    print(f"  Degradation: {mse_ood_baseline/mse_const:.2f}x")
    
    # Test different TTA methods
    print("\n" + "="*60)
    print("TESTING TTA METHODS")
    print("="*60)
    
    tta_configs = [
        # Original TENT (for comparison)
        {'method': 'tent', 'lr': 1e-4, 'steps': 5, 'name': 'TENT (original)'},
        
        # Regression-specific TTA
        {'method': 'regression', 'lr': 1e-3, 'steps': 5, 'name': 'RegressionTTA (lr=1e-3)'},
        {'method': 'regression', 'lr': 5e-4, 'steps': 5, 'name': 'RegressionTTA (lr=5e-4)'},
        {'method': 'regression', 'lr': 1e-4, 'steps': 5, 'name': 'RegressionTTA (lr=1e-4)'},
        {'method': 'regression', 'lr': 1e-4, 'steps': 10, 'name': 'RegressionTTA (10 steps)'},
        
        # Physics-aware regression TTA
        {'method': 'physics_regression', 'lr': 5e-4, 'steps': 5, 'name': 'PhysicsRegressionTTA'},
        {'method': 'physics_regression', 'lr': 1e-4, 'steps': 10, 'name': 'PhysicsRegressionTTA (10 steps)'},
    ]
    
    results = []
    
    for config in tta_configs:
        print(f"\nTesting {config['name']}:")
        
        # Create TTA wrapper
        tta_kwargs = {
            'adaptation_steps': config['steps'],
            'learning_rate': config['lr'],
            'update_bn_only': True
        }
        
        if config['method'] == 'physics_regression':
            tta_kwargs['physics_weight'] = 0.5
        
        tta_model = TTAWrapper(model, tta_method=config['method'], **tta_kwargs)
        
        # Test on OOD data batch by batch
        batch_errors = []
        batch_size = 10
        
        for i in range(0, len(X_test_ood), batch_size):
            X_batch = X_test_ood[i:i+batch_size]
            y_batch = y_test_ood[i:i+batch_size]
            
            # Adapt and predict
            y_pred = tta_model.predict(X_batch, adapt=True)
            error = np.mean((y_pred - y_batch)**2)
            batch_errors.append(error)
            
            # Reset for next batch
            tta_model.reset()
        
        mean_mse = np.mean(batch_errors)
        improvement = (1 - mean_mse/mse_ood_baseline) * 100
        
        print(f"  MSE: {mean_mse:.4f}")
        print(f"  Improvement: {improvement:+.1f}%")
        
        results.append({
            'name': config['name'],
            'mse': mean_mse,
            'improvement': improvement
        })
    
    # Weight restoration check
    print("\n" + "="*60)
    print("WEIGHT RESTORATION CHECK")
    print("="*60)
    
    # Test with regression TTA
    test_input = X_test_ood[:1]
    pred_before = model.predict(test_input, verbose=0)
    
    tta_model = TTAWrapper(model, tta_method='regression', learning_rate=1e-3, adaptation_steps=10)
    _ = tta_model.predict(X_test_ood[:10], adapt=True)
    tta_model.reset()
    
    pred_after = model.predict(test_input, verbose=0)
    restoration_error = np.mean(np.abs(pred_before - pred_after))
    
    print(f"Restoration error: {restoration_error:.6f}")
    print(f"Weight restoration: {'✓ WORKING' if restoration_error < 1e-3 else '✗ FAILED'}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    # Sort by improvement
    results.sort(key=lambda x: x['improvement'], reverse=True)
    
    print("\nTop 3 configurations:")
    for i, result in enumerate(results[:3]):
        print(f"{i+1}. {result['name']}")
        print(f"   MSE: {result['mse']:.4f}, Improvement: {result['improvement']:+.1f}%")
    
    if results[0]['improvement'] > 0:
        print(f"\n✓ TTA is showing improvement on OOD data!")
    else:
        print(f"\n✗ TTA needs more tuning for this task")


if __name__ == "__main__":
    main()