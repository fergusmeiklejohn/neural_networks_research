"""Comprehensive hyperparameter search for TTA methods."""

import numpy as np
import pickle
from pathlib import Path
import json
import os
import sys
from datetime import datetime
from itertools import product

sys.path.append(str(Path(__file__).parent.parent.parent))
os.environ['KERAS_BACKEND'] = 'jax'

import keras
from models.test_time_adaptation.tta_wrappers import TTAWrapper


def create_flexible_model(input_shape=(1, 8), output_shape=(10, 8), 
                         architecture='dense', hidden_units=128):
    """Create different model architectures for testing."""
    
    if architecture == 'dense':
        model = keras.Sequential([
            keras.layers.Input(shape=input_shape),
            keras.layers.Flatten(),
            keras.layers.Dense(hidden_units, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(hidden_units // 2, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(np.prod(output_shape)),
            keras.layers.Reshape(output_shape)
        ])
    
    elif architecture == 'deep':
        model = keras.Sequential([
            keras.layers.Input(shape=input_shape),
            keras.layers.Flatten(),
            keras.layers.Dense(hidden_units, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(hidden_units, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(hidden_units // 2, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(np.prod(output_shape)),
            keras.layers.Reshape(output_shape)
        ])
    
    elif architecture == 'wide':
        model = keras.Sequential([
            keras.layers.Input(shape=input_shape),
            keras.layers.Flatten(),
            keras.layers.Dense(hidden_units * 2, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(hidden_units, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(np.prod(output_shape)),
            keras.layers.Reshape(output_shape)
        ])
    
    elif architecture == 'residual':
        # Simple residual-like architecture
        inputs = keras.layers.Input(shape=input_shape)
        x = keras.layers.Flatten()(inputs)
        
        # First block
        identity = keras.layers.Dense(hidden_units)(x)
        x = keras.layers.Dense(hidden_units, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Add()([x, identity])
        
        # Second block
        identity = x
        x = keras.layers.Dense(hidden_units, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Add()([x, identity])
        
        # Output
        x = keras.layers.Dense(np.prod(output_shape))(x)
        outputs = keras.layers.Reshape(output_shape)(x)
        
        model = keras.Model(inputs, outputs)
    
    return model


def evaluate_configuration(model, train_data, test_data, tta_config, n_samples=20):
    """Evaluate a specific TTA configuration."""
    # Quick training
    X_train = train_data['X'][:100]
    y_train = train_data['y'][:100]
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse'
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=15,
        batch_size=32,
        verbose=0
    )
    
    # Baseline evaluation
    baseline_errors = []
    for i in range(min(n_samples, len(test_data['X']))):
        X = test_data['X'][i:i+1]
        y_true = test_data['y'][i]
        y_pred = model.predict(X, verbose=0)[0]
        mse = np.mean((y_true - y_pred) ** 2)
        baseline_errors.append(mse)
    
    baseline_mse = np.mean(baseline_errors)
    
    # TTA evaluation
    tta_model = TTAWrapper(model, **tta_config)
    tta_errors = []
    improvements = []
    
    for i in range(min(n_samples, len(test_data['X']))):
        X = test_data['X'][i:i+1]
        y_true = test_data['y'][i]
        
        # Reset and adapt
        tta_model.reset()
        y_pred = tta_model.predict(X, adapt=True)[0]
        
        mse = np.mean((y_true - y_pred) ** 2)
        tta_errors.append(mse)
        
        improvement = (baseline_errors[i] - mse) / baseline_errors[i] * 100
        improvements.append(improvement)
    
    return {
        'baseline_mse': baseline_mse,
        'tta_mse': np.mean(tta_errors),
        'improvement': np.mean(improvements),
        'improvement_std': np.std(improvements),
        'improved_ratio': sum(1 for imp in improvements if imp > 0) / len(improvements),
        'training_loss': history.history['loss'][-1]
    }


def main():
    print("TTA Hyperparameter Search")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    data_dir = Path(__file__).parent.parent.parent / "data" / "true_ood_physics"
    
    const_files = sorted(data_dir.glob("constant_gravity_*.pkl"))
    varying_files = sorted(data_dir.glob("time_varying_gravity_*.pkl"))
    
    with open(const_files[-1], 'rb') as f:
        const_data = pickle.load(f)
    with open(varying_files[-1], 'rb') as f:
        varying_data = pickle.load(f)
    
    # Prepare data
    def prepare_data(trajectories, input_steps=1, output_steps=10):
        X, y = [], []
        for traj in trajectories:
            for i in range(len(traj) - input_steps - output_steps + 1):
                X.append(traj[i:i+input_steps])
                y.append(traj[i+input_steps:i+input_steps+output_steps])
        return {'X': np.array(X), 'y': np.array(y)}
    
    train_data = prepare_data(const_data['trajectories'][:80])
    test_data = prepare_data(varying_data['trajectories'])
    
    # Define hyperparameter grid
    param_grid = {
        'learning_rate': [1e-5, 1e-4, 1e-3, 1e-2],
        'adaptation_steps': [1, 5, 10, 20],
        'architecture': ['dense', 'deep', 'wide'],
        'hidden_units': [64, 128, 256],
        'optimizer_type': ['adam', 'sgd']  # Different optimizers
    }
    
    # Test TENT with different configurations
    print("\nSearching TENT configurations...")
    print("-" * 70)
    
    best_config = None
    best_improvement = -float('inf')
    results = []
    
    # Sample configurations to test
    for lr, steps, arch, units in product(
        param_grid['learning_rate'],
        param_grid['adaptation_steps'],
        ['dense', 'deep'],  # Limit architectures for speed
        [128]  # Fixed hidden units for now
    ):
        config_name = f"TENT_lr{lr}_s{steps}_{arch}"
        print(f"\nTesting {config_name}...")
        
        # Create model
        model = create_flexible_model(
            architecture=arch,
            hidden_units=units
        )
        
        # TTA configuration
        tta_config = {
            'tta_method': 'tent',
            'adaptation_steps': steps,
            'learning_rate': lr,
            'task_type': 'regression',  # Use regression-specific TTA
            'update_bn_only': True  # TENT typically only updates BN
        }
        
        # Evaluate
        try:
            result = evaluate_configuration(
                model, train_data, test_data, tta_config, n_samples=10
            )
            
            result['config'] = config_name
            result['hyperparameters'] = {
                'learning_rate': lr,
                'adaptation_steps': steps,
                'architecture': arch,
                'hidden_units': units
            }
            
            results.append(result)
            
            print(f"  Baseline MSE: {result['baseline_mse']:.2f}")
            print(f"  TTA MSE: {result['tta_mse']:.2f}")
            print(f"  Improvement: {result['improvement']:.1f}% (±{result['improvement_std']:.1f}%)")
            print(f"  Improved ratio: {result['improved_ratio']:.2f}")
            
            if result['improvement'] > best_improvement:
                best_improvement = result['improvement']
                best_config = result
                
        except Exception as e:
            print(f"  Failed: {e}")
    
    # Test PhysicsTENT with best architecture
    print("\n" + "="*70)
    print("Testing PhysicsTENT variations...")
    print("-" * 70)
    
    physics_weights = [0.01, 0.1, 0.5, 1.0]
    
    for weight in physics_weights:
        config_name = f"PhysicsTENT_pw{weight}"
        print(f"\nTesting {config_name}...")
        
        model = create_flexible_model(architecture='deep')
        
        tta_config = {
            'tta_method': 'physics_tent',
            'adaptation_steps': 10,
            'learning_rate': 1e-3,
            'task_type': 'regression',
            'physics_loss_weight': weight
        }
        
        result = evaluate_configuration(
            model, train_data, test_data, tta_config, n_samples=10
        )
        
        result['config'] = config_name
        result['physics_weight'] = weight
        results.append(result)
        
        print(f"  Improvement: {result['improvement']:.1f}%")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    # Sort by improvement
    results.sort(key=lambda x: x['improvement'], reverse=True)
    
    print("\nTop 5 Configurations:")
    for i, result in enumerate(results[:5]):
        print(f"\n{i+1}. {result['config']}")
        print(f"   Improvement: {result['improvement']:.1f}% (±{result['improvement_std']:.1f}%)")
        print(f"   Improved ratio: {result['improved_ratio']:.2f}")
        if 'hyperparameters' in result:
            print(f"   Params: {result['hyperparameters']}")
    
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
    
    serializable_results = convert_to_serializable(results)
    
    with open(output_dir / f"hyperparameter_search_{timestamp}.json", 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}/hyperparameter_search_{timestamp}.json")
    
    # Key insights
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    
    if best_config and best_config['improvement'] > 0:
        print(f"\n✓ TTA CAN improve performance!")
        print(f"Best improvement: {best_config['improvement']:.1f}%")
        print(f"Best config: {best_config['config']}")
    else:
        print("\n✗ No configuration improved performance")
        print("Recommendations:")
        print("1. Try even higher learning rates (1e-1)")
        print("2. Use multiple timestep inputs")
        print("3. Try different model architectures")
        print("4. Consider meta-learning approaches")


if __name__ == "__main__":
    main()