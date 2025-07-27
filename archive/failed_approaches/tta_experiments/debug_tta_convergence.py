"""Debug why all TTA methods converge to the same MSE value."""

import numpy as np
import pickle
from pathlib import Path
import os
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
os.environ['KERAS_BACKEND'] = 'jax'

import keras
from models.test_time_adaptation.tta_wrappers import TTAWrapper


def analyze_model_weights(model, name="Model"):
    """Analyze model weights statistics."""
    print(f"\n{name} Weight Analysis:")
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'weights') and layer.weights:
            for weight in layer.weights:
                w = weight.numpy()
                print(f"  Layer {i} ({layer.name}) - {weight.name}:")
                print(f"    Shape: {w.shape}, Mean: {np.mean(w):.6f}, Std: {np.std(w):.6f}")
                if 'batch_normalization' in layer.name:
                    print(f"    Min: {np.min(w):.6f}, Max: {np.max(w):.6f}")


def track_adaptation_changes(model, test_data, tta_method='tent', n_steps=5):
    """Track how weights change during adaptation."""
    print(f"\n\nTracking {tta_method.upper()} adaptation over {n_steps} steps...")
    
    # Create TTA wrapper
    tta_model = TTAWrapper(
        model,
        tta_method=tta_method,
        adaptation_steps=n_steps,
        learning_rate=1e-4
    )
    
    # Get initial predictions
    X = test_data[0:1].reshape(1, 1, 8)
    y_initial = model.predict(X, verbose=0)
    
    # Track predictions at each adaptation step
    predictions_per_step = []
    losses_per_step = []
    
    # Manually step through adaptation
    adapter = tta_model.tta_adapter
    
    # Do one forward pass to initialize shapes
    _ = model.predict(X, verbose=0)
    
    # Store initial weights after initialization
    initial_weights = {}
    for layer in model.layers:
        if 'batch_normalization' in layer.name:
            for weight in layer.weights:
                initial_weights[weight.name] = weight.numpy().copy()
    
    # Adapt step by step
    print("\nAdaptation progress:")
    for step in range(n_steps):
        # Get prediction before this step
        y_pred = model.predict(X, verbose=0)
        predictions_per_step.append(y_pred)
        
        # Perform one adaptation step
        if hasattr(adapter, 'adapt_step_simple'):
            _, loss = adapter.adapt_step_simple(X)
        else:
            _, loss = adapter.adapt_step(X)
        
        losses_per_step.append(float(loss))
        print(f"  Step {step+1}: Loss = {loss:.6f}")
        
        # Check weight changes
        for layer in model.layers:
            if 'batch_normalization' in layer.name:
                for weight in layer.weights:
                    if weight.name in initial_weights:
                        current = weight.numpy()
                        initial = initial_weights[weight.name]
                        # Handle shape mismatches
                        if current.shape != initial.shape:
                            print(f"    {weight.name} shape changed: {initial.shape} -> {current.shape}")
                        else:
                            change = np.mean(np.abs(current - initial))
                            if change > 1e-6:
                                print(f"    {weight.name} changed by {change:.6f}")
    
    # Final prediction
    y_final = model.predict(X, verbose=0)
    
    # Analyze prediction changes
    print(f"\nPrediction changes:")
    print(f"  Initial prediction mean: {np.mean(y_initial):.6f}")
    print(f"  Final prediction mean: {np.mean(y_final):.6f}")
    print(f"  Total change: {np.mean(np.abs(y_final - y_initial)):.6f}")
    
    # Check if predictions actually changed
    all_same = all(np.allclose(predictions_per_step[0], pred) for pred in predictions_per_step[1:])
    if all_same:
        print("  WARNING: Predictions did not change during adaptation!")
    
    return {
        'initial': y_initial,
        'final': y_final,
        'steps': predictions_per_step,
        'losses': losses_per_step,
        'predictions_changed': not all_same
    }


def main():
    print("TTA Convergence Debugging")
    print("="*60)
    
    # Load test data
    data_dir = Path(__file__).parent.parent.parent / "data" / "true_ood_physics"
    varying_files = sorted(data_dir.glob("time_varying_gravity_*.pkl"))
    
    with open(varying_files[-1], 'rb') as f:
        varying_data = pickle.load(f)
    
    print(f"Loaded time-varying gravity data: {varying_data['trajectories'].shape}")
    
    # Create a simple model
    print("\nCreating test model...")
    model = keras.Sequential([
        keras.layers.Input(shape=(1, 8)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(8 * 10),
        keras.layers.Reshape((10, 8))
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    # Quick training on constant gravity
    print("\nTraining on constant gravity...")
    const_files = sorted(data_dir.glob("constant_gravity_*.pkl"))
    with open(const_files[-1], 'rb') as f:
        const_data = pickle.load(f)
    
    X_train = const_data['trajectories'][:50, 0:1]
    y_train = const_data['trajectories'][:50, 1:11]
    
    model.fit(X_train, y_train, epochs=10, verbose=0)
    
    # Analyze initial model
    analyze_model_weights(model, "Initial Model")
    
    # Test each TTA method
    test_trajectory = varying_data['trajectories'][0]
    
    methods = ['tent', 'physics_tent', 'ttt']
    results = {}
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Testing {method.upper()}")
        print('='*60)
        
        # Create fresh model copy
        model_copy = keras.models.clone_model(model)
        model_copy.set_weights(model.get_weights())
        model_copy.compile(optimizer='adam', loss='mse')
        
        # Track adaptation
        result = track_adaptation_changes(model_copy, test_trajectory, method, n_steps=10)
        results[method] = result
        
        # Analyze final model
        analyze_model_weights(model_copy, f"After {method.upper()}")
    
    # Compare results
    print("\n" + "="*60)
    print("COMPARISON OF TTA METHODS")
    print("="*60)
    
    # Check if all methods produced same final prediction
    final_preds = [results[m]['final'] for m in methods]
    all_same = all(np.allclose(final_preds[0], pred) for pred in final_preds[1:])
    
    if all_same:
        print("\nWARNING: All TTA methods produced identical final predictions!")
        print("This explains why they all have the same MSE.")
    
    # Check adaptation effectiveness
    for method in methods:
        print(f"\n{method.upper()}:")
        print(f"  Predictions changed: {results[method]['predictions_changed']}")
        print(f"  Loss trajectory: {results[method]['losses'][:5]}...")
        print(f"  Initial vs Final prediction diff: {np.mean(np.abs(results[method]['final'] - results[method]['initial'])):.6f}")
    
    # Additional analysis
    print("\n" + "="*60)
    print("HYPOTHESIS TESTING")
    print("="*60)
    
    print("\n1. Are BatchNorm statistics being updated?")
    # This would need deeper inspection of BN running mean/variance
    
    print("\n2. Is the learning rate too small?")
    print("   Current LR: 1e-4")
    print("   Try: 1e-3, 1e-2 for stronger adaptation")
    
    print("\n3. Is single timestep input limiting?")
    print("   Current: 1 timestep → predict 10")
    print("   Try: 5 timesteps → predict 10")
    
    print("\n4. Are we in a local minimum?")
    print("   All methods converging to same value suggests yes")
    
    print("\nRecommendations:")
    print("1. Increase learning rate (1e-3 or 1e-2)")
    print("2. Use multi-timestep inputs")
    print("3. Add gradient clipping to prevent instability")
    print("4. Try different optimizers (SGD with momentum)")


if __name__ == "__main__":
    main()