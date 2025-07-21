"""Comprehensive TTA debugging script to understand adaptation behavior."""

import numpy as np
import pickle
from pathlib import Path
import sys
import json
from datetime import datetime
import keras
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.imports import setup_project_paths
setup_project_paths()

from utils.config import setup_environment
from utils.paths import get_data_path, get_output_path
from models.test_time_adaptation.tta_wrappers import TTAWrapper
from models.test_time_adaptation.regression_tta_v2 import RegressionTTAV2


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


def track_parameter_changes(model_before, model_after):
    """Track how parameters changed during adaptation."""
    changes = {}
    
    for i, (var_before, var_after) in enumerate(zip(model_before.variables, model_after.variables)):
        # Convert to numpy for analysis
        before_np = np.array(var_before)
        after_np = np.array(var_after)
        
        # Calculate statistics
        diff = after_np - before_np
        abs_diff = np.abs(diff)
        rel_diff = np.where(np.abs(before_np) > 1e-10, 
                           np.abs(diff) / (np.abs(before_np) + 1e-10), 
                           abs_diff)
        
        layer_name = var_before.path if hasattr(var_before, 'path') else f"var_{i}"
        
        changes[layer_name] = {
            'mean_abs_change': float(np.mean(abs_diff)),
            'max_abs_change': float(np.max(abs_diff)),
            'mean_rel_change': float(np.mean(rel_diff)),
            'max_rel_change': float(np.max(rel_diff)),
            'std_change': float(np.std(diff)),
            'num_changed': int(np.sum(abs_diff > 1e-10)),
            'total_params': int(before_np.size),
            'percent_changed': float(100 * np.sum(abs_diff > 1e-10) / before_np.size)
        }
    
    return changes


def debug_single_adaptation(model, trajectory, tta_kwargs, debug_info):
    """Debug a single TTA adaptation in detail."""
    print("\n" + "="*70)
    print(f"Debugging TTA: {debug_info['name']}")
    print("="*70)
    
    # Prepare single trajectory
    X = trajectory[0:1].reshape(1, 1, 8)
    y_true = trajectory[1:11]
    
    # Get baseline prediction (no adaptation)
    y_pred_baseline = model.predict(X, verbose=0)
    if len(y_pred_baseline.shape) == 3:
        y_pred_baseline = y_pred_baseline[0]
    baseline_mse = np.mean((y_true[:len(y_pred_baseline)] - y_pred_baseline)**2)
    
    print(f"\nBaseline MSE (no adaptation): {baseline_mse:.4f}")
    
    # Save original weights
    original_weights = [np.array(var) for var in model.variables]
    
    # Create TTA wrapper and adapt
    tta_model = TTAWrapper(model, **tta_kwargs)
    
    # Adapt the model
    y_pred_adapted = tta_model.predict(X, adapt=True)
    
    # Get adapted prediction
    y_pred_adapted = model.predict(X, verbose=0)
    if len(y_pred_adapted.shape) == 3:
        y_pred_adapted = y_pred_adapted[0]
    adapted_mse = np.mean((y_true[:len(y_pred_adapted)] - y_pred_adapted)**2)
    
    print(f"\nAdapted MSE: {adapted_mse:.4f}")
    print(f"Change in MSE: {adapted_mse - baseline_mse:.4f} ({(adapted_mse/baseline_mse - 1)*100:.1f}%)")
    
    # Track parameter changes
    param_changes = track_parameter_changes(
        type(model).from_config(model.get_config()),  # Create dummy model with original weights
        model
    )
    
    # Restore original weights for comparison
    for var, orig_val in zip(model.variables, original_weights):
        var.assign(orig_val)
    
    # Analyze parameter changes
    print("\nParameter Change Summary:")
    total_params = sum(info['total_params'] for info in param_changes.values())
    total_changed = sum(info['num_changed'] for info in param_changes.values())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Parameters changed: {total_changed:,} ({100*total_changed/total_params:.2f}%)")
    
    # Show top 5 most changed layers
    sorted_changes = sorted(param_changes.items(), 
                           key=lambda x: x[1]['mean_abs_change'], 
                           reverse=True)
    print("\n  Top 5 layers by mean absolute change:")
    for layer_name, info in sorted_changes[:5]:
        if info['mean_abs_change'] > 1e-10:
            print(f"    {layer_name}: mean_change={info['mean_abs_change']:.2e}, "
                  f"max_change={info['max_abs_change']:.2e}, "
                  f"changed={info['percent_changed']:.1f}%")
    
    # Analyze prediction differences
    print("\nPrediction Analysis:")
    pred_diff = y_pred_adapted - y_pred_baseline
    print(f"  Mean prediction change: {np.mean(np.abs(pred_diff)):.6f}")
    print(f"  Max prediction change: {np.max(np.abs(pred_diff)):.6f}")
    print(f"  Std of changes: {np.std(pred_diff):.6f}")
    
    # Visual comparison of first few predictions
    print("\n  First 3 time steps (first 2 features):")
    print("  Time | Feature | Baseline | Adapted | Difference")
    print("  -----|---------|----------|---------|------------")
    for t in range(min(3, len(y_pred_baseline))):
        for f in range(2):
            print(f"   {t:2d}  |    {f:2d}   | {y_pred_baseline[t,f]:8.4f} | "
                  f"{y_pred_adapted[t,f]:8.4f} | {pred_diff[t,f]:+8.4f}")
    
    # Reset model to original state
    tta_model.reset()
    
    return {
        'baseline_mse': float(baseline_mse),
        'adapted_mse': float(adapted_mse),
        'mse_change': float(adapted_mse - baseline_mse),
        'mse_change_percent': float((adapted_mse/baseline_mse - 1)*100),
        'param_changes': param_changes,
        'total_params_changed': total_changed,
        'mean_pred_change': float(np.mean(np.abs(pred_diff))),
        'step_info': step_info if 'step_info' in locals() else None
    }


def test_on_different_ood_levels(model, data_dir):
    """Test TTA on different OOD scenarios."""
    print("\n" + "="*70)
    print("Testing on Different OOD Levels")
    print("="*70)
    
    scenarios = []
    
    # 1. In-distribution (constant gravity = training)
    const_files = sorted(data_dir.glob("constant_gravity_*.pkl"))
    if const_files:
        with open(const_files[-1], 'rb') as f:
            const_data = pickle.load(f)
        scenarios.append({
            'name': 'In-distribution (constant gravity)',
            'data': const_data,
            'expected': 'Should perform well without adaptation'
        })
    
    # 2. Time-varying gravity (current test case)
    varying_files = sorted(data_dir.glob("time_varying_gravity_*.pkl"))
    if varying_files:
        with open(varying_files[-1], 'rb') as f:
            varying_data = pickle.load(f)
        scenarios.append({
            'name': 'Time-varying gravity',
            'data': varying_data,
            'expected': 'Current failing case'
        })
    
    # 3. Create simple 2x gravity scenario
    print("\nGenerating 2x gravity scenario...")
    if const_files:
        with open(const_files[-1], 'rb') as f:
            const_data = pickle.load(f)
        
        # Double gravity in trajectories
        double_grav_trajs = []
        for traj in const_data['trajectories'][:20]:
            # Assuming gravity affects y-acceleration (columns 3 and 7)
            traj_2x = traj.copy()
            # Double the acceleration due to gravity
            # This is approximate - would need proper physics sim for exact
            traj_2x[:, [3, 7]] *= 2.0  # y-velocities
            double_grav_trajs.append(traj_2x)
        
        scenarios.append({
            'name': '2x constant gravity (simpler OOD)',
            'data': {'trajectories': double_grav_trajs},
            'expected': 'Simpler OOD case - should be easier to adapt'
        })
    
    # Test each scenario
    results = {}
    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        print(f"Expected: {scenario['expected']}")
        
        # Prepare test data
        X_test, y_test = prepare_data(scenario['data']['trajectories'][:10])
        
        # Baseline evaluation
        baseline_mse = model.evaluate(X_test, y_test, verbose=0)[0]
        print(f"Baseline MSE: {baseline_mse:.4f}")
        
        # Test with simple TTA
        tta_model = TTAWrapper(
            model,
            tta_method='regression_v2',
            adaptation_steps=10,
            learning_rate=1e-6,
            bn_only_mode=False,
            consistency_loss_weight=0.1,
            smoothness_loss_weight=0.05
        )
        
        # Adapt and evaluate
        errors = []
        for i in range(min(5, len(scenario['data']['trajectories']))):
            traj = scenario['data']['trajectories'][i]
            X = traj[0:1].reshape(1, 1, 8)
            y_true = traj[1:11]
            
            y_pred = tta_model.predict(X, adapt=True)
            if len(y_pred.shape) == 3:
                y_pred = y_pred[0]
            
            mse = np.mean((y_true[:len(y_pred)] - y_pred)**2)
            errors.append(mse)
            tta_model.reset()
        
        adapted_mse = np.mean(errors)
        improvement = (1 - adapted_mse/baseline_mse) * 100
        
        print(f"Adapted MSE: {adapted_mse:.4f}")
        print(f"Improvement: {improvement:+.1f}%")
        
        results[scenario['name']] = {
            'baseline_mse': baseline_mse,
            'adapted_mse': adapted_mse,
            'improvement_percent': improvement
        }
    
    return results


def main():
    """Run comprehensive TTA debugging."""
    print("TTA V2 Comprehensive Debugging")
    print("="*70)
    
    # Setup
    config = setup_environment()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory
    output_dir = get_output_path() / "tta_debugging"
    output_dir.mkdir(exist_ok=True)
    
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
    
    # Train model
    print("\nTraining base model...")
    model = create_physics_model()
    
    X_train, y_train = prepare_data(const_data['trajectories'][:100])
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
    
    # Define configurations to debug
    debug_configs = [
        {
            'name': 'Minimal adaptation (1 step, lr=1e-7)',
            'kwargs': {
                'tta_method': 'regression_v2',
                'adaptation_steps': 1,
                'learning_rate': 1e-7,
                'bn_only_mode': False,
                'consistency_loss_weight': 0.0,
                'smoothness_loss_weight': 0.0
            }
        },
        {
            'name': 'Standard adaptation (10 steps, lr=1e-6)',
            'kwargs': {
                'tta_method': 'regression_v2',
                'adaptation_steps': 10,
                'learning_rate': 1e-6,
                'bn_only_mode': False,
                'consistency_loss_weight': 0.1,
                'smoothness_loss_weight': 0.05
            }
        },
        {
            'name': 'BN-only adaptation',
            'kwargs': {
                'tta_method': 'regression_v2',
                'adaptation_steps': 10,
                'learning_rate': 1e-5,
                'bn_only_mode': True,
                'consistency_loss_weight': 0.0,
                'smoothness_loss_weight': 0.0
            }
        }
    ]
    
    # Debug each configuration
    all_results = {}
    for config_info in debug_configs:
        result = debug_single_adaptation(
            model, 
            ood_data['trajectories'][0],
            config_info['kwargs'],
            config_info
        )
        all_results[config_info['name']] = result
    
    # Test on different OOD levels
    ood_level_results = test_on_different_ood_levels(model, data_dir)
    
    # Save detailed results
    results_file = output_dir / f"tta_debug_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'detailed_results': all_results,
            'ood_level_results': ood_level_results
        }, f, indent=2, default=str)
    
    print(f"\n\nResults saved to: {results_file}")
    
    # Final analysis
    print("\n" + "="*70)
    print("DEBUGGING SUMMARY")
    print("="*70)
    
    # Check if any configuration improved
    any_improvement = False
    for name, result in all_results.items():
        if result['mse_change_percent'] < 0:
            any_improvement = True
            print(f"\n✓ {name}: {result['mse_change_percent']:.1f}% improvement")
    
    if not any_improvement:
        print("\n✗ No configuration showed improvement")
        
        # Analyze why
        print("\nKey observations:")
        
        # Check gradient magnitudes
        if all_results and 'step_info' in list(all_results.values())[0]:
            first_result = list(all_results.values())[0]
            if first_result['step_info']:
                first_grads = first_result['step_info'][0]['gradients']
                max_grad = max(g['max_abs_grad'] for g in first_grads.values())
                mean_grad = np.mean([g['mean_abs_grad'] for g in first_grads.values()])
                print(f"- Max gradient magnitude: {max_grad:.2e}")
                print(f"- Mean gradient magnitude: {mean_grad:.2e}")
                
                if max_grad < 1e-6:
                    print("  → Gradients are extremely small!")
        
        # Check parameter changes
        total_changed = sum(r['total_params_changed'] for r in all_results.values()) / len(all_results)
        print(f"- Average parameters changed: {total_changed:.0f}")
        
        if total_changed < 100:
            print("  → Very few parameters are being updated!")
        
        # Check prediction changes
        mean_pred_change = np.mean([r['mean_pred_change'] for r in all_results.values()])
        print(f"- Mean prediction change: {mean_pred_change:.6f}")
        
        if mean_pred_change < 0.01:
            print("  → Predictions barely change after adaptation!")
    
    # OOD level analysis
    print("\nOOD Level Analysis:")
    for scenario_name, result in ood_level_results.items():
        print(f"- {scenario_name}: {result['improvement_percent']:+.1f}% improvement")
    
    print("\nNext steps:")
    print("1. If gradients are too small, try higher learning rates")
    print("2. If predictions don't change, check loss function formulation")
    print("3. If simpler OOD works better, progressively increase complexity")
    print("4. Consider that time-varying gravity may require different approach")


if __name__ == "__main__":
    main()