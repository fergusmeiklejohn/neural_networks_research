"""Quick test of TTA on existing baseline models."""

import os
import sys
import numpy as np
import pickle
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.imports import setup_project_paths
setup_project_paths()

from utils.config import setup_environment
from models.test_time_adaptation import TENT, PhysicsTENT, PhysicsTTT
from models.test_time_adaptation.tta_wrappers import TTAWrapper
import keras


def load_test_data():
    """Load existing test data."""
    data_path = Path(__file__).parent / "data/processed/physics_worlds"
    test_file = data_path / "test_data.pkl"
    
    if test_file.exists():
        with open(test_file, 'rb') as f:
            data = pickle.load(f)
        # Check data format
        if isinstance(data, dict) and 'trajectories' in data:
            print(f"Loaded test data with {len(data['trajectories'])} trajectories")
            return data
        elif isinstance(data, list):
            # Old format - list of trajectories
            print(f"Loaded test data with {len(data)} trajectories")
            return {'trajectories': np.array(data), 'metadata': {}}
    else:
        print(f"Test data not found at {test_file}")
        # Create synthetic test data for demonstration
        print("Creating synthetic test data...")
        n_samples = 10
        time_steps = 50
        features = 8  # x1,y1,vx1,vy1,x2,y2,vx2,vy2
        
        # Simple falling ball trajectories
        trajectories = []
        for i in range(n_samples):
            traj = np.zeros((time_steps, features))
            # Initial positions and velocities
            traj[0] = np.random.randn(features) * 0.1
            traj[0, [1, 5]] = 1.0  # Start at y=1
            
            # Simulate with gravity
            g = -9.8
            dt = 0.1
            for t in range(1, time_steps):
                # Update positions
                traj[t, [0, 4]] = traj[t-1, [0, 4]] + traj[t-1, [2, 6]] * dt
                traj[t, [1, 5]] = traj[t-1, [1, 5]] + traj[t-1, [3, 7]] * dt
                # Update velocities (gravity on y)
                traj[t, [2, 6]] = traj[t-1, [2, 6]]
                traj[t, [3, 7]] = traj[t-1, [3, 7]] + g * dt
            
            trajectories.append(traj)
        
        return {
            'trajectories': np.array(trajectories),
            'metadata': {'gravity': g, 'synthetic': True}
        }


def evaluate_single_trajectory(model, trajectory, adapt=False):
    """Evaluate model on a single trajectory."""
    # Use first timestep as input
    input_state = trajectory[0:1]
    target = trajectory[1:]
    
    # Expand dims for batch
    input_batch = np.expand_dims(input_state, 0)
    
    # Predict
    if hasattr(model, 'predict'):
        pred = model.predict(input_batch, adapt=adapt)
    else:
        pred = model(input_batch, training=False)
    
    # Handle different output shapes
    if isinstance(pred, (list, tuple)):
        pred = pred[0]
    
    pred = np.array(pred)
    
    # Compute MSE
    if len(pred.shape) == 3:  # (batch, time, features)
        # Ensure same length
        min_len = min(pred.shape[1], len(target))
        mse = np.mean((pred[0, :min_len] - target[:min_len]) ** 2)
    else:
        # Single timestep prediction
        mse = np.mean((pred[0] - target[0]) ** 2)
    
    return mse, pred


def test_model_with_tta(model_path, test_data, model_type='generic'):
    """Test a model with different TTA methods."""
    print(f"\n{'='*60}")
    print(f"Testing model: {os.path.basename(model_path)}")
    print(f"{'='*60}")
    
    # Load model
    try:
        model = keras.models.load_model(model_path, compile=False)
        print(f"Model loaded successfully")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    results = {}
    trajectories = test_data['trajectories'][:5]  # Test on first 5 trajectories
    
    # Test without TTA (baseline)
    print("\n1. Testing without TTA...")
    mse_scores = []
    for traj in trajectories:
        mse, _ = evaluate_single_trajectory(model, traj, adapt=False)
        mse_scores.append(mse)
    
    results['no_tta'] = {
        'mse': np.mean(mse_scores),
        'mse_std': np.std(mse_scores)
    }
    print(f"   MSE: {results['no_tta']['mse']:.4f} ± {results['no_tta']['mse_std']:.4f}")
    
    # Test with TENT
    print("\n2. Testing with TENT...")
    try:
        tent_model = TTAWrapper(model, tta_method='tent', adaptation_steps=1, learning_rate=1e-4)
        mse_scores = []
        for traj in trajectories:
            mse, _ = evaluate_single_trajectory(tent_model, traj, adapt=True)
            mse_scores.append(mse)
            tent_model.reset()  # Reset after each trajectory
        
        results['tent'] = {
            'mse': np.mean(mse_scores),
            'mse_std': np.std(mse_scores)
        }
        print(f"   MSE: {results['tent']['mse']:.4f} ± {results['tent']['mse_std']:.4f}")
        
        # Calculate improvement
        improvement = (results['no_tta']['mse'] - results['tent']['mse']) / results['no_tta']['mse'] * 100
        print(f"   Improvement: {improvement:.1f}%")
    except Exception as e:
        print(f"   Error with TENT: {e}")
        results['tent'] = {'error': str(e)}
    
    # Test with PhysicsTENT
    print("\n3. Testing with PhysicsTENT...")
    try:
        physics_tent_model = TTAWrapper(
            model, 
            tta_method='physics_tent', 
            adaptation_steps=1, 
            learning_rate=1e-4,
            physics_loss_weight=0.1
        )
        mse_scores = []
        for traj in trajectories:
            mse, _ = evaluate_single_trajectory(physics_tent_model, traj, adapt=True)
            mse_scores.append(mse)
            physics_tent_model.reset()
        
        results['physics_tent'] = {
            'mse': np.mean(mse_scores),
            'mse_std': np.std(mse_scores)
        }
        print(f"   MSE: {results['physics_tent']['mse']:.4f} ± {results['physics_tent']['mse_std']:.4f}")
        
        improvement = (results['no_tta']['mse'] - results['physics_tent']['mse']) / results['no_tta']['mse'] * 100
        print(f"   Improvement: {improvement:.1f}%")
    except Exception as e:
        print(f"   Error with PhysicsTENT: {e}")
        results['physics_tent'] = {'error': str(e)}
    
    # Test with TTT (if model supports it)
    if model_type != 'pinn':  # PINNs might not work well with TTT
        print("\n4. Testing with TTT...")
        try:
            ttt_model = TTAWrapper(
                model,
                tta_method='ttt',
                adaptation_steps=5,
                learning_rate=1e-4,
                trajectory_length=50,
                adaptation_window=5
            )
            mse_scores = []
            for traj in trajectories:
                mse, _ = evaluate_single_trajectory(ttt_model, traj, adapt=True)
                mse_scores.append(mse)
                ttt_model.reset()
            
            results['ttt'] = {
                'mse': np.mean(mse_scores),
                'mse_std': np.std(mse_scores)
            }
            print(f"   MSE: {results['ttt']['mse']:.4f} ± {results['ttt']['mse_std']:.4f}")
            
            improvement = (results['no_tta']['mse'] - results['ttt']['mse']) / results['no_tta']['mse'] * 100
            print(f"   Improvement: {improvement:.1f}%")
        except Exception as e:
            print(f"   Error with TTT: {e}")
            results['ttt'] = {'error': str(e)}
    
    return results


def main():
    """Main test script."""
    # Setup
    config = setup_environment()
    
    # Load test data
    test_data = load_test_data()
    
    # Models to test - use absolute paths from current directory
    base_dir = Path(__file__).parent
    model_paths = [
        (base_dir / "outputs/baseline_results/gflownet_model_20250715_062359.keras", "gflownet"),
        (base_dir / "outputs/baseline_results/maml_model_20250715_062721.keras", "maml"),
        (base_dir / "outputs/minimal_pinn/model_20250715_063020.keras", "pinn"),
    ]
    
    all_results = {}
    
    for model_path, model_type in model_paths:
        if model_path.exists():
            results = test_model_with_tta(str(model_path), test_data, model_type)
            if results:
                all_results[model_type] = results
        else:
            print(f"\nModel not found: {model_path}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY OF RESULTS")
    print("="*60)
    
    for model_type, results in all_results.items():
        print(f"\n{model_type.upper()}:")
        
        # Find best method
        best_method = None
        best_mse = float('inf')
        baseline_mse = results.get('no_tta', {}).get('mse', float('inf'))
        
        for method, metrics in results.items():
            if 'mse' in metrics and metrics['mse'] < best_mse:
                best_mse = metrics['mse']
                best_method = method
        
        if best_method and baseline_mse < float('inf'):
            improvement = (baseline_mse - best_mse) / baseline_mse * 100
            print(f"  Best method: {best_method}")
            print(f"  Baseline MSE: {baseline_mse:.4f}")
            print(f"  Best MSE: {best_mse:.4f}")
            print(f"  Improvement: {improvement:.1f}%")
    
    print("\nTest complete!")


if __name__ == "__main__":
    main()