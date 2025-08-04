"""Evaluate GFlowNet and MAML baselines on time-varying gravity data."""

import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.imports import setup_project_paths

setup_project_paths()

from baseline_models_physics import (
    PhysicsGFlowNetBaseline,
    PhysicsGraphExtrapolationBaseline,
    PhysicsMAMLBaseline,
)

from utils.config import setup_environment
from utils.paths import get_data_path, get_output_path


def prepare_data(trajectories, input_steps=1, output_steps=10):
    """Prepare trajectory data for training/evaluation."""
    X, y = [], []
    for traj in trajectories:
        for i in range(len(traj) - input_steps - output_steps + 1):
            X.append(traj[i : i + input_steps])
            y.append(traj[i + input_steps : i + input_steps + output_steps])
    return np.array(X), np.array(y)


def create_baseline_config(model_type="physics"):
    """Create configuration for baseline models."""
    return {
        "task_type": "physics",
        "input_shape": (1, 8),  # Single timestep, 8 features
        "output_shape": (10, 8),  # 10 timesteps prediction
        "hidden_dim": 128,
        "num_layers": 3,
        "learning_rate": 1e-3,
    }


def evaluate_baseline(baseline_model, test_data, model_name):
    """Evaluate a baseline model on test data."""
    X_test, y_test = test_data

    # Get predictions
    predictions = baseline_model.predict(X_test)

    # Calculate MSE
    mse = np.mean((predictions - y_test) ** 2)

    # Calculate per-timestep MSE
    timestep_mse = np.mean((predictions - y_test) ** 2, axis=(0, 2))

    return {
        "model": model_name,
        "overall_mse": float(mse),
        "timestep_mse": timestep_mse.tolist(),
        "mean_prediction": float(np.mean(np.abs(predictions))),
        "mean_target": float(np.mean(np.abs(y_test))),
    }


def test_gflownet_baseline(train_data, val_data, test_data_dict):
    """Test GFlowNet baseline on physics prediction."""
    print("\n" + "=" * 70)
    print("Testing GFlowNet Baseline")
    print("=" * 70)

    # Create model
    config = create_baseline_config()
    gflownet = PhysicsGFlowNetBaseline(config)
    gflownet.build_model()

    print("\nTraining GFlowNet...")
    # GFlowNet training is exploration-based
    history = gflownet.train(
        train_data=train_data,
        val_data=val_data,
        epochs=30,  # Shorter for testing
        exploration_steps=50,
    )

    # Evaluate on different test sets
    results = {}
    for test_name, test_data in test_data_dict.items():
        print(f"\nEvaluating on {test_name}...")
        result = evaluate_baseline(gflownet, test_data, "GFlowNet")
        results[test_name] = result
        print(f"  MSE: {result['overall_mse']:.2f}")

    return results, gflownet


def test_maml_baseline(train_data, val_data, test_data_dict):
    """Test MAML baseline on physics prediction."""
    print("\n" + "=" * 70)
    print("Testing MAML Baseline")
    print("=" * 70)

    # Create model
    config = create_baseline_config()
    config["inner_lr"] = 0.01  # MAML-specific
    config["inner_steps"] = 5
    config["meta_batch_size"] = 8  # Smaller for our data size

    maml = PhysicsMAMLBaseline(config)
    maml.build_model()

    print("\nTraining MAML...")
    # MAML uses episodic training
    history = maml.train(
        train_data=train_data,
        val_data=val_data,
        epochs=30,  # Shorter for testing
        tasks_per_epoch=50,
    )

    # Test adaptation capability
    print("\nTesting adaptation on new physics...")
    # For time-varying gravity, we can create a few-shot task
    X_support, y_support = test_data_dict["time_varying"]
    X_support_few = X_support[:10]  # 10 examples for adaptation
    y_support_few = y_support[:10]

    # Adapt to new task
    adapted_model = maml.adapt_to_task(X_support_few, y_support_few)

    # Evaluate on different test sets
    results = {}
    for test_name, test_data in test_data_dict.items():
        print(f"\nEvaluating on {test_name}...")

        # Test both base and adapted models
        base_result = evaluate_baseline(maml, test_data, "MAML_base")
        results[f"{test_name}_base"] = base_result

        if test_name == "time_varying":
            # Use adapted model for time-varying
            adapted_result = {
                "model": "MAML_adapted",
                "overall_mse": float(
                    np.mean(
                        (adapted_model.predict(test_data[0][10:]) - test_data[1][10:])
                        ** 2
                    )
                ),
                "adaptation_examples": 10,
            }
            results[f"{test_name}_adapted"] = adapted_result
            print(f"  Base MSE: {base_result['overall_mse']:.2f}")
            print(f"  Adapted MSE: {adapted_result['overall_mse']:.2f}")
        else:
            print(f"  MSE: {base_result['overall_mse']:.2f}")

    return results, maml


def test_graphextrap_baseline(train_data, val_data, test_data_dict):
    """Test GraphExtrapolation baseline for comparison."""
    print("\n" + "=" * 70)
    print("Testing GraphExtrapolation Baseline (for reference)")
    print("=" * 70)

    # Create model
    config = create_baseline_config()
    config["use_geometric_features"] = True  # Key for GraphExtrap

    graph_model = PhysicsGraphExtrapolationBaseline(config)
    graph_model.build_model()

    print("\nTraining GraphExtrapolation...")
    history = graph_model.train(train_data=train_data, val_data=val_data, epochs=30)

    # Evaluate
    results = {}
    for test_name, test_data in test_data_dict.items():
        print(f"\nEvaluating on {test_name}...")
        result = evaluate_baseline(graph_model, test_data, "GraphExtrapolation")
        results[test_name] = result
        print(f"  MSE: {result['overall_mse']:.2f}")

    return results, graph_model


def main():
    """Run baseline evaluations on time-varying gravity data."""
    print("Baseline Evaluation on Time-Varying Gravity")
    print("=" * 70)

    # Setup
    setup_environment()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output directory
    output_dir = get_output_path() / "baseline_evaluation"
    output_dir.mkdir(exist_ok=True)

    # Load data (same as TTA experiments)
    data_dir = get_data_path() / "true_ood_physics"

    print("Loading data...")
    # Training data - constant gravity
    const_files = sorted(data_dir.glob("constant_gravity_*.pkl"))
    with open(const_files[-1], "rb") as f:
        const_data = pickle.load(f)

    # Test data - time-varying gravity
    varying_files = sorted(data_dir.glob("time_varying_gravity_*.pkl"))
    with open(varying_files[-1], "rb") as f:
        ood_data = pickle.load(f)

    print(f"Loaded {len(const_data['trajectories'])} constant gravity trajectories")
    print(f"Loaded {len(ood_data['trajectories'])} time-varying gravity trajectories")

    # Prepare data
    print("\nPreparing data...")
    X_train, y_train = prepare_data(const_data["trajectories"][:80])
    X_val, y_val = prepare_data(const_data["trajectories"][80:100])

    # Prepare test sets
    X_test_id, y_test_id = prepare_data(const_data["trajectories"][100:120])
    X_test_ood, y_test_ood = prepare_data(ood_data["trajectories"][:50])

    train_data = (X_train, y_train)
    val_data = (X_val, y_val)
    test_data_dict = {
        "in_distribution": (X_test_id, y_test_id),
        "time_varying": (X_test_ood, y_test_ood),
    }

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test ID samples: {len(X_test_id)}")
    print(f"Test OOD samples: {len(X_test_ood)}")

    # Run evaluations
    all_results = {}

    # Test GFlowNet
    try:
        gflownet_results, gflownet_model = test_gflownet_baseline(
            train_data, val_data, test_data_dict
        )
        all_results["GFlowNet"] = gflownet_results
    except Exception as e:
        print(f"GFlowNet failed: {e}")
        all_results["GFlowNet"] = {"error": str(e)}

    # Test MAML
    try:
        maml_results, maml_model = test_maml_baseline(
            train_data, val_data, test_data_dict
        )
        all_results["MAML"] = maml_results
    except Exception as e:
        print(f"MAML failed: {e}")
        all_results["MAML"] = {"error": str(e)}

    # Test GraphExtrapolation for comparison
    try:
        graph_results, graph_model = test_graphextrap_baseline(
            train_data, val_data, test_data_dict
        )
        all_results["GraphExtrapolation"] = graph_results
    except Exception as e:
        print(f"GraphExtrapolation failed: {e}")
        all_results["GraphExtrapolation"] = {"error": str(e)}

    # Save results
    results_file = output_dir / f"baseline_results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "results": all_results,
                "data_stats": {
                    "train_size": len(X_train),
                    "val_size": len(X_val),
                    "test_id_size": len(X_test_id),
                    "test_ood_size": len(X_test_ood),
                },
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to: {results_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print("\nIn-Distribution Performance:")
    for model_name in ["GFlowNet", "MAML", "GraphExtrapolation"]:
        if model_name in all_results and "in_distribution" in all_results[model_name]:
            mse = all_results[model_name]["in_distribution"]["overall_mse"]
            print(f"  {model_name}: MSE = {mse:.2f}")

    print("\nTime-Varying Gravity Performance:")
    for model_name in ["GFlowNet", "MAML", "GraphExtrapolation"]:
        if model_name in all_results:
            if "time_varying" in all_results[model_name]:
                mse = all_results[model_name]["time_varying"]["overall_mse"]
                print(f"  {model_name}: MSE = {mse:.2f}")
            if "time_varying_adapted" in all_results[model_name]:
                mse = all_results[model_name]["time_varying_adapted"]["overall_mse"]
                print(f"  {model_name} (adapted): MSE = {mse:.2f}")

    # Compare with TTA results
    print("\nComparison with TTA:")
    print("  TTA on time-varying: MSE = 6,935 (235% degradation)")
    print("  Baseline on time-varying: MSE = 2,070")

    # Calculate degradation/improvement
    baseline_mse = 2070
    for model_name in ["GFlowNet", "MAML"]:
        if model_name in all_results and "time_varying" in all_results[model_name]:
            mse = all_results[model_name]["time_varying"]["overall_mse"]
            change = (mse / baseline_mse - 1) * 100
            if change > 0:
                print(f"  {model_name}: {change:+.1f}% degradation vs no adaptation")
            else:
                print(f"  {model_name}: {-change:.1f}% improvement vs no adaptation")


if __name__ == "__main__":
    main()
