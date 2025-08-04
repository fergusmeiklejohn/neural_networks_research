"""Comprehensive TTA V2 hyperparameter tuning with wider search range."""

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import keras
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.imports import setup_project_paths

setup_project_paths()

from models.test_time_adaptation.tta_wrappers import TTAWrapper
from utils.config import setup_environment
from utils.paths import get_data_path, get_output_path


def create_physics_model(input_steps=1, output_steps=10):
    """Create a physics prediction model."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_steps, 8)),
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(8 * output_steps),
            keras.layers.Reshape((output_steps, 8)),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mae"]
    )
    return model


def prepare_data(trajectories, input_steps=1, output_steps=10):
    """Prepare trajectory data for training/evaluation."""
    X, y = [], []
    for traj in trajectories:
        for i in range(len(traj) - input_steps - output_steps + 1):
            X.append(traj[i : i + input_steps])
            y.append(traj[i + input_steps : i + input_steps + output_steps])
    return np.array(X), np.array(y)


def evaluate_tta_config(
    model,
    test_trajectories,
    tta_method="regression_v2",
    learning_rate=1e-5,
    adaptation_steps=10,
    consistency_loss_weight=0.1,
    smoothness_loss_weight=0.05,
    energy_loss_weight=0.1,
    momentum_loss_weight=0.1,
    bn_only_mode=False,
    n_test_trajectories=20,
    verbose=False,
):
    """Evaluate a specific TTA configuration."""
    try:
        # Create TTA wrapper with specific config
        tta_kwargs = {
            "adaptation_steps": adaptation_steps,
            "learning_rate": learning_rate,
            "reset_after_batch": True,
        }

        if tta_method == "regression_v2":
            tta_kwargs["consistency_loss_weight"] = consistency_loss_weight
            tta_kwargs["smoothness_loss_weight"] = smoothness_loss_weight
            tta_kwargs["bn_only_mode"] = bn_only_mode
        elif tta_method == "physics_regression_v2":
            tta_kwargs["consistency_loss_weight"] = consistency_loss_weight
            tta_kwargs["smoothness_loss_weight"] = smoothness_loss_weight
            tta_kwargs["energy_loss_weight"] = energy_loss_weight
            tta_kwargs["momentum_loss_weight"] = momentum_loss_weight
            tta_kwargs["bn_only_mode"] = bn_only_mode

        tta_model = TTAWrapper(model, tta_method=tta_method, **tta_kwargs)

        # Evaluate on test trajectories
        errors = []
        for i, traj in enumerate(test_trajectories[:n_test_trajectories]):
            X = traj[0:1].reshape(1, 1, 8)
            y_true = traj[1:11]

            # Adapt and predict
            y_pred = tta_model.predict(X, adapt=True)
            if len(y_pred.shape) == 3:
                y_pred = y_pred[0]

            # Compute error
            min_len = min(len(y_true), len(y_pred))
            mse = np.mean((y_true[:min_len] - y_pred[:min_len]) ** 2)
            errors.append(mse)

            # Reset model for next trajectory
            tta_model.reset()

        return {
            "mean_mse": np.mean(errors),
            "std_mse": np.std(errors),
            "errors": errors,
            "success": True,
        }

    except Exception as e:
        print(f"  ERROR: {str(e)}")
        return {
            "mean_mse": float("inf"),
            "std_mse": 0.0,
            "errors": [],
            "success": False,
            "error": str(e),
        }


def run_hyperparameter_search(debug_mode=False):
    """Run comprehensive hyperparameter search for TTA V2."""
    print(
        "TTA V2 Hyperparameter Tuning"
        + (" (Debug Mode)" if debug_mode else " (Full Search)")
    )
    print("=" * 70)

    # Setup
    setup_environment()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load data
    data_dir = get_data_path() / "true_ood_physics"

    # Load constant gravity (training)
    print("Loading data...")
    const_files = sorted(data_dir.glob("constant_gravity_*.pkl"))
    if not const_files:
        print("ERROR: No constant gravity data found!")
        return

    with open(const_files[-1], "rb") as f:
        const_data = pickle.load(f)

    # Load time-varying gravity (OOD test)
    varying_files = sorted(data_dir.glob("time_varying_gravity_*.pkl"))
    if not varying_files:
        print("ERROR: No time-varying gravity data found!")
        return

    with open(varying_files[-1], "rb") as f:
        ood_data = pickle.load(f)

    print(f"Loaded {len(const_data['trajectories'])} constant gravity trajectories")
    print(f"Loaded {len(ood_data['trajectories'])} time-varying gravity trajectories")

    # Train base model
    print("\nTraining base model...")
    model = create_physics_model()

    # Adjust for debug mode
    n_train = 30 if debug_mode else 80
    n_epochs = 5 if debug_mode else 20

    X_train, y_train = prepare_data(const_data["trajectories"][:n_train])
    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")

    model.fit(X_train, y_train, epochs=n_epochs, batch_size=32, verbose=0)

    # Baseline evaluation (no TTA)
    print("\nBaseline evaluation (no TTA)...")
    X_test_const, y_test_const = prepare_data(const_data["trajectories"][80:90])
    baseline_const = model.evaluate(X_test_const, y_test_const, verbose=0)
    print(f"Constant gravity MSE: {baseline_const[0]:.4f}")

    X_test_ood, y_test_ood = prepare_data(ood_data["trajectories"][:10])
    baseline_ood = model.evaluate(X_test_ood, y_test_ood, verbose=0)
    print(f"Time-varying gravity MSE (no TTA): {baseline_ood[0]:.4f}")
    print(f"Degradation: {baseline_ood[0]/baseline_const[0]:.2f}x")

    # Define hyperparameter grid
    if debug_mode:
        # Quick debug search
        hp_grid = {
            "learning_rates": [1e-4, 1e-5, 1e-6, 1e-7],  # Wider range
            "adaptation_steps": [5, 10, 20],
            "consistency_weights": [
                0.0,
                0.1,
                0.5,
            ],  # Include 0 to test if losses are the issue
            "smoothness_weights": [0.0, 0.05],
            "energy_weights": [0.0, 0.1],
            "momentum_weights": [0.0, 0.1],
        }
        n_test_traj = 10
    else:
        # Comprehensive search focusing on very low learning rates
        hp_grid = {
            "learning_rates": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7],
            "adaptation_steps": [1, 3, 5, 10, 20, 30, 50],
            "consistency_weights": [0.0, 0.01, 0.05, 0.1, 0.2, 0.5],
            "smoothness_weights": [0.0, 0.01, 0.05, 0.1],
            "energy_weights": [0.0, 0.05, 0.1, 0.2],
            "momentum_weights": [0.0, 0.05, 0.1, 0.2],
        }
        n_test_traj = 30

    # Results storage
    results = []
    best_config = None
    best_mse = float("inf")

    print("\nStarting hyperparameter search...")
    print(
        f"Testing {len(hp_grid['learning_rates'])} learning rates from {hp_grid['learning_rates'][-1]} to {hp_grid['learning_rates'][0]}"
    )
    print(f"Evaluating on {n_test_traj} trajectories per configuration")

    # Test regression_v2 first (simpler, no physics losses)
    print("\n--- Testing RegressionTTAV2 ---")
    config_count = 0

    for lr in hp_grid["learning_rates"]:
        for steps in hp_grid["adaptation_steps"]:
            # Test with different loss weights
            for cons_w in hp_grid["consistency_weights"][:3]:  # Limit in debug
                for smooth_w in hp_grid["smoothness_weights"][:2]:
                    # Test both BN-only and full parameter updates
                    for bn_only in [True, False]:
                        config_count += 1

                        # Skip some configs in debug mode to save time
                        if debug_mode and config_count % 3 != 0:
                            continue

                        mode_str = "BN-only" if bn_only else "All params"
                        print(
                            f"\nConfig {config_count}: lr={lr:.0e}, steps={steps}, "
                            f"cons={cons_w}, smooth={smooth_w}, {mode_str}"
                        )

                        result = evaluate_tta_config(
                            model,
                            ood_data["trajectories"],
                            tta_method="regression_v2",
                            learning_rate=lr,
                            adaptation_steps=steps,
                            consistency_loss_weight=cons_w,
                            smoothness_loss_weight=smooth_w,
                            bn_only_mode=bn_only,
                            n_test_trajectories=n_test_traj,
                        )

                        if result["success"]:
                            improvement = (
                                1 - result["mean_mse"] / baseline_ood[0]
                            ) * 100
                            config_data = {
                                "config_id": config_count,
                                "tta_method": "regression_v2",
                                "learning_rate": lr,
                                "adaptation_steps": steps,
                                "consistency_loss_weight": cons_w,
                                "smoothness_loss_weight": smooth_w,
                                "bn_only_mode": bn_only,
                                "mean_mse": float(result["mean_mse"]),
                                "std_mse": float(result["std_mse"]),
                                "improvement": float(improvement),
                            }
                            results.append(config_data)

                            print(
                                f"  MSE: {result['mean_mse']:.4f} ± {result['std_mse']:.4f}"
                            )
                            print(f"  Improvement: {improvement:.1f}%")

                            if result["mean_mse"] < best_mse:
                                best_mse = result["mean_mse"]
                                best_config = config_data

                            # Early indication of good config
                            if improvement > 10:
                                print("  ✓ Positive improvement!")

    # Test physics_regression_v2 (only if regression_v2 shows promise)
    if best_config and best_config["improvement"] > -20:
        print("\n--- Testing PhysicsRegressionTTAV2 ---")

        # Use best learning rates from regression_v2
        promising_lrs = sorted(
            list(set([r["learning_rate"] for r in results if r["improvement"] > -30]))
        )[:3]
        if not promising_lrs:
            promising_lrs = [1e-6, 1e-7]

        for lr in promising_lrs:
            for steps in [10, 20]:  # Fewer steps to test
                for energy_w in hp_grid["energy_weights"][:2]:
                    for momentum_w in hp_grid["momentum_weights"][:2]:
                        for bn_only in [True, False]:
                            config_count += 1

                            mode_str = "BN-only" if bn_only else "All params"
                            print(
                                f"\nConfig {config_count}: physics_v2, lr={lr:.0e}, steps={steps}, "
                                f"energy={energy_w}, momentum={momentum_w}, {mode_str}"
                            )

                            result = evaluate_tta_config(
                                model,
                                ood_data["trajectories"],
                                tta_method="physics_regression_v2",
                                learning_rate=lr,
                                adaptation_steps=steps,
                                consistency_loss_weight=0.1,
                                smoothness_loss_weight=0.05,
                                energy_loss_weight=energy_w,
                                momentum_loss_weight=momentum_w,
                                bn_only_mode=bn_only,
                                n_test_trajectories=n_test_traj,
                            )

                            if result["success"]:
                                improvement = (
                                    1 - result["mean_mse"] / baseline_ood[0]
                                ) * 100
                                config_data = {
                                    "config_id": config_count,
                                    "tta_method": "physics_regression_v2",
                                    "learning_rate": lr,
                                    "adaptation_steps": steps,
                                    "consistency_loss_weight": 0.1,
                                    "smoothness_loss_weight": 0.05,
                                    "energy_loss_weight": energy_w,
                                    "momentum_loss_weight": momentum_w,
                                    "bn_only_mode": bn_only,
                                    "mean_mse": float(result["mean_mse"]),
                                    "std_mse": float(result["std_mse"]),
                                    "improvement": float(improvement),
                                }
                                results.append(config_data)

                                print(
                                    f"  MSE: {result['mean_mse']:.4f} ± {result['std_mse']:.4f}"
                                )
                                print(f"  Improvement: {improvement:.1f}%")

                                if result["mean_mse"] < best_mse:
                                    best_mse = result["mean_mse"]
                                    best_config = config_data

    # Save results
    output_dir = get_output_path() / "tta_tuning"
    output_dir.mkdir(exist_ok=True)

    mode_str = "debug" if debug_mode else "full"
    results_file = output_dir / f"tta_v2_tuning_results_{mode_str}_{timestamp}.json"

    summary = {
        "timestamp": timestamp,
        "debug_mode": debug_mode,
        "baseline_const_mse": float(baseline_const[0]),
        "baseline_ood_mse": float(baseline_ood[0]),
        "baseline_degradation": float(baseline_ood[0] / baseline_const[0]),
        "n_configs_tested": len(results),
        "hyperparameter_grid": {
            "learning_rates": hp_grid["learning_rates"],
            "adaptation_steps": hp_grid["adaptation_steps"],
            "n_test_trajectories": n_test_traj,
        },
        "results": results,
    }

    if best_config:
        summary["best_config"] = best_config

    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("HYPERPARAMETER SEARCH COMPLETE")
    print("=" * 70)
    print(f"\nBaseline OOD MSE (no TTA): {baseline_ood[0]:.4f}")

    if best_config:
        print(f"\nBest configuration found:")
        print(f"  Method: {best_config['tta_method']}")
        print(f"  Learning rate: {best_config['learning_rate']:.0e}")
        print(f"  Adaptation steps: {best_config['adaptation_steps']}")
        print(f"  BN-only mode: {best_config['bn_only_mode']}")
        print(f"  MSE: {best_config['mean_mse']:.4f} ± {best_config['std_mse']:.4f}")
        print(f"  Improvement: {best_config['improvement']:.1f}%")

        # Show top 5 configurations
        print("\nTop 5 configurations:")
        sorted_results = sorted(results, key=lambda x: x["mean_mse"])
        for i, cfg in enumerate(sorted_results[:5]):
            mode_str = "BN-only" if cfg["bn_only_mode"] else "All params"
            print(
                f"\n{i+1}. {cfg['tta_method']} - MSE: {cfg['mean_mse']:.4f} "
                f"(improvement: {cfg['improvement']:.1f}%)"
            )
            print(
                f"   lr={cfg['learning_rate']:.0e}, steps={cfg['adaptation_steps']}, {mode_str}"
            )

        # Analyze results
        print("\n--- Analysis ---")
        positive_configs = [r for r in results if r["improvement"] > 0]
        if positive_configs:
            print(
                f"Found {len(positive_configs)} configurations with positive improvement!"
            )
            best_positive = max(positive_configs, key=lambda x: x["improvement"])
            print(
                f"Best improvement: {best_positive['improvement']:.1f}% with lr={best_positive['learning_rate']:.0e}"
            )
        else:
            print("No configurations showed positive improvement.")

            # Find patterns
            lr_performance = {}
            for r in results:
                lr = r["learning_rate"]
                if lr not in lr_performance:
                    lr_performance[lr] = []
                lr_performance[lr].append(r["improvement"])

            print("\nAverage improvement by learning rate:")
            for lr in sorted(lr_performance.keys()):
                avg_imp = np.mean(lr_performance[lr])
                print(f"  lr={lr:.0e}: {avg_imp:.1f}%")

    print(f"\nResults saved to: {results_file}")
    print("\nTo read results later:")
    print(f"  with open('{results_file}', 'r') as f:")
    print("      results = json.load(f)")

    return results_file


def main():
    """Run hyperparameter tuning."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug", action="store_true", help="Run in debug mode with fewer configs"
    )
    args = parser.parse_args()

    run_hyperparameter_search(debug_mode=args.debug)


if __name__ == "__main__":
    main()
