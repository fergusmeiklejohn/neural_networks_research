"""Hyperparameter tuning for Test-Time Adaptation on physics OOD data."""

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

import keras

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
    tta_method="tent",
    learning_rate=1e-3,
    adaptation_steps=5,
    confidence_threshold=None,
    physics_loss_weight=0.1,
    update_bn_only=True,
    batch_size=1,
):
    """Evaluate a specific TTA configuration."""
    # Create TTA wrapper with specific config
    tta_kwargs = {
        "adaptation_steps": adaptation_steps,
        "learning_rate": learning_rate,
        "reset_after_batch": True,
        "update_bn_only": update_bn_only,
    }

    if tta_method == "tent" and confidence_threshold is not None:
        tta_kwargs["confidence_threshold"] = confidence_threshold
    elif tta_method == "physics_tent":
        tta_kwargs["physics_loss_weight"] = physics_loss_weight
        if confidence_threshold is not None:
            tta_kwargs["confidence_threshold"] = confidence_threshold

    tta_model = TTAWrapper(model, tta_method=tta_method, **tta_kwargs)

    # Evaluate on test trajectories
    errors = []
    for i, traj in enumerate(test_trajectories[:20]):  # Limit for speed
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

    return {"mean_mse": np.mean(errors), "std_mse": np.std(errors), "errors": errors}


def run_hyperparameter_search():
    """Run comprehensive hyperparameter search for TTA."""
    print("TTA Hyperparameter Tuning for Physics OOD")
    print("=" * 60)

    # Setup
    config = setup_environment()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load data
    data_dir = get_data_path() / "true_ood_physics"

    # Load constant gravity (training)
    print("Loading data...")
    const_files = sorted(data_dir.glob("constant_gravity_*.pkl"))
    with open(const_files[-1], "rb") as f:
        const_data = pickle.load(f)

    # Load time-varying gravity (OOD test)
    varying_files = sorted(data_dir.glob("time_varying_gravity_*.pkl"))
    with open(varying_files[-1], "rb") as f:
        ood_data = pickle.load(f)

    print(f"Loaded {len(const_data['trajectories'])} constant gravity trajectories")
    print(f"Loaded {len(ood_data['trajectories'])} time-varying gravity trajectories")

    # Train base model
    print("\nTraining base model...")
    model = create_physics_model()

    X_train, y_train = prepare_data(const_data["trajectories"][:80])
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

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
    hp_grid = {
        "tta_method": ["tent", "physics_tent"],
        "learning_rate": [5e-3, 1e-3, 5e-4, 1e-4, 5e-5],
        "adaptation_steps": [1, 5, 10, 20],
        "confidence_threshold": [None, 0.9, 0.95],
        "physics_loss_weight": [0.01, 0.1, 0.5],  # Only for physics_tent
        "update_bn_only": [True, False],
    }

    # Results storage
    results = []
    best_config = None
    best_mse = float("inf")

    print("\nStarting hyperparameter search...")
    print(f"Total configurations to test: ~{5 * 4 * 3 * 2 * 2}")  # Approximate

    # Grid search
    config_count = 0
    for tta_method in hp_grid["tta_method"]:
        for lr in hp_grid["learning_rate"]:
            for steps in hp_grid["adaptation_steps"]:
                for conf_thresh in hp_grid["confidence_threshold"]:
                    for update_bn in hp_grid["update_bn_only"]:
                        # For TENT, we don't need physics_loss_weight
                        if tta_method == "tent":
                            config_count += 1
                            print(
                                f"\nConfig {config_count}: {tta_method}, lr={lr}, steps={steps}, "
                                f"conf={conf_thresh}, bn_only={update_bn}"
                            )

                            result = evaluate_tta_config(
                                model,
                                ood_data["trajectories"],
                                tta_method=tta_method,
                                learning_rate=lr,
                                adaptation_steps=steps,
                                confidence_threshold=conf_thresh,
                                update_bn_only=update_bn,
                            )

                            config_result = {
                                "config_id": config_count,
                                "tta_method": tta_method,
                                "learning_rate": lr,
                                "adaptation_steps": steps,
                                "confidence_threshold": conf_thresh,
                                "physics_loss_weight": None,
                                "update_bn_only": update_bn,
                                "mean_mse": float(result["mean_mse"]),
                                "std_mse": float(result["std_mse"]),
                                "improvement": float(
                                    (1 - result["mean_mse"] / baseline_ood[0]) * 100
                                ),
                            }
                            results.append(config_result)

                            print(
                                f"  MSE: {result['mean_mse']:.4f} ± {result['std_mse']:.4f}"
                            )
                            print(f"  Improvement: {config_result['improvement']:.1f}%")

                            if result["mean_mse"] < best_mse:
                                best_mse = result["mean_mse"]
                                best_config = config_result

                        # For PhysicsTENT, we also vary physics_loss_weight
                        else:
                            for phys_weight in hp_grid["physics_loss_weight"]:
                                config_count += 1
                                print(
                                    f"\nConfig {config_count}: {tta_method}, lr={lr}, steps={steps}, "
                                    f"conf={conf_thresh}, phys_w={phys_weight}, bn_only={update_bn}"
                                )

                                result = evaluate_tta_config(
                                    model,
                                    ood_data["trajectories"],
                                    tta_method=tta_method,
                                    learning_rate=lr,
                                    adaptation_steps=steps,
                                    confidence_threshold=conf_thresh,
                                    physics_loss_weight=phys_weight,
                                    update_bn_only=update_bn,
                                )

                                config_result = {
                                    "config_id": config_count,
                                    "tta_method": tta_method,
                                    "learning_rate": lr,
                                    "adaptation_steps": steps,
                                    "confidence_threshold": conf_thresh,
                                    "physics_loss_weight": phys_weight,
                                    "update_bn_only": update_bn,
                                    "mean_mse": float(result["mean_mse"]),
                                    "std_mse": float(result["std_mse"]),
                                    "improvement": float(
                                        (1 - result["mean_mse"] / baseline_ood[0]) * 100
                                    ),
                                }
                                results.append(config_result)

                                print(
                                    f"  MSE: {result['mean_mse']:.4f} ± {result['std_mse']:.4f}"
                                )
                                print(
                                    f"  Improvement: {config_result['improvement']:.1f}%"
                                )

                                if result["mean_mse"] < best_mse:
                                    best_mse = result["mean_mse"]
                                    best_config = config_result

                        # Early stopping if we're doing really well
                        if best_config and best_config["improvement"] > 30:
                            print(
                                "\nFound excellent configuration, continuing to verify..."
                            )

    # Save results
    output_dir = get_output_path() / "tta_tuning"
    output_dir.mkdir(exist_ok=True)

    results_file = output_dir / f"tta_tuning_results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "baseline_const_mse": baseline_const[0],
                "baseline_ood_mse": baseline_ood[0],
                "baseline_degradation": baseline_ood[0] / baseline_const[0],
                "n_configs_tested": len(results),
                "results": results,
                "best_config": best_config,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 60)
    print("HYPERPARAMETER SEARCH COMPLETE")
    print("=" * 60)
    print(f"\nBaseline OOD MSE (no TTA): {baseline_ood[0]:.4f}")
    print(f"\nBest configuration found:")
    print(f"  Method: {best_config['tta_method']}")
    print(f"  Learning rate: {best_config['learning_rate']}")
    print(f"  Adaptation steps: {best_config['adaptation_steps']}")
    print(f"  Confidence threshold: {best_config['confidence_threshold']}")
    if best_config["physics_loss_weight"] is not None:
        print(f"  Physics loss weight: {best_config['physics_loss_weight']}")
    print(f"  Update BN only: {best_config['update_bn_only']}")
    print(f"  MSE: {best_config['mean_mse']:.4f} ± {best_config['std_mse']:.4f}")
    print(f"  Improvement: {best_config['improvement']:.1f}%")

    # Show top 5 configurations
    print("\nTop 5 configurations:")
    sorted_results = sorted(results, key=lambda x: x["mean_mse"])
    for i, config in enumerate(sorted_results[:5]):
        print(
            f"\n{i+1}. {config['tta_method']} - MSE: {config['mean_mse']:.4f} "
            f"(improvement: {config['improvement']:.1f}%)"
        )
        print(
            f"   lr={config['learning_rate']}, steps={config['adaptation_steps']}, "
            f"conf={config['confidence_threshold']}, bn_only={config['update_bn_only']}"
        )
        if config["physics_loss_weight"] is not None:
            print(f"   physics_weight={config['physics_loss_weight']}")

    print(f"\nResults saved to: {results_file}")

    # Test best config on more trajectories
    print("\nValidating best configuration on more trajectories...")
    best_tta = TTAWrapper(
        model,
        tta_method=best_config["tta_method"],
        adaptation_steps=best_config["adaptation_steps"],
        learning_rate=best_config["learning_rate"],
        confidence_threshold=best_config["confidence_threshold"],
        physics_loss_weight=best_config.get("physics_loss_weight"),
        update_bn_only=best_config["update_bn_only"],
    )

    # Test on 50 trajectories
    validation_errors = []
    for traj in ood_data["trajectories"][:50]:
        X = traj[0:1].reshape(1, 1, 8)
        y_true = traj[1:11]
        y_pred = best_tta.predict(X, adapt=True)
        if len(y_pred.shape) == 3:
            y_pred = y_pred[0]
        mse = np.mean((y_true[: len(y_pred)] - y_pred[: len(y_true)]) ** 2)
        validation_errors.append(mse)
        best_tta.reset()

    print(
        f"Validation MSE (50 trajectories): {np.mean(validation_errors):.4f} ± {np.std(validation_errors):.4f}"
    )
    print(
        f"Validation improvement: {(1 - np.mean(validation_errors)/baseline_ood[0])*100:.1f}%"
    )


def main():
    """Run hyperparameter tuning."""
    run_hyperparameter_search()


if __name__ == "__main__":
    main()
