"""
Train baseline models on pendulum data
Adapted from baseline_models_physics.py for pendulum state representation
"""

import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import keras
import matplotlib.pyplot as plt
import numpy as np

# Import baseline model architectures
from baseline_models_physics import (
    PhysicsGFlowNetBaseline,
    PhysicsGraphExtrapolationBaseline,
    PhysicsMAMLBaseline,
)
from keras import layers


class PendulumDataLoader:
    """Load and preprocess pendulum data for baseline models"""

    def __init__(self, data_dir: str = "data/processed/pendulum"):
        self.data_dir = Path(data_dir)

    def load_dataset(self, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load pendulum dataset and convert to model input format"""
        filepath = self.data_dir / f"{filename}.pkl"

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        # Extract trajectories
        trajectories = data["trajectories"]

        # Convert to input/output pairs
        X, y = [], []

        for traj_data in trajectories:
            traj = traj_data["trajectory"]

            # Create state representation: [x, y, theta, theta_dot, length]
            # Note: Including length even for fixed pendulum to match dimensions
            states = np.column_stack(
                [traj["x"], traj["y"], traj["theta"], traj["theta_dot"], traj["length"]]
            )

            # Create input/output pairs (predict next 10 steps)
            for i in range(len(states) - 11):
                X.append(states[i : i + 1])  # Current state
                y.append(states[i + 1 : i + 11])  # Next 10 states

        return np.array(X), np.array(y)

    def prepare_data(self):
        """Load all datasets"""
        print("Loading pendulum datasets...")

        # Load training data
        X_train, y_train = self.load_dataset("pendulum_train")
        X_val, y_val = self.load_dataset("pendulum_val")
        X_test_fixed, y_test_fixed = self.load_dataset("pendulum_test_fixed")
        X_test_ood, y_test_ood = self.load_dataset("pendulum_test_ood")

        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test (fixed) samples: {len(X_test_fixed)}")
        print(f"Test (OOD) samples: {len(X_test_ood)}")

        return {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test_fixed": (X_test_fixed, y_test_fixed),
            "test_ood": (X_test_ood, y_test_ood),
        }


class PendulumBaselineTrainer:
    """Train and evaluate baseline models on pendulum data"""

    def __init__(self, output_dir: str = "outputs/pendulum_baselines"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def adapt_model_for_pendulum(self, model_class, config: Dict[str, Any]):
        """Adapt baseline model for pendulum state dimensions"""
        # Override dimensions for pendulum
        if hasattr(model_class, "__name__"):
            if "GFlowNet" in model_class.__name__:
                # Modify config for pendulum
                model = model_class(config)
                model.input_dim = 5  # [x, y, theta, theta_dot, length]
                model.output_dim = 5
                model.build_model()
                return model
            elif "GraphExtrapolation" in model_class.__name__:
                model = model_class(config)
                model.state_dim = 5
                model.build_model()
                return model
            elif "MAML" in model_class.__name__:
                model = model_class(config)
                model.input_dim = 5
                model.output_dim = 5
                model.build_model()
                return model

        # Default case
        model = model_class(config)
        if hasattr(model, "input_dim"):
            model.input_dim = 5
        if hasattr(model, "output_dim"):
            model.output_dim = 5
        if hasattr(model, "state_dim"):
            model.state_dim = 5
        model.build_model()
        return model

    def train_erm_baseline(self, data: Dict):
        """Train standard ERM baseline with data augmentation"""
        print("\n=== Training ERM + Augmentation Baseline ===")

        X_train, y_train = data["train"]
        X_val, y_val = data["val"]

        # Build simple feedforward model
        model = keras.Sequential(
            [
                layers.Input(shape=(1, 5)),
                layers.Flatten(),
                layers.Dense(256, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.1),
                layers.Dense(256, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.1),
                layers.Dense(128, activation="relu"),
                layers.Dense(10 * 5),  # 10 steps * 5 features
                layers.Reshape((10, 5)),
            ]
        )

        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

        # Data augmentation: add noise to training data
        def augment_data(X, y, noise_scale=0.01):
            X_aug = X + np.random.normal(0, noise_scale, X.shape)
            y_aug = y + np.random.normal(0, noise_scale, y.shape)
            return X_aug, y_aug

        # Train with augmentation
        best_val_loss = float("inf")
        for epoch in range(50):
            # Augment training data each epoch
            X_aug, y_aug = augment_data(X_train, y_train)

            # Train for one epoch
            history = model.fit(
                X_aug,
                y_aug,
                validation_data=(X_val, y_val),
                epochs=1,
                batch_size=32,
                verbose=0,
            )

            val_loss = history.history["val_loss"][0]
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save(self.output_dir / "erm_best.keras")

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: val_loss = {val_loss:.4f}")

        # Load best model
        model = keras.models.load_model(self.output_dir / "erm_best.keras")

        return model

    def evaluate_model(self, model, data: Dict, model_name: str):
        """Evaluate model on all test sets"""
        results = {}

        # Evaluate on fixed-length test set
        X_test, y_test = data["test_fixed"]
        y_pred = model.predict(X_test, verbose=0)
        mse_fixed = np.mean((y_pred - y_test) ** 2)
        results["mse_fixed"] = float(mse_fixed)

        # Evaluate on OOD (time-varying) test set
        X_ood, y_ood = data["test_ood"]
        y_pred_ood = model.predict(X_ood, verbose=0)
        mse_ood = np.mean((y_pred_ood - y_ood) ** 2)
        results["mse_ood"] = float(mse_ood)

        # Compute degradation factor
        results["ood_degradation"] = mse_ood / mse_fixed

        print(f"\n{model_name} Results:")
        print(f"  Fixed-length MSE: {mse_fixed:.4f}")
        print(f"  Time-varying MSE: {mse_ood:.4f}")
        print(f"  OOD Degradation: {results['ood_degradation']:.1f}x")

        return results

    def train_all_baselines(self, data: Dict):
        """Train all baseline models"""
        results = {}

        # 1. ERM + Augmentation
        erm_model = self.train_erm_baseline(data)
        results["ERM+Aug"] = self.evaluate_model(erm_model, data, "ERM+Aug")

        # 2. GFlowNet
        print("\n=== Training GFlowNet Baseline ===")
        gflownet = self.adapt_model_for_pendulum(
            PhysicsGFlowNetBaseline, {"exploration_bonus": 0.1, "flow_steps": 5}
        )
        gflownet.train(data["train"], data["val"], epochs=30)
        results["GFlowNet"] = self.evaluate_model(gflownet.predictor, data, "GFlowNet")

        # 3. Graph Extrapolation
        print("\n=== Training Graph Extrapolation Baseline ===")
        graphextrap = self.adapt_model_for_pendulum(
            PhysicsGraphExtrapolationBaseline, {"graph_layers": 3, "node_dim": 64}
        )
        graphextrap.train(data["train"], data["val"], epochs=30)
        results["GraphExtrap"] = self.evaluate_model(
            graphextrap.model, data, "GraphExtrap"
        )

        # 4. MAML
        print("\n=== Training MAML Baseline ===")
        maml = self.adapt_model_for_pendulum(
            PhysicsMAMLBaseline, {"inner_lr": 0.01, "inner_steps": 5}
        )
        maml.train(data["train"], data["val"], epochs=30)
        results["MAML"] = self.evaluate_model(maml.model, data, "MAML")

        return results

    def save_results(self, results: Dict):
        """Save results and create comparison plot"""
        # Save JSON results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"pendulum_baseline_results_{timestamp}.json"

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        # Create comparison plot
        models = list(results.keys())
        mse_fixed = [results[m]["mse_fixed"] for m in models]
        mse_ood = [results[m]["mse_ood"] for m in models]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # MSE comparison
        x = np.arange(len(models))
        width = 0.35

        ax1.bar(
            x - width / 2,
            mse_fixed,
            width,
            label="Fixed Length",
            color="blue",
            alpha=0.7,
        )
        ax1.bar(
            x + width / 2, mse_ood, width, label="Time-Varying", color="red", alpha=0.7
        )
        ax1.set_xlabel("Model")
        ax1.set_ylabel("MSE")
        ax1.set_title("Pendulum Prediction Error")
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45)
        ax1.legend()
        ax1.set_yscale("log")

        # Degradation factor
        degradation = [results[m]["ood_degradation"] for m in models]
        ax2.bar(models, degradation, color="orange", alpha=0.7)
        ax2.set_xlabel("Model")
        ax2.set_ylabel("OOD Degradation Factor")
        ax2.set_title("Performance Degradation on Time-Varying Pendulum")
        ax2.set_xticklabels(models, rotation=45)
        ax2.axhline(y=1, color="k", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(self.output_dir / "pendulum_baseline_comparison.png", dpi=150)
        print(f"\nResults saved to {results_file}")
        print(f"Plot saved to {self.output_dir / 'pendulum_baseline_comparison.png'}")

        # Print summary table
        print("\n=== Summary Table ===")
        print(f"{'Model':<15} {'Fixed MSE':<12} {'OOD MSE':<12} {'Degradation':<12}")
        print("-" * 55)
        for model in models:
            r = results[model]
            print(
                f"{model:<15} {r['mse_fixed']:<12.4f} {r['mse_ood']:<12.4f} {r['ood_degradation']:<12.1f}x"
            )


def main():
    """Main training pipeline"""
    print("Starting pendulum baseline training...")

    # First generate the data if it doesn't exist
    pendulum_data_path = Path("data/processed/pendulum")
    if not (pendulum_data_path / "pendulum_train.pkl").exists():
        print("\nPendulum data not found. Generating datasets first...")
        from pendulum_data_generator import PendulumDataConfig, PendulumDataGenerator

        config = PendulumDataConfig(num_samples=10000)
        generator = PendulumDataGenerator(config)

        # Generate all datasets
        print("Generating training data...")
        train_data = generator.generate_dataset(mechanism="fixed", num_samples=8000)
        generator.save_dataset(train_data, "pendulum_train")

        print("Generating validation data...")
        val_data = generator.generate_dataset(mechanism="fixed", num_samples=1000)
        generator.save_dataset(val_data, "pendulum_val")

        print("Generating test data...")
        test_fixed = generator.generate_dataset(mechanism="fixed", num_samples=1000)
        generator.save_dataset(test_fixed, "pendulum_test_fixed")

        print("Generating OOD test data...")
        test_ood = generator.generate_dataset(
            mechanism="time-varying", num_samples=1000
        )
        generator.save_dataset(test_ood, "pendulum_test_ood")

    # Load data
    loader = PendulumDataLoader()
    data = loader.prepare_data()

    # Train baselines
    trainer = PendulumBaselineTrainer()
    results = trainer.train_all_baselines(data)

    # Save results
    trainer.save_results(results)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
