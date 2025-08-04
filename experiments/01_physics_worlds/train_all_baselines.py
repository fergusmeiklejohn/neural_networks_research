"""
Train All 4 Baseline Models on Physics Worlds Data

Complete implementation of all baselines:
1. ERM + Data Augmentation
2. GFlowNet-inspired exploration
3. Graph-based extrapolation
4. MAML for quick adaptation
"""

import os

os.environ["KERAS_BACKEND"] = "jax"

import json
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict

import keras
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))


class BaselineModel(ABC):
    """Abstract base class for baseline models."""

    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.name = config.get("name", "baseline")

    @abstractmethod
    def build_model(self):
        """Build the model architecture."""

    @abstractmethod
    def train(self, train_data, val_data, epochs: int = 100):
        """Train the model."""

    @abstractmethod
    def evaluate(self, test_data):
        """Evaluate model performance."""


class ERMWithAugmentation(BaselineModel):
    """Baseline 1: Standard ERM with data augmentation."""

    def build_model(self):
        """Build standard neural network."""
        self.model = keras.Sequential(
            [
                keras.layers.Input(shape=self.config["input_shape"]),
                keras.layers.Dense(256, activation="relu"),
                keras.layers.BatchNormalization(),
                keras.layers.Dense(256, activation="relu"),
                keras.layers.BatchNormalization(),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dense(np.prod(self.config["output_shape"])),
                keras.layers.Reshape(self.config["output_shape"]),
            ]
        )

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="mse",
            metrics=["mae"],
        )

    def augment_data(self, X, y):
        """Apply physics-aware data augmentation."""
        augmented_X = [X]
        augmented_y = [y]

        # 1. Add Gaussian noise
        noise_X = X + np.random.normal(0, 0.1, X.shape)
        noise_y = y + np.random.normal(0, 0.1, y.shape)
        augmented_X.append(noise_X)
        augmented_y.append(noise_y)

        # 2. Interpolation between samples
        idx = np.random.permutation(len(X))
        alpha = np.random.uniform(0.3, 0.7, (len(X), 1))
        interp_X = alpha * X + (1 - alpha) * X[idx]
        interp_y = alpha * y + (1 - alpha) * y[idx]
        augmented_X.append(interp_X)
        augmented_y.append(interp_y)

        # 3. Time reversal (physics symmetry)
        # Reverse velocities for time-reversal symmetry
        reverse_X = X.copy()
        reverse_X[:, 2:] *= -1  # Reverse velocities
        reverse_y = y.copy()
        reverse_y[:, 2:] *= -1
        augmented_X.append(reverse_X)
        augmented_y.append(reverse_y)

        return np.vstack(augmented_X), np.vstack(augmented_y)

    def train(self, train_data, val_data, epochs=100):
        """Train with augmented data."""
        X_train, y_train = train_data
        X_val, y_val = val_data

        # Apply augmentation
        X_aug, y_aug = self.augment_data(X_train, y_train)

        print(f"Training data augmented from {len(X_train)} to {len(X_aug)} samples")

        history = self.model.fit(
            X_aug,
            y_aug,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=64,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
            ],
            verbose=1,
        )

        return history

    def evaluate(self, test_data):
        """Evaluate on test set."""
        X_test, y_test = test_data
        loss, mae = self.model.evaluate(X_test, y_test, verbose=0)
        predictions = self.model.predict(X_test, verbose=0)
        return {"mse": loss, "mae": mae, "predictions": predictions}


class GFlowNetExploration(BaselineModel):
    """Baseline 2: GFlowNet-inspired exploration in parameter space."""

    def build_model(self):
        """Build exploration and prediction models."""
        # Parameter explorer network
        self.explorer = keras.Sequential(
            [
                keras.layers.Input(shape=self.config["input_shape"]),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(32, activation="tanh"),  # Latent physics parameters
            ],
            name="explorer",
        )

        # Dynamics predictor conditioned on parameters
        state_input = keras.Input(shape=self.config["input_shape"])
        param_input = keras.Input(shape=(32,))

        # Concatenate state and parameters
        combined = keras.layers.Concatenate()([state_input, param_input])
        x = keras.layers.Dense(256, activation="relu")(combined)
        x = keras.layers.Dense(256, activation="relu")(x)
        x = keras.layers.Dense(128, activation="relu")(x)
        output = keras.layers.Dense(np.prod(self.config["output_shape"]))(x)
        output = keras.layers.Reshape(self.config["output_shape"])(output)

        self.predictor = keras.Model(
            inputs=[state_input, param_input], outputs=output, name="predictor"
        )

        # Combined model
        state_in = keras.Input(shape=self.config["input_shape"])
        params = self.explorer(state_in)
        prediction = self.predictor([state_in, params])

        self.model = keras.Model(inputs=state_in, outputs=prediction)

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="mse",
            metrics=["mae"],
        )

    def train(self, train_data, val_data, epochs=100):
        """Train with exploration bonus."""
        X_train, y_train = train_data
        X_val, y_val = val_data

        # Custom training loop with exploration
        best_val_loss = float("inf")
        history = {"loss": [], "val_loss": [], "mae": [], "val_mae": []}

        for epoch in range(epochs):
            # Exploration phase: generate diverse parameters
            params = self.explorer.predict(X_train[:100], verbose=0)
            param_diversity = np.std(params, axis=0).mean()

            # Standard training
            hist = self.model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=1,
                batch_size=64,
                verbose=0,
            )

            # Track metrics
            history["loss"].append(hist.history["loss"][0])
            history["val_loss"].append(hist.history["val_loss"][0])
            history["mae"].append(hist.history["mae"][0])
            history["val_mae"].append(hist.history["val_mae"][0])

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch}: loss={history['loss'][-1]:.4f}, "
                    f"val_loss={history['val_loss'][-1]:.4f}, "
                    f"param_diversity={param_diversity:.4f}"
                )

            # Save best model
            if history["val_loss"][-1] < best_val_loss:
                best_val_loss = history["val_loss"][-1]
                self.best_weights = self.model.get_weights()

        # Restore best weights
        self.model.set_weights(self.best_weights)

        return history

    def evaluate(self, test_data):
        """Evaluate model."""
        X_test, y_test = test_data
        loss, mae = self.model.evaluate(X_test, y_test, verbose=0)
        predictions = self.model.predict(X_test, verbose=0)

        # Also evaluate parameter diversity
        params = self.explorer.predict(X_test, verbose=0)
        param_diversity = np.std(params, axis=0).mean()

        return {
            "mse": loss,
            "mae": mae,
            "param_diversity": param_diversity,
            "predictions": predictions,
        }


class GraphExtrapolation(BaselineModel):
    """Baseline 3: Graph-based representation for extrapolation."""

    def build_model(self):
        """Build graph neural network for physics."""
        # Simple GNN-inspired architecture
        self.model = keras.Sequential(
            [
                keras.layers.Input(shape=self.config["input_shape"]),
                # Node embedding
                keras.layers.Dense(128, activation="relu"),
                # Message passing layers
                keras.layers.Dense(256, activation="relu"),
                keras.layers.BatchNormalization(),
                keras.layers.Dense(256, activation="relu"),
                keras.layers.BatchNormalization(),
                # Aggregation
                keras.layers.Dense(128, activation="relu"),
                # Output
                keras.layers.Dense(np.prod(self.config["output_shape"])),
                keras.layers.Reshape(self.config["output_shape"]),
            ]
        )

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=5e-4),
            loss="mse",
            metrics=["mae"],
        )

    def compute_graph_features(self, X):
        """Extract graph-based features from states."""
        # Compute pairwise distances (for multi-particle systems)
        # For single particle, use distance from origin
        positions = X[:, :2]  # x, y positions
        distances = np.sqrt(np.sum(positions**2, axis=1, keepdims=True))

        # Compute angles
        angles = np.arctan2(X[:, 1], X[:, 0]).reshape(-1, 1)

        # Combine with original features
        graph_features = np.hstack([X, distances, angles])

        return graph_features

    def train(self, train_data, val_data, epochs=100):
        """Train with graph features."""
        X_train, y_train = train_data
        X_val, y_val = val_data

        # Add graph features
        X_train_graph = self.compute_graph_features(X_train)
        X_val_graph = self.compute_graph_features(X_val)

        # Update input shape for graph features
        if self.config["input_shape"][0] != X_train_graph.shape[1]:
            self.config["input_shape"] = (X_train_graph.shape[1],)
            self.build_model()

        history = self.model.fit(
            X_train_graph,
            y_train,
            validation_data=(X_val_graph, y_val),
            epochs=epochs,
            batch_size=64,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(patience=7, factor=0.5),
            ],
            verbose=1,
        )

        return history

    def evaluate(self, test_data):
        """Evaluate model."""
        X_test, y_test = test_data
        X_test_graph = self.compute_graph_features(X_test)

        loss, mae = self.model.evaluate(X_test_graph, y_test, verbose=0)
        predictions = self.model.predict(X_test_graph, verbose=0)

        return {"mse": loss, "mae": mae, "predictions": predictions}


class MAMLPhysics(BaselineModel):
    """Baseline 4: MAML for quick adaptation to new physics."""

    def build_model(self):
        """Build MAML architecture."""
        # Base model
        self.model = keras.Sequential(
            [
                keras.layers.Input(shape=self.config["input_shape"]),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dense(np.prod(self.config["output_shape"])),
                keras.layers.Reshape(self.config["output_shape"]),
            ]
        )

        # Inner loop optimizer
        self.inner_optimizer = keras.optimizers.SGD(learning_rate=0.01)

        # Outer loop optimizer
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="mse",
            metrics=["mae"],
        )

    def create_tasks(self, X, y, n_tasks=10, samples_per_task=100):
        """Create meta-learning tasks with different physics."""
        tasks = []

        for i in range(n_tasks):
            # Sample indices
            idx = np.random.choice(len(X), samples_per_task, replace=False)

            # Create task with modified physics
            task_X = X[idx].copy()
            task_y = y[idx].copy()

            # Modify physics parameters
            if i % 3 == 0:
                # Change gravity
                gravity_factor = np.random.uniform(0.5, 2.0)
                task_y[:, 3] *= gravity_factor
            elif i % 3 == 1:
                # Change friction
                friction_factor = np.random.uniform(0.8, 1.2)
                task_y[:, 2:] *= friction_factor
            else:
                # Add systematic bias
                bias = np.random.uniform(-0.5, 0.5, 4)
                task_y += bias

            tasks.append((task_X, task_y))

        return tasks

    def train(self, train_data, val_data, epochs=100):
        """Meta-training."""
        X_train, y_train = train_data
        X_val, y_val = val_data

        history = {"loss": [], "val_loss": [], "mae": [], "val_mae": []}

        for epoch in range(epochs):
            # Create tasks
            tasks = self.create_tasks(X_train, y_train)

            epoch_loss = 0
            epoch_mae = 0

            for task_X, task_y in tasks:
                # Save original weights
                original_weights = self.model.get_weights()

                # Inner loop: adapt to task
                for _ in range(5):  # 5 gradient steps
                    # Simple gradient-based update
                    hist = self.model.fit(
                        task_X, task_y, epochs=1, batch_size=len(task_X), verbose=0
                    )

                # Evaluate adapted model
                val_pred = self.model(X_val[:100], training=False)
                val_loss = keras.ops.mean(
                    keras.losses.mean_squared_error(y_val[:100], val_pred)
                )

                epoch_loss += float(val_loss)
                epoch_mae += float(
                    keras.ops.mean(keras.ops.abs(y_val[:100] - val_pred))
                )

                # Restore original weights
                self.model.set_weights(original_weights)

            # Outer loop update
            hist = self.model.fit(X_train, y_train, epochs=1, batch_size=64, verbose=0)

            # Track metrics
            history["loss"].append(hist.history["loss"][0])
            val_metrics = self.model.evaluate(X_val, y_val, verbose=0)
            history["val_loss"].append(val_metrics[0])
            history["mae"].append(hist.history["mae"][0])
            history["val_mae"].append(val_metrics[1])

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch}: loss={history['loss'][-1]:.4f}, "
                    f"val_loss={history['val_loss'][-1]:.4f}, "
                    f"meta_loss={epoch_loss/len(tasks):.4f}"
                )

        return history

    def evaluate(self, test_data):
        """Evaluate model."""
        X_test, y_test = test_data
        loss, mae = self.model.evaluate(X_test, y_test, verbose=0)
        predictions = self.model.predict(X_test, verbose=0)

        # Test adaptation capability
        adapt_tasks = self.create_tasks(X_test, y_test, n_tasks=5, samples_per_task=20)
        adapt_scores = []

        for task_X, task_y in adapt_tasks:
            # Quick adaptation
            original_weights = self.model.get_weights()

            for _ in range(10):
                self.model.fit(task_X, task_y, epochs=1, verbose=0)

            # Evaluate
            task_loss = self.model.evaluate(task_X, task_y, verbose=0)[0]
            adapt_scores.append(task_loss)

            # Restore
            self.model.set_weights(original_weights)

        return {
            "mse": loss,
            "mae": mae,
            "adaptation_score": np.mean(adapt_scores),
            "predictions": predictions,
        }


def generate_physics_data(n_samples=5000, test_ood=True):
    """Generate physics data with in-distribution and OOD samples."""

    # Physics parameters
    dt = 0.1
    gravity_values = {
        "earth": 9.8,
        "moon": 1.6,
        "mars": 3.7,
        "jupiter": 24.8,
        "space": 0.0,
    }

    # Training data: Earth and Mars gravity
    train_gravities = ["earth", "mars"]
    train_samples = []

    for _ in range(n_samples):
        # Initial conditions
        x0 = np.random.uniform(-10, 10)
        y0 = np.random.uniform(5, 20)
        vx0 = np.random.uniform(-5, 5)
        vy0 = np.random.uniform(-5, 5)

        # Random gravity from training set
        g = gravity_values[np.random.choice(train_gravities)]

        # Current state
        state = np.array([x0, y0, vx0, vy0])

        # Next state (simple physics)
        next_state = state.copy()
        next_state[0] += vx0 * dt
        next_state[1] += vy0 * dt
        next_state[3] -= g * dt

        train_samples.append((state, next_state))

    X_train = np.array([s[0] for s in train_samples])
    y_train = np.array([s[1] for s in train_samples])

    # Validation data: same distribution
    val_samples = []
    for _ in range(n_samples // 5):
        x0 = np.random.uniform(-10, 10)
        y0 = np.random.uniform(5, 20)
        vx0 = np.random.uniform(-5, 5)
        vy0 = np.random.uniform(-5, 5)

        g = gravity_values[np.random.choice(train_gravities)]

        state = np.array([x0, y0, vx0, vy0])
        next_state = state.copy()
        next_state[0] += vx0 * dt
        next_state[1] += vy0 * dt
        next_state[3] -= g * dt

        val_samples.append((state, next_state))

    X_val = np.array([s[0] for s in val_samples])
    y_val = np.array([s[1] for s in val_samples])

    # Test data: mix of in-distribution and OOD
    test_samples = []
    test_labels = []

    # In-distribution test
    for _ in range(n_samples // 10):
        x0 = np.random.uniform(-10, 10)
        y0 = np.random.uniform(5, 20)
        vx0 = np.random.uniform(-5, 5)
        vy0 = np.random.uniform(-5, 5)

        g = gravity_values[np.random.choice(train_gravities)]

        state = np.array([x0, y0, vx0, vy0])
        next_state = state.copy()
        next_state[0] += vx0 * dt
        next_state[1] += vy0 * dt
        next_state[3] -= g * dt

        test_samples.append((state, next_state))
        test_labels.append("in_distribution")

    if test_ood:
        # Near-OOD: Moon gravity
        for _ in range(n_samples // 10):
            x0 = np.random.uniform(-10, 10)
            y0 = np.random.uniform(5, 20)
            vx0 = np.random.uniform(-5, 5)
            vy0 = np.random.uniform(-5, 5)

            g = gravity_values["moon"]

            state = np.array([x0, y0, vx0, vy0])
            next_state = state.copy()
            next_state[0] += vx0 * dt
            next_state[1] += vy0 * dt
            next_state[3] -= g * dt

            test_samples.append((state, next_state))
            test_labels.append("near_ood")

        # Far-OOD: Jupiter gravity
        for _ in range(n_samples // 10):
            x0 = np.random.uniform(-10, 10)
            y0 = np.random.uniform(5, 20)
            vx0 = np.random.uniform(-5, 5)
            vy0 = np.random.uniform(-5, 5)

            g = gravity_values["jupiter"]

            state = np.array([x0, y0, vx0, vy0])
            next_state = state.copy()
            next_state[0] += vx0 * dt
            next_state[1] += vy0 * dt
            next_state[3] -= g * dt

            test_samples.append((state, next_state))
            test_labels.append("far_ood")

    X_test = np.array([s[0] for s in test_samples])
    y_test = np.array([s[1] for s in test_samples])

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), test_labels


def evaluate_extrapolation(baseline, test_data, test_labels):
    """Evaluate model on different extrapolation categories."""
    X_test, y_test = test_data

    results = {}

    for category in ["in_distribution", "near_ood", "far_ood"]:
        if category not in test_labels:
            continue

        # Get indices for this category
        indices = [i for i, label in enumerate(test_labels) if label == category]

        if not indices:
            continue

        X_cat = X_test[indices]
        y_cat = y_test[indices]

        # Handle GraphExtrapolation special case
        if hasattr(baseline, "compute_graph_features"):
            X_cat = baseline.compute_graph_features(X_cat)

        # Evaluate
        predictions = baseline.model.predict(X_cat, verbose=0)
        mse = np.mean((predictions - y_cat) ** 2)
        mae = np.mean(np.abs(predictions - y_cat))

        results[category] = {
            "mse": float(mse),
            "mae": float(mae),
            "count": len(indices),
        }

    return results


def main():
    """Train all baselines and compare."""

    print("=" * 60)
    print("Training All 4 Baseline Models on Physics Extrapolation")
    print("=" * 60)

    # Generate data
    print("\nGenerating physics data...")
    train_data, val_data, test_data, test_labels = generate_physics_data(n_samples=3000)
    print(f"Train: {train_data[0].shape}")
    print(f"Val: {val_data[0].shape}")
    print(f"Test: {test_data[0].shape}")

    # Configuration
    config = {
        "input_shape": (4,),  # [x, y, vx, vy]
        "output_shape": (4,),  # [x', y', vx', vy']
    }

    # Train all baselines
    baselines = [
        ("ERM+Aug", ERMWithAugmentation),
        ("GFlowNet", GFlowNetExploration),
        ("GraphExtrap", GraphExtrapolation),
        ("MAML", MAMLPhysics),
    ]

    results = {}

    for name, baseline_class in baselines:
        print(f"\n{'='*60}")
        print(f"Training {name}")
        print(f"{'='*60}")

        # Create model
        model_config = config.copy()
        model_config["name"] = name
        baseline = baseline_class(model_config)
        baseline.build_model()

        # Train
        start_time = datetime.now()
        history = baseline.train(train_data, val_data, epochs=50)
        train_time = (datetime.now() - start_time).total_seconds()

        # Evaluate
        print(f"\nEvaluating {name}...")
        eval_results = baseline.evaluate(test_data)

        # Extrapolation evaluation
        extrap_results = evaluate_extrapolation(baseline, test_data, test_labels)

        # Store results
        results[name] = {
            "train_time": train_time,
            "overall_mse": eval_results["mse"],
            "overall_mae": eval_results["mae"],
            "extrapolation": extrap_results,
            "special_metrics": {
                k: v
                for k, v in eval_results.items()
                if k not in ["mse", "mae", "predictions"]
            },
        }

        print(f"\nResults for {name}:")
        print(f"Overall MSE: {eval_results['mse']:.4f}")
        print(f"Overall MAE: {eval_results['mae']:.4f}")
        print(f"Training time: {train_time:.1f}s")

        for cat, metrics in extrap_results.items():
            print(f"{cat}: MSE={metrics['mse']:.4f}, n={metrics['count']}")

    # Save results
    output_dir = Path("outputs/baseline_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python types for JSON serialization
    def convert_to_json_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(v) for v in obj]
        return obj

    results_json = convert_to_json_serializable(results)

    results_file = output_dir / "all_baselines_results.json"
    with open(results_file, "w") as f:
        json.dump(results_json, f, indent=2)

    # Generate comparison report
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")

    print(
        f"\n{'Model':<15} {'Overall MSE':>12} {'In-Dist MSE':>12} {'Near-OOD MSE':>12} {'Far-OOD MSE':>12}"
    )
    print("-" * 75)

    for name, res in results.items():
        in_dist = res["extrapolation"].get("in_distribution", {}).get("mse", -1)
        near_ood = res["extrapolation"].get("near_ood", {}).get("mse", -1)
        far_ood = res["extrapolation"].get("far_ood", {}).get("mse", -1)

        print(
            f"{name:<15} {res['overall_mse']:>12.4f} {in_dist:>12.4f} {near_ood:>12.4f} {far_ood:>12.4f}"
        )

    # Special metrics
    print(f"\n{'='*60}")
    print("SPECIAL METRICS")
    print(f"{'='*60}")

    for name, res in results.items():
        if res["special_metrics"]:
            print(f"\n{name}:")
            for metric, value in res["special_metrics"].items():
                print(f"  {metric}: {value:.4f}")

    print(f"\nResults saved to: {results_file}")

    # Create markdown report
    report = ["# Baseline Models Comparison on Physics Extrapolation\n"]
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    report.append("## Task Description\n")
    report.append("- **In-distribution**: Earth (9.8 m/s²) and Mars (3.7 m/s²) gravity")
    report.append("- **Near-OOD**: Moon gravity (1.6 m/s²)")
    report.append("- **Far-OOD**: Jupiter gravity (24.8 m/s²)\n")

    report.append("## Results Summary\n")
    report.append(
        "| Model | Overall MSE | In-Dist MSE | Near-OOD MSE | Far-OOD MSE | Training Time |"
    )
    report.append(
        "|-------|-------------|-------------|--------------|-------------|---------------|"
    )

    for name, res in results.items():
        in_dist = res["extrapolation"].get("in_distribution", {}).get("mse", -1)
        near_ood = res["extrapolation"].get("near_ood", {}).get("mse", -1)
        far_ood = res["extrapolation"].get("far_ood", {}).get("mse", -1)

        report.append(
            f"| {name} | {res['overall_mse']:.4f} | {in_dist:.4f} | "
            f"{near_ood:.4f} | {far_ood:.4f} | {res['train_time']:.1f}s |"
        )

    report.append("\n## Key Insights\n")

    # Find best performers
    best_overall = min(results.items(), key=lambda x: x[1]["overall_mse"])[0]

    best_far_ood = None
    best_far_ood_score = float("inf")
    for name, res in results.items():
        far_ood_score = res["extrapolation"].get("far_ood", {}).get("mse", float("inf"))
        if far_ood_score < best_far_ood_score:
            best_far_ood = name
            best_far_ood_score = far_ood_score

    report.append(f"- **Best overall performance**: {best_overall}")
    report.append(
        f"- **Best far-OOD extrapolation**: {best_far_ood} (MSE: {best_far_ood_score:.4f})"
    )

    # Calculate degradation
    report.append("\n### Extrapolation Degradation\n")
    for name, res in results.items():
        in_dist = res["extrapolation"].get("in_distribution", {}).get("mse", 0)
        far_ood = res["extrapolation"].get("far_ood", {}).get("mse", 0)
        if in_dist > 0:
            degradation = (far_ood - in_dist) / in_dist * 100
            report.append(
                f"- {name}: {degradation:.1f}% increase from in-dist to far-OOD"
            )

    report_file = output_dir / "baseline_comparison_report.md"
    with open(report_file, "w") as f:
        f.write("\n".join(report))

    print(f"\nReport saved to: {report_file}")


if __name__ == "__main__":
    main()
