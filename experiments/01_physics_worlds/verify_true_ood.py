"""Verify that time-varying gravity data is truly out-of-distribution."""

import pickle
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.imports import setup_project_paths

setup_project_paths()

import keras

from utils.config import setup_environment
from utils.paths import get_data_path, get_output_path


class OODVerifier:
    """Verify if test data is truly out-of-distribution."""

    def __init__(self, model=None):
        """Initialize verifier with optional pre-trained model."""
        self.model = model
        self.train_representations = None
        self.knn = None

    def create_physics_model(self, input_dim=8, hidden_dim=64):
        """Create a simple model for learning representations."""
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(1, input_dim)),
                keras.layers.Flatten(),
                keras.layers.Dense(hidden_dim, activation="relu"),
                keras.layers.BatchNormalization(),
                keras.layers.Dense(
                    hidden_dim, activation="relu", name="representation"
                ),
                keras.layers.BatchNormalization(),
                keras.layers.Dense(input_dim * 49),  # Predict next 49 timesteps
                keras.layers.Reshape((49, input_dim)),
            ]
        )

        model.compile(optimizer="adam", loss="mse")
        return model

    def train_on_constant_physics(self, const_data, epochs=20):
        """Train model on constant gravity physics."""
        print("Training model on constant gravity data...")

        # Prepare training data
        trajectories = const_data["trajectories"]
        len(trajectories)

        # Use first timestep to predict rest of trajectory
        X_train = trajectories[:, 0:1, :]  # Shape: (n_samples, 1, 8)
        y_train = trajectories[:, 1:, :]  # Shape: (n_samples, 49, 8)

        # Create and train model
        if self.model is None:
            self.model = self.create_physics_model()

        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            verbose=1,
        )

        return history

    def extract_representations(self, data):
        """Extract learned representations from data."""
        trajectories = data["trajectories"]
        X = trajectories[:, 0:1, :]  # First timestep

        # Get representation layer
        repr_layer = None
        for layer in self.model.layers:
            if layer.name == "representation":
                repr_layer = layer
                break

        if repr_layer is None:
            # Use second-to-last layer
            repr_layer = self.model.layers[-3]

        # Create representation extractor
        # Get intermediate output using functional API
        intermediate_output = repr_layer.output

        # Get representations by calling model up to representation layer
        inp = self.model.inputs[0]
        repr_model = keras.Model(inputs=inp, outputs=intermediate_output)

        # Extract representations
        representations = repr_model.predict(X, verbose=0)
        return representations

    def fit_training_manifold(self, train_data):
        """Fit k-NN model to training representations."""
        print("\nFitting training manifold...")

        # Extract training representations
        self.train_representations = self.extract_representations(train_data)

        # Fit k-NN
        self.knn = NearestNeighbors(n_neighbors=10, metric="euclidean")
        self.knn.fit(self.train_representations)

        # Compute training distances for threshold
        train_distances, _ = self.knn.kneighbors(self.train_representations)
        self.train_mean_dist = train_distances.mean(axis=1)
        self.ood_threshold = np.percentile(self.train_mean_dist, 95)

        print(
            f"Training manifold fitted with {len(self.train_representations)} samples"
        )
        print(f"OOD threshold (95th percentile): {self.ood_threshold:.4f}")

    def verify_ood(self, test_data):
        """Verify what percentage of test data is truly OOD."""
        print("\nVerifying OOD status...")

        # Extract test representations
        test_representations = self.extract_representations(test_data)

        # Compute distances to training manifold
        test_distances, test_indices = self.knn.kneighbors(test_representations)
        test_mean_dist = test_distances.mean(axis=1)

        # Classify as OOD
        is_ood = test_mean_dist > self.ood_threshold
        ood_percentage = np.mean(is_ood) * 100

        results = {
            "ood_percentage": ood_percentage,
            "test_distances": test_mean_dist,
            "is_ood": is_ood,
            "threshold": self.ood_threshold,
            "train_distances": self.train_mean_dist,
        }

        print(f"OOD percentage: {ood_percentage:.1f}%")
        print(f"Mean test distance: {test_mean_dist.mean():.4f}")
        print(f"Mean train distance: {self.train_mean_dist.mean():.4f}")
        print(
            f"Distance ratio: {test_mean_dist.mean() / self.train_mean_dist.mean():.2f}x"
        )

        return results

    def visualize_representations(self, train_data, test_data, save_path=None):
        """Visualize representations using PCA."""
        print("\nVisualizing representations...")

        # Extract representations
        train_repr = self.extract_representations(train_data)
        test_repr = self.extract_representations(test_data)

        # Combine and apply PCA
        all_repr = np.vstack([train_repr, test_repr])
        pca = PCA(n_components=2)
        all_pca = pca.fit_transform(all_repr)

        # Split back
        n_train = len(train_repr)
        train_pca = all_pca[:n_train]
        test_pca = all_pca[n_train:]

        # Plot
        plt.figure(figsize=(10, 8))

        # Training data
        plt.scatter(
            train_pca[:, 0],
            train_pca[:, 1],
            c="blue",
            alpha=0.6,
            label="Training (constant g)",
            s=50,
        )

        # Test data colored by OOD status
        test_distances, _ = self.knn.kneighbors(test_repr)
        test_mean_dist = test_distances.mean(axis=1)
        is_ood = test_mean_dist > self.ood_threshold

        plt.scatter(
            test_pca[~is_ood, 0],
            test_pca[~is_ood, 1],
            c="green",
            alpha=0.6,
            label="Test (interpolation)",
            s=50,
            marker="^",
        )
        plt.scatter(
            test_pca[is_ood, 0],
            test_pca[is_ood, 1],
            c="red",
            alpha=0.6,
            label="Test (true OOD)",
            s=50,
            marker="s",
        )

        plt.xlabel("First Principal Component")
        plt.ylabel("Second Principal Component")
        plt.title("Representation Space: Constant vs Time-Varying Gravity")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add explained variance
        var_explained = pca.explained_variance_ratio_
        plt.text(
            0.02,
            0.98,
            f"Variance explained: {var_explained[0]:.2%}, {var_explained[1]:.2%}",
            transform=plt.gca().transAxes,
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

        return pca


def main():
    """Verify true OOD status of time-varying gravity data."""
    print("True OOD Verification for Time-Varying Gravity")
    print("=" * 60)

    # Load data
    data_dir = get_data_path() / "true_ood_physics"

    # Find most recent files
    const_files = list(data_dir.glob("constant_gravity_*.pkl"))
    varying_files = list(data_dir.glob("time_varying_gravity_*.pkl"))

    if not const_files or not varying_files:
        print(
            "Error: Could not find data files. Run generate_true_ood_data_minimal.py first."
        )
        return

    # Load most recent
    const_file = sorted(const_files)[-1]
    varying_file = sorted(varying_files)[-1]

    print(f"Loading constant gravity: {const_file.name}")
    with open(const_file, "rb") as f:
        const_data = pickle.load(f)

    print(f"Loading time-varying gravity: {varying_file.name}")
    with open(varying_file, "rb") as f:
        varying_data = pickle.load(f)

    # Create verifier
    verifier = OODVerifier()

    # Train on constant gravity
    history = verifier.train_on_constant_physics(const_data, epochs=30)

    # Fit training manifold
    verifier.fit_training_manifold(const_data)

    # Verify OOD status
    results = verifier.verify_ood(varying_data)

    # Visualize
    viz_path = (
        get_output_path()
        / "ood_verification"
        / f"representation_space_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )
    viz_path.parent.mkdir(parents=True, exist_ok=True)
    pca = verifier.visualize_representations(
        const_data, varying_data, save_path=viz_path
    )

    # Additional analysis
    print("\n" + "=" * 60)
    print("OOD Verification Summary")
    print("=" * 60)
    print(f"Training: Constant gravity (g = -392 pixels/s²)")
    print(f"Test: Time-varying gravity (g(t) = -392 * (1 + 0.3*sin(0.5*t)))")
    print(f"True OOD percentage: {results['ood_percentage']:.1f}%")

    if results["ood_percentage"] > 50:
        print("✓ SUCCESS: Majority of test samples are truly out-of-distribution!")
    else:
        print("✗ WARNING: Most test samples are still within training distribution")

    # Save results
    results_path = (
        get_output_path()
        / "ood_verification"
        / f"verification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    )
    with open(results_path, "wb") as f:
        pickle.dump(results, f)

    print(f"\nResults saved to: {results_path}")
    print(f"Visualization saved to: {viz_path}")

    return results


if __name__ == "__main__":
    config = setup_environment()
    results = main()
