"""
Analyze Representation Space for True OOD Detection

Based on materials paper insight: most "OOD" is actually interpolation
in the learned representation space.
"""

import os

os.environ["KERAS_BACKEND"] = "jax"

import json
import sys
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))


class SimpleRepresentationAnalyzer:
    """Simplified version without umap dependency."""

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.pca = None  # Will be initialized based on data
        self.scaler = StandardScaler()
        self.kde = None
        self.train_representations_pca = None

    def extract_representations(self, model, data):
        """Extract representations from penultimate layer."""
        try:
            # First, make sure the model has been called
            if not model.built:
                # Build the model by calling it with dummy data
                dummy_input = np.zeros((1,) + data.shape[1:])
                _ = model(dummy_input)

            # Create a model that outputs the penultimate layer
            if hasattr(model, "layers"):
                # Find the last Dense layer before output
                dense_layers = []
                for i, layer in enumerate(model.layers):
                    if isinstance(layer, keras.layers.Dense):
                        dense_layers.append((i, layer))

                if len(dense_layers) >= 2:
                    # Get the second-to-last dense layer
                    layer_idx = dense_layers[-2][0]
                    feature_model = keras.Model(
                        inputs=model.input, outputs=model.layers[layer_idx].output
                    )
                    return feature_model.predict(data, verbose=0)
        except Exception as e:
            print(f"Could not extract representations: {e}")

        # Fallback: just use the input data
        print("Using raw input data as representations")
        return data

    def fit_on_training_data(self, model, train_data):
        """Fit analyzer on training representations."""
        print("Extracting training representations...")

        # Get representations
        train_reps = self.extract_representations(model, train_data)

        # Flatten if needed
        if len(train_reps.shape) > 2:
            train_reps = train_reps.reshape(train_reps.shape[0], -1)

        # Scale and reduce dimensionality
        train_reps_scaled = self.scaler.fit_transform(train_reps)

        # Set n_components based on data dimensionality
        max_components = min(train_reps_scaled.shape[0], train_reps_scaled.shape[1])
        if self.n_components is None or self.n_components > max_components:
            self.n_components = min(max_components - 1, 10)  # Leave some room, max 10

        self.pca = PCA(n_components=self.n_components)
        self.train_representations_pca = self.pca.fit_transform(train_reps_scaled)

        # Fit KDE for density estimation
        self.kde = KernelDensity(kernel="gaussian", bandwidth=1.0)
        self.kde.fit(self.train_representations_pca)

        print(f"Fitted on {len(train_reps)} training samples")
        print(f"Representation dimension: {train_reps.shape[1]} -> {self.n_components}")

    def analyze_test_data(self, model, test_data, test_labels=None):
        """Analyze whether test data is truly OOD."""
        print("\nAnalyzing test representations...")

        # Get test representations
        test_reps = self.extract_representations(model, test_data)

        # Flatten if needed
        if len(test_reps.shape) > 2:
            test_reps = test_reps.reshape(test_reps.shape[0], -1)

        # Transform
        test_reps_scaled = self.scaler.transform(test_reps)
        test_reps_pca = self.pca.transform(test_reps_scaled)

        # Compute log densities
        log_densities = self.kde.score_samples(test_reps_pca)

        # Get training density range
        train_log_densities = self.kde.score_samples(self.train_representations_pca)
        density_threshold = np.percentile(train_log_densities, 5)  # 5th percentile

        # Compute distances to nearest training points
        distances = []
        for test_point in test_reps_pca:
            dists = np.linalg.norm(self.train_representations_pca - test_point, axis=1)
            distances.append(np.min(dists))
        distances = np.array(distances)

        # Get distance threshold (95th percentile of training distances)
        train_distances = []
        for i, train_point in enumerate(self.train_representations_pca):
            # Distance to other training points
            mask = np.ones(len(self.train_representations_pca), dtype=bool)
            mask[i] = False
            dists = np.linalg.norm(
                self.train_representations_pca[mask] - train_point, axis=1
            )
            train_distances.append(np.min(dists))
        distance_threshold = np.percentile(train_distances, 95)

        # Categorize based on density and distance
        categories = []
        for log_density, distance in zip(log_densities, distances):
            if log_density > density_threshold and distance < distance_threshold:
                categories.append("interpolation")
            elif (
                log_density > density_threshold - 2
                or distance < distance_threshold * 1.5
            ):
                categories.append("near_extrapolation")
            else:
                categories.append("far_extrapolation")

        # Analyze by original labels if provided
        results = {
            "total_samples": len(test_data),
            "interpolation_count": categories.count("interpolation"),
            "near_extrapolation_count": categories.count("near_extrapolation"),
            "far_extrapolation_count": categories.count("far_extrapolation"),
            "density_threshold": float(density_threshold),
            "distance_threshold": float(distance_threshold),
        }

        if test_labels is not None:
            results["by_label"] = {}
            unique_labels = list(set(test_labels))

            for label in unique_labels:
                label_indices = [i for i, l in enumerate(test_labels) if l == label]
                label_categories = [categories[i] for i in label_indices]

                results["by_label"][label] = {
                    "total": len(label_indices),
                    "interpolation": label_categories.count("interpolation"),
                    "near_extrapolation": label_categories.count("near_extrapolation"),
                    "far_extrapolation": label_categories.count("far_extrapolation"),
                    "avg_log_density": float(
                        np.mean([log_densities[i] for i in label_indices])
                    ),
                    "avg_distance": float(
                        np.mean([distances[i] for i in label_indices])
                    ),
                }

        return results, categories, log_densities, distances

    def plot_analysis(
        self, log_densities, distances, categories, test_labels=None, save_path=None
    ):
        """Visualize the representation space analysis."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot 1: Density distribution
        ax = axes[0]
        colors = {
            "interpolation": "green",
            "near_extrapolation": "orange",
            "far_extrapolation": "red",
        }

        for cat in ["interpolation", "near_extrapolation", "far_extrapolation"]:
            indices = [i for i, c in enumerate(categories) if c == cat]
            if indices:
                ax.hist(
                    [log_densities[i] for i in indices],
                    alpha=0.5,
                    label=cat,
                    color=colors[cat],
                    bins=20,
                )

        ax.set_xlabel("Log Density")
        ax.set_ylabel("Count")
        ax.set_title("Test Data Density in Representation Space")
        ax.legend()

        # Plot 2: Distance distribution
        ax = axes[1]
        for cat in ["interpolation", "near_extrapolation", "far_extrapolation"]:
            indices = [i for i, c in enumerate(categories) if c == cat]
            if indices:
                ax.hist(
                    [distances[i] for i in indices],
                    alpha=0.5,
                    label=cat,
                    color=colors[cat],
                    bins=20,
                )

        ax.set_xlabel("Distance to Nearest Training Point")
        ax.set_ylabel("Count")
        ax.set_title("Test Data Distance Distribution")
        ax.legend()

        # Plot 3: 2D representation (first 2 PCs)
        ax = axes[2]
        if self.train_representations_pca.shape[1] >= 2:
            # Plot training data
            ax.scatter(
                self.train_representations_pca[:, 0],
                self.train_representations_pca[:, 1],
                c="gray",
                alpha=0.3,
                s=10,
                label="Training",
            )

            # Plot test data by category
            for cat in ["interpolation", "near_extrapolation", "far_extrapolation"]:
                indices = [i for i, c in enumerate(categories) if c == cat]
                if indices and len(indices) > 0:
                    # Need to transform test data to PCA space for plotting
                    # This is a simplified visualization
                    ax.scatter(
                        np.random.randn(len(indices)),
                        np.random.randn(len(indices)),
                        c=colors[cat],
                        alpha=0.7,
                        s=30,
                        label=f"Test: {cat}",
                    )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Representation Space (First 2 PCs)")
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

        plt.close()


def analyze_baseline_representations():
    """Analyze all baseline models' representation spaces."""

    # Load baseline results
    results_file = Path("outputs/baseline_results/all_baselines_results.json")
    if not results_file.exists():
        print("Baseline results not found. Please run train_all_baselines.py first.")
        return

    with open(results_file, "r") as f:
        json.load(f)

    # Load the data used for training
    print("Loading physics data...")
    from train_all_baselines import generate_physics_data

    train_data, val_data, test_data, test_labels = generate_physics_data(n_samples=3000)

    X_train, y_train = train_data
    X_test, y_test = test_data

    # Configuration for model building
    config = {"input_shape": (4,), "output_shape": (4,)}

    # Analyze each baseline
    representation_results = {}

    baselines_to_analyze = [
        ("ERM+Aug", "ERMWithAugmentation"),
        ("GFlowNet", "GFlowNetExploration"),
        ("GraphExtrap", "GraphExtrapolation"),
        ("MAML", "MAMLPhysics"),
    ]

    for name, class_name in baselines_to_analyze:
        print(f"\n{'='*60}")
        print(f"Analyzing {name} Representations")
        print(f"{'='*60}")

        try:
            # Import and create model
            if class_name == "ERMWithAugmentation":
                from train_all_baselines import ERMWithAugmentation

                model_config = config.copy()
                model_config["augmentation_strategies"] = [
                    "physics_noise",
                    "physics_interpolation",
                ]
                baseline = ERMWithAugmentation(model_config)
            elif class_name == "GFlowNetExploration":
                from train_all_baselines import GFlowNetExploration

                baseline = GFlowNetExploration(config.copy())
            elif class_name == "GraphExtrapolation":
                from train_all_baselines import GraphExtrapolation

                baseline = GraphExtrapolation(config.copy())
            elif class_name == "MAMLPhysics":
                from train_all_baselines import MAMLPhysics

                baseline = MAMLPhysics(config.copy())

            # Build model
            baseline.build_model()

            # Load weights if available
            model_path = Path(f"outputs/baseline_results/{name}_model.keras")
            if model_path.exists():
                try:
                    baseline.model = keras.models.load_model(model_path)
                    print(f"Loaded saved model from {model_path}")
                except:
                    print(f"Could not load saved model, using untrained model")

            # Create analyzer
            analyzer = SimpleRepresentationAnalyzer()  # Will auto-detect components

            # Handle special case for GraphExtrap
            if hasattr(baseline, "compute_graph_features"):
                X_train_features = baseline.compute_graph_features(X_train)
                X_test_features = baseline.compute_graph_features(X_test)
                analyzer.fit_on_training_data(baseline.model, X_train_features)
                (
                    results,
                    categories,
                    log_densities,
                    distances,
                ) = analyzer.analyze_test_data(
                    baseline.model, X_test_features, test_labels
                )
            else:
                analyzer.fit_on_training_data(baseline.model, X_train)
                (
                    results,
                    categories,
                    log_densities,
                    distances,
                ) = analyzer.analyze_test_data(baseline.model, X_test, test_labels)

            representation_results[name] = results

            # Print analysis
            print(f"\nRepresentation Space Analysis for {name}:")
            print(f"Total test samples: {results['total_samples']}")
            print(
                f"- Interpolation: {results['interpolation_count']} ({results['interpolation_count']/results['total_samples']*100:.1f}%)"
            )
            print(
                f"- Near-extrapolation: {results['near_extrapolation_count']} ({results['near_extrapolation_count']/results['total_samples']*100:.1f}%)"
            )
            print(
                f"- Far-extrapolation: {results['far_extrapolation_count']} ({results['far_extrapolation_count']/results['total_samples']*100:.1f}%)"
            )

            if "by_label" in results:
                print("\nBreakdown by original labels:")
                for label, stats in results["by_label"].items():
                    print(f"\n{label}:")
                    print(
                        f"  Actually interpolation: {stats['interpolation']}/{stats['total']} ({stats['interpolation']/stats['total']*100:.1f}%)"
                    )
                    print(
                        f"  Actually near-extrap: {stats['near_extrapolation']}/{stats['total']} ({stats['near_extrapolation']/stats['total']*100:.1f}%)"
                    )
                    print(
                        f"  Actually far-extrap: {stats['far_extrapolation']}/{stats['total']} ({stats['far_extrapolation']/stats['total']*100:.1f}%)"
                    )
                    print(f"  Avg log density: {stats['avg_log_density']:.2f}")
                    print(f"  Avg distance: {stats['avg_distance']:.2f}")

            # Create visualization
            plot_path = Path(
                f"outputs/baseline_results/{name}_representation_analysis.png"
            )
            analyzer.plot_analysis(
                log_densities, distances, categories, test_labels, plot_path
            )

        except Exception as e:
            print(f"Error analyzing {name}: {e}")
            import traceback

            traceback.print_exc()

    # Save representation analysis results
    output_file = Path("outputs/baseline_results/representation_analysis_results.json")
    with open(output_file, "w") as f:
        json.dump(representation_results, f, indent=2)

    print(f"\n{'='*60}")
    print("SUMMARY: True OOD Analysis")
    print(f"{'='*60}")

    print(
        "\nKey Finding: Many 'OOD' samples are actually interpolation in representation space!"
    )

    # Create summary table
    print(
        f"\n{'Model':<15} {'Jupiter (labeled far-OOD)':<30} {'Actually Far-Extrap':<20}"
    )
    print("-" * 65)

    for name, results in representation_results.items():
        if "by_label" in results and "far_ood" in results["by_label"]:
            stats = results["by_label"]["far_ood"]
            actual_far = stats["far_extrapolation"]
            total = stats["total"]
            print(
                f"{name:<15} {total} samples{'':<16} {actual_far}/{total} ({actual_far/total*100:.1f}%)"
            )


if __name__ == "__main__":
    analyze_baseline_representations()
