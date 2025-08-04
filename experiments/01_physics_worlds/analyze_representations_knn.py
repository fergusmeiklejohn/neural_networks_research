"""
Improved representation space analysis using k-NN distance metric
Addresses reviewer concern about convex hull in high dimensions
"""

import json
import sys
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.append("../../models")


class KNNRepresentationAnalyzer:
    """Analyze OOD using k-nearest neighbor distances"""

    def __init__(self, k_neighbors=10):
        """
        Args:
            k_neighbors: Number of nearest neighbors to consider
        """
        self.k = k_neighbors
        self.scaler = StandardScaler()

    def extract_representations(self, model, data):
        """Extract representations from penultimate layer"""
        # Create a model that outputs the second-to-last layer
        representation_model = keras.Model(
            inputs=model.input, outputs=model.layers[-2].output
        )

        # Extract representations
        representations = []
        batch_size = 32

        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            reps = representation_model.predict(batch, verbose=0)
            representations.append(reps)

        return np.vstack(representations)

    def analyze_ood_with_knn(self, train_reps, test_reps, test_labels):
        """
        Analyze OOD using k-NN distances

        Returns:
            dict: Analysis results including distances and thresholds
        """
        # Normalize representations
        train_reps_norm = self.scaler.fit_transform(train_reps)
        test_reps_norm = self.scaler.transform(test_reps)

        # Fit k-NN on training data
        knn = NearestNeighbors(n_neighbors=self.k, metric="euclidean")
        knn.fit(train_reps_norm)

        # Compute distances for test samples
        distances, indices = knn.kneighbors(test_reps_norm)
        mean_distances = distances.mean(axis=1)

        # Compute threshold using training data self-distances
        train_distances, _ = knn.kneighbors(train_reps_norm)
        train_mean_distances = train_distances.mean(axis=1)

        # Use 95th percentile of training distances as threshold
        threshold_95 = np.percentile(train_mean_distances, 95)
        threshold_99 = np.percentile(train_mean_distances, 99)

        # Classify samples
        results = {
            "total_samples": len(test_reps),
            "k_neighbors": self.k,
            "threshold_95": float(threshold_95),
            "threshold_99": float(threshold_99),
            "by_label": {},
        }

        # Analyze by label
        unique_labels = np.unique(test_labels)
        for label in unique_labels:
            mask = test_labels == label
            label_distances = mean_distances[mask]

            results["by_label"][label] = {
                "total": int(np.sum(mask)),
                "within_95": int(np.sum(label_distances <= threshold_95)),
                "within_99": int(np.sum(label_distances <= threshold_99)),
                "beyond_99": int(np.sum(label_distances > threshold_99)),
                "mean_distance": float(np.mean(label_distances)),
                "median_distance": float(np.median(label_distances)),
                "max_distance": float(np.max(label_distances)),
            }

        # Overall statistics
        results["overall"] = {
            "within_95": int(np.sum(mean_distances <= threshold_95)),
            "within_99": int(np.sum(mean_distances <= threshold_99)),
            "beyond_99": int(np.sum(mean_distances > threshold_99)),
            "pct_within_95": float(
                100 * np.sum(mean_distances <= threshold_95) / len(mean_distances)
            ),
            "pct_within_99": float(
                100 * np.sum(mean_distances <= threshold_99) / len(mean_distances)
            ),
        }

        return results, mean_distances

    def plot_distance_distribution(
        self, distances, labels, threshold_95, threshold_99, save_path
    ):
        """Plot distribution of k-NN distances by label"""
        plt.figure(figsize=(10, 6))

        # Create violin plot
        unique_labels = np.unique(labels)
        label_names = {
            "in_distribution": "In-Distribution",
            "near_ood": "Near-OOD",
            "far_ood": "Far-OOD (Jupiter)",
        }

        data_for_plot = []
        labels_for_plot = []

        for label in unique_labels:
            mask = labels == label
            data_for_plot.extend(distances[mask])
            labels_for_plot.extend([label_names.get(label, label)] * np.sum(mask))

        # Create violin plot
        ax = sns.violinplot(x=labels_for_plot, y=data_for_plot, inner="box")

        # Add threshold lines
        ax.axhline(
            y=threshold_95,
            color="orange",
            linestyle="--",
            label="95th percentile of training",
            linewidth=2,
        )
        ax.axhline(
            y=threshold_99,
            color="red",
            linestyle="--",
            label="99th percentile of training",
            linewidth=2,
        )

        plt.ylabel("Mean k-NN Distance", fontsize=12)
        plt.xlabel("Distribution Type", fontsize=12)
        plt.title(f"k-NN Distance Analysis (k={self.k})", fontsize=14)
        plt.legend()
        plt.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


def extract_trajectory_features(data_list):
    """Extract trajectory features from list of data dictionaries"""
    trajectories = []

    for sample in data_list:
        traj_array = np.array(sample["trajectory"])
        if traj_array.shape[1] >= 17:  # Has enough features for 2 balls
            # Extract core features: positions and velocities
            x1 = traj_array[:, 1:2]
            y1 = traj_array[:, 2:3]
            vx1 = traj_array[:, 3:4]
            vy1 = traj_array[:, 4:5]
            x2 = traj_array[:, 9:10]
            y2 = traj_array[:, 10:11]
            vx2 = traj_array[:, 11:12]
            vy2 = traj_array[:, 12:13]

            core_features = np.concatenate([x1, y1, x2, y2, vx1, vy1, vx2, vy2], axis=1)

            # Truncate to 50 timesteps
            if core_features.shape[0] > 50:
                core_features = core_features[:50]
            elif core_features.shape[0] < 50:
                padding = np.zeros((50 - core_features.shape[0], 8))
                core_features = np.concatenate([core_features, padding], axis=0)

            # Flatten for input
            trajectories.append(core_features.flatten())

    return np.array(trajectories)


def run_knn_analysis():
    """Run k-NN analysis on all models"""

    # Load test data
    import pickle

    with open(
        "data/processed/physics_worlds_v2/test_interpolation_data.pkl", "rb"
    ) as f:
        test_interp_data = pickle.load(f)
    with open(
        "data/processed/physics_worlds_v2/test_extrapolation_data.pkl", "rb"
    ) as f:
        test_extrap_data = pickle.load(f)
    with open("data/processed/physics_worlds_v2/test_novel_data.pkl", "rb") as f:
        test_novel_data = pickle.load(f)

    # Load training data for representations
    with open("data/processed/physics_worlds_v2/train_data.pkl", "rb") as f:
        train_data = pickle.load(f)

    # Extract features
    print("Extracting trajectory features...")
    X_train = extract_trajectory_features(train_data)
    X_test_interp = extract_trajectory_features(test_interp_data)
    X_test_extrap = extract_trajectory_features(test_extrap_data)
    X_test_novel = extract_trajectory_features(test_novel_data)

    print(f"Train shape: {X_train.shape}")
    print(f"Test interpolation shape: {X_test_interp.shape}")
    print(f"Test extrapolation shape: {X_test_extrap.shape}")
    print(f"Test novel shape: {X_test_novel.shape}")

    # Combine test data
    X_test = np.vstack([X_test_interp, X_test_extrap, X_test_novel])

    # Create labels
    test_labels = np.array(
        ["in_distribution"] * len(X_test_interp)
        + ["near_ood"] * len(X_test_extrap)
        + ["far_ood"] * len(X_test_novel)
    )

    # Models to analyze - only use models that exist
    models = {
        "GFlowNet": "outputs/baseline_results/gflownet_model_20250715_062359.keras",
        "MAML": "outputs/baseline_results/maml_model_20250715_062721.keras",
    }

    # Check which models actually exist
    existing_models = {}
    for model_name, model_path in models.items():
        if Path(model_path).exists():
            existing_models[model_name] = model_path
        else:
            print(f"Warning: {model_name} model not found at {model_path}")

    models = existing_models

    analyzer = KNNRepresentationAnalyzer(k_neighbors=10)
    all_results = {}

    for model_name, model_path in models.items():
        print(f"\nAnalyzing {model_name}...")

        try:
            # Load model
            model = keras.models.load_model(model_path, compile=False)

            # Extract representations
            train_reps = analyzer.extract_representations(model, X_train)
            test_reps = analyzer.extract_representations(model, X_test)

            # Run analysis
            results, distances = analyzer.analyze_ood_with_knn(
                train_reps, test_reps, test_labels
            )

            all_results[model_name] = results

            # Create visualization
            analyzer.plot_distance_distribution(
                distances,
                test_labels,
                results["threshold_95"],
                results["threshold_99"],
                f"outputs/baseline_results/{model_name}_knn_distances.png",
            )

            # Print summary
            print(f"Results for {model_name}:")
            print(
                f"  Within 95th percentile: {results['overall']['pct_within_95']:.1f}%"
            )
            print(
                f"  Within 99th percentile: {results['overall']['pct_within_99']:.1f}%"
            )
            print(
                f"  Beyond 99th percentile: {100 - results['overall']['pct_within_99']:.1f}%"
            )

        except Exception as e:
            print(f"Error analyzing {model_name}: {e}")
            continue

    # Save results
    with open("outputs/baseline_results/knn_analysis_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


if __name__ == "__main__":
    results = run_knn_analysis()

    # Create comparison table
    print("\n" + "=" * 60)
    print("k-NN Distance Analysis Summary")
    print("=" * 60)
    print(f"{'Model':<15} {'Within 95%':<12} {'Within 99%':<12} {'True OOD':<12}")
    print("-" * 60)

    for model, res in results.items():
        if "overall" in res:
            pct_95 = res["overall"]["pct_within_95"]
            pct_99 = res["overall"]["pct_within_99"]
            pct_ood = 100 - pct_99
            print(f"{model:<15} {pct_95:>10.1f}% {pct_99:>10.1f}% {pct_ood:>10.1f}%")
