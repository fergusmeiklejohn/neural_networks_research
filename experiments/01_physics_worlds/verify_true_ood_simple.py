"""Simple verification that time-varying gravity is truly OOD."""

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.imports import setup_project_paths

setup_project_paths()

from utils.config import setup_environment
from utils.paths import get_data_path, get_output_path


def extract_physics_features(trajectories):
    """Extract simple physics features from trajectories."""
    features = []

    for traj in trajectories:
        # Extract features from trajectory
        # 1. Average y-position (height)
        avg_y = np.mean(traj[:, [1, 5]])

        # 2. Average y-velocity
        avg_vy = np.mean(traj[:, [3, 7]])

        # 3. Y-position range
        y_range = np.max(traj[:, [1, 5]]) - np.min(traj[:, [1, 5]])

        # 4. Final y-position
        final_y = np.mean(traj[-1, [1, 5]])

        # 5. Maximum downward velocity
        max_down_v = np.min(traj[:, [3, 7]])

        # 6. Time to reach bottom (first time y < 20)
        y_positions = traj[:, [1, 5]].mean(axis=1)
        bottom_indices = np.where(y_positions < 20)[0]
        time_to_bottom = bottom_indices[0] if len(bottom_indices) > 0 else 50

        # 7. Total distance traveled
        total_dist = 0
        for t in range(1, len(traj)):
            for ball in [0, 1]:
                dx = traj[t, ball * 4] - traj[t - 1, ball * 4]
                dy = traj[t, ball * 4 + 1] - traj[t - 1, ball * 4 + 1]
                total_dist += np.sqrt(dx**2 + dy**2)

        # 8. Energy proxy (kinetic + potential)
        avg_kinetic = np.mean(traj[:, [2, 3, 6, 7]] ** 2)
        avg_potential = avg_y * 9.8  # Simplified

        features.append(
            [
                avg_y,
                avg_vy,
                y_range,
                final_y,
                max_down_v,
                time_to_bottom,
                total_dist,
                avg_kinetic,
                avg_potential,
            ]
        )

    return np.array(features)


def verify_ood_with_features(const_data, varying_data):
    """Verify OOD using physics features."""
    print("Extracting physics features...")

    # Extract features
    const_features = extract_physics_features(const_data["trajectories"])
    varying_features = extract_physics_features(varying_data["trajectories"])

    print(f"Constant gravity features shape: {const_features.shape}")
    print(f"Time-varying features shape: {varying_features.shape}")

    # Fit k-NN on constant gravity features
    print("\nFitting k-NN model...")
    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(const_features)

    # Get distances for training data
    train_distances, _ = knn.kneighbors(const_features)
    train_mean_dist = train_distances.mean(axis=1)

    # Set OOD threshold
    ood_threshold = np.percentile(train_mean_dist, 95)
    print(f"OOD threshold (95th percentile): {ood_threshold:.4f}")

    # Check time-varying gravity
    test_distances, _ = knn.kneighbors(varying_features)
    test_mean_dist = test_distances.mean(axis=1)

    # Classify as OOD
    is_ood = test_mean_dist > ood_threshold
    ood_percentage = np.mean(is_ood) * 100

    print(f"\nResults:")
    print(f"Time-varying gravity OOD percentage: {ood_percentage:.1f}%")
    print(f"Mean distance (constant g): {train_mean_dist.mean():.4f}")
    print(f"Mean distance (varying g): {test_mean_dist.mean():.4f}")
    print(f"Distance ratio: {test_mean_dist.mean() / train_mean_dist.mean():.2f}x")

    # Visualize with PCA
    plt.figure(figsize=(10, 8))

    # Combine and apply PCA
    all_features = np.vstack([const_features, varying_features])
    pca = PCA(n_components=2)
    all_pca = pca.fit_transform(all_features)

    # Split back
    n_const = len(const_features)
    const_pca = all_pca[:n_const]
    varying_pca = all_pca[n_const:]

    # Plot
    plt.scatter(
        const_pca[:, 0],
        const_pca[:, 1],
        c="blue",
        alpha=0.6,
        label="Constant gravity",
        s=50,
    )

    # Color varying gravity by OOD status
    plt.scatter(
        varying_pca[~is_ood, 0],
        varying_pca[~is_ood, 1],
        c="green",
        alpha=0.6,
        label=f"Varying g (interpolation: {100-ood_percentage:.0f}%)",
        s=50,
        marker="^",
    )
    plt.scatter(
        varying_pca[is_ood, 0],
        varying_pca[is_ood, 1],
        c="red",
        alpha=0.6,
        label=f"Varying g (true OOD: {ood_percentage:.0f}%)",
        s=50,
        marker="s",
    )

    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.title("Physics Feature Space: Constant vs Time-Varying Gravity")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add explained variance
    var_explained = pca.explained_variance_ratio_
    plt.text(
        0.02,
        0.98,
        f"Variance explained: {var_explained[0]:.1%}, {var_explained[1]:.1%}",
        transform=plt.gca().transAxes,
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Save plot
    plot_path = get_output_path() / "ood_verification" / "feature_space_simple.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.show()

    return {
        "ood_percentage": ood_percentage,
        "threshold": ood_threshold,
        "const_distances": train_mean_dist,
        "varying_distances": test_mean_dist,
        "is_ood": is_ood,
    }


def main():
    """Main verification function."""
    print("Simple OOD Verification for Time-Varying Gravity")
    print("=" * 60)

    # Load data
    data_dir = get_data_path() / "true_ood_physics"

    # Find most recent files
    const_files = list(data_dir.glob("constant_gravity_*.pkl"))
    varying_files = list(data_dir.glob("time_varying_gravity_*.pkl"))

    if not const_files or not varying_files:
        print("Error: Could not find data files.")
        return

    # Load most recent
    const_file = sorted(const_files)[-1]
    varying_file = sorted(varying_files)[-1]

    print(f"Loading: {const_file.name}")
    with open(const_file, "rb") as f:
        const_data = pickle.load(f)

    print(f"Loading: {varying_file.name}")
    with open(varying_file, "rb") as f:
        varying_data = pickle.load(f)

    # Verify OOD
    results = verify_ood_with_features(const_data, varying_data)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Physics type comparison:")
    print(f"  Training: Constant gravity (g = -392 pixels/s²)")
    print(f"  Test: Time-varying gravity (g(t) = -392 * (1 + 0.3*sin(0.5*t)))")
    print(f"\nTrue OOD percentage: {results['ood_percentage']:.1f}%")

    if results["ood_percentage"] > 50:
        print("✓ SUCCESS: Majority of test samples are truly out-of-distribution!")
    else:
        print("✗ WARNING: Most test samples are still within training distribution")
        print("\nThis suggests we need more extreme physics modifications.")

    return results


if __name__ == "__main__":
    config = setup_environment()
    results = main()
