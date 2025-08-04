"""
Generate publication-quality figures for OOD evaluation paper
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set publication-quality defaults
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.figsize"] = (8, 6)

# Set color palette
colors = sns.color_palette("husl", 4)
model_colors = {
    "ERM+Aug": colors[0],
    "GFlowNet": colors[1],
    "GraphExtrap": colors[2],
    "MAML": colors[3],
}


def load_results():
    """Load experimental results"""
    # Load k-NN analysis results
    with open(
        "../../experiments/01_physics_worlds/outputs/baseline_results/knn_analysis_results.json",
        "r",
    ) as f:
        knn_results = json.load(f)

    # Load baseline results
    with open(
        "../../experiments/01_physics_worlds/outputs/baseline_results/all_baselines_results.json",
        "r",
    ) as f:
        baseline_results = json.load(f)

    return knn_results, baseline_results


def create_interpolation_bar_chart(knn_results):
    """Create bar chart showing k-NN distance analysis percentages"""
    fig, ax = plt.subplots(figsize=(10, 6))

    models = list(knn_results.keys())
    within_95_pcts = []
    within_99_pcts = []
    beyond_99_pcts = []

    for model in models:
        overall = knn_results[model]["overall"]
        within_95_pcts.append(overall["pct_within_95"])
        within_99_pcts.append(overall["pct_within_99"])
        beyond_99_pcts.append(100 - overall["pct_within_99"])

    x = np.arange(len(models))
    width = 0.25

    bars1 = ax.bar(
        x - width,
        within_95_pcts,
        width,
        label="Within 95th percentile",
        color="skyblue",
        edgecolor="black",
        linewidth=1,
    )
    bars2 = ax.bar(
        x,
        within_99_pcts,
        width,
        label="Within 99th percentile",
        color="lightgreen",
        edgecolor="black",
        linewidth=1,
    )
    bars3 = ax.bar(
        x + width,
        beyond_99_pcts,
        width,
        label="Beyond 99th percentile",
        color="lightcoral",
        edgecolor="black",
        linewidth=1,
    )

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Percentage of Test Samples", fontsize=12)
    ax.set_title("k-NN Distance Analysis: Distribution of Test Samples", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("figure1_interpolation_analysis.pdf", bbox_inches="tight")
    plt.savefig("figure1_interpolation_analysis.png", bbox_inches="tight")
    plt.close()


def create_performance_comparison(baseline_results):
    """Create log-scale performance comparison"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Data for plotting
    models = [
        "GraphExtrap\n(published)",
        "MAML\n(published)",
        "GFlowNet\n(published)",
        "ERM+Aug\n(published)",
        "GFlowNet\n(our test)",
        "MAML\n(our test)",
        "Standard PINN",
        "Minimal PINN",
    ]

    mse_values = [0.766, 0.823, 0.850, 1.128, 2229.38, 3298.69, 880.879, 42532.14]

    # Color code by category
    colors = ["green", "green", "green", "green", "orange", "orange", "red", "darkred"]

    # Create horizontal bar chart
    y_pos = np.arange(len(models))
    bars = ax.barh(y_pos, mse_values, color=colors, edgecolor="black", linewidth=1)

    # Use log scale
    ax.set_xscale("log")

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, mse_values)):
        if val < 10:
            label = f"{val:.3f}"
        elif val < 1000:
            label = f"{val:.1f}"
        else:
            label = f"{val:,.0f}"

        ax.text(val * 1.5, i, label, va="center", fontsize=10)

    # Add vertical lines for reference
    ax.axvline(x=1, color="gray", linestyle="--", alpha=0.5, label="MSE = 1")
    ax.axvline(x=1000, color="gray", linestyle="--", alpha=0.5, label="MSE = 1000")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlabel("Mean Squared Error (log scale)", fontsize=12)
    ax.set_title("Performance Comparison: Published vs Reproduced Results", fontsize=14)
    ax.set_xlim(0.1, 100000)
    ax.grid(axis="x", alpha=0.3)

    # Add legend for color coding
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="green", edgecolor="black", label="Published results"),
        Patch(facecolor="orange", edgecolor="black", label="Our reproductions"),
        Patch(facecolor="red", edgecolor="black", label="Physics-informed models"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    plt.savefig("figure2_performance_comparison.pdf", bbox_inches="tight")
    plt.savefig("figure2_performance_comparison.png", bbox_inches="tight")
    plt.close()


def create_distribution_breakdown(knn_results):
    """Create detailed breakdown by distribution type"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Prepare data
    models = list(knn_results.keys())
    dist_types = ["in_distribution", "near_ood", "far_ood"]
    dist_labels = ["In-Distribution", "Near-OOD", "Far-OOD (Jupiter)"]

    # Plot 1: Within 99th percentile rates by distribution
    x = np.arange(len(dist_labels))
    width = 0.2

    for i, model in enumerate(models):
        within_99_rates = []
        for dist in dist_types:
            total = knn_results[model]["by_label"][dist]["total"]
            within_99 = knn_results[model]["by_label"][dist]["within_99"]
            rate = 100 * within_99 / total
            within_99_rates.append(rate)

        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax1.bar(
            x + offset, within_99_rates, width, label=model, color=model_colors[model]
        )

    ax1.set_xlabel("Distribution Type", fontsize=12)
    ax1.set_ylabel("Within 99th Percentile (%)", fontsize=12)
    ax1.set_title("k-NN Distance Analysis by Distribution Type", fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(dist_labels)
    ax1.legend()
    ax1.set_ylim(85, 96)
    ax1.grid(axis="y", alpha=0.3)

    # Plot 2: Mean k-NN distances
    for i, model in enumerate(models):
        distances = []
        for dist in dist_types:
            mean_dist = knn_results[model]["by_label"][dist]["mean_distance"]
            distances.append(mean_dist)

        ax2.plot(
            dist_labels,
            distances,
            "o-",
            label=model,
            color=model_colors[model],
            linewidth=2,
            markersize=8,
        )

    ax2.set_xlabel("Distribution Type", fontsize=12)
    ax2.set_ylabel("Mean k-NN Distance", fontsize=12)
    ax2.set_title("k-NN Distance Analysis", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figure3_distribution_breakdown.pdf", bbox_inches="tight")
    plt.savefig("figure3_distribution_breakdown.png", bbox_inches="tight")
    plt.close()


def create_conceptual_diagram():
    """Create conceptual diagram of interpolation vs extrapolation"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Generate synthetic data for visualization
    np.random.seed(42)

    # Plot 1: Input space
    # Training data (Earth and Mars gravity)
    earth_x = np.random.normal(-9.8, 0.5, 100)
    earth_y = np.random.normal(0, 1, 100)
    mars_x = np.random.normal(-3.7, 0.5, 100)
    mars_y = np.random.normal(0, 1, 100)

    # Test data (Jupiter)
    jupiter_x = np.random.normal(-24.8, 0.5, 50)
    jupiter_y = np.random.normal(0, 1, 50)

    ax1.scatter(earth_x, earth_y, c="blue", alpha=0.5, label="Earth (training)", s=30)
    ax1.scatter(mars_x, mars_y, c="red", alpha=0.5, label="Mars (training)", s=30)
    ax1.scatter(
        jupiter_x,
        jupiter_y,
        c="green",
        marker="^",
        s=50,
        label="Jupiter (test)",
        edgecolor="black",
        linewidth=1,
    )

    ax1.set_xlabel("Gravity (m/s²)", fontsize=12)
    ax1.set_ylabel("Other Physics Parameters", fontsize=12)
    ax1.set_title("(a) Input Space View", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-30, 0)

    # Plot 2: Representation space
    # Create overlapping distributions
    train_x = np.random.normal(0, 2, 200)
    train_y = np.random.normal(0, 2, 200)
    test_x = np.random.normal(0.5, 2, 50)  # Mostly overlapping
    test_y = np.random.normal(0.5, 2, 50)

    # Draw convex hull
    from scipy.spatial import ConvexHull

    train_points = np.column_stack([train_x, train_y])
    hull = ConvexHull(train_points)

    for simplex in hull.simplices:
        ax2.plot(train_points[simplex, 0], train_points[simplex, 1], "k-", alpha=0.3)

    ax2.scatter(train_x, train_y, c="gray", alpha=0.3, label="Training", s=30)
    ax2.scatter(
        test_x,
        test_y,
        c="green",
        marker="^",
        s=50,
        label="Test (91.7% inside hull)",
        edgecolor="black",
        linewidth=1,
    )

    ax2.set_xlabel("Representation Dimension 1", fontsize=12)
    ax2.set_ylabel("Representation Dimension 2", fontsize=12)
    ax2.set_title("(b) Learned Representation Space", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-6, 6)
    ax2.set_ylim(-6, 6)

    plt.tight_layout()
    plt.savefig("figure4_conceptual_diagram.pdf", bbox_inches="tight")
    plt.savefig("figure4_conceptual_diagram.png", bbox_inches="tight")
    plt.close()


def main():
    """Generate all figures"""
    print("Loading results...")
    knn_results, baseline_results = load_results()

    print("Creating Figure 1: k-NN analysis...")
    create_interpolation_bar_chart(knn_results)

    print("Creating Figure 2: Performance comparison...")
    create_performance_comparison(baseline_results)

    print("Creating Figure 3: Distribution breakdown...")
    create_distribution_breakdown(knn_results)

    print("Creating Figure 4: Conceptual diagram...")
    create_conceptual_diagram()

    print("\nAll figures generated successfully!")
    print("Files created:")
    print("- figure1_interpolation_analysis.pdf/png")
    print("- figure2_performance_comparison.pdf/png")
    print("- figure3_distribution_breakdown.pdf/png")
    print("- figure4_conceptual_diagram.pdf/png")


if __name__ == "__main__":
    main()
