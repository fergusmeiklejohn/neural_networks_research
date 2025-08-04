"""
True OOD Verifier using Convex Hull Analysis

Based on Li et al. (2025) - "Most OOD benchmarks test interpolation"

This tool analyzes whether test samples are truly out-of-distribution (extrapolation)
or just interpolation within the convex hull of training representations.

Key insight: Many claimed "OOD" successes are actually interpolation when analyzed
in the model's learned representation space.
"""

from utils.imports import setup_project_paths

setup_project_paths()

import json
import logging
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils.config import setup_environment

# Set up environment
config = setup_environment()
logger = logging.getLogger(__name__)


class TrueOODVerifier:
    """
    Verifies whether test samples are truly OOD using convex hull analysis.

    Based on Li et al. (2025) - Most "OOD" is actually interpolation
    in the learned representation space.
    """

    def __init__(
        self,
        use_pca: bool = True,
        pca_components: int = 50,
        distance_thresholds: Dict[str, float] = None,
    ):
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.pca = None
        self.scaler = StandardScaler()

        # Distance thresholds for categorization
        if distance_thresholds is None:
            self.distance_thresholds = {
                "interpolation": 0.01,  # Within hull + small margin
                "near_extrapolation": 0.1,  # Close to hull
                "far_extrapolation": 0.1,  # Far from hull (> threshold)
            }
        else:
            self.distance_thresholds = distance_thresholds

        self.hull = None
        self.delaunay = None  # For efficient point-in-hull tests

    def fit_convex_hull(self, train_representations: np.ndarray):
        """
        Fit convex hull to training representations.

        Args:
            train_representations: (n_samples, n_features) array
        """
        logger.info(
            f"Fitting convex hull to {train_representations.shape[0]} training samples"
        )

        # Standardize features
        train_scaled = self.scaler.fit_transform(train_representations)

        # Apply PCA if requested
        if self.use_pca and train_representations.shape[1] > self.pca_components:
            self.pca = PCA(n_components=self.pca_components)
            train_reduced = self.pca.fit_transform(train_scaled)
            logger.info(
                f"Reduced dimensions from {train_representations.shape[1]} to {self.pca_components}"
            )
            logger.info(
                f"Explained variance: {self.pca.explained_variance_ratio_.sum():.3f}"
            )
        else:
            train_reduced = train_scaled

        # Compute convex hull
        try:
            self.hull = ConvexHull(train_reduced)
            self.delaunay = Delaunay(train_reduced)
            logger.info(f"Convex hull computed with {len(self.hull.vertices)} vertices")
        except Exception as e:
            logger.error(f"Failed to compute convex hull: {e}")
            raise

        self.train_representations = train_reduced

    def _transform_representations(self, representations: np.ndarray) -> np.ndarray:
        """Apply same transformations as training data."""
        scaled = self.scaler.transform(representations)

        if self.pca is not None:
            return self.pca.transform(scaled)
        else:
            return scaled

    def _distance_to_hull(self, point: np.ndarray) -> float:
        """
        Compute distance from point to convex hull.

        Negative distance means inside hull, positive means outside.
        """
        # First check if point is inside hull
        if self.delaunay.find_simplex(point) >= 0:
            # Point is inside - compute negative distance to boundary
            # This is approximate - true signed distance is complex to compute
            return -0.001  # Small negative value for points inside

        # Point is outside - compute distance to closest facet
        # For each facet (simplex), compute distance
        min_distance = float("inf")

        for simplex in self.hull.simplices:
            # Get vertices of this facet
            facet_points = self.train_representations[simplex]

            # Compute distance to facet (simplified - uses centroid)
            facet_center = facet_points.mean(axis=0)
            distance = np.linalg.norm(point - facet_center)

            min_distance = min(min_distance, distance)

        return min_distance

    def analyze_dataset(
        self,
        train_data: Any,
        test_data: Any,
        model: Any,
        representation_layer: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze whether test data is truly OOD.

        Args:
            train_data: Training data (format depends on model)
            test_data: Test data to analyze
            model: Model with method to extract representations
            representation_layer: Which layer to use for representations

        Returns:
            Dictionary with categorized results and statistics
        """
        logger.info("Extracting representations...")

        # Get representations from model
        if hasattr(model, "get_representations"):
            train_repr = model.get_representations(
                train_data, layer=representation_layer
            )
            test_repr = model.get_representations(test_data, layer=representation_layer)
        elif hasattr(model, "encode"):
            train_repr = model.encode(train_data)
            test_repr = model.encode(test_data)
        else:
            raise ValueError("Model must have 'get_representations' or 'encode' method")

        # Fit convex hull to training data
        self.fit_convex_hull(train_repr)

        # Transform test representations
        test_transformed = self._transform_representations(test_repr)

        # Analyze each test point
        results = {
            "interpolation": [],
            "near_extrapolation": [],
            "far_extrapolation": [],
            "distances": [],
            "indices": {
                "interpolation": [],
                "near_extrapolation": [],
                "far_extrapolation": [],
            },
        }

        for i, point in enumerate(test_transformed):
            distance = self._distance_to_hull(point)
            results["distances"].append(distance)

            # Categorize based on distance
            if distance < self.distance_thresholds["interpolation"]:
                category = "interpolation"
            elif distance < self.distance_thresholds["near_extrapolation"]:
                category = "near_extrapolation"
            else:
                category = "far_extrapolation"

            results[category].append(point)
            results["indices"][category].append(i)

        # Compute statistics
        n_test = len(test_transformed)
        stats = {
            "n_interpolation": len(results["interpolation"]),
            "n_near_extrapolation": len(results["near_extrapolation"]),
            "n_far_extrapolation": len(results["far_extrapolation"]),
            "pct_interpolation": len(results["interpolation"]) / n_test * 100,
            "pct_near_extrapolation": len(results["near_extrapolation"]) / n_test * 100,
            "pct_far_extrapolation": len(results["far_extrapolation"]) / n_test * 100,
            "mean_distance": np.mean(results["distances"]),
            "median_distance": np.median(results["distances"]),
            "max_distance": np.max(results["distances"]),
            "n_inside_hull": sum(1 for d in results["distances"] if d < 0),
        }

        results["statistics"] = stats

        logger.info("\nOOD Analysis Results:")
        logger.info(
            f"  Interpolation: {stats['n_interpolation']} ({stats['pct_interpolation']:.1f}%)"
        )
        logger.info(
            f"  Near-extrapolation: {stats['n_near_extrapolation']} ({stats['pct_near_extrapolation']:.1f}%)"
        )
        logger.info(
            f"  Far-extrapolation: {stats['n_far_extrapolation']} ({stats['pct_far_extrapolation']:.1f}%)"
        )
        logger.info(f"  Points inside hull: {stats['n_inside_hull']}")

        return results

    def visualize_analysis(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None,
        show_hull: bool = True,
        max_dims: int = 3,
    ):
        """
        Visualize the OOD analysis results.

        Args:
            results: Results from analyze_dataset
            save_path: Path to save figure
            show_hull: Whether to show convex hull
            max_dims: Maximum dimensions to plot (2 or 3)
        """
        if self.train_representations.shape[1] > max_dims:
            # Further reduce dimensions for visualization
            viz_pca = PCA(n_components=max_dims)
            train_viz = viz_pca.fit_transform(self.train_representations)

            # Transform test points
            test_viz = {
                category: viz_pca.transform(np.array(points))
                if len(points) > 0
                else np.array([])
                for category, points in results.items()
                if category
                in ["interpolation", "near_extrapolation", "far_extrapolation"]
            }
        else:
            train_viz = self.train_representations
            test_viz = {
                category: np.array(points) if len(points) > 0 else np.array([])
                for category, points in results.items()
                if category
                in ["interpolation", "near_extrapolation", "far_extrapolation"]
            }

        # Create figure
        fig = plt.figure(figsize=(12, 10))

        if max_dims == 2 or train_viz.shape[1] == 2:
            ax = fig.add_subplot(111)

            # Plot training points
            ax.scatter(
                train_viz[:, 0],
                train_viz[:, 1],
                c="lightgray",
                alpha=0.3,
                s=20,
                label="Training",
            )

            # Plot convex hull
            if show_hull and self.hull is not None:
                for simplex in self.hull.simplices:
                    ax.plot(
                        train_viz[simplex, 0], train_viz[simplex, 1], "k-", alpha=0.3
                    )

            # Plot test points by category
            colors = {
                "interpolation": "green",
                "near_extrapolation": "orange",
                "far_extrapolation": "red",
            }

            for category, points in test_viz.items():
                if len(points) > 0:
                    ax.scatter(
                        points[:, 0],
                        points[:, 1],
                        c=colors[category],
                        alpha=0.7,
                        s=50,
                        label=f"{category} ({len(points)})",
                    )

            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")

        else:  # 3D plot
            ax = fig.add_subplot(111, projection="3d")

            # Plot training points
            ax.scatter(
                train_viz[:, 0],
                train_viz[:, 1],
                train_viz[:, 2],
                c="lightgray",
                alpha=0.3,
                s=20,
                label="Training",
            )

            # Plot test points by category
            colors = {
                "interpolation": "green",
                "near_extrapolation": "orange",
                "far_extrapolation": "red",
            }

            for category, points in test_viz.items():
                if len(points) > 0:
                    ax.scatter(
                        points[:, 0],
                        points[:, 1],
                        points[:, 2],
                        c=colors[category],
                        alpha=0.7,
                        s=50,
                        label=f"{category} ({len(points)})",
                    )

            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_zlabel("PC3")

        ax.legend()
        ax.set_title("OOD Analysis: Interpolation vs Extrapolation")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Figure saved to {save_path}")

        plt.show()

    def generate_report(self, results: Dict[str, Any], output_path: str):
        """Generate detailed analysis report."""
        report = {
            "analysis_type": "convex_hull_ood_verification",
            "configuration": {
                "use_pca": self.use_pca,
                "pca_components": self.pca_components if self.use_pca else None,
                "distance_thresholds": self.distance_thresholds,
            },
            "results": {
                "statistics": results["statistics"],
                "distance_distribution": {
                    "min": float(np.min(results["distances"])),
                    "max": float(np.max(results["distances"])),
                    "mean": float(np.mean(results["distances"])),
                    "std": float(np.std(results["distances"])),
                    "percentiles": {
                        "25": float(np.percentile(results["distances"], 25)),
                        "50": float(np.percentile(results["distances"], 50)),
                        "75": float(np.percentile(results["distances"], 75)),
                        "95": float(np.percentile(results["distances"], 95)),
                    },
                },
            },
            "interpretation": self._generate_interpretation(results),
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report saved to {output_path}")

    def _generate_interpretation(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Generate human-readable interpretation of results."""
        stats = results["statistics"]

        interpretation = {}

        # Overall assessment
        if stats["pct_interpolation"] > 80:
            interpretation["overall"] = (
                "The vast majority of 'OOD' test samples are actually interpolation. "
                "The model is not truly extrapolating to new regions."
            )
        elif stats["pct_far_extrapolation"] > 50:
            interpretation["overall"] = (
                "Many test samples require true extrapolation. "
                "This is a genuinely challenging OOD scenario."
            )
        else:
            interpretation["overall"] = (
                "The test set contains a mix of interpolation and extrapolation. "
                "Care should be taken when interpreting performance."
            )

        # Specific insights
        interpretation["interpolation_insight"] = (
            f"{stats['pct_interpolation']:.1f}% of test samples fall within or very close to "
            f"the convex hull of training representations. Performance on these should be "
            f"similar to in-distribution performance."
        )

        interpretation["extrapolation_insight"] = (
            f"{stats['pct_far_extrapolation']:.1f}% of test samples require significant "
            f"extrapolation. These truly test the model's ability to generalize beyond "
            f"its training distribution."
        )

        return interpretation


def test_verifier_on_simple_data():
    """Test the verifier on simple 2D data for visualization."""
    np.random.seed(42)

    # Create simple 2D training data (two clusters)
    cluster1 = np.random.randn(50, 2) + np.array([0, 0])
    cluster2 = np.random.randn(50, 2) + np.array([5, 0])
    train_data = np.vstack([cluster1, cluster2])

    # Create test data with different categories
    test_interpolation = np.random.randn(20, 2) + np.array([2.5, 0])  # Between clusters
    test_near = np.random.randn(20, 2) + np.array([7, 0])  # Just outside
    test_far = np.random.randn(20, 2) + np.array([10, 0])  # Far outside
    test_data = np.vstack([test_interpolation, test_near, test_far])

    # Create dummy model that returns data as-is
    class DummyModel:
        def get_representations(self, data, layer=None):
            return data

    model = DummyModel()

    # Run analysis
    verifier = TrueOODVerifier(use_pca=False)
    results = verifier.analyze_dataset(train_data, test_data, model)

    # Visualize
    verifier.visualize_analysis(results, show_hull=True, max_dims=2)

    return results


if __name__ == "__main__":
    logger.info("Testing True OOD Verifier...")
    results = test_verifier_on_simple_data()
    logger.info("\nTest complete!")
