#!/usr/bin/env python3
"""
Diagnostic analysis of why imagination approaches underperform.
Analyzes the gap between V7 baseline (66.6%) and imagination-augmented (45.5%).
"""

from utils.imports import setup_project_paths

setup_project_paths()

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Import the solvers
from enhanced_arc_solver_v7 import EnhancedARCSolverV7
from final_benchmark_test import WorkingAugmentedV7
from fix_invention_solver import FixedDistributionInventionSolver


@dataclass
class FailureAnalysis:
    """Analysis of a single failure case."""

    task_id: str
    v7_accuracy: float
    imagination_accuracy: float
    hypothesis_count: int
    hypothesis_diversity: float
    constraint_violations: List[str]
    failure_mode: str


class ImaginationDiagnostics:
    """Analyzes why imagination approaches fail."""

    def __init__(self):
        self.v7_solver = EnhancedARCSolverV7(
            use_synthesis=True, use_position_learning=True, confidence_threshold=0.85
        )
        self.imagination_solver = FixedDistributionInventionSolver(invention_budget=30)
        self.augmented_solver = WorkingAugmentedV7()

    def analyze_hypothesis_quality(
        self,
        hypotheses: List[np.ndarray],
        test_input: np.ndarray,
        expected_output: np.ndarray,
    ) -> Dict:
        """Analyze the quality of generated hypotheses."""
        if not hypotheses:
            return {
                "count": 0,
                "diversity": 0.0,
                "best_accuracy": 0.0,
                "constraint_violations": ["no_hypotheses_generated"],
            }

        # Calculate diversity (how different hypotheses are from each other)
        diversity_scores = []
        for i, h1 in enumerate(hypotheses):
            for h2 in hypotheses[i + 1 :]:
                if h1.shape == h2.shape:
                    diversity_scores.append(np.mean(h1 != h2))

        diversity = np.mean(diversity_scores) if diversity_scores else 0.0

        # Check constraint violations
        violations = []
        input_colors = set(test_input.flatten())

        for hypothesis in hypotheses:
            # Check if sizes match expected
            if hypothesis.shape != expected_output.shape:
                violations.append("size_mismatch")

            # Check if new colors introduced
            hypothesis_colors = set(hypothesis.flatten())
            new_colors = hypothesis_colors - input_colors - {0}
            if new_colors and len(new_colors) > 2:  # Allow some new colors
                violations.append("too_many_new_colors")

            # Check if pattern is too random (high entropy)
            if hypothesis.size > 0:
                unique_ratio = len(np.unique(hypothesis)) / hypothesis.size
                if unique_ratio > 0.8:  # Too many unique values
                    violations.append("too_random")

        # Calculate best accuracy
        best_accuracy = 0.0
        for hypothesis in hypotheses:
            if hypothesis.shape == expected_output.shape:
                accuracy = np.mean(hypothesis == expected_output)
                best_accuracy = max(best_accuracy, accuracy)

        return {
            "count": len(hypotheses),
            "diversity": float(diversity),
            "best_accuracy": float(best_accuracy),
            "constraint_violations": list(set(violations)),
        }

    def identify_failure_mode(
        self, v7_accuracy: float, imagination_accuracy: float, hypothesis_analysis: Dict
    ) -> str:
        """Identify the primary failure mode."""
        if hypothesis_analysis["count"] == 0:
            return "no_hypotheses_generated"

        if "size_mismatch" in hypothesis_analysis["constraint_violations"]:
            return "incorrect_size_prediction"

        if "too_random" in hypothesis_analysis["constraint_violations"]:
            return "excessive_randomness"

        if hypothesis_analysis["diversity"] < 0.1:
            return "insufficient_diversity"

        if hypothesis_analysis["best_accuracy"] > imagination_accuracy:
            return "poor_hypothesis_selection"

        if v7_accuracy > 0.7 and imagination_accuracy < 0.3:
            return "unnecessary_imagination"

        return "general_quality_issue"

    def analyze_task(self, task_file: str) -> Optional[FailureAnalysis]:
        """Analyze a single task."""
        try:
            # Load task
            with open(task_file, "r") as f:
                task = json.load(f)

            # Get examples and test
            examples = [
                (np.array(ex["input"]), np.array(ex["output"])) for ex in task["train"]
            ]
            test_input = np.array(task["test"][0]["input"])
            test_output = np.array(task["test"][0]["output"])

            # Test V7 baseline
            v7_result = self.v7_solver.solve(examples, test_input)
            if hasattr(v7_result, "output_grid"):
                v7_result = v7_result.output_grid
            v7_accuracy = (
                np.mean(v7_result == test_output) if v7_result is not None else 0.0
            )

            # Generate imagination hypotheses for analysis
            hints = {"task_type": "unknown", "confidence": 0.3}
            hypotheses = []

            # Try to get hypotheses from the imagination solver
            unique_test = sorted(set(test_input.flatten()) - {0})
            if unique_test:
                distributions = self.imagination_solver.invent_distributions(
                    test_input, hints
                )

                # Convert distributions to hypotheses (simplified)
                for dist in distributions[:10]:  # Analyze first 10
                    # This is a simplified hypothesis generation
                    hypothesis = np.zeros_like(test_output)
                    # Apply distribution as a simple pattern
                    for i, val in enumerate(dist):
                        if i < hypothesis.size:
                            hypothesis.flat[i] = val
                    hypotheses.append(hypothesis)

            # Analyze hypotheses
            hypothesis_analysis = self.analyze_hypothesis_quality(
                hypotheses, test_input, test_output
            )

            # Test augmented solver
            augmented_result = self.augmented_solver.solve(examples, test_input)
            if hasattr(augmented_result, "output_grid"):
                augmented_result = augmented_result.output_grid
            imagination_accuracy = (
                np.mean(augmented_result == test_output)
                if augmented_result is not None
                else 0.0
            )

            # Identify failure mode
            failure_mode = self.identify_failure_mode(
                v7_accuracy, imagination_accuracy, hypothesis_analysis
            )

            return FailureAnalysis(
                task_id=Path(task_file).stem,
                v7_accuracy=float(v7_accuracy),
                imagination_accuracy=float(imagination_accuracy),
                hypothesis_count=hypothesis_analysis["count"],
                hypothesis_diversity=hypothesis_analysis["diversity"],
                constraint_violations=hypothesis_analysis["constraint_violations"],
                failure_mode=failure_mode,
            )

        except Exception as e:
            print(f"Error analyzing {task_file}: {e}")
            return None

    def generate_report(self, analyses: List[FailureAnalysis]) -> Dict:
        """Generate comprehensive failure report."""
        # Group by failure mode
        by_mode = defaultdict(list)
        for analysis in analyses:
            by_mode[analysis.failure_mode].append(analysis)

        # Calculate statistics
        report = {
            "summary": {
                "total_tasks": len(analyses),
                "avg_v7_accuracy": np.mean([a.v7_accuracy for a in analyses]),
                "avg_imagination_accuracy": np.mean(
                    [a.imagination_accuracy for a in analyses]
                ),
                "avg_hypothesis_count": np.mean([a.hypothesis_count for a in analyses]),
                "avg_hypothesis_diversity": np.mean(
                    [a.hypothesis_diversity for a in analyses]
                ),
            },
            "failure_modes": {},
        }

        # Analyze each failure mode
        for mode, cases in by_mode.items():
            report["failure_modes"][mode] = {
                "count": len(cases),
                "percentage": len(cases) / len(analyses) * 100,
                "avg_accuracy_drop": np.mean(
                    [c.v7_accuracy - c.imagination_accuracy for c in cases]
                ),
                "example_tasks": [c.task_id for c in cases[:3]],
            }

        # Identify key issues
        report["key_issues"] = []

        if report["summary"]["avg_hypothesis_diversity"] < 0.3:
            report["key_issues"].append(
                "Low hypothesis diversity - hypotheses too similar to each other"
            )

        if report["summary"]["avg_hypothesis_count"] < 5:
            report["key_issues"].append(
                "Insufficient hypothesis generation - need more candidates"
            )

        if "unnecessary_imagination" in report["failure_modes"]:
            count = report["failure_modes"]["unnecessary_imagination"]["count"]
            if count > len(analyses) * 0.3:
                report["key_issues"].append(
                    f"Imagination triggered unnecessarily in {count} tasks where V7 already performs well"
                )

        if "poor_hypothesis_selection" in report["failure_modes"]:
            report["key_issues"].append(
                "Hypothesis selection mechanism failing - best hypothesis not chosen"
            )

        return report


def main():
    """Run the diagnostic analysis."""
    print("=" * 60)
    print("IMAGINATION FAILURE ANALYSIS")
    print("=" * 60)

    diagnostics = ImaginationDiagnostics()

    # Get ARC task files
    data_dir = Path(
        "/Users/fergusmeiklejohn/dev/neural_networks_research/experiments/04_distribution_invention_mechanisms/data/arc_agi_official/ARC-AGI/data/training"
    )
    task_files = list(data_dir.glob("*.json"))[:20]  # Analyze first 20 tasks

    print(f"\nAnalyzing {len(task_files)} tasks...")

    # Analyze each task
    analyses = []
    for i, task_file in enumerate(task_files):
        print(f"  [{i+1}/{len(task_files)}] Analyzing {task_file.name}...")
        analysis = diagnostics.analyze_task(task_file)
        if analysis:
            analyses.append(analysis)

            # Print immediate insights
            if analysis.v7_accuracy > 0.7 and analysis.imagination_accuracy < 0.4:
                print(
                    f"    ⚠️  Imagination hurt performance: {analysis.v7_accuracy:.1%} → {analysis.imagination_accuracy:.1%}"
                )
            elif analysis.hypothesis_count == 0:
                print(f"    ❌ No hypotheses generated")

    # Generate report
    print("\n" + "=" * 60)
    print("ANALYSIS REPORT")
    print("=" * 60)

    report = diagnostics.generate_report(analyses)

    # Print summary
    print("\n📊 SUMMARY:")
    print(f"  V7 Baseline Accuracy: {report['summary']['avg_v7_accuracy']:.1%}")
    print(
        f"  Imagination Accuracy: {report['summary']['avg_imagination_accuracy']:.1%}"
    )
    print(
        f"  Performance Drop: {report['summary']['avg_v7_accuracy'] - report['summary']['avg_imagination_accuracy']:.1%}"
    )
    print(
        f"  Avg Hypotheses Generated: {report['summary']['avg_hypothesis_count']:.1f}"
    )
    print(
        f"  Hypothesis Diversity: {report['summary']['avg_hypothesis_diversity']:.2f}"
    )

    # Print failure modes
    print("\n🔍 FAILURE MODES:")
    for mode, data in sorted(
        report["failure_modes"].items(), key=lambda x: x[1]["count"], reverse=True
    ):
        print(f"\n  {mode}:")
        print(f"    Count: {data['count']} ({data['percentage']:.1f}%)")
        print(f"    Avg accuracy drop: {data['avg_accuracy_drop']:.1%}")
        print(f"    Examples: {', '.join(data['example_tasks'])}")

    # Print key issues
    print("\n⚠️  KEY ISSUES:")
    for issue in report["key_issues"]:
        print(f"  • {issue}")

    # Print recommendations
    print("\n💡 RECOMMENDATIONS:")
    print("  1. Improve hypothesis diversity through structured variations")
    print(
        "  2. Better triggering - only use imagination when V7 confidence is genuinely low"
    )
    print("  3. Add constraint checking to avoid invalid hypotheses")
    print(
        "  4. Implement better hypothesis selection based on partial pattern matching"
    )
    print("  5. Create 'imagination operators' that make systematic changes")

    # Save detailed report
    output_file = Path("imagination_failure_analysis.json")
    with open(output_file, "w") as f:
        # Convert analyses to dict for JSON serialization
        json_report = {
            "summary": report["summary"],
            "failure_modes": report["failure_modes"],
            "key_issues": report["key_issues"],
            "detailed_analyses": [
                {
                    "task_id": a.task_id,
                    "v7_accuracy": a.v7_accuracy,
                    "imagination_accuracy": a.imagination_accuracy,
                    "hypothesis_count": a.hypothesis_count,
                    "hypothesis_diversity": a.hypothesis_diversity,
                    "constraint_violations": a.constraint_violations,
                    "failure_mode": a.failure_mode,
                }
                for a in analyses
            ],
        }
        json.dump(json_report, f, indent=2)

    print(f"\n✅ Full report saved to: {output_file}")


if __name__ == "__main__":
    main()
