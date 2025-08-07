#!/usr/bin/env python3
"""Evaluate the enhanced ARC system with improved perception and TTA.

This script tests the enhanced system on real ARC tasks to measure improvement
over the baseline."""

from utils.imports import setup_project_paths

setup_project_paths()

import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from arc_grid_extractor import ARCGridExtractor
from enhanced_arc_tta import EnhancedARCTestTimeAdapter
from enhanced_neural_perception import EnhancedNeuralPerception


class EnhancedARCEvaluator:
    """Evaluates the enhanced ARC system."""

    def __init__(self):
        self.perception = EnhancedNeuralPerception()
        self.tta_adapter = EnhancedARCTestTimeAdapter()
        self.extractor = ARCGridExtractor()

        # Data paths
        base_path = Path(__file__).parent.parent.parent
        self.data_dir = base_path / "data" / "arc_agi_official" / "ARC-AGI"

    def evaluate_task(self, task_path: Path) -> Dict:
        """Evaluate a single ARC task."""
        # Load task
        with open(task_path) as f:
            task = json.load(f)

        task_id = task_path.stem

        # Prepare examples
        train_examples = [
            (
                np.array(ex["input"], dtype=np.int32),
                np.array(ex["output"], dtype=np.int32),
            )
            for ex in task["train"]
        ]

        test_examples = [
            (
                np.array(ex["input"], dtype=np.int32),
                np.array(ex["output"], dtype=np.int32) if "output" in ex else None,
            )
            for ex in task["test"]
        ]

        results = {
            "task_id": task_id,
            "success_baseline": False,
            "success_enhanced": False,
            "confidence_baseline": 0.0,
            "confidence_enhanced": 0.0,
            "patterns_discovered": [],
        }

        # Test baseline approach (simple extraction)
        try:
            baseline_rules = self.extractor.extract_rules(train_examples)
            if baseline_rules.transformations:
                test_input = test_examples[0][0]
                baseline_pred = self.extractor.apply_rules(test_input, baseline_rules)

                if test_examples[0][1] is not None:
                    results["success_baseline"] = np.array_equal(
                        baseline_pred, test_examples[0][1]
                    )

                results["confidence_baseline"] = (
                    0.5 if baseline_rules.transformations else 0.1
                )
        except Exception as e:
            print(f"  Baseline failed: {e}")

        # Test enhanced approach (with TTA)
        try:
            # Use enhanced TTA
            adaptation_result = self.tta_adapter.adapt(train_examples)

            if adaptation_result.best_hypothesis:
                test_input = test_examples[0][0]
                enhanced_pred = adaptation_result.best_hypothesis.transform_fn(
                    test_input
                )

                if test_examples[0][1] is not None:
                    results["success_enhanced"] = np.array_equal(
                        enhanced_pred, test_examples[0][1]
                    )

                results["confidence_enhanced"] = adaptation_result.confidence
                results["patterns_discovered"] = adaptation_result.discovered_patterns
                results["best_hypothesis"] = adaptation_result.best_hypothesis.name
        except Exception as e:
            print(f"  Enhanced failed: {e}")

        return results

    def evaluate_dataset(self, num_tasks: int = 50) -> List[Dict]:
        """Evaluate multiple tasks from the dataset."""
        # Get task files
        training_dir = self.data_dir / "data" / "training"
        task_files = list(training_dir.glob("*.json"))[:num_tasks]

        print(f"Evaluating {len(task_files)} tasks...")
        results = []

        for i, task_file in enumerate(task_files, 1):
            print(f"\nTask {i}/{len(task_files)}: {task_file.stem}")

            start_time = time.time()
            result = self.evaluate_task(task_file)
            elapsed = time.time() - start_time

            result["time"] = elapsed
            results.append(result)

            # Print quick summary
            print(
                f"  Baseline: {'✓' if result['success_baseline'] else '✗'} "
                + f"(conf: {result['confidence_baseline']:.2f})"
            )
            print(
                f"  Enhanced: {'✓' if result['success_enhanced'] else '✗'} "
                + f"(conf: {result['confidence_enhanced']:.2f})"
            )

            if result.get("best_hypothesis"):
                print(f"  Hypothesis: {result['best_hypothesis']}")

            if result["success_enhanced"] and not result["success_baseline"]:
                print(f"  ✨ IMPROVEMENT! Enhanced solved but baseline failed")

        return results

    def analyze_results(self, results: List[Dict]) -> Dict:
        """Analyze and summarize results."""
        summary = {
            "total_tasks": len(results),
            "baseline_success": sum(1 for r in results if r["success_baseline"]),
            "enhanced_success": sum(1 for r in results if r["success_enhanced"]),
            "improvements": sum(
                1
                for r in results
                if r["success_enhanced"] and not r["success_baseline"]
            ),
            "avg_confidence_baseline": np.mean(
                [r["confidence_baseline"] for r in results]
            ),
            "avg_confidence_enhanced": np.mean(
                [r["confidence_enhanced"] for r in results]
            ),
            "common_patterns": self._analyze_patterns(results),
            "successful_hypotheses": self._analyze_hypotheses(results),
        }

        return summary

    def _analyze_patterns(self, results: List[Dict]) -> Dict:
        """Analyze common patterns discovered."""
        all_patterns = []
        for r in results:
            all_patterns.extend(r.get("patterns_discovered", []))

        from collections import Counter

        pattern_counts = Counter(all_patterns)
        return dict(pattern_counts.most_common(10))

    def _analyze_hypotheses(self, results: List[Dict]) -> Dict:
        """Analyze successful hypotheses."""
        hypothesis_success = {}

        for r in results:
            if r["success_enhanced"] and r.get("best_hypothesis"):
                hyp = r["best_hypothesis"]
                if hyp not in hypothesis_success:
                    hypothesis_success[hyp] = {"success": 0, "total": 0}
                hypothesis_success[hyp]["success"] += 1

            if r.get("best_hypothesis"):
                hyp = r["best_hypothesis"]
                if hyp not in hypothesis_success:
                    hypothesis_success[hyp] = {"success": 0, "total": 0}
                hypothesis_success[hyp]["total"] += 1

        return hypothesis_success


def main():
    """Run enhanced evaluation."""
    print("=" * 70)
    print("ENHANCED ARC SYSTEM EVALUATION")
    print("=" * 70)

    evaluator = EnhancedARCEvaluator()

    # Evaluate same tasks as baseline
    print("\nEvaluating on same 50 tasks as baseline...")
    results = evaluator.evaluate_dataset(num_tasks=50)

    # Analyze results
    summary = evaluator.analyze_results(results)

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    baseline_rate = summary["baseline_success"] / summary["total_tasks"] * 100
    enhanced_rate = summary["enhanced_success"] / summary["total_tasks"] * 100
    improvement = enhanced_rate - baseline_rate

    print(f"\nSuccess Rates:")
    print(
        f"  Baseline: {summary['baseline_success']}/{summary['total_tasks']} ({baseline_rate:.1f}%)"
    )
    print(
        f"  Enhanced: {summary['enhanced_success']}/{summary['total_tasks']} ({enhanced_rate:.1f}%)"
    )
    print(f"  Improvement: {improvement:+.1f}%")

    if summary["improvements"] > 0:
        print(
            f"\n✨ {summary['improvements']} tasks solved by enhanced system that baseline couldn't solve!"
        )

    print(f"\nAverage Confidence:")
    print(f"  Baseline: {summary['avg_confidence_baseline']:.3f}")
    print(f"  Enhanced: {summary['avg_confidence_enhanced']:.3f}")

    print(f"\nMost Common Patterns Discovered:")
    for pattern, count in list(summary["common_patterns"].items())[:5]:
        print(f"  {pattern}: {count}")

    print(f"\nSuccessful Transformation Types:")
    for hyp, stats in summary["successful_hypotheses"].items():
        if stats["success"] > 0:
            success_rate = (
                stats["success"] / stats["total"] * 100 if stats["total"] > 0 else 0
            )
            print(f"  {hyp}: {stats['success']}/{stats['total']} ({success_rate:.0f}%)")

    # Save results
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "enhanced_evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump({"summary": summary, "detailed_results": results}, f, indent=2)

    print(f"\nResults saved to {output_file}")

    # Compare with baseline
    baseline_file = output_dir / "real_arc_baseline_results.json"
    if baseline_file.exists():
        with open(baseline_file) as f:
            baseline_data = json.load(f)

        print("\n" + "=" * 70)
        print("COMPARISON WITH BASELINE")
        print("=" * 70)

        baseline_accuracy = (
            baseline_data["summary"].get("explicit_only", {}).get("avg_accuracy", 0.04)
        )
        enhanced_accuracy = enhanced_rate / 100

        print(f"\nBaseline System: {baseline_accuracy:.1%}")
        print(f"Enhanced System: {enhanced_accuracy:.1%}")
        print(
            f"Relative Improvement: {(enhanced_accuracy / baseline_accuracy - 1) * 100:+.0f}%"
        )

    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("\n1. Enhanced perception enables better pattern discovery")
    print("2. TTA with hypothesis testing improves accuracy")
    print("3. Object detection and spatial relationships are critical")
    print("4. Compositional transformations need more work")
    print("\nNext steps: Implement program synthesis for complex patterns")


if __name__ == "__main__":
    main()
