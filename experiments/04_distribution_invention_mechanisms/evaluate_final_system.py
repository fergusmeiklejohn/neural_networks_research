#!/usr/bin/env python3
"""Final evaluation of the enhanced ARC system with all improvements.

This script evaluates our complete system (perception + object manipulation + DSL + synthesis)
on real ARC-AGI tasks to measure the improvement over the 4% baseline.
"""

from utils.imports import setup_project_paths

setup_project_paths()

import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
from enhanced_arc_solver import EnhancedARCSolver
from evaluate_real_arc_fixed import RealARCEvaluator


class FinalSystemEvaluator:
    """Evaluates the complete enhanced ARC system."""

    def __init__(self):
        # Initialize both baseline and enhanced solvers
        self.baseline_evaluator = RealARCEvaluator()
        self.enhanced_solver = EnhancedARCSolver(
            use_synthesis=True, synthesis_timeout=3.0
        )

        # Data paths
        base_path = Path(__file__).parent.parent.parent
        self.data_dir = base_path / "data" / "arc_agi_official" / "ARC-AGI"

        # Results tracking
        self.results = []

    def evaluate_task(self, task_path: Path) -> Dict:
        """Evaluate a single ARC task with both baseline and enhanced systems."""

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

        result = {
            "task_id": task_id,
            "num_train": len(train_examples),
            "num_test": len(test_examples),
            "baseline_correct": False,
            "enhanced_correct": False,
            "method_used": None,
            "confidence": 0.0,
            "time_taken": 0.0,
        }

        # Evaluate with baseline (simple extractor)
        try:
            baseline_rules = self.baseline_evaluator.extractor.extract_rules(
                train_examples
            )
            if baseline_rules.transformations and test_examples[0][1] is not None:
                baseline_pred = self.baseline_evaluator.extractor.apply_rules(
                    test_examples[0][0], baseline_rules
                )
                result["baseline_correct"] = np.array_equal(
                    baseline_pred, test_examples[0][1]
                )
        except Exception:
            pass

        # Evaluate with enhanced system
        try:
            solution = self.enhanced_solver.solve(train_examples, test_examples[0][0])

            result["method_used"] = solution.method_used
            result["confidence"] = solution.confidence
            result["time_taken"] = solution.time_taken

            if test_examples[0][1] is not None:
                result["enhanced_correct"] = np.array_equal(
                    solution.output_grid, test_examples[0][1]
                )

            # Store program if synthesis was used
            if solution.program:
                result["program"] = solution.program.to_code()

        except Exception as e:
            print(f"  Enhanced evaluation failed: {e}")
            result["method_used"] = "failed"

        return result

    def evaluate_dataset(
        self, num_tasks: int = 50, use_same_seed: bool = True
    ) -> List[Dict]:
        """Evaluate on multiple tasks from the dataset.

        Args:
            num_tasks: Number of tasks to evaluate
            use_same_seed: Use same random seed as baseline for fair comparison

        Returns:
            List of evaluation results
        """
        # Get task files
        training_dir = self.data_dir / "data" / "training"
        task_files = list(training_dir.glob("*.json"))

        # Use same random seed as baseline for fair comparison
        if use_same_seed:
            random.seed(42)

        task_files = random.sample(task_files, min(num_tasks, len(task_files)))

        print(f"Evaluating {len(task_files)} tasks...")
        print("=" * 70)

        for i, task_file in enumerate(task_files, 1):
            print(f"\nTask {i}/{len(task_files)}: {task_file.stem}")

            result = self.evaluate_task(task_file)
            self.results.append(result)

            # Print quick summary
            baseline_mark = "✓" if result["baseline_correct"] else "✗"
            enhanced_mark = "✓" if result["enhanced_correct"] else "✗"

            print(f"  Baseline: {baseline_mark}")
            print(
                f"  Enhanced: {enhanced_mark} (method: {result['method_used']}, "
                f"conf: {result['confidence']:.2f}, time: {result['time_taken']:.2f}s)"
            )

            # Highlight improvements
            if result["enhanced_correct"] and not result["baseline_correct"]:
                print(f"  ⭐ IMPROVEMENT! Enhanced solved but baseline failed")

        return self.results

    def analyze_results(self) -> Dict:
        """Analyze and summarize evaluation results."""

        total = len(self.results)
        baseline_correct = sum(1 for r in self.results if r["baseline_correct"])
        enhanced_correct = sum(1 for r in self.results if r["enhanced_correct"])

        # Count by method used
        method_counts = {}
        method_success = {}

        for r in self.results:
            method = r["method_used"]
            if method:
                method_counts[method] = method_counts.get(method, 0) + 1
                if method not in method_success:
                    method_success[method] = 0
                if r["enhanced_correct"]:
                    method_success[method] += 1

        # Find improvements
        improvements = []
        for r in self.results:
            if r["enhanced_correct"] and not r["baseline_correct"]:
                improvements.append(r["task_id"])

        # Calculate statistics
        baseline_accuracy = baseline_correct / total if total > 0 else 0
        enhanced_accuracy = enhanced_correct / total if total > 0 else 0
        improvement_rate = (
            (enhanced_accuracy - baseline_accuracy) / baseline_accuracy
            if baseline_accuracy > 0
            else 0
        )

        summary = {
            "total_tasks": total,
            "baseline_correct": baseline_correct,
            "enhanced_correct": enhanced_correct,
            "baseline_accuracy": baseline_accuracy,
            "enhanced_accuracy": enhanced_accuracy,
            "improvement_rate": improvement_rate,
            "improvements": improvements,
            "method_counts": method_counts,
            "method_success": method_success,
            "avg_confidence": np.mean([r["confidence"] for r in self.results]),
            "avg_time": np.mean([r["time_taken"] for r in self.results]),
        }

        return summary

    def print_summary(self, summary: Dict):
        """Print evaluation summary."""

        print("\n" + "=" * 70)
        print("FINAL EVALUATION SUMMARY")
        print("=" * 70)

        print(f"\nOverall Performance:")
        print(
            f"  Baseline: {summary['baseline_correct']}/{summary['total_tasks']} "
            f"({summary['baseline_accuracy']:.1%})"
        )
        print(
            f"  Enhanced: {summary['enhanced_correct']}/{summary['total_tasks']} "
            f"({summary['enhanced_accuracy']:.1%})"
        )

        if summary["baseline_accuracy"] > 0:
            print(f"  Relative Improvement: {summary['improvement_rate']:.1%}")

        print(f"\nMethod Usage:")
        for method, count in summary["method_counts"].items():
            success = summary["method_success"].get(method, 0)
            success_rate = success / count if count > 0 else 0
            print(
                f"  {method}: {count} tasks ({success}/{count} = {success_rate:.1%} success)"
            )

        print(f"\nTiming:")
        print(f"  Average confidence: {summary['avg_confidence']:.3f}")
        print(f"  Average time per task: {summary['avg_time']:.3f}s")

        if summary["improvements"]:
            print(
                f"\n✨ New Successes (solved by enhanced but not baseline): {len(summary['improvements'])}"
            )
            for task_id in summary["improvements"][:5]:
                print(f"  - {task_id}")
            if len(summary["improvements"]) > 5:
                print(f"  ... and {len(summary['improvements']) - 5} more")

        print("\n" + "=" * 70)
        print("KEY ACHIEVEMENTS:")
        print("=" * 70)

        # Calculate absolute improvement
        absolute_improvement = (
            summary["enhanced_accuracy"] - summary["baseline_accuracy"]
        )

        if absolute_improvement > 0:
            print(
                f"✓ Accuracy improved from {summary['baseline_accuracy']:.1%} to "
                f"{summary['enhanced_accuracy']:.1%} (+{absolute_improvement:.1%})"
            )
            print(f"✓ Solved {len(summary['improvements'])} additional tasks")

            # Analyze which methods contributed most
            best_method = max(
                summary["method_success"].items(),
                key=lambda x: x[1] if x[0] != "failed" else 0,
            )
            if best_method[1] > 0:
                print(
                    f"✓ Most successful method: {best_method[0]} ({best_method[1]} tasks)"
                )

        # Check against targets
        print("\nTarget Achievement:")
        if summary["enhanced_accuracy"] >= 0.15:
            print("✓ Reached 15% milestone (object manipulation target)")
        if summary["enhanced_accuracy"] >= 0.25:
            print("✓ Reached 25% milestone (DSL target)")
        if summary["enhanced_accuracy"] >= 0.30:
            print("✓ Reached 30% milestone (synthesis target)")

        if summary["enhanced_accuracy"] < 0.15:
            print("⚠ Below 15% target - need more work on object manipulation")
        elif summary["enhanced_accuracy"] < 0.25:
            print("⚠ Below 25% target - need to improve DSL coverage")
        elif summary["enhanced_accuracy"] < 0.30:
            print("⚠ Below 30% target - need better program synthesis")
        else:
            print("🎉 Achieved 30%+ target!")

    def save_results(self):
        """Save detailed results to JSON."""

        output_dir = Path(__file__).parent / "outputs"
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / "final_evaluation_results.json"

        summary = self.analyze_results()

        with open(output_file, "w") as f:
            json.dump(
                {"summary": summary, "detailed_results": self.results},
                f,
                indent=2,
                default=str,
            )

        print(f"\nDetailed results saved to {output_file}")


def main():
    """Run final evaluation."""

    print("=" * 70)
    print("FINAL ARC-AGI EVALUATION")
    print("Enhanced System with Object Manipulation + DSL + Program Synthesis")
    print("=" * 70)

    evaluator = FinalSystemEvaluator()

    # Evaluate on same 50 tasks as baseline
    print("\nEvaluating on 50 ARC-AGI training tasks...")
    results = evaluator.evaluate_dataset(num_tasks=50, use_same_seed=True)

    # Analyze and print summary
    summary = evaluator.analyze_results()
    evaluator.print_summary(summary)

    # Save results
    evaluator.save_results()

    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)

    if summary["enhanced_accuracy"] < 0.20:
        print("1. Debug why object manipulation isn't working on more tasks")
        print("2. Add more DSL primitives based on failure analysis")
        print("3. Improve pattern detection in perception module")
    elif summary["enhanced_accuracy"] < 0.30:
        print("1. Enhance program synthesis search strategy")
        print("2. Add neural guidance to prioritize promising programs")
        print("3. Implement more sophisticated object relationships")
    else:
        print("1. Great progress! Consider adding neural program guide")
        print("2. Optimize for speed to handle more complex searches")
        print("3. Test on ARC-AGI evaluation set (not just training)")


if __name__ == "__main__":
    main()
