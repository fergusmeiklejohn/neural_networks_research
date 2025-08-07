#!/usr/bin/env python3
"""Evaluate our hybrid approach on real ARC-AGI tasks.

This runs our explicit extraction + neural perception + TTA on actual ARC tasks.
"""

from utils.imports import setup_project_paths

setup_project_paths()

import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from arc_evaluation_pipeline import ARCEvaluationPipeline
from arc_grid_extractor import ARCGridExtractor
from arc_test_time_adapter import ARCTestTimeAdapter
from hybrid_arc_solver import HybridARCSolver
from neural_perception import NeuralPerceptionModule


class RealARCEvaluator:
    """Evaluates our approach on real ARC-AGI tasks."""

    def __init__(self, data_dir: str = "data/arc_agi_official"):
        self.data_dir = Path(data_dir)
        self.repo_path = self.data_dir / "ARC-AGI"

        # Initialize our solvers
        self.extractor = ARCGridExtractor()
        self.perception = NeuralPerceptionModule()
        self.adapter = ARCTestTimeAdapter()
        self.hybrid_solver = HybridARCSolver()
        self.pipeline = ARCEvaluationPipeline()

    def load_task(self, task_path: Path) -> Dict:
        """Load a single ARC task."""
        with open(task_path) as f:
            task = json.load(f)
        task["task_id"] = task_path.stem
        return task

    def evaluate_task(self, task: Dict, use_tta: bool = True) -> Dict:
        """Evaluate a single task with our approach."""
        task_id = task["task_id"]

        # Prepare training examples
        train_examples = [
            (
                np.array(ex["input"], dtype=np.int32),
                np.array(ex["output"], dtype=np.int32),
            )
            for ex in task["train"]
        ]

        # Prepare test inputs
        test_inputs = [np.array(ex["input"], dtype=np.int32) for ex in task["test"]]

        # Get expected outputs (if available)
        expected_outputs = []
        for ex in task["test"]:
            if "output" in ex:
                expected_outputs.append(np.array(ex["output"], dtype=np.int32))

        # Track results
        result = {
            "task_id": task_id,
            "num_train": len(train_examples),
            "num_test": len(test_inputs),
            "has_solutions": len(expected_outputs) > 0,
        }

        # Try different approaches
        approaches = {
            "explicit_only": self.evaluate_explicit_only,
            "neural_only": self.evaluate_neural_only,
            "hybrid_no_tta": self.evaluate_hybrid_no_tta,
            "hybrid_with_tta": self.evaluate_hybrid_with_tta,
        }

        for approach_name, approach_func in approaches.items():
            if approach_name == "hybrid_with_tta" and not use_tta:
                continue

            start_time = time.time()
            predictions, confidence = approach_func(train_examples, test_inputs)
            elapsed = time.time() - start_time

            # Evaluate accuracy if we have solutions
            accuracy = 0.0
            if expected_outputs:
                correct = sum(
                    1
                    for pred, expected in zip(predictions, expected_outputs)
                    if np.array_equal(pred, expected)
                )
                accuracy = correct / len(expected_outputs)

            result[f"{approach_name}_accuracy"] = accuracy
            result[f"{approach_name}_confidence"] = confidence
            result[f"{approach_name}_time"] = elapsed

        return result

    def evaluate_explicit_only(
        self, train_examples: List[Tuple], test_inputs: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], float]:
        """Evaluate using only explicit extraction."""
        try:
            # Extract rules
            rules = self.extractor.extract_rules(train_examples)

            # Apply to test inputs
            predictions = []
            for test_input in test_inputs:
                try:
                    pred = self.extractor.apply_rules(test_input, rules)
                    predictions.append(pred)
                except Exception:
                    # If rule application fails, return input unchanged
                    predictions.append(test_input)

            # Confidence based on extraction success
            confidence = 0.5 if rules.transformations else 0.1
            return predictions, confidence

        except Exception:
            # Fallback: return inputs unchanged
            return test_inputs, 0.0

    def evaluate_neural_only(
        self, train_examples: List[Tuple], test_inputs: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], float]:
        """Evaluate using only neural perception (baseline)."""
        # This is a simplified neural-only approach
        # In practice, you'd train a neural model on the examples

        # For now, just detect patterns and make simple predictions
        predictions = []
        for test_input in test_inputs:
            # Detect patterns in input
            patterns = self.perception.detect_spatial_patterns(test_input)

            # Simple heuristic: if symmetric, maintain symmetry
            # Otherwise, return input unchanged
            if any(p.pattern_type.endswith("symmetry") for p in patterns):
                # Apply simple symmetry-preserving transform
                predictions.append(np.flip(test_input, axis=0))
            else:
                predictions.append(test_input)

        return predictions, 0.3

    def evaluate_hybrid_no_tta(
        self, train_examples: List[Tuple], test_inputs: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], float]:
        """Evaluate using hybrid approach without TTA."""
        predictions = []
        confidences = []

        for test_input in test_inputs:
            # Hybrid solver returns ARCPrediction object
            pred_obj = self.hybrid_solver.solve(train_examples, test_input)
            predictions.append(pred_obj.grid)
            confidences.append(pred_obj.confidence)

        avg_confidence = np.mean(confidences) if confidences else 0.0
        return predictions, avg_confidence

    def evaluate_hybrid_with_tta(
        self, train_examples: List[Tuple], test_inputs: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], float]:
        """Evaluate using hybrid approach with TTA."""
        predictions = []
        confidences = []

        # Use TTA adapter for this approach
        for test_input in test_inputs:
            # First get initial rules with hybrid solver
            initial_pred = self.hybrid_solver.solve(train_examples, test_input)

            # Then adapt with TTA
            initial_rules = self.extractor.extract_rules(train_examples)
            adapted_result = self.adapter.adapt(
                train_examples, initial_rules, max_steps=5
            )

            # Apply adapted rules
            try:
                pred = self.extractor.apply_rules(
                    test_input, adapted_result.refined_rules
                )
                predictions.append(pred)
                confidences.append(adapted_result.confidence)
            except Exception:
                # Fallback to initial prediction
                predictions.append(initial_pred.grid)
                confidences.append(initial_pred.confidence)

        avg_confidence = np.mean(confidences) if confidences else 0.0
        return predictions, avg_confidence

    def evaluate_dataset(
        self, dataset: str = "training", max_tasks: int = 10, random_sample: bool = True
    ) -> List[Dict]:
        """Evaluate on multiple tasks from a dataset."""
        dataset_path = self.repo_path / "data" / dataset

        if not dataset_path.exists():
            print(f"Dataset path {dataset_path} not found!")
            return []

        # Get task files
        task_files = list(dataset_path.glob("*.json"))
        print(f"Found {len(task_files)} tasks in {dataset}")

        # Sample tasks
        if random_sample and len(task_files) > max_tasks:
            task_files = random.sample(task_files, max_tasks)
        else:
            task_files = task_files[:max_tasks]

        results = []
        for i, task_file in enumerate(task_files, 1):
            print(f"\nEvaluating task {i}/{len(task_files)}: {task_file.stem}")

            task = self.load_task(task_file)
            result = self.evaluate_task(task)
            results.append(result)

            # Print summary
            if result["has_solutions"]:
                print(f"  Explicit only: {result['explicit_only_accuracy']:.1%}")
                print(f"  Neural only: {result['neural_only_accuracy']:.1%}")
                print(f"  Hybrid (no TTA): {result['hybrid_no_tta_accuracy']:.1%}")
                if "hybrid_with_tta_accuracy" in result:
                    print(
                        f"  Hybrid (with TTA): {result['hybrid_with_tta_accuracy']:.1%}"
                    )

        return results

    def analyze_results(self, results: List[Dict]) -> None:
        """Analyze and print summary of results."""
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)

        # Calculate averages
        approaches = [
            "explicit_only",
            "neural_only",
            "hybrid_no_tta",
            "hybrid_with_tta",
        ]

        for approach in approaches:
            accuracy_key = f"{approach}_accuracy"
            time_key = f"{approach}_time"

            accuracies = [r[accuracy_key] for r in results if accuracy_key in r]
            times = [r[time_key] for r in results if time_key in r]

            if accuracies:
                avg_accuracy = np.mean(accuracies)
                avg_time = np.mean(times)

                print(f"\n{approach.replace('_', ' ').title()}:")
                print(f"  Average accuracy: {avg_accuracy:.1%}")
                print(f"  Average time: {avg_time:.3f}s")
                print(
                    f"  Tasks solved (>0%): {sum(1 for a in accuracies if a > 0)}/{len(accuracies)}"
                )
                print(
                    f"  Perfect solves (100%): {sum(1 for a in accuracies if a == 1.0)}/{len(accuracies)}"
                )

        # Find best approach per task
        print("\n" + "-" * 70)
        print("Best approach per task:")
        for result in results:
            task_id = result["task_id"]
            best_approach = None
            best_accuracy = 0.0

            for approach in approaches:
                accuracy_key = f"{approach}_accuracy"
                if accuracy_key in result and result[accuracy_key] > best_accuracy:
                    best_accuracy = result[accuracy_key]
                    best_approach = approach

            if best_approach and best_accuracy > 0:
                print(f"  {task_id}: {best_approach} ({best_accuracy:.1%})")

        # Identify where explicit extraction excels
        print("\n" + "-" * 70)
        print("Tasks where explicit extraction performs well:")
        for result in results:
            if result.get("explicit_only_accuracy", 0) >= 0.5:
                print(f"  {result['task_id']}: {result['explicit_only_accuracy']:.1%}")

        # Save results
        output_file = Path(
            "experiments/04_distribution_invention_mechanisms/outputs/real_arc_results.json"
        )
        output_file.parent.mkdir(exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")


def main():
    """Run evaluation on real ARC tasks."""
    print("=" * 70)
    print("REAL ARC-AGI EVALUATION")
    print("=" * 70)

    evaluator = RealARCEvaluator()

    # Evaluate on a sample of training tasks
    print("\nEvaluating on training set...")
    results = evaluator.evaluate_dataset(
        dataset="training", max_tasks=10, random_sample=True  # Start with 10 tasks
    )

    # Analyze results
    evaluator.analyze_results(results)

    print("\n" + "=" * 70)
    print("KEY INSIGHTS:")
    print("1. Compare performance across approaches")
    print("2. Identify task types where explicit extraction excels")
    print("3. Measure TTA effectiveness on real tasks")
    print("4. Find patterns in failure cases")
    print("=" * 70)


if __name__ == "__main__":
    main()
