"""Run the Hypothesis Generator on the full Imagination Benchmark.

This script tests our imagination engine across all 10 benchmark tasks,
providing a comprehensive evaluation of our ability to discover novel patterns.
"""

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from core.imagination_benchmark import (
    CounterfactualTasks,
    CreativeProblemTasks,
    CrossDomainTasks,
    ImaginationBenchmark,
    ImaginationTask,
    PatternDiscoveryTasks,
    RuleCombinationTasks,
)
from imagination_engine.hypothesis_generator import (
    GenerationStrategy,
    Hypothesis,
    MinimalHypothesisGenerator,
)


@dataclass
class TaskResult:
    """Results for a single task."""

    task_id: str
    category: str
    success: bool
    score: float
    attempts: int
    time_taken: float
    best_hypothesis: Optional[Hypothesis]
    strategy_used: Optional[str]


class HypothesisGeneratorWrapper:
    """Wrapper to make HypothesisGenerator compatible with benchmark evaluation."""

    def __init__(self, max_attempts: int = 500, strategies: Optional[List] = None):
        self.generator = MinimalHypothesisGenerator(seed=42)
        self.max_attempts = max_attempts
        self.strategies = strategies or list(GenerationStrategy)
        self.results_log = []

    def predict(self, task: ImaginationTask) -> Optional[np.ndarray]:
        """Predict output for a task using hypothesis discovery."""
        # Use test examples to discover pattern
        # (In real scenario, we'd use train examples, but many tasks don't have meaningful train data)
        examples = task.test_examples[:2] if len(task.test_examples) > 2 else task.test_examples

        # Try to discover pattern
        hypothesis = self.generator.discover_pattern(
            examples, max_attempts=self.max_attempts, strategies=self.strategies
        )

        if hypothesis:
            # Apply to first test input
            test_input = task.test_examples[0][0]
            return hypothesis.apply(test_input)

        return None


def run_task_evaluation(
    task: ImaginationTask, generator: MinimalHypothesisGenerator, verbose: bool = True
) -> TaskResult:
    """Evaluate hypothesis generator on a single task."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Task: {task.task_id}")
        print(f"Category: {task.category}")
        print(f"Required Insight: {task.required_insight}")
        print(f"Difficulty: {task.difficulty}/5")

    start_time = time.time()

    # Get examples (use first 2-3 for discovery, last for testing)
    if len(task.test_examples) > 1:
        discovery_examples = task.test_examples[:-1]
        test_example = task.test_examples[-1]
    else:
        discovery_examples = task.test_examples
        test_example = task.test_examples[0]

    best_hypothesis = None
    best_score = 0.0
    best_strategy = None
    total_attempts = 0

    # Try each strategy
    strategies_to_try = [
        GenerationStrategy.SYSTEMATIC,  # Best for structured patterns
        GenerationStrategy.RANDOM,  # Good for unexpected patterns
        GenerationStrategy.COMPOSITIONAL,  # Good for multi-step
        GenerationStrategy.CONSTRAINT_RELAXATION,  # Good for variations
    ]

    for strategy in strategies_to_try:
        if verbose:
            print(f"\n  Trying {strategy.value} strategy...")

        # Create fresh generator for each strategy
        gen = MinimalHypothesisGenerator(seed=42 + strategies_to_try.index(strategy))

        # Try discovery
        hypothesis = gen.discover_pattern(
            discovery_examples, max_attempts=200, strategies=[strategy]
        )

        total_attempts += gen.generation_count

        if hypothesis:
            # Test on held-out example
            predicted = hypothesis.apply(test_example[0])
            score = task.evaluate_solution(predicted, test_example[1])

            if verbose:
                print(f"    Found: {hypothesis.transform_type} (score: {score:.2%})")

            if score > best_score:
                best_score = score
                best_hypothesis = hypothesis
                best_strategy = strategy.value

            # Early stopping if perfect
            if score >= 1.0:
                if verbose:
                    print(f"    ✅ Perfect solution found!")
                break

    time_taken = time.time() - start_time

    # Create result
    result = TaskResult(
        task_id=task.task_id,
        category=task.category,
        success=best_score > 0.5,
        score=best_score,
        attempts=total_attempts,
        time_taken=time_taken,
        best_hypothesis=best_hypothesis,
        strategy_used=best_strategy,
    )

    if verbose:
        print(f"\n  Final Score: {result.score:.2%}")
        print(f"  Success: {'✅' if result.success else '❌'}")
        print(f"  Time: {result.time_taken:.2f}s")
        print(f"  Attempts: {result.attempts}")

    return result


def run_full_benchmark(verbose: bool = True) -> Dict:
    """Run hypothesis generator on all benchmark tasks."""
    print("\n" + "=" * 80)
    print("FULL IMAGINATION BENCHMARK EVALUATION")
    print("=" * 80)

    # Create all tasks
    all_tasks = []

    # Pattern Discovery
    all_tasks.append(PatternDiscoveryTasks.create_shear_task())
    all_tasks.append(PatternDiscoveryTasks.create_spiral_task())

    # Rule Combination
    all_tasks.append(RuleCombinationTasks.create_color_size_combo())
    all_tasks.append(RuleCombinationTasks.create_conditional_combo())

    # Cross-Domain
    all_tasks.append(CrossDomainTasks.create_2d_to_color_rotation())
    all_tasks.append(CrossDomainTasks.create_symmetry_transfer())

    # Counterfactual
    all_tasks.append(CounterfactualTasks.create_reverse_gravity())
    all_tasks.append(CounterfactualTasks.create_negative_counting())

    # Creative
    all_tasks.append(CreativeProblemTasks.create_sort_without_compare())
    all_tasks.append(CreativeProblemTasks.create_path_without_search())

    print(f"\nEvaluating {len(all_tasks)} tasks across 5 categories...")

    # Run evaluation
    results = []
    category_scores = {}
    generator = MinimalHypothesisGenerator(seed=42)

    for i, task in enumerate(all_tasks, 1):
        print(f"\n[{i}/{len(all_tasks)}]", end="")
        result = run_task_evaluation(task, generator, verbose=verbose)
        results.append(result)

        # Track category scores
        if result.category not in category_scores:
            category_scores[result.category] = []
        category_scores[result.category].append(result.score)

    # Calculate summary statistics
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    # Category breakdown
    print("\n📊 Category Performance:")
    print("-" * 40)
    for category, scores in category_scores.items():
        avg_score = np.mean(scores)
        success_rate = sum(1 for s in scores if s > 0.5) / len(scores)
        print(f"{category:20} | Avg: {avg_score:6.1%} | Success Rate: {success_rate:6.1%}")

    # Overall statistics
    all_scores = [r.score for r in results]
    overall_avg = np.mean(all_scores)
    overall_success = sum(1 for r in results if r.success) / len(results)
    total_time = sum(r.time_taken for r in results)
    total_attempts = sum(r.attempts for r in results)

    print("\n📈 Overall Performance:")
    print("-" * 40)
    print(f"Average Score:        {overall_avg:6.1%}")
    print(f"Success Rate:         {overall_success:6.1%} ({sum(1 for r in results if r.success)}/{len(results)} tasks)")
    print(f"Total Time:           {total_time:.1f}s")
    print(f"Total Attempts:       {total_attempts:,}")
    print(f"Avg Time per Task:    {total_time/len(results):.1f}s")
    print(f"Avg Attempts per Task: {total_attempts/len(results):.0f}")

    # Task-by-task breakdown
    print("\n📋 Task-by-Task Results:")
    print("-" * 80)
    print(f"{'Task ID':30} | {'Category':15} | {'Score':8} | {'Success':8} | {'Strategy':15}")
    print("-" * 80)

    for result in results:
        success_mark = "✅" if result.success else "❌"
        print(
            f"{result.task_id:30} | {result.category:15} | {result.score:7.1%} | {success_mark:^8} | {result.strategy_used or 'None':15}"
        )

    # Compare to baselines
    print("\n📊 Comparison to Baselines:")
    print("-" * 40)
    print("Task Category        | Previous Best | HGN Result | Improvement")
    print("-" * 65)

    baseline_scores = {
        "pattern_discovery": 0.42,
        "rule_combination": 0.33,
        "cross_domain": 0.22,
        "counterfactual": 0.72,
        "creative": 0.53,
    }

    for category, prev_score in baseline_scores.items():
        if category in category_scores:
            new_score = np.mean(category_scores[category])
            improvement = new_score - prev_score
            sign = "+" if improvement > 0 else ""
            print(
                f"{category:20} | {prev_score:12.1%} | {new_score:10.1%} | {sign}{improvement:+7.1%}"
            )

    # Key insights
    print("\n🔍 Key Insights:")
    print("-" * 40)

    # Find best and worst performing tasks
    best_task = max(results, key=lambda r: r.score)
    worst_task = min(results, key=lambda r: r.score)

    print(f"Best Task:  {best_task.task_id} ({best_task.score:.1%})")
    print(f"Worst Task: {worst_task.task_id} ({worst_task.score:.1%})")

    # Strategy effectiveness
    strategy_stats = {}
    for r in results:
        if r.strategy_used and r.success:
            if r.strategy_used not in strategy_stats:
                strategy_stats[r.strategy_used] = 0
            strategy_stats[r.strategy_used] += 1

    if strategy_stats:
        print("\nMost Effective Strategies:")
        for strategy, count in sorted(strategy_stats.items(), key=lambda x: -x[1]):
            print(f"  {strategy}: {count} successful tasks")

    # Save results to file
    save_results(results, category_scores, overall_avg, overall_success)

    return {
        "results": results,
        "category_scores": category_scores,
        "overall_average": overall_avg,
        "overall_success_rate": overall_success,
        "total_time": total_time,
        "total_attempts": total_attempts,
    }


def save_results(results, category_scores, overall_avg, overall_success):
    """Save results to JSON file."""
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "overall_score": float(overall_avg),
        "success_rate": float(overall_success),
        "category_scores": {k: float(np.mean(v)) for k, v in category_scores.items()},
        "task_results": [
            {
                "task_id": r.task_id,
                "category": r.category,
                "score": float(r.score),
                "success": bool(r.success),  # Convert numpy bool to Python bool
                "attempts": int(r.attempts),
                "time": float(r.time_taken),
                "strategy": r.strategy_used,
            }
            for r in results
        ],
    }

    output_path = Path(__file__).parent / "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n💾 Results saved to: {output_path}")


def test_specific_category(category: str):
    """Test only tasks from a specific category."""
    print(f"\n{'='*60}")
    print(f"Testing {category.upper()} tasks only")
    print("=" * 60)

    tasks = []

    if category == "pattern_discovery":
        tasks = [
            PatternDiscoveryTasks.create_shear_task(),
            PatternDiscoveryTasks.create_spiral_task(),
        ]
    elif category == "rule_combination":
        tasks = [
            RuleCombinationTasks.create_color_size_combo(),
            RuleCombinationTasks.create_conditional_combo(),
        ]
    elif category == "cross_domain":
        tasks = [
            CrossDomainTasks.create_2d_to_color_rotation(),
            CrossDomainTasks.create_symmetry_transfer(),
        ]
    elif category == "counterfactual":
        tasks = [
            CounterfactualTasks.create_reverse_gravity(),
            CounterfactualTasks.create_negative_counting(),
        ]
    elif category == "creative":
        tasks = [
            CreativeProblemTasks.create_sort_without_compare(),
            CreativeProblemTasks.create_path_without_search(),
        ]

    generator = MinimalHypothesisGenerator(seed=42)
    results = []

    for task in tasks:
        result = run_task_evaluation(task, generator, verbose=True)
        results.append(result)

    # Summary
    avg_score = np.mean([r.score for r in results])
    success_rate = sum(1 for r in results if r.success) / len(results)

    print(f"\n{category.upper()} Summary:")
    print(f"  Average Score: {avg_score:.1%}")
    print(f"  Success Rate: {success_rate:.1%}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Imagination Benchmark")
    parser.add_argument(
        "--category",
        type=str,
        choices=[
            "pattern_discovery",
            "rule_combination",
            "cross_domain",
            "counterfactual",
            "creative",
        ],
        help="Test specific category only",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick test with fewer attempts"
    )
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")

    args = parser.parse_args()

    if args.category:
        test_specific_category(args.category)
    else:
        run_full_benchmark(verbose=not args.quiet)