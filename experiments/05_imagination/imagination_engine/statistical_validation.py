"""Statistical validation of Imagination Engine with multiple seeds.

This script runs the full benchmark multiple times with different random seeds
to calculate confidence intervals and verify robustness of results.
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from final_integrated_system import FinalIntegratedSystem
from core.imagination_benchmark import (
    CounterfactualTasks,
    CreativeProblemTasks,
    CrossDomainTasks,
    PatternDiscoveryTasks,
    RuleCombinationTasks,
)


def run_single_evaluation(seed: int) -> Dict:
    """Run a complete benchmark evaluation with a specific seed.
    
    Args:
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with results for all tasks
    """
    print(f"\n{'='*60}")
    print(f"Running evaluation with seed {seed}")
    print(f"{'='*60}")
    
    # Set random seeds
    np.random.seed(seed)
    
    # Initialize system with this seed
    system = FinalIntegratedSystem()
    system.hypothesis_gen.rng = np.random.RandomState(seed)
    
    # Create all tasks
    tasks = [
        (PatternDiscoveryTasks.create_shear_task(), "pattern_discovery", "shear"),
        (PatternDiscoveryTasks.create_spiral_task(), "pattern_discovery", "spiral"),
        (RuleCombinationTasks.create_color_size_combo(), "rule_combination", "color_size"),
        (RuleCombinationTasks.create_conditional_combo(), "rule_combination", "conditional"),
        (CrossDomainTasks.create_2d_to_color_rotation(), "cross_domain", "2d_to_color"),
        (CrossDomainTasks.create_symmetry_transfer(), "cross_domain", "symmetry"),
        (CounterfactualTasks.create_reverse_gravity(), "counterfactual", "reverse_gravity"),
        (CounterfactualTasks.create_negative_counting(), "counterfactual", "negative_counting"),
        (CreativeProblemTasks.create_sort_without_compare(), "creative", "creative_sort"),
        (CreativeProblemTasks.create_path_without_search(), "creative", "path_finding"),
    ]
    
    results = {
        "seed": seed,
        "tasks": {},
        "categories": {},
        "overall": 0.0,
        "timing": {}
    }
    
    category_scores = {}
    
    for task, category, task_name in tasks:
        start_time = time.time()
        score, method = system.solve_task(task, category)
        elapsed = time.time() - start_time
        
        # Store task result
        results["tasks"][task_name] = {
            "score": float(score),
            "method": method,
            "time": elapsed
        }
        
        # Accumulate category scores
        if category not in category_scores:
            category_scores[category] = []
        category_scores[category].append(float(score))
        
        print(f"  {task_name:20} | {score:6.1%} | {elapsed:.2f}s")
    
    # Calculate category averages
    for category, scores in category_scores.items():
        results["categories"][category] = float(np.mean(scores))
    
    # Calculate overall average
    all_scores = [r["score"] for r in results["tasks"].values()]
    results["overall"] = float(np.mean(all_scores))
    
    print(f"\nOverall score for seed {seed}: {results['overall']:.1%}")
    
    return results


def calculate_statistics(all_results: List[Dict]) -> Dict:
    """Calculate statistics across multiple runs.
    
    Args:
        all_results: List of result dictionaries from multiple runs
        
    Returns:
        Dictionary with statistical analysis
    """
    stats_dict = {
        "n_runs": len(all_results),
        "tasks": {},
        "categories": {},
        "overall": {}
    }
    
    # Collect scores by task
    task_scores = {}
    for result in all_results:
        for task_name, task_data in result["tasks"].items():
            if task_name not in task_scores:
                task_scores[task_name] = []
            task_scores[task_name].append(task_data["score"])
    
    # Calculate task statistics
    for task_name, scores in task_scores.items():
        scores_array = np.array(scores)
        stats_dict["tasks"][task_name] = {
            "mean": float(np.mean(scores_array)),
            "std": float(np.std(scores_array)),
            "min": float(np.min(scores_array)),
            "max": float(np.max(scores_array)),
            "ci_95": calculate_confidence_interval(scores_array),
            "success_rate": float(np.mean(scores_array > 0.5))
        }
    
    # Collect scores by category
    category_scores = {}
    for result in all_results:
        for category, score in result["categories"].items():
            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(score)
    
    # Calculate category statistics
    for category, scores in category_scores.items():
        scores_array = np.array(scores)
        stats_dict["categories"][category] = {
            "mean": float(np.mean(scores_array)),
            "std": float(np.std(scores_array)),
            "min": float(np.min(scores_array)),
            "max": float(np.max(scores_array)),
            "ci_95": calculate_confidence_interval(scores_array)
        }
    
    # Calculate overall statistics
    overall_scores = np.array([r["overall"] for r in all_results])
    stats_dict["overall"] = {
        "mean": float(np.mean(overall_scores)),
        "std": float(np.std(overall_scores)),
        "min": float(np.min(overall_scores)),
        "max": float(np.max(overall_scores)),
        "ci_95": calculate_confidence_interval(overall_scores),
        "all_scores": overall_scores.tolist()
    }
    
    return stats_dict


def calculate_confidence_interval(scores: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate confidence interval for scores.
    
    Args:
        scores: Array of scores
        confidence: Confidence level (default 0.95 for 95% CI)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if len(scores) < 2:
        return (float(scores[0]), float(scores[0])) if len(scores) == 1 else (0.0, 0.0)
    
    mean = np.mean(scores)
    sem = stats.sem(scores)  # Standard error of the mean
    ci = stats.t.interval(confidence, len(scores) - 1, loc=mean, scale=sem)
    
    return (float(ci[0]), float(ci[1]))


def print_statistical_report(statistics: Dict):
    """Print a formatted statistical report.
    
    Args:
        statistics: Dictionary with statistical analysis
    """
    print("\n" + "=" * 80)
    print("STATISTICAL VALIDATION REPORT")
    print("=" * 80)
    
    print(f"\nNumber of runs: {statistics['n_runs']}")
    
    # Overall performance
    overall = statistics["overall"]
    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"  Mean:          {overall['mean']:.1%}")
    print(f"  Std Dev:       {overall['std']:.1%}")
    print(f"  95% CI:        [{overall['ci_95'][0]:.1%}, {overall['ci_95'][1]:.1%}]")
    print(f"  Range:         [{overall['min']:.1%}, {overall['max']:.1%}]")
    
    # Category performance
    print(f"\n📈 CATEGORY PERFORMANCE:")
    print("-" * 60)
    print(f"{'Category':20} | {'Mean':7} | {'Std':7} | {'95% CI':20}")
    print("-" * 60)
    
    for category, stats in statistics["categories"].items():
        ci_str = f"[{stats['ci_95'][0]:.1%}, {stats['ci_95'][1]:.1%}]"
        print(f"{category:20} | {stats['mean']:6.1%} | {stats['std']:6.1%} | {ci_str:20}")
    
    # Task performance
    print(f"\n📋 INDIVIDUAL TASK PERFORMANCE:")
    print("-" * 70)
    print(f"{'Task':20} | {'Mean':7} | {'Std':7} | {'Success Rate':12} | {'95% CI':20}")
    print("-" * 70)
    
    for task, stats in statistics["tasks"].items():
        ci_str = f"[{stats['ci_95'][0]:.1%}, {stats['ci_95'][1]:.1%}]"
        print(f"{task:20} | {stats['mean']:6.1%} | {stats['std']:6.1%} | {stats['success_rate']:11.1%} | {ci_str:20}")
    
    # Robustness analysis
    print(f"\n🔍 ROBUSTNESS ANALYSIS:")
    
    # Find most/least stable tasks
    task_stabilities = [(task, stats["std"]) for task, stats in statistics["tasks"].items()]
    task_stabilities.sort(key=lambda x: x[1])
    
    print(f"\nMost stable tasks (lowest std dev):")
    for task, std in task_stabilities[:3]:
        print(f"  - {task}: {std:.1%} std dev")
    
    print(f"\nLeast stable tasks (highest std dev):")
    for task, std in task_stabilities[-3:]:
        print(f"  - {task}: {std:.1%} std dev")
    
    # Statistical significance
    print(f"\n📊 STATISTICAL SIGNIFICANCE:")
    baseline = 0.15  # Original baseline performance
    overall_mean = overall['mean']
    overall_ci = overall['ci_95']
    
    if overall_ci[0] > baseline:
        improvement = (overall_mean - baseline) / baseline * 100
        print(f"  ✅ Significant improvement over baseline ({baseline:.1%})")
        print(f"     Mean improvement: {improvement:.0f}%")
        print(f"     Lower bound of 95% CI ({overall_ci[0]:.1%}) > baseline")
    else:
        print(f"  ⚠️  95% CI overlaps with baseline ({baseline:.1%})")


def main():
    """Run statistical validation with multiple seeds."""
    
    print("=" * 80)
    print("IMAGINATION ENGINE - STATISTICAL VALIDATION")
    print("=" * 80)
    
    # Configuration
    n_seeds = 10
    seeds = list(range(42, 42 + n_seeds))  # Deterministic seeds
    
    print(f"\nRunning {n_seeds} evaluations with seeds: {seeds}")
    
    # Run evaluations
    all_results = []
    start_time = time.time()
    
    for i, seed in enumerate(seeds, 1):
        print(f"\n[{i}/{n_seeds}] Evaluation with seed {seed}")
        result = run_single_evaluation(seed)
        all_results.append(result)
        
        # Show running average
        current_scores = [r["overall"] for r in all_results]
        running_mean = np.mean(current_scores)
        running_std = np.std(current_scores) if len(current_scores) > 1 else 0
        print(f"\nRunning average: {running_mean:.1%} ± {running_std:.1%}")
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    statistics = calculate_statistics(all_results)
    
    # Print report
    print_statistical_report(statistics)
    
    # Save results
    output_file = Path(__file__).parent / "statistical_validation_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "configuration": {
                "n_seeds": n_seeds,
                "seeds": seeds,
                "total_time": total_time
            },
            "raw_results": all_results,
            "statistics": statistics
        }, f, indent=2)
    
    print(f"\n💾 Results saved to {output_file}")
    print(f"⏱️  Total time: {total_time:.1f} seconds")
    
    # Final verdict
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    
    mean_score = statistics["overall"]["mean"]
    ci_lower, ci_upper = statistics["overall"]["ci_95"]
    
    if ci_lower >= 0.70:
        print(f"✅ ROBUST SUCCESS: Lower bound of 95% CI ({ci_lower:.1%}) exceeds 70% target")
    elif mean_score >= 0.70:
        print(f"🔶 PARTIAL SUCCESS: Mean ({mean_score:.1%}) exceeds 70% but CI includes values below")
    else:
        print(f"❌ Target not robustly achieved: Mean {mean_score:.1%}, CI [{ci_lower:.1%}, {ci_upper:.1%}]")
    
    return statistics


if __name__ == "__main__":
    statistics = main()