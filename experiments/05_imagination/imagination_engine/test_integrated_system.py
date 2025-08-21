"""Test the Integrated Imagination System on the full benchmark.

This tests whether integrating all components improves overall performance
toward our 70% target.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from core.imagination_benchmark import (
    CounterfactualTasks,
    CreativeProblemTasks,
    CrossDomainTasks,
    ImaginationBenchmark,
    PatternDiscoveryTasks,
    RuleCombinationTasks,
)
from integrated_imagination_system import IntegratedImaginationSystem


def run_integrated_benchmark(verbose: bool = True) -> Dict:
    """Run the integrated system on all benchmark tasks."""
    
    print("\n" + "=" * 80)
    print("INTEGRATED IMAGINATION SYSTEM - FULL BENCHMARK")
    print("=" * 80)
    
    # Initialize system
    system = IntegratedImaginationSystem(verbose=False)  # Reduce verbosity
    
    # Create all tasks with categories
    tasks_with_categories = [
        # Pattern Discovery
        (PatternDiscoveryTasks.create_shear_task(), "pattern_discovery"),
        (PatternDiscoveryTasks.create_spiral_task(), "pattern_discovery"),
        
        # Rule Combination
        (RuleCombinationTasks.create_color_size_combo(), "rule_combination"),
        (RuleCombinationTasks.create_conditional_combo(), "rule_combination"),
        
        # Cross-Domain
        (CrossDomainTasks.create_2d_to_color_rotation(), "cross_domain"),
        (CrossDomainTasks.create_symmetry_transfer(), "cross_domain"),
        
        # Counterfactual
        (CounterfactualTasks.create_reverse_gravity(), "counterfactual"),
        (CounterfactualTasks.create_negative_counting(), "counterfactual"),
        
        # Creative
        (CreativeProblemTasks.create_sort_without_compare(), "creative"),
        (CreativeProblemTasks.create_path_without_search(), "creative"),
    ]
    
    print(f"\nTesting {len(tasks_with_categories)} tasks with integrated system...")
    
    # Track results
    results = []
    category_scores = {}
    
    # Run each task
    for i, (task, category) in enumerate(tasks_with_categories, 1):
        if verbose:
            print(f"\n[{i}/{len(tasks_with_categories)}] {task.task_id} ({category})")
        else:
            print(f"[{i}/{len(tasks_with_categories)}]", end=" ")
        
        start_time = time.time()
        
        # Run integrated system
        result = system.imagine(
            task.test_examples,
            task_category=category,
            max_attempts=500  # Balance speed and thoroughness
        )
        
        elapsed = time.time() - start_time
        
        # Calculate score
        if result.hypothesis or result.principle or result.rule or result.program:
            # Test the solution
            test_score = 0.0
            for inp, out in task.test_examples:
                predicted = None
                
                # Apply the solution based on what was found
                if result.hypothesis:
                    predicted = result.hypothesis.apply(inp)
                elif result.principle:
                    # This would need proper implementation
                    pass
                elif result.rule:
                    # This would need proper implementation
                    pass
                
                if predicted is not None and predicted.shape == out.shape:
                    test_score += task.evaluate_solution(predicted, out)
            
            final_score = test_score / len(task.test_examples) if task.test_examples else 0.0
        else:
            final_score = result.score
        
        results.append({
            "task_id": task.task_id,
            "category": category,
            "score": final_score,
            "method": result.method_used,
            "success": result.success,
            "time": elapsed
        })
        
        # Track category scores
        if category not in category_scores:
            category_scores[category] = []
        category_scores[category].append(final_score)
        
        if verbose:
            print(f"  Score: {final_score:.1%} | Method: {result.method_used} | Time: {elapsed:.1f}s")
    
    # Calculate summary statistics
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    # Category breakdown
    print("\n📊 Category Performance:")
    print("-" * 40)
    for category, scores in category_scores.items():
        avg_score = np.mean(scores)
        print(f"{category:20} | {avg_score:6.1%}")
    
    # Overall statistics
    all_scores = [r["score"] for r in results]
    overall_avg = np.mean(all_scores)
    success_count = sum(1 for r in results if r["success"])
    total_time = sum(r["time"] for r in results)
    
    print("\n📈 Overall Performance:")
    print("-" * 40)
    print(f"Average Score:        {overall_avg:6.1%}")
    print(f"Success Rate:         {success_count}/{len(results)} tasks")
    print(f"Total Time:           {total_time:.1f}s")
    
    # Method usage
    method_counts = {}
    for r in results:
        method = r["method"]
        if method not in method_counts:
            method_counts[method] = 0
        method_counts[method] += 1
    
    print("\n🔧 Methods Used:")
    print("-" * 40)
    for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
        print(f"{method:25} | {count} tasks")
    
    # Compare to previous best
    print("\n📊 Comparison to Previous Results:")
    print("-" * 65)
    print("Component            | Previous | Integrated | Improvement")
    print("-" * 65)
    
    previous_scores = {
        "pattern_discovery": 0.92,
        "rule_combination": 0.00,
        "cross_domain": 0.00,
        "counterfactual": 0.50,
        "creative": 0.44,
    }
    
    for category in previous_scores:
        prev = previous_scores[category]
        new = np.mean(category_scores.get(category, [0]))
        diff = new - prev
        sign = "+" if diff > 0 else ""
        print(f"{category:20} | {prev:7.1%} | {new:10.1%} | {sign}{diff:+7.1%}")
    
    print("-" * 65)
    prev_overall = 0.372  # Previous result
    diff = overall_avg - prev_overall
    sign = "+" if diff > 0 else ""
    print(f"{'OVERALL':20} | {prev_overall:7.1%} | {overall_avg:10.1%} | {sign}{diff:+7.1%}")
    
    # System statistics
    print("\n📊 System Statistics:")
    print("-" * 40)
    stats = system.get_statistics()
    for key, value in stats.items():
        if key != "method_performance":
            print(f"{key}: {value}")
    
    # Success analysis
    print("\n🎯 Progress Toward 70% Goal:")
    print("-" * 40)
    print(f"Current: {overall_avg:.1%}")
    print(f"Target:  70.0%")
    print(f"Gap:     {70 - overall_avg*100:.1f}%")
    
    if overall_avg >= 0.70:
        print("\n🎉 GOAL ACHIEVED! 70% benchmark reached!")
    elif overall_avg >= 0.50:
        print("\n📈 Significant progress! Over 50% achieved.")
    elif overall_avg > prev_overall:
        print(f"\n✅ Improvement! +{(overall_avg - prev_overall)*100:.1f}% from previous.")
    else:
        print("\n📝 Integration needs refinement.")
    
    return {
        "results": results,
        "category_scores": category_scores,
        "overall_average": overall_avg,
        "success_count": success_count,
        "total_time": total_time
    }


def test_specific_failures():
    """Test on tasks that previously failed."""
    
    print("\n" + "=" * 60)
    print("Testing Previously Failed Tasks")
    print("=" * 60)
    
    system = IntegratedImaginationSystem(verbose=True)
    
    # Test rule combination (previously 0%)
    print("\n🔬 Rule Combination Task:")
    task = RuleCombinationTasks.create_color_size_combo()
    result = system.imagine(task.test_examples, task_category="rule_combination")
    print(f"Result: {result.score:.1%} using {result.method_used}")
    if result.success:
        print(system.explain_solution(result))
    
    # Test cross-domain (previously 0%)
    print("\n🔬 Cross-Domain Task:")
    task = CrossDomainTasks.create_2d_to_color_rotation()
    result = system.imagine(task.test_examples, task_category="cross_domain")
    print(f"Result: {result.score:.1%} using {result.method_used}")
    if result.success:
        print(system.explain_solution(result))


def main():
    """Run all integrated system tests."""
    
    print("=" * 80)
    print("INTEGRATED IMAGINATION SYSTEM TEST")
    print("=" * 80)
    
    # Test on specific failures first
    # test_specific_failures()
    
    # Run full benchmark
    results = run_integrated_benchmark(verbose=False)
    
    # Final message
    print("\n" + "=" * 80)
    if results["overall_average"] >= 0.50:
        print("✨ Integration shows significant improvement!")
    else:
        print("📝 Further integration refinement needed.")
    print("=" * 80)


if __name__ == "__main__":
    main()