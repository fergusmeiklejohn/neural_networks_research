"""Evaluate Imagination Engine V3 on real ARC-AGI-2 dataset."""

import numpy as np
from pathlib import Path
import time
import json
from typing import Dict, List, Any
from tqdm import tqdm

from imagination_engine_v3 import ImaginationEngineV3
from arc_data_loader import load_arc_training_data, prepare_task_for_hti


def evaluate_solution(predictions: List[np.ndarray], test_examples: List) -> float:
    """Evaluate predictions against test examples."""
    
    if not predictions or not test_examples:
        return 0.0
    
    correct = 0
    total = 0
    
    for pred, (inp, expected) in zip(predictions, test_examples):
        if pred is not None and np.array_equal(pred, expected):
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0.0


def run_arc_evaluation(
    max_tasks: int = 100,
    timeout_per_task: float = 10.0,
    verbose: bool = False
) -> Dict[str, Any]:
    """Run evaluation on ARC-AGI-2 training dataset.
    
    Args:
        max_tasks: Maximum number of tasks to evaluate
        timeout_per_task: Maximum time per task in seconds
        verbose: Whether to print detailed progress
        
    Returns:
        Dictionary with evaluation results
    """
    
    print("=" * 70)
    print("IMAGINATION ENGINE V3 - REAL ARC-AGI-2 EVALUATION")
    print("=" * 70)
    
    # Load ARC training data
    print("\nLoading ARC-AGI-2 training data...")
    try:
        arc_tasks = load_arc_training_data(max_tasks=max_tasks)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return {}
    
    print(f"Loaded {len(arc_tasks)} tasks for evaluation")
    
    # Initialize engine
    print("\nInitializing Imagination Engine V3...")
    engine = ImaginationEngineV3(
        memory_path=Path("arc_real_evaluation_memory.json"),
        memory_capacity=500,
        enable_learning=True,
        verbose=verbose
    )
    
    # Try to load existing memory
    engine.load_memory()
    initial_inventions = len(engine.invention_memory.inventions)
    print(f"Starting with {initial_inventions} stored inventions")
    
    # Initialize results
    results = {
        "total_tasks": len(arc_tasks),
        "successful": 0,
        "perfect": 0,
        "partial": 0,
        "failed": 0,
        "by_strategy": {},
        "by_accuracy": {
            "100%": 0,
            "80-99%": 0,
            "50-79%": 0,
            "1-49%": 0,
            "0%": 0
        },
        "timing": {
            "total_time": 0,
            "avg_time": 0,
            "max_time": 0,
            "min_time": float('inf')
        },
        "memory_performance": {
            "initial_inventions": initial_inventions,
            "final_inventions": 0,
            "memory_hits": 0,
            "new_inventions": 0
        },
        "task_details": []
    }
    
    # Process tasks
    print(f"\nEvaluating {len(arc_tasks)} tasks...")
    print("=" * 70)
    
    overall_start = time.time()
    
    for i, task in enumerate(tqdm(arc_tasks, desc="Processing")):
        task_start = time.time()
        task_id = task['id']
        
        try:
            # Prepare task data
            train_examples, test_examples = prepare_task_for_hti(task)
            
            # Convert to engine format
            engine_task = {
                'train': [{'input': inp.tolist(), 'output': out.tolist()} 
                         for inp, out in train_examples],
                'test': [{'input': inp.tolist()} for inp, _ in test_examples]
            }
            
            # Solve task
            solution = engine.solve(engine_task, timeout=timeout_per_task)
            
            # Evaluate accuracy
            if test_examples:
                accuracy = evaluate_solution(solution.predictions, test_examples)
            else:
                accuracy = solution.accuracy  # Use training accuracy
            
            # Update statistics
            if accuracy >= 1.0:
                results["perfect"] += 1
                results["successful"] += 1
                results["by_accuracy"]["100%"] += 1
            elif accuracy >= 0.8:
                results["successful"] += 1
                results["by_accuracy"]["80-99%"] += 1
            elif accuracy >= 0.5:
                results["partial"] += 1
                results["by_accuracy"]["50-79%"] += 1
            elif accuracy > 0:
                results["by_accuracy"]["1-49%"] += 1
            else:
                results["failed"] += 1
                results["by_accuracy"]["0%"] += 1
            
            # Track strategy
            strategy = solution.strategy_used
            results["by_strategy"][strategy] = results["by_strategy"].get(strategy, 0) + 1
            
            # Track timing
            task_time = time.time() - task_start
            results["timing"]["max_time"] = max(results["timing"]["max_time"], task_time)
            results["timing"]["min_time"] = min(results["timing"]["min_time"], task_time)
            
            # Store task details
            results["task_details"].append({
                "task_id": task_id,
                "accuracy": float(accuracy),
                "strategy": strategy,
                "time": task_time,
                "invention_used": solution.invention_used,
                "new_invention": solution.new_invention
            })
            
            # Print progress for interesting cases
            if verbose or (i < 5) or accuracy >= 0.8:
                status = "✓" if accuracy >= 0.8 else "✗"
                print(f"{status} Task {task_id}: {accuracy:.1%} via {strategy} ({task_time:.2f}s)")
            
        except Exception as e:
            results["failed"] += 1
            results["by_accuracy"]["0%"] += 1
            results["task_details"].append({
                "task_id": task_id,
                "accuracy": 0.0,
                "strategy": "error",
                "error": str(e)
            })
            if verbose:
                print(f"✗ Task {task_id}: Error - {e}")
    
    # Compute final statistics
    total_time = time.time() - overall_start
    results["timing"]["total_time"] = total_time
    results["timing"]["avg_time"] = total_time / len(arc_tasks) if arc_tasks else 0
    
    # Get engine statistics
    engine_stats = engine.get_statistics()
    results["memory_performance"]["final_inventions"] = len(engine.invention_memory.inventions)
    results["memory_performance"]["memory_hits"] = engine_stats["engine"]["memory_hits"]
    results["memory_performance"]["new_inventions"] = engine_stats["engine"]["new_inventions"]
    results["memory_performance"]["cache_hit_rate"] = engine_stats["memory"]["cache_hit_rate"]
    
    # Save memory
    engine.save_memory()
    
    # Save results
    results_file = Path(f"arc_real_evaluation_results_{int(time.time())}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f"\nResults saved to {results_file}")
    
    return results


def print_summary(results: Dict[str, Any]):
    """Print evaluation summary."""
    
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    
    total = results["total_tasks"]
    
    # Overall performance
    print(f"\nOverall Performance:")
    print(f"  Total tasks: {total}")
    print(f"  Perfect (100%): {results['perfect']} ({results['perfect']/total*100:.1f}%)")
    print(f"  Successful (≥80%): {results['successful']} ({results['successful']/total*100:.1f}%)")
    print(f"  Partial (≥50%): {results['partial']} ({results['partial']/total*100:.1f}%)")
    print(f"  Failed: {results['failed']} ({results['failed']/total*100:.1f}%)")
    
    # Accuracy distribution
    print(f"\nAccuracy Distribution:")
    for range_name, count in results["by_accuracy"].items():
        pct = count / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {range_name:8s}: {count:4d} ({pct:5.1f}%) {bar}")
    
    # Top strategies
    print(f"\nTop Strategies:")
    sorted_strategies = sorted(results["by_strategy"].items(), 
                              key=lambda x: x[1], reverse=True)[:5]
    for strategy, count in sorted_strategies:
        pct = count / total * 100
        print(f"  {strategy:30s}: {count:3d} ({pct:5.1f}%)")
    
    # Memory performance
    print(f"\nMemory Performance:")
    mem = results["memory_performance"]
    print(f"  Initial inventions: {mem['initial_inventions']}")
    print(f"  Final inventions: {mem['final_inventions']}")
    print(f"  New inventions created: {mem['new_inventions']}")
    print(f"  Memory hits: {mem['memory_hits']}")
    if 'cache_hit_rate' in mem:
        print(f"  Cache hit rate: {mem['cache_hit_rate']:.1%}")
    
    # Timing
    print(f"\nTiming Statistics:")
    timing = results["timing"]
    print(f"  Total time: {timing['total_time']:.1f}s")
    print(f"  Average per task: {timing['avg_time']:.2f}s")
    print(f"  Max time: {timing['max_time']:.2f}s")
    if timing['min_time'] != float('inf'):
        print(f"  Min time: {timing['min_time']:.2f}s")
    
    # Best performing tasks
    if results["task_details"]:
        sorted_tasks = sorted(results["task_details"], 
                            key=lambda x: x["accuracy"], reverse=True)
        
        print(f"\nTop 5 Best Solved Tasks:")
        for task in sorted_tasks[:5]:
            if task["accuracy"] > 0:
                print(f"  {task['task_id']}: {task['accuracy']:.1%} via {task['strategy']}")
        
        # Tasks that used memory
        memory_tasks = [t for t in results["task_details"] if t.get("invention_used")]
        if memory_tasks:
            print(f"\nTasks Solved Using Memory ({len(memory_tasks)} total):")
            for task in memory_tasks[:5]:
                print(f"  {task['task_id']}: used {task['invention_used']}")


def main():
    """Main evaluation function."""
    
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Imagination Engine V3 on ARC-AGI-2")
    parser.add_argument("--max-tasks", type=int, default=100,
                       help="Maximum number of tasks to evaluate")
    parser.add_argument("--timeout", type=float, default=10.0,
                       help="Timeout per task in seconds")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed progress")
    
    args = parser.parse_args()
    
    # Run evaluation
    results = run_arc_evaluation(
        max_tasks=args.max_tasks,
        timeout_per_task=args.timeout,
        verbose=args.verbose
    )
    
    if results:
        # Print summary
        print_summary(results)
        
        # Final message
        success_rate = results["successful"] / results["total_tasks"] * 100
        print("\n" + "=" * 70)
        print(f"FINAL: {success_rate:.1f}% success rate on {results['total_tasks']} ARC tasks")
        print(f"Created {results['memory_performance']['new_inventions']} new inventions")
        print("=" * 70)


if __name__ == "__main__":
    main()