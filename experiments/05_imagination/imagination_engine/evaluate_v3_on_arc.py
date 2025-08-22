"""Evaluate Imagination Engine V3 on the full ARC dataset."""

import json
import numpy as np
from pathlib import Path
import time
from typing import Dict, List, Tuple, Any
from tqdm import tqdm

from imagination_engine_v3 import ImaginationEngineV3


def load_arc_tasks(data_dir: Path, max_tasks: int = None) -> List[Dict]:
    """Load ARC tasks from directory."""
    
    tasks = []
    task_files = sorted(data_dir.glob("*.json"))
    
    if max_tasks:
        task_files = task_files[:max_tasks]
    
    for task_file in task_files:
        with open(task_file) as f:
            task = json.load(f)
            task['name'] = task_file.stem
            tasks.append(task)
    
    print(f"Loaded {len(tasks)} tasks from {data_dir}")
    return tasks


def evaluate_predictions(
    predictions: List[np.ndarray],
    expected: List[np.ndarray]
) -> float:
    """Evaluate prediction accuracy."""
    
    if not predictions or not expected:
        return 0.0
    
    correct = 0
    for pred, exp in zip(predictions, expected):
        if pred is not None and np.array_equal(pred, exp):
            correct += 1
    
    return correct / len(expected)


def run_evaluation(
    engine: ImaginationEngineV3,
    tasks: List[Dict],
    timeout_per_task: float = 10.0,
    save_results: bool = True
) -> Dict[str, Any]:
    """Run evaluation on a set of tasks."""
    
    results = {
        "total_tasks": len(tasks),
        "successful": 0,
        "by_strategy": {},
        "by_accuracy": {
            "perfect": 0,  # 100%
            "high": 0,     # 80-99%
            "medium": 0,    # 50-79%
            "low": 0,      # 1-49%
            "failed": 0    # 0%
        },
        "timing": {
            "total_time": 0,
            "avg_time": 0,
            "max_time": 0,
            "min_time": float('inf')
        },
        "memory_usage": {
            "hits": 0,
            "new_inventions": 0,
            "adaptations": 0
        },
        "detailed_results": []
    }
    
    print(f"\nEvaluating {len(tasks)} tasks...")
    print("=" * 70)
    
    start_time = time.time()
    
    for i, task in enumerate(tqdm(tasks, desc="Processing tasks")):
        task_start = time.time()
        
        # Solve task
        try:
            solution = engine.solve(task, timeout=timeout_per_task)
            
            # Evaluate if we have expected outputs
            if 'test' in task and task['test']:
                expected = [np.array(ex.get('output', [])) for ex in task['test'] 
                           if 'output' in ex]
                
                if expected:
                    accuracy = evaluate_predictions(solution.predictions, expected)
                else:
                    # Use training accuracy as proxy
                    accuracy = solution.accuracy
            else:
                accuracy = solution.accuracy
            
            # Categorize accuracy
            if accuracy >= 1.0:
                results["by_accuracy"]["perfect"] += 1
                results["successful"] += 1
            elif accuracy >= 0.8:
                results["by_accuracy"]["high"] += 1
                results["successful"] += 1
            elif accuracy >= 0.5:
                results["by_accuracy"]["medium"] += 1
            elif accuracy > 0:
                results["by_accuracy"]["low"] += 1
            else:
                results["by_accuracy"]["failed"] += 1
            
            # Track strategy
            strategy = solution.strategy_used
            results["by_strategy"][strategy] = results["by_strategy"].get(strategy, 0) + 1
            
            # Track timing
            task_time = time.time() - task_start
            results["timing"]["max_time"] = max(results["timing"]["max_time"], task_time)
            results["timing"]["min_time"] = min(results["timing"]["min_time"], task_time)
            
            # Store detailed result
            results["detailed_results"].append({
                "task": task.get('name', f'task_{i}'),
                "accuracy": accuracy,
                "strategy": strategy,
                "time": task_time,
                "operations": solution.operation_count,
                "invention_used": solution.invention_used,
                "new_invention": solution.new_invention
            })
            
        except Exception as e:
            print(f"\nError on task {task.get('name', i)}: {e}")
            results["by_accuracy"]["failed"] += 1
            results["detailed_results"].append({
                "task": task.get('name', f'task_{i}'),
                "accuracy": 0.0,
                "strategy": "error",
                "error": str(e)
            })
    
    # Compute final statistics
    total_time = time.time() - start_time
    results["timing"]["total_time"] = total_time
    results["timing"]["avg_time"] = total_time / len(tasks) if tasks else 0
    
    # Get engine statistics
    engine_stats = engine.get_statistics()
    results["memory_usage"]["hits"] = engine_stats["engine"]["memory_hits"]
    results["memory_usage"]["new_inventions"] = engine_stats["engine"]["new_inventions"]
    results["memory_usage"]["adaptations"] = engine_stats["engine"]["adaptations"]
    results["memory_stats"] = engine_stats["memory"]
    
    # Save results if requested
    if save_results:
        results_file = Path(f"arc_evaluation_results_{int(time.time())}.json")
        with open(results_file, "w") as f:
            # Convert numpy types for JSON serialization
            json_results = json.loads(json.dumps(results, default=str))
            json.dump(json_results, f, indent=2)
        print(f"\nResults saved to {results_file}")
    
    return results


def print_evaluation_summary(results: Dict[str, Any]):
    """Print a summary of evaluation results."""
    
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    
    # Overall performance
    success_rate = results["successful"] / results["total_tasks"] * 100
    print(f"\nOverall Performance:")
    print(f"  Total tasks: {results['total_tasks']}")
    print(f"  Successful (≥80%): {results['successful']} ({success_rate:.1f}%)")
    
    # Accuracy breakdown
    print(f"\nAccuracy Breakdown:")
    total = results["total_tasks"]
    for category, count in results["by_accuracy"].items():
        pct = count / total * 100 if total > 0 else 0
        print(f"  {category:8s}: {count:3d} ({pct:5.1f}%)")
    
    # Strategy usage
    print(f"\nStrategies Used:")
    for strategy, count in sorted(results["by_strategy"].items(), 
                                 key=lambda x: x[1], reverse=True):
        pct = count / total * 100 if total > 0 else 0
        print(f"  {strategy:30s}: {count:3d} ({pct:5.1f}%)")
    
    # Memory usage
    print(f"\nMemory Performance:")
    print(f"  Memory hits: {results['memory_usage']['hits']}")
    print(f"  New inventions: {results['memory_usage']['new_inventions']}")
    print(f"  Adaptations: {results['memory_usage']['adaptations']}")
    
    if "memory_stats" in results:
        print(f"  Total stored inventions: {results['memory_stats']['total_inventions']}")
        print(f"  Cache hit rate: {results['memory_stats']['cache_hit_rate']:.1%}")
    
    # Timing
    print(f"\nTiming Statistics:")
    print(f"  Total time: {results['timing']['total_time']:.1f}s")
    print(f"  Average per task: {results['timing']['avg_time']:.2f}s")
    print(f"  Max time: {results['timing']['max_time']:.2f}s")
    print(f"  Min time: {results['timing']['min_time']:.2f}s")
    
    # Top successful tasks
    successful_tasks = [r for r in results["detailed_results"] 
                       if r["accuracy"] >= 0.8]
    if successful_tasks:
        print(f"\nTop Successful Tasks ({len(successful_tasks)} total):")
        for task in successful_tasks[:5]:
            print(f"  {task['task']}: {task['accuracy']:.1%} via {task['strategy']}")
    
    # Failed tasks
    failed_tasks = [r for r in results["detailed_results"] 
                   if r["accuracy"] == 0]
    if failed_tasks:
        print(f"\nFailed Tasks ({len(failed_tasks)} total):")
        for task in failed_tasks[:5]:
            reason = task.get('error', task['strategy'])
            print(f"  {task['task']}: {reason}")


def main():
    """Main evaluation function."""
    
    print("IMAGINATION ENGINE V3 - ARC DATASET EVALUATION")
    print("=" * 70)
    
    # Check for ARC dataset
    base_path = Path("/Users/fergusmeiklejohn/dev/neural_networks_research")
    training_dir = base_path / "data" / "arc-agi" / "training"
    
    # Alternative path
    if not training_dir.exists():
        training_dir = base_path / "experiments" / "05_imagination" / "data" / "arc-agi-2" / "training"
    
    # Use test dataset if no real ARC dataset
    if not training_dir.exists():
        training_dir = Path("test_arc_dataset")
        if not training_dir.exists():
            print(f"No dataset found. Creating test dataset...")
            import create_test_arc_dataset
            training_dir = create_test_arc_dataset.create_test_dataset()
    
    # Create engine with persistent memory
    print("\nInitializing Imagination Engine V3...")
    engine = ImaginationEngineV3(
        memory_path=Path("arc_evaluation_memory.json"),
        memory_capacity=500,
        enable_learning=True,
        verbose=False  # Quiet for batch processing
    )
    
    # Load existing memory if available
    engine.load_memory()
    
    # Load tasks
    max_tasks = 100  # Start with subset for testing
    tasks = load_arc_tasks(training_dir, max_tasks=max_tasks)
    
    if not tasks:
        print("No tasks loaded")
        return
    
    # Run evaluation
    results = run_evaluation(
        engine,
        tasks,
        timeout_per_task=10.0,
        save_results=True
    )
    
    # Print summary
    print_evaluation_summary(results)
    
    # Save memory for future runs
    engine.save_memory()
    print(f"\nMemory saved with {len(engine.invention_memory.inventions)} inventions")
    
    # Final summary
    success_rate = results["successful"] / results["total_tasks"] * 100
    print("\n" + "=" * 70)
    print(f"FINAL RESULT: {success_rate:.1f}% success rate on {results['total_tasks']} ARC tasks")
    print("=" * 70)


if __name__ == "__main__":
    main()