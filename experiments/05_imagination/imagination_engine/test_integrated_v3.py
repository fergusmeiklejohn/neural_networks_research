"""Test the integrated Imagination Engine V3."""

import numpy as np
from pathlib import Path
import json
import time

from imagination_engine_v3 import ImaginationEngineV3


def create_test_tasks():
    """Create test ARC-style tasks."""
    
    tasks = []
    
    # Task 1: Simple increment (should be solved quickly)
    task1 = {
        "name": "increment",
        "train": [
            {"input": [[1, 2], [3, 4]], "output": [[2, 3], [4, 5]]},
            {"input": [[5, 6], [7, 8]], "output": [[6, 7], [8, 9]]}
        ],
        "test": [
            {"input": [[10, 11], [12, 13]]}
        ]
    }
    tasks.append(task1)
    
    # Task 2: Color swap
    task2 = {
        "name": "color_swap",
        "train": [
            {"input": [[1, 2, 1], [2, 1, 2]], "output": [[2, 1, 2], [1, 2, 1]]},
            {"input": [[3, 4, 3], [4, 3, 4]], "output": [[4, 3, 4], [3, 4, 3]]}
        ],
        "test": [
            {"input": [[5, 6, 5], [6, 5, 6]]}
        ]
    }
    tasks.append(task2)
    
    # Task 3: Cross pattern (requires geometric reasoning)
    task3 = {
        "name": "cross_pattern",
        "train": [
            {
                "input": [
                    [0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 2, 0],
                    [0, 0, 0, 0, 0]
                ],
                "output": [
                    [0, 1, 0, 2, 0],
                    [1, 1, 1, 2, 1],
                    [0, 1, 0, 2, 0],
                    [2, 2, 2, 2, 2],
                    [0, 1, 0, 2, 0]
                ]
            }
        ],
        "test": [
            {
                "input": [
                    [0, 0, 0, 0, 0],
                    [0, 0, 3, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 4, 0, 0, 0],
                    [0, 0, 0, 0, 0]
                ]
            }
        ]
    }
    tasks.append(task3)
    
    # Task 4: Diagonal fill
    task4 = {
        "name": "diagonal_fill",
        "train": [
            {"input": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 
             "output": [[3, 0, 0], [0, 3, 0], [0, 0, 3]]},
            {"input": [[2, 0, 0], [0, 2, 0], [0, 0, 2]], 
             "output": [[3, 0, 0], [0, 3, 0], [0, 0, 3]]}
        ],
        "test": [
            {"input": [[5, 0, 0], [0, 5, 0], [0, 0, 5]]}
        ]
    }
    tasks.append(task4)
    
    return tasks


def test_basic_solving():
    """Test basic solving capabilities."""
    
    print("=" * 70)
    print("TESTING BASIC SOLVING CAPABILITIES")
    print("=" * 70)
    
    # Create engine
    engine = ImaginationEngineV3(
        memory_path=Path("test_v3_memory.json"),
        memory_capacity=100,
        enable_learning=True,
        verbose=True
    )
    
    # Get test tasks
    tasks = create_test_tasks()
    
    results = []
    
    for i, task in enumerate(tasks):
        print(f"\n{'='*50}")
        print(f"Task {i+1}: {task['name']}")
        print(f"{'='*50}")
        
        # Solve task
        start_time = time.time()
        solution = engine.solve(task, timeout=10.0)
        solve_time = time.time() - start_time
        
        # Check if we have expected output for validation
        has_expected = False
        if 'test' in task and task['test'] and 'output' in task['test'][0]:
            expected = np.array(task['test'][0]['output'])
            predicted = solution.predictions[0] if solution.predictions else None
            
            if predicted is not None:
                accuracy = np.mean(predicted == expected)
                has_expected = True
            else:
                accuracy = 0.0
        else:
            accuracy = solution.accuracy
        
        # Report results
        print(f"\nResults:")
        print(f"  Strategy: {solution.strategy_used}")
        print(f"  Accuracy: {accuracy:.1%}")
        print(f"  Time: {solve_time:.2f}s")
        
        if solution.invention_used:
            print(f"  Used invention: {solution.invention_used}")
        if solution.new_invention:
            print(f"  Created invention: {solution.new_invention}")
        if solution.operation_count:
            print(f"  Operations: {solution.operation_count}")
        
        results.append({
            "task": task['name'],
            "strategy": solution.strategy_used,
            "accuracy": accuracy,
            "time": solve_time,
            "success": accuracy > 0.8
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    successful = sum(1 for r in results if r['success'])
    print(f"\nSolved {successful}/{len(tasks)} tasks successfully")
    
    for result in results:
        status = "✓" if result['success'] else "✗"
        print(f"  {status} {result['task']}: {result['strategy']} ({result['accuracy']:.1%})")
    
    return engine, results


def test_memory_learning():
    """Test that the engine learns from experience."""
    
    print("\n" + "=" * 70)
    print("TESTING MEMORY AND LEARNING")
    print("=" * 70)
    
    # Create engine
    engine = ImaginationEngineV3(
        memory_path=Path("learning_test.json"),
        enable_learning=True,
        verbose=False
    )
    
    # Create a task
    task = {
        "name": "test_pattern",
        "train": [
            {"input": [[1, 0], [0, 1]], "output": [[2, 0], [0, 2]]},
            {"input": [[3, 0], [0, 3]], "output": [[4, 0], [0, 4]]}
        ],
        "test": [
            {"input": [[5, 0], [0, 5]]}
        ]
    }
    
    print("\n1. First solve (should invent):")
    solution1 = engine.solve(task)
    print(f"   Strategy: {solution1.strategy_used}")
    print(f"   New invention: {solution1.new_invention}")
    
    # Solve same task again
    print("\n2. Second solve (should use memory):")
    solution2 = engine.solve(task)
    print(f"   Strategy: {solution2.strategy_used}")
    print(f"   Used invention: {solution2.invention_used}")
    
    # Check statistics
    stats = engine.get_statistics()
    print("\n3. Engine Statistics:")
    print(f"   Memory hits: {stats['engine']['memory_hits']}")
    print(f"   New inventions: {stats['engine']['new_inventions']}")
    print(f"   Total stored: {stats['memory']['total_inventions']}")
    
    # Clean up
    Path("learning_test.json").unlink(missing_ok=True)
    Path("learning_test.pkl").unlink(missing_ok=True)
    
    return engine


def test_with_real_arc_task():
    """Test with a real ARC task structure."""
    
    print("\n" + "=" * 70)
    print("TESTING WITH REAL ARC TASK")
    print("=" * 70)
    
    # Load a sample ARC task (if available)
    arc_path = Path("/Users/fergusmeiklejohn/dev/neural_networks_research/data/arc-agi/training")
    
    if arc_path.exists():
        # Load first task
        task_files = list(arc_path.glob("*.json"))[:1]
        
        if task_files:
            with open(task_files[0]) as f:
                task = json.load(f)
            
            print(f"\nLoaded task: {task_files[0].stem}")
            print(f"Training examples: {len(task.get('train', []))}")
            print(f"Test examples: {len(task.get('test', []))}")
            
            # Create engine
            engine = ImaginationEngineV3(verbose=True)
            
            # Solve
            solution = engine.solve(task, timeout=20.0)
            
            print(f"\nResult: {solution.strategy_used}")
            print(f"Accuracy: {solution.accuracy:.1%}")
            
            return engine, solution
    else:
        print("ARC dataset not found at expected location")
        return None, None


def analyze_performance():
    """Analyze overall performance of the integrated system."""
    
    print("\n" + "=" * 70)
    print("PERFORMANCE ANALYSIS")
    print("=" * 70)
    
    engine, results = test_basic_solving()
    
    # Get final statistics
    stats = engine.get_statistics()
    
    print("\n1. Solution Strategies Used:")
    strategy_counts = {}
    for r in results:
        strategy = r['strategy']
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
    
    for strategy, count in strategy_counts.items():
        print(f"   {strategy}: {count}")
    
    print("\n2. Memory Statistics:")
    print(f"   Total inventions stored: {stats['memory']['total_inventions']}")
    print(f"   Cache hit rate: {stats['memory']['cache_hit_rate']:.1%}")
    print(f"   Average accuracy: {stats['memory']['avg_accuracy']:.1%}")
    print(f"   Average operations: {stats['memory']['avg_operation_count']:.1f}")
    
    print("\n3. Performance Metrics:")
    avg_time = np.mean([r['time'] for r in results])
    avg_accuracy = np.mean([r['accuracy'] for r in results])
    print(f"   Average solving time: {avg_time:.2f}s")
    print(f"   Average accuracy: {avg_accuracy:.1%}")
    print(f"   Success rate: {sum(r['success'] for r in results)}/{len(results)}")
    
    # Save memory for future use
    engine.save_memory()
    print("\n4. Memory saved for future sessions")
    
    return engine, stats


if __name__ == "__main__":
    # Run all tests
    print("IMAGINATION ENGINE V3 - INTEGRATED TESTING")
    print("=" * 70)
    
    # Test basic solving
    engine1, results = test_basic_solving()
    
    # Test memory and learning
    engine2 = test_memory_learning()
    
    # Test with real ARC if available
    test_with_real_arc_task()
    
    # Analyze performance
    final_engine, final_stats = analyze_performance()
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)
    
    # Clean up test files
    for f in ["test_v3_memory.json", "test_v3_memory.pkl", "imagination_v3_memory.json", "imagination_v3_memory.pkl"]:
        Path(f).unlink(missing_ok=True)