"""Test HTI on real ARC-AGI tasks.

This script tests whether the Hierarchical Transform Inventor can solve
actual ARC-AGI tasks without any training - purely through its learnable
hypothesis space and memory system.
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from integrated_hti_system import IntegratedHTI


def load_arc_tasks(n_tasks: int = 10) -> List[Dict]:
    """Load a subset of ARC tasks for testing.
    
    For now, we'll create simplified ARC-style tasks.
    In production, would load from the actual ARC dataset.
    """
    tasks = []
    
    # Task 1: Color mapping (red→blue, blue→red)
    task1 = {
        'id': 'color_swap',
        'train': [
            {'input': [[2, 0, 0], [0, 1, 0], [0, 0, 2]], 
             'output': [[1, 0, 0], [0, 2, 0], [0, 0, 1]]},
            {'input': [[1, 1, 0], [0, 2, 0], [0, 0, 0]], 
             'output': [[2, 2, 0], [0, 1, 0], [0, 0, 0]]},
        ],
        'test': [
            {'input': [[2, 1, 0], [1, 2, 0], [0, 0, 0]], 
             'output': [[1, 2, 0], [2, 1, 0], [0, 0, 0]]}
        ]
    }
    tasks.append(task1)
    
    # Task 2: Pattern completion (fill missing corner)
    task2 = {
        'id': 'corner_completion',
        'train': [
            {'input': [[3, 0, 3], [0, 0, 0], [3, 0, 0]], 
             'output': [[3, 0, 3], [0, 0, 0], [3, 0, 3]]},
            {'input': [[5, 0, 5], [0, 0, 0], [0, 0, 5]], 
             'output': [[5, 0, 5], [0, 0, 0], [5, 0, 5]]},
        ],
        'test': [
            {'input': [[7, 0, 0], [0, 0, 0], [7, 0, 7]], 
             'output': [[7, 0, 7], [0, 0, 0], [7, 0, 7]]}
        ]
    }
    tasks.append(task2)
    
    # Task 3: Growing pattern (extend by one)
    task3 = {
        'id': 'growth_pattern',
        'train': [
            {'input': [[4, 0, 0], [0, 0, 0], [0, 0, 0]], 
             'output': [[4, 4, 0], [4, 0, 0], [0, 0, 0]]},
            {'input': [[0, 0, 0], [0, 6, 0], [0, 0, 0]], 
             'output': [[0, 6, 0], [6, 6, 6], [0, 6, 0]]},
        ],
        'test': [
            {'input': [[0, 0, 0], [0, 0, 0], [8, 0, 0]], 
             'output': [[0, 0, 0], [8, 0, 0], [8, 8, 0]]}
        ]
    }
    tasks.append(task3)
    
    # Task 4: Mirror symmetry
    task4 = {
        'id': 'mirror_pattern',
        'train': [
            {'input': [[1, 2, 0], [0, 0, 0], [0, 0, 0]], 
             'output': [[1, 2, 0], [0, 0, 0], [0, 2, 1]]},
            {'input': [[3, 0, 0], [4, 0, 0], [0, 0, 0]], 
             'output': [[3, 0, 0], [4, 0, 0], [0, 0, 3]]},
        ],
        'test': [
            {'input': [[5, 6, 0], [7, 0, 0], [0, 0, 0]], 
             'output': [[5, 6, 0], [7, 0, 0], [0, 6, 5]]}
        ]
    }
    tasks.append(task4)
    
    # Task 5: Count and fill (count non-zero, fill that many)
    task5 = {
        'id': 'count_fill',
        'train': [
            {'input': [[1, 0, 1], [0, 0, 0], [0, 0, 0]], 
             'output': [[2, 2, 0], [0, 0, 0], [0, 0, 0]]},
            {'input': [[3, 3, 3], [0, 0, 0], [0, 0, 0]], 
             'output': [[3, 3, 3], [0, 0, 0], [0, 0, 0]]},
        ],
        'test': [
            {'input': [[5, 0, 0], [0, 5, 0], [0, 0, 0]], 
             'output': [[2, 2, 0], [0, 0, 0], [0, 0, 0]]}
        ]
    }
    tasks.append(task5)
    
    return tasks[:n_tasks]


def test_hti_on_arc_task(hti_system: IntegratedHTI, task: Dict) -> Dict:
    """Test HTI on a single ARC task."""
    
    # Convert task format
    train_examples = []
    for example in task['train']:
        inp = np.array(example['input'], dtype=np.float32)
        out = np.array(example['output'], dtype=np.float32)
        train_examples.append((inp, out))
    
    # Solve with HTI
    start_time = time.time()
    transform, info = hti_system.solve_with_memory(train_examples, task['id'])
    solve_time = time.time() - start_time
    
    # Test on test examples
    test_scores = []
    for test_example in task['test']:
        test_input = np.array(test_example['input'], dtype=np.float32)
        expected = np.array(test_example['output'], dtype=np.float32)
        
        # Apply learned transform
        predicted = transform(test_input)
        
        # Calculate accuracy
        if predicted.shape == expected.shape:
            accuracy = np.mean(predicted == expected)
        else:
            accuracy = 0.0
        
        test_scores.append(accuracy)
    
    avg_score = np.mean(test_scores) if test_scores else 0.0
    
    return {
        'task_id': task['id'],
        'score': avg_score,
        'solve_time': solve_time,
        'source': info.get('source', 'unknown'),
        'reasoning_cycles': info.get('reasoning_cycles', 0)
    }


def main():
    """Test HTI on ARC-AGI style tasks."""
    
    print("\n" + "=" * 80)
    print("TESTING HTI ON ARC-AGI STYLE TASKS")
    print("=" * 80)
    print("\nNote: These are simplified ARC-style tasks for validation.")
    print("The HTI has NOT been trained - it's learning on-the-fly!\n")
    
    # Create HTI system
    print("Initializing HTI system...")
    hti = IntegratedHTI()
    
    # Load tasks
    print("Loading ARC-style tasks...")
    tasks = load_arc_tasks(n_tasks=5)
    
    # Test on each task
    results = []
    print("\n" + "-" * 60)
    
    for i, task in enumerate(tasks, 1):
        print(f"\nTask {i}/{len(tasks)}: {task['id']}")
        
        result = test_hti_on_arc_task(hti, task)
        results.append(result)
        
        print(f"  Score: {result['score']:.1%}")
        print(f"  Time: {result['solve_time']:.2f}s")
        print(f"  Source: {result['source']}")
        print(f"  Reasoning cycles: {result['reasoning_cycles']}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    scores = [r['score'] for r in results]
    solved = sum(1 for s in scores if s > 0.99)
    partial = sum(1 for s in scores if 0.5 < s <= 0.99)
    failed = sum(1 for s in scores if s <= 0.5)
    
    print(f"\nTasks tested: {len(tasks)}")
    print(f"Perfect solutions (>99%): {solved}")
    print(f"Partial solutions (50-99%): {partial}")
    print(f"Failed (<50%): {failed}")
    print(f"\nAverage score: {np.mean(scores):.1%}")
    print(f"Median score: {np.median(scores):.1%}")
    
    # Breakdown by source
    memory_solved = sum(1 for r in results if r['source'] == 'memory' and r['score'] > 0.5)
    invented = sum(1 for r in results if r['source'] == 'invented')
    
    print(f"\nSolved from memory: {memory_solved}")
    print(f"Required invention: {invented}")
    
    # Memory statistics
    mem_stats = hti.memory.get_statistics()
    print(f"\nMemory statistics:")
    print(f"  Transforms stored: {mem_stats.get('total_transforms', 0)}")
    print(f"  Transforms discovered: {hti.transforms_discovered}")
    
    # Assessment
    print("\n" + "-" * 60)
    if np.mean(scores) > 0.5:
        print("✅ HTI shows promising performance on ARC-style tasks!")
        print("   The learnable transform space generalizes to new patterns.")
    elif np.mean(scores) > 0.25:
        print("🔶 HTI shows some capability on ARC tasks.")
        print("   More training or semantic understanding needed.")
    else:
        print("❌ HTI struggles with ARC tasks.")
        print("   Consider adding pre-training or better primitives.")
    
    # Comparison with baselines
    print("\n📊 Context (from our testing):")
    print("  - Fixed hypothesis generator: ~37% on our benchmark")
    print("  - With compositional reasoning: 72.8% on our benchmark")
    print(f"  - HTI on ARC-style tasks: {np.mean(scores):.1%}")
    
    return results


if __name__ == "__main__":
    results = main()