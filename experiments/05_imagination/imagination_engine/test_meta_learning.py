"""Test the meta-learning capabilities of Imagination Engine V4."""

import numpy as np
from pathlib import Path
import json
import time
from typing import Dict, List

from imagination_engine_v4 import ImaginationEngineV4
from arc_data_loader import load_arc_training_data, prepare_task_for_hti


def run_meta_learning_experiment(
    num_tasks: int = 50,
    learning_rounds: int = 3
) -> Dict:
    """Run experiment to test if meta-learning improves performance over time.
    
    Args:
        num_tasks: Number of tasks to use
        learning_rounds: Number of times to process the same tasks
        
    Returns:
        Dictionary with experiment results
    """
    
    print("=" * 70)
    print("META-LEARNING EXPERIMENT")
    print("=" * 70)
    print(f"Testing with {num_tasks} tasks over {learning_rounds} rounds")
    print("Hypothesis: Performance should improve in later rounds due to learning")
    print()
    
    # Load tasks
    try:
        all_tasks = load_arc_training_data(max_tasks=num_tasks)
    except:
        # Use test dataset if no real ARC data
        print("Using test dataset...")
        from create_test_arc_dataset import create_test_dataset
        test_dir = create_test_dataset()
        all_tasks = []
        for file in test_dir.glob("*.json"):
            with open(file) as f:
                task = json.load(f)
                task['id'] = file.stem
                all_tasks.append(task)
    
    # Initialize engine with fresh memory
    engine = ImaginationEngineV4(
        memory_path=Path("meta_test_memory.json"),
        meta_learning_path=Path("meta_test_learning.json"),
        enable_meta_learning=True,
        verbose=False
    )
    
    results = {
        'rounds': [],
        'overall_improvement': 0.0,
        'strategy_learning': {},
        'meta_patterns_discovered': 0
    }
    
    # Run multiple rounds
    for round_num in range(learning_rounds):
        print(f"\n{'='*50}")
        print(f"ROUND {round_num + 1}/{learning_rounds}")
        print(f"{'='*50}")
        
        round_start = time.time()
        round_results = {
            'round': round_num + 1,
            'successes': 0,
            'failures': 0,
            'accuracy_sum': 0.0,
            'meta_learning_used': 0,
            'strategies_used': {},
            'time_taken': 0.0
        }
        
        # Process all tasks
        for i, task in enumerate(all_tasks):
            if i % 10 == 0:
                print(f"  Processing task {i+1}/{len(all_tasks)}...")
            
            # Prepare task
            engine_task = {
                'id': task.get('id', f'task_{i}'),
                'train': task.get('train', []),
                'test': task.get('test', [])
            }
            
            # Solve
            solution = engine.solve(engine_task, timeout=5.0)
            
            # Track results
            if solution.accuracy >= 0.8:
                round_results['successes'] += 1
            else:
                round_results['failures'] += 1
            
            round_results['accuracy_sum'] += solution.accuracy
            
            if solution.meta_learning_applied:
                round_results['meta_learning_used'] += 1
            
            strategy = solution.strategy_used
            round_results['strategies_used'][strategy] = round_results['strategies_used'].get(strategy, 0) + 1
        
        round_results['time_taken'] = time.time() - round_start
        round_results['avg_accuracy'] = round_results['accuracy_sum'] / len(all_tasks)
        round_results['success_rate'] = round_results['successes'] / len(all_tasks)
        
        results['rounds'].append(round_results)
        
        # Print round summary
        print(f"\nRound {round_num + 1} Results:")
        print(f"  Success rate: {round_results['success_rate']:.1%} ({round_results['successes']}/{len(all_tasks)})")
        print(f"  Average accuracy: {round_results['avg_accuracy']:.1%}")
        print(f"  Meta-learning used: {round_results['meta_learning_used']} times")
        print(f"  Time: {round_results['time_taken']:.1f}s")
        
        # Get meta-learning insights
        meta_summary = engine.meta_learner.get_learning_summary()
        print(f"\n  Meta-Learning Status:")
        print(f"    Strategies learned: {meta_summary['strategies_learned']}")
        print(f"    Overall success rate: {meta_summary['overall_success_rate']:.1%}")
        print(f"    Meta-patterns discovered: {meta_summary['meta_patterns_discovered']}")
    
    # Calculate improvement
    if len(results['rounds']) >= 2:
        first_round = results['rounds'][0]
        last_round = results['rounds'][-1]
        
        results['overall_improvement'] = (
            last_round['success_rate'] - first_round['success_rate']
        )
        
        print("\n" + "=" * 70)
        print("EXPERIMENT SUMMARY")
        print("=" * 70)
        
        print(f"\nPerformance over rounds:")
        for r in results['rounds']:
            print(f"  Round {r['round']}: {r['success_rate']:.1%} success rate")
        
        print(f"\nOverall improvement: {results['overall_improvement']:.1%}")
        
        if results['overall_improvement'] > 0:
            print("✓ Meta-learning is working! Performance improved over time.")
        else:
            print("⚠ No improvement observed. May need more tasks or rounds.")
        
        # Analyze strategy learning
        print("\nStrategy evolution:")
        for round_result in results['rounds']:
            top_strategy = max(round_result['strategies_used'].items(), 
                             key=lambda x: x[1])[0] if round_result['strategies_used'] else "none"
            print(f"  Round {round_result['round']}: Most used = {top_strategy}")
    
    # Save engine state
    engine.save_all()
    
    return results


def test_specific_learning():
    """Test if the system learns from specific failure patterns."""
    
    print("\n" + "=" * 70)
    print("SPECIFIC LEARNING TEST")
    print("=" * 70)
    print("Testing if system learns from specific failure patterns...")
    
    # Create tasks with known patterns
    tasks = []
    
    # Pattern 1: Simple rotation (should learn geometric_reasoning works)
    for i in range(5):
        task = {
            'id': f'rotation_{i}',
            'train': [
                {'input': [[1, 2], [3, 4]], 'output': [[3, 1], [4, 2]]},
                {'input': [[5, 6], [7, 8]], 'output': [[7, 5], [8, 6]]}
            ],
            'test': [
                {'input': [[9, 10], [11, 12]]}
            ]
        }
        tasks.append(task)
    
    # Pattern 2: Value increment (should learn trace works)
    for i in range(5):
        task = {
            'id': f'increment_{i}',
            'train': [
                {'input': [[1, 2], [3, 4]], 'output': [[2, 3], [4, 5]]},
                {'input': [[5, 6], [7, 8]], 'output': [[6, 7], [8, 9]]}
            ],
            'test': [
                {'input': [[10, 11], [12, 13]]}
            ]
        }
        tasks.append(task)
    
    # Initialize engine
    engine = ImaginationEngineV4(
        memory_path=Path("specific_test_memory.json"),
        meta_learning_path=Path("specific_test_learning.json"),
        enable_meta_learning=True,
        verbose=False
    )
    
    print("\nRound 1: Initial learning...")
    round1_correct = 0
    
    for task in tasks:
        solution = engine.solve(task, timeout=5.0)
        if solution.accuracy >= 0.8:
            round1_correct += 1
    
    print(f"  Correct: {round1_correct}/{len(tasks)}")
    
    print("\nRound 2: After learning...")
    round2_correct = 0
    round2_meta = 0
    
    for task in tasks:
        # Change IDs to simulate new tasks
        task['id'] = task['id'] + '_v2'
        solution = engine.solve(task, timeout=5.0)
        if solution.accuracy >= 0.8:
            round2_correct += 1
        if solution.meta_learning_applied:
            round2_meta += 1
    
    print(f"  Correct: {round2_correct}/{len(tasks)}")
    print(f"  Meta-learning used: {round2_meta} times")
    
    # Check what was learned
    meta_summary = engine.meta_learner.get_learning_summary()
    
    print("\nLearning Summary:")
    print(f"  Total tasks seen: {meta_summary['total_tasks_seen']}")
    
    if meta_summary['strategy_performance']:
        print("\n  Strategy Performance:")
        for strategy, perf in meta_summary['strategy_performance'].items():
            if perf['attempts'] > 0:
                print(f"    {strategy}: {perf['success_rate']:.1%} success ({perf['attempts']} attempts)")
    
    if meta_summary['top_errors']:
        print("\n  Top Errors Learned:")
        for error, count in meta_summary['top_errors'].items():
            print(f"    {error}: {count} occurrences")
    
    # Save state
    engine.save_all()
    
    return round2_correct > round1_correct


def analyze_meta_patterns():
    """Analyze what meta-patterns the system discovers."""
    
    print("\n" + "=" * 70)
    print("META-PATTERN ANALYSIS")
    print("=" * 70)
    
    # Load existing meta-learning knowledge if available
    engine = ImaginationEngineV4(
        meta_learning_path=Path("meta_test_learning.json"),
        enable_meta_learning=True,
        verbose=False
    )
    
    # Extract meta-patterns
    patterns = engine.meta_learner.extract_meta_patterns()
    
    if patterns:
        print(f"Discovered {len(patterns)} meta-patterns:")
        
        for i, pattern in enumerate(patterns[:5]):  # Show first 5
            print(f"\n  Pattern {i+1}:")
            print(f"    Type: {pattern.get('type')}")
            
            if pattern['type'] == 'strategy_affinity':
                print(f"    Strategy: {pattern.get('strategy')}")
                print(f"    Success rate: {pattern.get('success_rate', 0):.1%}")
                print(f"    Confidence: {pattern.get('confidence', 0):.1%}")
            
            elif pattern['type'] == 'task_cluster':
                print(f"    Strategy: {pattern.get('strategy')}")
                print(f"    Tasks in cluster: {pattern.get('num_tasks', 0)}")
    else:
        print("No meta-patterns discovered yet. Run more experiments to build knowledge.")


if __name__ == "__main__":
    # Run main experiment
    print("TESTING META-LEARNING CAPABILITIES")
    print("=" * 70)
    
    # Test 1: Does performance improve over rounds?
    results = run_meta_learning_experiment(num_tasks=20, learning_rounds=3)
    
    # Test 2: Does it learn from specific patterns?
    print("\n")
    learned = test_specific_learning()
    
    if learned:
        print("\n✓ System successfully learned from specific patterns!")
    else:
        print("\n⚠ System did not show clear learning on specific patterns")
    
    # Test 3: Analyze discovered patterns
    analyze_meta_patterns()
    
    print("\n" + "=" * 70)
    print("META-LEARNING TESTS COMPLETE")
    print("=" * 70)
    
    # Clean up test files
    for f in ["meta_test_memory.json", "meta_test_memory.pkl", 
              "meta_test_learning.json", "specific_test_memory.json",
              "specific_test_memory.pkl", "specific_test_learning.json"]:
        Path(f).unlink(missing_ok=True)