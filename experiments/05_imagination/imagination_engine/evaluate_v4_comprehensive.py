"""
Comprehensive evaluation of Imagination Engine V4 with all improvements.

Tests:
1. Performance on ARC tasks
2. Learning improvement over rounds
3. Strategy effectiveness
4. Failure analysis
"""

import numpy as np
from pathlib import Path
import time
import json
from typing import Dict, List, Any, Tuple
from tqdm import tqdm
import pickle
from collections import defaultdict

from imagination_engine_v4 import ImaginationEngineV4
from arc_data_loader import load_arc_training_data, prepare_task_for_hti
from meta_learner import MetaLearner


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


def analyze_failure_modes(engine: ImaginationEngineV4) -> Dict[str, Any]:
    """Analyze failure modes from meta-learner."""
    analysis = {
        'common_errors': defaultdict(int),
        'missing_capabilities': defaultdict(int),
        'strategy_failures': defaultdict(int)
    }
    
    # Get failure analysis from meta-learner
    if hasattr(engine, 'meta_learner') and hasattr(engine.meta_learner, 'failure_history'):
        for failure in engine.meta_learner.failure_history:
            # Count error types
            if 'error_type' in failure:
                analysis['common_errors'][failure['error_type']] += 1
            
            # Count missing capabilities
            if 'missing_capability' in failure:
                analysis['missing_capabilities'][failure['missing_capability']] += 1
            
            # Count strategy failures
            if 'failed_strategies' in failure:
                for strategy in failure['failed_strategies']:
                    analysis['strategy_failures'][strategy] += 1
    
    return analysis


def run_comprehensive_evaluation(
    max_tasks: int = 50,
    learning_rounds: int = 3,
    timeout_per_task: float = 30.0,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run comprehensive evaluation of V4 engine.
    
    Args:
        max_tasks: Maximum number of tasks to evaluate
        learning_rounds: Number of rounds to test learning
        timeout_per_task: Maximum time per task in seconds
        verbose: Whether to print detailed progress
        
    Returns:
        Dictionary with comprehensive evaluation results
    """
    
    print("=" * 70)
    print("IMAGINATION ENGINE V4 - COMPREHENSIVE EVALUATION")
    print("=" * 70)
    print(f"Tasks: {max_tasks} | Rounds: {learning_rounds} | Timeout: {timeout_per_task}s")
    print("-" * 70)
    
    # Initialize engine
    engine = ImaginationEngineV4(verbose=False)
    
    # Load ARC tasks
    print("\nLoading ARC-AGI-2 training data...")
    tasks = load_arc_training_data(max_tasks=max_tasks)
    print(f"Loaded {len(tasks)} tasks")
    
    # Results tracking
    results = {
        'rounds': [],
        'overall_stats': {},
        'strategy_effectiveness': defaultdict(lambda: {'attempts': 0, 'successes': 0}),
        'learning_metrics': {},
        'failure_analysis': {},
        'task_categories': defaultdict(list)
    }
    
    # Run multiple rounds to test learning
    for round_num in range(learning_rounds):
        print(f"\n{'='*30} ROUND {round_num + 1} {'='*30}")
        
        round_results = {
            'round': round_num + 1,
            'tasks_solved': 0,
            'total_accuracy': 0.0,
            'strategy_usage': defaultdict(int),
            'time_taken': 0.0,
            'task_results': []
        }
        
        # Evaluate each task
        progress_bar = tqdm(enumerate(tasks), total=len(tasks), desc=f"Round {round_num + 1}")
        for task_idx, task_data in progress_bar:
            # Generate task ID
            task_id = f"task_{task_idx:03d}"
            
            # Prepare task
            train_examples, test_examples = prepare_task_for_hti(task_data)
            
            if not train_examples or not test_examples:
                continue
            
            # Time the solution
            start_time = time.time()
            
            try:
                # Solve with timeout
                # Convert to format expected by engine
                train_data = [{'input': inp.tolist(), 'output': out.tolist()} 
                             for inp, out in train_examples]
                test_data = [{'input': inp.tolist()} for inp, _ in test_examples]
                
                task_dict = {
                    'id': task_id,
                    'train': train_data, 
                    'test': test_data
                }
                solution = engine.solve(task_dict, timeout=timeout_per_task)
                
                elapsed = time.time() - start_time
                
                if verbose:
                    print(f"\nTask {task_id}: Solution found = {solution is not None}")
                
                # Evaluate solution
                if solution and hasattr(solution, 'predictions'):
                    accuracy = evaluate_solution(solution.predictions, test_examples)
                    
                    # Track results
                    task_result = {
                        'task_id': task_id,
                        'accuracy': accuracy,
                        'solved': accuracy > 0.5,
                        'strategy_used': getattr(solution, 'strategy_used', 'unknown'),
                        'time': elapsed
                    }
                    
                    round_results['task_results'].append(task_result)
                    
                    if accuracy > 0.5:
                        round_results['tasks_solved'] += 1
                    
                    round_results['total_accuracy'] += accuracy
                    
                    # Track strategy usage
                    strategy = getattr(solution, 'strategy_used', 'unknown')
                    round_results['strategy_usage'][strategy] += 1
                    results['strategy_effectiveness'][strategy]['attempts'] += 1
                    if accuracy > 0.5:
                        results['strategy_effectiveness'][strategy]['successes'] += 1
                    
                    # Categorize task
                    if accuracy == 1.0:
                        results['task_categories']['perfect'].append(task_id)
                    elif accuracy > 0.5:
                        results['task_categories']['partial'].append(task_id)
                    else:
                        results['task_categories']['failed'].append(task_id)
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'solved': round_results['tasks_solved'],
                        'accuracy': f"{accuracy:.1%}"
                    })
                    
                else:
                    # No solution found
                    task_result = {
                        'task_id': task_id,
                        'accuracy': 0.0,
                        'solved': False,
                        'strategy_used': 'none',
                        'time': elapsed
                    }
                    round_results['task_results'].append(task_result)
                    results['task_categories']['failed'].append(task_id)
                    
            except Exception as e:
                if verbose:
                    print(f"\nError on task {task_id}: {str(e)}")
                task_result = {
                    'task_id': task_id,
                    'accuracy': 0.0,
                    'solved': False,
                    'strategy_used': 'error',
                    'time': 0.0,
                    'error': str(e)
                }
                round_results['task_results'].append(task_result)
                results['task_categories']['error'].append(task_id)
        
        # Calculate round statistics
        n_tasks = len(round_results['task_results'])
        if n_tasks > 0:
            round_results['solve_rate'] = round_results['tasks_solved'] / n_tasks
            round_results['avg_accuracy'] = round_results['total_accuracy'] / n_tasks
        else:
            round_results['solve_rate'] = 0.0
            round_results['avg_accuracy'] = 0.0
        
        results['rounds'].append(round_results)
        
        # Print round summary
        print(f"\nRound {round_num + 1} Summary:")
        print(f"  Tasks Solved: {round_results['tasks_solved']}/{n_tasks} ({round_results['solve_rate']:.1%})")
        print(f"  Average Accuracy: {round_results['avg_accuracy']:.1%}")
        print(f"  Strategy Usage: {dict(round_results['strategy_usage'])}")
    
    # Analyze learning improvement
    print("\n" + "=" * 70)
    print("LEARNING ANALYSIS")
    print("-" * 70)
    
    if len(results['rounds']) >= 2:
        first_round = results['rounds'][0]
        last_round = results['rounds'][-1]
        
        results['learning_metrics'] = {
            'solve_rate_improvement': last_round['solve_rate'] - first_round['solve_rate'],
            'accuracy_improvement': last_round['avg_accuracy'] - first_round['avg_accuracy'],
            'first_round_solve_rate': first_round['solve_rate'],
            'last_round_solve_rate': last_round['solve_rate'],
            'first_round_accuracy': first_round['avg_accuracy'],
            'last_round_accuracy': last_round['avg_accuracy']
        }
        
        print(f"Solve Rate: {first_round['solve_rate']:.1%} → {last_round['solve_rate']:.1%} "
              f"({'↑' if results['learning_metrics']['solve_rate_improvement'] > 0 else '↓'} "
              f"{abs(results['learning_metrics']['solve_rate_improvement']):.1%})")
        print(f"Accuracy: {first_round['avg_accuracy']:.1%} → {last_round['avg_accuracy']:.1%} "
              f"({'↑' if results['learning_metrics']['accuracy_improvement'] > 0 else '↓'} "
              f"{abs(results['learning_metrics']['accuracy_improvement']):.1%})")
    
    # Analyze strategy effectiveness
    print("\n" + "=" * 70)
    print("STRATEGY EFFECTIVENESS")
    print("-" * 70)
    
    for strategy, stats in sorted(results['strategy_effectiveness'].items(), 
                                 key=lambda x: x[1]['attempts'], reverse=True):
        if stats['attempts'] > 0:
            success_rate = stats['successes'] / stats['attempts']
            print(f"{strategy:30s}: {stats['successes']:3d}/{stats['attempts']:3d} ({success_rate:.1%})")
    
    # Analyze failures
    print("\n" + "=" * 70)
    print("FAILURE ANALYSIS")
    print("-" * 70)
    
    results['failure_analysis'] = analyze_failure_modes(engine)
    
    if results['failure_analysis']['common_errors']:
        print("\nCommon Errors:")
        for error, count in sorted(results['failure_analysis']['common_errors'].items(), 
                                  key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {error}: {count}")
    
    if results['failure_analysis']['missing_capabilities']:
        print("\nMissing Capabilities:")
        for capability, count in sorted(results['failure_analysis']['missing_capabilities'].items(),
                                       key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {capability}: {count}")
    
    # Calculate overall statistics
    all_results = []
    for round_res in results['rounds']:
        all_results.extend(round_res['task_results'])
    
    results['overall_stats'] = {
        'total_tasks': len(all_results),
        'unique_tasks': len(set(r['task_id'] for r in all_results)),
        'total_solved': sum(1 for r in all_results if r['solved']),
        'overall_solve_rate': sum(1 for r in all_results if r['solved']) / len(all_results) if all_results else 0,
        'overall_accuracy': sum(r['accuracy'] for r in all_results) / len(all_results) if all_results else 0,
        'perfect_solutions': len(set(results['task_categories']['perfect'])),
        'partial_solutions': len(set(results['task_categories']['partial'])),
        'failed_tasks': len(set(results['task_categories']['failed'])),
        'error_tasks': len(set(results['task_categories']['error']))
    }
    
    # Print overall summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("-" * 70)
    print(f"Total Evaluations: {results['overall_stats']['total_tasks']}")
    print(f"Unique Tasks: {results['overall_stats']['unique_tasks']}")
    print(f"Overall Solve Rate: {results['overall_stats']['overall_solve_rate']:.1%}")
    print(f"Overall Accuracy: {results['overall_stats']['overall_accuracy']:.1%}")
    print(f"\nTask Categories:")
    print(f"  Perfect Solutions: {results['overall_stats']['perfect_solutions']}")
    print(f"  Partial Solutions: {results['overall_stats']['partial_solutions']}")
    print(f"  Failed Tasks: {results['overall_stats']['failed_tasks']}")
    print(f"  Error Tasks: {results['overall_stats']['error_tasks']}")
    
    # Save results
    results_file = f"evaluation_results_v4_{time.strftime('%Y%m%d_%H%M%S')}.json"
    
    # Convert defaultdicts to regular dicts for JSON serialization
    results_to_save = {
        'rounds': results['rounds'],
        'overall_stats': results['overall_stats'],
        'strategy_effectiveness': dict(results['strategy_effectiveness']),
        'learning_metrics': results['learning_metrics'],
        'failure_analysis': {
            'common_errors': dict(results['failure_analysis']['common_errors']),
            'missing_capabilities': dict(results['failure_analysis']['missing_capabilities']),
            'strategy_failures': dict(results['failure_analysis']['strategy_failures'])
        },
        'task_categories': dict(results['task_categories'])
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    print(f"\n✅ Results saved to {results_file}")
    
    # Get learning summary from meta-learner
    if hasattr(engine, 'meta_learner'):
        print("\n" + "=" * 70)
        print("META-LEARNING SUMMARY")
        print("-" * 70)
        summary = engine.meta_learner.get_learning_summary()
        print(f"Strategies Learned: {summary.get('strategies_learned', 0)}")
        print(f"Total Experiences: {summary.get('total_experiences', 0)}")
        print(f"Success Patterns: {summary.get('success_patterns', 0)}")
        print(f"Failure Patterns: {summary.get('failure_patterns', 0)}")
        
        if summary.get('top_strategies'):
            print("\nTop Performing Strategies:")
            for strategy, score in summary['top_strategies'][:5]:
                print(f"  {strategy}: {score:.2f}")
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    
    return results


def main():
    """Run the comprehensive evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Imagination Engine V4')
    parser.add_argument('--max-tasks', type=int, default=30,
                       help='Maximum number of tasks to evaluate')
    parser.add_argument('--rounds', type=int, default=3,
                       help='Number of learning rounds')
    parser.add_argument('--timeout', type=float, default=30.0,
                       help='Timeout per task in seconds')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed progress')
    
    args = parser.parse_args()
    
    results = run_comprehensive_evaluation(
        max_tasks=args.max_tasks,
        learning_rounds=args.rounds,
        timeout_per_task=args.timeout,
        verbose=args.verbose
    )
    
    return results


if __name__ == "__main__":
    main()