"""Comprehensive testing on the complete ARC training dataset.

This script tests our enhanced ARC system on all 1000 training tasks to:
1. Measure true performance across diverse task types
2. Detect potential overfitting
3. Understand which primitives are most valuable
4. Track learning and transfer effects
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
import time
from datetime import datetime
import multiprocessing as mp
from functools import partial
import traceback

from arc_data_loader import load_arc_training_data
from program_synthesis_v2 import EnhancedProgramSynthesizer, Transform
from hypothesis_generator import MinimalHypothesisGenerator, GenerationStrategy
from compound_primitive_learner import CompoundPrimitiveLearner
from arc_primitives_extended import ARCPrimitivesExtended


class ComprehensiveARCEngine:
    """Enhanced ARC engine with compound learning and comprehensive tracking."""
    
    def __init__(self, use_compounds: bool = True):
        self.synthesizer = EnhancedProgramSynthesizer()
        self.hypothesis_gen = MinimalHypothesisGenerator()
        self.use_compounds = use_compounds
        
        if use_compounds:
            self.compound_learner = CompoundPrimitiveLearner("arc_learned_compounds.json")
        else:
            self.compound_learner = None
        
        self.solve_times = []
        self.primitives_used = {}
        
    def solve(self, 
             examples: List[Tuple[np.ndarray, np.ndarray]], 
             task_id: str = None,
             max_time: float = 10.0) -> Optional[Transform]:
        """Try to solve an ARC task with comprehensive tracking."""
        
        start_time = time.time()
        
        # Extract task features for compound matching
        task_features = self._extract_task_features(examples)
        
        # Try relevant compound primitives first
        if self.compound_learner:
            relevant_compounds = self.compound_learner.get_relevant_compounds(task_features)
            for compound in relevant_compounds:
                primitive = compound.to_primitive()
                score = self._evaluate_transform(primitive, examples)
                if score > 0.95:
                    elapsed = time.time() - start_time
                    self.solve_times.append(elapsed)
                    self._track_primitive_use(compound.name)
                    return primitive
        
        # Try program synthesis with beam search
        solution = self.synthesizer.synthesize(examples, max_time * 0.6)
        
        if solution:
            # Learn from successful solution
            if self.compound_learner:
                test_score = self._evaluate_transform(solution, examples)
                if test_score > 0.95:
                    self.compound_learner.learn_from_solution(solution, task_features, test_score)
            
            elapsed = time.time() - start_time
            self.solve_times.append(elapsed)
            self._track_primitive_use(solution.to_string())
            return solution
        
        # Try hypothesis generation if time permits
        if time.time() - start_time < max_time:
            for strategy in [GenerationStrategy.SYSTEMATIC, 
                            GenerationStrategy.CONSTRAINT_RELAXATION]:
                if time.time() - start_time >= max_time:
                    break
                    
                hypotheses = self.hypothesis_gen.generate_hypotheses(
                    examples, n_hypotheses=3, strategy=strategy
                )
                
                for hypothesis in hypotheses:
                    score = self._evaluate_hypothesis(hypothesis.transform_fn, examples)
                    if score > 0.8:
                        from program_synthesis_v2 import Primitive
                        result = Primitive(f"hypothesis_{strategy.value}", hypothesis.transform_fn)
                        
                        elapsed = time.time() - start_time
                        self.solve_times.append(elapsed)
                        self._track_primitive_use(result.name)
                        return result
        
        return None
    
    def _extract_task_features(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> Dict:
        """Extract features from task examples."""
        if not examples:
            return {}
        
        return {
            'input_shape': examples[0][0].shape,
            'output_shape': examples[0][1].shape,
            'input_colors': tuple(sorted(np.unique(examples[0][0]))),
            'output_colors': tuple(sorted(np.unique(examples[0][1]))),
            'num_examples': len(examples)
        }
    
    def _evaluate_transform(self, transform: Transform, examples: List[Tuple]) -> float:
        """Evaluate a transform on examples."""
        if not examples:
            return 0.0
        
        total_score = 0.0
        for input_grid, expected_output in examples:
            try:
                predicted = transform.apply(input_grid)
                if predicted.shape == expected_output.shape:
                    accuracy = np.mean(predicted == expected_output)
                    total_score += accuracy
            except:
                continue
        
        return total_score / len(examples)
    
    def _evaluate_hypothesis(self, transform, examples):
        """Evaluate a hypothesis transform."""
        if not examples:
            return 0.0
        
        total = 0
        for inp, out in examples:
            try:
                pred = transform(inp)
                if pred.shape == out.shape:
                    total += np.mean(pred == out)
            except:
                pass
        
        return total / len(examples)
    
    def _track_primitive_use(self, primitive_name: str):
        """Track which primitives are being used."""
        self.primitives_used[primitive_name] = self.primitives_used.get(primitive_name, 0) + 1


def process_single_task(task_data: Dict, 
                       engine: ComprehensiveARCEngine,
                       timeout: float = 10.0) -> Dict:
    """Process a single ARC task and return results."""
    
    task_id = task_data.get('id', 'unknown')
    
    try:
        # Convert to numpy arrays
        train_examples = [
            (np.array(ex['input']), np.array(ex['output']))
            for ex in task_data['train']
        ]
        
        # Categorize task
        task_type = categorize_task(train_examples)
        
        # Try to solve
        start_time = time.time()
        solution = engine.solve(train_examples, task_id=task_id, max_time=timeout)
        solve_time = time.time() - start_time
        
        # Evaluate solution
        if solution and task_data.get('test'):
            test_input = np.array(task_data['test'][0]['input'])
            test_output = np.array(task_data['test'][0]['output']) if 'output' in task_data['test'][0] else None
            
            if test_output is not None:
                try:
                    predicted = solution.apply(test_input)
                    
                    if predicted.shape == test_output.shape:
                        accuracy = float(np.mean(predicted == test_output))
                        success = accuracy > 0.9
                    else:
                        accuracy = 0.0
                        success = False
                        
                except Exception as e:
                    accuracy = 0.0
                    success = False
            else:
                accuracy = -1.0  # No test output available
                success = False
        else:
            accuracy = 0.0
            success = False
        
        return {
            'task_id': task_id,
            'task_type': task_type,
            'solved': solution is not None,
            'accuracy': accuracy,
            'success': success,
            'solve_time': solve_time,
            'solution_type': solution.to_string() if solution else None
        }
        
    except Exception as e:
        return {
            'task_id': task_id,
            'task_type': 'error',
            'solved': False,
            'accuracy': 0.0,
            'success': False,
            'solve_time': 0.0,
            'error': str(e)
        }


def categorize_task(examples: List[Tuple[np.ndarray, np.ndarray]]) -> str:
    """Categorize a task based on its characteristics."""
    
    if not examples:
        return 'other'
    
    inp, out = examples[0]
    
    # Check for resize
    if inp.shape != out.shape:
        return 'resize'
    
    # Check for new colors
    inp_colors = set(np.unique(inp))
    out_colors = set(np.unique(out))
    if out_colors - inp_colors:
        return 'color_mapping'
    
    # Check for duplication
    inp_nonzero = np.count_nonzero(inp)
    out_nonzero = np.count_nonzero(out)
    if out_nonzero > inp_nonzero * 1.5:
        return 'object_duplication'
    
    # Check for simple transformations
    if np.array_equal(np.rot90(inp), out) or np.array_equal(np.flip(inp, 0), out):
        return 'pattern'
    
    return 'other'


def test_full_dataset(num_tasks: Optional[int] = None,
                     checkpoint_interval: int = 100,
                     timeout_per_task: float = 10.0,
                     use_compounds: bool = True,
                     parallel: bool = False):
    """Test on the full ARC dataset with checkpointing."""
    
    print("=" * 70)
    print("COMPREHENSIVE ARC DATASET TESTING")
    print("=" * 70)
    print(f"Start time: {datetime.now()}")
    
    # Load all tasks
    print(f"\nLoading ARC training tasks...")
    all_tasks = load_arc_training_data(max_tasks=num_tasks)
    print(f"Loaded {len(all_tasks)} tasks")
    
    # Split for overfitting detection
    split_point = len(all_tasks) // 2
    first_half = all_tasks[:split_point]
    second_half = all_tasks[split_point:]
    
    # Initialize engine
    engine = ComprehensiveARCEngine(use_compounds=use_compounds)
    
    # Results storage
    results = []
    checkpoint_path = Path("results/checkpoints")
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # Process tasks
    print(f"\nTesting with timeout={timeout_per_task}s per task...")
    print(f"Using compounds: {use_compounds}")
    print(f"Parallel processing: {parallel}")
    print("-" * 70)
    
    start_time = time.time()
    
    if parallel and False:  # Disabled for now due to serialization issues
        # Parallel processing
        with mp.Pool(processes=mp.cpu_count() - 1) as pool:
            process_func = partial(process_single_task, 
                                  engine=engine, 
                                  timeout=timeout_per_task)
            results = pool.map(process_func, all_tasks)
    else:
        # Sequential processing with progress updates
        for i, task_data in enumerate(all_tasks):
            # Process task
            result = process_single_task(task_data, engine, timeout_per_task)
            results.append(result)
            
            # Progress update
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (i + 1)
                remaining = avg_time * (len(all_tasks) - i - 1)
                
                solved = sum(1 for r in results if r['success'])
                print(f"Progress: {i+1}/{len(all_tasks)} | "
                      f"Solved: {solved}/{i+1} ({solved/(i+1)*100:.1f}%) | "
                      f"ETA: {remaining/60:.1f} min")
            
            # Checkpoint
            if (i + 1) % checkpoint_interval == 0:
                checkpoint_file = checkpoint_path / f"checkpoint_{i+1}.json"
                with open(checkpoint_file, 'w') as f:
                    json.dump({
                        'num_processed': i + 1,
                        'results': results,
                        'timestamp': datetime.now().isoformat()
                    }, f, indent=2)
                print(f"  Checkpoint saved: {checkpoint_file}")
    
    # Calculate final statistics
    total_time = time.time() - start_time
    
    # Overall metrics
    num_solved = sum(1 for r in results if r['success'])
    num_partial = sum(1 for r in results if r['accuracy'] > 0.5 and not r['success'])
    avg_accuracy = np.mean([r['accuracy'] for r in results if r['accuracy'] >= 0])
    
    # Performance by task type
    task_types = {}
    for result in results:
        task_type = result['task_type']
        if task_type not in task_types:
            task_types[task_type] = {'total': 0, 'solved': 0, 'accuracies': []}
        
        task_types[task_type]['total'] += 1
        if result['success']:
            task_types[task_type]['solved'] += 1
        if result['accuracy'] >= 0:
            task_types[task_type]['accuracies'].append(result['accuracy'])
    
    # Overfitting detection
    first_half_results = results[:split_point]
    second_half_results = results[split_point:]
    
    first_half_solved = sum(1 for r in first_half_results if r['success'])
    second_half_solved = sum(1 for r in second_half_results if r['success'])
    
    # Primitive usage statistics
    if hasattr(engine, 'primitives_used'):
        top_primitives = sorted(engine.primitives_used.items(), 
                              key=lambda x: -x[1])[:20]
    else:
        top_primitives = []
    
    # Generate report
    report = {
        'metadata': {
            'total_tasks': len(all_tasks),
            'total_time_seconds': total_time,
            'avg_time_per_task': total_time / len(all_tasks),
            'use_compounds': use_compounds,
            'timestamp': datetime.now().isoformat()
        },
        'overall_performance': {
            'tasks_solved': num_solved,
            'solve_rate': num_solved / len(all_tasks),
            'tasks_partial': num_partial,
            'average_accuracy': avg_accuracy,
            'perfect_solutions': sum(1 for r in results if r['accuracy'] == 1.0)
        },
        'performance_by_type': {
            task_type: {
                'total': stats['total'],
                'solved': stats['solved'],
                'solve_rate': stats['solved'] / stats['total'] if stats['total'] > 0 else 0,
                'avg_accuracy': np.mean(stats['accuracies']) if stats['accuracies'] else 0
            }
            for task_type, stats in task_types.items()
        },
        'overfitting_analysis': {
            'first_half_solved': first_half_solved,
            'first_half_rate': first_half_solved / len(first_half_results) if first_half_results else 0,
            'second_half_solved': second_half_solved,
            'second_half_rate': second_half_solved / len(second_half_results) if second_half_results else 0,
            'performance_change': (second_half_solved / len(second_half_results) - 
                                 first_half_solved / len(first_half_results))
                                if first_half_results and second_half_results else 0
        },
        'top_primitives': top_primitives,
        'detailed_results': results
    }
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"full_test_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TESTING COMPLETE")
    print("=" * 70)
    print(f"\nOverall Performance:")
    print(f"  Tasks solved: {num_solved}/{len(all_tasks)} ({num_solved/len(all_tasks)*100:.1f}%)")
    print(f"  Average accuracy: {avg_accuracy:.1%}")
    print(f"  Perfect solutions: {report['overall_performance']['perfect_solutions']}")
    print(f"  Total time: {total_time/60:.1f} minutes")
    
    print(f"\nPerformance by Task Type:")
    for task_type, stats in report['performance_by_type'].items():
        print(f"  {task_type:20s}: {stats['solved']}/{stats['total']} "
              f"({stats['solve_rate']*100:.1f}%), "
              f"avg accuracy: {stats['avg_accuracy']:.1%}")
    
    print(f"\nOverfitting Analysis:")
    print(f"  First half:  {first_half_solved}/{len(first_half_results)} "
          f"({report['overfitting_analysis']['first_half_rate']*100:.1f}%)")
    print(f"  Second half: {second_half_solved}/{len(second_half_results)} "
          f"({report['overfitting_analysis']['second_half_rate']*100:.1f}%)")
    print(f"  Performance change: {report['overfitting_analysis']['performance_change']*100:+.1f}%")
    
    if top_primitives:
        print(f"\nTop Used Primitives:")
        for primitive, count in top_primitives[:10]:
            print(f"  {primitive[:50]:50s}: {count} uses")
    
    print(f"\nResults saved to: {results_file}")
    
    # Save learned compounds if using them
    if use_compounds and hasattr(engine, 'compound_learner'):
        engine.compound_learner.save_compounds()
        stats = engine.compound_learner.analyze_compounds()
        print(f"\nCompound Learning Statistics:")
        print(f"  Total compounds learned: {stats['total']}")
        print(f"  Total successful uses: {stats.get('total_uses', 0)}")
    
    return report


if __name__ == "__main__":
    # Test on a subset first to verify everything works
    print("Starting with subset test (first 100 tasks)...")
    report = test_full_dataset(
        num_tasks=100,  # Start with 100 tasks
        checkpoint_interval=50,
        timeout_per_task=5.0,
        use_compounds=True,
        parallel=False
    )
    
    print("\n" + "=" * 70)
    print("Subset test complete. To run full dataset test, modify num_tasks=None")