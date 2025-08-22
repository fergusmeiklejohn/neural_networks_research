"""Test the enhanced ARC system with extended primitives."""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
import time

from arc_data_loader import load_arc_training_data
from program_synthesis_v2 import EnhancedProgramSynthesizer
from hypothesis_generator import MinimalHypothesisGenerator, GenerationStrategy
from arc_primitives_extended import ARCPrimitivesExtended


class EnhancedARCEngine:
    """Enhanced ARC engine with extended primitives and beam search."""
    
    def __init__(self):
        self.synthesizer = EnhancedProgramSynthesizer()
        self.hypothesis_gen = MinimalHypothesisGenerator()
        self.memory = []
        
    def solve(self, examples: List[Tuple[np.ndarray, np.ndarray]], 
             max_time: float = 10.0) -> Optional[Any]:
        """Try to solve an ARC task."""
        
        start_time = time.time()
        
        # First try program synthesis with beam search
        print("  Attempting enhanced synthesis with beam search...")
        solution = self.synthesizer.synthesize(examples, max_time * 0.7)
        
        if solution:
            return solution
        
        # If that fails, try hypothesis generation
        if time.time() - start_time < max_time:
            print("  Trying hypothesis generation...")
            for strategy in [GenerationStrategy.SYSTEMATIC, 
                            GenerationStrategy.CONSTRAINT_RELAXATION]:
                hypotheses = self.hypothesis_gen.generate_hypotheses(examples, n_hypotheses=5, strategy=strategy)
                for hypothesis in hypotheses:
                    score = self._evaluate_hypothesis(hypothesis.transform, examples)
                    if score > 0.8:
                        # Convert to a primitive for consistency
                        from program_synthesis_v2 import Primitive
                        return Primitive(f"hypothesis_{strategy.value}", hypothesis.transform)
        
        return None
    
    def _evaluate_hypothesis(self, transform, examples):
        """Evaluate a hypothesis."""
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


def test_on_tasks(num_tasks: int = 50):
    """Test enhanced system on more tasks."""
    
    print(f"Loading {num_tasks} ARC tasks...")
    tasks = load_arc_training_data(max_tasks=num_tasks)
    
    engine = EnhancedARCEngine()
    
    results = []
    task_types = {
        'resize': [],
        'color_mapping': [],
        'object_duplication': [],
        'pattern': [],
        'other': []
    }
    
    print(f"\nTesting on {len(tasks)} tasks...")
    print("=" * 60)
    
    for i, task_data in enumerate(tasks):
        task_id = task_data.get('id', f'task_{i}')
        print(f"\nTask {i+1}/{len(tasks)}: {task_id}")
        
        # Convert to numpy
        train_examples = [
            (np.array(ex['input']), np.array(ex['output']))
            for ex in task_data['train']
        ]
        
        # Categorize task
        task_type = categorize_task(train_examples)
        print(f"  Task type: {task_type}")
        
        # Try to solve
        solution = engine.solve(train_examples, max_time=5.0)
        
        if solution:
            # Test on test set
            test_examples = task_data.get('test', [])
            if test_examples:
                test_input = np.array(test_examples[0]['input'])
                test_output = np.array(test_examples[0]['output']) if 'output' in test_examples[0] else None
                
                try:
                    predicted = solution.apply(test_input)
                    
                    if test_output is not None and predicted.shape == test_output.shape:
                        accuracy = np.mean(predicted == test_output)
                        success = accuracy > 0.9
                        
                        results.append({
                            'task': task_id,
                            'type': task_type,
                            'accuracy': accuracy,
                            'success': success
                        })
                        
                        task_types[task_type].append(accuracy)
                        
                        status = "✓" if success else "✗"
                        print(f"  Result: {status} Accuracy: {accuracy:.1%}")
                    else:
                        print(f"  Result: ✗ Shape mismatch or no output")
                        task_types[task_type].append(0)
                except Exception as e:
                    print(f"  Result: ✗ Error: {e}")
                    task_types[task_type].append(0)
        else:
            print(f"  Result: ✗ No solution found")
            task_types[task_type].append(0)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    successful = [r for r in results if r.get('success', False)]
    print(f"\nOverall Performance:")
    print(f"  Tasks solved: {len(successful)}/{len(tasks)} ({len(successful)/len(tasks)*100:.1f}%)")
    
    if results:
        avg_accuracy = sum(r['accuracy'] for r in results) / len(results)
        print(f"  Average accuracy: {avg_accuracy:.1%}")
    
    print(f"\nPerformance by Task Type:")
    for task_type, scores in task_types.items():
        if scores:
            avg = sum(scores) / len(scores)
            solved = sum(1 for s in scores if s > 0.9)
            print(f"  {task_type:20s}: {solved}/{len(scores)} solved, {avg:.1%} avg accuracy")
    
    # Save detailed results
    with open('enhanced_results.json', 'w') as f:
        json.dump({
            'summary': {
                'total_tasks': len(tasks),
                'tasks_solved': len(successful),
                'success_rate': len(successful) / len(tasks) if tasks else 0,
                'average_accuracy': sum(r['accuracy'] for r in results) / len(results) if results else 0
            },
            'by_type': {
                task_type: {
                    'count': len(scores),
                    'solved': sum(1 for s in scores if s > 0.9),
                    'average': sum(scores) / len(scores) if scores else 0
                }
                for task_type, scores in task_types.items()
            },
            'detailed_results': results
        }, f, indent=2)
    
    print(f"\nDetailed results saved to enhanced_results.json")
    
    return results


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
    
    # Check for pattern
    if np.array_equal(np.rot90(inp), out) or np.array_equal(np.flip(inp, 0), out):
        return 'pattern'
    
    return 'other'


if __name__ == "__main__":
    results = test_on_tasks(num_tasks=50)
    
    print("\n" + "=" * 60)
    print("Testing complete!")