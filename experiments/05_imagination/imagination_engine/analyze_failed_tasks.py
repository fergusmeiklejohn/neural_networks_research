"""Analyze failed ARC tasks to identify missing primitives."""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from arc_imagination_engine import ARCImaginationEngine
from arc_data_loader import load_arc_training_data

def analyze_task_characteristics(task_data: Dict) -> Dict:
    """Analyze characteristics of a task to understand what's needed."""
    characteristics = {
        'name': task_data.get('name', 'unknown'),
        'num_examples': len(task_data['train']),
        'grid_changes': [],
        'potential_operations': []
    }
    
    for example in task_data['train']:
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])
        
        # Size changes
        if input_grid.shape != output_grid.shape:
            characteristics['potential_operations'].append('resize')
        
        # Color analysis
        input_colors = set(np.unique(input_grid))
        output_colors = set(np.unique(output_grid))
        new_colors = output_colors - input_colors
        if new_colors:
            characteristics['potential_operations'].append('color_mapping')
        
        # Object counting
        non_zero_input = np.count_nonzero(input_grid)
        non_zero_output = np.count_nonzero(output_grid)
        if non_zero_output > non_zero_input * 1.5:
            characteristics['potential_operations'].append('object_duplication')
        
        # Pattern detection
        if output_grid.shape[0] == output_grid.shape[1]:
            # Check for rotation
            for k in range(1, 4):
                if np.array_equal(np.rot90(input_grid, k), output_grid):
                    characteristics['potential_operations'].append(f'rotate_{k*90}')
                    break
            
            # Check for reflection
            if np.array_equal(np.flip(input_grid, axis=0), output_grid):
                characteristics['potential_operations'].append('flip_vertical')
            if np.array_equal(np.flip(input_grid, axis=1), output_grid):
                characteristics['potential_operations'].append('flip_horizontal')
        
        # Check for sorting/ordering
        input_flat = input_grid.flatten()
        output_flat = output_grid.flatten()
        if np.array_equal(np.sort(input_flat), output_flat):
            characteristics['potential_operations'].append('sorting')
    
    # Remove duplicates
    characteristics['potential_operations'] = list(set(characteristics['potential_operations']))
    
    return characteristics

def test_on_sample_tasks(num_tasks: int = 20):
    """Test engine on sample tasks and analyze failures."""
    
    # Load tasks
    tasks = load_arc_training_data(max_tasks=num_tasks)
    engine = ARCImaginationEngine()
    
    results = []
    failed_tasks = []
    
    print(f"Testing on {len(tasks)} tasks...")
    
    for i, task_data in enumerate(tasks):
        task_name = task_data.get('id', f"task_{i}")
        print(f"\nTask {i+1}/{len(tasks)}: {task_name}")
        
        # Analyze task characteristics
        characteristics = analyze_task_characteristics(task_data)
        
        # Convert to numpy arrays
        train_examples = [
            (np.array(ex['input']), np.array(ex['output']))
            for ex in task_data['train']
        ]
        
        # Try to solve
        solution = engine.solve(train_examples, max_time=5.0)
        
        if solution:
            # Test on test examples
            test_examples = task_data.get('test', [])
            if test_examples:
                test_input = np.array(test_examples[0]['input'])
                test_output = np.array(test_examples[0]['output']) if 'output' in test_examples[0] else None
                
                # Apply the transform (solution is a Transform object)
                predicted = solution.apply(test_input)
                
                if test_output is not None:
                    accuracy = np.mean(predicted == test_output)
                    success = accuracy > 0.9
                else:
                    accuracy = -1
                    success = False
                
                results.append({
                    'task': task_name,
                    'accuracy': accuracy,
                    'success': success,
                    'characteristics': characteristics
                })
                
                print(f"  Accuracy: {accuracy:.2%}")
                print(f"  Operations detected: {characteristics['potential_operations']}")
                
                if not success:
                    failed_tasks.append({
                        'task': task_name,
                        'accuracy': accuracy,
                        'characteristics': characteristics,
                        'missing_operations': characteristics['potential_operations']
                    })
        else:
            print(f"  Failed to find solution")
            failed_tasks.append({
                'task': task_name,
                'accuracy': 0,
                'characteristics': characteristics,
                'missing_operations': characteristics['potential_operations']
            })
    
    # Analyze failures
    print("\n" + "="*50)
    print("FAILURE ANALYSIS")
    print("="*50)
    
    if failed_tasks:
        # Count missing operations
        missing_ops_count = {}
        for task in failed_tasks:
            for op in task['missing_operations']:
                missing_ops_count[op] = missing_ops_count.get(op, 0) + 1
        
        print("\nMost common missing operations:")
        for op, count in sorted(missing_ops_count.items(), key=lambda x: -x[1]):
            print(f"  {op}: {count} tasks")
        
        print(f"\nTotal tasks tested: {len(tasks)}")
        print(f"Tasks solved: {len(tasks) - len(failed_tasks)}")
        print(f"Tasks failed: {len(failed_tasks)}")
        print(f"Success rate: {(len(tasks) - len(failed_tasks)) / len(tasks):.1%}")
    else:
        print("All tasks solved successfully!")
    
    return results, failed_tasks

if __name__ == "__main__":
    results, failed_tasks = test_on_sample_tasks(num_tasks=20)
    
    # Save analysis
    with open('failed_tasks_analysis.json', 'w') as f:
        json.dump({
            'results': results,
            'failed_tasks': failed_tasks
        }, f, indent=2)
    
    print("\nAnalysis saved to failed_tasks_analysis.json")