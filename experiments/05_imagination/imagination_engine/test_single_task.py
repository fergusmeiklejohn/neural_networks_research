"""Test a single ARC task to see what's happening."""

import numpy as np
from imagination_engine_v4 import ImaginationEngineV4
from arc_data_loader import load_arc_training_data

# Load one task
tasks = load_arc_training_data(max_tasks=1)
task_data = tasks[0]

# Prepare task for engine
train_examples = []
for example in task_data['train']:
    inp = np.array(example['input'], dtype=np.float32)
    out = np.array(example['output'], dtype=np.float32)
    train_examples.append({'input': inp.tolist(), 'output': out.tolist()})

test_examples = []
for example in task_data['test']:
    inp = np.array(example['input'], dtype=np.float32)
    out = np.array(example['output'], dtype=np.float32)
    test_examples.append({'input': inp.tolist()})

task_dict = {
    'id': 'test_task',
    'train': train_examples,
    'test': test_examples
}

print("Task prepared:")
print(f"  Train examples: {len(train_examples)}")
print(f"  Test examples: {len(test_examples)}")
print(f"  First train input shape: {np.array(train_examples[0]['input']).shape}")
print(f"  First train output shape: {np.array(train_examples[0]['output']).shape}")

# Create engine with verbose output
engine = ImaginationEngineV4(verbose=True)

# Solve
print("\nSolving task...")
solution = engine.solve(task_dict, timeout=10.0)

print(f"\nSolution found: {solution is not None}")
if solution:
    print(f"  Strategy used: {solution.strategy_used}")
    print(f"  Accuracy: {solution.accuracy:.2%}")
    print(f"  Predictions: {len(solution.predictions)}")
    
    if solution.predictions:
        pred = solution.predictions[0]
        expected = np.array(task_data['test'][0]['output'])
        
        print(f"\nFirst prediction shape: {pred.shape if pred is not None else 'None'}")
        print(f"Expected shape: {expected.shape}")
        
        if pred is not None:
            print(f"Match: {np.array_equal(pred, expected)}")
            if pred.shape == expected.shape:
                accuracy = np.sum(pred == expected) / pred.size
                print(f"Pixel accuracy: {accuracy:.2%}")

print("\nEngine statistics:")
print(f"  Memory hits: {engine.memory_hits}")
print(f"  New inventions: {engine.new_inventions}")
print(f"  Meta-learning successes: {engine.meta_learning_successes}")
print(f"  Total tasks solved: {engine.total_tasks_solved}")