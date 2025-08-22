"""Quick test of enhanced system."""

import numpy as np
from program_synthesis_v2 import EnhancedProgramSynthesizer
from arc_data_loader import load_arc_training_data

# Load a few tasks
tasks = load_arc_training_data(max_tasks=5)
synthesizer = EnhancedProgramSynthesizer()

results = []

for i, task in enumerate(tasks):
    print(f"\nTask {i+1}: {task['id']}")
    
    # Get examples
    examples = [
        (np.array(ex['input']), np.array(ex['output']))
        for ex in task['train']
    ]
    
    # Try synthesis
    solution = synthesizer.synthesize(examples, max_time=3.0)
    
    if solution:
        # Test it
        test = task['test'][0] if task['test'] else None
        if test:
            inp = np.array(test['input'])
            out = np.array(test['output']) if 'output' in test else None
            
            pred = solution.apply(inp)
            
            if out is not None and pred.shape == out.shape:
                acc = np.mean(pred == out)
                print(f"  Solution: {solution.to_string()}")
                print(f"  Accuracy: {acc:.1%}")
                results.append(acc)
            else:
                print(f"  Shape mismatch")
                results.append(0)
    else:
        print(f"  No solution found")
        results.append(0)

print(f"\n{'='*50}")
print(f"Summary: {sum(r > 0.9 for r in results)}/{len(results)} tasks solved")
print(f"Average accuracy: {sum(results)/len(results):.1%}")