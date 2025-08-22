"""Test program synthesis on ARC tasks."""

import numpy as np
import json
from pathlib import Path
from program_synthesis import ProgramSynthesizer, Primitive, Sequence
from arc_primitives import ARCPrimitives


def test_on_fill_task():
    """Test on the enclosed region filling task."""
    print("\n=== Testing on Fill Task (00d62c1b) ===")
    
    task_file = Path(__file__).parent / "arc_agi_2_data" / "training" / "00d62c1b.json"
    
    if task_file.exists():
        with open(task_file, 'r') as f:
            task = json.load(f)
        
        # Get training examples
        examples = []
        for ex in task['train'][:3]:  # Use first 3 examples
            input_grid = np.array(ex['input'])
            output_grid = np.array(ex['output'])
            examples.append((input_grid, output_grid))
        
        print(f"Training on {len(examples)} examples")
        
        # Synthesize program
        synthesizer = ProgramSynthesizer()
        program = synthesizer.synthesize(examples, max_depth=2)
        
        if program:
            print(f"Synthesized program: {program.to_string()}")
            
            # Test on test example
            test_input = np.array(task['test'][0]['input'])
            test_output = np.array(task['test'][0]['output'])
            
            predicted = program.apply(test_input)
            accuracy = np.mean(predicted == test_output) if predicted.shape == test_output.shape else 0
            
            print(f"Test accuracy: {accuracy:.2%}")
            
            if accuracy == 1.0:
                print("✓ Perfect solution found!")
            elif accuracy > 0.8:
                print("✓ Good solution found")
            else:
                print("✗ Solution needs improvement")
                print(f"Expected shape: {test_output.shape}, Got: {predicted.shape}")
        else:
            print("✗ No program synthesized")
    else:
        print("Task file not found")


def test_on_tiling_task():
    """Test on the 3x3 tiling task."""
    print("\n=== Testing on Tiling Task (007bbfb7) ===")
    
    task_file = Path(__file__).parent / "arc_agi_2_data" / "training" / "007bbfb7.json"
    
    if task_file.exists():
        with open(task_file, 'r') as f:
            task = json.load(f)
        
        # Get training examples
        examples = []
        for ex in task['train'][:2]:  # Use first 2 examples
            input_grid = np.array(ex['input'])
            output_grid = np.array(ex['output'])
            examples.append((input_grid, output_grid))
        
        print(f"Training on {len(examples)} examples")
        
        # Synthesize program
        synthesizer = ProgramSynthesizer()
        program = synthesizer.synthesize(examples, max_depth=2)
        
        if program:
            print(f"Synthesized program: {program.to_string()}")
            
            # Test on test example
            test_input = np.array(task['test'][0]['input'])
            test_output = np.array(task['test'][0]['output'])
            
            predicted = program.apply(test_input)
            accuracy = np.mean(predicted == test_output) if predicted.shape == test_output.shape else 0
            
            print(f"Test accuracy: {accuracy:.2%}")
            
            if accuracy > 0.8:
                print("✓ Good solution found")
            else:
                print("Note: This task has complex tiling logic")
        else:
            print("✗ No program synthesized")
    else:
        print("Task file not found")


def test_simple_transformation():
    """Test on a simple synthetic transformation."""
    print("\n=== Testing on Simple Synthetic Task ===")
    
    # Create a simple task: fill all 0s with 5
    input1 = np.array([
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    output1 = np.array([
        [1, 1, 5],
        [5, 1, 5],
        [5, 5, 1]
    ])
    
    input2 = np.array([
        [2, 0, 2],
        [0, 0, 0],
        [2, 0, 2]
    ])
    output2 = np.array([
        [2, 5, 2],
        [5, 5, 5],
        [2, 5, 2]
    ])
    
    examples = [(input1, output1), (input2, output2)]
    
    print("Task: Replace all 0s with 5")
    
    # Create a simple program manually
    replace_zeros = Primitive(
        "replace_zeros",
        lambda g: np.where(g == 0, 5, g),
        {}
    )
    
    # Test manual program
    test_input = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    predicted = replace_zeros.apply(test_input)
    expected = np.array([[5, 1, 5], [1, 5, 1], [5, 1, 5]])
    
    print(f"Manual program works: {np.array_equal(predicted, expected)}")
    
    # Now try synthesis
    synthesizer = ProgramSynthesizer()
    
    # Add the replace primitive to the library
    synthesizer.primitive_library.append(replace_zeros)
    
    program = synthesizer.synthesize(examples, max_depth=1)
    
    if program:
        print(f"Synthesized: {program.to_string()}")
        print("✓ Synthesis successful")
    else:
        print("✗ Synthesis failed")


def test_multiple_arc_tasks():
    """Test on multiple ARC tasks to measure overall performance."""
    print("\n=== Testing on Multiple ARC Tasks ===")
    
    arc_dir = Path(__file__).parent / "arc_agi_2_data" / "training"
    
    if not arc_dir.exists():
        print("ARC data directory not found")
        return
    
    # Test on first 10 tasks
    task_files = list(arc_dir.glob("*.json"))[:10]
    
    synthesizer = ProgramSynthesizer()
    results = []
    
    for i, task_file in enumerate(task_files):
        with open(task_file, 'r') as f:
            task = json.load(f)
        
        # Get examples
        examples = []
        for ex in task['train'][:3]:
            input_grid = np.array(ex['input'])
            output_grid = np.array(ex['output'])
            examples.append((input_grid, output_grid))
        
        # Try to synthesize
        program = synthesizer.synthesize(examples, max_depth=2)
        
        if program and task['test']:
            # Test on first test example
            test_input = np.array(task['test'][0]['input'])
            test_output = np.array(task['test'][0]['output'])
            
            try:
                predicted = program.apply(test_input)
                if predicted.shape == test_output.shape:
                    accuracy = np.mean(predicted == test_output)
                else:
                    accuracy = 0.0
            except:
                accuracy = 0.0
            
            results.append(accuracy)
            status = "✓" if accuracy > 0.8 else "✗"
            print(f"Task {i+1} ({task_file.stem}): {accuracy:.1%} {status}")
        else:
            results.append(0.0)
            print(f"Task {i+1} ({task_file.stem}): No solution ✗")
    
    # Summary
    avg_accuracy = sum(results) / len(results) if results else 0
    solved = sum(1 for r in results if r > 0.8)
    
    print(f"\nSummary:")
    print(f"  Average accuracy: {avg_accuracy:.1%}")
    print(f"  Tasks solved (>80%): {solved}/{len(results)}")


def run_all_tests():
    """Run all synthesis tests."""
    print("=" * 60)
    print("Testing Program Synthesis on ARC Tasks")
    print("=" * 60)
    
    test_simple_transformation()
    test_on_fill_task()
    test_on_tiling_task()
    test_multiple_arc_tasks()
    
    print("\n" + "=" * 60)
    print("Program Synthesis Tests Complete")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()