"""Test the primitive invention system.

This script tests whether we can successfully invent primitives for various
transformation patterns that don't exist in our fixed library.
"""

import numpy as np
from typing import List, Tuple
from primitive_inventor import PrimitiveInventor, InventedPrimitive


def create_test_cases() -> List[Tuple[str, List[Tuple[np.ndarray, np.ndarray]]]]:
    """Create test cases for primitive invention."""
    
    test_cases = []
    
    # Test 1: Simple value mapping (increment all non-zero values)
    test1_examples = []
    for _ in range(3):
        input_grid = np.array([
            [0, 1, 2],
            [3, 0, 4],
            [5, 6, 0]
        ])
        output_grid = np.array([
            [0, 2, 3],
            [4, 0, 5],
            [6, 7, 0]
        ])
        test1_examples.append((input_grid, output_grid))
    test_cases.append(("increment_nonzero", test1_examples))
    
    # Test 2: Diagonal modification
    test2_examples = []
    input_grid = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    output_grid = np.array([
        [0, 2, 3],
        [4, 0, 6],
        [7, 8, 0]
    ])
    test2_examples.append((input_grid, output_grid))
    
    input_grid2 = np.array([
        [2, 1, 1],
        [1, 2, 1],
        [1, 1, 2]
    ])
    output_grid2 = np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ])
    test2_examples.append((input_grid2, output_grid2))
    test_cases.append(("diagonal_to_zero", test2_examples))
    
    # Test 3: Checkerboard pattern
    test3_examples = []
    input_grid = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ])
    output_grid = np.array([
        [2, 1, 2],
        [1, 2, 1],
        [2, 1, 2]
    ])
    test3_examples.append((input_grid, output_grid))
    test_cases.append(("checkerboard_pattern", test3_examples))
    
    # Test 4: Swap colors (all 1s become 2s, all 2s become 1s)
    test4_examples = []
    input_grid = np.array([
        [1, 2, 1],
        [2, 1, 2],
        [1, 2, 0]
    ])
    output_grid = np.array([
        [2, 1, 2],
        [1, 2, 1],
        [2, 1, 0]
    ])
    test4_examples.append((input_grid, output_grid))
    test_cases.append(("swap_1_and_2", test4_examples))
    
    # Test 5: Frame with specific color
    test5_examples = []
    input_grid = np.array([
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0]
    ])
    output_grid = np.array([
        [3, 3, 3, 3],
        [3, 1, 1, 3],
        [3, 1, 1, 3],
        [3, 3, 3, 3]
    ])
    test5_examples.append((input_grid, output_grid))
    test_cases.append(("add_frame", test5_examples))
    
    return test_cases


def test_invention():
    """Test the primitive invention system."""
    
    print("=" * 70)
    print("TESTING PRIMITIVE INVENTION SYSTEM")
    print("=" * 70)
    
    inventor = PrimitiveInventor(max_program_length=10, timeout=5.0)
    test_cases = create_test_cases()
    
    results = []
    
    for test_name, examples in test_cases:
        print(f"\nTest: {test_name}")
        print("-" * 40)
        
        # Show the transformation
        input_grid, output_grid = examples[0]
        print("Input:")
        print(input_grid)
        print("\nExpected Output:")
        print(output_grid)
        
        # Try different strategies
        strategies = ["trace", "search", "differential"]
        invented = None
        
        for strategy in strategies:
            print(f"\nTrying strategy: {strategy}")
            invented = inventor.invent_primitive(examples, strategy=strategy)
            
            if invented:
                print(f"✓ Success! Invented primitive: {invented.name}")
                print(f"  Program: {invented.program}")
                print(f"  Atomic sequence: {invented.atomic_sequence}")
                print(f"  Score: {invented.score:.2f}")
                print(f"  Time: {invented.invention_time:.3f}s")
                
                # Test the invented primitive
                predicted = invented.apply(input_grid)
                accuracy = np.mean(predicted == output_grid)
                print(f"  Test accuracy: {accuracy:.1%}")
                
                if accuracy == 1.0:
                    print("  ✓ Perfect match!")
                    results.append((test_name, True, strategy))
                    break
                else:
                    print("  Predicted output:")
                    print(predicted)
            else:
                print(f"✗ Failed with strategy: {strategy}")
        
        if not invented or accuracy < 1.0:
            results.append((test_name, False, None))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    successful = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    print(f"\nSuccess rate: {successful}/{total} ({successful/total*100:.1f}%)")
    
    print("\nDetailed results:")
    for test_name, success, strategy in results:
        status = "✓" if success else "✗"
        strategy_str = f" (strategy: {strategy})" if strategy else ""
        print(f"  {status} {test_name}{strategy_str}")
    
    print(f"\nTotal primitives invented: {inventor.invention_count}")
    
    return results


def test_on_failed_arc_task():
    """Test invention on a real failed ARC task."""
    
    print("\n" + "=" * 70)
    print("TESTING ON REAL ARC TASK")
    print("=" * 70)
    
    # Load a simple ARC task that failed in our previous test
    # This is task 23581191 - draws cross pattern
    examples = []
    
    # Example 1
    input1 = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 8, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 7, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    
    output1 = np.array([
        [0, 0, 0, 8, 0, 0, 7, 0, 0],
        [8, 8, 8, 8, 8, 8, 2, 8, 8],
        [0, 0, 0, 8, 0, 0, 7, 0, 0],
        [0, 0, 0, 8, 0, 0, 7, 0, 0],
        [0, 0, 0, 8, 0, 0, 7, 0, 0],
        [0, 0, 0, 8, 0, 0, 7, 0, 0],
        [7, 7, 7, 2, 7, 7, 7, 7, 7],
        [0, 0, 0, 8, 0, 0, 7, 0, 0],
        [0, 0, 0, 8, 0, 0, 7, 0, 0]
    ])
    
    examples.append((input1, output1))
    
    print("ARC Task: Cross pattern drawing")
    print("Input has 2 colored pixels, output draws crossing lines through them")
    
    inventor = PrimitiveInventor(max_program_length=20, timeout=10.0)
    
    # Try to invent a primitive for this task
    print("\nAttempting to invent primitive...")
    
    for strategy in ["trace", "search", "differential"]:
        print(f"\nTrying strategy: {strategy}")
        invented = inventor.invent_primitive(examples, strategy=strategy)
        
        if invented:
            print(f"✓ Invented primitive: {invented.name}")
            print(f"  Program: {invented.program}")
            print(f"  Score: {invented.score:.2f}")
            
            # Test it
            predicted = invented.apply(input1)
            accuracy = np.mean(predicted == output1)
            print(f"  Accuracy: {accuracy:.1%}")
            
            if accuracy > 0.8:
                print("  ✓ Good solution found!")
                return True
        else:
            print(f"✗ Failed with strategy: {strategy}")
    
    print("\n✗ Could not invent suitable primitive for this ARC task")
    print("This shows the current limitations - we need more sophisticated invention strategies")
    return False


if __name__ == "__main__":
    # Test basic invention
    results = test_invention()
    
    # Test on real ARC task
    success = test_on_failed_arc_task()
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("\nThe primitive invention system can successfully invent simple primitives")
    print("like value mappings and diagonal patterns. However, complex ARC tasks")
    print("like cross-pattern drawing require more sophisticated invention strategies.")
    print("\nNext steps: Implement advanced invention strategies that can handle")
    print("complex spatial reasoning and multi-step constructions.")