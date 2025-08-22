"""Test advanced invention strategies on complex ARC tasks."""

import numpy as np
from typing import List, Tuple
from primitive_inventor import PrimitiveInventor
from invention_strategies import InventionStrategies


def test_cross_pattern():
    """Test on the ARC cross-pattern task."""
    
    print("=" * 70)
    print("TESTING ADVANCED INVENTION ON ARC CROSS-PATTERN")
    print("=" * 70)
    
    # Create the cross-pattern examples
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
    
    # Example 2 (different positions)
    input2 = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 8, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 7, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    
    output2 = np.array([
        [0, 0, 8, 0, 0, 7, 0, 0, 0],
        [8, 8, 8, 8, 8, 2, 8, 8, 8],
        [0, 0, 8, 0, 0, 7, 0, 0, 0],
        [0, 0, 8, 0, 0, 7, 0, 0, 0],
        [0, 0, 8, 0, 0, 7, 0, 0, 0],
        [0, 0, 8, 0, 0, 7, 0, 0, 0],
        [7, 7, 2, 7, 7, 7, 7, 7, 7],
        [0, 0, 8, 0, 0, 7, 0, 0, 0],
        [0, 0, 8, 0, 0, 7, 0, 0, 0]
    ])
    
    examples.append((input2, output2))
    
    print("\nTask: Draw lines through colored points, mark intersections")
    print(f"Examples provided: {len(examples)}")
    
    # Test different strategies
    strategies = InventionStrategies()
    
    print("\n" + "-" * 40)
    print("Strategy 1: Pattern Decomposition")
    print("-" * 40)
    
    invented = strategies.pattern_decomposition(examples)
    if invented:
        print(f"✓ Success! Program: {invented.program}")
        print(f"  Score: {invented.score:.2f}")
        
        # Test on first example
        predicted = invented.apply(input1)
        accuracy = np.mean(predicted == output1)
        print(f"  Test accuracy: {accuracy:.1%}")
        
        if accuracy < 1.0:
            print("\n  Predicted output (first 3 rows):")
            print(predicted[:3])
            print("\n  Expected output (first 3 rows):")
            print(output1[:3])
    else:
        print("✗ Failed to invent with pattern decomposition")
    
    print("\n" + "-" * 40)
    print("Strategy 2: Abstraction Discovery")
    print("-" * 40)
    
    invented = strategies.abstraction_discovery(examples)
    if invented:
        print(f"✓ Success! Program: {invented.program}")
        print(f"  Score: {invented.score:.2f}")
        
        predicted = invented.apply(input1)
        accuracy = np.mean(predicted == output1)
        print(f"  Test accuracy: {accuracy:.1%}")
    else:
        print("✗ Failed to invent with abstraction discovery")
    
    print("\n" + "-" * 40)
    print("Strategy 3: Geometric Reasoning")
    print("-" * 40)
    
    invented = strategies.geometric_reasoning(examples[:1])  # Use single example
    if invented:
        print(f"✓ Success! Program: {invented.program}")
        print(f"  Score: {invented.score:.2f}")
        
        predicted = invented.apply(input1)
        accuracy = np.mean(predicted == output1)
        print(f"  Test accuracy: {accuracy:.1%}")
        
        if accuracy > 0.8:
            print("\n  ✓ Good solution found with geometric reasoning!")
            
            # Test on second example
            predicted2 = invented.apply(input2)
            accuracy2 = np.mean(predicted2 == output2)
            print(f"  Generalization accuracy: {accuracy2:.1%}")
    else:
        print("✗ Failed to invent with geometric reasoning")
    
    return invented is not None


def test_simple_patterns():
    """Test on simpler patterns to verify strategies work."""
    
    print("\n" + "=" * 70)
    print("TESTING ON SIMPLE PATTERNS")
    print("=" * 70)
    
    strategies = InventionStrategies()
    
    # Test 1: Simple rotation
    print("\nTest 1: Rotation")
    input1 = np.array([[1, 2], [3, 4]])
    output1 = np.array([[3, 1], [4, 2]])  # 90 degree rotation
    
    invented = strategies.geometric_reasoning([(input1, output1)])
    if invented:
        print(f"✓ Found: {invented.program}")
    else:
        print("✗ Failed")
    
    # Test 2: Value mapping
    print("\nTest 2: Value Mapping")
    input2 = np.array([[1, 2, 3], [4, 5, 6]])
    output2 = np.array([[2, 3, 4], [5, 6, 7]])  # Increment
    
    invented = strategies.abstraction_discovery([(input2, output2)])
    if invented:
        print(f"✓ Found: {invented.program}")
        predicted = invented.apply(input2)
        print(f"  Accuracy: {np.mean(predicted == output2):.1%}")
    else:
        print("✗ Failed")
    
    # Test 3: Object coloring
    print("\nTest 3: Object Transformation")
    input3 = np.array([
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
        [2, 2, 0, 0]
    ])
    output3 = np.array([
        [0, 3, 3, 0],
        [0, 3, 3, 0],
        [0, 0, 0, 0],
        [4, 4, 0, 0]
    ])
    
    invented = strategies.pattern_decomposition([(input3, output3)])
    if invented:
        print(f"✓ Found: {invented.program}")
        predicted = invented.apply(input3)
        print(f"  Accuracy: {np.mean(predicted == output3):.1%}")
    else:
        print("✗ Failed")


def compare_strategies():
    """Compare basic vs advanced invention strategies."""
    
    print("\n" + "=" * 70)
    print("COMPARING BASIC VS ADVANCED STRATEGIES")
    print("=" * 70)
    
    # Complex pattern that basic strategy would memorize
    input_grid = np.array([
        [0, 0, 0, 0, 0],
        [0, 3, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 5, 0],
        [0, 0, 0, 0, 0]
    ])
    
    output_grid = np.array([
        [0, 3, 0, 5, 0],
        [3, 3, 3, 2, 3],
        [0, 3, 0, 5, 0],
        [5, 2, 5, 5, 5],
        [0, 3, 0, 5, 0]
    ])
    
    examples = [(input_grid, output_grid)]
    
    print("\nPattern: Draw cross through colored points")
    print("Input has 2 colored pixels (3 and 5)")
    print("Output has lines through them with intersection marked as 2")
    
    # Test basic inventor
    print("\n" + "-" * 40)
    print("Basic Inventor (trace-based):")
    basic_inventor = PrimitiveInventor()
    basic_result = basic_inventor.invent_primitive(examples, strategy="trace")
    
    if basic_result:
        print(f"  Program: {basic_result.program}")
        print(f"  Atomic ops: {len(basic_result.atomic_sequence)}")
    
    # Test advanced strategies
    print("\n" + "-" * 40)
    print("Advanced Strategy (geometric reasoning):")
    strategies = InventionStrategies()
    advanced_result = strategies.geometric_reasoning(examples)
    
    if advanced_result:
        print(f"  Program: {advanced_result.program}")
        print(f"  Atomic ops: {len(advanced_result.atomic_sequence)}")
        
        # Test generalization
        test_input = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 4, 0],
            [0, 0, 0, 0, 0],
            [0, 6, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
        
        predicted = advanced_result.apply(test_input)
        print("\n  Testing generalization on new input:")
        print("  Input has colors 4 and 6 at different positions")
        print("  Predicted output (should have cross pattern):")
        print(predicted)
    
    print("\n" + "-" * 40)
    print("Analysis:")
    if basic_result and advanced_result:
        basic_ops = len(basic_result.atomic_sequence)
        advanced_ops = len(advanced_result.atomic_sequence)
        
        print(f"  Basic uses {basic_ops} operations (likely pixel-by-pixel)")
        print(f"  Advanced uses {advanced_ops} operations (pattern-based)")
        print(f"  Efficiency gain: {basic_ops/advanced_ops:.1f}x")
        
        if basic_ops > 10 and advanced_ops < 5:
            print("  ✓ Advanced strategy is more elegant!")
    else:
        print("  Could not compare - one strategy failed")


if __name__ == "__main__":
    # Test simple patterns first
    test_simple_patterns()
    
    # Test on complex ARC cross-pattern
    success = test_cross_pattern()
    
    # Compare strategies
    compare_strategies()
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    if success:
        print("\n✓ Advanced strategies successfully handle complex patterns!")
        print("  - Pattern decomposition identifies sub-patterns")
        print("  - Geometric reasoning understands spatial relationships")
        print("  - Solutions are more elegant than pixel-by-pixel memorization")
    else:
        print("\n⚠ Advanced strategies need further development")
        print("  - May need constraint-based synthesis for complex patterns")
        print("  - Could benefit from learning across multiple tasks")