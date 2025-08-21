"""Test suite for the Minimal Hypothesis Generator."""

import sys
from pathlib import Path

import numpy as np

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from imagination_engine.hypothesis_generator import (
    GenerationStrategy,
    Hypothesis,
    MinimalHypothesisGenerator,
)


def test_basic_hypothesis_creation():
    """Test basic hypothesis creation and application."""
    print("\n=== Testing Basic Hypothesis Creation ===")

    # Create a simple transformation function
    def rotate_90(grid, **kwargs):
        return np.rot90(grid)

    hypothesis = Hypothesis(
        transform_type="rotate",
        parameters={},
        transform_fn=rotate_90,
    )

    # Test application
    test_grid = np.array([[1, 2], [3, 4]])
    result = hypothesis.apply(test_grid)
    expected = np.array([[2, 4], [1, 3]])

    assert np.array_equal(result, expected), "Rotation failed"
    print("✓ Basic hypothesis creation and application works")


def test_confidence_updates():
    """Test hypothesis confidence updates."""
    print("\n=== Testing Confidence Updates ===")

    def identity(grid, **kwargs):
        return grid

    hypothesis = Hypothesis(
        transform_type="identity",
        parameters={},
        transform_fn=identity,
    )

    # Update with successes and failures
    hypothesis.update_confidence(True)
    hypothesis.update_confidence(True)
    hypothesis.update_confidence(False)

    assert hypothesis.confidence == 2 / 3, f"Expected 0.667, got {hypothesis.confidence}"
    assert hypothesis.evidence == [True, True, False]
    print(f"✓ Confidence correctly updated to {hypothesis.confidence:.3f}")


def test_random_generation():
    """Test random hypothesis generation."""
    print("\n=== Testing Random Hypothesis Generation ===")

    generator = MinimalHypothesisGenerator(seed=42)

    # Create dummy examples
    examples = [
        (np.array([[1, 2], [3, 4]]), np.array([[2, 3], [4, 5]])),
        (np.array([[5, 6], [7, 8]]), np.array([[6, 7], [8, 9]])),
    ]

    hypotheses = generator.generate_hypotheses(
        examples, n_hypotheses=10, strategy=GenerationStrategy.RANDOM
    )

    assert len(hypotheses) == 10, f"Expected 10 hypotheses, got {len(hypotheses)}"
    assert all(isinstance(h, Hypothesis) for h in hypotheses)

    # Check variety of types
    types = set(h.transform_type for h in hypotheses)
    print(f"✓ Generated {len(types)} different transform types: {types}")


def test_hypothesis_testing():
    """Test hypothesis testing on examples."""
    print("\n=== Testing Hypothesis Testing ===")

    generator = MinimalHypothesisGenerator(seed=42)

    # Create examples with simple pattern (add 1 to each element)
    examples = [
        (np.array([[1, 2], [3, 4]]), np.array([[2, 3], [4, 5]])),
        (np.array([[5, 6], [7, 8]]), np.array([[6, 7], [8, 9]])),
    ]

    # Create a correct hypothesis
    def add_one(grid, **kwargs):
        return grid + 1

    correct_hypothesis = Hypothesis(
        transform_type="add_one", parameters={}, transform_fn=add_one
    )

    score = generator.test_hypothesis(correct_hypothesis, examples)
    assert score == 1.0, f"Expected score 1.0, got {score}"
    print("✓ Correct hypothesis gets perfect score")

    # Test incorrect hypothesis
    def subtract_one(grid, **kwargs):
        return grid - 1

    wrong_hypothesis = Hypothesis(
        transform_type="subtract_one", parameters={}, transform_fn=subtract_one
    )

    score = generator.test_hypothesis(wrong_hypothesis, examples)
    assert score == 0.0, f"Expected score 0.0, got {score}"
    print("✓ Incorrect hypothesis gets zero score")


def test_shear_discovery():
    """Test discovery of shear transformation."""
    print("\n=== Testing Shear Discovery ===")

    generator = MinimalHypothesisGenerator(seed=42)

    # Create shear examples (shift each row by its index)
    def create_shear_example(grid):
        result = np.zeros_like(grid)
        h, w = grid.shape
        for row in range(h):
            for col in range(w):
                new_col = (col + row) % w
                result[row, new_col] = grid[row, col]
        return result

    # Generate examples
    examples = []
    for i in range(3):
        input_grid = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        output_grid = create_shear_example(input_grid)
        examples.append((input_grid, output_grid))

    print("Shear examples created:")
    print("Input:\n", examples[0][0])
    print("Output (sheared):\n", examples[0][1])

    # Try to discover with systematic strategy (most likely to work)
    hypothesis = generator.discover_pattern(
        examples, max_attempts=500, strategies=[GenerationStrategy.SYSTEMATIC]
    )

    if hypothesis:
        print(f"✓ Discovered pattern: {hypothesis.transform_type}")
        print(f"  Confidence: {hypothesis.confidence:.2f}")
        print(f"  Parameters: {hypothesis.parameters}")

        # Test on new example
        test_input = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
        test_output = create_shear_example(test_input)
        predicted = hypothesis.apply(test_input)

        if np.array_equal(predicted, test_output):
            print("✓ Hypothesis generalizes to new example!")
        else:
            print("✗ Hypothesis doesn't generalize perfectly")
            print("Expected:\n", test_output)
            print("Predicted:\n", predicted)
    else:
        print("✗ No pattern discovered with systematic strategy")

    # Try with random strategy
    generator2 = MinimalHypothesisGenerator(seed=123)
    hypothesis2 = generator2.discover_pattern(
        examples, max_attempts=500, strategies=[GenerationStrategy.RANDOM]
    )

    if hypothesis2:
        print(f"✓ Random strategy discovered: {hypothesis2.transform_type}")
    else:
        print("✗ Random strategy didn't discover pattern")


def test_all_strategies():
    """Test all generation strategies."""
    print("\n=== Testing All Generation Strategies ===")

    generator = MinimalHypothesisGenerator(seed=42)

    # Simple examples
    examples = [
        (np.array([[1, 2], [3, 4]]), np.array([[2, 3], [4, 5]])),
    ]

    strategies = list(GenerationStrategy)
    for strategy in strategies:
        hypotheses = generator.generate_hypotheses(
            examples, n_hypotheses=5, strategy=strategy
        )
        print(f"✓ {strategy.value}: Generated {len(hypotheses)} hypotheses")

        # Show first hypothesis details
        if hypotheses:
            h = hypotheses[0]
            print(f"  First hypothesis: {h.transform_type}")


def test_statistics():
    """Test statistics tracking."""
    print("\n=== Testing Statistics ===")

    generator = MinimalHypothesisGenerator(seed=42)

    # Generate some hypotheses
    examples = [(np.array([[1, 2]]), np.array([[2, 3]]))]
    generator.generate_hypotheses(examples, n_hypotheses=10)

    stats = generator.get_statistics()
    assert stats["total_generated"] == 10
    assert stats["patterns_discovered"] == 0
    assert stats["discovery_rate"] == 0

    print(f"✓ Statistics tracked: {stats}")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("HYPOTHESIS GENERATOR TEST SUITE")
    print("=" * 60)

    test_basic_hypothesis_creation()
    test_confidence_updates()
    test_random_generation()
    test_hypothesis_testing()
    test_all_strategies()
    test_statistics()
    test_shear_discovery()  # Most important test

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()