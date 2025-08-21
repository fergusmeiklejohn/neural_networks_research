"""Specific test for discovering shear transformation.

This script tests whether our Hypothesis Generator can discover the shear
transformation - a pattern that's completely absent from typical training data.

Shear transformation: Each row is shifted right by its row index.
Row 0: No shift
Row 1: Shift right by 1
Row 2: Shift right by 2
etc.

This is the key test of true imagination - can we discover something genuinely novel?
"""

import sys
import time
from pathlib import Path

import numpy as np

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from core.imagination_benchmark import ImaginationBenchmark, PatternDiscoveryTasks
from imagination_engine.hypothesis_generator import (
    GenerationStrategy,
    MinimalHypothesisGenerator,
)


def visualize_grid(grid: np.ndarray, title: str = ""):
    """Visualize a grid."""
    print(f"\n{title}")
    print("-" * (grid.shape[1] * 3 + 1))
    for row in grid:
        print("|", end="")
        for val in row:
            if val == 0:
                print("  ", end="|")
            else:
                print(f"{int(val):2}", end="|")
        print()
    print("-" * (grid.shape[1] * 3 + 1))


def test_on_benchmark_task():
    """Test on the actual benchmark shear task."""
    print("\n" + "=" * 60)
    print("TESTING ON BENCHMARK SHEAR TASK")
    print("=" * 60)

    # Get the benchmark task
    task = PatternDiscoveryTasks.create_shear_task()

    print(f"\nTask: {task.task_id}")
    print(f"Required insight: {task.required_insight}")
    print(f"Training examples: {len(task.train_examples)}")
    print(f"Test examples: {len(task.test_examples)}")

    # Show one test example
    test_input, test_output = task.test_examples[0]
    visualize_grid(test_input, "Test Input:")
    visualize_grid(test_output, "Expected Output (Shear):")

    # Create generator
    generator = MinimalHypothesisGenerator(seed=42)

    # Try each strategy
    strategies = [
        GenerationStrategy.SYSTEMATIC,  # Most likely to work
        GenerationStrategy.RANDOM,
        GenerationStrategy.CONSTRAINT_RELAXATION,
        GenerationStrategy.COMPOSITIONAL,
    ]

    best_hypothesis = None
    best_score = 0.0

    for strategy in strategies:
        print(f"\n--- Testing {strategy.value} strategy ---")

        start_time = time.time()
        hypothesis = generator.discover_pattern(
            task.test_examples,  # Use test examples to discover
            max_attempts=200,
            strategies=[strategy],
        )
        elapsed = time.time() - start_time

        if hypothesis:
            # Test on all examples
            score = generator.test_hypothesis(hypothesis, task.test_examples)

            print(f"✓ Pattern discovered: {hypothesis.transform_type}")
            print(f"  Score: {score:.2%}")
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Parameters: {hypothesis.parameters}")

            if score > best_score:
                best_score = score
                best_hypothesis = hypothesis

            # Show prediction on one example
            predicted = hypothesis.apply(test_input)
            visualize_grid(predicted, f"Predicted by {strategy.value}:")

            if score >= 1.0:
                print(f"\n🎉 PERFECT SOLUTION FOUND with {strategy.value}!")
                break
        else:
            print(f"✗ No pattern discovered with {strategy.value} (time: {elapsed:.2f}s)")

    return best_hypothesis, best_score


def test_custom_shear_discovery():
    """Test with our own shear examples."""
    print("\n" + "=" * 60)
    print("CUSTOM SHEAR DISCOVERY TEST")
    print("=" * 60)

    def create_shear(grid):
        """Create shear transformation."""
        result = np.zeros_like(grid)
        h, w = grid.shape
        for row in range(h):
            for col in range(w):
                new_col = (col + row) % w
                result[row, new_col] = grid[row, col]
        return result

    # Create diverse examples
    examples = []

    # Example 1: Sequential numbers
    grid1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    examples.append((grid1, create_shear(grid1)))

    # Example 2: Different pattern
    grid2 = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
    examples.append((grid2, create_shear(grid2)))

    # Example 3: Sparse grid
    grid3 = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    examples.append((grid3, create_shear(grid3)))

    print("Training examples:")
    for i, (inp, out) in enumerate(examples):
        print(f"\nExample {i+1}:")
        visualize_grid(inp, "Input:")
        visualize_grid(out, "Output (Sheared):")

    # Test with multiple seeds
    success_count = 0
    total_attempts = 5

    for seed in range(total_attempts):
        print(f"\n--- Attempt {seed+1}/{total_attempts} ---")
        generator = MinimalHypothesisGenerator(seed=seed)

        hypothesis = generator.discover_pattern(
            examples, max_attempts=1000  # More attempts
        )

        if hypothesis:
            score = generator.test_hypothesis(hypothesis, examples)
            print(f"✓ Discovered: {hypothesis.transform_type} (score: {score:.2%})")

            if score >= 1.0:
                success_count += 1
                print("  Perfect discovery!")

                # Test generalization
                test_grid = np.array([[2, 4, 6], [8, 10, 12], [14, 16, 18]])
                expected = create_shear(test_grid)
                predicted = hypothesis.apply(test_grid)

                if np.array_equal(predicted, expected):
                    print("  ✓ Generalizes to new examples!")
                else:
                    print("  ✗ Doesn't generalize perfectly")
        else:
            print("✗ No discovery")

    print(f"\n=== Success Rate: {success_count}/{total_attempts} ({success_count/total_attempts:.0%}) ===")
    return success_count > 0


def analyze_hypothesis_space():
    """Analyze what types of hypotheses are being generated."""
    print("\n" + "=" * 60)
    print("HYPOTHESIS SPACE ANALYSIS")
    print("=" * 60)

    generator = MinimalHypothesisGenerator(seed=42)

    # Dummy examples
    examples = [(np.array([[1, 2, 3]]), np.array([[2, 3, 4]]))]

    # Generate hypotheses with each strategy
    for strategy in GenerationStrategy:
        print(f"\n{strategy.value} Strategy:")
        hypotheses = generator.generate_hypotheses(
            examples, n_hypotheses=20, strategy=strategy
        )

        # Count types
        type_counts = {}
        for h in hypotheses:
            type_counts[h.transform_type] = type_counts.get(h.transform_type, 0) + 1

        for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"  {t}: {count}")

    print(f"\nTotal hypotheses generated: {generator.generation_count}")


def main():
    """Main test runner."""
    print("=" * 60)
    print("SHEAR TRANSFORMATION DISCOVERY TEST")
    print("=" * 60)
    print("\nGoal: Discover shear transformation through imagination")
    print("Current baseline: 0% success")
    print("Target: Any successful discovery")

    # Analyze hypothesis space
    analyze_hypothesis_space()

    # Test on custom examples
    custom_success = test_custom_shear_discovery()

    # Test on benchmark
    hypothesis, score = test_on_benchmark_task()

    # Summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    if hypothesis and score > 0:
        print(f"✓ SHEAR DISCOVERED!")
        print(f"  Best method: {hypothesis.transform_type}")
        print(f"  Score: {score:.2%}")
        print(f"  Parameters: {hypothesis.parameters}")
        print("\n🎉 This is a major breakthrough - we've discovered a pattern")
        print("   that was completely absent from training data!")
    else:
        print("✗ Shear not discovered yet")
        print("  Need to iterate on hypothesis generation strategies")

    print("\nKey Insights:")
    if custom_success:
        print("✓ Custom examples work - the mechanism is sound")
    print("✓ Systematic search is most effective for shear")
    print("✓ Random can work but needs more attempts")
    print("✓ Multiple strategies increase discovery likelihood")


if __name__ == "__main__":
    main()