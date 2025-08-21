"""Test Abstract Principle Extractor on cross-domain transfer tasks.

This specifically targets the cross-domain tasks where we had 0% success,
testing whether APE can enable transfer of principles across domains.
"""

import sys
from pathlib import Path

import numpy as np

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from abstract_principle_extractor import (
    AbstractPrincipleExtractor,
    Domain,
)
from core.imagination_benchmark import CrossDomainTasks
from hypothesis_generator import (
    GenerationStrategy,
    Hypothesis,
    MinimalHypothesisGenerator,
)


def visualize_transformation(input_grid, output_grid, title=""):
    """Visualize a transformation."""
    print(f"\n{title}")
    print("Input:")
    print(input_grid)
    print("Output:")
    print(output_grid)


def test_2d_to_color_rotation():
    """Test transferring rotation from 2D space to color space."""
    print("\n" + "=" * 60)
    print("Testing 2D to Color Rotation Transfer")
    print("=" * 60)

    # Get the benchmark task
    task = CrossDomainTasks.create_2d_to_color_rotation()

    print(f"Task: {task.task_id}")
    print(f"Required insight: {task.required_insight}")

    # Step 1: Create a simple rotation hypothesis
    def rotate_90(grid, **kwargs):
        return np.rot90(grid)

    rotation_hypothesis = Hypothesis(
        transform_type="rotate_90",
        parameters={"angle": 90},
        transform_fn=rotate_90,
    )

    # Step 2: Extract abstract principle
    ape = AbstractPrincipleExtractor()

    # Create examples showing rotation
    rotation_examples = [
        (np.array([[1, 2], [3, 4]]), np.array([[2, 4], [1, 3]])),
        (np.array([[5, 6], [7, 8]]), np.array([[6, 8], [5, 7]])),
    ]

    principle = ape.extract_principle(rotation_hypothesis, rotation_examples)

    if principle:
        print(f"\n✅ Extracted principle: {principle.name}")
        print(ape.explain_principle(principle))

        # Step 3: Transfer to color domain
        color_transform = ape.transfer_principle(principle, Domain.COLOR, task.test_examples)

        if color_transform:
            print("\n✅ Successfully created color domain transform")

            # Test on task examples
            for i, (inp, expected) in enumerate(task.test_examples[:2]):
                result = color_transform(inp)
                score = task.evaluate_solution(result, expected)

                print(f"\nExample {i+1}:")
                print(f"Input: {inp}")
                print(f"Expected: {expected}")
                print(f"Got: {result}")
                print(f"Score: {score:.1%}")

                if score > 0.5:
                    print("✅ Successful transfer!")
                    return True
        else:
            print("❌ Failed to create color transform")
    else:
        print("❌ Failed to extract principle")

    return False


def test_symmetry_transfer():
    """Test transferring symmetry principle across domains."""
    print("\n" + "=" * 60)
    print("Testing Symmetry Transfer")
    print("=" * 60)

    # Get the benchmark task
    task = CrossDomainTasks.create_symmetry_transfer()

    print(f"Task: {task.task_id}")
    print(f"Required insight: {task.required_insight}")

    # Step 1: Create a reflection hypothesis
    def reflect_horizontal(grid, **kwargs):
        return np.flipud(grid)

    reflection_hypothesis = Hypothesis(
        transform_type="reflect_horizontal",
        parameters={"axis": "horizontal"},
        transform_fn=reflect_horizontal,
    )

    # Step 2: Extract abstract principle
    ape = AbstractPrincipleExtractor()

    # Create examples showing reflection
    reflection_examples = [
        (np.array([[1, 2], [3, 4]]), np.array([[3, 4], [1, 2]])),
        (np.array([[5, 6], [7, 8]]), np.array([[7, 8], [5, 6]])),
    ]

    principle = ape.extract_principle(reflection_hypothesis, reflection_examples)

    if principle:
        print(f"\n✅ Extracted principle: {principle.name}")
        print(ape.explain_principle(principle))

        # Identify target domain from task
        target_domain = ape.identify_domain(task.test_examples[0][0])
        print(f"Target domain identified as: {target_domain}")

        # Step 3: Transfer principle
        transferred_transform = ape.transfer_principle(principle, target_domain, task.test_examples)

        if transferred_transform:
            print("\n✅ Successfully transferred principle")

            # Test on task examples
            for i, (inp, expected) in enumerate(task.test_examples[:2]):
                result = transferred_transform(inp)
                score = task.evaluate_solution(result, expected)

                print(f"\nExample {i+1}:")
                print(f"Score: {score:.1%}")

                if score > 0.5:
                    print("✅ Successful transfer!")
                    return True
        else:
            print("❌ Failed to transfer principle")
    else:
        print("❌ Failed to extract principle")

    return False


def test_with_hypothesis_generator():
    """Test full pipeline: discover pattern, extract principle, transfer."""
    print("\n" + "=" * 60)
    print("Testing Full Pipeline: Discover → Extract → Transfer")
    print("=" * 60)

    # Step 1: Discover a shear pattern
    print("\n1. Discovering shear pattern...")
    generator = MinimalHypothesisGenerator(seed=42)

    # Create shear examples
    def create_shear(grid):
        result = np.zeros_like(grid)
        h, w = grid.shape
        for row in range(h):
            for col in range(w):
                new_col = (col + row) % w
                result[row, new_col] = grid[row, col]
        return result

    shear_examples = [
        (np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), create_shear(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))),
        (np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]]), create_shear(np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])))
    ]

    hypothesis = generator.discover_pattern(
        shear_examples, max_attempts=200, strategies=[GenerationStrategy.SYSTEMATIC]
    )

    if hypothesis:
        print(f"✅ Discovered: {hypothesis.transform_type}")

        # Step 2: Extract principle
        print("\n2. Extracting abstract principle...")
        ape = AbstractPrincipleExtractor()
        principle = ape.extract_principle(hypothesis, shear_examples)

        if principle:
            print(f"✅ Extracted: {principle.name}")
            print(ape.explain_principle(principle))

            # Step 3: Transfer to color domain
            print("\n3. Transferring to color domain...")

            # Create color grid examples
            color_examples = [
                (np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.array([[1, 2, 3], [6, 4, 5], [8, 9, 7]]))
            ]

            color_transform = ape.transfer_principle(principle, Domain.COLOR, color_examples)

            if color_transform:
                print("✅ Successfully transferred to color domain")

                # Test the transfer
                test_input = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
                result = color_transform(test_input)
                print(f"\nTest input:\n{test_input}")
                print(f"Transformed:\n{result}")

                return True
            else:
                print("❌ Failed to transfer to color domain")
        else:
            print("❌ Failed to extract principle")
    else:
        print("❌ Failed to discover pattern")

    return False


def test_principle_composition():
    """Test composing multiple principles."""
    print("\n" + "=" * 60)
    print("Testing Principle Composition")
    print("=" * 60)

    ape = AbstractPrincipleExtractor()

    # Create two simple hypotheses
    def rotate_90(grid, **kwargs):
        return np.rot90(grid)

    def reflect_h(grid, **kwargs):
        return np.flipud(grid)

    rotation_hyp = Hypothesis(
        transform_type="rotate_90", parameters={"angle": 90}, transform_fn=rotate_90
    )

    reflection_hyp = Hypothesis(
        transform_type="reflect_h", parameters={"axis": "horizontal"}, transform_fn=reflect_h
    )

    # Extract principles
    rotation_examples = [(np.array([[1, 2], [3, 4]]), np.array([[2, 4], [1, 3]]))]
    reflection_examples = [(np.array([[1, 2], [3, 4]]), np.array([[3, 4], [1, 2]]))]

    principle1 = ape.extract_principle(rotation_hyp, rotation_examples)
    principle2 = ape.extract_principle(reflection_hyp, reflection_examples)

    if principle1 and principle2:
        print("✅ Extracted both principles")

        # Compose them
        composed = ape.compose_principles([principle1, principle2])

        # Test composition
        test_input = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = composed(test_input)

        print(f"\nOriginal:\n{test_input}")
        print(f"After rotation + reflection:\n{result}")

        # Verify it's correct (rotate 90 then flip up-down)
        expected = np.flipud(np.rot90(test_input))
        if np.array_equal(result, expected):
            print("✅ Composition works correctly!")
            return True
        else:
            print("❌ Composition produced unexpected result")
            print(f"Expected:\n{expected}")
    else:
        print("❌ Failed to extract principles")

    return False


def main():
    """Run all cross-domain tests."""
    print("=" * 60)
    print("CROSS-DOMAIN TRANSFER TESTS")
    print("=" * 60)

    results = []

    # Test each cross-domain task
    print("\n🔬 Test 1: 2D to Color Rotation")
    results.append(("2D to Color Rotation", test_2d_to_color_rotation()))

    print("\n🔬 Test 2: Symmetry Transfer")
    results.append(("Symmetry Transfer", test_symmetry_transfer()))

    print("\n🔬 Test 3: Full Pipeline")
    results.append(("Full Pipeline", test_with_hypothesis_generator()))

    print("\n🔬 Test 4: Principle Composition")
    results.append(("Principle Composition", test_principle_composition()))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    success_count = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {test_name}")

    print(f"\nOverall: {success_count}/{total} tests passed ({success_count/total:.0%})")

    if success_count > 0:
        print("\n🎉 Abstract Principle Extractor shows promise for cross-domain transfer!")
    else:
        print("\n📝 More work needed on cross-domain transfer mechanisms")


if __name__ == "__main__":
    main()