#!/usr/bin/env python3
"""Test the extended synthesis system on ARC task 00d62c1b."""

from utils.imports import setup_project_paths

setup_project_paths()

import json
from pathlib import Path

import numpy as np
from bidirectional_synthesis import BidirectionalSynthesizer, BottomUpEnumerator
from extended_compositional_dsl import ExtendedCompositionalDSL
from neural_guided_synthesis import NeuralGuidedSynthesizer
from neural_program_ranker import NeuralProgramRanker


def load_arc_task(task_id: str):
    """Load an ARC task."""
    data_dir = Path("data/arc_agi_official/ARC-AGI/data/training")
    task_file = data_dir / f"{task_id}.json"

    with open(task_file, "r") as f:
        task = json.load(f)

    train_examples = [
        (np.array(ex["input"]), np.array(ex["output"])) for ex in task["train"]
    ]
    test_examples = [
        (np.array(ex["input"]), np.array(ex["output"])) for ex in task["test"]
    ]

    return train_examples, test_examples


def test_fill_interior_directly():
    """Test FillInterior primitive directly on task 00d62c1b."""
    print("Testing FillInterior directly on task 00d62c1b\n")

    # Load task
    train_examples, test_examples = load_arc_task("00d62c1b")

    # Create DSL
    dsl = ExtendedCompositionalDSL()

    # Create FillInterior program
    from advanced_dsl_primitives import FillInterior

    program = FillInterior(boundary_color=3, fill_color=4)

    # Test on training examples
    print("Testing on training examples:")
    for i, (inp, expected) in enumerate(train_examples[:3]):
        result = dsl.execute_program(program, inp)
        match = np.array_equal(result, expected)
        print(f"  Example {i+1}: {'✓' if match else '✗'}")
        if not match and i == 0:
            print(f"    Input shape: {inp.shape}")
            print(f"    Expected unique colors: {np.unique(expected)}")
            print(f"    Got unique colors: {np.unique(result)}")
            # Show first few rows for debugging
            print("    Expected (first 6 rows):")
            print(expected[:6])
            print("    Got (first 6 rows):")
            print(result[:6])

    # Test on test example
    print("\nTesting on test example:")
    for i, (inp, expected) in enumerate(test_examples):
        result = dsl.execute_program(program, inp)
        match = np.array_equal(result, expected)
        print(f"  Test {i+1}: {'✓' if match else '✗'}")


def test_synthesis_with_extended_dsl():
    """Test synthesis with the extended DSL."""
    print("\n" + "=" * 60)
    print("Testing Synthesis with Extended DSL")
    print("=" * 60)

    # Create extended DSL
    dsl = ExtendedCompositionalDSL()

    # Load task
    train_examples, test_examples = load_arc_task("00d62c1b")

    print(f"\nTask 00d62c1b:")
    print(f"  Train examples: {len(train_examples)}")
    print(f"  Test examples: {len(test_examples)}")

    # Analyze task
    suggestions = dsl.suggest_primitives_for_task(train_examples)
    print(f"  Suggested primitives: {suggestions}")

    # Update the enumerator to know about FillInterior
    class ExtendedBottomUpEnumerator(BottomUpEnumerator):
        def _generate_parameters(self, primitive_name, examples):
            """Extended parameter generation including FillInterior."""
            if primitive_name == "fill_interior":
                # Analyze examples for boundary and fill colors
                params_list = []
                for inp, out in examples:
                    in_colors = set(np.unique(inp))
                    out_colors = set(np.unique(out))
                    new_colors = out_colors - in_colors

                    # Common pattern: boundary is existing color, fill is new color
                    for boundary in in_colors:
                        if boundary != 0:  # Not background
                            for fill in new_colors:
                                params_list.append(
                                    {
                                        "boundary_color": int(boundary),
                                        "fill_color": int(fill),
                                    }
                                )

                    # Also try existing colors as fill
                    for boundary in in_colors:
                        if boundary != 0:
                            for fill in in_colors:
                                if fill != boundary and fill != 0:
                                    params_list.append(
                                        {
                                            "boundary_color": int(boundary),
                                            "fill_color": int(fill),
                                        }
                                    )

                return params_list[:10]  # Limit combinations
            else:
                # Use parent implementation
                return super()._generate_parameters(primitive_name, examples)

    # Create extended bidirectional synthesizer
    class ExtendedBidirectionalSynthesizer(BidirectionalSynthesizer):
        def __init__(self, dsl, timeout=30.0):
            super().__init__(dsl, timeout)
            self.bottom_up = ExtendedBottomUpEnumerator(dsl)

    # Test bidirectional synthesis
    print("\nBidirectional Synthesis:")
    synthesizer = ExtendedBidirectionalSynthesizer(dsl, timeout=20.0)
    program = synthesizer.synthesize(train_examples)

    if program:
        print(f"  Found program: {program}")

        # Test on examples
        train_correct = 0
        for inp, expected in train_examples:
            result = dsl.execute_program(program, inp)
            if np.array_equal(result, expected):
                train_correct += 1

        test_correct = 0
        for inp, expected in test_examples:
            result = dsl.execute_program(program, inp)
            if np.array_equal(result, expected):
                test_correct += 1

        print(
            f"  Train accuracy: {train_correct}/{len(train_examples)} ({train_correct/len(train_examples)*100:.0f}%)"
        )
        print(
            f"  Test accuracy: {test_correct}/{len(test_examples)} ({test_correct/len(test_examples)*100:.0f}%)"
        )

    else:
        print("  No program found")

    # Test neural-guided synthesis with extended DSL
    print("\nNeural-Guided Synthesis (with extended DSL):")

    # Create untrained ranker
    ranker = NeuralProgramRanker(
        vocab_size=100, hidden_dim=128, num_heads=4, num_layers=2
    )

    # Need to extend the neural guided synthesizer too
    class ExtendedNeuralGuidedSynthesizer(NeuralGuidedSynthesizer):
        def __init__(
            self,
            dsl,
            neural_ranker=None,
            beam_width=50,
            max_depth=5,
            neural_weight=0.3,
            timeout=30.0,
        ):
            super().__init__(
                dsl, neural_ranker, beam_width, max_depth, neural_weight, timeout
            )
            self.enumerator = ExtendedBottomUpEnumerator(dsl)

    neural_synthesizer = ExtendedNeuralGuidedSynthesizer(
        dsl, neural_ranker=ranker, beam_width=30, neural_weight=0.1, timeout=20.0
    )

    program = neural_synthesizer.synthesize(train_examples)

    if program:
        print(f"  Found program: {program}")

        # Test accuracy
        train_correct = sum(
            1
            for inp, expected in train_examples
            if np.array_equal(dsl.execute_program(program, inp), expected)
        )
        test_correct = sum(
            1
            for inp, expected in test_examples
            if np.array_equal(dsl.execute_program(program, inp), expected)
        )

        print(
            f"  Train accuracy: {train_correct}/{len(train_examples)} ({train_correct/len(train_examples)*100:.0f}%)"
        )
        print(
            f"  Test accuracy: {test_correct}/{len(test_examples)} ({test_correct/len(test_examples)*100:.0f}%)"
        )
    else:
        print("  No program found")


def test_more_arc_tasks():
    """Test on multiple ARC tasks with extended DSL."""
    print("\n" + "=" * 60)
    print("Testing Multiple ARC Tasks")
    print("=" * 60)

    dsl = ExtendedCompositionalDSL()

    # Tasks to test (selected based on likely solvability)
    test_tasks = [
        "00d62c1b",  # Fill interior (should work now!)
        "0520fde7",  # Transformation task
        "08ed6ac7",  # Pattern filling
        "25ff71a9",  # Simple transformation
        "3c9b0459",  # Object manipulation
    ]

    results = []

    for task_id in test_tasks:
        print(f"\nTask {task_id}:")
        try:
            train_examples, test_examples = load_arc_task(task_id)

            # Quick synthesis with timeout
            class ExtendedBottomUpEnumerator(BottomUpEnumerator):
                def _generate_parameters(self, primitive_name, examples):
                    if primitive_name == "fill_interior":
                        params_list = []
                        for inp, out in examples:
                            in_colors = set(np.unique(inp))
                            out_colors = set(np.unique(out))
                            new_colors = out_colors - in_colors
                            for boundary in in_colors:
                                if boundary != 0:
                                    for fill in new_colors:
                                        params_list.append(
                                            {
                                                "boundary_color": int(boundary),
                                                "fill_color": int(fill),
                                            }
                                        )
                        return params_list[:5]
                    elif primitive_name == "flood_fill":
                        # Generate flood fill parameters
                        return []  # Skip for now
                    else:
                        return super()._generate_parameters(primitive_name, examples)

            class QuickSynthesizer(BidirectionalSynthesizer):
                def __init__(self, dsl):
                    super().__init__(dsl, timeout=10.0)
                    self.bottom_up = ExtendedBottomUpEnumerator(dsl)

            synthesizer = QuickSynthesizer(dsl)
            program = synthesizer.synthesize(train_examples)

            if program:
                # Test accuracy
                test_correct = sum(
                    1
                    for inp, expected in test_examples
                    if np.array_equal(dsl.execute_program(program, inp), expected)
                )
                accuracy = test_correct / len(test_examples) if test_examples else 0
                print(f"  ✓ Solved! Program: {program}")
                print(f"  Test accuracy: {accuracy*100:.0f}%")
                results.append((task_id, True, accuracy))
            else:
                print(f"  ✗ No program found")
                results.append((task_id, False, 0))

        except Exception as e:
            print(f"  Error: {e}")
            results.append((task_id, False, 0))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    solved = sum(1 for _, success, _ in results if success)
    print(f"Solved: {solved}/{len(results)} tasks ({solved/len(results)*100:.0f}%)")
    for task_id, success, accuracy in results:
        status = "✓" if success else "✗"
        print(f"  {status} {task_id}: {accuracy*100:.0f}% accuracy")


if __name__ == "__main__":
    # First test FillInterior directly
    test_fill_interior_directly()

    # Then test synthesis
    test_synthesis_with_extended_dsl()

    # Finally test on multiple tasks
    test_more_arc_tasks()
