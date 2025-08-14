#!/usr/bin/env python3
"""Test the program synthesis system on actual ARC tasks."""

from utils.imports import setup_project_paths

setup_project_paths()

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from bidirectional_synthesis import BidirectionalSynthesizer
from compositional_dsl import CompositionalDSL
from neural_guided_synthesis import NeuralGuidedSynthesizer
from neural_program_ranker import NeuralProgramRanker


def load_arc_task(
    task_id: str,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[Tuple[np.ndarray, np.ndarray]]]:
    """Load an ARC task by ID."""
    data_dir = Path("data/arc_agi_official/ARC-AGI/data/training")
    task_file = data_dir / f"{task_id}.json"

    if not task_file.exists():
        raise FileNotFoundError(f"Task {task_id} not found")

    with open(task_file, "r") as f:
        task = json.load(f)

    # Convert to numpy arrays
    train_examples = [
        (np.array(ex["input"]), np.array(ex["output"])) for ex in task["train"]
    ]

    test_examples = [
        (np.array(ex["input"]), np.array(ex["output"])) for ex in task["test"]
    ]

    return train_examples, test_examples


def evaluate_program_on_task(program, train_examples, test_examples, dsl):
    """Evaluate a program on train and test examples."""
    # Check train accuracy
    train_correct = 0
    for inp, expected in train_examples:
        try:
            result = dsl.execute_program(program, inp)
            if np.array_equal(result, expected):
                train_correct += 1
        except Exception:
            pass

    # Check test accuracy
    test_correct = 0
    test_results = []
    for inp, expected in test_examples:
        try:
            result = dsl.execute_program(program, inp)
            test_results.append(result)
            if np.array_equal(result, expected):
                test_correct += 1
        except Exception:
            test_results.append(None)

    return {
        "train_accuracy": train_correct / len(train_examples) if train_examples else 0,
        "test_accuracy": test_correct / len(test_examples) if test_examples else 0,
        "test_results": test_results,
    }


def test_on_selected_tasks():
    """Test synthesis on a selection of ARC tasks."""
    # Create synthesizers
    dsl = CompositionalDSL()
    bidirectional = BidirectionalSynthesizer(dsl, timeout=15.0)

    # Create neural-guided synthesizer (with untrained ranker for now)
    neural_ranker = NeuralProgramRanker(
        vocab_size=50, hidden_dim=128, num_heads=4, num_layers=2
    )
    neural_guided = NeuralGuidedSynthesizer(
        dsl, neural_ranker=neural_ranker, beam_width=30, timeout=15.0, neural_weight=0.1
    )

    # Selected tasks that should be solvable with our current DSL
    test_tasks = [
        "00d62c1b",  # Simple color mapping
        "0520fde7",  # Horizontal flip
        "08ed6ac7",  # Pattern filling
        "0a938d79",  # Grid transformation
        "0b148d64",  # Object manipulation
    ]

    results = []

    for task_id in test_tasks:
        print(f"\n{'='*60}")
        print(f"Testing task: {task_id}")
        print("=" * 60)

        try:
            train_examples, test_examples = load_arc_task(task_id)
            print(
                f"Train examples: {len(train_examples)}, Test examples: {len(test_examples)}"
            )

            # Analyze task
            print("\nTask analysis:")
            for i, (inp, out) in enumerate(train_examples[:2]):
                print(f"  Example {i+1}: {inp.shape} -> {out.shape}")
                print(f"    Unique colors in: {np.unique(inp)}")
                print(f"    Unique colors out: {np.unique(out)}")

            # Try bidirectional synthesis
            print("\nBidirectional synthesis:")
            program_bi = bidirectional.synthesize(train_examples)

            if program_bi:
                print(f"  Found program: {program_bi}")
                eval_bi = evaluate_program_on_task(
                    program_bi, train_examples, test_examples, dsl
                )
                print(f"  Train accuracy: {eval_bi['train_accuracy']:.1%}")
                print(f"  Test accuracy: {eval_bi['test_accuracy']:.1%}")
            else:
                print("  No program found")
                eval_bi = {"train_accuracy": 0, "test_accuracy": 0}

            # Try neural-guided synthesis
            print("\nNeural-guided synthesis:")
            program_ng = neural_guided.synthesize(train_examples)

            if program_ng:
                print(f"  Found program: {program_ng}")
                eval_ng = evaluate_program_on_task(
                    program_ng, train_examples, test_examples, dsl
                )
                print(f"  Train accuracy: {eval_ng['train_accuracy']:.1%}")
                print(f"  Test accuracy: {eval_ng['test_accuracy']:.1%}")
            else:
                print("  No program found")
                eval_ng = {"train_accuracy": 0, "test_accuracy": 0}

            # Store results
            results.append(
                {
                    "task_id": task_id,
                    "bidirectional": {
                        "program": str(program_bi) if program_bi else None,
                        "train_acc": eval_bi["train_accuracy"],
                        "test_acc": eval_bi["test_accuracy"],
                    },
                    "neural_guided": {
                        "program": str(program_ng) if program_ng else None,
                        "train_acc": eval_ng["train_accuracy"],
                        "test_acc": eval_ng["test_accuracy"],
                    },
                }
            )

        except FileNotFoundError:
            print(f"  Task file not found")
            continue
        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)

    solved_bi = sum(1 for r in results if r["bidirectional"]["test_acc"] > 0)
    solved_ng = sum(1 for r in results if r["neural_guided"]["test_acc"] > 0)

    print(f"Tasks attempted: {len(results)}")
    print(
        f"Solved by bidirectional: {solved_bi}/{len(results)} ({solved_bi/len(results)*100:.1f}%)"
    )
    print(
        f"Solved by neural-guided: {solved_ng}/{len(results)} ({solved_ng/len(results)*100:.1f}%)"
    )

    print("\nPer-task results:")
    for r in results:
        print(f"\n{r['task_id']}:")
        print(
            f"  Bidirectional: Train={r['bidirectional']['train_acc']:.1%}, Test={r['bidirectional']['test_acc']:.1%}"
        )
        print(
            f"  Neural-guided: Train={r['neural_guided']['train_acc']:.1%}, Test={r['neural_guided']['test_acc']:.1%}"
        )

    return results


def test_single_task_detailed():
    """Test synthesis on a single task with detailed output."""
    task_id = "00d62c1b"  # Simple task
    print(f"Detailed test on task: {task_id}\n")

    # Load task
    train_examples, test_examples = load_arc_task(task_id)

    # Show examples
    print("Training examples:")
    for i, (inp, out) in enumerate(train_examples):
        print(f"\nExample {i+1}:")
        print(f"Input ({inp.shape}):")
        print(inp)
        print(f"Output ({out.shape}):")
        print(out)

    # Create DSL and synthesizer
    dsl = CompositionalDSL()
    synthesizer = BidirectionalSynthesizer(dsl, timeout=30.0)

    # Synthesize
    print("\nSynthesizing program...")
    program = synthesizer.synthesize(train_examples)

    if program:
        print(f"\nFound program: {program}")

        # Test on all examples
        print("\nTesting on training examples:")
        for i, (inp, expected) in enumerate(train_examples):
            result = dsl.execute_program(program, inp)
            match = np.array_equal(result, expected)
            print(f"  Example {i+1}: {'✓' if match else '✗'}")
            if not match:
                print(f"    Expected:\n{expected}")
                print(f"    Got:\n{result}")

        print("\nTesting on test examples:")
        for i, (inp, expected) in enumerate(test_examples):
            result = dsl.execute_program(program, inp)
            match = np.array_equal(result, expected)
            print(f"  Test {i+1}: {'✓' if match else '✗'}")
            if not match:
                print(
                    f"    Expected shape: {expected.shape}, Got shape: {result.shape}"
                )
    else:
        print("No program found!")


if __name__ == "__main__":
    # First test on single task
    print("=" * 60)
    print("SINGLE TASK TEST")
    print("=" * 60)
    test_single_task_detailed()

    # Then test on multiple tasks
    print("\n" + "=" * 60)
    print("MULTI-TASK TEST")
    print("=" * 60)
    results = test_on_selected_tasks()
