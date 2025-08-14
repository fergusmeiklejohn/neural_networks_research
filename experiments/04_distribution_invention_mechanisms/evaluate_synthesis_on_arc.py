#!/usr/bin/env python3
"""Evaluate the program synthesis system on a larger set of ARC tasks."""

from utils.imports import setup_project_paths

setup_project_paths()

import json
import random
import time
from pathlib import Path

import numpy as np
from improved_bidirectional_synthesis import ImprovedBidirectionalSynthesizer
from tqdm import tqdm


def load_arc_task(task_id: str, split: str = "training"):
    """Load an ARC task."""
    data_dir = Path(f"data/arc_agi_official/ARC-AGI/data/{split}")
    task_file = data_dir / f"{task_id}.json"

    if not task_file.exists():
        return None, None

    with open(task_file, "r") as f:
        task = json.load(f)

    train_examples = [
        (np.array(ex["input"]), np.array(ex["output"])) for ex in task["train"]
    ]
    test_examples = [
        (np.array(ex["input"]), np.array(ex["output"])) for ex in task["test"]
    ]

    return train_examples, test_examples


def get_all_task_ids(split: str = "training"):
    """Get all task IDs from a split."""
    data_dir = Path(f"data/arc_agi_official/ARC-AGI/data/{split}")
    return [f.stem for f in data_dir.glob("*.json")]


def evaluate_task(
    task_id: str, synthesizer: ImprovedBidirectionalSynthesizer, timeout: float = 10.0
):
    """Evaluate synthesis on a single task."""
    train_examples, test_examples = load_arc_task(task_id)

    if train_examples is None:
        return None

    # Set timeout for this task
    synthesizer.timeout = timeout

    # Try to synthesize
    start_time = time.time()
    try:
        program = synthesizer.synthesize(train_examples)
        synthesis_time = time.time() - start_time
    except Exception as e:
        return {
            "task_id": task_id,
            "solved": False,
            "program": None,
            "train_acc": 0.0,
            "test_acc": 0.0,
            "synthesis_time": timeout,
            "error": str(e),
        }

    if program is None:
        return {
            "task_id": task_id,
            "solved": False,
            "program": None,
            "train_acc": 0.0,
            "test_acc": 0.0,
            "synthesis_time": synthesis_time,
            "error": "No program found",
        }

    # Evaluate accuracy
    train_correct = 0
    for inp, expected in train_examples:
        try:
            result = synthesizer.dsl.execute_program(program, inp)
            if np.array_equal(result, expected):
                train_correct += 1
        except:
            pass

    test_correct = 0
    for inp, expected in test_examples:
        try:
            result = synthesizer.dsl.execute_program(program, inp)
            if np.array_equal(result, expected):
                test_correct += 1
        except:
            pass

    train_acc = train_correct / len(train_examples) if train_examples else 0
    test_acc = test_correct / len(test_examples) if test_examples else 0

    return {
        "task_id": task_id,
        "solved": test_acc == 1.0,
        "program": str(program),
        "train_acc": train_acc,
        "test_acc": test_acc,
        "synthesis_time": synthesis_time,
        "error": None,
    }


def evaluate_on_sample(num_tasks: int = 50, seed: int = 42):
    """Evaluate on a random sample of ARC tasks."""
    print(f"Evaluating on {num_tasks} random ARC tasks\n")

    # Get all task IDs
    all_tasks = get_all_task_ids("training")
    print(f"Total available tasks: {len(all_tasks)}")

    # Sample tasks
    random.seed(seed)
    sampled_tasks = random.sample(all_tasks, min(num_tasks, len(all_tasks)))

    # Create synthesizer
    synthesizer = ImprovedBidirectionalSynthesizer(timeout=10.0)

    # Evaluate each task
    results = []
    solved_count = 0

    print("\nEvaluating tasks:")
    for task_id in tqdm(sampled_tasks):
        result = evaluate_task(task_id, synthesizer, timeout=10.0)

        if result:
            results.append(result)
            if result["solved"]:
                solved_count += 1
                tqdm.write(f"  ✓ {task_id}: {result['program'][:50]}...")

    # Calculate statistics
    solved_tasks = [r for r in results if r["solved"]]
    avg_train_acc = np.mean([r["train_acc"] for r in results])
    avg_test_acc = np.mean([r["test_acc"] for r in results])
    avg_time = np.mean([r["synthesis_time"] for r in results])

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Tasks evaluated: {len(results)}")
    print(
        f"Tasks solved: {len(solved_tasks)} ({len(solved_tasks)/len(results)*100:.1f}%)"
    )
    print(f"Average train accuracy: {avg_train_acc:.1%}")
    print(f"Average test accuracy: {avg_test_acc:.1%}")
    print(f"Average synthesis time: {avg_time:.2f}s")

    # Show solved tasks
    if solved_tasks:
        print("\nSolved tasks:")
        for r in solved_tasks:
            print(f"  {r['task_id']}: {r['program'][:70]}")

    # Analyze program types
    program_types = {}
    for r in solved_tasks:
        # Extract main primitive from program
        prog_str = r["program"]
        if "FillInterior" in prog_str:
            prog_type = "FillInterior"
        elif "Rotate" in prog_str:
            prog_type = "Rotate"
        elif "FlipH" in prog_str or "FlipV" in prog_str:
            prog_type = "Flip"
        elif "SetColor" in prog_str:
            prog_type = "ColorTransform"
        elif "TilePattern" in prog_str or "RepeatGrid" in prog_str:
            prog_type = "Tiling"
        elif "MirrorSymmetry" in prog_str:
            prog_type = "Symmetry"
        else:
            prog_type = "Other"

        program_types[prog_type] = program_types.get(prog_type, 0) + 1

    if program_types:
        print("\nProgram types found:")
        for prog_type, count in sorted(program_types.items(), key=lambda x: -x[1]):
            print(f"  {prog_type}: {count}")

    return results


def test_known_solvable_tasks():
    """Test on tasks we know should be solvable."""
    print("Testing on known solvable tasks\n")

    # Tasks we've identified as solvable
    known_tasks = [
        ("00d62c1b", "FillInterior"),  # Fill interior
        ("3c9b0459", "Rotate"),  # Rotation
        ("25ff71a9", "ColorTransform"),  # Color mapping
        ("0520fde7", "Transform"),  # Some transformation
        ("08ed6ac7", "Pattern"),  # Pattern task
    ]

    synthesizer = ImprovedBidirectionalSynthesizer(timeout=15.0)
    results = []

    for task_id, expected_type in known_tasks:
        print(f"\nTask {task_id} (expecting {expected_type}):")

        result = evaluate_task(task_id, synthesizer, timeout=15.0)

        if result:
            if result["solved"]:
                print(f"  ✓ Solved with: {result['program']}")
                print(f"    Test accuracy: {result['test_acc']*100:.0f}%")
            else:
                print(f"  ✗ Failed. Best program: {result['program']}")
                print(
                    f"    Train acc: {result['train_acc']:.1%}, Test acc: {result['test_acc']:.1%}"
                )

            results.append(result)

    # Summary
    solved = sum(1 for r in results if r["solved"])
    print(
        f"\n{solved}/{len(results)} known tasks solved ({solved/len(results)*100:.0f}%)"
    )

    return results


if __name__ == "__main__":
    # First test on known solvable tasks
    print("=" * 60)
    print("TESTING KNOWN SOLVABLE TASKS")
    print("=" * 60)
    known_results = test_known_solvable_tasks()

    # Then evaluate on random sample
    print("\n" + "=" * 60)
    print("EVALUATING ON RANDOM SAMPLE")
    print("=" * 60)
    sample_results = evaluate_on_sample(num_tasks=30, seed=42)

    # Save results
    import json

    with open("synthesis_evaluation_results.json", "w") as f:
        json.dump(
            {"known_tasks": known_results, "random_sample": sample_results},
            f,
            indent=2,
            default=lambda x: str(x),
        )

    print("\nResults saved to synthesis_evaluation_results.json")
