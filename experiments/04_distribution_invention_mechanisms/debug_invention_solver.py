#!/usr/bin/env python3
"""Debug the invention solver to see why it's not selecting the right distribution."""

from utils.imports import setup_project_paths

setup_project_paths()

import json
from pathlib import Path

import numpy as np
from invention_based_solver import DistributionInventionSolver

# Get data path
DATA_DIR = (
    Path(__file__).parent.parent.parent
    / "data"
    / "arc_agi_official"
    / "ARC-AGI"
    / "data"
    / "training"
)


def load_arc_task(task_path: Path):
    """Load an ARC task from JSON file."""
    with open(task_path) as f:
        return json.load(f)


def debug_solver():
    """Debug the solver's selection process."""
    task_path = DATA_DIR / "05269061.json"
    task = load_arc_task(task_path)

    solver = DistributionInventionSolver(invention_budget=50)

    # Get examples
    train_examples = [
        (np.array(ex["input"]), np.array(ex["output"])) for ex in task["train"]
    ]

    test_input = np.array(task["test"][0]["input"])
    expected_output = np.array(task["test"][0]["output"])
    output_shape = expected_output.shape

    # Extract hints
    hints = solver.extract_distribution_hints(train_examples)

    # Invent distributions
    distributions = solver.invent_distributions(test_input, hints)

    # Get correct pattern
    unique_test = sorted(set(test_input.flatten()) - {0})
    correct_pattern = list(expected_output.flatten()[: len(unique_test)])

    print("Evaluating distributions:")
    print("=" * 60)

    # Evaluate each distribution
    scores = []
    for i, dist in enumerate(distributions[:10]):
        score = solver.evaluate_distribution(dist, train_examples, output_shape)
        is_correct = "✓" if dist == correct_pattern else " "
        scores.append((dist, score, is_correct))
        print(f"{is_correct} {i+1:2}. {dist} -> score: {score:.3f}")

    # Sort by score
    scores.sort(key=lambda x: x[1], reverse=True)

    print("\nTop 5 by score:")
    print("-" * 40)
    for dist, score, is_correct in scores[:5]:
        print(f"{is_correct} {dist} -> {score:.3f}")

    print("\nThe issue: Correct pattern has low score!")
    print("This shows that pure example-matching isn't enough.")
    print("\nWe need a different approach...")

    # Let's try each distribution and see which gives best result
    print("\n" + "=" * 60)
    print("Testing each distribution directly:")

    best_accuracy = 0
    best_dist = None

    for i, dist in enumerate(distributions[:10]):
        # Create output
        output_size = output_shape[0] * output_shape[1]
        result = np.tile(dist, (output_size // len(dist)) + 1)[:output_size]
        result = result.reshape(output_shape)

        # Check accuracy
        accuracy = np.sum(result == expected_output) / expected_output.size

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_dist = dist

        is_correct = "✓" if dist == correct_pattern else " "
        if accuracy > 0.9 or dist == correct_pattern:
            print(f"{is_correct} {i+1:2}. {dist} -> accuracy: {accuracy:.1%}")

    print(f"\nBest distribution by actual accuracy: {best_dist}")
    print(f"Best accuracy: {best_accuracy:.1%}")

    if best_accuracy == 1.0:
        print("\n✅ Found perfect solution!")
        print("The key: We need to actually TEST distributions, not just score them!")


if __name__ == "__main__":
    debug_solver()
