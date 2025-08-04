#!/usr/bin/env python3
"""Progressive complexity dataset for testing variable binding architectures.

This module creates datasets with increasing complexity levels to systematically
test the ability of neural architectures to learn and apply variable bindings.
"""

from utils.imports import setup_project_paths

setup_project_paths()

import random
from typing import Any, Dict, List


# Define vocabulary and actions (from existing code)
VOCAB = {
    "PAD": 0,
    "do": 1,
    "means": 2,
    "is": 3,
    "and": 4,
    "or": 5,
    "then": 6,
    "twice": 7,
    "thrice": 8,
    "while": 9,
    "X": 10,
    "Y": 11,
    "Z": 12,
    "W": 13,
    "jump": 14,
    "walk": 15,
    "run": 16,
    "turn": 17,
    "true": 18,
}

ACTIONS = ["JUMP", "WALK", "RUN", "TURN"]
ACTION_TO_IDX = {action: idx for idx, action in enumerate(ACTIONS)}


def tokenize(command: str) -> List[int]:
    """Convert command string to token indices."""
    tokens = []
    for word in command.split():
        if word in VOCAB:
            tokens.append(VOCAB[word])
        else:
            # Unknown token, use PAD
            tokens.append(VOCAB["PAD"])
    return tokens


def actions_to_indices(actions: List[str]) -> List[int]:
    """Convert action names to indices."""
    return [ACTION_TO_IDX[action] for action in actions]


class ProgressiveComplexityDataset:
    """Generate datasets with progressive complexity levels."""

    def __init__(self):
        self.variables = ["X", "Y", "Z", "W"]
        self.actions = ["jump", "walk", "run", "turn"]

    def generate_level_1(self, num_samples: int = 100) -> List[Dict[str, Any]]:
        """Level 1: Simple single bindings.

        Examples:
        - "X means jump do X" → ['JUMP']
        - "Y means walk do Y" → ['WALK']
        """
        data = []

        for _ in range(num_samples):
            var = random.choice(self.variables[:2])  # Only X, Y for simplicity
            action = random.choice(self.actions)

            command = f"{var} means {action} do {var}"
            expected = [action.upper()]

            data.append(
                {
                    "command": command,
                    "tokens": tokenize(command),
                    "expected_actions": expected,
                    "expected_indices": actions_to_indices(expected),
                    "complexity_level": 1,
                    "pattern_type": "simple_binding",
                }
            )

        return data

    def generate_level_2(self, num_samples: int = 100) -> List[Dict[str, Any]]:
        """Level 2: Multiple bindings and simple compositions.

        Examples:
        - "X means jump Y means walk do X and Y" → ['JUMP', 'WALK']
        - "X means run Y means turn do Y then X" → ['TURN', 'RUN']
        """
        data = []

        patterns = [
            # Parallel execution
            (
                "{v1} means {a1} {v2} means {a2} do {v1} and {v2}",
                lambda a1, a2: [a1.upper(), a2.upper()],
                "parallel",
            ),
            # Sequential execution
            (
                "{v1} means {a1} {v2} means {a2} do {v1} then {v2}",
                lambda a1, a2: [a1.upper(), a2.upper()],
                "sequential",
            ),
            # Reversed order
            (
                "{v1} means {a1} {v2} means {a2} do {v2} then {v1}",
                lambda a1, a2: [a2.upper(), a1.upper()],
                "reversed_sequential",
            ),
        ]

        for _ in range(num_samples):
            pattern, action_fn, pattern_type = random.choice(patterns)
            v1, v2 = random.sample(self.variables[:2], 2)
            a1, a2 = random.sample(self.actions, 2)

            command = pattern.format(v1=v1, v2=v2, a1=a1, a2=a2)
            expected = action_fn(a1, a2)

            data.append(
                {
                    "command": command,
                    "tokens": tokenize(command),
                    "expected_actions": expected,
                    "expected_indices": actions_to_indices(expected),
                    "complexity_level": 2,
                    "pattern_type": pattern_type,
                }
            )

        return data

    def generate_level_3(self, num_samples: int = 100) -> List[Dict[str, Any]]:
        """Level 3: Rebinding and temporal modifiers.

        Examples:
        - "X means jump do X then X means walk do X" → ['JUMP', 'WALK']
        - "X means jump do X twice then Y means walk do Y" → ['JUMP', 'JUMP', 'WALK']
        """
        data = []

        patterns = [
            # Simple rebinding
            (
                "{v} means {a1} do {v} then {v} means {a2} do {v}",
                lambda a1, a2: [a1.upper(), a2.upper()],
                "rebinding",
            ),
            # Temporal modifier
            (
                "{v} means {a1} do {v} twice",
                lambda a1: [a1.upper(), a1.upper()],
                "temporal_twice",
            ),
            # Temporal with sequential
            (
                "{v1} means {a1} do {v1} twice then {v2} means {a2} do {v2}",
                lambda a1, a2: [a1.upper(), a1.upper(), a2.upper()],
                "temporal_sequential",
            ),
            # Rebinding with different variable
            (
                "{v1} means {a1} do {v1} then {v2} means {a2} do {v2} then {v1} means {a3} do {v1}",
                lambda a1, a2, a3: [a1.upper(), a2.upper(), a3.upper()],
                "complex_rebinding",
            ),
        ]

        for _ in range(num_samples):
            pattern_choice = random.randint(0, len(patterns) - 1)
            pattern, action_fn, pattern_type = patterns[pattern_choice]

            if pattern_choice < 2:  # Single variable patterns
                v = random.choice(self.variables[:2])
                if pattern_choice == 1:  # Temporal pattern needs only one action
                    a1 = random.choice(self.actions)
                    command = pattern.format(v=v, a1=a1)
                    expected = action_fn(a1)
                else:  # Rebinding pattern needs two actions
                    a1, a2 = random.sample(self.actions, 2)
                    command = pattern.format(v=v, a1=a1, a2=a2)
                    expected = action_fn(a1, a2)
            elif pattern_choice == 2:  # Two variable pattern
                v1, v2 = random.sample(self.variables[:2], 2)
                a1, a2 = random.sample(self.actions, 2)
                command = pattern.format(v1=v1, v2=v2, a1=a1, a2=a2)
                expected = action_fn(a1, a2)
            else:  # Three action pattern
                v1, v2 = random.sample(self.variables[:2], 2)
                a1, a2, a3 = random.sample(self.actions, 3)
                command = pattern.format(v1=v1, v2=v2, a1=a1, a2=a2, a3=a3)
                expected = action_fn(a1, a2, a3)

            data.append(
                {
                    "command": command,
                    "tokens": tokenize(command),
                    "expected_actions": expected,
                    "expected_indices": actions_to_indices(expected),
                    "complexity_level": 3,
                    "pattern_type": pattern_type,
                }
            )

        return data

    def generate_level_4(self, num_samples: int = 100) -> List[Dict[str, Any]]:
        """Level 4: Complex compositions with long-range dependencies.

        Examples:
        - "X means jump do X then Y means walk do Y and X" → ['JUMP', 'WALK', 'JUMP']
        - "X means jump Y means walk do X and Y then X or Y" → ['JUMP', 'WALK', 'JUMP']
        """
        data = []

        patterns = [
            # Long-range reference
            (
                "{v1} means {a1} do {v1} then {v2} means {a2} do {v2} and {v1}",
                lambda a1, a2: [a1.upper(), a2.upper(), a1.upper()],
                "long_range",
            ),
            # Complex composition
            (
                "{v1} means {a1} {v2} means {a2} do {v1} and {v2} then {v1}",
                lambda a1, a2: [a1.upper(), a2.upper(), a1.upper()],
                "complex_composition",
            ),
            # Or operator (choose first)
            (
                "{v1} means {a1} {v2} means {a2} do {v1} or {v2}",
                lambda a1, a2: [a1.upper()],
                "or_operator",
            ),
            # Multiple segments
            (
                "{v1} means {a1} do {v1} then {v2} means {a2} do {v2} then do {v1} and {v2}",
                lambda a1, a2: [a1.upper(), a2.upper(), a1.upper(), a2.upper()],
                "multiple_segments",
            ),
        ]

        for _ in range(num_samples):
            pattern, action_fn, pattern_type = random.choice(patterns)
            v1, v2 = random.sample(self.variables[:2], 2)
            a1, a2 = random.sample(self.actions, 2)

            command = pattern.format(v1=v1, v2=v2, a1=a1, a2=a2)
            expected = action_fn(a1, a2)

            data.append(
                {
                    "command": command,
                    "tokens": tokenize(command),
                    "expected_actions": expected,
                    "expected_indices": actions_to_indices(expected),
                    "complexity_level": 4,
                    "pattern_type": pattern_type,
                }
            )

        return data

    def generate_all_levels(
        self, samples_per_level: int = 100
    ) -> Dict[str, List[Dict]]:
        """Generate datasets for all complexity levels."""
        return {
            "level_1": self.generate_level_1(samples_per_level),
            "level_2": self.generate_level_2(samples_per_level),
            "level_3": self.generate_level_3(samples_per_level),
            "level_4": self.generate_level_4(samples_per_level),
        }

    def generate_mixed_dataset(self, total_samples: int = 400) -> List[Dict[str, Any]]:
        """Generate a mixed dataset with all complexity levels."""
        samples_per_level = total_samples // 4
        all_data = []

        all_data.extend(self.generate_level_1(samples_per_level))
        all_data.extend(self.generate_level_2(samples_per_level))
        all_data.extend(self.generate_level_3(samples_per_level))
        all_data.extend(self.generate_level_4(samples_per_level))

        # Shuffle the data
        random.shuffle(all_data)
        return all_data


def test_dataset_generation():
    """Test the dataset generation."""
    print("Testing Progressive Complexity Dataset Generation...")

    dataset = ProgressiveComplexityDataset()

    # Test each level
    for level in range(1, 5):
        print(f"\n=== Level {level} Examples ===")
        samples = getattr(dataset, f"generate_level_{level}")(5)

        for i, sample in enumerate(samples):
            print(f"\nExample {i+1}:")
            print(f"  Command: {sample['command']}")
            print(f"  Expected: {sample['expected_actions']}")
            print(f"  Pattern: {sample['pattern_type']}")

    # Test mixed dataset
    print("\n=== Mixed Dataset Stats ===")
    mixed = dataset.generate_mixed_dataset(100)
    level_counts = {}
    pattern_counts = {}

    for sample in mixed:
        level = sample["complexity_level"]
        pattern = sample["pattern_type"]

        level_counts[level] = level_counts.get(level, 0) + 1
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

    print(f"Level distribution: {level_counts}")
    print(f"Total patterns: {len(pattern_counts)}")

    print("\nDataset generation test passed!")


if __name__ == "__main__":
    test_dataset_generation()
