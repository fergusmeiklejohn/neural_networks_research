#!/usr/bin/env python3
"""Download and prepare ARC-AGI dataset for evaluation.

The ARC-AGI dataset structure:
- 400 training tasks (individual JSON files)
- 400 evaluation tasks (individual JSON files)
- Test tasks are held private

We'll use a pre-packaged version or create our own sample tasks for testing.
"""

from utils.imports import setup_project_paths

setup_project_paths()

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def create_sample_arc_tasks() -> Dict:
    """Create sample ARC tasks for testing our system.

    These represent different complexity levels and transformation types.
    """
    tasks = {}

    # Task 1: Simple color mapping (Easy)
    tasks["sample_color_map"] = {
        "train": [
            {
                "input": [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
                "output": [[2, 0, 2], [0, 2, 0], [2, 0, 2]],  # 1→2
            },
            {
                "input": [[1, 1, 0], [0, 1, 1], [1, 0, 0]],
                "output": [[2, 2, 0], [0, 2, 2], [2, 0, 0]],  # 1→2
            },
        ],
        "test": [
            {
                "input": [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
                "output": [[0, 2, 2], [2, 0, 2], [2, 2, 0]],  # Expected
            }
        ],
    }

    # Task 2: Object movement (Medium)
    tasks["sample_movement"] = {
        "train": [
            {
                "input": [[3, 0, 0], [0, 0, 0], [0, 0, 0]],
                "output": [[0, 0, 0], [3, 0, 0], [0, 0, 0]],  # Move down
            },
            {
                "input": [[0, 5, 0], [0, 0, 0], [0, 0, 0]],
                "output": [[0, 0, 0], [0, 5, 0], [0, 0, 0]],  # Move down
            },
        ],
        "test": [
            {
                "input": [[0, 0, 7], [0, 0, 0], [0, 0, 0]],
                "output": [[0, 0, 0], [0, 0, 7], [0, 0, 0]],  # Expected
            }
        ],
    }

    # Task 3: Pattern completion (Hard)
    tasks["sample_pattern"] = {
        "train": [
            {
                "input": [[1, 2, 0], [3, 4, 0], [0, 0, 0]],
                "output": [[1, 2, 1], [3, 4, 3], [1, 3, 1]],  # Pattern fill
            },
            {
                "input": [[5, 6, 0], [7, 8, 0], [0, 0, 0]],
                "output": [[5, 6, 5], [7, 8, 7], [5, 7, 5]],  # Same pattern
            },
        ],
        "test": [
            {
                "input": [[2, 3, 0], [4, 5, 0], [0, 0, 0]],
                "output": [[2, 3, 2], [4, 5, 4], [2, 4, 2]],  # Expected
            }
        ],
    }

    # Task 4: Symmetry creation (Medium)
    tasks["sample_symmetry"] = {
        "train": [
            {
                "input": [[1, 2, 0, 0], [3, 4, 0, 0]],
                "output": [[1, 2, 2, 1], [3, 4, 4, 3]],  # Mirror horizontally
            },
            {
                "input": [[5, 6, 0, 0], [7, 8, 0, 0]],
                "output": [[5, 6, 6, 5], [7, 8, 8, 7]],  # Mirror horizontally
            },
        ],
        "test": [
            {
                "input": [[2, 3, 0, 0], [4, 5, 0, 0]],
                "output": [[2, 3, 3, 2], [4, 5, 5, 4]],  # Expected
            }
        ],
    }

    # Task 5: Conditional fill (Hard - needs TTA)
    tasks["sample_conditional"] = {
        "train": [
            {
                "input": [[1, 0, 1], [0, 0, 0], [1, 0, 1]],
                "output": [[1, 2, 1], [2, 2, 2], [1, 2, 1]],  # Fill 0s with 2
            },
            {
                "input": [[3, 0, 0], [0, 0, 3], [0, 3, 0]],
                "output": [[3, 2, 2], [2, 2, 3], [2, 3, 2]],  # Fill 0s with 2
            },
        ],
        "test": [
            {
                "input": [[5, 0, 5], [0, 5, 0], [5, 0, 5]],
                "output": [[5, 2, 5], [2, 5, 2], [5, 2, 5]],  # Expected
            }
        ],
    }

    # Task 6: Scaling (Medium)
    tasks["sample_scaling"] = {
        "train": [
            {
                "input": [[1, 2], [3, 4]],
                "output": [
                    [1, 1, 2, 2],
                    [1, 1, 2, 2],
                    [3, 3, 4, 4],
                    [3, 3, 4, 4],
                ],  # 2x scale
            }
        ],
        "test": [
            {
                "input": [[5, 6], [7, 8]],
                "output": [
                    [5, 5, 6, 6],
                    [5, 5, 6, 6],
                    [7, 7, 8, 8],
                    [7, 7, 8, 8],
                ],  # Expected
            }
        ],
    }

    return tasks


class ARCDataset:
    """Manages ARC dataset for evaluation."""

    def __init__(self, data_dir: str = "data/arc"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Load or create sample tasks
        self.tasks = self.load_or_create_tasks()

    def load_or_create_tasks(self) -> Dict:
        """Load existing tasks or create sample ones."""
        tasks_file = self.data_dir / "sample_tasks.json"

        if tasks_file.exists():
            print(f"Loading tasks from {tasks_file}")
            with open(tasks_file) as f:
                return json.load(f)
        else:
            print("Creating sample ARC tasks...")
            tasks = create_sample_arc_tasks()

            # Save for future use
            with open(tasks_file, "w") as f:
                json.dump(tasks, f, indent=2)

            print(f"Saved {len(tasks)} sample tasks to {tasks_file}")
            return tasks

    def get_task(self, task_id: str) -> Dict:
        """Get a specific task."""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
        return self.tasks[task_id]

    def get_task_examples(
        self, task_id: str
    ) -> Tuple[
        List[Tuple[np.ndarray, np.ndarray]], List[Tuple[np.ndarray, np.ndarray]]
    ]:
        """Get task examples in format for our solver.

        Returns:
            train_examples: List of (input, output) pairs
            test_examples: List of (input, output) pairs
        """
        task = self.get_task(task_id)

        # Convert train examples
        train_examples = []
        for example in task["train"]:
            input_grid = np.array(example["input"], dtype=np.int32)
            output_grid = np.array(example["output"], dtype=np.int32)
            train_examples.append((input_grid, output_grid))

        # Convert test examples
        test_examples = []
        for example in task["test"]:
            input_grid = np.array(example["input"], dtype=np.int32)
            output_grid = np.array(example["output"], dtype=np.int32)
            test_examples.append((input_grid, output_grid))

        return train_examples, test_examples

    def list_tasks(self) -> List[str]:
        """List all available task IDs."""
        return list(self.tasks.keys())

    def print_summary(self):
        """Print summary of available tasks."""
        print("\n" + "=" * 70)
        print("ARC DATASET SUMMARY")
        print("=" * 70)

        print(f"\nTotal tasks: {len(self.tasks)}")
        print("\nTasks by difficulty:")

        easy = ["sample_color_map"]
        medium = ["sample_movement", "sample_symmetry", "sample_scaling"]
        hard = ["sample_pattern", "sample_conditional"]

        print(f"  Easy: {easy}")
        print(f"  Medium: {medium}")
        print(f"  Hard (needs TTA): {hard}")

        print("\nSample task analysis:")
        for task_id in self.list_tasks():
            task = self.get_task(task_id)
            print(f"\n  {task_id}:")
            print(f"    Train examples: {len(task['train'])}")
            print(f"    Test examples: {len(task['test'])}")
            if task["train"]:
                input_shape = np.array(task["train"][0]["input"]).shape
                output_shape = np.array(task["train"][0]["output"]).shape
                print(f"    Input shape: {input_shape}")
                print(f"    Output shape: {output_shape}")


def main():
    """Setup ARC dataset."""
    dataset = ARCDataset()
    dataset.print_summary()

    # Show a sample task
    print("\n" + "=" * 70)
    print("SAMPLE TASK: sample_color_map")
    print("=" * 70)

    train_examples, test_examples = dataset.get_task_examples("sample_color_map")

    print(f"\nTrain examples: {len(train_examples)}")
    for i, (input_grid, output_grid) in enumerate(train_examples):
        print(f"\nExample {i+1}:")
        print(f"Input:\n{input_grid}")
        print(f"Output:\n{output_grid}")

    print(f"\nTest examples: {len(test_examples)}")
    for i, (input_grid, expected_output) in enumerate(test_examples):
        print(f"\nTest {i+1}:")
        print(f"Input:\n{input_grid}")
        print(f"Expected:\n{expected_output}")

    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("1. Build evaluation pipeline (arc_evaluation_pipeline.py)")
    print("2. Test hybrid solver on these tasks")
    print("3. Test with and without TTA")
    print("4. Measure performance improvement")
    print("=" * 70)

    print("\nNote: Using sample tasks for testing. For full ARC evaluation,")
    print("download the complete dataset from: https://github.com/fchollet/ARC-AGI")


if __name__ == "__main__":
    main()
