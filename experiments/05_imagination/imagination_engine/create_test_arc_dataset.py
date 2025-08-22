"""Create a small test ARC dataset for evaluation."""

import json
import numpy as np
from pathlib import Path


def create_test_dataset():
    """Create test ARC tasks."""
    
    output_dir = Path("test_arc_dataset")
    output_dir.mkdir(exist_ok=True)
    
    tasks = []
    
    # Task 1: Increment values
    task1 = {
        "train": [
            {"input": [[1, 2], [3, 4]], "output": [[2, 3], [4, 5]]},
            {"input": [[5, 6], [7, 8]], "output": [[6, 7], [8, 9]]}
        ],
        "test": [
            {"input": [[10, 11], [12, 13]], "output": [[11, 12], [13, 14]]}
        ]
    }
    
    # Task 2: Diagonal pattern
    task2 = {
        "train": [
            {"input": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 
             "output": [[2, 0, 0], [0, 2, 0], [0, 0, 2]]},
            {"input": [[3, 0, 0], [0, 3, 0], [0, 0, 3]], 
             "output": [[4, 0, 0], [0, 4, 0], [0, 0, 4]]}
        ],
        "test": [
            {"input": [[5, 0, 0], [0, 5, 0], [0, 0, 5]], 
             "output": [[6, 0, 0], [0, 6, 0], [0, 0, 6]]}
        ]
    }
    
    # Task 3: Border addition
    task3 = {
        "train": [
            {"input": [[1, 1], [1, 1]], 
             "output": [[3, 3, 3, 3], [3, 1, 1, 3], [3, 1, 1, 3], [3, 3, 3, 3]]}
        ],
        "test": [
            {"input": [[2, 2], [2, 2]], 
             "output": [[3, 3, 3, 3], [3, 2, 2, 3], [3, 2, 2, 3], [3, 3, 3, 3]]}
        ]
    }
    
    # Task 4: Color swap
    task4 = {
        "train": [
            {"input": [[1, 2, 1], [2, 1, 2]], "output": [[2, 1, 2], [1, 2, 1]]},
            {"input": [[3, 4, 3], [4, 3, 4]], "output": [[4, 3, 4], [3, 4, 3]]}
        ],
        "test": [
            {"input": [[5, 6, 5], [6, 5, 6]], "output": [[6, 5, 6], [5, 6, 5]]}
        ]
    }
    
    # Task 5: Cross pattern
    task5 = {
        "train": [
            {
                "input": [
                    [0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 2, 0],
                    [0, 0, 0, 0, 0]
                ],
                "output": [
                    [0, 1, 0, 2, 0],
                    [1, 1, 1, 2, 1],
                    [0, 1, 0, 2, 0],
                    [2, 2, 2, 2, 2],
                    [0, 1, 0, 2, 0]
                ]
            }
        ],
        "test": [
            {
                "input": [
                    [0, 0, 0, 0, 0],
                    [0, 0, 3, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 4, 0, 0, 0],
                    [0, 0, 0, 0, 0]
                ],
                "output": [
                    [0, 4, 3, 0, 0],
                    [4, 4, 3, 4, 4],
                    [0, 4, 3, 0, 0],
                    [3, 3, 3, 3, 3],
                    [0, 4, 3, 0, 0]
                ]
            }
        ]
    }
    
    # Task 6: Fill enclosed regions
    task6 = {
        "train": [
            {
                "input": [
                    [1, 1, 1, 1],
                    [1, 0, 0, 1],
                    [1, 0, 0, 1],
                    [1, 1, 1, 1]
                ],
                "output": [
                    [1, 1, 1, 1],
                    [1, 2, 2, 1],
                    [1, 2, 2, 1],
                    [1, 1, 1, 1]
                ]
            }
        ],
        "test": [
            {
                "input": [
                    [3, 3, 3, 3, 3],
                    [3, 0, 0, 0, 3],
                    [3, 0, 0, 0, 3],
                    [3, 3, 3, 3, 3]
                ],
                "output": [
                    [3, 3, 3, 3, 3],
                    [3, 4, 4, 4, 3],
                    [3, 4, 4, 4, 3],
                    [3, 3, 3, 3, 3]
                ]
            }
        ]
    }
    
    # Task 7: Rotation
    task7 = {
        "train": [
            {"input": [[1, 2], [3, 4]], "output": [[3, 1], [4, 2]]},
            {"input": [[5, 6], [7, 8]], "output": [[7, 5], [8, 6]]}
        ],
        "test": [
            {"input": [[9, 10], [11, 12]], "output": [[11, 9], [12, 10]]}
        ]
    }
    
    # Task 8: Pattern continuation
    task8 = {
        "train": [
            {
                "input": [[1, 2, 1], [2, 1, 2], [0, 0, 0]],
                "output": [[1, 2, 1], [2, 1, 2], [1, 2, 1]]
            }
        ],
        "test": [
            {
                "input": [[3, 4, 3], [4, 3, 4], [0, 0, 0]],
                "output": [[3, 4, 3], [4, 3, 4], [3, 4, 3]]
            }
        ]
    }
    
    # Task 9: Mirror symmetry
    task9 = {
        "train": [
            {"input": [[1, 2, 0], [3, 4, 0]], "output": [[1, 2, 2], [3, 4, 4]]}
        ],
        "test": [
            {"input": [[5, 6, 0], [7, 8, 0]], "output": [[5, 6, 6], [7, 8, 8]]}
        ]
    }
    
    # Task 10: Count and replace
    task10 = {
        "train": [
            {"input": [[1, 1, 2], [2, 1, 2]], "output": [[3, 3, 3], [3, 3, 3]]},
            {"input": [[4, 4, 5], [5, 4, 5]], "output": [[3, 3, 3], [3, 3, 3]]}
        ],
        "test": [
            {"input": [[6, 6, 7], [7, 6, 7]], "output": [[3, 3, 3], [3, 3, 3]]}
        ]
    }
    
    tasks = [task1, task2, task3, task4, task5, task6, task7, task8, task9, task10]
    
    # Save tasks
    for i, task in enumerate(tasks):
        filename = output_dir / f"test_task_{i:03d}.json"
        with open(filename, "w") as f:
            json.dump(task, f, indent=2)
    
    print(f"Created {len(tasks)} test tasks in {output_dir}")
    return output_dir


if __name__ == "__main__":
    dataset_dir = create_test_dataset()
    print(f"Test dataset ready at: {dataset_dir}")