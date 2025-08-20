"""Debug specific failed tasks to understand exact transformations needed."""

import json
from pathlib import Path

import numpy as np


def load_and_analyze_task(task_id: str):
    """Load and deeply analyze a specific task."""
    data_dir = Path("data/arc_agi_official/ARC-AGI/data/training")
    with open(data_dir / f"{task_id}.json", "r") as f:
        task = json.load(f)

    print(f"\n{'='*60}")
    print(f"Task {task_id} - Detailed Analysis")
    print("=" * 60)

    for i, example in enumerate(task["train"][:3]):
        inp = np.array(example["input"])
        out = np.array(example["output"])

        print(f"\nExample {i+1}:")
        print(f"Input ({inp.shape}):")
        print(inp)
        print(f"\nOutput ({out.shape}):")
        print(out)

        # Analyze the transformation
        print("\nTransformation analysis:")

        # For task 007bbfb7 - looks like 3x3 expansion
        if task_id == "007bbfb7" and inp.shape == (3, 3) and out.shape == (9, 9):
            print("This appears to be a 3x3 expansion pattern")
            # Check if each cell is expanded to 3x3
            for y in range(3):
                for x in range(3):
                    cell_value = inp[y, x]
                    sub_grid = out[y * 3 : (y + 1) * 3, x * 3 : (x + 1) * 3]
                    print(f"  Cell [{y},{x}] = {cell_value} -> subgrid:")
                    print(f"    {sub_grid}")

        # For task a416b8f3 - horizontal doubling
        elif task_id == "a416b8f3" and out.shape[1] == inp.shape[1] * 2:
            print("This appears to be horizontal doubling/mirroring")
            left_half = out[:, : inp.shape[1]]
            right_half = out[:, inp.shape[1] :]
            print(f"  Left half matches input: {np.array_equal(left_half, inp)}")
            print(f"  Right half matches input: {np.array_equal(right_half, inp)}")
            print(
                f"  Right half is flipped: {np.array_equal(right_half, np.fliplr(inp))}"
            )

        # For task 1cf80156 - extraction
        elif task_id == "1cf80156" and inp.shape != out.shape:
            print("This appears to be object extraction")
            # Find non-zero regions
            inp_nonzero = np.argwhere(inp != 0)
            if len(inp_nonzero) > 0:
                min_y, min_x = inp_nonzero.min(axis=0)
                max_y, max_x = inp_nonzero.max(axis=0)
                print(
                    f"  Non-zero bounding box: [{min_y}:{max_y+1}, {min_x}:{max_x+1}]"
                )
                extracted = inp[min_y : max_y + 1, min_x : max_x + 1]
                print(f"  Extracted shape: {extracted.shape}")
                print(f"  Output shape: {out.shape}")
                print(f"  Extracted matches output: {np.array_equal(extracted, out)}")

        # For task 25ff71a9 - rotation/flip
        elif task_id == "25ff71a9" and inp.shape == out.shape:
            print("This appears to be rotation or flipping")
            print(f"  90° rotation: {np.array_equal(out, np.rot90(inp))}")
            print(f"  180° rotation: {np.array_equal(out, np.rot90(inp, 2))}")
            print(f"  270° rotation: {np.array_equal(out, np.rot90(inp, 3))}")
            print(f"  Horizontal flip: {np.array_equal(out, np.fliplr(inp))}")
            print(f"  Vertical flip: {np.array_equal(out, np.flipud(inp))}")
            print(f"  Transpose: {np.array_equal(out, inp.T)}")

        # For task 3aa6fb7a - color addition
        elif task_id == "3aa6fb7a":
            print("This appears to involve adding new colors")
            diff = out - inp
            new_pixels = np.argwhere(diff != 0)
            if len(new_pixels) > 0:
                print(f"  {len(new_pixels)} pixels changed")
                print(f"  New pixel locations: {new_pixels[:5].tolist()}...")
                print(
                    f"  Colors added at those locations: {[out[y,x] for y,x in new_pixels[:5]]}"
                )


# Analyze each failed task
failed_tasks = ["1cf80156", "25ff71a9", "3aa6fb7a", "a416b8f3", "007bbfb7"]

for task_id in failed_tasks:
    load_and_analyze_task(task_id)
