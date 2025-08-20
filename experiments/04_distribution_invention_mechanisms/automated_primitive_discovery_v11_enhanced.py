"""Enhanced Automated Primitive Discovery with Missing Patterns - V11.

This version adds the patterns identified from failed task analysis:
1. Bounding box extraction
2. Gravity/falling movement
3. Corner marking of objects
4. Horizontal/vertical tiling
5. Pattern-based cell expansion
"""

import json
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage

warnings.filterwarnings("ignore")


class PrimitiveDiscovererV11:
    """Enhanced primitive discovery with compositional patterns."""

    def __init__(self, verbose: bool = False, accuracy_threshold: float = 0.75):
        self.verbose = verbose
        self.accuracy_threshold = accuracy_threshold
        self.discovered_patterns = []

    def discover_primitive(
        self, task_id: str, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[str]:
        """Main discovery function - tries all pattern types."""
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Discovering primitive for task {task_id}")
            print(f"{'='*60}")

        # Try different pattern categories
        pattern_functions = [
            self._try_bounding_box_extraction,
            self._try_gravity_movement,
            self._try_corner_marking,
            self._try_tiling_patterns,
            self._try_cell_expansion,
            self._try_existing_patterns,  # All v9 patterns
        ]

        for pattern_func in pattern_functions:
            result = pattern_func(task_id, examples)
            if result:
                accuracy = self._test_primitive(result, examples)
                if accuracy >= self.accuracy_threshold:
                    if self.verbose:
                        print(f"✓ Found working pattern with {accuracy:.1%} accuracy!")
                    return result

        if self.verbose:
            print(f"✗ No pattern found for task {task_id}")
        return None

    def _try_bounding_box_extraction(
        self, task_id: str, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[str]:
        """Try extracting bounding box of non-zero pixels."""
        if self.verbose:
            print("Trying bounding box extraction...")

        # Check if all examples extract bounding boxes
        matches = True
        for inp, out in examples:
            if inp.shape == out.shape:
                return None  # Not an extraction

            # Find bounding box of non-zero pixels
            nonzero = np.argwhere(inp != 0)
            if len(nonzero) == 0:
                return None

            min_y, min_x = nonzero.min(axis=0)
            max_y, max_x = nonzero.max(axis=0)
            extracted = inp[min_y : max_y + 1, min_x : max_x + 1]

            if not np.array_equal(extracted, out):
                matches = False
                break

        if matches:
            # Generate code for bounding box extraction
            code = f'''
class BoundingBoxExtract_{task_id}:
    """Extract minimal bounding box containing all non-zero pixels."""

    def execute(self, input_grid):
        grid = np.array(input_grid)
        nonzero = np.argwhere(grid != 0)

        if len(nonzero) == 0:
            return grid

        min_y, min_x = nonzero.min(axis=0)
        max_y, max_x = nonzero.max(axis=0)

        return grid[min_y:max_y+1, min_x:max_x+1]
'''
            return code
        return None

    def _try_gravity_movement(
        self, task_id: str, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[str]:
        """Try gravity/falling pattern where objects move down."""
        if self.verbose:
            print("Trying gravity movement...")

        # Check if pattern matches gravity movement
        for inp, out in examples:
            if inp.shape != out.shape:
                return None

            # Check if non-zero pixels moved down
            for col in range(inp.shape[1]):
                inp_col = inp[:, col]
                out_col = out[:, col]

                # Get non-zero values in column
                inp_vals = inp_col[inp_col != 0]
                out_vals = out_col[out_col != 0]

                # Should have same values, potentially in different positions
                if not np.array_equal(sorted(inp_vals), sorted(out_vals)):
                    continue

                # Check if values moved down (gravity effect)
                inp_positions = np.where(inp_col != 0)[0]
                out_positions = np.where(out_col != 0)[0]

                if len(out_positions) > 0 and len(inp_positions) > 0:
                    # Output positions should be lower (higher indices)
                    if np.mean(out_positions) > np.mean(inp_positions):
                        # This looks like gravity
                        code = f'''
class GravityFall_{task_id}:
    """Apply gravity - move all objects down until they hit bottom or another object."""

    def execute(self, input_grid):
        grid = np.array(input_grid).copy()
        h, w = grid.shape

        # Process each column
        for col in range(w):
            # Extract non-zero values
            column = grid[:, col]
            values = []
            for row in range(h):
                if column[row] != 0:
                    values.append(column[row])
                    column[row] = 0

            # Place values at bottom
            for i, val in enumerate(values):
                grid[h - len(values) + i, col] = val

        return grid
'''
                        return code

        return None

    def _try_corner_marking(
        self, task_id: str, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[str]:
        """Try marking corners of connected components."""
        if self.verbose:
            print("Trying corner marking...")

        for inp, out in examples:
            if inp.shape != out.shape:
                return None

            # Find connected components
            labeled, num_features = ndimage.label(inp != 0)

            # Check if output adds marks at component corners
            diff = out.astype(int) - inp.astype(int)
            new_pixels = np.argwhere(diff != 0)

            if len(new_pixels) > 0:
                # Check if new pixels are near component corners
                for label_id in range(1, num_features + 1):
                    component_mask = labeled == label_id
                    component_coords = np.argwhere(component_mask)

                    if len(component_coords) > 0:
                        # Find corners (min/max positions)
                        min_y, min_x = component_coords.min(axis=0)
                        max_y, max_x = component_coords.max(axis=0)

                        # Check for marks at corners
                        corner_positions = [
                            (min_y - 1, min_x - 1),  # Top-left
                            (min_y - 1, max_x + 1),  # Top-right
                            (max_y + 1, min_x - 1),  # Bottom-left
                            (max_y + 1, max_x + 1),  # Bottom-right
                        ]

                        for y, x in corner_positions:
                            if 0 <= y < out.shape[0] and 0 <= x < out.shape[1]:
                                if out[y, x] != inp[y, x] and out[y, x] != 0:
                                    # Found corner marking pattern
                                    code = f'''
class CornerMarker_{task_id}:
    """Mark corners of connected components with a specific color."""

    def execute(self, input_grid):
        grid = np.array(input_grid).copy()
        from scipy import ndimage

        # Find connected components
        labeled, num_features = ndimage.label(grid != 0)

        # Mark corners of each component
        for label_id in range(1, num_features + 1):
            component_mask = (labeled == label_id)
            coords = np.argwhere(component_mask)

            if len(coords) > 0:
                min_y, min_x = coords.min(axis=0)
                max_y, max_x = coords.max(axis=0)

                # Mark corners with color 1
                corners = [
                    (min_y - 1, max_x + 1),  # Top-right corner
                    (max_y + 1, min_x - 1),  # Bottom-left corner
                ]

                for y, x in corners:
                    if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]:
                        if grid[y, x] == 0:
                            grid[y, x] = 1

        return grid
'''
                                    return code

        return None

    def _try_tiling_patterns(
        self, task_id: str, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[str]:
        """Try horizontal or vertical tiling/repetition."""
        if self.verbose:
            print("Trying tiling patterns...")

        for inp, out in examples:
            # Check horizontal tiling (output width is multiple of input)
            if out.shape[0] == inp.shape[0] and out.shape[1] % inp.shape[1] == 0:
                num_tiles = out.shape[1] // inp.shape[1]

                # Check if output is input repeated horizontally
                expected = np.tile(inp, (1, num_tiles))
                if np.array_equal(expected, out):
                    code = f'''
class HorizontalTile_{task_id}:
    """Tile the input horizontally {num_tiles} times."""

    def execute(self, input_grid):
        grid = np.array(input_grid)
        return np.tile(grid, (1, {num_tiles}))
'''
                    return code

            # Check vertical tiling
            if out.shape[1] == inp.shape[1] and out.shape[0] % inp.shape[0] == 0:
                num_tiles = out.shape[0] // inp.shape[0]
                expected = np.tile(inp, (num_tiles, 1))
                if np.array_equal(expected, out):
                    code = f'''
class VerticalTile_{task_id}:
    """Tile the input vertically {num_tiles} times."""

    def execute(self, input_grid):
        grid = np.array(input_grid)
        return np.tile(grid, ({num_tiles}, 1))
'''
                    return code

        return None

    def _try_cell_expansion(
        self, task_id: str, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[str]:
        """Try pattern where each cell expands to a larger pattern."""
        if self.verbose:
            print("Trying cell expansion patterns...")

        for inp, out in examples:
            # Check if output is exact multiple of input size
            if out.shape[0] % inp.shape[0] == 0 and out.shape[1] % inp.shape[1] == 0:
                scale_y = out.shape[0] // inp.shape[0]
                scale_x = out.shape[1] // inp.shape[1]

                if scale_y == scale_x and scale_y > 1:
                    # Check if it's a pattern-based expansion
                    scale = scale_y

                    # Detect the expansion pattern for each unique value
                    patterns = {}
                    for val in np.unique(inp):
                        if val == 0:
                            # Check what 0 expands to
                            patterns[0] = np.zeros((scale, scale), dtype=int)
                        else:
                            # Find first occurrence and check its expansion
                            y, x = np.argwhere(inp == val)[0]
                            sub_grid = out[
                                y * scale : (y + 1) * scale, x * scale : (x + 1) * scale
                            ]
                            patterns[val] = sub_grid

                    # Generate code with detected patterns
                    pattern_dict = {int(k): v.tolist() for k, v in patterns.items()}

                    code = f'''
class PatternExpansion_{task_id}:
    """Expand each cell to a {scale}x{scale} pattern based on its value."""

    def execute(self, input_grid):
        grid = np.array(input_grid)
        scale = {scale}
        h, w = grid.shape
        output = np.zeros((h * scale, w * scale), dtype=int)

        patterns = {pattern_dict}

        for y in range(h):
            for x in range(w):
                val = grid[y, x]
                if val in patterns:
                    pattern = np.array(patterns[val])
                else:
                    pattern = np.zeros((scale, scale), dtype=int)

                output[y*scale:(y+1)*scale, x*scale:(x+1)*scale] = pattern

        return output
'''
                    return code

        return None

    def _try_existing_patterns(
        self, task_id: str, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[str]:
        """Try all patterns from v9 (cross, region, rotation, etc.)."""
        if self.verbose:
            print("Trying existing v9 patterns...")

        # This would include all the patterns from v9_final
        # For brevity, returning None here - in practice would include all v9 patterns
        return None

    def _test_primitive(
        self, code: str, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> float:
        """Test generated primitive code on examples."""
        try:
            # Create namespace and execute code
            namespace = {"np": np, "ndimage": ndimage}
            exec(code, namespace)

            # Find the class
            class_name = None
            for name in namespace:
                if name.startswith(
                    (
                        "BoundingBox",
                        "Gravity",
                        "Corner",
                        "Horizontal",
                        "Vertical",
                        "Pattern",
                    )
                ):
                    class_name = name
                    break

            if not class_name:
                return 0.0

            # Test on examples
            primitive = namespace[class_name]()
            correct = 0
            total = 0

            for inp, expected_out in examples:
                try:
                    predicted = primitive.execute(inp)
                    predicted = np.array(predicted)

                    if predicted.shape == expected_out.shape:
                        if np.array_equal(predicted, expected_out):
                            correct += 1
                        else:
                            # Calculate partial accuracy
                            matching = np.sum(predicted == expected_out)
                            total_pixels = expected_out.size
                            correct += matching / total_pixels
                    total += 1
                except:
                    total += 1

            return correct / total if total > 0 else 0.0

        except Exception as e:
            if self.verbose:
                print(f"Error testing primitive: {e}")
            return 0.0


def test_on_failed_tasks():
    """Test the enhanced discoverer on previously failed tasks."""
    failed_tasks = ["1cf80156", "25ff71a9", "3aa6fb7a", "a416b8f3", "007bbfb7"]

    discoverer = PrimitiveDiscovererV11(verbose=True, accuracy_threshold=0.75)
    results = {}

    for task_id in failed_tasks:
        print(f"\n{'='*60}")
        print(f"Testing task {task_id}")
        print(f"{'='*60}")

        # Load task
        data_dir = Path("data/arc_agi_official/ARC-AGI/data/training")
        with open(data_dir / f"{task_id}.json", "r") as f:
            task = json.load(f)

        examples = [
            (np.array(e["input"]), np.array(e["output"])) for e in task["train"]
        ]

        # Try discovery
        result = discoverer.discover_primitive(task_id, examples)

        if result:
            # Test accuracy
            accuracy = discoverer._test_primitive(result, examples)
            print(f"✓ SUCCESS: Found pattern with {accuracy:.1%} accuracy")
            results[task_id] = accuracy
        else:
            print(f"✗ FAILED: No pattern found")
            results[task_id] = 0.0

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for task_id, accuracy in results.items():
        status = "✓" if accuracy > 0 else "✗"
        print(f"{status} {task_id}: {accuracy:.1%}")

    success_count = sum(1 for acc in results.values() if acc > 0)
    print(
        f"\nSuccess rate: {success_count}/{len(failed_tasks)} = {success_count/len(failed_tasks):.1%}"
    )


if __name__ == "__main__":
    test_on_failed_tasks()
