#!/usr/bin/env python3
"""Automated primitive discovery system for ARC tasks.

This module implements the key breakthrough: automatically discovering
task-specific primitives by analyzing failed synthesis attempts and
extracting transformation patterns.
"""

from utils.imports import setup_project_paths

setup_project_paths()

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from compositional_dsl import ExecutionContext, Primitive
from scipy import ndimage
from tqdm import tqdm


class PrimitiveDiscoverer:
    """Discovers task-specific primitives from examples."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.discovered_primitives = {}

    def discover_primitive(
        self, task_id: str, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[Primitive]:
        """Discover a task-specific primitive from input-output examples.

        Args:
            task_id: Task identifier
            examples: List of (input, output) grid pairs

        Returns:
            A new Primitive if pattern found, None otherwise
        """
        if self.verbose:
            print(f"\nAnalyzing task {task_id}...")

        # Extract transformation patterns
        patterns = self._extract_patterns(examples)

        if not patterns:
            return None

        # Find most consistent pattern
        best_pattern = self._find_best_pattern(patterns, examples)

        if best_pattern is None:
            return None

        # Generate primitive code
        primitive_code = self._synthesize_primitive(best_pattern, task_id)

        if primitive_code:
            # Test the primitive
            if self._test_primitive(primitive_code, examples):
                if self.verbose:
                    print(f"✅ Discovered primitive for {task_id}!")
                self.discovered_primitives[task_id] = primitive_code
                return primitive_code

        return None

    def _extract_patterns(self, examples):
        """Extract potential transformation patterns."""
        patterns = []

        for inp, out in examples:
            # Color mapping patterns
            color_map = self._analyze_color_mapping(inp, out)
            if color_map:
                patterns.append({"type": "color_map", "data": color_map})

            # Spatial patterns
            spatial = self._analyze_spatial_pattern(inp, out)
            if spatial:
                patterns.append({"type": "spatial", "data": spatial})

            # Object-based patterns
            objects = self._analyze_object_pattern(inp, out)
            if objects:
                patterns.append({"type": "objects", "data": objects})

            # Conditional patterns
            conditional = self._analyze_conditional_pattern(inp, out)
            if conditional:
                patterns.append({"type": "conditional", "data": conditional})

        return patterns

    def _analyze_color_mapping(self, inp, out):
        """Analyze color transformation patterns."""
        # Check if it's a simple color remapping
        set(inp.flatten())
        set(out.flatten())

        if inp.shape != out.shape:
            return None

        # Build color mapping
        color_map = {}
        for i in range(inp.shape[0]):
            for j in range(inp.shape[1]):
                if inp[i, j] != 0:
                    if inp[i, j] not in color_map:
                        color_map[inp[i, j]] = out[i, j]
                    elif color_map[inp[i, j]] != out[i, j]:
                        # Not a consistent mapping
                        return None

        if color_map and len(color_map) > 0:
            return color_map

        return None

    def _analyze_spatial_pattern(self, inp, out):
        """Analyze spatial transformation patterns."""
        if inp.shape != out.shape:
            return None

        # Check for cross pattern (like in ae3edfdc)
        pattern = self._detect_cross_pattern(inp, out)
        if pattern:
            return {"pattern": "cross", "details": pattern}

        # Check for line patterns
        pattern = self._detect_line_pattern(inp, out)
        if pattern:
            return {"pattern": "lines", "details": pattern}

        # Check for region filling
        pattern = self._detect_region_fill(inp, out)
        if pattern:
            return {"pattern": "region_fill", "details": pattern}

        return None

    def _detect_cross_pattern(self, inp, out):
        """Detect cross pattern formation."""
        # Find positions where output has cross patterns not in input
        diff = (out != 0) & (inp == 0)

        crosses = []
        for i in range(1, inp.shape[0] - 1):
            for j in range(1, inp.shape[1] - 1):
                # Check if this forms a cross center
                if (
                    diff[i, j]
                    and diff[i - 1, j]
                    and diff[i + 1, j]
                    and diff[i, j - 1]
                    and diff[i, j + 1]
                ):
                    # Found a cross pattern
                    crosses.append((i, j))

        if crosses:
            # Analyze what triggers cross formation
            triggers = []
            for ci, cj in crosses:
                # Check original input for patterns
                if inp[ci, cj] != 0:
                    triggers.append({"type": "center_colored", "pos": (ci, cj)})
                # Check for aligned colored pixels
                elif np.any(inp[ci, :] != 0) and np.any(inp[:, cj] != 0):
                    triggers.append({"type": "aligned_colors", "pos": (ci, cj)})

            return {"crosses": crosses, "triggers": triggers}

        return None

    def _detect_line_pattern(self, inp, out):
        """Detect line drawing patterns."""
        # Find new lines in output
        diff = (out != 0) & (inp == 0)

        # Check for horizontal lines
        for i in range(inp.shape[0]):
            if np.sum(diff[i, :]) > inp.shape[1] * 0.7:  # Most of row is new
                # Found horizontal line
                return {"type": "horizontal", "positions": [i]}

        # Check for vertical lines
        for j in range(inp.shape[1]):
            if np.sum(diff[:, j]) > inp.shape[0] * 0.7:  # Most of column is new
                # Found vertical line
                return {"type": "vertical", "positions": [j]}

        return None

    def _detect_region_fill(self, inp, out):
        """Detect region filling patterns."""
        # Check if output fills regions defined in input
        diff = (out != inp) & (out != 0)

        if np.sum(diff) > 0:
            # Find what regions were filled
            filled_regions = []

            # Check for enclosed regions
            for color in np.unique(inp):
                if color != 0:
                    mask = inp == color
                    # Check if this color forms a boundary
                    filled = ndimage.binary_fill_holes(mask)
                    interior = filled & ~mask

                    if np.any(interior):
                        # This color encloses a region
                        filled_regions.append(
                            {"boundary_color": color, "interior": interior}
                        )

            if filled_regions:
                return {"type": "enclosed", "regions": filled_regions}

        return None

    def _analyze_object_pattern(self, inp, out):
        """Analyze object-based transformation patterns."""
        # Find objects in input
        input_objects = self._extract_objects(inp)
        output_objects = self._extract_objects(out)

        if not input_objects:
            return None

        # Check for object counting/sorting
        if len(output_objects) != len(input_objects):
            # Objects were added/removed
            return {
                "type": "count_based",
                "input_count": len(input_objects),
                "output_count": len(output_objects),
            }

        # Check for object rearrangement
        if self._objects_rearranged(input_objects, output_objects):
            return {"type": "rearrangement", "objects": input_objects}

        return None

    def _extract_objects(self, grid):
        """Extract distinct objects from grid."""
        objects = []

        for color in np.unique(grid):
            if color != 0:
                mask = (grid == color).astype(int)
                labeled, num = ndimage.label(mask)

                for i in range(1, num + 1):
                    obj_mask = labeled == i
                    positions = np.argwhere(obj_mask)

                    if len(positions) > 0:
                        objects.append(
                            {
                                "color": color,
                                "positions": positions,
                                "size": len(positions),
                                "center": positions.mean(axis=0),
                            }
                        )

        return objects

    def _objects_rearranged(self, objects1, objects2):
        """Check if objects were rearranged."""
        if len(objects1) != len(objects2):
            return False

        # Check if centers moved
        for obj1, obj2 in zip(objects1, objects2):
            if not np.allclose(obj1["center"], obj2["center"]):
                return True

        return False

    def _analyze_conditional_pattern(self, inp, out):
        """Analyze conditional transformation patterns."""
        # Check for patterns based on neighbor conditions
        changes = []

        for i in range(inp.shape[0]):
            for j in range(inp.shape[1]):
                if inp[i, j] != out[i, j]:
                    # Analyze why this pixel changed
                    neighbors = self._get_neighbors(inp, i, j)
                    changes.append(
                        {
                            "pos": (i, j),
                            "from": inp[i, j],
                            "to": out[i, j],
                            "neighbors": neighbors,
                        }
                    )

        if changes:
            # Find common patterns in changes
            pattern = self._find_conditional_rule(changes)
            if pattern:
                return pattern

        return None

    def _get_neighbors(self, grid, i, j):
        """Get neighbor values for a position."""
        neighbors = {}
        h, w = grid.shape

        if i > 0:
            neighbors["top"] = grid[i - 1, j]
        if i < h - 1:
            neighbors["bottom"] = grid[i + 1, j]
        if j > 0:
            neighbors["left"] = grid[i, j - 1]
        if j < w - 1:
            neighbors["right"] = grid[i, j + 1]

        return neighbors

    def _find_conditional_rule(self, changes):
        """Find common conditional rule in changes."""
        # Look for patterns like "if has 2+ colored neighbors, fill with color X"
        rules = []

        for change in changes:
            colored_neighbors = sum(1 for v in change["neighbors"].values() if v != 0)

            if colored_neighbors >= 2 and change["from"] == 0:
                # Filling based on neighbor count
                rules.append(
                    {
                        "type": "neighbor_count_fill",
                        "min_neighbors": colored_neighbors,
                        "fill_color": change["to"],
                    }
                )

        if rules:
            # Find most common rule
            return rules[0]  # Simplified - could do voting

        return None

    def _find_best_pattern(self, patterns, examples):
        """Find the pattern that best explains all examples."""
        if not patterns:
            return None

        # Score each pattern by consistency
        pattern_scores = {}

        for pattern in patterns:
            score = 0
            pattern_key = str(pattern)

            # Check if pattern appears in all examples
            for inp, out in examples:
                if self._pattern_matches(pattern, inp, out):
                    score += 1

            if pattern_key not in pattern_scores:
                pattern_scores[pattern_key] = (score, pattern)

        # Return pattern with highest score
        if pattern_scores:
            best_key = max(pattern_scores, key=lambda k: pattern_scores[k][0])
            return pattern_scores[best_key][1]

        return None

    def _pattern_matches(self, pattern, inp, out):
        """Check if pattern matches this example."""
        if pattern["type"] == "color_map":
            # Check color mapping
            for i in range(inp.shape[0]):
                for j in range(inp.shape[1]):
                    if inp[i, j] in pattern["data"]:
                        if out[i, j] != pattern["data"][inp[i, j]]:
                            return False
            return True

        elif pattern["type"] == "spatial":
            # Check spatial pattern
            if pattern["data"]["pattern"] == "cross":
                # Verify cross patterns exist
                return self._detect_cross_pattern(inp, out) is not None

        # Add more pattern type checks as needed
        return False

    def _synthesize_primitive(self, pattern, task_id):
        """Generate primitive code from pattern."""
        if pattern["type"] == "spatial" and pattern["data"]["pattern"] == "cross":
            # Generate cross pattern primitive
            return self._generate_cross_primitive(pattern["data"], task_id)

        elif pattern["type"] == "color_map":
            # Generate color mapping primitive
            return self._generate_color_map_primitive(pattern["data"], task_id)

        elif pattern["type"] == "conditional":
            # Generate conditional fill primitive
            return self._generate_conditional_primitive(pattern, task_id)

        return None

    def _generate_cross_primitive(self, pattern_data, task_id):
        """Generate a cross pattern primitive."""
        class_name = f"CrossPattern_{task_id.replace('-', '_')}"

        code = f'''
class {class_name}(Primitive):
    """Auto-discovered cross pattern for {task_id}."""

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        result = context.copy()
        grid = result.current_grid.copy()

        # Find positions to form crosses
        for i in range(1, grid.shape[0] - 1):
            for j in range(1, grid.shape[1] - 1):
                # Check trigger condition
                if self._should_form_cross(context.input_grid, i, j):
                    # Form cross
                    color = self._get_cross_color(context.input_grid, i, j)
                    grid[i-1, j] = color
                    grid[i+1, j] = color
                    grid[i, j-1] = color
                    grid[i, j+1] = color
                    grid[i, j] = color

        result.current_grid = grid
        return result

    def _should_form_cross(self, grid, i, j):
        # Check if position should have a cross
        # Based on pattern: aligned colored pixels
        return (np.any(grid[i, :] != 0) and np.any(grid[:, j] != 0) and
                grid[i, j] == 0)

    def _get_cross_color(self, grid, i, j):
        # Get color for the cross
        row_colors = grid[i, grid[i, :] != 0]
        col_colors = grid[grid[:, j] != 0, j]

        if len(row_colors) > 0:
            return row_colors[0]
        elif len(col_colors) > 0:
            return col_colors[0]
        return 2  # Default

    def __str__(self):
        return "{class_name}()"
'''

        # Create the class dynamically
        namespace = {
            "Primitive": Primitive,
            "ExecutionContext": ExecutionContext,
            "np": np,
        }
        exec(code, namespace)

        return namespace[class_name]

    def _generate_color_map_primitive(self, color_map, task_id):
        """Generate a color mapping primitive."""
        class_name = f"ColorMap_{task_id.replace('-', '_')}"
        map_str = str(color_map)

        code = f'''
class {class_name}(Primitive):
    """Auto-discovered color mapping for {task_id}."""

    def __init__(self):
        self.color_map = {map_str}

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        result = context.copy()
        grid = result.current_grid.copy()

        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] in self.color_map:
                    grid[i, j] = self.color_map[grid[i, j]]

        result.current_grid = grid
        return result

    def __str__(self):
        return "{class_name}()"
'''

        namespace = {"Primitive": Primitive, "ExecutionContext": ExecutionContext}
        exec(code, namespace)

        return namespace[class_name]

    def _generate_conditional_primitive(self, pattern, task_id):
        """Generate a conditional fill primitive."""
        class_name = f"ConditionalFill_{task_id.replace('-', '_')}"

        if pattern.get("type") == "neighbor_count_fill":
            min_neighbors = pattern.get("min_neighbors", 2)
            fill_color = pattern.get("fill_color", 2)

            code = f'''
class {class_name}(Primitive):
    """Auto-discovered conditional fill for {task_id}."""

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        result = context.copy()
        grid = result.current_grid.copy()

        for i in range(1, grid.shape[0] - 1):
            for j in range(1, grid.shape[1] - 1):
                if grid[i, j] == 0:
                    # Count colored neighbors
                    neighbors = [
                        grid[i-1, j], grid[i+1, j],
                        grid[i, j-1], grid[i, j+1]
                    ]
                    colored = sum(1 for n in neighbors if n != 0)

                    if colored >= {min_neighbors}:
                        grid[i, j] = {fill_color}

        result.current_grid = grid
        return result

    def __str__(self):
        return "{class_name}()"
'''

            namespace = {"Primitive": Primitive, "ExecutionContext": ExecutionContext}
            exec(code, namespace)

            return namespace[class_name]

        return None

    def _test_primitive(self, primitive_class, examples):
        """Test if primitive produces correct outputs."""
        try:
            primitive = primitive_class()

            for inp, expected_out in examples:
                context = ExecutionContext(input_grid=inp, current_grid=inp.copy())
                result = primitive.execute(context)

                if not np.array_equal(result.current_grid, expected_out):
                    return False

            return True

        except Exception as e:
            if self.verbose:
                print(f"  Error testing primitive: {e}")
            return False

    def save_discovered_primitives(self, output_path: Path):
        """Save discovered primitives to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        primitives_info = []
        for task_id, primitive_class in self.discovered_primitives.items():
            primitives_info.append(
                {
                    "task_id": task_id,
                    "class_name": primitive_class.__name__,
                    "doc": primitive_class.__doc__,
                }
            )

        with open(output_path, "w") as f:
            json.dump(primitives_info, f, indent=2)

        print(f"\nSaved {len(primitives_info)} discovered primitives to {output_path}")


def test_primitive_discovery_on_failed_tasks(num_tasks: int = 5):
    """Test automated primitive discovery on failed tasks."""
    print("Testing Automated Primitive Discovery")
    print("=" * 60)

    # Load failed task IDs
    failed_tasks_file = Path("failed_tasks_detailed.json")
    if not failed_tasks_file.exists():
        print(f"Error: {failed_tasks_file} not found")
        return

    with open(failed_tasks_file, "r") as f:
        failed_data = json.load(f)

    failed_task_ids = failed_data["failed_task_ids"][:num_tasks]
    print(f"\nTesting on {len(failed_task_ids)} failed tasks")

    # Initialize discoverer
    discoverer = PrimitiveDiscoverer(verbose=True)

    # Discover primitives for each task
    discovered_count = 0
    for task_id in tqdm(failed_task_ids, desc="Discovering primitives"):
        # Load task examples
        examples = load_task_examples(task_id)

        if examples:
            primitive = discoverer.discover_primitive(task_id, examples)
            if primitive:
                discovered_count += 1
                print(f"  ✅ Discovered primitive for {task_id}")

                # Test the primitive
                test_accuracy = test_primitive_accuracy(primitive, examples)
                print(f"     Accuracy: {test_accuracy:.1%}")

    # Summary
    print("\n" + "=" * 60)
    print(f"Discovery Results:")
    print(f"  Tasks analyzed: {len(failed_task_ids)}")
    print(f"  Primitives discovered: {discovered_count}")
    print(f"  Discovery rate: {discovered_count/len(failed_task_ids)*100:.1f}%")

    # Save discovered primitives
    if discovered_count > 0:
        discoverer.save_discovered_primitives(
            Path("discovered_primitives_catalog.json")
        )

    return discoverer


def load_task_examples(task_id: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Load examples for a specific task."""
    # Try training directory first
    data_dir = Path("data/arc_agi_official/ARC-AGI/data/training")
    task_file = data_dir / f"{task_id}.json"

    if not task_file.exists():
        # Try evaluation directory
        data_dir = Path("data/arc_agi_official/ARC-AGI/data/evaluation")
        task_file = data_dir / f"{task_id}.json"

    if not task_file.exists():
        return []

    with open(task_file, "r") as f:
        task = json.load(f)

    examples = [(np.array(ex["input"]), np.array(ex["output"])) for ex in task["train"]]

    return examples


def test_primitive_accuracy(primitive_class, examples):
    """Test primitive accuracy on examples."""
    try:
        primitive = primitive_class()
        correct = 0

        for inp, expected_out in examples:
            context = ExecutionContext(input_grid=inp, current_grid=inp.copy())
            result = primitive.execute(context)

            if np.array_equal(result.current_grid, expected_out):
                correct += 1

        return correct / len(examples)

    except Exception:
        return 0.0


def main():
    """Main function to run primitive discovery."""
    import argparse

    parser = argparse.ArgumentParser(description="Automated Primitive Discovery")
    parser.add_argument(
        "--num_tasks",
        type=int,
        default=10,
        help="Number of failed tasks to analyze",
    )
    parser.add_argument(
        "--task_id",
        type=str,
        help="Specific task ID to analyze",
    )

    args = parser.parse_args()

    if args.task_id:
        # Analyze specific task
        print(f"Analyzing task {args.task_id}")
        examples = load_task_examples(args.task_id)

        if not examples:
            print(f"Could not load examples for {args.task_id}")
            return

        discoverer = PrimitiveDiscoverer(verbose=True)
        primitive = discoverer.discover_primitive(args.task_id, examples)

        if primitive:
            print(f"\n✅ Successfully discovered primitive!")
            accuracy = test_primitive_accuracy(primitive, examples)
            print(f"Accuracy on training examples: {accuracy:.1%}")
        else:
            print(f"\n❌ Could not discover primitive for this task")

    else:
        # Test on multiple failed tasks
        test_primitive_discovery_on_failed_tasks(args.num_tasks)


if __name__ == "__main__":
    main()
