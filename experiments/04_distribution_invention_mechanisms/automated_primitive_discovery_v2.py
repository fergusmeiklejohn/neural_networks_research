#!/usr/bin/env python3
"""Automated primitive discovery system for ARC tasks - Version 2.

Fixed version that properly handles cross pattern detection and synthesis.
"""

from utils.imports import setup_project_paths

setup_project_paths()

from typing import List, Optional, Tuple

import numpy as np
from compositional_dsl import ExecutionContext, Primitive
from scipy import ndimage


class PrimitiveDiscovererV2:
    """Discovers task-specific primitives from examples - improved version."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.discovered_primitives = {}

    def discover_primitive(
        self, task_id: str, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[str]:
        """Discover a task-specific primitive from input-output examples.

        Args:
            task_id: Task identifier
            examples: List of (input, output) grid pairs

        Returns:
            Primitive code string if pattern found, None otherwise
        """
        if self.verbose:
            print(f"\nAnalyzing task {task_id}...")

        # Extract transformation patterns
        patterns = self._extract_patterns(examples)

        if not patterns:
            if self.verbose:
                print("No patterns found")
            return None

        # Find most consistent pattern
        best_pattern = self._find_best_pattern(patterns, examples)

        if best_pattern is None:
            if self.verbose:
                print("No consistent pattern found")
            return None

        if self.verbose:
            print(f"Best pattern: {best_pattern['type']}")

        # Generate primitive code
        primitive_code = self._synthesize_primitive(best_pattern, task_id)

        if primitive_code:
            # Test the primitive
            if self._test_primitive(primitive_code, examples, task_id):
                if self.verbose:
                    print(f"✅ Discovered primitive for {task_id}!")
                self.discovered_primitives[task_id] = primitive_code
                return primitive_code
            else:
                if self.verbose:
                    print("Generated primitive failed testing")

        return None

    def _extract_patterns(self, examples):
        """Extract potential transformation patterns."""
        patterns = []

        for inp, out in examples:
            # Spatial patterns (including crosses)
            spatial = self._analyze_spatial_pattern(inp, out)
            if spatial:
                patterns.append({"type": "spatial", "data": spatial})

            # Color mapping patterns
            color_map = self._analyze_color_mapping(inp, out)
            if color_map:
                patterns.append({"type": "color_map", "data": color_map})

            # Object-based patterns
            objects = self._analyze_object_pattern(inp, out)
            if objects:
                patterns.append({"type": "objects", "data": objects})

            # Conditional patterns
            conditional = self._analyze_conditional_pattern(inp, out)
            if conditional:
                patterns.append({"type": "conditional", "data": conditional})

        return patterns

    def _analyze_spatial_pattern(self, inp, out):
        """Analyze spatial transformation patterns - FIXED version."""
        # Check for cross patterns
        crosses = self._detect_cross_pattern(inp, out)
        if crosses:
            return {"pattern": "cross", "details": crosses}

        # Check for line patterns
        lines = self._detect_line_pattern(inp, out)
        if lines:
            return {"pattern": "line", "details": lines}

        # Check for region fills
        regions = self._detect_region_fill(inp, out)
        if regions:
            return {"pattern": "region", "details": regions}

        return None

    def _detect_cross_pattern(self, inp, out):
        """Detect cross pattern formations - with fuzzy matching."""
        h, w = inp.shape
        crosses = []

        # Look for positions where crosses formed
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                # Check if this is a cross center in output
                center_val = out[i, j]

                # Count how many arms changed (relaxed criteria - 3+ arms)
                arms_changed = 0
                arm_colors = []

                if i > 0 and out[i - 1, j] != inp[i - 1, j]:
                    arms_changed += 1
                    arm_colors.append(out[i - 1, j])
                if i < h - 1 and out[i + 1, j] != inp[i + 1, j]:
                    arms_changed += 1
                    arm_colors.append(out[i + 1, j])
                if j > 0 and out[i, j - 1] != inp[i, j - 1]:
                    arms_changed += 1
                    arm_colors.append(out[i, j - 1])
                if j < w - 1 and out[i, j + 1] != inp[i, j + 1]:
                    arms_changed += 1
                    arm_colors.append(out[i, j + 1])

                # Consider it a cross if 3+ arms changed to the same color
                if arms_changed >= 3 and center_val != 0:
                    # Check if arms are the same color
                    if arm_colors and all(
                        c == arm_colors[0] for c in arm_colors if c != 0
                    ):
                        cross_info = {
                            "center": (i, j),
                            "center_color": int(center_val),
                            "cross_color": int(arm_colors[0]) if arm_colors else 0,
                            "arms_changed": arms_changed,
                        }

                        # Analyze what triggered this cross
                        # Look for markers on the same row/column
                        horizontal_markers = []
                        vertical_markers = []

                        for jj in range(w):
                            if jj != j and inp[i, jj] != 0:
                                horizontal_markers.append(
                                    {"pos": jj, "color": int(inp[i, jj])}
                                )

                        for ii in range(h):
                            if ii != i and inp[ii, j] != 0:
                                vertical_markers.append(
                                    {"pos": ii, "color": int(inp[ii, j])}
                                )

                        cross_info["h_markers"] = horizontal_markers
                        cross_info["v_markers"] = vertical_markers

                        crosses.append(cross_info)

        return crosses if crosses else None

    def _detect_line_pattern(self, inp, out):
        """Detect line drawing patterns."""
        diff = (out != 0) & (inp == 0)

        lines = []
        # Check for horizontal lines
        for i in range(inp.shape[0]):
            if np.sum(diff[i, :]) > inp.shape[1] * 0.5:
                lines.append({"type": "horizontal", "row": i})

        # Check for vertical lines
        for j in range(inp.shape[1]):
            if np.sum(diff[:, j]) > inp.shape[0] * 0.5:
                lines.append({"type": "vertical", "col": j})

        return lines if lines else None

    def _detect_region_fill(self, inp, out):
        """Detect region filling patterns."""
        diff = (out != inp) & (out != 0)

        if np.sum(diff) > 0:
            filled_regions = []

            for color in np.unique(inp):
                if color != 0:
                    mask = inp == color
                    filled = ndimage.binary_fill_holes(mask)
                    interior = filled & ~mask

                    if np.any(interior):
                        filled_regions.append(
                            {"boundary_color": int(color), "filled": True}
                        )

            return filled_regions if filled_regions else None

        return None

    def _analyze_color_mapping(self, inp, out):
        """Analyze color transformation patterns."""
        # Don't consider identity mappings or trivial changes
        if np.array_equal(inp, out):
            return None

        inp_colors = set(inp.flatten())
        out_colors = set(out.flatten())

        # Only consider if colors are preserved (no new colors)
        if not out_colors.issubset(inp_colors):
            return None

        # Check for consistent color remapping
        color_map = {}
        for color in inp_colors:
            if color != 0:
                inp_mask = inp == color
                out_vals = out[inp_mask]
                unique_out = np.unique(out_vals)

                # Only valid if this color maps to exactly one other color
                if len(unique_out) == 1:
                    color_map[int(color)] = int(unique_out[0])
                else:
                    # Not a simple color mapping
                    return None

        # Only return if there's actual remapping
        if color_map and any(k != v for k, v in color_map.items()):
            return color_map

        return None

    def _analyze_object_pattern(self, inp, out):
        """Analyze object-based transformation patterns."""
        input_objects = self._extract_objects(inp)
        output_objects = self._extract_objects(out)

        if not input_objects:
            return None

        # Check for object counting
        if len(output_objects) != len(input_objects):
            return {
                "type": "count_based",
                "input_count": len(input_objects),
                "output_count": len(output_objects),
            }

        # Check for sorting by size
        input_sizes = sorted([obj["size"] for obj in input_objects])
        output_sizes = [obj["size"] for obj in output_objects]
        if input_sizes == output_sizes and self._objects_rearranged(
            input_objects, output_objects
        ):
            return {"type": "sort_by_size", "objects": input_objects}

        # Check for object movement
        if self._detect_uniform_movement(input_objects, output_objects):
            return {"type": "movement", "objects": input_objects}

        # Check for duplication
        if len(output_objects) > len(input_objects):
            return {
                "type": "duplication",
                "factor": len(output_objects) // len(input_objects),
            }

        return None

    def _extract_objects(self, grid):
        """Extract connected components as objects."""
        objects = []
        for color in np.unique(grid):
            if color != 0:
                mask = grid == color
                labeled, num = ndimage.label(mask)
                for i in range(1, num + 1):
                    obj_mask = labeled == i
                    objects.append(
                        {
                            "color": int(color),
                            "mask": obj_mask,
                            "size": np.sum(obj_mask),
                        }
                    )
        return objects

    def _objects_rearranged(self, input_objects, output_objects):
        """Check if objects were rearranged."""
        if len(input_objects) != len(output_objects):
            return False

        # Simple check - if sizes match but positions differ
        return True  # Simplified for now

    def _detect_uniform_movement(self, input_objects, output_objects):
        """Check if all objects moved uniformly."""
        if len(input_objects) != len(output_objects):
            return False

        # Simplified check
        return False

    def _analyze_conditional_pattern(self, inp, out):
        """Analyze conditional transformation patterns."""
        diff_mask = inp != out

        if not np.any(diff_mask):
            return None

        # Check for neighbor-based conditions
        for i in range(inp.shape[0]):
            for j in range(inp.shape[1]):
                if diff_mask[i, j]:
                    neighbors = self._get_neighbors(inp, i, j)
                    non_zero = [n for n in neighbors if n != 0]

                    if len(non_zero) >= 2:
                        return {
                            "type": "neighbor_based",
                            "min_neighbors": 2,
                            "fill_color": int(out[i, j]),
                        }

        return None

    def _get_neighbors(self, grid, i, j):
        """Get neighboring values."""
        neighbors = []
        h, w = grid.shape

        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w:
                    neighbors.append(grid[ni, nj])

        return neighbors

    def _find_best_pattern(self, patterns, examples):
        """Find the pattern that best explains all examples."""
        if not patterns:
            return None

        # Score each pattern by consistency
        pattern_scores = {}

        for pattern in patterns:
            score = 0

            # Check if pattern appears consistently
            for inp, out in examples:
                if self._pattern_matches(pattern, inp, out):
                    score += 1

            # Use a more unique key
            pattern_key = f"{pattern['type']}_{id(pattern)}"
            pattern_scores[pattern_key] = (score, pattern)

        # Return pattern with highest score
        if pattern_scores:
            best_key = max(pattern_scores, key=lambda k: pattern_scores[k][0])
            best_score, best_pattern = pattern_scores[best_key]

            if self.verbose:
                print(f"Best pattern score: {best_score}/{len(examples)}")

            # Only return if pattern matches majority of examples
            if best_score >= len(examples) * 0.8:
                return best_pattern

        return None

    def _pattern_matches(self, pattern, inp, out):
        """Check if pattern matches this example - FIXED version."""
        if pattern["type"] == "spatial":
            if pattern["data"]["pattern"] == "cross":
                # Check if crosses exist in this example
                detected = self._detect_cross_pattern(inp, out)
                return detected is not None and len(detected) > 0
            elif pattern["data"]["pattern"] == "line":
                detected = self._detect_line_pattern(inp, out)
                return detected is not None
            elif pattern["data"]["pattern"] == "region":
                detected = self._detect_region_fill(inp, out)
                return detected is not None

        elif pattern["type"] == "color_map":
            # Check color mapping
            for i in range(inp.shape[0]):
                for j in range(inp.shape[1]):
                    if inp[i, j] in pattern["data"]:
                        if out[i, j] != pattern["data"][inp[i, j]]:
                            return False
            return True

        elif pattern["type"] == "conditional":
            # Check conditional pattern
            return self._analyze_conditional_pattern(inp, out) is not None

        return False

    def _synthesize_primitive(self, pattern, task_id):
        """Generate primitive code from pattern - IMPROVED version."""
        if pattern["type"] == "spatial":
            if pattern["data"]["pattern"] == "cross":
                return self._generate_cross_primitive_v2(
                    pattern["data"]["details"], task_id
                )
            elif pattern["data"]["pattern"] == "line":
                return self._generate_line_primitive(
                    pattern["data"]["details"], task_id
                )
            elif pattern["data"]["pattern"] == "region":
                return self._generate_region_primitive(
                    pattern["data"]["details"], task_id
                )

        elif pattern["type"] == "color_map":
            return self._generate_color_map_primitive(pattern["data"], task_id)

        elif pattern["type"] == "conditional":
            return self._generate_conditional_primitive(pattern["data"], task_id)

        elif pattern["type"] == "objects":
            return self._generate_object_primitive(pattern["data"], task_id)

        return None

    def _generate_cross_primitive_v2(self, cross_details, task_id):
        """Generate an improved cross pattern primitive based on detected patterns."""
        if not cross_details:
            return None

        # Analyze the cross patterns to extract rules
        center_colors = set()
        cross_colors = set()

        for cross in cross_details:
            center_colors.add(cross["center_color"])
            cross_colors.add(cross["cross_color"])

        # Generate code
        class_name = f"CrossPattern_{task_id.replace('-', '_')}"

        code = f'''
class {class_name}(Primitive):
    """Auto-discovered cross pattern for {task_id}."""

    def __init__(self):
        self.center_colors = {list(center_colors)}
        self.marker_colors = {list(cross_colors)}

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        result = context.copy()
        grid = result.current_grid.copy()
        h, w = grid.shape

        # Find positions to form crosses based on detected pattern
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                # Check if this position should have a cross
                if grid[i, j] in self.center_colors:
                    # Check for markers on same row/column
                    has_h_markers = False
                    has_v_markers = False
                    marker_color = None

                    # Check horizontal
                    for jj in range(w):
                        if jj != j and grid[i, jj] in self.marker_colors:
                            has_h_markers = True
                            marker_color = grid[i, jj]
                            break

                    # Check vertical
                    for ii in range(h):
                        if ii != i and grid[ii, j] in self.marker_colors:
                            has_v_markers = True
                            if marker_color is None:
                                marker_color = grid[ii, j]
                            break

                    # Form cross if markers found
                    if has_h_markers or has_v_markers:
                        if marker_color is not None:
                            # Form the cross
                            if i > 0:
                                grid[i-1, j] = marker_color
                            if i < h-1:
                                grid[i+1, j] = marker_color
                            if j > 0:
                                grid[i, j-1] = marker_color
                            if j < w-1:
                                grid[i, j+1] = marker_color

                            # Clear original markers
                            for jj in range(w):
                                if abs(jj - j) > 1 and grid[i, jj] == marker_color:
                                    grid[i, jj] = 0
                            for ii in range(h):
                                if abs(ii - i) > 1 and grid[ii, j] == marker_color:
                                    grid[ii, j] = 0

        result.current_grid = grid
        return result

    def __str__(self):
        return "{class_name}()"
'''
        return code

    def _generate_color_map_primitive(self, color_map, task_id):
        """Generate color mapping primitive."""
        class_name = f"ColorMap_{task_id.replace('-', '_')}"

        code = f'''
class {class_name}(Primitive):
    """Auto-discovered color mapping for {task_id}."""

    def __init__(self):
        self.color_map = {color_map}

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
        return code

    def _generate_line_primitive(self, lines, task_id):
        """Generate line drawing primitive."""
        if not lines:
            return None

        class_name = f"LinePattern_{task_id.replace('-', '_')}"

        # Analyze line patterns
        h_lines = [l for l in lines if l.get("type") == "horizontal"]
        v_lines = [l for l in lines if l.get("type") == "vertical"]

        code = f'''
class {class_name}(Primitive):
    """Auto-discovered line pattern for {task_id}."""

    def __init__(self):
        self.h_lines = {[l.get('row') for l in h_lines]}
        self.v_lines = {[l.get('col') for l in v_lines]}

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        result = context.copy()
        grid = result.current_grid.copy()

        # Draw horizontal lines
        for row in self.h_lines:
            if 0 <= row < grid.shape[0]:
                # Find the color to use (most common non-zero in row)
                row_colors = grid[row, :]
                non_zero = row_colors[row_colors != 0]
                if len(non_zero) > 0:
                    color = np.bincount(non_zero).argmax()
                    grid[row, :] = color

        # Draw vertical lines
        for col in self.v_lines:
            if 0 <= col < grid.shape[1]:
                # Find the color to use
                col_colors = grid[:, col]
                non_zero = col_colors[col_colors != 0]
                if len(non_zero) > 0:
                    color = np.bincount(non_zero).argmax()
                    grid[:, col] = color

        result.current_grid = grid
        return result

    def __str__(self):
        return "{class_name}()"
'''
        return code

    def _generate_region_primitive(self, regions, task_id):
        """Generate region filling primitive."""
        if not regions:
            return None

        class_name = f"RegionFill_{task_id.replace('-', '_')}"

        # Get boundary colors
        boundary_colors = list(
            set(r.get("boundary_color", 0) for r in regions if r.get("boundary_color"))
        )

        code = f'''
class {class_name}(Primitive):
    """Auto-discovered region filling for {task_id}."""

    def __init__(self):
        self.boundary_colors = {boundary_colors}
        self.fill_color = 4  # Common fill color in ARC

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        result = context.copy()
        grid = result.current_grid.copy()
        from scipy import ndimage

        # Fill regions enclosed by boundary colors
        for boundary_color in self.boundary_colors:
            if boundary_color != 0:
                # Create mask of boundary
                boundary_mask = (grid == boundary_color)

                # Fill holes in the boundary
                filled = ndimage.binary_fill_holes(boundary_mask)

                # Get interior (filled minus boundary)
                interior = filled & ~boundary_mask

                # Fill interior with fill color
                grid[interior] = self.fill_color

        result.current_grid = grid
        return result

    def __str__(self):
        return "{class_name}()"
'''
        return code

    def _generate_conditional_primitive(self, conditional_data, task_id):
        """Generate conditional transformation primitive."""
        if not conditional_data:
            return None

        class_name = f"ConditionalFill_{task_id.replace('-', '_')}"

        min_neighbors = conditional_data.get("min_neighbors", 2)
        fill_color = conditional_data.get("fill_color", 4)

        code = f'''
class {class_name}(Primitive):
    """Auto-discovered conditional fill for {task_id}."""

    def __init__(self):
        self.min_neighbors = {min_neighbors}
        self.fill_color = {fill_color}

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        result = context.copy()
        grid = result.current_grid.copy()
        h, w = grid.shape

        # Apply conditional fill based on neighbors
        for i in range(h):
            for j in range(w):
                if grid[i, j] == 0:  # Only fill empty cells
                    # Count non-zero neighbors
                    neighbors = []
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < h and 0 <= nj < w:
                                if grid[ni, nj] != 0:
                                    neighbors.append(grid[ni, nj])

                    # Fill if enough neighbors
                    if len(neighbors) >= self.min_neighbors:
                        grid[i, j] = self.fill_color

        result.current_grid = grid
        return result

    def __str__(self):
        return "{class_name}()"
'''
        return code

    def _generate_object_primitive(self, object_data, task_id):
        """Generate object manipulation primitive."""
        if not object_data:
            return None

        class_name = f"ObjectManip_{task_id.replace('-', '_')}"

        # Determine object operation type
        if object_data.get("type") == "count_based":
            # Object counting
            code = f'''
class {class_name}(Primitive):
    """Auto-discovered object counting for {task_id}."""

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        result = context.copy()
        grid = context.input_grid.copy()
        from scipy import ndimage

        # Count objects per color
        counts = {{}}
        for color in np.unique(grid):
            if color != 0:
                mask = grid == color
                labeled, num = ndimage.label(mask)
                counts[color] = num

        # Create output encoding counts
        output = np.zeros_like(grid)
        for i, (color, count) in enumerate(counts.items()):
            if i < output.shape[0] and count < 10:
                output[i, 0] = count

        result.current_grid = output
        return result

    def __str__(self):
        return "{class_name}()"
'''
        elif object_data.get("type") == "sort_by_size":
            # Object sorting
            code = f'''
class {class_name}(Primitive):
    """Auto-discovered object sorting for {task_id}."""

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        result = context.copy()
        grid = context.input_grid.copy()
        from scipy import ndimage

        # Extract and sort objects by size
        objects = []
        for color in np.unique(grid):
            if color != 0:
                mask = grid == color
                labeled, num = ndimage.label(mask)
                for i in range(1, num + 1):
                    obj_mask = labeled == i
                    size = np.sum(obj_mask)
                    positions = np.argwhere(obj_mask)
                    objects.append({{'color': color, 'size': size, 'positions': positions}})

        objects.sort(key=lambda x: x['size'])

        # Arrange sorted objects
        output = np.zeros_like(grid)
        col = 0
        for obj in objects:
            for pos in obj['positions']:
                if col < output.shape[1]:
                    output[pos[0], col] = obj['color']
            col += 2

        result.current_grid = output
        return result

    def __str__(self):
        return "{class_name}()"
'''
        else:
            # Default object operation
            code = f'''
class {class_name}(Primitive):
    """Auto-discovered object manipulation for {task_id}."""

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        result = context.copy()
        # Simple pass-through for now
        return result

    def __str__(self):
        return "{class_name}()"
'''

        return code

    def _test_primitive(self, primitive_code, examples, task_id):
        """Test the generated primitive on examples."""
        try:
            # Create a namespace for execution
            namespace = {
                "Primitive": Primitive,
                "ExecutionContext": ExecutionContext,
                "np": np,
            }

            # Execute the code to define the class
            exec(primitive_code, namespace)

            # Get the class
            class_name = f"CrossPattern_{task_id.replace('-', '_')}"
            if class_name not in namespace:
                # Try other pattern types
                for name in namespace:
                    if name.startswith(
                        (
                            "CrossPattern_",
                            "ColorMap_",
                            "LinePattern_",
                            "RegionFill_",
                            "ConditionalFill_",
                            "ObjectManip_",
                        )
                    ):
                        class_name = name
                        break

            if class_name not in namespace:
                if self.verbose:
                    print(f"Could not find generated class in namespace")
                return False

            PrimitiveClass = namespace[class_name]

            # Test on examples
            total_accuracy = 0
            for inp, expected in examples:
                primitive = PrimitiveClass()
                context = ExecutionContext(
                    input_grid=inp.copy(), current_grid=inp.copy()
                )

                result_context = primitive.execute(context)
                result = result_context.current_grid

                accuracy = np.mean(result == expected)
                total_accuracy += accuracy

            avg_accuracy = total_accuracy / len(examples)

            if self.verbose:
                print(f"Test accuracy: {avg_accuracy:.1%}")

            # Consider it successful if accuracy is high
            return avg_accuracy > 0.95

        except Exception as e:
            if self.verbose:
                print(f"Error testing primitive: {e}")
            return False


def test_improved_discovery():
    """Test the improved discovery system."""
    import json
    from pathlib import Path

    # Load task
    data_dir = Path("data/arc_agi_official/ARC-AGI/data/training")
    task_file = data_dir / "ae3edfdc.json"

    with open(task_file, "r") as f:
        task = json.load(f)

    # Get examples
    train_examples = [
        (np.array(ex["input"]), np.array(ex["output"])) for ex in task["train"]
    ]

    print("=" * 60)
    print("Testing Improved Primitive Discovery")
    print("=" * 60)

    discoverer = PrimitiveDiscovererV2(verbose=True)

    # Try discovery
    discovered_code = discoverer.discover_primitive("ae3edfdc", train_examples)

    if discovered_code:
        print(f"\n✅ Successfully discovered primitive!")
        print(f"\nGenerated code:\n{discovered_code}")

        # Save to file
        with open("discovered_ae3edfdc.py", "w") as f:
            f.write("#!/usr/bin/env python3\n")
            f.write("from compositional_dsl import Primitive, ExecutionContext\n")
            f.write("import numpy as np\n\n")
            f.write(discovered_code)
        print("\nSaved to discovered_ae3edfdc.py")
    else:
        print(f"\n❌ Failed to discover primitive")


if __name__ == "__main__":
    test_improved_discovery()
