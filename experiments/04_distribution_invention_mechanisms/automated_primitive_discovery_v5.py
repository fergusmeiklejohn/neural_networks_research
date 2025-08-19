#!/usr/bin/env python3
"""Automated primitive discovery system for ARC tasks - Version 5.

Fixed to handle different input/output sizes and improved shape accuracy.
"""

from utils.imports import setup_project_paths

setup_project_paths()

import json
from pathlib import Path

import numpy as np
from automated_primitive_discovery_v3 import PrimitiveDiscovererV3
from diagonal_pattern_detector import (
    DiagonalPatternDetector,
    generate_diagonal_primitive,
)
from scipy import ndimage


class PrimitiveDiscovererV5(PrimitiveDiscovererV3):
    """Enhanced discoverer with size mismatch handling and improved patterns."""

    def __init__(
        self, verbose: bool = True, library_path: str = "arc_pattern_library.json"
    ):
        super().__init__(verbose, library_path)
        self.diagonal_detector = DiagonalPatternDetector()

    def _extract_patterns(self, examples):
        """Extract potential transformation patterns - size-aware version."""
        patterns = []

        for inp, out in examples:
            # Check for size changes first
            if inp.shape != out.shape:
                # Handle size change patterns separately
                size_pattern = self._analyze_size_change_pattern(inp, out)
                if size_pattern:
                    patterns.append({"type": "size_change", "data": size_pattern})

            # Only analyze spatial patterns if sizes match
            if inp.shape == out.shape:
                # Original spatial patterns (crosses, lines, regions)
                spatial = self._analyze_spatial_pattern(inp, out)
                if spatial:
                    patterns.append({"type": "spatial", "data": spatial})

                # Diagonal patterns
                diagonal = self._analyze_diagonal_pattern(inp, out)
                if diagonal:
                    patterns.append({"type": "diagonal", "data": diagonal})

            # Shape patterns (can work with different sizes)
            shapes = self._analyze_shape_pattern(inp, out)
            if shapes:
                patterns.append({"type": "shape", "data": shapes})

            # Color mapping patterns (size-independent)
            color_map = self._analyze_color_mapping(inp, out)
            if color_map:
                patterns.append({"type": "color_map", "data": color_map})

            # Object-based patterns (size-independent)
            objects = self._analyze_object_pattern(inp, out)
            if objects:
                patterns.append({"type": "objects", "data": objects})

            # Conditional patterns (only if same size)
            if inp.shape == out.shape:
                conditional = self._analyze_conditional_pattern(inp, out)
                if conditional:
                    patterns.append({"type": "conditional", "data": conditional})

        return patterns

    def _analyze_size_change_pattern(self, inp, out):
        """Analyze patterns involving size changes."""
        h_in, w_in = inp.shape
        h_out, w_out = out.shape

        patterns = []

        # Check for scaling
        if h_out == h_in * 2 and w_out == w_in * 2:
            patterns.append({"type": "scale_2x", "factor": 2})
        elif h_out == h_in * 3 and w_out == w_in * 3:
            patterns.append({"type": "scale_3x", "factor": 3})

        # Check for cropping
        elif h_out < h_in or w_out < w_in:
            # Try to find the crop region
            for i in range(h_in - h_out + 1):
                for j in range(w_in - w_out + 1):
                    crop = inp[i : i + h_out, j : j + w_out]
                    if np.array_equal(crop, out):
                        patterns.append(
                            {
                                "type": "crop",
                                "top": i,
                                "left": j,
                                "height": h_out,
                                "width": w_out,
                            }
                        )
                        break

        # Check for padding
        elif h_out > h_in or w_out > w_in:
            # Check if input is contained in output
            for i in range(h_out - h_in + 1):
                for j in range(w_out - w_in + 1):
                    if i + h_in <= h_out and j + w_in <= w_out:
                        region = out[i : i + h_in, j : j + w_in]
                        if np.array_equal(region, inp):
                            patterns.append(
                                {
                                    "type": "pad",
                                    "top": i,
                                    "left": j,
                                    "pad_color": 0,  # Assuming 0 padding
                                }
                            )
                            break

        return patterns if patterns else None

    def _analyze_spatial_pattern(self, inp, out):
        """Analyze spatial transformation patterns - FIXED for size safety."""
        # Only analyze if same size
        if inp.shape != out.shape:
            return None

        # Check for cross patterns
        crosses = self._detect_cross_pattern_safe(inp, out)
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

    def _detect_cross_pattern_safe(self, inp, out):
        """Detect cross pattern formations - size-safe version."""
        # Ensure same size
        if inp.shape != out.shape:
            return None

        h, w = out.shape
        crosses = []

        # Look for positions where crosses formed
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                # Check if this is a cross center in output
                center_val = out[i, j]

                # Count how many arms changed
                arms_changed = 0
                arm_colors = []

                # Check each arm safely
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
                        crosses.append(cross_info)

        return crosses if crosses else None

    def _analyze_diagonal_pattern(self, inp, out):
        """Analyze diagonal transformation patterns - size-safe."""
        # Only works if same size
        if inp.shape != out.shape:
            return None

        # Check for diagonal lines
        diagonals = self.diagonal_detector.detect_diagonal_lines(inp, out)
        if diagonals:
            return {"pattern": "diagonal_line", "details": diagonals}

        # Check for diagonal fills
        fills = self.diagonal_detector.detect_diagonal_fill(inp, out)
        if fills:
            return {"pattern": "diagonal_fill", "details": fills}

        # Check for diagonal symmetry
        symmetry = self.diagonal_detector.detect_diagonal_symmetry(inp, out)
        if symmetry:
            return {"pattern": "diagonal_symmetry", "details": symmetry}

        return None

    def _analyze_shape_pattern(self, inp, out):
        """Analyze shape-based transformation patterns - improved accuracy."""
        shapes = []

        # Detect rectangles with better accuracy
        rectangles = self._detect_rectangles_improved(inp, out)
        if rectangles:
            shapes.append({"type": "rectangle", "details": rectangles})

        # Detect triangles
        triangles = self._detect_triangles(inp, out)
        if triangles:
            shapes.append({"type": "triangle", "details": triangles})

        # Detect diamonds
        diamonds = self._detect_diamonds(inp, out)
        if diamonds:
            shapes.append({"type": "diamond", "details": diamonds})

        return shapes if shapes else None

    def _detect_rectangles_improved(self, inp, out):
        """Improved rectangle detection with better parameter extraction."""
        h_out, w_out = out.shape
        h_inp, w_inp = inp.shape
        rectangles = []

        # Look for filled rectangular regions in output
        for color in np.unique(out):
            if color == 0:
                continue

            # Find connected components of this color
            mask = out == color
            labeled, num = ndimage.label(mask)

            for i in range(1, num + 1):
                component = labeled == i

                # Check if component is rectangular
                rows, cols = np.where(component)
                if len(rows) < 4:  # Too small
                    continue

                min_row, max_row = rows.min(), rows.max()
                min_col, max_col = cols.min(), cols.max()

                # Check if it's a filled rectangle
                expected_size = (max_row - min_row + 1) * (max_col - min_col + 1)
                actual_size = component.sum()

                if actual_size >= expected_size * 0.9:  # Allow 10% tolerance
                    rect_info = {
                        "top_left": (int(min_row), int(min_col)),
                        "bottom_right": (int(max_row), int(max_col)),
                        "color": int(color),
                        "filled": actual_size == expected_size,
                        "width": int(max_col - min_col + 1),
                        "height": int(max_row - min_row + 1),
                    }

                    # Check if this rectangle was NOT in input (new rectangle)
                    is_new = True
                    if (
                        min_row < h_inp
                        and max_row < h_inp
                        and min_col < w_inp
                        and max_col < w_inp
                    ):
                        input_region = inp[min_row : max_row + 1, min_col : max_col + 1]
                        if np.all(input_region == color):
                            is_new = False

                    if is_new:
                        rectangles.append(rect_info)

        return rectangles if rectangles else None

    def _detect_triangles(self, inp, out):
        """Detect triangle patterns in transformation."""
        h, w = out.shape
        triangles = []

        # Check for right triangles in corners
        # Top-left triangle
        triangle_mask = np.zeros_like(out, dtype=bool)
        for i in range(min(h, w)):
            for j in range(i + 1):
                if i < h and j < w:
                    triangle_mask[i, j] = True

        # Check if this region was filled
        if np.any(triangle_mask):
            triangle_values = out[triangle_mask]
            non_zero = triangle_values[triangle_values != 0]
            if len(non_zero) > len(triangle_values) * 0.5:  # >50% filled
                # Find most common color
                if len(non_zero) > 0:
                    unique, counts = np.unique(non_zero, return_counts=True)
                    color = unique[np.argmax(counts)]
                    triangles.append(
                        {"type": "top_left", "size": min(h, w), "color": int(color)}
                    )

        return triangles if triangles else None

    def _detect_diamonds(self, inp, out):
        """Detect diamond/rhombus patterns in transformation."""
        h, w = out.shape
        center_i, center_j = h // 2, w // 2

        # Check for diamond pattern around center
        diamond_coords = []
        radius = min(h, w) // 2

        for i in range(h):
            for j in range(w):
                manhattan_dist = abs(i - center_i) + abs(j - center_j)
                if manhattan_dist <= radius:
                    # Check if sizes match before comparison
                    if inp.shape == out.shape:
                        if inp[i, j] != out[i, j] and out[i, j] != 0:
                            diamond_coords.append((i, j))
                    elif out[i, j] != 0:  # Different sizes, just check output
                        diamond_coords.append((i, j))

        if len(diamond_coords) > 4:
            # Find the color used
            colors = [out[i, j] for i, j in diamond_coords]
            if colors:
                unique, counts = np.unique(colors, return_counts=True)
                most_common_color = unique[np.argmax(counts)]
                return [
                    {
                        "center": (center_i, center_j),
                        "radius": radius,
                        "points": len(diamond_coords),
                        "color": int(most_common_color),
                    }
                ]

        return None

    def _synthesize_primitive(self, pattern, task_id):
        """Generate primitive code from pattern - extended version."""
        if pattern["type"] == "size_change":
            return self._generate_size_change_primitive(pattern["data"], task_id)
        elif pattern["type"] == "diagonal":
            return generate_diagonal_primitive(pattern["data"]["details"], task_id)
        elif pattern["type"] == "shape":
            return self._generate_shape_primitive_improved(pattern["data"], task_id)

        # Fall back to parent implementation
        return super()._synthesize_primitive(pattern, task_id)

    def _generate_size_change_primitive(self, size_data, task_id):
        """Generate size change primitive."""
        if not size_data:
            return None

        first_pattern = size_data[0]
        pattern_type = first_pattern["type"]
        class_name = f"SizeChange_{task_id.replace('-', '_')}"

        if pattern_type == "scale_2x" or pattern_type == "scale_3x":
            factor = first_pattern["factor"]
            code = f'''
class {class_name}(Primitive):
    """Auto-discovered scaling pattern for {task_id}."""

    def __init__(self):
        self.scale_factor = {factor}

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        result = context.copy()
        grid = context.input_grid.copy()
        h, w = grid.shape

        # Create scaled output
        import numpy as np
        output = np.zeros((h * self.scale_factor, w * self.scale_factor), dtype=grid.dtype)

        for i in range(h):
            for j in range(w):
                for di in range(self.scale_factor):
                    for dj in range(self.scale_factor):
                        output[i * self.scale_factor + di, j * self.scale_factor + dj] = grid[i, j]

        result.current_grid = output
        return result

    def __str__(self):
        return "{class_name}()"
'''
            return code

        elif pattern_type == "crop":
            top = first_pattern["top"]
            left = first_pattern["left"]
            height = first_pattern["height"]
            width = first_pattern["width"]

            code = f'''
class {class_name}(Primitive):
    """Auto-discovered cropping pattern for {task_id}."""

    def __init__(self):
        self.top = {top}
        self.left = {left}
        self.height = {height}
        self.width = {width}

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        result = context.copy()
        grid = context.input_grid.copy()

        # Crop the grid
        result.current_grid = grid[self.top:self.top+self.height, self.left:self.left+self.width]
        return result

    def __str__(self):
        return "{class_name}()"
'''
            return code

        return None

    def _generate_shape_primitive_improved(self, shape_data, task_id):
        """Generate shape-based primitive with improved accuracy."""
        if not shape_data:
            return None

        # Get first shape type
        first_shape = shape_data[0]
        shape_type = first_shape["type"]

        class_name = f"ShapePattern_{task_id.replace('-', '_')}"

        if shape_type == "rectangle":
            rectangles = first_shape["details"]
            if not rectangles:
                return None

            # Extract common properties
            rectangles[0]

            code = f'''
class {class_name}(Primitive):
    """Auto-discovered rectangle pattern for {task_id}."""

    def __init__(self):
        self.rectangles = {rectangles}

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        result = context.copy()
        # Start with input grid to preserve existing content
        grid = context.input_grid.copy()

        for rect in self.rectangles:
            top, left = rect["top_left"]
            bottom, right = rect["bottom_right"]
            color = rect["color"]

            # Draw rectangle
            for i in range(top, min(bottom + 1, grid.shape[0])):
                for j in range(left, min(right + 1, grid.shape[1])):
                    grid[i, j] = color

        result.current_grid = grid
        return result

    def __str__(self):
        return "{class_name}()"
'''
            return code

        # Similar improvements for triangle and diamond...
        return super()._generate_shape_primitive(shape_data, task_id)


def test_improved_discovery():
    """Test the improved discovery system with size handling."""

    # Test tasks including those with size mismatches
    test_tasks = [
        "ae3edfdc",  # Cross pattern - baseline
        "00d62c1b",  # Region pattern - baseline
        "0520fde7",  # Previously failed with index error
        "045e512c",  # Shape patterns
        "0a938d79",  # Diagonal symmetry
        "0b148d64",  # Previously failed with index error
        "0ca9ddb6",  # Test case
        "0d3d703e",  # Test case
        "05f2a901",  # Additional test
        "06df4c85",  # Additional test
    ]

    data_dir = Path("data/arc_agi_official/ARC-AGI/data/training")

    print("=" * 80)
    print("Testing Improved Primitive Discovery (V5)")
    print("=" * 80)

    discoverer = PrimitiveDiscovererV5(verbose=True)

    # Show library stats
    stats = discoverer.library.export_statistics()
    print(
        f"\nLibrary contains {stats['total_patterns']} patterns from {stats['tasks_covered']} tasks"
    )

    results = []

    for task_id in test_tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task_id}")
        print("-" * 60)

        task_file = data_dir / f"{task_id}.json"

        if not task_file.exists():
            print(f"  ❌ Task file not found")
            results.append({"task": task_id, "success": False})
            continue

        try:
            with open(task_file, "r") as f:
                task = json.load(f)

            train_examples = [
                (np.array(ex["input"]), np.array(ex["output"])) for ex in task["train"]
            ]

            # Check for size mismatches
            for inp, out in train_examples:
                if inp.shape != out.shape:
                    print(f"  Size mismatch: {inp.shape} -> {out.shape}")
                    break

            # Try discovery
            discovered_code = discoverer.discover_primitive(task_id, train_examples)

            if discovered_code:
                print(f"  ✅ Discovery successful!")
                results.append({"task": task_id, "success": True})

                # Show discovered pattern type
                if "SizeChange" in discovered_code:
                    print("  Pattern type: Size Change")
                elif "Diagonal" in discovered_code:
                    print("  Pattern type: Diagonal")
                elif "Shape" in discovered_code:
                    print("  Pattern type: Shape")
                elif "Cross" in discovered_code:
                    print("  Pattern type: Cross")
                elif "Region" in discovered_code:
                    print("  Pattern type: Region")
            else:
                print(f"  ❌ Discovery failed")
                results.append({"task": task_id, "success": False})

        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({"task": task_id, "success": False})

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    successful = sum(1 for r in results if r["success"])
    total = len(results)

    print(f"Success rate: {successful}/{total} ({successful/total*100:.1f}%)")

    # Show updated library stats
    new_stats = discoverer.library.export_statistics()
    print(
        f"\nLibrary now contains {new_stats['total_patterns']} patterns (+{new_stats['total_patterns'] - stats['total_patterns']})"
    )

    print("\nImprovements in V5:")
    print("✅ Handles different input/output sizes")
    print("✅ Size change patterns (scaling, cropping, padding)")
    print("✅ Improved rectangle detection accuracy")
    print("✅ Better shape parameter extraction")
    print("✅ Safe spatial pattern detection")


if __name__ == "__main__":
    test_improved_discovery()
