#!/usr/bin/env python3
"""Automated primitive discovery system for ARC tasks - Version 4 Fixed.

Enhanced with diagonal patterns, shape detection, and fixed testing.
"""

from utils.imports import setup_project_paths

setup_project_paths()

import json
from pathlib import Path

import numpy as np
from automated_primitive_discovery_v3 import PrimitiveDiscovererV3
from compositional_dsl import ExecutionContext, Primitive
from diagonal_pattern_detector import (
    DiagonalPatternDetector,
    generate_diagonal_primitive,
)
from scipy import ndimage


class PrimitiveDiscovererV4Fixed(PrimitiveDiscovererV3):
    """Enhanced discoverer with diagonal and shape patterns - fixed version."""

    def __init__(
        self, verbose: bool = True, library_path: str = "arc_pattern_library.json"
    ):
        super().__init__(verbose, library_path)
        self.diagonal_detector = DiagonalPatternDetector()

    def _extract_patterns(self, examples):
        """Extract potential transformation patterns - enhanced version."""
        patterns = []

        for inp, out in examples:
            # Original spatial patterns (crosses, lines, regions)
            spatial = self._analyze_spatial_pattern(inp, out)
            if spatial:
                patterns.append({"type": "spatial", "data": spatial})

            # NEW: Diagonal patterns
            diagonal = self._analyze_diagonal_pattern(inp, out)
            if diagonal:
                patterns.append({"type": "diagonal", "data": diagonal})

            # NEW: Shape patterns
            shapes = self._analyze_shape_pattern(inp, out)
            if shapes:
                patterns.append({"type": "shape", "data": shapes})

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

    def _analyze_diagonal_pattern(self, inp, out):
        """Analyze diagonal transformation patterns."""
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
        """Analyze shape-based transformation patterns."""
        shapes = []

        # Detect rectangles
        rectangles = self._detect_rectangles(inp, out)
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

    def _detect_rectangles(self, inp, out):
        """Detect rectangle patterns in transformation."""
        h, w = out.shape
        rectangles = []

        # Look for filled rectangular regions
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

                if actual_size == expected_size:
                    # It's a rectangle!
                    rect_info = {
                        "top_left": (int(min_row), int(min_col)),
                        "bottom_right": (int(max_row), int(max_col)),
                        "color": int(color),
                        "filled": True,
                    }

                    # Check if this rectangle was in input
                    input_region = inp[min_row : max_row + 1, min_col : max_col + 1]
                    if not np.all(input_region == color):
                        # This is a new rectangle
                        rectangles.append(rect_info)

        return rectangles if rectangles else None

    def _detect_triangles(self, inp, out):
        """Detect triangle patterns in transformation."""
        # Simplified: Look for triangular fills
        h, w = out.shape

        # Check for right triangles in corners
        triangles = []

        # Top-left triangle
        is_triangle = True
        triangle_color = None
        for i in range(min(h, w)):
            for j in range(i + 1):
                if out[i, j] != 0 and inp[i, j] != out[i, j]:
                    if triangle_color is None:
                        triangle_color = out[i, j]
                    elif out[i, j] != triangle_color:
                        is_triangle = False
                        break
            if not is_triangle:
                break

        if is_triangle and triangle_color is not None:
            triangles.append(
                {"type": "top_left", "size": min(h, w), "color": int(triangle_color)}
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
                    if inp[i, j] != out[i, j] and out[i, j] != 0:
                        diamond_coords.append((i, j))

        if len(diamond_coords) > 4:
            # Find the color used
            colors = [out[i, j] for i, j in diamond_coords]
            if colors:
                most_common_color = max(set(colors), key=colors.count)
                return [
                    {
                        "center": (center_i, center_j),
                        "radius": radius,
                        "points": len(diamond_coords),
                        "color": int(most_common_color),
                    }
                ]

        return None

    def _pattern_matches(self, pattern, inp, out):
        """Check if pattern matches this example - extended version."""
        # Handle new pattern types
        if pattern["type"] == "diagonal":
            if pattern["data"]["pattern"] == "diagonal_line":
                detected = self.diagonal_detector.detect_diagonal_lines(inp, out)
                return detected is not None
            elif pattern["data"]["pattern"] == "diagonal_fill":
                detected = self.diagonal_detector.detect_diagonal_fill(inp, out)
                return detected is not None
            elif pattern["data"]["pattern"] == "diagonal_symmetry":
                detected = self.diagonal_detector.detect_diagonal_symmetry(inp, out)
                return detected is not None

        elif pattern["type"] == "shape":
            # Check if any shape matches
            detected_shapes = self._analyze_shape_pattern(inp, out)
            if detected_shapes:
                for shape in pattern["data"]:
                    for detected in detected_shapes:
                        if shape["type"] == detected["type"]:
                            return True
            return False

        # Fall back to parent implementation
        return super()._pattern_matches(pattern, inp, out)

    def _synthesize_primitive(self, pattern, task_id):
        """Generate primitive code from pattern - extended version."""
        if pattern["type"] == "diagonal":
            return generate_diagonal_primitive(pattern["data"]["details"], task_id)

        elif pattern["type"] == "shape":
            return self._generate_shape_primitive(pattern["data"], task_id)

        # Fall back to parent implementation
        return super()._synthesize_primitive(pattern, task_id)

    def _generate_shape_primitive(self, shape_data, task_id):
        """Generate shape-based primitive."""
        if not shape_data:
            return None

        # Get first shape type
        first_shape = shape_data[0]
        shape_type = first_shape["type"]

        class_name = f"ShapePattern_{task_id.replace('-', '_')}"

        if shape_type == "rectangle":
            rectangles = first_shape["details"]
            rectangles[0] if rectangles else {}

            code = f'''
class {class_name}(Primitive):
    """Auto-discovered rectangle pattern for {task_id}."""

    def __init__(self):
        self.rectangles = {rectangles}

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        result = context.copy()
        grid = result.current_grid.copy()

        for rect in self.rectangles:
            top, left = rect["top_left"]
            bottom, right = rect["bottom_right"]
            color = rect["color"]

            for i in range(top, bottom + 1):
                for j in range(left, right + 1):
                    if 0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]:
                        grid[i, j] = color

        result.current_grid = grid
        return result

    def __str__(self):
        return "{class_name}()"
'''
            return code

        elif shape_type == "triangle":
            details = first_shape["details"][0] if first_shape["details"] else {}
            triangle_color = details.get("color", 4)

            code = f'''
class {class_name}(Primitive):
    """Auto-discovered triangle pattern for {task_id}."""

    def __init__(self):
        self.fill_color = {triangle_color}

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        result = context.copy()
        grid = result.current_grid.copy()

        # Fill triangular region
        for i in range(grid.shape[0]):
            for j in range(i + 1):  # Upper triangular
                if j < grid.shape[1]:
                    if grid[i, j] == 0:
                        grid[i, j] = self.fill_color

        result.current_grid = grid
        return result

    def __str__(self):
        return "{class_name}()"
'''
            return code

        elif shape_type == "diamond":
            details = first_shape["details"][0] if first_shape["details"] else {}
            center = details.get("center", (5, 5))
            radius = details.get("radius", 3)
            diamond_color = details.get("color", 4)

            code = f'''
class {class_name}(Primitive):
    """Auto-discovered diamond pattern for {task_id}."""

    def __init__(self):
        self.center = {center}
        self.radius = {radius}
        self.fill_color = {diamond_color}

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        result = context.copy()
        grid = result.current_grid.copy()
        h, w = grid.shape

        center_i, center_j = self.center

        for i in range(h):
            for j in range(w):
                manhattan_dist = abs(i - center_i) + abs(j - center_j)
                if manhattan_dist <= self.radius:
                    if grid[i, j] == 0:
                        grid[i, j] = self.fill_color

        result.current_grid = grid
        return result

    def __str__(self):
        return "{class_name}()"
'''
            return code

        return None

    def _test_primitive(self, primitive_code, examples, task_id):
        """Test the generated primitive on examples - fixed version."""
        try:
            # Create a namespace for execution
            namespace = {
                "Primitive": Primitive,
                "ExecutionContext": ExecutionContext,
                "np": np,
            }

            # Execute the code to define the class
            exec(primitive_code, namespace)

            # Get the class - search for any class that extends Primitive
            class_name = None
            for name in namespace:
                if name.startswith(
                    (
                        "CrossPattern_",
                        "ColorMap_",
                        "LinePattern_",
                        "RegionFill_",
                        "ConditionalFill_",
                        "ObjectManip_",
                        "DiagonalPattern_",
                        "DiagonalFill_",
                        "DiagonalTranspose_",
                        "ShapePattern_",
                    )
                ):
                    class_name = name
                    break

            if class_name is None:
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


def test_enhanced_discovery():
    """Test the enhanced discovery system with diagonal and shape patterns."""

    # Test tasks that might have diagonal or shape patterns
    test_tasks = [
        "ae3edfdc",  # Cross pattern - baseline
        "00d62c1b",  # Region pattern - baseline
        "0520fde7",  # Might have diagonal
        "045e512c",  # Might have shapes
        "0a938d79",  # Might have diagonal symmetry
        "0b148d64",  # Might have rectangles
        "0ca9ddb6",  # Another test case
        "0d3d703e",  # Another test case
    ]

    data_dir = Path("data/arc_agi_official/ARC-AGI/data/training")

    print("=" * 80)
    print("Testing Enhanced Primitive Discovery (V4 Fixed)")
    print("=" * 80)

    discoverer = PrimitiveDiscovererV4Fixed(verbose=True)

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
            continue

        try:
            with open(task_file, "r") as f:
                task = json.load(f)

            train_examples = [
                (np.array(ex["input"]), np.array(ex["output"])) for ex in task["train"]
            ]

            # Try discovery
            discovered_code = discoverer.discover_primitive(task_id, train_examples)

            if discovered_code:
                print(f"  ✅ Discovery successful!")
                results.append({"task": task_id, "success": True})

                # Show discovered pattern type
                if "Diagonal" in discovered_code:
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
            import traceback

            traceback.print_exc()
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

    print("\nEnhanced patterns implemented:")
    print("✅ Diagonal lines (45°, 135°)")
    print("✅ Diagonal fills (triangular regions)")
    print("✅ Diagonal symmetry (transpose)")
    print("✅ Rectangle detection")
    print("✅ Triangle detection")
    print("✅ Diamond detection")


if __name__ == "__main__":
    test_enhanced_discovery()
