#!/usr/bin/env python3
"""Analyze failed ARC-AGI tasks to identify missing primitives and patterns.

This script systematically analyzes tasks that failed in our evaluation to
understand what specific transformations ARC uses that we're missing.
"""

from utils.imports import setup_project_paths

setup_project_paths()

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
from scipy import ndimage

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from enhanced_arc_solver import EnhancedARCSolver
from enhanced_neural_perception import EnhancedNeuralPerception


class FailedTaskAnalyzer:
    """Analyzes failed ARC tasks to identify missing capabilities."""

    def __init__(self):
        self.data_dir = Path("data/arc_agi_official/ARC-AGI/data")
        self.solver = EnhancedARCSolver(use_synthesis=True)
        self.perception = EnhancedNeuralPerception()

        # Track failure modes
        self.failure_modes = defaultdict(list)
        self.missing_operations = defaultdict(int)
        self.pattern_types = defaultdict(list)

        # Load evaluation results if available
        self.load_previous_results()

    def load_previous_results(self):
        """Load previous evaluation results to identify failed tasks."""
        results_path = Path("outputs/arc_evaluation_results.json")
        if results_path.exists():
            with open(results_path) as f:
                data = json.load(f)
                # Handle both dict and list formats
                if isinstance(data, dict):
                    self.previous_results = data
                    self.failed_tasks = [
                        task_id
                        for task_id, correct in self.previous_results.items()
                        if not correct
                    ]
                else:
                    # Assume it's a list of results
                    self.previous_results = {}
                    self.failed_tasks = []
                    for item in data:
                        if isinstance(item, dict) and "task_id" in item:
                            task_id = item["task_id"]
                            correct = item.get("correct", False)
                            self.previous_results[task_id] = correct
                            if not correct:
                                self.failed_tasks.append(task_id)
        else:
            self.previous_results = {}
            self.failed_tasks = []

    def analyze_transformation(
        self, input_grid: np.ndarray, output_grid: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze the transformation between input and output."""
        analysis = {
            "size_change": False,
            "color_mapping": {},
            "object_count_change": 0,
            "spatial_pattern": None,
            "arithmetic_operation": None,
            "conditional_logic": None,
            "structural_change": None,
        }

        # Size change analysis
        if input_grid.shape != output_grid.shape:
            analysis["size_change"] = True
            analysis["size_ratio"] = (
                output_grid.shape[0] / input_grid.shape[0],
                output_grid.shape[1] / input_grid.shape[1],
            )

        # Color analysis
        input_colors = set(input_grid.flatten())
        output_colors = set(output_grid.flatten())

        # Check for color mappings
        if input_colors != output_colors and input_grid.shape == output_grid.shape:
            # Try to infer mapping
            for in_color in input_colors:
                in_mask = input_grid == in_color
                if in_mask.any():
                    # Find what this color became
                    corresponding_output = output_grid[in_mask]
                    if len(set(corresponding_output)) == 1:
                        analysis["color_mapping"][int(in_color)] = int(
                            corresponding_output[0]
                        )

        # Object count analysis
        input_objects = self.extract_objects(input_grid)
        output_objects = self.extract_objects(output_grid)
        analysis["object_count_change"] = len(output_objects) - len(input_objects)

        # Arithmetic detection
        analysis["arithmetic_operation"] = self.detect_arithmetic(
            input_grid, output_grid, input_colors, output_colors
        )

        # Spatial pattern detection
        analysis["spatial_pattern"] = self.detect_spatial_pattern(
            input_grid, output_grid
        )

        # Conditional logic detection
        analysis["conditional_logic"] = self.detect_conditional_logic(
            input_grid, output_grid, input_objects
        )

        # Structural change
        analysis["structural_change"] = self.detect_structural_change(
            input_objects, output_objects
        )

        return analysis

    def extract_objects(self, grid: np.ndarray) -> List[Dict]:
        """Extract objects from grid."""
        objects = []
        colors = set(grid.flatten()) - {0}  # Ignore background

        for color in colors:
            mask = grid == color
            labeled, num = ndimage.label(mask)

            for i in range(1, num + 1):
                component = labeled == i
                pixels = np.argwhere(component)

                if len(pixels) > 0:
                    min_r, min_c = pixels.min(axis=0)
                    max_r, max_c = pixels.max(axis=0)

                    objects.append(
                        {
                            "color": int(color),
                            "pixels": pixels.tolist(),
                            "bbox": (min_r, min_c, max_r, max_c),
                            "size": len(pixels),
                        }
                    )

        return objects

    def detect_arithmetic(
        self,
        input_grid: np.ndarray,
        output_grid: np.ndarray,
        input_colors: Set[int],
        output_colors: Set[int],
    ) -> Optional[str]:
        """Detect arithmetic operations."""
        # Check if colors are shifted by constant
        if (
            len(input_colors) == len(output_colors)
            and input_grid.shape == output_grid.shape
        ):
            color_diffs = []
            for in_c in sorted(input_colors):
                if in_c == 0:
                    continue
                # Find corresponding output color
                in_positions = np.argwhere(input_grid == in_c)
                if len(in_positions) > 0:
                    # Check bounds
                    r, c = in_positions[0]
                    if r < output_grid.shape[0] and c < output_grid.shape[1]:
                        out_color = output_grid[r, c]
                        if out_color != 0:
                            color_diffs.append(out_color - in_c)

            if color_diffs and all(d == color_diffs[0] for d in color_diffs):
                return f"add_{color_diffs[0]}_to_colors"

        # Check for counting patterns
        unique_input = len(input_colors)
        unique_output = len(output_colors)

        if unique_output > unique_input:
            # Check if output encodes count
            for color in output_colors:
                count = np.sum(output_grid == color)
                if count > 1 and color > 0 and color <= 9:
                    # Might be encoding count as color
                    return "count_encoding"

        return None

    def detect_spatial_pattern(
        self, input_grid: np.ndarray, output_grid: np.ndarray
    ) -> Optional[str]:
        """Detect spatial transformation patterns."""
        patterns = []

        # Check for diagonal patterns
        if self.has_diagonal_pattern(output_grid):
            patterns.append("diagonal")

        # Check for spiral patterns
        if self.has_spiral_pattern(output_grid):
            patterns.append("spiral")

        # Check for border patterns
        if self.has_border_pattern(output_grid):
            patterns.append("border")

        # Check for repeating patterns
        if self.has_repeating_pattern(output_grid):
            patterns.append("repeating")

        # Check for symmetry
        if self.is_symmetric(output_grid):
            patterns.append("symmetric")

        return ",".join(patterns) if patterns else None

    def has_diagonal_pattern(self, grid: np.ndarray) -> bool:
        """Check for diagonal patterns."""
        h, w = grid.shape

        # Check main diagonal
        if h == w:
            diagonal = [grid[i, i] for i in range(min(h, w))]
            if len(set(diagonal)) == 1 and diagonal[0] != 0:
                return True

        # Check anti-diagonal
        anti_diagonal = [grid[i, w - 1 - i] for i in range(min(h, w))]
        if len(set(anti_diagonal)) == 1 and anti_diagonal[0] != 0:
            return True

        return False

    def has_spiral_pattern(self, grid: np.ndarray) -> bool:
        """Check for spiral patterns."""
        # Simplified check - look for specific ordering
        h, w = grid.shape
        if h < 3 or w < 3:
            return False

        # Check if perimeter has consistent pattern
        perimeter = []
        # Top row
        perimeter.extend(grid[0, :])
        # Right column (excluding corners)
        if h > 1:
            perimeter.extend(grid[1:-1, -1])
        # Bottom row (reversed)
        if h > 1:
            perimeter.extend(grid[-1, ::-1])
        # Left column (excluding corners, reversed)
        if h > 2 and w > 1:
            perimeter.extend(grid[-2:0:-1, 0])

        # Check if perimeter has a pattern
        if len(set(perimeter)) > 1 and len(set(perimeter)) < len(perimeter) / 2:
            return True

        return False

    def has_border_pattern(self, grid: np.ndarray) -> bool:
        """Check for border patterns."""
        h, w = grid.shape

        # Check if border is different from interior
        border = set()
        border.update(grid[0, :])  # Top
        border.update(grid[-1, :])  # Bottom
        border.update(grid[:, 0])  # Left
        border.update(grid[:, -1])  # Right

        if h > 2 and w > 2:
            interior = set(grid[1:-1, 1:-1].flatten())
            if border and interior and not border.intersection(interior):
                return True

        return False

    def has_repeating_pattern(self, grid: np.ndarray) -> bool:
        """Check for repeating patterns."""
        h, w = grid.shape

        # Check for 2x2 repeating
        if h >= 4 and w >= 4:
            pattern = grid[:2, :2]
            for i in range(0, h - 1, 2):
                for j in range(0, w - 1, 2):
                    if not np.array_equal(grid[i : i + 2, j : j + 2], pattern):
                        break
                else:
                    continue
                break
            else:
                return True

        # Check for row-wise repeating
        if h >= 2:
            for i in range(h - 1):
                if np.array_equal(grid[i], grid[i + 1]):
                    return True

        return False

    def is_symmetric(self, grid: np.ndarray) -> bool:
        """Check for symmetry."""
        # Horizontal symmetry
        if np.array_equal(grid, np.flipud(grid)):
            return True
        # Vertical symmetry
        if np.array_equal(grid, np.fliplr(grid)):
            return True
        # Rotational symmetry (180 degrees)
        if np.array_equal(grid, np.rot90(grid, 2)):
            return True

        return False

    def detect_conditional_logic(
        self, input_grid: np.ndarray, output_grid: np.ndarray, input_objects: List[Dict]
    ) -> Optional[str]:
        """Detect conditional logic patterns."""
        conditions = []

        # Check if output depends on object properties
        for obj in input_objects:
            # Check size-based conditions
            if obj["size"] > 4:
                # Check if large objects are treated differently
                bbox = obj["bbox"]
                region = output_grid[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1]
                if len(set(region.flatten())) == 1:
                    conditions.append("if_large_then_fill")

            # Check shape-based conditions
            if self.is_square_object(obj):
                bbox = obj["bbox"]
                region = output_grid[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1]
                if not np.array_equal(
                    input_grid[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1], region
                ):
                    conditions.append("if_square_then_transform")

        # Check color-based conditions
        if input_grid.shape == output_grid.shape:
            for color in set(input_grid.flatten()) - {0}:
                mask = input_grid == color
                output_at_mask = output_grid[mask]
                if len(set(output_at_mask)) == 1 and output_at_mask[0] != color:
                    conditions.append(f"if_color_{color}_then_{output_at_mask[0]}")

        return ",".join(conditions) if conditions else None

    def is_square_object(self, obj: Dict) -> bool:
        """Check if an object forms a square."""
        bbox = obj["bbox"]
        height = bbox[2] - bbox[0] + 1
        width = bbox[3] - bbox[1] + 1
        return height == width

    def detect_structural_change(
        self, input_objects: List[Dict], output_objects: List[Dict]
    ) -> Optional[str]:
        """Detect structural changes between objects."""
        changes = []

        # Check for object merging
        if len(output_objects) < len(input_objects):
            changes.append("objects_merged")

        # Check for object splitting
        if len(output_objects) > len(input_objects):
            changes.append("objects_split")

        # Check for object rearrangement
        if len(input_objects) == len(output_objects) and len(input_objects) > 1:
            input_positions = [obj["bbox"][:2] for obj in input_objects]
            output_positions = [obj["bbox"][:2] for obj in output_objects]

            if set(input_positions) != set(output_positions):
                changes.append("objects_rearranged")

        return ",".join(changes) if changes else None

    def analyze_task(self, task_id: str) -> Dict[str, Any]:
        """Analyze a single task."""
        task_path = self.data_dir / "training" / f"{task_id}.json"
        if not task_path.exists():
            task_path = self.data_dir / "evaluation" / f"{task_id}.json"

        if not task_path.exists():
            return {"error": f"Task {task_id} not found"}

        with open(task_path) as f:
            task = json.load(f)

        # Analyze all training examples
        transformations = []
        for example in task["train"]:
            input_grid = np.array(example["input"])
            output_grid = np.array(example["output"])

            transformation = self.analyze_transformation(input_grid, output_grid)
            transformations.append(transformation)

        # Try to solve with our current system
        train_examples = [
            (np.array(ex["input"]), np.array(ex["output"])) for ex in task["train"]
        ]
        test_input = np.array(task["test"][0]["input"])

        solution = self.solver.solve(train_examples, test_input)

        # Categorize failure mode
        failure_mode = self.categorize_failure(
            transformations, solution.method_used, solution.confidence
        )

        return {
            "task_id": task_id,
            "transformations": transformations,
            "method_tried": solution.method_used,
            "confidence": solution.confidence,
            "failure_mode": failure_mode,
            "missing_capabilities": self.identify_missing_capabilities(transformations),
        }

    def categorize_failure(
        self, transformations: List[Dict], method_used: str, confidence: float
    ) -> str:
        """Categorize the failure mode."""
        # Check what patterns were present
        patterns = set()
        for t in transformations:
            if t["arithmetic_operation"]:
                patterns.add("arithmetic")
            if t["conditional_logic"]:
                patterns.add("conditional")
            if t["spatial_pattern"]:
                patterns.add("spatial")
            if t["structural_change"]:
                patterns.add("structural")
            if t["object_count_change"] != 0:
                patterns.add("counting")

        if "arithmetic" in patterns:
            return "missing_arithmetic"
        elif "conditional" in patterns:
            return "missing_conditional_logic"
        elif "spatial" in patterns:
            return "missing_spatial_reasoning"
        elif "counting" in patterns:
            return "missing_counting"
        elif "structural" in patterns:
            return "missing_structural_ops"
        elif confidence < 0.5:
            return "low_confidence_pattern"
        else:
            return "unknown_pattern"

    def identify_missing_capabilities(self, transformations: List[Dict]) -> List[str]:
        """Identify specific missing capabilities."""
        missing = set()

        for t in transformations:
            # Check arithmetic
            if t["arithmetic_operation"]:
                missing.add(f"arithmetic:{t['arithmetic_operation']}")

            # Check conditional
            if t["conditional_logic"]:
                for cond in t["conditional_logic"].split(","):
                    missing.add(f"conditional:{cond}")

            # Check spatial
            if t["spatial_pattern"]:
                for pattern in t["spatial_pattern"].split(","):
                    missing.add(f"spatial:{pattern}")

            # Check structural
            if t["structural_change"]:
                for change in t["structural_change"].split(","):
                    missing.add(f"structural:{change}")

            # Check counting
            if t["object_count_change"] != 0:
                if t["object_count_change"] > 0:
                    missing.add("counting:duplication")
                else:
                    missing.add("counting:reduction")

        return sorted(list(missing))

    def analyze_all_failed_tasks(self, limit: int = 50):
        """Analyze all failed tasks."""
        # Get list of training tasks
        training_tasks = sorted(
            [f.stem for f in (self.data_dir / "training").glob("*.json")]
        )

        if not training_tasks:
            print("No training tasks found!")
            return

        print(f"Found {len(training_tasks)} training tasks")
        print(f"Analyzing up to {limit} tasks...")
        print("=" * 60)

        results = []
        failure_categories = Counter()
        all_missing_capabilities = Counter()

        for i, task_id in enumerate(training_tasks[:limit]):
            print(
                f"\nAnalyzing task {i+1}/{min(limit, len(training_tasks))}: {task_id}"
            )

            result = self.analyze_task(task_id)
            results.append(result)

            if "error" not in result:
                failure_categories[result["failure_mode"]] += 1
                for cap in result["missing_capabilities"]:
                    all_missing_capabilities[cap] += 1

                print(f"  Failure mode: {result['failure_mode']}")
                print(f"  Method tried: {result['method_tried']}")
                print(f"  Confidence: {result['confidence']:.3f}")
                if result["missing_capabilities"]:
                    print(f"  Missing: {', '.join(result['missing_capabilities'][:3])}")

        # Generate report
        self.generate_report(results, failure_categories, all_missing_capabilities)

    def generate_report(
        self,
        results: List[Dict],
        failure_categories: Counter,
        missing_capabilities: Counter,
    ):
        """Generate comprehensive failure analysis report."""
        report_path = Path("ARC_FAILURE_ANALYSIS.md")

        with open(report_path, "w") as f:
            f.write("# ARC-AGI Failure Analysis Report\n\n")
            f.write(f"Analyzed {len(results)} tasks\n\n")

            # Failure categories
            f.write("## Failure Categories\n\n")
            total = sum(failure_categories.values())
            for category, count in failure_categories.most_common():
                percentage = (count / total) * 100 if total > 0 else 0
                f.write(f"- **{category}**: {count} tasks ({percentage:.1f}%)\n")

            # Missing capabilities
            f.write("\n## Top Missing Capabilities\n\n")
            for capability, count in missing_capabilities.most_common(20):
                f.write(f"- `{capability}`: {count} occurrences\n")

            # Detailed breakdown
            f.write("\n## Capability Categories\n\n")

            # Group by category
            cap_groups = defaultdict(list)
            for cap, count in missing_capabilities.items():
                category = cap.split(":")[0]
                operation = cap.split(":")[1] if ":" in cap else cap
                cap_groups[category].append((operation, count))

            for category in [
                "arithmetic",
                "conditional",
                "spatial",
                "counting",
                "structural",
            ]:
                if category in cap_groups:
                    f.write(f"### {category.capitalize()} Operations\n\n")
                    for op, count in sorted(cap_groups[category], key=lambda x: -x[1]):
                        f.write(f"- {op}: {count} tasks\n")
                    f.write("\n")

            # Recommendations
            f.write("## Recommendations for DSL Enhancement\n\n")
            f.write("Based on this analysis, we should add:\n\n")

            f.write("### 1. Arithmetic Primitives\n")
            f.write("- `AddConstant(value)`: Add constant to all non-zero colors\n")
            f.write("- `CountObjects()`: Count objects and encode as color\n")
            f.write("- `MultiplyColors(factor)`: Scale color values\n\n")

            f.write("### 2. Conditional Logic\n")
            f.write("- `IfSize(threshold, then_op, else_op)`: Size-based conditions\n")
            f.write("- `IfColor(color, then_op)`: Color-based conditions\n")
            f.write("- `IfShape(shape_test, then_op)`: Shape-based conditions\n\n")

            f.write("### 3. Spatial Patterns\n")
            f.write("- `DrawDiagonal(color)`: Draw diagonal lines\n")
            f.write("- `DrawBorder(color, thickness)`: Draw borders\n")
            f.write("- `FillSpiral(start_color)`: Fill in spiral pattern\n")
            f.write("- `RepeatPattern(pattern, times)`: Repeat spatial pattern\n\n")

            f.write("### 4. Counting and Indexing\n")
            f.write("- `EnumerateObjects()`: Number objects sequentially\n")
            f.write("- `DuplicateNTimes(n)`: Controlled duplication\n")
            f.write("- `SelectNth(n)`: Select nth object\n\n")

            f.write("### 5. Structural Operations\n")
            f.write("- `MergeAdjacent()`: Merge touching objects\n")
            f.write("- `SplitByColor()`: Split multi-color objects\n")
            f.write("- `ConnectObjects(method)`: Connect objects with lines\n\n")

            # Task-specific insights
            f.write("\n## Task-Specific Insights\n\n")

            # Find interesting examples
            for result in results[:10]:
                if "error" not in result and result["missing_capabilities"]:
                    f.write(f"### Task {result['task_id']}\n")
                    f.write(f"- **Failure**: {result['failure_mode']}\n")
                    f.write(
                        f"- **Missing**: {', '.join(result['missing_capabilities'])}\n"
                    )

                    # Show specific transformation details
                    if result["transformations"]:
                        t = result["transformations"][0]
                        if t["arithmetic_operation"]:
                            f.write(f"- **Arithmetic**: {t['arithmetic_operation']}\n")
                        if t["spatial_pattern"]:
                            f.write(f"- **Spatial**: {t['spatial_pattern']}\n")
                        if t["conditional_logic"]:
                            f.write(f"- **Conditional**: {t['conditional_logic']}\n")
                    f.write("\n")

        print(f"\n{'=' * 60}")
        print(f"Report saved to: {report_path}")
        print(f"{'=' * 60}")


def main():
    """Run the failure analysis."""
    analyzer = FailedTaskAnalyzer()
    analyzer.analyze_all_failed_tasks(limit=50)


if __name__ == "__main__":
    main()
