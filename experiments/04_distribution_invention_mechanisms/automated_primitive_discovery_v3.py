#!/usr/bin/env python3
"""Automated primitive discovery system for ARC tasks - Version 3.

Enhanced with pattern library integration for reuse and faster discovery.
"""

from utils.imports import setup_project_paths

setup_project_paths()

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from automated_primitive_discovery_v2 import PrimitiveDiscovererV2
from compositional_dsl import ExecutionContext, Primitive
from pattern_library import PatternLibrary


class PrimitiveDiscovererV3(PrimitiveDiscovererV2):
    """Enhanced discoverer with pattern library integration."""

    def __init__(
        self, verbose: bool = True, library_path: str = "arc_pattern_library.json"
    ):
        super().__init__(verbose)
        self.library = PatternLibrary(library_path)
        self.reuse_threshold = 0.85  # Minimum accuracy to reuse a pattern

    def discover_primitive(
        self, task_id: str, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[str]:
        """Discover primitive - first try library, then discover new."""

        if self.verbose:
            print(f"\nAnalyzing task {task_id}...")

        # Step 1: Try patterns from library first
        library_code = self._try_library_patterns(task_id, examples)
        if library_code:
            if self.verbose:
                print(f"✅ Reused pattern from library!")
            return library_code

        # Step 2: If library fails, discover new pattern
        if self.verbose:
            print("No suitable library pattern, discovering new...")

        # Use parent class discovery
        new_code = super().discover_primitive(task_id, examples)

        # Step 3: If successful, add to library
        if new_code:
            self._add_to_library(task_id, new_code, examples)

        return new_code

    def _try_library_patterns(
        self, task_id: str, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[str]:
        """Try patterns from the library."""

        if self.verbose:
            print(f"Checking {len(self.library.patterns)} library patterns...")

        # Extract patterns from current examples for matching
        current_patterns = self._extract_patterns(examples[:1])

        best_accuracy = 0
        best_code = None

        for pattern in current_patterns:
            pattern_type = pattern["type"]

            # Handle spatial patterns specially
            if pattern_type == "spatial" and pattern["data"]:
                pattern_type = pattern["data"].get("pattern", "spatial")

            # Find similar patterns in library
            similar = self.library.find_similar_patterns(
                pattern_type=pattern_type,
                pattern_data=pattern.get("data", {}),
                examples=examples,
                similarity_threshold=0.5,  # Lower threshold for initial search
            )

            if self.verbose and similar:
                print(f"  Found {len(similar)} similar {pattern_type} patterns")

            # Try each similar pattern
            for key, entry, similarity in similar[:3]:  # Try top 3
                accuracy = self.library.try_pattern(entry, examples)

                if accuracy and accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_code = entry.code_template

                    if self.verbose:
                        print(
                            f"    Pattern from {entry.task_id}: {accuracy:.1%} accuracy"
                        )

                    # If good enough, use it
                    if accuracy >= self.reuse_threshold:
                        # Adapt the code for new task
                        adapted_code = self._adapt_code_for_task(
                            best_code, task_id, entry.task_id
                        )
                        return adapted_code

        return None

    def _adapt_code_for_task(
        self, code: str, new_task_id: str, old_task_id: str
    ) -> str:
        """Adapt code from one task to another."""
        # Replace task ID in class names
        old_id_safe = old_task_id.replace("-", "_")
        new_id_safe = new_task_id.replace("-", "_")

        adapted = code.replace(old_id_safe, new_id_safe)
        adapted = adapted.replace(old_task_id, new_task_id)

        return adapted

    def _add_to_library(
        self, task_id: str, code: str, examples: List[Tuple[np.ndarray, np.ndarray]]
    ):
        """Add successful pattern to library."""

        # Determine pattern type from code
        if "CrossPattern" in code:
            pattern_type = "cross"
        elif "ColorMap" in code:
            pattern_type = "color_map"
        elif "RegionFill" in code:
            pattern_type = "region"
        elif "LinePattern" in code:
            pattern_type = "line"
        elif "ConditionalFill" in code:
            pattern_type = "conditional"
        else:
            pattern_type = "unknown"

        # Extract pattern data (simplified for now)
        pattern_data = {}

        # Calculate accuracy
        accuracy = self._calculate_accuracy(code, examples)

        # Add to library
        key = self.library.add_pattern(
            task_id=task_id,
            pattern_type=pattern_type,
            pattern_data=pattern_data,
            code_template=code,
            accuracy=accuracy,
            examples=examples,
        )

        if self.verbose:
            print(f"  Added to library as: {key}")

    def _calculate_accuracy(
        self, code: str, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> float:
        """Calculate accuracy of generated code."""
        try:
            namespace = {
                "Primitive": Primitive,
                "ExecutionContext": ExecutionContext,
                "np": np,
            }

            exec(code, namespace)

            # Find the class
            class_name = None
            for name in namespace:
                if name.startswith(
                    (
                        "CrossPattern_",
                        "ColorMap_",
                        "LinePattern_",
                        "RegionFill_",
                        "ConditionalFill_",
                    )
                ):
                    class_name = name
                    break

            if not class_name:
                return 0.0

            PrimitiveClass = namespace[class_name]
            primitive = PrimitiveClass()

            total_accuracy = 0
            for inp, expected in examples:
                context = ExecutionContext(
                    input_grid=inp.copy(), current_grid=inp.copy()
                )
                result_context = primitive.execute(context)
                result = result_context.current_grid

                accuracy = np.mean(result == expected)
                total_accuracy += accuracy

            return total_accuracy / len(examples)

        except:
            return 0.0


def test_with_library():
    """Test discovery with library integration."""

    # Test tasks - some should benefit from library
    test_tasks = [
        "ae3edfdc",  # Cross pattern - in library
        "00d62c1b",  # Region pattern - in library
        "05269061",  # Object manipulation - new
        "08ed6ac7",  # Pattern propagation - new
    ]

    data_dir = Path("data/arc_agi_official/ARC-AGI/data/training")

    print("=" * 80)
    print("Testing Primitive Discovery with Pattern Library")
    print("=" * 80)

    discoverer = PrimitiveDiscovererV3(verbose=True)

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
                results.append({"task": task_id, "success": True})
            else:
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


if __name__ == "__main__":
    test_with_library()
