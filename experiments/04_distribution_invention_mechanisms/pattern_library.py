#!/usr/bin/env python3
"""Pattern library system for storing and reusing discovered patterns.

This system allows us to:
1. Save successful patterns with metadata
2. Search for similar patterns
3. Reuse patterns on new tasks
4. Build a knowledge base of transformations
"""

from utils.imports import setup_project_paths

setup_project_paths()

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PatternEntry:
    """Entry in the pattern library."""

    task_id: str
    pattern_type: str  # cross, line, region, color_map, conditional
    pattern_data: dict  # Pattern-specific parameters
    code_template: str  # Generated Python code
    accuracy: float  # Test accuracy achieved
    examples_hash: str  # Hash of training examples for validation
    metadata: dict  # Additional info (grid sizes, colors used, etc.)


class PatternLibrary:
    """Manages a library of discovered patterns."""

    def __init__(self, library_path: str = "pattern_library.json"):
        self.library_path = Path(library_path)
        self.patterns = self._load_library()

    def _load_library(self) -> Dict[str, PatternEntry]:
        """Load existing pattern library."""
        if self.library_path.exists():
            with open(self.library_path, "r") as f:
                data = json.load(f)
                return {k: PatternEntry(**v) for k, v in data.items()}
        return {}

    def save_library(self):
        """Save pattern library to disk."""
        data = {k: asdict(v) for k, v in self.patterns.items()}
        with open(self.library_path, "w") as f:
            json.dump(data, f, indent=2)

    def add_pattern(
        self,
        task_id: str,
        pattern_type: str,
        pattern_data: dict,
        code_template: str,
        accuracy: float,
        examples: List[Tuple[np.ndarray, np.ndarray]],
    ) -> str:
        """Add a new pattern to the library."""
        # Create hash of examples for validation
        examples_str = str([(inp.tolist(), out.tolist()) for inp, out in examples])
        examples_hash = hashlib.md5(examples_str.encode()).hexdigest()[:8]

        # Extract metadata
        metadata = {
            "input_shapes": [inp.shape for inp, _ in examples],
            "output_shapes": [out.shape for _, out in examples],
            "input_colors": list(
                set().union(*[set(inp.flatten().tolist()) for inp, _ in examples])
            ),
            "output_colors": list(
                set().union(*[set(out.flatten().tolist()) for _, out in examples])
            ),
            "num_examples": len(examples),
        }

        # Create entry
        entry = PatternEntry(
            task_id=task_id,
            pattern_type=pattern_type,
            pattern_data=pattern_data,
            code_template=code_template,
            accuracy=accuracy,
            examples_hash=examples_hash,
            metadata=metadata,
        )

        # Generate unique key
        key = f"{task_id}_{pattern_type}_{examples_hash}"
        self.patterns[key] = entry

        self.save_library()
        return key

    def find_similar_patterns(
        self,
        pattern_type: str,
        pattern_data: dict,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        similarity_threshold: float = 0.8,
    ) -> List[Tuple[str, PatternEntry, float]]:
        """Find similar patterns in the library."""
        similar = []

        # Extract features from query
        query_metadata = {
            "input_shapes": [inp.shape for inp, _ in examples],
            "output_shapes": [out.shape for _, out in examples],
            "input_colors": set().union(
                *[set(inp.flatten().tolist()) for inp, _ in examples]
            ),
            "output_colors": set().union(
                *[set(out.flatten().tolist()) for _, out in examples]
            ),
        }

        for key, entry in self.patterns.items():
            if entry.pattern_type != pattern_type:
                continue

            # Calculate similarity score
            score = self._calculate_similarity(
                pattern_data,
                entry.pattern_data,
                query_metadata,
                entry.metadata,
                pattern_type,
            )

            if score >= similarity_threshold:
                similar.append((key, entry, score))

        # Sort by similarity score
        similar.sort(key=lambda x: x[2], reverse=True)
        return similar

    def _calculate_similarity(
        self,
        pattern_data1: dict,
        pattern_data2: dict,
        metadata1: dict,
        metadata2: dict,
        pattern_type: str,
    ) -> float:
        """Calculate similarity between two patterns."""
        scores = []

        # Type-specific similarity
        if pattern_type == "cross":
            # Compare center and marker colors
            if "center_colors" in pattern_data1 and "center_colors" in pattern_data2:
                colors1 = set(pattern_data1.get("center_colors", []))
                colors2 = set(pattern_data2.get("center_colors", []))
                if colors1 and colors2:
                    scores.append(len(colors1 & colors2) / len(colors1 | colors2))

        elif pattern_type == "color_map":
            # Compare color mappings
            if isinstance(pattern_data1, dict) and isinstance(pattern_data2, dict):
                keys1 = set(pattern_data1.keys())
                keys2 = set(pattern_data2.keys())
                if keys1 and keys2:
                    scores.append(len(keys1 & keys2) / len(keys1 | keys2))

        elif pattern_type == "region":
            # Compare boundary colors
            colors1 = set(pattern_data1.get("boundary_colors", []))
            colors2 = set(pattern_data2.get("boundary_colors", []))
            if colors1 and colors2:
                scores.append(len(colors1 & colors2) / len(colors1 | colors2))

        # Metadata similarity
        # Color overlap
        colors1 = set(metadata1.get("input_colors", []))
        colors2 = set(metadata2.get("input_colors", []))
        if colors1 and colors2:
            scores.append(len(colors1 & colors2) / len(colors1 | colors2))

        # Shape similarity (rough)
        shapes1 = metadata1.get("input_shapes", [])
        shapes2 = metadata2.get("input_shapes", [])
        if shapes1 and shapes2:
            # Check if any shapes match
            shape_match = any(s1 == s2 for s1 in shapes1 for s2 in shapes2)
            scores.append(1.0 if shape_match else 0.5)

        return np.mean(scores) if scores else 0.0

    def get_pattern_by_task(self, task_id: str) -> List[PatternEntry]:
        """Get all patterns for a specific task."""
        return [entry for entry in self.patterns.values() if entry.task_id == task_id]

    def get_patterns_by_type(self, pattern_type: str) -> List[PatternEntry]:
        """Get all patterns of a specific type."""
        return [
            entry
            for entry in self.patterns.values()
            if entry.pattern_type == pattern_type
        ]

    def try_pattern(
        self, pattern_entry: PatternEntry, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[float]:
        """Try a pattern from the library on new examples."""
        try:
            from compositional_dsl import ExecutionContext, Primitive

            # Execute the code template
            namespace = {
                "Primitive": Primitive,
                "ExecutionContext": ExecutionContext,
                "np": np,
            }

            exec(pattern_entry.code_template, namespace)

            # Find the generated class
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
                return None

            PrimitiveClass = namespace[class_name]
            primitive = PrimitiveClass()

            # Test on examples
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

        except Exception as e:
            print(f"Error trying pattern: {e}")
            return None

    def export_statistics(self) -> dict:
        """Export library statistics."""
        stats = {
            "total_patterns": len(self.patterns),
            "tasks_covered": len(set(p.task_id for p in self.patterns.values())),
            "pattern_types": {},
            "average_accuracy": (
                np.mean([p.accuracy for p in self.patterns.values()])
                if self.patterns
                else 0
            ),
        }

        # Count by type
        for pattern_type in ["cross", "line", "region", "color_map", "conditional"]:
            count = sum(
                1 for p in self.patterns.values() if p.pattern_type == pattern_type
            )
            stats["pattern_types"][pattern_type] = count

        return stats


def demonstrate_library():
    """Demonstrate the pattern library system."""
    print("Pattern Library System Demo")
    print("=" * 60)

    # Create library
    library = PatternLibrary("arc_pattern_library.json")

    # Check if we have any discovered patterns to add
    discovered_files = list(Path(".").glob("discovered_*.py"))

    if discovered_files:
        print(f"\nFound {len(discovered_files)} discovered patterns")

        for file_path in discovered_files[:3]:  # Add first 3
            # Extract task_id from filename
            task_id = file_path.stem.replace("discovered_", "")

            print(f"\nAdding pattern from {task_id}...")

            # Read the code
            with open(file_path, "r") as f:
                code = f.read()

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

            # Add to library (with dummy data for demo)
            key = library.add_pattern(
                task_id=task_id,
                pattern_type=pattern_type,
                pattern_data={},  # Would extract from actual discovery
                code_template=code.split("\n\n", 1)[1] if "\n\n" in code else code,
                accuracy=0.99,  # Placeholder
                examples=[],  # Would use actual examples
            )

            print(f"  Added as: {key}")

    # Show statistics
    stats = library.export_statistics()
    print("\n" + "=" * 60)
    print("Library Statistics:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")

    # Demonstrate similarity search
    if library.patterns:
        print("\n" + "=" * 60)
        print("Pattern Search Demo:")

        # Get a cross pattern if available
        cross_patterns = library.get_patterns_by_type("cross")
        if cross_patterns:
            print(f"\nFound {len(cross_patterns)} cross patterns")
            print(f"First cross pattern is from task: {cross_patterns[0].task_id}")

    print("\n" + "=" * 60)
    print("Library saved to: arc_pattern_library.json")


if __name__ == "__main__":
    demonstrate_library()
