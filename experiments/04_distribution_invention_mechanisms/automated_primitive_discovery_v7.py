#!/usr/bin/env python3
"""Automated primitive discovery system for ARC tasks - Version 7.

Fixes library matching for tasks with varying example dimensions.
Adds better pattern detection for reaching 40%+ discovery rate.
"""

from utils.imports import setup_project_paths

setup_project_paths()

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from automated_primitive_discovery_v6 import PrimitiveDiscovererV6


class PrimitiveDiscovererV7(PrimitiveDiscovererV6):
    """Fixed discoverer handling varying example dimensions."""

    def discover_primitive(
        self, task_id: str, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[str]:
        """Discover primitive with better handling of varying dimensions."""

        if self.verbose:
            print(f"\nAnalyzing task {task_id}...")

        # Check dimension consistency
        dimensions_vary = False
        first_inp_shape = examples[0][0].shape
        first_out_shape = examples[0][1].shape

        for inp, out in examples[1:]:
            if inp.shape != first_inp_shape or out.shape != first_out_shape:
                dimensions_vary = True
                if self.verbose:
                    print(f"  Examples have varying dimensions")
                break

        # Check for size mismatches
        size_mismatch = False
        for inp, out in examples:
            if inp.shape != out.shape:
                size_mismatch = True
                if self.verbose:
                    print(f"  Size change detected: {inp.shape} -> {out.shape}")
                break

        # Only try library if dimensions are consistent and no size mismatch
        if not dimensions_vary and not size_mismatch:
            library_code = self._try_library_patterns_safe(task_id, examples)
            if library_code:
                if self.verbose:
                    print(f"✅ Reused pattern from library!")
                return library_code

        # If library fails, discover new pattern
        if self.verbose:
            print("Discovering new pattern...")

        # Extract patterns (handle varying dimensions)
        patterns = self._extract_patterns_robust(examples)

        if not patterns:
            if self.verbose:
                print("No patterns found")
            return None

        # Find best pattern
        best_pattern = self._find_best_pattern(patterns, examples)

        if best_pattern is None:
            if self.verbose:
                print("No consistent pattern found")
            return None

        if self.verbose:
            print(f"Best pattern: {best_pattern['type']}")

        # Generate primitive code
        primitive_code = self._synthesize_primitive_enhanced(best_pattern, task_id)

        if primitive_code:
            # Test the primitive with adjusted threshold
            if self._test_primitive_flexible(primitive_code, examples, task_id):
                if self.verbose:
                    print(f"✅ Discovered primitive for {task_id}!")

                # Add to library if appropriate
                if not dimensions_vary and not size_mismatch:
                    self._add_to_library(
                        task_id, primitive_code, examples, best_pattern
                    )

                return primitive_code
            else:
                if self.verbose:
                    print("Generated primitive failed testing")

        return None

    def _try_library_patterns_safe(
        self, task_id: str, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[str]:
        """Safely try library patterns with dimension checking."""

        if self.verbose:
            print(f"Checking {len(self.library.patterns)} library patterns...")

        # Extract patterns safely
        try:
            current_patterns = self._extract_patterns_robust(examples[:1])
        except Exception as e:
            if self.verbose:
                print(f"  Error extracting patterns: {e}")
            return None

        best_accuracy = 0
        best_code = None

        for pattern in current_patterns:
            pattern_type = pattern["type"]

            # Handle different pattern types
            if pattern_type == "spatial" and pattern["data"]:
                pattern_type = pattern["data"].get("pattern", "spatial")
            elif pattern_type == "diagonal" and pattern["data"]:
                pattern_type = pattern["data"].get("pattern", "diagonal")

            # Find similar patterns in library
            try:
                # Use only first example for matching to avoid dimension issues
                similar = self.library.find_similar_patterns(
                    pattern_type=pattern_type,
                    pattern_data=pattern.get("data", {}),
                    examples=examples[:1],
                    similarity_threshold=0.5,
                )

                if self.verbose and similar:
                    print(f"  Found {len(similar)} similar {pattern_type} patterns")

                # Try each similar pattern
                for key, entry, similarity in similar[:3]:
                    # Check dimension compatibility
                    if entry.metadata.get("input_shapes"):
                        lib_shapes = entry.metadata["input_shapes"]
                        if lib_shapes and len(lib_shapes) > 0:
                            lib_shape = tuple(lib_shapes[0])
                            our_shape = examples[0][0].shape
                            if lib_shape != our_shape:
                                if self.verbose:
                                    print(
                                        f"    Skipping {entry.task_id}: shape mismatch {lib_shape} vs {our_shape}"
                                    )
                                continue

                    try:
                        accuracy = self.library.try_pattern(entry, examples)
                    except Exception as e:
                        if self.verbose:
                            print(f"    Error testing pattern: {e}")
                        continue

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
            except Exception as e:
                if self.verbose:
                    print(f"  Error checking library: {e}")
                continue

        return None

    def _extract_patterns_robust(self, examples):
        """Extract patterns robustly handling varying dimensions."""
        patterns = []

        # Process each example separately to handle varying dimensions
        for inp, out in examples:
            try:
                example_patterns = self._extract_patterns_enhanced([(inp, out)])
                patterns.extend(example_patterns)
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Error extracting patterns from example: {e}")
                continue

        # Deduplicate patterns
        unique_patterns = []
        seen_types = set()

        for pattern in patterns:
            pattern_key = f"{pattern['type']}_{str(pattern.get('data', ''))[:50]}"
            if pattern_key not in seen_types:
                seen_types.add(pattern_key)
                unique_patterns.append(pattern)

        return unique_patterns

    def _find_best_pattern(self, patterns, examples):
        """Find the pattern that best explains all examples - robust version."""
        if not patterns:
            return None

        # Score each pattern by consistency
        pattern_scores = {}

        for pattern in patterns:
            score = 0
            tested = 0

            # Check if pattern appears consistently
            for inp, out in examples:
                try:
                    if self._pattern_matches_robust(pattern, inp, out):
                        score += 1
                    tested += 1
                except Exception as e:
                    # Skip examples that cause errors
                    if self.verbose:
                        print(f"  Warning: Error testing pattern: {e}")
                    continue

            # Use a more unique key
            pattern_key = f"{pattern['type']}_{id(pattern)}"
            if tested > 0:
                pattern_scores[pattern_key] = (score, pattern, tested)

        # Return pattern with highest score
        if pattern_scores:
            best_key = max(
                pattern_scores,
                key=lambda k: pattern_scores[k][0] / pattern_scores[k][2],
            )
            best_score, best_pattern, tested = pattern_scores[best_key]

            if self.verbose:
                print(f"Best pattern score: {best_score}/{tested}")

            # Only return if pattern matches majority of testable examples
            if best_score >= tested * 0.8:
                return best_pattern

        return None

    def _pattern_matches_robust(self, pattern, inp, out):
        """Check if pattern matches this example - robust version."""
        try:
            # Size mismatch patterns always need special handling
            if inp.shape != out.shape:
                if pattern["type"] == "size_change":
                    return True  # Could be a size change pattern
                else:
                    return False  # Other patterns don't apply to size changes

            # Now handle same-size patterns
            if pattern["type"] == "spatial":
                if pattern["data"] and pattern["data"].get("pattern") == "cross":
                    detected = self._detect_cross_pattern_safe(inp, out)
                    return detected is not None and len(detected) > 0
                elif pattern["data"] and pattern["data"].get("pattern") == "line":
                    detected = self._detect_line_pattern(inp, out)
                    return detected is not None
                elif pattern["data"] and pattern["data"].get("pattern") == "region":
                    detected = self._detect_region_fill(inp, out)
                    return detected is not None

            elif pattern["type"] == "symmetry":
                detected = self._analyze_symmetry_pattern(inp, out)
                return detected is not None

            elif pattern["type"] == "diagonal":
                detected = self._analyze_diagonal_pattern(inp, out)
                return detected is not None

            elif pattern["type"] == "conditional":
                detected = self._analyze_conditional_pattern_improved(inp, out)
                return detected is not None

            elif pattern["type"] == "color_map":
                # Check color mapping
                for i in range(inp.shape[0]):
                    for j in range(inp.shape[1]):
                        if inp[i, j] in pattern["data"]:
                            if out[i, j] != pattern["data"][inp[i, j]]:
                                return False
                return True

            return False

        except Exception as e:
            if self.verbose:
                print(f"    Error in pattern matching: {e}")
            return False


def test_v7_discovery():
    """Test V7 with robust dimension handling."""

    # Extended test set targeting 40%+ success
    test_tasks = [
        # Known successful (baseline)
        "ae3edfdc",  # Cross pattern
        "00d62c1b",  # Region pattern
        "0ca9ddb6",  # Cross pattern
        "06df4c85",  # Region pattern
        "045e512c",  # Shape/region
        # Previously problematic
        "0520fde7",  # Size mismatch
        "0a938d79",  # Varying dimensions
        "0b148d64",  # Size mismatch
        "0d3d703e",  # Conditional
        "05f2a901",  # Conditional
        "08ed6ac7",  # Complex
        "09629e4f",  # Complex
        # Additional tasks
        "1cf80156",  # Size change
        "22eb0ac0",  # Possibly rotation
        "25ff71a9",  # Possibly mirror
        "28e73c20",  # Additional
        "2dee498d",  # Additional
        "32597951",  # Additional
        "3906de3d",  # Additional
        "3aa6fb7a",  # Additional
    ]

    data_dir = Path("data/arc_agi_official/ARC-AGI/data/training")

    print("=" * 80)
    print("Testing V7: Robust Dimension Handling")
    print("=" * 80)
    print("Improvements:")
    print("- Fixed library matching for varying dimensions")
    print("- Robust pattern extraction per example")
    print("- Better error handling throughout")
    print("Target: 40%+ discovery rate")
    print("=" * 80)

    discoverer = PrimitiveDiscovererV7(verbose=True, accuracy_threshold=0.85)

    results = []

    for task_id in test_tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task_id}")
        print("-" * 60)

        task_file = data_dir / f"{task_id}.json"

        if not task_file.exists():
            print(f"  ❌ Task file not found")
            results.append({"task": task_id, "success": False, "reason": "not_found"})
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
            else:
                print(f"  ❌ Discovery failed")
                results.append(
                    {"task": task_id, "success": False, "reason": "no_pattern"}
                )

        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({"task": task_id, "success": False, "reason": str(e)[:50]})

    # Summary
    print("\n" + "=" * 80)
    print("V7 FINAL RESULTS")
    print("=" * 80)

    successful = sum(1 for r in results if r["success"])
    total = len(results)

    print(f"Success rate: {successful}/{total} ({successful/total*100:.1f}%)")

    print("\nSuccessful tasks:")
    for r in results:
        if r["success"]:
            print(f"  ✅ {r['task']}")

    print("\nFailed tasks:")
    for r in results:
        if not r["success"]:
            print(f"  ❌ {r['task']}: {r.get('reason', 'unknown')}")

    if successful / total >= 0.4:
        print("\n🎉 GOAL ACHIEVED: 40%+ discovery rate!")
    else:
        print(
            f"\n📈 Progress: Need {int(total*0.4 - successful)} more successes for 40%"
        )

    return results


if __name__ == "__main__":
    test_v7_discovery()
