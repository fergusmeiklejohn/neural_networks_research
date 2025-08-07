#!/usr/bin/env python3
"""Test pattern discovery to debug why patterns aren't being found."""

from utils.imports import setup_project_paths

setup_project_paths()

import numpy as np
from arc_test_time_adapter import ARCTestTimeAdapter
from neural_perception import NeuralPerceptionModule


def test_pattern_detection():
    """Test pattern detection on sample grids."""
    perception = NeuralPerceptionModule()

    print("=" * 60)
    print("TESTING PATTERN DETECTION")
    print("=" * 60)

    # Test 1: Checkerboard pattern
    print("\nTest 1: Checkerboard Pattern")
    checkerboard = np.array([[1, 2, 1, 2], [2, 1, 2, 1], [1, 2, 1, 2], [2, 1, 2, 1]])
    patterns = perception.detect_spatial_patterns(checkerboard)
    print(f"Detected {len(patterns)} patterns:")
    for p in patterns:
        print(
            f"  - {p.pattern_type}: confidence={p.confidence:.2f}, params={p.parameters}"
        )

    # Test 2: Symmetric pattern
    print("\nTest 2: Symmetric Pattern")
    symmetric = np.array([[1, 2, 3, 2, 1], [4, 5, 6, 5, 4], [1, 2, 3, 2, 1]])
    patterns = perception.detect_spatial_patterns(symmetric)
    print(f"Detected {len(patterns)} patterns:")
    for p in patterns:
        print(
            f"  - {p.pattern_type}: confidence={p.confidence:.2f}, params={p.parameters}"
        )

    # Test 3: Progression pattern
    print("\nTest 3: Progression Pattern")
    progression = np.array([[1, 1, 2, 2, 2, 3, 3, 3, 3], [1, 1, 2, 2, 2, 3, 3, 3, 3]])
    patterns = perception.detect_spatial_patterns(progression)
    print(f"Detected {len(patterns)} patterns:")
    for p in patterns:
        print(
            f"  - {p.pattern_type}: confidence={p.confidence:.2f}, params={p.parameters}"
        )

    # Test 4: Alternation pattern
    print("\nTest 4: Alternation Pattern")
    alternation = np.array([[1, 2, 1, 2, 1, 2], [3, 4, 3, 4, 3, 4]])
    patterns = perception.detect_spatial_patterns(alternation)
    print(f"Detected {len(patterns)} patterns:")
    for p in patterns:
        print(
            f"  - {p.pattern_type}: confidence={p.confidence:.2f}, params={p.parameters}"
        )

    print("\n" + "=" * 60)

    # Now test TTA adapter's pattern discovery
    print("\nTesting TTA Adapter Pattern Discovery")
    print("-" * 60)

    adapter = ARCTestTimeAdapter()

    # Create sample task with patterns
    examples = [
        (
            np.array([[1, 2, 1], [2, 1, 2], [1, 2, 1]]),  # Checkerboard-like
            np.array([[2, 1, 2], [1, 2, 1], [2, 1, 2]]),  # Inverted
        ),
        (np.array([[3, 4, 3], [4, 3, 4]]), np.array([[4, 3, 4], [3, 4, 3]])),
    ]

    # Extract initial rules
    initial_rules = adapter.extractor.extract_rules(examples)
    print(f"Initial rules: {len(initial_rules.transformations)} transformations")

    # Try to adapt with pattern discovery
    result = adapter.adapt(examples, initial_rules, max_steps=5)
    print(f"Adaptation result:")
    print(f"  - Steps: {result.adaptation_steps}")
    print(f"  - Discovered patterns: {result.discovered_patterns}")
    print(f"  - Final confidence: {result.confidence:.2f}")

    # Check what patterns were found in the intermediate steps
    print("\nDetailed pattern search:")
    all_patterns = []
    for input_grid, output_grid in examples:
        input_patterns = perception.detect_spatial_patterns(input_grid)
        output_patterns = perception.detect_spatial_patterns(output_grid)
        print(f"  Input patterns: {[p.pattern_type for p in input_patterns]}")
        print(f"  Output patterns: {[p.pattern_type for p in output_patterns]}")
        all_patterns.append((input_patterns, output_patterns))

    # Check what consistent patterns would be found
    consistent = adapter._find_consistent_patterns(all_patterns)
    print(f"\nConsistent patterns found: {list(consistent.keys())}")


if __name__ == "__main__":
    test_pattern_detection()
