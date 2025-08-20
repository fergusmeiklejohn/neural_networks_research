#!/usr/bin/env python3
"""Test specifically for cross pattern detection."""

from utils.imports import setup_project_paths

setup_project_paths()

import json
from pathlib import Path

import numpy as np
from automated_primitive_discovery_v2 import PrimitiveDiscovererV2


def test_cross_detection():
    """Test cross pattern detection specifically."""

    # Load task
    data_dir = Path("data/arc_agi_official/ARC-AGI/data/training")
    task_file = data_dir / "ae3edfdc.json"

    with open(task_file, "r") as f:
        task = json.load(f)

    # Get examples
    train_examples = [
        (np.array(ex["input"]), np.array(ex["output"])) for ex in task["train"]
    ]

    print("Testing Cross Pattern Detection")
    print("=" * 60)

    discoverer = PrimitiveDiscovererV2(verbose=False)

    # Test cross detection on first example
    inp, out = train_examples[0]

    print(f"Input shape: {inp.shape}")
    print(f"Output shape: {out.shape}")

    # Test the cross detection method directly
    crosses = discoverer._detect_cross_pattern(inp, out)

    if crosses:
        print(f"\n✅ Found {len(crosses)} cross patterns!")
        for i, cross in enumerate(crosses):
            print(f"\nCross {i+1}:")
            print(f"  Center: {cross['center']}")
            print(f"  Center color: {cross['center_color']}")
            print(f"  Cross color: {cross['cross_color']}")
            print(f"  H markers: {cross['h_markers']}")
            print(f"  V markers: {cross['v_markers']}")
    else:
        print("\n❌ No crosses detected")

    # Now test full pattern extraction
    print("\n" + "=" * 60)
    print("Testing Full Pattern Extraction:")

    patterns = discoverer._extract_patterns([train_examples[0]])

    print(f"\nFound {len(patterns)} patterns:")
    for p in patterns:
        print(f"  - Type: {p['type']}")
        if p["type"] == "spatial" and p["data"]:
            print(f"    Pattern: {p['data'].get('pattern', 'unknown')}")
            if "details" in p["data"]:
                print(f"    Details: {len(p['data']['details'])} items")

    # Test with all examples
    print("\n" + "=" * 60)
    print("Testing Pattern Consistency Across All Examples:")

    all_patterns = []
    for i, (inp, out) in enumerate(train_examples):
        crosses = discoverer._detect_cross_pattern(inp, out)
        if crosses:
            print(f"  Example {i+1}: {len(crosses)} crosses")
            all_patterns.append(crosses)
        else:
            print(f"  Example {i+1}: No crosses")

    if all_patterns:
        print(
            f"\n✅ Cross patterns found in {len(all_patterns)}/{len(train_examples)} examples"
        )
    else:
        print("\n❌ No cross patterns found in any example")


if __name__ == "__main__":
    test_cross_detection()
