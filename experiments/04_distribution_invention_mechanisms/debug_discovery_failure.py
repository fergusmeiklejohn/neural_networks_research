#!/usr/bin/env python3
"""Debug why discovery fails even with consistent patterns."""

from utils.imports import setup_project_paths

setup_project_paths()

import json
from pathlib import Path

import numpy as np
from automated_primitive_discovery_v2 import PrimitiveDiscovererV2


def debug_task(task_id):
    """Debug discovery process for a specific task."""

    print(f"Debugging Task: {task_id}")
    print("=" * 60)

    data_dir = Path("data/arc_agi_official/ARC-AGI/data/training")
    task_file = data_dir / f"{task_id}.json"

    with open(task_file, "r") as f:
        task = json.load(f)

    train_examples = [
        (np.array(ex["input"]), np.array(ex["output"])) for ex in task["train"]
    ]

    discoverer = PrimitiveDiscovererV2(verbose=False)

    # Step 1: Extract patterns
    print("\n1. PATTERN EXTRACTION:")
    patterns = discoverer._extract_patterns(train_examples)
    print(f"   Found {len(patterns)} patterns")

    # Step 2: Find best pattern
    print("\n2. FINDING BEST PATTERN:")
    best_pattern = discoverer._find_best_pattern(patterns, train_examples)

    if best_pattern:
        print(f"   Best pattern type: {best_pattern['type']}")
        if best_pattern["type"] == "spatial":
            print(
                f"   Spatial pattern: {best_pattern['data'].get('pattern', 'unknown')}"
            )

        # Step 3: Try synthesis
        print("\n3. PRIMITIVE SYNTHESIS:")
        primitive_code = discoverer._synthesize_primitive(best_pattern, task_id)

        if primitive_code:
            print("   ✅ Code generated successfully")
            print(f"   Code length: {len(primitive_code)} chars")

            # Show first few lines
            lines = primitive_code.strip().split("\n")[:5]
            for line in lines:
                print(f"     {line}")

            # Step 4: Test primitive
            print("\n4. TESTING PRIMITIVE:")
            success = discoverer._test_primitive(
                primitive_code, train_examples, task_id
            )

            if success:
                print("   ✅ Testing passed!")
            else:
                print("   ❌ Testing failed")

                # Try to run the code and see what happens
                print("\n   Debugging test failure:")
                try:
                    namespace = {
                        "Primitive": Primitive,
                        "ExecutionContext": ExecutionContext,
                        "np": np,
                    }
                    exec(primitive_code, namespace)

                    # Find the class
                    class_name = None
                    for name in namespace:
                        if name.startswith(("CrossPattern_", "ColorMap_", "Region_")):
                            class_name = name
                            break

                    if class_name:
                        PrimitiveClass = namespace[class_name]
                        primitive = PrimitiveClass()

                        # Test on first example
                        inp, expected = train_examples[0]
                        from compositional_dsl import ExecutionContext

                        context = ExecutionContext(
                            input_grid=inp.copy(), current_grid=inp.copy()
                        )

                        result_context = primitive.execute(context)
                        result = result_context.current_grid

                        accuracy = np.mean(result == expected)
                        print(f"     First example accuracy: {accuracy:.1%}")

                        if accuracy < 1.0:
                            # Show differences
                            diff_count = np.sum(result != expected)
                            print(f"     Differences: {diff_count} pixels")

                            # Show first few differences
                            diff_positions = np.argwhere(result != expected)[:5]
                            for pos in diff_positions:
                                i, j = pos
                                print(
                                    f"       ({i},{j}): {result[i,j]} should be {expected[i,j]}"
                                )
                    else:
                        print("     Could not find generated class")

                except Exception as e:
                    print(f"     Error executing: {e}")
        else:
            print("   ❌ Code generation failed")
            print(
                f"   Pattern type '{best_pattern['type']}' may not have implementation"
            )
    else:
        print("   ❌ No best pattern found")

        # Debug pattern scoring
        print("\n   Pattern scoring details:")

        for pattern in patterns[:5]:  # Check first 5
            score = 0
            for inp, out in train_examples:
                if discoverer._pattern_matches(pattern, inp, out):
                    score += 1

            print(f"     {pattern['type']}: {score}/{len(train_examples)} matches")

            # Show why it doesn't meet threshold
            threshold = len(train_examples) * 0.8
            if score < threshold:
                print(f"       (below threshold of {threshold:.1f})")


def main():
    # Debug the failed task
    debug_task("00d62c1b")

    print("\n" + "=" * 60)
    print("DIAGNOSIS:")
    print("=" * 60)

    print(
        """
The issue is likely one of:
1. Pattern matches but code generation not implemented for that type
2. Generated code doesn't correctly implement the pattern
3. Pattern scoring threshold too high (80%)
4. Multiple valid patterns confusing the selection
"""
    )


if __name__ == "__main__":
    from compositional_dsl import ExecutionContext, Primitive

    main()
