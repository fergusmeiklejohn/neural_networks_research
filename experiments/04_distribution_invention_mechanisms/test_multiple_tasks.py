#!/usr/bin/env python3
"""Test automated primitive discovery on multiple ARC tasks."""

from utils.imports import setup_project_paths

setup_project_paths()

import json
import time
from pathlib import Path

import numpy as np
from automated_primitive_discovery_v2 import PrimitiveDiscovererV2


def test_multiple_tasks():
    """Test discovery on multiple tasks from failed synthesis analysis."""

    # Tasks identified in failed synthesis analysis
    test_tasks = [
        "ae3edfdc",  # Cross pattern - already working
        "00d62c1b",  # Line drawing
        "0520fde7",  # Conditional fill
        "05269061",  # Object manipulation
        "05f2a901",  # Color mapping
        "0692e18c",  # Region filling
        "08ed6ac7",  # Pattern propagation (V7 solved this)
        "09629e4f",  # Object counting
        "0a938d79",  # Spatial transformation
    ]

    data_dir = Path("data/arc_agi_official/ARC-AGI/data/training")

    results = []
    discoverer = PrimitiveDiscovererV2(verbose=False)

    print("=" * 80)
    print("Testing Automated Primitive Discovery on Multiple Tasks")
    print("=" * 80)

    for task_id in test_tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task_id}")
        print("-" * 60)

        task_file = data_dir / f"{task_id}.json"

        if not task_file.exists():
            print(f"  ❌ Task file not found")
            results.append({"task": task_id, "status": "file_not_found", "accuracy": 0})
            continue

        try:
            with open(task_file, "r") as f:
                task = json.load(f)

            train_examples = [
                (np.array(ex["input"]), np.array(ex["output"])) for ex in task["train"]
            ]

            print(f"  Examples: {len(train_examples)}")
            print(f"  Grid sizes: {train_examples[0][0].shape}")

            # Try discovery
            start_time = time.time()
            discovered_code = discoverer.discover_primitive(task_id, train_examples)
            discovery_time = time.time() - start_time

            if discovered_code:
                print(f"  ✅ Successfully discovered primitive!")
                print(f"  Discovery time: {discovery_time:.2f}s")

                # Extract pattern type from code
                if "CrossPattern" in discovered_code:
                    pattern_type = "cross"
                elif "ColorMap" in discovered_code:
                    pattern_type = "color_map"
                elif "Line" in discovered_code:
                    pattern_type = "line"
                elif "Region" in discovered_code:
                    pattern_type = "region"
                elif "Conditional" in discovered_code:
                    pattern_type = "conditional"
                else:
                    pattern_type = "unknown"

                print(f"  Pattern type: {pattern_type}")

                # Save discovered primitive
                output_file = f"discovered_{task_id}.py"
                with open(output_file, "w") as f:
                    f.write("#!/usr/bin/env python3\n")
                    f.write(
                        "from compositional_dsl import Primitive, ExecutionContext\n"
                    )
                    f.write("import numpy as np\n\n")
                    f.write(discovered_code)
                print(f"  Saved to: {output_file}")

                results.append(
                    {
                        "task": task_id,
                        "status": "success",
                        "pattern_type": pattern_type,
                        "time": discovery_time,
                    }
                )
            else:
                print(f"  ❌ Failed to discover primitive")

                # Analyze why it failed
                patterns = discoverer._extract_patterns(train_examples[:1])
                if patterns:
                    print(f"  Patterns found but not consistent:")
                    for p in patterns:
                        print(f"    - {p['type']}")
                else:
                    print(f"  No patterns detected")

                results.append(
                    {
                        "task": task_id,
                        "status": "failed",
                        "patterns_found": len(patterns) if patterns else 0,
                    }
                )

        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({"task": task_id, "status": "error", "error": str(e)})

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    errors = [r for r in results if r["status"] in ["error", "file_not_found"]]

    print(f"\nTotal tasks tested: {len(test_tasks)}")
    print(
        f"✅ Successful discoveries: {len(successful)} ({len(successful)/len(test_tasks)*100:.1f}%)"
    )
    print(
        f"❌ Failed discoveries: {len(failed)} ({len(failed)/len(test_tasks)*100:.1f}%)"
    )
    print(f"⚠️ Errors: {len(errors)}")

    if successful:
        print("\nSuccessful discoveries:")
        for r in successful:
            print(
                f"  - {r['task']}: {r.get('pattern_type', 'unknown')} pattern ({r.get('time', 0):.2f}s)"
            )

    if failed:
        print("\nFailed discoveries:")
        for r in failed:
            print(
                f"  - {r['task']}: {r.get('patterns_found', 0)} patterns found but inconsistent"
            )

    return results


if __name__ == "__main__":
    results = test_multiple_tasks()
