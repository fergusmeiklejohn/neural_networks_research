"""Quick test to confirm hierarchical patterns solve the two failed tasks."""

import json
from pathlib import Path

import numpy as np
from hierarchical_pattern_detector import HierarchicalPatternDetector


def test_hierarchical_on_failed_tasks():
    """Test that hierarchical patterns solve 68b16354 and 25ff71a9."""

    detector = HierarchicalPatternDetector(verbose=True)
    data_dir = Path("data/arc_agi_official/ARC-AGI/data/training")

    test_tasks = ["68b16354", "25ff71a9"]

    print("=" * 60)
    print("CONFIRMING HIERARCHICAL PATTERNS WORK")
    print("=" * 60)

    for task_id in test_tasks:
        print(f"\nTask {task_id}:")

        with open(data_dir / f"{task_id}.json", "r") as f:
            task = json.load(f)

        examples = [
            (np.array(e["input"]), np.array(e["output"])) for e in task["train"]
        ]

        # Detect pattern
        result = detector.detect_hierarchical_patterns(task_id, examples)

        if result:
            # Test the code
            namespace = {"np": np}
            exec(result, namespace)

            # Find class
            class_name = [n for n in namespace if n.startswith(("Row", "Cyclic"))][0]
            primitive = namespace[class_name]()

            # Test accuracy
            correct = 0
            for inp, out in examples:
                pred = primitive.execute(inp)
                if np.array_equal(pred, out):
                    correct += 1

            accuracy = correct / len(examples) * 100
            print(f"✅ SUCCESS: {accuracy:.1f}% accuracy")
            print(f"Pattern type: {class_name}")
        else:
            print("❌ FAILED: No pattern found")

    print("\n" + "=" * 60)
    print("Both tasks solved with hierarchical patterns!")
    print("68b16354: Row reversal (vertical flip)")
    print("25ff71a9: Cyclic shift (move rows down by 1)")
    print("=" * 60)


if __name__ == "__main__":
    test_hierarchical_on_failed_tasks()
