"""Test our discovery system on a larger, random sample of ARC tasks.

This will reveal if we're overfitting to our small test set.
"""

import json
import random
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Import our best discoverers
from automated_primitive_discovery_v9_final import PrimitiveDiscovererV9Final
from automated_primitive_discovery_v11_enhanced import PrimitiveDiscovererV11
from hierarchical_pattern_detector import HierarchicalPatternDetector
from scipy import ndimage


class ComprehensiveDiscoverer:
    """Combines all our pattern detection capabilities."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.v9_discoverer = PrimitiveDiscovererV9Final(
            verbose=False, accuracy_threshold=0.75
        )
        self.v11_discoverer = PrimitiveDiscovererV11(
            verbose=False, accuracy_threshold=0.75
        )
        self.hierarchical = HierarchicalPatternDetector(verbose=False)

    def discover(
        self, task_id: str, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Tuple[bool, str, float]:
        """Try all discovery methods."""

        # Try hierarchical first (often simpler)
        result = self.hierarchical.detect_hierarchical_patterns(task_id, examples)
        if result:
            accuracy = self._test_primitive(result, examples)
            if accuracy >= 0.75:
                return True, "hierarchical", accuracy

        # Try v11 patterns (has the new patterns)
        result = self.v11_discoverer.discover_primitive(task_id, examples)
        if result:
            accuracy = self.v11_discoverer._test_primitive(result, examples)
            if accuracy >= 0.75:
                return True, "v11_patterns", accuracy

        # Try v9 patterns (comprehensive base patterns)
        result = self.v9_discoverer.discover_primitive(task_id, examples)
        if result:
            return True, "v9_patterns", 1.0  # v9 already tests internally

        return False, "none", 0.0

    def _test_primitive(
        self, code: str, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> float:
        """Test generated code."""
        try:
            namespace = {"np": np, "ndimage": ndimage}
            exec(code, namespace)

            # Find the class
            class_name = None
            for name in namespace:
                if name[0].isupper() and "_" in name:
                    class_name = name
                    break

            if not class_name:
                return 0.0

            primitive = namespace[class_name]()
            correct = 0

            for inp, out in examples:
                try:
                    pred = primitive.execute(inp)
                    if np.array_equal(pred, out):
                        correct += 1
                except:
                    pass

            return correct / len(examples)
        except:
            return 0.0


def get_all_training_tasks():
    """Get list of all training tasks."""
    data_dir = Path("data/arc_agi_official/ARC-AGI/data/training")
    task_files = list(data_dir.glob("*.json"))
    return [f.stem for f in task_files]


def test_on_random_sample(sample_size: int = 50, seed: int = 42):
    """Test on random sample of tasks."""

    print("=" * 60)
    print(f"TESTING ON {sample_size} RANDOM ARC TASKS")
    print("=" * 60)
    print("This will reveal true performance without overfitting")
    print("=" * 60)

    # Get all tasks and sample randomly
    all_tasks = get_all_training_tasks()
    print(f"Total training tasks available: {len(all_tasks)}")

    # Set seed for reproducibility
    random.seed(seed)

    # Sample tasks (excluding our test set to be fair)
    test_set = {
        "ae3edfdc",
        "00d62c1b",
        "0ca9ddb6",
        "ed36ccf7",
        "68b16354",
        "32597951",
        "045e512c",
        "05f2a901",
        "42a50994",
        "1cf80156",
        "25ff71a9",
        "3aa6fb7a",
        "a416b8f3",
        "007bbfb7",
    }

    available_tasks = [t for t in all_tasks if t not in test_set]
    sampled_tasks = random.sample(
        available_tasks, min(sample_size, len(available_tasks))
    )

    print(f"Sampled {len(sampled_tasks)} tasks (excluding our test set)")
    print(f"Random seed: {seed}")
    print()

    # Test each task
    discoverer = ComprehensiveDiscoverer(verbose=False)
    data_dir = Path("data/arc_agi_official/ARC-AGI/data/training")

    results = {
        "success": [],
        "failure": [],
        "pattern_types": {
            "hierarchical": 0,
            "v11_patterns": 0,
            "v9_patterns": 0,
            "none": 0,
        },
        "accuracies": [],
    }

    start_time = time.time()

    for i, task_id in enumerate(sampled_tasks):
        if i % 10 == 0:
            print(f"Progress: {i}/{len(sampled_tasks)} tasks...")

        try:
            with open(data_dir / f"{task_id}.json", "r") as f:
                task = json.load(f)

            examples = [
                (np.array(e["input"]), np.array(e["output"])) for e in task["train"]
            ]

            # Try discovery
            success, pattern_type, accuracy = discoverer.discover(task_id, examples)

            if success:
                results["success"].append(task_id)
                results["accuracies"].append(accuracy)
                results["pattern_types"][pattern_type] += 1
                if discoverer.verbose:
                    print(f"  ✓ {task_id}: {pattern_type} ({accuracy:.1%})")
            else:
                results["failure"].append(task_id)
                results["pattern_types"]["none"] += 1
                if discoverer.verbose:
                    print(f"  ✗ {task_id}: no pattern found")

        except Exception as e:
            results["failure"].append(task_id)
            results["pattern_types"]["none"] += 1
            if discoverer.verbose:
                print(f"  ✗ {task_id}: error - {e}")

    elapsed = time.time() - start_time

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    success_rate = len(results["success"]) / len(sampled_tasks) * 100
    print(
        f"Success rate: {len(results['success'])}/{len(sampled_tasks)} = {success_rate:.1f}%"
    )

    if results["accuracies"]:
        avg_accuracy = np.mean(results["accuracies"]) * 100
        print(f"Average accuracy when successful: {avg_accuracy:.1f}%")

    print(f"\nPattern type distribution:")
    for ptype, count in results["pattern_types"].items():
        pct = count / len(sampled_tasks) * 100
        print(f"  {ptype}: {count} ({pct:.1f}%)")

    print(f"\nTime taken: {elapsed:.1f} seconds")
    print(f"Average per task: {elapsed/len(sampled_tasks):.2f} seconds")

    # Show some failures for analysis
    print("\n" + "=" * 60)
    print("SAMPLE OF FAILED TASKS (for analysis)")
    print("=" * 60)

    for task_id in results["failure"][:5]:
        print(f"- {task_id}")

    if len(results["failure"]) > 5:
        print(f"... and {len(results['failure']) - 5} more")

    print("\n" + "=" * 60)
    print("COMPARISON WITH CLAIMED PERFORMANCE")
    print("=" * 60)
    print(f"Claimed on test set: 85-100%")
    print(f"Actual on random sample: {success_rate:.1f}%")

    if success_rate < 70:
        print("\n⚠️ WARNING: Significant overfitting detected!")
        print("The system was too specialized to the test set.")
    elif success_rate < 80:
        print("\n⚠️ Some overfitting detected, but reasonable generalization")
    else:
        print("\n✅ Good generalization! Performance holds on random tasks")

    return results


def analyze_failure_patterns(failed_tasks: List[str], sample_size: int = 3):
    """Analyze a few failed tasks to understand what patterns we're missing."""

    print("\n" + "=" * 60)
    print("FAILURE ANALYSIS")
    print("=" * 60)

    data_dir = Path("data/arc_agi_official/ARC-AGI/data/training")

    for task_id in failed_tasks[:sample_size]:
        print(f"\nTask {task_id}:")

        try:
            with open(data_dir / f"{task_id}.json", "r") as f:
                task = json.load(f)

            example = task["train"][0]
            inp = np.array(example["input"])
            out = np.array(example["output"])

            print(f"  Input shape: {inp.shape}")
            print(f"  Output shape: {out.shape}")
            print(f"  Size change: {inp.shape != out.shape}")

            # Basic analysis
            if inp.shape != out.shape:
                print(f"  Transformation type: Size transformation")
            else:
                # Check what changed
                diff = np.sum(inp != out)
                pct_changed = diff / inp.size * 100
                print(f"  Pixels changed: {diff}/{inp.size} ({pct_changed:.1f}%)")

            print(f"  Input colors: {np.unique(inp).tolist()}")
            print(f"  Output colors: {np.unique(out).tolist()}")

        except Exception as e:
            print(f"  Error analyzing: {e}")


if __name__ == "__main__":
    # Test on progressively larger samples
    for sample_size in [30, 50]:
        print(f"\n{'='*60}")
        print(f"TESTING WITH {sample_size} TASKS")
        print(f"{'='*60}\n")

        results = test_on_random_sample(sample_size=sample_size, seed=42)

        # Analyze some failures
        if results["failure"]:
            analyze_failure_patterns(results["failure"], sample_size=3)

        # If performance is very bad, stop early
        success_rate = len(results["success"]) / sample_size * 100
        if success_rate < 20:
            print("\n⚠️ Performance too low, stopping further tests")
            break
