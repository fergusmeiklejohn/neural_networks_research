"""Analyze why we're overfitting and what patterns we're missing."""

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np


def analyze_successful_vs_failed():
    """Compare what worked vs what didn't."""

    # Our test set that we "optimized" for
    test_set_tasks = [
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
    ]

    # Tasks that succeeded in random sample (from the test output)
    # These are estimated from the ~30-36% success rate
    random_successes = []  # We'd need to run again to get exact list

    print("=" * 60)
    print("OVERFITTING ANALYSIS")
    print("=" * 60)

    print("\n1. PERFORMANCE COMPARISON:")
    print(f"   Test set (14 tasks): ~85-100% success")
    print(f"   Random sample (50 tasks): ~36% success")
    print(f"   Overfitting factor: ~2.5x")

    print("\n2. WHAT THIS MEANS:")
    print("   - We unconsciously selected patterns that fit our test set")
    print("   - The test set is not representative of all ARC tasks")
    print("   - Many ARC patterns are still missing from our system")

    print("\n3. PATTERN DISTRIBUTION IN RANDOM SAMPLE:")
    print("   - v11_patterns (new): 26.7%")
    print("   - v9_patterns (original): 3.3%")
    print("   - hierarchical: 0.0% (!)")
    print("   - none (failed): 70.0%")

    print("\n4. KEY OBSERVATIONS:")
    print("   - Hierarchical patterns (row reversal, cyclic shift) are RARE")
    print("   - Most ARC tasks need patterns we haven't implemented")
    print("   - V11 patterns (bounding box, tiling, etc.) are more common")


def sample_more_failed_tasks(n=10):
    """Look at more failed tasks to understand missing patterns."""

    data_dir = Path("data/arc_agi_official/ARC-AGI/data/training")

    # Get all tasks
    all_tasks = list(data_dir.glob("*.json"))

    # Exclude our test set
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

    available = [f for f in all_tasks if f.stem not in test_set]
    random.seed(43)  # Different seed
    sampled = random.sample(available, n)

    print("\n" + "=" * 60)
    print(f"ANALYZING {n} MORE RANDOM TASKS")
    print("=" * 60)

    pattern_hints = []

    for task_file in sampled:
        task_id = task_file.stem
        with open(task_file, "r") as f:
            task = json.load(f)

        example = task["train"][0]
        inp = np.array(example["input"])
        out = np.array(example["output"])

        print(f"\n{task_id}:")
        print(f"  Input: {inp.shape}, Output: {out.shape}")

        # Analyze transformation type
        if inp.shape != out.shape:
            h_ratio = out.shape[0] / inp.shape[0] if inp.shape[0] > 0 else 0
            w_ratio = out.shape[1] / inp.shape[1] if inp.shape[1] > 0 else 0

            if h_ratio == w_ratio and h_ratio in [2, 3, 4]:
                pattern_hints.append("uniform_scaling")
            elif out.shape[0] < inp.shape[0] or out.shape[1] < inp.shape[1]:
                pattern_hints.append("extraction/cropping")
            else:
                pattern_hints.append("complex_size_change")
        else:
            # Same size - analyze changes
            changes = np.sum(inp != out)
            pct = changes / inp.size * 100

            if pct < 10:
                pattern_hints.append("sparse_modification")
            elif pct > 80:
                pattern_hints.append("major_transformation")
            else:
                pattern_hints.append("moderate_transformation")

            # Check for new colors
            inp_colors = set(np.unique(inp))
            out_colors = set(np.unique(out))
            if out_colors - inp_colors:
                pattern_hints.append("color_generation")

    print("\n" + "=" * 60)
    print("PATTERN HINTS FROM RANDOM SAMPLE")
    print("=" * 60)

    hint_counts = Counter(pattern_hints)
    for hint, count in hint_counts.most_common():
        print(f"  {hint}: {count}")


def recommendations():
    """Recommendations for improving the system."""

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS TO FIX OVERFITTING")
    print("=" * 60)

    print(
        """
1. EXPAND PATTERN LIBRARY:
   - We need 50+ more pattern types (not just 10-15)
   - Focus on the 70% that currently fail
   - Each pattern should be more general/flexible

2. IMPLEMENT FUZZY MATCHING:
   - Current patterns require exact matches
   - Real tasks have variations and noise
   - Need approximate pattern matching

3. USE LARGER DEVELOPMENT SET:
   - Current 14-task set is too small
   - Use 100+ tasks for development
   - Regularly test on held-out sets

4. PATTERN CATEGORIES WE'RE MISSING:
   - Object manipulation (move, copy, delete specific objects)
   - Counting and arithmetic patterns
   - Symmetry operations beyond simple flips
   - Conditional logic (if-then-else patterns)
   - Relative positioning (above, below, left-of, right-of)
   - Pattern continuation/completion
   - Sorting and ordering operations
   - Grid partitioning and merging

5. REALISTIC EXPECTATIONS:
   - Current SOTA on ARC is ~30-40% (similar to our result!)
   - 85-100% was unrealistic based on tiny test set
   - Focus on robust 40-50% as next milestone
   - Need fundamentally different approach for 80%+
"""
    )


if __name__ == "__main__":
    analyze_successful_vs_failed()
    sample_more_failed_tasks(n=10)
    recommendations()
