#!/usr/bin/env python3
"""
Quick test of the simple baseline model with minimal data.
"""

import subprocess
import sys


def test_simple_baseline():
    """Run simple baseline with minimal settings for testing."""

    print("Testing Simple Baseline Model")
    print("=" * 50)

    # Run with minimal settings
    cmd = [
        sys.executable,
        "train_simple_baseline.py",
        "--epochs",
        "2",
        "--batch_size",
        "16",
        "--mixed_from_start",  # Include modifications from start
        "--use_proper_validation",
    ]

    print(f"Running command: {' '.join(cmd)}")
    print()

    # Run the training
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        print("\nTest completed successfully!")
    else:
        print(f"\nTest failed with return code: {result.returncode}")

    return result.returncode


if __name__ == "__main__":
    exit(test_simple_baseline())
