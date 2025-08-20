#!/usr/bin/env python3
"""Investigate why THEN passes tests but fails evaluation."""

from utils.imports import setup_project_paths

setup_project_paths()

import mlx.core as mx
from progressive_complexity_dataset import ProgressiveComplexityDataset
from then_fix_final import DefinitiveTHENExtractor
from two_stage_compiler_v2 import TwoStageCompilerV2

VOCAB = {
    "PAD": 0,
    "do": 1,
    "means": 2,
    "is": 3,
    "and": 4,
    "or": 5,
    "then": 6,
    "twice": 7,
    "thrice": 8,
    "while": 9,
    "X": 10,
    "Y": 11,
    "Z": 12,
    "W": 13,
    "jump": 14,
    "walk": 15,
    "run": 16,
    "turn": 17,
    "true": 18,
}

# Create models
baseline = TwoStageCompilerV2(len(VOCAB), 4)
baseline.set_vocab(VOCAB)

fixed = TwoStageCompilerV2(len(VOCAB), 4)
fixed.extractor = DefinitiveTHENExtractor(VOCAB)

# Generate data
dataset = ProgressiveComplexityDataset()
level_2_data = dataset.generate_level_2(100)

# Find THEN patterns
then_patterns = []
for sample in level_2_data:
    if (
        " then " in sample["command"]
        and "means" not in sample["command"].split(" then ")[1]
    ):
        then_patterns.append(sample)

print(f"Found {len(then_patterns)} THEN patterns in Level 2\n")

# Test each one
success_count = 0
failure_examples = []

for i, sample in enumerate(then_patterns):
    tokens = mx.array([sample["tokens"]])
    expected = sample["expected_actions"]

    # Get baseline prediction
    baseline_out = baseline(tokens)
    baseline_pred = []
    if baseline_out.shape[0] > 0:
        indices = mx.argmax(baseline_out, axis=-1)
        for idx in indices:
            baseline_pred.append(["JUMP", "WALK", "RUN", "TURN"][int(idx)])

    # Get fixed prediction
    fixed_out = fixed(tokens)
    fixed_pred = []
    if fixed_out.shape[0] > 0:
        indices = mx.argmax(fixed_out, axis=-1)
        for idx in indices:
            fixed_pred.append(["JUMP", "WALK", "RUN", "TURN"][int(idx)])

    # Check if fixed is correct
    if fixed_pred == expected:
        success_count += 1
    else:
        failure_examples.append(
            {
                "command": sample["command"],
                "expected": expected,
                "baseline": baseline_pred,
                "fixed": fixed_pred,
            }
        )

print(
    f"Fixed model success rate: {success_count}/{len(then_patterns)} = {success_count/len(then_patterns)*100:.1f}%"
)

# Analyze failures
if failure_examples:
    print("\nAnalyzing failures...")

    # Show first 5 failures
    for i, fail in enumerate(failure_examples[:5]):
        print(f"\nFailure {i+1}:")
        print(f"  Command: {fail['command']}")
        print(f"  Expected: {fail['expected']}")
        print(f"  Baseline: {fail['baseline']}")
        print(f"  Fixed: {fail['fixed']}")

        # Debug the fixed model
        tokens = [VOCAB.get(w, 0) for w in fail["command"].split()]
        tokens_mx = mx.array([tokens])

        analysis = fixed.analyze(tokens_mx)
        print(f"  Segments created: {len(analysis['segments'])}")
        for j, seg in enumerate(analysis["segments"]):
            print(f"    Segment {j}: {seg}")

# Check if it's a consistency issue
print("\n" + "=" * 60)
print("Checking Consistency")
print("=" * 60)

# Run the same pattern multiple times
test_command = "X means jump Y means walk do X then Y"
test_expected = ["JUMP", "WALK"]

print(f"\nTesting '{test_command}' multiple times:")
for i in range(5):
    tokens = [VOCAB.get(w, 0) for w in test_command.split()]
    tokens_mx = mx.array([tokens])

    outputs = fixed(tokens_mx)
    predicted = []
    if outputs.shape[0] > 0:
        indices = mx.argmax(outputs, axis=-1)
        for idx in indices:
            predicted.append(["JUMP", "WALK", "RUN", "TURN"][int(idx)])

    print(f"  Run {i+1}: {predicted} - {'✓' if predicted == test_expected else '✗'}")
