#!/usr/bin/env python3
"""Deep analysis of why THEN patterns fail."""

from utils.imports import setup_project_paths

setup_project_paths()

import mlx.core as mx
from final_then_fix import FinalBindingExtractor
from progressive_complexity_dataset import ProgressiveComplexityDataset
from two_stage_compiler_v2 import TwoStageCompilerV2

# Setup
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

# Create both models
original_model = TwoStageCompilerV2(len(VOCAB), 4)
original_model.set_vocab(VOCAB)

improved_model = TwoStageCompilerV2(len(VOCAB), 4)
improved_model.extractor = FinalBindingExtractor(VOCAB)

# Generate comprehensive dataset
dataset = ProgressiveComplexityDataset()
all_data = []
for i in range(1, 5):
    level_data = getattr(dataset, f"generate_level_{i}")(200)
    all_data.extend(level_data)

# Find ALL THEN patterns
then_patterns = []
for sample in all_data:
    if " then " in sample["command"]:
        # Exclude rebinding patterns
        if "means" not in sample["command"].split(" then ")[1]:
            then_patterns.append(sample)

print(f"Found {len(then_patterns)} pure THEN patterns\n")

# Categorize THEN patterns
categories = {
    "simple": [],  # "do X then Y"
    "mixed_ops": [],  # "do X and Y then Z"
    "multiple_then": [],  # "do X then Y then Z"
    "with_modifiers": [],  # "do X twice then Y"
    "complex": [],  # Everything else
}

for pattern in then_patterns:
    cmd = pattern["command"]
    then_count = cmd.count(" then ")

    if then_count > 1:
        categories["multiple_then"].append(pattern)
    elif " and " in cmd:
        categories["mixed_ops"].append(pattern)
    elif "twice" in cmd or "thrice" in cmd:
        categories["with_modifiers"].append(pattern)
    elif then_count == 1 and cmd.count("do") == 1:
        categories["simple"].append(pattern)
    else:
        categories["complex"].append(pattern)

# Analyze each category
print("THEN Pattern Categories:")
print("=" * 60)
for cat_name, patterns in categories.items():
    if patterns:
        print(f"\n{cat_name}: {len(patterns)} patterns")
        print("Examples:")
        for p in patterns[:3]:
            print(f"  - {p['command']}")
            print(f"    Expected: {p['expected_actions']}")

# Test each category with both models
print("\n" + "=" * 60)
print("Testing by Category:")
print("=" * 60)

for cat_name, patterns in categories.items():
    if not patterns:
        continue

    orig_correct = 0
    imp_correct = 0

    for pattern in patterns[:20]:  # Test first 20 of each
        tokens = mx.array([pattern["tokens"]])
        expected = pattern["expected_actions"]

        # Original model
        orig_out = original_model(tokens)
        orig_pred = []
        if orig_out.shape[0] > 0:
            indices = mx.argmax(orig_out, axis=-1)
            for idx in indices:
                orig_pred.append(["JUMP", "WALK", "RUN", "TURN"][int(idx)])

        # Improved model
        imp_out = improved_model(tokens)
        imp_pred = []
        if imp_out.shape[0] > 0:
            indices = mx.argmax(imp_out, axis=-1)
            for idx in indices:
                imp_pred.append(["JUMP", "WALK", "RUN", "TURN"][int(idx)])

        if orig_pred == expected:
            orig_correct += 1
        if imp_pred == expected:
            imp_correct += 1

        # Show first failure in each category
        if imp_pred != expected and imp_correct == 0:
            print(f"\n{cat_name} failure example:")
            print(f"  Command: {pattern['command']}")
            print(f"  Expected: {expected}")
            print(f"  Got: {imp_pred}")

            # Analyze why
            analysis = improved_model.analyze(tokens)
            print(f"  Segments: {len(analysis['segments'])}")
            for seg in analysis["segments"]:
                print(f"    - {seg}")

    tested = min(20, len(patterns))
    print(
        f"\n{cat_name}: Original {orig_correct}/{tested}, Improved {imp_correct}/{tested}"
    )

# Focus on simple THEN patterns
print("\n" + "=" * 60)
print("Deep Dive: Simple THEN Patterns")
print("=" * 60)

if categories["simple"]:
    print(f"\nTesting all {len(categories['simple'])} simple patterns...")

    simple_correct = 0
    for pattern in categories["simple"]:
        tokens = mx.array([pattern["tokens"]])
        outputs = improved_model(tokens)

        predicted = []
        if outputs.shape[0] > 0:
            indices = mx.argmax(outputs, axis=-1)
            for idx in indices:
                predicted.append(["JUMP", "WALK", "RUN", "TURN"][int(idx)])

        if predicted == pattern["expected_actions"]:
            simple_correct += 1

    print(
        f"Simple THEN accuracy: {simple_correct}/{len(categories['simple'])} = {simple_correct/len(categories['simple'])*100:.1f}%"
    )

    # Show pattern of failures
    print("\nFailure analysis:")
    fail_types = {}
    for pattern in categories["simple"]:
        tokens = mx.array([pattern["tokens"]])
        outputs = improved_model(tokens)

        predicted = []
        if outputs.shape[0] > 0:
            indices = mx.argmax(outputs, axis=-1)
            for idx in indices:
                predicted.append(["JUMP", "WALK", "RUN", "TURN"][int(idx)])

        if predicted != pattern["expected_actions"]:
            # Categorize failure
            if len(predicted) != len(pattern["expected_actions"]):
                fail_type = f"wrong_length ({len(predicted)} vs {len(pattern['expected_actions'])})"
            elif len(predicted) == 1:
                fail_type = "only_first_action"
            else:
                fail_type = "wrong_actions"

            if fail_type not in fail_types:
                fail_types[fail_type] = []
            fail_types[fail_type].append(pattern)

    for fail_type, patterns in fail_types.items():
        print(f"\n{fail_type}: {len(patterns)} failures")
        if patterns:
            print(f"  Example: {patterns[0]['command']}")
            print(f"  Expected: {patterns[0]['expected_actions']}")
