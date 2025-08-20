#!/usr/bin/env python3
"""Debug THEN pattern handling."""

from utils.imports import setup_project_paths

setup_project_paths()

import mlx.core as mx
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

# Create models
original_model = TwoStageCompilerV2(len(VOCAB), 4)
original_model.set_vocab(VOCAB)

# Also test improved version
from train_then_simple import ImprovedBindingExtractorV2, THENAwareExecutor

improved_model = TwoStageCompilerV2(len(VOCAB), 4)
improved_model.extractor = ImprovedBindingExtractorV2(VOCAB)
improved_model.executor = THENAwareExecutor(len(VOCAB), 4)

# Generate data and find THEN patterns
dataset = ProgressiveComplexityDataset()
level_2_data = dataset.generate_level_2(100)

print("Analyzing THEN patterns in Level 2...\n")

then_patterns = []
for sample in level_2_data:
    if " then " in sample["command"]:
        then_patterns.append(sample)

print(f"Found {len(then_patterns)} THEN patterns\n")

# Analyze first 5 THEN patterns
for i, sample in enumerate(then_patterns[:5]):
    print(f"\nPattern {i+1}:")
    print(f"Command: {sample['command']}")
    print(f"Expected: {sample['expected_actions']}")

    tokens = mx.array([sample["tokens"]])

    # Get analysis from original model
    analysis = original_model.analyze(tokens)
    print(f"Original Bindings: {analysis['bindings']}")
    print(f"Original Segments: {analysis['segments']}")
    print(f"Original Actions: {analysis['actions']}")

    # Get analysis from improved model
    improved_analysis = improved_model.analyze(tokens)
    print(f"Improved Bindings: {improved_analysis['bindings']}")
    print(f"Improved Segments: {improved_analysis['segments']}")
    print(f"Improved Actions: {improved_analysis['actions']}")

    # Check correctness
    orig_correct = analysis["actions"] == sample["expected_actions"]
    imp_correct = improved_analysis["actions"] == sample["expected_actions"]
    print(f"Original Correct: {orig_correct}")
    print(f"Improved Correct: {imp_correct}")

    if imp_correct and not orig_correct:
        print("✅ IMPROVED VERSION FIXED IT!")

# Check what types of THEN patterns exist
print("\n" + "=" * 60)
print("THEN Pattern Types:")
print("=" * 60)

pattern_types = {}
for sample in then_patterns:
    cmd = sample["command"]
    # Classify pattern
    if "means" in cmd.split("then")[1]:
        pattern_type = "rebinding"
    elif "do" in cmd and cmd.count("do") > 1:
        pattern_type = "multiple_do"
    else:
        pattern_type = "simple_then"

    if pattern_type not in pattern_types:
        pattern_types[pattern_type] = []
    pattern_types[pattern_type].append(sample)

for ptype, samples in pattern_types.items():
    print(f"\n{ptype}: {len(samples)} samples")
    # Show first example
    if samples:
        print(f"  Example: {samples[0]['command']}")
        print(f"  Expected: {samples[0]['expected_actions']}")
