#!/usr/bin/env python3
"""Debug ablation issue."""

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

# Create model
model = TwoStageCompilerV2(len(VOCAB), 4)
model.set_vocab(VOCAB)

# Generate one test sample
dataset = ProgressiveComplexityDataset()
samples = dataset.generate_level_1(1)
sample = samples[0]

print(f"Command: {sample['command']}")
print(f"Tokens: {sample['tokens']}")
print(f"Expected: {sample['expected_actions']}")

# Test model
tokens = mx.array([sample["tokens"]])
outputs = model(tokens)

print(f"\nOutputs shape: {outputs.shape}")
print(f"Outputs: {outputs}")

# Convert to actions
if outputs.shape[0] > 0:
    indices = mx.argmax(outputs, axis=-1)
    action_names = ["jump", "walk", "run", "turn"]
    predicted = []
    for idx in indices:
        predicted.append(action_names[int(idx)])
    print(f"Predicted: {predicted}")

# Analyze
analysis = model.analyze(tokens)
print(f"\nAnalysis: {analysis}")
