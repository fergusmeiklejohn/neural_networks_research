#!/usr/bin/env python3
"""Verify THEN success - why the discrepancy?"""

from utils.imports import setup_project_paths

setup_project_paths()

import mlx.core as mx
import numpy as np
from final_then_fix import FinalBindingExtractor
from progressive_complexity_dataset import ProgressiveComplexityDataset
from train_two_stage_simple import evaluate_model
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

# Create model with final fix
model = TwoStageCompilerV2(len(VOCAB), 4)
model.extractor = FinalBindingExtractor(VOCAB)

# Generate fresh test data
dataset = ProgressiveComplexityDataset()
test_data = {
    f"level_{i}": getattr(dataset, f"generate_level_{i}")(100) for i in range(1, 5)
}

# Run full evaluation
print("Running full evaluation with FinalBindingExtractor...\n")
results = evaluate_model(model, test_data, VOCAB)

print("\nResults by level:")
for level in range(1, 5):
    level_name = f"level_{level}"
    if level_name in results:
        print(f"Level {level}: {results[level_name]:.2%}")

avg_accuracy = np.mean(list(results.values()))
print(f"\nAverage: {avg_accuracy:.2%}")

# Now specifically test THEN patterns
print("\n" + "=" * 60)
print("THEN Pattern Analysis")
print("=" * 60)

# Count THEN patterns by level
then_by_level = {}
correct_by_level = {}

for level_name, level_data in test_data.items():
    then_count = 0
    then_correct = 0

    for sample in level_data:
        # Pure THEN pattern (not rebinding)
        if (
            " then " in sample["command"]
            and "means" not in sample["command"].split(" then ")[1]
        ):
            then_count += 1

            tokens = mx.array([sample["tokens"]])
            outputs = model(tokens)

            predicted = []
            if outputs.shape[0] > 0:
                indices = mx.argmax(outputs, axis=-1)
                for idx in indices:
                    predicted.append(["JUMP", "WALK", "RUN", "TURN"][int(idx)])

            if predicted == sample["expected_actions"]:
                then_correct += 1
            elif then_count <= 3:  # Show first few failures
                print(f"\nFailed in {level_name}:")
                print(f"  Command: {sample['command']}")
                print(f"  Expected: {sample['expected_actions']}")
                print(f"  Got: {predicted}")

    then_by_level[level_name] = then_count
    correct_by_level[level_name] = then_correct

    if then_count > 0:
        acc = then_correct / then_count * 100
        print(f"\n{level_name}: {then_correct}/{then_count} = {acc:.1f}%")

# Overall THEN accuracy
total_then = sum(then_by_level.values())
total_correct = sum(correct_by_level.values())
if total_then > 0:
    overall_then = total_correct / total_then * 100
    print(f"\nOverall THEN: {total_correct}/{total_then} = {overall_then:.1f}%")

# Compare with original model
print("\n" + "=" * 60)
print("Comparison with Original Model")
print("=" * 60)

original_model = TwoStageCompilerV2(len(VOCAB), 4)
original_model.set_vocab(VOCAB)

orig_results = evaluate_model(original_model, test_data, VOCAB)
print("\nOriginal model results:")
for level in range(1, 5):
    level_name = f"level_{level}"
    if level_name in orig_results:
        print(f"Level {level}: {orig_results[level_name]:.2%}")

orig_avg = np.mean(list(orig_results.values()))
print(f"\nOriginal average: {orig_avg:.2%}")
print(f"Improved average: {avg_accuracy:.2%}")
print(f"Improvement: {avg_accuracy - orig_avg:.2%}")
