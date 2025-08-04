#!/usr/bin/env python3
"""Debug THEN evaluation to find the issue."""

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

# Create model with fix
model = TwoStageCompilerV2(len(VOCAB), 4)
model.extractor = DefinitiveTHENExtractor(VOCAB)

# Generate same test data
dataset = ProgressiveComplexityDataset()
test_data = {
    f"level_{i}": getattr(dataset, f"generate_level_{i}")(100) for i in range(1, 5)
}

# Count THEN patterns exactly as in the evaluation
then_correct = 0
then_total = 0
debug_info = []

for level_name, level_data in test_data.items():
    level_then_count = 0
    level_correct = 0

    for sample in level_data:
        # Same condition as in evaluation
        if (
            " then " in sample["command"]
            and "means" not in sample["command"].split(" then ")[1]
        ):
            then_total += 1
            level_then_count += 1

            tokens = mx.array([sample["tokens"]])
            outputs = model(tokens)

            predicted = []
            if outputs.shape[0] > 0:
                indices = mx.argmax(outputs, axis=-1)
                for idx in indices:
                    predicted.append(["JUMP", "WALK", "RUN", "TURN"][int(idx)])

            is_correct = predicted == sample["expected_actions"]
            if is_correct:
                then_correct += 1
                level_correct += 1

            # Track first few from each level
            if level_then_count <= 3:
                debug_info.append(
                    {
                        "level": level_name,
                        "command": sample["command"],
                        "expected": sample["expected_actions"],
                        "predicted": predicted,
                        "correct": is_correct,
                        "output_shape": outputs.shape if outputs is not None else None,
                    }
                )

    if level_then_count > 0:
        print(f"{level_name}: {level_correct}/{level_then_count} THEN patterns correct")

print(
    f"\nTotal THEN: {then_correct}/{then_total} = {then_correct/max(then_total,1)*100:.1f}%"
)

# Show debug info
print("\n" + "=" * 60)
print("Sample THEN patterns from each level:")
print("=" * 60)

for info in debug_info:
    print(f"\n{info['level']}:")
    print(f"  Command: {info['command']}")
    print(f"  Expected: {info['expected']}")
    print(f"  Predicted: {info['predicted']}")
    print(f"  Output shape: {info['output_shape']}")
    print(f"  Correct: {info['correct']}")
