#!/usr/bin/env python3
"""Debug what segments are actually being created."""

from utils.imports import setup_project_paths

setup_project_paths()

import mlx.core as mx
from final_then_fix import FinalBindingExtractor
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

model = TwoStageCompilerV2(len(VOCAB), 4)
model.extractor = FinalBindingExtractor(VOCAB)

# Test failing cases
test_cases = [
    ("Y means walk X means turn do X then Y", ["TURN", "WALK"]),
    ("X means jump Y means run do X and Y then X", ["JUMP", "RUN", "JUMP"]),
]

for command, expected in test_cases:
    print(f"\nCommand: {command}")
    print(f"Expected: {expected}")

    tokens = [VOCAB.get(w, 0) for w in command.split()]
    tokens_mx = mx.array([tokens])

    # Get segments
    bindings, segments = model.extractor.extract(tokens_mx)

    print(f"\nSegments created: {len(segments)}")
    for i, seg in enumerate(segments):
        print(f"\nSegment {i}:")
        print(f"  Position: {seg.start_pos} to {seg.end_pos}")
        print(f"  Token indices: {tokens[seg.start_pos:seg.end_pos]}")
        print(f"  Words: {command.split()[seg.start_pos:seg.end_pos]}")
        print(f"  Parse tree: {seg.parse_tree}")

    # Get full output
    outputs = model(tokens_mx)
    predicted = []
    if outputs.shape[0] > 0:
        indices = mx.argmax(outputs, axis=-1)
        for idx in indices:
            predicted.append(["JUMP", "WALK", "RUN", "TURN"][int(idx)])

    print(f"\nPredicted: {predicted}")
    print(f"Correct: {predicted == expected}")

    # Show what the executor sees
    print("\nExecutor receives:")
    active_bindings = {}
    for b in bindings:
        active_bindings[b.variable] = b.action
    print(f"  Bindings: {active_bindings}")
    print(f"  Segments to execute: {len(segments)}")
