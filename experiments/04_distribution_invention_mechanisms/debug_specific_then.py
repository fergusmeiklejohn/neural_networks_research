#!/usr/bin/env python3
"""Debug specific THEN failure."""

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

# Debug the failing case
command = "Y means run X means jump do Y then X"
expected = ["RUN", "JUMP"]

print(f"Command: {command}")
print(f"Expected: {expected}")

tokens = [VOCAB.get(w, 0) for w in command.split()]
print(f"Tokens: {tokens}")
print(f"Token words: {[command.split()[i] for i in range(len(tokens))]}")

# Manual analysis
for i, (token, word) in enumerate(zip(tokens, command.split())):
    print(f"  {i}: {token} = '{word}'")

tokens_mx = mx.array([tokens])

# Extract bindings and segments
bindings, segments = model.extractor.extract(tokens_mx)

print(f"\nBindings:")
for b in bindings:
    print(f"  {b}")

print(f"\nSegments:")
for s in segments:
    print(f"  {s}")

# Get full analysis
analysis = model.analyze(tokens_mx)
print(f"\nFull analysis:")
print(f"Actions: {analysis['actions']}")

# Check the parse tree details
print("\nDetailed segment analysis:")
for i, seg in enumerate(segments):
    print(f"\nSegment {i}:")
    print(f"  Range: {seg.start_pos} to {seg.end_pos}")
    print(f"  Tokens in range: {tokens[seg.start_pos:seg.end_pos]}")
    print(f"  Words in range: {command.split()[seg.start_pos:seg.end_pos]}")
    if seg.parse_tree:
        print(f"  Parse tree: {seg.parse_tree}")
        print(f"    Type: {seg.parse_tree.node_type}")
        print(f"    Value: {seg.parse_tree.value}")
