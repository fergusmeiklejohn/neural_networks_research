#!/usr/bin/env python3
"""Isolate and debug the failing rebinding test case."""

from utils.imports import setup_project_paths

setup_project_paths()

from utils.config import setup_environment

config = setup_environment()

import os

import mlx.core as mx
from mlx_model_io import load_model_simple
from train_integrated_model import ACTIONS, VOCAB
from train_nested_temporal_model_fixed import NestedTemporalBindingModel

# Ensure temporal modifiers are in VOCAB
for mod in ["twice", "thrice"]:
    if mod not in VOCAB:
        VOCAB[mod] = len(VOCAB)

print("=== Debugging Variable Rebinding Issue ===\n")

# Create or load model
model_path = os.path.join(
    "/Users/fergusmeiklejohn/dev/neural_networks_research/outputs",
    "nested_temporal_best.pkl",
)
if os.path.exists(model_path):
    print(f"Loading model from {model_path}")
    model = NestedTemporalBindingModel(len(VOCAB), len(ACTIONS))
    load_model_simple(model_path, model)
else:
    print("Creating new model for testing")
    model = NestedTemporalBindingModel(len(VOCAB), len(ACTIONS))

# Test cases from the integrated test
test_cases = [
    # Basic nested temporal (should pass)
    ("X means jump do X twice", ["JUMP", "JUMP"]),
    # Variable rebinding with nested temporal (this might be failing)
    (
        "X means jump do X twice then X means walk do X twice twice",
        ["JUMP", "JUMP"] + ["WALK"] * 4,
    ),
    # Simpler rebinding case
    ("X means jump do X then X means walk do X", ["JUMP", "WALK"]),
    # Rebinding with single temporal
    ("X means jump do X twice then X means walk do X", ["JUMP", "JUMP", "WALK"]),
]

for i, (command, expected) in enumerate(test_cases):
    print(f"\nTest {i+1}: {command}")
    print(f"Expected: {expected}")

    tokens = [VOCAB.get(word, VOCAB["<PAD>"]) for word in command.split()]
    print(f"Token IDs: {tokens}")

    inputs = {"command": mx.array([tokens])}

    # Get predictions
    outputs = model(inputs, stage="full")
    print(f"Output shape: {outputs.shape}")

    predictions = mx.argmax(outputs, axis=1)

    predicted_actions = []
    for pred in predictions:
        for name, idx in ACTIONS.items():
            if idx == int(pred):
                predicted_actions.append(name)
                break

    print(f"Predicted: {predicted_actions}")

    correct = predicted_actions == expected
    print(f"Correct: {correct}")

    if not correct:
        print("MISMATCH DETAILS:")
        print(f"  Expected length: {len(expected)}")
        print(f"  Predicted length: {len(predicted_actions)}")
        for j, (exp, pred) in enumerate(zip(expected, predicted_actions)):
            if exp != pred:
                print(f"  Position {j}: expected {exp}, got {pred}")

print("\n=== Analysis ===")
print("If rebinding cases fail, check:")
print("1. Is versioned memory being cleared between 'then' segments?")
print("2. Are nested temporal modifiers processed correctly after rebinding?")
print("3. Is the model maintaining separate contexts for each segment?")
