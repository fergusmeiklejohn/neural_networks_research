#!/usr/bin/env python3
"""
Test temporal pattern handling in the enhanced model
"""

import sys

sys.path.append(".")

from pathlib import Path

import mlx.core as mx
import numpy as np
from train_binding_curriculum import ACTIONS, VOCAB
from train_temporal_curriculum import TemporalDynamicMemoryModel


def test_temporal_patterns():
    """Test specific temporal patterns to verify implementation"""

    # Load the trained model
    model_path = Path("outputs/temporal_curriculum/temporal_model_final.npz")
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return

    # Initialize model
    model = TemporalDynamicMemoryModel(
        vocab_size=len(VOCAB),
        num_actions=len(ACTIONS),
        embed_dim=64,
        num_slots=4,
        num_heads=8,
        initial_temperature=0.1,  # Low temperature for deterministic behavior
    )

    # Load weights - they are stored as a flat dict
    weights = np.load(model_path, allow_pickle=True)

    # Convert numpy arrays to mx arrays
    model_params = {}
    for key in weights.files:
        value = weights[key]
        model_params[key] = mx.array(value)

    # MLX expects nested dict structure, so we need to unflatten
    # The keys are in format "module.submodule.param"
    nested_params = {}
    for key, value in model_params.items():
        parts = key.split(".")
        current = nested_params
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    # Update model with loaded parameters
    try:
        model.update(nested_params)
    except Exception as e:
        print(f"Error updating model parameters: {e}")
        print(f"Flat keys: {list(model_params.keys())[:5]}...")
        print(f"Model expects: {list(model.parameters().keys())[:5]}...")
        # Try direct update with flat params
        model.load_weights(model_path)
        return
    model.eval()

    # Test cases specifically for temporal patterns
    test_cases = [
        # Basic temporal patterns
        ("X means jump do X twice", ["JUMP", "JUMP"]),
        ("Y means walk do Y twice", ["WALK", "WALK"]),
        ("Z means turn do Z thrice", ["TURN", "TURN", "TURN"]),
        # With distractors
        ("X means jump Y means walk do X twice", ["JUMP", "JUMP"]),
        ("X means jump Y means walk do Y twice", ["WALK", "WALK"]),
        # Edge cases
        ("X means run do X", ["RUN"]),  # No temporal modifier
        ("X means look do X twice", ["LOOK", "LOOK"]),
    ]

    print("Testing Temporal Patterns")
    print("=" * 60)

    successes = 0
    for cmd_str, expected in test_cases:
        # Tokenize
        tokens = cmd_str.split()
        token_ids = [VOCAB.get(t, VOCAB["<PAD>"]) for t in tokens]
        cmd_batch = mx.array([token_ids], dtype=mx.int32)

        # Get predictions
        outputs = model(cmd_batch, stage="full", training=False)
        logits = outputs["action_logits"]

        # Extract predictions
        predictions = []

        # Check if temporal actions were generated
        num_temporal = outputs.get("temporal_actions", 0)

        if num_temporal > 0:
            # Get temporal predictions from the end
            start_pos = logits.shape[1] - num_temporal
            for i in range(num_temporal):
                if start_pos + i < logits.shape[1]:
                    pred_id = mx.argmax(logits[0, start_pos + i]).item()
                    pred_action = [k for k, v in ACTIONS.items() if v == pred_id][0]
                    predictions.append(pred_action)
        else:
            # Look for regular predictions after 'do'
            for i, token in enumerate(tokens):
                if i > 0 and tokens[i - 1] == "do" and token in ["X", "Y", "Z"]:
                    if i < logits.shape[1]:
                        pred_id = mx.argmax(logits[0, i]).item()
                        pred_action = [k for k, v in ACTIONS.items() if v == pred_id][0]
                        predictions.append(pred_action)

        # Check correctness
        correct = predictions == expected
        if correct:
            successes += 1
            status = "✓"
        else:
            status = "✗"

        print(f"{status} '{cmd_str}'")
        print(f"   Expected: {expected}")
        print(f"   Got:      {predictions}")
        print(f"   Temporal actions: {num_temporal}")
        print()

    print(
        f"Overall Success Rate: {successes}/{len(test_cases)} = {successes/len(test_cases)*100:.1f}%"
    )

    # Analyze internal representations
    print("\nAnalyzing Internal Representations")
    print("=" * 60)

    # Test a specific case in detail
    test_cmd = "Y means turn do Y twice"
    tokens = test_cmd.split()
    token_ids = [VOCAB.get(t, VOCAB["<PAD>"]) for t in tokens]
    cmd_batch = mx.array([token_ids], dtype=mx.int32)

    outputs = model(cmd_batch, stage="full", training=False)

    print(f"Command: {test_cmd}")
    print(f"Storage mask shape: {outputs['storage_mask'].shape}")
    print(f"Storage positions: {mx.where(outputs['storage_mask'][0] > 0)}")
    print(f"Slot values norm: {mx.norm(outputs['slot_values'][0], axis=1)}")
    print(f"Temporal actions generated: {outputs['temporal_actions']}")

    # Check binding consistency
    bindings = outputs["bindings"][0]
    print(f"Variable bindings: {bindings}")


if __name__ == "__main__":
    test_temporal_patterns()
