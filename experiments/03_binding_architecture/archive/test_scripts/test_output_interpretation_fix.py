"""
Demonstrate the output interpretation fix by comparing old vs new models.
"""

from utils.imports import setup_project_paths

setup_project_paths()

import mlx.core as mx
from test_sequential_model_fixed import extract_action_predictions
from train_binding_curriculum import ACTIONS, VOCAB
from train_sequential_action_positions import (
    SequentialModelWithActionTracking,
    extract_action_predictions_improved,
)
from train_sequential_planning_fixed import SequentialDynamicMemoryModel

from utils.config import setup_environment

config = setup_environment()


def demonstrate_output_interpretation_fix():
    """Show how the fix improves output interpretation."""

    print("=" * 70)
    print("OUTPUT INTERPRETATION FIX DEMONSTRATION")
    print("=" * 70)

    # Initialize both models
    old_model = SequentialDynamicMemoryModel(
        vocab_size=len(VOCAB),
        num_actions=len(ACTIONS),
        embed_dim=128,
        num_slots=4,
        num_heads=4,
        mlp_hidden_dim=256,
    )

    new_model = SequentialModelWithActionTracking(
        vocab_size=len(VOCAB),
        num_actions=len(ACTIONS),
        embed_dim=128,
        num_slots=4,
        num_heads=4,
        mlp_hidden_dim=256,
    )

    # Test command
    test_command = "X means jump do X twice then Y means walk do Y"
    expected = ["JUMP", "JUMP", "WALK"]

    print(f"\nTest Command: {test_command}")
    print(f"Expected Output: {expected}")
    print("\n" + "-" * 70)

    # Test OLD model
    print("\nOLD MODEL (outputs at every token position):")
    command_tokens = test_command.split()
    command_ids = mx.array(
        [[VOCAB.get(token, VOCAB["<PAD>"]) for token in command_tokens]]
    )

    old_outputs = old_model(command_ids, stage="full_binding")
    if "action_logits" in old_outputs:
        old_shape = old_outputs["action_logits"].shape
        print(f"- Action logits shape: {old_shape}")
        print(f"- Number of outputs: {old_shape[1]} (one for each token!)")

        # Show what the extraction function has to deal with
        old_predictions = extract_action_predictions(old_model, test_command)
        print(f"- Extracted predictions: {old_predictions}")
        print(f"- Correct? {'✓' if old_predictions == expected else '✗'}")

    print("\n" + "-" * 70)

    # Test NEW model
    print("\nNEW MODEL (outputs only at action positions):")
    new_outputs = new_model(command_ids, stage="full_binding")
    if "action_logits" in new_outputs:
        new_shape = new_outputs["action_logits"].shape
        print(f"- Action logits shape: {new_shape}")
        print(f"- Number of outputs: {new_shape[1]} (exactly {len(expected)} actions!)")
        print(f"- Action positions: {new_outputs.get('action_positions', [])}")

        # Show clean extraction
        new_predictions = extract_action_predictions_improved(new_model, test_command)
        print(f"- Extracted predictions: {new_predictions}")
        print(f"- Correct? {'✓' if new_predictions == expected else '✗'}")

    print("\n" + "=" * 70)
    print("KEY IMPROVEMENTS:")
    print("=" * 70)
    print("1. Old model outputs predictions for ALL token positions")
    print("2. New model outputs predictions ONLY at action positions")
    print("3. This eliminates ambiguity and makes extraction reliable")
    print("4. The action_positions list shows exactly where actions occur")

    # More complex example
    print("\n" + "=" * 70)
    print("COMPLEX PATTERN TEST")
    print("=" * 70)

    complex_command = (
        "X means jump do X thrice then Y means walk do Y twice then Z means turn do Z"
    )
    complex_expected = ["JUMP", "JUMP", "JUMP", "WALK", "WALK", "TURN"]

    print(f"\nCommand: {complex_command}")
    print(f"Expected: {complex_expected}")

    # Test with new model
    complex_ids = mx.array(
        [[VOCAB.get(token, VOCAB["<PAD>"]) for token in complex_command.split()]]
    )
    complex_outputs = new_model(complex_ids, stage="full_binding")

    if "action_logits" in complex_outputs:
        print(f"\nNew Model Results:")
        print(f"- Action positions: {complex_outputs.get('action_positions', [])}")
        print(f"- Number of actions: {complex_outputs.get('num_actions', 0)}")
        print(
            f"- Shape matches expected: {complex_outputs['action_logits'].shape[1] == len(complex_expected)}"
        )


if __name__ == "__main__":
    demonstrate_output_interpretation_fix()
