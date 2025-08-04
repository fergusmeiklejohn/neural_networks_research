"""
Minimal training script for sequential planning model with debugging.
"""

from utils.imports import setup_project_paths

setup_project_paths()

import os
from datetime import datetime

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from train_sequential_planning import (
    ACTIONS,
    VOCAB,
    SequentialDynamicMemoryModel,
    create_sequential_dataset,
)

from utils.config import setup_environment
from utils.paths import get_output_path

config = setup_environment()


def simple_train_step(model, batch, optimizer, stage="full_binding"):
    """Simplified training step for debugging."""

    def loss_fn(model, command_ids, action_ids):
        outputs = model(command_ids, stage=stage)

        if "action_logits" in outputs:
            action_logits = outputs["action_logits"]

            # Simple loss calculation
            batch_size, seq_len = action_ids.shape
            pred_len = action_logits.shape[1]

            if pred_len >= seq_len:
                action_logits = action_logits[:, :seq_len, :]
            else:
                action_ids = action_ids[:, :pred_len]

            loss = nn.losses.cross_entropy(
                action_logits.reshape(-1, action_logits.shape[-1]),
                action_ids.reshape(-1),
                reduction="mean",
            )
            return loss

        return mx.array(0.0)

    # Compute loss and gradients
    loss_and_grad_fn = mx.value_and_grad(loss_fn)
    loss, grads = loss_and_grad_fn(model, batch["command_ids"], batch["action_ids"])

    # Update model
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)

    return loss.item()


def evaluate_patterns(model, test_patterns):
    """Evaluate specific test patterns."""
    print("\nEvaluating test patterns:")
    print("=" * 50)

    for pattern, expected in test_patterns:
        # Tokenize
        tokens = pattern.split()
        token_ids = [VOCAB.get(token, VOCAB["<PAD>"]) for token in tokens]
        command_ids = mx.array([token_ids])

        # Forward pass
        outputs = model(command_ids, stage="full_binding")

        if "action_logits" in outputs:
            predictions = mx.argmax(outputs["action_logits"], axis=-1)

            # Extract predicted actions
            predicted_actions = []
            # Convert predictions to numpy for easier indexing
            predictions_np = np.array(predictions)

            for j in range(min(predictions_np.shape[1], 10)):  # Limit to 10 for safety
                try:
                    pred_id = (
                        predictions_np[0, j].item()
                        if hasattr(predictions_np[0, j], "item")
                        else int(predictions_np[0, j])
                    )
                    for action_name, action_id in ACTIONS.items():
                        if action_id == pred_id and action_name != "<PAD>":
                            predicted_actions.append(action_name)
                            break
                except Exception:
                    break

            # Trim to expected length
            predicted_actions = predicted_actions[: len(expected)]

            print(f"\nPattern: {pattern}")
            print(f"Expected: {expected}")
            print(f"Predicted: {predicted_actions}")
            print(f"Correct: {predicted_actions == expected}")


def main():
    # Small configuration for debugging
    num_epochs = 10
    batch_size = 4  # Small batch size
    learning_rate = 0.001
    num_samples = 100  # Small dataset

    print("Creating small dataset...")
    dataset = create_sequential_dataset(num_samples=num_samples)

    # Initialize model
    model = SequentialDynamicMemoryModel(
        vocab_size=len(VOCAB),
        num_actions=len(ACTIONS),
        embed_dim=128,  # Smaller embedding
        num_slots=4,
        num_heads=4,
        mlp_hidden_dim=256,
    )

    # Initialize optimizer
    optimizer = optim.Adam(learning_rate=learning_rate)

    # Test patterns
    test_patterns = [
        ("X means jump do X", ["JUMP"]),
        ("X means jump do X then Y means walk do Y", ["JUMP", "WALK"]),
        ("Z means turn do Z twice", ["TURN", "TURN"]),
    ]

    # Initial evaluation
    print("\nInitial evaluation (before training):")
    evaluate_patterns(model, test_patterns)

    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        epoch_losses = []

        # Create small batches
        indices = np.random.permutation(len(dataset["commands"]))

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i : i + batch_size]

            # Skip incomplete batches
            if len(batch_indices) < batch_size:
                continue

            # Prepare batch
            batch_commands = [dataset["commands"][idx] for idx in batch_indices]
            batch_actions = [dataset["actions"][idx] for idx in batch_indices]

            # Convert to token IDs
            max_cmd_len = max(len(cmd.split()) for cmd in batch_commands)
            max_act_len = max(len(acts) for acts in batch_actions)

            command_ids = []
            action_ids = []

            for cmd, acts in zip(batch_commands, batch_actions):
                # Tokenize command
                tokens = cmd.split()
                cmd_ids = [VOCAB.get(token, VOCAB["<PAD>"]) for token in tokens]
                cmd_ids += [VOCAB["<PAD>"]] * (max_cmd_len - len(cmd_ids))
                command_ids.append(cmd_ids)

                # Tokenize actions
                act_ids = [ACTIONS.get(act, ACTIONS["<PAD>"]) for act in acts]
                act_ids += [ACTIONS["<PAD>"]] * (max_act_len - len(act_ids))
                action_ids.append(act_ids)

            batch = {
                "command_ids": mx.array(command_ids),
                "action_ids": mx.array(action_ids),
            }

            # Train step
            try:
                loss = simple_train_step(model, batch, optimizer)
                epoch_losses.append(loss)
            except Exception as e:
                print(f"Error in batch {i//batch_size}: {e}")
                # Debug info
                print(f"Command shape: {batch['command_ids'].shape}")
                print(f"Action shape: {batch['action_ids'].shape}")
                if "temporal" in str(e).lower():
                    print("Skipping batch with temporal pattern issues...")
                    continue
                else:
                    raise

        # Print epoch summary
        if epoch_losses:
            avg_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")

        # Evaluate periodically
        if (epoch + 1) % 5 == 0:
            evaluate_patterns(model, test_patterns)

    # Final evaluation
    print("\nFinal evaluation:")
    evaluate_patterns(model, test_patterns)

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = get_output_path("models")
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, f"sequential_minimal_{timestamp}.npz")
    mx.save(model_path, dict(model.parameters()))
    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()
