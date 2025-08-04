#!/usr/bin/env python3
"""Integrated Variable Binding Model - Combines all 4 architectural components.

This model integrates:
1. Dynamic Memory - Variable binding with input-specific storage
2. Temporal Action Buffer - Handles "twice"/"thrice" patterns
3. Sequential Planning - Supports "then" operator for sequential composition
4. Versioned Memory - Enables variable rebinding over time

The goal is a single unified model that can handle complex patterns like:
"X means jump do X twice then X means walk do X and Y means run do Y thrice"
"""

from utils.imports import setup_project_paths

setup_project_paths()

import os
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx_model_io import save_model_simple
from tqdm import tqdm

# Import existing components and utilities
from train_binding_curriculum import (
    ACTIONS,
    VOCAB,
    generate_stage1_data,
    generate_stage2_data,
    generate_stage3_data,
)
from train_sequential_action_positions import ActionPositionTracker
from train_sequential_planning_fixed import BindingAttentionFixed as BindingAttention
from train_sequential_planning_fixed import SequencePlanner
from train_temporal_curriculum import TemporalActionBuffer

from utils.config import setup_environment
from utils.paths import get_output_path

config = setup_environment()


class VersionedMemory:
    """Memory module that supports versioning for rebinding variables.

    This is a simplified version that maintains a history of bindings
    with timestamps, allowing variables to be rebound over time.
    """

    def __init__(self, num_slots: int = 4, max_versions: int = 5):
        self.num_slots = num_slots
        self.max_versions = max_versions
        self.memory = {}  # slot_id -> list of (value, timestamp, position)

    def bind(self, slot_id: int, value: mx.array, position: int) -> None:
        """Add a new binding version."""
        if slot_id not in self.memory:
            self.memory[slot_id] = []

        self.memory[slot_id].append(
            {
                "value": value,
                "position": position,
                "timestamp": position,  # Use position as timestamp
            }
        )

        # Keep only recent versions
        if len(self.memory[slot_id]) > self.max_versions:
            self.memory[slot_id].pop(0)

    def retrieve(
        self, slot_id: int, position: Optional[int] = None
    ) -> Optional[mx.array]:
        """Retrieve the most recent value for a slot."""
        if slot_id not in self.memory or not self.memory[slot_id]:
            return None

        # Return most recent binding
        return self.memory[slot_id][-1]["value"]

    def get_history(self, slot_id: int) -> List[Dict]:
        """Get full history for debugging."""
        return self.memory.get(slot_id, [])

    def clear(self) -> None:
        """Clear all memory."""
        self.memory = {}


class IntegratedBindingModel(nn.Module):
    """Unified model combining all 4 architectural components."""

    def __init__(
        self,
        vocab_size: int,
        num_actions: int,
        embed_dim: int = 256,
        num_slots: int = 4,
        num_heads: int = 8,
        mlp_hidden_dim: int = 512,
    ):
        super().__init__()

        # Core dimensions
        self.vocab_size = vocab_size
        self.num_actions = num_actions
        self.embed_dim = embed_dim
        self.num_slots = num_slots

        # Embeddings
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = nn.Embedding(
            100, embed_dim
        )  # Support up to 100 positions

        # Component 1: Dynamic Memory with binding attention
        self.slot_keys = mx.random.normal((num_slots, embed_dim))
        self.binder = BindingAttention(embed_dim, num_heads)

        # Component 2: Temporal Action Buffer
        self.temporal_buffer = TemporalActionBuffer()

        # Component 3: Sequential Planning
        self.sequence_planner = SequencePlanner()

        # Component 4: Versioned Memory
        self.versioned_memory = VersionedMemory(num_slots)

        # Action tracking
        self.action_tracker = ActionPositionTracker(VOCAB)

        # Decoders
        self.recognition_decoder = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, vocab_size),
        )

        self.action_decoder = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, num_actions),
        )

    def process_segment_versioned(
        self,
        command_ids: mx.array,
        segment: Tuple[int, int],
        bindings: Dict,
        stage: str,
    ) -> List[mx.array]:
        """Process a command segment with versioned memory support."""
        outputs = []

        # Ensure command_ids has correct shape
        if len(command_ids.shape) == 1:
            command_ids = command_ids.reshape(1, -1)

        # Extract segment
        segment_tokens = command_ids[:, segment[0] : segment[1]]
        segment_length = segment[1] - segment[0]

        # Track slot values for this segment (batch, num_slots, embed_dim)
        slot_values = mx.zeros((1, self.num_slots, self.embed_dim))

        for t in range(segment_length):
            # Get token embedding - work around MLX indexing issue
            token_val = int(segment_tokens[0, t].item())
            token_embed = self.token_embeddings.weight[token_val].reshape(1, 1, -1)
            pos_embed = self.position_embeddings.weight[t].reshape(1, 1, -1)
            current_embed = token_embed + pos_embed

            # Check for storage pattern (X means/is Y)
            is_storage = False
            if t + 2 < segment_length:
                next_token = segment_tokens[0, t + 1].item()
                is_means = next_token == VOCAB.get("means", -1)
                is_is = next_token == VOCAB.get("is", -1)
                is_storage = is_means or is_is

            # Get binding scores
            binding_scores, _ = self.binder(
                current_embed, self.slot_keys, training=(stage != "recognition")
            )

            # Handle storage
            if is_storage and t + 2 < segment_length:
                var_id = segment_tokens[0, t].item()
                slot_idx = mx.argmax(binding_scores[0]).item()

                # Store in versioned memory
                value_token_idx = int(segment_tokens[0, t + 2].item())
                value_embed = self.token_embeddings.weight[value_token_idx].reshape(
                    1, 1, -1
                )

                # Update versioned memory
                self.versioned_memory.bind(
                    slot_idx,
                    value_embed.squeeze(1),  # Remove sequence dim
                    segment[0] + t,  # Global position
                )

                # Update current bindings
                bindings[var_id] = slot_idx

                # Update slot values for immediate use
                mask = mx.arange(self.num_slots) == slot_idx
                mask = mask[None, :, None]
                value_broadcast = value_embed[:, None, :]
                slot_values = mx.where(mask, value_broadcast, slot_values)

            # Check for temporal patterns
            is_temporal, repeat_count, var_name = self.detect_temporal_pattern(
                segment_tokens, t
            )

            if is_temporal and var_name:
                var_id = VOCAB[var_name]
                if var_id in bindings:
                    slot_idx = bindings[var_id]
                    # Retrieve from versioned memory
                    retrieved_value = self.versioned_memory.retrieve(
                        slot_idx, segment[0] + t
                    )
                    if retrieved_value is not None:
                        # Generate repeated actions
                        for _ in range(repeat_count):
                            outputs.append(self.action_decoder(retrieved_value))
                continue

            # Check if current token is an action position
            # Simple check: is this a variable token that should be executed?
            token_id = segment_tokens[0, t].item()
            is_do_context = t > 0 and segment_tokens[0, t - 1].item() == VOCAB.get(
                "do", -1
            )
            is_variable = token_id in [VOCAB["X"], VOCAB["Y"], VOCAB["Z"]]
            is_action = is_do_context and is_variable

            if is_action:
                # Variable or direct action
                token_id = segment_tokens[0, t].item()

                if token_id in bindings:
                    # Retrieved from versioned memory
                    slot_idx = bindings[token_id]
                    retrieved = self.versioned_memory.retrieve(slot_idx, segment[0] + t)

                    if retrieved is not None:
                        action_logits = self.action_decoder(retrieved)
                    else:
                        # Fallback to slot values
                        action_logits = self.action_decoder(slot_values[0, slot_idx])

                    outputs.append(action_logits)
                elif token_id == VOCAB.get("do", -1):
                    # Skip 'do' token
                    continue
                else:
                    # Direct action
                    output = self.recognition_decoder(current_embed[0])
                    outputs.append(output)

        return outputs

    def detect_temporal_pattern(
        self, tokens: mx.array, position: int
    ) -> Tuple[bool, int, Optional[str]]:
        """Detect temporal patterns like 'do X twice'."""
        if position >= len(tokens[0]):
            return False, 0, None

        token_id = tokens[0, position].item()

        # Check for temporal modifiers
        if token_id == VOCAB.get("twice", -1):
            # Look back for variable
            for i in range(position - 1, max(0, position - 3), -1):
                prev_token = tokens[0, i].item()
                for var_name in ["X", "Y", "Z"]:
                    if prev_token == VOCAB.get(var_name, -1):
                        return True, 2, var_name

        elif token_id == VOCAB.get("thrice", -1):
            # Look back for variable
            for i in range(position - 1, max(0, position - 3), -1):
                prev_token = tokens[0, i].item()
                for var_name in ["X", "Y", "Z"]:
                    if prev_token == VOCAB.get(var_name, -1):
                        return True, 3, var_name

        return False, 0, None

    def __call__(self, inputs: Dict[str, mx.array], stage: str = "full") -> mx.array:
        """Forward pass supporting all architectural components."""
        command_ids = inputs["command"]

        # Clear versioned memory for new sequence
        self.versioned_memory.clear()

        # Parse sequence for 'then' operators
        segments = self.sequence_planner.parse_sequence(command_ids)

        # Process each segment
        all_outputs = []
        bindings = {}  # Shared across segments

        for seg_start, seg_end in segments:
            segment_outputs = self.process_segment_versioned(
                command_ids, (seg_start, seg_end), bindings, stage
            )
            all_outputs.extend(segment_outputs)

        # Stack outputs
        if all_outputs:
            # Squeeze out extra dimensions and stack
            squeezed_outputs = []
            for out in all_outputs:
                if len(out.shape) > 1:
                    squeezed_outputs.append(out.squeeze())
                else:
                    squeezed_outputs.append(out)
            return mx.stack(squeezed_outputs)
        else:
            # Return dummy output if no actions
            return mx.zeros((1, self.num_actions))


def generate_rebinding_data(num_samples: int = 100) -> List[Dict[str, Any]]:
    """Generate data with variable rebinding patterns."""
    data = []

    # Ensure necessary tokens are in vocabulary
    if "then" not in VOCAB:
        VOCAB["then"] = len(VOCAB)
    if "and" not in VOCAB:
        VOCAB["and"] = len(VOCAB)

    patterns = [
        # Simple rebinding
        ("X means jump do X then X means walk do X", ["JUMP", "WALK"]),
        # Multiple rebindings
        (
            "X is turn do X then X is jump do X then X is walk do X",
            ["TURN", "JUMP", "WALK"],
        ),
        # Rebinding with temporal
        ("Y means jump do Y twice then Y means turn do Y", ["JUMP", "JUMP", "TURN"]),
        # Complex rebinding
        (
            "X means walk Y means jump do X and Y then X means turn do X",
            ["WALK", "JUMP", "TURN"],
        ),
    ]

    # Generate samples
    for _ in range(num_samples):
        pattern_idx = np.random.randint(len(patterns))
        command, expected = patterns[pattern_idx]

        # Tokenize
        tokens = [VOCAB.get(word, VOCAB["<PAD>"]) for word in command.split()]

        # Get expected action indices
        action_indices = [ACTIONS[action] for action in expected]

        data.append(
            {
                "command": np.array([tokens]),
                "target": np.array([action_indices]),
                "labels": np.array([action_indices]),
                "mask": np.ones((1, len(action_indices))),
            }
        )

    return data


def train_integrated_model():
    """Train the integrated model on all pattern types."""
    print("=== Training Integrated Variable Binding Model ===")
    print(
        "Components: Dynamic Memory + Temporal Buffer + Sequential Planning + Versioned Memory\n"
    )

    # Ensure necessary tokens are in vocabulary
    if "then" not in VOCAB:
        VOCAB["then"] = len(VOCAB)
    if "and" not in VOCAB:
        VOCAB["and"] = len(VOCAB)

    # Model setup
    model = IntegratedBindingModel(
        vocab_size=len(VOCAB), num_actions=len(ACTIONS), embed_dim=256, num_slots=4
    )

    optimizer = optim.Adam(learning_rate=0.0001)

    # Generate mixed training data
    print("Generating training data...")

    # Convert batch data to list format
    def batch_to_list(batch_data: Dict[str, mx.array], stage: str = None) -> List[Dict]:
        """Convert batched data to list of individual examples."""
        data_list = []
        batch_size = batch_data["command"].shape[0]

        for i in range(batch_size):
            item = {"command": batch_data["command"][i : i + 1], "stage": stage}

            if "target" in batch_data:
                item["target"] = batch_data["target"][i : i + 1]
            if "labels" in batch_data:
                item["labels"] = batch_data["labels"][i : i + 1]
                item["mask"] = mx.ones_like(item["labels"])

            data_list.append(item)

        return data_list

    stage1_data = batch_to_list(generate_stage1_data(500), "recognition")
    stage2_data = batch_to_list(generate_stage2_data(500), "retrieval")
    stage3_data = batch_to_list(generate_stage3_data(500), "full")
    rebinding_data = generate_rebinding_data(500)  # Already in list format

    # Combine all data
    all_data = stage1_data + stage2_data + stage3_data + rebinding_data

    # Training loop
    num_epochs = 50
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        # Shuffle data
        np.random.shuffle(all_data)

        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(all_data, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in progress_bar:
            # Forward pass
            inputs = {
                "command": mx.array(batch["command"]),
                "target": mx.array(batch["target"]) if "target" in batch else None,
            }

            # Determine stage based on data
            if "stage" in batch:
                stage = batch["stage"]
            else:
                # Infer stage from pattern
                stage = "full"

            # Compute loss
            def loss_fn(model):
                outputs = model(inputs, stage=stage)

                # Get targets
                if "labels" in batch:
                    labels = mx.array(batch["labels"][0])
                else:
                    labels = mx.array(batch["target"][0])

                # Ensure outputs match label length
                if len(outputs) > len(labels):
                    outputs = outputs[: len(labels)]
                elif len(outputs) < len(labels):
                    # Pad outputs
                    padding = mx.zeros((len(labels) - len(outputs), outputs.shape[-1]))
                    outputs = mx.concatenate([outputs, padding])

                # Cross entropy loss
                loss = mx.mean(nn.losses.cross_entropy(outputs, labels))

                # Accuracy
                predictions = mx.argmax(outputs, axis=1)
                accuracy = mx.mean(predictions == labels)

                return loss, accuracy

            # Gradient step - MLX doesn't support has_aux
            loss_val, accuracy = loss_fn(model)

            # Compute gradients separately
            def loss_only(model):
                outputs = model(inputs, stage=stage)

                # Get targets
                if "labels" in batch:
                    labels = mx.array(batch["labels"][0])
                else:
                    labels = mx.array(batch["target"][0])

                # Ensure outputs match label length
                if len(outputs) > len(labels):
                    outputs = outputs[: len(labels)]
                elif len(outputs) < len(labels):
                    # Pad outputs
                    padding = mx.zeros((len(labels) - len(outputs), outputs.shape[-1]))
                    outputs = mx.concatenate([outputs, padding])

                # Cross entropy loss
                return mx.mean(nn.losses.cross_entropy(outputs, labels))

            grad_fn = mx.grad(loss_only)
            grads = grad_fn(model)

            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            # Update stats
            total_loss += float(loss_val)

            # Get the actual label count
            if "labels" in batch:
                label_count = (
                    len(batch["labels"][0])
                    if isinstance(batch["labels"], list)
                    else batch["labels"].shape[1]
                )
            else:
                label_count = (
                    len(batch["target"][0])
                    if isinstance(batch["target"], list)
                    else batch["target"].shape[1]
                )

            correct += float(accuracy) * label_count
            total += label_count

            # Update progress bar
            progress_bar.set_postfix(
                {"loss": f"{float(loss_val):.4f}", "acc": f"{correct/total:.2%}"}
            )

        # Epoch summary
        epoch_accuracy = correct / total
        print(
            f"\nEpoch {epoch+1}: Loss={total_loss/len(all_data):.4f}, Accuracy={epoch_accuracy:.2%}"
        )

        # Save best model
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            save_path = os.path.join(get_output_path(), "integrated_model_best.pkl")
            save_model_simple(save_path, model)
            print(f"Saved best model with accuracy {best_accuracy:.2%}")

    print(f"\nTraining complete! Best accuracy: {best_accuracy:.2%}")

    # Save final model
    final_path = os.path.join(get_output_path(), "integrated_model_final.pkl")
    save_model_simple(final_path, model)
    print(f"Saved final model to {final_path}")

    return model


def test_integrated_model(model: IntegratedBindingModel):
    """Test the integrated model on various pattern types."""
    print("\n=== Testing Integrated Model ===")

    test_cases = [
        # Basic binding
        ("X means jump do X", ["JUMP"]),
        # Temporal patterns
        ("Y means walk do Y twice", ["WALK", "WALK"]),
        ("Z means turn do Z thrice", ["TURN", "TURN", "TURN"]),
        # Sequential patterns
        ("X means jump do X then Y means walk do Y", ["JUMP", "WALK"]),
        # Rebinding patterns
        ("X means jump do X then X means walk do X", ["JUMP", "WALK"]),
        # Complex combinations
        (
            "X means jump do X twice then X means turn do X thrice",
            ["JUMP", "JUMP", "TURN", "TURN", "TURN"],
        ),
    ]

    for command, expected in test_cases:
        print(f"\nTest: {command}")
        print(f"Expected: {expected}")

        # Tokenize
        tokens = [VOCAB.get(word, VOCAB["<PAD>"]) for word in command.split()]
        inputs = {"command": mx.array([tokens])}

        # Get predictions
        outputs = model(inputs, stage="full")
        predictions = mx.argmax(outputs, axis=1)

        # Convert to actions
        predicted_actions = []
        for pred in predictions:
            for action_name, action_id in ACTIONS.items():
                if action_id == int(pred):
                    predicted_actions.append(action_name)
                    break

        print(f"Predicted: {predicted_actions}")
        print(f"Correct: {predicted_actions == expected}")


if __name__ == "__main__":
    # Train model
    model = train_integrated_model()

    # Test model
    test_integrated_model(model)
