"""
Train variable binding model with sequence planning - MLX-compatible version.
Fixed autodiff issues by replacing problematic operations.
"""

from utils.imports import setup_project_paths

setup_project_paths()

import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from tqdm import tqdm
from train_binding_curriculum import (
    ACTIONS,
    VOCAB,
    generate_stage3_data,
)
from train_temporal_curriculum import TemporalActionBuffer

from utils.config import setup_environment
from utils.paths import get_output_path

config = setup_environment()


class SequencePlanner:
    """Parse and plan execution of sequential commands with 'then' operator."""

    def __init__(self):
        self.then_token = VOCAB.get("then", None)

    def parse_sequence(self, tokens: mx.array) -> List[Tuple[int, int]]:
        """Parse tokens into sequential segments separated by 'then'."""
        segments = []
        current_start = 0

        # Convert to numpy for easier manipulation
        if hasattr(tokens, "numpy"):
            tokens_np = tokens.numpy()
        else:
            tokens_np = np.array(tokens)

        # Handle batch dimension if present
        if len(tokens_np.shape) > 1:
            tokens_np = tokens_np[0]  # Take first batch item

        # Find all 'then' positions
        then_positions = []
        if self.then_token is not None:
            for i, token in enumerate(tokens_np):
                if token == self.then_token:
                    then_positions.append(i)

        # Create segments
        if not then_positions:
            segments.append((0, len(tokens_np)))
        else:
            for then_pos in then_positions:
                segments.append((current_start, then_pos))
                current_start = then_pos + 1
            segments.append((current_start, len(tokens_np)))

        return segments


class SequentialDynamicMemoryModel(nn.Module):
    """Variable binding model with sequence planning - MLX compatible."""

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

        # Core components
        self.vocab_size = vocab_size
        self.num_actions = num_actions
        self.embed_dim = embed_dim
        self.num_slots = num_slots

        # Embeddings and encoding
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = nn.Embedding(50, embed_dim)

        # Sequence planning
        self.sequence_planner = SequencePlanner()

        # Temporal action buffer
        self.temporal_buffer = TemporalActionBuffer()

        # Memory components
        self.slot_keys = mx.random.normal((num_slots, embed_dim))
        self.binder = BindingAttentionFixed(embed_dim, num_heads)

        # Prediction heads
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

    def detect_temporal_modifiers(
        self, tokens: mx.array, position: int
    ) -> Tuple[bool, int, Optional[str]]:
        """Detect patterns like 'do X twice' or 'do Y thrice'."""
        if position < len(tokens[0]) and position >= 0:
            token_id = tokens[0, position].item()

            # Check if this is a temporal modifier
            if token_id == VOCAB.get("twice", -1):
                # Look back for the variable
                for i in range(position - 1, max(0, position - 3), -1):
                    prev_token = tokens[0, i].item()
                    if prev_token in [VOCAB["X"], VOCAB["Y"], VOCAB["Z"]]:
                        # Find which variable it is
                        for var_name, var_id in VOCAB.items():
                            if var_id == prev_token and var_name in ["X", "Y", "Z"]:
                                return True, 2, var_name

            elif token_id == VOCAB.get("thrice", -1):
                # Look back for the variable
                for i in range(position - 1, max(0, position - 3), -1):
                    prev_token = tokens[0, i].item()
                    if prev_token in [VOCAB["X"], VOCAB["Y"], VOCAB["Z"]]:
                        # Find which variable it is
                        for var_name, var_id in VOCAB.items():
                            if var_id == prev_token and var_name in ["X", "Y", "Z"]:
                                return True, 3, var_name

        return False, 0, None

    def process_segment(
        self,
        command_ids: mx.array,
        segment: Tuple[int, int],
        slot_values: mx.array,
        bindings: Dict,
        stage: str,
    ) -> Tuple[List[mx.array], List[mx.array]]:
        """Process a single command segment."""
        outputs = []
        temporal_actions = []
        action_buffer = TemporalActionBuffer()

        # Extract segment tokens
        segment_tokens = command_ids[:, segment[0] : segment[1]]
        segment_length = segment[1] - segment[0]

        # Process each token in segment
        for t in range(segment_length):
            # Get embeddings
            token_embed = self.token_embeddings(segment_tokens[:, t : t + 1])
            pos_embed = self.position_embeddings(mx.array([t]))
            current_embed = token_embed + pos_embed

            # Check for storage pattern
            is_storage = False
            if t + 2 < segment_length:
                is_means = segment_tokens[:, t + 1] == VOCAB["means"]
                is_is = segment_tokens[:, t + 1] == VOCAB["is"]
                is_storage = mx.any(is_means | is_is)

            # Compute binding scores
            binding_scores, raw_scores = self.binder(
                current_embed, self.slot_keys, training=(stage != "recognition")
            )

            # Update bindings if storage pattern
            if is_storage:
                var_id = segment_tokens[0, t].item()
                slot_idx = mx.argmax(binding_scores[0]).item()
                bindings[var_id] = slot_idx

                # Store value in slot
                if t + 2 < segment_length:
                    value_embed = self.token_embeddings(
                        segment_tokens[:, t + 2 : t + 3]
                    )
                    # Create mask for the slot to update
                    mask = mx.arange(self.num_slots) == slot_idx
                    # Expand mask to match slot_values shape
                    mask = mask[None, :, None]  # (1, num_slots, 1)
                    # Update slot_values
                    # Ensure value_embed has correct shape
                    if len(value_embed.shape) == 3:
                        value_embed = value_embed.squeeze(1)  # Remove sequence dim

                    # Broadcast value_embed to slot shape
                    value_broadcast = mx.zeros_like(slot_values)
                    value_broadcast = mx.where(
                        mask,
                        value_embed[:, None, :],  # (batch, 1, embed_dim)
                        value_broadcast,
                    )

                    # Update only the selected slot
                    slot_values = mx.where(mask, value_broadcast, slot_values)

            # Check for temporal patterns
            is_temporal, repeat_count, target_var = self.detect_temporal_modifiers(
                segment_tokens, t
            )

            if is_temporal and target_var is not None:
                # Handle temporal repetition using continuous retrieval
                var_id = VOCAB[target_var]
                if var_id in bindings:
                    # Use continuous retrieval instead of discrete indexing
                    # Create one-hot binding for this variable
                    slot_idx = bindings[var_id]
                    one_hot_binding = mx.zeros((1, self.num_slots))
                    one_hot_binding = mx.where(
                        mx.arange(self.num_slots) == slot_idx, 1.0, one_hot_binding
                    )
                    # Retrieve using weighted sum
                    retrieved = (one_hot_binding[:, None, :] @ slot_values).squeeze(1)

                    for _ in range(repeat_count):
                        temporal_actions.append(retrieved)
            else:
                # Normal retrieval - already continuous
                binding_scores_unsqueezed = binding_scores[:, None, :]

                matmul_result = binding_scores_unsqueezed @ slot_values

                if matmul_result.shape[1] == 1:
                    retrieved = matmul_result.squeeze(1)
                else:
                    # Fallback: use element-wise multiply and sum
                    binding_expanded = binding_scores[:, :, None]
                    weighted = slot_values * binding_expanded
                    retrieved = mx.sum(weighted, axis=1)

                # Check if this is an action position
                if t > 0:
                    prev_was_do = segment_tokens[:, t - 1] == VOCAB["do"]
                    is_var = mx.logical_or(
                        segment_tokens[:, t] == VOCAB["X"],
                        mx.logical_or(
                            segment_tokens[:, t] == VOCAB["Y"],
                            segment_tokens[:, t] == VOCAB["Z"],
                        ),
                    )

                    if mx.any(prev_was_do & is_var):
                        action_buffer.push(retrieved, segment_tokens[0, t].item())

            # Prepare output
            if stage == "recognition":
                if current_embed.shape[1] == 1:
                    output = current_embed.squeeze(1)
                else:
                    output = current_embed[:, 0, :]
            else:
                output = retrieved
                if len(output.shape) == 1:
                    output = output[None, :]

            outputs.append(output)

        return outputs, temporal_actions

    def __call__(
        self,
        command_ids: mx.array,
        action_ids: Optional[mx.array] = None,
        stage: str = "recognition",
    ) -> Dict:
        batch_size = command_ids.shape[0]

        # Parse sequence into segments
        segments = self.sequence_planner.parse_sequence(command_ids)

        # Initialize memory
        slot_values = mx.zeros((batch_size, self.num_slots, self.embed_dim))
        bindings = {}

        # Process each segment sequentially
        all_outputs = []
        all_temporal_actions = []

        for segment in segments:
            segment_outputs, segment_temporal = self.process_segment(
                command_ids, segment, slot_values, bindings, stage
            )
            all_outputs.extend(segment_outputs)
            all_temporal_actions.extend(segment_temporal)

        # Stack outputs
        outputs_stacked = mx.stack(all_outputs, axis=1)

        # Add temporal actions if any
        if all_temporal_actions:
            # Check shape and stack appropriately
            if len(all_temporal_actions[0].shape) == 1:
                temporal_stacked = mx.stack(all_temporal_actions, axis=0)
                temporal_stacked = temporal_stacked[None, :, :]
            else:
                temporal_stacked = mx.stack(all_temporal_actions, axis=1)

            outputs_stacked = mx.concatenate(
                [outputs_stacked, temporal_stacked], axis=1
            )

        # Final decoding
        if stage == "recognition":
            recognition_logits = self.recognition_decoder(outputs_stacked)
            return {
                "recognition_logits": recognition_logits,
                "bindings": bindings,
                "slot_values": slot_values,
                "segments": segments,
                "temporal_actions": len(all_temporal_actions),
            }
        else:
            action_logits = self.action_decoder(outputs_stacked)
            return {
                "action_logits": action_logits,
                "bindings": bindings,
                "slot_values": slot_values,
                "segments": segments,
                "temporal_actions": len(all_temporal_actions),
            }


class BindingAttentionFixed(nn.Module):
    """Attention mechanism for variable-slot binding - MLX compatible version."""

    def __init__(
        self, embed_dim: int, num_heads: int = 8, initial_temperature: float = 1.0
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.temperature = initial_temperature

        # Multi-head projections
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)

    def __call__(
        self, query: mx.array, keys: mx.array, training: bool = True
    ) -> Tuple[mx.array, mx.array]:
        batch_size = query.shape[0]
        num_slots = keys.shape[0]

        # Project query and keys
        Q = self.query_proj(query)  # (batch, 1, embed_dim)
        K = self.key_proj(keys)  # (num_slots, embed_dim)

        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, 1, self.num_heads, self.head_dim)
        K = K.reshape(num_slots, self.num_heads, self.head_dim)
        K = mx.repeat(mx.expand_dims(K, 0), batch_size, axis=0)

        # Compute attention scores
        K_transposed = mx.transpose(K, (0, 2, 3, 1))
        Q_reshaped = mx.transpose(Q, (0, 2, 1, 3))
        scores = Q_reshaped @ K_transposed
        scores = scores.squeeze(2)
        scores = scores / mx.sqrt(mx.array(self.head_dim))

        # Average over heads
        scores = mx.mean(scores, axis=1)  # (batch, num_slots)

        # Apply softmax with temperature for continuous attention
        if training:
            # Add small Gumbel noise for exploration during training
            noise = -mx.log(
                -mx.log(mx.random.uniform(shape=scores.shape) + 1e-10) + 1e-10
            )
            scores_with_noise = (scores + noise * 0.1) / self.temperature
            binding_weights = nn.softmax(scores_with_noise, axis=-1)
        else:
            # During inference, use standard softmax
            binding_weights = nn.softmax(scores / self.temperature, axis=-1)

        return binding_weights, scores


def create_sequential_dataset(num_samples: int = 1000) -> Dict:
    """Create dataset with sequential composition patterns."""
    # Ensure 'then' is in vocabulary
    if "then" not in VOCAB:
        VOCAB["then"] = len(VOCAB)

    # Generate basic patterns from stage 3
    stage3_data = generate_stage3_data(num_samples // 2)
    basic_commands = []
    basic_actions = []

    for i in range(len(stage3_data["command"])):
        # Convert command IDs back to string
        cmd_ids = stage3_data["command"][i]
        cmd_tokens = []
        for token_id in cmd_ids:
            if token_id == VOCAB["<PAD>"]:
                break
            for token, tid in VOCAB.items():
                if tid == token_id:
                    cmd_tokens.append(token)
                    break

        # Convert action IDs
        act_ids = stage3_data["labels"][i]
        act_tokens = []
        for action_id in act_ids:
            if action_id == ACTIONS["<PAD>"]:
                break
            for action, aid in ACTIONS.items():
                if aid == action_id:
                    act_tokens.append(action)
                    break

        basic_commands.append(" ".join(cmd_tokens))
        basic_actions.append(act_tokens)

    # Generate sequential patterns
    sequential_commands = []
    sequential_actions = []

    variables = ["X", "Y", "Z"]
    actions = ["jump", "walk", "turn", "run"]

    for _ in range(num_samples // 2):
        # Random pattern: "V1 means A1 do V1 then V2 means A2 do V2"
        v1, v2 = np.random.choice(variables, 2, replace=False)
        a1, a2 = np.random.choice(actions, 2, replace=False)

        command = f"{v1} means {a1} do {v1} then {v2} means {a2} do {v2}"
        expected = [a1.upper(), a2.upper()]

        sequential_commands.append(command)
        sequential_actions.append(expected)

        # Also add some with temporal patterns
        if np.random.random() > 0.5:
            v3 = np.random.choice(variables)
            a3 = np.random.choice(actions)
            modifier = np.random.choice(["twice", "thrice"])
            repeat = 2 if modifier == "twice" else 3

            command = f"{v3} means {a3} do {v3} {modifier} then {v1} means {a1} do {v1}"
            expected = [a3.upper()] * repeat + [a1.upper()]

            sequential_commands.append(command)
            sequential_actions.append(expected)

    # Combine datasets
    all_commands = basic_commands + sequential_commands
    all_actions = basic_actions + sequential_actions

    return {
        "commands": all_commands,
        "actions": all_actions,
        "vocab": VOCAB,
        "actions_vocab": ACTIONS,
        "max_seq_len": 30,
    }


def train_step(
    model: SequentialDynamicMemoryModel,
    batch: Dict,
    optimizer: optim.Optimizer,
    stage: str,
) -> Dict:
    """Single training step."""

    def loss_fn(model, command_ids, action_ids, stage):
        # Forward pass
        outputs = model(command_ids, stage=stage)

        # Compute loss based on stage
        if stage == "recognition" and "recognition_logits" in outputs:
            # Recognition loss
            var_mask = mx.logical_or(
                command_ids == VOCAB["X"],
                mx.logical_or(command_ids == VOCAB["Y"], command_ids == VOCAB["Z"]),
            )
            if mx.any(var_mask):
                recog_logits = outputs["recognition_logits"]

                # Compute loss only on masked positions
                # Ensure we only use positions that exist in both logits and targets
                batch_size, cmd_len = command_ids.shape
                pred_len = recog_logits.shape[1]

                # Truncate to minimum length
                min_len = min(cmd_len, pred_len)
                recog_logits_truncated = recog_logits[:, :min_len, :]
                command_ids_truncated = command_ids[:, :min_len]
                var_mask_truncated = var_mask[:, :min_len]

                # Reshape for loss computation
                logits_flat = recog_logits_truncated.reshape(
                    -1, recog_logits_truncated.shape[-1]
                )
                targets_flat = command_ids_truncated.reshape(-1)
                mask_flat = var_mask_truncated.reshape(-1)

                # Compute cross entropy for all positions
                all_losses = nn.losses.cross_entropy(
                    logits_flat, targets_flat, reduction="none"
                )

                # Apply mask and take mean only over valid positions
                masked_losses = all_losses * mask_flat
                num_valid = mx.sum(mask_flat)

                if num_valid > 0:
                    loss = mx.sum(masked_losses) / num_valid
                else:
                    loss = mx.array(0.0)
            else:
                loss = mx.array(0.0)

        elif "action_logits" in outputs:
            # Action prediction loss
            action_logits = outputs["action_logits"]

            # Handle variable length outputs
            batch_size, seq_len = action_ids.shape
            pred_len = action_logits.shape[1]

            if pred_len >= seq_len:
                action_logits = action_logits[:, :seq_len, :]
            else:
                action_ids = action_ids[:, :pred_len]

            # Mask out padding
            mask = action_ids != ACTIONS["<PAD>"]
            if mx.any(mask):
                # Compute loss without boolean indexing
                logits_flat = action_logits.reshape(-1, action_logits.shape[-1])
                targets_flat = action_ids.reshape(-1)
                mask_flat = mask.reshape(-1)

                # Compute cross entropy for all positions
                all_losses = nn.losses.cross_entropy(
                    logits_flat, targets_flat, reduction="none"
                )

                # Apply mask and take mean only over valid positions
                masked_losses = all_losses * mask_flat
                num_valid = mx.sum(mask_flat)

                if num_valid > 0:
                    loss = mx.sum(masked_losses) / num_valid
                else:
                    loss = mx.array(0.0)
            else:
                loss = mx.array(0.0)
        else:
            loss = mx.array(0.0)

        return loss

    # Compute loss and gradients
    loss_and_grad_fn = mx.value_and_grad(loss_fn)
    loss, grads = loss_and_grad_fn(
        model, batch["command_ids"], batch["action_ids"], stage
    )

    # Update model
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)

    return {
        "loss": loss.item() if hasattr(loss, "item") else float(loss),
        "stage": stage,
    }


def evaluate_model(
    model: SequentialDynamicMemoryModel, dataset: Dict, stage: str = "full_binding"
) -> Dict:
    """Evaluate model performance."""
    model.eval()

    correct = 0
    total = 0
    correct_sequential = 0
    total_sequential = 0

    for i in range(min(100, len(dataset["commands"]))):  # Evaluate on subset
        command = dataset["commands"][i]
        expected_actions = dataset["actions"][i]

        # Check if this is a sequential pattern
        is_sequential = "then" in command

        # Prepare batch
        command_tokens = command.split()
        command_ids = mx.array(
            [[VOCAB.get(token, VOCAB["<PAD>"]) for token in command_tokens]]
        )

        # Forward pass
        outputs = model(command_ids, stage=stage)

        if "action_logits" in outputs:
            # Get predictions
            predictions = mx.argmax(outputs["action_logits"], axis=-1)
            predictions_np = np.array(predictions)

            # Extract predicted actions
            predicted_actions = []
            for j in range(min(predictions_np.shape[1], len(expected_actions))):
                pred_id = int(predictions_np[0, j])
                for action_name, action_id in ACTIONS.items():
                    if action_id == pred_id and action_name != "<PAD>":
                        predicted_actions.append(action_name)
                        break

            # Check correctness
            is_correct = predicted_actions == expected_actions[: len(predicted_actions)]

            if is_correct and len(predicted_actions) == len(expected_actions):
                correct += 1
                if is_sequential:
                    correct_sequential += 1

            if is_sequential:
                total_sequential += 1

        total += 1

    results = {
        "accuracy": correct / total if total > 0 else 0,
        "total_evaluated": total,
    }

    if total_sequential > 0:
        results["sequential_accuracy"] = correct_sequential / total_sequential
        results["sequential_total"] = total_sequential

    return results


def main():
    # Configuration
    num_epochs = 10
    batch_size = 16
    learning_rate = 0.001
    num_samples = 500
    eval_interval = 2

    # Create dataset with sequential patterns
    print("Creating dataset with sequential patterns...")
    dataset = create_sequential_dataset(num_samples=num_samples)

    # Initialize model
    model = SequentialDynamicMemoryModel(
        vocab_size=len(VOCAB),
        num_actions=len(ACTIONS),
        embed_dim=128,
        num_slots=4,
        num_heads=4,
        mlp_hidden_dim=256,
    )

    # Initialize optimizer
    optimizer = optim.Adam(learning_rate=learning_rate)

    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        epoch_losses = []

        # Create random batches
        indices = np.random.permutation(len(dataset["commands"]))

        for i in tqdm(
            range(0, len(indices), batch_size), desc=f"Epoch {epoch+1}/{num_epochs}"
        ):
            batch_indices = indices[i : i + batch_size]

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

            # Train on both stages
            stage = "full_binding" if epoch > 2 else "recognition"

            # Train step
            step_results = train_step(model, batch, optimizer, stage)
            epoch_losses.append(step_results["loss"])

        # Print epoch summary
        avg_loss = np.mean(epoch_losses)
        print(f"\nEpoch {epoch+1} - Average Loss: {avg_loss:.4f}")

        # Evaluate periodically
        if (epoch + 1) % eval_interval == 0:
            print("\nEvaluating...")
            results = evaluate_model(model, dataset, stage="full_binding")
            print(f"Overall Accuracy: {results['accuracy']*100:.1f}%")
            if "sequential_accuracy" in results:
                print(
                    f"Sequential Pattern Accuracy: {results['sequential_accuracy']*100:.1f}% "
                    f"({results['sequential_total']} patterns)"
                )

    # Final evaluation
    print("\n" + "=" * 50)
    print("FINAL EVALUATION")
    print("=" * 50)

    final_results = evaluate_model(model, dataset, stage="full_binding")
    print(f"Final Accuracy: {final_results['accuracy']*100:.1f}%")
    if "sequential_accuracy" in final_results:
        print(
            f"Sequential Pattern Accuracy: {final_results['sequential_accuracy']*100:.1f}%"
        )

    # Test specific patterns
    print("\n" + "=" * 50)
    print("TESTING SPECIFIC SEQUENTIAL PATTERNS")
    print("=" * 50)

    test_patterns = [
        "X means jump do X",
        "X means jump do X then Y means walk do Y",
        "Z means turn do Z twice then X means run do X",
    ]

    for pattern in test_patterns:
        command_tokens = pattern.split()
        command_ids = mx.array(
            [[VOCAB.get(token, VOCAB["<PAD>"]) for token in command_tokens]]
        )

        results = model(command_ids, stage="full_binding")
        segments = results.get("segments", [])

        # Get predictions
        if "action_logits" in results:
            predictions = mx.argmax(results["action_logits"], axis=-1)
            predictions_np = np.array(predictions)

            # Extract predicted actions
            predicted_actions = []
            for j in range(predictions_np.shape[1]):
                if j >= 10:  # Limit output
                    break
                pred_id = int(predictions_np[0, j])
                for action_name, action_id in ACTIONS.items():
                    if action_id == pred_id and action_name != "<PAD>":
                        predicted_actions.append(action_name)
                        break

        print(f"\nPattern: {pattern}")
        print(f"Segments: {segments}")
        print(f"Predicted: {predicted_actions[:5]}")  # Show first 5 predictions

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = get_output_path("models")
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, f"sequential_fixed_{timestamp}.npz")
    # MLX save format - need to save each parameter separately
    weights = dict(model.parameters())
    mx.savez(model_path, **weights)
    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()
