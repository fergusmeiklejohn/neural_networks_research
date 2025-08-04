"""
Train variable binding model with sequence planning for "then" operator support.

This extends the temporal dynamic memory model to handle sequential composition
patterns like "do X then do Y".
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
        """
        Parse tokens into sequential segments separated by 'then'.
        Returns list of (start_idx, end_idx) tuples for each segment.
        """
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
            # No 'then' found, entire sequence is one segment
            segments.append((0, len(tokens_np)))
        else:
            # Multiple segments
            for then_pos in then_positions:
                segments.append((current_start, then_pos))
                current_start = then_pos + 1
            # Add final segment
            segments.append((current_start, len(tokens_np)))

        return segments

    def extract_command_from_segment(
        self, tokens: mx.array, segment: Tuple[int, int]
    ) -> mx.array:
        """Extract the command tokens from a segment."""
        start, end = segment
        if len(tokens.shape) > 1:
            return tokens[:, start:end]
        else:
            return tokens[start:end]


class SequentialDynamicMemoryModel(nn.Module):
    """Variable binding model with sequence planning and temporal action support."""

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
        self.binder = BindingAttention(embed_dim, num_heads)

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
        """
        Detect patterns like "do X twice" or "do Y thrice"
        Returns: (is_modifier, repeat_count, target_variable)
        """
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
                    slot_values = mx.where(
                        mask,
                        value_embed[:, None, :],  # (batch, 1, embed_dim)
                        slot_values,
                    )

            # Check for temporal patterns
            is_temporal, repeat_count, target_var = self.detect_temporal_modifiers(
                segment_tokens, t
            )

            if is_temporal and target_var is not None:
                # Handle temporal repetition
                var_id = VOCAB[target_var]
                if var_id in bindings:
                    slot_idx = bindings[var_id]
                    # Directly retrieve from the bound slot
                    # slot_values: (batch, num_slots, embed_dim)
                    try:
                        retrieved = slot_values[:, slot_idx, :]  # (batch, embed_dim)
                        # Ensure batch dimension is preserved
                        if len(retrieved.shape) == 1:
                            retrieved = retrieved[None, :]  # Add batch dimension back

                        for _ in range(repeat_count):
                            temporal_actions.append(retrieved)
                    except Exception:
                        # If retrieval fails, skip temporal actions
                        pass
            else:
                # Normal retrieval
                # binding_scores: (batch, num_slots)
                # slot_values: (batch, num_slots, embed_dim)
                # We want: (batch, embed_dim)
                # Use matrix multiplication: (batch, 1, num_slots) @ (batch, num_slots, embed_dim) = (batch, 1, embed_dim)
                binding_scores_unsqueezed = binding_scores[
                    :, None, :
                ]  # (batch, 1, num_slots)
                retrieved = (binding_scores_unsqueezed @ slot_values).squeeze(
                    1
                )  # (batch, embed_dim)

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
                # current_embed is (batch, 1, embed_dim), squeeze the sequence dimension
                if current_embed.shape[1] == 1:
                    output = current_embed.squeeze(1)
                else:
                    # If not 1, just take the current position
                    output = current_embed[:, 0, :]  # Take first position
            else:
                output = retrieved
                if len(output.shape) == 1:
                    output = output[None, :]
                # No need to squeeze if already 2D

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
                # Missing batch dimension, stack on axis 0 and add batch dim
                temporal_stacked = mx.stack(all_temporal_actions, axis=0)
                temporal_stacked = temporal_stacked[None, :, :]  # Add batch dimension
            else:
                # Has batch dimension, stack normally
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


class BindingAttention(nn.Module):
    """Attention mechanism for variable-slot binding with Gumbel-Softmax"""

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

    def gumbel_softmax(
        self, logits: mx.array, temperature: float = 1.0, hard: bool = True
    ) -> mx.array:
        """Gumbel-Softmax for differentiable discrete sampling"""
        # Sample from Gumbel(0, 1)
        gumbels = -mx.log(
            -mx.log(mx.random.uniform(shape=logits.shape) + 1e-10) + 1e-10
        )

        # Add Gumbel noise to logits and apply temperature
        y = nn.softmax((logits + gumbels) / temperature, axis=-1)

        if hard:
            # Straight-through estimator
            index = mx.argmax(y, axis=-1, keepdims=True)
            y_hard = mx.zeros_like(y)
            y_hard = mx.put_along_axis(y_hard, index, mx.array(1.0), axis=-1)
            y = y_hard - mx.stop_gradient(y) + y

        return y

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
        K = mx.repeat(
            mx.expand_dims(K, 0), batch_size, axis=0
        )  # (batch, num_slots, num_heads, head_dim)

        # Compute attention scores using proper matrix multiplication
        # Q: (batch, 1, num_heads, head_dim)
        # K: (batch, num_slots, num_heads, head_dim)
        # K transposed: (batch, num_heads, head_dim, num_slots)
        K_transposed = mx.transpose(K, (0, 2, 3, 1))
        # Q reshaped: (batch, num_heads, 1, head_dim)
        Q_reshaped = mx.transpose(Q, (0, 2, 1, 3))
        # scores: (batch, num_heads, 1, num_slots)
        scores = Q_reshaped @ K_transposed
        # scores: (batch, num_heads, num_slots)
        scores = scores.squeeze(2)
        scores = scores / mx.sqrt(mx.array(self.head_dim))

        # Average over heads
        scores = mx.mean(scores, axis=1)  # (batch, num_slots)

        # Apply Gumbel-Softmax for differentiable discrete selection
        if training:
            binding_weights = self.gumbel_softmax(
                scores, temperature=self.temperature, hard=True
            )
        else:
            # During inference, use hard argmax
            binding_weights = mx.zeros_like(scores)
            indices = mx.argmax(scores, axis=-1, keepdims=True)
            binding_weights = mx.put_along_axis(
                binding_weights, indices, mx.array(1.0), axis=-1
            )

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
        "max_seq_len": 30,  # Increased for sequential patterns
    }


def train_step(
    model: SequentialDynamicMemoryModel,
    batch: Dict,
    optimizer: optim.Optimizer,
    stage: str,
) -> Dict:
    """Single training step with mixed curriculum."""

    def loss_fn(model, command_ids, action_ids, stage):
        # Get all three stages' outputs in one pass
        results = {}

        # Recognition stage
        recog_out = model(command_ids, action_ids, stage="recognition")
        results["recognition"] = recog_out

        # Retrieval stage
        retrieval_out = model(command_ids, action_ids, stage="retrieval")
        results["retrieval"] = retrieval_out

        # Full binding stage
        binding_out = model(command_ids, action_ids, stage="full_binding")
        results["binding"] = binding_out

        # Compute losses for all stages
        losses = {}

        # Recognition loss (identify variables)
        if "recognition_logits" in recog_out:
            var_mask = mx.logical_or(
                command_ids == VOCAB["X"],
                mx.logical_or(command_ids == VOCAB["Y"], command_ids == VOCAB["Z"]),
            )
            if mx.any(var_mask):
                recog_logits = recog_out["recognition_logits"]
                recog_targets = mx.where(var_mask, command_ids, -100)
                losses["recognition"] = nn.losses.cross_entropy(
                    recog_logits[var_mask], recog_targets[var_mask], reduction="mean"
                )
            else:
                losses["recognition"] = mx.array(0.0)

        # Action prediction losses
        for stage_name, stage_out in [
            ("retrieval", retrieval_out),
            ("binding", binding_out),
        ]:
            if "action_logits" in stage_out:
                action_logits = stage_out["action_logits"]

                # Handle variable length outputs
                batch_size, seq_len = action_ids.shape
                pred_len = action_logits.shape[1]

                if pred_len >= seq_len:
                    # Predictions are longer (due to temporal actions)
                    action_logits_truncated = action_logits[:, :seq_len, :]
                    losses[stage_name] = nn.losses.cross_entropy(
                        action_logits_truncated.reshape(-1, action_logits.shape[-1]),
                        action_ids.reshape(-1),
                        reduction="mean",
                    )
                else:
                    # Predictions are shorter (shouldn't happen)
                    losses[stage_name] = nn.losses.cross_entropy(
                        action_logits.reshape(-1, action_logits.shape[-1]),
                        action_ids[:, :pred_len].reshape(-1),
                        reduction="mean",
                    )

        # Combine losses with weighting
        total_loss = (
            losses["recognition"] * 0.2
            + losses.get("retrieval", 0) * 0.3
            + losses.get("binding", 0) * 0.5
        )

        # Store results for later
        model._last_results = (losses, results)

        return total_loss

    # Compute loss and gradients
    loss_and_grad_fn = mx.value_and_grad(loss_fn)
    loss, grads = loss_and_grad_fn(
        model, batch["command_ids"], batch["action_ids"], stage
    )

    # Retrieve stored results
    losses, results = model._last_results

    # Update model
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)

    return {
        "loss": loss.item(),
        "losses": {k: v.item() if hasattr(v, "item") else v for k, v in losses.items()},
        "segments": results["binding"].get("segments", []),
    }


def evaluate_model(
    model: SequentialDynamicMemoryModel, dataset: Dict, stage: str = "full_binding"
) -> Dict:
    """Evaluate model performance."""
    model.eval()

    correct_stage1 = 0
    correct_stage2 = 0
    correct_stage3 = 0
    correct_sequential = 0
    total_sequential = 0
    total = 0

    for i in range(len(dataset["commands"])):
        command = dataset["commands"][i]
        expected_actions = dataset["actions"][i]

        # Check if this is a sequential pattern
        is_sequential = "then" in command

        # Prepare batch
        command_tokens = command.split()
        command_ids = mx.array(
            [[VOCAB.get(token, VOCAB["<PAD>"]) for token in command_tokens]]
        )

        # Forward pass (MLX doesn't need no_grad)
        results = model(command_ids, stage=stage)

        # Evaluate based on stage
        if stage == "recognition":
            # Stage 1: Variable recognition
            predictions = mx.argmax(results["recognition_logits"], axis=-1)
            var_positions = [
                j for j, token in enumerate(command_tokens) if token in ["X", "Y", "Z"]
            ]

            if all(
                predictions[0, j].item() == VOCAB[command_tokens[j]]
                for j in var_positions
            ):
                correct_stage1 += 1

        elif stage in ["retrieval", "full_binding"]:
            # Get action predictions
            action_logits = results["action_logits"]
            predictions = mx.argmax(action_logits, axis=-1)

            # Extract predicted actions
            predicted_actions = []
            for j in range(predictions.shape[1]):
                pred_id = predictions[0, j].item()
                # Look for action tokens
                for action_name, action_id in ACTIONS.items():
                    if action_id == pred_id and action_name != "<PAD>":
                        predicted_actions.append(action_name)
                        break

            # Trim to expected length
            predicted_actions = predicted_actions[: len(expected_actions)]

            # Check correctness
            is_correct = predicted_actions == expected_actions

            if stage == "retrieval":
                if is_correct:
                    correct_stage2 += 1
            else:  # full_binding
                if is_correct:
                    correct_stage3 += 1

                # Track sequential performance
                if is_sequential:
                    total_sequential += 1
                    if is_correct:
                        correct_sequential += 1

        total += 1

    results = {
        "stage1_accuracy": correct_stage1 / total if total > 0 else 0,
        "stage2_accuracy": correct_stage2 / total if total > 0 else 0,
        "stage3_accuracy": correct_stage3 / total if total > 0 else 0,
        "total_evaluated": total,
    }

    if total_sequential > 0:
        results["sequential_accuracy"] = correct_sequential / total_sequential
        results["sequential_total"] = total_sequential

    return results


def main():
    # Configuration
    num_epochs = 20
    batch_size = 32
    learning_rate = 0.001
    num_samples = 2000
    eval_interval = 5

    # Create dataset with sequential patterns
    print("Creating dataset with sequential patterns...")
    dataset = create_sequential_dataset(num_samples=num_samples)

    # Initialize model
    model = SequentialDynamicMemoryModel(
        vocab_size=len(VOCAB),
        num_actions=len(ACTIONS),
        embed_dim=256,
        num_slots=4,
        num_heads=8,
        mlp_hidden_dim=512,
    )

    # Initialize optimizer
    optimizer = optim.Adam(learning_rate=learning_rate)

    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    print("Training with mixed curriculum (all stages simultaneously)")

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

            # Randomly select stage for this batch (for mixed training)
            stage = np.random.choice(["recognition", "retrieval", "full_binding"])

            # Train step
            step_results = train_step(model, batch, optimizer, stage)
            epoch_losses.append(step_results["loss"])

        # Print epoch summary
        avg_loss = np.mean(epoch_losses)
        print(f"\nEpoch {epoch+1} - Average Loss: {avg_loss:.4f}")

        # Evaluate periodically
        if (epoch + 1) % eval_interval == 0:
            print("\nEvaluating on all stages...")

            # Evaluate each stage
            stage1_results = evaluate_model(model, dataset, stage="recognition")
            stage2_results = evaluate_model(model, dataset, stage="retrieval")
            stage3_results = evaluate_model(model, dataset, stage="full_binding")

            print(
                f"Stage 1 (Recognition): {stage1_results['stage1_accuracy']*100:.1f}%"
            )
            print(f"Stage 2 (Retrieval): {stage2_results['stage2_accuracy']*100:.1f}%")
            print(
                f"Stage 3 (Full Binding): {stage3_results['stage3_accuracy']*100:.1f}%"
            )

            if "sequential_accuracy" in stage3_results:
                print(
                    f"Sequential Patterns: {stage3_results['sequential_accuracy']*100:.1f}% "
                    f"({stage3_results['sequential_total']} patterns)"
                )

    # Final evaluation
    print("\n" + "=" * 50)
    print("FINAL EVALUATION")
    print("=" * 50)

    final_results = evaluate_model(model, dataset, stage="full_binding")
    print(f"Final Accuracy: {final_results['stage3_accuracy']*100:.1f}%")
    if "sequential_accuracy" in final_results:
        print(
            f"Sequential Pattern Accuracy: {final_results['sequential_accuracy']*100:.1f}%"
        )

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = get_output_path("models")
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, f"sequential_model_{timestamp}.npz")
    mx.save(model_path, dict(model.parameters()))
    print(f"\nModel saved to: {model_path}")

    # Test specific sequential patterns
    print("\n" + "=" * 50)
    print("TESTING SPECIFIC SEQUENTIAL PATTERNS")
    print("=" * 50)

    test_patterns = [
        "X means jump do X then Y means walk do Y",
        "Z means turn do Z twice then X means run do X",
        "Y means walk do Y then Z means turn do Z thrice",
    ]

    for pattern in test_patterns:
        command_tokens = pattern.split()
        command_ids = mx.array(
            [[VOCAB.get(token, VOCAB["<PAD>"]) for token in command_tokens]]
        )

        # MLX doesn't need no_grad
        results = model(command_ids, stage="full_binding")
        segments = results.get("segments", [])

        # Get predictions
        action_logits = results["action_logits"]
        predictions = mx.argmax(action_logits, axis=-1)

        # Extract predicted actions
        predicted_actions = []
        for j in range(predictions.shape[1]):
            pred_id = predictions[0, j].item()
            for action_name, action_id in ACTIONS.items():
                if action_id == pred_id and action_name != "<PAD>":
                    predicted_actions.append(action_name)
                    break

        print(f"\nPattern: {pattern}")
        print(f"Segments: {segments}")
        print(f"Predicted: {predicted_actions}")


if __name__ == "__main__":
    main()
