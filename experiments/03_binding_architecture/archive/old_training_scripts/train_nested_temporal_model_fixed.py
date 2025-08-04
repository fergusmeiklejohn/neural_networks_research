#!/usr/bin/env python3
"""Training script for the integrated model with nested temporal patterns.

This version properly handles variable-length sequences from different data sources.
"""

from utils.imports import setup_project_paths

setup_project_paths()

import os
from typing import Any, Dict, List

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from compositional_operators import (
    CompositionalExecutor,
    CompositionalParser,
)
from mlx_model_io import save_model_simple
from nested_temporal_patterns import (
    NestedTemporalExecutor,
    NestedTemporalParser,
)
from train_integrated_model import ACTIONS, VOCAB, IntegratedBindingModel

from utils.config import setup_environment
from utils.paths import get_output_path

config = setup_environment()


class NestedTemporalBindingModel(IntegratedBindingModel):
    """Extended binding model with nested temporal pattern support."""

    def __init__(
        self,
        vocab_size: int,
        num_actions: int,
        embed_dim: int = 256,
        num_slots: int = 4,
        num_heads: int = 8,
        mlp_hidden_dim: int = 512,
    ):
        super().__init__(
            vocab_size, num_actions, embed_dim, num_slots, num_heads, mlp_hidden_dim
        )

        # Add parsers and executors
        self.compositional_parser = CompositionalParser(VOCAB)
        self.compositional_executor = CompositionalExecutor(self, VOCAB)
        self.temporal_parser = NestedTemporalParser(VOCAB)
        self.temporal_executor = NestedTemporalExecutor(self, VOCAB, ACTIONS)

    def __call__(self, inputs: Dict[str, mx.array], stage: str = "full") -> mx.array:
        """Forward pass with nested temporal pattern support."""
        command_ids = inputs["command"]

        # Clear versioned memory for new sequence
        self.versioned_memory.clear()

        # Ensure command_ids is in the right format
        if len(command_ids.shape) == 1:
            command_ids = command_ids.reshape(1, -1)

        # Check if this is a nested temporal pattern
        if hasattr(command_ids, "numpy"):
            token_array = command_ids.numpy()
        else:
            token_array = command_ids

        if len(token_array.shape) > 1:
            token_array = token_array[0]

        token_list = token_array.tolist()
        token_list = [int(t) for t in token_list]

        # Check for nested temporal patterns
        temporal_nodes = self.temporal_parser.parse(token_list)

        if temporal_nodes and self._is_pure_temporal_pattern(token_list):
            # Handle as nested temporal pattern
            bindings = self._extract_bindings(token_list)
            outputs = self.temporal_executor.execute_nested_pattern(
                token_list, bindings
            )
        else:
            # Fall back to compositional parsing
            # Ensure command_ids has correct shape for compositional parser
            if len(command_ids.shape) == 1:
                command_ids = command_ids.reshape(1, -1)
            parse_tree = self.compositional_parser.parse(command_ids)
            bindings = {}
            outputs = self.compositional_executor.execute(
                parse_tree, command_ids, bindings, stage
            )

        # Stack outputs
        if outputs:
            squeezed_outputs = []
            for out in outputs:
                if len(out.shape) > 1:
                    squeezed_outputs.append(out.squeeze())
                else:
                    squeezed_outputs.append(out)
            return mx.stack(squeezed_outputs)
        else:
            return mx.zeros((1, self.num_actions))

    def _is_pure_temporal_pattern(self, tokens: List[int]) -> bool:
        """Check if this is a pure nested temporal pattern (no compositional operators)."""
        # Get token names
        id_to_word = {v: k for k, v in VOCAB.items()}
        words = [id_to_word.get(t, "<PAD>") for t in tokens]

        # Check for compositional operators
        compositional_ops = {"and", "or", "while"}
        return not any(word in compositional_ops for word in words)

    def _extract_bindings(self, tokens: List[int]) -> Dict[str, str]:
        """Extract variable bindings from the command."""
        id_to_word = {v: k for k, v in VOCAB.items()}
        words = [id_to_word.get(t, "<PAD>") for t in tokens]

        bindings = {}
        i = 0
        while i < len(words) - 2:
            if words[i] in ["X", "Y", "Z"] and words[i + 1] == "means":
                var_name = words[i]
                action = words[i + 2].upper()
                if action in ACTIONS:
                    bindings[var_name] = action
                i += 3
            else:
                i += 1

        return bindings


def normalize_data_format(data_item: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize data format to ensure consistent labels shape and key names."""
    normalized = data_item.copy()

    # Convert 'target' to 'labels' if needed
    if "target" in normalized and "labels" not in normalized:
        normalized["labels"] = normalized["target"]
        del normalized["target"]

    # Ensure labels exists
    if "labels" not in normalized:
        raise ValueError(f"Data item missing 'labels' key: {list(normalized.keys())}")

    # Get labels
    labels = normalized["labels"]

    # Convert to MLX array if needed
    if isinstance(labels, (list, np.ndarray)):
        labels = mx.array(labels)

    # Flatten labels to 1D
    # Handle various shapes: (1, N), (N,), or scalar
    if len(labels.shape) == 0:  # Scalar
        labels = mx.array([labels])
    elif len(labels.shape) == 2:  # (1, N) shape
        labels = labels.squeeze(axis=0)
    elif len(labels.shape) > 2:  # Unexpected shape
        labels = labels.flatten()

    # Ensure labels is 1D
    if len(labels.shape) != 1:
        raise ValueError(f"Unable to normalize labels shape: {labels.shape}")

    normalized["labels"] = labels
    normalized["num_actions"] = len(labels)

    return normalized


def generate_nested_temporal_data(num_samples: int = 100) -> List[Dict[str, Any]]:
    """Generate training data with nested temporal patterns."""
    # Ensure temporal modifiers are in vocabulary
    for word in ["twice", "thrice", "do", "then"]:
        if word not in VOCAB:
            VOCAB[word] = len(VOCAB)

    data = []

    # Pattern 1: Simple nested (do X twice twice)
    for _ in range(num_samples // 4):
        var = np.random.choice(["X", "Y", "Z"])
        action = np.random.choice(["WALK", "JUMP", "TURN"])

        # Choose nesting level
        if np.random.random() < 0.5:
            # Two levels: "do X twice twice" = 4 repetitions
            command = f"{var} means {action.lower()} do {var} twice twice"
            num_reps = 4
        else:
            # Two levels with different modifiers: "do X thrice twice" = 6 repetitions
            command = f"{var} means {action.lower()} do {var} thrice twice"
            num_reps = 6

        tokens = [VOCAB.get(word, VOCAB["<PAD>"]) for word in command.split()]
        labels = [ACTIONS[action]] * num_reps

        data.append(
            {"command": mx.array([tokens]), "labels": mx.array(labels), "stage": "full"}
        )

    # Pattern 2: Three-level nesting
    for _ in range(num_samples // 4):
        var = np.random.choice(["X", "Y"])
        action = np.random.choice(["WALK", "JUMP", "TURN"])

        # "do X twice twice twice" = 8 repetitions
        command = f"{var} means {action.lower()} do {var} twice twice twice"
        num_reps = 8

        tokens = [VOCAB.get(word, VOCAB["<PAD>"]) for word in command.split()]
        labels = [ACTIONS[action]] * num_reps

        data.append(
            {"command": mx.array([tokens]), "labels": mx.array(labels), "stage": "full"}
        )

    # Pattern 3: Sequential with nested temporal
    for _ in range(num_samples // 4):
        var1 = "X"
        var2 = "Y"
        action1 = np.random.choice(["WALK", "JUMP"])
        action2 = np.random.choice(["TURN", "RUN"])

        # "do X twice twice then do Y thrice"
        command = f"{var1} means {action1.lower()} {var2} means {action2.lower()} do {var1} twice twice then do {var2} thrice"

        tokens = [VOCAB.get(word, VOCAB["<PAD>"]) for word in command.split()]
        labels = [ACTIONS[action1]] * 4 + [ACTIONS[action2]] * 3

        data.append(
            {"command": mx.array([tokens]), "labels": mx.array(labels), "stage": "full"}
        )

    # Pattern 4: Mixed modifiers
    for _ in range(num_samples // 4):
        var = np.random.choice(["X", "Y", "Z"])
        action = np.random.choice(["WALK", "JUMP", "TURN"])

        if np.random.random() < 0.5:
            # "do X twice thrice" = 6 repetitions (2 * 3)
            command = f"{var} means {action.lower()} do {var} twice thrice"
            num_reps = 6
        else:
            # "do X thrice thrice" = 9 repetitions
            command = f"{var} means {action.lower()} do {var} thrice thrice"
            num_reps = 9

        tokens = [VOCAB.get(word, VOCAB["<PAD>"]) for word in command.split()]
        labels = [ACTIONS[action]] * num_reps

        data.append(
            {"command": mx.array([tokens]), "labels": mx.array(labels), "stage": "full"}
        )

    return data


def batch_to_list(batch_data: Dict[str, mx.array], stage: str = None) -> List[Dict]:
    """Convert batched data to list of individual examples."""
    data_list = []
    batch_size = batch_data["command"].shape[0]

    for i in range(batch_size):
        # Ensure command tokens are integers
        command = batch_data["command"][i : i + 1]
        if command.dtype != mx.int32 and command.dtype != mx.int64:
            command = command.astype(mx.int32)

        item = {"command": command, "stage": stage}

        if "target" in batch_data:
            target = batch_data["target"][i : i + 1]
            if target.dtype != mx.int32 and target.dtype != mx.int64:
                target = target.astype(mx.int32)
            item["target"] = target

        if "labels" in batch_data:
            labels = batch_data["labels"][i : i + 1]
            if labels.dtype != mx.int32 and labels.dtype != mx.int64:
                labels = labels.astype(mx.int32)
            item["labels"] = labels
            item["mask"] = mx.ones_like(item["labels"])

        data_list.append(item)

    return data_list


def compute_sequence_loss(outputs: mx.array, labels: mx.array) -> mx.array:
    """Compute loss for variable-length sequences.

    Args:
        outputs: Model outputs of shape (seq_len, num_actions)
        labels: Target labels of shape (seq_len,)

    Returns:
        Average cross-entropy loss
    """
    # Ensure both are 2D for cross_entropy
    if len(outputs.shape) == 1:
        outputs = outputs.reshape(1, -1)

    if len(labels.shape) == 0:  # Scalar
        labels = mx.array([labels])

    # Handle length mismatch
    min_len = min(outputs.shape[0], labels.shape[0])
    if min_len == 0:
        return mx.array(0.0)

    outputs = outputs[:min_len]
    labels = labels[:min_len]

    # Compute cross-entropy for each position
    losses = []
    for i in range(min_len):
        loss = nn.losses.cross_entropy(outputs[i : i + 1], labels[i : i + 1])
        losses.append(loss)

    return mx.mean(mx.stack(losses))


def train_nested_temporal_model(num_epochs: int = 10, batch_size: int = 1):
    """Train the model with nested temporal patterns."""
    print("=== Training Nested Temporal Model (Fixed) ===")

    # Generate training data FIRST (this modifies VOCAB)
    print("\nGenerating training data...")

    # Include all previous patterns plus nested temporal
    from train_compositional_model import generate_compositional_data
    from train_integrated_model import (
        generate_rebinding_data,
        generate_stage1_data,
        generate_stage2_data,
        generate_stage3_data,
    )

    # Convert to lists
    stage1_data = batch_to_list(generate_stage1_data(100), "recognition")
    stage2_data = batch_to_list(generate_stage2_data(100), "retrieval")
    stage3_data = batch_to_list(generate_stage3_data(100), "full")
    rebinding_data = generate_rebinding_data(100)
    compositional_data = generate_compositional_data(100)
    nested_temporal_data = generate_nested_temporal_data(
        200
    )  # More samples for complex patterns

    # Normalize all data
    all_data = []
    data_sources = [
        ("Basic Stage 1", stage1_data),
        ("Basic Stage 2", stage2_data),
        ("Basic Stage 3", stage3_data),
        ("Rebinding", rebinding_data),
        ("Compositional", compositional_data),
        ("Nested Temporal", nested_temporal_data),
    ]

    for source_name, dataset in data_sources:
        normalized_count = 0
        for item in dataset:
            try:
                normalized = normalize_data_format(item)
                all_data.append(normalized)
                normalized_count += 1
            except Exception as e:
                print(f"Warning: Failed to normalize item from {source_name}: {e}")
        print(f"  - {source_name}: {normalized_count} samples")

    print(f"\nTotal training samples: {len(all_data)}")

    # Create model AFTER data generation (which modifies VOCAB)
    vocab_size = len(VOCAB)
    num_actions = len(ACTIONS)
    print(f"\nCreating model with vocab_size={vocab_size}, num_actions={num_actions}")
    print(f"VOCAB size after data generation: {len(VOCAB)}")
    model = NestedTemporalBindingModel(vocab_size, num_actions)

    # Shuffle data
    indices = np.random.permutation(len(all_data))
    train_data = [all_data[i] for i in indices]

    # Setup optimizer
    optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=0.01)

    # Training loop
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")

        total_loss = 0.0
        correct = 0
        total = 0

        # Process each batch
        for i, batch in enumerate(train_data):
            # Prepare inputs
            inputs = {"command": batch["command"], "stage": batch.get("stage", "full")}

            # Forward pass and loss
            def loss_fn(model):
                outputs = model(inputs, stage=batch.get("stage", "full"))
                labels = batch["labels"]

                # Compute sequence loss
                loss = compute_sequence_loss(outputs, labels)

                # Compute accuracy
                predictions = mx.argmax(outputs, axis=1)
                min_len = min(len(predictions), len(labels))
                if min_len > 0:
                    accuracy = mx.mean(predictions[:min_len] == labels[:min_len])
                else:
                    accuracy = mx.array(0.0)

                return loss, accuracy

            loss_val, accuracy = loss_fn(model)

            # Gradient computation
            def loss_only(model):
                outputs = model(inputs, stage=batch.get("stage", "full"))
                labels = batch["labels"]
                return compute_sequence_loss(outputs, labels)

            grad_fn = mx.grad(loss_only)
            grads = grad_fn(model)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            # Update stats
            total_loss += float(loss_val)
            num_actions = batch.get("num_actions", 1)
            correct += float(accuracy) * num_actions
            total += num_actions

            if i % 100 == 0:
                print(
                    f"Batch {i}/{len(train_data)}, Loss: {float(loss_val):.4f}, Acc: {float(accuracy):.2%}"
                )

        # Epoch summary
        epoch_acc = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(train_data) if len(train_data) > 0 else 0.0
        print(
            f"\nEpoch {epoch+1}: Avg Loss = {avg_loss:.4f}, Accuracy = {epoch_acc:.2%}"
        )

        if epoch_acc > best_accuracy:
            best_accuracy = epoch_acc
            save_path = os.path.join(
                get_output_path(), "nested_temporal_model_best.pkl"
            )
            save_model_simple(save_path, model)
            print(f"Saved best model with accuracy {best_accuracy:.2%}")

    print(f"\nTraining complete! Best accuracy: {best_accuracy:.2%}")
    return model


def test_nested_temporal_model(model: NestedTemporalBindingModel):
    """Test the nested temporal model."""
    print("\n=== Testing Nested Temporal Model ===")

    test_cases = [
        # Basic nested temporal
        ("X means jump do X twice", ["JUMP", "JUMP"]),
        ("X means jump do X twice twice", ["JUMP", "JUMP", "JUMP", "JUMP"]),
        ("X means walk do X thrice twice", ["WALK"] * 6),
        ("X means turn do X twice thrice", ["TURN"] * 6),
        # Three-level nesting
        ("X means jump do X twice twice twice", ["JUMP"] * 8),
        # Sequential with nested
        (
            "X means walk Y means jump do X twice twice then do Y thrice",
            ["WALK"] * 4 + ["JUMP"] * 3,
        ),
        # Variable rebinding with nested temporal
        (
            "X means jump do X twice then X means walk do X twice twice",
            ["JUMP", "JUMP"] + ["WALK"] * 4,
        ),
        # Complex mixed pattern
        (
            "X means turn Y means jump do X thrice then do Y twice twice",
            ["TURN"] * 3 + ["JUMP"] * 4,
        ),
    ]

    correct_count = 0
    for command, expected in test_cases:
        print(f"\nCommand: {command}")
        print(f"Expected: {expected}")

        tokens = [VOCAB.get(word, VOCAB["<PAD>"]) for word in command.split()]
        inputs = {"command": mx.array([tokens])}

        # Get predictions
        outputs = model(inputs, stage="full")
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
        if correct:
            correct_count += 1

        # Additional analysis for nested patterns
        token_list = [int(t) for t in tokens]
        temporal_nodes = model.temporal_parser.parse(token_list)
        if temporal_nodes:
            print("Temporal patterns found:")
            for node in temporal_nodes:
                from nested_temporal_patterns import describe_temporal_node

                print(f"  {describe_temporal_node(node)}")
                print(f"  Total repetitions: {node.compute_repetitions()}")

    print(f"\n=== Test Summary ===")
    print(
        f"Correct: {correct_count}/{len(test_cases)} = {correct_count/len(test_cases)*100:.1f}%"
    )


if __name__ == "__main__":
    model = train_nested_temporal_model(num_epochs=5)
    test_nested_temporal_model(model)
