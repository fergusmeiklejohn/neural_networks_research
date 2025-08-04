#!/usr/bin/env python3
"""Train baseline models on variable binding tasks."""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath("../.."))

# Import baseline models and data generation
from baseline_models import RuleBasedBaseline, create_baseline_model
from mlx_model_io import save_model_simple
from train_binding_curriculum import (
    generate_stage1_data,
    generate_stage2_data,
    generate_stage3_data,
)


def train_baseline(
    model: nn.Module,
    train_data: List[Tuple],
    val_data: List[Tuple],
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    model_name: str = "baseline",
) -> Dict:
    """Train a baseline model on binding tasks.

    Returns:
        Dictionary of training metrics
    """
    print(f"\nTraining {model_name} baseline...")

    # Create optimizer
    optimizer = optim.Adam(learning_rate=learning_rate)

    # Training history
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    def loss_fn(model, inputs, targets, mask):
        """Compute cross-entropy loss."""
        logits = model(inputs)

        # Only compute loss at action positions
        batch_size, seq_len, num_classes = logits.shape
        logits_flat = logits.reshape(-1, num_classes)
        targets_flat = targets.reshape(-1)
        mask_flat = mask.reshape(-1)

        # Compute losses
        losses = nn.losses.cross_entropy(logits_flat, targets_flat, reduction="none")

        # Apply mask
        masked_losses = losses * mask_flat
        loss = mx.sum(masked_losses) / mx.maximum(mx.sum(mask_flat), 1.0)

        # Compute accuracy
        predictions = mx.argmax(logits_flat, axis=1)
        correct = (predictions == targets_flat) * mask_flat
        accuracy = mx.sum(correct) / mx.maximum(mx.sum(mask_flat), 1.0)

        return loss, accuracy

    @mx.compile
    def train_step(model, inputs, targets, mask):
        """Single training step."""

        def compute_loss(model):
            loss, acc = loss_fn(model, inputs, targets, mask)
            return loss, (loss, acc)

        grad_fn = nn.value_and_grad(model, compute_loss, has_aux=True)
        (loss, (loss_val, acc)), grads = grad_fn(model)
        optimizer.update(model, grads)

        return loss_val, acc

    # Training loop
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_losses = []
        train_accs = []

        for inputs, targets, mask in train_data:
            loss, acc = train_step(model, inputs, targets, mask)
            train_losses.append(float(loss))
            train_accs.append(float(acc))

        # Evaluate
        model.eval()
        val_losses = []
        val_accs = []

        for inputs, targets, mask in val_data:
            loss, acc = loss_fn(model, inputs, targets, mask)
            val_losses.append(float(loss))
            val_accs.append(float(acc))

        # Record metrics
        train_loss = np.mean(train_losses)
        train_acc = np.mean(train_accs)
        val_loss = np.mean(val_losses)
        val_acc = np.mean(val_accs)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
            )

    print(f"Final validation accuracy: {val_acc:.4f}")

    return history


def evaluate_rule_based(model: RuleBasedBaseline, data: List[Tuple]) -> float:
    """Evaluate rule-based baseline."""
    correct = 0
    total = 0

    for inputs, targets, mask in data:
        predictions = model.predict(inputs)
        pred_labels = mx.argmax(predictions, axis=-1)

        # Compare at masked positions
        for b in range(inputs.shape[0]):
            for i in range(inputs.shape[1]):
                if mask[b, i] > 0:
                    if pred_labels[b, i] == targets[b, i]:
                        correct += 1
                    total += 1

    accuracy = correct / max(total, 1)
    return accuracy


def main():
    """Train all baseline models."""
    # Create vocabulary
    vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
        "X": 2,
        "Y": 3,
        "Z": 4,
        "W": 5,
        "means": 6,
        "do": 7,
        "twice": 8,
        "thrice": 9,
        "then": 10,
        "jump": 11,
        "walk": 12,
        "turn": 13,
        "look": 14,
        "run": 15,
        "stop": 16,
        "JUMP": 17,
        "WALK": 18,
        "TURN": 19,
        "LOOK": 20,
        "RUN": 21,
        "STOP": 22,
    }
    vocab_size = len(vocab)

    # Generate mixed training data (all stages)
    print("Generating training data...")
    stage1_train = generate_stage1_data(32, max_examples=200)
    stage2_train = generate_stage2_data(32, max_examples=200)
    stage3_train = generate_stage3_data(32, max_examples=200)

    stage1_val = generate_stage1_data(32, max_examples=50)
    stage2_val = generate_stage2_data(32, max_examples=50)
    stage3_val = generate_stage3_data(32, max_examples=50)

    # Convert to list of tuples for training
    def dict_to_tuples(data_dict):
        """Convert data dictionary to list of (input, target, mask) tuples."""
        inputs = data_dict["command"]

        # Stage 1 has 'target', stages 2-3 have 'labels'
        labels = data_dict.get("labels", data_dict.get("target"))

        # Create mask based on where we have labels
        if "labels" in data_dict:
            # For stage 2/3, mask where labels are not padding (0)
            mask = (labels != 0).astype(mx.float32)
        else:
            # For stage 1, mask is just at the target position
            batch_size = inputs.shape[0]
            seq_len = inputs.shape[1]
            mask = mx.zeros((batch_size, seq_len))
            # Stage 1 only has single targets, need to handle differently

        # Split into individual examples
        tuples = []
        for i in range(inputs.shape[0]):
            tuples.append(
                (
                    inputs[i : i + 1],  # Keep batch dimension
                    labels[i : i + 1]
                    if len(labels.shape) > 1
                    else mx.array([[labels[i]]]),
                    mask[i : i + 1] if len(mask.shape) > 1 else mx.ones((1, 1)),
                )
            )
        return tuples

    # Combine all stages
    train_data = (
        dict_to_tuples(stage1_train)
        + dict_to_tuples(stage2_train)
        + dict_to_tuples(stage3_train)
    )
    val_data = (
        dict_to_tuples(stage1_val)
        + dict_to_tuples(stage2_val)
        + dict_to_tuples(stage3_val)
    )

    print(f"Total training samples: {len(train_data)}")
    print(f"Total validation samples: {len(val_data)}")

    # Results storage
    results = {}

    # Train LSTM baseline
    lstm_model = create_baseline_model("lstm", vocab_size)
    lstm_history = train_baseline(
        lstm_model,
        train_data,
        val_data,
        num_epochs=5,
        model_name="LSTM",  # Quick test with 5 epochs
    )
    results["lstm"] = {
        "final_val_acc": lstm_history["val_acc"][-1],
        "history": lstm_history,
    }

    # Save LSTM model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_model_simple(f"models/lstm_baseline_{timestamp}.pkl", lstm_model)

    # Train Transformer baseline
    transformer_model = create_baseline_model("transformer", vocab_size)
    transformer_history = train_baseline(
        transformer_model,
        train_data,
        val_data,
        num_epochs=5,
        model_name="Transformer",  # Quick test
    )
    results["transformer"] = {
        "final_val_acc": transformer_history["val_acc"][-1],
        "history": transformer_history,
    }

    # Save Transformer model
    save_model_simple(f"models/transformer_baseline_{timestamp}.pkl", transformer_model)

    # Train Feedforward baseline
    ff_model = create_baseline_model("feedforward", vocab_size)
    ff_history = train_baseline(
        ff_model,
        train_data,
        val_data,
        num_epochs=5,
        model_name="Feedforward",  # Quick test
    )
    results["feedforward"] = {
        "final_val_acc": ff_history["val_acc"][-1],
        "history": ff_history,
    }

    # Save Feedforward model
    save_model_simple(f"models/feedforward_baseline_{timestamp}.pkl", ff_model)

    # Evaluate rule-based baseline
    print("\nEvaluating rule-based baseline...")
    rule_model = RuleBasedBaseline(vocab)
    rule_acc = evaluate_rule_based(rule_model, val_data)
    print(f"Rule-based accuracy: {rule_acc:.4f}")
    results["rule_based"] = {"final_val_acc": rule_acc}

    # Save results
    os.makedirs("results", exist_ok=True)
    with open(f"results/baseline_results_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 50)
    print("BASELINE COMPARISON SUMMARY")
    print("=" * 50)
    for model_name, result in results.items():
        print(f"{model_name}: {result['final_val_acc']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
