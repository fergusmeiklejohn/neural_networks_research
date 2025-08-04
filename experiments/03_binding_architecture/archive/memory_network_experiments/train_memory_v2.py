#!/usr/bin/env python3
"""Train the improved memory network with compositional support."""

from utils.imports import setup_project_paths

setup_project_paths()

from utils.config import setup_environment

config = setup_environment()

from typing import Dict, List

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from memory_network_v2 import ImprovedMemoryNetwork
from progressive_complexity_dataset import ACTIONS, VOCAB, ProgressiveComplexityDataset


def train_step(model, optimizer, sample: Dict) -> float:
    """Train on a single sample."""
    tokens = mx.array([sample["tokens"]])
    expected_indices = sample["expected_indices"]

    def loss_fn(model):
        predictions = model({"command": tokens})

        if predictions.shape[0] == 0 or len(expected_indices) == 0:
            return mx.array(0.0)

        # Match lengths
        n_expected = len(expected_indices)
        n_predicted = predictions.shape[0]

        if n_predicted != n_expected:
            # For now, just use min length
            min_len = min(n_predicted, n_expected)
            predictions = predictions[:min_len]
            expected_trimmed = expected_indices[:min_len]
        else:
            expected_trimmed = expected_indices

        targets = mx.array(expected_trimmed)
        loss = nn.losses.cross_entropy(predictions, targets)
        return mx.mean(loss)

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad_fn(model)

    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)

    return loss.item()


def evaluate(model, dataset: List[Dict], verbose: bool = False) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()

    correct_by_level = {1: 0, 2: 0, 3: 0, 4: 0}
    total_by_level = {1: 0, 2: 0, 3: 0, 4: 0}
    partial_correct_by_level = {1: 0, 2: 0, 3: 0, 4: 0}

    for sample in dataset:
        tokens = mx.array([sample["tokens"]])
        expected = sample["expected_indices"]
        level = sample["complexity_level"]

        predictions = model({"command": tokens})

        if predictions.shape[0] > 0:
            pred_indices = mx.argmax(predictions, axis=1).tolist()

            # Check exact match
            if pred_indices == expected:
                correct_by_level[level] += 1
                partial_correct_by_level[level] += 1
            else:
                # Check partial match (at least getting length right)
                if len(pred_indices) == len(expected):
                    # Count how many are correct
                    correct_actions = sum(
                        p == e for p, e in zip(pred_indices, expected)
                    )
                    if correct_actions > 0:
                        partial_correct_by_level[level] += correct_actions / len(
                            expected
                        )

            if verbose and pred_indices != expected:
                print(f"\nMismatch:")
                print(f"  Command: {sample['command']}")
                print(f"  Expected: {[ACTIONS[i] for i in expected]}")
                print(f"  Predicted: {[ACTIONS[i] for i in pred_indices]}")

        total_by_level[level] += 1

    # Calculate metrics
    results = {}
    for level in range(1, 5):
        if total_by_level[level] > 0:
            results[f"level_{level}_exact"] = (
                correct_by_level[level] / total_by_level[level]
            )
            results[f"level_{level}_partial"] = (
                partial_correct_by_level[level] / total_by_level[level]
            )

    total_correct = sum(correct_by_level.values())
    total_samples = sum(total_by_level.values())
    results["overall_exact"] = total_correct / total_samples if total_samples > 0 else 0

    return results


def analyze_memory_usage(model, sample: Dict):
    """Analyze what's happening in memory."""
    tokens = mx.array([sample["tokens"]])

    # Reset and run forward pass
    model.reset_memory()
    bindings, execution_plan = model.parse_execution_plan(tokens)

    print(f"\nCommand: {sample['command']}")
    print(f"Bindings: {bindings}")
    print(f"Execution plan: {execution_plan}")

    # Check memory values after storage
    print("\nMemory values after storage:")
    for i, var in enumerate(["X", "Y", "Z", "W"]):
        var_token = VOCAB[var]
        if var_token in bindings:
            slot = bindings[var_token]
            value = model.memory_values[slot]
            print(
                f"  {var} (slot {slot}): mean={mx.mean(value).item():.4f}, std={mx.std(value).item():.4f}"
            )


def main():
    """Train improved memory network."""
    print("Training Improved Memory Network...")

    # Create model
    model = ImprovedMemoryNetwork(
        vocab_size=len(VOCAB),
        num_actions=len(ACTIONS),
        embed_dim=64,
        hidden_dim=128,
        num_vars=4,
    )

    # Create dataset
    dataset_gen = ProgressiveComplexityDataset()

    # Generate training data
    print("\nGenerating training data...")
    train_data = []
    train_data.extend(dataset_gen.generate_level_1(100))
    train_data.extend(dataset_gen.generate_level_2(200))  # More level 2 data

    # Test data
    test_data = []
    for level in range(1, 5):
        test_data.extend(getattr(dataset_gen, f"generate_level_{level}")(25))

    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")

    # Analyze a few samples before training
    print("\n=== Pre-training Analysis ===")
    for i in range(3):
        analyze_memory_usage(model, train_data[i])

    # Training
    optimizer = optim.Adam(learning_rate=1e-3)

    print("\n=== Training ===")
    num_epochs = 30

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        np.random.shuffle(train_data)

        for sample in train_data:
            loss = train_step(model, optimizer, sample)
            epoch_loss += loss

        avg_loss = epoch_loss / len(train_data)

        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            results = evaluate(model, test_data[:50])  # Quick eval
            print(f"\nEpoch {epoch+1}/{num_epochs}:")
            print(f"  Loss: {avg_loss:.4f}")
            print(
                f"  Level 1: {results.get('level_1_exact', 0):.1%} exact, {results.get('level_1_partial', 0):.1%} partial"
            )
            print(
                f"  Level 2: {results.get('level_2_exact', 0):.1%} exact, {results.get('level_2_partial', 0):.1%} partial"
            )
            print(f"  Overall: {results['overall_exact']:.1%}")

    # Final evaluation
    print("\n=== Final Evaluation ===")
    final_results = evaluate(model, test_data, verbose=True)

    print("\nFinal Results:")
    for level in range(1, 5):
        exact = final_results.get(f"level_{level}_exact", 0)
        partial = final_results.get(f"level_{level}_partial", 0)
        print(f"  Level {level}: {exact:.1%} exact match, {partial:.1%} partial match")
    print(f"\nOverall Exact Match: {final_results['overall_exact']:.1%}")

    # Post-training analysis
    print("\n=== Post-training Analysis ===")
    test_samples = [
        {
            "command": "X means jump Y means walk do X and Y",
            "tokens": tokenize_command("X means jump Y means walk do X and Y"),
            "expected_indices": [0, 1],
        },
        {
            "command": "X means run Y means turn do Y then X",
            "tokens": tokenize_command("X means run Y means turn do Y then X"),
            "expected_indices": [3, 2],
        },
    ]

    for sample in test_samples:
        analyze_memory_usage(model, sample)

        # Also show predictions
        predictions = model({"command": mx.array([sample["tokens"]])})
        if predictions.shape[0] > 0:
            pred_indices = mx.argmax(predictions, axis=1).tolist()
            print(f"  Predicted: {[ACTIONS[i] for i in pred_indices]}")

    return model, final_results


def tokenize_command(command: str) -> List[int]:
    """Tokenize a command string."""
    return [VOCAB.get(word, 0) for word in command.split()]


if __name__ == "__main__":
    model, results = main()
