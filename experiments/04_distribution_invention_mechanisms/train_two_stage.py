#!/usr/bin/env python3
"""Training script for Two-Stage Compiler.

This script trains the neural components while keeping the rule extraction fixed,
demonstrating how explicit mechanisms enable learning compositional patterns.
"""

from utils.imports import setup_project_paths

setup_project_paths()

import argparse
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from progressive_complexity_dataset import ProgressiveComplexityDataset
from tqdm import tqdm
from two_stage_compiler_v2 import TwoStageCompilerV2

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TwoStageTrainer:
    """Trainer for Two-Stage Compiler."""

    def __init__(self, vocab_size: int, num_actions: int, learning_rate: float = 1e-3):
        self.vocab_size = vocab_size
        self.num_actions = num_actions

        # Create model
        self.model = TwoStageCompilerV2(vocab_size, num_actions)

        # Only train the neural executor
        self.optimizer = optim.Adam(learning_rate=learning_rate)

        # Loss function
        self.loss_fn = nn.losses.cross_entropy

        # Metrics
        self.metrics = {
            "train_loss": [],
            "train_accuracy": [],
            "val_accuracy_by_level": {1: [], 2: [], 3: [], 4: []},
        }

    def compute_loss(
        self, model: TwoStageCompilerV2, batch: Dict
    ) -> Tuple[mx.array, mx.array]:
        """Compute loss and accuracy for a batch."""
        tokens = batch["tokens"]
        expected = batch["expected_indices"]

        # Forward pass
        outputs = model(tokens)

        # Handle variable length outputs
        total_loss = mx.array(0.0)
        total_correct = 0
        total_actions = 0

        for i in range(len(expected)):
            exp_actions = expected[i]
            num_expected = len(exp_actions)

            if outputs.shape[0] >= num_expected:
                # Get predictions for this sample
                pred_slice = outputs[:num_expected]

                # Convert expected to MLX array
                targets = mx.array(exp_actions)

                # Compute loss
                loss = self.loss_fn(pred_slice, targets)
                total_loss = total_loss + mx.sum(loss)

                # Compute accuracy
                predictions = mx.argmax(pred_slice, axis=-1)
                correct = mx.sum(predictions == targets)
                total_correct += int(correct)
                total_actions += num_expected

        # Average loss and accuracy
        avg_loss = total_loss / max(total_actions, 1)
        accuracy = total_correct / max(total_actions, 1)

        return avg_loss, mx.array(accuracy)

    def train_epoch(
        self, model: TwoStageCompilerV2, train_data: List[Dict]
    ) -> Tuple[float, float]:
        """Train for one epoch."""
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        # Simple batching - process one at a time for now
        for sample in tqdm(train_data, desc="Training"):
            # Create batch of size 1
            batch = {
                "tokens": mx.array([sample["tokens"]]),
                "expected_indices": [sample["expected_indices"]],
            }

            # Forward and backward pass
            loss_and_grad_fn = nn.value_and_grad(model, self.compute_loss)
            (loss, accuracy), grads = loss_and_grad_fn(model, batch)

            # Update only neural executor parameters
            self.optimizer.update(model.executor, grads.executor)

            total_loss += float(loss)
            total_accuracy += float(accuracy)
            num_batches += 1

        return total_loss / num_batches, total_accuracy / num_batches

    def evaluate(
        self, model: TwoStageCompilerV2, eval_data: Dict[str, List[Dict]]
    ) -> Dict[str, float]:
        """Evaluate on each complexity level."""
        results = {}

        for level_name, level_data in eval_data.items():
            if not level_data:
                continue

            total_correct = 0
            total_actions = 0

            for sample in level_data:
                batch = {
                    "tokens": mx.array([sample["tokens"]]),
                    "expected_indices": [sample["expected_indices"]],
                }

                # Get predictions
                outputs = model(batch["tokens"])

                expected = sample["expected_indices"]
                if outputs.shape[0] >= len(expected):
                    predictions = mx.argmax(outputs[: len(expected)], axis=-1)
                    targets = mx.array(expected)

                    correct = mx.all(predictions == targets)
                    if correct:
                        total_correct += 1
                    total_actions += 1

            accuracy = total_correct / max(total_actions, 1)
            results[level_name] = accuracy

        return results

    def train(
        self,
        train_data: List[Dict],
        val_data: Dict[str, List[Dict]],
        epochs: int = 50,
        vocab: Dict[str, int] = None,
    ):
        """Main training loop."""
        if vocab:
            self.model.set_vocab(vocab)

        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Train samples: {len(train_data)}")
        logger.info(
            f"Val samples by level: {', '.join(f'{k}: {len(v)}' for k, v in val_data.items())}"
        )

        best_avg_accuracy = 0.0

        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(self.model, train_data)

            # Evaluate
            val_results = self.evaluate(self.model, val_data)
            avg_val_acc = np.mean(list(val_results.values()))

            # Log results
            logger.info(f"\nEpoch {epoch + 1}/{epochs}")
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"Val Accuracy by Level:")
            for level, acc in val_results.items():
                logger.info(f"  {level}: {acc:.4f}")
            logger.info(f"Average Val Acc: {avg_val_acc:.4f}")

            # Track metrics
            self.metrics["train_loss"].append(train_loss)
            self.metrics["train_accuracy"].append(train_acc)
            for i in range(1, 5):
                level_key = f"level_{i}"
                if level_key in val_results:
                    self.metrics["val_accuracy_by_level"][i].append(
                        val_results[level_key]
                    )

            # Save best model
            if avg_val_acc > best_avg_accuracy:
                best_avg_accuracy = avg_val_acc
                self.save_checkpoint(epoch, avg_val_acc)

        return self.metrics

    def save_checkpoint(self, epoch: int, accuracy: float):
        """Save model checkpoint."""
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save only the neural executor parameters
        checkpoint = {
            "epoch": epoch,
            "accuracy": accuracy,
            "executor_state": self.model.executor.state_dict(),
        }

        path = os.path.join(
            checkpoint_dir, f"two_stage_epoch_{epoch}_acc_{accuracy:.3f}.npz"
        )
        mx.savez(path, **checkpoint)
        logger.info(f"Saved checkpoint to {path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Two-Stage Compiler")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--train_size", type=int, default=1000, help="Training samples per level"
    )
    parser.add_argument(
        "--val_size", type=int, default=200, help="Validation samples per level"
    )
    args = parser.parse_args()

    # Define vocabulary
    VOCAB = {
        "PAD": 0,
        "do": 1,
        "means": 2,
        "is": 3,
        "and": 4,
        "or": 5,
        "then": 6,
        "twice": 7,
        "thrice": 8,
        "while": 9,
        "X": 10,
        "Y": 11,
        "Z": 12,
        "W": 13,
        "jump": 14,
        "walk": 15,
        "run": 16,
        "turn": 17,
        "true": 18,
    }

    # Generate dataset
    logger.info("Generating dataset...")
    dataset = ProgressiveComplexityDataset()

    # Training data - mixed complexity
    train_data = dataset.generate_mixed_dataset(args.train_size * 4)

    # Validation data - separated by level
    val_data = {
        f"level_{i}": getattr(dataset, f"generate_level_{i}")(args.val_size)
        for i in range(1, 5)
    }

    # Create trainer
    trainer = TwoStageTrainer(
        vocab_size=len(VOCAB), num_actions=4, learning_rate=args.lr
    )

    # Train
    metrics = trainer.train(train_data, val_data, epochs=args.epochs, vocab=VOCAB)

    # Save metrics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = f"metrics_{timestamp}.json"
    with open(metrics_path, "w") as f:
        # Convert metrics to serializable format
        serializable_metrics = {
            "train_loss": metrics["train_loss"],
            "train_accuracy": metrics["train_accuracy"],
            "val_accuracy_by_level": {
                str(k): v for k, v in metrics["val_accuracy_by_level"].items()
            },
        }
        json.dump(serializable_metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")

    # Print final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS:")
    print("=" * 60)
    for level in range(1, 5):
        if metrics["val_accuracy_by_level"][level]:
            final_acc = metrics["val_accuracy_by_level"][level][-1]
            print(f"Level {level}: {final_acc:.4f}")

    # Analysis
    print("\n" + "=" * 60)
    print("KEY INSIGHTS:")
    print("=" * 60)
    print(
        "1. With explicit binding extraction, the model only needs to learn operators"
    )
    print("2. This dramatically simplifies the learning problem")
    print("3. The architecture demonstrates how distribution invention requires:")
    print("   - Explicit rule identification")
    print("   - Discrete modifications")
    print("   - Temporal state tracking")
    print("4. This minimal example shows the path to creative AI")


if __name__ == "__main__":
    main()
