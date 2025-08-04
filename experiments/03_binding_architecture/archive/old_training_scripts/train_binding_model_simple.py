"""
Simplified training script for Minimal Binding Model

This version uses Keras 3's built-in training loop instead of manual gradient computation.
"""

from utils.imports import setup_project_paths

setup_project_paths()

import json
import logging
import os
from datetime import datetime

import keras
import numpy as np
from dereferencing_tasks import DereferencingTaskGenerator
from keras import ops
from minimal_binding_scan import MinimalBindingModel

from utils.config import setup_environment
from utils.paths import get_output_path

# Set up environment
config = setup_environment()
logger = logging.getLogger(__name__)


class CustomMinimalBindingModel(MinimalBindingModel):
    """Extended model that works with Keras compile/fit."""

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        """Compute the loss for training."""
        if y_pred is None:
            y_pred = self(x, training=True)

        # Extract action logits from the output dictionary
        action_logits = y_pred["action_logits"]

        # Use the compiled loss function
        loss = self.compiled_loss(y, action_logits, sample_weight=sample_weight)

        return loss

    def compute_metrics(self, x, y, y_pred, sample_weight=None):
        """Update metrics."""
        # Extract action logits
        action_logits = y_pred["action_logits"]

        # Update compiled metrics
        self.compiled_metrics.update_state(
            y, action_logits, sample_weight=sample_weight
        )

        return {m.name: m.result() for m in self.metrics}


def prepare_data_for_keras(dataset, generator):
    """Convert dataset to format expected by Keras."""
    # Get commands and actions
    commands = dataset["commands"]
    actions = dataset["actions"]

    # Create dataset that returns proper format
    return {"command": commands}, actions


def evaluate_model(model, dataset, generator):
    """Custom evaluation function."""
    commands = dataset["commands"]
    actions = dataset["actions"]

    total_correct = 0
    total_samples = len(commands)

    # Process in batches
    batch_size = 32
    for i in range(0, len(commands), batch_size):
        batch_commands = commands[i : i + batch_size]
        batch_actions = actions[i : i + batch_size]

        # Get predictions
        outputs = model({"command": batch_commands}, training=False)
        predictions = ops.argmax(outputs["action_logits"], axis=-1)

        # Check accuracy per sequence
        for j in range(len(batch_commands)):
            pred = predictions[j]
            target = batch_actions[j]

            # Find non-padding positions
            non_pad = target != 0

            if len(non_pad) > 0 and ops.all(pred[non_pad] == target[non_pad]):
                total_correct += 1

    accuracy = total_correct / total_samples
    return accuracy


def test_modification_capability(model, generator):
    """Test if model can handle modifications."""
    test_cases = generator.generate_modification_test_set()

    results = []

    for test_case in test_cases:
        # Test original
        orig_cmd = generator.encode_words(test_case["original"]["command"])
        orig_act = generator.encode_actions(test_case["original"]["actions"])

        orig_cmd = np.expand_dims(orig_cmd, 0)
        outputs = model({"command": orig_cmd}, training=False)
        orig_pred = ops.argmax(outputs["action_logits"], axis=-1)

        # Test modified
        mod_cmd = generator.encode_words(test_case["modified"]["command"])
        mod_act = generator.encode_actions(test_case["modified"]["actions"])

        mod_cmd = np.expand_dims(mod_cmd, 0)
        outputs = model({"command": mod_cmd}, training=False)
        mod_pred = ops.argmax(outputs["action_logits"], axis=-1)

        # Check if predictions match expected
        orig_correct = ops.all(orig_pred[0, : len(orig_act)] == orig_act)
        mod_correct = ops.all(mod_pred[0, : len(mod_act)] == mod_act)

        results.append(
            {
                "modification_type": test_case["modification_type"],
                "original_correct": bool(orig_correct),
                "modified_correct": bool(mod_correct),
                "original_command": " ".join(test_case["original"]["command"]),
                "modified_command": " ".join(test_case["modified"]["command"]),
            }
        )

    return results


class ModificationTestCallback(keras.callbacks.Callback):
    """Callback to test modification capability during training."""

    def __init__(self, generator, output_dir):
        super().__init__()
        self.generator = generator
        self.output_dir = output_dir
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            logger.info(f"\nEpoch {epoch + 1}: Testing modification capability...")
            results = test_modification_capability(self.model, self.generator)

            n_correct = sum(r["modified_correct"] for r in results)
            success_rate = n_correct / len(results) if results else 0

            logger.info(f"Modification success rate: {success_rate:.2%}")

            # Show examples
            for result in results[:2]:
                logger.info(
                    f"  {result['original_command']} -> {result['modified_command']}"
                )
                logger.info(
                    f"  Original: {'✓' if result['original_correct'] else '✗'}, "
                    f"Modified: {'✓' if result['modified_correct'] else '✗'}"
                )

            self.history.append(
                {"epoch": epoch + 1, "success_rate": success_rate, "details": results}
            )


def main():
    """Main training script."""
    logger.info("Starting Minimal Binding Model training (simplified version)...")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = get_output_path() / f"binding_model_simple_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize components
    generator = DereferencingTaskGenerator(seed=42)

    # Generate dataset
    logger.info("Generating training data...")
    dataset = generator.generate_dataset(n_samples=1000)

    # Prepare data
    train_x, train_y = prepare_data_for_keras(dataset["train"], generator)
    val_x, val_y = prepare_data_for_keras(dataset["val"], generator)

    logger.info(f"Train samples: {len(train_y)}")
    logger.info(f"Val samples: {len(val_y)}")

    # Create model
    model = CustomMinimalBindingModel(
        vocab_size=len(generator.word_vocab),
        action_vocab_size=len(generator.action_vocab),
        n_slots=10,
        embed_dim=128,
        hidden_dim=256,
    )

    # Build model
    dummy_input = {"command": np.zeros((1, 10), dtype=np.int32)}
    _ = model(dummy_input)

    logger.info(f"Model initialized with {model.count_params()} parameters")

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # Callbacks
    callbacks = [
        ModificationTestCallback(generator, output_dir),
        keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, "best_model.keras"),
            save_best_only=True,
            monitor="val_loss",
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
    ]

    # Train model
    history = model.fit(
        x=train_x,
        y=train_y,
        validation_data=(val_x, val_y),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    # Save final model
    final_model_path = os.path.join(output_dir, "binding_model_final.keras")
    model.save(final_model_path)
    logger.info(f"Final model saved to: {final_model_path}")

    # Final evaluation
    logger.info("\nFinal evaluation:")
    train_acc = evaluate_model(model, dataset["train"], generator)
    val_acc = evaluate_model(model, dataset["val"], generator)
    test_acc = evaluate_model(model, dataset["test"], generator)

    logger.info(f"Train accuracy: {train_acc:.2%}")
    logger.info(f"Val accuracy: {val_acc:.2%}")
    logger.info(f"Test accuracy: {test_acc:.2%}")

    # Final modification test
    logger.info("\nFinal modification capability test:")
    final_results = test_modification_capability(model, generator)

    n_correct = sum(r["modified_correct"] for r in final_results)
    final_success_rate = n_correct / len(final_results) if final_results else 0

    logger.info(f"Final modification success rate: {final_success_rate:.2%}")

    # Save results
    results_data = {
        "final_modification_success_rate": final_success_rate,
        "modification_test_results": final_results,
        "accuracies": {"train": train_acc, "val": val_acc, "test": test_acc},
        "training_history": history.history,
    }

    results_path = os.path.join(output_dir, "final_results.json")
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)

    logger.info(f"Results saved to: {results_path}")

    # Save modification callback history
    if callbacks[0].history:
        mod_history_path = os.path.join(output_dir, "modification_history.json")
        with open(mod_history_path, "w") as f:
            json.dump(callbacks[0].history, f, indent=2)
        logger.info(f"Modification history saved to: {mod_history_path}")


if __name__ == "__main__":
    main()
