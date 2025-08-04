"""
Variable Binding training with TensorFlow backend for proper gradients.
"""

from utils.imports import setup_project_paths

setup_project_paths()

# Force TensorFlow backend
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import json
import logging
from datetime import datetime

import numpy as np

from utils.config import setup_environment
from utils.paths import get_output_path

# Set up environment
config = setup_environment()
logger = logging.getLogger(__name__)

# Import keras and TensorFlow
import keras
import tensorflow as tf

logger.info(f"Using Keras backend: {keras.backend.backend()}")

from dereferencing_tasks import DereferencingTaskGenerator
from minimal_binding_scan import MinimalBindingModel


def train_step(model, optimizer, loss_fn, batch_commands, batch_actions):
    """Single training step with TensorFlow."""
    with tf.GradientTape() as tape:
        outputs = model({"command": batch_commands}, training=True)
        logits = outputs["action_logits"]

        # Pad actions to match logits sequence length if needed
        action_seq_len = tf.shape(batch_actions)[1]
        logits_seq_len = tf.shape(logits)[1]

        if action_seq_len < logits_seq_len:
            # Pad actions with zeros to match logits length
            padding = [[0, 0], [0, logits_seq_len - action_seq_len]]
            batch_actions_padded = tf.pad(batch_actions, padding, constant_values=0)
        else:
            batch_actions_padded = batch_actions[:, :logits_seq_len]

        # Truncate logits to match actions if needed
        if logits_seq_len > action_seq_len:
            logits = logits[:, :action_seq_len, :]
            batch_actions_padded = batch_actions

        loss = loss_fn(batch_actions_padded, logits)

    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    # Compute accuracy
    predictions = tf.argmax(logits, axis=-1)
    mask = tf.not_equal(batch_actions_padded, 0)  # Non-padding positions
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(predictions, batch_actions_padded), tf.float32)
        * tf.cast(mask, tf.float32)
    )

    return loss, accuracy


def evaluate(model, data, batch_size=32):
    """Evaluate model on dataset."""
    total_loss = 0
    total_accuracy = 0
    n_batches = 0

    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    for i in range(0, len(data["commands"]), batch_size):
        batch_commands = data["commands"][i : i + batch_size]
        batch_actions = data["actions"][i : i + batch_size]

        outputs = model({"command": batch_commands}, training=False)
        logits = outputs["action_logits"]

        # Handle sequence length mismatch
        action_seq_len = tf.shape(batch_actions)[1]
        logits_seq_len = tf.shape(logits)[1]

        if action_seq_len < logits_seq_len:
            # Pad actions with zeros to match logits length
            padding = [[0, 0], [0, logits_seq_len - action_seq_len]]
            batch_actions_padded = tf.pad(batch_actions, padding, constant_values=0)
        else:
            batch_actions_padded = batch_actions[:, :logits_seq_len]

        # Truncate logits to match actions if needed
        if logits_seq_len > action_seq_len:
            logits = logits[:, :action_seq_len, :]
            batch_actions_padded = batch_actions

        loss = loss_fn(batch_actions_padded, logits)
        predictions = tf.argmax(logits, axis=-1)
        mask = tf.not_equal(batch_actions_padded, 0)
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(predictions, batch_actions_padded), tf.float32)
            * tf.cast(mask, tf.float32)
        )

        total_loss += float(loss)
        total_accuracy += float(accuracy)
        n_batches += 1

    return total_loss / n_batches, total_accuracy / n_batches


def test_specific_examples(model, generator):
    """Test specific binding examples."""
    test_cases = [
        ("X jump", "JUMP"),
        ("Y walk", "WALK"),
        ("X jump and Y walk", "JUMP WALK"),
        ("define X as jump then X", "JUMP"),
        ("define Y as walk then Y Y", "WALK WALK"),
    ]

    results = []
    for command_str, expected_str in test_cases:
        # Encode
        command_tokens = command_str.split()
        command_ids = generator.encode_words(command_tokens)

        # Predict
        outputs = model({"command": np.expand_dims(command_ids, 0)}, training=False)
        pred_ids = tf.argmax(outputs["action_logits"][0], axis=-1).numpy()

        # Decode
        pred_tokens = []
        for idx in pred_ids:
            if idx == 0:  # PAD
                break
            token = generator.id_to_action.get(idx, "<UNK>")
            pred_tokens.append(token)

        predicted_str = " ".join(pred_tokens)
        correct = predicted_str == expected_str

        results.append(
            {
                "command": command_str,
                "expected": expected_str,
                "predicted": predicted_str,
                "correct": correct,
            }
        )

        logger.info(
            f"Command: '{command_str}' -> Expected: '{expected_str}', "
            f"Got: '{predicted_str}' {'✓' if correct else '✗'}"
        )

    return results


def main(epochs=20, batch_size=32, learning_rate=0.001):
    """Main training function."""
    logger.info("=== Variable Binding Training (TensorFlow) ===")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = get_output_path("binding_training", f"tensorflow_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Generate data
    logger.info("Generating dereferencing dataset...")
    generator = DereferencingTaskGenerator(seed=42)
    dataset = generator.generate_dataset(n_samples=2000)

    logger.info(
        f"Dataset sizes - Train: {len(dataset['train']['commands'])}, "
        f"Val: {len(dataset['val']['commands'])}, "
        f"Test: {len(dataset['test']['commands'])}"
    )

    # Create model
    model = MinimalBindingModel(
        vocab_size=len(generator.word_vocab),
        action_vocab_size=len(generator.action_vocab),
        n_slots=10,
        embed_dim=128,
        hidden_dim=256,
    )

    # Build model
    dummy_input = {"command": dataset["train"]["commands"][:1]}
    _ = model(dummy_input)
    logger.info(f"Model initialized with {model.count_params():,} parameters")

    # Create optimizer and loss
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Training history
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    # Training loop
    logger.info("\nStarting training...")
    best_val_accuracy = 0

    for epoch in range(epochs):
        logger.info(f"\nEpoch {epoch + 1}/{epochs}")

        # Shuffle training data
        indices = np.random.permutation(len(dataset["train"]["commands"]))

        # Training
        epoch_loss = 0
        epoch_accuracy = 0
        n_batches = 0

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i : i + batch_size]
            batch_commands = dataset["train"]["commands"][batch_indices]
            batch_actions = dataset["train"]["actions"][batch_indices]

            loss, accuracy = train_step(
                model, optimizer, loss_fn, batch_commands, batch_actions
            )

            epoch_loss += float(loss)
            epoch_accuracy += float(accuracy)
            n_batches += 1

            if n_batches % 20 == 0:
                logger.info(f"  Batch {n_batches}: loss={loss:.4f}, acc={accuracy:.4f}")

        # Average training metrics
        avg_train_loss = epoch_loss / n_batches
        avg_train_accuracy = epoch_accuracy / n_batches

        # Validation
        val_loss, val_accuracy = evaluate(model, dataset["val"], batch_size)

        # Save history
        history["train_loss"].append(avg_train_loss)
        history["train_accuracy"].append(avg_train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        logger.info(
            f"  Train - Loss: {avg_train_loss:.4f}, Accuracy: {avg_train_accuracy:.4f}"
        )
        logger.info(f"  Val   - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model.save_weights(os.path.join(output_dir, "best_model.weights.h5"))
            logger.info(f"  Saved best model (val_acc: {val_accuracy:.4f})")

        # Test specific examples every 5 epochs
        if (epoch + 1) % 5 == 0:
            logger.info("\n  Testing specific examples:")
            test_results = test_specific_examples(model, generator)
            n_correct = sum(r["correct"] for r in test_results)
            logger.info(f"  Specific examples: {n_correct}/{len(test_results)} correct")

    # Final evaluation
    logger.info("\n=== Final Evaluation ===")

    # Load best weights
    model.load_weights(os.path.join(output_dir, "best_model.weights.h5"))

    # Test set evaluation
    test_loss, test_accuracy = evaluate(model, dataset["test"], batch_size)
    logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Final specific examples
    logger.info("\nFinal specific examples:")
    final_test_results = test_specific_examples(model, generator)

    # Test modifications
    logger.info("\nTesting modifications:")
    mod_test_cases = generator.generate_modification_test_set()
    mod_results = []

    for case in mod_test_cases:
        # Original
        orig_cmd = generator.encode_words(case["original"]["command"])
        orig_out = model({"command": np.expand_dims(orig_cmd, 0)}, training=False)
        orig_pred = tf.argmax(orig_out["action_logits"][0], axis=-1).numpy()

        # Modified
        mod_cmd = generator.encode_words(case["modified"]["command"])
        mod_out = model({"command": np.expand_dims(mod_cmd, 0)}, training=False)
        mod_pred = tf.argmax(mod_out["action_logits"][0], axis=-1).numpy()

        # Check if modification worked
        orig_expected = generator.encode_actions(case["original"]["actions"])
        mod_expected = generator.encode_actions(case["modified"]["actions"])

        orig_correct = np.array_equal(orig_pred[: len(orig_expected)], orig_expected)
        mod_correct = np.array_equal(mod_pred[: len(mod_expected)], mod_expected)

        mod_results.append(
            {
                "type": case["modification_type"],
                "original_correct": orig_correct,
                "modified_correct": mod_correct,
            }
        )

        logger.info(
            f"  {case['modification_type']}: "
            f"Original {'✓' if orig_correct else '✗'}, "
            f"Modified {'✓' if mod_correct else '✗'}"
        )

    mod_success_rate = sum(r["modified_correct"] for r in mod_results) / len(
        mod_results
    )
    logger.info(f"\nModification success rate: {mod_success_rate:.2%}")

    # Save results
    results = {
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "model_params": model.count_params(),
        },
        "final_metrics": {
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "best_val_accuracy": best_val_accuracy,
            "modification_success_rate": mod_success_rate,
        },
        "history": history,
        "specific_examples": final_test_results,
        "modification_results": mod_results,
    }

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {output_dir}")

    if test_accuracy > 0.8 and mod_success_rate > 0.5:
        logger.info("\n✓ SUCCESS: Variable binding achieved!")
    else:
        logger.info("\n✗ Model needs more training or architectural improvements")

    return model, results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        logger.info("Running quick test (5 epochs)...")
        model, results = main(epochs=5, batch_size=32)
    else:
        logger.info("Running full training (20 epochs)...")
        model, results = main(epochs=20, batch_size=32)
