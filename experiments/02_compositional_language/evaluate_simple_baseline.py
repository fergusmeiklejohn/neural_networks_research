#!/usr/bin/env python3
"""
Evaluate the simple baseline model using proper validation sets.

This script loads the trained simple baseline model and evaluates it
on all modification-specific validation sets to reveal its true performance.
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import argparse
import json
from pathlib import Path

import numpy as np

# Import evaluation utilities
from evaluation_v2 import ModificationAwareEvaluator
from tensorflow import keras
from train_progressive_curriculum import SCANTokenizer


def load_simple_baseline_model(model_path: Path, config_path: Path):
    """Load the simple baseline model."""
    # Load config
    with open(config_path, "r") as f:
        config = json.load(f)

    # Load model
    model = keras.models.load_model(str(model_path))

    return model, config


def create_generate_function(model, tokenizer, max_length=50):
    """Create a generate function that matches the expected interface."""

    def generate_action(
        command_ids, modification=None, start_token=None, end_token=None, **kwargs
    ):
        """Generate action sequence given command."""
        # Ensure inputs are numpy arrays
        if isinstance(command_ids, list):
            command_ids = np.array(command_ids)

        # Add batch dimension if needed
        if len(command_ids.shape) == 1:
            command_ids = np.expand_dims(command_ids, axis=0)
            single_sample = True
        else:
            single_sample = False

        batch_size = command_ids.shape[0]

        # Prepare modification input
        if modification is None:
            modification_input = np.zeros((batch_size, 8), dtype=np.float32)
        else:
            if isinstance(modification, (list, tuple)):
                modification_input = np.array(modification, dtype=np.float32)
            else:
                modification_input = modification

            if len(modification_input.shape) == 1:
                modification_input = np.expand_dims(modification_input, axis=0)

        # Start with START token
        current_token = np.full(
            (batch_size, 1), tokenizer.action_to_id["<START>"], dtype=np.int32
        )
        generated_tokens = []

        for _ in range(max_length):
            # Predict next token
            predictions = model.predict(
                [command_ids, current_token, modification_input], verbose=0
            )

            # Get the last token prediction
            next_token_logits = predictions[:, -1, :]
            next_token = np.argmax(next_token_logits, axis=-1)
            next_token = np.expand_dims(next_token, axis=1)

            generated_tokens.append(next_token)

            # Check for END token
            if np.all(next_token == tokenizer.action_to_id["<END>"]):
                break

            # Update current sequence
            current_token = np.concatenate([current_token, next_token], axis=1)

        # Concatenate all generated tokens
        if generated_tokens:
            result = np.concatenate(generated_tokens, axis=1)
        else:
            result = np.zeros((batch_size, 1), dtype=np.int32)

        # Remove batch dimension if single sample
        if single_sample:
            result = result[0]

        return result

    return generate_action


def main():
    parser = argparse.ArgumentParser(description="Evaluate simple baseline model")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="outputs/simple_baseline_v2_20250725_072805",
        help="Directory containing the trained model",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for evaluation"
    )

    args = parser.parse_args()

    print("Evaluating Simple Baseline Model")
    print("=" * 60)

    # Setup paths
    base_dir = Path(__file__).parent
    model_dir = base_dir / args.model_dir
    model_path = model_dir / "model.h5"
    config_path = model_dir / "config.json"

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    # Load model and config
    print(f"Loading model from: {model_path}")
    model, config = load_simple_baseline_model(model_path, config_path)
    print(f"Model config: {json.dumps(config, indent=2)}")

    # Setup tokenizer
    data_path = base_dir / "data" / "processed"
    vocab_path = (
        base_dir
        / "compositional_language_complete_20250722_185804"
        / "outputs"
        / "safeguarded_training"
        / "vocabulary.json"
    )

    if not vocab_path.exists():
        vocab_path = data_path / "vocabulary.json"

    with open(vocab_path, "r") as f:
        vocab_data = json.load(f)

    tokenizer = SCANTokenizer(None)
    if "command_to_id" in vocab_data:
        tokenizer.command_to_id = vocab_data["command_to_id"]
        tokenizer.action_to_id = vocab_data["action_to_id"]
    else:
        tokenizer.command_to_id = {
            token: i for i, token in enumerate(vocab_data["command_vocab"])
        }
        tokenizer.action_to_id = {
            token: i for i, token in enumerate(vocab_data["action_vocab"])
        }

    tokenizer.id_to_command = {v: k for k, v in tokenizer.command_to_id.items()}
    tokenizer.id_to_action = {v: k for k, v in tokenizer.action_to_id.items()}

    # Create evaluator
    evaluator = ModificationAwareEvaluator(tokenizer)

    # Add generate_action method to model
    model.generate_action = create_generate_function(model, tokenizer)

    # Evaluate on all validation sets
    validation_dir = data_path / "proper_validation_sets"
    print(f"\nEvaluating on validation sets from: {validation_dir}")

    results = evaluator.evaluate_all_sets(
        model, validation_dir, batch_size=args.batch_size
    )

    # Save results
    results_path = model_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    for set_name, set_results in results.items():
        if isinstance(set_results, dict) and "accuracy" in set_results:
            accuracy = set_results["accuracy"]
            total = set_results.get("total", 0)
            print(f"{set_name:.<30} {accuracy:>6.2%} ({total} examples)")

    # Print aggregate metrics
    if "aggregate_metrics" in results:
        print("\nAggregate Metrics:")
        metrics = results["aggregate_metrics"]
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")

    # Check for evaluation illusion
    print("\n" + "=" * 60)
    print("EVALUATION ILLUSION CHECK")
    print("=" * 60)

    base_acc = results.get("val_base", {}).get("accuracy", 0)
    mod_accs = [
        results.get(k, {}).get("accuracy", 0)
        for k in results
        if k.startswith("val_mod_") and isinstance(results[k], dict)
    ]

    if mod_accs:
        avg_mod_acc = np.mean(mod_accs)
        print(f"Base accuracy: {base_acc:.2%}")
        print(f"Average modification accuracy: {avg_mod_acc:.2%}")
        print(f"Performance drop: {base_acc - avg_mod_acc:.2%}")

        if avg_mod_acc < 0.1 and base_acc > 0.1:
            print("\n⚠️  EVALUATION ILLUSION CONFIRMED!")
            print("The model performs on base examples but fails on modifications.")
            print("This indicates it never learned to handle rule modifications.")
        elif avg_mod_acc < base_acc * 0.5:
            print("\n⚠️  SIGNIFICANT MODIFICATION WEAKNESS DETECTED!")
            print(
                "The model's modification performance is less than half of base performance."
            )
        else:
            print("\n✓ Model shows some ability to handle modifications.")
    else:
        print("No modification results found.")


if __name__ == "__main__":
    main()
