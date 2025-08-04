#!/usr/bin/env python3
"""
Compare model outputs between Stage 1 and Stage 2 to understand modification behavior.

This script recreates the model and loads weights to compare predictions.
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import json

import tensorflow as tf
from scan_data_loader import SCANDataLoader
from train_progressive_curriculum import SCANTokenizer

# Import model creation function
from models import create_model


def load_vocabulary(vocab_path):
    """Load vocabulary from JSON file."""
    with open(vocab_path, "r") as f:
        return json.load(f)


def recreate_tokenizer(vocab_data):
    """Recreate tokenizer from saved vocabulary."""
    tokenizer = SCANTokenizer()
    tokenizer.command_word_index = vocab_data["command_to_id"]
    tokenizer.action_word_index = vocab_data["action_to_id"]
    tokenizer.command_index_word = {
        v: k for k, v in tokenizer.command_word_index.items()
    }
    tokenizer.action_index_word = {v: k for k, v in tokenizer.action_word_index.items()}
    return tokenizer


def generate_predictions(model, tokenizer, samples, use_modification=False):
    """Generate predictions for a set of samples."""
    predictions = []

    for sample in samples:
        # Encode command
        command_tokens = tokenizer.encode_command(sample.command)
        command_tensor = tf.constant([command_tokens])

        # Create modification tensor (zeros if not using modification)
        mod_tensor = tf.zeros_like(command_tensor)

        # Create inputs
        inputs = {"command": command_tensor, "modification": mod_tensor}

        # Get rule embeddings
        rule_outputs = model(inputs, training=False)
        rule_embeddings = rule_outputs["rule_embeddings"]

        # Generate sequence
        generated = model.sequence_generator.generate(
            rule_embeddings,
            start_token=tokenizer.action_word_index["<START>"],
            end_token=tokenizer.action_word_index["<END>"],
            max_length=50,
        )

        # Decode
        predicted_action = tokenizer.decode_action(generated[0].numpy())
        predictions.append(
            {
                "command": sample.command,
                "expected": sample.action,
                "predicted": predicted_action,
                "correct": predicted_action == sample.action,
            }
        )

    return predictions


def compare_stages(stage1_path, stage2_path, vocab_path, num_samples=10):
    """Compare outputs from two different stage checkpoints."""

    print("Loading vocabulary...")
    vocab_data = load_vocabulary(vocab_path)
    tokenizer = recreate_tokenizer(vocab_data)

    print("Creating model...")
    model = create_model(
        len(tokenizer.command_word_index), len(tokenizer.action_word_index)
    )

    print("Loading test data...")
    data_loader = SCANDataLoader()
    data_loader.load_all_data()
    splits = data_loader.create_isolated_splits()
    test_samples = splits["test_interpolation"][:num_samples]

    print(f"\nComparing {num_samples} samples between stages...")
    print("=" * 80)

    # Stage 1 predictions
    print("\n1. Loading Stage 1 weights...")
    model.load_weights(stage1_path)
    stage1_predictions = generate_predictions(model, tokenizer, test_samples)

    # Stage 2 predictions
    print("2. Loading Stage 2 weights...")
    model.load_weights(stage2_path)
    stage2_predictions = generate_predictions(model, tokenizer, test_samples)

    # Compare results
    print("\n3. Comparison Results:")
    print("-" * 80)

    differences = 0
    stage1_correct = 0
    stage2_correct = 0

    for i, (pred1, pred2) in enumerate(zip(stage1_predictions, stage2_predictions)):
        stage1_correct += pred1["correct"]
        stage2_correct += pred2["correct"]

        if pred1["predicted"] != pred2["predicted"]:
            differences += 1
            print(f"\nDifference found in example {i+1}:")
            print(f"Command: {pred1['command']}")
            print(f"Expected: {pred1['expected']}")
            print(f"Stage 1: {pred1['predicted']} {'✓' if pred1['correct'] else '✗'}")
            print(f"Stage 2: {pred2['predicted']} {'✓' if pred2['correct'] else '✗'}")

    print(f"\n\nSummary:")
    print(f"Total samples: {num_samples}")
    print(
        f"Stage 1 accuracy: {stage1_correct}/{num_samples} ({stage1_correct/num_samples*100:.1f}%)"
    )
    print(
        f"Stage 2 accuracy: {stage2_correct}/{num_samples} ({stage2_correct/num_samples*100:.1f}%)"
    )
    print(f"Different predictions: {differences} ({differences/num_samples*100:.1f}%)")

    # Analyze the nature of differences
    if differences > 0:
        print("\n⚠️ WARNING: Stage 2 produces different outputs than Stage 1!")
        print("This suggests the modification training is affecting base behavior.")
    else:
        print(
            "\n✓ Stage 2 maintains identical predictions to Stage 1 on unmodified inputs."
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compare model outputs between stages")
    parser.add_argument(
        "--stage1",
        default="compositional_language_complete_20250722_185804/outputs/safeguarded_training/checkpoints/stage_1_epoch_5.h5",
    )
    parser.add_argument(
        "--stage2",
        default="compositional_language_complete_20250722_185804/outputs/safeguarded_training/checkpoints/stage_2_epoch_1.h5",
    )
    parser.add_argument(
        "--vocabulary",
        default="compositional_language_complete_20250722_185804/outputs/safeguarded_training/vocabulary.json",
    )
    parser.add_argument("--num_samples", type=int, default=20)
    args = parser.parse_args()

    compare_stages(args.stage1, args.stage2, args.vocabulary, args.num_samples)


if __name__ == "__main__":
    main()
