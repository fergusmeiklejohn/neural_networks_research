#!/usr/bin/env python3
"""
Analyze model predictions to understand modification behavior.

This script loads a checkpoint and examines:
1. Whether modifications are being applied at all
2. How predictions differ between modified and unmodified inputs
3. Attention patterns in the modification component
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import argparse
import json

# Add project root to path
import sys
from pathlib import Path

import tensorflow as tf

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from modification_generator import ModificationGenerator
from scan_data_loader import SCANDataLoader
from train_progressive_curriculum import SCANTokenizer

from models import CompositionalLanguageModel


def load_model_and_tokenizer(checkpoint_path, vocabulary_path):
    """Load model from checkpoint and tokenizer from vocabulary."""
    # Load vocabulary
    with open(vocabulary_path, "r") as f:
        vocab_data = json.load(f)

    # Create tokenizer
    tokenizer = SCANTokenizer()
    tokenizer.command_word_index = vocab_data["command_to_id"]
    tokenizer.action_word_index = vocab_data["action_to_id"]
    tokenizer.command_index_word = {
        v: k for k, v in tokenizer.command_word_index.items()
    }
    tokenizer.action_index_word = {v: k for k, v in tokenizer.action_word_index.items()}

    # Create model
    model = CompositionalLanguageModel(
        command_vocab_size=len(tokenizer.command_word_index),
        action_vocab_size=len(tokenizer.action_word_index),
        d_model=128,
    )

    # Load weights
    model.load_weights(checkpoint_path)

    return model, tokenizer


def analyze_predictions(model, tokenizer, test_samples, modification_pairs):
    """Analyze model predictions on test samples with and without modifications."""

    print("\n=== ANALYZING MODEL PREDICTIONS ===\n")

    # Test 1: Basic predictions without modifications
    print("1. Testing basic predictions (no modifications):")
    print("-" * 50)

    for i, sample in enumerate(test_samples[:5]):
        command_tokens = tokenizer.encode_command(sample["command"])
        command_tensor = tf.constant([command_tokens])

        # Create dummy modification (all zeros)
        dummy_mod = tf.zeros_like(command_tensor)

        # Get prediction
        inputs = {"command": command_tensor, "modification": dummy_mod}

        # Get rule embeddings for generation
        rule_outputs = model(inputs, training=False)
        rule_embeddings = rule_outputs["rule_embeddings"]

        # Generate sequence
        predicted_sequence = model.sequence_generator.generate(
            rule_embeddings,
            start_token=tokenizer.action_word_index["<START>"],
            end_token=tokenizer.action_word_index["<END>"],
            max_length=50,
        )

        # Decode prediction
        predicted_action = tokenizer.decode_action(predicted_sequence[0].numpy())

        print(f"\nExample {i+1}:")
        print(f"Command: {sample['command']}")
        print(f"Expected: {sample['action']}")
        print(f"Predicted: {predicted_action}")
        print(f"Correct: {predicted_action == sample['action']}")

    # Test 2: Predictions with modifications
    print("\n\n2. Testing predictions with modifications:")
    print("-" * 50)

    # Get some modification examples
    mod_examples = modification_pairs[:5]

    for i, mod_pair in enumerate(mod_examples):
        # Encode original command
        command_tokens = tokenizer.encode_command(mod_pair.original_sample.command)
        command_tensor = tf.constant([command_tokens])

        # Encode modification
        mod_tokens = tokenizer.encode_command(mod_pair.modification_description)
        mod_tensor = tf.constant([mod_tokens])

        # Get prediction with modification
        inputs = {"command": command_tensor, "modification": mod_tensor}

        # Get modified rule embeddings
        rule_outputs = model(inputs, training=False)
        rule_embeddings = rule_outputs["rule_embeddings"]

        # Generate sequence
        predicted_sequence = model.sequence_generator.generate(
            rule_embeddings,
            start_token=tokenizer.action_word_index["<START>"],
            end_token=tokenizer.action_word_index["<END>"],
            max_length=50,
        )

        # Decode prediction
        predicted_action = tokenizer.decode_action(predicted_sequence[0].numpy())

        print(f"\nModification Example {i+1}:")
        print(f"Original Command: {mod_pair.original_sample.command}")
        print(f"Original Action: {mod_pair.original_sample.action}")
        print(f"Modification: {mod_pair.modification_description}")
        print(f"Expected Modified Action: {mod_pair.modified_sample.action}")
        print(f"Predicted Action: {predicted_action}")
        print(
            f"Modification Applied: {predicted_action != mod_pair.original_sample.action}"
        )
        print(
            f"Correct Modification: {predicted_action == mod_pair.modified_sample.action}"
        )

    # Test 3: Analyze rule embeddings
    print("\n\n3. Analyzing rule embedding changes:")
    print("-" * 50)

    # Compare embeddings with and without modification
    sample = modification_pairs[0]
    command_tokens = tokenizer.encode_command(sample.original_sample.command)
    command_tensor = tf.constant([command_tokens])

    # Get embeddings without modification
    dummy_mod = tf.zeros_like(command_tensor)
    inputs_no_mod = {"command": command_tensor, "modification": dummy_mod}
    outputs_no_mod = model(inputs_no_mod, training=False)
    embeddings_no_mod = outputs_no_mod["rule_embeddings"]

    # Get embeddings with modification
    mod_tokens = tokenizer.encode_command(sample.modification_description)
    mod_tensor = tf.constant([mod_tokens])
    inputs_with_mod = {"command": command_tensor, "modification": mod_tensor}
    outputs_with_mod = model(inputs_with_mod, training=False)
    embeddings_with_mod = outputs_with_mod["rule_embeddings"]

    # Calculate embedding difference
    embedding_diff = tf.reduce_mean(tf.abs(embeddings_with_mod - embeddings_no_mod))

    print(f"Command: {sample.original_sample.command}")
    print(f"Modification: {sample.modification_description}")
    print(f"Average embedding difference: {embedding_diff:.4f}")
    print(
        f"Max embedding difference: {tf.reduce_max(tf.abs(embeddings_with_mod - embeddings_no_mod)):.4f}"
    )

    # Check if modification is actually being processed
    if embedding_diff < 0.01:
        print("\n⚠️ WARNING: Modifications are having minimal effect on embeddings!")
        print("This suggests the modification component may not be working properly.")


def main():
    parser = argparse.ArgumentParser(description="Analyze model predictions")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--vocabulary", help="Path to vocabulary.json (default: same dir as checkpoint)"
    )
    parser.add_argument(
        "--num_examples", type=int, default=10, help="Number of examples to analyze"
    )
    args = parser.parse_args()

    # Set vocabulary path
    if args.vocabulary is None:
        checkpoint_dir = Path(args.checkpoint).parent
        args.vocabulary = checkpoint_dir / "vocabulary.json"

    print(f"Loading model from: {args.checkpoint}")
    print(f"Loading vocabulary from: {args.vocabulary}")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.checkpoint, args.vocabulary)

    # Load test data
    print("\nLoading test data...")
    data_loader = SCANDataLoader()
    data_loader.load_data()
    train_data = [
        {"command": s.command, "action": s.action} for s in data_loader.train_samples
    ]
    test_data_interpolation = [
        {"command": s.command, "action": s.action}
        for s in data_loader.test_samples_interpolation
    ]

    # Load modifications
    print("Loading modifications...")
    mod_gen = ModificationGenerator()
    modification_pairs = mod_gen.load_modifications()

    # Analyze predictions
    analyze_predictions(
        model,
        tokenizer,
        test_data_interpolation[: args.num_examples],
        modification_pairs,
    )


if __name__ == "__main__":
    main()
