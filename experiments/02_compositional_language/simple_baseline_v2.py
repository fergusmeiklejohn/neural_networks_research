#!/usr/bin/env python3
"""
Simplified Baseline Model V2 - Even simpler approach

This version uses the most straightforward possible architecture
to establish a true baseline without any complexity.
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
from tensorflow import keras

# Import from current directory
from train_progressive_curriculum import SCANTokenizer


def create_simple_model(
    command_vocab_size: int, action_vocab_size: int, d_model: int = 128
) -> keras.Model:
    """Create a simple encoder-decoder model using Functional API."""

    # Input layers
    command_input = keras.Input(shape=(None,), dtype="int32", name="commands")
    action_input = keras.Input(shape=(None,), dtype="int32", name="actions")
    modification_input = keras.Input(shape=(8,), dtype="float32", name="modification")

    # Shared embedding layer
    embedding = keras.layers.Embedding(
        max(command_vocab_size, action_vocab_size), d_model, mask_zero=True
    )

    # Encode commands
    command_embedded = embedding(command_input)

    # Add modification signal to embeddings (simple broadcast addition)
    mod_projection = keras.layers.Dense(d_model, activation="tanh")(modification_input)
    mod_projection = keras.layers.Reshape((1, d_model))(mod_projection)
    # Broadcast addition using keras.layers.Add
    command_embedded = keras.layers.Add()([command_embedded, mod_projection])

    # Encoder LSTM
    encoder_output, state_h, state_c = keras.layers.LSTM(
        d_model, return_sequences=True, return_state=True
    )(command_embedded)
    encoder_states = [state_h, state_c]

    # Decoder
    action_embedded = embedding(action_input)
    decoder_lstm = keras.layers.LSTM(d_model, return_sequences=True)
    decoder_output = decoder_lstm(action_embedded, initial_state=encoder_states)

    # Output projection
    output = keras.layers.Dense(action_vocab_size, name="output")(decoder_output)

    # Create model
    model = keras.Model(
        inputs=[command_input, action_input, modification_input], outputs=output
    )

    return model


def load_and_prepare_data(data_path: Path, tokenizer: SCANTokenizer):
    """Load data and prepare for training."""

    # Load base training data
    with open(data_path / "train.pkl", "rb") as f:
        train_data = pickle.load(f)

    # Convert to numpy arrays
    commands = []
    actions = []
    modifications = []

    print("Preparing base training data...")
    for sample in train_data[:10000]:  # Use subset for testing
        # Handle both dict and object formats
        if isinstance(sample, dict):
            command = sample["command"]
            action = sample["action"]
        else:
            command = sample.command
            action = sample.action

        # Tokenize
        command_ids = tokenizer.encode_command(command)
        action_ids = tokenizer.encode_action(action)

        commands.append(command_ids)
        actions.append(action_ids[:-1])  # Input (remove last token)
        modifications.append(np.zeros(8, dtype=np.float32))  # No modification

    # Optionally add some modified examples
    mod_path = data_path / "modification_pairs.pkl"
    if mod_path.exists():
        print("Adding modified examples...")
        with open(mod_path, "rb") as f:
            mod_pairs = pickle.load(f)

        mod_types = ["walk_skip", "jump_hop", "look_scan", "left_right"]

        for item in mod_pairs[:2000]:  # Add some modified examples
            if len(item) == 3:
                base, modified, mod_type = item

                # Handle formats
                if isinstance(base, dict):
                    base_command = base["command"]
                else:
                    base_command = base.command

                if isinstance(modified, dict):
                    modified_action = modified["action"]
                else:
                    modified_action = modified.action

                # Tokenize
                command_ids = tokenizer.encode_command(base_command)
                action_ids = tokenizer.encode_action(modified_action)

                # Create modification vector
                mod_vector = np.zeros(8, dtype=np.float32)
                if mod_type in mod_types:
                    mod_vector[mod_types.index(mod_type)] = 1.0

                commands.append(command_ids)
                actions.append(action_ids[:-1])  # Input
                modifications.append(mod_vector)

    # Convert to numpy arrays
    commands = np.array(commands)
    actions = np.array(actions)
    modifications = np.array(modifications)

    # Create target labels (shifted actions)
    targets = []
    for sample in train_data[:10000]:
        if isinstance(sample, dict):
            action = sample["action"]
        else:
            action = sample.action
        action_ids = tokenizer.encode_action(action)
        targets.append(action_ids[1:])  # Output (remove first token)

    # Add targets for modified examples
    if mod_path.exists():
        for item in mod_pairs[:2000]:
            if len(item) == 3:
                _, modified, _ = item
                if isinstance(modified, dict):
                    modified_action = modified["action"]
                else:
                    modified_action = modified.action
                action_ids = tokenizer.encode_action(modified_action)
                targets.append(action_ids[1:])

    targets = np.array(targets)

    print(f"Prepared {len(commands)} training examples")
    print(f"Commands shape: {commands.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Modifications shape: {modifications.shape}")

    return commands, actions, targets, modifications


def main():
    parser = argparse.ArgumentParser(description="Train simple baseline model V2")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.001)

    args = parser.parse_args()

    print("Simple Baseline V2 Training")
    print("=" * 50)

    # Setup paths
    base_dir = Path(__file__).parent
    data_path = base_dir / "data" / "processed"
    output_dir = (
        base_dir
        / "outputs"
        / f'simple_baseline_v2_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load vocabulary
    trained_model_vocab = (
        base_dir
        / "compositional_language_complete_20250722_185804"
        / "outputs"
        / "safeguarded_training"
        / "vocabulary.json"
    )
    if trained_model_vocab.exists():
        vocab_path = trained_model_vocab
    else:
        vocab_path = data_path / "vocabulary.json"

    with open(vocab_path, "r") as f:
        vocab_data = json.load(f)

    # Initialize tokenizer
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

    command_vocab_size = len(tokenizer.command_to_id)
    action_vocab_size = len(tokenizer.action_to_id)

    print(f"Command vocabulary size: {command_vocab_size}")
    print(f"Action vocabulary size: {action_vocab_size}")

    # Create model
    model = create_simple_model(command_vocab_size, action_vocab_size, args.d_model)

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    # Load and prepare data
    commands, actions, targets, modifications = load_and_prepare_data(
        data_path, tokenizer
    )

    # Train model
    print("\nTraining model...")
    history = model.fit(
        [commands, actions, modifications],
        targets,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=0.1,
        verbose=1,
    )

    # Save model
    model_path = output_dir / "model.h5"
    model.save(str(model_path))
    print(f"\nModel saved to: {model_path}")

    # Save history
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history.history, f, indent=2)

    # Save config
    config = {
        "model_type": "simple_baseline_v2",
        "command_vocab_size": command_vocab_size,
        "action_vocab_size": action_vocab_size,
        "d_model": args.d_model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
    }

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nTraining complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
