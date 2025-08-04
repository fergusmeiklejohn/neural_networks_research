#!/usr/bin/env python3
"""
Full training script for simple baseline with all data and better parameters.
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
from simple_baseline_v2 import create_simple_model
from tensorflow import keras

# Import from current directory
from train_progressive_curriculum import SCANTokenizer


def load_full_dataset(data_path: Path, tokenizer: SCANTokenizer):
    """Load complete dataset with all examples."""

    # Load base training data
    with open(data_path / "train.pkl", "rb") as f:
        train_data = pickle.load(f)

    print(f"Loaded {len(train_data)} base training examples")

    # Convert to numpy arrays
    commands = []
    actions = []
    targets = []
    modifications = []

    print("Preparing base training data...")
    for i, sample in enumerate(train_data):
        if i % 5000 == 0:
            print(f"  Processed {i}/{len(train_data)} examples...")

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
        targets.append(action_ids[1:])  # Output (remove first token)
        modifications.append(np.zeros(8, dtype=np.float32))  # No modification

    # Add modified examples
    mod_path = data_path / "modification_pairs.pkl"
    if mod_path.exists():
        print("\nAdding modified examples...")
        with open(mod_path, "rb") as f:
            mod_pairs = pickle.load(f)

        print(f"Loaded {len(mod_pairs)} modification pairs")

        mod_types = ["walk_skip", "jump_hop", "look_scan", "left_right"]

        # Use a good portion of modifications
        num_mods = min(len(mod_pairs), len(train_data) // 3)  # 33% modifications

        for i, item in enumerate(mod_pairs[:num_mods]):
            if i % 5000 == 0:
                print(f"  Processed {i}/{num_mods} modification examples...")

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
                targets.append(action_ids[1:])  # Output
                modifications.append(mod_vector)

    # Convert to numpy arrays
    commands = np.array(commands)
    actions = np.array(actions)
    targets = np.array(targets)
    modifications = np.array(modifications)

    print(f"\nTotal training examples: {len(commands)}")
    print(f"  Base examples: {len(train_data)}")
    print(f"  Modified examples: {len(commands) - len(train_data)}")
    print(f"Commands shape: {commands.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Modifications shape: {modifications.shape}")

    return commands, actions, targets, modifications


def main():
    parser = argparse.ArgumentParser(
        description="Train simple baseline with full dataset"
    )
    parser.add_argument(
        "--d_model", type=int, default=256, help="Model dimension (default: 256)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size (default: 64)"
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of epochs (default: 30)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0005,
        help="Learning rate (default: 0.0005)",
    )
    parser.add_argument(
        "--patience", type=int, default=5, help="Early stopping patience (default: 5)"
    )

    args = parser.parse_args()

    print("Simple Baseline Full Training")
    print("=" * 50)
    print(f"Configuration:")
    print(f"  d_model: {args.d_model}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  epochs: {args.epochs}")
    print(f"  learning_rate: {args.learning_rate}")
    print(f"  patience: {args.patience}")
    print()

    # Setup paths
    base_dir = Path(__file__).parent
    data_path = base_dir / "data" / "processed"
    output_dir = (
        base_dir
        / "outputs"
        / f'simple_baseline_full_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
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

    # Compile with a lower learning rate and gradient clipping
    optimizer = keras.optimizers.Adam(
        learning_rate=args.learning_rate, clipnorm=1.0  # Gradient clipping
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    # Load full dataset
    commands, actions, targets, modifications = load_full_dataset(data_path, tokenizer)

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            str(output_dir / "best_model.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # Train model
    print("\nTraining model...")
    history = model.fit(
        [commands, actions, modifications],
        targets,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )

    # Save final model
    model_path = output_dir / "model.h5"
    model.save(str(model_path))
    print(f"\nModel saved to: {model_path}")

    # Save history
    history_path = output_dir / "training_history.json"
    history_dict = {k: [float(v) for v in history.history[k]] for k in history.history}
    with open(history_path, "w") as f:
        json.dump(history_dict, f, indent=2)

    # Save config
    config = {
        "model_type": "simple_baseline_full",
        "command_vocab_size": command_vocab_size,
        "action_vocab_size": action_vocab_size,
        "d_model": args.d_model,
        "epochs": len(history.history["loss"]),  # Actual epochs trained
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "total_examples": len(commands),
        "base_examples": len(commands) - np.sum(modifications.sum(axis=1) > 0),
        "modified_examples": int(np.sum(modifications.sum(axis=1) > 0)),
    }

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nTraining complete! Results saved to: {output_dir}")
    print(f"\nFinal metrics:")
    print(f"  Training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"  Validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"  Training loss: {history.history['loss'][-1]:.4f}")
    print(f"  Validation loss: {history.history['val_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
