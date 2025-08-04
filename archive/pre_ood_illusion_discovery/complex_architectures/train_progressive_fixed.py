#!/usr/bin/env python3
"""
Fixed progressive training without mixed precision and proper model handling
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import gc
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict

import tensorflow as tf
import wandb
from tensorflow import keras
from tqdm import tqdm

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"GPU memory growth setting failed: {e}")

from scan_data_loader import SCANDataLoader
from train_progressive_curriculum import (
    ProgressiveCurriculum,
    SCANTokenizer,
    compute_accuracy,
    create_dataset,
)

# Import from current directory
from models import create_model


def train_progressive_curriculum_fixed(config: Dict):
    """Fixed training with proper model handling and no mixed precision"""

    # Setup
    print("Setting up fixed progressive curriculum training...")
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb if enabled
    if config.get("use_wandb", False):
        wandb.init(
            project=config.get("wandb_project", "compositional-language"),
            name=f"fixed_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=config,
        )

    # Load data
    print("\nLoading data...")
    loader = SCANDataLoader(data_dir="data")
    splits = loader.load_processed_splits()

    # Load modifications
    mod_path = Path("data/processed/modification_pairs.pkl")
    if mod_path.exists():
        with open(mod_path, "rb") as f:
            modifications = pickle.load(f)
    else:
        modifications = []

    # Create tokenizer
    print("Building vocabulary...")
    tokenizer = SCANTokenizer()
    tokenizer.build_vocabulary(splits["train"])
    tokenizer.save_vocabulary(output_dir / "vocabulary.json")

    print(f"Command vocabulary size: {len(tokenizer.command_to_id)}")
    print(f"Action vocabulary size: {len(tokenizer.action_to_id)}")

    # Create model ONCE
    print("\nCreating model...")
    model = create_model(
        command_vocab_size=len(tokenizer.command_to_id),
        action_vocab_size=len(tokenizer.action_to_id),
        d_model=config.get("d_model", 128),
    )

    # Build model
    print("Building model...")
    dummy_command = tf.zeros((1, 50), dtype=tf.int32)
    dummy_target = tf.zeros((1, 99), dtype=tf.int32)
    dummy_modification = tf.zeros((1, 20), dtype=tf.int32)
    _ = model(
        {
            "command": dummy_command,
            "target": dummy_target,
            "modification": dummy_modification,
        },
        training=False,
    )
    print("Model built successfully")

    # Count parameters
    total_params = sum([tf.size(v).numpy() for v in model.trainable_variables])
    print(f"Total trainable parameters: {total_params:,}")

    # Initialize curriculum
    curriculum = ProgressiveCurriculum(config)

    # Loss function
    loss_fn = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )

    # Training loop
    for stage in range(1, 5):
        stage_config = curriculum.get_stage_config(stage)
        print(f"\n{'='*60}")
        print(f"{stage_config['name']}")
        print(f"{stage_config['description']}")
        print(f"{'='*60}")

        # Garbage collect but DON'T clear TF session
        gc.collect()

        # Create stage-specific dataset
        if stage_config.get("use_modifications", False):
            dataset = create_dataset(
                splits["train"],
                tokenizer,
                modifications=modifications,
                modification_ratio=stage_config.get("modification_ratio", 0.0),
                batch_size=config["batch_size"],
            )
        else:
            dataset = create_dataset(
                splits["train"], tokenizer, batch_size=config["batch_size"]
            )

        # Create validation dataset
        val_dataset = create_dataset(
            splits["val_interpolation"], tokenizer, batch_size=config["batch_size"]
        )

        # Setup optimizer for this stage
        optimizer = keras.optimizers.Adam(learning_rate=stage_config["lr"])

        # Training epochs
        best_val_acc = 0
        for epoch in range(stage_config["epochs"]):
            print(f"\nEpoch {epoch + 1}/{stage_config['epochs']}")

            # Training metrics
            train_loss = 0
            num_batches = 0

            # Progress bar
            pbar = tqdm(dataset, desc="Training")

            for i, batch in enumerate(pbar):
                with tf.GradientTape() as tape:
                    # Prepare inputs
                    inputs = {
                        "command": batch["command"],
                        "target": batch["action"][:, :-1],
                        "modification": batch["modification"],
                    }

                    # Forward pass
                    outputs = model(inputs, training=True)

                    # Compute loss
                    logits = outputs["logits"]
                    targets = batch["action"][:, 1:]

                    # Mask padding
                    mask = tf.cast(
                        targets != tokenizer.action_to_id["<PAD>"], tf.float32
                    )
                    loss_per_token = loss_fn(targets, logits)
                    masked_loss = loss_per_token * mask
                    loss = tf.reduce_sum(masked_loss) / tf.reduce_sum(mask)

                # Compute gradients
                gradients = tape.gradient(loss, model.trainable_variables)

                # Clip gradients for stability
                gradients = [
                    tf.clip_by_norm(g, 1.0) if g is not None else None
                    for g in gradients
                ]

                # Filter None gradients
                valid_grads_and_vars = [
                    (g, v)
                    for g, v in zip(gradients, model.trainable_variables)
                    if g is not None
                ]

                # Apply gradients
                optimizer.apply_gradients(valid_grads_and_vars)

                # Update metrics
                train_loss += loss.numpy()
                num_batches += 1

                # Update progress bar
                pbar.set_postfix({"loss": f"{train_loss/num_batches:.4f}"})

                # Periodic garbage collection (not TF session clear)
                if i > 0 and i % 500 == 0:
                    gc.collect()

            # Validation at end of epoch
            print("Validating...")
            val_acc = compute_accuracy(model, val_dataset, tokenizer, max_samples=200)
            avg_train_loss = train_loss / num_batches

            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Accuracy: {val_acc:.2%}")

            # Track best validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # Save best model for this stage
                best_checkpoint_path = output_dir / f"stage_{stage}_best_model.h5"
                model.save_weights(str(best_checkpoint_path))
                print(f"  New best! Saved to {best_checkpoint_path}")

            if config.get("use_wandb", False):
                wandb.log(
                    {
                        f"stage_{stage}/train_loss": avg_train_loss,
                        f"stage_{stage}/val_accuracy": val_acc,
                        f"stage_{stage}/best_val_accuracy": best_val_acc,
                        "epoch": sum(
                            [
                                curriculum.get_stage_config(s)["epochs"]
                                for s in range(1, stage)
                            ]
                        )
                        + epoch,
                    }
                )

        # Save stage checkpoint
        checkpoint_path = output_dir / f"stage_{stage}_final_model.h5"
        model.save_weights(str(checkpoint_path))
        print(f"Stage {stage} complete. Best validation accuracy: {best_val_acc:.2%}")

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    # Test on different splits
    test_results = {}

    for split_name in ["test_interpolation", "test_primitive_extrap"]:
        if split_name in splits:
            print(f"\nEvaluating {split_name}...")
            test_dataset = create_dataset(
                splits[split_name], tokenizer, batch_size=config["batch_size"]
            )
            accuracy = compute_accuracy(model, test_dataset, tokenizer, max_samples=500)
            test_results[split_name] = accuracy
            print(f"{split_name}: {accuracy:.2%}")

    # Save results
    results_path = output_dir / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "config": config,
                "test_results": test_results,
                "vocabulary_sizes": {
                    "command": len(tokenizer.command_to_id),
                    "action": len(tokenizer.action_to_id),
                },
                "total_parameters": int(total_params),
                "optimizations": {
                    "mixed_precision": False,
                    "memory_growth": True,
                    "gradient_clipping": True,
                    "tf_function": False,
                    "periodic_gc": True,
                },
            },
            f,
            indent=2,
        )

    if config.get("use_wandb", False):
        # Log final results
        for split_name, accuracy in test_results.items():
            wandb.log({f"final/{split_name}": accuracy})
        wandb.finish()

    print(f"\nTraining complete! Results saved to {output_dir}")

    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model size: {total_params:,} parameters")
    print(f"Best validation accuracy during training: {best_val_acc:.2%}")
    for split_name, accuracy in test_results.items():
        print(f"{split_name}: {accuracy:.2%}")


def main():
    """Run fixed training"""

    config = {
        # Model parameters
        "d_model": 128,
        "batch_size": 8,
        # Training epochs
        "stage1_epochs": 10,
        "stage2_epochs": 10,
        "stage3_epochs": 10,
        "stage4_epochs": 10,
        # Learning rates
        "stage1_lr": 1e-3,
        "stage2_lr": 5e-4,
        "stage3_lr": 2e-4,
        "stage4_lr": 1e-4,
        # Output
        "output_dir": "outputs/fixed_training",
        "use_wandb": True,
        "wandb_project": "compositional-language-fixed",
    }

    train_progressive_curriculum_fixed(config)


if __name__ == "__main__":
    main()
