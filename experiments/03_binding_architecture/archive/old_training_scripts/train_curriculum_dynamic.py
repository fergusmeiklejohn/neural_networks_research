#!/usr/bin/env python3
"""
Curriculum Learning with Dynamic Memory Architecture

Combines the 3-stage curriculum with dynamic memory storage
to achieve true variable binding.
"""

import argparse

# Import data generation and other utilities from existing files
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

sys.path.append(".")
from train_binding_curriculum import (
    ACTIONS,
    VOCAB,
    generate_stage1_data,
    generate_stage2_data,
    generate_stage3_data,
)
from train_binding_dynamic import DynamicMemoryModel


def train_step(model, batch, stage, optimizer):
    """Single training step for the appropriate stage"""

    def loss_fn(model):
        outputs = model(batch["command"], stage=stage, training=True)

        if stage == "recognition":
            # Cross-entropy loss for variable recognition
            logits = outputs["recognition_logits"]
            # Only compute loss on the query position (last non-pad token)
            mask = batch["command"] != VOCAB["<PAD>"]
            last_positions = mx.sum(mask, axis=1) - 1

            # Gather logits at query positions
            batch_indices = mx.arange(logits.shape[0])
            query_logits = logits[batch_indices, last_positions]

            loss = mx.mean(nn.losses.cross_entropy(query_logits, batch["target"]))
        else:
            # Action prediction loss
            logits = outputs["action_logits"]
            labels = batch["labels"]

            # For stage 2, we only care about the last token prediction
            if stage == "retrieval":
                # Find last non-pad position for each sequence
                mask = batch["command"] != VOCAB["<PAD>"]
                last_positions = mx.sum(mask, axis=1) - 1
                batch_indices = mx.arange(logits.shape[0])

                # Get logits at last positions
                last_logits = logits[batch_indices, last_positions]
                # Labels are already single actions for stage 2
                loss = mx.mean(nn.losses.cross_entropy(last_logits, labels.squeeze()))
            else:
                # For stage 3, we need to handle variable-length outputs
                # The model outputs predictions at every position, but we only care about
                # positions after "do" where variables appear

                total_loss = mx.array(0.0)
                total_count = 0

                for i in range(logits.shape[0]):  # For each example in batch
                    # Find where "do" appears
                    cmd = batch["command"][i]
                    do_pos = -1
                    for j in range(len(cmd)):
                        if cmd[j].item() == VOCAB["do"]:
                            do_pos = j
                            break

                    if do_pos >= 0:
                        # Get labels for this example
                        example_labels = labels[i]
                        valid_labels = []
                        for l in example_labels:
                            if l.item() != ACTIONS["<PAD>"]:
                                valid_labels.append(l.item())

                        # Match predictions to labels
                        # Starting from position after "do", look for predictions
                        pred_idx = 0
                        for j in range(do_pos + 1, logits.shape[1]):
                            if pred_idx < len(valid_labels):
                                # This position should predict an action
                                pred_logits = logits[i, j]
                                target = valid_labels[pred_idx]
                                loss_j = nn.losses.cross_entropy(
                                    pred_logits[None, :], mx.array([target])
                                )
                                total_loss = total_loss + loss_j.squeeze()
                                total_count += 1
                                pred_idx += 1

                if total_count > 0:
                    loss = total_loss / total_count
                else:
                    loss = mx.array(0.0)

        return loss

    loss_and_grad_fn = mx.value_and_grad(loss_fn)
    loss, grads = loss_and_grad_fn(model)
    optimizer.update(model, grads)
    mx.eval(loss)

    return loss.item()


def evaluate_stage(model, stage: str, num_eval: int = 100) -> float:
    """Evaluate performance on a specific stage"""
    model.eval()
    correct = 0
    total = 0

    # Generate evaluation data
    if stage == "recognition":
        data_fn = generate_stage1_data
    elif stage == "retrieval":
        data_fn = generate_stage2_data
    else:  # full
        data_fn = generate_stage3_data

    for _ in range(num_eval // 32):
        batch = data_fn(32)
        outputs = model(batch["command"], stage=stage, training=False)

        if stage == "recognition":
            # Check if predicted variable matches target
            logits = outputs["recognition_logits"]
            mask = batch["command"] != VOCAB["<PAD>"]
            last_positions = mx.sum(mask, axis=1) - 1
            batch_indices = mx.arange(logits.shape[0])
            query_logits = logits[batch_indices, last_positions]
            predictions = mx.argmax(query_logits, axis=-1)
            correct += mx.sum(predictions == batch["target"]).item()
            total += batch["target"].shape[0]
        else:
            # Check action predictions
            logits = outputs["action_logits"]

            if stage == "retrieval":
                # For stage 2, check last position only
                mask = batch["command"] != VOCAB["<PAD>"]
                last_positions = mx.sum(mask, axis=1) - 1
                batch_indices = mx.arange(logits.shape[0])
                last_logits = logits[batch_indices, last_positions]
                predictions = mx.argmax(last_logits, axis=-1)
                labels = batch["labels"].squeeze()
                matches = predictions == labels
                correct += mx.sum(matches).item()
                total += labels.shape[0]
            else:
                # For stage 3, we need to match predictions to labels more carefully
                for i in range(logits.shape[0]):
                    # Find where "do" appears
                    cmd = batch["command"][i]
                    do_pos = -1
                    for j in range(len(cmd)):
                        if cmd[j].item() == VOCAB["do"]:
                            do_pos = j
                            break

                    if do_pos >= 0:
                        # Get labels for this example
                        example_labels = batch["labels"][i]
                        valid_labels = []
                        for l in example_labels:
                            if l.item() != ACTIONS["<PAD>"]:
                                valid_labels.append(l.item())

                        # Get predictions starting from position after "do"
                        example_correct = True
                        pred_idx = 0
                        for j in range(do_pos + 1, logits.shape[1]):
                            if pred_idx < len(valid_labels):
                                pred = mx.argmax(logits[i, j]).item()
                                if pred != valid_labels[pred_idx]:
                                    example_correct = False
                                    break
                                pred_idx += 1

                        # Check if we predicted the right number of actions
                        if pred_idx == len(valid_labels) and example_correct:
                            correct += 1

                    total += 1

    return correct / total if total > 0 else 0.0


def test_modification_capability(model):
    """Test if the model can handle variable substitutions"""
    model.eval()

    test_cases = [
        {
            "original": "X means jump do X",
            "modified": "X means walk do X",
            "expected_orig": ["JUMP"],
            "expected_mod": ["WALK"],
        },
        {
            "original": "Y means turn do Y twice",
            "modified": "Y means run do Y twice",
            "expected_orig": ["TURN", "TURN"],
            "expected_mod": ["RUN", "RUN"],
        },
        {
            "original": "Z means look do Z",
            "modified": "Z means jump do Z",
            "expected_orig": ["LOOK"],
            "expected_mod": ["JUMP"],
        },
    ]

    successes = 0

    for i, test in enumerate(test_cases):
        print(f"\nTest {i+1}:")
        print(f"  Original: {test['original']}")

        # Test original
        orig_ids = [
            VOCAB.get(token, VOCAB["<PAD>"]) for token in test["original"].split()
        ]
        orig_batch = mx.array([orig_ids], dtype=mx.int32)
        orig_outputs = model(orig_batch, stage="full", training=False)
        orig_logits = orig_outputs["action_logits"]

        # Get predictions for non-pad positions
        orig_preds = []
        for j in range(len(test["expected_orig"])):
            if j < orig_logits.shape[1]:
                pred_id = mx.argmax(orig_logits[0, j]).item()
                action = [k for k, v in ACTIONS.items() if v == pred_id][0]
                orig_preds.append(action)

        print(f"  Expected: {test['expected_orig']}")
        print(f"  Predicted: {orig_preds}")

        # Test modified
        print(f"  Modified: {test['modified']}")
        mod_ids = [
            VOCAB.get(token, VOCAB["<PAD>"]) for token in test["modified"].split()
        ]
        mod_batch = mx.array([mod_ids], dtype=mx.int32)
        mod_outputs = model(mod_batch, stage="full", training=False)
        mod_logits = mod_outputs["action_logits"]

        # Get predictions for non-pad positions
        mod_preds = []
        for j in range(len(test["expected_mod"])):
            if j < mod_logits.shape[1]:
                pred_id = mx.argmax(mod_logits[0, j]).item()
                action = [k for k, v in ACTIONS.items() if v == pred_id][0]
                mod_preds.append(action)

        print(f"  Expected: {test['expected_mod']}")
        print(f"  Predicted: {mod_preds}")

        # Check success
        if orig_preds == test["expected_orig"] and mod_preds == test["expected_mod"]:
            print("  Success: ✓")
            successes += 1
        else:
            print("  Success: ✗")

    print(
        f"\nModification Success Rate: {successes}/{len(test_cases)} = {successes/len(test_cases)*100:.1f}%"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Curriculum learning with dynamic memory"
    )
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_slots", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs_per_stage", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--stage_threshold",
        type=float,
        default=0.8,
        help="Accuracy threshold to progress to next stage",
    )
    parser.add_argument("--output_dir", type=str, default="outputs/curriculum_dynamic")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model with dynamic memory
    model = DynamicMemoryModel(
        vocab_size=len(VOCAB),
        num_actions=len(ACTIONS),
        embed_dim=args.embed_dim,
        num_slots=args.num_slots,
        num_heads=args.num_heads,
        initial_temperature=1.0,
    )
    mx.eval(model.parameters())

    # Initialize optimizer
    optimizer = optim.Adam(learning_rate=args.lr)

    # Temperature annealing parameters
    initial_temperature = 1.0
    min_temperature = 0.1
    temperature_decay = 0.95

    # Training stages
    stages = ["recognition", "retrieval", "full"]

    for stage_idx, stage in enumerate(stages):
        print(f"\n{'='*60}")
        print(f"Starting Stage {stage_idx + 1}: {stage.upper()}")
        print(f"{'='*60}")

        # Data generator for this stage
        if stage == "recognition":
            data_fn = generate_stage1_data
        elif stage == "retrieval":
            data_fn = generate_stage2_data
        else:
            data_fn = generate_stage3_data

        best_accuracy = 0.0
        epochs_without_improvement = 0

        for epoch in range(args.epochs_per_stage):
            # Update temperature
            current_temp = max(
                min_temperature,
                initial_temperature
                * (temperature_decay ** (stage_idx * args.epochs_per_stage + epoch)),
            )
            model.binder.temperature = current_temp

            # Training
            epoch_loss = 0.0
            num_batches = 100  # Fixed number of batches per epoch

            start_time = time.time()
            for _ in range(num_batches):
                batch = data_fn(args.batch_size)
                loss = train_step(model, batch, stage, optimizer)
                epoch_loss += loss

            avg_loss = epoch_loss / num_batches

            # Evaluation
            accuracy = evaluate_stage(model, stage)

            print(
                f"Epoch {epoch+1}/{args.epochs_per_stage} - "
                f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2%}, "
                f"Temperature: {current_temp:.3f}, Time: {time.time()-start_time:.2f}s"
            )

            # Check for improvement
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                epochs_without_improvement = 0

                # Save checkpoint
                checkpoint_path = output_dir / f"{stage}_best_model.npz"
                # Convert parameters to flat arrays for saving
                flat_params = {}

                def flatten_params(params, prefix=""):
                    for k, v in params.items():
                        key = f"{prefix}.{k}" if prefix else k
                        if isinstance(v, dict):
                            flatten_params(v, key)
                        else:
                            flat_params[key] = v

                flatten_params(dict(model.parameters()))
                np.savez(
                    checkpoint_path, **{k: np.array(v) for k, v in flat_params.items()}
                )
                print(f"  Saved best model with accuracy {accuracy:.2%}")
            else:
                epochs_without_improvement += 1

            # Check if we should move to next stage
            if accuracy >= args.stage_threshold:
                print(f"\n✓ Stage {stage} completed! Accuracy: {accuracy:.2%}")
                break

            # Early stopping within stage
            if epochs_without_improvement >= 5:
                print(f"\n! No improvement for 5 epochs. Moving to next stage.")
                break

        print(f"\nStage {stage} final accuracy: {best_accuracy:.2%}")

    # Final evaluation on all stages
    print(f"\n{'='*60}")
    print("FINAL EVALUATION")
    print(f"{'='*60}")

    for stage in stages:
        accuracy = evaluate_stage(model, stage, num_eval=200)
        print(f"{stage.capitalize()} accuracy: {accuracy:.2%}")

    # Test modification capability on full task
    print(f"\n{'='*60}")
    print("MODIFICATION TEST")
    print(f"{'='*60}")

    test_modification_capability(model)

    # Save final model
    final_path = output_dir / "curriculum_dynamic_final.npz"
    flat_params = {}

    def flatten_params(params, prefix=""):
        for k, v in params.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flatten_params(v, key)
            else:
                flat_params[key] = v

    flatten_params(dict(model.parameters()))
    np.savez(final_path, **{k: np.array(v) for k, v in flat_params.items()})
    print(f"\nFinal model saved to {final_path}")


if __name__ == "__main__":
    main()
