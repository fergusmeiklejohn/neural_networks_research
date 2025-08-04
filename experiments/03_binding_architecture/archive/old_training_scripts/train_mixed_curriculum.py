#!/usr/bin/env python3
"""
Mixed Curriculum Training - Train all stages simultaneously for better stability
"""

import argparse

# Import components
import sys
from pathlib import Path
from typing import Dict, List

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
from train_temporal_curriculum import TemporalDynamicMemoryModel


def generate_mixed_batch(
    batch_size: int = 32, stage_weights: List[float] = [0.2, 0.3, 0.5]
) -> Dict[str, mx.array]:
    """
    Generate a mixed batch with samples from all stages

    Args:
        batch_size: Total batch size
        stage_weights: Probability of each stage [stage1, stage2, stage3]
    """
    assert abs(sum(stage_weights) - 1.0) < 1e-6, "Stage weights must sum to 1"

    # Determine number of samples per stage
    stage_sizes = np.random.multinomial(batch_size, stage_weights)

    all_commands = []
    all_labels = []
    all_stages = []

    # Generate Stage 1 samples (Recognition)
    if stage_sizes[0] > 0:
        stage1 = generate_stage1_data(int(stage_sizes[0]))
        for i in range(stage1["command"].shape[0]):
            cmd = stage1["command"][i].tolist()
            # For stage 1, create a dummy label
            label = [stage1["target"][i].item()]
            all_commands.append(cmd)
            all_labels.append(label)
            all_stages.append("recognition")

    # Generate Stage 2 samples (Retrieval)
    if stage_sizes[1] > 0:
        stage2 = generate_stage2_data(int(stage_sizes[1]))
        for i in range(stage2["command"].shape[0]):
            cmd = stage2["command"][i].tolist()
            label = stage2["labels"][i].tolist()
            all_commands.append(cmd)
            all_labels.append(label)
            all_stages.append("retrieval")

    # Generate Stage 3 samples (Full Binding)
    if stage_sizes[2] > 0:
        stage3 = generate_stage3_data(int(stage_sizes[2]))
        for i in range(stage3["command"].shape[0]):
            cmd = stage3["command"][i].tolist()
            label = stage3["labels"][i].tolist()
            all_commands.append(cmd)
            all_labels.append(label)
            all_stages.append("full")

    # Pad everything to same length
    max_cmd_len = max(len(cmd) for cmd in all_commands)
    max_label_len = max(len(label) for label in all_labels)

    padded_commands = []
    padded_labels = []

    for cmd, label in zip(all_commands, all_labels):
        # Remove existing padding
        cmd = [t for t in cmd if t != VOCAB["<PAD>"]]
        label = [l for l in label if l != ACTIONS["<PAD>"]]

        # Re-pad to consistent length
        padded_cmd = cmd + [VOCAB["<PAD>"]] * (max_cmd_len - len(cmd))
        padded_label = label + [ACTIONS["<PAD>"]] * (max_label_len - len(label))

        padded_commands.append(padded_cmd)
        padded_labels.append(padded_label)

    return {
        "command": mx.array(padded_commands, dtype=mx.int32),
        "labels": mx.array(padded_labels, dtype=mx.int32),
        "stages": all_stages,  # Keep track of which stage each sample is from
    }


def mixed_loss_fn(model, batch):
    """Compute loss for mixed batch with appropriate handling per stage"""

    # We'll compute loss per sample based on its stage
    total_loss = mx.array(0.0)
    valid_count = 0

    # Process in sub-batches by stage for efficiency
    stage_groups = {}
    for i, stage in enumerate(batch["stages"]):
        if stage not in stage_groups:
            stage_groups[stage] = []
        stage_groups[stage].append(i)

    # Process each stage group
    for stage, indices in stage_groups.items():
        if not indices:
            continue

        # Extract samples for this stage
        stage_commands = batch["command"][indices]
        stage_labels = batch["labels"][indices]

        # Get model outputs
        outputs = model(stage_commands, stage=stage, training=True)

        if stage == "recognition":
            # Recognition task - predict variable at query position
            logits = outputs["recognition_logits"]
            mask = stage_commands != VOCAB["<PAD>"]
            last_positions = mx.sum(mask, axis=1) - 1

            batch_indices = mx.arange(len(indices))
            query_logits = logits[batch_indices, last_positions]

            # Target is the first label (variable ID)
            targets = stage_labels[:, 0]
            stage_loss = mx.mean(nn.losses.cross_entropy(query_logits, targets))

        elif stage == "retrieval":
            # Retrieval task - predict action at last position
            logits = outputs["action_logits"]
            mask = stage_commands != VOCAB["<PAD>"]
            last_positions = mx.sum(mask, axis=1) - 1

            batch_indices = mx.arange(len(indices))
            last_logits = logits[batch_indices, last_positions]

            targets = stage_labels[:, 0]  # Single action
            stage_loss = mx.mean(nn.losses.cross_entropy(last_logits, targets))

        else:  # stage == 'full'
            # Full binding - handle temporal patterns
            logits = outputs["action_logits"]
            stage_loss = mx.array(0.0)
            stage_count = 0

            for i in range(len(indices)):
                cmd = stage_commands[i]
                labels = stage_labels[i]

                # Find valid labels
                valid_labels = []
                for label in labels:
                    if label.item() != ACTIONS["<PAD>"]:
                        valid_labels.append(label)
                    else:
                        break

                if not valid_labels:
                    continue

                # Check for temporal patterns
                has_temporal = mx.any(
                    (cmd == VOCAB.get("twice", -1)) | (cmd == VOCAB.get("thrice", -1))
                )

                if has_temporal and outputs.get("temporal_actions", 0) > 0:
                    # Handle temporal predictions
                    num_temporal = outputs["temporal_actions"]
                    start_pos = logits.shape[1] - num_temporal

                    for k, label in enumerate(valid_labels):
                        if start_pos + k < logits.shape[1]:
                            pred_logits = logits[i, start_pos + k]
                            loss_k = nn.losses.cross_entropy(pred_logits, label)
                            stage_loss = stage_loss + loss_k
                            stage_count += 1
                else:
                    # Regular pattern
                    do_pos = -1
                    for j in range(len(cmd)):
                        if cmd[j].item() == VOCAB["do"]:
                            do_pos = j
                            break

                    if do_pos >= 0:
                        pred_idx = 0
                        for j in range(do_pos + 1, min(len(cmd), logits.shape[1])):
                            if cmd[j].item() in [VOCAB["X"], VOCAB["Y"], VOCAB["Z"]]:
                                if pred_idx < len(valid_labels):
                                    pred_logits = logits[i, j]
                                    loss_k = nn.losses.cross_entropy(
                                        pred_logits, valid_labels[pred_idx]
                                    )
                                    stage_loss = stage_loss + loss_k
                                    stage_count += 1
                                    pred_idx += 1

            if stage_count > 0:
                stage_loss = stage_loss / stage_count
            else:
                stage_loss = mx.array(0.0)

        total_loss = total_loss + stage_loss * len(indices)
        valid_count += len(indices)

    return total_loss / max(valid_count, 1)


def evaluate_all_stages(model, num_samples: int = 100):
    """Evaluate model on all stages"""
    model.eval()

    results = {}

    # Stage 1: Recognition
    stage1 = generate_stage1_data(num_samples)
    outputs1 = model(stage1["command"], stage="recognition", training=False)
    logits1 = outputs1["recognition_logits"]
    mask1 = stage1["command"] != VOCAB["<PAD>"]
    last_pos1 = mx.sum(mask1, axis=1) - 1
    batch_idx1 = mx.arange(num_samples)
    preds1 = mx.argmax(logits1[batch_idx1, last_pos1], axis=1)
    acc1 = mx.mean(preds1 == stage1["target"]).item()
    results["recognition"] = acc1

    # Stage 2: Retrieval
    stage2 = generate_stage2_data(num_samples)
    outputs2 = model(stage2["command"], stage="retrieval", training=False)
    logits2 = outputs2["action_logits"]
    mask2 = stage2["command"] != VOCAB["<PAD>"]
    last_pos2 = mx.sum(mask2, axis=1) - 1
    batch_idx2 = mx.arange(num_samples)
    preds2 = mx.argmax(logits2[batch_idx2, last_pos2], axis=1)
    acc2 = mx.mean(preds2 == stage2["labels"][:, 0]).item()
    results["retrieval"] = acc2

    # Stage 3: Full (simplified evaluation)
    stage3 = generate_stage3_data(num_samples)
    outputs3 = model(stage3["command"], stage="full", training=False)
    # For simplicity, just check first action prediction
    correct = 0
    total = 0

    for i in range(num_samples):
        cmd = stage3["command"][i]
        labels = stage3["labels"][i]

        # Find first valid label
        first_label = None
        for label in labels:
            if label.item() != ACTIONS["<PAD>"]:
                first_label = label.item()
                break

        if first_label is None:
            continue

        # Find do position
        do_pos = -1
        for j in range(len(cmd)):
            if cmd[j].item() == VOCAB["do"]:
                do_pos = j
                break

        if do_pos >= 0 and do_pos + 1 < len(cmd):
            # Check prediction at first variable after 'do'
            for j in range(do_pos + 1, len(cmd)):
                if cmd[j].item() in [VOCAB["X"], VOCAB["Y"], VOCAB["Z"]]:
                    if j < outputs3["action_logits"].shape[1]:
                        pred = mx.argmax(outputs3["action_logits"][i, j]).item()
                        if pred == first_label:
                            correct += 1
                        total += 1
                        break

    acc3 = correct / total if total > 0 else 0.0
    results["full"] = acc3

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--num_slots", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--stage_weights",
        type=float,
        nargs=3,
        default=[0.2, 0.3, 0.5],
        help="Sampling weights for [stage1, stage2, stage3]",
    )
    parser.add_argument("--output_dir", type=str, default="outputs/mixed_curriculum")
    args = parser.parse_args()

    # Normalize stage weights
    args.stage_weights = [w / sum(args.stage_weights) for w in args.stage_weights]

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model
    model = TemporalDynamicMemoryModel(
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

    # Training loop
    print("Mixed Curriculum Training")
    print("=" * 60)
    print(f"Stage weights: {args.stage_weights}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("=" * 60)

    best_avg_acc = 0.0

    for epoch in range(args.epochs):
        # Training
        model.train()
        total_loss = 0
        num_batches = 20

        for _ in range(num_batches):
            batch = generate_mixed_batch(args.batch_size, args.stage_weights)

            # Compute loss and gradients
            loss_and_grad_fn = mx.value_and_grad(lambda m: mixed_loss_fn(m, batch))
            loss, grads = loss_and_grad_fn(model)

            # Update
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            total_loss += loss.item()

        avg_loss = total_loss / num_batches

        # Evaluation every 5 epochs
        if (epoch + 1) % 5 == 0:
            results = evaluate_all_stages(model, num_samples=50)
            avg_acc = np.mean(list(results.values()))

            print(f"Epoch {epoch+1}/{args.epochs}: Loss={avg_loss:.4f}")
            print(f"  Recognition: {results['recognition']:.1%}")
            print(f"  Retrieval: {results['retrieval']:.1%}")
            print(f"  Full: {results['full']:.1%}")
            print(f"  Average: {avg_acc:.1%}")

            if avg_acc > best_avg_acc:
                best_avg_acc = avg_acc
                # Save checkpoint
                checkpoint_path = output_dir / "best_mixed_model.npz"
                flat_params = {}
                for name, param in model.parameters().items():
                    flat_params[name] = np.array(param)
                np.savez(checkpoint_path, **flat_params)
                print(f"  ✓ New best model saved!")
        else:
            print(f"Epoch {epoch+1}/{args.epochs}: Loss={avg_loss:.4f}")

    print("\n" + "=" * 60)
    print(f"Training complete! Best average accuracy: {best_avg_acc:.1%}")

    # Test modification capability
    print("\nTesting Modification Capability...")
    from train_temporal_curriculum import test_modification_capability

    mod_success = test_modification_capability(model)
    print(f"Modification Success Rate: {mod_success:.1%}")


if __name__ == "__main__":
    main()
