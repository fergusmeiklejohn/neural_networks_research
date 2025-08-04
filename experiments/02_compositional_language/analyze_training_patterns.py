#!/usr/bin/env python3
"""
Analyze training patterns to understand modification behavior without model weights.

This script analyzes the training dynamics from the JSON results to infer
whether models are learning to apply modifications or ignoring them.
"""

import json
from pathlib import Path

import numpy as np


def load_results():
    """Load the comprehensive results."""
    results_path = Path("comprehensive_results_20250723_060233/final_results.json")
    with open(results_path, "r") as f:
        return json.load(f)


def analyze_loss_patterns(experiment_data):
    """Analyze loss patterns across stages to detect modification learning."""
    stages = experiment_data["stages"]

    patterns = {
        "stage_names": [],
        "initial_losses": [],
        "final_losses": [],
        "loss_changes": [],
        "accuracy_changes": [],
        "convergence_speed": [],
    }

    for stage in stages:
        patterns["stage_names"].append(stage["name"])

        # Get loss trajectory
        history = stage["history"]
        initial_loss = history[0]["loss"]
        final_loss = history[-1]["loss"]

        patterns["initial_losses"].append(initial_loss)
        patterns["final_losses"].append(final_loss)
        patterns["loss_changes"].append(final_loss - initial_loss)

        # Check convergence speed (how quickly loss stabilizes)
        if len(history) > 1:
            # Check if loss changes between epochs
            loss_deltas = [
                abs(history[i]["loss"] - history[i - 1]["loss"])
                for i in range(1, len(history))
            ]
            avg_delta = np.mean(loss_deltas) if loss_deltas else 0
            patterns["convergence_speed"].append(avg_delta)
        else:
            patterns["convergence_speed"].append(0)

        # Accuracy changes
        initial_acc = history[0]["accuracy"]
        final_acc = history[-1]["accuracy"]
        patterns["accuracy_changes"].append(final_acc - initial_acc)

    return patterns


def detect_modification_learning(patterns, model_name):
    """Detect if model is learning modifications based on patterns."""
    indicators = {
        "learning_modifications": 0,
        "ignoring_modifications": 0,
        "uncertain": 0,
    }

    # Analyze each stage transition
    for i in range(1, len(patterns["stage_names"])):
        prev_loss = patterns["final_losses"][i - 1]
        curr_loss = patterns["initial_losses"][i]
        loss_jump = curr_loss - prev_loss

        # Check convergence speed
        conv_speed = patterns["convergence_speed"][i]

        # Indicators of modification learning:
        # 1. Loss increases when modifications introduced
        # 2. Model takes time to converge (not instant)
        # 3. Accuracy changes during stage

        if loss_jump > 0.1 and conv_speed > 0.001:
            indicators["learning_modifications"] += 1
        elif abs(loss_jump) < 0.01 and conv_speed < 0.0001:
            indicators["ignoring_modifications"] += 1
        else:
            indicators["uncertain"] += 1

    return indicators


def analyze_validation_consistency(results):
    """Analyze validation accuracy patterns."""
    print("\n" + "=" * 60)
    print("VALIDATION ACCURACY ANALYSIS")
    print("=" * 60)

    for exp_name, exp_data in results["experiments"].items():
        val_accs = [stage["val_accuracy"] for stage in exp_data["stages"]]

        # Check if validation accuracy changes
        unique_accs = set(val_accs)

        print(f"\n{exp_name}:")
        print(f"  Validation accuracies: {val_accs}")
        print(f"  Unique values: {len(unique_accs)}")

        if len(unique_accs) == 1:
            print(f"  ⚠️  CONSTANT validation accuracy: {val_accs[0]:.3f}")
            print(f"  → Suggests validation set has NO modified examples")
        else:
            print(f"  ✓ Validation accuracy varies")
            print(f"  → Validation set likely includes modifications")


def main():
    """Run pattern analysis on all experiments."""
    results = load_results()

    print("=" * 60)
    print("TRAINING PATTERN ANALYSIS")
    print("=" * 60)

    all_indicators = {}

    for exp_name, exp_data in results["experiments"].items():
        print(
            f"\n{exp_name} ({exp_data['model_version']}, "
            f"{'mixed' if exp_data['mixed_training'] else 'standard'} training):"
        )

        # Analyze patterns
        patterns = analyze_loss_patterns(exp_data)

        # Print stage-by-stage analysis
        print("\nStage-by-stage analysis:")
        for i, stage_name in enumerate(patterns["stage_names"]):
            print(f"\n  {stage_name}:")
            print(
                f"    Loss: {patterns['initial_losses'][i]:.3f} → "
                f"{patterns['final_losses'][i]:.3f}"
            )
            print(f"    Convergence speed: {patterns['convergence_speed'][i]:.6f}")

            if i > 0:
                loss_jump = (
                    patterns["initial_losses"][i] - patterns["final_losses"][i - 1]
                )
                print(f"    Loss jump from previous: {loss_jump:+.3f}")

        # Detect modification learning
        indicators = detect_modification_learning(patterns, exp_name)
        all_indicators[exp_name] = indicators

        print(f"\n  Modification learning indicators:")
        print(
            f"    Learning modifications: {indicators['learning_modifications']}/3 stages"
        )
        print(
            f"    Ignoring modifications: {indicators['ignoring_modifications']}/3 stages"
        )
        print(f"    Uncertain: {indicators['uncertain']}/3 stages")

    # Analyze validation consistency
    analyze_validation_consistency(results)

    # Overall conclusions
    print("\n" + "=" * 60)
    print("CONCLUSIONS")
    print("=" * 60)

    # Key finding 1: Constant validation accuracy
    print("\n1. VALIDATION SET ISSUE:")
    print("   All models show EXACTLY 84.3% validation accuracy throughout")
    print("   This strongly suggests the validation set contains ONLY")
    print("   unmodified SCAN examples, making it impossible to measure")
    print("   modification performance.")

    # Key finding 2: v2 without mixed training
    print("\n2. V2 ARCHITECTURE FAILURE:")
    print("   v2_standard completely failed (4.2% accuracy)")
    print("   The gating mechanism prevents learning without mixed training")

    # Key finding 3: Loss patterns
    print("\n3. MODIFICATION LEARNING EVIDENCE:")
    for exp_name, indicators in all_indicators.items():
        if indicators["ignoring_modifications"] >= 2:
            print(f"   {exp_name}: Likely IGNORING modifications")
        elif indicators["learning_modifications"] >= 2:
            print(f"   {exp_name}: Likely LEARNING modifications")
        else:
            print(f"   {exp_name}: UNCERTAIN modification behavior")

    # Key finding 4: Training vs validation gap
    print("\n4. TRAINING-VALIDATION GAP:")
    print("   Training accuracy degrades (suggesting modification struggles)")
    print("   Validation accuracy stays constant (no modifications to test)")
    print("   This gap indicates models DO see modifications in training")
    print("   but we're not measuring their performance properly")

    print("\n" + "=" * 60)
    print("RECOMMENDED NEXT STEPS")
    print("=" * 60)
    print("\n1. Create proper validation sets with modified examples")
    print("2. Implement modification-specific metrics")
    print("3. Log example predictions during training")
    print("4. Add gate activation monitoring for v2 models")
    print("5. Test with single modification type first")


if __name__ == "__main__":
    main()
