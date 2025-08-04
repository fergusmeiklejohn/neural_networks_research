#!/usr/bin/env python3
"""Comprehensive test suite for variable rebinding patterns.

Tests the ability of models to handle variable rebinding - when the same
variable is bound to different values at different points in the sequence.
"""

import sys
from typing import Dict, List

import mlx.core as mx
import numpy as np

sys.path.append("../..")

from train_sequential_action_positions import (
    ACTIONS,
    VOCAB,
    SequentialModelWithActionTracking,
)
from train_versioned_memory import VersionedMemoryModel


class RebindingTestSuite:
    """Test suite for evaluating rebinding capabilities."""

    def __init__(self):
        self.test_categories = {
            "basic_rebinding": self.generate_basic_rebinding,
            "temporal_rebinding": self.generate_temporal_rebinding,
            "sequential_rebinding": self.generate_sequential_rebinding,
            "interleaved_rebinding": self.generate_interleaved_rebinding,
            "nested_rebinding": self.generate_nested_rebinding,
        }

    def generate_basic_rebinding(self) -> List[Dict]:
        """Basic rebinding: X changes value once."""
        patterns = [
            {
                "command": "X means jump do X now X means walk do X",
                "expected": ["JUMP", "WALK"],
                "description": "Single variable rebinding with 'now'",
            },
            {
                "command": "Y means turn do Y then Y means run do Y",
                "expected": ["TURN", "RUN"],
                "description": "Single variable rebinding with 'then'",
            },
            {
                "command": "Z is stop recall Z now Z is look recall Z",
                "expected": ["STOP", "LOOK"],
                "description": "Storage pattern rebinding",
            },
        ]
        return patterns

    def generate_temporal_rebinding(self) -> List[Dict]:
        """Rebinding with temporal modifiers."""
        patterns = [
            {
                "command": "X means jump do X twice then X means walk do X",
                "expected": ["JUMP", "JUMP", "WALK"],
                "description": "Rebinding after temporal pattern",
            },
            {
                "command": "Y means turn do Y then Y means run do Y thrice",
                "expected": ["TURN", "RUN", "RUN", "RUN"],
                "description": "Rebinding with temporal on second binding",
            },
            {
                "command": "Z means jump do Z twice then Z means walk do Z twice",
                "expected": ["JUMP", "JUMP", "WALK", "WALK"],
                "description": "Both bindings have temporal modifiers",
            },
        ]
        return patterns

    def generate_sequential_rebinding(self) -> List[Dict]:
        """Multiple rebindings in sequence."""
        patterns = [
            {
                "command": "X means jump do X then X means walk do X then X means turn do X",
                "expected": ["JUMP", "WALK", "TURN"],
                "description": "Triple rebinding",
            },
            {
                "command": "Y means run do Y then Y means stop do Y then Y means look do Y then Y means walk do Y",
                "expected": ["RUN", "STOP", "LOOK", "WALK"],
                "description": "Quadruple rebinding",
            },
        ]
        return patterns

    def generate_interleaved_rebinding(self) -> List[Dict]:
        """Multiple variables with interleaved rebinding."""
        patterns = [
            {
                "command": "X means jump Y means walk do X then do Y then X means turn do X",
                "expected": ["JUMP", "WALK", "TURN"],
                "description": "Rebind X while Y remains constant",
            },
            {
                "command": "X means jump Y means walk do X then X means turn Y means run do Y then do X",
                "expected": ["JUMP", "RUN", "TURN"],
                "description": "Both variables rebound",
            },
            {
                "command": "X means jump do X then Y means walk do Y then X means turn do X then do Y",
                "expected": ["JUMP", "WALK", "TURN", "WALK"],
                "description": "Interleaved usage and rebinding",
            },
        ]
        return patterns

    def generate_nested_rebinding(self) -> List[Dict]:
        """Complex nested patterns with rebinding."""
        patterns = [
            {
                "command": "X means jump do X twice then X means walk do X then do X twice",
                "expected": ["JUMP", "JUMP", "WALK", "WALK", "WALK"],
                "description": "Rebinding affects subsequent temporal patterns",
            },
            {
                "command": "X means jump Y means walk do X then do Y then X means Y do X",
                "expected": ["JUMP", "WALK", "WALK"],
                "description": "Variable bound to another variable's value",
            },
        ]
        return patterns

    def tokenize_command(self, command: str) -> mx.array:
        """Convert command string to token IDs."""
        tokens = command.split()
        token_ids = [VOCAB.get(token, VOCAB["<PAD>"]) for token in tokens]
        return mx.array([token_ids])

    def extract_predictions(
        self, model, command: str, model_type: str = "versioned"
    ) -> List[str]:
        """Extract action predictions from model."""
        command_ids = self.tokenize_command(command)

        if model_type == "versioned":
            outputs = model(command_ids, stage="full_binding")
        else:
            outputs = model(command_ids, stage="full_binding")

        if "action_logits" not in outputs or outputs.get("num_actions", 0) == 0:
            return []

        # Get predictions
        predictions = mx.argmax(outputs["action_logits"], axis=-1)
        predictions_np = np.array(predictions)

        # Convert to action names
        predicted_actions = []
        for j in range(predictions_np.shape[1]):
            pred_id = int(predictions_np[0, j])
            for action_name, action_id in ACTIONS.items():
                if action_id == pred_id and action_name != "<PAD>":
                    predicted_actions.append(action_name)
                    break

        return predicted_actions

    def evaluate_model(
        self, model, model_name: str, model_type: str = "versioned"
    ) -> Dict:
        """Evaluate a model on all rebinding test patterns."""
        results = {
            "model_name": model_name,
            "category_results": {},
            "overall_accuracy": 0,
            "total_tests": 0,
            "correct_tests": 0,
        }

        print(f"\nEvaluating {model_name}...")
        print("=" * 60)

        for category_name, generator_func in self.test_categories.items():
            patterns = generator_func()
            category_correct = 0
            category_total = len(patterns)

            print(f"\n{category_name}:")
            print("-" * 50)

            for pattern in patterns:
                command = pattern["command"]
                expected = pattern["expected"]

                # Get predictions
                predicted = self.extract_predictions(model, command, model_type)

                # Check correctness
                correct = predicted == expected
                if correct:
                    category_correct += 1
                    results["correct_tests"] += 1

                results["total_tests"] += 1

                # Print results
                status = "✓" if correct else "✗"
                print(f"{status} {pattern['description']}")
                if not correct:
                    print(f"  Command: {command}")
                    print(f"  Expected: {expected}")
                    print(f"  Got: {predicted}")

            # Category summary
            category_accuracy = (
                category_correct / category_total if category_total > 0 else 0
            )
            results["category_results"][category_name] = {
                "accuracy": category_accuracy,
                "correct": category_correct,
                "total": category_total,
            }
            print(
                f"\nCategory accuracy: {category_accuracy:.1%} ({category_correct}/{category_total})"
            )

        # Overall summary
        results["overall_accuracy"] = (
            results["correct_tests"] / results["total_tests"]
            if results["total_tests"] > 0
            else 0
        )

        print("\n" + "=" * 60)
        print(
            f"OVERALL ACCURACY: {results['overall_accuracy']:.1%} ({results['correct_tests']}/{results['total_tests']})"
        )
        print("=" * 60)

        return results

    def compare_models(self):
        """Compare versioned memory model with standard model."""
        print("\nVARIABLE REBINDING TEST SUITE")
        print("=" * 80)

        # Test versioned memory model
        print("\n1. Testing Versioned Memory Model")
        versioned_model = VersionedMemoryModel()
        versioned_results = self.evaluate_model(
            versioned_model, "Versioned Memory Model", model_type="versioned"
        )

        # Test standard sequential model
        print("\n\n2. Testing Standard Sequential Model")
        standard_model = SequentialModelWithActionTracking()
        standard_results = self.evaluate_model(
            standard_model, "Standard Sequential Model", model_type="standard"
        )

        # Comparison summary
        print("\n\nCOMPARISON SUMMARY")
        print("=" * 80)
        print(
            f"{'Category':<25} {'Versioned':<15} {'Standard':<15} {'Improvement':<15}"
        )
        print("-" * 70)

        for category in versioned_results["category_results"]:
            v_acc = versioned_results["category_results"][category]["accuracy"]
            s_acc = standard_results["category_results"][category]["accuracy"]
            improvement = v_acc - s_acc

            print(f"{category:<25} {v_acc:>14.1%} {s_acc:>14.1%} {improvement:>+14.1%}")

        print("-" * 70)
        v_overall = versioned_results["overall_accuracy"]
        s_overall = standard_results["overall_accuracy"]
        overall_improvement = v_overall - s_overall

        print(
            f"{'OVERALL':<25} {v_overall:>14.1%} {s_overall:>14.1%} {overall_improvement:>+14.1%}"
        )
        print("=" * 80)

        return versioned_results, standard_results


def test_memory_versions():
    """Test the versioning mechanism directly."""
    print("\nTesting memory versioning mechanism...")

    from train_versioned_memory import VersionedMemory

    memory = VersionedMemory(num_slots=4, embed_dim=16, max_versions=3)

    # Create test embeddings
    var_embed = mx.random.normal((1, 16))
    value1 = mx.ones((1, 16)) * 1.0
    value2 = mx.ones((1, 16)) * 2.0
    value3 = mx.ones((1, 16)) * 3.0

    # Test binding scores (assign to slot 0)
    binding_scores = mx.zeros((1, 4))
    binding_scores = binding_scores.at[:, 0].set(1.0)

    # Test versioning
    memory_state = {}

    # First binding
    memory_state = memory.bind_versioned(
        var_embed, value1, binding_scores, 0, memory_state
    )
    print(
        f"After first binding: {len(memory_state)} slots, slot 0 has {len(memory_state[0]['versions'])} versions"
    )

    # Second binding (rebind)
    memory_state = memory.bind_versioned(
        var_embed, value2, binding_scores, 10, memory_state
    )
    print(
        f"After rebinding: {len(memory_state)} slots, slot 0 has {len(memory_state[0]['versions'])} versions"
    )

    # Third binding
    memory_state = memory.bind_versioned(
        var_embed, value3, binding_scores, 20, memory_state
    )
    print(
        f"After third binding: {len(memory_state)} slots, slot 0 has {len(memory_state[0]['versions'])} versions"
    )

    # Test retrieval
    retrieved = memory.retrieve_versioned(binding_scores, memory_state, 25)
    print(f"Retrieved shape: {retrieved.shape}")
    print(f"Should retrieve version 3 (most recent)")

    print("\n✓ Memory versioning test complete!")


def main():
    """Run all rebinding tests."""
    # Test memory versioning
    test_memory_versions()

    # Run full test suite
    test_suite = RebindingTestSuite()
    test_suite.compare_models()


if __name__ == "__main__":
    main()
