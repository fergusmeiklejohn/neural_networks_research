#!/usr/bin/env python3
"""Final definitive fix for THEN operator."""

from utils.imports import setup_project_paths

setup_project_paths()

import logging
from typing import List, Tuple

import mlx.core as mx
import numpy as np
from progressive_complexity_dataset import ProgressiveComplexityDataset
from two_stage_compiler_v2 import (
    ExecutionSegment,
    ImprovedBindingExtractor,
    TemporalBinding,
    TwoStageCompilerV2,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DefinitiveTHENExtractor(ImprovedBindingExtractor):
    """Definitive fix that properly handles all THEN patterns."""

    def extract(
        self, tokens: mx.array
    ) -> Tuple[List[TemporalBinding], List[ExecutionSegment]]:
        """Extract with definitive THEN handling."""
        token_list = self._to_list(tokens)

        bindings = []
        segments = []
        i = 0

        # First pass: extract all bindings
        while i < len(token_list):
            if i + 2 < len(token_list) and token_list[i + 1] == self.means_token:
                var_token = token_list[i]
                action_token = token_list[i + 2]

                var_name = self._get_var_name(var_token)
                action_name = self.action_map.get(action_token)

                if var_name and action_name:
                    # Update previous binding's scope
                    for prev_binding in bindings:
                        if (
                            prev_binding.variable == var_name
                            and prev_binding.scope_end is None
                        ):
                            prev_binding.scope_end = i

                    # Create new binding
                    binding = TemporalBinding(
                        variable=var_name, action=action_name, scope_start=i
                    )
                    bindings.append(binding)

                i += 3
            else:
                i += 1

        # Second pass: extract execution segments
        i = 0
        while i < len(token_list):
            if token_list[i] == self.do_token:
                # Found a "do", now parse the execution
                exec_start = i
                i += 1  # Skip "do"

                # Collect tokens until "then" or end
                current_segment_start = i

                while i < len(token_list):
                    # Check if we hit "then"
                    if token_list[i] == self.then_token:
                        # Parse segment before "then"
                        if i > current_segment_start:
                            tree = self._parse_segment(
                                token_list, current_segment_start, i
                            )
                            if tree:
                                segments.append(ExecutionSegment(exec_start, i, tree))

                        # Move past "then" and continue parsing
                        i += 1
                        current_segment_start = i

                        # Keep going to parse what's after "then"
                        continue

                    # Check if we hit a new binding (stop execution)
                    elif (
                        i + 1 < len(token_list)
                        and token_list[i + 1] == self.means_token
                    ):
                        # Parse final segment
                        if i > current_segment_start:
                            tree = self._parse_segment(
                                token_list, current_segment_start, i
                            )
                            if tree:
                                segments.append(ExecutionSegment(exec_start, i, tree))
                        break

                    # Check if we hit another "do" (stop execution)
                    elif token_list[i] == self.do_token:
                        # Parse final segment
                        if i > current_segment_start:
                            tree = self._parse_segment(
                                token_list, current_segment_start, i
                            )
                            if tree:
                                segments.append(ExecutionSegment(exec_start, i, tree))
                        break

                    else:
                        i += 1

                # If we reached the end, parse any remaining segment
                if i >= len(token_list) and i > current_segment_start:
                    tree = self._parse_segment(token_list, current_segment_start, i)
                    if tree:
                        segments.append(ExecutionSegment(exec_start, i, tree))
            else:
                i += 1

        return bindings, segments


def test_definitive_fix():
    """Test the definitive THEN fix."""
    print("\n" + "=" * 80)
    print("TESTING DEFINITIVE THEN FIX")
    print("=" * 80 + "\n")

    VOCAB = {
        "PAD": 0,
        "do": 1,
        "means": 2,
        "is": 3,
        "and": 4,
        "or": 5,
        "then": 6,
        "twice": 7,
        "thrice": 8,
        "while": 9,
        "X": 10,
        "Y": 11,
        "Z": 12,
        "W": 13,
        "jump": 14,
        "walk": 15,
        "run": 16,
        "turn": 17,
        "true": 18,
    }

    # Create model with definitive fix
    model = TwoStageCompilerV2(len(VOCAB), 4)
    model.extractor = DefinitiveTHENExtractor(VOCAB)

    # Test problematic cases
    test_cases = [
        # Simple THEN
        ("X means jump Y means walk do X then Y", ["JUMP", "WALK"]),
        ("Y means walk X means turn do X then Y", ["TURN", "WALK"]),
        # THEN with AND
        ("X means jump Y means run do X and Y then X", ["JUMP", "RUN", "JUMP"]),
        # Rebinding
        ("X means jump do X then X means walk do X", ["JUMP", "WALK"]),
        # With modifiers
        ("X means jump do X twice then Y means walk do Y", ["JUMP", "JUMP", "WALK"]),
    ]

    print("Testing specific patterns:\n")

    all_pass = True
    for command, expected in test_cases:
        print(f"Command: {command}")
        print(f"Expected: {expected}")

        tokens = [VOCAB.get(w, 0) for w in command.split()]
        tokens_mx = mx.array([tokens])

        # Analyze
        analysis = model.analyze(tokens_mx)
        print(f"Segments: {len(analysis['segments'])}")
        print(f"Actions: {analysis['actions']}")

        is_correct = analysis["actions"] == expected
        print(f"Result: {'✅ PASS' if is_correct else '❌ FAIL'}")

        if not is_correct:
            all_pass = False
            # Debug
            for i, seg in enumerate(analysis["segments"]):
                print(f"  Segment {i}: {seg}")
        print()

    if not all_pass:
        print("Some tests failed - implementation needs more work")
        return

    # Full evaluation
    print("\n" + "=" * 60)
    print("FULL EVALUATION")
    print("=" * 60 + "\n")

    dataset = ProgressiveComplexityDataset()
    test_data = {
        f"level_{i}": getattr(dataset, f"generate_level_{i}")(100) for i in range(1, 5)
    }

    from train_two_stage_simple import evaluate_model

    results = evaluate_model(model, test_data, VOCAB)

    for level in range(1, 5):
        level_name = f"level_{level}"
        if level_name in results:
            print(f"Level {level}: {results[level_name]:.2%}")

    avg_accuracy = np.mean(list(results.values()))
    print(f"\nAverage: {avg_accuracy:.2%}")

    # THEN-specific analysis
    print("\n" + "=" * 60)
    print("THEN ACCURACY")
    print("=" * 60 + "\n")

    then_correct = 0
    then_total = 0

    for level_data in test_data.values():
        for sample in level_data:
            if (
                " then " in sample["command"]
                and "means" not in sample["command"].split(" then ")[1]
            ):
                then_total += 1

                tokens = mx.array([sample["tokens"]])
                outputs = model(tokens)

                predicted = []
                if outputs.shape[0] > 0:
                    indices = mx.argmax(outputs, axis=-1)
                    for idx in indices:
                        predicted.append(["JUMP", "WALK", "RUN", "TURN"][int(idx)])

                if predicted == sample["expected_actions"]:
                    then_correct += 1

    then_accuracy = then_correct / max(then_total, 1)
    print(f"THEN patterns: {then_correct}/{then_total} = {then_accuracy:.2%}")

    # Compare with baseline
    print("\n" + "=" * 60)
    print("COMPARISON WITH BASELINE")
    print("=" * 60)

    baseline = TwoStageCompilerV2(len(VOCAB), 4)
    baseline.set_vocab(VOCAB)
    baseline_results = evaluate_model(baseline, test_data, VOCAB)
    baseline_avg = np.mean(list(baseline_results.values()))

    print(f"\nBaseline average: {baseline_avg:.2%}")
    print(f"With THEN fix: {avg_accuracy:.2%}")
    print(f"Improvement: +{avg_accuracy - baseline_avg:.2%}")

    if avg_accuracy > 90:
        print("\n✅ SUCCESS! Achieved >90% accuracy!")
        print("The Two-Stage Compiler now properly handles all operators.")
    elif avg_accuracy > 85:
        print("\n⚠️ Good progress! Close to target.")
    else:
        print("\n❌ More work needed to reach >90% target.")


if __name__ == "__main__":
    test_definitive_fix()
