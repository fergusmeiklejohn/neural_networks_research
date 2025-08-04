#!/usr/bin/env python3
"""Final fix for THEN operator - properly parse "do X then Y" into two segments."""

from utils.imports import setup_project_paths

setup_project_paths()

import logging
from typing import List, Optional, Tuple

import mlx.core as mx
import numpy as np
from progressive_complexity_dataset import ProgressiveComplexityDataset
from two_stage_compiler_v2 import (
    ExecutionSegment,
    ImprovedBindingExtractor,
    ParseNode,
    TemporalBinding,
    TwoStageCompilerV2,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinalBindingExtractor(ImprovedBindingExtractor):
    """Final extractor that properly handles 'do X then Y' patterns."""

    def _parse_execution_with_then(
        self, tokens: List[int], start: int
    ) -> List[Tuple[int, int, Optional[ParseNode]]]:
        """Parse execution handling THEN as segment separator."""
        segments = []
        current_start = start
        i = start

        while i < len(tokens):
            # Look for THEN as segment boundary
            if tokens[i] == self.then_token:
                # Parse segment before THEN
                if i > current_start:
                    tree = self._parse_segment(tokens, current_start, i)
                    if tree:
                        segments.append((current_start - 1, i, tree))
                # Start new segment after THEN
                current_start = i + 1
                i += 1
            # Stop at next binding or do
            elif (i + 1 < len(tokens) and tokens[i + 1] == self.means_token) or tokens[
                i
            ] == self.do_token:
                # Parse final segment
                if i > current_start:
                    tree = self._parse_segment(tokens, current_start, i)
                    if tree:
                        segments.append((current_start - 1, i, tree))
                break
            else:
                i += 1

        # Parse any remaining segment
        if i >= len(tokens) and i > current_start:
            tree = self._parse_segment(tokens, current_start, i)
            if tree:
                segments.append((current_start - 1, i, tree))

        return segments

    def extract(
        self, tokens: mx.array
    ) -> Tuple[List[TemporalBinding], List[ExecutionSegment]]:
        """Extract with proper THEN handling."""
        token_list = self._to_list(tokens)

        bindings = []
        segments = []
        i = 0

        while i < len(token_list):
            # Check for binding pattern
            if i + 2 < len(token_list) and token_list[i + 1] == self.means_token:
                var_token = token_list[i]
                action_token = token_list[i + 2]

                var_name = self._get_var_name(var_token)
                action_name = self.action_map.get(action_token)

                if var_name and action_name:
                    # Create temporal binding
                    binding = TemporalBinding(
                        variable=var_name, action=action_name, scope_start=i
                    )

                    # Update previous binding's scope
                    for prev_binding in bindings:
                        if (
                            prev_binding.variable == var_name
                            and prev_binding.scope_end is None
                        ):
                            prev_binding.scope_end = i

                    bindings.append(binding)
                    i += 3

                    # Check for immediate "do" after binding
                    if i < len(token_list) and token_list[i] == self.do_token:
                        # Parse execution with THEN handling
                        exec_segments = self._parse_execution_with_then(
                            token_list, i + 1
                        )
                        for seg_start, seg_end, seg_tree in exec_segments:
                            segments.append(
                                ExecutionSegment(seg_start, seg_end, seg_tree)
                            )
                        # Jump to end of execution
                        if exec_segments:
                            i = exec_segments[-1][1]
                        else:
                            i += 1
                else:
                    i += 1

            elif token_list[i] == self.do_token:
                # Standalone "do" segment
                exec_segments = self._parse_execution_with_then(token_list, i + 1)
                for seg_start, seg_end, seg_tree in exec_segments:
                    segments.append(ExecutionSegment(i, seg_end, seg_tree))
                # Jump to end
                if exec_segments:
                    i = exec_segments[-1][1]
                else:
                    i += 1
            else:
                i += 1

        return bindings, segments


def test_final_fix():
    """Test the final THEN fix."""
    print("\n" + "=" * 80)
    print("TESTING FINAL THEN FIX")
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

    # Create model with final fix
    model = TwoStageCompilerV2(len(VOCAB), 4)
    model.extractor = FinalBindingExtractor(VOCAB)

    # Test specific THEN patterns
    test_cases = [
        ("X means jump Y means walk do X then Y", ["JUMP", "WALK"]),
        ("X means jump Y means walk do Y then X", ["WALK", "JUMP"]),
        ("X means jump do X then X means walk do X", ["JUMP", "WALK"]),
        ("X means jump do X twice then Y means walk do Y", ["JUMP", "JUMP", "WALK"]),
    ]

    print("Testing specific patterns:\n")

    for command, expected in test_cases:
        print(f"Command: {command}")
        print(f"Expected: {expected}")

        tokens = [VOCAB.get(w, 0) for w in command.split()]
        tokens_mx = mx.array([tokens])

        # Analyze
        analysis = model.analyze(tokens_mx)
        print(f"Segments: {len(analysis['segments'])} segments")
        for seg in analysis["segments"]:
            print(f"  - {seg}")
        print(f"Actions: {analysis['actions']}")

        is_correct = analysis["actions"] == expected
        print(f"Result: {'✅ PASS' if is_correct else '❌ FAIL'}\n")

    # Full evaluation
    print("\n" + "=" * 60)
    print("FULL EVALUATION WITH FINAL FIX")
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

    # Specific THEN analysis
    print("\n" + "=" * 60)
    print("THEN PATTERN PERFORMANCE")
    print("=" * 60 + "\n")

    then_correct = 0
    then_total = 0

    for level_data in test_data.values():
        for sample in level_data:
            # Check for THEN patterns (not rebinding)
            if (
                " then " in sample["command"]
                and "means" not in sample["command"].split("then")[1]
            ):
                then_total += 1
                tokens = mx.array([sample["tokens"]])

                # Get predictions
                outputs = model(tokens)
                predicted = []
                if outputs.shape[0] > 0:
                    indices = mx.argmax(outputs, axis=-1)
                    action_names = ["JUMP", "WALK", "RUN", "TURN"]
                    for idx in indices:
                        predicted.append(action_names[int(idx)])

                if predicted == sample["expected_actions"]:
                    then_correct += 1
                elif then_total <= 5:  # Show first few failures
                    print(f"Failed: {sample['command']}")
                    print(f"  Expected: {sample['expected_actions']}")
                    print(f"  Got: {predicted}")

                    # Debug why it failed
                    analysis = model.analyze(tokens)
                    print(f"  Segments: {len(analysis['segments'])}")
                    print(f"  Analysis actions: {analysis['actions']}")

    then_accuracy = then_correct / max(then_total, 1)
    print(f"\nTHEN accuracy: {then_correct}/{then_total} = {then_accuracy:.2%}")

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)

    if avg_accuracy > 90:
        print(f"✅ SUCCESS! Achieved {avg_accuracy:.2%} accuracy!")
        print("The Two-Stage Compiler now properly handles all operators.")
        print("This validates that distribution invention needs:")
        print("- Explicit rule extraction (100% on bindings)")
        print("- Discrete state tracking (temporal scoping)")
        print("- Minimal learning (only compositional patterns)")
    else:
        print(f"Current accuracy: {avg_accuracy:.2%}")
        print(f"THEN accuracy: {then_accuracy:.2%}")
        print("Further improvements needed for THEN handling.")


if __name__ == "__main__":
    test_final_fix()
