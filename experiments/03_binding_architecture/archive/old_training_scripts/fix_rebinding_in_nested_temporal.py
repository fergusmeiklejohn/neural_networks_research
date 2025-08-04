#!/usr/bin/env python3
"""Fix for the rebinding issue in nested temporal model.

The issue: NestedTemporalBindingModel doesn't use sequence_planner to handle 'then'.
This causes all segments to share the same versioned memory context.
"""

from utils.imports import setup_project_paths

setup_project_paths()

from typing import Dict, List

import mlx.core as mx
from compositional_operators import (
    CompositionalExecutor,
    CompositionalParser,
)
from nested_temporal_patterns import (
    NestedTemporalExecutor,
    NestedTemporalParser,
)
from train_integrated_model import ACTIONS, VOCAB, IntegratedBindingModel

from utils.config import setup_environment

config = setup_environment()


class FixedNestedTemporalBindingModel(IntegratedBindingModel):
    """Fixed version that properly handles 'then' for sequence segmentation."""

    def __init__(
        self,
        vocab_size: int,
        num_actions: int,
        embed_dim: int = 256,
        num_slots: int = 4,
        num_heads: int = 8,
        mlp_hidden_dim: int = 512,
    ):
        super().__init__(
            vocab_size, num_actions, embed_dim, num_slots, num_heads, mlp_hidden_dim
        )

        # Add parsers and executors
        self.compositional_parser = CompositionalParser(VOCAB)
        self.compositional_executor = CompositionalExecutor(self, VOCAB)
        self.temporal_parser = NestedTemporalParser(VOCAB)
        self.temporal_executor = NestedTemporalExecutor(self, VOCAB, ACTIONS)

    def __call__(self, inputs: Dict[str, mx.array], stage: str = "full") -> mx.array:
        """Forward pass with proper 'then' handling."""
        command_ids = inputs["command"]

        # Clear versioned memory for new sequence
        self.versioned_memory.clear()

        # CRITICAL FIX: Use sequence planner to parse 'then' segments
        segments = self.sequence_planner.parse_sequence(command_ids)

        # Process each segment separately
        all_outputs = []
        bindings = {}  # Shared across segments for basic bindings

        for seg_idx, (seg_start, seg_end) in enumerate(segments):
            # Extract segment tokens
            if hasattr(command_ids, "numpy"):
                full_tokens = command_ids.numpy()
            else:
                full_tokens = command_ids

            if len(full_tokens.shape) > 1:
                full_tokens = full_tokens[0]

            segment_tokens = full_tokens[seg_start:seg_end]
            segment_token_list = [int(t) for t in segment_tokens.tolist()]

            # Check if this segment has nested temporal patterns
            temporal_nodes = self.temporal_parser.parse(segment_token_list)

            if temporal_nodes and self._is_pure_temporal_pattern(segment_token_list):
                # Handle as nested temporal pattern
                seg_bindings = self._extract_bindings_for_segment(
                    command_ids, seg_start, seg_end, bindings
                )
                outputs = self.temporal_executor.execute_nested_pattern(
                    segment_token_list, seg_bindings
                )
            else:
                # Use regular segment processing
                outputs = self.process_segment_versioned(
                    command_ids, (seg_start, seg_end), bindings, stage
                )

            all_outputs.extend(outputs)

        # Stack outputs
        if all_outputs:
            squeezed_outputs = []
            for out in all_outputs:
                if len(out.shape) > 1:
                    squeezed_outputs.append(out.squeeze())
                else:
                    squeezed_outputs.append(out)
            return mx.stack(squeezed_outputs)
        else:
            return mx.zeros((1, self.num_actions))

    def _is_pure_temporal_pattern(self, tokens: List[int]) -> bool:
        """Check if this is a pure nested temporal pattern (no compositional operators)."""
        id_to_word = {v: k for k, v in VOCAB.items()}
        words = [id_to_word.get(t, "<PAD>") for t in tokens]

        # Check for compositional operators (but NOT 'then' - that's handled by segments)
        compositional_ops = {"and", "or", "while"}
        return not any(word in compositional_ops for word in words)

    def _extract_bindings_for_segment(
        self,
        command_ids: mx.array,
        seg_start: int,
        seg_end: int,
        existing_bindings: Dict,
    ) -> Dict:
        """Extract bindings for a specific segment, including any from the segment itself."""
        # Start with existing bindings
        bindings = existing_bindings.copy()

        # Process bindings in this segment
        if hasattr(command_ids, "numpy"):
            tokens = command_ids.numpy()
        else:
            tokens = command_ids

        if len(tokens.shape) > 1:
            tokens = tokens[0]

        # Look for "X means Y" patterns in segment
        means_token = VOCAB.get("means", -1)
        i = seg_start
        while i < seg_end - 2:
            if int(tokens[i + 1]) == means_token:
                var_token = int(tokens[i])
                action_token = int(tokens[i + 2])

                # Find variable and action names
                var_name = None
                action_name = None
                for word, wid in VOCAB.items():
                    if wid == var_token:
                        var_name = word
                    if wid == action_token:
                        for act_word, act_id in ACTIONS.items():
                            if act_word.lower() == word:
                                action_name = act_word
                                break

                if var_name and action_name:
                    bindings[var_name] = action_name

                i += 3
            else:
                i += 1

        return bindings

    def _extract_bindings(self, tokens: List[int]) -> Dict[str, str]:
        """Extract variable bindings from token sequence."""
        bindings = {}
        means_token = VOCAB.get("means", -1)

        i = 0
        while i < len(tokens) - 2:
            if tokens[i + 1] == means_token:
                # Found "X means Y" pattern
                var_token = tokens[i]
                action_token = tokens[i + 2]

                # Convert tokens to words
                var_name = None
                action_name = None
                for word, wid in VOCAB.items():
                    if wid == var_token:
                        var_name = word
                    if wid == action_token:
                        # Map to action
                        for act_word, act_id in ACTIONS.items():
                            if act_word.lower() == word:
                                action_name = act_word
                                break

                if var_name and action_name:
                    bindings[var_name] = action_name

                i += 3
            else:
                i += 1

        return bindings


def test_rebinding_fix():
    """Test that the fix resolves the rebinding issue."""
    print("=== Testing Rebinding Fix ===\n")

    # Ensure temporal modifiers are in VOCAB
    for mod in ["twice", "thrice", "then"]:
        if mod not in VOCAB:
            VOCAB[mod] = len(VOCAB)

    # Create model
    model = FixedNestedTemporalBindingModel(len(VOCAB), len(ACTIONS))

    # Test cases
    test_cases = [
        # Simple rebinding
        ("X means jump do X then X means walk do X", ["JUMP", "WALK"]),
        # Rebinding with nested temporal
        (
            "X means jump do X twice then X means walk do X twice twice",
            ["JUMP", "JUMP"] + ["WALK"] * 4,
        ),
        # Multiple rebindings
        (
            "X means jump do X then X means walk do X then X means turn do X",
            ["JUMP", "WALK", "TURN"],
        ),
    ]

    for command, expected in test_cases:
        print(f"Command: {command}")
        print(f"Expected: {expected}")

        tokens = [VOCAB.get(word, VOCAB["<PAD>"]) for word in command.split()]
        inputs = {"command": mx.array([tokens])}

        # Debug: show segments
        segments = model.sequence_planner.parse_sequence(mx.array([tokens]))
        print(f"Segments: {segments}")

        # Get predictions
        outputs = model(inputs, stage="full")
        predictions = mx.argmax(outputs, axis=1)

        predicted_actions = []
        for pred in predictions:
            for name, idx in ACTIONS.items():
                if idx == int(pred):
                    predicted_actions.append(name)
                    break

        print(f"Predicted: {predicted_actions}")

        # Note: Without training, predictions will be random
        # The important thing is that we get the right number of outputs
        # and that segments are processed separately
        print(f"Output count correct: {len(predicted_actions) == len(expected)}")
        print()


if __name__ == "__main__":
    test_rebinding_fix()
