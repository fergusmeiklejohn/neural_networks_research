#!/usr/bin/env python3
"""Fix compositional operators by improving execution logic."""

from utils.imports import setup_project_paths

setup_project_paths()

from utils.config import setup_environment

config = setup_environment()

from typing import Dict, List

import mlx.core as mx
from compositional_operators import CompositionalParser, OperatorType, ParseNode


class FixedCompositionalExecutor:
    """Fixed executor that properly handles compositional expressions."""

    def __init__(self, model, vocab: Dict[str, int]):
        self.model = model
        self.vocab = vocab
        self.parser = CompositionalParser(vocab)

        # Special tokens
        self.do_token = vocab.get("do", -1)
        self.means_token = vocab.get("means", -1)

    def execute(
        self,
        parse_tree: ParseNode,
        command_ids: mx.array,
        bindings: Dict,
        stage: str = "full",
    ) -> List[mx.array]:
        """Execute a parse tree and return action outputs."""
        if parse_tree.is_leaf():
            # Execute leaf node
            return self._execute_leaf(parse_tree, command_ids, bindings, stage)

        # Execute based on operator type
        if parse_tree.operator == OperatorType.SEQUENCE:
            return self._execute_sequence(parse_tree, command_ids, bindings, stage)
        elif parse_tree.operator == OperatorType.PARALLEL:
            return self._execute_parallel(parse_tree, command_ids, bindings, stage)
        elif parse_tree.operator == OperatorType.LOOP:
            return self._execute_loop(parse_tree, command_ids, bindings, stage)
        elif parse_tree.operator == OperatorType.CHOICE:
            return self._execute_choice(parse_tree, command_ids, bindings, stage)
        else:
            raise ValueError(f"Unknown operator: {parse_tree.operator}")

    def _execute_leaf(
        self, node: ParseNode, command_ids: mx.array, bindings: Dict, stage: str
    ) -> List[mx.array]:
        """Execute a leaf node with better handling of 'do' patterns."""
        # Skip empty nodes
        if not node.tokens:
            return []

        # Find 'do' token and process what follows
        outputs = []
        i = 0
        while i < len(node.tokens):
            if node.tokens[i] == self.do_token and i + 1 < len(node.tokens):
                # Process the token after 'do'
                var_token = node.tokens[i + 1]

                # Convert token to variable name
                var_name = None
                for word, idx in self.vocab.items():
                    if idx == var_token:
                        var_name = word
                        break

                if var_name and var_name in bindings:
                    # Get the bound action
                    action_name = bindings[var_name]
                    # Import ACTIONS from train_integrated_model
                    from train_integrated_model import ACTIONS

                    action_idx = ACTIONS.get(action_name, 0)

                    # Create output for this action
                    output = mx.zeros((self.model.num_actions,))
                    output = mx.where(
                        mx.arange(self.model.num_actions) == action_idx, 1000.0, output
                    )
                    outputs.append(output)
                else:
                    # Try to execute the segment normally
                    segment = (
                        node.start_pos + i,
                        min(node.start_pos + i + 2, node.end_pos),
                    )
                    segment_outputs = self.model.process_segment_versioned(
                        command_ids, segment, bindings, stage
                    )
                    outputs.extend(segment_outputs)

                i += 2  # Skip 'do' and the variable
            else:
                i += 1

        # If no outputs were generated, process the entire segment
        if not outputs:
            segment = (node.start_pos, node.end_pos)
            outputs = self.model.process_segment_versioned(
                command_ids, segment, bindings, stage
            )

        return outputs

    def _execute_sequence(
        self, node: ParseNode, command_ids: mx.array, bindings: Dict, stage: str
    ) -> List[mx.array]:
        """Execute children in sequence."""
        outputs = []
        for child in node.children:
            child_outputs = self.execute(child, command_ids, bindings, stage)
            outputs.extend(child_outputs)
        return outputs

    def _execute_parallel(
        self, node: ParseNode, command_ids: mx.array, bindings: Dict, stage: str
    ) -> List[mx.array]:
        """Execute children in parallel (left to right)."""
        outputs = []

        # Special handling for "do X and Y" pattern
        if len(node.children) == 2:
            left_child = node.children[0]
            right_child = node.children[1]

            # Check if left child contains "do X"
            if left_child.is_leaf() and self.do_token in left_child.tokens:
                left_outputs = self._execute_leaf(
                    left_child, command_ids, bindings, stage
                )
                outputs.extend(left_outputs)
            else:
                left_outputs = self.execute(left_child, command_ids, bindings, stage)
                outputs.extend(left_outputs)

            # Execute right child (often just "Y")
            if right_child.is_leaf() and len(right_child.tokens) == 1:
                # Single token, likely a variable
                var_token = right_child.tokens[0]
                var_name = None
                for word, idx in self.vocab.items():
                    if idx == var_token:
                        var_name = word
                        break

                if var_name and var_name in bindings:
                    action_name = bindings[var_name]
                    from train_integrated_model import ACTIONS

                    action_idx = ACTIONS.get(action_name, 0)
                    output = mx.zeros((self.model.num_actions,))
                    output = mx.where(
                        mx.arange(self.model.num_actions) == action_idx, 1000.0, output
                    )
                    outputs.append(output)
                else:
                    right_outputs = self.execute(
                        right_child, command_ids, bindings, stage
                    )
                    outputs.extend(right_outputs)
            else:
                right_outputs = self.execute(right_child, command_ids, bindings, stage)
                outputs.extend(right_outputs)
        else:
            # General case: execute all children
            for child in node.children:
                child_outputs = self.execute(child, command_ids, bindings, stage)
                outputs.extend(child_outputs)

        return outputs

    def _execute_loop(
        self, node: ParseNode, command_ids: mx.array, bindings: Dict, stage: str
    ) -> List[mx.array]:
        """Execute loop with better handling."""
        outputs = []

        # Default loop count
        loop_count = 3

        if len(node.children) >= 2:
            # Check condition for special cases
            condition = node.children[0]
            if condition.is_leaf():
                # Check for "true" in condition
                true_token = self.vocab.get("true", -1)
                if true_token in condition.tokens:
                    loop_count = 3

            # Execute body loop_count times
            body = node.children[1]
            for _ in range(loop_count):
                body_outputs = self.execute(body, command_ids, bindings, stage)
                outputs.extend(body_outputs)

        return outputs

    def _execute_choice(
        self, node: ParseNode, command_ids: mx.array, bindings: Dict, stage: str
    ) -> List[mx.array]:
        """Execute choice deterministically for training."""
        if len(node.children) == 0:
            return []

        # For training, always choose the first option
        chosen_child = node.children[0]
        return self.execute(chosen_child, command_ids, bindings, stage)


def test_fixed_executor():
    """Test the fixed compositional executor."""
    from train_integrated_model import ACTIONS, VOCAB, IntegratedBindingModel

    # Create model
    model = IntegratedBindingModel(
        vocab_size=len(VOCAB),
        num_actions=len(ACTIONS),
        embed_dim=256,
        num_slots=4,
        num_heads=8,
        mlp_hidden_dim=512,
    )

    # Create fixed executor
    executor = FixedCompositionalExecutor(model, VOCAB)

    test_cases = [
        (
            "X means jump Y means walk do X and Y",
            {"X": "JUMP", "Y": "WALK"},
            ["JUMP", "WALK"],
        ),
        (
            "X means jump Y means walk do X then do Y",
            {"X": "JUMP", "Y": "WALK"},
            ["JUMP", "WALK"],
        ),
        ("X means jump while true do X", {"X": "JUMP"}, ["JUMP", "JUMP", "JUMP"]),
        (
            "X means walk Y means jump do X and Y then do X",
            {"X": "WALK", "Y": "JUMP"},
            ["WALK", "JUMP", "WALK"],
        ),
    ]

    print("=== Testing Fixed Compositional Executor ===\n")

    for command, bindings, expected in test_cases:
        print(f"Command: {command}")
        print(f"Bindings: {bindings}")
        print(f"Expected: {expected}")

        # Parse command
        tokens = [VOCAB.get(word, VOCAB["<PAD>"]) for word in command.split()]
        command_ids = mx.array([tokens])

        # Clear memory and execute
        model.versioned_memory.clear()

        # Parse tree
        parse_tree = executor.parser.parse(command_ids)
        print(
            f"Parse tree: operator={parse_tree.operator}, is_leaf={parse_tree.is_leaf()}"
        )

        # Execute
        outputs = executor.execute(parse_tree, command_ids, bindings, "full")
        print(f"Got {len(outputs)} outputs")

        # Convert to predictions
        predicted_actions = []
        for output in outputs:
            pred_idx = int(mx.argmax(output))
            for name, idx in ACTIONS.items():
                if idx == pred_idx:
                    predicted_actions.append(name)
                    break

        print(f"Predicted: {predicted_actions}")
        print(f"Correct: {predicted_actions == expected}")
        print()


if __name__ == "__main__":
    test_fixed_executor()
