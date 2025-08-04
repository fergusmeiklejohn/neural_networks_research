#!/usr/bin/env python3
"""Fixed compositional operators with proper binding separation.

This module provides the corrected implementation that separates variable
bindings from execution parsing, solving the LEAF node issue.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

# Note: When used in actual training scripts, these imports will come from MLX
# For now, we define minimal stubs for standalone use
try:
    import mlx.core as mx
except ImportError:
    # Stub for testing
    class mx:
        @staticmethod
        def array(data):
            return np.array(data)


class OperatorType(Enum):
    """Types of compositional operators."""

    SEQUENCE = "then"
    PARALLEL = "and"
    LOOP = "while"
    CHOICE = "or"


@dataclass
class ParseNode:
    """Node in the parse tree for compositional expressions."""

    operator: Optional[OperatorType] = None
    children: List["ParseNode"] = None
    tokens: List[int] = None
    start_pos: int = 0
    end_pos: int = 0

    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.tokens is None:
            self.tokens = []

    def is_leaf(self) -> bool:
        """Check if this is a leaf node (contains tokens, not operators)."""
        return len(self.children) == 0 and len(self.tokens) > 0

    def is_operator(self) -> bool:
        """Check if this is an operator node."""
        return self.operator is not None


class CompositionalParser:
    """Fixed parser that properly separates bindings from execution."""

    def __init__(self, vocab: Dict[str, int]):
        self.vocab = vocab
        self.vocab_reverse = {v: k for k, v in vocab.items()}
        self.operator_tokens = {
            vocab.get("then", -1): OperatorType.SEQUENCE,
            vocab.get("and", -1): OperatorType.PARALLEL,
            vocab.get("while", -1): OperatorType.LOOP,
            vocab.get("or", -1): OperatorType.CHOICE,
        }
        # Remove any -1 values
        self.operator_tokens = {
            k: v for k, v in self.operator_tokens.items() if k != -1
        }

        # Special tokens
        self.do_token = vocab.get("do", -1)
        self.means_token = vocab.get("means", -1)

        # Operator precedence (higher number = higher precedence)
        self.precedence = {
            OperatorType.CHOICE: 1,  # Lowest precedence
            OperatorType.PARALLEL: 2,
            OperatorType.SEQUENCE: 3,
            OperatorType.LOOP: 4,  # Highest precedence
        }

    def parse(self, tokens) -> ParseNode:
        """Parse token sequence with proper binding separation.

        This is the main fix: we separate bindings from execution before parsing.
        """
        # Convert to list
        token_list = self._tokens_to_list(tokens)

        # Find execution start (after 'do')
        exec_start = self._find_execution_start(token_list)

        if exec_start > 0:
            # Parse only the execution part
            tree = self._parse_expression(token_list, exec_start, len(token_list))
        else:
            # No 'do' found, parse entire sequence
            tree = self._parse_expression(token_list, 0, len(token_list))

        return tree

    def parse_with_bindings(self, tokens) -> Tuple[ParseNode, Dict[str, str]]:
        """Parse tokens and extract bindings separately.

        Returns:
            - ParseNode: The parse tree for the execution part
            - Dict[str, str]: Variable bindings extracted from the command
        """
        token_list = self._tokens_to_list(tokens)

        # Extract bindings
        bindings = self.extract_bindings(tokens)

        # Find execution start
        exec_start = self._find_execution_start(token_list)

        # Parse execution part
        if exec_start > 0:
            tree = self._parse_expression(token_list, exec_start, len(token_list))
        else:
            tree = self._parse_expression(token_list, 0, len(token_list))

        return tree, bindings

    def extract_bindings(self, tokens) -> Dict[str, str]:
        """Extract variable bindings from the token sequence.

        Returns dict mapping variable names to action names.
        """
        token_list = self._tokens_to_list(tokens)

        bindings = {}
        i = 0

        # Look for "VAR means ACTION" patterns before 'do'
        while i < len(token_list) - 2:
            if token_list[i] == self.do_token:
                break  # Stop at execution start

            if token_list[i + 1] == self.means_token:
                var_name = self.vocab_reverse.get(token_list[i], "")
                action_name = self.vocab_reverse.get(token_list[i + 2], "")
                if var_name and action_name:
                    bindings[var_name] = action_name
                i += 3
            else:
                i += 1

        return bindings

    def _tokens_to_list(self, tokens) -> List[int]:
        """Convert various token formats to a list of ints."""
        if hasattr(tokens, "numpy"):
            token_array = tokens.numpy()
        elif hasattr(tokens, "tolist"):
            token_array = tokens.tolist()
        else:
            token_array = tokens

        # Handle batch dimension
        if isinstance(token_array, np.ndarray) and len(token_array.shape) > 1:
            token_array = token_array[0]

        # Ensure all are ints
        if isinstance(token_array, np.ndarray):
            token_list = token_array.tolist()
        else:
            token_list = list(token_array)

        return [int(t) for t in token_list]

    def _find_execution_start(self, token_list: List[int]) -> int:
        """Find where execution starts (position after 'do')."""
        for i, token in enumerate(token_list):
            if token == self.do_token:
                return i + 1
        return 0  # No 'do' found

    def _parse_expression(self, tokens: List[int], start: int, end: int) -> ParseNode:
        """Parse an expression within the given range."""
        # Find the operator with lowest precedence (rightmost if tie)
        split_pos = -1
        split_op = None
        min_precedence = float("inf")

        # Scan for operators
        for i in range(start, end):
            if tokens[i] in self.operator_tokens:
                op_type = self.operator_tokens[tokens[i]]
                prec = self.precedence[op_type]

                # Use <= to prefer rightmost operator of same precedence
                if prec <= min_precedence:
                    min_precedence = prec
                    split_pos = i
                    split_op = op_type

        # If no operator found, this is a leaf node
        if split_pos == -1:
            return ParseNode(tokens=tokens[start:end], start_pos=start, end_pos=end)

        # Handle special case for "while" operator
        if split_op == OperatorType.LOOP:
            # "while X do Y" pattern - X is condition, Y is body
            condition_end = split_pos
            body_start = split_pos + 1

            # Look for "do" token after "while"
            for i in range(body_start, end):
                if tokens[i] == self.do_token:
                    condition_end = i
                    body_start = i + 1
                    break

            node = ParseNode(operator=split_op, start_pos=start, end_pos=end)
            if condition_end > start:
                node.children.append(
                    self._parse_expression(tokens, start, condition_end)
                )
            if body_start < end:
                node.children.append(self._parse_expression(tokens, body_start, end))
            return node

        # For other operators, split into left and right
        node = ParseNode(operator=split_op, start_pos=start, end_pos=end)

        # Parse left side
        if split_pos > start:
            node.children.append(self._parse_expression(tokens, start, split_pos))

        # Parse right side
        if split_pos + 1 < end:
            node.children.append(self._parse_expression(tokens, split_pos + 1, end))

        return node


class CompositionalExecutor:
    """Fixed executor that properly handles separated bindings."""

    def __init__(self, model, vocab: Dict[str, int]):
        self.model = model
        self.vocab = vocab
        self.vocab_reverse = {v: k for k, v in vocab.items()}
        self.parser = CompositionalParser(vocab)

    def execute(
        self,
        parse_tree: ParseNode,
        command_ids,
        bindings: Dict[str, str],
        stage: str = "full",
    ) -> List:
        """Execute a parse tree with proper variable resolution."""
        if parse_tree.is_leaf():
            # Execute leaf node with variable resolution
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

    def execute_command(self, command_ids, stage: str = "full") -> List:
        """Execute a command with automatic binding extraction."""
        # Parse and extract bindings
        tree, bindings = self.parser.parse_with_bindings(command_ids)

        # Execute with bindings
        return self.execute(tree, command_ids, bindings, stage)

    def _execute_leaf(
        self, node: ParseNode, command_ids, bindings: Dict[str, str], stage: str
    ) -> List:
        """Execute a leaf node with proper variable resolution."""
        # For compositional operators, leaf nodes contain only the execution part
        # (e.g., just "X" or "Y", not "X means jump Y means walk do X")

        # If the model has a process_segment_versioned method, use it
        if hasattr(self.model, "process_segment_versioned"):
            # Use the model's existing segment processing
            segment = (node.start_pos, node.end_pos)
            return self.model.process_segment_versioned(
                command_ids, segment, bindings, stage
            )
        else:
            # Fallback: resolve variables manually
            outputs = []

            for token_id in node.tokens:
                token_str = self.vocab_reverse.get(token_id, "")

                # Skip operators and special tokens
                if token_str in ["and", "then", "or", "while", "do"]:
                    continue

                # Check if it's a variable that needs resolution
                if token_str in bindings:
                    action_name = bindings[token_str]
                    outputs.append(self._create_action_output(action_name))
                else:
                    # Direct action
                    if token_str in ["jump", "walk", "turn", "run"]:
                        outputs.append(self._create_action_output(token_str))

            return outputs

    def _create_action_output(self, action_name: str):
        """Create action output for fallback mode."""
        try:
            import mlx.core as mx

            action_map = {"jump": 0, "walk": 1, "turn": 2, "run": 3}
            if action_name in action_map:
                return mx.array([action_map[action_name]])
        except ImportError:
            # Return as numpy array if MLX not available
            action_map = {"jump": 0, "walk": 1, "turn": 2, "run": 3}
            if action_name in action_map:
                return np.array([action_map[action_name]])
        return None

    def _execute_sequence(
        self, node: ParseNode, command_ids, bindings: Dict[str, str], stage: str
    ) -> List:
        """Execute children in sequence (then operator)."""
        outputs = []
        for child in node.children:
            child_outputs = self.execute(child, command_ids, bindings, stage)
            outputs.extend(child_outputs)
        return outputs

    def _execute_parallel(
        self, node: ParseNode, command_ids, bindings: Dict[str, str], stage: str
    ) -> List:
        """Execute children in parallel (and operator)."""
        outputs = []
        for child in node.children:
            child_outputs = self.execute(child, command_ids, bindings, stage)
            outputs.extend(child_outputs)
        return outputs

    def _execute_loop(
        self, node: ParseNode, command_ids, bindings: Dict[str, str], stage: str
    ) -> List:
        """Execute loop (while operator) - simplified to 2 iterations."""
        outputs = []
        if len(node.children) >= 2:
            # Execute body twice (simplified)
            body = node.children[1]
            for _ in range(2):
                body_outputs = self.execute(body, command_ids, bindings, stage)
                outputs.extend(body_outputs)
        return outputs

    def _execute_choice(
        self, node: ParseNode, command_ids, bindings: Dict[str, str], stage: str
    ) -> List:
        """Execute choice (or operator) - execute first option."""
        if node.children:
            return self.execute(node.children[0], command_ids, bindings, stage)
        return []
