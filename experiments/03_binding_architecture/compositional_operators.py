#!/usr/bin/env python3
"""Implementation of compositional operators (and, while, or) for variable binding.

This module extends the integrated model to support:
1. "and" operator - Parallel execution of actions
2. "while" operator - Conditional loops (simplified for now)
3. "or" operator - Choice between alternatives
"""

from utils.imports import setup_project_paths

setup_project_paths()

from utils.config import setup_environment

config = setup_environment()

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import mlx.core as mx
import numpy as np


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
    """Parse commands with compositional operators into tree structure."""

    def __init__(self, vocab: Dict[str, int]):
        self.vocab = vocab
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

        # Operator precedence (higher number = higher precedence)
        self.precedence = {
            OperatorType.CHOICE: 1,  # Lowest precedence
            OperatorType.PARALLEL: 2,
            OperatorType.SEQUENCE: 3,
            OperatorType.LOOP: 4,  # Highest precedence
        }

    def parse(self, tokens: mx.array) -> ParseNode:
        """Parse token sequence into hierarchical structure."""
        # Convert to list for easier manipulation
        if hasattr(tokens, "numpy"):
            token_array = tokens.numpy()
        else:
            token_array = tokens

        # Handle batch dimension
        if len(token_array.shape) > 1:
            token_array = token_array[0]

        # Convert to python list
        token_list = token_array.tolist()

        # Ensure all are ints
        token_list = [int(t) for t in token_list]

        return self._parse_expression(token_list, 0, len(token_list))

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
            # For now, we'll treat it as "repeat Y while X is true"
            # In our simple case, we'll execute Y a fixed number of times
            condition_end = split_pos
            body_start = split_pos + 1

            # Look for "do" token after "while"
            do_token = self.vocab.get("do", -1)
            for i in range(body_start, end):
                if tokens[i] == do_token:
                    condition_end = i
                    body_start = i + 1
                    break

            node = ParseNode(operator=split_op, start_pos=start, end_pos=end)
            node.children.append(self._parse_expression(tokens, start, condition_end))
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
    """Execute parsed compositional expressions."""

    def __init__(self, model, vocab: Dict[str, int]):
        self.model = model
        self.vocab = vocab
        self.parser = CompositionalParser(vocab)

    def execute(
        self,
        parse_tree: ParseNode,
        command_ids: mx.array,
        bindings: Dict,
        stage: str = "full",
    ) -> List[mx.array]:
        """Execute a parse tree and return action outputs."""
        if parse_tree.is_leaf():
            # Execute leaf node (basic command)
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
        """Execute a leaf node (basic command segment)."""
        # Use the model's existing segment processing
        segment = (node.start_pos, node.end_pos)
        return self.model.process_segment_versioned(
            command_ids, segment, bindings, stage
        )

    def _execute_sequence(
        self, node: ParseNode, command_ids: mx.array, bindings: Dict, stage: str
    ) -> List[mx.array]:
        """Execute children in sequence (then operator)."""
        outputs = []
        for child in node.children:
            child_outputs = self.execute(child, command_ids, bindings, stage)
            outputs.extend(child_outputs)
        return outputs

    def _execute_parallel(
        self, node: ParseNode, command_ids: mx.array, bindings: Dict, stage: str
    ) -> List[mx.array]:
        """Execute children in parallel (and operator)."""
        # For "do X and Y", we execute both and return both actions
        outputs = []
        for child in node.children:
            child_outputs = self.execute(child, command_ids, bindings, stage)
            outputs.extend(child_outputs)
        return outputs

    def _execute_loop(
        self, node: ParseNode, command_ids: mx.array, bindings: Dict, stage: str
    ) -> List[mx.array]:
        """Execute loop (while operator)."""
        # Simplified: execute body a fixed number of times
        # In a full implementation, we'd evaluate the condition
        outputs = []

        # For now, execute 3 times (could be made dynamic)
        loop_count = 3

        if len(node.children) >= 2:
            # node.children[0] is condition (ignored for now)
            # node.children[1] is body
            body = node.children[1]

            for _ in range(loop_count):
                body_outputs = self.execute(body, command_ids, bindings, stage)
                outputs.extend(body_outputs)

        return outputs

    def _execute_choice(
        self, node: ParseNode, command_ids: mx.array, bindings: Dict, stage: str
    ) -> List[mx.array]:
        """Execute choice (or operator)."""
        # Randomly choose one child to execute
        if len(node.children) == 0:
            return []

        # Choose randomly (could be made more sophisticated)
        choice_idx = np.random.randint(len(node.children))
        chosen_child = node.children[choice_idx]

        return self.execute(chosen_child, command_ids, bindings, stage)


def test_compositional_parsing():
    """Test the compositional parser."""
    from train_integrated_model import VOCAB

    # Ensure operators are in vocabulary
    if "then" not in VOCAB:
        VOCAB["then"] = len(VOCAB)
    if "and" not in VOCAB:
        VOCAB["and"] = len(VOCAB)
    if "or" not in VOCAB:
        VOCAB["or"] = len(VOCAB)
    if "while" not in VOCAB:
        VOCAB["while"] = len(VOCAB)

    parser = CompositionalParser(VOCAB)

    test_cases = [
        "do X and Y",
        "do X then do Y",
        "do X or Y twice",
        "do X and Y then do Z",
        "while X do Y",
        "do X then do Y and Z",
    ]

    print("=== Testing Compositional Parser ===")
    print(
        f"Vocabulary: then={VOCAB.get('then')}, and={VOCAB.get('and')}, or={VOCAB.get('or')}, while={VOCAB.get('while')}"
    )
    print(f"Parser operator tokens: {parser.operator_tokens}\n")

    for command in test_cases:
        print(f"Command: {command}")
        tokens = [VOCAB.get(word, VOCAB["<PAD>"]) for word in command.split()]
        print(f"Tokens: {tokens}")

        tree = parser.parse(mx.array(tokens))
        print(
            f"Tree operator: {tree.operator}, is_leaf: {tree.is_leaf()}, children: {len(tree.children)}"
        )
        print_tree(tree, tokens, vocab=VOCAB)
        print()


def print_tree(
    node: ParseNode, tokens: List[int], indent: int = 0, vocab: Dict[str, int] = None
):
    """Pretty print a parse tree."""
    prefix = "  " * indent

    if node.is_leaf():
        # Convert token IDs back to words for display
        words = []
        for tid in node.tokens:
            if vocab:
                for word, wid in vocab.items():
                    if wid == tid:
                        words.append(word)
                        break
                else:
                    words.append(f"<{tid}>")
            else:
                words.append(f"<{tid}>")
        print(f"{prefix}LEAF: {' '.join(words)}")
    else:
        if node.operator:
            print(f"{prefix}{node.operator.value.upper()}:")
        else:
            print(f"{prefix}UNKNOWN:")
        for child in node.children:
            print_tree(child, tokens, indent + 1, vocab)


if __name__ == "__main__":
    pass

    test_compositional_parsing()
