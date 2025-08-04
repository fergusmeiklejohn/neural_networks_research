#!/usr/bin/env python3
"""Improved implementation of compositional operators with better parsing and execution logic."""

from utils.imports import setup_project_paths

setup_project_paths()

from utils.config import setup_environment

config = setup_environment()

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import mlx.core as mx


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


class ImprovedCompositionalParser:
    """Improved parser for commands with compositional operators."""

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
        # Updated precedence to better handle common patterns
        self.precedence = {
            OperatorType.SEQUENCE: 1,  # Lowest - sequences bind loosest
            OperatorType.CHOICE: 2,
            OperatorType.PARALLEL: 3,  # Parallel binds tighter than sequence
            OperatorType.LOOP: 4,  # Highest - loops bind tightest
        }

        # Special tokens
        self.do_token = vocab.get("do", -1)
        self.means_token = vocab.get("means", -1)

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

        # Remove leading/trailing padding if present
        pad_token = self.vocab.get("<PAD>", -1)
        while token_list and token_list[0] == pad_token:
            token_list.pop(0)
        while token_list and token_list[-1] == pad_token:
            token_list.pop()

        return self._parse_expression(token_list, 0, len(token_list))

    def _find_matching_do(self, tokens: List[int], start: int, end: int) -> int:
        """Find the matching 'do' for a while/condition construct."""
        depth = 0
        for i in range(start, end):
            if tokens[i] == self.do_token and depth == 0:
                return i
            elif (
                tokens[i] in self.operator_tokens
                and self.operator_tokens[tokens[i]] == OperatorType.LOOP
            ):
                depth += 1
            elif tokens[i] == self.do_token and depth > 0:
                depth -= 1
        return -1

    def _parse_expression(self, tokens: List[int], start: int, end: int) -> ParseNode:
        """Parse an expression within the given range."""
        if start >= end:
            return ParseNode(tokens=[], start_pos=start, end_pos=end)

        # Handle special case for "while X do Y" first
        if start < end and tokens[start] in self.operator_tokens:
            if self.operator_tokens[tokens[start]] == OperatorType.LOOP:
                # Find the matching 'do'
                do_pos = self._find_matching_do(tokens, start + 1, end)
                if do_pos != -1:
                    node = ParseNode(
                        operator=OperatorType.LOOP, start_pos=start, end_pos=end
                    )
                    # Condition is between 'while' and 'do'
                    node.children.append(
                        self._parse_expression(tokens, start + 1, do_pos)
                    )
                    # Body is after 'do'
                    node.children.append(
                        self._parse_expression(tokens, do_pos + 1, end)
                    )
                    return node

        # Find the operator with lowest precedence (rightmost if tie)
        split_pos = -1
        split_op = None
        min_precedence = float("inf")

        # Track depth for nested structures
        paren_depth = 0

        # Scan for operators from right to left to prefer rightmost
        for i in range(end - 1, start - 1, -1):
            # Skip operators inside loops
            if tokens[i] == self.do_token:
                paren_depth += 1
            elif (
                i > start
                and tokens[i - 1] in self.operator_tokens
                and self.operator_tokens[tokens[i - 1]] == OperatorType.LOOP
            ):
                paren_depth -= 1

            if paren_depth == 0 and tokens[i] in self.operator_tokens:
                op_type = self.operator_tokens[tokens[i]]
                prec = self.precedence[op_type]

                # Use < to prefer rightmost operator of same precedence
                if prec < min_precedence:
                    min_precedence = prec
                    split_pos = i
                    split_op = op_type

        # If no operator found, this is a leaf node
        if split_pos == -1:
            return ParseNode(tokens=tokens[start:end], start_pos=start, end_pos=end)

        # For other operators, split into left and right
        node = ParseNode(operator=split_op, start_pos=start, end_pos=end)

        # Parse left side
        if split_pos > start:
            node.children.append(self._parse_expression(tokens, start, split_pos))

        # Parse right side
        if split_pos + 1 < end:
            node.children.append(self._parse_expression(tokens, split_pos + 1, end))

        return node


class ImprovedCompositionalExecutor:
    """Improved executor for parsed compositional expressions."""

    def __init__(self, model, vocab: Dict[str, int]):
        self.model = model
        self.vocab = vocab
        self.parser = ImprovedCompositionalParser(vocab)

        # Special tokens
        self.do_token = vocab.get("do", -1)
        self.twice_token = vocab.get("twice", -1)
        self.thrice_token = vocab.get("thrice", -1)

        # For choice operator - track previous choices for consistency
        self.choice_history = {}

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
        # Skip empty nodes
        if not node.tokens:
            return []

        # Check if this is just "do" - skip it
        if len(node.tokens) == 1 and node.tokens[0] == self.do_token:
            return []

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
        # The order matters - we preserve left-to-right execution
        outputs = []
        for child in node.children:
            child_outputs = self.execute(child, command_ids, bindings, stage)
            outputs.extend(child_outputs)
        return outputs

    def _execute_loop(
        self, node: ParseNode, command_ids: mx.array, bindings: Dict, stage: str
    ) -> List[mx.array]:
        """Execute loop (while operator)."""
        outputs = []

        if len(node.children) >= 2:
            # node.children[0] is condition
            # node.children[1] is body

            # Evaluate condition to determine loop count
            # For now, we'll check if condition contains "true" or a number
            condition_tokens = node.children[0].tokens
            loop_count = 3  # Default

            # Check for specific patterns in condition
            true_token = self.vocab.get("true", -1)
            if true_token in condition_tokens:
                loop_count = 3  # Standard interpretation of "while true"

            # Check for numbers in condition
            for token in condition_tokens:
                for word, idx in self.vocab.items():
                    if idx == token and word.isdigit():
                        loop_count = min(int(word), 5)  # Cap at 5 for safety
                        break

            # Execute body loop_count times
            body = node.children[1]
            for _ in range(loop_count):
                body_outputs = self.execute(body, command_ids, bindings, stage)
                outputs.extend(body_outputs)

        return outputs

    def _execute_choice(
        self, node: ParseNode, command_ids: mx.array, bindings: Dict, stage: str
    ) -> List[mx.array]:
        """Execute choice (or operator)."""
        if len(node.children) == 0:
            return []

        # Make choice deterministic based on command for consistency
        # Use hash of command to select which branch
        command_hash = hash(tuple(command_ids.flatten().tolist()))
        choice_idx = command_hash % len(node.children)

        # Store choice for debugging
        self.choice_history[command_hash] = choice_idx

        chosen_child = node.children[choice_idx]
        return self.execute(chosen_child, command_ids, bindings, stage)


def test_improved_parser():
    """Test the improved compositional parser."""
    from train_integrated_model import VOCAB

    # Ensure operators are in vocabulary
    operators = ["then", "and", "or", "while", "do", "true"]
    for op in operators:
        if op not in VOCAB:
            VOCAB[op] = len(VOCAB)

    parser = ImprovedCompositionalParser(VOCAB)

    test_cases = [
        "do X and Y",
        "do X then do Y",
        "do X or Y twice",
        "do X and Y then do Z",
        "while X do Y",
        "while true do X",
        "do X then do Y and Z",
        "do X and Y or Z",
        "do X then Y then Z",
    ]

    print("=== Testing Improved Compositional Parser ===")
    print(f"Vocabulary operators: {[(op, VOCAB.get(op)) for op in operators]}")
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
    test_improved_parser()
