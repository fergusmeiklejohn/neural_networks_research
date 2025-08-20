#!/usr/bin/env python3
"""Rule-based binding extractor for Two-Stage Compiler.

This module extracts variable bindings explicitly and creates an execution plan,
separating discrete rule extraction from continuous neural execution.
"""

from utils.imports import setup_project_paths

setup_project_paths()

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import mlx.core as mx

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OperatorType(Enum):
    """Types of compositional operators."""

    AND = "AND"
    OR = "OR"
    WHILE = "WHILE"
    THEN = "THEN"


@dataclass
class ExecutionNode:
    """Node in execution tree."""

    node_type: str  # "variable", "operator", "modifier"
    value: Union[str, OperatorType]  # Variable name or operator type
    children: List["ExecutionNode"] = None
    modifier: Optional[str] = None  # For 'twice', 'thrice'

    def __post_init__(self):
        if self.children is None:
            self.children = []

    def is_leaf(self) -> bool:
        return self.node_type == "variable"

    def __repr__(self) -> str:
        if self.is_leaf():
            mod_str = f" {self.modifier}" if self.modifier else ""
            return f"Var({self.value}{mod_str})"
        else:
            return f"Op({self.value.value})"


@dataclass
class BindingEntry:
    """Single binding with temporal information."""

    variable: str
    action: str
    start_pos: int  # When this binding becomes active
    end_pos: Optional[
        int
    ] = None  # When this binding is replaced (None if still active)


class RuleBasedBindingExtractor:
    """Extracts bindings and execution plan from tokenized commands."""

    def __init__(self, vocab: Dict[str, int]):
        self.vocab = vocab
        self.reverse_vocab = {v: k for k, v in vocab.items()}

        # Key tokens
        self.do_token = vocab.get("do", -1)
        self.means_token = vocab.get("means", -1)
        self.while_token = vocab.get("while", -1)
        self.true_token = vocab.get("true", -1)
        self.twice_token = vocab.get("twice", -1)
        self.thrice_token = vocab.get("thrice", -1)

        # Operator tokens
        self.operator_tokens = {
            vocab.get("and", -1): OperatorType.AND,
            vocab.get("or", -1): OperatorType.OR,
            vocab.get("while", -1): OperatorType.WHILE,
            vocab.get("then", -1): OperatorType.THEN,
        }

        # Variable tokens
        self.variable_tokens = {
            vocab.get("X", -1): "X",
            vocab.get("Y", -1): "Y",
            vocab.get("Z", -1): "Z",
            vocab.get("W", -1): "W",
        }

        # Action tokens
        self.action_tokens = {
            vocab.get("jump", -1): "JUMP",
            vocab.get("walk", -1): "WALK",
            vocab.get("run", -1): "RUN",
            vocab.get("turn", -1): "TURN",
        }

    def extract(
        self, tokens: mx.array
    ) -> Tuple[Dict[str, List[BindingEntry]], ExecutionNode]:
        """Extract bindings and execution plan from tokens.

        Returns:
            - bindings: Dict mapping variable names to list of temporal bindings
            - execution_plan: Tree structure representing execution order
        """
        token_list = self._tokens_to_list(tokens)
        logger.debug(f"Extracting from: {self._tokens_to_words(token_list)}")

        # Phase 1: Extract all bindings with temporal information
        bindings = self._extract_temporal_bindings(token_list)

        # Phase 2: Extract execution plan
        execution_plan = self._extract_execution_plan(token_list)

        return bindings, execution_plan

    def _tokens_to_list(self, tokens: mx.array) -> List[int]:
        """Convert MLX array to Python list."""
        if len(tokens.shape) > 1:
            tokens = tokens[0]
        import numpy as np

        token_array = np.array(tokens)
        if len(token_array.shape) > 1:
            token_array = token_array.flatten()
        return [int(t) for t in token_array]

    def _tokens_to_words(self, token_list: List[int]) -> str:
        """Convert tokens back to words for debugging."""
        return " ".join(self.reverse_vocab.get(t, f"?{t}") for t in token_list)

    def _extract_temporal_bindings(
        self, token_list: List[int]
    ) -> Dict[str, List[BindingEntry]]:
        """Extract all bindings with temporal information."""
        bindings = {}
        i = 0

        while i < len(token_list):
            # Look for "variable means action" pattern
            if i + 2 < len(token_list) and token_list[i + 1] == self.means_token:
                var_token = token_list[i]
                action_token = token_list[i + 2]

                if (
                    var_token in self.variable_tokens
                    and action_token in self.action_tokens
                ):
                    var_name = self.variable_tokens[var_token]
                    action_name = self.action_tokens[action_token]

                    # Create binding entry
                    entry = BindingEntry(
                        variable=var_name, action=action_name, start_pos=i
                    )

                    # Check if this variable was already bound (rebinding)
                    if var_name in bindings:
                        # Mark the previous binding as ended
                        bindings[var_name][-1].end_pos = i
                        bindings[var_name].append(entry)
                    else:
                        bindings[var_name] = [entry]

                    i += 3
                else:
                    i += 1
            else:
                i += 1

        return bindings

    def _extract_execution_plan(self, token_list: List[int]) -> Optional[ExecutionNode]:
        """Extract execution plan from token list."""
        # Find all execution segments (starting with 'do' or 'while')
        segments = []
        i = 0

        while i < len(token_list):
            if token_list[i] == self.do_token:
                # Find the end of this execution segment
                start = i + 1
                end = self._find_segment_end(token_list, start)
                if end > start:
                    segments.append((start, end, "do"))
                i = end
            elif token_list[i] == self.while_token:
                # Handle "while true do X" pattern
                if (
                    i + 3 < len(token_list)
                    and token_list[i + 1] == self.true_token
                    and token_list[i + 2] == self.do_token
                ):
                    start = i + 3
                    end = self._find_segment_end(token_list, start)
                    if end > start:
                        segments.append((start, end, "while"))
                    i = end
                else:
                    i += 1
            else:
                i += 1

        if not segments:
            return None

        # Parse each segment and connect them
        nodes = []
        for start, end, seg_type in segments:
            node = self._parse_segment(token_list, start, end)
            if seg_type == "while" and node:
                # Wrap in while operator
                node = ExecutionNode(
                    node_type="operator", value=OperatorType.WHILE, children=[node]
                )
            if node:
                nodes.append(node)

        # Connect segments with THEN
        if len(nodes) == 0:
            return None
        elif len(nodes) == 1:
            return nodes[0]
        else:
            result = nodes[0]
            for node in nodes[1:]:
                result = ExecutionNode(
                    node_type="operator",
                    value=OperatorType.THEN,
                    children=[result, node],
                )
            return result

    def _find_segment_end(self, token_list: List[int], start: int) -> int:
        """Find the end of an execution segment."""
        i = start
        while i < len(token_list):
            # Stop at next 'do' or 'means' (new binding)
            if token_list[i] == self.do_token:
                return i
            if i + 1 < len(token_list) and token_list[i + 1] == self.means_token:
                return i
            i += 1
        return len(token_list)

    def _parse_segment(
        self, token_list: List[int], start: int, end: int
    ) -> Optional[ExecutionNode]:
        """Parse a single execution segment."""
        # Find operators in this segment
        operators = []
        for i in range(start, end):
            if token_list[i] in self.operator_tokens:
                operators.append((i, self.operator_tokens[token_list[i]]))

        if not operators:
            # No operators - parse as simple variable reference
            return self._parse_simple_segment(token_list, start, end)
        else:
            # Has operators - parse recursively
            # For now, handle simple cases (can be extended)
            if len(operators) == 1:
                op_pos, op_type = operators[0]
                left = self._parse_segment(token_list, start, op_pos)
                right = self._parse_segment(token_list, op_pos + 1, end)
                if left and right:
                    return ExecutionNode(
                        node_type="operator", value=op_type, children=[left, right]
                    )
            # Handle multiple operators with precedence if needed

        return None

    def _parse_simple_segment(
        self, token_list: List[int], start: int, end: int
    ) -> Optional[ExecutionNode]:
        """Parse a segment without operators."""
        # Look for variable and optional modifier
        var_node = None
        modifier = None

        for i in range(start, end):
            token = token_list[i]
            if token in self.variable_tokens:
                var_name = self.variable_tokens[token]
                var_node = ExecutionNode(node_type="variable", value=var_name)
            elif token == self.twice_token:
                modifier = "twice"
            elif token == self.thrice_token:
                modifier = "thrice"

        if var_node:
            var_node.modifier = modifier
            return var_node

        return None

    def get_active_binding(
        self, variable: str, position: int, bindings: Dict[str, List[BindingEntry]]
    ) -> Optional[str]:
        """Get the active binding for a variable at a given position."""
        if variable not in bindings:
            return None

        for entry in bindings[variable]:
            if entry.start_pos <= position:
                if entry.end_pos is None or position < entry.end_pos:
                    return entry.action

        return None


def test_binding_extractor():
    """Test the binding extractor with various cases."""
    print("=== Testing Rule-Based Binding Extractor ===\n")

    # Define vocab
    VOCAB = {
        "PAD": 0,
        "do": 1,
        "means": 2,
        "and": 3,
        "or": 4,
        "then": 5,
        "twice": 6,
        "thrice": 7,
        "while": 8,
        "true": 9,
        "X": 10,
        "Y": 11,
        "Z": 12,
        "W": 13,
        "jump": 14,
        "walk": 15,
        "run": 16,
        "turn": 17,
    }

    extractor = RuleBasedBindingExtractor(VOCAB)

    test_cases = [
        "X means jump do X",
        "X means jump Y means walk do X and Y",
        "X means jump do X then X means walk do X",
        "X means jump do X twice",
        "X means jump while true do X",
        "X means jump do X then Y means walk do Y and X",
    ]

    for command in test_cases:
        print(f"Command: {command}")

        # Tokenize
        tokens = [VOCAB.get(word, VOCAB["PAD"]) for word in command.split()]
        tokens_mx = mx.array([tokens])

        # Extract
        bindings, execution = extractor.extract(tokens_mx)

        # Display bindings
        print("Bindings:")
        for var, entries in bindings.items():
            for entry in entries:
                end_str = f"-{entry.end_pos}" if entry.end_pos else "-∞"
                print(f"  {var} -> {entry.action} (pos {entry.start_pos}{end_str})")

        # Display execution plan
        print(f"Execution: {execution}")
        print("-" * 60)


if __name__ == "__main__":
    test_binding_extractor()
