#!/usr/bin/env python3
"""Improved Neural Memory Networks with compositional operator support."""

from utils.imports import setup_project_paths

setup_project_paths()

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

import mlx.core as mx
import mlx.nn as nn


class OperatorType(Enum):
    """Types of compositional operators."""

    AND = "and"
    THEN = "then"
    OR = "or"
    NONE = "none"


@dataclass
class ExecutionNode:
    """Node in execution plan."""

    operator: OperatorType
    variables: List[int]  # Variable token IDs
    children: List["ExecutionNode"] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []


class ImprovedMemoryNetwork(nn.Module):
    """Memory network with explicit compositional operator handling."""

    def __init__(
        self,
        vocab_size: int,
        num_actions: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_vars: int = 4,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.num_actions = num_actions
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_vars = num_vars

        # Vocabulary mappings
        self.vocab = {
            "do": 1,
            "means": 2,
            "is": 3,
            "and": 4,
            "or": 5,
            "then": 6,
            "twice": 7,
            "thrice": 8,
            "X": 10,
            "Y": 11,
            "Z": 12,
            "W": 13,
        }
        self.var_tokens = [10, 11, 12, 13]  # X, Y, Z, W

        # Embeddings
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)

        # Memory for variable bindings (fixed size)
        self.memory_keys = mx.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]  # X  # Y  # Z
        )  # W
        self.memory_values = mx.zeros((num_vars, hidden_dim))

        # Networks for encoding and decoding
        self.action_encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.action_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def reset_memory(self):
        """Reset memory to zeros."""
        self.memory_values = mx.zeros((self.num_vars, self.hidden_dim))

    def parse_execution_plan(
        self, tokens: mx.array
    ) -> Tuple[Dict[int, int], ExecutionNode]:
        """Parse tokens into bindings and execution plan.

        Returns:
            bindings: Dict mapping variable token to memory slot
            execution_root: Root of execution tree
        """
        bindings = {}
        token_list = tokens[0].tolist() if len(tokens.shape) > 1 else tokens.tolist()

        # Phase 1: Extract all bindings
        i = 0
        while i < len(token_list):
            token = token_list[i]

            # Check for binding pattern: VAR means/is ACTION
            if token in self.var_tokens and i + 2 < len(token_list):
                next_token = token_list[i + 1]
                if next_token in [self.vocab["means"], self.vocab["is"]]:
                    var_idx = self.var_tokens.index(token)
                    bindings[token] = var_idx
                    i += 3  # Skip VAR means ACTION
                    continue
            i += 1

        # Phase 2: Parse execution structure
        execution_root = self._parse_execution(token_list, bindings)

        return bindings, execution_root

    def _parse_execution(
        self, tokens: List[int], bindings: Dict[int, int]
    ) -> ExecutionNode:
        """Parse execution structure from tokens."""
        # Find all "do" positions
        do_positions = []
        for i, token in enumerate(tokens):
            if token == self.vocab["do"]:
                do_positions.append(i)

        if not do_positions:
            return ExecutionNode(OperatorType.NONE, [])

        # Simple parser for common patterns
        root = ExecutionNode(OperatorType.NONE, [])

        for do_pos in do_positions:
            # Look for variables after "do"
            vars_to_execute = []
            i = do_pos + 1

            while i < len(tokens):
                token = tokens[i]

                if token in bindings:
                    vars_to_execute.append(token)

                # Check for operators
                if token == self.vocab["and"]:
                    # Continue collecting variables
                    i += 1
                    continue
                elif token == self.vocab["then"]:
                    # Create node for current variables
                    if vars_to_execute:
                        if root.operator == OperatorType.NONE:
                            root = ExecutionNode(OperatorType.THEN, [])
                        node = ExecutionNode(OperatorType.AND, vars_to_execute)
                        root.children.append(node)
                        vars_to_execute = []
                    break
                elif token == self.vocab["or"]:
                    # For simplicity, treat OR as taking first variable
                    if vars_to_execute:
                        vars_to_execute = [vars_to_execute[0]]
                    break
                elif token not in bindings and token != self.vocab["and"]:
                    # End of this execution segment
                    break

                i += 1

            # Add remaining variables
            if vars_to_execute:
                if root.operator == OperatorType.NONE:
                    root = ExecutionNode(OperatorType.AND, vars_to_execute)
                else:
                    node = ExecutionNode(OperatorType.AND, vars_to_execute)
                    root.children.append(node)

        return root

    def execute_plan(
        self, plan: ExecutionNode, bindings: Dict[int, int]
    ) -> List[mx.array]:
        """Execute the plan and return action predictions."""
        outputs = []

        if plan.operator == OperatorType.NONE:
            return outputs

        # Handle different operators
        if plan.operator == OperatorType.AND:
            # Execute all variables in parallel
            for var_token in plan.variables:
                if var_token in bindings:
                    slot_idx = bindings[var_token]
                    value = self.memory_values[slot_idx]
                    action_logits = self.action_decoder(value)
                    outputs.append(action_logits)

        elif plan.operator == OperatorType.THEN:
            # Execute children sequentially
            for child in plan.children:
                child_outputs = self.execute_plan(child, bindings)
                outputs.extend(child_outputs)

        elif plan.operator == OperatorType.OR:
            # Execute first variable only
            if plan.variables:
                var_token = plan.variables[0]
                if var_token in bindings:
                    slot_idx = bindings[var_token]
                    value = self.memory_values[slot_idx]
                    action_logits = self.action_decoder(value)
                    outputs.append(action_logits)

        return outputs

    def forward(self, tokens: mx.array) -> mx.array:
        """Process command and return action predictions."""
        self.reset_memory()

        # Parse command
        bindings, execution_plan = self.parse_execution_plan(tokens)

        # Store bindings in memory
        token_list = tokens[0].tolist() if len(tokens.shape) > 1 else tokens.tolist()

        for i in range(len(token_list)):
            token = token_list[i]

            # Check for binding pattern
            if token in bindings and i + 2 < len(token_list):
                next_token = token_list[i + 1]
                if next_token in [self.vocab["means"], self.vocab["is"]]:
                    # Get action embedding
                    action_token = token_list[i + 2]
                    action_embed = self.token_embeddings(mx.array([action_token]))
                    action_value = self.action_encoder(action_embed[0])

                    # Store in memory
                    slot_idx = bindings[token]
                    self.memory_values = mx.array(
                        self.memory_values
                    )  # Ensure it's mutable
                    self.memory_values[slot_idx] = action_value

        # Execute plan
        outputs = self.execute_plan(execution_plan, bindings)

        if outputs:
            return mx.stack(outputs)
        else:
            return mx.zeros((1, self.num_actions))

    def __call__(self, inputs: Dict[str, mx.array]) -> mx.array:
        """Interface compatible with training code."""
        return self.forward(inputs["command"])


def test_improved_memory():
    """Test the improved memory network."""
    from progressive_complexity_dataset import ACTIONS, VOCAB

    print("Testing Improved Memory Network...")

    model = ImprovedMemoryNetwork(
        vocab_size=len(VOCAB), num_actions=len(ACTIONS), embed_dim=64, hidden_dim=128
    )

    # Test cases
    test_commands = [
        "X means jump do X",
        "X means jump Y means walk do X and Y",
        "X means jump Y means walk do Y then X",
        "X means jump do X then Y means walk do Y",
    ]

    for command in test_commands:
        # Tokenize
        tokens = []
        for word in command.split():
            tokens.append(VOCAB.get(word, 0))

        tokens_array = mx.array([tokens])

        # Forward pass
        outputs = model({"command": tokens_array})

        print(f"\nCommand: {command}")
        print(f"Output shape: {outputs.shape}")

        if outputs.shape[0] > 0:
            predictions = mx.argmax(outputs, axis=1).tolist()
            print(f"Predicted indices: {predictions}")
            print(f"Predicted actions: {[ACTIONS[i] for i in predictions]}")

    print("\nTest complete!")


if __name__ == "__main__":
    test_improved_memory()
