#!/usr/bin/env python3
"""Binding-aware transformer for neural execution in Two-Stage Compiler.

This module implements a transformer that takes explicit bindings as input,
focusing learning on compositional execution rather than binding discovery.
"""

from utils.imports import setup_project_paths

setup_project_paths()

import logging
from typing import Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from rule_based_binding_extractor import BindingEntry, ExecutionNode, OperatorType

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BindingEmbedding(nn.Module):
    """Embeds binding information for use in transformer."""

    def __init__(self, num_variables: int, num_actions: int, hidden_dim: int):
        super().__init__()
        self.num_variables = num_variables
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim

        # Embeddings for variables and actions
        self.var_embedding = nn.Embedding(num_variables, hidden_dim)
        self.action_embedding = nn.Embedding(num_actions, hidden_dim)

        # Projection to combine variable and action
        self.binding_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def __call__(
        self,
        bindings: Dict[str, str],
        var_to_idx: Dict[str, int],
        action_to_idx: Dict[str, int],
    ) -> mx.array:
        """Convert bindings dict to embedding matrix.

        Returns:
            mx.array of shape [num_bindings, hidden_dim]
        """
        if not bindings:
            # Return empty tensor if no bindings
            return mx.zeros((0, self.hidden_dim))

        binding_embeds = []

        for var_name, action_name in bindings.items():
            # Get indices
            var_idx = var_to_idx.get(var_name, 0)
            action_idx = action_to_idx.get(action_name, 0)

            # Get embeddings
            var_embed = self.var_embedding(mx.array(var_idx))
            action_embed = self.action_embedding(mx.array(action_idx))

            # Combine
            combined = mx.concatenate([var_embed, action_embed], axis=-1)
            binding_embed = self.binding_proj(combined)

            binding_embeds.append(binding_embed)

        return mx.stack(binding_embeds)


class ExecutionTreeEncoder(nn.Module):
    """Encodes the execution tree structure."""

    def __init__(self, hidden_dim: int, num_operators: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Embeddings for operators
        self.operator_embedding = nn.Embedding(num_operators, hidden_dim)

        # TreeLSTM-like combination for operators
        self.combine_left = nn.Linear(hidden_dim, hidden_dim)
        self.combine_right = nn.Linear(hidden_dim, hidden_dim)
        self.combine_op = nn.Linear(hidden_dim, hidden_dim)
        self.combine_proj = nn.Linear(hidden_dim * 3, hidden_dim)

    def __call__(
        self,
        node: ExecutionNode,
        var_embeddings: Dict[str, mx.array],
        op_to_idx: Dict[OperatorType, int],
    ) -> mx.array:
        """Encode execution tree recursively."""
        if node.is_leaf():
            # Return variable embedding
            return var_embeddings.get(node.value, mx.zeros(self.hidden_dim))
        else:
            # Encode operator node
            op_idx = op_to_idx.get(node.value, 0)
            op_embed = self.operator_embedding(mx.array(op_idx))

            # Encode children
            if len(node.children) == 1:
                # Unary operator (WHILE)
                child_embed = self(node.children[0], var_embeddings, op_to_idx)
                combined = mx.concatenate(
                    [
                        op_embed,
                        child_embed,
                        mx.zeros(self.hidden_dim),  # Padding for consistent size
                    ],
                    axis=-1,
                )
            elif len(node.children) == 2:
                # Binary operator
                left_embed = self(node.children[0], var_embeddings, op_to_idx)
                right_embed = self(node.children[1], var_embeddings, op_to_idx)

                # Combine with operator-specific transformations
                left_transformed = self.combine_left(left_embed)
                right_transformed = self.combine_right(right_embed)
                op_transformed = self.combine_op(op_embed)

                combined = mx.concatenate(
                    [op_transformed, left_transformed, right_transformed], axis=-1
                )
            else:
                # Shouldn't happen with our operators
                combined = mx.concatenate(
                    [op_embed, mx.zeros(self.hidden_dim), mx.zeros(self.hidden_dim)],
                    axis=-1,
                )

            return mx.tanh(self.combine_proj(combined))


class BindingAwareTransformer(nn.Module):
    """Transformer that uses explicit bindings for execution."""

    def __init__(
        self,
        vocab_size: int,
        num_actions: int,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(256, hidden_dim)  # Max seq length

        # Binding embeddings
        self.binding_embedding = BindingEmbedding(
            num_variables=4,  # X, Y, Z, W
            num_actions=num_actions,
            hidden_dim=hidden_dim,
        )

        # Execution tree encoder
        self.tree_encoder = ExecutionTreeEncoder(
            hidden_dim=hidden_dim, num_operators=4  # AND, OR, THEN, WHILE
        )

        # Transformer layers with cross-attention to bindings
        self.layers = []
        for _ in range(num_layers):
            self.layers.append(
                TransformerLayerWithBindings(hidden_dim=hidden_dim, num_heads=num_heads)
            )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, num_actions)

        # Learnable tokens for special purposes
        self.cls_token = mx.random.normal((1, hidden_dim))

    def __call__(
        self,
        tokens: mx.array,
        bindings: Dict[str, List[BindingEntry]],
        execution_tree: Optional[ExecutionNode],
    ) -> mx.array:
        """Forward pass using tokens, bindings, and execution tree.

        Args:
            tokens: Input token ids [batch_size, seq_len]
            bindings: Variable bindings with temporal info
            execution_tree: Parse tree for execution

        Returns:
            Action predictions [num_actions, num_actions]
        """
        tokens.shape[0]
        seq_len = tokens.shape[1]

        # Embed tokens
        token_embeds = self.token_embedding(tokens)
        pos_ids = mx.arange(seq_len)
        pos_embeds = self.position_embedding(pos_ids)

        # Add positional embeddings
        x = token_embeds + pos_embeds[None, :, :]

        # Get current bindings (latest for each variable)
        current_bindings = {}
        for var_name, entries in bindings.items():
            if entries:
                current_bindings[var_name] = entries[-1].action

        # Embed bindings
        var_to_idx = {"X": 0, "Y": 1, "Z": 2, "W": 3}
        action_to_idx = {"JUMP": 0, "WALK": 1, "RUN": 2, "TURN": 3}
        binding_embeds = self.binding_embedding(
            current_bindings, var_to_idx, action_to_idx
        )

        # Apply transformer layers with binding attention
        for layer in self.layers:
            x = layer(x, binding_embeds)

        # Execute based on tree structure
        if execution_tree:
            outputs = self._execute_tree(
                execution_tree, x, current_bindings, action_to_idx
            )
            return outputs
        else:
            # No execution - return empty
            return mx.zeros((0, self.num_actions))

    def _execute_tree(
        self,
        node: ExecutionNode,
        hidden_states: mx.array,
        bindings: Dict[str, str],
        action_to_idx: Dict[str, int],
    ) -> mx.array:
        """Execute the tree and produce action outputs."""
        outputs = []
        self._execute_node(node, hidden_states, bindings, action_to_idx, outputs)

        if outputs:
            return mx.stack(outputs)
        else:
            return mx.zeros((0, self.num_actions))

    def _execute_node(
        self,
        node: ExecutionNode,
        hidden_states: mx.array,
        bindings: Dict[str, str],
        action_to_idx: Dict[str, int],
        outputs: List[mx.array],
    ):
        """Recursively execute nodes."""
        if node.is_leaf():
            # Execute variable
            if node.value in bindings:
                action_name = bindings[node.value]
                action_idx = action_to_idx.get(action_name, 0)

                # Create one-hot action vector
                action_vec = mx.zeros(self.num_actions)
                action_vec = mx.where(
                    mx.arange(self.num_actions) == action_idx, 1.0, action_vec
                )

                # Apply modifier
                if node.modifier == "twice":
                    outputs.extend([action_vec, action_vec])
                elif node.modifier == "thrice":
                    outputs.extend([action_vec, action_vec, action_vec])
                else:
                    outputs.append(action_vec)
        else:
            # Execute operator
            if node.value == OperatorType.AND:
                # Execute both children
                self._execute_node(
                    node.children[0], hidden_states, bindings, action_to_idx, outputs
                )
                self._execute_node(
                    node.children[1], hidden_states, bindings, action_to_idx, outputs
                )

            elif node.value == OperatorType.OR:
                # Execute first child only
                self._execute_node(
                    node.children[0], hidden_states, bindings, action_to_idx, outputs
                )

            elif node.value == OperatorType.THEN:
                # Execute children in sequence
                self._execute_node(
                    node.children[0], hidden_states, bindings, action_to_idx, outputs
                )
                self._execute_node(
                    node.children[1], hidden_states, bindings, action_to_idx, outputs
                )

            elif node.value == OperatorType.WHILE:
                # Execute child 3 times (simplified)
                for _ in range(3):
                    self._execute_node(
                        node.children[0],
                        hidden_states,
                        bindings,
                        action_to_idx,
                        outputs,
                    )


class TransformerLayerWithBindings(nn.Module):
    """Transformer layer with cross-attention to bindings."""

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Self-attention
        self.self_attn = nn.MultiHeadAttention(hidden_dim, num_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)

        # Cross-attention to bindings
        self.cross_attn = nn.MultiHeadAttention(hidden_dim, num_heads)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm3 = nn.LayerNorm(hidden_dim)

    def __call__(self, x: mx.array, binding_embeds: mx.array) -> mx.array:
        """Forward pass with binding attention."""
        # Self-attention
        attn_out = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)

        # Cross-attention to bindings (if any)
        if binding_embeds.shape[0] > 0:
            # Expand binding embeds to match batch dimension
            batch_size = x.shape[0]
            binding_embeds_expanded = mx.expand_dims(binding_embeds, axis=0)
            binding_embeds_expanded = mx.repeat(
                binding_embeds_expanded, batch_size, axis=0
            )

            cross_out = self.cross_attn(
                x, binding_embeds_expanded, binding_embeds_expanded
            )
            x = self.norm2(x + cross_out)

        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm3(x + ff_out)

        return x


def test_binding_aware_transformer():
    """Test the binding-aware transformer."""
    print("=== Testing Binding-Aware Transformer ===\n")

    # Initialize transformer
    transformer = BindingAwareTransformer(
        vocab_size=20, num_actions=4, hidden_dim=64, num_heads=4, num_layers=2
    )

    # Test with simple binding
    tokens = mx.array([[10, 2, 14, 1, 10]])  # "X means jump do X"

    # Create bindings
    bindings = {"X": [BindingEntry(variable="X", action="JUMP", start_pos=0)]}

    # Create execution tree
    from rule_based_binding_extractor import ExecutionNode

    exec_tree = ExecutionNode(node_type="variable", value="X")

    # Forward pass
    outputs = transformer(tokens, bindings, exec_tree)

    print(f"Input shape: {tokens.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Output: {outputs}")

    # Check if output is correct (should be one-hot for JUMP)
    action_probs = mx.softmax(outputs, axis=-1)
    predicted_action = mx.argmax(action_probs, axis=-1)
    print(f"Predicted action index: {predicted_action}")
    print(f"Expected: 0 (JUMP)")


if __name__ == "__main__":
    test_binding_aware_transformer()
