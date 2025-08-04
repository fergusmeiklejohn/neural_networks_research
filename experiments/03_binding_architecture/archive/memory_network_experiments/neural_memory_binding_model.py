#!/usr/bin/env python3
"""Differentiable Neural Memory Networks for explicit variable binding.

This model implements explicit key-value memory for variable bindings,
making the binding operation a first-class differentiable operation
rather than an implicit side effect.
"""

from utils.imports import setup_project_paths
setup_project_paths()

from utils.config import setup_environment
config = setup_environment()

import mlx.core as mx
import mlx.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np


class NeuralMemoryBinding(nn.Module):
    """Explicit key-value memory for variable bindings.
    
    This module maintains fixed slots for variables (X, Y, Z, W) and provides
    differentiable read/write operations for binding and retrieval.
    """
    
    def __init__(self, num_vars: int = 4, key_dim: int = 64, value_dim: int = 256):
        super().__init__()
        self.num_vars = num_vars
        self.key_dim = key_dim
        self.value_dim = value_dim
        
        # Fixed keys for variables (learnable for better matching)
        self.memory_keys = nn.Embedding(num_vars, key_dim)
        
        # Memory values (will be updated during forward pass)
        # Initialize as parameters but treat as buffers
        self.register_buffer('memory_values', mx.zeros((num_vars, value_dim)))
        
        # Write gate network - decides when to write to memory
        self.write_gate = nn.Sequential(
            nn.Linear(value_dim * 2, 128),  # Current value + existing value
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Read attention - for soft reading from memory
        self.read_attention = nn.Linear(key_dim, key_dim)
        
    def register_buffer(self, name: str, tensor: mx.array):
        """Register a buffer (non-parameter tensor)."""
        setattr(self, name, tensor)
        
    def write(self, var_idx: int, value: mx.array) -> mx.array:
        """Write a value to memory slot with differentiable gating.
        
        Args:
            var_idx: Variable index (0-3 for X, Y, Z, W)
            value: Value embedding to store (can be batched)
            
        Returns:
            Write gate value (for regularization)
        """
        # Handle batched values
        if len(value.shape) == 2:
            # Take first element of batch for simplicity
            value = value[0:1]
        
        # Get existing value
        existing = self.memory_values[var_idx:var_idx+1]
        
        # Compute write gate
        combined = mx.concatenate([value, existing], axis=-1)
        gate = self.write_gate(combined)
        
        # Update memory with gating
        new_value = gate * value + (1 - gate) * existing
        
        # Update memory values
        mask = mx.arange(self.num_vars) == var_idx
        mask = mask[:, None]  # Shape: (num_vars, 1)
        self.memory_values = mx.where(mask, new_value, self.memory_values)
        
        return gate
        
    def read(self, query: mx.array) -> Tuple[mx.array, mx.array]:
        """Read from memory using attention mechanism.
        
        Args:
            query: Query embedding
            
        Returns:
            value: Retrieved value
            attention_weights: Attention distribution over memory slots
        """
        # Transform query
        query_transformed = self.read_attention(query)
        
        # Compute attention scores with all keys
        keys = self.memory_keys.weight  # Shape: (num_vars, key_dim)
        scores = mx.matmul(query_transformed, keys.T) / mx.sqrt(float(self.key_dim))
        attention_weights = mx.softmax(scores, axis=-1)
        
        # Read value as weighted sum
        if len(attention_weights.shape) == 1:
            attention_weights = attention_weights[None, :]
        value = mx.matmul(attention_weights, self.memory_values)
        
        return value, attention_weights
        
    def reset(self):
        """Reset memory to zeros."""
        self.memory_values = mx.zeros((self.num_vars, self.value_dim))


class MemoryBasedBindingModel(nn.Module):
    """Complete model using explicit memory for variable binding."""
    
    def __init__(self, vocab_size: int, num_actions: int, embed_dim: int = 256,
                 hidden_dim: int = 512, num_vars: int = 4):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.num_actions = num_actions
        self.embed_dim = embed_dim
        self.num_vars = num_vars
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        
        # Sequential processor (LSTM for maintaining context)
        self.lstm = nn.LSTM(embed_dim, hidden_dim)
        
        # Memory module
        self.memory = NeuralMemoryBinding(num_vars, key_dim=hidden_dim//2, value_dim=hidden_dim)
        
        # Variable encoder - maps variable tokens to queries
        self.var_encoder = nn.Linear(embed_dim, hidden_dim//2)
        
        # Action encoder - maps action tokens to values for storage
        self.action_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Action decoder - maps retrieved values to action predictions
        self.action_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
        
        # Pattern detection layers
        self.binding_detector = nn.Linear(hidden_dim, 1)  # Detects "X means Y"
        self.execution_detector = nn.Linear(hidden_dim, 1)  # Detects "do X"
        
    def detect_binding_pattern(self, tokens: mx.array, hidden_states: mx.array, 
                             position: int) -> Tuple[bool, Optional[int], Optional[int]]:
        """Detect if current position is part of a binding pattern.
        
        Returns:
            is_binding: Whether this is a binding pattern
            var_idx: Variable index if binding
            action_pos: Position of action token if binding
        """
        if position + 2 >= tokens.shape[1]:
            return False, None, None
            
        # Use hidden state to detect binding pattern
        score = mx.sigmoid(self.binding_detector(hidden_states[:, position]))
        is_binding = score.item() > 0.5
        
        if is_binding:
            # Assume pattern is "VAR means/is ACTION"
            var_token = tokens[0, position].item()
            var_idx = self._token_to_var_idx(var_token)
            if var_idx is not None:
                return True, var_idx, position + 2
                
        return False, None, None
        
    def detect_execution_pattern(self, tokens: mx.array, hidden_states: mx.array,
                               position: int) -> Tuple[bool, Optional[int]]:
        """Detect if current position is an execution pattern.
        
        Returns:
            is_execution: Whether this is an execution pattern
            var_idx: Variable index if execution
        """
        if position >= tokens.shape[1]:
            return False, None
            
        # Use hidden state to detect execution pattern
        score = mx.sigmoid(self.execution_detector(hidden_states[:, position]))
        is_execution = score.item() > 0.5
        
        if is_execution:
            var_token = tokens[0, position].item()
            var_idx = self._token_to_var_idx(var_token)
            if var_idx is not None:
                return True, var_idx
                
        return False, None
        
    def _token_to_var_idx(self, token: int) -> Optional[int]:
        """Map token to variable index."""
        # This should be configured based on vocabulary
        # For now, assume X=0, Y=1, Z=2, W=3
        var_map = {10: 0, 11: 1, 12: 2, 13: 3}  # Placeholder mapping
        return var_map.get(token)
        
    def forward(self, tokens: mx.array) -> List[mx.array]:
        """Process command and return action predictions.
        
        Args:
            tokens: Input token sequence (batch_size, seq_len)
            
        Returns:
            List of action predictions
        """
        batch_size, seq_len = tokens.shape
        outputs = []
        
        # Reset memory for new sequence
        self.memory.reset()
        
        # Embed tokens
        embeddings = self.token_embeddings(tokens)
        
        # Process sequence with LSTM (MLX expects seq_len first)
        # Transpose to (seq_len, batch_size, embed_dim)
        embeddings_t = mx.transpose(embeddings, (1, 0, 2))
        hidden_states_t, _ = self.lstm(embeddings_t)
        # Transpose back to (batch_size, seq_len, hidden_dim)
        hidden_states = mx.transpose(hidden_states_t, (1, 0, 2))
        
        # Process each position
        for t in range(seq_len):
            # Check for binding pattern
            is_binding, var_idx, action_pos = self.detect_binding_pattern(
                tokens, hidden_states, t
            )
            
            if is_binding and action_pos is not None:
                # Encode action and store in memory
                action_hidden = hidden_states[:, action_pos]
                action_value = self.action_encoder(action_hidden)
                self.memory.write(var_idx, action_value)
                
            # Check for execution pattern
            is_execution, var_idx = self.detect_execution_pattern(
                tokens, hidden_states, t
            )
            
            if is_execution and var_idx is not None:
                # Read from memory and predict action
                var_embedding = embeddings[:, t]
                query = self.var_encoder(var_embedding)
                value, _ = self.memory.read(query)
                # Handle batch dimension properly
                if len(value.shape) == 3:
                    value = value[0]  # Take first batch element
                elif len(value.shape) == 2 and value.shape[0] == 1:
                    value = value.squeeze(0)
                action_logits = self.action_decoder(value)
                outputs.append(action_logits)
                
        return outputs
        
    def __call__(self, inputs: Dict[str, mx.array]) -> mx.array:
        """Interface compatible with existing training code."""
        tokens = inputs['command']
        if len(tokens.shape) == 1:
            tokens = tokens[None, :]
            
        outputs = self.forward(tokens)
        
        if outputs:
            return mx.stack(outputs)
        else:
            # Return dummy output if no actions
            return mx.zeros((1, self.num_actions))


def test_memory_module():
    """Test the memory module in isolation."""
    print("Testing NeuralMemoryBinding module...")
    
    # Create memory
    memory = NeuralMemoryBinding(num_vars=4, key_dim=32, value_dim=64)
    
    # Test write
    test_value = mx.random.normal((1, 64))
    gate = memory.write(0, test_value)  # Write to X (index 0)
    print(f"Write gate value: {gate.item():.3f}")
    
    # Test read
    query = mx.random.normal((1, 32))
    value, attention = memory.read(query)
    print(f"Attention weights: {attention}")
    print(f"Retrieved value shape: {value.shape}")
    
    print("Memory module test passed!")


if __name__ == "__main__":
    test_memory_module()