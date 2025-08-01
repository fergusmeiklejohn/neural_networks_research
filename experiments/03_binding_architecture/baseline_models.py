#!/usr/bin/env python3
"""Baseline models for variable binding comparison.

Implements several baseline approaches:
1. Standard LSTM/GRU model
2. Transformer-based model  
3. Simple feedforward model
4. Rule-based baseline
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


class LSTMBaseline(nn.Module):
    """Standard LSTM model for sequence-to-sequence prediction."""
    
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, 6)  # 6 action types
        
    def __call__(self, input_ids: mx.array) -> mx.array:
        # Embed inputs
        x = self.embedding(input_ids)
        
        # Run LSTM
        output, _ = self.lstm(x)
        
        # Project to action space
        logits = self.output_proj(output)
        
        return logits


class TransformerBaseline(nn.Module):
    """Transformer model for variable binding."""
    
    def __init__(self, vocab_size: int, embed_dim: int = 128, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(512, embed_dim)  # Max sequence length
        
        # Transformer layers
        self.layers = []
        for _ in range(num_layers):
            self.layers.append(nn.TransformerEncoderLayer(
                dims=embed_dim,
                num_heads=num_heads,
                mlp_dims=embed_dim * 4
            ))
        
        self.output_proj = nn.Linear(embed_dim, 6)
        
    def __call__(self, input_ids: mx.array) -> mx.array:
        batch_size, seq_len = input_ids.shape
        
        # Create position ids
        pos_ids = mx.broadcast_to(mx.arange(seq_len)[None, :], (batch_size, seq_len))
        
        # Embed tokens and positions
        x = self.embedding(input_ids) + self.pos_embedding(pos_ids)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, mask=None)
        
        # Project to actions
        logits = self.output_proj(x)
        
        return logits


class FeedforwardBaseline(nn.Module):
    """Simple feedforward model with fixed context window."""
    
    def __init__(self, vocab_size: int, context_window: int = 10, hidden_dim: int = 256):
        super().__init__()
        self.context_window = context_window
        self.embedding = nn.Embedding(vocab_size, 64)
        
        # Feedforward layers
        input_dim = context_window * 64
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6)
        )
        
    def __call__(self, input_ids: mx.array) -> mx.array:
        batch_size, seq_len = input_ids.shape
        
        # Embed inputs
        x = self.embedding(input_ids)
        
        # Process with sliding window
        outputs = []
        for i in range(seq_len):
            # Get context window
            start = max(0, i - self.context_window + 1)
            end = i + 1
            context = x[:, start:end]
            
            # Pad if needed
            if context.shape[1] < self.context_window:
                pad_len = self.context_window - context.shape[1]
                padding = mx.zeros((batch_size, pad_len, 64))
                context = mx.concatenate([padding, context], axis=1)
            
            # Flatten and process
            context_flat = context.reshape(batch_size, -1)
            output = self.mlp(context_flat)
            outputs.append(output[:, None, :])
        
        return mx.concatenate(outputs, axis=1)


class RuleBasedBaseline:
    """Rule-based baseline using pattern matching."""
    
    def __init__(self, vocab: Dict[str, int]):
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        
        # Action mappings
        self.action_map = {
            'jump': 0, 'JUMP': 0,
            'walk': 1, 'WALK': 1,
            'turn': 2, 'TURN': 2,
            'look': 3, 'LOOK': 3,
            'run': 4, 'RUN': 4,
            'stop': 5, 'STOP': 5
        }
        
    def predict(self, input_ids: mx.array) -> mx.array:
        """Predict actions using rules."""
        batch_size, seq_len = input_ids.shape
        predictions = mx.zeros((batch_size, seq_len, 6))
        
        for b in range(batch_size):
            # Convert to tokens
            tokens = [self.inv_vocab.get(int(input_ids[b, i]), '<UNK>') 
                     for i in range(seq_len)]
            
            # Track variable bindings
            bindings = {}
            
            # Find patterns
            i = 0
            while i < len(tokens):
                # Pattern: "X means ACTION"
                if i + 2 < len(tokens) and tokens[i + 1] == 'means':
                    var = tokens[i]
                    action = tokens[i + 2]
                    if action in self.action_map:
                        bindings[var] = self.action_map[action]
                    i += 3
                    
                # Pattern: "do VAR"
                elif tokens[i] == 'do' and i + 1 < len(tokens):
                    var = tokens[i + 1]
                    if var in bindings:
                        # Set prediction at action position
                        predictions[b, i + 1, bindings[var]] = 1.0
                        
                    # Handle "twice"/"thrice"
                    if i + 2 < len(tokens):
                        if tokens[i + 2] == 'twice' and var in bindings:
                            predictions[b, i + 2, bindings[var]] = 1.0
                        elif tokens[i + 2] == 'thrice' and var in bindings:
                            predictions[b, i + 2, bindings[var]] = 1.0
                            # Add third repetition (heuristic placement)
                            if i + 3 < len(tokens):
                                predictions[b, i + 3, bindings[var]] = 1.0
                    i += 1
                else:
                    i += 1
        
        return predictions


def create_baseline_model(model_type: str, vocab_size: int, **kwargs) -> nn.Module:
    """Factory function to create baseline models.
    
    Args:
        model_type: One of 'lstm', 'transformer', 'feedforward'
        vocab_size: Vocabulary size
        **kwargs: Additional model-specific arguments
        
    Returns:
        Initialized model
    """
    if model_type == 'lstm':
        return LSTMBaseline(vocab_size, **kwargs)
    elif model_type == 'transformer':
        return TransformerBaseline(vocab_size, **kwargs)
    elif model_type == 'feedforward':
        return FeedforwardBaseline(vocab_size, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def test_baselines():
    """Test baseline models."""
    vocab_size = 50
    batch_size = 2
    seq_len = 20
    
    # Create dummy input
    input_ids = mx.random.randint(0, vocab_size, (batch_size, seq_len))
    
    print("Testing baseline models...")
    
    # Test LSTM
    print("\nLSTM Baseline:")
    lstm = LSTMBaseline(vocab_size)
    output = lstm(input_ids)
    print(f"Output shape: {output.shape}")
    
    # Test Transformer
    print("\nTransformer Baseline:")
    transformer = TransformerBaseline(vocab_size)
    output = transformer(input_ids)
    print(f"Output shape: {output.shape}")
    
    # Test Feedforward
    print("\nFeedforward Baseline:")
    ff = FeedforwardBaseline(vocab_size)
    output = ff(input_ids)
    print(f"Output shape: {output.shape}")
    
    # Test Rule-based
    print("\nRule-based Baseline:")
    vocab = {f'token_{i}': i for i in range(vocab_size)}
    vocab.update({'X': 10, 'Y': 11, 'means': 12, 'do': 13, 'jump': 14, 'walk': 15, 'twice': 16})
    
    rule_based = RuleBasedBaseline(vocab)
    # Create test pattern: "X means jump do X twice"
    test_input = mx.array([[10, 12, 14, 13, 10, 16] + [0] * 14])  # X means jump do X twice
    output = rule_based.predict(test_input)
    print(f"Output shape: {output.shape}")
    print(f"Predictions at key positions: {output[0, 4:7]}")  # Should predict jump at positions 4 and 5
    
    print("\n✓ All baselines tested successfully!")


if __name__ == "__main__":
    test_baselines()