#!/usr/bin/env python3
"""
Improved Neural Network Models for Compositional Language Distribution Invention

Key improvements over v1:
1. Explicit gating mechanism for selective rule modification
2. Stronger modification signal propagation (concatenated to all layers)
3. Memory component to maintain base knowledge
4. Residual connections to prevent catastrophic forgetting
"""

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Dict, List, Tuple, Optional
import json


class PositionalEncoding(keras.layers.Layer):
    """Add positional encoding to embeddings"""
    
    def __init__(self, max_length=100, d_model=256):
        super().__init__()
        self.max_length = max_length
        self.d_model = d_model
        
        # Create positional encoding matrix
        position = np.arange(max_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pos_encoding = np.zeros((max_length, d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        self.pos_encoding = self.add_weight(
            name='pos_encoding',
            shape=(1, max_length, d_model),
            initializer=keras.initializers.Constant(pos_encoding[np.newaxis, ...]),
            trainable=False
        )
        
    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]


class GatedModificationLayer(keras.layers.Layer):
    """
    Gated modification layer that selectively applies modifications.
    
    Key innovation: Explicit gating mechanism that determines:
    1. Which parts of the representation to modify
    2. How strongly to apply the modification
    3. What to preserve from the original representation
    """
    
    def __init__(self, d_model, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Gate computation network
        self.gate_network = keras.Sequential([
            keras.layers.Dense(d_model * 2, activation='relu'),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(d_model, activation='sigmoid')  # Gate values [0, 1]
        ])
        
        # Modification transformation network
        self.modification_network = keras.Sequential([
            keras.layers.Dense(d_model * 2, activation='relu'),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(d_model)
        ])
        
        # Layer normalization
        self.layer_norm = keras.layers.LayerNormalization()
        
    def call(self, original_embedding, modification_signal, training=None):
        """
        Apply gated modification to embeddings.
        
        Args:
            original_embedding: Original rule embeddings
            modification_signal: Encoded modification request
            
        Returns:
            Modified embeddings with selective gating
        """
        # Concatenate original and modification for gate computation
        combined = tf.concat([original_embedding, modification_signal], axis=-1)
        
        # Compute gate values (0 = keep original, 1 = apply modification)
        gate = self.gate_network(combined, training=training)
        
        # Compute modification
        modification = self.modification_network(modification_signal, training=training)
        
        # Apply gated modification: output = gate * modification + (1 - gate) * original
        modified_embedding = gate * modification + (1 - gate) * original_embedding
        
        # Add residual connection and normalize
        output = self.layer_norm(modified_embedding + original_embedding)
        
        return output, gate  # Return gate for analysis


class ImprovedRuleModificationComponent(keras.Model):
    """
    Improved component that modifies extracted rules with explicit gating.
    
    Key improvements:
    1. Modification signal is propagated to ALL layers, not just cross-attention
    2. Explicit gating mechanism for selective modification
    3. Multiple modification layers for progressive refinement
    """
    
    def __init__(self,
                 vocab_size: int,
                 d_model: int = 256,
                 num_layers: int = 4,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        # Modification request encoder
        self.mod_embedding = keras.layers.Embedding(vocab_size, d_model)
        self.mod_pos_encoding = PositionalEncoding(max_length=20, d_model=d_model)
        
        # Global modification signal extractor
        self.mod_pooling = keras.layers.GlobalAveragePooling1D()
        self.mod_projection = keras.layers.Dense(d_model)
        
        # Stack of gated modification layers
        self.gated_layers = []
        for _ in range(num_layers):
            self.gated_layers.append(GatedModificationLayer(d_model, dropout_rate))
        
        # Final projection
        self.output_projection = keras.layers.Dense(d_model)
        
    def call(self, rule_embeddings, modification_request, training=None):
        # Encode modification request
        mod_embed = self.mod_embedding(modification_request)
        mod_embed = self.mod_pos_encoding(mod_embed)
        
        # Extract global modification signal
        mod_signal = self.mod_pooling(mod_embed)
        mod_signal = self.mod_projection(mod_signal)
        
        # Expand modification signal to match sequence length
        seq_len = tf.shape(rule_embeddings)[1]
        mod_signal_expanded = tf.expand_dims(mod_signal, axis=1)
        mod_signal_expanded = tf.tile(mod_signal_expanded, [1, seq_len, 1])
        
        # Apply progressive gated modifications
        modified_embeddings = rule_embeddings
        gates = []
        
        for gated_layer in self.gated_layers:
            modified_embeddings, gate = gated_layer(
                modified_embeddings, 
                mod_signal_expanded, 
                training=training
            )
            gates.append(gate)
        
        # Final projection
        output = self.output_projection(modified_embeddings)
        
        return output, gates  # Return gates for analysis


class CompositionalLanguageModelV2(keras.Model):
    """
    Improved model for compositional language distribution invention.
    
    Key improvements:
    1. Uses ImprovedRuleModificationComponent with gating
    2. Maintains separate pathways for base and modified knowledge
    3. Better handling of modification signals
    """
    
    def __init__(self,
                 command_vocab_size: int,
                 action_vocab_size: int,
                 d_model: int = 256,
                 **kwargs):
        super().__init__()
        
        # Import the original components we're keeping
        from models import CompositionalRuleExtractor, SequenceGenerator
        
        self.rule_extractor = CompositionalRuleExtractor(
            vocab_size=command_vocab_size,
            d_model=d_model,
            **kwargs.get('extractor_kwargs', {})
        )
        
        self.rule_modifier = ImprovedRuleModificationComponent(
            vocab_size=command_vocab_size,
            d_model=d_model,
            **kwargs.get('modifier_kwargs', {})
        )
        
        self.sequence_generator = SequenceGenerator(
            vocab_size=action_vocab_size,
            d_model=d_model,
            **kwargs.get('generator_kwargs', {})
        )
        
        # Additional components for v2
        self.base_memory = self.add_weight(
            name='base_memory',
            shape=(1, 1, d_model),
            initializer='zeros',
            trainable=True
        )
        
    def call(self, inputs, training=None):
        command = inputs['command']
        modification = inputs.get('modification', None)
        target = inputs.get('target', None)
        
        # Extract rules from command
        rule_outputs = self.rule_extractor(command, training=training)
        rule_embeddings = rule_outputs['embeddings']
        
        # Apply modification if provided and not all zeros
        gates = None
        if modification is not None:
            # Check if modification is not just padding
            mod_sum = tf.reduce_sum(tf.abs(modification))
            
            # Use conditional logic for modification
            def apply_modification():
                modified, gates = self.rule_modifier(
                    rule_embeddings, 
                    modification, 
                    training=training
                )
                return modified, gates
            
            def keep_original():
                return rule_embeddings, None
            
            rule_embeddings, gates = tf.cond(
                mod_sum > 0,
                apply_modification,
                keep_original
            )
        
        # Generate sequence if target provided
        if target is not None:
            logits = self.sequence_generator(
                rule_embeddings, target, training=training
            )
            return logits
        
        # Otherwise return embeddings and analysis info
        return {
            'rule_embeddings': rule_embeddings,
            'rule_outputs': rule_outputs,
            'modification_gates': gates
        }


def create_model_v2(command_vocab_size: int,
                    action_vocab_size: int,
                    d_model: int = 256,
                    **kwargs) -> CompositionalLanguageModelV2:
    """Create an instance of the improved model"""
    
    model = CompositionalLanguageModelV2(
        command_vocab_size=command_vocab_size,
        action_vocab_size=action_vocab_size,
        d_model=d_model,
        **kwargs
    )
    
    # Build model
    dummy_command = tf.constant([[1, 2, 3, 4, 5]])
    dummy_target = tf.constant([[1, 2, 3, 4, 5, 6]])
    dummy_modification = tf.constant([[1, 2, 3]])
    
    _ = model({
        'command': dummy_command,
        'target': dummy_target,
        'modification': dummy_modification
    })
    
    return model


if __name__ == '__main__':
    # Test the improved model
    print("Testing improved compositional language model...")
    
    model = create_model_v2(
        command_vocab_size=20,
        action_vocab_size=10,
        d_model=128
    )
    
    print(f"\nModel created successfully!")
    print(f"Total parameters: {model.count_params():,}")
    
    # Test forward pass
    test_inputs = {
        'command': tf.constant([[1, 2, 3, 4, 5]]),
        'target': tf.constant([[1, 2, 3, 4, 5, 6]]),
        'modification': tf.constant([[7, 8, 9]])
    }
    
    output = model(test_inputs, training=True)
    print(f"Output shape: {output.shape}")
    print("\n✓ Model v2 is ready for training!")