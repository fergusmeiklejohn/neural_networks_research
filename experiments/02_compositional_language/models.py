#!/usr/bin/env python3
"""
Neural Network Models for Compositional Language Distribution Invention

This module implements the core models for learning and modifying compositional
rules in the SCAN dataset.
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"


import numpy as np
import tensorflow as tf
from tensorflow import keras


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
            name="pos_encoding",
            shape=(1, max_length, d_model),
            initializer=keras.initializers.Constant(pos_encoding[np.newaxis, ...]),
            trainable=False,
        )

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]


class CompositionalRuleExtractor(keras.Model):
    """
    Transformer-based model for extracting compositional rules from SCAN commands.

    This model learns to identify:
    1. Primitive actions (walk, run, jump, look, turn)
    2. Modifiers (twice, thrice, opposite)
    3. Compositional patterns (how primitives and modifiers combine)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 1024,
        max_length: int = 50,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model

        # Embedding layers
        self.embedding = keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(max_length, d_model)
        self.dropout = keras.layers.Dropout(dropout_rate)

        # Transformer encoder layers
        self.encoder_layers = []
        for _ in range(num_layers):
            self.encoder_layers.append(
                {
                    "attention": keras.layers.MultiHeadAttention(
                        num_heads=num_heads,
                        key_dim=d_model // num_heads,
                        dropout=dropout_rate,
                    ),
                    "ffn": keras.Sequential(
                        [
                            keras.layers.Dense(d_ff, activation="relu"),
                            keras.layers.Dropout(dropout_rate),
                            keras.layers.Dense(d_model),
                        ]
                    ),
                    "norm1": keras.layers.LayerNormalization(),
                    "norm2": keras.layers.LayerNormalization(),
                }
            )

        # Rule extraction heads
        self.primitive_head = keras.layers.Dense(
            5, activation="sigmoid"
        )  # 5 primitives
        self.direction_head = keras.layers.Dense(
            3, activation="sigmoid"
        )  # left, right, around
        self.modifier_head = keras.layers.Dense(
            3, activation="sigmoid"
        )  # twice, thrice, opposite
        self.connector_head = keras.layers.Dense(2, activation="sigmoid")  # and, after

        # Global pooling for sequence-level features
        self.global_pool = keras.layers.GlobalAveragePooling1D()

    def call(self, inputs, training=None):
        # Embed and add positional encoding
        x = self.embedding(inputs)
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)

        # Apply transformer layers
        attention_weights = []
        for layer in self.encoder_layers:
            # Self-attention
            attn_output, attn_weights = layer["attention"](
                x, x, x, training=training, return_attention_scores=True
            )
            x = layer["norm1"](x + attn_output)

            # Feed-forward
            ffn_output = layer["ffn"](x, training=training)
            x = layer["norm2"](x + ffn_output)

            attention_weights.append(attn_weights)

        # Global pooling for sequence-level representation
        pooled = self.global_pool(x)

        # Extract rules
        primitives = self.primitive_head(pooled)
        directions = self.direction_head(pooled)
        modifiers = self.modifier_head(pooled)
        connectors = self.connector_head(pooled)

        return {
            "primitives": primitives,
            "directions": directions,
            "modifiers": modifiers,
            "connectors": connectors,
            "embeddings": x,
            "attention_weights": attention_weights,
        }


class RuleModificationComponent(keras.Model):
    """
    Component that modifies extracted rules based on modification requests.

    Takes:
    1. Original rule embeddings from CompositionalRuleExtractor
    2. Modification request (e.g., "jump means walk")

    Outputs:
    1. Modified rule embeddings
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        # Modification request encoder
        self.mod_embedding = keras.layers.Embedding(vocab_size, d_model)
        self.mod_pos_encoding = PositionalEncoding(max_length=20, d_model=d_model)

        # Cross-attention layers for modification
        self.cross_attention_layers = []
        for _ in range(num_layers):
            self.cross_attention_layers.append(
                keras.layers.MultiHeadAttention(
                    num_heads=num_heads,
                    key_dim=d_model // num_heads,
                    dropout=dropout_rate,
                )
            )

        # Rule update network
        self.rule_updater = keras.Sequential(
            [
                keras.layers.Dense(d_model * 2, activation="relu"),
                keras.layers.Dropout(dropout_rate),
                keras.layers.Dense(d_model),
            ]
        )

        # Layer normalization
        self.layer_norms = [
            keras.layers.LayerNormalization() for _ in range(num_layers)
        ]

    def call(self, rule_embeddings, modification_request, training=None):
        # Encode modification request
        mod_embed = self.mod_embedding(modification_request)
        mod_embed = self.mod_pos_encoding(mod_embed)

        # Apply cross-attention to modify rules
        modified_embeddings = rule_embeddings

        for i, (cross_attn, layer_norm) in enumerate(
            zip(self.cross_attention_layers, self.layer_norms)
        ):
            # Cross-attention: rules attend to modification request
            attn_output = cross_attn(
                query=modified_embeddings,
                key=mod_embed,
                value=mod_embed,
                training=training,
            )

            # Residual connection and normalization
            modified_embeddings = layer_norm(modified_embeddings + attn_output)

            # Rule update
            updated = self.rule_updater(modified_embeddings)
            modified_embeddings = modified_embeddings + updated

        return modified_embeddings


class SequenceGenerator(keras.Model):
    """
    Generates action sequences from (modified) rule embeddings.

    Uses transformer decoder architecture with beam search for high-quality outputs.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 1024,
        max_length: int = 100,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_length = max_length

        # Output embedding
        self.output_embedding = keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(max_length, d_model)

        # Transformer decoder layers
        self.decoder_layers = []
        for _ in range(num_layers):
            self.decoder_layers.append(
                {
                    "self_attention": keras.layers.MultiHeadAttention(
                        num_heads=num_heads,
                        key_dim=d_model // num_heads,
                        dropout=dropout_rate,
                    ),
                    "cross_attention": keras.layers.MultiHeadAttention(
                        num_heads=num_heads,
                        key_dim=d_model // num_heads,
                        dropout=dropout_rate,
                    ),
                    "ffn": keras.Sequential(
                        [
                            keras.layers.Dense(d_ff, activation="relu"),
                            keras.layers.Dropout(dropout_rate),
                            keras.layers.Dense(d_model),
                        ]
                    ),
                    "norm1": keras.layers.LayerNormalization(),
                    "norm2": keras.layers.LayerNormalization(),
                    "norm3": keras.layers.LayerNormalization(),
                }
            )

        # Output projection
        self.output_projection = keras.layers.Dense(vocab_size)

    def call(self, encoder_output, target_sequence, training=None):
        # Embed target sequence
        target_embed = self.output_embedding(target_sequence)
        target_embed = self.pos_encoding(target_embed)

        # Create causal mask for self-attention
        seq_len = tf.shape(target_sequence)[1]
        causal_mask = self.create_causal_mask(seq_len)

        # Apply decoder layers
        x = target_embed
        for layer in self.decoder_layers:
            # Self-attention with causal mask
            self_attn = layer["self_attention"](
                query=x, key=x, value=x, attention_mask=causal_mask, training=training
            )
            x = layer["norm1"](x + self_attn)

            # Cross-attention to encoder output
            cross_attn = layer["cross_attention"](
                query=x, key=encoder_output, value=encoder_output, training=training
            )
            x = layer["norm2"](x + cross_attn)

            # Feed-forward network
            ffn_output = layer["ffn"](x)
            x = layer["norm3"](x + ffn_output)

        # Project to vocabulary
        logits = self.output_projection(x)

        return logits

    def create_causal_mask(self, seq_len):
        """Create causal mask for self-attention"""
        mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        return mask[tf.newaxis, tf.newaxis, :, :]

    def generate(self, encoder_output, start_token, end_token, max_length=None):
        """Generate sequence using greedy decoding"""
        if max_length is None:
            max_length = self.max_length

        batch_size = tf.shape(encoder_output)[0]

        # Start with start token
        output_sequence = tf.fill([batch_size, 1], start_token)

        for _ in range(max_length - 1):
            # Get predictions for next token
            logits = self(encoder_output, output_sequence, training=False)
            next_token = tf.argmax(logits[:, -1, :], axis=-1, output_type=tf.int32)
            next_token = tf.expand_dims(next_token, axis=1)

            # Append to sequence
            output_sequence = tf.concat([output_sequence, next_token], axis=1)

            # Check if all sequences have produced end token
            if tf.reduce_all(tf.equal(next_token, end_token)):
                break

        return output_sequence


class CompositionalLanguageModel(keras.Model):
    """
    Complete model for compositional language distribution invention.

    Combines:
    1. CompositionalRuleExtractor
    2. RuleModificationComponent
    3. SequenceGenerator
    """

    def __init__(
        self,
        command_vocab_size: int,
        action_vocab_size: int,
        d_model: int = 256,
        **kwargs,
    ):
        super().__init__()

        self.rule_extractor = CompositionalRuleExtractor(
            vocab_size=command_vocab_size,
            d_model=d_model,
            **kwargs.get("extractor_kwargs", {}),
        )

        self.rule_modifier = RuleModificationComponent(
            vocab_size=command_vocab_size,
            d_model=d_model,
            **kwargs.get("modifier_kwargs", {}),
        )

        self.sequence_generator = SequenceGenerator(
            vocab_size=action_vocab_size,
            d_model=d_model,
            **kwargs.get("generator_kwargs", {}),
        )

    def call(self, inputs, training=None):
        command = inputs["command"]
        modification = inputs.get("modification", None)
        target = inputs.get("target", None)

        # Extract rules from command
        rule_outputs = self.rule_extractor(command, training=training)
        rule_embeddings = rule_outputs["embeddings"]

        # Apply modification if provided and not all zeros (dummy modification)
        if modification is not None:
            # Check if modification is not just padding (all zeros)
            # Use tf.cond for graph mode compatibility
            mod_sum = tf.reduce_sum(tf.abs(modification))
            rule_embeddings = tf.cond(
                mod_sum > 0,
                lambda: self.rule_modifier(
                    rule_embeddings, modification, training=training
                ),
                lambda: rule_embeddings,
            )

        # Generate sequence if target provided (training mode)
        if target is not None:
            logits = self.sequence_generator(rule_embeddings, target, training=training)
            # For training with model.fit(), return just the logits tensor
            # Keras expects the model output to match the target shape
            return logits

        # Otherwise return embeddings for generation
        return {"rule_embeddings": rule_embeddings, "rule_outputs": rule_outputs}

    def generate_action(self, command, modification=None, start_token=0, end_token=1):
        """Generate action sequence for a command with optional modification"""
        # Get rule embeddings
        outputs = self(
            {"command": command, "modification": modification}, training=False
        )

        rule_embeddings = outputs["rule_embeddings"]

        # Generate sequence
        action_sequence = self.sequence_generator.generate(
            rule_embeddings, start_token, end_token
        )

        return action_sequence


def create_model(
    command_vocab_size: int, action_vocab_size: int, d_model: int = 256
) -> CompositionalLanguageModel:
    """Create and compile the compositional language model"""

    # Adjust FFN size based on d_model for memory efficiency
    d_ff = d_model * 2 if d_model <= 128 else d_model * 4

    model = CompositionalLanguageModel(
        command_vocab_size=command_vocab_size,
        action_vocab_size=action_vocab_size,
        d_model=d_model,
        extractor_kwargs={
            "num_heads": 4 if d_model <= 128 else 8,
            "num_layers": 4 if d_model <= 128 else 6,
            "d_ff": d_ff,
            "dropout_rate": 0.1,
        },
        modifier_kwargs={
            "num_heads": 4,
            "num_layers": 2 if d_model <= 128 else 3,
            "dropout_rate": 0.1,
        },
        generator_kwargs={
            "num_heads": 4 if d_model <= 128 else 8,
            "num_layers": 4 if d_model <= 128 else 6,
            "d_ff": d_ff,
            "dropout_rate": 0.1,
        },
    )

    # Compile the model with standard settings
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def test_models():
    """Test the model components"""
    print("Testing Compositional Language Models...")

    # Test parameters
    command_vocab_size = 20
    action_vocab_size = 10
    batch_size = 2
    seq_len = 10

    # Create model
    model = create_model(command_vocab_size, action_vocab_size)

    # Test data
    command = tf.random.uniform(
        (batch_size, seq_len), 0, command_vocab_size, dtype=tf.int32
    )
    modification = tf.random.uniform(
        (batch_size, 5), 0, command_vocab_size, dtype=tf.int32
    )
    target = tf.random.uniform((batch_size, 15), 0, action_vocab_size, dtype=tf.int32)

    # Test forward pass
    outputs = model(
        {"command": command, "modification": modification, "target": target},
        training=True,
    )

    print(f"Output logits shape: {outputs['logits'].shape}")
    print(f"Rule outputs: {outputs['rule_outputs'].keys()}")

    # Test generation
    generated = model.generate_action(command, modification)
    print(f"Generated sequence shape: {generated.shape}")

    print("Model test successful!")


if __name__ == "__main__":
    test_models()
