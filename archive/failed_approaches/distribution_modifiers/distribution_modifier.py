#!/usr/bin/env python3
"""
Distribution Modification Component for Physics Worlds Experiment.

This component learns to modify physics parameters based on natural language
or numerical modification requests, maintaining consistency while applying
the requested changes.
"""

import os

os.environ["KERAS_BACKEND"] = "jax"

import json
from pathlib import Path
from typing import Dict, List, Tuple

import keras
import numpy as np
from keras import layers, ops


@keras.saving.register_keras_serializable()
class ModificationEncoder(keras.Model):
    """Encodes modification requests into a latent representation."""

    def __init__(
        self, vocab_size: int = 1000, embed_dim: int = 64, hidden_dim: int = 128
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.embedding = layers.Embedding(vocab_size, embed_dim)
        self.lstm = layers.LSTM(hidden_dim, return_sequences=False)
        self.dense = layers.Dense(hidden_dim, activation="relu")

    def get_config(self):
        return {
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
        }

    def call(self, inputs):
        # inputs: (batch_size, sequence_length)
        x = self.embedding(inputs)
        x = self.lstm(x)
        x = self.dense(x)
        return x


@keras.saving.register_keras_serializable()
class ParameterModifier(keras.Model):
    """Modifies physics parameters based on encoded request and current params."""

    def __init__(self, n_params: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.n_params = n_params

        # Encode current parameters
        self.param_encoder = keras.Sequential(
            [
                layers.Dense(hidden_dim, activation="relu"),
                layers.Dense(hidden_dim, activation="relu"),
            ]
        )

        # Combine request and params
        self.combiner = keras.Sequential(
            [
                layers.Dense(hidden_dim * 2, activation="relu"),
                layers.Dropout(0.2),
                layers.Dense(hidden_dim, activation="relu"),
                layers.Dropout(0.2),
            ]
        )

        # Output modification factors (multiplicative)
        self.modifier = layers.Dense(n_params, activation="linear")

    def get_config(self):
        return {"n_params": self.n_params, "hidden_dim": 128}  # Fixed for now

    def call(self, params, request_encoding, training=None):
        # Encode current parameters
        param_features = self.param_encoder(params)

        # Combine with request
        combined = ops.concatenate([param_features, request_encoding], axis=-1)
        features = self.combiner(combined, training=training)

        # Output modification factors (centered around 1.0)
        mod_factors = self.modifier(features)
        mod_factors = ops.exp(ops.clip(mod_factors, -2, 2))  # Range: [0.135, 7.39]

        # Apply modifications
        modified_params = params * mod_factors

        return modified_params, mod_factors


@keras.saving.register_keras_serializable()
class DistributionModifier(keras.Model):
    """Complete distribution modification pipeline."""

    def __init__(self, vocab_size: int = 1000, n_params: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_params = n_params

        # Components
        self.request_encoder = ModificationEncoder(vocab_size=vocab_size)
        self.param_modifier = ParameterModifier(n_params=n_params)

        # Consistency checker - ensures non-modified params stay similar
        self.consistency_checker = keras.Sequential(
            [
                layers.Dense(64, activation="relu"),
                layers.Dense(32, activation="relu"),
                layers.Dense(n_params, activation="sigmoid"),
            ]
        )

    def get_config(self):
        return {"vocab_size": self.vocab_size, "n_params": self.n_params}

    def call(self, inputs, training=None):
        params, request_tokens = inputs

        # Encode request
        request_features = self.request_encoder(request_tokens)

        # Modify parameters
        modified_params, mod_factors = self.param_modifier(
            params, request_features, training=training
        )

        # Check consistency (which params should change)
        change_mask = self.consistency_checker(request_features)

        # Apply selective modification
        final_params = params * (1 - change_mask) + modified_params * change_mask

        return final_params, mod_factors, change_mask

    def compute_loss(self, params, request_tokens, target_params, target_changes):
        """Compute modification loss with consistency constraints."""
        pred_params, mod_factors, change_mask = self(
            [params, request_tokens], training=True
        )

        # Parameter prediction loss
        param_loss = ops.mean(ops.square(pred_params - target_params))

        # Change detection loss (which parameters should change)
        actual_changes = ops.abs(target_params - params) / (ops.abs(params) + 1e-6)
        change_threshold = 0.05  # 5% change threshold
        change_targets = ops.cast(actual_changes > change_threshold, "float32")
        change_loss = keras.losses.binary_crossentropy(change_targets, change_mask)
        change_loss = ops.mean(change_loss)

        # Magnitude consistency loss
        pred_changes = ops.abs(pred_params - params) / (ops.abs(params) + 1e-6)
        target_change_ratios = ops.abs(target_params - params) / (
            ops.abs(params) + 1e-6
        )

        # Only compute magnitude loss for parameters that actually changed
        change_weights = ops.cast(actual_changes > change_threshold, "float32")
        magnitude_loss = ops.sum(
            change_weights * ops.square(pred_changes - target_change_ratios)
        ) / (ops.sum(change_weights) + 1e-6)

        # Total loss
        total_loss = param_loss + 0.5 * change_loss + 0.3 * magnitude_loss

        return {
            "total_loss": total_loss,
            "param_loss": param_loss,
            "change_loss": change_loss,
            "magnitude_loss": magnitude_loss,
        }


class ModificationDataProcessor:
    """Processes modification pairs data for training."""

    def __init__(self, vocab_size: int = 1000, max_length: int = 20):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.word_to_idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx_to_word = {0: "<PAD>", 1: "<UNK>"}
        self.next_idx = 2

        # Physics parameter names and indices
        self.param_names = ["gravity", "friction", "elasticity", "damping"]
        self.param_indices = {name: i for i, name in enumerate(self.param_names)}

    def build_vocabulary(self, descriptions: List[str]):
        """Build vocabulary from modification descriptions."""
        word_counts = {}

        for desc in descriptions:
            words = desc.lower().split()
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

        # Sort by frequency and take top vocab_size words
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

        for word, _ in sorted_words[
            : self.vocab_size - 2
        ]:  # Reserve 2 for special tokens
            if word not in self.word_to_idx:
                self.word_to_idx[word] = self.next_idx
                self.idx_to_word[self.next_idx] = word
                self.next_idx += 1

    def encode_description(self, description: str) -> np.ndarray:
        """Convert description to token indices."""
        words = description.lower().split()
        indices = []

        for word in words[: self.max_length]:
            idx = self.word_to_idx.get(word, 1)  # Use <UNK> for unknown words
            indices.append(idx)

        # Pad to max_length
        while len(indices) < self.max_length:
            indices.append(0)  # <PAD>

        return np.array(indices[: self.max_length])

    def extract_params(self, config: Dict) -> np.ndarray:
        """Extract physics parameters in consistent order."""
        params = []
        for name in self.param_names:
            # Handle absolute values for gravity (stored as negative)
            if name == "gravity":
                params.append(abs(config[name]))
            else:
                params.append(config[name])
        return np.array(params, dtype=np.float32)

    def process_modification_pairs(
        self, mod_pairs: List[Dict]
    ) -> Tuple[np.ndarray, ...]:
        """Process modification pairs into training data."""
        # Build vocabulary first
        descriptions = [pair["modification_description"] for pair in mod_pairs]
        self.build_vocabulary(descriptions)

        # Process data
        base_params = []
        mod_descriptions = []
        target_params = []
        param_changes = []

        for pair in mod_pairs:
            # Extract parameters
            base_p = self.extract_params(pair["base_config"])

            # Create full target parameters (base + modifications)
            target_p = base_p.copy()
            for param_name in pair["modification_params"]:
                if param_name in self.param_indices:
                    idx = self.param_indices[param_name]
                    if param_name == "gravity":
                        target_p[idx] = abs(pair["modification_params"][param_name])
                    else:
                        target_p[idx] = pair["modification_params"][param_name]

            # Encode description
            desc_tokens = self.encode_description(pair["modification_description"])

            # Calculate parameter changes
            changes = (target_p - base_p) / (base_p + 1e-6)

            base_params.append(base_p)
            mod_descriptions.append(desc_tokens)
            target_params.append(target_p)
            param_changes.append(changes)

        return (
            np.array(base_params),
            np.array(mod_descriptions),
            np.array(target_params),
            np.array(param_changes),
        )

    def save_vocabulary(self, path: Path):
        """Save vocabulary for later use."""
        vocab_data = {
            "word_to_idx": self.word_to_idx,
            "idx_to_word": self.idx_to_word,
            "param_names": self.param_names,
            "vocab_size": self.vocab_size,
            "max_length": self.max_length,
        }
        with open(path, "w") as f:
            json.dump(vocab_data, f, indent=2)

    def load_vocabulary(self, path: Path):
        """Load saved vocabulary."""
        with open(path, "r") as f:
            vocab_data = json.load(f)
        self.word_to_idx = vocab_data["word_to_idx"]
        self.idx_to_word = {int(k): v for k, v in vocab_data["idx_to_word"].items()}
        self.param_names = vocab_data["param_names"]
        self.vocab_size = vocab_data["vocab_size"]
        self.max_length = vocab_data["max_length"]
        self.next_idx = max(self.idx_to_word.keys()) + 1


def test_modification_component():
    """Test the modification component with sample data."""
    print("Testing Distribution Modification Component...")

    # Create dummy data
    batch_size = 32
    vocab_size = 100
    n_params = 4
    max_length = 10

    # Initialize model
    model = DistributionModifier(vocab_size=vocab_size, n_params=n_params)

    # Create dummy inputs
    params = np.random.uniform(0.1, 2.0, (batch_size, n_params)).astype(np.float32)
    request_tokens = np.random.randint(0, vocab_size, (batch_size, max_length))

    # Forward pass
    outputs = model([params, request_tokens], training=True)
    final_params, mod_factors, change_mask = outputs

    print(f"Input params shape: {params.shape}")
    print(f"Request tokens shape: {request_tokens.shape}")
    print(f"Output params shape: {final_params.shape}")
    print(f"Modification factors shape: {mod_factors.shape}")
    print(f"Change mask shape: {change_mask.shape}")

    # Test loss computation
    target_params = params * np.random.uniform(0.5, 1.5, params.shape).astype(
        np.float32
    )
    target_changes = np.abs(target_params - params) / params

    losses = model.compute_loss(params, request_tokens, target_params, target_changes)
    print(f"\nLosses:")
    for name, value in losses.items():
        print(f"  {name}: {float(value):.4f}")

    print("\nTest passed!")


if __name__ == "__main__":
    test_modification_component()
