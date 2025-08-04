"""
Minimal Variable Binding Model for SCAN

Based on Wu et al. (2025) - Transformers learning variable binding
Key insight: Force binding through dereferencing tasks

This implements the core architecture to enable true rule modifications
by explicitly binding words to memory slots that can be modified.
"""

from utils.imports import setup_project_paths

setup_project_paths()

import logging
from typing import Dict, Tuple

import numpy as np

from utils.config import setup_environment

# Set up environment and logging
config = setup_environment()
logger = logging.getLogger(__name__)

# Import ML libraries
import keras
from keras import layers, ops


class VariableMemory(layers.Layer):
    """
    Explicit variable slots for storing word-meaning bindings.

    Unlike traditional embeddings, these are discrete slots that can be
    selectively modified without affecting other bindings.
    """

    def __init__(self, n_slots: int = 10, slot_dim: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.n_slots = n_slots
        self.slot_dim = slot_dim

    def build(self, input_shape):
        # Initialize memory slots
        self.slots = self.add_weight(
            name="memory_slots",
            shape=(self.n_slots, self.slot_dim),
            initializer="glorot_uniform",
            trainable=True,
        )

        # Slot keys for content-based addressing
        self.slot_keys = self.add_weight(
            name="slot_keys",
            shape=(self.n_slots, self.slot_dim),
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, query, training=None):
        """
        Retrieve memory contents based on query.

        Args:
            query: Tensor of shape (batch_size, query_dim)

        Returns:
            memory_contents: Retrieved slot contents
            attention_weights: Attention over slots
        """
        # Project query to slot dimension
        query_proj = layers.Dense(self.slot_dim)(query)

        # Compute attention over slots
        scores = ops.matmul(query_proj, ops.transpose(self.slot_keys))
        attention_weights = ops.softmax(scores / ops.sqrt(float(self.slot_dim)))

        # Retrieve weighted combination of slots
        memory_contents = ops.matmul(attention_weights, self.slots)

        return memory_contents, attention_weights

    def update_slot(self, slot_idx: int, new_value):
        """Update a specific memory slot with new value."""
        self.slots[slot_idx].assign(new_value)


class BindingAttention(layers.Layer):
    """
    Associates words with specific memory slots.

    This creates explicit bindings that can be tracked and modified,
    unlike distributed representations in standard transformers.
    """

    def __init__(self, hidden_dim: int = 256, n_heads: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

    def build(self, input_shape):
        # Multi-head attention components
        self.q_proj = layers.Dense(self.hidden_dim)
        self.k_proj = layers.Dense(self.hidden_dim)
        self.v_proj = layers.Dense(self.hidden_dim)
        self.out_proj = layers.Dense(self.hidden_dim)

        # Binding predictor - outputs slot assignments
        self.binding_predictor = layers.Dense(1)

    def call(self, word_embeds, memory_keys, training=None):
        """
        Create bindings between words and memory slots.

        Args:
            word_embeds: Word embeddings (batch_size, seq_len, embed_dim)
            memory_keys: Memory slot keys (n_slots, slot_dim)

        Returns:
            bindings: Slot assignments for each word
            binding_scores: Raw scores for interpretability
        """
        batch_size, seq_len = ops.shape(word_embeds)[:2]

        # Project to multi-head space
        Q = self.q_proj(word_embeds)  # (batch, seq_len, hidden_dim)
        K = self.k_proj(memory_keys)  # (n_slots, hidden_dim)
        V = self.v_proj(memory_keys)  # (n_slots, hidden_dim)

        # Reshape for multi-head attention
        Q = ops.reshape(Q, (batch_size, seq_len, self.n_heads, self.head_dim))
        Q = ops.transpose(Q, (0, 2, 1, 3))  # (batch, n_heads, seq_len, head_dim)

        # Expand K, V for batch dimension
        K = ops.expand_dims(K, 0)  # (1, n_slots, hidden_dim)
        K = ops.reshape(K, (1, -1, self.n_heads, self.head_dim))
        K = ops.transpose(K, (0, 2, 1, 3))  # (1, n_heads, n_slots, head_dim)
        K = ops.tile(K, (batch_size, 1, 1, 1))  # (batch, n_heads, n_slots, head_dim)

        V = ops.expand_dims(V, 0)
        V = ops.reshape(V, (1, -1, self.n_heads, self.head_dim))
        V = ops.transpose(V, (0, 2, 1, 3))
        V = ops.tile(V, (batch_size, 1, 1, 1))

        # Compute attention scores
        scores = ops.matmul(Q, ops.transpose(K, (0, 1, 3, 2)))
        scores = scores / ops.sqrt(float(self.head_dim))

        # Apply softmax to get binding probabilities
        binding_probs = ops.softmax(
            scores, axis=-1
        )  # (batch, n_heads, seq_len, n_slots)

        # Average across heads
        binding_probs = ops.mean(binding_probs, axis=1)  # (batch, seq_len, n_slots)

        # Get hard bindings (argmax) for discrete slot assignment
        bindings = ops.argmax(binding_probs, axis=-1)  # (batch, seq_len)

        return bindings, binding_probs


class BoundVariableExecutor(layers.Layer):
    """
    Executes commands using bound variables from memory.

    This allows the same command structure to produce different outputs
    based on current variable bindings.
    """

    def __init__(
        self,
        action_vocab_size: int,
        hidden_dim: int = 256,
        embed_dim: int = 128,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.action_vocab_size = action_vocab_size
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

    def build(self, input_shape):
        # Command embedding (needed to convert tokens to vectors)
        # Note: vocab_size will be set from parent model
        self.command_embedding = None  # Will use parent's embedding

        # Command encoder
        self.command_encoder = layers.LSTM(self.hidden_dim, return_sequences=True)

        # Action decoder with attention
        self.decoder_lstm = layers.LSTM(self.hidden_dim, return_sequences=True)
        self.action_predictor = layers.Dense(self.action_vocab_size)

        # Cross-attention for incorporating bindings
        self.cross_attention = layers.MultiHeadAttention(
            num_heads=4, key_dim=self.hidden_dim // 4
        )

    def call(self, command_embeds, bound_variables, training=None):
        """
        Execute command using current variable bindings.

        Args:
            command_embeds: Embedded command (batch_size, seq_len, embed_dim)
            bound_variables: Variables retrieved from memory based on bindings

        Returns:
            action_sequence: Predicted action sequence
        """
        # Encode command structure
        command_encoded = self.command_encoder(command_embeds)

        # Incorporate bound variables via cross-attention
        attended_command = self.cross_attention(
            query=command_encoded, value=bound_variables, key=bound_variables
        )

        # Decode to action sequence
        decoder_out = self.decoder_lstm(attended_command)
        action_logits = self.action_predictor(decoder_out)

        return action_logits


class MinimalBindingModel(keras.Model):
    """
    Complete model combining variable binding components.

    Based on Wu et al. (2025) - Transformers learning variable binding
    Key insight: Force binding through dereferencing tasks
    """

    def __init__(
        self,
        vocab_size: int,
        action_vocab_size: int,
        n_slots: int = 10,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.action_vocab_size = action_vocab_size
        self.n_slots = n_slots
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Components
        self.embedding = layers.Embedding(vocab_size, embed_dim)
        self.variable_memory = VariableMemory(n_slots, embed_dim)
        self.binder = BindingAttention(hidden_dim)
        self.executor = BoundVariableExecutor(action_vocab_size, hidden_dim, embed_dim)

    def parse_to_variables(self, command):
        """Parse command into variables that can be bound."""
        # Embed command tokens
        embeds = self.embedding(command)
        return embeds

    def apply_modification(self, bindings, memory_contents, modification):
        """
        Apply modifications to specific bindings.

        This is the key operation that enables rule changes.
        """
        # For now, return unmodified - will implement modification logic
        # in training script based on specific modification rules
        return memory_contents

    def call(self, inputs, training=None):
        """
        Forward pass with optional modifications.

        Args:
            inputs: Dict with 'command' and optional 'modification'

        Returns:
            action_logits: Predicted action sequence
            bindings: Variable bindings for analysis
        """
        command = inputs["command"]
        modification = inputs.get("modification", None)

        # Parse command into variables
        variables = self.parse_to_variables(command)

        # Get memory keys
        _, memory_keys = self.variable_memory(
            ops.zeros((ops.shape(command)[0], self.embed_dim))
        )

        # Bind variables to memory slots
        bindings, binding_scores = self.binder(
            variables, self.variable_memory.slot_keys
        )

        # Retrieve bound variables from memory
        batch_size, seq_len = ops.shape(bindings)
        bound_variables = []

        for i in range(seq_len):
            slot_indices = bindings[:, i]  # (batch_size,)
            retrieved = ops.take(self.variable_memory.slots, slot_indices, axis=0)
            bound_variables.append(retrieved)

        bound_variables = ops.stack(
            bound_variables, axis=1
        )  # (batch, seq_len, slot_dim)

        # Apply modifications if provided
        if modification is not None:
            bound_variables = self.apply_modification(
                bindings, bound_variables, modification
            )

        # Execute with bound variables (pass embedded command, not raw tokens)
        action_logits = self.executor(variables, bound_variables)

        # Return logits and bindings for analysis
        return {
            "action_logits": action_logits,
            "bindings": bindings,
            "binding_scores": binding_scores,
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "action_vocab_size": self.action_vocab_size,
                "n_slots": self.n_slots,
                "embed_dim": self.embed_dim,
                "hidden_dim": self.hidden_dim,
            }
        )
        return config


def create_dereferencing_task(
    vocab: Dict[str, int], action_vocab: Dict[str, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create training tasks that force variable binding through dereferencing.

    Example:
        "X means jump. Do X twice." -> "JUMP JUMP"

    This forces the model to bind X to jump and dereference it.
    """
    # Implementation will be added based on specific SCAN vocabulary


def test_binding_capability():
    """Test if model can perform basic variable binding."""

    # Create small test vocabulary
    vocab = {"<PAD>": 0, "<START>": 1, "<END>": 2, "jump": 3, "walk": 4, "X": 5, "Y": 6}
    action_vocab = {"<PAD>": 0, "JUMP": 1, "WALK": 2}

    # Create model
    model = MinimalBindingModel(
        vocab_size=len(vocab),
        action_vocab_size=len(action_vocab),
        n_slots=5,
        embed_dim=32,
        hidden_dim=64,
    )

    # Test forward pass
    test_command = np.array([[vocab["X"], vocab["jump"]]])  # "X jump"

    outputs = model({"command": test_command})

    logger.info(f"Model output shape: {outputs['action_logits'].shape}")
    logger.info(f"Bindings: {outputs['bindings']}")

    return model


if __name__ == "__main__":
    logger.info("Testing minimal binding model...")
    model = test_binding_capability()
    logger.info("Basic test complete!")
