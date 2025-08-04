"""
Variable Binding model implemented in MLX for Apple Silicon optimization.

This implementation uses MLX's native operations to achieve better GPU utilization
on Apple Silicon compared to our Keras multi-backend approach.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map, tree_flatten, tree_unflatten

import numpy as np
import time
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

from utils.imports import setup_project_paths
setup_project_paths()

from utils.config import setup_environment
from utils.paths import get_output_path
import logging

# Set up environment
config = setup_environment()
logger = logging.getLogger(__name__)


class VariableMemoryMLX(nn.Module):
    """
    Explicit variable slots for storing word-meaning bindings in MLX.
    
    Unlike traditional embeddings, these are discrete slots that can be
    selectively modified without affecting other bindings.
    """
    
    def __init__(self, n_slots: int = 10, slot_dim: int = 128):
        super().__init__()
        self.n_slots = n_slots
        self.slot_dim = slot_dim
        
        # Initialize memory slots and keys
        self.slots = mx.random.uniform(-0.1, 0.1, (n_slots, slot_dim))
        self.slot_keys = mx.random.uniform(-0.1, 0.1, (n_slots, slot_dim))
    
    def __call__(self, query):
        """
        Retrieve memory contents based on query.
        
        Args:
            query: Array of shape (batch_size, query_dim)
            
        Returns:
            memory_contents: Retrieved slot contents
            attention_weights: Attention over slots
        """
        # Project query to slot dimension
        query_proj = nn.Linear(query.shape[-1], self.slot_dim)(query)
        
        # Compute attention over slots
        scores = mx.matmul(query_proj, self.slot_keys.T)
        attention_weights = mx.softmax(scores / mx.sqrt(float(self.slot_dim)), axis=-1)
        
        # Retrieve weighted combination of slots
        memory_contents = mx.matmul(attention_weights, self.slots)
        
        return memory_contents, attention_weights


class BindingAttentionMLX(nn.Module):
    """
    Associates words with specific memory slots using MLX operations.
    
    This creates explicit bindings that can be tracked and modified,
    unlike distributed representations in standard transformers.
    """
    
    def __init__(self, hidden_dim: int = 256, n_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        
        # Multi-head attention components
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def __call__(self, word_embeds, memory_keys):
        """
        Create bindings between words and memory slots.
        
        Args:
            word_embeds: Word embeddings (batch_size, seq_len, embed_dim)
            memory_keys: Memory slot keys (n_slots, slot_dim)
            
        Returns:
            bindings: Slot assignments for each word
            binding_scores: Raw scores for interpretability
        """
        batch_size, seq_len = word_embeds.shape[:2]
        
        # Project memory keys to hidden_dim first
        memory_keys_proj = nn.Linear(memory_keys.shape[-1], self.hidden_dim)(memory_keys)
        
        # Project to multi-head space
        Q = self.q_proj(word_embeds)  # (batch, seq_len, hidden_dim)
        K = self.k_proj(memory_keys_proj)   # (n_slots, hidden_dim)
        V = self.v_proj(memory_keys_proj)   # (n_slots, hidden_dim)
        
        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        Q = Q.transpose(0, 2, 1, 3)  # (batch, n_heads, seq_len, head_dim)
        
        # Expand K, V for batch dimension
        K = mx.expand_dims(K, axis=0)  # (1, n_slots, hidden_dim)
        K = K.reshape(1, -1, self.n_heads, self.head_dim)
        K = K.transpose(0, 2, 1, 3)  # (1, n_heads, n_slots, head_dim)
        K = mx.repeat(K, batch_size, axis=0)  # (batch, n_heads, n_slots, head_dim)
        
        V = mx.expand_dims(V, axis=0)
        V = V.reshape(1, -1, self.n_heads, self.head_dim)
        V = V.transpose(0, 2, 1, 3)
        V = mx.repeat(V, batch_size, axis=0)
        
        # Compute attention scores
        scores = mx.matmul(Q, K.transpose(0, 1, 3, 2))
        scores = scores / mx.sqrt(float(self.head_dim))
        
        # Apply softmax to get binding probabilities
        binding_probs = mx.softmax(scores, axis=-1)  # (batch, n_heads, seq_len, n_slots)
        
        # Average across heads
        binding_probs = mx.mean(binding_probs, axis=1)  # (batch, seq_len, n_slots)
        
        # Get hard bindings (argmax) for discrete slot assignment
        bindings = mx.argmax(binding_probs, axis=-1)  # (batch, seq_len)
        
        return bindings, binding_probs


class BoundVariableExecutorMLX(nn.Module):
    """
    Executes commands using bound variables from memory.
    
    This allows the same command structure to produce different outputs
    based on current variable bindings.
    """
    
    def __init__(self, action_vocab_size: int, hidden_dim: int = 256):
        super().__init__()
        self.action_vocab_size = action_vocab_size
        self.hidden_dim = hidden_dim
        
        # Command encoder (LSTM equivalent using GRU for simplicity in MLX)
        self.encoder_cell = nn.GRU(hidden_dim, hidden_dim)
        self.decoder_cell = nn.GRU(hidden_dim, hidden_dim)
        
        # Action predictor
        self.action_predictor = nn.Linear(hidden_dim, action_vocab_size)
        
        # Cross-attention layers
        self.cross_attn_q = nn.Linear(hidden_dim, hidden_dim)
        self.cross_attn_k = nn.Linear(hidden_dim, hidden_dim)
        self.cross_attn_v = nn.Linear(hidden_dim, hidden_dim)
    
    def __call__(self, command_embeds, bound_variables):
        """
        Execute command using current variable bindings.
        
        Args:
            command_embeds: Embedded command (batch_size, seq_len, embed_dim)
            bound_variables: Variables retrieved from memory based on bindings
            
        Returns:
            action_sequence: Predicted action sequence
        """
        batch_size, seq_len, _ = command_embeds.shape
        
        # Encode command - process sequentially
        hidden = mx.zeros((batch_size, self.hidden_dim))
        encoded_states = []
        
        for t in range(seq_len):
            hidden = self.encoder_cell(command_embeds[:, t, :], hidden)
            encoded_states.append(hidden)
        
        command_encoded = mx.stack(encoded_states, axis=1)
        
        # Cross-attention with bound variables
        Q = self.cross_attn_q(command_encoded)
        K = self.cross_attn_k(bound_variables)
        V = self.cross_attn_v(bound_variables)
        
        # Compute attention
        scores = mx.matmul(Q, K.transpose(0, 2, 1)) / mx.sqrt(float(self.hidden_dim))
        attn_weights = mx.softmax(scores, axis=-1)
        attended_command = mx.matmul(attn_weights, V)
        
        # Decode to action sequence
        decoder_hidden = mx.zeros((batch_size, self.hidden_dim))
        action_logits = []
        
        for t in range(seq_len):
            decoder_hidden = self.decoder_cell(attended_command[:, t, :], decoder_hidden)
            logits = self.action_predictor(decoder_hidden)
            action_logits.append(logits)
        
        return mx.stack(action_logits, axis=1)


class MinimalBindingModelMLX(nn.Module):
    """
    Complete model combining variable binding components in MLX.
    
    Optimized for Apple Silicon GPU utilization.
    """
    
    def __init__(
        self,
        vocab_size: int,
        action_vocab_size: int,
        n_slots: int = 10,
        embed_dim: int = 128,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.action_vocab_size = action_vocab_size
        self.n_slots = n_slots
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Components
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.variable_memory = VariableMemoryMLX(n_slots, embed_dim)
        self.binder = BindingAttentionMLX(hidden_dim)
        self.executor = BoundVariableExecutorMLX(action_vocab_size, hidden_dim)
        
        # Project embeddings to hidden dim for binder
        self.embed_proj = nn.Linear(embed_dim, hidden_dim)
    
    def __call__(self, inputs):
        """
        Forward pass with optional modifications.
        
        Args:
            inputs: Dict with 'command' and optional 'modification'
            
        Returns:
            Dict with action_logits, bindings, and binding_scores
        """
        command = inputs['command']
        
        # Parse command into variables
        embeds = self.embedding(command)
        embeds_proj = self.embed_proj(embeds)
        
        # Get memory keys
        batch_size = command.shape[0]
        query = mx.zeros((batch_size, self.embed_dim))
        _, _ = self.variable_memory(query)
        
        # Bind variables to memory slots
        bindings, binding_scores = self.binder(embeds_proj, self.variable_memory.slot_keys)
        
        # Retrieve bound variables from memory
        bound_variables = []
        for i in range(command.shape[1]):
            slot_indices = bindings[:, i]
            retrieved = self.variable_memory.slots[slot_indices]
            bound_variables.append(retrieved)
        
        bound_variables = mx.stack(bound_variables, axis=1)
        
        # Execute with bound variables
        action_logits = self.executor(embeds, bound_variables)
        
        return {
            'action_logits': action_logits,
            'bindings': bindings,
            'binding_scores': binding_scores
        }


def create_simple_binding_data(n_samples: int = 100):
    """Create simple dereferencing tasks for testing binding."""
    # Simple vocabularies
    command_vocab = {
        '<PAD>': 0, '<START>': 1, '<END>': 2,
        'jump': 3, 'walk': 4, 'run': 5,
        'X': 6, 'Y': 7, 'Z': 8,
        'twice': 9, 'and': 10
    }
    
    action_vocab = {
        '<PAD>': 0, '<START>': 1, '<END>': 2,
        'JUMP': 3, 'WALK': 4, 'RUN': 5
    }
    
    # Generate training data
    commands = []
    actions = []
    
    # Basic binding patterns
    patterns = [
        # Simple binding
        (['X', 'jump'], ['JUMP']),
        (['Y', 'walk'], ['WALK']),
        (['Z', 'run'], ['RUN']),
        # Repeated binding
        (['X', 'jump', 'X', 'jump'], ['JUMP', 'JUMP']),
        (['Y', 'walk', 'Y', 'walk'], ['WALK', 'WALK']),
        # Mixed binding
        (['X', 'jump', 'and', 'Y', 'walk'], ['JUMP', 'WALK']),
        (['Z', 'run', 'and', 'X', 'jump'], ['RUN', 'JUMP']),
    ]
    
    # Create samples
    for _ in range(n_samples):
        pattern_idx = np.random.randint(len(patterns))
        cmd_tokens, act_tokens = patterns[pattern_idx]
        
        # Convert to IDs and pad
        cmd_ids = [command_vocab.get(t, 0) for t in cmd_tokens]
        act_ids = [action_vocab.get(t, 0) for t in act_tokens]
        
        # Pad to fixed length
        max_cmd_len = 8
        max_act_len = 4
        
        cmd_ids = cmd_ids[:max_cmd_len] + [0] * (max_cmd_len - len(cmd_ids))
        act_ids = act_ids[:max_act_len] + [0] * (max_act_len - len(act_ids))
        
        commands.append(cmd_ids)
        actions.append(act_ids)
    
    return {
        'commands': mx.array(commands),
        'actions': mx.array(actions),
        'command_vocab': command_vocab,
        'action_vocab': action_vocab
    }


def loss_fn(model, x, y):
    """Compute cross-entropy loss."""
    outputs = model({'command': x})
    logits = outputs['action_logits']
    
    # Truncate to match target length
    if logits.shape[1] > y.shape[1]:
        logits = logits[:, :y.shape[1], :]
    
    # Pad targets if needed
    if y.shape[1] > logits.shape[1]:
        pad_len = y.shape[1] - logits.shape[1]
        y = y[:, :logits.shape[1]]
    
    return mx.mean(nn.losses.cross_entropy(logits, y))


def eval_accuracy(model, data):
    """Evaluate model accuracy."""
    correct = 0
    total = 0
    
    batch_size = 32
    for i in range(0, len(data['commands']), batch_size):
        batch_x = data['commands'][i:i+batch_size]
        batch_y = data['actions'][i:i+batch_size]
        
        outputs = model({'command': batch_x})
        predictions = mx.argmax(outputs['action_logits'], axis=-1)
        
        # Truncate to match lengths
        min_len = min(predictions.shape[1], batch_y.shape[1])
        predictions = predictions[:, :min_len]
        batch_y = batch_y[:, :min_len]
        
        # Count correct predictions (ignoring padding)
        mask = batch_y != 0
        correct += mx.sum(mask * (predictions == batch_y))
        total += mx.sum(mask)
    
    return float(correct) / float(total) if total > 0 else 0.0


def train_mlx(epochs=20, batch_size=32, learning_rate=0.001):
    """Main training function using MLX."""
    logger.info("=== Variable Binding Training (MLX) ===")
    logger.info(f"Device: {mx.default_device()}")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = get_output_path('binding_training', f'mlx_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate data
    logger.info("Generating training data...")
    train_data = create_simple_binding_data(n_samples=2000)
    val_data = create_simple_binding_data(n_samples=400)
    
    # Create model
    model = MinimalBindingModelMLX(
        vocab_size=len(train_data['command_vocab']),
        action_vocab_size=len(train_data['action_vocab']),
        n_slots=10,
        embed_dim=128,
        hidden_dim=256
    )
    
    # Initialize parameters
    mx.eval(model.parameters())
    num_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    logger.info(f"Model initialized with {num_params:,} parameters")
    
    # Create optimizer
    optimizer = optim.Adam(learning_rate=learning_rate)
    
    # Create value_and_grad function
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    
    # Training history
    history = {'loss': [], 'val_accuracy': []}
    
    # Training loop
    logger.info("\nStarting training...")
    n_samples = len(train_data['commands'])
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Shuffle data
        perm = mx.array(np.random.permutation(n_samples))
        
        # Training
        epoch_loss = 0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            # Get batch
            ids = perm[i:i+batch_size]
            batch_x = train_data['commands'][ids]
            batch_y = train_data['actions'][ids]
            
            # Compute loss and gradients
            loss, grads = loss_and_grad_fn(model, batch_x, batch_y)
            
            # Update model
            optimizer.update(model, grads)
            
            # Evaluate to materialize computation
            mx.eval(model.parameters(), optimizer.state)
            
            epoch_loss += float(loss)
            n_batches += 1
        
        # Validation
        val_accuracy = eval_accuracy(model, val_data)
        
        # Record metrics
        avg_loss = epoch_loss / n_batches
        history['loss'].append(avg_loss)
        history['val_accuracy'].append(val_accuracy)
        
        epoch_time = time.time() - start_time
        logger.info(f"Epoch {epoch + 1}/{epochs} - "
                   f"Loss: {avg_loss:.4f}, "
                   f"Val Acc: {val_accuracy:.4f}, "
                   f"Time: {epoch_time:.2f}s")
        
        # Test predictions every 5 epochs
        if (epoch + 1) % 5 == 0:
            logger.info("  Sample predictions:")
            for j in range(3):
                cmd = train_data['commands'][j:j+1]
                act_true = train_data['actions'][j]
                
                outputs = model({'command': cmd})
                pred = mx.argmax(outputs['action_logits'][0], axis=-1)
                
                logger.info(f"    True: {act_true}, Pred: {pred}")
    
    # Save results
    logger.info("\nSaving results...")
    
    # Save model weights
    weights_dict = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(os.path.join(output_dir, 'model_weights.safetensors'), weights_dict)
    
    # Save history and config
    results = {
        'config': {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'model_params': num_params,
            'device': str(mx.default_device())
        },
        'history': history,
        'final_metrics': {
            'final_loss': history['loss'][-1],
            'final_val_accuracy': history['val_accuracy'][-1]
        }
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"\nFinal validation accuracy: {history['val_accuracy'][-1]:.4f}")
    
    return model, history


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        logger.info("Running quick test (5 epochs)...")
        model, history = train_mlx(epochs=5, batch_size=64)
    else:
        logger.info("Running full training (20 epochs)...")
        model, history = train_mlx(epochs=20, batch_size=64)