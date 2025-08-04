"""
Simplified Variable Binding model in MLX for Apple Silicon.

This version uses simpler operations to avoid dimension mismatches
and focuses on demonstrating better GPU utilization.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map, tree_flatten

import numpy as np
import time
import json
import os
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleBindingModel(nn.Module):
    """
    Simplified binding model using MLX operations.
    """
    
    def __init__(self, vocab_size: int, action_vocab_size: int, 
                 embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        
        # Embeddings
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Simple encoder (using Linear layers instead of RNN)
        self.encoder1 = nn.Linear(embed_dim, hidden_dim)
        self.encoder2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Binding mechanism (simplified)
        self.bind_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Decoder
        self.decoder1 = nn.Linear(hidden_dim, hidden_dim)
        self.decoder2 = nn.Linear(hidden_dim, action_vocab_size)
        
    def __call__(self, x):
        # Embed input
        embeds = self.embedding(x)  # (batch, seq_len, embed_dim)
        
        # Encode with simple feedforward
        h = nn.relu(self.encoder1(embeds))
        h = nn.relu(self.encoder2(h))
        
        # Simple binding mechanism (attention-like)
        binding_scores = self.bind_proj(h)
        attention = mx.softmax(binding_scores, axis=-1)
        
        # Apply binding (weighted sum) - use attention as weights across hidden dim
        # attention shape: (batch, seq_len, hidden_dim)
        # h shape: (batch, seq_len, hidden_dim)
        bound_repr = mx.sum(h * attention, axis=1, keepdims=True)
        bound_repr = mx.repeat(bound_repr, h.shape[1], axis=1)
        
        # Decode
        output = h + bound_repr  # Residual connection
        output = nn.relu(self.decoder1(output))
        logits = self.decoder2(output)
        
        return logits


def create_data(n_samples=1000):
    """Create simple training data."""
    vocab_size = 20
    action_vocab_size = 10
    seq_len = 8
    
    # Random data for testing
    commands = np.random.randint(0, vocab_size, (n_samples, seq_len))
    actions = np.random.randint(0, action_vocab_size, (n_samples, seq_len))
    
    return {
        'commands': mx.array(commands),
        'actions': mx.array(actions),
        'vocab_size': vocab_size,
        'action_vocab_size': action_vocab_size
    }


def loss_fn(model, x, y):
    """Cross-entropy loss."""
    logits = model(x)
    return mx.mean(nn.losses.cross_entropy(logits, y))


def train():
    """Main training function."""
    logger.info("=== MLX Variable Binding Training (Simplified) ===")
    logger.info(f"Device: {mx.default_device()}")
    
    # Parameters
    epochs = 10
    batch_size = 128
    learning_rate = 0.001
    
    # Generate data
    logger.info("Generating data...")
    train_data = create_data(n_samples=5000)
    val_data = create_data(n_samples=1000)
    
    # Create model
    model = SimpleBindingModel(
        vocab_size=train_data['vocab_size'],
        action_vocab_size=train_data['action_vocab_size'],
        embed_dim=128,
        hidden_dim=256
    )
    
    # Initialize
    mx.eval(model.parameters())
    num_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    logger.info(f"Model parameters: {num_params:,}")
    
    # Optimizer
    optimizer = optim.Adam(learning_rate=learning_rate)
    
    # Training function
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    
    # Training loop
    n_samples = len(train_data['commands'])
    steps_per_epoch = n_samples // batch_size
    
    logger.info(f"\nTraining for {epochs} epochs, {steps_per_epoch} steps per epoch...")
    logger.info(f"Batch size: {batch_size}")
    
    history = {'loss': [], 'val_loss': [], 'gpu_time': []}
    
    for epoch in range(epochs):
        start_time = time.time()
        epoch_loss = 0
        
        # Shuffle
        perm = mx.array(np.random.permutation(n_samples))
        
        # Training steps
        for step in range(steps_per_epoch):
            # Get batch
            idx = perm[step * batch_size:(step + 1) * batch_size]
            batch_x = train_data['commands'][idx]
            batch_y = train_data['actions'][idx]
            
            # Forward and backward
            loss, grads = loss_and_grad_fn(model, batch_x, batch_y)
            
            # Update
            optimizer.update(model, grads)
            
            # Force evaluation
            mx.eval(model.parameters(), optimizer.state)
            
            epoch_loss += float(loss)
        
        # Validation
        val_loss = 0
        val_steps = 100  # Sample validation
        for i in range(val_steps):
            idx = mx.array(np.random.choice(len(val_data['commands']), batch_size))
            batch_x = val_data['commands'][idx]
            batch_y = val_data['actions'][idx]
            
            loss = loss_fn(model, batch_x, batch_y)
            mx.eval(loss)
            val_loss += float(loss)
        
        # Metrics
        avg_loss = epoch_loss / steps_per_epoch
        avg_val_loss = val_loss / val_steps
        epoch_time = time.time() - start_time
        
        history['loss'].append(avg_loss)
        history['val_loss'].append(avg_val_loss)
        history['gpu_time'].append(epoch_time)
        
        logger.info(f"Epoch {epoch+1}/{epochs} - "
                   f"Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                   f"Time: {epoch_time:.2f}s ({steps_per_epoch/epoch_time:.1f} steps/s)")
    
    # Summary
    logger.info("\n=== Training Complete ===")
    logger.info(f"Final training loss: {history['loss'][-1]:.4f}")
    logger.info(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    logger.info(f"Average epoch time: {np.mean(history['gpu_time']):.2f}s")
    logger.info(f"Total training time: {sum(history['gpu_time']):.2f}s")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"outputs/mlx_simple_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model weights
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(os.path.join(output_dir, "model.safetensors"), weights)
    
    # Save history
    with open(os.path.join(output_dir, "history.json"), 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_dir}")
    
    return model, history


if __name__ == "__main__":
    # Run training
    model, history = train()
    
    # Test inference speed
    logger.info("\n=== Testing Inference Speed ===")
    
    # Create test batch
    test_batch = mx.random.randint(0, 20, (256, 8))
    
    # Warmup
    for _ in range(10):
        _ = model(test_batch)
        mx.eval(model(test_batch))
    
    # Time inference
    n_iterations = 100
    start = time.time()
    
    for _ in range(n_iterations):
        output = model(test_batch)
        mx.eval(output)
    
    inference_time = time.time() - start
    logger.info(f"Inference time for {n_iterations} batches: {inference_time:.2f}s")
    logger.info(f"Average per batch: {inference_time/n_iterations*1000:.2f}ms")
    logger.info(f"Throughput: {n_iterations * 256 / inference_time:.0f} samples/s")