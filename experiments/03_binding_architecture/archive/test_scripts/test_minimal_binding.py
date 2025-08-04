"""Minimal test to debug binding gradient flow"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Test if gradients flow through soft attention
print("=== Testing Soft Attention Gradient Flow ===\n")

# Simple parameters
batch_size = 2
seq_len = 3
embed_dim = 8
num_slots = 4

# Create data
words = mx.random.normal((batch_size, seq_len, embed_dim))
slot_keys = mx.random.normal((num_slots, embed_dim))
slot_values = mx.random.normal((num_slots, embed_dim))
target = mx.array([[1, 2, 0], [2, 1, 0]])  # Target actions

# Define a simple model class
class SimpleBindingModel(nn.Module):
    def __init__(self, embed_dim, num_actions=3):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.decoder = nn.Linear(embed_dim, num_actions)
        
    def __call__(self, words, slot_keys, slot_values):
        batch_size, seq_len, embed_dim = words.shape
        num_slots = slot_keys.shape[0]
        
        # Compute attention
        Q = self.q_proj(words)
        K = self.k_proj(slot_keys[None, :, :])
        K = mx.broadcast_to(K, (batch_size, num_slots, embed_dim))
        
        # Attention scores
        scores = mx.mean(Q[:, :, None, :] * K[:, None, :, :], axis=-1)
        soft_scores = mx.softmax(scores, axis=-1)
        
        # Retrieve values using soft attention
        slot_values_exp = slot_values[None, None, :, :]
        slot_values_exp = mx.broadcast_to(slot_values_exp, 
                                         (batch_size, seq_len, num_slots, embed_dim))
        soft_scores_exp = soft_scores[:, :, :, None]
        retrieved = mx.sum(slot_values_exp * soft_scores_exp, axis=2)
        
        # Decode to actions
        logits = self.decoder(retrieved)
        
        return logits, soft_scores

# Create model
model = SimpleBindingModel(embed_dim)

# Forward pass and loss
def compute_loss(model):
    logits, soft_scores = model(words, slot_keys, slot_values)
    
    # Cross entropy loss
    loss_fn = nn.losses.cross_entropy
    loss = mx.mean(loss_fn(logits.reshape(-1, 3), target.reshape(-1)))
    
    return loss

# Compute gradients
loss_and_grad_fn = mx.value_and_grad(compute_loss)
loss, grads = loss_and_grad_fn(model)

print(f"Loss: {loss.item():.4f}")

# Check gradients
print("\nGradient norms:")
for name, grad in grads.items():
    if isinstance(grad, dict):
        for sub_name, sub_grad in grad.items():
            norm = mx.sqrt(mx.sum(sub_grad * sub_grad)).item()
            print(f"  {name}.{sub_name}: {norm:.6f}")

# Now test with Gumbel-Softmax
print("\n\n=== Testing Gumbel-Softmax Gradient Flow ===\n")

class GumbelBindingModel(nn.Module):
    def __init__(self, embed_dim, num_actions=3, temperature=1.0):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.decoder = nn.Linear(embed_dim, num_actions)
        self.temperature = temperature
        
    def gumbel_softmax(self, logits):
        # Add Gumbel noise
        shape = logits.shape
        uniform = mx.random.uniform(shape=shape, low=1e-8, high=1.0)
        gumbel = -mx.log(-mx.log(uniform))
        
        # Apply temperature
        y_soft = mx.softmax((logits + gumbel) / self.temperature, axis=-1)
        return y_soft
        
    def __call__(self, words, slot_keys, slot_values):
        batch_size, seq_len, embed_dim = words.shape
        num_slots = slot_keys.shape[0]
        
        # Compute attention
        Q = self.q_proj(words)
        K = self.k_proj(slot_keys[None, :, :])
        K = mx.broadcast_to(K, (batch_size, num_slots, embed_dim))
        
        # Attention scores
        scores = mx.mean(Q[:, :, None, :] * K[:, None, :, :], axis=-1)
        
        # Use Gumbel-Softmax
        soft_scores = self.gumbel_softmax(scores)
        
        # Retrieve values using soft attention
        slot_values_exp = slot_values[None, None, :, :]
        slot_values_exp = mx.broadcast_to(slot_values_exp, 
                                         (batch_size, seq_len, num_slots, embed_dim))
        soft_scores_exp = soft_scores[:, :, :, None]
        retrieved = mx.sum(slot_values_exp * soft_scores_exp, axis=2)
        
        # Decode to actions
        logits = self.decoder(retrieved)
        
        return logits, soft_scores

# Test with different temperatures
for temp in [5.0, 2.0, 1.0, 0.5]:
    print(f"\nTemperature = {temp}")
    gumbel_model = GumbelBindingModel(embed_dim, temperature=temp)
    
    # Compute loss and gradients
    def compute_gumbel_loss(model):
        logits, _ = model(words, slot_keys, slot_values)
        loss_fn = nn.losses.cross_entropy
        loss = mx.mean(loss_fn(logits.reshape(-1, 3), target.reshape(-1)))
        return loss
    
    loss_and_grad_fn = mx.value_and_grad(compute_gumbel_loss)
    loss, grads = loss_and_grad_fn(gumbel_model)
    
    print(f"  Loss: {loss.item():.4f}")
    
    # Check q_proj and k_proj gradients
    q_grad_norm = mx.sqrt(mx.sum(grads['q_proj']['weight'] ** 2)).item()
    k_grad_norm = mx.sqrt(mx.sum(grads['k_proj']['weight'] ** 2)).item()
    
    print(f"  Q projection grad norm: {q_grad_norm:.6f}")
    print(f"  K projection grad norm: {k_grad_norm:.6f}")