"""Debug why binder projections have zero gradients"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

def test_simple_attention_gradients():
    """Test gradient flow through a simple attention mechanism"""
    print("=== Testing Simple Attention Gradient Flow ===\n")
    
    # Create simple test data
    batch_size = 2
    seq_len = 3
    embed_dim = 8
    num_slots = 4
    
    # Initialize simple attention components
    q_proj = nn.Linear(embed_dim, embed_dim)
    k_proj = nn.Linear(embed_dim, embed_dim)
    
    # Random inputs
    words = mx.random.normal((batch_size, seq_len, embed_dim))
    slot_keys = mx.random.normal((num_slots, embed_dim))
    
    # Target for loss (random actions)
    target = mx.array(np.random.randint(0, 5, (batch_size, seq_len)))
    
    def forward():
        # Project inputs
        Q = q_proj(words)  # (batch, seq_len, embed_dim)
        K = k_proj(slot_keys[None, :, :])  # (1, num_slots, embed_dim)
        K = mx.broadcast_to(K, (batch_size, num_slots, embed_dim))
        
        # Compute attention scores
        scores = mx.mean(Q[:, :, None, :] * K[:, None, :, :], axis=-1)  # (batch, seq_len, num_slots)
        
        # Soft attention (differentiable)
        soft_scores = mx.softmax(scores, axis=-1)
        
        # Use soft scores to retrieve values (simulate slot values)
        slot_values = mx.random.normal((num_slots, embed_dim))
        slot_values_expanded = slot_values[None, None, :, :]
        slot_values_expanded = mx.broadcast_to(slot_values_expanded, 
                                              (batch_size, seq_len, num_slots, embed_dim))
        
        # Weighted sum
        soft_scores_expanded = soft_scores[:, :, :, None]
        retrieved = mx.sum(slot_values_expanded * soft_scores_expanded, axis=2)
        
        # Simple decoder (just project to actions)
        decoder = nn.Linear(embed_dim, 5)
        logits = decoder(retrieved)
        
        # Loss
        loss_fn = nn.losses.cross_entropy
        loss = mx.mean(loss_fn(logits.reshape(-1, 5), target.reshape(-1)))
        
        return loss, soft_scores
    
    # Create a model dict for gradient computation
    model = {'q_proj': q_proj, 'k_proj': k_proj}
    
    def loss_wrapper(model):
        return forward()[0]
    
    # Compute gradients
    value_and_grad_fn = mx.value_and_grad(loss_wrapper)
    loss, grads = value_and_grad_fn(model)
    
    print(f"Loss: {loss.item():.4f}")
    print(f"\nQ projection grad norm: {mx.sqrt(mx.sum(q_proj.weight * q_proj.weight)).item():.6f}")
    print(f"K projection grad norm: {mx.sqrt(mx.sum(k_proj.weight * k_proj.weight)).item():.6f}")
    
    # Check if parameters would change
    lr = 0.001
    q_before = mx.array(q_proj.weight)
    k_before = mx.array(k_proj.weight)
    
    # Simulate update
    optimizer = optim.Adam(learning_rate=lr)
    optimizer.update({'q': q_proj, 'k': k_proj}, grads)
    
    q_change = mx.mean(mx.abs(q_proj.weight - q_before)).item()
    k_change = mx.mean(mx.abs(k_proj.weight - k_before)).item()
    
    print(f"\nQ weight change: {q_change:.6f}")
    print(f"K weight change: {k_change:.6f}")


def test_gumbel_softmax_gradients():
    """Test gradient flow through Gumbel-Softmax"""
    print("\n\n=== Testing Gumbel-Softmax Gradient Flow ===\n")
    
    # Simple setup
    logits = mx.array([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0]])
    target = mx.array([2, 0])  # Target indices
    
    def gumbel_softmax(logits, temperature=1.0):
        # Sample from Gumbel(0, 1)
        shape = logits.shape
        uniform = mx.random.uniform(shape=shape, low=1e-8, high=1.0)
        gumbel = -mx.log(-mx.log(uniform))
        
        # Add Gumbel noise and apply temperature
        y_soft = mx.softmax((logits + gumbel) / temperature, axis=-1)
        return y_soft
    
    def loss_fn():
        soft_probs = gumbel_softmax(logits, temperature=1.0)
        # Use soft probabilities to compute loss
        loss = -mx.mean(mx.log(soft_probs[mx.arange(2), target] + 1e-8))
        return loss
    
    # Compute gradients
    value_and_grad_fn = mx.value_and_grad(loss_fn)
    loss, grads = value_and_grad_fn()
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Logits gradient: {grads}")
    print(f"Gradient norm: {mx.sqrt(mx.sum(grads * grads)).item():.6f}")


def test_binding_attention_simplified():
    """Test our actual BindingAttention mechanism in isolation"""
    print("\n\n=== Testing BindingAttention Gradient Flow ===\n")
    
    from train_binding_mlx_proper import BindingAttention
    
    # Create binder
    binder = BindingAttention(embed_dim=32, num_heads=4, temperature=2.0)
    
    # Test inputs
    batch_size = 2
    seq_len = 4
    num_slots = 5
    embed_dim = 32
    
    words = mx.random.normal((batch_size, seq_len, embed_dim))
    slot_keys = mx.random.normal((num_slots, embed_dim))
    
    # Forward pass
    def compute_loss():
        bindings, soft_scores = binder(words, slot_keys, training=True)
        
        # Simulate using soft scores for something differentiable
        # (In real model, this is used for slot value retrieval)
        slot_values = mx.random.normal((num_slots, embed_dim))
        
        # Expand for batch processing
        slot_values_exp = slot_values[None, None, :, :]
        slot_values_exp = mx.broadcast_to(slot_values_exp, 
                                         (batch_size, seq_len, num_slots, embed_dim))
        soft_scores_exp = soft_scores[:, :, :, None]
        
        # Weighted retrieval
        retrieved = mx.sum(slot_values_exp * soft_scores_exp, axis=2)
        
        # Simple loss - encourage diversity in slot usage
        slot_usage = mx.mean(soft_scores, axis=(0, 1))  # Average over batch and seq
        entropy = -mx.sum(slot_usage * mx.log(slot_usage + 1e-8))
        
        # Also add a reconstruction-like loss
        reconstruction_loss = mx.mean((retrieved - words) ** 2)
        
        total_loss = reconstruction_loss - 0.1 * entropy
        return total_loss
    
    # Get gradients
    value_and_grad_fn = mx.value_and_grad(compute_loss)
    loss, grads = value_and_grad_fn()
    
    print(f"Loss: {loss.item():.4f}")
    
    # Check gradients for binder components
    if 'binder' in grads:
        binder_grads = grads['binder']
        for name, grad in binder_grads.items():
            if hasattr(grad, 'shape'):
                norm = mx.sqrt(mx.sum(grad * grad)).item()
                print(f"{name} gradient norm: {norm:.6f}")


if __name__ == "__main__":
    test_simple_attention_gradients()
    test_gumbel_softmax_gradients()
    test_binding_attention_simplified()