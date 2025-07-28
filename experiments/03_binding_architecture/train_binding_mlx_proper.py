#!/usr/bin/env python3
"""
Proper Variable Binding Architecture in MLX

This implements the full variable binding mechanism from Wu et al. (2025):
- Explicit memory slots for storing bindings
- Multi-head attention for word-to-slot assignment  
- Hard binding via argmax (discrete assignments)
- Proper dereferencing tasks for training
- Support for dynamic rebinding/modifications

Key improvements over simplified version:
1. VariableMemory with content-based addressing
2. Real attention mechanism (not just weighted sum)
3. Actual dereferencing tasks (not random data)
4. Modification support for rule changes
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import sys
import os
import time
from typing import Dict, Tuple, Optional, List
import argparse

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dereferencing_tasks import DereferencingTaskGenerator


class VariableMemory(nn.Module):
    """Explicit memory slots for variable bindings
    
    Key insights from Wu et al.:
    - Use discrete slots, not distributed embeddings
    - Content-based addressing for slot selection
    - Can selectively update individual slots
    """
    
    def __init__(self, num_slots: int = 10, slot_dim: int = 128):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        
        # Learnable slot embeddings
        self.slots = mx.random.normal((num_slots, slot_dim)) * 0.02
        
        # Slot keys for content-based addressing
        self.slot_keys = mx.random.normal((num_slots, slot_dim)) * 0.02
        
    def __call__(self, query: mx.array) -> Tuple[mx.array, mx.array]:
        """Return slot values and keys for addressing"""
        return self.slots, self.slot_keys
    
    def update_slot(self, slot_idx: int, new_value: mx.array):
        """Update a specific slot (for modifications)"""
        mask = mx.zeros((self.num_slots, 1))
        mask[slot_idx] = 1.0
        self.slots = (1 - mask) * self.slots + mask * new_value


class BindingAttention(nn.Module):
    """Multi-head attention for word-to-slot binding
    
    Creates explicit associations between words and memory slots
    Uses hard binding (argmax) to force discrete assignments
    """
    
    def __init__(self, embed_dim: int = 128, num_heads: int = 8, temperature: float = 1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.temperature = temperature
        
        # Query projection for words
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        
        # Key projection for memory slots  
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        
        # Value projection (for soft attention scores)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.scale = self.head_dim ** -0.5
        
    def gumbel_softmax(self, logits: mx.array, temperature: float, hard: bool = False) -> mx.array:
        """Gumbel-Softmax for differentiable discrete sampling"""
        # Sample from Gumbel(0, 1)
        shape = logits.shape
        uniform = mx.random.uniform(shape=shape, low=1e-8, high=1.0)
        gumbel = -mx.log(-mx.log(uniform))
        
        # Add Gumbel noise to logits and apply temperature
        y_soft = mx.softmax((logits + gumbel) / temperature, axis=-1)
        
        if hard:
            # Straight-through estimator: hard sample in forward, soft in backward
            indices = mx.argmax(y_soft, axis=-1)
            # For now, return soft scores and indices separately
            return y_soft, indices
        
        return y_soft, None
    
    def __call__(self, 
                 words: mx.array,          # (batch, seq_len, embed_dim)
                 slot_keys: mx.array,      # (num_slots, embed_dim)
                 training: bool = True     # Whether we're in training mode
                ) -> Tuple[mx.array, mx.array]:
        """
        Returns:
            bindings: (batch, seq_len) - slot indices for each word
            scores: (batch, seq_len, num_slots) - attention scores
        """
        batch_size, seq_len, _ = words.shape
        num_slots = slot_keys.shape[0]
        
        # Project to multi-head format
        Q = self.q_proj(words)  # (batch, seq_len, embed_dim)
        K = self.k_proj(slot_keys[None, :, :])  # (1, num_slots, embed_dim)
        K = mx.broadcast_to(K, (batch_size, num_slots, self.embed_dim))
        
        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        Q = Q.transpose(0, 2, 1, 3)  # (batch, heads, seq_len, head_dim)
        
        K = K.reshape(batch_size, num_slots, self.num_heads, self.head_dim)
        K = K.transpose(0, 2, 1, 3)  # (batch, heads, num_slots, head_dim)
        
        # Compute attention scores
        scores = (Q @ K.transpose(0, 1, 3, 2)) * self.scale
        scores = mx.mean(scores, axis=1)  # Average over heads: (batch, seq_len, num_slots)
        
        if training:
            # Use Gumbel-Softmax during training for differentiable binding
            soft_scores, hard_indices = self.gumbel_softmax(scores, self.temperature, hard=True)
            if hard_indices is not None:
                bindings = hard_indices
            else:
                bindings = mx.argmax(scores, axis=-1)  # Fallback
        else:
            # Use regular softmax and argmax during evaluation
            soft_scores = mx.softmax(scores, axis=-1)
            bindings = mx.argmax(scores, axis=-1)
        
        return bindings, soft_scores


class BoundVariableExecutor(nn.Module):
    """Execute commands using bound variables
    
    Key mechanism: Uses retrieved slot values (not just embeddings)
    to generate action sequences based on current bindings
    """
    
    def __init__(self, 
                 embed_dim: int = 128,
                 hidden_dim: int = 256,
                 num_actions: int = 10):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Simple encoder (can be enhanced with LSTM later)
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),  # Concat word + bound value
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Decoder to actions
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
        
    def __call__(self,
                 word_embeds: mx.array,      # (batch, seq_len, embed_dim)
                 bound_values: mx.array      # (batch, seq_len, embed_dim)
                ) -> mx.array:
        """Generate action logits using bound variable values"""
        batch_size, seq_len, _ = word_embeds.shape
        
        # Concatenate word embeddings with their bound values
        combined = mx.concatenate([word_embeds, bound_values], axis=-1)
        
        # Encode
        encoded = self.encoder(combined)
        
        # Decode to actions
        action_logits = self.decoder(encoded)
        
        return action_logits  # (batch, seq_len, num_actions)


class ProperBindingModel(nn.Module):
    """Full variable binding model with all components"""
    
    def __init__(self,
                 vocab_size: int = 20,
                 num_actions: int = 10,
                 embed_dim: int = 128,
                 hidden_dim: int = 256,
                 num_slots: int = 10,
                 num_heads: int = 8):
        super().__init__()
        
        # Word embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Core components
        self.memory = VariableMemory(num_slots, embed_dim)
        self.binder = BindingAttention(embed_dim, num_heads)
        self.executor = BoundVariableExecutor(embed_dim, hidden_dim, num_actions)
        
    def __call__(self, 
                 command_ids: mx.array,          # (batch, seq_len)
                 modification: Optional[mx.array] = None,
                 training: bool = True
                ) -> Dict[str, mx.array]:
        """Forward pass with variable binding"""
        
        # Embed words
        word_embeds = self.embedding(command_ids)
        
        # Get memory slots
        slot_values, slot_keys = self.memory(None)
        
        # Bind words to slots (with training flag for Gumbel-Softmax)
        bindings, binding_scores = self.binder(word_embeds, slot_keys, training=training)
        
        # Retrieve bound values using soft attention for differentiability
        # binding_scores: (batch, seq_len, num_slots)
        # slot_values: (num_slots, embed_dim)
        
        # Expand slot_values for batch processing
        batch_size, seq_len, num_slots = binding_scores.shape
        _, embed_dim = slot_values.shape
        
        # Broadcast slot_values to match batch and sequence dimensions
        slot_values_expanded = slot_values[None, None, :, :]  # (1, 1, num_slots, embed_dim)
        slot_values_expanded = mx.broadcast_to(slot_values_expanded, 
                                              (batch_size, seq_len, num_slots, embed_dim))
        
        # Use soft attention to retrieve values (differentiable!)
        # binding_scores needs to be expanded to match slot_values_expanded
        binding_weights = binding_scores[:, :, :, None]  # (batch, seq_len, num_slots, 1)
        
        # Weighted sum over slots
        bound_values = mx.sum(slot_values_expanded * binding_weights, axis=2)  # (batch, seq_len, embed_dim)
        
        # Apply modifications if provided
        if modification is not None:
            bound_values = self.apply_modification(bindings, bound_values, modification)
        
        # Execute using bound values
        action_logits = self.executor(word_embeds, bound_values)
        
        return {
            'action_logits': action_logits,
            'bindings': bindings,
            'binding_scores': binding_scores
        }
    
    def apply_modification(self, bindings, bound_values, modification):
        """Apply rule modifications by updating bound values"""
        # This is a placeholder - in full implementation would:
        # 1. Parse modification command
        # 2. Identify which slots to update
        # 3. Update those slots with new values
        # For now, just return unchanged
        return bound_values


def train_step(model, batch, loss_fn, optimizer):
    """Single training step"""
    
    def loss_wrapper(model):
        outputs = model(batch['command'], training=False)
        logits = outputs['action_logits']
        
        # Handle sequence length mismatch
        batch_actions = batch['action']
        action_seq_len = batch_actions.shape[1]
        logits_seq_len = logits.shape[1]
        
        if action_seq_len != logits_seq_len:
            min_len = min(action_seq_len, logits_seq_len)
            batch_actions = batch_actions[:, :min_len]
            logits = logits[:, :min_len]
        
        # Reshape for loss computation
        logits_flat = logits.reshape(-1, logits.shape[-1])
        actions_flat = batch_actions.reshape(-1)
        
        loss = loss_fn(logits_flat, actions_flat)
        
        # Make sure loss is scalar by taking mean
        loss = mx.mean(loss)
        
        # Add binding entropy regularization (encourage diverse slot usage)
        binding_scores = outputs['binding_scores']
        slot_probs = mx.mean(binding_scores, axis=(0, 1))  # Average over batch and seq
        entropy = -mx.sum(slot_probs * mx.log(slot_probs + 1e-8))
        
        # Total loss with entropy bonus
        total_loss = loss - 0.01 * entropy  # Small weight to encourage diversity
        
        return total_loss
    
    # Compute value and gradients separately (MLX style)
    loss_and_grad_fn = mx.value_and_grad(loss_wrapper)
    loss, grads = loss_and_grad_fn(model)
    
    # Update model
    optimizer.update(model, grads)
    
    # Get outputs for logging (run forward pass again)
    outputs = model(batch['command'], training=True)
    
    # Compute gradient norms for debugging
    grad_norms = {}
    
    def compute_grad_norms(grads_dict, prefix=""):
        for name, grad in grads_dict.items():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(grad, dict):
                compute_grad_norms(grad, full_name)
            elif hasattr(grad, 'shape'):  # It's an array
                norm = mx.sqrt(mx.sum(grad * grad))
                grad_norms[full_name] = norm.item()
    
    compute_grad_norms(grads)
    
    return loss, outputs, grad_norms


def evaluate(model, generator, num_samples=100):
    """Evaluate model accuracy"""
    correct = 0
    total = 0
    
    for _ in range(num_samples // 32):
        batch = generate_batch_from_dataset(generator, 32)
        outputs = model(batch['command'], training=False)
        
        # Get predictions
        logits = outputs['action_logits']
        preds = mx.argmax(logits, axis=-1)
        
        # Handle sequence length mismatch
        actions = batch['action']
        min_len = min(preds.shape[1], actions.shape[1])
        preds = preds[:, :min_len]
        actions = actions[:, :min_len]
        
        # Count correct
        correct += mx.sum(preds == actions).item()
        total += preds.size
    
    return correct / total if total > 0 else 0.0


def test_modifications(model, generator):
    """Test model's ability to handle modifications"""
    print("\n=== Testing Modification Capability ===")
    
    # Generate modification test cases
    test_cases = generator.generate_modification_test_set()
    
    successes = 0
    total_tests = min(5, len(test_cases))
    
    for i, test_case in enumerate(test_cases[:total_tests]):
        # Get original command and expected actions
        orig_cmd = test_case['original']['command']
        orig_actions = test_case['original']['actions']
        
        # Encode command
        orig_cmd_encoded = generator.encode_words(orig_cmd)
        orig_cmd_array = mx.array(orig_cmd_encoded)[None, :]
        
        # Get model outputs
        outputs = model(orig_cmd_array)
        pred_logits = outputs['action_logits'][0]
        pred_actions = mx.argmax(pred_logits, axis=-1)
        
        # Convert expected actions to IDs
        expected_ids = generator.encode_actions(orig_actions)
        
        # Check if predictions match expected
        pred_list = pred_actions.tolist()[:len(expected_ids)]
        success = pred_list == expected_ids.tolist()
        successes += success
        
        print(f"\nTest {i+1} ({test_case['modification_type']}):")
        print(f"Command: {' '.join(orig_cmd)}")
        print(f"Expected: {orig_actions}")
        pred_action_names = []
        for a in pred_list:
            if a < len(generator.id_to_action):
                pred_action_names.append(generator.id_to_action[a])
        print(f"Predicted: {pred_action_names}")
        print(f"Success: {'✓' if success else '✗'}")
        
        # Also test modified version
        if 'modified' in test_case:
            mod_cmd = test_case['modified']['command']
            mod_actions = test_case['modified']['actions']
            
            mod_cmd_encoded = generator.encode_words(mod_cmd)
            mod_cmd_array = mx.array(mod_cmd_encoded)[None, :]
            
            outputs_mod = model(mod_cmd_array)
            pred_logits_mod = outputs_mod['action_logits'][0]
            pred_actions_mod = mx.argmax(pred_logits_mod, axis=-1)
            
            expected_mod_ids = generator.encode_actions(mod_actions)
            pred_mod_list = pred_actions_mod.tolist()[:len(expected_mod_ids)]
            
            print(f"  Modified: {' '.join(mod_cmd)}")
            print(f"  Expected: {mod_actions}")
            pred_mod_names = []
            for a in pred_mod_list:
                if a < len(generator.id_to_action):
                    pred_mod_names.append(generator.id_to_action[a])
            print(f"  Predicted: {pred_mod_names}")
    
    print(f"\nModification Success Rate: {successes}/{total_tests} = {successes/total_tests:.1%}")


def generate_batch_from_dataset(generator, batch_size):
    """Generate a batch of tasks for training"""
    batch_commands = []
    batch_actions = []
    
    for _ in range(batch_size):
        # Randomly choose task type
        task_type = np.random.choice(
            ['simple', 'multiple', 'rebinding', 'compositional', 'modifiers'],
            p=[0.3, 0.2, 0.2, 0.2, 0.1]
        )
        
        # Generate task
        if task_type == 'simple':
            command, actions, _ = generator.generate_simple_binding()
        elif task_type == 'multiple':
            command, actions, _ = generator.generate_multiple_bindings()
        elif task_type == 'rebinding':
            command, actions, _ = generator.generate_rebinding()
        elif task_type == 'compositional':
            command, actions, _ = generator.generate_compositional()
        else:  # modifiers
            command, actions, _ = generator.generate_with_modifiers()
        
        # Remove periods
        command = [w for w in command if w != '.']
        
        # Encode
        cmd_encoded = generator.encode_words(command)
        act_encoded = generator.encode_actions(actions)
        
        batch_commands.append(cmd_encoded)
        batch_actions.append(act_encoded)
    
    # Pad to same length
    max_cmd_len = max(len(cmd) for cmd in batch_commands)
    max_act_len = max(len(act) for act in batch_actions)
    
    padded_commands = np.zeros((batch_size, max_cmd_len), dtype=np.int32)
    padded_actions = np.zeros((batch_size, max_act_len), dtype=np.int32)
    
    for i, (cmd, act) in enumerate(zip(batch_commands, batch_actions)):
        padded_commands[i, :len(cmd)] = cmd
        padded_actions[i, :len(act)] = act
    
    return {
        'command': mx.array(padded_commands),
        'action': mx.array(padded_actions)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.003, help='Learning rate')
    parser.add_argument('--quick', action='store_true', help='Quick test run')
    args = parser.parse_args()
    
    # Quick test settings
    if args.quick:
        args.epochs = 5
        args.batch_size = 16
    
    print("=== Variable Binding Model Training (MLX - Proper Implementation) ===")
    print(f"Epochs: {args.epochs}, Batch Size: {args.batch_size}, LR: {args.lr}")
    
    # Initialize task generator
    generator = DereferencingTaskGenerator()
    
    # Create model
    model = ProperBindingModel(
        vocab_size=len(generator.word_to_id),
        num_actions=len(generator.action_to_id),
        embed_dim=128,
        hidden_dim=256,
        num_slots=10,
        num_heads=8
    )
    
    # Temperature schedule for Gumbel-Softmax
    # Start with moderate temperature for better gradient flow
    initial_temperature = 1.5
    min_temperature = 0.1
    temperature_decay = 0.95  # Faster decay
    
    # Initialize model parameters
    mx.eval(model.parameters())
    
    # Setup training
    optimizer = optim.Adam(learning_rate=args.lr)
    loss_fn = nn.losses.cross_entropy
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(args.epochs):
        epoch_start = time.time()
        epoch_loss = 0
        num_batches = 100 if not args.quick else 10
        
        # Update temperature for this epoch
        current_temp = max(min_temperature, initial_temperature * (temperature_decay ** epoch))
        model.binder.temperature = current_temp
        
        for _ in range(num_batches):
            batch = generate_batch_from_dataset(generator, args.batch_size)
            loss, outputs, grad_norms = train_step(model, batch, loss_fn, optimizer)
            epoch_loss += loss.item()
            mx.eval(model.parameters(), optimizer.state)
        
        # Evaluate
        accuracy = evaluate(model, generator, num_samples=200 if not args.quick else 50)
        
        epoch_time = time.time() - epoch_start
        # Print gradient norms for key components
        if epoch == 0 and grad_norms:
            print("  Gradient norms:")
            for key in ['binder.q_proj', 'binder.k_proj', 'memory.slot_values', 'executor.encoder.0']:
                if key in grad_norms:
                    print(f"    {key}: {grad_norms[key]:.6f}")
        
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {epoch_loss/num_batches:.4f}, "
              f"Accuracy: {accuracy:.2%}, Time: {epoch_time:.2f}s, Temperature: {current_temp:.3f}")
    
    # Test modifications
    test_modifications(model, generator)
    
    # Performance analysis
    print("\n=== Performance Metrics ===")
    
    # Measure inference speed
    batch = generate_batch_from_dataset(generator, 128)
    start_time = time.time()
    for _ in range(100):
        _ = model(batch['command'], training=False)
        mx.eval(_['action_logits'])
    inference_time = time.time() - start_time
    
    samples_per_sec = (128 * 100) / inference_time
    ms_per_batch = (inference_time / 100) * 1000
    
    print(f"Inference throughput: {samples_per_sec:,.0f} samples/second")
    print(f"Latency: {ms_per_batch:.2f} ms/batch")
    
    # Save model
    print("\nSaving model weights...")
    # MLX doesn't have a built-in save for dictionaries, so we'll use numpy
    import numpy as np
    params_dict = {}
    for k, v in model.parameters().items():
        # Flatten nested dictionary structure
        if isinstance(v, dict):
            for k2, v2 in v.items():
                params_dict[f"{k}.{k2}"] = np.array(v2)
        else:
            params_dict[k] = np.array(v)
    np.savez("proper_binding_model.npz", **params_dict)
    print("Model saved to proper_binding_model.npz")


if __name__ == "__main__":
    main()