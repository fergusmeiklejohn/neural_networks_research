"""Variable binding with contrastive learning and focused attention"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import argparse
import time

class FocusedBindingAttention(nn.Module):
    """Binding attention with stronger inductive biases"""
    
    def __init__(self, embed_dim: int = 128, num_heads: int = 8, 
                 num_slots: int = 4, temperature: float = 1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_slots = num_slots
        self.temperature = temperature
        
        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        
        # Variable type embedding (helps distinguish X, Y, A, B etc)
        self.var_type_embed = nn.Embedding(10, embed_dim)  # Support up to 10 variable types
        
        self.scale = self.head_dim ** -0.5
        
    def __call__(self, words, slot_keys, word_ids, training=True):
        batch_size, seq_len, _ = words.shape
        
        # Add variable type information to queries
        # Assume variables are uppercase letters (need to identify them)
        var_mask = self.identify_variables(word_ids)  # (batch, seq_len)
        var_embeds = self.var_type_embed(var_mask)
        
        # Combine word embeddings with variable type info
        enhanced_words = words + 0.1 * var_embeds  # Small contribution
        
        # Project
        Q = self.q_proj(enhanced_words)
        K = self.k_proj(slot_keys[None, :, :])
        K = mx.broadcast_to(K, (batch_size, self.num_slots, self.embed_dim))
        
        # Reshape for multi-head
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        Q = Q.transpose(0, 2, 1, 3)
        
        K = K.reshape(batch_size, self.num_slots, self.num_heads, self.head_dim)
        K = K.transpose(0, 2, 1, 3)
        
        # Attention scores
        scores = (Q @ K.transpose(0, 1, 3, 2)) * self.scale
        scores = mx.mean(scores, axis=1)  # (batch, seq_len, num_slots)
        
        if training:
            # Gumbel-Softmax
            shape = scores.shape
            uniform = mx.random.uniform(shape=shape, low=1e-8, high=1.0)
            gumbel = -mx.log(-mx.log(uniform))
            soft_scores = mx.softmax((scores + gumbel) / self.temperature, axis=-1)
        else:
            soft_scores = mx.softmax(scores, axis=-1)
        
        # For visualization, get hard assignments
        bindings = mx.argmax(scores, axis=-1)
        
        return bindings, soft_scores, var_mask
    
    def identify_variables(self, word_ids):
        """Identify which words are variables (X, Y, A, B, etc)"""
        # This is a simplified version - in practice would use vocabulary
        # For now, assign IDs based on word position patterns
        # Variables typically appear at positions 0 and 4 in "X means action do X"
        batch_size, seq_len = word_ids.shape
        var_ids = mx.zeros((batch_size, seq_len), dtype=mx.int32)
        
        # Simple heuristic: words at positions 0, 4, 7, etc might be variables
        for i in range(seq_len):
            if i % 3 == 0:  # Rough pattern
                var_ids[:, i] = (word_ids[:, i] % 4) + 1  # Map to 1-4 for X,Y,A,B
        
        return var_ids


class ContrastiveBindingModel(nn.Module):
    """Model with contrastive loss for variable identity"""
    
    def __init__(self, vocab_size: int, num_actions: int, embed_dim: int = 128,
                 hidden_dim: int = 256, num_slots: int = 4, num_heads: int = 8):
        super().__init__()
        
        self.num_slots = num_slots
        self.embed_dim = embed_dim
        
        # Core components
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Make slot keys and values proper parameters
        self.slot_keys = mx.random.normal((num_slots, embed_dim)) * 0.02
        self.slot_values = mx.random.normal((num_slots, embed_dim)) * 0.02
        
        self.binder = FocusedBindingAttention(embed_dim, num_heads, num_slots)
        
        # Action decoder
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
        
        # Contrastive projection head
        self.contrast_proj = nn.Linear(embed_dim, 64)
        
    def __call__(self, command_ids, training=True):
        # Embed
        word_embeds = self.embedding(command_ids)
        
        # Bind
        bindings, soft_scores, var_mask = self.binder(
            word_embeds, self.slot_keys, command_ids, training
        )
        
        # Retrieve values using soft attention
        batch_size, seq_len = command_ids.shape
        slot_values_exp = self.slot_values[None, None, :, :]
        slot_values_exp = mx.broadcast_to(slot_values_exp,
                                         (batch_size, seq_len, self.num_slots, self.embed_dim))
        soft_scores_exp = soft_scores[:, :, :, None]
        retrieved = mx.sum(slot_values_exp * soft_scores_exp, axis=2)
        
        # Decode
        combined = mx.concatenate([word_embeds, retrieved], axis=-1)
        action_logits = self.decoder(combined)
        
        # Contrastive features for variables
        var_features = None
        if training:
            # Get features for variable positions
            var_positions = var_mask > 0
            if mx.any(var_positions):
                var_embeds = word_embeds * var_positions[:, :, None]
                var_retrieved = retrieved * var_positions[:, :, None]
                var_combined = var_embeds + var_retrieved
                var_features = self.contrast_proj(var_combined)
        
        return {
            'action_logits': action_logits,
            'bindings': bindings,
            'soft_scores': soft_scores,
            'var_mask': var_mask,
            'var_features': var_features
        }


def contrastive_loss(var_features, var_mask, temperature=0.5):
    """Contrastive loss to ensure same variables get similar representations"""
    if var_features is None:
        return mx.array(0.0)
    
    batch_size, seq_len, feat_dim = var_features.shape
    
    # Flatten and normalize
    features_flat = var_features.reshape(-1, feat_dim)
    var_ids_flat = var_mask.reshape(-1)
    
    # Only consider actual variables (non-zero mask)
    valid_mask = var_ids_flat > 0
    if not mx.any(valid_mask):
        return mx.array(0.0)
    
    valid_features = features_flat * valid_mask[:, None]
    valid_ids = var_ids_flat * valid_mask
    
    # Normalize features
    norms = mx.sqrt(mx.sum(valid_features ** 2, axis=1, keepdims=True) + 1e-8)
    normalized = valid_features / norms
    
    # Compute similarity matrix
    sim_matrix = normalized @ normalized.T
    sim_matrix = sim_matrix / temperature
    
    # Create label matrix (1 if same variable, 0 otherwise)
    labels = valid_ids[:, None] == valid_ids[None, :]
    labels = labels * valid_mask[:, None] * valid_mask[None, :]
    
    # Mask out diagonal
    mask = 1 - mx.eye(len(valid_ids))
    labels = labels * mask
    
    # Compute contrastive loss
    exp_sim = mx.exp(sim_matrix) * mask
    pos_sim = mx.sum(exp_sim * labels, axis=1)
    all_sim = mx.sum(exp_sim, axis=1)
    
    loss = -mx.log(pos_sim / (all_sim + 1e-8) + 1e-8)
    
    # Only average over positions that have positive pairs
    has_pos = mx.sum(labels, axis=1) > 0
    if mx.any(has_pos):
        loss = mx.sum(loss * has_pos) / (mx.sum(has_pos) + 1e-8)
    else:
        loss = mx.array(0.0)
    
    return loss


def generate_binding_batch(batch_size, vocab_size):
    """Generate training batch with explicit variable binding"""
    commands = []
    actions = []
    
    # Simplified vocabulary indices
    var_to_id = {'X': 4, 'Y': 5, 'A': 6, 'B': 7}
    action_to_id = {'JUMP': 1, 'WALK': 2, 'TURN': 3, 'RUN': 4}
    word_to_id = {
        'means': 8, 'do': 9, 'then': 10, 'twice': 11,
        'jump': 12, 'walk': 13, 'turn': 14, 'run': 15
    }
    
    for _ in range(batch_size):
        if np.random.random() < 0.3:
            # Simple: "do action"
            action = np.random.choice(['JUMP', 'WALK', 'TURN', 'RUN'])
            cmd = [word_to_id['do'], word_to_id[action.lower()]]
            act = [action_to_id[action]]
        elif np.random.random() < 0.7:
            # Binding: "X means action do X"
            var = np.random.choice(['X', 'Y', 'A', 'B'])
            action = np.random.choice(['jump', 'walk', 'turn', 'run'])
            cmd = [var_to_id[var], word_to_id['means'], word_to_id[action], 
                   word_to_id['do'], var_to_id[var]]
            act = [action_to_id[action.upper()]]
        else:
            # Multiple: "X means action1 Y means action2 do X then Y"
            vars = np.random.choice(['X', 'Y', 'A', 'B'], size=2, replace=False)
            actions_chosen = np.random.choice(['jump', 'walk', 'turn', 'run'], size=2, replace=False)
            cmd = [var_to_id[vars[0]], word_to_id['means'], word_to_id[actions_chosen[0]],
                   var_to_id[vars[1]], word_to_id['means'], word_to_id[actions_chosen[1]],
                   word_to_id['do'], var_to_id[vars[0]], word_to_id['then'], var_to_id[vars[1]]]
            act = [action_to_id[actions_chosen[0].upper()], action_to_id[actions_chosen[1].upper()]]
        
        commands.append(cmd)
        actions.append(act)
    
    # Pad
    max_cmd_len = max(len(c) for c in commands)
    max_act_len = max(len(a) for a in actions)
    
    padded_commands = np.zeros((batch_size, max_cmd_len), dtype=np.int32)
    padded_actions = np.zeros((batch_size, max_act_len), dtype=np.int32)
    
    for i, (cmd, act) in enumerate(zip(commands, actions)):
        padded_commands[i, :len(cmd)] = cmd
        padded_actions[i, :len(act)] = act
    
    return mx.array(padded_commands), mx.array(padded_actions)


def train_step(model, commands, actions, optimizer):
    """Training step with contrastive loss"""
    def loss_fn(model):
        outputs = model(commands, training=True)
        
        # Action prediction loss
        logits = outputs['action_logits']
        batch_size, cmd_len, num_actions = logits.shape
        _, act_len = actions.shape
        
        # Handle sequence length mismatch
        min_len = min(cmd_len, act_len)
        logits_trimmed = logits[:, :min_len, :]
        actions_trimmed = actions[:, :min_len]
        
        logits_flat = logits_trimmed.reshape(-1, num_actions)
        actions_flat = actions_trimmed.reshape(-1)
        
        # Mask out padding (action 0)
        mask = actions_flat > 0
        if mx.any(mask):
            ce_loss = mx.sum(nn.losses.cross_entropy(logits_flat, actions_flat) * mask) / mx.sum(mask)
        else:
            ce_loss = mx.array(0.0)
        
        # Contrastive loss for variable identity
        contrast_loss = contrastive_loss(outputs['var_features'], outputs['var_mask'])
        
        # Attention entropy regularization (encourage focused attention)
        soft_scores = outputs['soft_scores']
        entropy = -mx.sum(soft_scores * mx.log(soft_scores + 1e-8), axis=-1)
        entropy_loss = mx.mean(entropy)  # We want LOW entropy (focused attention)
        
        # Total loss
        total_loss = ce_loss + 0.1 * contrast_loss + 0.01 * entropy_loss
        
        return total_loss, {
            'ce_loss': ce_loss,
            'contrast_loss': contrast_loss,
            'entropy': mx.mean(entropy)
        }
    
    # MLX doesn't support has_aux, so return metrics separately
    def loss_wrapper(model):
        total_loss, metrics = loss_fn(model)
        return total_loss
    
    loss_and_grad_fn = mx.value_and_grad(loss_wrapper)
    loss, grads = loss_and_grad_fn(model)
    optimizer.update(model, grads)
    
    # Get metrics by running forward pass again
    _, metrics = loss_fn(model)
    
    return loss, metrics


def evaluate_binding(model, num_tests=100):
    """Evaluate variable binding capability"""
    correct = 0
    consistent = 0
    
    var_to_id = {'X': 4, 'Y': 5, 'A': 6, 'B': 7}
    action_to_id = {'JUMP': 1, 'WALK': 2, 'TURN': 3, 'RUN': 4}
    id_to_action = {v: k for k, v in action_to_id.items()}
    word_to_id = {
        'means': 8, 'do': 9, 'then': 10, 'twice': 11,
        'jump': 12, 'walk': 13, 'turn': 14, 'run': 15
    }
    
    for _ in range(num_tests):
        # Test: "X means jump do X" should output JUMP
        var = np.random.choice(['X', 'Y', 'A', 'B'])
        action = np.random.choice(['jump', 'walk', 'turn', 'run'])
        
        cmd = mx.array([[var_to_id[var], word_to_id['means'], word_to_id[action], 
                        word_to_id['do'], var_to_id[var]]])
        expected = action_to_id[action.upper()]
        
        outputs = model(cmd, training=False)
        pred = mx.argmax(outputs['action_logits'][0, 0]).item()
        
        if pred == expected:
            correct += 1
        
        # Check binding consistency
        bindings = outputs['bindings'][0].tolist()
        if bindings[0] == bindings[4]:  # Same variable should bind to same slot
            consistent += 1
    
    return correct / num_tests, consistent / num_tests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_slots', type=int, default=4)
    args = parser.parse_args()
    
    print("=== Contrastive Variable Binding Training ===")
    print(f"Slots: {args.num_slots}, Epochs: {args.epochs}, LR: {args.lr}")
    
    # Model
    model = ContrastiveBindingModel(
        vocab_size=20,
        num_actions=6,  # PAD, JUMP, WALK, TURN, RUN, END
        embed_dim=128,
        hidden_dim=256,
        num_slots=args.num_slots,
        num_heads=8
    )
    
    optimizer = optim.Adam(learning_rate=args.lr)
    
    # Training
    steps_per_epoch = 100
    
    for epoch in range(args.epochs):
        epoch_losses = {'total': 0, 'ce': 0, 'contrast': 0, 'entropy': 0}
        
        # Anneal temperature
        model.binder.temperature = max(0.1, 1.0 * (0.95 ** epoch))
        
        for _ in range(steps_per_epoch):
            commands, actions = generate_binding_batch(args.batch_size, 20)
            loss, metrics = train_step(model, commands, actions, optimizer)
            
            epoch_losses['total'] += loss.item()
            epoch_losses['ce'] += metrics['ce_loss'].item()
            epoch_losses['contrast'] += metrics['contrast_loss'].item()
            epoch_losses['entropy'] += metrics['entropy'].item()
            
            mx.eval(model.parameters())
        
        # Evaluate
        accuracy, consistency = evaluate_binding(model)
        
        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"  Loss: {epoch_losses['total']/steps_per_epoch:.3f} "
              f"(CE: {epoch_losses['ce']/steps_per_epoch:.3f}, "
              f"Contrast: {epoch_losses['contrast']/steps_per_epoch:.3f})")
        print(f"  Entropy: {epoch_losses['entropy']/steps_per_epoch:.3f}, "
              f"Temperature: {model.binder.temperature:.3f}")
        print(f"  Accuracy: {accuracy:.1%}, Consistency: {consistency:.1%}")
    
    # Final test on modifications
    print("\n=== Testing Modifications ===")
    test_modification_capability(model)


def test_modification_capability(model):
    """Test if model can handle variable reassignment"""
    var_to_id = {'X': 4, 'Y': 5}
    action_to_id = {'JUMP': 1, 'WALK': 2}
    id_to_action = {v: k for k, v in action_to_id.items()}
    word_to_id = {'means': 8, 'do': 9, 'jump': 12, 'walk': 13}
    
    # Test 1: X means jump, do X (should output JUMP)
    cmd1 = mx.array([[4, 8, 12, 9, 4]])  # X means jump do X
    out1 = model(cmd1, training=False)
    pred1 = mx.argmax(out1['action_logits'][0, 0]).item()
    
    # Test 2: X means walk, do X (should output WALK)
    cmd2 = mx.array([[4, 8, 13, 9, 4]])  # X means walk do X
    out2 = model(cmd2, training=False)
    pred2 = mx.argmax(out2['action_logits'][0, 0]).item()
    
    print(f"X means jump do X -> {id_to_action.get(pred1, 'UNK')} {'✓' if pred1 == 1 else '✗'}")
    print(f"X means walk do X -> {id_to_action.get(pred2, 'UNK')} {'✓' if pred2 == 2 else '✗'}")
    
    # Check if bindings are consistent
    bind1 = out1['bindings'][0].tolist()
    bind2 = out2['bindings'][0].tolist()
    
    print(f"\nBinding analysis:")
    print(f"  X bindings in test 1: position 0 -> slot {bind1[0]}, position 4 -> slot {bind1[4]}")
    print(f"  X bindings in test 2: position 0 -> slot {bind2[0]}, position 4 -> slot {bind2[4]}")
    print(f"  Consistent: {'✓' if bind1[0] == bind1[4] and bind2[0] == bind2[4] else '✗'}")


if __name__ == "__main__":
    main()