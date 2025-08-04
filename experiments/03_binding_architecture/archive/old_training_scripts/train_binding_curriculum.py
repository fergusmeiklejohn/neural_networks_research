#!/usr/bin/env python3
"""
Curriculum Learning for Variable Binding

This implements a 3-stage curriculum based on our key insight:
1. Stage 1: Variable Recognition - Learn which variable is which
2. Stage 2: Direct Retrieval - Learn to retrieve stored values
3. Stage 3: Full Binding - Combine recognition and retrieval with execution

Key differences from previous approaches:
- Explicit stages with different data patterns
- Stage transitions based on performance thresholds
- Focused learning objectives per stage
"""

import argparse
import time
from typing import Dict, List, Tuple, Optional
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from pathlib import Path

# Actions mapping
ACTIONS = {
    'WALK': 0,
    'JUMP': 1,
    'TURN': 2,
    'LOOK': 3,
    'RUN': 4,
    '<PAD>': 5
}

# Vocabulary - extended for curriculum stages
VOCAB = {
    '<PAD>': 0,
    'X': 1, 'Y': 2, 'Z': 3,
    'means': 4, 'do': 5,
    'walk': 6, 'jump': 7, 'turn': 8, 'look': 9, 'run': 10,
    'twice': 11, 'thrice': 12,
    'what': 13, 'is': 14, 'recall': 15,  # New tokens for curriculum
}

class CurriculumBindingModel(nn.Module):
    """Variable binding model with support for curriculum learning"""
    
    def __init__(self, 
                 vocab_size: int = len(VOCAB),
                 num_actions: int = len(ACTIONS),
                 embed_dim: int = 128,
                 num_slots: int = 4,
                 num_heads: int = 8,
                 initial_temperature: float = 1.0):
        super().__init__()
        
        # Token embeddings
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        
        # Binding attention mechanism
        self.binder = BindingAttention(embed_dim, num_heads, initial_temperature)
        
        # Memory slots (learnable parameters)
        self.slot_keys = mx.random.normal((num_slots, embed_dim))
        self.slot_values = mx.random.normal((num_slots, embed_dim))
        
        # Action decoder - simplified for better learning
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_actions)
        )
        
        # Stage-specific heads for curriculum
        self.recognition_head = nn.Linear(embed_dim, vocab_size)  # For "what is X?" queries
        
    def __call__(self, 
                 command_ids: mx.array,
                 stage: str = "full",
                 training: bool = True) -> Dict[str, mx.array]:
        """
        Forward pass with stage-specific processing
        
        Args:
            command_ids: Input token ids (batch, seq_len)
            stage: One of "recognition", "retrieval", "full"
            training: Whether in training mode
        """
        batch_size, seq_len = command_ids.shape
        
        # Embed tokens
        word_embeds = self.token_embed(command_ids)  # (batch, seq_len, embed_dim)
        
        # Get bindings using soft attention
        bindings, binding_scores = self.binder(
            word_embeds, self.slot_keys, training=training
        )
        
        # Retrieve bound values using soft attention
        bound_values = binding_scores @ self.slot_values  # (batch, seq_len, embed_dim)
        
        # Stage-specific processing
        if stage == "recognition":
            # For "what is X?" - output the variable name
            recognition_logits = self.recognition_head(bound_values)
            return {
                'recognition_logits': recognition_logits,
                'bindings': bindings,
                'binding_scores': binding_scores,
                'temperature': self.binder.temperature
            }
        else:
            # For retrieval and full stages - decode to actions
            action_logits = self.decoder(bound_values)
            return {
                'action_logits': action_logits,
                'bindings': bindings,
                'binding_scores': binding_scores,
                'temperature': self.binder.temperature
            }


class BindingAttention(nn.Module):
    """Attention mechanism for binding variables to slots"""
    
    def __init__(self, embed_dim: int = 128, num_heads: int = 8, initial_temperature: float = 1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.temperature = initial_temperature
        
    def gumbel_softmax(self, logits: mx.array, temperature: float, hard: bool = False) -> mx.array:
        """Gumbel-Softmax for differentiable discrete sampling"""
        gumbel_noise = -mx.log(-mx.log(mx.random.uniform(shape=logits.shape) + 1e-8) + 1e-8)
        y = logits + gumbel_noise
        y = mx.softmax(y / temperature, axis=-1)
        
        if hard:
            # Straight-through estimator
            y_hard = mx.zeros_like(y)
            indices = mx.argmax(y, axis=-1, keepdims=True)
            ones = mx.ones(indices.shape)
            y_hard = mx.put_along_axis(y_hard, indices, ones, axis=-1)
            y = mx.stop_gradient(y_hard - y) + y
            
        return y
    
    def __call__(self, 
                 words: mx.array,
                 slot_keys: mx.array,
                 training: bool = True) -> Tuple[mx.array, mx.array]:
        """
        Compute bindings between words and slots
        
        Returns:
            bindings: Hard assignments (batch, seq_len)
            soft_scores: Soft attention scores (batch, seq_len, num_slots)
        """
        batch_size, seq_len, embed_dim = words.shape
        num_slots = slot_keys.shape[0]
        
        # Project to multi-head format
        Q = self.q_proj(words).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        Q = Q.transpose(0, 2, 1, 3)  # (batch, heads, seq_len, head_dim)
        
        K = self.k_proj(slot_keys).reshape(num_slots, self.num_heads, self.head_dim)
        K = mx.broadcast_to(K[None, :, :, :], (batch_size, num_slots, self.num_heads, self.head_dim))
        K = K.transpose(0, 2, 1, 3)  # (batch, heads, num_slots, head_dim)
        
        # Compute attention scores
        scores = (Q @ K.transpose(0, 1, 3, 2)) * self.scale  # (batch, heads, seq_len, num_slots)
        scores = mx.mean(scores, axis=1)  # (batch, seq_len, num_slots)
        
        # Apply Gumbel-Softmax for differentiable selection
        soft_scores = self.gumbel_softmax(scores, self.temperature, hard=training)
        
        # Get hard bindings for analysis
        bindings = mx.argmax(scores, axis=-1)  # (batch, seq_len)
        
        return bindings, soft_scores


def generate_stage1_data(batch_size: int = 32, max_examples: int = 10000) -> Dict[str, mx.array]:
    """
    Stage 1: Variable Recognition
    Pattern: "X means jump Y means walk what is X" -> X (token id)
    """
    commands = []
    targets = []
    
    variables = ['X', 'Y', 'Z']
    actions = ['walk', 'jump', 'turn', 'look', 'run']
    
    for _ in range(min(batch_size, max_examples)):
        # Sample 2-3 variable definitions
        num_vars = np.random.randint(2, 4)
        used_vars = np.random.choice(variables, size=num_vars, replace=False)
        var_actions = np.random.choice(actions, size=num_vars, replace=False)
        
        # Build command
        cmd = []
        for var, action in zip(used_vars, var_actions):
            cmd.extend([var, 'means', action])
        
        # Add query
        query_var = np.random.choice(used_vars)
        cmd.extend(['what', 'is', query_var])
        
        # Target is the variable token id
        target = VOCAB[query_var]
        
        # Convert to ids
        cmd_ids = [VOCAB.get(token, VOCAB['<PAD>']) for token in cmd]
        commands.append(cmd_ids)
        targets.append(target)
    
    # Pad sequences
    max_len = max(len(cmd) for cmd in commands)
    padded_commands = []
    for cmd in commands:
        padded = cmd + [VOCAB['<PAD>']] * (max_len - len(cmd))
        padded_commands.append(padded)
    
    return {
        'command': mx.array(padded_commands, dtype=mx.int32),
        'target': mx.array(targets, dtype=mx.int32)
    }


def generate_stage2_data(batch_size: int = 32, max_examples: int = 10000) -> Dict[str, mx.array]:
    """
    Stage 2: Direct Retrieval
    Pattern: "X is jump recall X" -> JUMP
    """
    commands = []
    labels = []
    
    variables = ['X', 'Y', 'Z']
    actions = ['walk', 'jump', 'turn', 'look', 'run']
    
    for _ in range(min(batch_size, max_examples)):
        var = np.random.choice(variables)
        action = np.random.choice(actions)
        
        # Build command: "X is jump recall X"
        cmd = [var, 'is', action, 'recall', var]
        label = [ACTIONS[action.upper()]]
        
        # Convert to ids
        cmd_ids = [VOCAB.get(token, VOCAB['<PAD>']) for token in cmd]
        commands.append(cmd_ids)
        labels.append(label)
    
    # Pad sequences
    max_cmd_len = max(len(cmd) for cmd in commands)
    padded_commands = []
    padded_labels = []
    
    for cmd, label in zip(commands, labels):
        padded_cmd = cmd + [VOCAB['<PAD>']] * (max_cmd_len - len(cmd))
        padded_label = label + [ACTIONS['<PAD>']] * (1 - len(label))
        padded_commands.append(padded_cmd)
        padded_labels.append(padded_label)
    
    return {
        'command': mx.array(padded_commands, dtype=mx.int32),
        'labels': mx.array(padded_labels, dtype=mx.int32)
    }


def generate_stage3_data(batch_size: int = 32, max_examples: int = 10000) -> Dict[str, mx.array]:
    """
    Stage 3: Full Binding (original task)
    Pattern: "X means jump do X" -> JUMP
    """
    commands = []
    labels = []
    
    variables = ['X', 'Y', 'Z']
    actions = ['walk', 'jump', 'turn', 'look', 'run']
    modifiers = ['twice', 'thrice']
    
    for _ in range(min(batch_size, max_examples)):
        var = np.random.choice(variables)
        action = np.random.choice(actions)
        
        # Build command
        cmd = [var, 'means', action, 'do', var]
        label = [ACTIONS[action.upper()]]
        
        # Sometimes add modifiers
        if np.random.rand() < 0.3:
            modifier = np.random.choice(modifiers)
            cmd.append(modifier)
            num_reps = 2 if modifier == 'twice' else 3
            label = label * num_reps
        
        # Convert to ids
        cmd_ids = [VOCAB.get(token, VOCAB['<PAD>']) for token in cmd]
        commands.append(cmd_ids)
        labels.append(label)
    
    # Pad sequences
    max_cmd_len = max(len(cmd) for cmd in commands)
    max_label_len = max(len(label) for label in labels)
    
    padded_commands = []
    padded_labels = []
    
    for cmd, label in zip(commands, labels):
        padded_cmd = cmd + [VOCAB['<PAD>']] * (max_cmd_len - len(cmd))
        padded_label = label + [ACTIONS['<PAD>']] * (max_label_len - len(label))
        padded_commands.append(padded_cmd)
        padded_labels.append(padded_label)
    
    return {
        'command': mx.array(padded_commands, dtype=mx.int32),
        'labels': mx.array(padded_labels, dtype=mx.int32)
    }


def train_step(model, batch, stage, optimizer):
    """Single training step for the appropriate stage"""
    
    def loss_fn(model):
        outputs = model(batch['command'], stage=stage, training=True)
        
        if stage == "recognition":
            # Cross-entropy loss for variable recognition
            logits = outputs['recognition_logits']
            # Only compute loss on the query position (last non-pad token)
            mask = batch['command'] != VOCAB['<PAD>']
            last_positions = mx.sum(mask, axis=1) - 1
            
            # Gather logits at query positions
            batch_indices = mx.arange(logits.shape[0])
            query_logits = logits[batch_indices, last_positions]
            
            loss = mx.mean(nn.losses.cross_entropy(query_logits, batch['target']))
        else:
            # Action prediction loss
            logits = outputs['action_logits']
            labels = batch['labels']
            
            # For stage 2, we only care about the last token prediction
            if stage == "retrieval":
                # Find last non-pad position for each sequence
                mask = batch['command'] != VOCAB['<PAD>']
                last_positions = mx.sum(mask, axis=1) - 1
                batch_indices = mx.arange(logits.shape[0])
                
                # Get logits at last positions
                last_logits = logits[batch_indices, last_positions]  # (batch, num_actions)
                # Labels are already single actions for stage 2
                loss = mx.mean(nn.losses.cross_entropy(last_logits, labels.squeeze()))
            else:
                # For stage 3, handle sequence predictions
                # Flatten for loss computation
                logits_flat = logits.reshape(-1, logits.shape[-1])
                labels_flat = labels.flatten()
                
                # Mask padding
                mask = labels_flat != ACTIONS['<PAD>']
                if mx.sum(mask) > 0:
                    # MLX doesn't support boolean indexing yet, so we use multiplication
                    # to zero out padding positions in the loss
                    ce_loss = nn.losses.cross_entropy(logits_flat, labels_flat, reduction='none')
                    masked_loss = ce_loss * mask
                    loss = mx.sum(masked_loss) / mx.sum(mask)
                else:
                    loss = mx.array(0.0)
        
        return loss
    
    loss_and_grad_fn = mx.value_and_grad(loss_fn)
    loss, grads = loss_and_grad_fn(model)
    optimizer.update(model, grads)
    mx.eval(loss)
    
    return loss.item()


def evaluate_stage(model, stage: str, num_eval: int = 100) -> float:
    """Evaluate performance on a specific stage"""
    model.eval()
    correct = 0
    total = 0
    
    # Generate evaluation data
    if stage == "recognition":
        data_fn = generate_stage1_data
    elif stage == "retrieval":
        data_fn = generate_stage2_data
    else:  # full
        data_fn = generate_stage3_data
    
    for _ in range(num_eval // 32):
        batch = data_fn(32)
        outputs = model(batch['command'], stage=stage, training=False)
        
        if stage == "recognition":
            # Check if predicted variable matches target
            logits = outputs['recognition_logits']
            mask = batch['command'] != VOCAB['<PAD>']
            last_positions = mx.sum(mask, axis=1) - 1
            batch_indices = mx.arange(logits.shape[0])
            query_logits = logits[batch_indices, last_positions]
            predictions = mx.argmax(query_logits, axis=-1)
            correct += mx.sum(predictions == batch['target']).item()
            total += batch['target'].shape[0]
        else:
            # Check action predictions
            logits = outputs['action_logits']
            predictions = mx.argmax(logits, axis=-1)
            labels = batch['labels']
            
            # Compare predictions to labels
            mask = labels != ACTIONS['<PAD>']
            matches = (predictions == labels) * mask
            
            # Count sequences where all non-pad predictions are correct
            sequence_correct = mx.all(matches | ~mask, axis=1)
            correct += mx.sum(sequence_correct).item()
            total += labels.shape[0]
    
    return correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description='Curriculum learning for variable binding')
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--num_slots', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--epochs_per_stage', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--stage_threshold', type=float, default=0.8,
                       help='Accuracy threshold to progress to next stage')
    parser.add_argument('--output_dir', type=str, default='outputs/curriculum')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    model = CurriculumBindingModel(
        vocab_size=len(VOCAB),
        num_actions=len(ACTIONS),
        embed_dim=args.embed_dim,
        num_slots=args.num_slots,
        num_heads=args.num_heads,
        initial_temperature=1.0
    )
    mx.eval(model.parameters())
    
    # Initialize optimizer
    optimizer = optim.Adam(learning_rate=args.lr)
    
    # Temperature annealing parameters
    initial_temperature = 1.0
    min_temperature = 0.1
    temperature_decay = 0.95
    
    # Training stages
    stages = ["recognition", "retrieval", "full"]
    
    for stage_idx, stage in enumerate(stages):
        print(f"\n{'='*60}")
        print(f"Starting Stage {stage_idx + 1}: {stage.upper()}")
        print(f"{'='*60}")
        
        # Data generator for this stage
        if stage == "recognition":
            data_fn = generate_stage1_data
        elif stage == "retrieval":
            data_fn = generate_stage2_data
        else:
            data_fn = generate_stage3_data
        
        best_accuracy = 0.0
        epochs_without_improvement = 0
        
        for epoch in range(args.epochs_per_stage):
            # Update temperature
            current_temp = max(min_temperature, 
                             initial_temperature * (temperature_decay ** (stage_idx * args.epochs_per_stage + epoch)))
            model.binder.temperature = current_temp
            
            # Training
            epoch_loss = 0.0
            num_batches = 100  # Fixed number of batches per epoch
            
            start_time = time.time()
            for _ in range(num_batches):
                batch = data_fn(args.batch_size)
                loss = train_step(model, batch, stage, optimizer)
                epoch_loss += loss
            
            avg_loss = epoch_loss / num_batches
            
            # Evaluation
            accuracy = evaluate_stage(model, stage)
            
            print(f"Epoch {epoch+1}/{args.epochs_per_stage} - "
                  f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2%}, "
                  f"Temperature: {current_temp:.3f}, Time: {time.time()-start_time:.2f}s")
            
            # Check for improvement
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                epochs_without_improvement = 0
                
                # Save checkpoint
                checkpoint_path = output_dir / f"{stage}_best_model.npz"
                # Convert parameters to flat arrays for saving
                flat_params = {}
                def flatten_params(params, prefix=''):
                    for k, v in params.items():
                        key = f"{prefix}.{k}" if prefix else k
                        if isinstance(v, dict):
                            flatten_params(v, key)
                        else:
                            flat_params[key] = v
                flatten_params(dict(model.parameters()))
                np.savez(checkpoint_path, **{k: np.array(v) for k, v in flat_params.items()})
                print(f"  Saved best model with accuracy {accuracy:.2%}")
            else:
                epochs_without_improvement += 1
            
            # Check if we should move to next stage
            if accuracy >= args.stage_threshold:
                print(f"\n✓ Stage {stage} completed! Accuracy: {accuracy:.2%}")
                break
            
            # Early stopping within stage
            if epochs_without_improvement >= 5:
                print(f"\n! No improvement for 5 epochs. Moving to next stage.")
                break
        
        print(f"\nStage {stage} final accuracy: {best_accuracy:.2%}")
    
    # Final evaluation on all stages
    print(f"\n{'='*60}")
    print("FINAL EVALUATION")
    print(f"{'='*60}")
    
    for stage in stages:
        accuracy = evaluate_stage(model, stage, num_eval=200)
        print(f"{stage.capitalize()} accuracy: {accuracy:.2%}")
    
    # Test modification capability on full task
    print(f"\n{'='*60}")
    print("MODIFICATION TEST")
    print(f"{'='*60}")
    
    test_modification_capability(model)
    
    # Save final model
    final_path = output_dir / "curriculum_final_model.npz"
    # Convert parameters to flat arrays for saving
    flat_params = {}
    def flatten_params(params, prefix=''):
        for k, v in params.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flatten_params(v, key)
            else:
                flat_params[key] = v
    flatten_params(dict(model.parameters()))
    np.savez(final_path, **{k: np.array(v) for k, v in flat_params.items()})
    print(f"\nFinal model saved to {final_path}")


def test_modification_capability(model):
    """Test if the model can handle variable substitutions"""
    model.eval()
    
    test_cases = [
        {
            'original': "X means jump do X",
            'modified': "X means walk do X",
            'expected_orig': ['JUMP'],
            'expected_mod': ['WALK']
        },
        {
            'original': "Y means turn do Y twice",
            'modified': "Y means run do Y twice", 
            'expected_orig': ['TURN', 'TURN'],
            'expected_mod': ['RUN', 'RUN']
        }
    ]
    
    successes = 0
    
    for i, test in enumerate(test_cases):
        print(f"\nTest {i+1}:")
        print(f"  Original: {test['original']}")
        
        # Test original
        orig_ids = [VOCAB.get(token, VOCAB['<PAD>']) for token in test['original'].split()]
        orig_batch = mx.array([orig_ids], dtype=mx.int32)
        orig_outputs = model(orig_batch, stage="full", training=False)
        orig_preds = mx.argmax(orig_outputs['action_logits'], axis=-1)[0]
        
        # Convert predictions to actions
        orig_actions = []
        for j in range(len(test['expected_orig'])):
            if j < orig_preds.shape[0]:
                action_id = orig_preds[j].item()
                action = [k for k, v in ACTIONS.items() if v == action_id][0]
                orig_actions.append(action)
        
        print(f"  Expected: {test['expected_orig']}")
        print(f"  Predicted: {orig_actions}")
        
        # Test modified
        print(f"  Modified: {test['modified']}")
        mod_ids = [VOCAB.get(token, VOCAB['<PAD>']) for token in test['modified'].split()]
        mod_batch = mx.array([mod_ids], dtype=mx.int32)
        mod_outputs = model(mod_batch, stage="full", training=False)
        mod_preds = mx.argmax(mod_outputs['action_logits'], axis=-1)[0]
        
        # Convert predictions to actions
        mod_actions = []
        for j in range(len(test['expected_mod'])):
            if j < mod_preds.shape[0]:
                action_id = mod_preds[j].item()
                action = [k for k, v in ACTIONS.items() if v == action_id][0]
                mod_actions.append(action)
        
        print(f"  Expected: {test['expected_mod']}")
        print(f"  Predicted: {mod_actions}")
        
        # Check success
        if orig_actions == test['expected_orig'] and mod_actions == test['expected_mod']:
            print("  Success: ✓")
            successes += 1
        else:
            print("  Success: ✗")
    
    print(f"\nModification Success Rate: {successes}/{len(test_cases)} = {successes/len(test_cases)*100:.1f}%")


if __name__ == "__main__":
    main()