#!/usr/bin/env python3
"""
Curriculum Learning with Temporal Action Buffer

Extends dynamic memory with temporal processing capabilities
to handle patterns like "do X twice" correctly.
"""

import argparse
import time
from typing import Dict, List, Tuple, Optional
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from pathlib import Path

# Import data generation and other utilities
import sys
sys.path.append('.')
from train_binding_curriculum import (
    generate_stage1_data, generate_stage2_data, generate_stage3_data,
    VOCAB, ACTIONS
)


class TemporalActionBuffer:
    """Tracks recent action predictions for temporal processing"""
    
    def __init__(self, max_history: int = 5):
        self.history = []  # List of (action_embedding, variable_id) pairs
        self.max_history = max_history
        
    def push(self, action_embedding: mx.array, variable_id: int):
        """Add a new action to history"""
        self.history.append((action_embedding, variable_id))
        if len(self.history) > self.max_history:
            self.history.pop(0)
            
    def get_last_for_variable(self, variable_id: int) -> Optional[mx.array]:
        """Get the most recent action for a specific variable"""
        for action_emb, var_id in reversed(self.history):
            if var_id == variable_id:
                return action_emb
        return None
        
    def clear(self):
        """Clear history for new sequence"""
        self.history = []


class TemporalDynamicMemoryModel(nn.Module):
    """
    Enhanced dynamic memory model with temporal action processing
    """
    
    def __init__(self,
                 vocab_size: int,
                 num_actions: int,
                 embed_dim: int = 64,
                 num_slots: int = 4,
                 num_heads: int = 8,
                 initial_temperature: float = 1.0):
        super().__init__()
        
        # Core components from DynamicMemoryModel
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.slot_keys = mx.random.normal((num_slots, embed_dim))
        self.binder = BindingAttention(embed_dim, num_heads, initial_temperature)
        
        # Value processing
        self.value_extractor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Temporal processing components
        self.temporal_processor = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),  # Combine current token + context
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Action decoder
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_actions)
        )
        
    def detect_storage_patterns(self, 
                              word_embeds: mx.array,
                              command_ids: mx.array) -> Tuple[mx.array, mx.array]:
        """Detect patterns like 'X is jump' or 'X means jump'"""
        batch_size, seq_len = command_ids.shape
        
        is_tokens = (command_ids == VOCAB['is']) | (command_ids == VOCAB['means'])
        
        storage_mask = mx.zeros((batch_size, seq_len))
        value_positions = mx.zeros((batch_size, seq_len), dtype=mx.int32)
        
        for i in range(seq_len - 2):
            var_is_variable = mx.logical_or(
                command_ids[:, i] == VOCAB['X'],
                mx.logical_or(
                    command_ids[:, i] == VOCAB['Y'],
                    command_ids[:, i] == VOCAB['Z']
                )
            )
            next_is_storage = is_tokens[:, i + 1]
            
            should_store = var_is_variable & next_is_storage
            storage_mask = mx.where(
                should_store[:, None] & (mx.arange(seq_len) == i),
                1.0,
                storage_mask
            )
            
            value_positions = mx.where(
                should_store[:, None] & (mx.arange(seq_len) == i),
                i + 2,
                value_positions
            )
        
        return storage_mask, value_positions
    
    def detect_temporal_patterns(self, 
                               command_ids: mx.array,
                               position: int) -> Tuple[bool, int, Optional[int]]:
        """
        Detect temporal modifiers like 'twice' or 'thrice'
        Returns: (is_temporal, repeat_count, variable_token_id)
        """
        batch_size = command_ids.shape[0]
        
        # Check if current position has a temporal modifier
        is_twice = command_ids[:, position] == VOCAB.get('twice', -1)
        is_thrice = command_ids[:, position] == VOCAB.get('thrice', -1)
        
        if not mx.any(is_twice | is_thrice):
            return False, 0, None
            
        # Look back for the variable (usually 1-2 positions back)
        for offset in [1, 2]:
            if position - offset >= 0:
                prev_token = command_ids[:, position - offset]
                is_variable = mx.logical_or(
                    prev_token == VOCAB['X'],
                    mx.logical_or(
                        prev_token == VOCAB['Y'],
                        prev_token == VOCAB['Z']
                    )
                )
                if mx.any(is_variable):
                    # Get repeat count
                    repeat_count = mx.where(is_twice, 2, 3)
                    # Return the variable token for the first batch item
                    # (simplified for now - could be extended for batched processing)
                    var_id = prev_token[0].item()
                    count = repeat_count[0].item()
                    return True, count, var_id
        
        return False, 0, None
    
    def __call__(self, 
                 command_ids: mx.array,
                 stage: str = "full",
                 training: bool = True) -> Dict[str, mx.array]:
        """Forward pass with temporal action processing"""
        batch_size, seq_len = command_ids.shape
        
        # Embed tokens
        word_embeds = self.token_embed(command_ids)
        
        # Detect storage patterns
        storage_mask, value_positions = self.detect_storage_patterns(word_embeds, command_ids)
        
        # Initialize dynamic slot values and temporal buffer
        slot_values = mx.zeros((batch_size, self.slot_keys.shape[0], self.slot_keys.shape[1]))
        
        # Process sequence with dynamic updates
        outputs = []
        temporal_actions = []  # Store expanded actions for temporal patterns
        action_buffer = TemporalActionBuffer()
        
        for t in range(seq_len):
            current_embed = word_embeds[:, t:t+1, :]
            
            # Get bindings
            bindings, binding_scores = self.binder(
                current_embed, self.slot_keys, training=training
            )
            
            # Check if this position stores a value
            stores_value = storage_mask[:, t:t+1]
            
            if mx.sum(stores_value) > 0:
                # Update slot values (same as before)
                batch_indices = mx.arange(batch_size)
                value_pos = value_positions[:, t]
                value_embeds = word_embeds[batch_indices, value_pos]
                processed_values = self.value_extractor(value_embeds)
                update_weights = binding_scores.squeeze(1) * stores_value
                updates = update_weights[:, :, None] * processed_values[:, None, :]
                slot_values = slot_values + updates
            
            # Check for temporal patterns
            is_temporal, repeat_count, var_id = self.detect_temporal_patterns(command_ids, t)
            
            if is_temporal and var_id is not None:
                # Handle temporal modifier
                # Get the binding for this variable
                # Create variable embeddings for the whole batch
                var_ids_batch = mx.full((batch_size,), var_id, dtype=mx.int32)
                var_embeds = self.token_embed(var_ids_batch)  # (batch, embed_dim)
                var_embeds = var_embeds[:, None, :]  # (batch, 1, embed_dim)
                
                _, var_binding_scores = self.binder(var_embeds, self.slot_keys, training=False)
                
                # Retrieve value using binding  
                retrieved = (var_binding_scores @ slot_values).squeeze(1)  # (batch, embed_dim)
                
                # Generate repeated actions
                for _ in range(repeat_count):
                    temporal_actions.append(retrieved)
            else:
                # Normal retrieval
                retrieved = (binding_scores @ slot_values).squeeze(1)
                
                # Check if this is an action position
                is_do_position = command_ids[:, t] == VOCAB['do']
                if t > 0:
                    prev_was_do = command_ids[:, t-1] == VOCAB['do']
                    is_var = mx.logical_or(
                        command_ids[:, t] == VOCAB['X'],
                        mx.logical_or(
                            command_ids[:, t] == VOCAB['Y'],
                            command_ids[:, t] == VOCAB['Z']
                        )
                    )
                    
                    if mx.any(prev_was_do & is_var):
                        # Store in action buffer for potential temporal use
                        action_buffer.push(retrieved, command_ids[0, t].item())
            
            # Prepare output - ensure consistent shapes
            if stage == "recognition":
                output = current_embed.squeeze(1)  # (batch, embed_dim)
            else:
                output = retrieved  # (batch, embed_dim)
                # Ensure output has proper shape
                if len(output.shape) == 1:
                    output = output[None, :]  # Add batch dim if missing
                if len(output.shape) == 3:
                    output = output.squeeze(1)  # Remove extra dim if present
            
            outputs.append(output)
        
        # Stack outputs
        try:
            outputs_stacked = mx.stack(outputs, axis=1)
        except ValueError as e:
            # Debug shape issue
            print(f"Stack error: {e}")
            print(f"Output shapes: {[o.shape for o in outputs]}")
            raise
        
        # Add temporal actions if any
        if temporal_actions:
            # temporal_actions is a list of (batch, embed_dim) arrays
            temporal_stacked = mx.stack(temporal_actions, axis=1)  # (batch, num_temporal, embed_dim)
            outputs_stacked = mx.concatenate([outputs_stacked, temporal_stacked], axis=1)
        
        # Final decoding
        if stage == "recognition":
            recognition_logits = self.decoder(outputs_stacked)
            return {
                'recognition_logits': recognition_logits,
                'bindings': bindings,
                'slot_values': slot_values,
                'storage_mask': storage_mask,
                'temporal_actions': len(temporal_actions)
            }
        else:
            action_logits = self.decoder(outputs_stacked)
            return {
                'action_logits': action_logits,
                'bindings': bindings,
                'slot_values': slot_values,
                'storage_mask': storage_mask,
                'temporal_actions': len(temporal_actions)
            }


class BindingAttention(nn.Module):
    """Attention mechanism for variable-slot binding with Gumbel-Softmax"""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, initial_temperature: float = 1.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.temperature = initial_temperature
        
        # Multi-head projections
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        
    def gumbel_softmax(self, logits: mx.array, temperature: float = 1.0, hard: bool = True) -> mx.array:
        """Gumbel-Softmax for differentiable discrete sampling"""
        # Sample from Gumbel(0, 1)
        gumbels = -mx.log(-mx.log(mx.random.uniform(shape=logits.shape) + 1e-10) + 1e-10)
        
        # Add Gumbel noise to logits and apply temperature
        y = nn.softmax((logits + gumbels) / temperature, axis=-1)
        
        if hard:
            # Straight-through estimator
            indices = mx.argmax(y, axis=-1, keepdims=True)
            y_hard = mx.zeros_like(y)
            y_hard = mx.put_along_axis(y_hard, indices, mx.ones(indices.shape), axis=-1)
            y = mx.stop_gradient(y_hard - y) + y
            
        return y
    
    def __call__(self, 
                 query: mx.array,  # (batch, 1, embed_dim)
                 keys: mx.array,   # (num_slots, embed_dim)
                 training: bool = True) -> Tuple[mx.array, mx.array]:
        """
        Returns:
            bindings: (batch, 1) - hard slot assignment
            scores: (batch, 1, num_slots) - soft attention scores
        """
        batch_size = query.shape[0]
        num_slots = keys.shape[0]
        
        # Project query
        Q = self.query_proj(query)  # (batch, 1, embed_dim)
        Q = Q.reshape(batch_size, 1, self.num_heads, self.head_dim)
        Q = mx.transpose(Q, (0, 2, 1, 3))  # (batch, num_heads, 1, head_dim)
        
        # Project keys
        K = self.key_proj(keys)  # (num_slots, embed_dim)
        K = K.reshape(num_slots, self.num_heads, self.head_dim)
        K = mx.transpose(K, (1, 0, 2))  # (num_heads, num_slots, head_dim)
        K = mx.broadcast_to(K[None, :, :, :], (batch_size, self.num_heads, num_slots, self.head_dim))
        
        # Compute attention scores using matmul
        # Q: (batch, num_heads, 1, head_dim)
        # K: (batch, num_heads, num_slots, head_dim)
        # scores = Q @ K^T
        scores = (Q @ mx.transpose(K, (0, 1, 3, 2))).squeeze(2)  # (batch, num_heads, num_slots)
        scores = scores / mx.sqrt(mx.array(self.head_dim))
        scores = mx.mean(scores, axis=1, keepdims=True)  # (batch, 1, num_slots)
        
        # Apply Gumbel-Softmax for discrete slot selection
        slot_probs = self.gumbel_softmax(scores, temperature=self.temperature, hard=training)
        
        # Get hard bindings
        bindings = mx.argmax(slot_probs, axis=-1)  # (batch, 1)
        
        return bindings, slot_probs


def train_step(model, batch, stage, optimizer):
    """Single training step for the appropriate stage"""
    
    def loss_fn(model):
        outputs = model(batch['command'], stage=stage, training=True)
        
        if stage == "recognition":
            # Cross-entropy loss for variable recognition
            logits = outputs['recognition_logits']
            mask = batch['command'] != VOCAB['<PAD>']
            last_positions = mx.sum(mask, axis=1) - 1
            
            batch_indices = mx.arange(logits.shape[0])
            query_logits = logits[batch_indices, last_positions]
            
            loss = mx.mean(nn.losses.cross_entropy(query_logits, batch['target']))
        else:
            # Action prediction loss
            logits = outputs['action_logits']
            labels = batch['labels']
            
            if stage == "retrieval":
                # For stage 2, focus on last prediction
                mask = batch['command'] != VOCAB['<PAD>']
                last_positions = mx.sum(mask, axis=1) - 1
                
                batch_indices = mx.arange(logits.shape[0])
                last_logits = logits[batch_indices, last_positions]
                
                loss = mx.mean(nn.losses.cross_entropy(last_logits, labels.squeeze()))
            else:
                # Stage 3: Handle variable-length outputs with temporal actions
                total_loss = mx.array(0.0)
                valid_count = 0
                
                for i in range(batch['command'].shape[0]):
                    cmd = batch['command'][i]
                    valid_labels = []
                    
                    # Find 'do' position
                    do_pos = -1
                    for j in range(len(cmd)):
                        if cmd[j].item() == VOCAB['do']:
                            do_pos = j
                            break
                    
                    if do_pos == -1:
                        continue
                    
                    # Extract valid labels
                    label_idx = 0
                    for j in range(len(labels[i])):
                        if labels[i, j].item() != VOCAB['<PAD>']:
                            valid_labels.append(labels[i, j])
                        else:
                            break
                    
                    if not valid_labels:
                        continue
                    
                    # Check for temporal modifiers
                    has_temporal = False
                    temporal_count = 0
                    
                    for j in range(do_pos + 1, len(cmd)):
                        token = cmd[j].item()
                        if token == VOCAB.get('twice', -1):
                            has_temporal = True
                            temporal_count = 2
                            break
                        elif token == VOCAB.get('thrice', -1):
                            has_temporal = True
                            temporal_count = 3
                            break
                    
                    # Calculate positions for predictions
                    if has_temporal:
                        # For temporal patterns, predictions come after the sequence
                        # and in the temporal_actions part
                        num_temporal = outputs['temporal_actions']
                        if num_temporal > 0:
                            start_pos = logits.shape[1] - num_temporal
                            for k, label in enumerate(valid_labels[:temporal_count]):
                                if start_pos + k < logits.shape[1]:
                                    pred_logits = logits[i, start_pos + k]
                                    loss_j = nn.losses.cross_entropy(pred_logits, label)
                                    total_loss = total_loss + loss_j.squeeze()
                                    valid_count += 1
                    else:
                        # Normal pattern - predictions at variable positions
                        pred_idx = 0
                        for j in range(do_pos + 1, len(cmd)):
                            if cmd[j].item() in [VOCAB['X'], VOCAB['Y'], VOCAB['Z']]:
                                if pred_idx < len(valid_labels) and j < logits.shape[1]:
                                    pred_logits = logits[i, j]
                                    loss_j = nn.losses.cross_entropy(pred_logits, valid_labels[pred_idx])
                                    total_loss = total_loss + loss_j.squeeze()
                                    valid_count += 1
                                    pred_idx += 1
                
                loss = total_loss / max(valid_count, 1)
        
        return loss
    
    # Compute loss and gradients
    loss_and_grad_fn = mx.value_and_grad(loss_fn)
    loss, grads = loss_and_grad_fn(model)
    
    # Update model
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    
    return loss.item()


def evaluate_stage(model, stage: str, num_samples: int = 100) -> float:
    """Evaluate model on specific stage"""
    model.eval()
    
    # Generate evaluation data
    if stage == "recognition":
        data_fn = generate_stage1_data
    elif stage == "retrieval":
        data_fn = generate_stage2_data
    else:
        data_fn = generate_stage3_data
    
    batch = data_fn(num_samples)
    outputs = model(batch['command'], stage=stage, training=False)
    
    if stage == "recognition":
        # Check if model correctly identifies variables
        logits = outputs['recognition_logits']
        mask = batch['command'] != VOCAB['<PAD>']
        last_positions = mx.sum(mask, axis=1) - 1
        
        batch_indices = mx.arange(logits.shape[0])
        query_logits = logits[batch_indices, last_positions]
        predictions = mx.argmax(query_logits, axis=1)
        
        accuracy = mx.mean(predictions == batch['target']).item()
    else:
        # Check action predictions
        logits = outputs['action_logits']
        labels = batch['labels']
        
        if stage == "retrieval":
            # Simple accuracy for stage 2
            mask = batch['command'] != VOCAB['<PAD>']
            last_positions = mx.sum(mask, axis=1) - 1
            
            batch_indices = mx.arange(logits.shape[0])
            last_logits = logits[batch_indices, last_positions]
            predictions = mx.argmax(last_logits, axis=1)
            
            accuracy = mx.mean(predictions == labels.squeeze()).item()
        else:
            # Stage 3: Complex evaluation with temporal patterns
            correct = 0
            total = 0
            
            for i in range(batch['command'].shape[0]):
                cmd = batch['command'][i]
                valid_labels = []
                
                # Extract valid labels
                for j in range(len(labels[i])):
                    if labels[i, j].item() != VOCAB['<PAD>']:
                        valid_labels.append(labels[i, j].item())
                    else:
                        break
                
                if not valid_labels:
                    continue
                
                # Check for temporal patterns
                has_temporal = False
                temporal_count = 0
                
                for j in range(len(cmd)):
                    token = cmd[j].item()
                    if token == VOCAB.get('twice', -1):
                        has_temporal = True
                        temporal_count = 2
                        break
                    elif token == VOCAB.get('thrice', -1):
                        has_temporal = True
                        temporal_count = 3
                        break
                
                # Get predictions
                if has_temporal and outputs['temporal_actions'] > 0:
                    # Check temporal predictions
                    num_temporal = outputs['temporal_actions']
                    start_pos = logits.shape[1] - num_temporal
                    
                    all_correct = True
                    for k in range(min(temporal_count, len(valid_labels))):
                        if start_pos + k < logits.shape[1]:
                            pred = mx.argmax(logits[i, start_pos + k]).item()
                            if pred != valid_labels[k]:
                                all_correct = False
                                break
                    
                    if all_correct:
                        correct += 1
                else:
                    # Normal evaluation
                    do_pos = -1
                    for j in range(len(cmd)):
                        if cmd[j].item() == VOCAB['do']:
                            do_pos = j
                            break
                    
                    if do_pos >= 0:
                        pred_idx = 0
                        all_correct = True
                        
                        for j in range(do_pos + 1, len(cmd)):
                            if cmd[j].item() in [VOCAB['X'], VOCAB['Y'], VOCAB['Z']]:
                                if pred_idx < len(valid_labels) and j < logits.shape[1]:
                                    pred = mx.argmax(logits[i, j]).item()
                                    if pred != valid_labels[pred_idx]:
                                        all_correct = False
                                        break
                                    pred_idx += 1
                        
                        if all_correct and pred_idx == len(valid_labels):
                            correct += 1
                
                total += 1
            
            accuracy = correct / total if total > 0 else 0.0
    
    return accuracy


def test_modification_capability(model):
    """Test if the model can handle variable substitutions including temporal patterns"""
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
        },
        {
            'original': "Z means look do Z",
            'modified': "Z means jump do Z",
            'expected_orig': ['LOOK'],
            'expected_mod': ['JUMP']
        }
    ]
    
    successes = 0
    
    for test in test_cases:
        # Test original
        orig_tokens = test['original'].split()
        orig_ids = [VOCAB.get(t, VOCAB['<PAD>']) for t in orig_tokens]
        orig_cmd = mx.array([orig_ids], dtype=mx.int32)
        
        outputs = model(orig_cmd, stage="full", training=False)
        logits = outputs['action_logits']
        
        # Get predictions
        predictions = []
        
        # Check if temporal actions were generated
        if 'twice' in test['original'] and outputs['temporal_actions'] > 0:
            # Get temporal predictions
            num_temporal = outputs['temporal_actions']
            start_pos = logits.shape[1] - num_temporal
            
            for i in range(num_temporal):
                if start_pos + i < logits.shape[1]:
                    pred_id = mx.argmax(logits[0, start_pos + i]).item()
                    pred_action = [k for k, v in ACTIONS.items() if v == pred_id][0]
                    predictions.append(pred_action)
        else:
            # Normal predictions
            for i, token in enumerate(orig_tokens):
                if i > 0 and orig_tokens[i-1] == 'do' and token in ['X', 'Y', 'Z']:
                    if i < logits.shape[1]:
                        pred_id = mx.argmax(logits[0, i]).item()
                        pred_action = [k for k, v in ACTIONS.items() if v == pred_id][0]
                        predictions.append(pred_action)
        
        orig_correct = predictions == test['expected_orig']
        
        # Test modified
        mod_tokens = test['modified'].split()
        mod_ids = [VOCAB.get(t, VOCAB['<PAD>']) for t in mod_tokens]
        mod_cmd = mx.array([mod_ids], dtype=mx.int32)
        
        outputs = model(mod_cmd, stage="full", training=False)
        logits = outputs['action_logits']
        
        predictions = []
        
        # Check temporal actions again
        if 'twice' in test['modified'] and outputs['temporal_actions'] > 0:
            num_temporal = outputs['temporal_actions']
            start_pos = logits.shape[1] - num_temporal
            
            for i in range(num_temporal):
                if start_pos + i < logits.shape[1]:
                    pred_id = mx.argmax(logits[0, start_pos + i]).item()
                    pred_action = [k for k, v in ACTIONS.items() if v == pred_id][0]
                    predictions.append(pred_action)
        else:
            for i, token in enumerate(mod_tokens):
                if i > 0 and mod_tokens[i-1] == 'do' and token in ['X', 'Y', 'Z']:
                    if i < logits.shape[1]:
                        pred_id = mx.argmax(logits[0, i]).item()
                        pred_action = [k for k, v in ACTIONS.items() if v == pred_id][0]
                        predictions.append(pred_action)
        
        mod_correct = predictions == test['expected_mod']
        
        if orig_correct and mod_correct:
            successes += 1
            print(f"✓ Test {successes}: {test['original']} → Success")
        else:
            print(f"✗ Test failed: {test['original']}")
            print(f"  Original: expected {test['expected_orig']}, got {predictions}")
    
    return successes / len(test_cases)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--num_slots', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs_per_stage', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--stage_threshold', type=float, default=0.8,
                       help='Accuracy threshold to progress to next stage')
    parser.add_argument('--output_dir', type=str, default='outputs/temporal_curriculum')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize model with temporal capabilities
    model = TemporalDynamicMemoryModel(
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
        
        for epoch in range(args.epochs_per_stage):
            # Training
            model.train()
            total_loss = 0
            num_batches = 10  # Fixed for consistency
            
            for _ in range(num_batches):
                batch = data_fn(args.batch_size)
                loss = train_step(model, batch, stage, optimizer)
                total_loss += loss
            
            avg_loss = total_loss / num_batches
            
            # Evaluation
            accuracy = evaluate_stage(model, stage)
            
            # Update temperature
            if hasattr(model.binder, 'temperature'):
                model.binder.temperature = max(
                    min_temperature,
                    initial_temperature * (temperature_decay ** (epoch + 1))
                )
            
            print(f"Epoch {epoch+1}/{args.epochs_per_stage}: "
                  f"Loss={avg_loss:.4f}, Accuracy={accuracy:.2%}, "
                  f"Temp={getattr(model.binder, 'temperature', 'N/A'):.3f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                # Save checkpoint
                checkpoint_path = output_dir / f"stage_{stage}_best.npz"
                flat_params = {}
                for name, param in model.parameters().items():
                    flat_params[name] = np.array(param)
                np.savez(checkpoint_path, **flat_params)
            
            # Check if we should progress to next stage
            if accuracy >= args.stage_threshold:
                print(f"✓ Stage {stage} complete! Achieved {accuracy:.2%} accuracy")
                break
        else:
            print(f"⚠ Stage {stage} completed {args.epochs_per_stage} epochs "
                  f"with best accuracy {best_accuracy:.2%}")
    
    # Test modification capability with temporal patterns
    print("\n" + "="*60)
    print("Testing Modification Capability (including temporal patterns)")
    print("="*60)
    
    mod_success_rate = test_modification_capability(model)
    print(f"\nModification Success Rate: {mod_success_rate:.1%}")
    
    # Save final model
    final_path = output_dir / "temporal_model_final.npz"
    flat_params = {}
    for name, param in model.parameters().items():
        flat_params[name] = np.array(param)
    np.savez(final_path, **flat_params)
    
    print(f"\n✓ Training complete! Final model saved to {final_path}")
    print(f"  Stage accuracies: {best_accuracy:.2%}")
    print(f"  Modification success: {mod_success_rate:.1%}")


if __name__ == "__main__":
    main()