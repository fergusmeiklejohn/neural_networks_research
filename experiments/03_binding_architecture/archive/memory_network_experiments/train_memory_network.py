#!/usr/bin/env python3
"""Train and test the Neural Memory Binding model on progressive complexity data."""

from utils.imports import setup_project_paths
setup_project_paths()

from utils.config import setup_environment
config = setup_environment()

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from typing import Dict, List, Tuple
import numpy as np
import os
from utils.paths import get_output_path

from neural_memory_binding_model import MemoryBasedBindingModel
from progressive_complexity_dataset import (
    ProgressiveComplexityDataset, VOCAB, ACTIONS, 
    tokenize, actions_to_indices
)


class ImprovedMemoryBindingModel(MemoryBasedBindingModel):
    """Memory model with improved pattern detection for our vocabulary."""
    
    def __init__(self, vocab_size: int, num_actions: int, **kwargs):
        super().__init__(vocab_size, num_actions, **kwargs)
        
        # Store vocabulary for pattern detection
        self.vocab = VOCAB
        self.inv_vocab = {v: k for k, v in VOCAB.items()}
        
        # Variable token IDs
        self.var_tokens = {
            'X': VOCAB['X'],
            'Y': VOCAB['Y'], 
            'Z': VOCAB['Z'],
            'W': VOCAB['W']
        }
        
        # Binding words
        self.means_token = VOCAB['means']
        self.is_token = VOCAB['is']
        self.do_token = VOCAB['do']
        
    def _token_to_var_idx(self, token: int) -> int:
        """Map token to variable index."""
        var_map = {
            self.var_tokens['X']: 0,
            self.var_tokens['Y']: 1,
            self.var_tokens['Z']: 2,
            self.var_tokens['W']: 3
        }
        return var_map.get(token, None)
    
    def detect_binding_pattern(self, tokens: mx.array, hidden_states: mx.array, 
                             position: int) -> Tuple[bool, int, int]:
        """Improved binding detection using explicit pattern matching."""
        if position + 2 >= tokens.shape[1]:
            return False, None, None
            
        # Check for "VAR means/is ACTION" pattern
        current_token = tokens[0, position].item()
        
        # Is this a variable token?
        if current_token not in [self.var_tokens['X'], self.var_tokens['Y'], 
                                 self.var_tokens['Z'], self.var_tokens['W']]:
            return False, None, None
            
        # Check next token is "means" or "is"
        next_token = tokens[0, position + 1].item()
        if next_token not in [self.means_token, self.is_token]:
            return False, None, None
            
        # Get variable index
        var_idx = self._token_to_var_idx(current_token)
        
        # Action is at position + 2
        return True, var_idx, position + 2
        
    def detect_execution_pattern(self, tokens: mx.array, hidden_states: mx.array,
                               position: int) -> Tuple[bool, int]:
        """Improved execution detection."""
        if position == 0:
            return False, None
            
        current_token = tokens[0, position].item()
        
        # Is this a variable token?
        if current_token not in [self.var_tokens['X'], self.var_tokens['Y'],
                                 self.var_tokens['Z'], self.var_tokens['W']]:
            return False, None
            
        # Check if previous token is "do"
        prev_token = tokens[0, position - 1].item()
        if prev_token != self.do_token:
            return False, None
            
        # Get variable index
        var_idx = self._token_to_var_idx(current_token)
        return True, var_idx


def collate_batch(samples: List[Dict]) -> Dict[str, mx.array]:
    """Collate samples into a batch."""
    # Find max length
    max_len = max(len(s['tokens']) for s in samples)
    
    # Pad sequences
    padded_tokens = []
    action_sequences = []
    
    for sample in samples:
        tokens = sample['tokens']
        # Pad tokens
        padded = tokens + [VOCAB['PAD']] * (max_len - len(tokens))
        padded_tokens.append(padded)
        
        # Action sequence
        action_sequences.append(sample['expected_indices'])
    
    return {
        'command': mx.array(padded_tokens),
        'expected_actions': action_sequences  # Keep as list for now
    }


def evaluate_model(model: ImprovedMemoryBindingModel, dataset: List[Dict], 
                  verbose: bool = False) -> Dict[str, float]:
    """Evaluate model on dataset."""
    correct_by_level = {1: 0, 2: 0, 3: 0, 4: 0}
    total_by_level = {1: 0, 2: 0, 3: 0, 4: 0}
    correct_by_pattern = {}
    total_by_pattern = {}
    
    model.eval()
    
    for sample in dataset:
        level = sample['complexity_level']
        pattern = sample['pattern_type']
        
        # Process single sample
        batch = collate_batch([sample])
        predictions = model(batch)
        
        # Get predicted actions
        if predictions.shape[0] > 0:
            pred_actions = mx.argmax(predictions, axis=1).tolist()
        else:
            pred_actions = []
            
        expected = sample['expected_indices']
        
        # Check if correct
        correct = pred_actions == expected
        
        if verbose and not correct:
            print(f"\nMismatch on level {level} ({pattern}):")
            print(f"  Command: {sample['command']}")
            print(f"  Expected: {sample['expected_actions']}")
            print(f"  Predicted: {[ACTIONS[i] for i in pred_actions] if pred_actions else []}")
        
        # Update statistics
        if correct:
            correct_by_level[level] += 1
            correct_by_pattern[pattern] = correct_by_pattern.get(pattern, 0) + 1
            
        total_by_level[level] += 1
        total_by_pattern[pattern] = total_by_pattern.get(pattern, 0) + 1
    
    # Calculate accuracies
    results = {}
    
    # By level
    for level in range(1, 5):
        if total_by_level[level] > 0:
            acc = correct_by_level[level] / total_by_level[level]
            results[f'level_{level}_accuracy'] = acc
            
    # By pattern
    for pattern in total_by_pattern:
        if total_by_pattern[pattern] > 0:
            acc = correct_by_pattern.get(pattern, 0) / total_by_pattern[pattern]
            results[f'{pattern}_accuracy'] = acc
            
    # Overall
    total_correct = sum(correct_by_level.values())
    total_samples = sum(total_by_level.values())
    results['overall_accuracy'] = total_correct / total_samples if total_samples > 0 else 0
    
    return results


def train_memory_network():
    """Train the memory network on progressive complexity data."""
    print("Training Neural Memory Binding Model...")
    
    # Create model
    model = ImprovedMemoryBindingModel(
        vocab_size=len(VOCAB),
        num_actions=len(ACTIONS),
        embed_dim=128,
        hidden_dim=256,
        num_vars=4
    )
    
    # Create dataset
    dataset_gen = ProgressiveComplexityDataset()
    
    # Start with level 1 and 2 data
    print("\n=== Generating Training Data ===")
    train_data = []
    train_data.extend(dataset_gen.generate_level_1(200))
    train_data.extend(dataset_gen.generate_level_2(200))
    
    print(f"Training samples: {len(train_data)}")
    
    # Generate test data
    test_data = []
    test_data.extend(dataset_gen.generate_level_1(50))
    test_data.extend(dataset_gen.generate_level_2(50))
    test_data.extend(dataset_gen.generate_level_3(50))  # Test generalization
    test_data.extend(dataset_gen.generate_level_4(50))  # Test generalization
    
    print(f"Test samples: {len(test_data)}")
    
    # Training setup
    optimizer = optim.Adam(learning_rate=1e-3)
    
    def loss_fn(model, batch):
        predictions = model(batch)
        expected = batch['expected_actions']
        
        # Compute cross-entropy loss
        total_loss = 0
        num_predictions = 0
        
        for i, exp_seq in enumerate(expected):
            if len(exp_seq) == 0:
                continue
                
            # Match prediction length to expected length
            pred_seq = predictions[:len(exp_seq)]
            exp_array = mx.array(exp_seq)
            
            # Cross entropy loss
            loss = mx.mean(nn.losses.cross_entropy(pred_seq, exp_array))
            total_loss = total_loss + loss
            num_predictions += 1
            
        return total_loss / max(num_predictions, 1)
    
    # Training loop
    print("\n=== Training ===")
    batch_size = 32
    num_epochs = 50
    best_accuracy = 0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        # Shuffle data
        np.random.shuffle(train_data)
        
        # Process in batches
        for i in range(0, len(train_data), batch_size):
            batch_samples = train_data[i:i+batch_size]
            batch = collate_batch(batch_samples)
            
            # Forward and backward
            loss_and_grad = nn.value_and_grad(model, loss_fn)
            loss, grads = loss_and_grad(model, batch)
            
            # Update
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            
            epoch_loss += loss.item()
            num_batches += 1
            
        avg_loss = epoch_loss / num_batches
        
        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            results = evaluate_model(model, test_data[:100])  # Quick eval on subset
            
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  Level 1 Acc: {results.get('level_1_accuracy', 0):.2%}")
            print(f"  Level 2 Acc: {results.get('level_2_accuracy', 0):.2%}")
            print(f"  Overall Acc: {results['overall_accuracy']:.2%}")
            
            if results['overall_accuracy'] > best_accuracy:
                best_accuracy = results['overall_accuracy']
    
    # Final evaluation
    print("\n=== Final Evaluation ===")
    model.eval()
    final_results = evaluate_model(model, test_data, verbose=True)
    
    print("\nFinal Results by Level:")
    for level in range(1, 5):
        acc = final_results.get(f'level_{level}_accuracy', 0)
        print(f"  Level {level}: {acc:.2%}")
        
    print(f"\nOverall Accuracy: {final_results['overall_accuracy']:.2%}")
    
    # Pattern analysis
    print("\nAccuracy by Pattern Type:")
    pattern_accs = [(k.replace('_accuracy', ''), v) 
                    for k, v in final_results.items() 
                    if k.endswith('_accuracy') and not k.startswith('level')]
    pattern_accs.sort(key=lambda x: x[1], reverse=True)
    
    for pattern, acc in pattern_accs[:10]:  # Top 10 patterns
        if pattern != 'overall':
            print(f"  {pattern}: {acc:.2%}")
    
    return model, final_results


if __name__ == "__main__":
    model, results = train_memory_network()