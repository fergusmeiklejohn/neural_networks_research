#!/usr/bin/env python3
"""Simplified training script focusing only on nested temporal patterns."""

from utils.imports import setup_project_paths
setup_project_paths()

from utils.config import setup_environment
from utils.paths import get_output_path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import os
from typing import Dict, List, Any
from train_integrated_model import IntegratedBindingModel, VOCAB, ACTIONS
from nested_temporal_patterns import NestedTemporalParser, NestedTemporalExecutor
from mlx_model_io import save_model_simple

config = setup_environment()


class SimpleNestedTemporalModel(IntegratedBindingModel):
    """Simplified model focusing on nested temporal patterns."""
    
    def __init__(self, vocab_size: int, num_actions: int, embed_dim: int = 256,
                 num_slots: int = 4, num_heads: int = 8, mlp_hidden_dim: int = 512):
        super().__init__(vocab_size, num_actions, embed_dim, num_slots, num_heads, mlp_hidden_dim)
        
        self.temporal_parser = NestedTemporalParser(VOCAB)
        self.temporal_executor = NestedTemporalExecutor(self, VOCAB, ACTIONS)
    
    def __call__(self, inputs: Dict[str, mx.array], stage: str = "full") -> mx.array:
        """Forward pass for nested temporal patterns only."""
        command_ids = inputs['command']
        
        # Clear versioned memory
        self.versioned_memory.clear()
        
        # Convert to list
        if hasattr(command_ids, 'numpy'):
            token_array = command_ids.numpy()
        else:
            token_array = command_ids
        
        if len(token_array.shape) > 1:
            token_array = token_array[0]
        
        token_list = token_array.tolist()
        token_list = [int(t) for t in token_list]
        
        # Extract bindings from command
        bindings = self._extract_bindings(token_list)
        
        # Execute nested temporal pattern
        outputs = self.temporal_executor.execute_nested_pattern(token_list, bindings)
        
        # Stack outputs
        if outputs:
            return mx.stack(outputs)
        else:
            return mx.zeros((1, self.num_actions))
    
    def _extract_bindings(self, tokens: List[int]) -> Dict[str, str]:
        """Extract variable bindings from command tokens."""
        id_to_word = {v: k for k, v in VOCAB.items()}
        words = [id_to_word.get(t, '<PAD>') for t in tokens]
        
        bindings = {}
        i = 0
        while i < len(words) - 2:
            if words[i] in ['X', 'Y', 'Z'] and words[i+1] == 'means':
                var_name = words[i]
                action = words[i+2].upper()
                if action in ACTIONS:
                    bindings[var_name] = action
                i += 3
            else:
                i += 1
        
        return bindings


def generate_simple_nested_data(num_samples: int = 100) -> List[Dict[str, Any]]:
    """Generate only nested temporal pattern data."""
    # Ensure temporal modifiers in vocabulary
    for word in ['twice', 'thrice', 'do', 'then', 'means']:
        if word not in VOCAB:
            VOCAB[word] = len(VOCAB)
    
    data = []
    
    for i in range(num_samples):
        # Choose pattern type
        pattern_type = i % 4
        
        if pattern_type == 0:
            # Simple nested: "X means walk do X twice twice" = 4 walks
            var = np.random.choice(['X', 'Y', 'Z'])
            action = np.random.choice(['WALK', 'JUMP', 'TURN'])
            command = f"{var} means {action.lower()} do {var} twice twice"
            num_actions = 4
            
        elif pattern_type == 1:
            # Mixed modifiers: "X means jump do X thrice twice" = 6 jumps
            var = np.random.choice(['X', 'Y', 'Z'])
            action = np.random.choice(['WALK', 'JUMP', 'TURN'])
            command = f"{var} means {action.lower()} do {var} thrice twice"
            num_actions = 6
            
        elif pattern_type == 2:
            # Three levels: "X means turn do X twice twice twice" = 8 turns
            var = np.random.choice(['X', 'Y'])
            action = np.random.choice(['WALK', 'JUMP', 'TURN'])
            command = f"{var} means {action.lower()} do {var} twice twice twice"
            num_actions = 8
            
        else:
            # Sequential with nested: "X means walk Y means jump do X twice twice then do Y thrice"
            var1, var2 = 'X', 'Y'
            action1 = np.random.choice(['WALK', 'TURN'])
            action2 = np.random.choice(['JUMP', 'RUN'])
            command = f"{var1} means {action1.lower()} {var2} means {action2.lower()} do {var1} twice twice then do {var2} thrice"
            labels = [ACTIONS[action1]] * 4 + [ACTIONS[action2]] * 3
            
            tokens = [VOCAB.get(word, VOCAB['<PAD>']) for word in command.split()]
            data.append({
                'command': mx.array([tokens]),
                'labels': mx.array(labels),
                'stage': 'full'
            })
            continue
        
        # For non-sequential patterns
        tokens = [VOCAB.get(word, VOCAB['<PAD>']) for word in command.split()]
        labels = [ACTIONS[action]] * num_actions
        
        data.append({
            'command': mx.array([tokens]),
            'labels': mx.array(labels),
            'stage': 'full'
        })
    
    return data


def train_simple_nested(num_epochs: int = 10):
    """Train only on nested temporal patterns."""
    print("=== Training Simple Nested Temporal Model ===")
    
    # Create model
    vocab_size = len(VOCAB)
    num_actions = len(ACTIONS)
    model = SimpleNestedTemporalModel(vocab_size, num_actions)
    
    # Generate training data
    print("\nGenerating nested temporal data...")
    train_data = generate_simple_nested_data(200)
    print(f"Total training samples: {len(train_data)}")
    
    # Setup optimizer
    optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=0.01)
    
    # Training loop
    best_accuracy = 0.0
    
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Shuffle data
        indices = np.random.permutation(len(train_data))
        train_data_shuffled = [train_data[i] for i in indices]
        
        for i, batch in enumerate(train_data_shuffled):
            inputs = {
                'command': batch['command'],
                'stage': 'full'
            }
            
            # Forward pass
            outputs = model(inputs, stage='full')
            labels = batch['labels']
            
            # Compute loss
            if outputs.shape[0] != labels.shape[0]:
                print(f"Shape mismatch: outputs {outputs.shape} vs labels {labels.shape}")
                if outputs.shape[0] > labels.shape[0]:
                    outputs = outputs[:labels.shape[0]]
                else:
                    padding = mx.zeros((labels.shape[0] - outputs.shape[0], outputs.shape[-1]))
                    outputs = mx.concatenate([outputs, padding])
            
            loss = mx.mean(nn.losses.cross_entropy(outputs, labels))
            predictions = mx.argmax(outputs, axis=1)
            accuracy = mx.mean(predictions == labels)
            
            # Gradient update
            def loss_fn(model):
                out = model(inputs, stage='full')
                if out.shape[0] != labels.shape[0]:
                    if out.shape[0] > labels.shape[0]:
                        out = out[:labels.shape[0]]
                    else:
                        pad = mx.zeros((labels.shape[0] - out.shape[0], out.shape[-1]))
                        out = mx.concatenate([out, pad])
                return mx.mean(nn.losses.cross_entropy(out, labels))
            
            grad_fn = mx.grad(loss_fn)
            grads = grad_fn(model)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            
            # Update stats
            total_loss += float(loss)
            correct += float(accuracy) * len(labels)
            total += len(labels)
            
            if i % 50 == 0:
                print(f"Batch {i}/{len(train_data)}, Loss: {float(loss):.4f}, Acc: {float(accuracy):.2%}")
        
        # Epoch summary
        epoch_acc = correct / total
        print(f"\nEpoch {epoch+1}: Avg Loss = {total_loss/len(train_data):.4f}, Accuracy = {epoch_acc:.2%}")
        
        if epoch_acc > best_accuracy:
            best_accuracy = epoch_acc
            save_path = os.path.join(get_output_path(), 'nested_temporal_simple_best.pkl')
            save_model_simple(save_path, model)
            print(f"Saved best model with accuracy {best_accuracy:.2%}")
    
    print(f"\nTraining complete! Best accuracy: {best_accuracy:.2%}")
    return model


def test_simple_nested(model: SimpleNestedTemporalModel):
    """Test the simple nested temporal model."""
    print("\n=== Testing Simple Nested Temporal Model ===")
    
    test_cases = [
        ("X means jump do X twice", ['JUMP', 'JUMP']),
        ("X means jump do X twice twice", ['JUMP', 'JUMP', 'JUMP', 'JUMP']),
        ("X means walk do X thrice twice", ['WALK'] * 6),
        ("X means turn do X twice twice twice", ['TURN'] * 8),
        ("X means walk Y means jump do X twice twice then do Y thrice", 
         ['WALK'] * 4 + ['JUMP'] * 3),
    ]
    
    for command, expected in test_cases:
        print(f"\nCommand: {command}")
        print(f"Expected: {expected}")
        
        tokens = [VOCAB.get(word, VOCAB['<PAD>']) for word in command.split()]
        inputs = {'command': mx.array([tokens])}
        
        outputs = model(inputs, stage='full')
        predictions = mx.argmax(outputs, axis=1)
        
        predicted_actions = []
        for pred in predictions:
            for name, idx in ACTIONS.items():
                if idx == int(pred):
                    predicted_actions.append(name)
                    break
        
        print(f"Predicted: {predicted_actions}")
        print(f"Correct: {predicted_actions == expected}")
        
        # Show parsed structure
        token_list = [int(t) for t in tokens]
        temporal_nodes = model.temporal_parser.parse(token_list)
        if temporal_nodes:
            from nested_temporal_patterns import describe_temporal_node
            print("Temporal structure:")
            for node in temporal_nodes:
                print(f"  {describe_temporal_node(node)}")


if __name__ == "__main__":
    model = train_simple_nested(num_epochs=5)
    test_simple_nested(model)