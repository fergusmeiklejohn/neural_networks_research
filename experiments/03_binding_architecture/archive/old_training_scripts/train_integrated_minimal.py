#!/usr/bin/env python3
"""Minimal training script for integrated model - focus on getting training loop working."""

from utils.imports import setup_project_paths
setup_project_paths()

from utils.config import setup_environment
config = setup_environment()

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from train_integrated_model import IntegratedBindingModel, VOCAB, ACTIONS

# Ensure necessary tokens
if 'then' not in VOCAB:
    VOCAB['then'] = len(VOCAB)
if 'and' not in VOCAB:
    VOCAB['and'] = len(VOCAB)

def create_simple_data(num_samples=100):
    """Create simple training data."""
    data = []
    
    # Pattern 1: Basic binding
    for _ in range(num_samples // 4):
        var = np.random.choice(['X', 'Y', 'Z'])
        action = np.random.choice(['WALK', 'JUMP', 'TURN'])
        
        command = f"{var} means {action.lower()} do {var}"
        tokens = [VOCAB.get(word, VOCAB['<PAD>']) for word in command.split()]
        
        data.append({
            'command': mx.array([tokens]),
            'labels': mx.array([ACTIONS[action]]),
            'stage': 'full'
        })
    
    # Pattern 2: Temporal
    for _ in range(num_samples // 4):
        var = np.random.choice(['X', 'Y', 'Z'])
        action = np.random.choice(['WALK', 'JUMP', 'TURN'])
        modifier = np.random.choice(['twice', 'thrice'])
        count = 2 if modifier == 'twice' else 3
        
        command = f"{var} means {action.lower()} do {var} {modifier}"
        tokens = [VOCAB.get(word, VOCAB['<PAD>']) for word in command.split()]
        
        data.append({
            'command': mx.array([tokens]),
            'labels': mx.array([ACTIONS[action]] * count),
            'stage': 'full'
        })
    
    # Pattern 3: Sequential
    for _ in range(num_samples // 4):
        var1 = 'X'
        var2 = 'Y'
        action1 = np.random.choice(['WALK', 'JUMP', 'TURN'])
        action2 = np.random.choice(['WALK', 'JUMP', 'TURN'])
        
        command = f"{var1} means {action1.lower()} do {var1} then {var2} means {action2.lower()} do {var2}"
        tokens = [VOCAB.get(word, VOCAB['<PAD>']) for word in command.split()]
        
        data.append({
            'command': mx.array([tokens]),
            'labels': mx.array([ACTIONS[action1], ACTIONS[action2]]),
            'stage': 'full'
        })
    
    # Pattern 4: Rebinding
    for _ in range(num_samples // 4):
        var = 'X'
        action1 = np.random.choice(['WALK', 'JUMP', 'TURN'])
        action2 = np.random.choice(['WALK', 'JUMP', 'TURN'])
        
        command = f"{var} means {action1.lower()} do {var} then {var} means {action2.lower()} do {var}"
        tokens = [VOCAB.get(word, VOCAB['<PAD>']) for word in command.split()]
        
        data.append({
            'command': mx.array([tokens]),
            'labels': mx.array([ACTIONS[action1], ACTIONS[action2]]),
            'stage': 'full'
        })
    
    return data


def train_minimal():
    """Minimal training loop."""
    print("=== Minimal Training for Integrated Model ===\n")
    
    # Create model
    model = IntegratedBindingModel(
        vocab_size=len(VOCAB),
        num_actions=len(ACTIONS),
        embed_dim=128,
        num_slots=4
    )
    
    optimizer = optim.Adam(learning_rate=0.001)
    
    # Create simple training data
    print("Creating training data...")
    train_data = create_simple_data(400)
    
    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Shuffle data
        np.random.shuffle(train_data)
        
        for i, batch in enumerate(train_data):
            inputs = {
                'command': batch['command']
            }
            
            # Forward pass and loss computation
            def loss_fn(model):
                outputs = model(inputs, stage=batch['stage'])
                labels = batch['labels']
                
                # Ensure shapes match
                if len(outputs) > len(labels):
                    outputs = outputs[:len(labels)]
                elif len(outputs) < len(labels):
                    # Pad with zeros
                    padding = mx.zeros((len(labels) - len(outputs), outputs.shape[-1]))
                    outputs = mx.concatenate([outputs, padding])
                
                # Cross entropy loss
                loss = mx.mean(nn.losses.cross_entropy(outputs, labels))
                
                # Accuracy
                predictions = mx.argmax(outputs, axis=1)
                accuracy = mx.mean(predictions == labels)
                
                return loss, accuracy
            
            loss_val, accuracy = loss_fn(model)
            
            # Compute gradients
            def loss_only(model):
                outputs = model(inputs, stage=batch['stage'])
                labels = batch['labels']
                
                if len(outputs) > len(labels):
                    outputs = outputs[:len(labels)]
                elif len(outputs) < len(labels):
                    padding = mx.zeros((len(labels) - len(outputs), outputs.shape[-1]))
                    outputs = mx.concatenate([outputs, padding])
                
                return mx.mean(nn.losses.cross_entropy(outputs, labels))
            
            # Gradient step
            grad_fn = mx.grad(loss_only)
            grads = grad_fn(model)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            
            # Update stats
            total_loss += float(loss_val)
            correct += float(accuracy) * len(batch['labels'])
            total += len(batch['labels'])
            
            if i % 50 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {i}/{len(train_data)}, "
                      f"Loss: {float(loss_val):.4f}, Acc: {float(accuracy):.2%}")
        
        # Epoch summary
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}: Avg Loss = {total_loss/len(train_data):.4f}, "
              f"Accuracy = {epoch_acc:.2%}\n")
    
    # Test the trained model
    test_cases = [
        ("X means jump do X", ['JUMP']),
        ("Y means walk do Y twice", ['WALK', 'WALK']),
        ("X means jump do X then Y means walk do Y", ['JUMP', 'WALK']),
        ("X means jump do X then X means walk do X", ['JUMP', 'WALK']),
    ]
    
    print("\n=== Testing Trained Model ===")
    for command, expected in test_cases:
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
        
        print(f"\nCommand: {command}")
        print(f"Expected: {expected}")
        print(f"Predicted: {predicted_actions}")
        print(f"Correct: {predicted_actions == expected}")
    
    print("\nTraining complete!")
    return model


if __name__ == "__main__":
    model = train_minimal()