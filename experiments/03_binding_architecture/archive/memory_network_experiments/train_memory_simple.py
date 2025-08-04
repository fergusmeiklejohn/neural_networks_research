#!/usr/bin/env python3
"""Simplified training script for Neural Memory Binding model."""

from utils.imports import setup_project_paths
setup_project_paths()

from utils.config import setup_environment
config = setup_environment()

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from typing import Dict, List
import numpy as np

from train_memory_network import ImprovedMemoryBindingModel
from progressive_complexity_dataset import (
    ProgressiveComplexityDataset, VOCAB, ACTIONS
)


def train_single_sample(model, optimizer, sample: Dict) -> float:
    """Train on a single sample."""
    # Get tokens and expected actions
    tokens = mx.array([sample['tokens']])  # Add batch dimension
    expected_indices = sample['expected_indices']
    
    def loss_fn(model):
        # Get predictions
        predictions = model({'command': tokens})
        
        if predictions.shape[0] == 0 or len(expected_indices) == 0:
            return mx.array(0.0)
        
        # Match prediction length to expected
        n_expected = len(expected_indices)
        if predictions.shape[0] < n_expected:
            # Pad predictions with zeros
            padding = mx.zeros((n_expected - predictions.shape[0], predictions.shape[1]))
            predictions = mx.concatenate([predictions, padding], axis=0)
        elif predictions.shape[0] > n_expected:
            # Truncate predictions
            predictions = predictions[:n_expected]
        
        # Compute loss
        targets = mx.array(expected_indices)
        loss = nn.losses.cross_entropy(predictions, targets)
        return mx.mean(loss)
    
    # Compute loss and gradients
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad_fn(model)
    
    # Update weights
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    
    return loss.item()


def evaluate_simple(model, dataset: List[Dict]) -> Dict[str, float]:
    """Simple evaluation."""
    model.eval()
    
    correct = 0
    total = 0
    level_correct = {1: 0, 2: 0, 3: 0, 4: 0}
    level_total = {1: 0, 2: 0, 3: 0, 4: 0}
    
    for sample in dataset:
        tokens = mx.array([sample['tokens']])
        expected = sample['expected_indices']
        level = sample['complexity_level']
        
        # Get predictions
        predictions = model({'command': tokens})
        
        if predictions.shape[0] > 0 and len(expected) > 0:
            # Get predicted actions
            pred_indices = mx.argmax(predictions, axis=1).tolist()
            
            # Truncate to match expected length
            pred_indices = pred_indices[:len(expected)]
            
            # Check if correct (must match exactly)
            if pred_indices == expected:
                correct += 1
                level_correct[level] += 1
        
        total += 1
        level_total[level] += 1
    
    # Calculate accuracies
    results = {
        'overall': correct / total if total > 0 else 0
    }
    
    for level in range(1, 5):
        if level_total[level] > 0:
            results[f'level_{level}'] = level_correct[level] / level_total[level]
        else:
            results[f'level_{level}'] = 0.0
            
    return results


def main():
    """Train memory network with simplified approach."""
    print("Training Neural Memory Binding Model (Simplified)...")
    
    # Create model
    model = ImprovedMemoryBindingModel(
        vocab_size=len(VOCAB),
        num_actions=len(ACTIONS),
        embed_dim=64,  # Smaller for faster training
        hidden_dim=128,
        num_vars=4
    )
    
    # Create dataset
    dataset_gen = ProgressiveComplexityDataset()
    
    # Generate data - start with just level 1
    print("\nGenerating Level 1 data...")
    train_data = dataset_gen.generate_level_1(100)
    test_data = dataset_gen.generate_level_1(20)
    
    # Add some level 2 data
    print("Adding Level 2 data...")
    train_data.extend(dataset_gen.generate_level_2(100))
    test_data.extend(dataset_gen.generate_level_2(20))
    
    print(f"\nTraining samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Training setup
    optimizer = optim.Adam(learning_rate=1e-3)
    
    # Training loop
    print("\n=== Training ===")
    num_epochs = 20
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        np.random.shuffle(train_data)
        
        # Train on each sample
        for i, sample in enumerate(train_data):
            loss = train_single_sample(model, optimizer, sample)
            epoch_loss += loss
            
            # Print progress
            if (i + 1) % 50 == 0:
                print(f"  Sample {i+1}/{len(train_data)}, Loss: {loss:.4f}")
        
        avg_loss = epoch_loss / len(train_data)
        
        # Evaluate
        model.eval()
        results = evaluate_simple(model, test_data)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Test Accuracy: {results['overall']:.2%}")
        print(f"  Level 1: {results.get('level_1', 0):.2%}, Level 2: {results.get('level_2', 0):.2%}")
        
        # Early stopping if we achieve good accuracy
        if results['overall'] > 0.90:
            print("\nAchieved >90% accuracy, stopping early!")
            break
    
    # Final detailed evaluation
    print("\n=== Final Evaluation ===")
    
    # Test on all levels
    all_test_data = []
    for level in range(1, 5):
        level_data = getattr(dataset_gen, f'generate_level_{level}')(25)
        all_test_data.extend(level_data)
    
    final_results = evaluate_simple(model, all_test_data)
    
    print("\nFinal Accuracy by Level:")
    for level in range(1, 5):
        print(f"  Level {level}: {final_results.get(f'level_{level}', 0):.2%}")
    print(f"\nOverall: {final_results['overall']:.2%}")
    
    # Test some specific examples
    print("\n=== Example Predictions ===")
    model.eval()
    
    test_examples = [
        "X means jump do X",
        "Y means walk do Y", 
        "X means jump Y means walk do X and Y",
        "X means run Y means turn do Y then X"
    ]
    
    for command in test_examples:
        tokens = []
        for word in command.split():
            if word in VOCAB:
                tokens.append(VOCAB[word])
            else:
                tokens.append(VOCAB['PAD'])
        
        tokens_array = mx.array([tokens])
        predictions = model({'command': tokens_array})
        
        if predictions.shape[0] > 0:
            pred_indices = mx.argmax(predictions, axis=1).tolist()
            pred_actions = [ACTIONS[i] for i in pred_indices]
        else:
            pred_actions = []
            
        print(f"\nCommand: {command}")
        print(f"Predicted: {pred_actions}")
    
    return model, final_results


if __name__ == "__main__":
    model, results = main()