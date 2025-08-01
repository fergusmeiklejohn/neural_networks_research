"""Test the trained sequential model on specific patterns."""

from utils.imports import setup_project_paths
setup_project_paths()

from utils.config import setup_environment
from utils.paths import get_data_path, get_output_path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import os
from typing import Dict, List

from train_binding_curriculum import VOCAB, ACTIONS
from train_sequential_planning_fixed import SequentialDynamicMemoryModel

config = setup_environment()


def extract_action_predictions(model, command: str) -> List[str]:
    """Extract action predictions from a command."""
    # Tokenize command
    command_tokens = command.split()
    command_ids = mx.array([[VOCAB.get(token, VOCAB['<PAD>']) for token in command_tokens]])
    
    # Get model outputs
    outputs = model(command_ids, stage="full_binding")
    
    if 'action_logits' not in outputs:
        return []
    
    # Get predictions
    action_logits = outputs['action_logits']
    predictions = mx.argmax(action_logits, axis=-1)
    
    # Extract only valid action predictions
    predicted_actions = []
    action_count = 0
    
    # Count expected actions
    for i, token in enumerate(command_tokens):
        if i > 0 and command_tokens[i-1] == 'do':
            if token in ['X', 'Y', 'Z']:
                # Check for modifiers
                if i + 1 < len(command_tokens):
                    if command_tokens[i+1] == 'twice':
                        action_count += 2
                    elif command_tokens[i+1] == 'thrice':
                        action_count += 3
                    else:
                        action_count += 1
                else:
                    action_count += 1
    
    # Extract that many predictions
    predictions_np = np.array(predictions)
    for j in range(min(action_count, predictions_np.shape[1])):
        pred_id = int(predictions_np[0, j])
        for action_name, action_id in ACTIONS.items():
            if action_id == pred_id and action_name != '<PAD>':
                predicted_actions.append(action_name)
                break
    
    return predicted_actions


def test_sequential_patterns():
    """Test various sequential patterns."""
    # Initialize model
    model = SequentialDynamicMemoryModel(
        vocab_size=len(VOCAB),
        num_actions=len(ACTIONS),
        embed_dim=128,
        num_slots=4,
        num_heads=4,
        mlp_hidden_dim=256
    )
    
    # Load latest model weights if available
    output_dir = get_output_path('models')
    model_files = [f for f in os.listdir(output_dir) if f.startswith('sequential_fixed_') and f.endswith('.npz')]
    
    if model_files:
        latest_model = sorted(model_files)[-1]
        model_path = os.path.join(output_dir, latest_model)
        print(f"Loading model from: {model_path}")
        
        weights = mx.load(model_path)
        # MLX models need manual parameter setting
        model.update(weights)
        print("Model loaded successfully!")
    else:
        print("No saved model found, using random initialization")
    
    # Test patterns
    test_cases = [
        # Basic patterns
        ("X means jump do X", ["JUMP"]),
        ("Y means walk do Y", ["WALK"]),
        ("Z means turn do Z", ["TURN"]),
        
        # Sequential patterns with "then"
        ("X means jump do X then Y means walk do Y", ["JUMP", "WALK"]),
        ("Y means walk do Y then Z means turn do Z", ["WALK", "TURN"]),
        ("X means run do X then Y means jump do Y then Z means walk do Z", ["RUN", "JUMP", "WALK"]),
        
        # Temporal patterns
        ("X means jump do X twice", ["JUMP", "JUMP"]),
        ("Y means walk do Y thrice", ["WALK", "WALK", "WALK"]),
        
        # Combined patterns
        ("X means jump do X twice then Y means walk do Y", ["JUMP", "JUMP", "WALK"]),
        ("Z means turn do Z then X means run do X thrice", ["TURN", "RUN", "RUN", "RUN"]),
        ("X means jump do X then Y means walk do Y twice then Z means turn do Z", ["JUMP", "WALK", "WALK", "TURN"]),
    ]
    
    print("\n" + "="*70)
    print("TESTING SEQUENTIAL PLANNING MODEL")
    print("="*70)
    
    correct = 0
    total = 0
    
    for command, expected in test_cases:
        predicted = extract_action_predictions(model, command)
        is_correct = predicted == expected
        
        print(f"\nCommand: {command}")
        print(f"Expected: {expected}")
        print(f"Predicted: {predicted}")
        print(f"✓ CORRECT" if is_correct else "✗ INCORRECT")
        
        if is_correct:
            correct += 1
        total += 1
    
    print("\n" + "="*70)
    print(f"OVERALL ACCURACY: {correct}/{total} ({correct/total*100:.1f}%)")
    print("="*70)
    
    # Detailed analysis of a complex pattern
    print("\n" + "="*70)
    print("DETAILED ANALYSIS: Sequential with Temporal Pattern")
    print("="*70)
    
    complex_command = "X means jump do X twice then Y means walk do Y then Z means turn do Z thrice"
    command_tokens = complex_command.split()
    command_ids = mx.array([[VOCAB.get(token, VOCAB['<PAD>']) for token in command_tokens]])
    
    outputs = model(command_ids, stage="full_binding")
    
    print(f"\nCommand: {complex_command}")
    print(f"Segments identified: {outputs.get('segments', [])}")
    print(f"Number of temporal actions: {outputs.get('temporal_actions', 0)}")
    print(f"Bindings: {outputs.get('bindings', {})}")
    
    predicted = extract_action_predictions(model, complex_command)
    expected = ["JUMP", "JUMP", "WALK", "TURN", "TURN", "TURN"]
    print(f"\nExpected: {expected}")
    print(f"Predicted: {predicted}")
    print(f"✓ CORRECT" if predicted == expected else "✗ INCORRECT")


if __name__ == "__main__":
    test_sequential_patterns()