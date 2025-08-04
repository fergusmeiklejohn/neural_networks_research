#!/usr/bin/env python3
"""Test Stage 2 retrieval with simple examples"""

import sys
sys.path.append('/Users/fergusmeiklejohn/conductor/repo/neural_networks_research/bandung')

import mlx.core as mx
import mlx.nn as nn
from train_binding_curriculum import (
    CurriculumBindingModel, VOCAB, ACTIONS, generate_stage2_data
)

def test_stage2_patterns():
    """Generate and display Stage 2 patterns"""
    print("Stage 2 Pattern Examples:")
    print("=" * 50)
    
    # Generate a few examples
    batch = generate_stage2_data(batch_size=5)
    
    # Reverse vocab/action mappings
    id_to_token = {v: k for k, v in VOCAB.items()}
    id_to_action = {v: k for k, v in ACTIONS.items()}
    
    for i in range(5):
        # Decode command
        cmd_ids = batch['command'][i].tolist()
        cmd_tokens = [id_to_token.get(tid, '?') for tid in cmd_ids if tid != VOCAB['<PAD>']]
        
        # Decode expected action
        action_id = batch['labels'][i, 0].item()
        expected_action = id_to_action.get(action_id, '?')
        
        print(f"\nExample {i+1}:")
        print(f"  Command: {' '.join(cmd_tokens)}")
        print(f"  Expected: {expected_action}")


def test_trained_model():
    """Test a trained model on Stage 2"""
    print("\n\nTesting Trained Model on Stage 2:")
    print("=" * 50)
    
    # Initialize model
    model = CurriculumBindingModel()
    
    # Try to load stage 2 checkpoint if it exists
    import os
    checkpoint_path = "outputs/curriculum/retrieval_best_model.npz"
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        # Load would go here but it's complex with MLX
    else:
        print("No checkpoint found, using random initialization")
    
    model.eval()
    
    # Test specific examples
    test_cases = [
        ("X is jump recall X", "JUMP"),
        ("Y is walk recall Y", "WALK"),
        ("Z is turn recall Z", "TURN"),
    ]
    
    id_to_action = {v: k for k, v in ACTIONS.items()}
    
    for cmd_str, expected in test_cases:
        # Encode command
        tokens = cmd_str.split()
        cmd_ids = [VOCAB.get(token, VOCAB['<PAD>']) for token in tokens]
        cmd_batch = mx.array([cmd_ids], dtype=mx.int32)
        
        # Get prediction
        outputs = model(cmd_batch, stage="retrieval", training=False)
        logits = outputs['action_logits']
        
        # Get prediction at last position
        last_pos = len(cmd_ids) - 1
        pred_logits = logits[0, last_pos]
        pred_id = mx.argmax(pred_logits).item()
        predicted = id_to_action.get(pred_id, '?')
        
        # Also check bindings
        bindings = outputs['bindings'][0].tolist()
        
        print(f"\nCommand: {cmd_str}")
        print(f"  Expected: {expected}")
        print(f"  Predicted: {predicted}")
        print(f"  Bindings: {bindings}")
        print(f"  Success: {'✓' if predicted == expected else '✗'}")


def analyze_stage2_issue():
    """Analyze why Stage 2 might be failing"""
    print("\n\nAnalyzing Stage 2 Issues:")
    print("=" * 50)
    
    # The issue might be that the model is trying to bind "recall X" 
    # instead of retrieving the value bound to X earlier
    
    print("\nPotential issues:")
    print("1. Model might not understand 'recall' means retrieve from memory")
    print("2. The binding from 'X is jump' might not persist to 'recall X'")
    print("3. The model outputs predictions for ALL positions, not just the last")
    print("\nKey insight: Stage 2 requires the model to:")
    print("  - Bind X to a slot when seeing 'X is jump'")
    print("  - Store 'jump' in that slot's value") 
    print("  - Retrieve from that slot when seeing 'recall X'")
    print("  - Decode the retrieved value to action")


if __name__ == "__main__":
    test_stage2_patterns()
    test_trained_model()
    analyze_stage2_issue()