#!/usr/bin/env python3
"""
Simple test of temporal pattern detection logic
"""

import sys
sys.path.append('.')

import mlx.core as mx
from train_binding_curriculum import VOCAB
from train_temporal_curriculum import TemporalDynamicMemoryModel


def test_temporal_detection():
    """Test the temporal pattern detection logic"""
    
    # Initialize a model (untrained)
    model = TemporalDynamicMemoryModel(
        vocab_size=len(VOCAB),
        num_actions=6,
        embed_dim=64,
        num_slots=4,
        num_heads=8
    )
    
    # Test cases
    test_cases = [
        "X means jump do X twice",
        "Y means walk do Y thrice", 
        "X means turn Y means run do X twice",
        "X means jump do X"  # No temporal modifier
    ]
    
    print("Testing Temporal Pattern Detection")
    print("=" * 60)
    
    for cmd_str in test_cases:
        tokens = cmd_str.split()
        token_ids = [VOCAB.get(t, VOCAB['<PAD>']) for t in tokens]
        cmd_batch = mx.array([token_ids], dtype=mx.int32)
        
        print(f"\nCommand: {cmd_str}")
        print(f"Tokens: {tokens}")
        
        # Check each position for temporal patterns
        for i, token in enumerate(tokens):
            is_temporal, repeat_count, var_id = model.detect_temporal_patterns(cmd_batch, i)
            if is_temporal:
                var_token = [k for k, v in VOCAB.items() if v == var_id][0]
                print(f"  Position {i} ('{token}'): Temporal modifier detected!")
                print(f"    Repeat count: {repeat_count}")
                print(f"    Variable: {var_token}")
        
        # Run forward pass to see if temporal actions are generated
        outputs = model(cmd_batch, stage="full", training=False)
        num_temporal = outputs.get('temporal_actions', 0)
        print(f"  Temporal actions generated: {num_temporal}")


if __name__ == "__main__":
    test_temporal_detection()