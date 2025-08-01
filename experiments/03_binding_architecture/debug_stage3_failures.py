#!/usr/bin/env python3
"""
Diagnostic tool to understand Stage 3 failures in detail
"""

import sys
sys.path.append('.')

import mlx.core as mx
import numpy as np
from train_binding_curriculum import (
    generate_stage3_data, VOCAB, ACTIONS
)
from train_temporal_curriculum import TemporalDynamicMemoryModel


def analyze_stage3_failures():
    """Analyze specific failure patterns in Stage 3"""
    
    # Initialize model (untrained for pattern analysis)
    model = TemporalDynamicMemoryModel(
        vocab_size=len(VOCAB),
        num_actions=len(ACTIONS),
        embed_dim=64,
        num_slots=4,
        num_heads=8
    )
    
    # Generate Stage 3 data samples
    batch = generate_stage3_data(20)
    
    print("Stage 3 Pattern Analysis")
    print("=" * 80)
    print("\nStage 3 characteristics:")
    print("- Multiple variables with bindings")
    print("- Distractors between binding and usage")
    print("- Complex patterns including temporal modifiers")
    print("\n" + "=" * 80)
    
    # Analyze each sample
    for i in range(min(5, batch['command'].shape[0])):  # Look at first 5
        cmd = batch['command'][i]
        labels = batch['labels'][i]
        
        # Decode command
        tokens = []
        for token_id in cmd:
            if token_id.item() == VOCAB['<PAD>']:
                break
            token = [k for k, v in VOCAB.items() if v == token_id.item()][0]
            tokens.append(token)
        
        # Find valid labels
        valid_labels = []
        for label_id in labels:
            if label_id.item() == VOCAB['<PAD>']:
                break
            action = [k for k, v in ACTIONS.items() if v == label_id.item()][0]
            valid_labels.append(action)
        
        print(f"\nExample {i+1}:")
        print(f"Command: {' '.join(tokens)}")
        print(f"Expected: {valid_labels}")
        
        # Analyze pattern complexity
        num_variables = sum(1 for t in tokens if t in ['X', 'Y', 'Z', 'A', 'B'])
        num_bindings = tokens.count('means')
        has_temporal = any(t in ['twice', 'thrice'] for t in tokens)
        do_position = tokens.index('do') if 'do' in tokens else -1
        
        print(f"Complexity:")
        print(f"  - Variables used: {num_variables}")
        print(f"  - Bindings: {num_bindings}")
        print(f"  - Has temporal: {has_temporal}")
        print(f"  - 'do' at position: {do_position}")
        
        # Identify potential confusion points
        if num_bindings > 1:
            print(f"  ⚠️  Multiple bindings - potential for confusion")
        
        if do_position > 10:
            print(f"  ⚠️  Long distance between binding and usage")
            
        if has_temporal and num_bindings > 1:
            print(f"  ⚠️  Complex: temporal + multiple variables")
    
    # Statistical analysis
    print("\n" + "=" * 80)
    print("Statistical Analysis of Stage 3 Patterns")
    print("=" * 80)
    
    total_samples = 100
    batch = generate_stage3_data(total_samples)
    
    # Collect statistics
    stats = {
        'single_var': 0,
        'multi_var': 0,
        'has_temporal': 0,
        'long_distance': 0,
        'avg_length': 0,
        'avg_bindings': 0
    }
    
    total_length = 0
    total_bindings = 0
    
    for i in range(batch['command'].shape[0]):
        cmd = batch['command'][i]
        
        # Decode and analyze
        tokens = []
        for token_id in cmd:
            if token_id.item() == VOCAB['<PAD>']:
                break
            token = [k for k, v in VOCAB.items() if v == token_id.item()][0]
            tokens.append(token)
        
        unique_vars = set(t for t in tokens if t in ['X', 'Y', 'Z', 'A', 'B'])
        num_bindings = tokens.count('means')
        
        if len(unique_vars) == 1:
            stats['single_var'] += 1
        else:
            stats['multi_var'] += 1
            
        if any(t in ['twice', 'thrice'] for t in tokens):
            stats['has_temporal'] += 1
            
        if 'do' in tokens:
            do_pos = tokens.index('do')
            if do_pos > 10:
                stats['long_distance'] += 1
                
        total_length += len(tokens)
        total_bindings += num_bindings
    
    stats['avg_length'] = total_length / total_samples
    stats['avg_bindings'] = total_bindings / total_samples
    
    print(f"Single variable patterns: {stats['single_var']}%")
    print(f"Multi-variable patterns: {stats['multi_var']}%") 
    print(f"Contains temporal modifier: {stats['has_temporal']}%")
    print(f"Long distance (>10 tokens to 'do'): {stats['long_distance']}%")
    print(f"Average sequence length: {stats['avg_length']:.1f} tokens")
    print(f"Average bindings per pattern: {stats['avg_bindings']:.1f}")
    
    # Hypothesis about failures
    print("\n" + "=" * 80)
    print("Hypotheses for Stage 3 Failures")
    print("=" * 80)
    print("\n1. **Interference between multiple bindings**")
    print("   - When multiple variables are bound, later bindings may overwrite earlier ones")
    print("   - Dynamic memory updates might not properly isolate different variables")
    print("\n2. **Long-range dependencies**")
    print("   - Distance between binding and usage challenges attention mechanisms")
    print("   - Distractors between binding and retrieval may corrupt memory")
    print("\n3. **Temporal + multi-variable complexity**")
    print("   - Combining temporal patterns with multiple variables exceeds model capacity")
    print("   - Need better compositional mechanisms")
    print("\n4. **Training curriculum imbalance**")
    print("   - Jump from Stage 2 (simple retrieval) to Stage 3 (full complexity) too large")
    print("   - May need intermediate stages")


if __name__ == "__main__":
    analyze_stage3_failures()