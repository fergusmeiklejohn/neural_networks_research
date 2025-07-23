#!/usr/bin/env python3
"""
Debug the errors found on Paperspace.
"""

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import tensorflow as tf
import numpy as np
from models import create_model
from models_v2 import create_model_v2
from train_progressive_curriculum import SCANTokenizer, create_dataset

print("=== DEBUGGING PAPERSPACE ERRORS ===\n")

# Create dummy tokenizer
tokenizer = SCANTokenizer()
tokenizer.command_to_id = {f'word{i}': i for i in range(20)}
tokenizer.action_to_id = {f'act{i}': i for i in range(10)}

# Test 1: v1 model count_params error
print("1. Testing v1 model count_params issue...")
try:
    model_v1 = create_model(20, 10, d_model=64)
    # This should fail
    params = model_v1.count_params()
    print(f"✗ ERROR: count_params worked without building! ({params:,} params)")
except ValueError as e:
    print(f"✓ Reproduced error: {str(e)[:100]}...")
    
    # Fix: Build the model first
    print("\nAttempting fix: Building model first...")
    dummy_inputs = {
        'command': tf.constant([[1, 2, 3, 4, 5]]),
        'target': tf.constant([[1, 2, 3, 4, 5, 6]]),
        'modification': tf.constant([[7, 8, 9]])
    }
    
    # Call model once to build it
    _ = model_v1(dummy_inputs, training=False)
    
    # Now count_params should work
    params = model_v1.count_params()
    print(f"✓ Fixed! Model has {params:,} parameters")

# Test 2: v2 model tf.cond error
print("\n\n2. Testing v2 model tf.cond issue...")
try:
    model_v2 = create_model_v2(20, 10, d_model=64)
    
    # Create test data that matches Paperspace shapes
    test_batch = {
        'command': tf.constant(np.random.randint(0, 20, size=(4, 50))),
        'target': tf.constant(np.random.randint(0, 10, size=(4, 99))),
        'modification': tf.constant(np.random.randint(0, 20, size=(4, 20)))
    }
    
    print("Input shapes:")
    for k, v in test_batch.items():
        print(f"  {k}: {v.shape}")
    
    # This should fail with [5, 2] error
    output = model_v2(test_batch, training=True)
    print(f"✗ ERROR: v2 model worked! Output shape: {output.shape}")
    
except Exception as e:
    print(f"✓ Reproduced error: {e}")
    print(f"Error type: {type(e).__name__}")
    
    # Debug the tf.cond issue
    print("\n\nDebugging tf.cond...")
    
    # Let's check what's happening in the model
    print("Checking model components...")
    
    # Test rule extractor
    try:
        rule_outputs = model_v2.rule_extractor(test_batch['command'], training=True)
        print(f"✓ Rule extractor works, embeddings shape: {rule_outputs['embeddings'].shape}")
    except Exception as e:
        print(f"✗ Rule extractor failed: {e}")
    
    # Test the problematic tf.cond
    print("\nTesting tf.cond logic...")
    
    # Simulate the logic from models_v2.py
    modification = test_batch['modification']
    mod_sum = tf.reduce_sum(tf.abs(modification))
    print(f"mod_sum: {mod_sum}")
    print(f"mod_sum > 0: {mod_sum > 0}")
    
    # The issue might be with the lambda functions
    rule_embeddings = rule_outputs['embeddings']
    
    def test_cond():
        def apply_mod():
            print("In apply_mod")
            # Test if rule_modifier works
            modified, gates = model_v2.rule_modifier(
                rule_embeddings, 
                modification, 
                training=True
            )
            return modified, gates
        
        def keep_orig():
            print("In keep_orig")
            return rule_embeddings, None
        
        # This is where the error happens
        result = tf.cond(
            mod_sum > 0,
            apply_mod,
            keep_orig
        )
        return result
    
    try:
        result = test_cond()
        print("✓ tf.cond worked!")
    except Exception as e:
        print(f"✗ tf.cond failed: {e}")
        
        # The issue is likely that the two branches return different structures
        # apply_mod returns (tensor, gates)
        # keep_orig returns (tensor, None)
        # tf.cond requires both branches to return the same structure
        
        print("\nThe issue: tf.cond branches return different structures!")
        print("apply_mod returns: (modified_embeddings, gates_list)")
        print("keep_orig returns: (embeddings, None)")
        print("\nNeed to make both branches return the same structure.")

# Test 3: Check if our validation would catch these
print("\n\n3. Checking if our validation catches these...")

# Simulate what our validation does
print("Our validation process:")
print("- Creates model")
print("- Calls model with test inputs") 
print("- Does NOT call count_params before building")
print("- Does NOT test with realistic batch sizes")

print("\n❌ Our validation missed these because:")
print("1. We didn't test count_params in the validation")
print("2. We used different input shapes than real training")
print("3. The tf.cond error only appears with certain input conditions")