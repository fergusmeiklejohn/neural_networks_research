#!/usr/bin/env python3
"""Debug the model v2 issue."""

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import tensorflow as tf
from models_v2 import create_model_v2, GatedModificationLayer

# Test the gated modification layer in isolation
print("Testing GatedModificationLayer...")

gated_layer = GatedModificationLayer(d_model=128)

# Create test inputs
batch_size = 2
seq_len = 10
d_model = 128

original = tf.random.normal((batch_size, seq_len, d_model))
mod_signal = tf.random.normal((batch_size, seq_len, d_model))

try:
    output, gate = gated_layer(original, mod_signal, training=True)
    print(f"✓ GatedModificationLayer works!")
    print(f"  Output shape: {output.shape}")
    print(f"  Gate shape: {gate.shape}")
except Exception as e:
    print(f"✗ GatedModificationLayer error: {e}")
    import traceback
    traceback.print_exc()

# Test the full model
print("\n\nTesting full model v2...")

try:
    model = create_model_v2(
        command_vocab_size=20,
        action_vocab_size=10,
        d_model=128
    )
    
    # Test with specific shapes from the error
    test_inputs = {
        'command': tf.constant([[1, 2, 3, 4, 5] * 10]),  # shape (1, 50)
        'target': tf.constant([[1, 2, 3, 4, 5, 6] * 16 + [0, 0, 0]]),  # shape (1, 99)
        'modification': tf.constant([[7, 8, 9] * 6 + [0, 0]])  # shape (1, 20)
    }
    
    print(f"Input shapes:")
    print(f"  command: {test_inputs['command'].shape}")
    print(f"  target: {test_inputs['target'].shape}")
    print(f"  modification: {test_inputs['modification'].shape}")
    
    output = model(test_inputs, training=True)
    print(f"\n✓ Model v2 forward pass successful!")
    print(f"  Output shape: {output.shape}")
    
except Exception as e:
    print(f"\n✗ Model v2 error: {e}")
    import traceback
    traceback.print_exc()
    
    # Try to isolate the problem
    print("\n\nDebugging step by step...")
    
    # Step 1: Rule extraction
    try:
        rule_outputs = model.rule_extractor(test_inputs['command'], training=True)
        print(f"✓ Rule extraction works, embeddings shape: {rule_outputs['embeddings'].shape}")
    except Exception as e:
        print(f"✗ Rule extraction failed: {e}")
    
    # Step 2: Rule modification
    try:
        rule_embeddings = rule_outputs['embeddings']
        modified, gates = model.rule_modifier(
            rule_embeddings, 
            test_inputs['modification'], 
            training=True
        )
        print(f"✓ Rule modification works, modified shape: {modified.shape}")
    except Exception as e:
        print(f"✗ Rule modification failed: {e}")
        traceback.print_exc()
    
    # Step 3: Sequence generation
    try:
        logits = model.sequence_generator(
            rule_embeddings, 
            test_inputs['target'], 
            training=True
        )
        print(f"✓ Sequence generation works, logits shape: {logits.shape}")
    except Exception as e:
        print(f"✗ Sequence generation failed: {e}")