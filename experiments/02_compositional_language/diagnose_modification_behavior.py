#!/usr/bin/env python3
"""
Diagnostic script to check if models are actually applying modifications.

This script will:
1. Load trained models from Paperspace results
2. Generate test examples with specific modifications
3. Compare predictions with and without modifications
4. Analyze if modifications are being applied
"""

import os
import sys
sys.path.append(os.path.abspath('../..'))

import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json
import pickle

from scan_data_loader import SCANDataLoader, SCANSample
from models_v2 import create_model_v2

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_v1_model(checkpoint_path, cmd_vocab, act_vocab):
    """Load v1 model from checkpoint."""
    # Import v1 model creation
    from models import create_model
    
    model = create_model(cmd_vocab, act_vocab, d_model=256)
    
    # Build model
    dummy_inputs = {
        'command': tf.constant([[1, 2, 3, 4, 5]]),
        'target': tf.constant([[1, 2, 3, 4, 5, 6]]),
        'modification': tf.constant([[1, 2, 3]])
    }
    _ = model(dummy_inputs, training=False)
    
    # Load weights
    model.load_weights(checkpoint_path)
    return model

def load_v2_model(checkpoint_path, cmd_vocab, act_vocab):
    """Load v2 model from checkpoint."""
    model = create_model_v2(cmd_vocab, act_vocab, d_model=256)
    
    # Build model
    dummy_inputs = {
        'command': tf.constant([[1, 2, 3, 4, 5]]),
        'target': tf.constant([[1, 2, 3, 4, 5, 6]]),
        'modification': tf.constant([[1, 2, 3]])
    }
    _ = model(dummy_inputs, training=False)
    
    # Load weights
    model.load_weights(checkpoint_path)
    return model

def tokenize_command(command, cmd_vocab):
    """Simple tokenizer for commands."""
    tokens = []
    for word in command.split():
        tokens.append(cmd_vocab.get(word, cmd_vocab['<UNK>']))
    return tokens

def tokenize_actions(actions, act_vocab):
    """Simple tokenizer for actions."""
    tokens = []
    for action in actions.split():
        tokens.append(act_vocab.get(action, act_vocab['<UNK>']))
    return tokens

def generate_test_cases_simple(cmd_vocab, act_vocab, num_examples=10):
    """Generate simple test cases without using SCANDataGenerator."""
    test_cases = []
    
    # Basic test cases with known transformations
    base_examples = [
        ("walk", "I_WALK"),
        ("walk twice", "I_WALK I_WALK"),
        ("look", "I_LOOK"),
        ("turn left", "I_TURN_LEFT"),
        ("jump", "I_JUMP")
    ]
    
    # Modifications to test
    modifications = [
        ("walk", "skip", "walk → skip"),
        ("look", "scan", "look → scan"),
        ("jump", "hop", "jump → hop")
    ]
    
    for i in range(min(num_examples, len(base_examples))):
        base_cmd, base_act = base_examples[i]
        
        # Always create a test case with or without modification
        if i < len(modifications):
            orig_word, new_word, mod_desc = modifications[i % len(modifications)]
            if orig_word in base_cmd:
                # Create modified version
                mod_cmd = base_cmd.replace(orig_word, new_word)
                mod_act = base_act.replace(f"I_{orig_word.upper()}", f"I_{new_word.upper()}")
            else:
                # Use a default modification if word not in command
                mod_cmd = base_cmd + " twice"
                mod_act = base_act + " " + base_act
                mod_desc = "double"
        else:
            # No modification for this test case
            mod_cmd = base_cmd
            mod_act = base_act
            mod_desc = "none"
        
        test_cases.append({
            'base_command': base_cmd,
            'base_tokens': tokenize_command(base_cmd, cmd_vocab),
            'base_actions': base_act,
            'base_action_tokens': tokenize_actions(base_act, act_vocab),
            'modification': mod_desc,
            'mod_tokens': [1, 0, 0] if mod_desc != "none" else [0, 0, 0],
            'modified_command': mod_cmd,
            'expected_actions': mod_act,
            'expected_tokens': tokenize_actions(mod_act, act_vocab)
        })
    
    return test_cases

def generate_test_cases(data_gen, num_examples=10):
    """Generate test cases with known modifications."""
    test_cases = []
    
    # Get some base examples
    cmd_train, act_train = data_gen.cmd_train[:num_examples], data_gen.act_train[:num_examples]
    
    # Create modifications
    modifications = [
        ("reverse", "Execute commands in reverse order"),
        ("double", "Execute each command twice"),
        ("skip", "Skip every other command")
    ]
    
    for i in range(min(num_examples, len(cmd_train))):
        cmd = cmd_train[i]
        act = act_train[i]
        
        # Apply each modification
        for mod_type, mod_desc in modifications:
            # Create modified action sequence
            if mod_type == "reverse":
                # Reverse the action sequence (excluding padding)
                act_tokens = [t for t in act if t != 0]  # Remove padding
                modified_act = act_tokens[::-1]
            elif mod_type == "double":
                # Double each action
                act_tokens = [t for t in act if t != 0]
                modified_act = []
                for token in act_tokens:
                    modified_act.extend([token, token])
            elif mod_type == "skip":
                # Skip every other action
                act_tokens = [t for t in act if t != 0]
                modified_act = act_tokens[::2]
            
            # Pad modified action
            modified_act = modified_act + [0] * (len(act) - len(modified_act))
            
            test_cases.append({
                'command': cmd,
                'original_action': act,
                'modified_action': np.array(modified_act),
                'modification_type': mod_type,
                'modification_desc': mod_desc
            })
    
    return test_cases

def predict_with_model(model, command, modification=None):
    """Get model prediction for a command with optional modification."""
    # Prepare inputs
    cmd_input = tf.constant([command])
    
    # Create dummy target for teacher forcing
    dummy_target = tf.constant([[1] * 50])  # Max length
    
    if modification is None:
        # No modification signal
        mod_input = tf.constant([[0, 0, 0]])  # Empty modification
    else:
        # Encode modification (simplified for testing)
        mod_map = {"reverse": [1, 0, 0], "double": [0, 1, 0], "skip": [0, 0, 1]}
        mod_input = tf.constant([mod_map.get(modification, [0, 0, 0])])
    
    # Get prediction
    inputs = {
        'command': cmd_input,
        'target': dummy_target,
        'modification': mod_input
    }
    
    # For inference, we need to generate autoregressively
    # This is a simplified version - in practice would need proper decoding
    outputs = model(inputs, training=False)
    
    # Get predicted tokens (argmax)
    predicted = tf.argmax(outputs, axis=-1).numpy()[0]
    
    return predicted

def analyze_modification_behavior(model, test_cases, model_name):
    """Analyze how model responds to modifications."""
    print(f"\n{'='*60}")
    print(f"Analyzing {model_name}")
    print(f"{'='*60}")
    
    modification_applied = 0
    modification_ignored = 0
    
    for i, test_case in enumerate(test_cases[:5]):  # Analyze first 5 cases
        cmd = test_case['command']
        original = test_case['original_action']
        expected_modified = test_case['modified_action']
        mod_type = test_case['modification_type']
        
        # Predict without modification
        pred_no_mod = predict_with_model(model, cmd, modification=None)
        
        # Predict with modification
        pred_with_mod = predict_with_model(model, cmd, modification=mod_type)
        
        # Compare predictions
        no_mod_tokens = [t for t in pred_no_mod if t > 0][:10]  # First 10 non-padding
        with_mod_tokens = [t for t in pred_with_mod if t > 0][:10]
        
        # Check if modification changed the output
        if np.array_equal(no_mod_tokens, with_mod_tokens):
            modification_ignored += 1
            status = "IGNORED"
        else:
            modification_applied += 1
            status = "APPLIED"
        
        print(f"\nExample {i+1} - Modification: {mod_type} - Status: {status}")
        print(f"  No mod:   {no_mod_tokens}")
        print(f"  With mod: {with_mod_tokens}")
        print(f"  Expected: {[t for t in expected_modified if t > 0][:10]}")
    
    print(f"\nSummary for {model_name}:")
    print(f"  Modifications applied: {modification_applied}/{len(test_cases[:5])}")
    print(f"  Modifications ignored: {modification_ignored}/{len(test_cases[:5])}")
    
    return {
        'applied': modification_applied,
        'ignored': modification_ignored,
        'total': len(test_cases[:5])
    }

def main():
    """Run diagnostic analysis on all trained models."""
    # Initialize data loader and load vocabulary
    print("Loading vocabulary...")
    vocab_path = Path("comprehensive_results_20250723_060233/vocabulary.json")
    if not vocab_path.exists():
        vocab_path = Path("compositional_language_complete_20250722_185804/outputs/safeguarded_training/vocabulary.json")
    
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
    
    cmd_vocab = vocab_data['command_to_id']
    act_vocab = vocab_data['action_to_id']
    
    # Generate test cases
    print("Generating test cases with known modifications...")
    test_cases = generate_test_cases_simple(cmd_vocab, act_vocab, num_examples=10)
    print(f"Generated {len(test_cases)} test cases")
    
    # Define model paths - using local safeguarded training results
    local_results_dir = Path("compositional_language_complete_20250722_185804/outputs/safeguarded_training")
    
    models_to_test = [
        ("v1_standard_local", local_results_dir / "final_model.h5", "v1"),
        # We only have v1 standard from local training
        # Paperspace results didn't save model checkpoints
    ]
    
    print("\nNOTE: Paperspace results only saved metrics, not model files.")
    print("Using local v1 standard model for diagnosis.")
    
    # Check which models exist
    print("\nChecking for model files...")
    for name, path, version in models_to_test:
        if path.exists():
            print(f"  ✓ Found {name} at {path}")
        else:
            print(f"  ✗ Missing {name} (expected at {path})")
    
    # Analyze each model
    results = {}
    for name, path, version in models_to_test:
        if not path.exists():
            print(f"\nSkipping {name} - file not found")
            continue
        
        try:
            # Load model
            if version == "v1":
                model = load_v1_model(str(path), len(cmd_vocab), len(act_vocab))
            else:
                model = load_v2_model(str(path), len(cmd_vocab), len(act_vocab))
            
            # Analyze behavior
            result = analyze_modification_behavior(model, test_cases, name)
            results[name] = result
            
        except Exception as e:
            print(f"\nError analyzing {name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*60)
    print("OVERALL SUMMARY")
    print("="*60)
    
    for name, result in results.items():
        apply_rate = result['applied'] / result['total'] * 100
        print(f"{name:15} - Modification apply rate: {apply_rate:.1f}%")
    
    print("\nConclusion:")
    if all(r['applied'] == 0 for r in results.values()):
        print("⚠️  ALL models are ignoring modifications!")
        print("This explains the constant validation accuracy.")
        print("Models have learned to ignore the modification signal entirely.")
    elif any(r['applied'] > r['total'] * 0.5 for r in results.values()):
        print("✓ Some models are applying modifications")
        print("Further investigation needed on modification quality.")
    else:
        print("⚠️  Models show limited modification application")
        print("May need stronger modification signals or different architecture.")

if __name__ == "__main__":
    main()