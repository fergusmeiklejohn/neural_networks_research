#!/usr/bin/env python3
"""
Quick test to see what the model is generating.
"""

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
from pathlib import Path

# Import utilities
from train_progressive_curriculum import SCANTokenizer
from evaluate_simple_baseline import create_generate_function


def test_generation(model_dir):
    """Test what the model generates for simple examples."""
    
    # Load model
    model_path = Path(model_dir) / 'model.h5'
    model = keras.models.load_model(str(model_path))
    
    # Setup tokenizer
    base_dir = Path(__file__).parent
    vocab_path = base_dir / 'compositional_language_complete_20250722_185804' / 'outputs' / 'safeguarded_training' / 'vocabulary.json'
    
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
    
    tokenizer = SCANTokenizer(None)
    tokenizer.command_to_id = vocab_data['command_to_id']
    tokenizer.action_to_id = vocab_data['action_to_id']
    tokenizer.id_to_command = {v: k for k, v in tokenizer.command_to_id.items()}
    tokenizer.id_to_action = {v: k for k, v in tokenizer.action_to_id.items()}
    
    # Test examples
    test_commands = [
        "walk",
        "run",
        "jump",
        "walk twice",
        "run left",
        "jump right"
    ]
    
    print("Testing model generation:")
    print("=" * 60)
    
    # Create generate function
    generate_action = create_generate_function(model, tokenizer)
    
    for command in test_commands:
        # Tokenize command
        command_ids = tokenizer.encode_command(command)
        
        # Generate without modification
        print(f"\nCommand: {command}")
        print(f"Command IDs: {command_ids[:10]}...")  # Show first 10
        
        # Try direct model prediction first
        try:
            # Prepare inputs for model
            batch_command = np.expand_dims(command_ids, axis=0)
            batch_action = np.array([[tokenizer.action_to_id['<START>']]])
            batch_mod = np.zeros((1, 8), dtype=np.float32)
            
            # Get one-step prediction
            logits = model.predict([batch_command, batch_action, batch_mod], verbose=0)
            next_token = np.argmax(logits[0, 0, :])
            next_word = tokenizer.id_to_action.get(next_token, f"<UNK:{next_token}>")
            print(f"First predicted token: {next_word} (id: {next_token})")
            
        except Exception as e:
            print(f"Direct prediction error: {e}")
        
        # Try full generation
        try:
            generated_ids = generate_action(command_ids)
            
            # Decode
            generated_tokens = []
            for token_id in generated_ids:
                if token_id == tokenizer.action_to_id.get('<END>', -1):
                    break
                if token_id == tokenizer.action_to_id.get('<PAD>', 0):
                    continue
                token = tokenizer.id_to_action.get(int(token_id), f"<UNK:{token_id}>")
                generated_tokens.append(token)
            
            generated_action = " ".join(generated_tokens)
            print(f"Generated: {generated_action}")
            
        except Exception as e:
            print(f"Generation error: {e}")
    
    # Also check vocabulary
    print("\n" + "=" * 60)
    print("Vocabulary check:")
    print(f"Command vocab size: {len(tokenizer.command_to_id)}")
    print(f"Action vocab size: {len(tokenizer.action_to_id)}")
    print(f"Sample commands: {list(tokenizer.command_to_id.keys())[:10]}")
    print(f"Sample actions: {list(tokenizer.action_to_id.keys())[:10]}")


if __name__ == "__main__":
    import sys
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs/simple_baseline_v2_20250725_082724"
    test_generation(model_dir)