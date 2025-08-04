"""
Minimal test of sequential model functionality.
"""

from utils.imports import setup_project_paths
setup_project_paths()

from utils.config import setup_environment
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from train_sequential_planning import SequentialDynamicMemoryModel, VOCAB, ACTIONS

config = setup_environment()

# Ensure 'then' is in vocabulary
if 'then' not in VOCAB:
    VOCAB['then'] = len(VOCAB)

# Initialize model
print("Initializing model...")
model = SequentialDynamicMemoryModel(
    vocab_size=len(VOCAB),
    num_actions=len(ACTIONS),
    embed_dim=128,
    num_slots=4,
    num_heads=4,
    mlp_hidden_dim=256
)

# Test patterns
test_patterns = [
    ("X means jump do X", ["JUMP"]),
    ("X means jump do X then Y means walk do Y", ["JUMP", "WALK"]),
    ("Z means turn do Z twice", ["TURN", "TURN"]),
    ("Z means turn do Z twice then X means run do X", ["TURN", "TURN", "RUN"]),
]

print("\nTesting Sequential Model:")
print("=" * 50)

for pattern, expected in test_patterns:
    print(f"\nPattern: {pattern}")
    print(f"Expected: {expected}")
    
    # Tokenize
    tokens = pattern.split()
    token_ids = [VOCAB.get(token, VOCAB['<PAD>']) for token in tokens]
    command_ids = mx.array([token_ids])  # Add batch dimension
    
    # Forward pass
    try:
        # MLX doesn't need no_grad context
        results = model(command_ids, stage="full_binding")
            
        print(f"Segments detected: {results.get('segments', [])}")
        print(f"Number of temporal actions: {results.get('temporal_actions', 0)}")
        
        # Get predictions
        if 'action_logits' in results:
            action_logits = results['action_logits']
            predictions = mx.argmax(action_logits, axis=-1)
            
            # Extract predicted actions
            predicted_actions = []
            for j in range(predictions.shape[1]):
                pred_id = predictions[0, j].item()
                for action_name, action_id in ACTIONS.items():
                    if action_id == pred_id and action_name != '<PAD>':
                        predicted_actions.append(action_name)
                        break
            
            print(f"Predicted (raw): {predicted_actions[:len(expected)+2]}")  # Show a few extra
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 50)
print("Model test complete!")