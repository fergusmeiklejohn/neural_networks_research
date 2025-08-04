"""
Test basic sequential patterns without temporal complexity.
"""

from utils.imports import setup_project_paths
setup_project_paths()

from utils.config import setup_environment
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from train_sequential_planning import SequentialDynamicMemoryModel, VOCAB, ACTIONS

config = setup_environment()

# Ensure 'then' is in vocabulary
if 'then' not in VOCAB:
    VOCAB['then'] = len(VOCAB)

def create_simple_sequential_data(num_samples=20):
    """Create very simple sequential data for testing."""
    commands = []
    expected_actions = []
    
    # Basic patterns
    patterns = [
        ("X means jump do X", ["JUMP"]),
        ("Y means walk do Y", ["WALK"]),
        ("Z means turn do Z", ["TURN"]),
        ("X means jump do X then Y means walk do Y", ["JUMP", "WALK"]),
        ("Y means walk do Y then Z means turn do Z", ["WALK", "TURN"]),
        ("Z means turn do Z then X means jump do X", ["TURN", "JUMP"]),
    ]
    
    # Repeat patterns to create dataset
    for _ in range(num_samples // len(patterns)):
        for cmd, actions in patterns:
            commands.append(cmd)
            expected_actions.append(actions)
    
    return commands, expected_actions


def train_simple_model():
    """Train model on simple sequential patterns."""
    
    # Create data
    commands, expected_actions = create_simple_sequential_data(30)
    
    # Initialize model
    model = SequentialDynamicMemoryModel(
        vocab_size=len(VOCAB),
        num_actions=len(ACTIONS),
        embed_dim=64,  # Small model
        num_slots=4,
        num_heads=2,
        mlp_hidden_dim=128
    )
    
    # Initialize optimizer
    optimizer = optim.Adam(learning_rate=0.01)
    
    print("Training on simple sequential patterns...")
    
    # Training loop
    for epoch in range(20):
        total_loss = 0.0
        
        for cmd, expected in zip(commands, expected_actions):
            # Tokenize
            tokens = cmd.split()
            cmd_ids = mx.array([[VOCAB.get(token, VOCAB['<PAD>']) for token in tokens]])
            act_ids = mx.array([[ACTIONS.get(act, ACTIONS['<PAD>']) for act in expected]])
            
            # Forward pass
            def loss_fn(model):
                outputs = model(cmd_ids, stage="full_binding")
                if 'action_logits' in outputs:
                    logits = outputs['action_logits'][:, :len(expected), :]
                    targets = act_ids[:, :len(expected)]
                    return nn.losses.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        targets.reshape(-1),
                        reduction='mean'
                    )
                return mx.array(0.0)
            
            # Compute gradients and update
            loss_and_grad = mx.value_and_grad(loss_fn)
            loss, grads = loss_and_grad(model)
            optimizer.update(model, grads)
            
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/20 - Avg Loss: {total_loss/len(commands):.4f}")
    
    # Test the model
    print("\nTesting trained model:")
    print("=" * 50)
    
    test_patterns = [
        ("X means jump do X", ["JUMP"]),
        ("X means jump do X then Y means walk do Y", ["JUMP", "WALK"]),
        ("Y means walk do Y then X means jump do X", ["WALK", "JUMP"]),
    ]
    
    for pattern, expected in test_patterns:
        tokens = pattern.split()
        cmd_ids = mx.array([[VOCAB.get(token, VOCAB['<PAD>']) for token in tokens]])
        
        outputs = model(cmd_ids, stage="full_binding")
        if 'action_logits' in outputs:
            predictions = mx.argmax(outputs['action_logits'], axis=-1)
            pred_np = np.array(predictions)
            
            predicted_actions = []
            for j in range(min(pred_np.shape[1], len(expected))):
                pred_id = pred_np[0, j].item() if hasattr(pred_np[0, j], 'item') else int(pred_np[0, j])
                for action_name, action_id in ACTIONS.items():
                    if action_id == pred_id and action_name != '<PAD>':
                        predicted_actions.append(action_name)
                        break
            
            print(f"\nPattern: {pattern}")
            print(f"Expected: {expected}")
            print(f"Predicted: {predicted_actions}")
            print(f"Correct: {predicted_actions == expected}")
            
            # Also show segments detected
            if 'segments' in outputs:
                print(f"Segments: {outputs['segments']}")


if __name__ == "__main__":
    train_simple_model()