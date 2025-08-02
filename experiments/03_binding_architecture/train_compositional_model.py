#!/usr/bin/env python3
"""Training script for the integrated model with compositional operators.

This extends the integrated model to support:
- "and" operator for parallel actions
- "or" operator for choice between actions
- "while" operator for repeated execution
"""

from utils.imports import setup_project_paths
setup_project_paths()

from utils.config import setup_environment
from utils.paths import get_output_path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import os
from typing import Dict, List, Any
from train_integrated_model import IntegratedBindingModel, VOCAB, ACTIONS
from compositional_operators import CompositionalParser, CompositionalExecutor, ParseNode
from mlx_model_io import save_model_simple

config = setup_environment()


class CompositionalBindingModel(IntegratedBindingModel):
    """Extended binding model with compositional operator support."""
    
    def __init__(self, vocab_size: int, num_actions: int, embed_dim: int = 256,
                 num_slots: int = 4, num_heads: int = 8, mlp_hidden_dim: int = 512):
        super().__init__(vocab_size, num_actions, embed_dim, num_slots, num_heads, mlp_hidden_dim)
        
        # Add compositional components
        self.compositional_parser = CompositionalParser(VOCAB)
        self.compositional_executor = CompositionalExecutor(self, VOCAB)
    
    def __call__(self, inputs: Dict[str, mx.array], stage: str = "full") -> mx.array:
        """Forward pass with compositional operator support."""
        command_ids = inputs['command']
        
        # Clear versioned memory for new sequence
        self.versioned_memory.clear()
        
        # Parse command for compositional structure
        parse_tree = self.compositional_parser.parse(command_ids)
        
        # Execute using compositional executor
        bindings = {}
        outputs = self.compositional_executor.execute(
            parse_tree, command_ids, bindings, stage
        )
        
        # Stack outputs
        if outputs:
            squeezed_outputs = []
            for out in outputs:
                if len(out.shape) > 1:
                    squeezed_outputs.append(out.squeeze())
                else:
                    squeezed_outputs.append(out)
            return mx.stack(squeezed_outputs)
        else:
            return mx.zeros((1, self.num_actions))


def generate_compositional_data(num_samples: int = 100) -> List[Dict[str, Any]]:
    """Generate training data with compositional operators."""
    # Ensure operators are in vocabulary
    for op in ['then', 'and', 'or', 'while']:
        if op not in VOCAB:
            VOCAB[op] = len(VOCAB)
    
    data = []
    
    # Pattern 1: "and" operator
    for _ in range(num_samples // 5):
        var1 = np.random.choice(['X', 'Y'])
        var2 = 'Y' if var1 == 'X' else 'Z'
        action1 = np.random.choice(['WALK', 'JUMP', 'TURN'])
        action2 = np.random.choice(['WALK', 'JUMP', 'TURN'])
        
        command = f"{var1} means {action1.lower()} {var2} means {action2.lower()} do {var1} and {var2}"
        tokens = [VOCAB.get(word, VOCAB['<PAD>']) for word in command.split()]
        
        # Both actions should be executed
        data.append({
            'command': mx.array([tokens]),
            'labels': mx.array([ACTIONS[action1], ACTIONS[action2]]),
            'stage': 'full'
        })
    
    # Pattern 2: "or" operator (randomly choose one)
    for _ in range(num_samples // 5):
        var1 = 'X'
        var2 = 'Y'
        action1 = np.random.choice(['WALK', 'JUMP', 'TURN'])
        action2 = np.random.choice(['WALK', 'JUMP', 'TURN'])
        
        command = f"{var1} means {action1.lower()} {var2} means {action2.lower()} do {var1} or {var2}"
        tokens = [VOCAB.get(word, VOCAB['<PAD>']) for word in command.split()]
        
        # One action chosen randomly
        chosen_action = action1 if np.random.random() > 0.5 else action2
        data.append({
            'command': mx.array([tokens]),
            'labels': mx.array([ACTIONS[chosen_action]]),
            'stage': 'full'
        })
    
    # Pattern 3: "while" operator (simplified - repeat 3 times)
    for _ in range(num_samples // 5):
        var = np.random.choice(['X', 'Y', 'Z'])
        action = np.random.choice(['WALK', 'JUMP', 'TURN'])
        
        command = f"{var} means {action.lower()} while true do {var}"
        tokens = [VOCAB.get(word, VOCAB['<PAD>']) for word in command.split()]
        
        # Action repeated 3 times
        data.append({
            'command': mx.array([tokens]),
            'labels': mx.array([ACTIONS[action]] * 3),
            'stage': 'full'
        })
    
    # Pattern 4: Combined operators
    for _ in range(num_samples // 5):
        var1 = 'X'
        var2 = 'Y'
        action1 = np.random.choice(['WALK', 'JUMP', 'TURN'])
        action2 = np.random.choice(['WALK', 'JUMP', 'TURN'])
        
        command = f"{var1} means {action1.lower()} {var2} means {action2.lower()} do {var1} and {var2} then do {var1}"
        tokens = [VOCAB.get(word, VOCAB['<PAD>']) for word in command.split()]
        
        data.append({
            'command': mx.array([tokens]),
            'labels': mx.array([ACTIONS[action1], ACTIONS[action2], ACTIONS[action1]]),
            'stage': 'full'
        })
    
    # Pattern 5: Basic patterns for stability
    for _ in range(num_samples // 5):
        var = np.random.choice(['X', 'Y', 'Z'])
        action = np.random.choice(['WALK', 'JUMP', 'TURN'])
        
        command = f"{var} means {action.lower()} do {var}"
        tokens = [VOCAB.get(word, VOCAB['<PAD>']) for word in command.split()]
        
        data.append({
            'command': mx.array([tokens]),
            'labels': mx.array([ACTIONS[action]]),
            'stage': 'full'
        })
    
    return data


def train_compositional_model():
    """Train model with compositional operators."""
    print("=== Training Compositional Variable Binding Model ===")
    print("Operators: AND (parallel), OR (choice), WHILE (loop), THEN (sequence)\n")
    
    # Ensure operators in vocabulary
    for op in ['then', 'and', 'or', 'while', 'true']:
        if op not in VOCAB:
            VOCAB[op] = len(VOCAB)
    
    # Create model
    model = CompositionalBindingModel(
        vocab_size=len(VOCAB),
        num_actions=len(ACTIONS),
        embed_dim=128,
        num_slots=4
    )
    
    optimizer = optim.Adam(learning_rate=0.001)
    
    # Generate training data
    print("Generating compositional training data...")
    train_data = generate_compositional_data(500)
    
    # Training loop
    num_epochs = 30
    best_accuracy = 0.0
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Shuffle data
        np.random.shuffle(train_data)
        
        for i, batch in enumerate(train_data):
            inputs = {'command': batch['command']}
            
            # Forward pass and loss
            def loss_fn(model):
                outputs = model(inputs, stage=batch['stage'])
                labels = batch['labels']
                
                # Handle different output lengths
                if len(outputs) > len(labels):
                    outputs = outputs[:len(labels)]
                elif len(outputs) < len(labels):
                    padding = mx.zeros((len(labels) - len(outputs), outputs.shape[-1]))
                    outputs = mx.concatenate([outputs, padding])
                
                loss = mx.mean(nn.losses.cross_entropy(outputs, labels))
                
                predictions = mx.argmax(outputs, axis=1)
                accuracy = mx.mean(predictions == labels)
                
                return loss, accuracy
            
            loss_val, accuracy = loss_fn(model)
            
            # Gradient computation
            def loss_only(model):
                outputs = model(inputs, stage=batch['stage'])
                labels = batch['labels']
                
                if len(outputs) > len(labels):
                    outputs = outputs[:len(labels)]
                elif len(outputs) < len(labels):
                    padding = mx.zeros((len(labels) - len(outputs), outputs.shape[-1]))
                    outputs = mx.concatenate([outputs, padding])
                
                return mx.mean(nn.losses.cross_entropy(outputs, labels))
            
            grad_fn = mx.grad(loss_only)
            grads = grad_fn(model)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            
            # Update stats
            total_loss += float(loss_val)
            correct += float(accuracy) * len(batch['labels'])
            total += len(batch['labels'])
            
            if i % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {i}/{len(train_data)}, "
                      f"Loss: {float(loss_val):.4f}, Acc: {float(accuracy):.2%}")
        
        # Epoch summary
        epoch_acc = correct / total
        print(f"\nEpoch {epoch+1}: Avg Loss = {total_loss/len(train_data):.4f}, "
              f"Accuracy = {epoch_acc:.2%}")
        
        if epoch_acc > best_accuracy:
            best_accuracy = epoch_acc
            save_path = os.path.join(get_output_path(), 'compositional_model_best.pkl')
            save_model_simple(save_path, model)
            print(f"Saved best model with accuracy {best_accuracy:.2%}")
    
    print(f"\nTraining complete! Best accuracy: {best_accuracy:.2%}")
    return model


def test_compositional_model(model: CompositionalBindingModel):
    """Test the compositional model."""
    print("\n=== Testing Compositional Model ===")
    
    test_cases = [
        # Basic operators
        ("X means jump Y means walk do X and Y", ['JUMP', 'WALK']),
        ("X means jump Y means walk do X or Y", ['JUMP']),  # Could be either
        ("X means jump while true do X", ['JUMP', 'JUMP', 'JUMP']),
        
        # Combined operators
        ("X means walk Y means jump do X and Y then do X", ['WALK', 'JUMP', 'WALK']),
        ("X means jump do X then Y means walk do Y and X", ['JUMP', 'WALK', 'JUMP']),
        
        # With temporal and rebinding
        ("X means jump do X twice and Y means walk do Y", ['JUMP', 'JUMP', 'WALK']),
        ("X means jump do X then X means walk do X or Y means run do Y", ['JUMP', 'WALK']),  # Should choose X
    ]
    
    for command, expected in test_cases:
        print(f"\nCommand: {command}")
        print(f"Expected: {expected} (or similar)")
        
        tokens = [VOCAB.get(word, VOCAB['<PAD>']) for word in command.split()]
        inputs = {'command': mx.array([tokens])}
        
        # Parse and show structure
        parse_tree = model.compositional_parser.parse(mx.array(tokens))
        print(f"Parse structure: {describe_tree(parse_tree)}")
        
        # Get predictions
        outputs = model(inputs, stage='full')
        predictions = mx.argmax(outputs, axis=1)
        
        predicted_actions = []
        for pred in predictions:
            for name, idx in ACTIONS.items():
                if idx == int(pred):
                    predicted_actions.append(name)
                    break
        
        print(f"Predicted: {predicted_actions}")
        
        # For OR operator, just check if one of the possible actions was chosen
        if 'or' in command and len(predicted_actions) == 1:
            possible_actions = [a for a in expected if a in ['JUMP', 'WALK', 'RUN']]
            correct = predicted_actions[0] in possible_actions
        else:
            correct = predicted_actions == expected
        
        print(f"Correct: {correct}")


def describe_tree(node: ParseNode) -> str:
    """Get a simple description of the parse tree."""
    if node.is_leaf():
        return "LEAF"
    else:
        child_descs = [describe_tree(c) for c in node.children]
        return f"{node.operator.value}({', '.join(child_descs)})"


if __name__ == "__main__":
    model = train_compositional_model()
    test_compositional_model(model)