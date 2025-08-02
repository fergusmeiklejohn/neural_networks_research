#!/usr/bin/env python3
"""Enhanced compositional training with better handling of bindings and execution.

Key improvement: Separate binding processing from compositional execution.
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
from typing import Dict, List, Any, Tuple
from train_integrated_model import IntegratedBindingModel, VOCAB, ACTIONS
from compositional_operators import CompositionalParser, CompositionalExecutor, ParseNode
from mlx_model_io import save_model_simple

config = setup_environment()

# CRITICAL FIX: Add all required operators to VOCAB FIRST
REQUIRED_OPERATORS = ['and', 'then', 'while', 'or', 'do', 'true']
print(f"Initial VOCAB size: {len(VOCAB)}")
for op in REQUIRED_OPERATORS:
    if op not in VOCAB:
        VOCAB[op] = len(VOCAB)
        print(f"Added '{op}' to VOCAB with ID {VOCAB[op]}")
print(f"VOCAB size after adding operators: {len(VOCAB)}\n")


class EnhancedCompositionalModel(IntegratedBindingModel):
    """Enhanced model that separates binding and execution phases."""
    
    def __init__(self, vocab_size: int, num_actions: int, embed_dim: int = 256,
                 num_slots: int = 4, num_heads: int = 8, mlp_hidden_dim: int = 512):
        super().__init__(vocab_size, num_actions, embed_dim, num_slots, num_heads, mlp_hidden_dim)
        
        # Add compositional components
        self.compositional_parser = CompositionalParser(VOCAB)
        self.compositional_executor = CompositionalExecutor(self, VOCAB)
    
    def separate_bindings_and_execution(self, command_ids: mx.array) -> Tuple[List[Tuple[int, int]], mx.array]:
        """Separate binding definitions from execution commands."""
        # Convert to numpy for easier manipulation
        if hasattr(command_ids, 'numpy'):
            tokens = command_ids.numpy()
        else:
            tokens = command_ids
            
        if len(tokens.shape) > 1:
            tokens = tokens[0]
        
        # Find where "do" commands start
        do_token = VOCAB.get('do', -1)
        binding_segments = []
        exec_start = -1
        
        # Process tokens to find bindings
        i = 0
        while i < len(tokens):
            if tokens[i] == do_token:
                exec_start = i
                break
            
            # Look for "X means Y" patterns
            if i + 2 < len(tokens):
                token_val = int(tokens[i])
                next_token = int(tokens[i + 1]) if i + 1 < len(tokens) else -1
                means_token = VOCAB.get('means', -1)
                
                if next_token == means_token:
                    # Found a binding
                    binding_start = i
                    binding_end = i + 3  # X means Y
                    
                    # Extend if there are more tokens before next variable or "do"
                    while binding_end < len(tokens) and tokens[binding_end] != do_token:
                        next_val = int(tokens[binding_end])
                        # Check if it's a new variable (uppercase letter token)
                        is_new_var = False
                        for word, wid in VOCAB.items():
                            if wid == next_val and word.isupper() and len(word) == 1:
                                is_new_var = True
                                break
                        if is_new_var:
                            break
                        binding_end += 1
                    
                    binding_segments.append((binding_start, binding_end))
                    i = binding_end
                    continue
            
            i += 1
        
        # Extract execution commands (everything after first "do")
        if exec_start >= 0:
            exec_tokens = mx.array(tokens[exec_start:])
        else:
            exec_tokens = mx.array([])
        
        return binding_segments, exec_tokens
    
    def __call__(self, inputs: Dict[str, mx.array], stage: str = "full") -> mx.array:
        """Forward pass with enhanced compositional support."""
        command_ids = inputs['command']
        
        # Clear versioned memory for new sequence
        self.versioned_memory.clear()
        
        # Separate bindings from execution
        binding_segments, exec_tokens = self.separate_bindings_and_execution(command_ids)
        
        # Process bindings first
        bindings = {}
        for start, end in binding_segments:
            segment = (start, end)
            # Use parent class method to process binding
            _ = self.process_segment_versioned(command_ids, segment, bindings, stage="binding")
        
        # Parse execution commands
        if len(exec_tokens) > 0:
            parse_tree = self.compositional_parser.parse(exec_tokens)
            
            # Execute using compositional executor
            outputs = self.compositional_executor.execute(
                parse_tree, exec_tokens, bindings, stage
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
        
        return mx.zeros((1, self.num_actions))


def generate_enhanced_compositional_data(num_samples: int = 100) -> List[Dict[str, Any]]:
    """Generate training data with clear separation of bindings and execution."""
    data = []
    
    # Pattern 1: "and" operator with clear structure
    for _ in range(num_samples // 5):
        var1 = np.random.choice(['X', 'Y'])
        var2 = 'Y' if var1 == 'X' else 'Z'
        action1 = np.random.choice(['WALK', 'JUMP', 'TURN'])
        action2 = np.random.choice(['WALK', 'JUMP', 'TURN'])
        
        # Clear separation: bindings first, then execution
        command = f"{var1} means {action1.lower()} {var2} means {action2.lower()} do {var1} and {var2}"
        tokens = [VOCAB.get(word, VOCAB['<PAD>']) for word in command.split()]
        
        data.append({
            'command': mx.array([tokens]),
            'labels': mx.array([ACTIONS[action1], ACTIONS[action2]]),
            'stage': 'full'
        })
    
    # Pattern 2: "then" operator
    for _ in range(num_samples // 5):
        var1 = 'X'
        var2 = 'Y'
        action1 = np.random.choice(['WALK', 'JUMP', 'TURN'])
        action2 = np.random.choice(['WALK', 'JUMP', 'TURN'])
        
        command = f"{var1} means {action1.lower()} {var2} means {action2.lower()} do {var1} then do {var2}"
        tokens = [VOCAB.get(word, VOCAB['<PAD>']) for word in command.split()]
        
        data.append({
            'command': mx.array([tokens]),
            'labels': mx.array([ACTIONS[action1], ACTIONS[action2]]),
            'stage': 'full'
        })
    
    # Pattern 3: Combined operators with rebinding
    for _ in range(num_samples // 5):
        var = 'X'
        action1 = np.random.choice(['WALK', 'JUMP', 'TURN'])
        action2 = np.random.choice(['WALK', 'JUMP', 'TURN'])
        
        # X means walk, do X, then rebind X and do it again
        command = f"{var} means {action1.lower()} do {var} then {var} means {action2.lower()} do {var}"
        tokens = [VOCAB.get(word, VOCAB['<PAD>']) for word in command.split()]
        
        data.append({
            'command': mx.array([tokens]),
            'labels': mx.array([ACTIONS[action1], ACTIONS[action2]]),
            'stage': 'full'
        })
    
    # Pattern 4: Three-way operations
    for _ in range(num_samples // 5):
        var1, var2, var3 = 'X', 'Y', 'Z'
        action1 = np.random.choice(['WALK', 'JUMP', 'TURN'])
        action2 = np.random.choice(['WALK', 'JUMP', 'TURN'])
        action3 = np.random.choice(['WALK', 'JUMP', 'TURN'])
        
        command = f"{var1} means {action1.lower()} {var2} means {action2.lower()} {var3} means {action3.lower()} do {var1} and {var2} and {var3}"
        tokens = [VOCAB.get(word, VOCAB['<PAD>']) for word in command.split()]
        
        data.append({
            'command': mx.array([tokens]),
            'labels': mx.array([ACTIONS[action1], ACTIONS[action2], ACTIONS[action3]]),
            'stage': 'full'
        })
    
    # Pattern 5: Basic patterns for baseline
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


def train_enhanced_compositional():
    """Train the enhanced compositional model."""
    print("=== Training Enhanced Compositional Model ===")
    print("Key improvement: Separate binding and execution phases\n")
    
    # Create model
    model = EnhancedCompositionalModel(
        vocab_size=len(VOCAB),
        num_actions=len(ACTIONS),
        embed_dim=128,
        num_slots=4
    )
    
    optimizer = optim.Adam(learning_rate=0.001)
    
    # Generate training data
    print("Generating enhanced training data...")
    train_data = generate_enhanced_compositional_data(500)
    
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
            save_path = os.path.join(get_output_path(), 'compositional_enhanced_best.pkl')
            save_model_simple(save_path, model)
            print(f"Saved best model with accuracy {best_accuracy:.2%}")
    
    print(f"\nTraining complete! Best accuracy: {best_accuracy:.2%}")
    return model


def test_enhanced_model(model: EnhancedCompositionalModel):
    """Test the enhanced compositional model."""
    print("\n=== Testing Enhanced Compositional Model ===")
    
    test_cases = [
        # Basic compositional
        ("X means jump Y means walk do X and Y", ['JUMP', 'WALK']),
        ("X means jump Y means walk do X then do Y", ['JUMP', 'WALK']),
        
        # Complex compositional
        ("X means walk Y means jump do X and Y then do X", ['WALK', 'JUMP', 'WALK']),
        ("X means walk Y means jump Z means turn do X and Y and Z", ['WALK', 'JUMP', 'TURN']),
        
        # With rebinding
        ("X means jump do X then X means walk do X", ['JUMP', 'WALK']),
        
        # Basic cases
        ("X means jump do X", ['JUMP']),
        ("X means walk do X twice", ['WALK', 'WALK']),
    ]
    
    correct_count = 0
    total_count = 0
    
    for command, expected in test_cases:
        print(f"\nCommand: {command}")
        print(f"Expected: {expected}")
        
        tokens = [VOCAB.get(word, VOCAB['<PAD>']) for word in command.split()]
        inputs = {'command': mx.array([tokens])}
        
        # Debug: show binding/execution separation
        binding_segs, exec_tokens = model.separate_bindings_and_execution(mx.array([tokens]))
        print(f"Binding segments: {binding_segs}")
        exec_words = []
        for tid in exec_tokens.tolist():
            for word, wid in VOCAB.items():
                if wid == tid:
                    exec_words.append(word)
                    break
        print(f"Execution: {' '.join(exec_words)}")
        
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
        
        # Check correctness
        correct = predicted_actions == expected
        print(f"Correct: {correct}")
        
        if correct:
            correct_count += 1
        total_count += 1
    
    accuracy = correct_count / total_count * 100
    print(f"\n=== Test Summary ===")
    print(f"Accuracy: {correct_count}/{total_count} = {accuracy:.1f}%")
    return accuracy


if __name__ == "__main__":
    # Test binding/execution separation
    print("=== Testing Binding/Execution Separation ===")
    model = EnhancedCompositionalModel(len(VOCAB), len(ACTIONS))
    
    test_cmd = "X means jump Y means walk do X and Y"
    tokens = [VOCAB.get(word, VOCAB['<PAD>']) for word in test_cmd.split()]
    binding_segs, exec_tokens = model.separate_bindings_and_execution(mx.array([tokens]))
    
    print(f"Command: {test_cmd}")
    print(f"Binding segments: {binding_segs}")
    print(f"Execution tokens: {exec_tokens}")
    
    print("\n" + "="*50 + "\n")
    
    # Train model
    model = train_enhanced_compositional()
    
    # Test model
    test_enhanced_model(model)