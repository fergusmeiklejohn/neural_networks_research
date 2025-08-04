#!/usr/bin/env python3
"""Train compositional model with improved parsing and execution logic."""

from utils.imports import setup_project_paths
setup_project_paths()

from utils.config import setup_environment
config = setup_environment()

import os
import pickle
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from typing import List, Dict, Tuple
import random

from train_integrated_model import (
    VOCAB, ACTIONS,
    generate_stage1_data, generate_stage2_data, generate_stage3_data,
    generate_rebinding_data, IntegratedBindingModel
)
from improved_compositional_operators import (
    ImprovedCompositionalParser, ImprovedCompositionalExecutor,
    ParseNode, OperatorType
)
from utils.paths import get_output_path


class ImprovedCompositionalBindingModel(IntegratedBindingModel):
    """Model with improved compositional operator support."""
    
    def __init__(self, vocab_size: int, num_actions: int, embed_dim: int = 256,
                 num_slots: int = 4, num_heads: int = 8, mlp_hidden_dim: int = 512):
        super().__init__(vocab_size, num_actions, embed_dim, num_slots, num_heads, mlp_hidden_dim)
        
        # Use improved parser and executor
        self.compositional_parser = ImprovedCompositionalParser(VOCAB)
        self.compositional_executor = ImprovedCompositionalExecutor(self, VOCAB)
    
    def __call__(self, inputs: Dict[str, mx.array], stage: str = "full") -> mx.array:
        """Process inputs and return action predictions."""
        command_ids = inputs['command']
        
        # Clear versioned memory at start
        self.versioned_memory.clear()
        
        # Parse command structure
        parse_tree = self.compositional_parser.parse(command_ids)
        
        # Execute parsed tree
        bindings = {}
        outputs = self.compositional_executor.execute(parse_tree, command_ids, bindings, stage)
        
        # Stack outputs into single tensor
        if outputs:
            return mx.stack(outputs)
        else:
            # Return single PAD output if no actions
            return mx.zeros((1, self.num_actions))


def generate_improved_compositional_data(num_samples: int = 1000) -> List[Dict]:
    """Generate training data with improved compositional patterns."""
    data = []
    
    # Basic operators
    operator_patterns = [
        # Sequence operator (then)
        {
            'pattern': "X means {} Y means {} do X then do Y",
            'actions': lambda x, y: [x, y],
            'type': 'sequence'
        },
        {
            'pattern': "X means {} do X then Y means {} do Y",
            'actions': lambda x, y: [x, y],
            'type': 'sequence_rebind'
        },
        {
            'pattern': "X means {} Y means {} Z means {} do X then do Y then do Z",
            'actions': lambda x, y, z: [x, y, z],
            'type': 'sequence_triple'
        },
        
        # Parallel operator (and)
        {
            'pattern': "X means {} Y means {} do X and Y",
            'actions': lambda x, y: [x, y],
            'type': 'parallel'
        },
        {
            'pattern': "X means {} Y means {} Z means {} do X and Y and Z",
            'actions': lambda x, y, z: [x, y, z],
            'type': 'parallel_triple'
        },
        
        # Loop operator (while)
        {
            'pattern': "X means {} while true do X",
            'actions': lambda x: [x, x, x],  # Execute 3 times
            'type': 'loop'
        },
        {
            'pattern': "X means {} Y means {} while Y do X",
            'actions': lambda x, y: [x, x, x],  # Execute X 3 times
            'type': 'loop_condition'
        },
        
        # Choice operator (or)
        {
            'pattern': "X means {} Y means {} do X or Y",
            'actions': lambda x, y: [x],  # Deterministic for training
            'type': 'choice'
        },
        
        # Combined operators
        {
            'pattern': "X means {} Y means {} do X and Y then do X",
            'actions': lambda x, y: [x, y, x],
            'type': 'combined_and_then'
        },
        {
            'pattern': "X means {} Y means {} Z means {} do X then do Y and Z",
            'actions': lambda x, y, z: [x, y, z],
            'type': 'combined_then_and'
        },
        {
            'pattern': "X means {} Y means {} do X twice then do Y",
            'actions': lambda x, y: [x, x, y],
            'type': 'temporal_then'
        },
        {
            'pattern': "X means {} while true do X then Y means {} do Y",
            'actions': lambda x, y: [x, x, x, y],
            'type': 'loop_then'
        },
    ]
    
    # Generate samples
    for _ in range(num_samples):
        pattern_info = random.choice(operator_patterns)
        pattern = pattern_info['pattern']
        
        # Choose random actions
        action_names = list(ACTIONS.keys())
        if pattern.count('{}') == 1:
            action1 = random.choice(action_names)
            command = pattern.format(action1.lower())
            actions = pattern_info['actions'](action1)
        elif pattern.count('{}') == 2:
            action1, action2 = random.sample(action_names, 2)
            command = pattern.format(action1.lower(), action2.lower())
            actions = pattern_info['actions'](action1, action2)
        else:  # 3 actions
            action1, action2, action3 = random.sample(action_names, 3)
            command = pattern.format(action1.lower(), action2.lower(), action3.lower())
            actions = pattern_info['actions'](action1, action2, action3)
        
        # Convert to tokens
        tokens = [VOCAB.get(word, VOCAB['<PAD>']) for word in command.split()]
        
        # Convert actions to indices
        action_indices = [ACTIONS[action] for action in actions]
        
        data.append({
            'command': mx.array([tokens]),
            'labels': mx.array(action_indices),
            'stage': 'full',
            'type': pattern_info['type'],
            'text': command
        })
    
    return data


def evaluate_compositional_accuracy(model: ImprovedCompositionalBindingModel, test_data: List[Dict]) -> Dict[str, float]:
    """Evaluate accuracy by operator type."""
    results = {}
    type_correct = {}
    type_total = {}
    
    for sample in test_data:
        command = sample['command']
        expected = sample['labels']
        op_type = sample.get('type', 'unknown')
        
        # Get prediction
        outputs = model({'command': command}, stage='full')
        predictions = mx.argmax(outputs, axis=1)
        
        # Check correctness
        correct = mx.array_equal(predictions, expected)
        
        # Update stats
        if op_type not in type_correct:
            type_correct[op_type] = 0
            type_total[op_type] = 0
        
        type_total[op_type] += 1
        if correct:
            type_correct[op_type] += 1
    
    # Calculate accuracies
    for op_type in type_total:
        accuracy = type_correct[op_type] / type_total[op_type] if type_total[op_type] > 0 else 0
        results[op_type] = accuracy
    
    # Overall accuracy
    total_correct = sum(type_correct.values())
    total_samples = sum(type_total.values())
    results['overall'] = total_correct / total_samples if total_samples > 0 else 0
    
    return results


def train_improved_compositional_model(num_epochs: int = 10, batch_size: int = 32):
    """Train the improved compositional model."""
    print("=== Training Improved Compositional Model ===")
    
    # Initialize model
    model = ImprovedCompositionalBindingModel(
        vocab_size=len(VOCAB),
        num_actions=len(ACTIONS),
        embed_dim=256,
        num_slots=4,
        num_heads=8,
        mlp_hidden_dim=512
    )
    
    # Initialize optimizer
    optimizer = optim.Adam(learning_rate=1e-3)
    
    # Generate training data
    print("\nGenerating training data...")
    
    # Mix of all data types for comprehensive training
    stage1_data = generate_stage1_data(200)  # Basic recognition
    stage2_data = generate_stage2_data(200)  # Basic retrieval
    stage3_data = generate_stage3_data(200)  # Full binding
    rebinding_data = generate_rebinding_data(200)  # Rebinding
    compositional_data = generate_improved_compositional_data(400)  # Focus on compositional
    
    # Convert batch data to lists
    def batch_to_list(batch_data):
        list_data = []
        for i in range(len(batch_data['command'])):
            item = {}
            for key in batch_data:
                if hasattr(batch_data[key], '__getitem__'):
                    item[key] = batch_data[key][i:i+1]
                else:
                    item[key] = batch_data[key]
            list_data.append(item)
        return list_data
    
    # Convert and ensure consistent format
    all_data = []
    for data_source in [stage1_data, stage2_data, stage3_data]:
        list_data = batch_to_list(data_source)
        for item in list_data:
            if 'target' in item and 'labels' not in item:
                item['labels'] = item['target']
                del item['target']
            # Ensure labels are properly shaped
            if len(item['labels'].shape) > 1:
                item['labels'] = item['labels'].squeeze()
        all_data.extend(list_data)
    
    # Add rebinding and compositional data
    for data_source in [rebinding_data, compositional_data]:
        for item in data_source:
            if 'target' in item and 'labels' not in item:
                item['labels'] = item['target']
                del item['target']
        all_data.extend(data_source)
    
    # Shuffle data
    random.shuffle(all_data)
    
    print(f"Total training samples: {len(all_data)}")
    print(f"- Stage 1 (recognition): {len(stage1_data)}")
    print(f"- Stage 2 (retrieval): {len(stage2_data)}")
    print(f"- Stage 3 (full binding): {len(stage3_data)}")
    print(f"- Rebinding: {len(rebinding_data)}")
    print(f"- Compositional: {len(compositional_data)}")
    
    # Training loop
    best_accuracy = 0
    
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
        # Shuffle data each epoch
        random.shuffle(all_data)
        
        total_loss = 0
        correct = 0
        total = 0
        
        for i, batch in enumerate(all_data):
            inputs = {'command': batch['command']}
            
            # Forward pass
            outputs = model(inputs, stage=batch.get('stage', 'full'))
            labels = batch['labels']
            
            # Ensure outputs and labels are properly shaped
            # Remove batch dimension if present
            if len(outputs.shape) == 3 and outputs.shape[0] == 1:
                outputs = outputs[0]  # (1, seq, vocab) -> (seq, vocab)
            if len(labels.shape) == 2 and labels.shape[0] == 1:
                labels = labels[0]  # (1, seq) -> (seq,)
            
            # Handle different sequence lengths
            if outputs.shape[0] != labels.shape[0]:
                if outputs.shape[0] > labels.shape[0]:
                    outputs = outputs[:labels.shape[0]]
                else:
                    # Pad outputs
                    padding = mx.zeros((labels.shape[0] - outputs.shape[0], outputs.shape[-1]))
                    outputs = mx.concatenate([outputs, padding])
            
            # Compute loss
            loss = mx.mean(nn.losses.cross_entropy(outputs, labels))
            
            # Compute gradients
            def loss_only(m):
                out = m(inputs, stage=batch.get('stage', 'full'))
                lbl = batch['labels']
                
                # Handle shapes
                if len(out.shape) == 3 and out.shape[0] == 1:
                    out = out[0]
                if len(lbl.shape) == 2 and lbl.shape[0] == 1:
                    lbl = lbl[0]
                
                # Handle length mismatch
                if out.shape[0] > lbl.shape[0]:
                    out = out[:lbl.shape[0]]
                elif out.shape[0] < lbl.shape[0]:
                    padding = mx.zeros((lbl.shape[0] - out.shape[0], out.shape[-1]))
                    out = mx.concatenate([out, padding])
                
                return mx.mean(nn.losses.cross_entropy(out, lbl))
            
            grad_fn = mx.grad(loss_only)
            grads = grad_fn(model)
            
            # Update
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            
            # Track metrics
            predictions = mx.argmax(outputs, axis=1)
            batch_correct = mx.sum(predictions == labels)
            
            total_loss += float(loss)
            correct += float(batch_correct)
            total += len(labels)
            
            if i % 100 == 0 and i > 0:
                print(f"  Batch {i}/{len(all_data)}: Loss = {float(loss):.4f}, "
                      f"Running Acc = {correct/total:.2%}")
        
        # Epoch summary
        epoch_acc = correct / total
        avg_loss = total_loss / len(all_data)
        print(f"\nEpoch {epoch+1} Summary: Loss = {avg_loss:.4f}, Accuracy = {epoch_acc:.2%}")
        
        # Save best model
        if epoch_acc > best_accuracy:
            best_accuracy = epoch_acc
            save_path = os.path.join(get_output_path(), 'improved_compositional_best.pkl')
            save_model_simple(save_path, model)
            print(f"Saved best model with accuracy {best_accuracy:.2%}")
    
    print(f"\nTraining complete! Best accuracy: {best_accuracy:.2%}")
    
    # Detailed evaluation
    print("\n=== Detailed Evaluation ===")
    test_data = generate_improved_compositional_data(200)
    accuracies = evaluate_compositional_accuracy(model, test_data)
    
    print("\nAccuracy by operator type:")
    for op_type, acc in sorted(accuracies.items()):
        if op_type != 'overall':
            print(f"  {op_type}: {acc:.2%}")
    print(f"\nOverall accuracy: {accuracies['overall']:.2%}")
    
    return model


def save_model_simple(path: str, model):
    """Save model weights in a simple format."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    weights = {k: v for k, v in model.parameters().items()}
    with open(path, 'wb') as f:
        pickle.dump(weights, f)
    print(f"Model saved to {path}")


def test_specific_cases(model: ImprovedCompositionalBindingModel):
    """Test specific compositional cases to understand failures."""
    print("\n=== Testing Specific Compositional Cases ===")
    
    test_cases = [
        ("X means jump Y means walk do X and Y", ['JUMP', 'WALK']),
        ("X means jump Y means walk do X then do Y", ['JUMP', 'WALK']),
        ("X means jump while true do X", ['JUMP', 'JUMP', 'JUMP']),
        ("X means walk Y means jump do X and Y then do X", ['WALK', 'JUMP', 'WALK']),
        ("X means jump do X then X means walk do X", ['JUMP', 'WALK']),
        ("X means jump Y means walk do X or Y", ['JUMP']),  # Should choose X deterministically
    ]
    
    for command, expected in test_cases:
        print(f"\nCommand: {command}")
        print(f"Expected: {expected}")
        
        # Tokenize
        tokens = [VOCAB.get(word, VOCAB['<PAD>']) for word in command.split()]
        inputs = {'command': mx.array([tokens])}
        
        # Parse and show structure
        parse_tree = model.compositional_parser.parse(mx.array(tokens))
        print(f"Parse structure: {describe_parse_tree(parse_tree)}")
        
        # Get predictions
        outputs = model(inputs, stage='full')
        predictions = mx.argmax(outputs, axis=1)
        
        # Convert to action names
        predicted_actions = []
        for pred in predictions:
            for name, idx in ACTIONS.items():
                if idx == int(pred):
                    predicted_actions.append(name)
                    break
        
        print(f"Predicted: {predicted_actions}")
        print(f"Correct: {predicted_actions == expected}")


def describe_parse_tree(node: ParseNode) -> str:
    """Get a string description of parse tree structure."""
    if node.is_leaf():
        return f"LEAF({len(node.tokens)} tokens)"
    else:
        children_desc = [describe_parse_tree(c) for c in node.children]
        return f"{node.operator.value}({', '.join(children_desc)})"


if __name__ == "__main__":
    # Train model
    model = train_improved_compositional_model(num_epochs=5)
    
    # Test specific cases
    test_specific_cases(model)