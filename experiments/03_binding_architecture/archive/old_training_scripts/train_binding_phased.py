"""Phased training approach for variable binding"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import argparse
import time
from dereferencing_tasks import DereferencingTaskGenerator
from train_binding_mlx_proper import ProperBindingModel, train_step, evaluate

def generate_phase1_batch(generator, batch_size):
    """Phase 1: Direct word-to-action mapping (no binding needed)"""
    batch_commands = []
    batch_actions = []
    
    for _ in range(batch_size):
        # Simple direct mappings
        word_action_pairs = [
            (['jump'], ['JUMP']),
            (['walk'], ['WALK']),
            (['turn'], ['TURN']),
            (['run'], ['RUN']),
            (['do', 'jump'], ['JUMP']),
            (['do', 'walk'], ['WALK']),
            (['do', 'turn'], ['TURN']),
            (['do', 'run'], ['RUN']),
        ]
        
        command, actions = word_action_pairs[np.random.randint(len(word_action_pairs))]
        
        cmd_encoded = generator.encode_words(command)
        act_encoded = generator.encode_actions(actions)
        
        batch_commands.append(cmd_encoded)
        batch_actions.append(act_encoded)
    
    # Pad to same length
    max_cmd_len = max(len(cmd) for cmd in batch_commands)
    max_act_len = max(len(act) for act in batch_actions)
    
    padded_commands = np.zeros((batch_size, max_cmd_len), dtype=np.int32)
    padded_actions = np.zeros((batch_size, max_act_len), dtype=np.int32)
    
    for i, (cmd, act) in enumerate(zip(batch_commands, batch_actions)):
        padded_commands[i, :len(cmd)] = cmd
        padded_actions[i, :len(act)] = act
    
    return {
        'command': mx.array(padded_commands),
        'action': mx.array(padded_actions)
    }

def generate_phase2_batch(generator, batch_size):
    """Phase 2: Simple variable binding (single variable)"""
    batch_commands = []
    batch_actions = []
    
    for _ in range(batch_size):
        # Only use simple binding patterns
        var = np.random.choice(['X', 'Y', 'A', 'B'])
        action = np.random.choice(['jump', 'walk', 'turn', 'run'])
        
        command = [var, 'means', action, 'do', var]
        actions = [action.upper()]
        
        cmd_encoded = generator.encode_words(command)
        act_encoded = generator.encode_actions(actions)
        
        batch_commands.append(cmd_encoded)
        batch_actions.append(act_encoded)
    
    # Pad to same length
    max_cmd_len = max(len(cmd) for cmd in batch_commands)
    max_act_len = max(len(act) for act in batch_actions)
    
    padded_commands = np.zeros((batch_size, max_cmd_len), dtype=np.int32)
    padded_actions = np.zeros((batch_size, max_act_len), dtype=np.int32)
    
    for i, (cmd, act) in enumerate(zip(batch_commands, batch_actions)):
        padded_commands[i, :len(cmd)] = cmd
        padded_actions[i, :len(act)] = act
    
    return {
        'command': mx.array(padded_commands),
        'action': mx.array(padded_actions)
    }

def generate_phase3_batch(generator, batch_size):
    """Phase 3: Multiple variables and compositional patterns"""
    batch_commands = []
    batch_actions = []
    
    for _ in range(batch_size):
        task_type = np.random.choice(['multiple', 'compositional'], p=[0.5, 0.5])
        
        if task_type == 'multiple':
            # Two variables
            var1, var2 = np.random.choice(['X', 'Y', 'A', 'B'], size=2, replace=False)
            action1, action2 = np.random.choice(['jump', 'walk', 'turn', 'run'], size=2, replace=False)
            
            command = [var1, 'means', action1, var2, 'means', action2, 'do', var1, 'then', var2]
            actions = [action1.upper(), action2.upper()]
        else:
            # Compositional
            var = np.random.choice(['X', 'Y', 'A', 'B'])
            action = np.random.choice(['jump', 'walk', 'turn', 'run'])
            modifier = np.random.choice(['twice', 'thrice'])
            
            command = [var, 'means', action, 'do', var, modifier]
            if modifier == 'twice':
                actions = [action.upper(), action.upper()]
            else:
                actions = [action.upper(), action.upper(), action.upper()]
        
        cmd_encoded = generator.encode_words(command)
        act_encoded = generator.encode_actions(actions)
        
        batch_commands.append(cmd_encoded)
        batch_actions.append(act_encoded)
    
    # Pad to same length
    max_cmd_len = max(len(cmd) for cmd in batch_commands)
    max_act_len = max(len(act) for act in batch_actions)
    
    padded_commands = np.zeros((batch_size, max_cmd_len), dtype=np.int32)
    padded_actions = np.zeros((batch_size, max_act_len), dtype=np.int32)
    
    for i, (cmd, act) in enumerate(zip(batch_commands, batch_actions)):
        padded_commands[i, :len(cmd)] = cmd
        padded_actions[i, :len(act)] = act
    
    return {
        'command': mx.array(padded_commands),
        'action': mx.array(padded_actions)
    }

def train_phase(model, generator, phase_name, generate_fn, optimizer, loss_fn, 
                num_epochs=5, batch_size=32, steps_per_epoch=100):
    """Train one phase"""
    print(f"\n=== {phase_name} ===")
    
    best_accuracy = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        
        # Update temperature
        current_temp = model.binder.temperature * 0.9  # Decay each epoch
        model.binder.temperature = max(0.1, current_temp)
        
        for _ in range(steps_per_epoch):
            batch = generate_fn(generator, batch_size)
            loss, outputs, _ = train_step(model, batch, loss_fn, optimizer)
            epoch_loss += loss.item()
            
            # Track accuracy
            logits = outputs['action_logits']
            preds = mx.argmax(logits, axis=-1)
            actions = batch['action']
            
            # Handle sequence length mismatch
            min_len = min(preds.shape[1], actions.shape[1])
            preds = preds[:, :min_len]
            actions = actions[:, :min_len]
            
            correct = mx.sum(preds == actions)
            total = preds.shape[0] * preds.shape[1]
            
            epoch_correct += correct.item()
            epoch_total += total
            
            mx.eval(model.parameters(), optimizer.state)
        
        accuracy = epoch_correct / epoch_total
        best_accuracy = max(best_accuracy, accuracy)
        
        print(f"  Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss/steps_per_epoch:.4f}, "
              f"Accuracy: {accuracy:.2%}, Temperature: {model.binder.temperature:.3f}")
    
    return best_accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase1_epochs', type=int, default=10, help='Phase 1 epochs')
    parser.add_argument('--phase2_epochs', type=int, default=15, help='Phase 2 epochs')
    parser.add_argument('--phase3_epochs', type=int, default=10, help='Phase 3 epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.003, help='Learning rate')
    args = parser.parse_args()
    
    print("=== Phased Variable Binding Training ===")
    print(f"Phase 1: {args.phase1_epochs} epochs - Direct word-to-action mapping")
    print(f"Phase 2: {args.phase2_epochs} epochs - Simple variable binding")
    print(f"Phase 3: {args.phase3_epochs} epochs - Complex patterns")
    
    # Initialize
    generator = DereferencingTaskGenerator()
    model = ProperBindingModel(
        vocab_size=len(generator.word_to_id),
        num_actions=len(generator.action_to_id),
        embed_dim=128,
        hidden_dim=256,
        num_slots=10,
        num_heads=8
    )
    
    # Start with low temperature for phase 1
    model.binder.temperature = 0.5
    
    optimizer = optim.Adam(learning_rate=args.lr)
    loss_fn = nn.losses.cross_entropy
    
    # Phase 1: Direct mapping
    phase1_acc = train_phase(
        model, generator, "Phase 1: Direct Mapping",
        generate_phase1_batch, optimizer, loss_fn,
        num_epochs=args.phase1_epochs, 
        batch_size=args.batch_size
    )
    
    # Test on simple patterns
    print("\n  Testing on direct patterns:")
    test_patterns = [
        (['jump'], ['JUMP']),
        (['walk'], ['WALK']),
        (['do', 'jump'], ['JUMP']),
        (['do', 'walk'], ['WALK']),
    ]
    
    correct = 0
    for command, expected in test_patterns:
        cmd_encoded = generator.encode_words(command)
        cmd_array = mx.array(cmd_encoded)[None, :]
        outputs = model(cmd_array, training=False)
        pred = mx.argmax(outputs['action_logits'], axis=-1)[0, 0].item()
        predicted = generator.id_to_action[pred] if pred < len(generator.id_to_action) else '<UNK>'
        match = predicted == expected[0]
        correct += match
        print(f"    {' '.join(command):15} -> {predicted:10} {'✓' if match else '✗'}")
    
    print(f"  Direct pattern accuracy: {correct}/{len(test_patterns)} = {correct/len(test_patterns):.0%}")
    
    # Phase 2: Simple binding
    phase2_acc = train_phase(
        model, generator, "Phase 2: Simple Variable Binding",
        generate_phase2_batch, optimizer, loss_fn,
        num_epochs=args.phase2_epochs,
        batch_size=args.batch_size
    )
    
    # Test binding
    print("\n  Testing simple binding:")
    test_bindings = [
        (['X', 'means', 'jump', 'do', 'X'], ['JUMP']),
        (['Y', 'means', 'walk', 'do', 'Y'], ['WALK']),
        (['A', 'means', 'turn', 'do', 'A'], ['TURN']),
        (['B', 'means', 'run', 'do', 'B'], ['RUN']),
    ]
    
    correct = 0
    for command, expected in test_bindings:
        cmd_encoded = generator.encode_words(command)
        cmd_array = mx.array(cmd_encoded)[None, :]
        outputs = model(cmd_array, training=False)
        pred = mx.argmax(outputs['action_logits'], axis=-1)[0, 0].item()
        predicted = generator.id_to_action[pred] if pred < len(generator.id_to_action) else '<UNK>'
        match = predicted == expected[0]
        correct += match
        print(f"    {' '.join(command):25} -> {predicted:10} {'✓' if match else '✗'}")
    
    print(f"  Binding accuracy: {correct}/{len(test_bindings)} = {correct/len(test_bindings):.0%}")
    
    # Phase 3: Complex patterns
    phase3_acc = train_phase(
        model, generator, "Phase 3: Complex Patterns",
        generate_phase3_batch, optimizer, loss_fn,
        num_epochs=args.phase3_epochs,
        batch_size=args.batch_size
    )
    
    # Final evaluation
    print("\n=== Final Evaluation ===")
    
    # Test modification capability
    from train_binding_mlx_proper import test_modifications
    test_modifications(model, generator)
    
    # Save model
    print("\nSaving model...")
    params_dict = {}
    for k, v in model.parameters().items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                params_dict[f"{k}.{k2}"] = np.array(v2)
        else:
            params_dict[k] = np.array(v)
    np.savez("phased_binding_model.npz", **params_dict)
    print("Model saved to phased_binding_model.npz")

if __name__ == "__main__":
    main()