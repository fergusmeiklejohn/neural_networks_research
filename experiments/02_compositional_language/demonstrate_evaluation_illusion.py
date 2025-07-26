#!/usr/bin/env python3
"""
Minimal demonstration of the Evaluation Illusion in compositional generalization.

This script shows how standard validation metrics can show high accuracy (>80%)
while the model completely fails (0%) at the actual task it was designed for.

Run this to see the illusion in action!
"""

import numpy as np
from typing import List, Tuple, Dict
import random


class SimpleCompositionalTask:
    """A toy compositional task to demonstrate the evaluation illusion."""
    
    def __init__(self):
        # Base mappings
        self.base_mappings = {
            "walk": "MOVE_FORWARD",
            "jump": "MOVE_UP",
            "look": "OBSERVE",
            "turn": "ROTATE"
        }
        
        # Modifications we want the model to learn
        self.modifications = {
            "quickly": lambda x: f"FAST_{x}",
            "slowly": lambda x: f"SLOW_{x}",
            "twice": lambda x: f"{x} {x}"
        }
        
    def generate_examples(self, n_examples: int, include_modifications: bool = True, 
                         exclude_modifier: str = None) -> List[Tuple[str, str]]:
        """Generate training/validation examples."""
        examples = []
        
        # Available modifiers (excluding any we want to test as truly unseen)
        available_modifiers = list(self.modifications.keys())
        if exclude_modifier and exclude_modifier in available_modifiers:
            available_modifiers.remove(exclude_modifier)
        
        for _ in range(n_examples):
            if include_modifications and random.random() > 0.5 and available_modifiers:
                # Modified example
                base_command = random.choice(list(self.base_mappings.keys()))
                modifier = random.choice(available_modifiers)
                
                command = f"{modifier} {base_command}"
                base_action = self.base_mappings[base_command]
                action = self.modifications[modifier](base_action)
                
                examples.append((command, action))
            else:
                # Base example
                command = random.choice(list(self.base_mappings.keys()))
                action = self.base_mappings[command]
                examples.append((command, action))
                
        return examples
    
    def generate_test_sets(self) -> Dict[str, List[Tuple[str, str]]]:
        """Generate proper test sets that reveal the illusion."""
        test_sets = {}
        
        # Test set 1: Only base commands (no modifications)
        test_sets["base_only"] = [
            (cmd, action) for cmd, action in self.base_mappings.items()
        ]
        
        # Test set 2: Only modifications
        test_sets["modifications_only"] = []
        for modifier, mod_func in self.modifications.items():
            for base_cmd, base_action in self.base_mappings.items():
                command = f"{modifier} {base_cmd}"
                action = mod_func(base_action)
                test_sets["modifications_only"].append((command, action))
        
        # Test set 3: Specific modification types
        for modifier, mod_func in self.modifications.items():
            test_sets[f"mod_{modifier}"] = []
            for base_cmd, base_action in self.base_mappings.items():
                command = f"{modifier} {base_cmd}"
                action = mod_func(base_action)
                test_sets[f"mod_{modifier}"].append((command, action))
                
        return test_sets


class MockModel:
    """A mock model that demonstrates the evaluation illusion."""
    
    def __init__(self):
        # This model "learns" to map inputs to outputs, but in a way that
        # creates the illusion of understanding while failing at the real task
        self.memory = {}
        
    def train(self, examples: List[Tuple[str, str]]):
        """'Train' by memorizing exact examples."""
        # Simulate a model that achieves high accuracy by memorizing
        # patterns but doesn't actually understand composition
        for command, action in examples:
            # Store exact mappings
            self.memory[command] = action
            
            # Also store some partial patterns (creates the illusion)
            words = command.split()
            if len(words) == 1:
                # Base command - store it
                self.memory[command] = action
            
            # IMPORTANT: We DON'T learn the "twice" pattern properly
            # This simulates a model that learns some modifications but not others
    
    def predict(self, command: str) -> str:
        """Predict action for a command."""
        # First try exact match (this is what creates high validation accuracy)
        if command in self.memory:
            return self.memory[command]
        
        # Try some spurious patterns
        words = command.split()
        if len(words) == 2:
            modifier, base = words
            if base in self.memory:
                # This will work for some modifiers but not others
                if modifier == "quickly":
                    return f"FAST_{self.memory[base]}"
                elif modifier == "slowly":
                    return f"SLOW_{self.memory[base]}"
        
        # Default failure
        return "UNKNOWN"


def evaluate(model: MockModel, test_set: List[Tuple[str, str]], name: str) -> float:
    """Evaluate model on a test set."""
    correct = 0
    total = len(test_set)
    
    for command, expected in test_set:
        predicted = model.predict(command)
        if predicted == expected:
            correct += 1
    
    accuracy = (correct / total) * 100
    print(f"{name:25} {accuracy:5.1f}% ({correct}/{total} correct)")
    return accuracy


def demonstrate_evaluation_illusion():
    """Main demonstration of the evaluation illusion."""
    print("=" * 60)
    print("DEMONSTRATING THE EVALUATION ILLUSION")
    print("=" * 60)
    
    # Create task
    task = SimpleCompositionalTask()
    
    # Generate training data (mixed base + modifications, but exclude "twice")
    print("\n1. Generating training data...")
    train_data = task.generate_examples(1000, include_modifications=True, exclude_modifier="twice")
    print(f"   - Generated {len(train_data)} training examples")
    print(f"   - Mix of base commands and modifications")
    print(f"   - NOTE: 'twice' modification excluded from training!")
    
    # Generate standard validation data (same distribution as training)
    val_data = task.generate_examples(200, include_modifications=True, exclude_modifier="twice")
    print(f"   - Generated {len(val_data)} validation examples")
    print(f"   - Also excluding 'twice' (mimicking training distribution)")
    
    # Train model
    print("\n2. Training model...")
    model = MockModel()
    model.train(train_data)
    print("   - Model 'trained' (memorized patterns)")
    
    # Standard evaluation (this creates the illusion)
    print("\n3. Standard Validation (creates the illusion):")
    print("-" * 50)
    standard_accuracy = evaluate(model, val_data, "Standard validation")
    
    # Proper evaluation (this reveals the truth)
    print("\n4. Proper Evaluation (reveals the truth):")
    print("-" * 50)
    test_sets = task.generate_test_sets()
    
    results = {}
    for test_name, test_data in test_sets.items():
        accuracy = evaluate(model, test_data, test_name)
        results[test_name] = accuracy
    
    # Show the illusion
    print("\n5. THE EVALUATION ILLUSION:")
    print("=" * 60)
    print(f"Standard validation suggests the model works: {standard_accuracy:.1f}%")
    print(f"But it completely fails on the modification 'twice': {results['mod_twice']:.1f}%")
    print("\nThe model memorized patterns rather than learning composition!")
    
    # Demonstrate specific failures
    print("\n6. Example failures:")
    print("-" * 50)
    test_commands = [
        ("walk", "Should output: MOVE_FORWARD"),
        ("quickly walk", "Should output: FAST_MOVE_FORWARD"),
        ("twice walk", "Should output: MOVE_FORWARD MOVE_FORWARD"),
        ("twice jump", "Should output: MOVE_UP MOVE_UP")
    ]
    
    for command, expected in test_commands:
        predicted = model.predict(command)
        status = "✓" if predicted != "UNKNOWN" else "✗"
        print(f"{status} '{command}' → '{predicted}' ({expected})")
    
    print("\n" + "=" * 60)
    print("CONCLUSION: High validation accuracy can hide complete failure!")
    print("Always test specific capabilities, not just aggregate performance.")
    print("=" * 60)


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Run demonstration
    demonstrate_evaluation_illusion()
    
    print("\n💡 Try modifying this script to:")
    print("   - Add new modification types")
    print("   - Change the model's learning strategy")
    print("   - Create different evaluation illusions")
    print("\nThe key insight: Standard evaluation practices can hide complete")
    print("failure modes in models that appear to be successful!")