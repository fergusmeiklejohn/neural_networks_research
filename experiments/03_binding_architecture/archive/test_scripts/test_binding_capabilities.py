#!/usr/bin/env python3
"""
Comprehensive Test Suite for Variable Binding

Tests the model's ability to perform true variable binding and modification,
not just pattern matching. Based on Wu et al. (2025) insights.

Key tests:
1. Basic dereferencing: "X means jump. Do X." → "JUMP"
2. Multiple variables: "X means jump. Y means walk. Do Y then X."
3. Rebinding: "X means jump. Now X means hop. Do X." → "HOP"
4. Compositional: "X means jump. Do X twice."
5. Chained references: "X means Y. Y means jump. Do X." → "JUMP"
"""

import mlx.core as mx
import numpy as np
import sys
import os
from typing import List, Tuple, Dict
from dataclasses import dataclass

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dereferencing_tasks import DereferencingTaskGenerator


@dataclass
class TestCase:
    """Single test case for variable binding"""
    name: str
    description: str
    commands: List[str]
    expected_actions: List[str]
    test_type: str  # 'basic', 'rebinding', 'compositional', 'chained'


class BindingTestSuite:
    """Comprehensive test suite for variable binding capabilities"""
    
    def __init__(self):
        self.generator = DereferencingTaskGenerator()
        self.test_cases = self._create_test_cases()
        
    def _create_test_cases(self) -> List[TestCase]:
        """Create comprehensive test cases"""
        return [
            # Basic dereferencing
            TestCase(
                name="basic_single",
                description="Single variable binding",
                commands=["X", "means", "jump", ".", "Do", "X", "."],
                expected_actions=["JUMP"],
                test_type="basic"
            ),
            TestCase(
                name="basic_multiple",
                description="Multiple independent variables",
                commands=["X", "means", "jump", ".", "Y", "means", "walk", ".", "Do", "Y", "."],
                expected_actions=["WALK"],
                test_type="basic"
            ),
            
            # Rebinding tests (critical for modification)
            TestCase(
                name="rebinding_simple",
                description="Variable rebinding",
                commands=["X", "means", "jump", ".", "X", "means", "hop", ".", "Do", "X", "."],
                expected_actions=["HOP"],
                test_type="rebinding"
            ),
            TestCase(
                name="rebinding_multiple",
                description="Multiple rebindings",
                commands=["X", "means", "jump", ".", "X", "means", "walk", ".", 
                         "X", "means", "turn", ".", "Do", "X", "."],
                expected_actions=["TURN"],
                test_type="rebinding"
            ),
            
            # Compositional tests
            TestCase(
                name="compositional_repeat",
                description="Compositional with repetition",
                commands=["X", "means", "jump", ".", "Do", "X", "twice", "."],
                expected_actions=["JUMP", "JUMP"],
                test_type="compositional"
            ),
            TestCase(
                name="compositional_sequence",
                description="Compositional sequence",
                commands=["X", "means", "jump", ".", "Y", "means", "walk", ".", 
                         "Do", "X", "then", "Y", "."],
                expected_actions=["JUMP", "WALK"],
                test_type="compositional"
            ),
            
            # Chained references
            TestCase(
                name="chained_simple",
                description="Simple chained reference",
                commands=["X", "means", "Y", ".", "Y", "means", "jump", ".", "Do", "X", "."],
                expected_actions=["JUMP"],
                test_type="chained"
            ),
            TestCase(
                name="chained_deep",
                description="Deep chained reference",
                commands=["X", "means", "Y", ".", "Y", "means", "Z", ".", 
                         "Z", "means", "turn", ".", "Do", "X", "."],
                expected_actions=["TURN"],
                test_type="chained"
            ),
            
            # Mixed complexity
            TestCase(
                name="mixed_rebind_compose",
                description="Rebinding with composition",
                commands=["X", "means", "jump", ".", "Do", "X", ".", 
                         "X", "means", "walk", ".", "Do", "X", "twice", "."],
                expected_actions=["JUMP", "WALK", "WALK"],
                test_type="mixed"
            ),
        ]
    
    def encode_test_case(self, test_case: TestCase) -> Dict[str, mx.array]:
        """Convert test case to model inputs"""
        # Convert commands to IDs
        command_ids = []
        for word in test_case.commands:
            if word in self.generator.word_to_id:
                command_ids.append(self.generator.word_to_id[word])
            else:
                command_ids.append(self.generator.word_to_id["<UNK>"])
        
        # Convert expected actions to IDs
        action_ids = []
        for action in test_case.expected_actions:
            if action in self.generator.action_to_id:
                action_ids.append(self.generator.action_to_id[action])
        
        return {
            'command': mx.array(command_ids)[None, :],  # Add batch dimension
            'expected_actions': action_ids
        }
    
    def evaluate_model(self, model, verbose=True) -> Dict[str, float]:
        """Evaluate model on all test cases"""
        results = {
            'basic': {'correct': 0, 'total': 0},
            'rebinding': {'correct': 0, 'total': 0},
            'compositional': {'correct': 0, 'total': 0},
            'chained': {'correct': 0, 'total': 0},
            'mixed': {'correct': 0, 'total': 0}
        }
        
        if verbose:
            print("\n=== Variable Binding Test Results ===\n")
        
        for test_case in self.test_cases:
            # Encode test case
            inputs = self.encode_test_case(test_case)
            
            # Get model predictions
            outputs = model(inputs['command'])
            logits = outputs['action_logits'][0]  # Remove batch dimension
            predictions = mx.argmax(logits, axis=-1)
            
            # Extract predicted actions up to expected length
            expected_len = len(inputs['expected_actions'])
            pred_actions = predictions[:expected_len].tolist()
            
            # Check correctness
            correct = pred_actions == inputs['expected_actions']
            results[test_case.test_type]['correct'] += correct
            results[test_case.test_type]['total'] += 1
            
            if verbose:
                print(f"Test: {test_case.name}")
                print(f"Description: {test_case.description}")
                print(f"Command: {' '.join(test_case.commands)}")
                print(f"Expected: {test_case.expected_actions}")
                pred_action_names = []
                for a in pred_actions:
                    if a < len(self.generator.id_to_action):
                        pred_action_names.append(self.generator.id_to_action[a])
                    else:
                        pred_action_names.append(f"<UNK_{a}>")
                print(f"Predicted: {pred_action_names}")
                print(f"Correct: {'✓' if correct else '✗'}")
                
                # Show bindings if available
                if 'bindings' in outputs:
                    bindings = outputs['bindings'][0].tolist()
                    print(f"Bindings: {bindings[:10]}...")  # Show first 10
                
                print("-" * 50)
        
        # Calculate accuracies
        accuracies = {}
        total_correct = 0
        total_count = 0
        
        for test_type, counts in results.items():
            if counts['total'] > 0:
                acc = counts['correct'] / counts['total']
                accuracies[test_type] = acc
                total_correct += counts['correct']
                total_count += counts['total']
        
        accuracies['overall'] = total_correct / total_count if total_count > 0 else 0
        
        if verbose:
            print("\n=== Summary by Test Type ===")
            for test_type, acc in accuracies.items():
                if test_type != 'overall':
                    print(f"{test_type.capitalize()}: {acc:.1%} "
                          f"({results[test_type]['correct']}/{results[test_type]['total']})")
            print(f"\nOverall Accuracy: {accuracies['overall']:.1%} "
                  f"({total_correct}/{total_count})")
        
        return accuracies
    
    def analyze_failure_modes(self, model):
        """Analyze common failure patterns"""
        print("\n=== Failure Mode Analysis ===\n")
        
        # Test specific failure modes
        failure_tests = [
            {
                'name': 'Binding Persistence',
                'commands': ["X", "means", "jump", ".", "Do", "X", ".", "Do", "X", "."],
                'expected': ["JUMP", "JUMP"],
                'description': "Tests if bindings persist across multiple uses"
            },
            {
                'name': 'Binding Isolation',
                'commands': ["X", "means", "jump", ".", "Y", "means", "walk", ".", 
                           "Do", "X", ".", "Do", "Y", "."],
                'expected': ["JUMP", "WALK"],
                'description': "Tests if different variables maintain separate bindings"
            },
            {
                'name': 'Rebinding Immediacy',
                'commands': ["X", "means", "jump", ".", "Do", "X", ".", 
                           "X", "means", "hop", ".", "Do", "X", "."],
                'expected': ["JUMP", "HOP"],
                'description': "Tests if rebinding takes effect immediately"
            }
        ]
        
        for test in failure_tests:
            # Encode and run
            command_ids = [self.generator.word_to_id.get(w, 0) for w in test['commands']]
            outputs = model(mx.array(command_ids)[None, :])
            predictions = mx.argmax(outputs['action_logits'][0], axis=-1)
            
            # Get predicted actions
            pred_actions = []
            action_positions = []
            for i, word in enumerate(test['commands']):
                if word == "." and i > 0 and test['commands'][i-1] in ["X", "Y", "Z"]:
                    action_positions.append(i-1)
            
            for pos in action_positions[:len(test['expected'])]:
                if pos < len(predictions):
                    pred_id = predictions[pos].item()
                    if pred_id < len(self.generator.id_to_action):
                        pred_actions.append(self.generator.id_to_action[pred_id])
            
            correct = pred_actions == test['expected']
            
            print(f"Test: {test['name']}")
            print(f"Description: {test['description']}")
            print(f"Expected: {test['expected']}")
            print(f"Predicted: {pred_actions}")
            print(f"Status: {'✓ PASS' if correct else '✗ FAIL'}")
            print("-" * 50)


def main():
    """Run comprehensive evaluation"""
    print("=== Variable Binding Capability Testing ===\n")
    
    # Load model if it exists
    try:
        # Try to load the proper binding model
        model_path = "proper_binding_model.npz"
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}...")
            # This is a placeholder - would need actual model loading code
            print("Note: Model loading not implemented - would test with trained model")
        else:
            print("No trained model found. Train a model first with train_binding_mlx_proper.py")
            return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create test suite
    test_suite = BindingTestSuite()
    
    # For demonstration, show test cases
    print("\nTest Suite Overview:")
    print(f"Total test cases: {len(test_suite.test_cases)}")
    for test_type in ['basic', 'rebinding', 'compositional', 'chained', 'mixed']:
        count = sum(1 for tc in test_suite.test_cases if tc.test_type == test_type)
        print(f"- {test_type.capitalize()}: {count} tests")
    
    # Note: Actual evaluation would happen here with loaded model
    # accuracies = test_suite.evaluate_model(model)
    # test_suite.analyze_failure_modes(model)


if __name__ == "__main__":
    main()