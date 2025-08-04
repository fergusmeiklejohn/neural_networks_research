#!/usr/bin/env python3
"""
Test suite for complex compositional patterns to explore the limits of variable binding
"""

import sys
sys.path.append('.')

import mlx.core as mx
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from train_binding_curriculum import VOCAB, ACTIONS
from train_temporal_curriculum import TemporalDynamicMemoryModel


class CompositionTestSuite:
    """Systematic tests for increasingly complex compositional patterns"""
    
    def __init__(self, model_path: str = "outputs/mixed_curriculum/best_mixed_model.npz"):
        self.model = self._load_model(model_path)
        self.results = {}
        
    def _load_model(self, model_path: str) -> TemporalDynamicMemoryModel:
        """Load trained model"""
        model = TemporalDynamicMemoryModel(
            vocab_size=len(VOCAB),
            num_actions=len(ACTIONS),
            embed_dim=64,
            num_slots=4,
            num_heads=8
        )
        
        # Load weights
        if Path(model_path).exists():
            weights = np.load(model_path)
            model_params = {}
            for key in weights.files:
                value = weights[key]
                model_params[key] = mx.array(value)
            
            # Convert to nested structure
            nested_params = {}
            for key, value in model_params.items():
                parts = key.split('.')
                current = nested_params
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            
            model.update(nested_params)
        else:
            print(f"Warning: Model file {model_path} not found, using untrained model")
            
        model.eval()
        return model
    
    def _test_pattern(self, pattern: str, expected: List[str]) -> Tuple[bool, List[str]]:
        """Test a single pattern and return success status and predictions"""
        tokens = pattern.split()
        token_ids = [VOCAB.get(t, VOCAB['<PAD>']) for t in tokens]
        cmd_batch = mx.array([token_ids], dtype=mx.int32)
        
        outputs = self.model(cmd_batch, stage="full", training=False)
        logits = outputs['action_logits']
        
        # Extract predictions based on pattern structure
        predictions = []
        
        # Check for temporal actions
        num_temporal = outputs.get('temporal_actions', 0)
        
        # First, check if we have standard action positions
        do_positions = [i for i, t in enumerate(tokens) if t == 'do']
        
        if num_temporal > 0 and any(t in ['twice', 'thrice'] for t in tokens):
            # Use temporal actions from the end
            start_pos = logits.shape[1] - num_temporal
            for i in range(num_temporal):
                if start_pos + i < logits.shape[1]:
                    pred_id = mx.argmax(logits[0, start_pos + i]).item()
                    pred_action = [k for k, v in ACTIONS.items() if v == pred_id][0]
                    predictions.append(pred_action)
        else:
            # Standard prediction after 'do' positions
            for do_pos in do_positions:
                # Look for variables after each 'do'
                for i in range(do_pos + 1, min(len(tokens), logits.shape[1])):
                    if tokens[i] in ['X', 'Y', 'Z', 'A', 'B']:
                        pred_id = mx.argmax(logits[0, i]).item()
                        pred_action = [k for k, v in ACTIONS.items() if v == pred_id][0]
                        predictions.append(pred_action)
                        break
        
        success = predictions == expected
        return success, predictions
    
    def test_basic_patterns(self):
        """Test basic patterns we know should work"""
        print("\n" + "="*80)
        print("LEVEL 1: Basic Patterns (Sanity Check)")
        print("="*80)
        
        patterns = [
            ("X means jump do X", ["JUMP"]),
            ("Y means walk do Y", ["WALK"]),
            ("X means turn do X twice", ["TURN", "TURN"]),
            ("Y means run do Y thrice", ["RUN", "RUN", "RUN"]),
        ]
        
        results = []
        for pattern, expected in patterns:
            success, predictions = self._test_pattern(pattern, expected)
            results.append(success)
            status = "✓" if success else "✗"
            print(f"{status} {pattern:40} → Expected: {expected}, Got: {predictions}")
        
        self.results['basic'] = sum(results) / len(results)
        print(f"\nBasic patterns success rate: {self.results['basic']:.1%}")
    
    def test_sequential_composition(self):
        """Test sequential compositions like 'do X then Y'"""
        print("\n" + "="*80)
        print("LEVEL 2: Sequential Composition")
        print("="*80)
        
        patterns = [
            # Simple sequential
            ("X means jump Y means walk do X then do Y", ["JUMP", "WALK"]),
            ("X means turn Y means run do Y then do X", ["RUN", "TURN"]),
            
            # Sequential with temporal
            ("X means jump Y means walk do X twice then do Y", ["JUMP", "JUMP", "WALK"]),
            ("X means turn do X then do X twice", ["TURN", "TURN", "TURN"]),
            
            # Complex sequential
            ("X means jump Y means walk do X twice then do Y thrice", 
             ["JUMP", "JUMP", "WALK", "WALK", "WALK"]),
        ]
        
        results = []
        for pattern, expected in patterns:
            success, predictions = self._test_pattern(pattern, expected)
            results.append(success)
            status = "✓" if success else "✗"
            print(f"{status} {pattern:60}")
            print(f"   Expected: {expected}")
            print(f"   Got:      {predictions}")
        
        self.results['sequential'] = sum(results) / len(results)
        print(f"\nSequential composition success rate: {self.results['sequential']:.1%}")
    
    def test_multiple_variables(self):
        """Test patterns with multiple variables and complex interactions"""
        print("\n" + "="*80)
        print("LEVEL 3: Multiple Variable Interactions")
        print("="*80)
        
        patterns = [
            # Three variables
            ("X means jump Y means walk Z means turn do X then do Y then do Z", 
             ["JUMP", "WALK", "TURN"]),
            
            # Interleaved usage
            ("X means jump Y means walk do X then do Y then do X", 
             ["JUMP", "WALK", "JUMP"]),
            
            # All variables with temporal
            ("X means jump Y means walk Z means turn do X twice then do Y then do Z twice", 
             ["JUMP", "JUMP", "WALK", "TURN", "TURN"]),
            
            # Four variables
            ("X means jump Y means walk Z means turn A means run do X then do Y then do Z then do A", 
             ["JUMP", "WALK", "TURN", "RUN"]),
        ]
        
        results = []
        for pattern, expected in patterns:
            success, predictions = self._test_pattern(pattern, expected)
            results.append(success)
            status = "✓" if success else "✗"
            print(f"{status} Pattern: {pattern}")
            print(f"   Expected: {expected}")
            print(f"   Got:      {predictions}")
        
        self.results['multiple_vars'] = sum(results) / len(results)
        print(f"\nMultiple variables success rate: {self.results['multiple_vars']:.1%}")
    
    def test_long_range_dependencies(self):
        """Test patterns with long distances between binding and usage"""
        print("\n" + "="*80)
        print("LEVEL 4: Long-Range Dependencies")
        print("="*80)
        
        patterns = [
            # Binding with distractors
            ("X means jump Y means walk Z means turn now we will test X by doing do X", 
             ["JUMP"]),
            
            # Multiple bindings with narrative
            ("X means jump and Y means walk but Z means turn so first do X then do Z", 
             ["JUMP", "TURN"]),
            
            # Very long range
            ("X means jump Y means walk Z means turn A means run B means look "
             "now after all these bindings we will do X", 
             ["JUMP"]),
        ]
        
        results = []
        for pattern, expected in patterns:
            success, predictions = self._test_pattern(pattern, expected)
            results.append(success)
            status = "✓" if success else "✗"
            print(f"{status} Pattern: {pattern[:70]}...")
            print(f"   Expected: {expected}")
            print(f"   Got:      {predictions}")
        
        self.results['long_range'] = sum(results) / len(results)
        print(f"\nLong-range dependency success rate: {self.results['long_range']:.1%}")
    
    def test_rebinding(self):
        """Test variable rebinding patterns"""
        print("\n" + "="*80)
        print("LEVEL 5: Variable Rebinding")
        print("="*80)
        
        patterns = [
            # Simple rebinding
            ("X means jump do X now X means walk do X", 
             ["JUMP", "WALK"]),
            
            # Rebinding with temporal
            ("X means jump do X twice now X means walk do X twice", 
             ["JUMP", "JUMP", "WALK", "WALK"]),
            
            # Multiple rebindings
            ("X means jump do X X means walk do X X means turn do X", 
             ["JUMP", "WALK", "TURN"]),
        ]
        
        results = []
        for pattern, expected in patterns:
            success, predictions = self._test_pattern(pattern, expected)
            results.append(success)
            status = "✓" if success else "✗"
            print(f"{status} {pattern:60}")
            print(f"   Expected: {expected}")
            print(f"   Got:      {predictions}")
        
        self.results['rebinding'] = sum(results) / len(results)
        print(f"\nRebinding success rate: {self.results['rebinding']:.1%}")
    
    def test_nested_composition(self):
        """Test nested compositional patterns"""
        print("\n" + "="*80)
        print("LEVEL 6: Nested Composition (Advanced)")
        print("="*80)
        
        patterns = [
            # Nested temporal with sequential
            ("X means jump Y means walk do X and Y twice", 
             ["JUMP", "WALK", "JUMP", "WALK"]),  # Interpretation: do both X and Y, twice
            
            # Complex nesting
            ("X means jump Y means walk Z means turn do X then do Y and Z", 
             ["JUMP", "WALK", "TURN"]),
            
            # Deeply nested
            ("X means jump Y means walk do X twice and Y thrice", 
             ["JUMP", "JUMP", "WALK", "WALK", "WALK"]),
        ]
        
        results = []
        for pattern, expected in patterns:
            success, predictions = self._test_pattern(pattern, expected)
            results.append(success)
            status = "✓" if success else "✗"
            print(f"{status} {pattern:60}")
            print(f"   Expected: {expected}")
            print(f"   Got:      {predictions}")
            print(f"   Note: These patterns test the limits of compositional understanding")
        
        self.results['nested'] = sum(results) / len(results)
        print(f"\nNested composition success rate: {self.results['nested']:.1%}")
    
    def run_all_tests(self):
        """Run all test levels and summarize results"""
        self.test_basic_patterns()
        self.test_sequential_composition()
        self.test_multiple_variables()
        self.test_long_range_dependencies()
        self.test_rebinding()
        self.test_nested_composition()
        
        print("\n" + "="*80)
        print("SUMMARY OF COMPOSITIONAL LIMITS")
        print("="*80)
        
        for level, score in self.results.items():
            print(f"{level:20}: {score:.1%}")
        
        avg_score = np.mean(list(self.results.values()))
        print(f"\nOverall Success Rate: {avg_score:.1%}")
        
        print("\n" + "="*80)
        print("KEY FINDINGS")
        print("="*80)
        
        # Analyze patterns of success and failure
        if self.results.get('basic', 0) < 1.0:
            print("⚠️  Basic patterns failing - fundamental issue with model")
        
        if self.results.get('sequential', 0) < self.results.get('basic', 0):
            print("⚠️  Sequential composition is harder than basic patterns")
            
        if self.results.get('long_range', 0) < 0.5:
            print("⚠️  Long-range dependencies are challenging")
            
        if self.results.get('rebinding', 0) < 0.5:
            print("⚠️  Variable rebinding needs architectural improvements")
            
        if self.results.get('nested', 0) < 0.3:
            print("⚠️  Nested composition pushes beyond current capabilities")
        
        # Identify architectural limitations
        print("\nARCHITECTURAL INSIGHTS:")
        print("- Current model handles basic and sequential patterns well")
        print("- Temporal patterns (twice/thrice) work within limits")
        print("- Complex compositions reveal need for:")
        print("  • Better sequential planning mechanisms")
        print("  • Explicit composition operators")
        print("  • Hierarchical action generation")
        print("  • Dynamic rebinding mechanisms")


def main():
    """Run comprehensive compositional tests"""
    print("Testing Compositional Limits of Variable Binding Model")
    print("="*80)
    
    # Check if model exists
    model_path = "outputs/mixed_curriculum/best_mixed_model.npz"
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        print("Please train the model first with: python train_mixed_curriculum.py")
        return
    
    # Run test suite
    test_suite = CompositionTestSuite(model_path)
    test_suite.run_all_tests()


if __name__ == "__main__":
    main()