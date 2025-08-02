#!/usr/bin/env python3
"""Implementation of nested temporal patterns for variable binding.

This module handles patterns like:
- "do X twice twice" → X repeated 4 times (2 * 2)
- "do X thrice twice" → X repeated 6 times (3 * 2)
- "do X twice then do Y thrice twice" → X X Y Y Y Y Y Y
"""

from utils.imports import setup_project_paths
setup_project_paths()

from utils.config import setup_environment
config = setup_environment()

import mlx.core as mx
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum


class TemporalModifier(Enum):
    """Types of temporal modifiers."""
    TWICE = 2
    THRICE = 3
    ONCE = 1  # Default


@dataclass
class TemporalNode:
    """Node representing a temporal pattern."""
    modifier: TemporalModifier
    children: List['TemporalNode'] = None
    base_action: Optional[str] = None  # For leaf nodes
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return self.base_action is not None
    
    def compute_repetitions(self) -> int:
        """Compute total repetitions for this node."""
        if self.is_leaf():
            return self.modifier.value
        else:
            # Multiply by children's repetitions
            child_reps = 1
            for child in self.children:
                child_reps *= child.compute_repetitions()
            return self.modifier.value * child_reps


class NestedTemporalParser:
    """Parse nested temporal patterns."""
    
    def __init__(self, vocab: Dict[str, int]):
        self.vocab = vocab
        self.temporal_tokens = {
            vocab.get('twice', -1): TemporalModifier.TWICE,
            vocab.get('thrice', -1): TemporalModifier.THRICE,
        }
        # Remove invalid entries
        self.temporal_tokens = {k: v for k, v in self.temporal_tokens.items() if k != -1}
        
        self.do_token = vocab.get('do', -1)
        self.var_tokens = {
            vocab.get('X', -1), 
            vocab.get('Y', -1), 
            vocab.get('Z', -1)
        }
        self.var_tokens = {t for t in self.var_tokens if t != -1}
    
    def parse_temporal_sequence(self, tokens: List[int], start: int, end: int) -> Tuple[Optional[TemporalNode], int]:
        """Parse a temporal sequence starting from position.
        
        Returns:
            (temporal_node, next_position) or (None, start) if no pattern found
        """
        i = start
        
        # Look for "do VARIABLE" pattern
        if i >= end or tokens[i] != self.do_token:
            return None, start
        
        i += 1  # Skip 'do'
        
        # Check for variable
        if i >= end or tokens[i] not in self.var_tokens:
            return None, start
        
        var_token = tokens[i]
        # Find variable name
        var_name = None
        for name, token_id in self.vocab.items():
            if token_id == var_token and name in ['X', 'Y', 'Z']:
                var_name = name
                break
        
        if not var_name:
            return None, start
        
        i += 1  # Skip variable
        
        # Now collect all temporal modifiers
        modifiers = []
        while i < end and tokens[i] in self.temporal_tokens:
            modifiers.append(self.temporal_tokens[tokens[i]])
            i += 1
        
        # Build nested structure from right to left
        if not modifiers:
            # No modifiers, just the action
            return TemporalNode(
                modifier=TemporalModifier.ONCE,
                base_action=var_name
            ), i
        
        # Build from innermost (rightmost) to outermost (leftmost)
        current_node = TemporalNode(
            modifier=modifiers[-1],
            base_action=var_name
        )
        
        # Wrap with remaining modifiers
        for modifier in reversed(modifiers[:-1]):
            current_node = TemporalNode(
                modifier=modifier,
                children=[current_node]
            )
        
        return current_node, i
    
    def parse(self, tokens: List[int]) -> List[TemporalNode]:
        """Parse entire token sequence for temporal patterns."""
        patterns = []
        i = 0
        
        while i < len(tokens):
            node, next_i = self.parse_temporal_sequence(tokens, i, len(tokens))
            if node:
                patterns.append(node)
                i = next_i
            else:
                i += 1
        
        return patterns


class NestedTemporalExecutor:
    """Execute nested temporal patterns."""
    
    def __init__(self, model, vocab: Dict[str, int], action_map: Dict[str, int]):
        self.model = model
        self.vocab = vocab
        self.action_map = action_map
        self.parser = NestedTemporalParser(vocab)
    
    def expand_temporal_node(self, node: TemporalNode, bindings: Dict[str, str]) -> List[str]:
        """Expand a temporal node into a sequence of actions."""
        if node.is_leaf():
            # Get the bound action for this variable
            var_name = node.base_action
            if var_name in bindings:
                action = bindings[var_name]
                return [action] * node.modifier.value
            else:
                # Default action if not bound
                return ['WALK'] * node.modifier.value
        else:
            # Recursive case - expand children first
            child_actions = []
            for child in node.children:
                child_actions.extend(self.expand_temporal_node(child, bindings))
            
            # Repeat the child sequence
            return child_actions * node.modifier.value
    
    def execute_nested_pattern(self, tokens: List[int], bindings: Dict[str, str]) -> List[mx.array]:
        """Execute a pattern with nested temporal modifiers."""
        # Parse temporal patterns
        temporal_nodes = self.parser.parse(tokens)
        
        # Expand each node
        all_actions = []
        for node in temporal_nodes:
            actions = self.expand_temporal_node(node, bindings)
            all_actions.extend(actions)
        
        # Convert to model outputs
        outputs = []
        for action in all_actions:
            if action in self.action_map:
                # Create one-hot encoded output
                output = mx.zeros(len(self.action_map))
                output[self.action_map[action]] = 1.0
                outputs.append(output)
        
        return outputs


def test_nested_temporal_patterns():
    """Test the nested temporal pattern parser."""
    from train_integrated_model import VOCAB, ACTIONS
    
    # Ensure temporal modifiers in vocabulary
    if 'twice' not in VOCAB:
        VOCAB['twice'] = len(VOCAB)
    if 'thrice' not in VOCAB:
        VOCAB['thrice'] = len(VOCAB)
    
    parser = NestedTemporalParser(VOCAB)
    
    test_cases = [
        "do X twice",
        "do X twice twice",
        "do X thrice twice",
        "do X twice thrice",
        "do X twice twice twice",
        "do X then do Y twice twice",
    ]
    
    print("=== Testing Nested Temporal Parser ===\n")
    
    for command in test_cases:
        print(f"Command: {command}")
        tokens = [VOCAB.get(word, VOCAB['<PAD>']) for word in command.split()]
        
        patterns = parser.parse_temporal_sequence(tokens, 0, len(tokens))
        if patterns[0]:
            node, _ = patterns
            repetitions = node.compute_repetitions()
            print(f"Parsed: {describe_temporal_node(node)}")
            print(f"Total repetitions: {repetitions}")
        else:
            print("No temporal pattern found")
        print()
    
    # Test full parsing
    print("\n=== Testing Full Pattern Parsing ===\n")
    
    complex_command = "do X twice twice then do Y thrice"
    print(f"Command: {complex_command}")
    tokens = [VOCAB.get(word, VOCAB['<PAD>']) for word in complex_command.split()]
    
    all_patterns = parser.parse(tokens)
    print(f"Found {len(all_patterns)} patterns:")
    for i, pattern in enumerate(all_patterns):
        print(f"  Pattern {i+1}: {describe_temporal_node(pattern)} - {pattern.compute_repetitions()} repetitions")
    
    # Test execution
    print("\n=== Testing Execution ===\n")
    
    executor = NestedTemporalExecutor(None, VOCAB, ACTIONS)
    bindings = {'X': 'JUMP', 'Y': 'WALK'}
    
    for command in ["do X twice twice", "do Y thrice twice"]:
        print(f"Command: {command}")
        print(f"Bindings: {bindings}")
        tokens = [VOCAB.get(word, VOCAB['<PAD>']) for word in command.split()]
        
        outputs = executor.execute_nested_pattern(tokens, bindings)
        actions = []
        for out in outputs:
            idx = int(mx.argmax(out))
            for action, action_id in ACTIONS.items():
                if action_id == idx:
                    actions.append(action)
                    break
        
        print(f"Executed actions: {actions}")
        print(f"Count: {len(actions)}")
        print()


def describe_temporal_node(node: TemporalNode, indent: int = 0) -> str:
    """Get string description of temporal node."""
    prefix = "  " * indent
    
    if node.is_leaf():
        return f"{prefix}{node.base_action} x{node.modifier.value}"
    else:
        parts = [f"{prefix}{node.modifier.name} x{node.modifier.value} of:"]
        for child in node.children:
            parts.append(describe_temporal_node(child, indent + 1))
        return "\n".join(parts)


if __name__ == "__main__":
    test_nested_temporal_patterns()