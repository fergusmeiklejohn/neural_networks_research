#!/usr/bin/env python3
"""Final complete fix for compositional model handling all edge cases."""

import mlx.core as mx
import mlx.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OperatorType(Enum):
    """Types of compositional operators."""
    AND = "AND"
    OR = "OR"
    WHILE = "WHILE"
    THEN = "THEN"


@dataclass
class ParseNode:
    """Node in parse tree."""
    operator: Optional[OperatorType]
    children: List['ParseNode']
    tokens: Optional[List[int]]  # For leaf nodes
    start_pos: int
    end_pos: int
    modifier: Optional[str] = None  # For 'twice', 'thrice'
    
    def is_leaf(self) -> bool:
        return self.operator is None


class FinalCompositionalParser:
    """Parser that handles ALL edge cases including modifiers."""
    
    def __init__(self, vocab: Dict[str, int]):
        self.vocab = vocab
        self.do_token = vocab.get('do', -1)
        self.means_token = vocab.get('means', -1)
        self.while_token = vocab.get('while', -1)
        self.true_token = vocab.get('true', -1)
        self.twice_token = vocab.get('twice', -1)
        self.thrice_token = vocab.get('thrice', -1)
        
        # Map tokens to operator types
        self.operator_tokens = {
            vocab.get('and', -1): OperatorType.AND,
            vocab.get('or', -1): OperatorType.OR,
            vocab.get('while', -1): OperatorType.WHILE,
            vocab.get('then', -1): OperatorType.THEN,
        }
        
        # Operator precedence
        self.precedence = {
            OperatorType.WHILE: 1,
            OperatorType.AND: 2,
            OperatorType.OR: 2,
            OperatorType.THEN: 3,
        }
    
    def parse_with_bindings(self, tokens: mx.array) -> Tuple[ParseNode, Dict[int, str]]:
        """Parse command, extracting bindings and execution tree."""
        token_list = self._tokens_to_list(tokens)
        logger.debug(f"Parsing: {self._tokens_to_words(token_list)}")
        
        # Extract bindings and execution ranges
        bindings, execution_ranges = self._extract_all_bindings(token_list)
        
        # If no execution ranges, return empty tree
        if not execution_ranges:
            return ParseNode(None, [], [], 0, 0), bindings
        
        # Parse execution ranges
        tree = self._parse_execution_ranges(token_list, execution_ranges)
        return tree, bindings
    
    def _tokens_to_list(self, tokens: mx.array) -> List[int]:
        """Convert MLX array to Python list."""
        if len(tokens.shape) > 1:
            tokens = tokens[0]
        import numpy as np
        token_array = np.array(tokens)
        if len(token_array.shape) > 1:
            token_array = token_array.flatten()
        return [int(t) for t in token_array]
    
    def _tokens_to_words(self, token_list: List[int]) -> str:
        """Convert tokens back to words for debugging."""
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        return ' '.join(reverse_vocab.get(t, f'?{t}') for t in token_list)
    
    def _extract_all_bindings(self, token_list: List[int]) -> Tuple[Dict[int, str], List[Tuple[int, int, Optional[str]]]]:
        """Extract bindings and execution ranges with modifiers."""
        bindings = {}
        execution_ranges = []
        i = 0
        
        while i < len(token_list):
            if i + 2 < len(token_list) and token_list[i + 1] == self.means_token:
                # Found binding
                var_token = token_list[i]
                action_token = token_list[i + 2]
                action_str = self._token_to_action(action_token)
                
                if action_str:
                    bindings[var_token] = action_str
                
                i += 3
                
                # Look for 'do' after binding
                if i < len(token_list) and token_list[i] == self.do_token:
                    start = i + 1
                    end = start
                    modifier = None
                    
                    # Find end and check for modifiers
                    while end < len(token_list):
                        if end + 1 < len(token_list) and token_list[end + 1] == self.means_token:
                            break
                        
                        # Check for modifier
                        if token_list[end] == self.twice_token:
                            modifier = 'twice'
                        elif token_list[end] == self.thrice_token:
                            modifier = 'thrice'
                        
                        end += 1
                    
                    if end > start:
                        execution_ranges.append((start, end, modifier))
                    
                    i = end
                
            elif token_list[i] == self.while_token:
                # Special case: "while true do X"
                if i + 3 < len(token_list) and token_list[i + 2] == self.do_token:
                    start = i
                    end = len(token_list)
                    
                    # Find where this while block ends
                    for j in range(i + 4, len(token_list)):
                        if j + 1 < len(token_list) and token_list[j + 1] == self.means_token:
                            end = j
                            break
                    
                    execution_ranges.append((start, end, None))
                    i = end
                else:
                    i += 1
                    
            elif token_list[i] == self.do_token and not any(start <= i < end for start, end, _ in execution_ranges):
                # Standalone 'do'
                start = i + 1
                end = start
                modifier = None
                
                while end < len(token_list):
                    if end + 1 < len(token_list) and token_list[end + 1] == self.means_token:
                        break
                    
                    if token_list[end] == self.twice_token:
                        modifier = 'twice'
                    elif token_list[end] == self.thrice_token:
                        modifier = 'thrice'
                    
                    end += 1
                
                if end > start:
                    execution_ranges.append((start, end, modifier))
                
                i = end
            else:
                i += 1
        
        return bindings, execution_ranges
    
    def _parse_execution_ranges(self, token_list: List[int], 
                               ranges: List[Tuple[int, int, Optional[str]]]) -> ParseNode:
        """Parse execution ranges with modifiers."""
        if len(ranges) == 1:
            start, end, modifier = ranges[0]
            
            # Check if this is a while block
            if start > 0 and token_list[start - 3] == self.while_token:
                return self._parse_while_expression(token_list, start - 3, end)
            else:
                node = self._parse_expression(token_list, start, end)
                
                # Apply modifier if present
                if modifier and node.is_leaf():
                    node.modifier = modifier
                
                return node
        else:
            # Multiple ranges - connect with THEN
            nodes = []
            for start, end, modifier in ranges:
                if start > 0 and token_list[start - 3] == self.while_token:
                    nodes.append(self._parse_while_expression(token_list, start - 3, end))
                else:
                    node = self._parse_expression(token_list, start, end)
                    if modifier and node.is_leaf():
                        node.modifier = modifier
                    nodes.append(node)
            
            # Chain with THEN
            result = nodes[0]
            for node in nodes[1:]:
                result = ParseNode(OperatorType.THEN, [result, node], None, 
                                 result.start_pos, node.end_pos)
            
            return result
    
    def _parse_while_expression(self, tokens: List[int], start: int, end: int) -> ParseNode:
        """Parse while expression."""
        true_node = ParseNode(None, [], [tokens[start + 1]], start + 1, start + 2)
        expr_start = start + 3
        expr_node = self._parse_expression(tokens, expr_start, end)
        return ParseNode(OperatorType.WHILE, [true_node, expr_node], None, start, end)
    
    def _parse_expression(self, tokens: List[int], start: int, end: int) -> ParseNode:
        """Parse expression, preserving modifiers on leaf nodes."""
        # Find operators (but not modifiers)
        split_pos = -1
        split_op = None
        min_precedence = float('inf')
        
        for i in range(start, end):
            if tokens[i] in self.operator_tokens:
                op_type = self.operator_tokens[tokens[i]]
                prec = self.precedence[op_type]
                
                if prec <= min_precedence:
                    min_precedence = prec
                    split_pos = i
                    split_op = op_type
        
        if split_pos == -1:
            # No operator - create leaf, keeping only the variable
            clean_tokens = []
            modifier = None
            
            for i in range(start, end):
                if tokens[i] == self.twice_token:
                    modifier = 'twice'
                elif tokens[i] == self.thrice_token:
                    modifier = 'thrice'
                else:
                    clean_tokens.append(tokens[i])
            
            node = ParseNode(None, [], clean_tokens, start, end)
            node.modifier = modifier
            return node
        
        # Create internal node
        left = self._parse_expression(tokens, start, split_pos)
        right = self._parse_expression(tokens, split_pos + 1, end)
        
        return ParseNode(split_op, [left, right], None, start, end)
    
    def _token_to_action(self, token: int) -> Optional[str]:
        """Convert action token to action string."""
        for word, idx in self.vocab.items():
            if idx == token and word in ['jump', 'walk', 'turn', 'run']:
                return word.upper()
        return None


class FinalExecutor:
    """Executor that handles modifiers correctly."""
    
    def __init__(self, model):
        self.model = model
        self.actions = model.actions
        self.num_actions = model.num_actions
    
    def execute(self, tree: ParseNode, bindings: Dict[int, str]) -> List[mx.array]:
        """Execute parse tree with bindings and modifiers."""
        outputs = []
        self._execute_node(tree, bindings, outputs)
        return outputs
    
    def _execute_node(self, node: ParseNode, bindings: Dict[int, str], 
                      outputs: List[mx.array]):
        """Execute node, handling modifiers."""
        if node.is_leaf():
            # Get base actions
            base_outputs = []
            for token in node.tokens:
                if token in bindings:
                    action_name = bindings[token]
                    action_idx = self.actions.get(action_name, 0)
                    
                    action_vec = mx.zeros(self.num_actions)
                    action_vec = mx.where(
                        mx.arange(self.num_actions) == action_idx, 
                        1.0, action_vec
                    )
                    base_outputs.append(action_vec)
            
            # Apply modifier
            if node.modifier == 'twice':
                for _ in range(2):
                    outputs.extend(base_outputs)
            elif node.modifier == 'thrice':
                for _ in range(3):
                    outputs.extend(base_outputs)
            else:
                outputs.extend(base_outputs)
        else:
            # Handle operators
            if node.operator == OperatorType.AND:
                self._execute_node(node.children[0], bindings, outputs)
                self._execute_node(node.children[1], bindings, outputs)
                
            elif node.operator == OperatorType.OR:
                self._execute_node(node.children[0], bindings, outputs)
                
            elif node.operator == OperatorType.THEN:
                self._execute_node(node.children[0], bindings, outputs)
                self._execute_node(node.children[1], bindings, outputs)
                
            elif node.operator == OperatorType.WHILE:
                # Execute 3 times
                for _ in range(3):
                    self._execute_node(node.children[1], bindings, outputs)


class FinalCompositionalModel(nn.Module):
    """Final model with complete fix."""
    
    def __init__(self, vocab_size: int, num_actions: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_actions = num_actions
        self.vocab = None
        self.actions = None
        self.parser = None
        self.executor = None
    
    def set_vocab_and_actions(self, vocab: Dict[str, int], actions: Dict[str, int]):
        """Initialize with vocab and actions."""
        self.vocab = vocab
        self.actions = actions
        self.parser = FinalCompositionalParser(vocab)
        self.executor = FinalExecutor(self)
    
    def __call__(self, inputs: Dict[str, mx.array]) -> mx.array:
        """Forward pass."""
        command_ids = inputs['command']
        tree, bindings = self.parser.parse_with_bindings(command_ids)
        outputs = self.executor.execute(tree, bindings)
        
        if outputs:
            return mx.stack(outputs)
        else:
            # Return empty tensor with correct shape
            return mx.zeros((0, self.num_actions))


def test_final_fix():
    """Test all cases including the problematic ones."""
    print("=== Testing FINAL Compositional Fix ===\n")
    
    # Define vocab and actions
    VOCAB = {
        '<PAD>': 0, '<UNK>': 1, 'X': 2, 'Y': 3, 'Z': 4,
        'means': 5, 'jump': 6, 'walk': 7, 'turn': 8, 'run': 9,
        'do': 10, 'twice': 11, 'thrice': 12, 'true': 13,
        'and': 14, 'then': 15, 'while': 16, 'or': 17
    }
    
    ACTIONS = {'JUMP': 0, 'WALK': 1, 'TURN': 2, 'RUN': 3}
    
    # Create model
    model = FinalCompositionalModel(len(VOCAB), len(ACTIONS))
    model.set_vocab_and_actions(VOCAB, ACTIONS)
    
    # Original failing test cases
    test_cases = [
        {
            "command": "X means run Y means turn do Y and X",
            "expected": ['TURN', 'RUN'],
            "description": "Basic AND operator"
        },
        {
            "command": "X means walk Y means jump do X and Y then X",
            "expected": ['WALK', 'JUMP', 'WALK'],
            "description": "AND followed by single variable"
        },
        {
            "command": "X means jump do X then Y means walk do Y and X",
            "expected": ['JUMP', 'WALK', 'JUMP'],
            "description": "Sequential bindings with execution"
        },
        {
            "command": "X means jump do X twice and Y means walk do Y",
            "expected": ['JUMP', 'JUMP', 'WALK'],
            "description": "Modifier 'twice' handling"
        },
        {
            "command": "X means jump while true do X",
            "expected": ['JUMP', 'JUMP', 'JUMP'],
            "description": "While loop (3 iterations)"
        },
        {
            "command": "X means jump Y means walk",
            "expected": [],
            "description": "No execution (no 'do')"
        }
    ]
    
    all_pass = True
    
    for test in test_cases:
        print(f"Test: {test['description']}")
        print(f"Command: {test['command']}")
        print(f"Expected: {test['expected']}")
        
        # Tokenize
        tokens = [VOCAB.get(word, VOCAB['<PAD>']) for word in test['command'].split()]
        inputs = {'command': mx.array([tokens])}
        
        # Execute
        outputs = model(inputs)
        
        # Convert to actions
        predicted_actions = []
        for i in range(outputs.shape[0]):
            action_idx = mx.argmax(outputs[i])
            for name, idx in ACTIONS.items():
                if idx == int(action_idx):
                    predicted_actions.append(name)
                    break
        
        print(f"Got: {predicted_actions}")
        
        # Check
        if predicted_actions == test['expected']:
            print("✅ PASS")
        else:
            print("❌ FAIL")
            all_pass = False
        
        print("-" * 60)
    
    print("\n" + "="*60)
    if all_pass:
        print("🎉 ALL TESTS PASS! The compositional model is FULLY FIXED! 🎉")
    else:
        print("❌ Some tests still failing")
    
    return all_pass


if __name__ == "__main__":
    test_final_fix()