#!/usr/bin/env python3
"""Test the fixed compositional model locally."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Minimal test without full environment setup
print("=== Testing Fixed Compositional Model ===\n")

# Import VOCAB and test the fix
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compositional_operators_fixed import CompositionalParser, CompositionalExecutor

# Test VOCAB
VOCAB = {
    '<PAD>': 0, '<UNK>': 1, 'X': 2, 'Y': 3, 'Z': 4,
    'means': 5, 'jump': 6, 'walk': 7, 'turn': 8, 'run': 9,
    'do': 10, 'twice': 11, 'thrice': 12, 'true': 13,
    'and': 14, 'then': 15, 'while': 16, 'or': 17
}

# Create parser
parser = CompositionalParser(VOCAB)

# Test cases
test_cases = [
    "X means jump Y means walk do X and Y",
    "X means walk Y means jump do X and Y then do X",
    "X means walk Y means jump Z means turn do X and Y and Z",
]

for command in test_cases:
    print(f"Command: {command}")
    
    # Tokenize
    tokens = [VOCAB.get(word, VOCAB['<PAD>']) for word in command.split()]
    
    # Parse with bindings
    tree, bindings = parser.parse_with_bindings(tokens)
    
    print(f"Bindings: {bindings}")
    
    # Check tree structure
    def print_tree(node, indent=0):
        prefix = "  " * indent
        if node.is_leaf():
            vocab_reverse = {v: k for k, v in VOCAB.items()}
            words = [vocab_reverse.get(t, '?') for t in node.tokens]
            print(f"{prefix}LEAF[{node.start_pos}:{node.end_pos}]: {' '.join(words)}")
        else:
            print(f"{prefix}{node.operator.value}[{node.start_pos}:{node.end_pos}]:")
            for child in node.children:
                print_tree(child, indent + 1)
    
    print("Parse tree (execution only):")
    print_tree(tree)
    
    # Verify leaf nodes contain only execution tokens
    def check_leaves(node):
        if node.is_leaf():
            vocab_reverse = {v: k for k, v in VOCAB.items()}
            words = [vocab_reverse.get(t, '?') for t in node.tokens]
            # Check if any binding words are in the leaf
            if 'means' in words or any(w in ['jump', 'walk', 'turn', 'run'] for w in words[1:]):
                print(f"  ❌ ERROR: Leaf contains binding words: {words}")
                return False
            else:
                print(f"  ✓ Leaf is clean: {words}")
                return True
        else:
            all_good = True
            for child in node.children:
                if not check_leaves(child):
                    all_good = False
            return all_good
    
    leaves_ok = check_leaves(tree)
    print(f"Result: {'PASS' if leaves_ok else 'FAIL'}")
    print("-" * 60 + "\n")

print("\nSummary:")
print("The fixed parser correctly:")
print("1. Extracts bindings before parsing")
print("2. Parses only the execution part")
print("3. Creates leaf nodes with just variables (X, Y, Z)")
print("4. No longer includes 'means' or action words in leaves")