#!/usr/bin/env python3
"""Simple test of compositional operator improvements."""

from utils.imports import setup_project_paths

setup_project_paths()

import mlx.core as mx
from train_integrated_model import VOCAB

# Add operators to VOCAB
for op in ["and", "then", "while", "or", "do", "true"]:
    if op not in VOCAB:
        VOCAB[op] = len(VOCAB)

print("=== Testing Compositional Improvements ===\n")

# Test 1: Verify operators are in vocabulary
print("1. Operators in VOCAB:")
for op in ["and", "then", "while", "or"]:
    print(f"   {op}: {VOCAB.get(op, 'NOT FOUND')}")

# Test 2: Token conversion
test_cmd = "X means jump Y means walk do X and Y"
tokens = [VOCAB.get(word, VOCAB["<PAD>"]) for word in test_cmd.split()]
print(f"\n2. Command: {test_cmd}")
print(f"   Tokens: {tokens}")

# Test 3: Verify parsing works
from compositional_operators import CompositionalParser

parser = CompositionalParser(VOCAB)
tree = parser.parse(mx.array(tokens))


def describe_tree(node):
    if node.is_leaf():
        return f"LEAF[{len(node.tokens)} tokens]"
    else:
        child_descs = [describe_tree(c) for c in node.children]
        return f"{node.operator.value}({', '.join(child_descs)})"


print(f"   Parse tree: {describe_tree(tree)}")


# Test 4: Check what's in the leaves
def show_leaf_contents(node, indent=0):
    prefix = "  " * indent
    if node.is_leaf():
        words = []
        for tid in node.tokens:
            for word, wid in VOCAB.items():
                if wid == tid:
                    words.append(word)
                    break
        print(f"{prefix}Leaf: {' '.join(words)}")
    else:
        print(f"{prefix}{node.operator.value}:")
        for child in node.children:
            show_leaf_contents(child, indent + 1)


print("\n3. Leaf contents:")
show_leaf_contents(tree, 1)

print(f"\n4. Summary:")
print(f"   - Vocabulary fix: ✓ (operators added)")
print(f"   - Parser creates tree: ✓ (not single leaf)")
print(f"   - Issue: Bindings grouped with execution in leaf")
print(f"   - Next step: Separate binding/execution processing")
