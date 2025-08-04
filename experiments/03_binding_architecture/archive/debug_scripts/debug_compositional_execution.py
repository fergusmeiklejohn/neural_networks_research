#!/usr/bin/env python3
"""Debug script to understand compositional execution issues."""

from utils.imports import setup_project_paths

setup_project_paths()

from utils.config import setup_environment

config = setup_environment()

import mlx.core as mx
from compositional_operators import CompositionalParser, ParseNode
from train_integrated_model import VOCAB

# Ensure operators are in VOCAB
for op in ["and", "then", "while", "or", "do", "true"]:
    if op not in VOCAB:
        VOCAB[op] = len(VOCAB)

parser = CompositionalParser(VOCAB)

# Test cases that are failing
test_cases = [
    "X means jump Y means walk do X and Y",
    "X means walk Y means jump do X and Y then do X",
    "X means walk Y means jump Z means turn do X and Y and Z",
]

print("=== Debugging Compositional Execution ===\n")

for command in test_cases:
    print(f"Command: {command}")
    tokens = [VOCAB.get(word, VOCAB["<PAD>"]) for word in command.split()]
    print(f"Token IDs: {tokens}")

    # Parse
    tree = parser.parse(mx.array(tokens))

    def debug_tree(node: ParseNode, indent=0):
        prefix = "  " * indent
        if node.is_leaf():
            words = []
            for tid in node.tokens:
                for word, wid in VOCAB.items():
                    if wid == tid:
                        words.append(word)
                        break
            print(f"{prefix}LEAF[{node.start_pos}:{node.end_pos}]: {' '.join(words)}")
        else:
            print(f"{prefix}{node.operator.value}[{node.start_pos}:{node.end_pos}]:")
            for child in node.children:
                debug_tree(child, indent + 1)

    debug_tree(tree)

    # Analyze segments
    print("\nSegments for execution:")

    def get_segments(node: ParseNode):
        if node.is_leaf():
            print(
                f"  Leaf segment: tokens[{node.start_pos}:{node.end_pos}] = {node.tokens}"
            )
            # Map tokens back to words
            words = []
            for tid in node.tokens:
                for word, wid in VOCAB.items():
                    if wid == tid:
                        words.append(word)
                        break
            print(f"    Words: {' '.join(words)}")

            # Check for variable bindings
            if "means" in words:
                var_idx = words.index("means") - 1
                action_idx = words.index("means") + 1
                if var_idx >= 0 and action_idx < len(words):
                    print(f"    Binding: {words[var_idx]} -> {words[action_idx]}")

            # Check for do commands
            if "do" in words:
                do_idx = words.index("do")
                if do_idx + 1 < len(words):
                    print(f"    Execute: {words[do_idx + 1]}")
        else:
            print(f"  Operator {node.operator.value}:")
            for child in node.children:
                get_segments(child)

    get_segments(tree)
    print("\n" + "=" * 50 + "\n")
