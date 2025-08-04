#!/usr/bin/env python3
"""Standalone implementation of the compositional parsing fix."""

from typing import Dict, List, Tuple

# Copy minimal necessary structures
VOCAB = {
    "<PAD>": 0,
    "<UNK>": 1,
    "X": 2,
    "Y": 3,
    "Z": 4,
    "means": 5,
    "jump": 6,
    "walk": 7,
    "turn": 8,
    "run": 9,
    "do": 10,
    "twice": 11,
    "thrice": 12,
    "true": 13,
    "and": 14,
    "then": 15,
    "while": 16,
    "or": 17,
}


def separate_bindings_and_execution(
    tokens: List[int],
) -> Tuple[Dict[int, int], List[int]]:
    """Separate variable bindings from execution commands.

    Returns:
        - bindings: Dict mapping variable tokens to action tokens
        - exec_tokens: List of tokens for the execution part only
    """
    do_token = VOCAB["do"]
    means_token = VOCAB["means"]

    # Extract bindings
    bindings = {}
    i = 0
    exec_start = 0

    while i < len(tokens):
        # Look for "do" to mark start of execution
        if tokens[i] == do_token:
            exec_start = i
            break

        # Look for "VAR means ACTION" pattern
        if i + 2 < len(tokens) and tokens[i + 1] == means_token:
            var_token = tokens[i]
            action_token = tokens[i + 2]
            bindings[var_token] = action_token
            i += 3
        else:
            i += 1

    # Extract execution part (everything after "do")
    exec_tokens = tokens[exec_start + 1 :] if exec_start < len(tokens) else []

    return bindings, exec_tokens


def parse_compositional_structure(tokens: List[int]) -> Dict:
    """Parse the compositional structure of execution tokens.

    Returns a simple dict representation of the parse tree.
    """
    and_token = VOCAB.get("and", -1)
    then_token = VOCAB.get("then", -1)
    or_token = VOCAB.get("or", -1)
    while_token = VOCAB.get("while", -1)

    # Find operators with precedence
    operators = []
    for i, token in enumerate(tokens):
        if token == then_token:
            operators.append((i, "then", 3))
        elif token == and_token:
            operators.append((i, "and", 2))
        elif token == or_token:
            operators.append((i, "or", 1))
        elif token == while_token:
            operators.append((i, "while", 4))

    # No operators - it's a leaf
    if not operators:
        return {"type": "leaf", "tokens": tokens}

    # Find operator with lowest precedence (rightmost if tie)
    operators.sort(key=lambda x: (x[2], -x[0]))
    pos, op_type, _ = operators[0]

    # Split at operator
    left_tokens = tokens[:pos]
    right_tokens = tokens[pos + 1 :]

    result = {"type": op_type}
    if left_tokens:
        result["left"] = parse_compositional_structure(left_tokens)
    if right_tokens:
        result["right"] = parse_compositional_structure(right_tokens)

    return result


def resolve_and_execute(tree: Dict, bindings: Dict[int, int]) -> List[str]:
    """Resolve variables and execute the parse tree."""
    # Reverse vocab for string lookup
    vocab_reverse = {v: k for k, v in VOCAB.items()}

    if tree["type"] == "leaf":
        # Resolve variables in leaf
        actions = []
        for token in tree["tokens"]:
            # Check if it's a variable that needs resolution
            if token in bindings:
                action_token = bindings[token]
                action_name = vocab_reverse.get(action_token, "").upper()
                if action_name in ["JUMP", "WALK", "TURN", "RUN"]:
                    actions.append(action_name)
            else:
                # Direct action
                action_name = vocab_reverse.get(token, "").upper()
                if action_name in ["JUMP", "WALK", "TURN", "RUN"]:
                    actions.append(action_name)
        return actions

    elif tree["type"] == "and":
        # Parallel execution - interleave actions
        left_actions = (
            resolve_and_execute(tree["left"], bindings) if "left" in tree else []
        )
        right_actions = (
            resolve_and_execute(tree["right"], bindings) if "right" in tree else []
        )
        # For 'and', we execute both (could interleave, but here we'll concatenate)
        return left_actions + right_actions

    elif tree["type"] == "then":
        # Sequential execution
        left_actions = (
            resolve_and_execute(tree["left"], bindings) if "left" in tree else []
        )
        right_actions = (
            resolve_and_execute(tree["right"], bindings) if "right" in tree else []
        )
        return left_actions + right_actions

    return []


def process_command(command: str) -> Tuple[List[str], Dict]:
    """Process a complete command with bindings and execution."""
    # Tokenize
    tokens = [VOCAB.get(word, VOCAB["<PAD>"]) for word in command.split()]

    # Separate bindings from execution
    bindings, exec_tokens = separate_bindings_and_execution(tokens)

    # Parse execution structure
    tree = parse_compositional_structure(exec_tokens)

    # Resolve and execute
    actions = resolve_and_execute(tree, bindings)

    # Debug info
    vocab_reverse = {v: k for k, v in VOCAB.items()}
    debug_info = {
        "bindings": {
            vocab_reverse.get(k, ""): vocab_reverse.get(v, "")
            for k, v in bindings.items()
        },
        "exec_part": " ".join([vocab_reverse.get(t, "") for t in exec_tokens]),
        "tree": tree,
    }

    return actions, debug_info


# Test the implementation
if __name__ == "__main__":
    print("=== Testing Compositional Parsing Fix ===\n")

    test_cases = [
        ("do X and Y", None),
        ("X means jump Y means walk do X and Y", ["JUMP", "WALK"]),
        ("X means walk Y means jump do X and Y then do X", ["WALK", "JUMP", "WALK"]),
        ("X means jump do X and X", ["JUMP", "JUMP"]),
    ]

    for command, expected in test_cases:
        print(f"Command: {command}")
        if expected:
            print(f"Expected: {expected}")

        actions, debug = process_command(command)

        print(f"Bindings: {debug['bindings']}")
        print(f"Execution: {debug['exec_part']}")
        print(f"Tree: {debug['tree']}")
        print(f"Result: {actions}")

        if expected:
            print(f"Correct: {actions == expected}")

        print("-" * 60 + "\n")
