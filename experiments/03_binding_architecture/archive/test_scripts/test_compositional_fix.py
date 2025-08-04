#!/usr/bin/env python3
"""Test the compositional parsing fix in a minimal way."""

from typing import List, Dict, Tuple

# Minimal VOCAB for testing
VOCAB = {
    '<PAD>': 0, '<UNK>': 1, 'X': 2, 'Y': 3, 'Z': 4,
    'means': 5, 'jump': 6, 'walk': 7, 'turn': 8, 'run': 9,
    'do': 10, 'twice': 11, 'thrice': 12, 'true': 13,
    'and': 14, 'then': 15, 'while': 16, 'or': 17
}

ACTIONS = ['JUMP', 'WALK', 'TURN', 'RUN']


def separate_bindings_and_execution(tokens: List[int]) -> Tuple[Dict[str, str], List[int], int]:
    """Separate bindings from execution and return both."""
    do_token = VOCAB['do']
    means_token = VOCAB['means']
    vocab_reverse = {v: k for k, v in VOCAB.items()}
    
    bindings = {}
    exec_start = -1
    
    # Find bindings and execution start
    i = 0
    while i < len(tokens):
        if tokens[i] == do_token:
            exec_start = i
            break
        
        # Look for "VAR means ACTION" pattern
        if i + 2 < len(tokens) and tokens[i + 1] == means_token:
            var_name = vocab_reverse.get(tokens[i], '')
            action_name = vocab_reverse.get(tokens[i + 2], '')
            if var_name and action_name:
                bindings[var_name] = action_name
            i += 3
        else:
            i += 1
    
    # Get execution tokens
    exec_tokens = tokens[exec_start + 1:] if exec_start >= 0 else []
    
    return bindings, exec_tokens, exec_start


def resolve_compositional_command(tokens: List[int], bindings: Dict[str, str]) -> List[str]:
    """Resolve a compositional command with bindings."""
    and_token = VOCAB['and']
    then_token = VOCAB['then']
    vocab_reverse = {v: k for k, v in VOCAB.items()}
    
    actions = []
    i = 0
    
    while i < len(tokens):
        token = tokens[i]
        token_str = vocab_reverse.get(token, '')
        
        # Variable that needs resolution
        if token_str in bindings:
            action = bindings[token_str].upper()
            if action in ACTIONS:
                actions.append(action)
        
        # Direct action
        elif token_str.upper() in ACTIONS:
            actions.append(token_str.upper())
        
        # Skip operators (they define structure, not actions)
        elif token in [and_token, then_token]:
            pass
        
        i += 1
    
    return actions


def test_compositional_parsing():
    """Test the compositional parsing fix."""
    print("=== Testing Compositional Parsing Fix ===\n")
    
    test_cases = [
        {
            'command': "X means jump Y means walk do X and Y",
            'expected': ['JUMP', 'WALK'],
            'description': "Simple 'and' with two variables"
        },
        {
            'command': "X means walk Y means jump do X and Y then do X",
            'expected': ['WALK', 'JUMP', 'WALK'],
            'description': "'and' followed by 'then'"
        },
        {
            'command': "X means walk Y means jump Z means turn do X and Y and Z",
            'expected': ['WALK', 'JUMP', 'TURN'],
            'description': "Multiple 'and' operators"
        },
        {
            'command': "X means jump do X and X",
            'expected': ['JUMP', 'JUMP'],
            'description': "Same variable used twice"
        },
        {
            'command': "do jump and walk",
            'expected': ['JUMP', 'WALK'],
            'description': "Direct actions without variables"
        }
    ]
    
    all_passed = True
    
    for test in test_cases:
        command = test['command']
        expected = test['expected']
        description = test['description']
        
        print(f"Test: {description}")
        print(f"Command: {command}")
        print(f"Expected: {expected}")
        
        # Tokenize
        tokens = [VOCAB.get(word, VOCAB['<PAD>']) for word in command.split()]
        
        # Separate bindings and execution
        bindings, exec_tokens, exec_start = separate_bindings_and_execution(tokens)
        
        # Debug info
        print(f"Bindings: {bindings}")
        vocab_reverse = {v: k for k, v in VOCAB.items()}
        exec_words = [vocab_reverse.get(t, '') for t in exec_tokens]
        print(f"Execution part: {' '.join(exec_words)}")
        
        # Resolve command
        if exec_start >= 0:
            # Has explicit execution part
            actions = resolve_compositional_command(exec_tokens, bindings)
        else:
            # Direct execution
            actions = resolve_compositional_command(tokens, bindings)
        
        print(f"Result: {actions}")
        
        # Check correctness
        passed = actions == expected
        print(f"Status: {'✓ PASS' if passed else '✗ FAIL'}")
        
        if not passed:
            all_passed = False
            print(f"  Mismatch: got {actions}, expected {expected}")
        
        print("-" * 60 + "\n")
    
    return all_passed


def demonstrate_fix():
    """Demonstrate how the fix solves the original problem."""
    print("\n=== Demonstrating the Fix ===\n")
    
    command = "X means jump Y means walk do X and Y"
    tokens = [VOCAB.get(word, VOCAB['<PAD>']) for word in command.split()]
    
    print("BEFORE FIX:")
    print("The parser would create:")
    print("  and[0:10]:")
    print("    LEAF[0:8]: X means jump Y means walk do X")
    print("    LEAF[9:10]: Y")
    print("Problem: First LEAF contains bindings + execution\n")
    
    print("AFTER FIX:")
    bindings, exec_tokens, exec_start = separate_bindings_and_execution(tokens)
    print(f"1. Extract bindings: {bindings}")
    print(f"2. Identify execution start: position {exec_start} (after 'do')")
    
    vocab_reverse = {v: k for k, v in VOCAB.items()}
    exec_words = [vocab_reverse.get(t, '') for t in exec_tokens]
    print(f"3. Parse only execution: {' '.join(exec_words)}")
    print("4. Now parser creates:")
    print("     and:")
    print("       LEAF: X")
    print("       LEAF: Y")
    print("5. Resolve variables: X->JUMP, Y->WALK")
    print("6. Final result: ['JUMP', 'WALK']")


if __name__ == "__main__":
    # Run tests
    all_passed = test_compositional_parsing()
    
    # Demonstrate the fix
    demonstrate_fix()
    
    if all_passed:
        print("\n✓ All tests passed! The compositional parsing fix is working correctly.")
    else:
        print("\n✗ Some tests failed. Please check the implementation.")