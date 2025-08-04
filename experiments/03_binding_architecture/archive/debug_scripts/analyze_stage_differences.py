#!/usr/bin/env python3
"""
Analyze the differences between Stage 2 and Stage 3 to understand accuracy drop
"""

import sys

sys.path.append(".")

from train_binding_curriculum import (
    ACTIONS,
    VOCAB,
    generate_stage2_data,
    generate_stage3_data,
)


def compare_stages():
    """Compare Stage 2 and Stage 3 patterns side by side"""

    print("Stage 2 vs Stage 3 Comparison")
    print("=" * 80)

    # Generate samples from both stages
    stage2_batch = generate_stage2_data(10)
    stage3_batch = generate_stage3_data(10)

    print("\nStage 2 Examples (100% accuracy):")
    print("-" * 40)
    for i in range(5):
        cmd = stage2_batch["command"][i]
        labels = stage2_batch["labels"][i]

        # Decode
        tokens = []
        for token_id in cmd:
            if token_id.item() == VOCAB["<PAD>"]:
                break
            token = [k for k, v in VOCAB.items() if v == token_id.item()][0]
            tokens.append(token)

        action = [k for k, v in ACTIONS.items() if v == labels[0].item()][0]
        print(f"{' '.join(tokens):30} → {action}")

    print("\nStage 3 Examples (37% accuracy):")
    print("-" * 40)
    for i in range(5):
        cmd = stage3_batch["command"][i]
        labels = stage3_batch["labels"][i]

        # Decode
        tokens = []
        for token_id in cmd:
            if token_id.item() == VOCAB["<PAD>"]:
                break
            token = [k for k, v in VOCAB.items() if v == token_id.item()][0]
            tokens.append(token)

        # Get expected actions
        expected = []
        for label_id in labels:
            if label_id.item() == ACTIONS["<PAD>"]:
                break
            action = [k for k, v in ACTIONS.items() if v == label_id.item()][0]
            expected.append(action)

        print(f"{' '.join(tokens):30} → {expected}")

    print("\n" + "=" * 80)
    print("Key Differences:")
    print("=" * 80)

    print("\n1. **Storage Pattern**")
    print("   Stage 2: 'X is jump' - uses 'is' token")
    print("   Stage 3: 'X means jump' - uses 'means' token")
    print("   → Model must handle BOTH storage patterns")

    print("\n2. **Retrieval Pattern**")
    print("   Stage 2: 'recall X' - explicit retrieval command")
    print("   Stage 3: 'do X' - implicit retrieval via 'do'")
    print("   → Different retrieval triggers")

    print("\n3. **Pattern Detection Logic**")
    print("   The model's storage detection looks for 'is' OR 'means'")
    print("   But does the model properly detect both in Stage 3 context?")

    # Test storage pattern detection
    print("\n" + "=" * 80)
    print("Testing Storage Pattern Detection")
    print("=" * 80)

    test_patterns = [
        "X is jump",
        "X means jump",
        "Y is walk recall Y",
        "Y means walk do Y",
        "Z means turn do Z twice",
    ]

    for pattern in test_patterns:
        tokens = pattern.split()
        [VOCAB.get(t, VOCAB["<PAD>"]) for t in tokens]

        # Check which positions would be marked for storage
        storage_positions = []
        for i in range(len(tokens) - 2):
            if tokens[i] in ["X", "Y", "Z"] and tokens[i + 1] in ["is", "means"]:
                storage_positions.append(i)

        print(f"\nPattern: {pattern}")
        print(f"Storage positions: {storage_positions}")
        if storage_positions:
            for pos in storage_positions:
                print(f"  → Store {tokens[pos]} with value {tokens[pos+2]}")

    print("\n" + "=" * 80)
    print("Hypothesis: Why Stage 3 Fails")
    print("=" * 80)

    print("\n1. **Training Order Effect**")
    print("   - Model learns 'is' pattern perfectly in Stage 2")
    print("   - When it sees 'means' in Stage 3, it might not adapt")
    print("   - Previous learning interferes with new pattern")

    print("\n2. **Retrieval Ambiguity**")
    print("   - 'recall X' is explicit - always retrieve")
    print("   - 'do X' could mean execute OR retrieve-then-execute")
    print("   - Model might be confused about when to retrieve")

    print("\n3. **Temporal Pattern Interference**")
    print("   - 30% of Stage 3 has temporal modifiers")
    print("   - Model might be trying to apply temporal logic to non-temporal patterns")
    print("   - Or failing to apply it when needed")

    print("\n4. **Gradient Interference**")
    print("   - Learning three different tasks sequentially")
    print("   - Later stages might corrupt earlier learning")
    print("   - Need better continual learning strategy")

    print("\nSuggested Solutions:")
    print("1. Mix stages during training instead of sequential")
    print("2. Use consistent storage patterns ('means' everywhere)")
    print("3. Add explicit stage indicators to inputs")
    print("4. Implement elastic weight consolidation")
    print("5. Create intermediate stages for gradual transition")


if __name__ == "__main__":
    compare_stages()
