#!/usr/bin/env python3
"""
Create Proper Validation Sets with Modified Examples

This script addresses the evaluation illusion discovered in our experiments
where validation sets contained no modified examples, masking complete failure
on modification tasks.

The new validation sets will include:
1. Base SCAN performance (unmodified)
2. Each modification type separately
3. Mixed modifications
4. Held-out modifications never seen in training
"""

import json
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from modification_generator import ModificationGenerator
from scan_data_loader import SCANDataLoader, SCANSample


def create_comprehensive_validation_sets(
    base_samples: List[SCANSample], output_dir: Path, seed: int = 42
) -> Dict[str, List[Dict]]:
    """
    Create validation sets that properly test both base and modified performance.

    Returns dict with keys:
    - val_base: Unmodified SCAN examples
    - val_mod_walk_skip: Examples with walk→skip modification
    - val_mod_jump_hop: Examples with jump→hop modification
    - val_mod_look_scan: Examples with look→scan modification
    - val_mod_mixed: Mix of different modifications
    - val_mod_unseen: Modifications never shown in training
    - val_mod_composed: Multiple modifications applied together
    """
    random.seed(seed)
    np.random.seed(seed)

    print("Creating comprehensive validation sets...")

    # Initialize modification generator
    mod_gen = ModificationGenerator()

    # Shuffle base samples
    samples = base_samples.copy()
    random.shuffle(samples)

    # Allocate samples for different validation sets
    n_per_set = min(500, len(samples) // 10)  # Max 500 per set

    validation_sets = {
        "val_base": [],
        "val_mod_walk_skip": [],
        "val_mod_jump_hop": [],
        "val_mod_look_scan": [],
        "val_mod_left_right": [],
        "val_mod_mixed": [],
        "val_mod_unseen": [],
        "val_mod_composed": [],
    }

    # 1. Base validation set (unmodified)
    base_samples_val = samples[:n_per_set]
    for sample in base_samples_val:
        validation_sets["val_base"].append(
            {
                "command": sample.command,
                "action": sample.action,
                "modification": "none",
                "modification_tokens": [0, 0, 0],  # No modification signal
            }
        )

    # 2. Specific modification validation sets
    modification_rules = [
        ("val_mod_walk_skip", {"walk": "skip"}, "walk → skip"),
        ("val_mod_jump_hop", {"jump": "hop"}, "jump → hop"),
        ("val_mod_look_scan", {"look": "scan"}, "look → scan"),
        ("val_mod_left_right", {"left": "right", "right": "left"}, "left ↔ right"),
    ]

    for val_key, swap_rules, description in modification_rules:
        # Find samples that contain the relevant words
        relevant_samples = [
            s
            for s in samples
            if any(word in s.command.split() for word in swap_rules.keys())
        ]
        selected = relevant_samples[:n_per_set]

        for sample in selected:
            # Apply modification with proper action token mapping
            mod_command = sample.command
            mod_action = sample.action

            # Apply swaps to command
            for orig, repl in swap_rules.items():
                mod_command = mod_command.replace(orig, repl)

            # Apply swaps to action with new tokens
            action_mappings = {
                "walk": ("I_WALK", "I_SKIP"),
                "jump": ("I_JUMP", "I_HOP"),
                "look": ("I_LOOK", "I_SCAN"),
                "left": ("LEFT", "RIGHT"),
                "right": ("RIGHT", "LEFT"),
            }

            for orig, repl in swap_rules.items():
                if orig in action_mappings:
                    old_action, new_action = action_mappings[orig]
                    mod_action = mod_action.replace(old_action, new_action)
                elif orig == "right" and repl == "left":
                    # Handle the direction swap case
                    mod_action = mod_action.replace("I_TURN_RIGHT", "I_TURN_LEFT_TEMP")
                    mod_action = mod_action.replace("I_TURN_LEFT", "I_TURN_RIGHT")
                    mod_action = mod_action.replace("I_TURN_LEFT_TEMP", "I_TURN_LEFT")

            # Skip if no actual modification occurred
            if mod_command == sample.command:
                continue

            validation_sets[val_key].append(
                {
                    "original_command": sample.command,
                    "original_action": sample.action,
                    "command": mod_command,
                    "action": mod_action,
                    "modification": description,
                    "modification_tokens": encode_modification(description),
                }
            )

    # 3. Mixed modifications validation set
    all_mods = []
    for key in [
        "val_mod_walk_skip",
        "val_mod_jump_hop",
        "val_mod_look_scan",
        "val_mod_left_right",
    ]:
        all_mods.extend(validation_sets[key][: n_per_set // 4])
    random.shuffle(all_mods)
    validation_sets["val_mod_mixed"] = all_mods[:n_per_set]

    # 4. Unseen modifications (not used in training)
    unseen_rules = [
        ({"turn": "spin"}, "turn → spin"),
        ({"run": "dash"}, "run → dash"),
        ({"twice": "double"}, "twice → double"),
        ({"around": "beside"}, "around → beside"),
    ]

    for swap_rules, description in unseen_rules:
        relevant_samples = [
            s
            for s in samples
            if any(word in s.command.split() for word in swap_rules.keys())
        ]
        selected = relevant_samples[: n_per_set // len(unseen_rules)]

        for sample in selected:
            mod_command, mod_action = apply_unseen_modification(sample, swap_rules)

            if mod_command != sample.command:
                validation_sets["val_mod_unseen"].append(
                    {
                        "original_command": sample.command,
                        "original_action": sample.action,
                        "command": mod_command,
                        "action": mod_action,
                        "modification": description,
                        "modification_tokens": encode_modification(description),
                    }
                )

    # 5. Composed modifications (multiple rules at once)
    composed_rules = [
        ({"walk": "skip", "left": "right"}, "walk → skip + left ↔ right"),
        ({"jump": "hop", "twice": "thrice"}, "jump → hop + twice → thrice"),
        ({"look": "scan", "around": "beside"}, "look → scan + around → beside"),
    ]

    for swap_rules, description in composed_rules:
        relevant_samples = [
            s
            for s in samples
            if any(word in s.command.split() for word in swap_rules.keys())
        ]
        selected = relevant_samples[: n_per_set // len(composed_rules)]

        for sample in selected:
            mod_command = sample.command
            mod_action = sample.action

            # Apply all swaps
            for orig, repl in swap_rules.items():
                mod_command = mod_command.replace(orig, repl)
                if orig in mod_gen.ACTION_MAP and repl in ["skip", "hop", "scan"]:
                    # Handle new action mappings
                    action_map = {"skip": "I_SKIP", "hop": "I_HOP", "scan": "I_SCAN"}
                    mod_gen.ACTION_MAP.get(orig, orig.upper())
                    mod_action = mod_action.replace(
                        f"I_{orig.upper()}", action_map.get(repl, f"I_{repl.upper()}")
                    )

            if mod_command != sample.command:
                validation_sets["val_mod_composed"].append(
                    {
                        "original_command": sample.command,
                        "original_action": sample.action,
                        "command": mod_command,
                        "action": mod_action,
                        "modification": description,
                        "modification_tokens": encode_modification(description),
                    }
                )

    # Print statistics
    print("\nValidation set statistics:")
    for key, samples in validation_sets.items():
        print(f"  {key}: {len(samples)} examples")

    # Save validation sets
    save_validation_sets(validation_sets, output_dir)

    return validation_sets


def encode_modification(description: str) -> List[int]:
    """
    Encode modification description into tokens.
    This is a simplified encoding - in practice would use learned embeddings.
    """
    # Simple one-hot style encoding for different modification types
    if "walk → skip" in description:
        return [1, 0, 0]
    elif "jump → hop" in description:
        return [0, 1, 0]
    elif "look → scan" in description:
        return [0, 0, 1]
    elif "left ↔ right" in description:
        return [1, 1, 0]
    elif "turn → spin" in description:
        return [1, 0, 1]
    elif "+" in description:  # Composed modifications
        return [1, 1, 1]
    else:
        return [0, 0, 0]  # No modification


def apply_unseen_modification(
    sample: SCANSample, swap_rules: Dict[str, str]
) -> Tuple[str, str]:
    """Apply modifications that create new action types not in standard SCAN."""
    mod_command = sample.command
    mod_action = sample.action

    for orig, repl in swap_rules.items():
        mod_command = mod_command.replace(orig, repl)

        # Create new action tokens for unseen words
        if orig in ["turn", "run", "walk", "jump", "look"]:
            mod_action = mod_action.replace(f"I_{orig.upper()}", f"I_{repl.upper()}")
        elif orig in ["twice", "thrice"]:
            # These don't have direct action mappings but affect repetition
            pass
        elif orig == "around":
            # 'around' affects turn commands
            if (
                "beside" == repl
                and "I_TURN_LEFT I_TURN_LEFT I_TURN_LEFT I_TURN_LEFT" in mod_action
            ):
                # "turn around" becomes "turn beside" - different action pattern
                mod_action = mod_action.replace(
                    "I_TURN_LEFT I_TURN_LEFT I_TURN_LEFT I_TURN_LEFT",
                    "I_TURN_LEFT I_TURN_RIGHT",  # New pattern for "beside"
                )

    return mod_command, mod_action


def save_validation_sets(validation_sets: Dict[str, List[Dict]], output_dir: Path):
    """Save validation sets in multiple formats for easy loading."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as pickle for Python loading
    with open(output_dir / "validation_sets_comprehensive.pkl", "wb") as f:
        pickle.dump(validation_sets, f)

    # Save as JSON for inspection
    with open(output_dir / "validation_sets_comprehensive.json", "w") as f:
        json.dump(validation_sets, f, indent=2)

    # Save each set separately for targeted evaluation
    for key, samples in validation_sets.items():
        with open(output_dir / f"{key}.pkl", "wb") as f:
            pickle.dump(samples, f)

        # Also save first 10 examples as JSON for easy inspection
        with open(output_dir / f"{key}_examples.json", "w") as f:
            json.dump(samples[:10], f, indent=2)

    # Create a metadata file
    metadata = {
        "creation_date": str(Path(__file__).stat().st_mtime),
        "validation_sets": {
            key: {"count": len(samples), "description": get_set_description(key)}
            for key, samples in validation_sets.items()
        },
    }

    with open(output_dir / "validation_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nValidation sets saved to {output_dir}")


def get_set_description(key: str) -> str:
    """Get human-readable description of validation set."""
    descriptions = {
        "val_base": "Unmodified SCAN examples for baseline performance",
        "val_mod_walk_skip": "Examples with walk → skip modification",
        "val_mod_jump_hop": "Examples with jump → hop modification",
        "val_mod_look_scan": "Examples with look → scan modification",
        "val_mod_left_right": "Examples with left ↔ right swap",
        "val_mod_mixed": "Mix of different single modifications",
        "val_mod_unseen": "Modifications never shown during training",
        "val_mod_composed": "Multiple modifications applied together",
    }
    return descriptions.get(key, "Unknown validation set")


def main():
    """Create comprehensive validation sets for compositional language experiments."""
    # Load base SCAN data
    print("Loading SCAN dataset...")
    loader = SCANDataLoader()
    loader.load_all_data()

    # Get samples from the standard validation split
    splits = loader.create_isolated_splits()
    base_validation_samples = splits["val_interpolation"] + splits["val_extrapolation"]

    print(f"Loaded {len(base_validation_samples)} base validation samples")

    # Create output directory
    output_dir = Path("data/processed/proper_validation_sets")

    # Generate comprehensive validation sets
    validation_sets = create_comprehensive_validation_sets(
        base_validation_samples, output_dir
    )

    print("\nValidation set creation complete!")
    print("\nThese new validation sets address the evaluation illusion by:")
    print("1. Including both base and modified examples")
    print("2. Testing each modification type separately")
    print("3. Including unseen modifications to test true generalization")
    print("4. Providing composed modifications to test complex reasoning")
    print(
        "\nUse these sets with evaluation_v2.py for accurate performance measurement."
    )


if __name__ == "__main__":
    main()
