#!/usr/bin/env python3
"""
Evaluate the trained Distribution Modification Component.
"""

import os

os.environ["KERAS_BACKEND"] = "jax"

from pathlib import Path

import keras
import numpy as np
from distribution_modifier import ModificationDataProcessor


def main():
    """Evaluate the trained model."""
    print("Loading trained model...")
    model = keras.models.load_model(
        "outputs/checkpoints/distribution_modifier_best.keras"
    )
    print(f"Model loaded successfully!")

    # Load vocabulary
    processor = ModificationDataProcessor()
    processor.load_vocabulary(Path("outputs/modifier_vocabulary.json"))

    print("\n" + "=" * 60)
    print("Testing Distribution Modification Component")
    print("=" * 60)

    # Test cases
    test_cases = [
        {
            "params": [800.0, 0.5, 0.7, 0.95],  # gravity, friction, elasticity, damping
            "modifications": [
                "increase gravity by 20%",
                "decrease gravity by 50%",
                "remove all friction",
                "increase friction significantly",
                "make objects perfectly bouncy",
                "make objects less bouncy",
                "underwater physics",
                "space physics",
                "bouncy castle",
            ],
        }
    ]

    for test in test_cases:
        base_params = np.array([test["params"]], dtype=np.float32)

        print(f"\nBase physics parameters:")
        for i, name in enumerate(processor.param_names):
            print(f"  {name:12s}: {base_params[0, i]:.3f}")

        print("\nModification results:")
        print("-" * 60)

        for mod_desc in test["modifications"]:
            # Encode description
            desc_tokens = processor.encode_description(mod_desc).reshape(1, -1)

            # Predict
            pred_params, mod_factors, change_mask = model(
                [base_params, desc_tokens], training=False
            )

            print(f"\n'{mod_desc}':")

            # Show changes
            for i, name in enumerate(processor.param_names):
                original = base_params[0, i]
                predicted = pred_params[0, i].numpy()
                factor = mod_factors[0, i].numpy()
                mask = change_mask[0, i].numpy()

                if mask > 0.3:  # Parameter likely changed
                    change_pct = ((predicted - original) / original) * 100
                    print(
                        f"  {name:12s}: {original:.3f} -> {predicted:.3f} "
                        f"({change_pct:+.1f}%, factor: {factor:.3f}, conf: {mask:.2f})"
                    )

    print("\n" + "=" * 60)
    print("Parameter Modification Patterns:")
    print("=" * 60)

    # Analyze which parameters the model learned to modify for each request type
    param_changes = {name: [] for name in processor.param_names}

    modification_types = {
        "gravity": ["gravity", "weight", "heavy", "light"],
        "friction": ["friction", "slippery", "sticky", "slide"],
        "elasticity": ["bouncy", "elastic", "bounce", "rigid"],
        "damping": ["damping", "air", "resistance", "drag"],
    }

    for param_type, keywords in modification_types.items():
        print(f"\n{param_type.upper()} related modifications:")

        for keyword in keywords:
            for direction in ["increase", "decrease", "more", "less"]:
                desc = f"{direction} {keyword}"
                desc_tokens = processor.encode_description(desc).reshape(1, -1)

                # Use neutral parameters
                test_params = np.array([[700.0, 0.5, 0.5, 0.9]], dtype=np.float32)

                pred_params, _, change_mask = model(
                    [test_params, desc_tokens], training=False
                )

                # Check which parameter changed most
                changes = np.abs(pred_params[0].numpy() - test_params[0])
                most_changed = processor.param_names[np.argmax(changes)]

                if np.max(change_mask[0].numpy()) > 0.3:
                    print(f"  '{desc}' -> primarily affects {most_changed}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
