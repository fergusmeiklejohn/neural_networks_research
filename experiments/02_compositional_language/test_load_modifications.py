#!/usr/bin/env python3
"""
Test that ModificationGenerator.load_modifications() works correctly
"""

from pathlib import Path

from modification_generator import ModificationGenerator


def test_load_modifications():
    """Test loading modifications"""
    print("Testing ModificationGenerator.load_modifications()...")

    # Create generator
    generator = ModificationGenerator()

    # Check if modifications exist
    mod_path = Path("data/processed/modification_pairs.pkl")
    print(f"Checking for modifications at: {mod_path}")
    print(f"File exists: {mod_path.exists()}")

    # Try to load
    try:
        modifications = generator.load_modifications()
        print(f"✅ Successfully loaded {len(modifications)} modifications")

        # Check structure
        if modifications:
            first_mod = modifications[0]
            print(f"\nFirst modification:")
            print(f"  Type: {first_mod.modification_type}")
            print(f"  Description: {first_mod.modification_description}")
            print(f"  Original: {first_mod.original_sample.command}")
            print(f"  Modified: {first_mod.modified_sample.command}")

    except Exception as e:
        print(f"❌ Error loading modifications: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_load_modifications()
