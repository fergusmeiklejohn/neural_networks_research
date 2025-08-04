#!/usr/bin/env python3
"""Inspect modification pairs data structure."""

import pickle
from pathlib import Path

import numpy as np


def inspect_modification_data():
    """Load and inspect modification pairs data."""
    data_dir = Path("data/processed/physics_worlds")
    mod_file = data_dir / "modification_pairs.pkl"

    if not mod_file.exists():
        print(f"Modification file not found: {mod_file}")
        return

    with open(mod_file, "rb") as f:
        mod_data = pickle.load(f)

    print(f"Modification data type: {type(mod_data)}")
    print(f"Number of modification pairs: {len(mod_data)}")

    if len(mod_data) > 0:
        print("\nFirst modification pair structure:")
        first_pair = mod_data[0]
        print(f"Pair type: {type(first_pair)}")

        if isinstance(first_pair, dict):
            print("Keys:", list(first_pair.keys()))
            for key, value in first_pair.items():
                if isinstance(value, (np.ndarray, list)):
                    print(f"  {key}: shape {np.array(value).shape}")
                else:
                    print(f"  {key}: {value}")

        print("\nSample modification descriptions:")
        for i in range(min(5, len(mod_data))):
            if "modification_description" in mod_data[i]:
                print(f"  {i}: {mod_data[i]['modification_description']}")


if __name__ == "__main__":
    inspect_modification_data()
