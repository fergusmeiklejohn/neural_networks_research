#!/usr/bin/env python3
"""Utilities for MLX model persistence."""

from typing import Any, Dict

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def flatten_params(params: Dict[str, Any], prefix: str = "") -> Dict[str, np.ndarray]:
    """Flatten nested parameter dictionaries into a single level dict with numpy arrays.

    Args:
        params: Nested dictionary of parameters from model.parameters()
        prefix: Prefix for parameter names (used for recursion)

    Returns:
        Flat dictionary with string keys and numpy array values
    """
    flat = {}
    for key, value in params.items():
        full_key = f"{prefix}{key}" if prefix else key

        if isinstance(value, dict):
            # Recursively flatten nested dicts
            flat.update(flatten_params(value, prefix=f"{full_key}."))
        elif isinstance(value, mx.array):
            # Convert MLX arrays to numpy
            flat[full_key] = np.array(value)
        elif isinstance(value, (list, tuple)):
            # Handle lists/tuples of parameters (e.g., from Sequential)
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    flat.update(flatten_params(item, prefix=f"{full_key}.{i}."))
                elif isinstance(item, mx.array):
                    flat[f"{full_key}.{i}"] = np.array(item)
        else:
            # Skip other types
            pass

    return flat


def unflatten_params(flat_params: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Unflatten a flat parameter dictionary back into nested structure.

    Args:
        flat_params: Flat dictionary with dot-separated keys

    Returns:
        Nested dictionary structure matching MLX model.parameters()
    """
    nested = {}

    for key, value in flat_params.items():
        parts = key.split(".")
        current = nested

        # Navigate/create nested structure
        for i, part in enumerate(parts[:-1]):
            next_part = parts[i + 1] if i + 1 < len(parts) else None

            # Check if next part is a number (list index)
            if next_part and next_part.isdigit():
                # Initialize as list if needed
                if part not in current:
                    current[part] = []
                current = current[part]

                # Extend list to required size
                idx = int(next_part)
                while len(current) <= idx:
                    current.append({})

            else:
                # Regular dict navigation
                if part not in current:
                    current[part] = {}
                current = current[part]

        # Set the final value
        final_key = parts[-1]
        if final_key.isdigit():
            # List index
            idx = int(final_key)
            while len(current) <= idx:
                current.append(None)
            current[idx] = mx.array(value)
        else:
            # Dict key
            current[final_key] = mx.array(value)

    return nested


def save_model(path: str, model: nn.Module) -> None:
    """Save MLX model parameters to file.

    Args:
        path: Path to save file (should end with .npz)
        model: MLX model to save
    """
    params = model.parameters()
    flat_params = flatten_params(params)
    np.savez(path, **flat_params)
    print(f"Model saved to {path} with {len(flat_params)} parameters")


def load_model(path: str, model: nn.Module) -> None:
    """Load MLX model parameters from file.

    Args:
        path: Path to saved file
        model: MLX model to load parameters into
    """
    # Load flat parameters with allow_pickle=True for compatibility
    with np.load(path, allow_pickle=True) as loaded:
        flat_params = {k: loaded[k] for k in loaded.files}

    # Unflatten to nested structure
    nested_params = unflatten_params(flat_params)

    # Update model parameters
    model.update(nested_params)
    print(f"Model loaded from {path} with {len(flat_params)} parameters")


def test_save_load():
    """Test the save/load functions."""

    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 30)
            )
            self.decoder = nn.Linear(30, 10)

        def __call__(self, x):
            x = self.encoder(x)
            return self.decoder(x)

    # Create model
    model = TestModel()

    # Generate random input and get output
    x = mx.random.normal((2, 10))
    y1 = model(x)

    # Save model
    save_model("test_model.npz", model)

    # Create new model and load
    model2 = TestModel()
    load_model("test_model.npz", model2)

    # Compare outputs
    y2 = model2(x)
    diff = mx.abs(y1 - y2).max()
    print(f"Max difference between outputs: {diff}")
    assert diff < 1e-6, "Model outputs differ after save/load!"

    # Clean up
    import os

    os.remove("test_model.npz")
    print("✓ Save/load test passed!")


if __name__ == "__main__":
    test_save_load()
