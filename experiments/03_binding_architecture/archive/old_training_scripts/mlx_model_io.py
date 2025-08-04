#!/usr/bin/env python3
"""Simple and robust MLX model save/load utilities."""

import pickle

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def save_model_simple(path: str, model: nn.Module) -> None:
    """Save MLX model using pickle for maximum compatibility.

    Args:
        path: Path to save file (should end with .pkl)
        model: MLX model to save
    """
    # Get all parameters and convert to numpy
    params = model.parameters()

    def convert_to_numpy(obj):
        """Recursively convert MLX arrays to numpy."""
        if isinstance(obj, mx.array):
            return np.array(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(convert_to_numpy(item) for item in obj)
        else:
            return obj

    np_params = convert_to_numpy(params)

    # Save with pickle
    with open(path, "wb") as f:
        pickle.dump(np_params, f)

    print(f"Model saved to {path}")


def load_model_simple(path: str, model: nn.Module) -> None:
    """Load MLX model from pickle file.

    Args:
        path: Path to saved file
        model: MLX model to load parameters into
    """
    # Load parameters
    with open(path, "rb") as f:
        np_params = pickle.load(f)

    def convert_to_mlx(obj):
        """Recursively convert numpy arrays to MLX."""
        if isinstance(obj, np.ndarray):
            return mx.array(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_mlx(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(convert_to_mlx(item) for item in obj)
        else:
            return obj

    mlx_params = convert_to_mlx(np_params)

    # Update model
    model.update(mlx_params)

    print(f"Model loaded from {path}")


def save_model_npz(path: str, model: nn.Module) -> None:
    """Save MLX model to npz format with flattened parameters.

    Args:
        path: Path to save file (should end with .npz)
        model: MLX model to save
    """
    flat_params = {}

    def flatten(obj, prefix=""):
        """Recursively flatten parameters."""
        if isinstance(obj, mx.array):
            flat_params[prefix] = np.array(obj)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                new_prefix = f"{prefix}.{k}" if prefix else k
                flatten(v, new_prefix)
        elif isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                flatten(item, f"{prefix}.{i}")

    # Flatten all parameters
    flatten(model.parameters())

    # Save as npz
    np.savez_compressed(path, **flat_params)
    print(f"Model saved to {path} with {len(flat_params)} parameters")


def test_save_load():
    """Test the save/load functions."""

    # Simple test model without Sequential
    class SimpleTestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 30)
            self.fc3 = nn.Linear(30, 10)

        def __call__(self, x):
            x = mx.tanh(self.fc1(x))
            x = mx.tanh(self.fc2(x))
            return self.fc3(x)

    print("Testing save/load with simple model...")

    # Create model
    model = SimpleTestModel()

    # Generate random input and get output
    x = mx.random.normal((2, 10))
    y1 = model(x)
    print(f"Original output shape: {y1.shape}")

    # Test pickle save/load
    print("\nTesting pickle save/load...")
    save_model_simple("test_model.pkl", model)

    model2 = SimpleTestModel()
    load_model_simple("test_model.pkl", model2)

    y2 = model2(x)
    diff = mx.abs(y1 - y2).max()
    print(f"Max difference after pickle save/load: {diff}")
    assert diff < 1e-6, "Outputs differ!"

    # Test npz save
    print("\nTesting npz save...")
    save_model_npz("test_model.npz", model)

    # Clean up
    import os

    os.remove("test_model.pkl")
    os.remove("test_model.npz")

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_save_load()
