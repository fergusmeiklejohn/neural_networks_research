#!/usr/bin/env python3
"""Debug MLX model persistence issues (std::bad_cast error)."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import os
import traceback

class SimpleModel(nn.Module):
    """Minimal model to test MLX persistence."""
    
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def __call__(self, x):
        x = self.fc1(x)
        x = mx.tanh(x)
        x = self.fc2(x)
        return x

def test_persistence_methods():
    """Test different MLX model saving methods."""
    print("Testing MLX model persistence methods...")
    
    # Create model and dummy data
    model = SimpleModel()
    x = mx.random.normal((2, 10))
    y = model(x)
    print(f"Model output shape: {y.shape}")
    
    # Get model parameters
    params = model.parameters()
    print(f"\nModel parameters: {list(params.keys())}")
    
    # Method 1: mx.save with dict
    print("\n1. Testing mx.save with dict(model.parameters())...")
    try:
        mx.save("test_model_dict.npz", dict(model.parameters()))
        print("✓ mx.save with dict succeeded")
        # Try loading
        loaded = mx.load("test_model_dict.npz")
        print(f"✓ Loaded keys: {list(loaded.keys())}")
        os.remove("test_model_dict.npz")
    except Exception as e:
        print(f"✗ mx.save with dict failed: {e}")
        traceback.print_exc()
    
    # Method 2: mx.savez with **dict
    print("\n2. Testing mx.savez with **dict(model.parameters())...")
    try:
        mx.savez("test_model_savez.npz", **dict(model.parameters()))
        print("✓ mx.savez with **dict succeeded")
        # Try loading
        loaded = mx.load("test_model_savez.npz")
        print(f"✓ Loaded keys: {list(loaded.keys())}")
        os.remove("test_model_savez.npz")
    except Exception as e:
        print(f"✗ mx.savez with **dict failed: {e}")
        traceback.print_exc()
    
    # Method 3: Convert to numpy first
    print("\n3. Testing numpy conversion before saving...")
    try:
        np_params = {}
        for k, v in model.parameters().items():
            np_params[k] = np.array(v)
        np.savez("test_model_numpy.npz", **np_params)
        print("✓ numpy.savez succeeded")
        # Try loading
        loaded = np.load("test_model_numpy.npz")
        print(f"✓ Loaded keys: {list(loaded.keys())}")
        os.remove("test_model_numpy.npz")
    except Exception as e:
        print(f"✗ numpy conversion failed: {e}")
        traceback.print_exc()
    
    # Method 4: Check if save_safetensors exists
    print("\n4. Checking for save_safetensors...")
    if hasattr(mx, 'save_safetensors'):
        print("✓ mx.save_safetensors exists")
        try:
            mx.save_safetensors("test_model.safetensors", dict(model.parameters()))
            print("✓ save_safetensors succeeded")
            os.remove("test_model.safetensors")
        except Exception as e:
            print(f"✗ save_safetensors failed: {e}")
    else:
        print("✗ mx.save_safetensors not available in this MLX version")
    
    # Method 5: Manual hierarchical saving
    print("\n5. Testing manual hierarchical saving...")
    try:
        # Flatten the nested parameter dict
        flat_params = {}
        for k, v in model.parameters().items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    flat_key = f"{k}.{sub_k}"
                    flat_params[flat_key] = np.array(sub_v)
            else:
                flat_params[k] = np.array(v)
        
        np.savez("test_model_flat.npz", **flat_params)
        print("✓ Flattened numpy save succeeded")
        print(f"✓ Saved keys: {list(flat_params.keys())}")
        os.remove("test_model_flat.npz")
    except Exception as e:
        print(f"✗ Manual hierarchical saving failed: {e}")
        traceback.print_exc()
    
    # Method 6: Test parameter structure
    print("\n6. Analyzing parameter structure...")
    for name, param in model.parameters().items():
        print(f"  {name}: type={type(param)}, shape={param.shape if hasattr(param, 'shape') else 'N/A'}")
        if isinstance(param, dict):
            for sub_name, sub_param in param.items():
                print(f"    {sub_name}: type={type(sub_param)}, shape={sub_param.shape if hasattr(sub_param, 'shape') else 'N/A'}")


if __name__ == "__main__":
    test_persistence_methods()