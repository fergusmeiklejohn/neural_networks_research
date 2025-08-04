#!/usr/bin/env python3
"""Test MLX embedding layer with exact same conditions."""

import mlx.core as mx
import mlx.nn as nn

# Test the exact same scenario
print("Testing MLX embedding layer...")

# Create embedding layer with same vocab size
vocab_size = 50  # Approximate vocab size based on the data
embed_dim = 256
embed = nn.Embedding(vocab_size, embed_dim)

# Test with exact same input
token_idx = mx.array([[2]], dtype=mx.int32)
print(f"Input shape: {token_idx.shape}, dtype: {token_idx.dtype}")
print(f"Input content: {token_idx}")

try:
    out = embed(token_idx)
    print(f"Success! Output shape: {out.shape}")
except Exception as e:
    print(f"Failed: {e}")
    print(f"Embedding weight shape: {embed.weight.shape}")
    print(f"Embedding weight dtype: {embed.weight.dtype}")

    # Try different approaches
    print("\nTrying squeeze approach:")
    try:
        squeezed = token_idx.squeeze()
        print(f"Squeezed shape: {squeezed.shape}")
        out = embed(squeezed)
        print(f"Success with squeeze! Output shape: {out.shape}")
    except Exception as e2:
        print(f"Squeeze failed: {e2}")

    print("\nTrying direct indexing:")
    try:
        idx_val = token_idx[0, 0]
        print(f"Direct index: {idx_val}, shape: {idx_val.shape}")
        out = embed.weight[idx_val]
        print(f"Direct indexing works! Output shape: {out.shape}")
    except Exception as e3:
        print(f"Direct indexing failed: {e3}")

# Test if it's a version issue
print("\nChecking MLX version:")
import mlx

print(f"MLX version: {mlx.__version__ if hasattr(mlx, '__version__') else 'Unknown'}")

# Test with numpy array conversion
print("\nTesting with numpy conversion:")
import numpy as np

np_idx = np.array([[2]], dtype=np.int32)
mx_idx = mx.array(np_idx)
try:
    out = embed(mx_idx)
    print(f"Numpy conversion works! Output shape: {out.shape}")
except Exception as e:
    print(f"Numpy conversion failed: {e}")
