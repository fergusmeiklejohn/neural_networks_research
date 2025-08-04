#!/usr/bin/env python3
"""Debug the embedding issue."""

import mlx.core as mx
import mlx.nn as nn

# Create a simple embedding layer
embed = nn.Embedding(100, 32)

# Test different input shapes
print("Testing embedding layer with different input shapes:")

# 1D input
try:
    tokens_1d = mx.array([1, 2, 3])
    out = embed(tokens_1d)
    print(f"1D input shape {tokens_1d.shape} -> output shape {out.shape}")
except Exception as e:
    print(f"1D input failed: {e}")

# 2D input (batch, seq)
try:
    tokens_2d = mx.array([[1, 2, 3]])
    out = embed(tokens_2d)
    print(f"2D input shape {tokens_2d.shape} -> output shape {out.shape}")
except Exception as e:
    print(f"2D input failed: {e}")

# Sliced 2D input
try:
    tokens_2d = mx.array([[1, 2, 3]])
    sliced = tokens_2d[:, 0:1]
    print(f"Sliced shape: {sliced.shape}, dtype: {sliced.dtype}")
    out = embed(sliced)
    print(f"Sliced 2D input shape {sliced.shape} -> output shape {out.shape}")
except Exception as e:
    print(f"Sliced 2D input failed: {e}")

# What happens with numpy arrays?
try:
    import numpy as np
    tokens_np = np.array([[1, 2, 3]])
    tokens_mx = mx.array(tokens_np)
    sliced = tokens_mx[:, 0:1]
    print(f"\nNumpy->MLX sliced shape: {sliced.shape}, dtype: {sliced.dtype}")
    out = embed(sliced)
    print(f"Numpy->MLX sliced output shape: {out.shape}")
except Exception as e:
    print(f"Numpy->MLX failed: {e}")

# Check the actual problematic case
try:
    segment_tokens = mx.array([[2, 4, 7, 3, 4, 12, 5, 6, 7, 8, 9, 10]])
    t = 0
    token_slice = segment_tokens[:, t:t+1]
    print(f"\nProblematic case - slice shape: {token_slice.shape}, dtype: {token_slice.dtype}")
    print(f"Slice content: {token_slice}")
    
    # Try different approaches
    # Approach 1: Direct slice
    try:
        out = embed(token_slice)
        print(f"Direct slice works! Output shape: {out.shape}")
    except Exception as e:
        print(f"Direct slice failed: {e}")
    
    # Approach 2: Squeeze
    try:
        squeezed = token_slice.squeeze()
        print(f"Squeezed shape: {squeezed.shape}")
        out = embed(squeezed)
        print(f"Squeezed works! Output shape: {out.shape}")
    except Exception as e:
        print(f"Squeezed failed: {e}")
    
    # Approach 3: Index directly
    try:
        token_idx = segment_tokens[0, t]
        print(f"Direct index shape: {token_idx.shape}, value: {token_idx}")
        out = embed(token_idx.reshape(1))
        print(f"Direct index works! Output shape: {out.shape}")
    except Exception as e:
        print(f"Direct index failed: {e}")
        
except Exception as e:
    print(f"Overall test failed: {e}")