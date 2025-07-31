"""
Debug shape issues in sequential planning model.
"""

from utils.imports import setup_project_paths
setup_project_paths()

from utils.config import setup_environment
import mlx.core as mx
import numpy as np

config = setup_environment()

# Test stacking
print("Testing array shapes:")

# Simulate outputs
output1 = mx.random.normal((1, 128))  # (batch, embed_dim)
output2 = mx.random.normal((1, 128))
output3 = mx.random.normal((1, 128))

outputs = [output1, output2, output3]
print(f"Individual output shape: {output1.shape}")

# Stack outputs
outputs_stacked = mx.stack(outputs, axis=1)
print(f"Stacked outputs shape: {outputs_stacked.shape}")  # Should be (1, 3, 128)

# Simulate temporal actions
temporal1 = mx.random.normal((1, 128))  # (batch, embed_dim)
temporal2 = mx.random.normal((1, 128))

temporal_actions = [temporal1, temporal2]
print(f"Individual temporal shape: {temporal1.shape}")

# Stack temporal
temporal_stacked = mx.stack(temporal_actions, axis=1)
print(f"Stacked temporal shape: {temporal_stacked.shape}")  # Should be (1, 2, 128)

# Try concatenation
try:
    combined = mx.concatenate([outputs_stacked, temporal_stacked], axis=1)
    print(f"Combined shape: {combined.shape}")  # Should be (1, 5, 128)
except Exception as e:
    print(f"Concatenation error: {e}")

print("\nNow testing potential issue:")

# What if temporal actions have wrong shape?
temporal_wrong = mx.random.normal((128,))  # Missing batch dimension
print(f"Wrong temporal shape: {temporal_wrong.shape}")

temporal_actions_wrong = [temporal_wrong, temporal_wrong]
try:
    temporal_stacked_wrong = mx.stack(temporal_actions_wrong, axis=1)
    print(f"Wrong stacked shape: {temporal_stacked_wrong.shape}")
except Exception as e:
    print(f"Stack error with wrong shape: {e}")

# Try stacking on axis 0 instead
temporal_stacked_axis0 = mx.stack(temporal_actions_wrong, axis=0)
print(f"Stacked on axis 0: {temporal_stacked_axis0.shape}")  # Should be (2, 128)

# This needs to be expanded
temporal_expanded = temporal_stacked_axis0[None, :, :]  # Add batch dimension
print(f"Expanded shape: {temporal_expanded.shape}")  # Should be (1, 2, 128)

# Now try concatenation
try:
    combined = mx.concatenate([outputs_stacked, temporal_expanded], axis=1)
    print(f"Combined after fix: {combined.shape}")  # Should be (1, 5, 128)
except Exception as e:
    print(f"Still error: {e}")