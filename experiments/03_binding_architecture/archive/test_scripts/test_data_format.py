#!/usr/bin/env python3
"""Test data format from curriculum stages."""

import sys

sys.path.append("../..")

from train_binding_curriculum import (
    generate_stage1_data,
    generate_stage2_data,
    generate_stage3_data,
)

# Generate small samples
print("Stage 1 data:")
stage1 = generate_stage1_data(2, max_examples=2)
print(f"Keys: {list(stage1.keys())}")
print(f"Command shape: {stage1['command'].shape}")
print(f"Target shape: {stage1['target'].shape}")
print()

print("Stage 2 data:")
stage2 = generate_stage2_data(2, max_examples=2)
print(f"Keys: {list(stage2.keys())}")
print(f"Command shape: {stage2['command'].shape}")
print(f"Labels shape: {stage2['labels'].shape}")
print()

print("Stage 3 data:")
stage3 = generate_stage3_data(2, max_examples=2)
print(f"Keys: {list(stage3.keys())}")
print(f"Command shape: {stage3['command'].shape}")
print(f"Labels shape: {stage3['labels'].shape}")
