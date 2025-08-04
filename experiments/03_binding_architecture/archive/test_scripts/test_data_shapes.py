#!/usr/bin/env python3
"""Test data shapes to debug training issues."""

from utils.imports import setup_project_paths
setup_project_paths()

from utils.config import setup_environment
config = setup_environment()

from train_integrated_model import (
    generate_stage1_data, generate_stage2_data, 
    generate_stage3_data, generate_rebinding_data
)
from train_compositional_model import generate_compositional_data
from train_nested_temporal_model import generate_nested_temporal_data, batch_to_list

# Generate small samples
print("=== Testing Data Shapes ===\n")

# Stage 1 data
stage1 = generate_stage1_data(2)
print("Stage 1 data (batched):")
print(f"  command shape: {stage1['command'].shape}")
print(f"  target shape: {stage1['target'].shape}")

stage1_list = batch_to_list(stage1, 'recognition')
print(f"\nStage 1 data (list): {len(stage1_list)} items")
for i, item in enumerate(stage1_list[:2]):
    print(f"  Item {i} keys: {list(item.keys())}")
    if 'labels' in item:
        print(f"    labels shape: {item['labels'].shape}")
    if 'target' in item:
        print(f"    target shape: {item['target'].shape}")

# Compositional data
print("\nCompositional data:")
comp_data = generate_compositional_data(2)
for i, item in enumerate(comp_data[:2]):
    print(f"  Item {i} keys: {list(item.keys())}")
    print(f"    command shape: {item['command'].shape}")
    print(f"    labels shape: {item['labels'].shape}")

# Nested temporal data
print("\nNested temporal data:")
nested_data = generate_nested_temporal_data(2)
for i, item in enumerate(nested_data[:2]):
    print(f"  Item {i} keys: {list(item.keys())}")
    print(f"    command shape: {item['command'].shape}")
    print(f"    labels shape: {item['labels'].shape}")

# Rebinding data
print("\nRebinding data:")
rebind_data = generate_rebinding_data(2)
for i, item in enumerate(rebind_data[:2]):
    print(f"  Item {i} keys: {list(item.keys())}")
    print(f"    command shape: {item['command'].shape}")
    print(f"    labels shape: {item['labels'].shape}")