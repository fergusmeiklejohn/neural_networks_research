import sys
sys.path.append('/Users/fergusmeiklejohn/conductor/repo/neural_networks_research/bandung')

from train_binding_curriculum import (
    generate_stage1_data, generate_stage2_data, generate_stage3_data
)

# Check shapes for each stage
print("Stage 1 data shapes:")
stage1 = generate_stage1_data(batch_size=4)
print(f"  command: {stage1['command'].shape}")
print(f"  target: {stage1['target'].shape}")
print(f"  command example: {stage1['command'][0]}")
print(f"  target example: {stage1['target'][0]}")

print("\nStage 2 data shapes:")
stage2 = generate_stage2_data(batch_size=4)
print(f"  command: {stage2['command'].shape}")
print(f"  labels: {stage2['labels'].shape}")
print(f"  command example: {stage2['command'][0]}")
print(f"  labels example: {stage2['labels'][0]}")

print("\nStage 3 data shapes:")
stage3 = generate_stage3_data(batch_size=4)
print(f"  command: {stage3['command'].shape}")
print(f"  labels: {stage3['labels'].shape}")
print(f"  command example: {stage3['command'][0]}")
print(f"  labels example: {stage3['labels'][0]}")