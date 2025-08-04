#!/usr/bin/env python3
"""Debug shapes in compositional model."""

from utils.imports import setup_project_paths
setup_project_paths()

from utils.config import setup_environment
config = setup_environment()

import mlx.core as mx
from train_integrated_model import VOCAB, ACTIONS, IntegratedBindingModel
from improved_compositional_operators import ImprovedCompositionalParser, ImprovedCompositionalExecutor

# Test the basic model behavior
model = IntegratedBindingModel(
    vocab_size=len(VOCAB),
    num_actions=len(ACTIONS),
    embed_dim=256,
    num_slots=4,
    num_heads=8,
    mlp_hidden_dim=512
)

# Test command
command = "X means jump Y means walk do X and Y"
tokens = [VOCAB.get(word, VOCAB['<PAD>']) for word in command.split()]
inputs = {'command': mx.array([tokens])}

print(f"Command: {command}")
print(f"Tokens shape: {mx.array([tokens]).shape}")

# Test parser
parser = ImprovedCompositionalParser(VOCAB)
parse_tree = parser.parse(mx.array(tokens))
print(f"\nParse tree structure:")
print(f"  Operator: {parse_tree.operator}")
print(f"  Is leaf: {parse_tree.is_leaf()}")
print(f"  Children: {len(parse_tree.children)}")

# Test basic model call
model.versioned_memory.clear()
outputs = model(inputs, stage='full')
print(f"\nModel output shape: {outputs.shape}")

# Test executor
executor = ImprovedCompositionalExecutor(model, VOCAB)
model.versioned_memory.clear()
exec_outputs = executor.execute(parse_tree, mx.array([tokens]), {}, 'full')
print(f"\nExecutor outputs: {len(exec_outputs)} items")
for i, out in enumerate(exec_outputs):
    print(f"  Output {i} shape: {out.shape}")

# Test stacking
if exec_outputs:
    stacked = mx.stack(exec_outputs)
    print(f"\nStacked output shape: {stacked.shape}")

# Test with expected labels
expected_actions = ['JUMP', 'WALK']
labels = mx.array([ACTIONS[a] for a in expected_actions])
print(f"\nLabels shape: {labels.shape}")
print(f"Labels: {labels}")

# Test specific problematic cases from training
print("\n=== Testing Stage Data Shapes ===")
from train_integrated_model import generate_stage1_data

# Get a single stage1 sample
stage1 = generate_stage1_data(1)
print("\nStage1 data keys:", stage1.keys())
print("Command shape:", stage1['command'].shape)
print("Target shape:", stage1['target'].shape)

# Batch to list conversion
def batch_to_list(batch_data):
    list_data = []
    for i in range(len(batch_data['command'])):
        item = {}
        for key in batch_data:
            if hasattr(batch_data[key], '__getitem__'):
                item[key] = batch_data[key][i:i+1]
            else:
                item[key] = batch_data[key]
        list_data.append(item)
    return list_data

stage1_list = batch_to_list(stage1)
print("\nAfter batch_to_list:")
print("Number of items:", len(stage1_list))
if stage1_list:
    item = stage1_list[0]
    print("First item keys:", item.keys())
    print("Command shape:", item['command'].shape)
    print("Target shape:", item['target'].shape)