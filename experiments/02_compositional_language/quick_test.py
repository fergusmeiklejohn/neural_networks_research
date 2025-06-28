#!/usr/bin/env python3
"""Quick test to verify basic functionality"""

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

print("Testing compositional language setup...")

# Test data loading
print("\n1. Testing data loader...")
from scan_data_loader import SCANDataLoader
loader = SCANDataLoader(data_dir='data')
splits = loader.load_processed_splits()
print(f"   Loaded {len(splits)} splits")
print(f"   Train samples: {len(splits.get('train', []))}")

# Test tokenizer
print("\n2. Testing tokenizer...")
from train_progressive_curriculum import SCANTokenizer
tokenizer = SCANTokenizer()
tokenizer.build_vocabulary(splits['train'][:100])
print(f"   Command vocab size: {len(tokenizer.command_to_id)}")
print(f"   Action vocab size: {len(tokenizer.action_to_id)}")

# Test encoding/decoding
sample = splits['train'][0]
cmd_enc = tokenizer.encode_command(sample['command'])
act_enc = tokenizer.encode_action(sample['action'])
print(f"   Sample command: {sample['command']}")
print(f"   Encoded shape: {cmd_enc.shape}")
print(f"   Sample action: {sample['action'][:50]}...")
print(f"   Encoded shape: {act_enc.shape}")

# Test model creation
print("\n3. Testing model creation...")
from models import create_model
model = create_model(
    command_vocab_size=len(tokenizer.command_to_id),
    action_vocab_size=len(tokenizer.action_to_id),
    d_model=64  # Small for testing
)
print("   Model created successfully!")

# Test dataset creation
print("\n4. Testing dataset creation...")
import tensorflow as tf
from train_progressive_curriculum import create_dataset

# Create small dataset
small_samples = splits['train'][:100]
dataset = create_dataset(small_samples, tokenizer, batch_size=4)
print(f"   Dataset created")

# Test one batch
for batch in dataset.take(1):
    print(f"   Batch keys: {batch.keys()}")
    print(f"   Command shape: {batch['command'].shape}")
    print(f"   Action shape: {batch['action'].shape}")

print("\nAll tests passed!")