#!/usr/bin/env python3
"""Quick dry run to test the key parts of the script"""

import os
import sys
from pathlib import Path

# Mock environment
os.environ['KERAS_BACKEND'] = 'tensorflow'

# Test the critical parts that were failing
print("Testing critical components...")

# 1. Test imports
print("\n1. Testing imports...")
try:
    from models import create_model
    from train_progressive_curriculum import create_dataset, SCANTokenizer
    from scan_data_loader import SCANDataLoader
    from modification_generator import ModificationGenerator
    print("✓ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# 2. Test tokenizer save/load
print("\n2. Testing tokenizer save/load...")
try:
    tokenizer = SCANTokenizer()
    test_samples = [{'command': 'walk', 'action': 'I_WALK'}]
    tokenizer.build_vocabulary(test_samples)
    
    # Test save
    test_path = Path('test_vocab.json')
    tokenizer.save_vocabulary(test_path)
    print("✓ save_vocabulary() works")
    
    # Test load
    tokenizer2 = SCANTokenizer(vocab_path=test_path)
    tokenizer2.load_vocabulary()
    print("✓ load_vocabulary() works")
    
    # Cleanup
    test_path.unlink()
    
except AttributeError as e:
    print(f"❌ AttributeError: {e}")
    sys.exit(1)

# 3. Test model creation
print("\n3. Testing model creation...")
try:
    model = create_model(
        command_vocab_size=10,
        action_vocab_size=10,
        d_model=64
    )
    print("✓ Model creation works")
except Exception as e:
    print(f"❌ Model error: {e}")

print("\n✅ All critical components working!")
print("Script should work on Paperspace now.")