#!/usr/bin/env python3
"""Test Stage 2 training locally to validate ModificationPair fix"""

import os
import sys

# Setup paths - add current directory first
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import tensorflow as tf
from modification_generator import ModificationGenerator
from scan_data_loader import SCANDataLoader
from train_progressive_curriculum import SCANTokenizer, create_dataset

from models import create_model


def test_stage2_setup():
    """Test Stage 2 data preparation and training setup"""
    print("Testing Stage 2 Setup...")
    print("-" * 60)

    # Load base data
    print("1. Loading SCAN data...")
    loader = SCANDataLoader()
    splits = loader.load_processed_splits()
    base_data = splits["train"]
    print(f"   ✓ Loaded {len(base_data)} training samples")

    # Load and convert modifications
    print("\n2. Loading modifications...")
    mod_generator = ModificationGenerator()
    modification_pairs = mod_generator.load_modifications()
    print(f"   ✓ Loaded {len(modification_pairs)} modification pairs")

    # Convert to training format (same as in paperspace script)
    print("\n3. Converting modifications to training format...")
    modifications = []
    for pair in modification_pairs:
        sample = {
            "command": pair.modified_sample.command,
            "action": pair.modified_sample.action,
            "modification": pair.modification_description,
        }
        modifications.append(sample)
    print(f"   ✓ Converted {len(modifications)} modifications")

    # Test Stage 2 data mixing
    print("\n4. Testing Stage 2 data mixing...")
    stage_data = base_data[: int(len(base_data) * 0.7)]  # 70% base
    stage_data.extend(modifications[:100])  # First 100 modifications
    print(f"   Base samples: {len(base_data[:int(len(base_data) * 0.7)])}")
    print(f"   Modification samples: {len(modifications[:100])}")
    print(f"   Total stage data: {len(stage_data)}")

    # Create tokenizer
    print("\n5. Building tokenizer...")
    tokenizer = SCANTokenizer()
    tokenizer.build_vocabulary(base_data + modifications[:100])
    print(f"   Command vocab size: {len(tokenizer.command_to_id)}")
    print(f"   Action vocab size: {len(tokenizer.action_to_id)}")

    # Create dataset
    print("\n6. Creating TensorFlow dataset...")
    try:
        dataset = create_dataset(stage_data[:50], tokenizer, batch_size=4)
        print("   ✓ Dataset created successfully")

        # Test one batch
        for inputs, targets in dataset.take(1):
            print(f"   Batch shapes:")
            print(f"     Commands: {inputs['command'].shape}")
            print(f"     Target inputs: {inputs['target'].shape}")
            print(f"     Modifications: {inputs['modification'].shape}")
            print(f"     Target outputs: {targets.shape}")
    except Exception as e:
        print(f"   ✗ Failed to create dataset: {e}")
        return False

    # Create and compile model
    print("\n7. Creating and compiling model...")
    try:
        model = create_model(
            command_vocab_size=len(tokenizer.command_to_id),
            action_vocab_size=len(tokenizer.action_to_id),
            d_model=128,
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        print("   ✓ Model compiled successfully")
    except Exception as e:
        print(f"   ✗ Failed to create model: {e}")
        return False

    # Test one training step
    print("\n8. Testing one training step...")
    try:
        history = model.fit(dataset.take(5), epochs=1, verbose=1)
        loss = history.history["loss"][0]
        acc = history.history["accuracy"][0]
        print(f"   ✓ Training step successful - Loss: {loss:.4f}, Accuracy: {acc:.4f}")
    except Exception as e:
        print(f"   ✗ Training failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = test_stage2_setup()
    if success:
        print("\n✅ Stage 2 test PASSED! Ready for Paperspace deployment.")
    else:
        print("\n❌ Stage 2 test FAILED! Fix issues before deploying.")
