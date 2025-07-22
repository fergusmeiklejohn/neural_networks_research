#!/usr/bin/env python3
"""
Simple local test: Run the actual training with minimal data.
This catches runtime errors before Paperspace deployment.
"""

import os
import sys
import time
from pathlib import Path

# Set up minimal test environment
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def test_training():
    """Run minimal training test"""
    print("🧪 Running local training test with minimal data...")
    print("="*60)
    
    # Import everything we need
    from models import create_model
    from train_progressive_curriculum import create_dataset, SCANTokenizer
    from scan_data_loader import SCANDataLoader
    from modification_generator import ModificationGenerator
    import tensorflow as tf
    
    try:
        # 1. Load data
        print("\n1️⃣ Loading data...")
        loader = SCANDataLoader()
        splits = loader.load_processed_splits()
        
        # Use only 10 samples for testing
        test_samples = splits['train'][:10]
        print(f"✓ Loaded {len(test_samples)} samples for testing")
        
        # 2. Build tokenizer
        print("\n2️⃣ Building tokenizer...")
        tokenizer = SCANTokenizer()
        tokenizer.build_vocabulary(test_samples)
        print(f"✓ Tokenizer built: {len(tokenizer.command_to_id)} commands, {len(tokenizer.action_to_id)} actions")
        
        # 3. Create dataset
        print("\n3️⃣ Creating dataset...")
        dataset = create_dataset(test_samples, tokenizer, batch_size=2)
        
        # Inspect dataset format
        for batch in dataset.take(1):
            if isinstance(batch, tuple) and len(batch) == 2:
                inputs, targets = batch
                print("✓ Dataset format: (inputs, targets)")
                print(f"  Input keys: {list(inputs.keys())}")
                for key, val in inputs.items():
                    print(f"    {key}: shape {val.shape}")
                print(f"  Targets shape: {targets.shape}")
            else:
                raise ValueError(f"Dataset returns {type(batch)}, not (inputs, targets)!")
        
        # 4. Create model
        print("\n4️⃣ Creating model...")
        model = create_model(
            command_vocab_size=len(tokenizer.command_to_id),
            action_vocab_size=len(tokenizer.action_to_id),
            d_model=128  # Use full size to match Paperspace
        )
        
        # 5. Compile model
        print("\n5️⃣ Compiling model...")
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        print("✓ Model compiled")
        
        # 6. Test model output shape
        print("\n6️⃣ Testing model output shape...")
        for batch in dataset.take(1):
            inputs, targets = batch
            predictions = model(inputs, training=False)
            print(f"✓ Model output shape: {predictions.shape}")
            print(f"  Target shape: {targets.shape}")
            
            # For sparse_categorical_crossentropy:
            # Model should output (batch, sequence, vocab_size)
            # Targets should be (batch, sequence) with integer indices
            if len(predictions.shape) == 3 and len(targets.shape) == 2:
                if predictions.shape[:2] == targets.shape:
                    print("✓ Shapes are compatible for sparse_categorical_crossentropy")
                else:
                    print(f"❌ ERROR: Shape mismatch!")
                    print(f"  Model batch/sequence dims: {predictions.shape[:2]}")
                    print(f"  Target batch/sequence dims: {targets.shape}")
                    return False
            elif predictions.shape == targets.shape:
                print("✓ Shapes match exactly")
            else:
                print(f"❌ ERROR: Unexpected shape configuration!")
                print(f"  Model output: {predictions.shape}")
                print(f"  Targets: {targets.shape}")
                return False
        
        # 7. Run one training step
        print("\n7️⃣ Running one training step...")
        start_time = time.time()
        
        history = model.fit(
            dataset.take(1),  # Just one batch
            epochs=1,
            verbose=1
        )
        
        elapsed = time.time() - start_time
        print(f"✓ Training step completed in {elapsed:.2f}s")
        print(f"  Loss: {history.history['loss'][0]:.4f}")
        
        # 8. Test all 4 stages (with 1 batch each)
        print("\n8️⃣ Testing all 4 training stages...")
        
        # Load modifications for later stages
        mod_generator = ModificationGenerator()
        modifications = mod_generator.load_modifications()
        
        stages = [
            ("Stage 1: Basic", test_samples, []),
            ("Stage 2: Simple Mods", test_samples[:7], modifications[:3]),
            ("Stage 3: Complex Mods", test_samples[:5], modifications[:5]),
            ("Stage 4: Novel", test_samples[:3], modifications[:7])
        ]
        
        for stage_name, base_data, mods in stages:
            print(f"\n  Testing {stage_name}...")
            
            # Create mixed dataset
            if mods:
                # Convert ModificationPair objects to dicts
                mod_dicts = []
                for m in mods[:3]:  # Just use first 3
                    if hasattr(m, 'modified_sample'):
                        # It's a ModificationPair object
                        mod_dict = m.modified_sample.to_dict()
                        mod_dict['modification'] = m.modification_description
                        mod_dicts.append(mod_dict)
                    else:
                        # It's already a dict
                        mod_dicts.append(m)
                stage_data = base_data + mod_dicts
            else:
                stage_data = base_data
            
            stage_dataset = create_dataset(stage_data, tokenizer, batch_size=2)
            
            # Train for 1 step
            history = model.fit(
                stage_dataset.take(1),
                epochs=1,
                verbose=0
            )
            print(f"  ✓ {stage_name} - Loss: {history.history['loss'][0]:.4f}")
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nThe training script should work on Paperspace.")
        print("No shape mismatches, no missing methods, no runtime errors.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        
        # Specific error handling
        if "Dimensions must be equal" in str(e):
            print("\n🔍 This is a shape mismatch error!")
            print("The model's output shape doesn't match the target shape.")
            print("Check the model's final layer and output dimensions.")
        elif "AttributeError" in str(e):
            print("\n🔍 This is a missing method/attribute error!")
            print("Something is trying to call a method that doesn't exist.")
        elif "Target data is missing" in str(e):
            print("\n🔍 This is a dataset format error!")
            print("The dataset must return (inputs, targets) tuples.")
        
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        
        return False

def main():
    print("🚀 LOCAL TRAINING TEST")
    print("This runs the actual training with minimal data to catch errors.\n")
    
    # Check prerequisites
    if not Path('data/processed/train.pkl').exists():
        print("❌ Training data not found!")
        print("Run: python scan_data_loader.py")
        sys.exit(1)
    
    if not Path('data/processed/modification_pairs.pkl').exists():
        print("❌ Modification data not found!")
        print("Run: python modification_generator.py")
        sys.exit(1)
    
    # Run test
    success = test_training()
    
    if not success:
        print("\n⚠️  Fix the errors before deploying to Paperspace!")
        print("This test just saved you hours of GPU debugging.")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()