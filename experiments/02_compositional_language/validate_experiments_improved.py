#!/usr/bin/env python3
"""
Improved validation script that catches issues we missed before.

New tests:
1. Test count_params before and after building
2. Test with realistic batch sizes and shapes
3. Test tf.cond with different modification conditions
4. Test model saving/loading
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from models_v2_fixed import create_model_v2_fixed
from scan_data_loader import SCANDataLoader
from train_progressive_curriculum import SCANTokenizer, create_dataset

# Import all our modules to test
from models import create_model


class ImprovedValidator:
    """Enhanced validation to catch more issues before Paperspace."""

    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "errors": [],
            "warnings": [],
        }
        self.passed = 0
        self.failed = 0

    def test(self, name, func):
        """Run a test and record results."""
        print(f"\n{'='*60}")
        print(f"TEST: {name}")
        print("=" * 60)

        try:
            result = func()
            self.results["tests"][name] = {"success": True, "details": result}
            self.passed += 1
            print(f"✓ PASSED")
            if result:
                print(f"Details: {result}")
        except Exception as e:
            self.results["tests"][name] = {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            self.failed += 1
            print(f"✗ FAILED: {e}")
            self.results["errors"].append(f"{name}: {str(e)}")

    def test_model_count_params(self):
        """Test count_params before and after building."""
        print("Testing model parameter counting...")

        # Test v1 model
        model_v1 = create_model(20, 10, d_model=64)

        # This should fail
        try:
            params = model_v1.count_params()
            raise AssertionError(
                f"count_params should have failed but returned {params}"
            )
        except ValueError:
            print("✓ Correctly caught count_params error before building")

        # Build model
        dummy_inputs = {
            "command": tf.constant([[1, 2, 3, 4, 5]]),
            "target": tf.constant([[1, 2, 3, 4, 5, 6]]),
            "modification": tf.constant([[1, 2, 3]]),
        }
        _ = model_v1(dummy_inputs, training=False)

        # Now it should work
        params = model_v1.count_params()
        print(f"✓ After building: {params:,} parameters")

        # Test v2 model
        model_v2 = create_model_v2_fixed(20, 10, d_model=64)
        params_v2 = model_v2.count_params()
        print(f"✓ V2 model: {params_v2:,} parameters")

        return f"v1: {params:,}, v2: {params_v2:,}"

    def test_realistic_shapes(self):
        """Test with shapes that match real training."""
        print("Testing with realistic Paperspace shapes...")

        # Create models
        model_v1 = create_model(20, 10, d_model=128)
        model_v2 = create_model_v2_fixed(20, 10, d_model=128)

        # Realistic batch from Paperspace error
        batch = {
            "command": tf.constant(np.random.randint(0, 20, size=(32, 50))),
            "target": tf.constant(np.random.randint(0, 10, size=(32, 99))),
            "modification": tf.constant(np.random.randint(0, 20, size=(32, 20))),
        }

        print(f"Batch shapes: {[(k, v.shape) for k, v in batch.items()]}")

        # Test v1
        output_v1 = model_v1(batch, training=True)
        print(f"✓ V1 output shape: {output_v1.shape}")

        # Test v2
        output_v2 = model_v2(batch, training=True)
        print(f"✓ V2 output shape: {output_v2.shape}")

        # Test with compile and fit
        for name, model in [("v1", model_v1), ("v2", model_v2)]:
            model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            # Create a small dataset
            dataset = tf.data.Dataset.from_tensor_slices((batch, batch["target"]))
            dataset = dataset.batch(8)

            # Try one training step
            history = model.fit(dataset.take(1), epochs=1, verbose=0)
            print(f"✓ {name} training step: loss={history.history['loss'][0]:.4f}")

        return "Both models work with realistic shapes"

    def test_modification_conditions(self):
        """Test tf.cond with various modification conditions."""
        print("Testing modification conditions...")

        model = create_model_v2_fixed(20, 10, d_model=64)

        # Test 1: No modification (empty)
        batch_no_mod = {
            "command": tf.constant([[1, 2, 3, 4, 5]]),
            "target": tf.constant([[1, 2, 3, 4, 5, 6]]),
            "modification": tf.constant([[0, 0, 0]]),  # All zeros
        }

        output1 = model(batch_no_mod, training=True)
        print(f"✓ No modification: {output1.shape}")

        # Test 2: With modification
        batch_with_mod = {
            "command": tf.constant([[1, 2, 3, 4, 5]]),
            "target": tf.constant([[1, 2, 3, 4, 5, 6]]),
            "modification": tf.constant([[7, 8, 9]]),  # Non-zero
        }

        output2 = model(batch_with_mod, training=True)
        print(f"✓ With modification: {output2.shape}")

        # Test 3: Mixed batch (some with, some without)
        batch_mixed = {
            "command": tf.constant([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]]),
            "target": tf.constant([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]]),
            "modification": tf.constant([[7, 8, 9], [0, 0, 0]]),
        }

        output3 = model(batch_mixed, training=True)
        print(f"✓ Mixed batch: {output3.shape}")

        return "tf.cond works with all modification conditions"

    def test_model_save_load(self):
        """Test model saving and loading."""
        print("Testing model save/load...")

        # Create and build models
        model_v1 = create_model(20, 10, d_model=64)
        model_v2 = create_model_v2_fixed(20, 10, d_model=64)

        # Build v1 model
        dummy = {
            "command": tf.constant([[1, 2, 3, 4, 5]]),
            "target": tf.constant([[1, 2, 3, 4, 5, 6]]),
            "modification": tf.constant([[7, 8, 9]]),
        }
        _ = model_v1(dummy, training=False)

        # Save weights (Keras now requires .weights.h5 extension)
        v1_path = Path("temp_v1_weights.weights.h5")
        v2_path = Path("temp_v2_weights.weights.h5")

        model_v1.save_weights(v1_path)
        model_v2.save_weights(v2_path)

        print("✓ Saved weights")

        # Create new models and load
        new_v1 = create_model(20, 10, d_model=64)
        new_v2 = create_model_v2_fixed(20, 10, d_model=64)

        # Build new v1
        _ = new_v1(dummy, training=False)

        # Load weights
        new_v1.load_weights(v1_path)
        new_v2.load_weights(v2_path)

        print("✓ Loaded weights")

        # Clean up
        v1_path.unlink()
        v2_path.unlink()

        return "Save/load works for both models"

    def test_full_training_pipeline(self):
        """Test the complete training pipeline with real data."""
        print("Testing full training pipeline...")

        # Load minimal data
        data_loader = SCANDataLoader()
        data_loader.load_all_data()
        splits = data_loader.create_isolated_splits()

        # Get small samples
        train_data = [
            {"command": s.command, "action": s.action} for s in splits["train"][:100]
        ]

        # Create tokenizer
        tokenizer = SCANTokenizer()
        tokenizer.build_vocabulary(train_data)

        # Create dataset
        dataset = create_dataset(train_data, tokenizer, batch_size=8)

        # Test both models
        for model_name, model_func in [
            (
                "v1",
                lambda: create_model(
                    len(tokenizer.command_to_id),
                    len(tokenizer.action_to_id),
                    d_model=64,
                ),
            ),
            (
                "v2",
                lambda: create_model_v2_fixed(
                    len(tokenizer.command_to_id),
                    len(tokenizer.action_to_id),
                    d_model=64,
                ),
            ),
        ]:
            print(f"\nTesting {model_name}...")

            model = model_func()

            # Build model if v1
            if model_name == "v1":
                for batch in dataset.take(1):
                    _ = model(batch[0], training=False)

            # Compile
            model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            # Train for 1 epoch
            history = model.fit(dataset.take(5), epochs=1, verbose=0)

            print(f"✓ {model_name} trained: loss={history.history['loss'][0]:.4f}")

        return "Full pipeline works for both models"

    def run_all_tests(self):
        """Run all validation tests."""
        print("=" * 60)
        print("IMPROVED PAPERSPACE VALIDATION")
        print("=" * 60)

        # Define all tests
        tests = [
            ("Model Parameter Counting", self.test_model_count_params),
            ("Realistic Training Shapes", self.test_realistic_shapes),
            ("Modification Conditions", self.test_modification_conditions),
            ("Model Save/Load", self.test_model_save_load),
            ("Full Training Pipeline", self.test_full_training_pipeline),
        ]

        # Run each test
        for name, test_func in tests:
            self.test(name, test_func)

        # Summary
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)

        print(f"\nTests: {self.passed} passed, {self.failed} failed")

        if self.results["errors"]:
            print(f"\n❌ Errors:")
            for error in self.results["errors"]:
                print(f"  - {error}")

        # Save results
        with open("validation_results_improved.json", "w") as f:
            json.dump(self.results, f, indent=2)

        if self.failed == 0:
            print("\n✅ ALL TESTS PASSED - Ready for Paperspace!")
            print("\nKey improvements over previous validation:")
            print("- Tests count_params before building (catches v1 error)")
            print("- Uses realistic batch sizes from Paperspace")
            print("- Tests various modification conditions")
            print("- Validates save/load functionality")
            print("- Full pipeline test with real data")
        else:
            print("\n❌ VALIDATION FAILED - Fix issues before deployment!")

        return self.failed == 0


def main():
    """Run improved validation."""
    validator = ImprovedValidator()
    success = validator.run_all_tests()

    if success:
        print("\n" + "=" * 60)
        print("NEXT STEPS:")
        print("1. Replace models_v2.py with models_v2_fixed.py")
        print("2. Use paperspace_comprehensive_experiments_fixed.py")
        print("3. Push to production and deploy to Paperspace")
        print("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
