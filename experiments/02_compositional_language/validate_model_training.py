#!/usr/bin/env python3
"""
Validation script that actually tests model.fit() to catch data format errors.
This would have caught the "Target data is missing" error.
"""

import os
import sys
from pathlib import Path


# Set backend
os.environ["KERAS_BACKEND"] = "tensorflow"


def validate_model_training(script_path: str):
    """Validate that model training actually works with the dataset"""
    print(f"🏋️ Validating model training for {script_path}...")
    print("=" * 60)

    # Change to script directory
    script_dir = Path(script_path).parent
    original_dir = os.getcwd()
    os.chdir(script_dir)

    try:
        # Import required modules
        print("\n1️⃣ Importing modules...")
        from train_progressive_curriculum import SCANTokenizer, create_dataset

        from models import create_model

        print("✓ Imports successful")

        # Create minimal test data
        print("\n2️⃣ Creating test data...")
        test_samples = [
            {"command": "walk", "action": "I_WALK", "modification": ""},
            {"command": "run left", "action": "I_TURN_LEFT I_RUN", "modification": ""},
            {
                "command": "jump right",
                "action": "I_TURN_RIGHT I_JUMP",
                "modification": "",
            },
            {
                "command": "look around left",
                "action": "I_TURN_LEFT I_LOOK I_TURN_LEFT I_LOOK I_TURN_LEFT I_LOOK I_TURN_LEFT I_LOOK",
                "modification": "",
            },
        ]

        # Build tokenizer
        tokenizer = SCANTokenizer()
        tokenizer.build_vocabulary(test_samples)
        print(
            f"✓ Tokenizer built: {len(tokenizer.command_to_id)} commands, {len(tokenizer.action_to_id)} actions"
        )

        # Create dataset
        print("\n3️⃣ Creating dataset...")
        dataset = create_dataset(test_samples, tokenizer, batch_size=2)

        # Inspect dataset format
        print("\n4️⃣ Inspecting dataset format...")
        for batch in dataset.take(1):
            if isinstance(batch, tuple) and len(batch) == 2:
                inputs, targets = batch
                print("✓ Dataset returns (inputs, targets) tuple")
                print(f"  Inputs type: {type(inputs)}")
                if isinstance(inputs, dict):
                    print(f"  Input keys: {list(inputs.keys())}")
                    for key, val in inputs.items():
                        print(f"    {key}: shape {val.shape}")
                print(f"  Targets shape: {targets.shape}")
            else:
                print(
                    f"❌ ERROR: Dataset returns {type(batch)}, not (inputs, targets) tuple!"
                )
                if isinstance(batch, dict):
                    print(f"  Keys: {list(batch.keys())}")
                return False

        # Create and compile model
        print("\n5️⃣ Creating and compiling model...")
        model = create_model(
            command_vocab_size=len(tokenizer.command_to_id),
            action_vocab_size=len(tokenizer.action_to_id),
            d_model=64,  # Small for testing
        )

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        print("✓ Model compiled")

        # Test model.fit() for 1 step
        print("\n6️⃣ Testing model.fit() for 1 step...")
        try:
            history = model.fit(
                dataset, epochs=1, steps_per_epoch=1, verbose=0  # Just 1 step
            )
            print("✓ model.fit() works!")
            print(f"  Loss: {history.history['loss'][0]:.4f}")
            if "accuracy" in history.history:
                print(f"  Accuracy: {history.history['accuracy'][0]:.4f}")
        except ValueError as e:
            print(f"❌ ERROR in model.fit(): {e}")
            if "Target data is missing" in str(e):
                print(
                    "\n🔧 DIAGNOSIS: Dataset is not returning (inputs, targets) format!"
                )
                print("   The dataset must return tuples of (inputs, targets)")
                print("   - inputs: dict with keys matching model input names")
                print("   - targets: array of target values")
                print("\n   Check that prepare_for_training() is defined and used!")
            return False
        except Exception as e:
            print(f"❌ ERROR: {type(e).__name__}: {e}")
            return False

        # Test model prediction
        print("\n7️⃣ Testing model prediction...")
        try:
            for batch in dataset.take(1):
                if isinstance(batch, tuple):
                    inputs, _ = batch
                else:
                    inputs = batch

                predictions = model(inputs, training=False)
                print(f"✓ Model prediction works! Output shape: {predictions.shape}")
        except Exception as e:
            print(f"❌ ERROR in prediction: {e}")
            return False

        print("\n" + "=" * 60)
        print("✅ Model training validation PASSED!")
        return True

    except Exception as e:
        print(f"\n❌ Validation failed: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        os.chdir(original_dir)


def check_dataset_functions():
    """Check if critical dataset functions exist"""
    print("\n🔍 Checking dataset functions...")

    try:
        from train_progressive_curriculum import prepare_for_training

        print("✓ prepare_for_training function exists")

        # Check function signature
        import inspect

        sig = inspect.signature(prepare_for_training)
        print(f"  Signature: {sig}")

        # Test the function with mock data
        import tensorflow as tf

        mock_data = {
            "command": tf.constant([[1, 2, 3, 0, 0]]),
            "action": tf.constant([[1, 4, 5, 6, 2]]),
            "has_modification": tf.constant([0]),
            "modification": tf.constant([[0, 0, 0, 0, 0]]),
        }

        result = prepare_for_training(mock_data)
        if isinstance(result, tuple) and len(result) == 2:
            print("✓ prepare_for_training returns (inputs, targets) tuple")
        else:
            print(f"❌ prepare_for_training returns {type(result)}, not tuple!")

    except ImportError:
        print("❌ ERROR: prepare_for_training function not found!")
        print("   This function must be defined to format data for model.fit()")
        return False
    except Exception as e:
        print(f"❌ ERROR testing prepare_for_training: {e}")
        return False

    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_model_training.py <script.py>")
        sys.exit(1)

    script_path = sys.argv[1]
    if not Path(script_path).exists():
        print(f"Error: {script_path} not found")
        sys.exit(1)

    # Run validations
    dataset_ok = check_dataset_functions()
    training_ok = validate_model_training(script_path)

    if dataset_ok and training_ok:
        print("\n✅ All model training validations passed!")
        sys.exit(0)
    else:
        print("\n❌ Model training validation FAILED!")
        print("\nCommon fixes:")
        print(
            "1. Ensure prepare_for_training() is defined in train_progressive_curriculum.py"
        )
        print("2. Check that dataset returns (inputs, targets) tuples")
        print("3. Verify model is compiled before fit()")
        print("4. Make sure all code is committed and pushed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
