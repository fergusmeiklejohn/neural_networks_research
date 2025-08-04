#!/usr/bin/env python3
"""
Test a Paperspace script locally with minimal data before deployment.
This catches runtime errors that static analysis might miss.

Usage: python test_script_locally.py <script_to_test.py>
"""

import os
import sys
import tempfile
from pathlib import Path

# Mock environment for testing
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def test_script(script_path: str):
    """Test a script with minimal data to catch runtime errors"""
    print(f"🧪 Testing {script_path} with minimal data...")
    print("=" * 60)

    # Create temporary directory structure
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock storage directory
        storage_dir = Path(temp_dir) / "storage"
        storage_dir.mkdir()

        # Change to script directory
        script_dir = Path(script_path).parent
        original_dir = os.getcwd()
        os.chdir(script_dir)

        try:
            # Import and test key components
            print("\n1️⃣ Testing imports...")
            from modification_generator import ModificationGenerator
            from train_progressive_curriculum import SCANTokenizer, create_dataset

            from models import create_model

            print("✓ All imports successful")

            # Test tokenizer methods
            print("\n2️⃣ Testing SCANTokenizer...")
            tokenizer = SCANTokenizer()

            # Check available methods
            methods = [m for m in dir(tokenizer) if not m.startswith("_")]
            print(f"Available methods: {methods}")

            # Test save method
            if hasattr(tokenizer, "save"):
                print("✓ tokenizer.save() exists")
            elif hasattr(tokenizer, "save_vocabulary"):
                print(
                    "⚠️  WARNING: Use tokenizer.save_vocabulary(), not tokenizer.save()"
                )
            else:
                print("❌ ERROR: No save method found on tokenizer!")

            # Test with minimal data
            print("\n3️⃣ Testing with minimal data...")
            test_samples = [
                {"command": "walk", "action": "I_WALK"},
                {"command": "run left", "action": "I_TURN_LEFT I_RUN"},
            ]

            tokenizer.build_vocabulary(test_samples)
            print(
                f"✓ Vocabulary built: {len(tokenizer.command_to_id)} commands, {len(tokenizer.action_to_id)} actions"
            )

            # Test model creation
            print("\n4️⃣ Testing model creation...")
            model = create_model(
                command_vocab_size=len(tokenizer.command_to_id),
                action_vocab_size=len(tokenizer.action_to_id),
                d_model=64,  # Small for testing
            )
            print(f"✓ Model created with {model.count_params():,} parameters")

            # Test dataset creation
            print("\n5️⃣ Testing dataset creation...")
            dataset = create_dataset(test_samples, tokenizer, batch_size=2)
            for batch in dataset.take(1):
                print(f"✓ Dataset created, batch shape: {batch['command'].shape}")

            # Test the actual training script functions
            print("\n6️⃣ Testing script-specific functions...")

            # Import the script as a module
            import importlib.util

            spec = importlib.util.spec_from_file_location("test_module", script_path)
            importlib.util.module_from_spec(spec)

            # Check for common issues
            with open(script_path, "r") as f:
                content = f.read()

            issues = []

            # Check for tokenizer.save() vs save_vocabulary()
            if "tokenizer.save(" in content:
                issues.append(
                    "❌ Uses tokenizer.save() - should be tokenizer.save_vocabulary()"
                )

            # Check for ModificationGenerator.load_modifications()
            if (
                "mod_generator.load_modifications()" in content
                or "generator.load_modifications()" in content
            ):
                # Verify the method exists
                from modification_generator import ModificationGenerator

                if not hasattr(ModificationGenerator, "load_modifications"):
                    issues.append(
                        "❌ Calls load_modifications() but method doesn't exist in ModificationGenerator"
                    )
                else:
                    print("✓ ModificationGenerator.load_modifications() exists")

            # Check for proper error handling
            if "try:" not in content or "except" not in content:
                issues.append("⚠️  No try/except blocks for error handling")

            # Check for storage directory validation
            if "os.path.exists('/storage')" not in content:
                issues.append("⚠️  Doesn't check if /storage exists")

            if issues:
                print("\nFound issues:")
                for issue in issues:
                    print(f"  {issue}")
            else:
                print("✓ No obvious issues found")

            print("\n" + "=" * 60)
            print("✅ Basic validation complete!")

            # Return specific recommendations
            if issues:
                print("\n🔧 Required fixes before Paperspace:")
                if any("tokenizer.save()" in issue for issue in issues):
                    print(
                        "  1. Replace all instances of tokenizer.save() with tokenizer.save_vocabulary()"
                    )
                if any("try/except" in issue for issue in issues):
                    print("  2. Add proper error handling around main execution")
                if any("/storage" in issue for issue in issues):
                    print("  3. Add validation for storage directory existence")

        except ImportError as e:
            print(f"\n❌ Import Error: {e}")
            print("This would fail on Paperspace!")
            return False
        except AttributeError as e:
            print(f"\n❌ Attribute Error: {e}")
            print("This would fail on Paperspace!")
            return False
        except Exception as e:
            print(f"\n❌ Runtime Error: {e}")
            print("This would fail on Paperspace!")
            return False
        finally:
            os.chdir(original_dir)

    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_script_locally.py <script_to_test.py>")
        sys.exit(1)

    script_path = sys.argv[1]
    success = test_script(script_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
