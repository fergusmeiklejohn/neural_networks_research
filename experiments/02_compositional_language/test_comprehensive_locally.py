#!/usr/bin/env python3
"""
Test comprehensive experiments locally with minimal data.
Run this before Paperspace to catch any errors.
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

from paperspace_comprehensive_experiments import ComprehensiveExperimentRunner


class LocalTestRunner(ComprehensiveExperimentRunner):
    """Modified runner for local testing with minimal data."""

    def __init__(self):
        super().__init__()
        # Override to use local test directory
        self.output_dir = (
            self.base_path
            / "experiments"
            / "02_compositional_language"
            / "test_comprehensive_local"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.storage_dir = self.output_dir

    def load_data(self):
        """Load minimal data for testing."""
        print("\n=== LOADING MINIMAL TEST DATA ===")

        # Call parent method
        super().load_data()

        # But use much less data
        self.train_data = self.train_data[:500]  # Only 500 samples
        self.val_data = self.val_data[:100]  # Only 100 validation
        self.all_modifications = self.all_modifications[:50]  # Only 50 modifications

        print(f"Reduced to {len(self.train_data)} train, {len(self.val_data)} val")
        print(f"Using only {len(self.all_modifications)} modifications")

    def run_experiment(
        self, experiment_name, model_version="v1", use_mixed_training=False
    ):
        """Run with minimal epochs for testing."""
        # Temporarily store original method

        # Override stages with minimal training
        if use_mixed_training:
            test_stages = [
                {"name": "Stage 1: Basic", "mix_ratio": 0.0, "epochs": 1, "lr": 0.001},
                {"name": "Stage 2: Mixed", "mix_ratio": 0.5, "epochs": 1, "lr": 0.0005},
            ]
        else:
            test_stages = [
                {"name": "Stage 1: Basic", "mix_ratio": 0.0, "epochs": 1, "lr": 0.001},
                {"name": "Stage 2: Mods", "mix_ratio": 1.0, "epochs": 1, "lr": 0.0005},
            ]

        # Monkey patch the stages
        self.run_experiment

        # Call parent with modified data
        print(f"\n{'='*60}")
        print(f"TEST EXPERIMENT: {experiment_name}")
        print(f"{'='*60}")

        # Create model
        cmd_vocab = len(self.tokenizer.command_to_id)
        act_vocab = len(self.tokenizer.action_to_id)

        if model_version == "v1":
            from models import create_model

            model = create_model(cmd_vocab, act_vocab, d_model=64)  # Smaller model
        else:
            from models_v2 import create_model_v2

            model = create_model_v2(cmd_vocab, act_vocab, d_model=64)  # Smaller model

        print(f"Model: {model_version}, Parameters: {model.count_params():,}")

        # Track history
        experiment_history = {
            "name": experiment_name,
            "model_version": model_version,
            "mixed_training": use_mixed_training,
            "stages": [],
        }

        # Train each stage
        for stage_idx, stage in enumerate(test_stages):
            print(f"\n{stage['name']} (TEST MODE)")
            print("-" * 40)

            # Create small dataset
            if stage["mix_ratio"] == 0.0:
                from train_progressive_curriculum import create_dataset

                dataset = create_dataset(
                    self.train_data[:100], self.tokenizer, batch_size=8
                )
            else:
                # Use very few modifications
                dataset = self.create_mixed_dataset(
                    self.train_data[:100],
                    self.all_modifications[:10],
                    mix_ratio=stage["mix_ratio"],
                    batch_size=8,
                )

            # Compile
            import tensorflow as tf

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=stage["lr"]),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            # Train for 1 epoch
            try:
                hist = model.fit(dataset.take(5), epochs=1, verbose=1)  # Only 5 batches

                metrics = {
                    "loss": float(hist.history["loss"][0]),
                    "accuracy": float(hist.history["accuracy"][0]),
                }

                experiment_history["stages"].append(
                    {
                        "name": stage["name"],
                        "mix_ratio": stage["mix_ratio"],
                        "history": [metrics],
                        "val_loss": 0.5,  # Dummy
                        "val_accuracy": 0.5,  # Dummy
                    }
                )

                print(f"✓ Stage completed - Loss: {metrics['loss']:.4f}")

            except Exception as e:
                print(f"✗ Error in stage: {e}")
                raise

        # Save test results
        self.results["experiments"][experiment_name] = experiment_history

        return experiment_history


def test_all():
    """Run minimal test of all experiments."""
    print("=" * 60)
    print("LOCAL TEST OF COMPREHENSIVE EXPERIMENTS")
    print("=" * 60)

    runner = LocalTestRunner()

    # Test each configuration
    configs = [
        ("v1_standard_test", "v1", False),
        ("v1_mixed_test", "v1", True),
        ("v2_standard_test", "v2", False),
        ("v2_mixed_test", "v2", True),
    ]

    # Load data once
    runner.load_data()

    # Quick test each
    failures = []
    for exp_name, model_version, use_mixed in configs:
        try:
            print(f"\nTesting: {exp_name}")
            runner.run_experiment(exp_name, model_version, use_mixed)
            print(f"✓ {exp_name} passed")
        except Exception as e:
            print(f"✗ {exp_name} failed: {e}")
            failures.append((exp_name, e))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    if not failures:
        print("\n✅ ALL TESTS PASSED!")
        print("\nReady for Paperspace deployment:")
        print("python paperspace_comprehensive_experiments.py")
    else:
        print(f"\n❌ {len(failures)} TESTS FAILED:")
        for name, error in failures:
            print(f"  - {name}: {error}")
        print("\nFix these issues before Paperspace deployment!")

    return len(failures) == 0


if __name__ == "__main__":
    import sys

    success = test_all()
    sys.exit(0 if success else 1)
