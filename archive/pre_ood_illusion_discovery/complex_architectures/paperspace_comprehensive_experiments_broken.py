#!/usr/bin/env python3
"""
Comprehensive Paperspace experiment runner for compositional language.

Runs multiple experiments to test:
1. Original model (v1) with standard training
2. Original model (v1) with mixed training
3. Improved model (v2) with standard training
4. Improved model (v2) with mixed training

All with comprehensive safeguards and result preservation.
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import json
import pickle
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from models_v2 import create_model_v2
from modification_generator import ModificationGenerator
from scan_data_loader import SCANDataLoader
from train_progressive_curriculum import SCANTokenizer, create_dataset

# Import all our modules
from models import create_model


class ComprehensiveExperimentRunner:
    """Runs all experiment variations with proper tracking."""

    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.setup_directories()
        self.results = {
            "timestamp": self.timestamp,
            "experiments": {},
            "comparisons": {},
        }

    def setup_directories(self):
        """Set up output and storage directories."""
        # Detect environment
        if os.path.exists("/notebooks/neural_networks_research"):
            self.base_path = Path("/notebooks/neural_networks_research")
        elif os.path.exists("/workspace/neural_networks_research"):
            self.base_path = Path("/workspace/neural_networks_research")
        else:
            self.base_path = Path.cwd().parent.parent

        # Main output directory
        self.output_dir = (
            self.base_path
            / "experiments"
            / "02_compositional_language"
            / f"comprehensive_results_{self.timestamp}"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Persistent storage for Paperspace
        if os.path.exists("/storage"):
            self.storage_dir = Path(
                f"/storage/compositional_comprehensive_{self.timestamp}"
            )
            self.storage_dir.mkdir(exist_ok=True)
            print(f"Using persistent storage: {self.storage_dir}")
        else:
            self.storage_dir = self.output_dir
            print(f"No /storage found, using local dir: {self.storage_dir}")

    def save_checkpoint(self, model, experiment_name, stage, epoch, metrics):
        """Save model checkpoint with emergency backup."""
        checkpoint_name = f"{experiment_name}_stage_{stage}_epoch_{epoch}.h5"

        # Save to both locations
        local_path = self.output_dir / "checkpoints" / checkpoint_name
        local_path.parent.mkdir(exist_ok=True)
        model.save_weights(local_path)

        if self.storage_dir != self.output_dir:
            storage_path = self.storage_dir / "checkpoints" / checkpoint_name
            storage_path.parent.mkdir(exist_ok=True)
            model.save_weights(storage_path)

        # Also save metrics
        metrics_data = {
            "experiment": experiment_name,
            "stage": stage,
            "epoch": epoch,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }

        metrics_path = (
            self.output_dir / "checkpoints" / f"{experiment_name}_metrics.jsonl"
        )
        with open(metrics_path, "a") as f:
            f.write(json.dumps(metrics_data) + "\n")

    def load_data(self):
        """Load and prepare all data."""
        print("\n=== LOADING DATA ===")

        # Load SCAN data
        data_loader = SCANDataLoader()
        data_loader.load_all_data()
        splits = data_loader.create_isolated_splits()

        # Keep samples
        self.train_samples = splits["train"]
        self.val_samples = splits["val_interpolation"]

        # Convert to dict format
        self.train_data = [
            {"command": s.command, "action": s.action} for s in self.train_samples
        ]
        self.val_data = [
            {"command": s.command, "action": s.action} for s in self.val_samples
        ]

        print(f"Loaded {len(self.train_data)} training samples")

        # Create tokenizer
        self.tokenizer = SCANTokenizer()
        self.tokenizer.build_vocabulary(self.train_data)
        self.tokenizer.save_vocabulary(self.output_dir / "vocabulary.json")

        # Generate modifications
        print("\nGenerating modifications...")
        mod_gen = ModificationGenerator()

        # Generate from subset for speed
        mod_dict = mod_gen.generate_all_modifications(self.train_samples[:2000])

        # Flatten modifications
        self.all_modifications = []
        for mod_type, pairs in mod_dict.items():
            self.all_modifications.extend(pairs)

        print(f"Generated {len(self.all_modifications)} modifications")

        # Save modifications
        with open(self.output_dir / "modifications.pkl", "wb") as f:
            pickle.dump(self.all_modifications, f)

    def create_mixed_dataset(self, base_data, modifications, mix_ratio, batch_size=32):
        """Create mixed dataset with base and modified examples."""
        # Convert modifications to training format
        mod_data = []
        for pair in modifications:
            mod_data.append(
                {
                    "command": pair.modified_sample.command,
                    "action": pair.modified_sample.action,
                    "modification": pair.modification_description,
                }
            )

        # Calculate sizes
        n_modified = int(len(mod_data) * mix_ratio)
        n_base = (
            int(n_modified / mix_ratio * (1 - mix_ratio))
            if mix_ratio > 0
            else len(base_data)
        )

        # Sample
        sampled_base = np.random.choice(
            base_data, size=min(n_base, len(base_data)), replace=False
        )
        sampled_mods = np.random.choice(
            mod_data, size=min(n_modified, len(mod_data)), replace=False
        )

        # Convert base to have empty modifications
        base_with_empty = []
        for sample in sampled_base:
            base_with_empty.append(
                {
                    "command": sample["command"],
                    "action": sample["action"],
                    "modification": "",
                }
            )

        # Combine
        combined = list(base_with_empty) + list(sampled_mods)
        np.random.shuffle(combined)

        return create_dataset(combined, self.tokenizer, batch_size)

    def run_experiment(
        self, experiment_name, model_version="v1", use_mixed_training=False
    ):
        """Run a single experiment configuration."""
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {experiment_name}")
        print(f"{'='*60}")

        # Create model
        cmd_vocab = len(self.tokenizer.command_to_id)
        act_vocab = len(self.tokenizer.action_to_id)

        if model_version == "v1":
            model = create_model(cmd_vocab, act_vocab, d_model=128)
        else:
            model = create_model_v2(cmd_vocab, act_vocab, d_model=128)

        print(f"Model: {model_version}, Parameters: {model.count_params():,}")

        # Training stages
        if use_mixed_training:
            stages = [
                {
                    "name": "Stage 1: Basic SCAN",
                    "mix_ratio": 0.0,
                    "epochs": 3,
                    "lr": 0.001,
                },
                {
                    "name": "Stage 2: Mixed 70/30",
                    "mix_ratio": 0.3,
                    "epochs": 3,
                    "lr": 0.0005,
                },
                {
                    "name": "Stage 3: Mixed 50/50",
                    "mix_ratio": 0.5,
                    "epochs": 3,
                    "lr": 0.0002,
                },
                {
                    "name": "Stage 4: Mixed 30/70",
                    "mix_ratio": 0.7,
                    "epochs": 3,
                    "lr": 0.0001,
                },
            ]
        else:
            stages = [
                {
                    "name": "Stage 1: Basic SCAN",
                    "mix_ratio": 0.0,
                    "epochs": 3,
                    "lr": 0.001,
                },
                {
                    "name": "Stage 2: Simple Mods",
                    "mix_ratio": 1.0,
                    "epochs": 3,
                    "lr": 0.0005,
                },
                {
                    "name": "Stage 3: Complex Mods",
                    "mix_ratio": 1.0,
                    "epochs": 3,
                    "lr": 0.0002,
                },
                {
                    "name": "Stage 4: Novel Gen",
                    "mix_ratio": 1.0,
                    "epochs": 3,
                    "lr": 0.0001,
                },
            ]

        # Track history
        experiment_history = {
            "name": experiment_name,
            "model_version": model_version,
            "mixed_training": use_mixed_training,
            "stages": [],
        }

        # Train each stage
        for stage_idx, stage in enumerate(stages):
            print(f"\n{stage['name']}")
            print("-" * 40)

            # Create dataset
            if stage["mix_ratio"] == 0.0:
                dataset = create_dataset(
                    self.train_data[:5000], self.tokenizer, batch_size=32
                )
            elif stage["mix_ratio"] == 1.0:
                # Pure modifications
                mod_data = []
                mods_to_use = self.all_modifications[: 300 * (stage_idx)]
                for pair in mods_to_use:
                    mod_data.append(
                        {
                            "command": pair.modified_sample.command,
                            "action": pair.modified_sample.action,
                            "modification": pair.modification_description,
                        }
                    )
                dataset = create_dataset(mod_data, self.tokenizer, batch_size=32)
            else:
                # Mixed dataset
                mods_to_use = self.all_modifications[: 300 * stage_idx]
                dataset = self.create_mixed_dataset(
                    self.train_data[:5000], mods_to_use, mix_ratio=stage["mix_ratio"]
                )

            # Compile
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=stage["lr"]),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            # Train
            stage_history = []
            for epoch in range(stage["epochs"]):
                print(f"Epoch {epoch+1}/{stage['epochs']}")

                try:
                    # Train
                    hist = model.fit(dataset, epochs=1, verbose=1)

                    # Record
                    metrics = {
                        "loss": float(hist.history["loss"][0]),
                        "accuracy": float(hist.history["accuracy"][0]),
                    }
                    stage_history.append(metrics)

                    # Save checkpoint
                    self.save_checkpoint(
                        model, experiment_name, stage_idx + 1, epoch + 1, metrics
                    )

                except Exception as e:
                    print(f"Error in training: {e}")
                    # Save emergency checkpoint
                    emergency_path = (
                        self.storage_dir
                        / f"emergency_{experiment_name}_s{stage_idx}_e{epoch}.h5"
                    )
                    try:
                        model.save_weights(emergency_path)
                        print(f"Emergency save to: {emergency_path}")
                    except:
                        pass
                    raise

            # Validate
            val_dataset = create_dataset(
                self.val_data[:1000], self.tokenizer, batch_size=32
            )
            val_metrics = model.evaluate(val_dataset, verbose=0)

            # Record stage
            experiment_history["stages"].append(
                {
                    "name": stage["name"],
                    "mix_ratio": stage["mix_ratio"],
                    "history": stage_history,
                    "val_loss": float(val_metrics[0]),
                    "val_accuracy": float(val_metrics[1]),
                }
            )

            print(f"Validation - Loss: {val_metrics[0]:.4f}, Acc: {val_metrics[1]:.4f}")

        # Save experiment results
        self.results["experiments"][experiment_name] = experiment_history

        # Save immediately
        with open(self.output_dir / "results.json", "w") as f:
            json.dump(self.results, f, indent=2)

        if self.storage_dir != self.output_dir:
            with open(self.storage_dir / "results.json", "w") as f:
                json.dump(self.results, f, indent=2)

        return experiment_history

    def run_all_experiments(self):
        """Run all experiment configurations."""
        experiments = [
            ("v1_standard", "v1", False),
            ("v1_mixed", "v1", True),
            ("v2_standard", "v2", False),
            ("v2_mixed", "v2", True),
        ]

        # Load data once
        self.load_data()

        # Run each experiment
        for exp_name, model_version, use_mixed in experiments:
            try:
                self.run_experiment(exp_name, model_version, use_mixed)
            except Exception as e:
                print(f"\nERROR in {exp_name}: {e}")
                import traceback

                traceback.print_exc()

                # Record failure
                self.results["experiments"][exp_name] = {
                    "name": exp_name,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }

        # Final analysis
        self.analyze_results()

    def analyze_results(self):
        """Analyze and compare all experiments."""
        print("\n\n" + "=" * 60)
        print("EXPERIMENT COMPARISON")
        print("=" * 60)

        comparisons = {}

        for exp_name, exp_data in self.results["experiments"].items():
            if "error" in exp_data:
                print(f"\n{exp_name}: FAILED - {exp_data['error']}")
                continue

            # Get key metrics
            stage1_acc = (
                exp_data["stages"][0]["val_accuracy"] if exp_data["stages"] else 0
            )
            final_acc = (
                exp_data["stages"][-1]["val_accuracy"] if exp_data["stages"] else 0
            )
            degradation = (
                (stage1_acc - final_acc) / stage1_acc * 100 if stage1_acc > 0 else 0
            )

            comparisons[exp_name] = {
                "stage1_accuracy": stage1_acc,
                "final_accuracy": final_acc,
                "degradation_percent": degradation,
                "model_version": exp_data.get("model_version", "unknown"),
                "mixed_training": exp_data.get("mixed_training", False),
            }

            print(f"\n{exp_name}:")
            print(f"  Stage 1 Accuracy: {stage1_acc:.4f}")
            print(f"  Final Accuracy: {final_acc:.4f}")
            print(f"  Degradation: {degradation:.1f}%")

        self.results["comparisons"] = comparisons

        # Save final results
        final_path = self.output_dir / "final_results.json"
        with open(final_path, "w") as f:
            json.dump(self.results, f, indent=2)

        if self.storage_dir != self.output_dir:
            shutil.copy(final_path, self.storage_dir / "final_results.json")

        print(f"\n\nResults saved to: {self.output_dir}")
        print(f"Storage backup: {self.storage_dir}")


def main():
    """Run comprehensive experiments."""
    print("COMPOSITIONAL LANGUAGE COMPREHENSIVE EXPERIMENTS")
    print("Starting at:", datetime.now().isoformat())

    runner = ComprehensiveExperimentRunner()
    runner.run_all_experiments()

    print("\nExperiments complete at:", datetime.now().isoformat())


if __name__ == "__main__":
    main()
