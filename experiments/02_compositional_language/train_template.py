#!/usr/bin/env python3
"""
Reusable Training Template for Distribution Invention Experiments
This template includes all lessons learned from compositional language training
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import gc
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

# Configure GPU properly
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ GPU memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"✗ GPU configuration failed: {e}")


class SafeTrainingPipeline:
    """Reusable training pipeline with comprehensive safety features"""

    def __init__(
        self,
        experiment_name: str,
        config: Dict[str, Any],
        use_storage: bool = True,
        use_wandb: bool = True,
    ):
        """
        Initialize training pipeline

        Args:
            experiment_name: Name of the experiment
            config: Training configuration dictionary
            use_storage: Whether to save to /storage (Paperspace)
            use_wandb: Whether to use Weights & Biases
        """
        self.experiment_name = experiment_name
        self.config = config
        self.use_storage = use_storage and os.path.exists("/storage")
        self.use_wandb = use_wandb

        # Setup directories
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.setup_directories()

        # Initialize logging
        self.setup_logging()

        # Initialize wandb if requested
        if self.use_wandb:
            self.init_wandb()

    def setup_directories(self):
        """Create all necessary directories"""
        # Local output directory
        self.output_dir = Path(f"outputs/{self.experiment_name}_{self.timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint directory
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Storage directory (if on Paperspace)
        if self.use_storage:
            self.storage_dir = Path(f"/storage/{self.experiment_name}_{self.timestamp}")
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created storage directory: {self.storage_dir}")
        else:
            self.storage_dir = None

    def setup_logging(self):
        """Setup comprehensive logging"""
        self.log_file = self.output_dir / "training.log"
        self.metrics_file = self.output_dir / "metrics.json"

        # Initialize metrics tracking
        self.metrics = {
            "experiment": self.experiment_name,
            "timestamp": self.timestamp,
            "config": self.config,
            "stages": {},
        }

    def init_wandb(self):
        """Initialize Weights & Biases"""
        try:
            import wandb

            wandb.init(
                project=self.config.get("wandb_project", "distribution-invention"),
                name=f"{self.experiment_name}_{self.timestamp}",
                config=self.config,
            )
            self.wandb = wandb
            print("✓ Initialized Weights & Biases")
        except Exception as e:
            print(f"✗ Failed to initialize wandb: {e}")
            self.use_wandb = False
            self.wandb = None

    def create_model(self) -> keras.Model:
        """
        Override this method to create your specific model

        Returns:
            Compiled Keras model
        """
        raise NotImplementedError("Subclass must implement create_model()")

    def load_data(self) -> Dict[str, Any]:
        """
        Override this method to load your data

        Returns:
            Dictionary with train/val/test datasets
        """
        raise NotImplementedError("Subclass must implement load_data()")

    def save_checkpoint(self, model: keras.Model, stage: int, metrics: Dict = None):
        """Save model checkpoint with safety redundancy"""
        checkpoint_name = f"stage_{stage}_checkpoint"

        # Save locally
        local_path = self.checkpoint_dir / f"{checkpoint_name}.h5"
        model.save_weights(str(local_path))
        print(f"✓ Saved checkpoint locally: {local_path}")

        # Save to storage if available
        if self.storage_dir:
            storage_path = self.storage_dir / f"{checkpoint_name}.h5"
            model.save_weights(str(storage_path))
            print(f"✓ Saved checkpoint to storage: {storage_path}")

        # Save metrics alongside
        if metrics:
            metrics_path = self.checkpoint_dir / f"{checkpoint_name}_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)

            if self.storage_dir:
                storage_metrics = self.storage_dir / f"{checkpoint_name}_metrics.json"
                with open(storage_metrics, "w") as f:
                    json.dump(metrics, f, indent=2)

    def log_metrics(self, stage: int, epoch: int, metrics: Dict):
        """Log metrics to multiple destinations"""
        # Update internal tracking
        if stage not in self.metrics["stages"]:
            self.metrics["stages"][stage] = []

        self.metrics["stages"][stage].append(
            {
                "epoch": epoch,
                "metrics": metrics,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Save to file
        with open(self.metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=2)

        # Log to wandb
        if self.use_wandb and self.wandb:
            log_dict = {f"stage_{stage}/{k}": v for k, v in metrics.items()}
            log_dict["epoch"] = epoch
            self.wandb.log(log_dict)

        # Print to console
        print(f"Stage {stage}, Epoch {epoch}: {metrics}")

    def train_stage(
        self,
        model: keras.Model,
        dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        stage: int,
        epochs: int,
        learning_rate: float,
    ) -> Dict[str, float]:
        """
        Train one stage with comprehensive safety features

        Returns:
            Dictionary of final metrics for the stage
        """
        print(f"\n{'='*60}")
        print(f"STAGE {stage} - Learning Rate: {learning_rate}")
        print(f"{'='*60}")

        # Setup optimizer - use legacy for compatibility
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)

        # Loss function
        loss_fn = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )

        best_val_metric = -float("inf")
        stage_start_time = time.time()

        for epoch in range(epochs):
            epoch_start_time = time.time()

            # Training
            train_loss = self.train_epoch(model, dataset, optimizer, loss_fn)

            # Validation
            val_metrics = self.validate(model, val_dataset)
            val_metrics["train_loss"] = train_loss

            # Log metrics
            self.log_metrics(stage, epoch, val_metrics)

            # Save best model
            val_metric = val_metrics.get(
                "accuracy", -val_metrics.get("loss", float("inf"))
            )
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                self.save_checkpoint(model, stage, val_metrics)
                print(f"✓ New best model saved!")

            # Periodic garbage collection
            if epoch % 5 == 0:
                gc.collect()

            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.1f}s")

        stage_time = time.time() - stage_start_time
        print(f"Stage {stage} completed in {stage_time/60:.1f} minutes")

        return val_metrics

    def train_epoch(
        self,
        model: keras.Model,
        dataset: tf.data.Dataset,
        optimizer: keras.optimizers.Optimizer,
        loss_fn,
    ) -> float:
        """Train for one epoch with error handling"""
        total_loss = 0
        num_batches = 0

        pbar = tqdm(dataset, desc="Training")
        for batch in pbar:
            try:
                loss = self.train_step(model, batch, optimizer, loss_fn)
                total_loss += loss
                num_batches += 1
                pbar.set_postfix({"loss": f"{total_loss/num_batches:.4f}"})
            except Exception as e:
                print(f"\nWarning: Batch failed with error: {e}")
                continue

        return total_loss / max(num_batches, 1)

    def train_step(self, model, batch, optimizer, loss_fn) -> float:
        """Single training step - override for custom training"""
        with tf.GradientTape() as tape:
            # Forward pass
            outputs = model(batch["inputs"], training=True)

            # Compute loss
            loss = self.compute_loss(outputs, batch["targets"], loss_fn)

        # Backward pass
        gradients = tape.gradient(loss, model.trainable_variables)

        # Clip gradients
        gradients = [
            tf.clip_by_norm(g, 1.0) if g is not None else None for g in gradients
        ]

        # Apply gradients
        optimizer.apply_gradients(
            [
                (g, v)
                for g, v in zip(gradients, model.trainable_variables)
                if g is not None
            ]
        )

        return loss.numpy()

    def compute_loss(self, outputs, targets, loss_fn):
        """Compute loss - override for custom loss computation"""
        return tf.reduce_mean(loss_fn(targets, outputs))

    def validate(
        self, model: keras.Model, dataset: tf.data.Dataset
    ) -> Dict[str, float]:
        """Validate model - override for custom validation"""
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch in dataset.take(100):  # Limit validation for speed
            outputs = model(batch["inputs"], training=False)

            # Compute metrics
            loss = self.compute_loss(
                outputs,
                batch["targets"],
                keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            )
            predictions = tf.argmax(outputs, axis=-1)
            correct = tf.reduce_sum(
                tf.cast(predictions == batch["targets"], tf.float32)
            )

            total_loss += loss.numpy() * batch["targets"].shape[0]
            total_correct += correct.numpy()
            total_samples += batch["targets"].shape[0]

        return {
            "loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples,
        }

    def run_progressive_curriculum(self, stages: List[Dict[str, Any]]):
        """Run full progressive curriculum training"""
        print(f"\nStarting {self.experiment_name} Progressive Curriculum")
        print(f"Output directory: {self.output_dir}")
        if self.storage_dir:
            print(f"Storage directory: {self.storage_dir}")

        # Load data
        print("\nLoading data...")
        data = self.load_data()

        # Create model
        print("\nCreating model...")
        model = self.create_model()

        # Run each stage
        for stage_idx, stage_config in enumerate(stages, 1):
            # Get stage-specific dataset if provided
            train_data = stage_config.get("train_data", data["train"])
            val_data = stage_config.get("val_data", data["val"])

            # Train stage
            metrics = self.train_stage(
                model=model,
                dataset=train_data,
                val_dataset=val_data,
                stage=stage_idx,
                epochs=stage_config["epochs"],
                learning_rate=stage_config["lr"],
            )

        # Final evaluation
        print("\n" + "=" * 60)
        print("FINAL EVALUATION")
        print("=" * 60)

        test_results = {}
        for test_name, test_data in data.get("test", {}).items():
            test_metrics = self.validate(model, test_data)
            test_results[test_name] = test_metrics
            print(f"{test_name}: {test_metrics}")

        # Save final results
        self.save_final_results(model, test_results)

        return test_results

    def save_final_results(self, model: keras.Model, test_results: Dict):
        """Save all final results with redundancy"""
        # Save final model
        self.save_checkpoint(model, stage=99, metrics=test_results)

        # Create comprehensive results file
        final_results = {
            "experiment": self.experiment_name,
            "timestamp": self.timestamp,
            "config": self.config,
            "test_results": test_results,
            "training_metrics": self.metrics,
        }

        # Save locally
        results_path = self.output_dir / "final_results.json"
        with open(results_path, "w") as f:
            json.dump(final_results, f, indent=2)

        # Save to storage
        if self.storage_dir:
            storage_results = self.storage_dir / "final_results.json"
            with open(storage_results, "w") as f:
                json.dump(final_results, f, indent=2)

        # Create downloadable archive
        self.create_results_archive()

        print(f"\n✓ All results saved successfully!")
        print(f"Local: {self.output_dir}")
        if self.storage_dir:
            print(f"Storage: {self.storage_dir}")

    def create_results_archive(self):
        """Create a downloadable archive of all results"""
        import zipfile

        archive_name = f"{self.experiment_name}_results_{self.timestamp}.zip"
        archive_path = Path(archive_name)

        with zipfile.ZipFile(archive_path, "w") as zf:
            # Add all files from output directory
            for file_path in self.output_dir.rglob("*"):
                if file_path.is_file():
                    zf.write(file_path, file_path.relative_to(self.output_dir.parent))

        print(f"\n✓ Created results archive: {archive_path}")
        print(f"Download command: sz {archive_path}")  # For terminal download


# Example usage:
class MyExperimentTraining(SafeTrainingPipeline):
    """Example implementation for a specific experiment"""

    def create_model(self):
        # Implement your model creation
        model = keras.Sequential(
            [keras.layers.Dense(128, activation="relu"), keras.layers.Dense(10)]
        )
        return model

    def load_data(self):
        # Implement your data loading
        # Return dict with 'train', 'val', and 'test' datasets
        pass


def example_usage():
    """Example of how to use the template"""
    config = {"d_model": 128, "batch_size": 32, "wandb_project": "my-experiment"}

    stages = [
        {"epochs": 10, "lr": 1e-3},
        {"epochs": 10, "lr": 5e-4},
        {"epochs": 10, "lr": 2e-4},
        {"epochs": 10, "lr": 1e-4},
    ]

    # Create and run training pipeline
    pipeline = MyExperimentTraining("my_experiment", config)
    results = pipeline.run_progressive_curriculum(stages)

    print(f"Training complete! Results: {results}")


if __name__ == "__main__":
    print("This is a template file. Create a subclass for your specific experiment.")
    print("See example_usage() for how to use this template.")
