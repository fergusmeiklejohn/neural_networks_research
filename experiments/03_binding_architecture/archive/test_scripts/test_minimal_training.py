"""
Minimal training test to catch API errors before full runs.

This script does a quick 1-epoch training to verify all APIs work correctly.
Run this before committing any training code!
"""

from utils.imports import setup_project_paths

setup_project_paths()

import logging
import sys

import numpy as np

from utils.config import setup_environment

# Set up environment
config = setup_environment()
logger = logging.getLogger(__name__)


def test_minimal_training():
    """Test training pipeline with minimal data."""
    try:
        # Import after environment setup to catch import errors
        from dereferencing_tasks import DereferencingTaskGenerator
        from minimal_binding_scan import MinimalBindingModel
        from train_binding_model import BindingTrainer

        logger.info("Imports successful")

        # Create minimal dataset
        generator = DereferencingTaskGenerator(seed=42)
        dataset = generator.generate_dataset(n_samples=20)  # Very small

        # Create model
        model = MinimalBindingModel(
            vocab_size=len(generator.word_vocab),
            action_vocab_size=len(generator.action_vocab),
            n_slots=5,
            embed_dim=32,
            hidden_dim=64,
        )

        # Build model
        dummy_input = {"command": np.zeros((1, 10), dtype=np.int32)}
        _ = model(dummy_input)
        logger.info(f"Model built successfully with {model.count_params()} parameters")

        # Create trainer
        trainer = BindingTrainer(model, generator, "/tmp/test_binding")

        # Test single training step
        batch_commands = dataset["train"]["commands"][:2]
        batch_actions = dataset["train"]["actions"][:2]

        logger.info("Testing single training step...")
        loss, bindings = trainer.train_step(batch_commands, batch_actions)
        logger.info(f"Training step successful! Loss: {float(loss):.4f}")

        # Test evaluation
        logger.info("Testing evaluation...")
        val_loss, val_acc = trainer.evaluate(dataset["val"], "val")
        logger.info(f"Evaluation successful! Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        # Test modification capability check
        logger.info("Testing modification capability...")
        results = trainer.test_modification_capability()
        logger.info(f"Modification test successful! Found {len(results)} test cases")

        return True

    except Exception as e:
        logger.error(f"Test failed with error: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    logger.info("Running minimal training test...")
    success = test_minimal_training()

    if success:
        logger.info("\n✓ All tests passed! Safe to proceed with full training.")
        sys.exit(0)
    else:
        logger.error("\n✗ Tests failed! Fix errors before committing.")
        sys.exit(1)
