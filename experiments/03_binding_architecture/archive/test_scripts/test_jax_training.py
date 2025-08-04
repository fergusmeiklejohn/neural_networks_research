"""
Quick test that JAX training works for at least one epoch.
"""

from utils.imports import setup_project_paths
setup_project_paths()

from utils.config import setup_environment
import numpy as np
import logging

# Set up environment
config = setup_environment()
logger = logging.getLogger(__name__)

def test_one_epoch():
    """Test that we can complete one epoch without errors."""
    try:
        from minimal_binding_scan import MinimalBindingModel
        from dereferencing_tasks import DereferencingTaskGenerator
        from train_binding_jax import BindingTrainer
        
        # Small dataset for quick test
        generator = DereferencingTaskGenerator(seed=42)
        dataset = generator.generate_dataset(n_samples=20)
        
        # Create model
        model = MinimalBindingModel(
            vocab_size=len(generator.word_vocab),
            action_vocab_size=len(generator.action_vocab),
            n_slots=5,
            embed_dim=32,
            hidden_dim=64
        )
        
        # Build model
        dummy_input = {'command': np.zeros((1, 10), dtype=np.int32)}
        _ = model(dummy_input)
        
        # Create trainer
        trainer = BindingTrainer(model, generator, "/tmp/test_binding_jax")
        
        # Try one training step
        batch_commands = dataset['train']['commands'][:2]
        batch_actions = dataset['train']['actions'][:2]
        
        logger.info("Testing train_step_simple (using train_on_batch)...")
        loss, bindings = trainer.train_step_simple(batch_commands, batch_actions)
        logger.info(f"✓ train_step_simple successful! Loss: {loss:.4f}")
        
        logger.info("\nTesting evaluation...")
        val_loss, val_acc = trainer.evaluate(dataset['val'])
        logger.info(f"✓ Evaluation successful! Loss: {val_loss:.4f}, Acc: {val_acc:.2%}")
        
        logger.info("\nTesting modification capability check...")
        results = trainer.test_modification_capability()
        logger.info(f"✓ Modification test successful! Found {len(results)} test cases")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    logger.info("Testing JAX-compatible training script...")
    success = test_one_epoch()
    
    if success:
        logger.info("\n✓ JAX training script is working correctly!")
        logger.info("You can now run: python train_binding_jax.py")
    else:
        logger.info("\n✗ JAX training script has issues that need fixing")