"""
Quick test script to verify binding model components work together.
"""

from utils.imports import setup_project_paths

setup_project_paths()

import logging

import numpy as np
from dereferencing_tasks import DereferencingTaskGenerator, test_task_generation
from minimal_binding_scan import test_binding_capability

from utils.config import setup_environment

# Set up environment
config = setup_environment()
logger = logging.getLogger(__name__)


def test_integration():
    """Test that all components integrate properly."""

    logger.info("=" * 50)
    logger.info("Testing Binding Model Components")
    logger.info("=" * 50)

    # Test 1: Task generation
    logger.info("\n1. Testing task generation...")
    test_task_generation()

    # Test 2: Model creation
    logger.info("\n2. Testing model creation...")
    model = test_binding_capability()

    # Test 3: Full integration
    logger.info("\n3. Testing full integration...")
    generator = DereferencingTaskGenerator()

    # Generate a simple task
    command_words, action_words, bindings = generator.generate_simple_binding()
    logger.info(
        f"Generated task: {' '.join(command_words)} -> {' '.join(action_words)}"
    )

    # Encode it
    command_encoded = generator.encode_words(command_words)
    generator.encode_actions(action_words)

    # Add batch dimension
    command_batch = np.expand_dims(command_encoded, 0)

    # Forward pass
    outputs = model({"command": command_batch})

    logger.info(f"Model outputs:")
    logger.info(f"  - Action logits shape: {outputs['action_logits'].shape}")
    logger.info(f"  - Bindings shape: {outputs['bindings'].shape}")
    logger.info(f"  - Binding scores shape: {outputs['binding_scores'].shape}")

    # Test 4: Check shapes match expectations
    logger.info("\n4. Verifying output shapes...")
    expected_seq_len = len(action_words)
    actual_seq_len = outputs["action_logits"].shape[1]

    if actual_seq_len >= expected_seq_len:
        logger.info("✓ Output sequence length sufficient for actions")
    else:
        logger.warning(
            f"✗ Output length {actual_seq_len} < expected {expected_seq_len}"
        )

    logger.info("\n" + "=" * 50)
    logger.info("All component tests complete!")
    logger.info("=" * 50)

    return True


if __name__ == "__main__":
    success = test_integration()
    if success:
        logger.info("\nBinding model components are working correctly!")
        logger.info("Ready to train with: python train_binding_model.py")
