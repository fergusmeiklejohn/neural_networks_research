#!/usr/bin/env python3
"""Test script to verify the dataset format fix for model.fit()"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

from train_progressive_curriculum import SCANTokenizer, create_dataset

from models import create_model


def test_dataset_format():
    """Test that create_dataset returns the correct format for model.fit()"""
    print("Testing dataset format fix...")

    # Create sample data
    samples = [
        {"command": "jump", "action": "JUMP"},
        {"command": "walk", "action": "WALK"},
        {"command": "run", "action": "RUN"},
        {"command": "jump twice", "action": "JUMP JUMP"},
        {"command": "walk and run", "action": "WALK RUN"},
    ]

    # Create tokenizer and build vocabulary
    tokenizer = SCANTokenizer()
    tokenizer.build_vocabulary(samples)
    print(f"Command vocab size: {len(tokenizer.command_to_id)}")
    print(f"Action vocab size: {len(tokenizer.action_to_id)}")

    # Create dataset
    dataset = create_dataset(samples, tokenizer, batch_size=2)

    # Check dataset format
    print("\nChecking dataset format...")
    for batch_inputs, batch_targets in dataset.take(1):
        print(f"Inputs type: {type(batch_inputs)}")
        print(f"Inputs keys: {batch_inputs.keys()}")
        print(f"Command shape: {batch_inputs['command'].shape}")
        print(f"Target shape: {batch_inputs['target'].shape}")
        print(f"Modification shape: {batch_inputs['modification'].shape}")
        print(f"Targets shape: {batch_targets.shape}")

        # Verify shapes match what the model expects
        assert isinstance(batch_inputs, dict), "Inputs should be a dictionary"
        assert "command" in batch_inputs, "Missing 'command' in inputs"
        assert "target" in batch_inputs, "Missing 'target' in inputs"
        assert "modification" in batch_inputs, "Missing 'modification' in inputs"
        assert (
            batch_inputs["target"].shape[1] == batch_targets.shape[1]
        ), "Target sequence lengths should match"

    print("\nDataset format is correct!")

    # Test with model
    print("\nTesting with actual model...")
    model = create_model(
        command_vocab_size=len(tokenizer.command_to_id),
        action_vocab_size=len(tokenizer.action_to_id),
        d_model=64,  # Small for testing
    )

    # Test fit with one batch
    try:
        history = model.fit(dataset.take(1), epochs=1, verbose=1)
        print("Model training successful!")
        print(f"Loss: {history.history['loss'][0]:.4f}")
        if "accuracy" in history.history:
            print(f"Accuracy: {history.history['accuracy'][0]:.4f}")
    except Exception as e:
        print(f"Error during model.fit(): {e}")
        raise

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_dataset_format()
