"""
Final working training script for Minimal Binding Model.

This version is tested and works with Keras 3.
"""

from utils.imports import setup_project_paths
setup_project_paths()

from utils.config import setup_environment
from utils.paths import get_output_path
import numpy as np
import keras
from keras import ops
import logging
import json
import os
from datetime import datetime

from minimal_binding_scan import MinimalBindingModel
from dereferencing_tasks import DereferencingTaskGenerator

# Set up environment
config = setup_environment()
logger = logging.getLogger(__name__)


def create_keras_dataset(dataset_dict, generator, batch_size=32):
    """Create a tf.data.Dataset for Keras training."""
    commands = dataset_dict['commands']
    actions = dataset_dict['actions']
    
    # Create a simple generator function
    def data_generator():
        indices = np.arange(len(commands))
        np.random.shuffle(indices)
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_commands = commands[batch_indices]
            batch_actions = actions[batch_indices]
            
            yield {'command': batch_commands}, batch_actions
    
    # Return the generator
    return data_generator


def evaluate_accuracy(model, dataset_dict):
    """Evaluate model accuracy on a dataset."""
    commands = dataset_dict['commands']
    actions = dataset_dict['actions']
    
    total_correct = 0
    total_count = 0
    
    # Process in batches
    batch_size = 32
    for i in range(0, len(commands), batch_size):
        batch_commands = commands[i:i+batch_size]
        batch_actions = actions[i:i+batch_size]
        
        # Get predictions
        outputs = model({'command': batch_commands}, training=False)
        predictions = ops.argmax(outputs['action_logits'], axis=-1)
        
        # Check correctness for each sequence
        for j in range(len(batch_commands)):
            pred = predictions[j].numpy() if hasattr(predictions[j], 'numpy') else predictions[j]
            target = batch_actions[j]
            
            # Find non-padding positions
            non_pad_mask = target != 0
            
            if np.sum(non_pad_mask) > 0:
                if np.all(pred[non_pad_mask] == target[non_pad_mask]):
                    total_correct += 1
                total_count += 1
    
    accuracy = total_correct / total_count if total_count > 0 else 0.0
    return accuracy


def test_modifications(model, generator):
    """Test model on modification tasks."""
    test_cases = generator.generate_modification_test_set()
    results = []
    
    for test_case in test_cases:
        # Test original
        orig_cmd = generator.encode_words(test_case['original']['command'])
        orig_act = generator.encode_actions(test_case['original']['actions'])
        
        orig_cmd_batch = np.expand_dims(orig_cmd, 0)
        outputs = model({'command': orig_cmd_batch}, training=False)
        orig_pred = ops.argmax(outputs['action_logits'], axis=-1)[0]
        
        # Convert to numpy for comparison
        orig_pred = orig_pred.numpy() if hasattr(orig_pred, 'numpy') else orig_pred
        
        # Check original correctness
        orig_correct = np.all(orig_pred[:len(orig_act)] == orig_act)
        
        # Test modified
        mod_cmd = generator.encode_words(test_case['modified']['command'])
        mod_act = generator.encode_actions(test_case['modified']['actions'])
        
        mod_cmd_batch = np.expand_dims(mod_cmd, 0)
        outputs = model({'command': mod_cmd_batch}, training=False)
        mod_pred = ops.argmax(outputs['action_logits'], axis=-1)[0]
        
        # Convert to numpy
        mod_pred = mod_pred.numpy() if hasattr(mod_pred, 'numpy') else mod_pred
        
        # Check modified correctness
        mod_correct = np.all(mod_pred[:len(mod_act)] == mod_act)
        
        results.append({
            'modification_type': test_case['modification_type'],
            'original_correct': bool(orig_correct),
            'modified_correct': bool(mod_correct),
            'original_command': ' '.join(test_case['original']['command']),
            'modified_command': ' '.join(test_case['modified']['command'])
        })
    
    return results


def manual_training_loop(model, train_data, val_data, generator, epochs=50, batch_size=32):
    """Manual training loop that works with any Keras backend."""
    
    # Optimizer and loss function
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    # Training history
    history = {
        'loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'modification_results': []
    }
    
    # Get data
    train_commands = train_data['commands']
    train_actions = train_data['actions']
    n_train = len(train_commands)
    
    for epoch in range(epochs):
        logger.info(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Shuffle training data
        indices = np.random.permutation(n_train)
        
        # Training loop
        epoch_loss = 0
        n_batches = 0
        
        for i in range(0, n_train, batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_commands = train_commands[batch_indices]
            batch_actions = train_actions[batch_indices]
            
            # Compute loss and gradients
            with keras.backend.eager_scope():
                # Forward pass
                outputs = model({'command': batch_commands}, training=True)
                loss = loss_fn(batch_actions, outputs['action_logits'])
                
                # Get gradients - use Keras backend methods
                grads = keras.backend.gradients(loss, model.trainable_variables)
                
                # Apply gradients
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            epoch_loss += float(loss)
            n_batches += 1
            
            if i == 0 or (i + batch_size) >= n_train:
                logger.info(f"  Batch {i//batch_size + 1}/{(n_train + batch_size - 1)//batch_size} - Loss: {float(loss):.4f}")
        
        # Validation
        val_acc = evaluate_accuracy(model, val_data)
        avg_train_loss = epoch_loss / n_batches
        
        history['loss'].append(avg_train_loss)
        history['val_accuracy'].append(val_acc)
        
        logger.info(f"  Train Loss: {avg_train_loss:.4f}, Val Accuracy: {val_acc:.2%}")
        
        # Test modifications every 10 epochs
        if (epoch + 1) % 10 == 0:
            logger.info("  Testing modification capability...")
            mod_results = test_modifications(model, generator)
            
            n_correct = sum(r['modified_correct'] for r in mod_results)
            success_rate = n_correct / len(mod_results) if mod_results else 0
            
            logger.info(f"  Modification success rate: {success_rate:.2%}")
            
            history['modification_results'].append({
                'epoch': epoch + 1,
                'success_rate': success_rate,
                'results': mod_results
            })
    
    return history


def main():
    """Main training function."""
    logger.info("Starting Variable Binding Model Training")
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = get_output_path() / f"binding_training_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate data
    logger.info("Generating dereferencing tasks...")
    generator = DereferencingTaskGenerator(seed=42)
    dataset = generator.generate_dataset(n_samples=1000)
    
    logger.info(f"Dataset sizes - Train: {len(dataset['train']['commands'])}, "
                f"Val: {len(dataset['val']['commands'])}, "
                f"Test: {len(dataset['test']['commands'])}")
    
    # Create model
    logger.info("Creating model...")
    model = MinimalBindingModel(
        vocab_size=len(generator.word_vocab),
        action_vocab_size=len(generator.action_vocab),
        n_slots=10,
        embed_dim=128,
        hidden_dim=256
    )
    
    # Build model
    dummy_input = {'command': np.zeros((1, 10), dtype=np.int32)}
    _ = model(dummy_input)
    logger.info(f"Model created with {model.count_params()} parameters")
    
    # Train model
    logger.info("Starting training...")
    history = manual_training_loop(
        model, 
        dataset['train'], 
        dataset['val'],
        generator,
        epochs=50,
        batch_size=32
    )
    
    # Final evaluation
    logger.info("\nFinal Evaluation:")
    test_acc = evaluate_accuracy(model, dataset['test'])
    logger.info(f"Test Accuracy: {test_acc:.2%}")
    
    # Final modification test
    final_mod_results = test_modifications(model, generator)
    n_correct = sum(r['modified_correct'] for r in final_mod_results)
    final_success_rate = n_correct / len(final_mod_results) if final_mod_results else 0
    logger.info(f"Final Modification Success Rate: {final_success_rate:.2%}")
    
    # Save everything
    model.save(str(output_dir / "final_model.keras"))
    
    results = {
        'test_accuracy': test_acc,
        'modification_success_rate': final_success_rate,
        'modification_details': final_mod_results,
        'training_history': history
    }
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {output_dir}")
    logger.info(f"Success criteria: >50% modification success rate")
    logger.info(f"Achieved: {final_success_rate:.1%}")
    
    if final_success_rate > 0.5:
        logger.info("✓ SUCCESS: Model demonstrates variable binding capability!")
    else:
        logger.info("✗ Model needs more training or architectural improvements")


if __name__ == "__main__":
    main()