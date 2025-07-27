#!/usr/bin/env python3
"""
Train the Distribution Modification Component using Keras 3 training API.
"""

import os
os.environ['KERAS_BACKEND'] = 'jax'

import keras
import numpy as np
import pickle
from pathlib import Path
import wandb
from datetime import datetime
import json

from distribution_modifier import (
    DistributionModifier, 
    ModificationDataProcessor
)


def load_modification_data(data_path: Path):
    """Load modification pairs from pickle file."""
    with open(data_path, 'rb') as f:
        return pickle.load(f)


def create_train_val_split(data, val_ratio=0.1):
    """Split data into training and validation sets."""
    n_samples = len(data[0])
    n_val = int(n_samples * val_ratio)
    
    # Shuffle indices
    indices = np.random.permutation(n_samples)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    # Split data
    train_data = tuple(d[train_indices] for d in data)
    val_data = tuple(d[val_indices] for d in data)
    
    return train_data, val_data


class ModificationTrainer(keras.Model):
    """Trainer wrapper for the distribution modifier."""
    
    def __init__(self, modifier_model):
        super().__init__()
        self.modifier = modifier_model
        
        # Metrics
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.param_loss_tracker = keras.metrics.Mean(name="param_loss")
        self.change_loss_tracker = keras.metrics.Mean(name="change_loss")
        self.val_error_tracker = keras.metrics.Mean(name="val_error")
        
    def call(self, inputs, training=None):
        return self.modifier(inputs, training=training)
    
    def train_step(self, data):
        """Custom training step."""
        params, descriptions, target_params = data
        
        # Forward pass and loss computation
        with keras.tf.GradientTape() as tape:
            # Compute predictions
            pred_params, _, _ = self.modifier([params, descriptions], training=True)
            
            # Compute losses
            losses = self.modifier.compute_loss(
                params, descriptions, target_params, None
            )
            total_loss = losses['total_loss']
        
        # Compute gradients
        gradients = tape.gradient(total_loss, self.trainable_variables)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # Update metrics
        self.loss_tracker.update_state(total_loss)
        self.param_loss_tracker.update_state(losses['param_loss'])
        self.change_loss_tracker.update_state(losses['change_loss'])
        
        return {
            'loss': self.loss_tracker.result(),
            'param_loss': self.param_loss_tracker.result(),
            'change_loss': self.change_loss_tracker.result()
        }
    
    def test_step(self, data):
        """Custom validation step."""
        params, descriptions, target_params = data
        
        # Forward pass
        pred_params, _, _ = self.modifier([params, descriptions], training=False)
        
        # Compute error
        error = keras.ops.mean(keras.ops.abs(pred_params - target_params))
        self.val_error_tracker.update_state(error)
        
        return {'val_error': self.val_error_tracker.result()}
    
    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.param_loss_tracker,
            self.change_loss_tracker,
            self.val_error_tracker
        ]


def evaluate_modification_types(model, processor, test_data):
    """Evaluate performance on different modification types."""
    mod_types = {}
    
    # Group by modification type
    for i, pair in enumerate(test_data):
        mod_type = pair['modification_type']
        if mod_type not in mod_types:
            mod_types[mod_type] = []
        mod_types[mod_type].append(i)
    
    print("\nEvaluation by Modification Type:")
    print("-" * 60)
    
    results = {}
    
    for mod_type, indices in mod_types.items():
        if len(indices) == 0:
            continue
            
        # Process test samples
        test_pairs = [test_data[i] for i in indices]
        params, descs, targets, _ = processor.process_modification_pairs(test_pairs)
        
        # Predict
        pred_params, _, _ = model([params, descs], training=False)
        
        # Compute metrics
        param_error = np.mean(np.abs(pred_params - targets))
        relative_error = np.mean(np.abs(pred_params - targets) / (np.abs(targets) + 1e-6))
        
        # Direction accuracy
        target_dirs = np.sign(targets - params)
        pred_dirs = np.sign(pred_params - params)
        direction_acc = np.mean(target_dirs == pred_dirs)
        
        results[mod_type] = {
            'count': len(indices),
            'param_error': float(param_error),
            'relative_error': float(relative_error),
            'direction_accuracy': float(direction_acc)
        }
        
        print(f"{mod_type:25s} | n={len(indices):4d} | "
              f"Error: {param_error:.4f} | "
              f"Relative: {relative_error:.4f} | "
              f"Direction: {direction_acc:.4f}")
    
    return results


def main():
    """Main training function."""
    # Set random seed
    np.random.seed(42)
    keras.utils.set_random_seed(42)
    
    # Initialize wandb
    use_wandb = False  # Set to True to enable wandb logging
    wandb_run = None
    if use_wandb:
        wandb_run = wandb.init(
            project="distribution-invention",
            name=f"physics_modifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                'vocab_size': 500,
                'n_params': 4,
                'learning_rate': 1e-3,
                'batch_size': 32,
                'epochs': 75
            }
        )
    
    # Load data
    print("Loading modification pairs data...")
    data_path = Path("data/processed/physics_worlds/modification_pairs.pkl")
    mod_pairs = load_modification_data(data_path)
    print(f"Loaded {len(mod_pairs)} modification pairs")
    
    # Initialize processor
    processor = ModificationDataProcessor(vocab_size=500, max_length=15)
    
    # Process all data
    print("\nProcessing modification pairs...")
    all_data = processor.process_modification_pairs(mod_pairs)
    
    # Save vocabulary
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/checkpoints").mkdir(exist_ok=True)
    processor.save_vocabulary(Path("outputs/modifier_vocabulary.json"))
    
    # Create train/val split
    train_data, val_data = create_train_val_split(all_data, val_ratio=0.15)
    
    # Initialize model
    print("\nInitializing model...")
    base_model = DistributionModifier(vocab_size=processor.vocab_size, n_params=4)
    
    # Build model with dummy input
    dummy_params = np.ones((1, 4), dtype=np.float32)
    dummy_tokens = np.ones((1, processor.max_length), dtype=np.int32)
    _ = base_model([dummy_params, dummy_tokens], training=False)
    
    print(f"Model parameters: {base_model.count_params():,}")
    
    # Create trainer model
    model = ModificationTrainer(base_model)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        run_eagerly=False  # Use JIT compilation
    )
    
    # Create datasets
    batch_size = 32
    
    # Training dataset
    train_dataset = keras.utils.tf_dataset.from_tensor_slices(
        (train_data[0], train_data[1], train_data[2])
    ).batch(batch_size).prefetch(8)
    
    # Validation dataset
    val_dataset = keras.utils.tf_dataset.from_tensor_slices(
        (val_data[0], val_data[1], val_data[2])
    ).batch(batch_size).prefetch(8)
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            'outputs/checkpoints/distribution_modifier_best.keras',
            monitor='val_val_error',
            save_best_only=True,
            save_weights_only=False,
            mode='min'
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_val_error',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_val_error',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    if use_wandb:
        callbacks.append(wandb.keras.WandbCallback())
    
    # Train
    print("\nTraining model...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=75,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    base_model.save('outputs/checkpoints/distribution_modifier_final.keras')
    
    # Plot training history
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_val_error'], label='Val Error')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss/Error')
    ax1.legend()
    ax1.set_title('Training Progress')
    
    ax2.plot(history.history['param_loss'], label='Param Loss')
    ax2.plot(history.history['change_loss'], label='Change Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.set_title('Loss Components')
    
    plt.tight_layout()
    plt.savefig('outputs/modifier_training_history.png')
    plt.close()
    
    # Evaluate on different modification types
    print("\nEvaluating on different modification types...")
    eval_results = evaluate_modification_types(base_model, processor, mod_pairs[:1000])
    
    # Save evaluation results
    with open('outputs/modifier_evaluation_results.json', 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print("\nTraining complete!")
    print(f"Best validation error: {min(history.history['val_val_error']):.4f}")
    
    if wandb_run:
        wandb.finish()


if __name__ == "__main__":
    main()