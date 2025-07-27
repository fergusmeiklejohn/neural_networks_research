#!/usr/bin/env python3
"""
Memory-optimized Progressive Training Curriculum for Compositional Language

This version includes:
- GPU memory growth configuration
- Mixed precision training
- tf.function compilation
- Periodic memory clearing
- Gradient accumulation option
"""

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import gc
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import wandb
from tqdm import tqdm

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"GPU memory growth setting failed: {e}")

# Enable mixed precision for memory efficiency
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
print(f"Mixed precision policy: {policy.name}")

# Import from current directory
from models import create_model
from scan_data_loader import SCANDataLoader
from train_progressive_curriculum import (
    SCANTokenizer, ProgressiveCurriculum, create_dataset, compute_accuracy
)


class MemoryEfficientTrainer:
    """Trainer with memory optimization techniques"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
        
    @tf.function(reduce_retracing=True)
    def train_step(self, batch, tokenizer_pad_id, loss_fn, training=True):
        """Compiled training step for efficiency"""
        with tf.GradientTape() as tape:
            # Extract inputs
            command = batch['command']
            action = batch['action']
            has_modification = batch['has_modification']
            modification = batch['modification']
            
            # Prepare inputs - always include modification to avoid tf.function issues
            inputs = {
                'command': command,
                'target': action[:, :-1],  # Exclude last token
                'modification': modification  # Always include, model will ignore if not needed
            }
            
            # Forward pass
            outputs = self.model(inputs, training=training)
            
            # Compute loss
            logits = outputs['logits']
            targets = action[:, 1:]  # Exclude first token
            
            # Create padding mask
            mask = tf.cast(targets != tokenizer_pad_id, tf.float32)
            
            # Compute masked loss
            loss_per_token = loss_fn(targets, logits)
            masked_loss = loss_per_token * mask
            loss = tf.reduce_sum(masked_loss) / tf.reduce_sum(mask)
            
            # Scale loss for mixed precision
            scaled_loss = self.model.optimizer.get_scaled_loss(loss)
        
        # Compute gradients
        scaled_gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
        gradients = self.model.optimizer.get_unscaled_gradients(scaled_gradients)
        
        # Filter None gradients
        valid_grads_and_vars = [
            (g, v) for g, v in zip(gradients, self.model.trainable_variables)
            if g is not None
        ]
        
        return loss, valid_grads_and_vars
    
    def train_epoch(self, dataset, optimizer, loss_fn, tokenizer, desc="Training"):
        """Train for one epoch with memory management"""
        total_loss = 0
        num_batches = 0
        accumulated_grads = []
        
        # Progress bar
        pbar = tqdm(dataset, desc=desc)
        
        for i, batch in enumerate(pbar):
            # Compute loss and gradients
            loss, grads_and_vars = self.train_step(
                batch, 
                tokenizer.action_to_id['<PAD>'],
                loss_fn,
                training=True
            )
            
            # Accumulate gradients
            if not accumulated_grads:
                accumulated_grads = [(g, v) for g, v in grads_and_vars]
            else:
                accumulated_grads = [
                    (acc_g + g, v) 
                    for (acc_g, _), (g, v) in zip(accumulated_grads, grads_and_vars)
                ]
            
            # Apply gradients every N steps
            if (i + 1) % self.gradient_accumulation_steps == 0:
                # Average accumulated gradients
                averaged_grads_and_vars = [
                    (g / self.gradient_accumulation_steps, v)
                    for g, v in accumulated_grads
                ]
                
                # Apply gradients
                optimizer.apply_gradients(averaged_grads_and_vars)
                
                # Clear accumulation
                accumulated_grads = []
            
            # Update metrics
            total_loss += loss.numpy()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{total_loss/num_batches:.4f}'})
            
            # Periodic memory clearing
            if i % 100 == 0:
                tf.keras.backend.clear_session()
                gc.collect()
        
        # Apply any remaining gradients
        if accumulated_grads:
            averaged_grads_and_vars = [
                (g / len(accumulated_grads), v)
                for g, v in accumulated_grads
            ]
            optimizer.apply_gradients(averaged_grads_and_vars)
        
        return total_loss / num_batches


def train_progressive_curriculum_optimized(config: Dict):
    """Main training function with memory optimizations"""
    
    # Setup
    print("Setting up optimized progressive curriculum training...")
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb if enabled
    if config.get('use_wandb', False):
        wandb.init(
            project=config.get('wandb_project', 'compositional-language'),
            name=f"optimized_curriculum_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=config
        )
    
    # Load data
    print("\nLoading data...")
    loader = SCANDataLoader(data_dir='data')
    splits = loader.load_processed_splits()
    
    # Load modifications
    mod_path = Path('data/processed/modification_pairs.pkl')
    if mod_path.exists():
        with open(mod_path, 'rb') as f:
            modifications = pickle.load(f)
    else:
        modifications = []
    
    # Create tokenizer
    print("Building vocabulary...")
    tokenizer = SCANTokenizer()
    tokenizer.build_vocabulary(splits['train'])
    tokenizer.save_vocabulary(output_dir / 'vocabulary.json')
    
    print(f"Command vocabulary size: {len(tokenizer.command_to_id)}")
    print(f"Action vocabulary size: {len(tokenizer.action_to_id)}")
    
    # Create model with mixed precision
    print("\nCreating model...")
    model = create_model(
        command_vocab_size=len(tokenizer.command_to_id),
        action_vocab_size=len(tokenizer.action_to_id),
        d_model=config.get('d_model', 128)
    )
    
    # Build model
    print("Building model...")
    dummy_command = tf.zeros((1, 50), dtype=tf.int32)
    dummy_target = tf.zeros((1, 99), dtype=tf.int32)
    _ = model({
        'command': dummy_command,
        'target': dummy_target
    }, training=False)
    print("Model built successfully")
    
    # Create trainer
    trainer = MemoryEfficientTrainer(model, config)
    
    # Initialize curriculum
    curriculum = ProgressiveCurriculum(config)
    
    # Loss function with mixed precision
    loss_fn = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction=tf.keras.losses.Reduction.NONE
    )
    
    # Training loop
    for stage in range(1, 5):
        stage_config = curriculum.get_stage_config(stage)
        print(f"\n{'='*60}")
        print(f"{stage_config['name']}")
        print(f"{stage_config['description']}")
        print(f"{'='*60}")
        
        # Clear memory before each stage
        tf.keras.backend.clear_session()
        gc.collect()
        
        # Create stage-specific dataset
        if stage_config.get('use_modifications', False):
            dataset = create_dataset(
                splits['train'],
                tokenizer,
                modifications=modifications,
                modification_ratio=stage_config.get('modification_ratio', 0.0),
                batch_size=config['batch_size']
            )
        else:
            dataset = create_dataset(
                splits['train'],
                tokenizer,
                batch_size=config['batch_size']
            )
        
        # Create validation dataset
        val_dataset = create_dataset(
            splits['val_interpolation'],
            tokenizer,
            batch_size=config['batch_size']
        )
        
        # Setup optimizer with mixed precision
        optimizer = keras.optimizers.Adam(learning_rate=stage_config['lr'])
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        model.optimizer = optimizer  # Attach for mixed precision
        
        # Training epochs
        for epoch in range(stage_config['epochs']):
            print(f"\nEpoch {epoch + 1}/{stage_config['epochs']}")
            
            # Train epoch
            train_loss = trainer.train_epoch(
                dataset, optimizer, loss_fn, tokenizer
            )
            
            # Validation every 5 epochs
            if epoch % 5 == 0:
                val_acc = compute_accuracy(model, val_dataset, tokenizer, max_samples=100)
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Accuracy: {val_acc:.2%}")
                
                if config.get('use_wandb', False):
                    wandb.log({
                        f'stage_{stage}/train_loss': train_loss,
                        f'stage_{stage}/val_accuracy': val_acc,
                        'epoch': sum([curriculum.get_stage_config(s)['epochs'] 
                                     for s in range(1, stage)]) + epoch
                    })
        
        # Save stage checkpoint
        checkpoint_path = output_dir / f"stage_{stage}_model.h5"
        model.save_weights(str(checkpoint_path))
        print(f"Saved checkpoint: {checkpoint_path}")
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    # Test on different splits
    test_results = {}
    
    for split_name in ['test_interpolation', 'test_primitive_extrap']:
        if split_name in splits:
            test_dataset = create_dataset(
                splits[split_name],
                tokenizer,
                batch_size=config['batch_size']
            )
            accuracy = compute_accuracy(model, test_dataset, tokenizer)
            test_results[split_name] = accuracy
            print(f"{split_name}: {accuracy:.2%}")
    
    # Save results
    results_path = output_dir / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'config': config,
            'test_results': test_results,
            'vocabulary_sizes': {
                'command': len(tokenizer.command_to_id),
                'action': len(tokenizer.action_to_id)
            },
            'optimizations': {
                'mixed_precision': True,
                'memory_growth': True,
                'gradient_accumulation': config.get('gradient_accumulation_steps', 1),
                'tf_function': True
            }
        }, f, indent=2)
    
    if config.get('use_wandb', False):
        wandb.finish()
    
    print(f"\nTraining complete! Results saved to {output_dir}")


def main():
    """Run optimized progressive curriculum training"""
    
    config = {
        # Model parameters
        'd_model': 128,
        'batch_size': 8,
        'gradient_accumulation_steps': 2,  # Effective batch size of 16
        
        # Training epochs
        'stage1_epochs': 20,
        'stage2_epochs': 20,
        'stage3_epochs': 20,
        'stage4_epochs': 20,
        
        # Learning rates
        'stage1_lr': 1e-3,
        'stage2_lr': 5e-4,
        'stage3_lr': 2e-4,
        'stage4_lr': 1e-4,
        
        # Output
        'output_dir': 'outputs/optimized_training',
        'use_wandb': True,
        'wandb_project': 'compositional-language-optimized'
    }
    
    train_progressive_curriculum_optimized(config)


if __name__ == "__main__":
    main()