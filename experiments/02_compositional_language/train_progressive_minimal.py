#!/usr/bin/env python3
"""
Minimal progressive training - simplest possible version that works
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

# Import from current directory
from scan_data_loader import SCANDataLoader
from train_progressive_curriculum import (
    SCANTokenizer, ProgressiveCurriculum, create_dataset, compute_accuracy
)


class SimpleTransformerModel(keras.Model):
    """Simplified transformer model without complex nested structures"""
    
    def __init__(self, command_vocab_size, action_vocab_size, d_model=128):
        super().__init__()
        
        # Encoder
        self.command_embedding = keras.layers.Embedding(command_vocab_size, d_model)
        self.encoder = keras.layers.LSTM(d_model, return_sequences=True)
        
        # Modification embedding (if provided)
        self.mod_embedding = keras.layers.Embedding(command_vocab_size, d_model)
        self.mod_encoder = keras.layers.LSTM(d_model // 2)
        
        # Decoder
        self.decoder_embedding = keras.layers.Embedding(action_vocab_size, d_model)
        self.decoder = keras.layers.LSTM(d_model, return_sequences=True)
        
        # Combine modification with encoding
        self.combine_layer = keras.layers.Dense(d_model)
        
        # Output projection
        self.output_layer = keras.layers.Dense(action_vocab_size)
        
        # Store vocab sizes
        self.command_vocab_size = command_vocab_size
        self.action_vocab_size = action_vocab_size
        
    def call(self, inputs, training=None):
        command = inputs['command']
        target = inputs.get('target', None)
        modification = inputs.get('modification', None)
        
        # Encode command
        command_embed = self.command_embedding(command)
        encoded = self.encoder(command_embed, training=training)
        
        # Handle modification if present
        if modification is not None:
            # Check if modification is meaningful (not all zeros)
            mod_sum = tf.reduce_sum(tf.abs(modification))
            if mod_sum > 0:
                mod_embed = self.mod_embedding(modification)
                mod_encoded = self.mod_encoder(mod_embed, training=training)
                # Add modification signal to encoding
                mod_signal = tf.expand_dims(mod_encoded, 1)
                encoded = encoded + self.combine_layer(mod_signal)
        
        # If target provided (training mode)
        if target is not None:
            # Decode
            target_embed = self.decoder_embedding(target)
            
            # Create decoder initial state from encoder final state
            decoder_output = self.decoder(target_embed, training=training)
            
            # Add encoder context
            decoder_output = decoder_output + tf.reduce_mean(encoded, axis=1, keepdims=True)
            
            # Project to vocabulary
            logits = self.output_layer(decoder_output)
            
            return {
                'logits': logits,
                'rule_outputs': {}  # Compatibility
            }
        
        return {
            'rule_embeddings': encoded,
            'rule_outputs': {}
        }


def train_progressive_curriculum_minimal(config: Dict):
    """Minimal training loop that should work"""
    
    # Setup
    print("Setting up minimal progressive curriculum training...")
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb if enabled
    if config.get('use_wandb', False):
        wandb.init(
            project=config.get('wandb_project', 'compositional-language'),
            name=f"minimal_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
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
    
    # Create simple model
    print("\nCreating simple model...")
    model = SimpleTransformerModel(
        command_vocab_size=len(tokenizer.command_to_id),
        action_vocab_size=len(tokenizer.action_to_id),
        d_model=config.get('d_model', 128)
    )
    
    # Build model with explicit forward pass
    print("Building model...")
    dummy_inputs = {
        'command': tf.zeros((1, 50), dtype=tf.int32),
        'target': tf.zeros((1, 99), dtype=tf.int32),
        'modification': tf.zeros((1, 20), dtype=tf.int32)
    }
    _ = model(dummy_inputs, training=False)
    print("Model built successfully")
    
    # Count parameters
    total_params = sum([tf.size(v).numpy() for v in model.trainable_variables])
    print(f"Total trainable parameters: {total_params:,}")
    
    # Initialize curriculum
    curriculum = ProgressiveCurriculum(config)
    
    # Loss function
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
        
        # Garbage collect
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
        
        # Setup optimizer
        optimizer = keras.optimizers.Adam(learning_rate=stage_config['lr'])
        
        # Training epochs
        best_val_acc = 0
        for epoch in range(stage_config['epochs']):
            print(f"\nEpoch {epoch + 1}/{stage_config['epochs']}")
            
            # Training metrics
            train_loss = 0
            num_batches = 0
            
            # Progress bar
            pbar = tqdm(dataset, desc="Training")
            
            for i, batch in enumerate(pbar):
                try:
                    with tf.GradientTape() as tape:
                        # Prepare inputs
                        inputs = {
                            'command': batch['command'],
                            'target': batch['action'][:, :-1],
                            'modification': batch.get('modification', tf.zeros_like(batch['command'][:, :20]))
                        }
                        
                        # Forward pass
                        outputs = model(inputs, training=True)
                        
                        # Compute loss
                        logits = outputs['logits']
                        targets = batch['action'][:, 1:]
                        
                        # Mask padding
                        mask = tf.cast(targets != tokenizer.action_to_id['<PAD>'], tf.float32)
                        loss_per_token = loss_fn(targets, logits)
                        masked_loss = loss_per_token * mask
                        loss = tf.reduce_sum(masked_loss) / (tf.reduce_sum(mask) + 1e-7)
                    
                    # Compute and apply gradients
                    gradients = tape.gradient(loss, model.trainable_variables)
                    
                    # Clip gradients
                    gradients = [tf.clip_by_value(g, -1.0, 1.0) if g is not None else None 
                                for g in gradients]
                    
                    # Apply gradients
                    optimizer.apply_gradients(
                        [(g, v) for g, v in zip(gradients, model.trainable_variables) if g is not None]
                    )
                    
                    # Update metrics
                    train_loss += loss.numpy()
                    num_batches += 1
                    
                    # Update progress bar
                    pbar.set_postfix({'loss': f'{train_loss/num_batches:.4f}'})
                    
                    # Periodic garbage collection
                    if i > 0 and i % 500 == 0:
                        gc.collect()
                        
                except Exception as e:
                    print(f"\nError in batch {i}: {e}")
                    continue
            
            # Validation
            if num_batches > 0:
                avg_train_loss = train_loss / num_batches
                print(f"  Train Loss: {avg_train_loss:.4f}")
                
                # Quick validation on subset
                val_acc = compute_accuracy(model, val_dataset, tokenizer, max_samples=50)
                print(f"  Val Accuracy: {val_acc:.2%}")
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                
                if config.get('use_wandb', False):
                    wandb.log({
                        f'stage_{stage}/train_loss': avg_train_loss,
                        f'stage_{stage}/val_accuracy': val_acc,
                        'epoch': sum([curriculum.get_stage_config(s)['epochs'] 
                                     for s in range(1, stage)]) + epoch
                    })
        
        # Save checkpoint
        try:
            checkpoint_path = output_dir / f"stage_{stage}_model.h5"
            model.save_weights(str(checkpoint_path))
            print(f"Saved checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")
    
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
            accuracy = compute_accuracy(model, test_dataset, tokenizer, max_samples=100)
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
            'total_parameters': int(total_params)
        }, f, indent=2)
    
    if config.get('use_wandb', False):
        wandb.finish()
    
    print(f"\nTraining complete! Results saved to {output_dir}")


def main():
    """Run minimal training"""
    
    config = {
        # Model parameters
        'd_model': 128,
        'batch_size': 8,
        
        # Training epochs (very short for testing)
        'stage1_epochs': 5,
        'stage2_epochs': 5,
        'stage3_epochs': 5,
        'stage4_epochs': 5,
        
        # Learning rates
        'stage1_lr': 1e-3,
        'stage2_lr': 5e-4,
        'stage3_lr': 2e-4,
        'stage4_lr': 1e-4,
        
        # Output
        'output_dir': 'outputs/minimal_training',
        'use_wandb': True,
        'wandb_project': 'compositional-language-minimal'
    }
    
    train_progressive_curriculum_minimal(config)


if __name__ == "__main__":
    main()