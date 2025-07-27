#!/usr/bin/env python3
"""
Progressive Training Curriculum for Paperspace GPU Training

This is the full-scale version for running on Paperspace with proper parameters.
Key differences from test version:
- Full dataset (no subsampling)
- 50 epochs per stage (200 total)
- Wandb logging enabled
- GPU optimizations
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# Set backend before importing Keras
os.environ['KERAS_BACKEND'] = 'tensorflow'  # TensorFlow for GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging

import tensorflow as tf
from tensorflow import keras
import wandb
from tqdm import tqdm
import pickle

# Add project root to path
if os.path.exists('/notebooks/neural_networks_research'):
    sys.path.append('/notebooks/neural_networks_research')
elif os.path.exists('/workspace/neural_networks_research'):
    sys.path.append('/workspace/neural_networks_research')
else:
    sys.path.append(os.path.abspath('../..'))

# For now, we'll use a simple transformer instead of the physics-informed one
# due to the 2-ball vs 3-ball mismatch issue


class SimpleTrajectoryTransformer(keras.Model):
    """Simplified transformer for trajectory prediction"""
    
    def __init__(self, 
                 sequence_length=10,
                 feature_dim=25,
                 num_heads=8,
                 num_layers=4,
                 d_model=256,
                 dropout_rate=0.1):
        super().__init__()
        
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        
        # Input projection
        self.input_projection = keras.layers.Dense(d_model)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(sequence_length, d_model)
        
        # Transformer blocks
        self.transformer_blocks = [
            TransformerBlock(d_model, num_heads, dropout_rate) 
            for _ in range(num_layers)
        ]
        
        # Output projection - predict full trajectory features
        self.output_projection = keras.layers.Dense(feature_dim)
        
    def _create_positional_encoding(self, seq_len, d_model):
        """Create sinusoidal positional encoding"""
        pos = np.arange(seq_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angle_rads = pos * angle_rates
        
        # Apply sin to even indices
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        # Apply cos to odd indices
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        return tf.constant(angle_rads[np.newaxis, ...], dtype=tf.float32)
    
    def call(self, inputs, training=None):
        # Project input
        x = self.input_projection(inputs)
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Apply transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x, training=training)
        
        # Project to output
        outputs = self.output_projection(x)
        
        return outputs


class TransformerBlock(keras.layers.Layer):
    """Single transformer block"""
    
    def __init__(self, d_model, num_heads, dropout_rate=0.1):
        super().__init__()
        self.attention = keras.layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=d_model // num_heads
        )
        self.dropout1 = keras.layers.Dropout(dropout_rate)
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.ffn = keras.Sequential([
            keras.layers.Dense(d_model * 4, activation='relu'),
            keras.layers.Dense(d_model),
        ])
        self.dropout2 = keras.layers.Dropout(dropout_rate)
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, inputs, training=None):
        # Self-attention
        attn_output = self.attention(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(inputs + attn_output)
        
        # Feed-forward
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.norm2(out1 + ffn_output)
        
        return out2


def load_full_data():
    """Load the complete physics worlds dataset"""
    print("Loading full dataset...")
    
    # Load from pickle files
    # Auto-detect the correct base path
    if Path("/notebooks/neural_networks_research").exists():
        base_path = "/notebooks/neural_networks_research"
    elif Path("/workspace/neural_networks_research").exists():
        base_path = "/workspace/neural_networks_research"
    else:
        base_path = "../.."  # Fallback to relative path
    
    data_dir = Path(base_path) / "experiments/01_physics_worlds/data/processed/physics_worlds_v2_quick"
    
    # Load all data
    with open(data_dir / 'train_data.pkl', 'rb') as f:
        train_pkl = pickle.load(f)
    
    with open(data_dir / 'val_in_dist_data.pkl', 'rb') as f:
        val_pkl = pickle.load(f)
    
    with open(data_dir / 'test_interpolation_data.pkl', 'rb') as f:
        test_interp_pkl = pickle.load(f)
    with open(data_dir / 'test_extrapolation_data.pkl', 'rb') as f:
        test_extrap_pkl = pickle.load(f)
    
    # Convert to arrays (filter for 3 balls)
    def extract_data(data_list):
        trajectories = []
        future_trajectories = []
        gravity_values = []
        friction_values = []
        
        for sample in data_list:
            if sample['num_balls'] != 3:
                continue
            
            traj = np.array(sample['trajectory'])
            trajectories.append(traj[:10])
            future_trajectories.append(traj[10:20])
            gravity_values.append(sample['physics_config']['gravity'])
            friction_values.append(sample['physics_config']['friction'])
        
        return {
            'trajectories': np.array(trajectories, dtype=np.float32),
            'future_trajectories': np.array(future_trajectories, dtype=np.float32),
            'gravity': np.array(gravity_values, dtype=np.float32),
            'friction': np.array(friction_values, dtype=np.float32)
        }
    
    train_data = extract_data(train_pkl)
    val_data = extract_data(val_pkl)
    test_interp_data = extract_data(test_interp_pkl)
    test_extrap_data = extract_data(test_extrap_pkl)
    
    # Combine test data
    test_data = {
        'trajectories': np.concatenate([test_interp_data['trajectories'], test_extrap_data['trajectories']]),
        'future_trajectories': np.concatenate([test_interp_data['future_trajectories'], test_extrap_data['future_trajectories']]),
        'gravity': np.concatenate([test_interp_data['gravity'], test_extrap_data['gravity']]),
        'friction': np.concatenate([test_interp_data['friction'], test_extrap_data['friction']])
    }
    
    print(f"  Train: {len(train_data['trajectories'])} samples")
    print(f"  Val: {len(val_data['trajectories'])} samples")
    print(f"  Test: {len(test_data['trajectories'])} samples")
    
    return train_data, val_data, test_data, test_interp_data, test_extrap_data


def create_physics_loss(predictions, targets, gravity, friction):
    """Simple physics-based loss terms"""
    # Energy conservation approximation
    # Kinetic energy should be conserved (minus friction losses)
    velocities_pred = predictions[:, :, 7:13]  # Extract velocity components
    velocities_true = targets[:, :, 7:13]
    
    # Compute kinetic energies
    ke_pred = tf.reduce_sum(tf.square(velocities_pred), axis=-1)
    ke_true = tf.reduce_sum(tf.square(velocities_true), axis=-1)
    
    # Energy should decrease due to friction
    energy_loss = tf.reduce_mean(tf.square(ke_pred - ke_true))
    
    # Momentum conservation (no external forces except gravity)
    momentum_pred = tf.reduce_sum(velocities_pred, axis=-1)
    momentum_true = tf.reduce_sum(velocities_true, axis=-1)
    momentum_loss = tf.reduce_mean(tf.square(momentum_pred - momentum_true))
    
    return energy_loss + 0.5 * momentum_loss


def train_progressive_curriculum():
    """Main training function with progressive curriculum"""
    
    # Configuration
    config = {
        "batch_size": 32,
        "stage1_epochs": 50,
        "stage2_epochs": 50,
        "stage3_epochs": 50,
        "stage4_epochs": 50,
        "learning_rates": [1e-3, 5e-4, 2e-4, 1e-4],
        "physics_weight_schedule": np.linspace(0.1, 1.0, 50),
        "wandb_project": "physics-worlds-extrapolation",
        "wandb_enabled": True,
        "checkpoint_dir": "outputs/checkpoints"  # Relative path
    }
    
    # Initialize wandb
    if config['wandb_enabled']:
        wandb.init(
            project=config['wandb_project'],
            config=config,
            name=f"progressive_curriculum_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    # Load data
    train_data, val_data, test_data, test_interp_data, test_extrap_data = load_full_data()
    
    # Initialize model
    print("\nInitializing model...")
    model = SimpleTrajectoryTransformer(
        sequence_length=10,
        feature_dim=25,
        num_heads=8,
        num_layers=6,
        d_model=256,
        dropout_rate=0.1
    )
    
    # Build model
    model(train_data['trajectories'][:1])
    
    # Create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((
        train_data['trajectories'],
        train_data['future_trajectories'],
        train_data['gravity'],
        train_data['friction']
    )).batch(config['batch_size']).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((
        val_data['trajectories'],
        val_data['future_trajectories']
    )).batch(config['batch_size'])
    
    # Stage configurations
    stages = [
        {
            "name": "Stage 1: In-Distribution Learning",
            "epochs": config['stage1_epochs'],
            "lr": config['learning_rates'][0],
            "use_physics": False
        },
        {
            "name": "Stage 2: Physics Integration",
            "epochs": config['stage2_epochs'],
            "lr": config['learning_rates'][1],
            "use_physics": True,
            "physics_schedule": config['physics_weight_schedule']
        },
        {
            "name": "Stage 3: Domain Randomization",
            "epochs": config['stage3_epochs'],
            "lr": config['learning_rates'][2],
            "use_physics": True,
            "physics_weight": 1.0,
            "randomize": True
        },
        {
            "name": "Stage 4: Extrapolation Fine-tuning",
            "epochs": config['stage4_epochs'],
            "lr": config['learning_rates'][3],
            "use_physics": True,
            "physics_weight": 1.0,
            "focus_hard": True
        }
    ]
    
    # Training loop
    optimizer = keras.optimizers.Adam(config['learning_rates'][0])
    
    for stage_idx, stage_config in enumerate(stages):
        print(f"\n{'='*60}")
        print(f"{stage_config['name']}")
        print(f"{'='*60}")
        
        # Update learning rate
        optimizer.learning_rate.assign(stage_config['lr'])
        
        for epoch in range(stage_config['epochs']):
            print(f"\nEpoch {epoch + 1}/{stage_config['epochs']}")
            
            # Training step
            train_loss = 0
            physics_loss = 0
            num_batches = 0
            
            for batch in tqdm(train_dataset, desc="Training"):
                trajectories, targets, gravity, friction = batch
                
                # Apply domain randomization in stage 3
                if stage_config.get('randomize', False):
                    noise_g = tf.random.uniform(shape=gravity.shape, minval=-0.3, maxval=0.3)
                    noise_f = tf.random.uniform(shape=friction.shape, minval=-0.3, maxval=0.3)
                    gravity = gravity * (1 + noise_g)
                    friction = friction * (1 + noise_f)
                
                with tf.GradientTape() as tape:
                    predictions = model(trajectories, training=True)
                    
                    # MSE loss
                    mse_loss = tf.reduce_mean(tf.square(predictions - targets))
                    
                    # Physics loss
                    if stage_config.get('use_physics', False):
                        if stage_idx == 1:  # Progressive schedule
                            weight = stage_config['physics_schedule'][epoch]
                        else:
                            weight = stage_config.get('physics_weight', 1.0)
                        
                        phys_loss = create_physics_loss(predictions, targets, gravity, friction)
                        total_loss = mse_loss + weight * phys_loss
                        physics_loss += phys_loss.numpy()
                    else:
                        total_loss = mse_loss
                    
                    train_loss += total_loss.numpy()
                
                gradients = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                num_batches += 1
            
            # Validation
            val_loss = 0
            val_batches = 0
            for val_batch in val_dataset:
                val_traj, val_targets = val_batch
                val_pred = model(val_traj, training=False)
                val_loss += tf.reduce_mean(tf.square(val_pred - val_targets)).numpy()
                val_batches += 1
            
            # Compute extrapolation accuracy
            extrap_pred = model(test_extrap_data['trajectories'], training=False)
            extrap_error = np.mean(np.abs(extrap_pred - test_extrap_data['future_trajectories']))
            extrap_acc = max(0, 1 - extrap_error / 10.0)
            
            # Logging
            metrics = {
                f"stage_{stage_idx+1}/train_loss": train_loss / num_batches,
                f"stage_{stage_idx+1}/val_loss": val_loss / val_batches,
                f"stage_{stage_idx+1}/physics_loss": physics_loss / num_batches if stage_config.get('use_physics') else 0,
                f"stage_{stage_idx+1}/extrap_accuracy": extrap_acc,
                "epoch": sum([s['epochs'] for s in stages[:stage_idx]]) + epoch
            }
            
            print(f"  Train Loss: {metrics[f'stage_{stage_idx+1}/train_loss']:.4f}")
            print(f"  Val Loss: {metrics[f'stage_{stage_idx+1}/val_loss']:.4f}")
            print(f"  Extrap Accuracy: {extrap_acc:.2%}")
            
            if config['wandb_enabled']:
                wandb.log(metrics)
        
        # Save checkpoint
        checkpoint_path = Path(config['checkpoint_dir']) / f"stage_{stage_idx+1}_final.h5"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_weights(str(checkpoint_path))
        print(f"Saved checkpoint: {checkpoint_path}")
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    # Test on interpolation
    interp_pred = model(test_interp_data['trajectories'], training=False)
    interp_error = np.mean(np.abs(interp_pred - test_interp_data['future_trajectories']))
    interp_acc = max(0, 1 - interp_error / 10.0)
    
    # Test on extrapolation
    extrap_pred = model(test_extrap_data['trajectories'], training=False)
    extrap_error = np.mean(np.abs(extrap_pred - test_extrap_data['future_trajectories']))
    extrap_acc = max(0, 1 - extrap_error / 10.0)
    
    print(f"Interpolation Accuracy: {interp_acc:.2%}")
    print(f"Extrapolation Accuracy: {extrap_acc:.2%}")
    
    if config['wandb_enabled']:
        wandb.log({
            "final/interpolation_accuracy": interp_acc,
            "final/extrapolation_accuracy": extrap_acc
        })
        wandb.finish()
    
    # Save final model
    final_path = Path(config['checkpoint_dir']) / "final_model.h5"
    model.save_weights(str(final_path))
    print(f"\nFinal model saved to: {final_path}")


if __name__ == "__main__":
    # Check for GPU
    print("GPU Available:", tf.config.list_physical_devices('GPU'))
    
    # Run training
    train_progressive_curriculum()