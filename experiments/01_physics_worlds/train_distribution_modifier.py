"""
Distribution Modification Training - Learn to modify physics distributions
"""

import sys
import os
sys.path.append('../..')

import numpy as np
import pickle
import keras
from keras import layers, callbacks, ops
from tqdm import tqdm
from typing import Dict, List, Tuple

from models.core.distribution_modifier import ModifierConfig


class DistributionModificationTrainer(keras.Model):
    """Simplified distribution modifier for modification training"""
    
    def __init__(self, config: ModifierConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        
        # Input: original physics + modification request
        self.physics_projection = layers.Dense(64, activation='relu')
        self.modification_embedding = layers.Embedding(20, 32)  # 20 modification types
        
        # Fusion layer
        self.fusion = layers.Dense(128, activation='relu')
        self.dropout1 = layers.Dropout(0.2)
        
        # Modification network
        self.modifier_network = keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(32, activation='relu'),
            layers.Dense(4, name='modified_physics')  # Output modified physics
        ])
        
    def call(self, inputs, training=None):
        # inputs: [original_physics, modification_type]
        original_physics, modification_type = inputs
        
        # Project physics parameters
        physics_features = self.physics_projection(original_physics)
        
        # Embed modification type
        mod_features = self.modification_embedding(modification_type)
        # mod_features shape is already (batch, embedding_dim), no need to squeeze
        
        # Fuse features
        combined = ops.concatenate([physics_features, mod_features], axis=-1)
        fused = self.fusion(combined, training=training)
        fused = self.dropout1(fused, training=training)
        
        # Generate modified physics
        modified_physics = self.modifier_network(fused, training=training)
        
        return modified_physics


def load_and_prepare_modification_data(data_path: str, max_samples: int = None):
    """Load and prepare data for distribution modification training"""
    print(f"Loading modification data from {data_path}...")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    if max_samples:
        data = data[:max_samples]
    
    print(f"Using {len(data)} modification samples")
    
    # Prepare arrays
    original_physics = []
    modification_types = []
    target_physics = []
    
    # Define modification type mappings
    mod_type_map = {
        'gravity_increase': 0,
        'gravity_decrease': 1,
        'friction_increase': 2,
        'friction_decrease': 3,
        'elasticity_increase': 4,
        'elasticity_decrease': 5,
        'damping_increase': 6,
        'damping_decrease': 7,
        'underwater_physics': 8,
        'space_physics': 9,
        'bouncy_castle': 10
    }
    
    for sample in tqdm(data, desc="Processing modification pairs"):
        # Original physics (from base_config)
        orig_physics = sample['base_config']
        original_physics.append([
            orig_physics['gravity'],
            orig_physics['friction'], 
            orig_physics['elasticity'],
            orig_physics['damping']
        ])
        
        # Modification type
        mod_type = sample['modification_type']
        modification_types.append(mod_type_map.get(mod_type, 0))
        
        # Target modified physics (need to combine base_config with modification_params)
        base_physics = sample['base_config']
        mod_params = sample['modification_params']
        
        # Create target physics by updating base with modifications
        target_config = base_physics.copy()
        target_config.update(mod_params)
        
        target_physics.append([
            target_config['gravity'],
            target_config['friction'], 
            target_config['elasticity'],
            target_config['damping']
        ])
    
    X_original = np.array(original_physics)
    X_modifications = np.array(modification_types)
    y_target = np.array(target_physics)
    
    # Normalize physics parameters (same as other components)
    X_original_norm = normalize_physics_params(X_original)
    y_target_norm = normalize_physics_params(y_target)
    
    print(f"Data shapes: X_original={X_original_norm.shape}, X_modifications={X_modifications.shape}, y_target={y_target_norm.shape}")
    print(f"Original physics range: {X_original_norm.min():.3f} to {X_original_norm.max():.3f}")
    print(f"Target physics range: {y_target_norm.min():.3f} to {y_target_norm.max():.3f}")
    print(f"Modification types: {np.unique(X_modifications)}")
    
    return X_original_norm, X_modifications, y_target_norm, X_original, y_target


def normalize_physics_params(params):
    """Normalize physics parameters to similar scales"""
    normalized = params.copy()
    
    # Gravity: scale to 0-1 range
    gravity_min, gravity_max = -1500, -200
    normalized[:, 0] = (params[:, 0] - gravity_min) / (gravity_max - gravity_min)
    
    # Friction: already 0-1, but ensure it
    normalized[:, 1] = np.clip(params[:, 1], 0, 1)
    
    # Elasticity: already 0-1, but ensure it  
    normalized[:, 2] = np.clip(params[:, 2], 0, 1)
    
    # Damping: already close to 0-1, normalize to 0-1
    damping_min, damping_max = 0.8, 1.0
    normalized[:, 3] = (params[:, 3] - damping_min) / (damping_max - damping_min)
    
    return normalized


def denormalize_physics_params(normalized_params):
    """Convert normalized parameters back to original scales"""
    denormalized = normalized_params.copy()
    
    # Gravity: scale back
    gravity_min, gravity_max = -1500, -200
    denormalized[:, 0] = normalized_params[:, 0] * (gravity_max - gravity_min) + gravity_min
    
    # Friction: already correct scale
    denormalized[:, 1] = normalized_params[:, 1]
    
    # Elasticity: already correct scale
    denormalized[:, 2] = normalized_params[:, 2]
    
    # Damping: scale back
    damping_min, damping_max = 0.8, 1.0
    denormalized[:, 3] = normalized_params[:, 3] * (damping_max - damping_min) + damping_min
    
    return denormalized


def calculate_modification_consistency(y_true_orig, y_pred_orig, X_modifications, tolerance=0.15):
    """Calculate modification consistency metrics"""
    
    # Calculate directional accuracy for each modification type
    mod_types = np.unique(X_modifications)
    results = {}
    
    for mod_type in mod_types:
        mask = X_modifications == mod_type
        if np.sum(mask) == 0:
            continue
            
        true_mod = y_true_orig[mask]
        pred_mod = y_pred_orig[mask]
        
        # Calculate relative error
        rel_error = np.abs((pred_mod - true_mod) / (np.abs(true_mod) + 1e-8))
        
        # Consistency within tolerance
        consistency = np.mean(rel_error < tolerance)
        
        # Mean absolute error
        mae = np.mean(np.abs(pred_mod - true_mod))
        
        results[f'mod_type_{mod_type}'] = {
            'consistency': consistency,
            'mae': mae,
            'samples': np.sum(mask)
        }
    
    # Overall consistency
    overall_error = np.abs((y_pred_orig - y_true_orig) / (np.abs(y_true_orig) + 1e-8))
    overall_consistency = np.mean(overall_error < tolerance)
    overall_mae = np.mean(np.abs(y_pred_orig - y_true_orig))
    
    results['overall'] = {
        'consistency': overall_consistency,
        'mae': overall_mae
    }
    
    return results


def train_distribution_modifier():
    """Train distribution modification model"""
    
    # Configuration
    config = ModifierConfig(
        modification_embedding_dim=32,
        rule_embedding_dim=64,
        hidden_dim=128,
        learning_rate=1e-3
    )
    
    # Generate synthetic modification data if it doesn't exist
    modification_data_path = 'data/processed/physics_worlds/modification_pairs.pkl'
    if not os.path.exists(modification_data_path):
        print("Generating synthetic modification data...")
        generate_synthetic_modification_data(modification_data_path)
    
    # Load data
    X_original, X_modifications, y_target, X_original_orig, y_target_orig = load_and_prepare_modification_data(
        modification_data_path, 
        max_samples=2000
    )
    
    # Split into train/val
    split_idx = int(0.8 * len(X_original))
    X_orig_train, X_orig_val = X_original[:split_idx], X_original[split_idx:]
    X_mod_train, X_mod_val = X_modifications[:split_idx], X_modifications[split_idx:]
    y_train, y_val = y_target[:split_idx], y_target[split_idx:]
    X_orig_train_orig, X_orig_val_orig = X_original_orig[:split_idx], X_original_orig[split_idx:]
    y_train_orig, y_val_orig = y_target_orig[:split_idx], y_target_orig[split_idx:]
    
    # Create model
    model = DistributionModificationTrainer(config)
    
    # Build model with dummy input
    dummy_physics = X_orig_train[:1]
    dummy_modifications = X_mod_train[:1]
    _ = model([dummy_physics, dummy_modifications])
    
    print(f"Model parameters: {model.count_params():,}")
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    # Data generator for training
    def data_generator(X_orig, X_mod, y, batch_size=32, shuffle=True):
        n_samples = len(X_orig)
        indices = np.arange(n_samples)
        
        while True:
            if shuffle:
                np.random.shuffle(indices)
            
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                batch_orig = X_orig[batch_indices]
                batch_mod = X_mod[batch_indices]
                batch_target = y[batch_indices]
                
                yield [batch_orig, batch_mod], batch_target
    
    # Setup callbacks
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6)
    ]
    
    # Calculate steps
    batch_size = 32
    steps_per_epoch = len(X_orig_train) // batch_size
    validation_steps = len(X_orig_val) // batch_size
    
    # Create generators
    train_gen = data_generator(X_orig_train, X_mod_train, y_train, batch_size, shuffle=True)
    val_gen = data_generator(X_orig_val, X_mod_val, y_val, batch_size, shuffle=False)
    
    # Train
    print("Starting distribution modification training...")
    history = model.fit(
        train_gen,
        epochs=100,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Evaluate
    print("Evaluating distribution modification...")
    
    # Generate predictions
    y_pred = np.array(model([X_orig_val, X_mod_val]))
    
    # Denormalize for evaluation
    y_pred_orig = denormalize_physics_params(y_pred)
    
    # Calculate metrics
    metrics = calculate_modification_consistency(y_val_orig, y_pred_orig, X_mod_val, tolerance=0.15)
    
    print(f"\nDistribution Modification Results:")
    print(f"Overall consistency (15% tolerance): {metrics['overall']['consistency']:.3f}")
    print(f"Overall MAE: {metrics['overall']['mae']:.4f}")
    
    # Per modification type results
    print(f"\nPer-modification type results:")
    for key, value in metrics.items():
        if key.startswith('mod_type_'):
            mod_type = int(key.split('_')[-1])
            print(f"  Modification {mod_type}: {value['consistency']:.3f} consistency, {value['samples']} samples")
    
    # Check if target achieved
    target_consistency = 0.70
    achieved = metrics['overall']['consistency'] >= target_consistency
    
    print(f"\n{'='*60}")
    print(f"DISTRIBUTION MODIFIER PRE-TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Target consistency: {target_consistency:.1%}")
    print(f"Achieved consistency: {metrics['overall']['consistency']:.1%}")
    print(f"Target achieved: {'✅ YES' if achieved else '❌ NO'}")
    
    if achieved:
        print(f"\n🎉 Distribution modification pre-training successful!")
        
        # Save model
        os.makedirs('outputs/checkpoints', exist_ok=True)
        model.save('outputs/checkpoints/distribution_modifier.keras')
        print(f"Model saved to: outputs/checkpoints/distribution_modifier.keras")
    else:
        print(f"\n📈 Training completed but target not yet reached.")
        print(f"Best consistency so far: {metrics['overall']['consistency']:.1%}")
    
    # Show some sample predictions
    print(f"\nSample modification predictions:")
    for i in range(min(3, len(y_val_orig))):
        print(f"Sample {i} (mod type {X_mod_val[i]}):")
        print(f"  Original: G={X_orig_val_orig[i,0]:.0f}, F={X_orig_val_orig[i,1]:.3f}, E={X_orig_val_orig[i,2]:.3f}, D={X_orig_val_orig[i,3]:.3f}")
        print(f"  Target:   G={y_val_orig[i,0]:.0f}, F={y_val_orig[i,1]:.3f}, E={y_val_orig[i,2]:.3f}, D={y_val_orig[i,3]:.3f}")
        print(f"  Predict:  G={y_pred_orig[i,0]:.0f}, F={y_pred_orig[i,1]:.3f}, E={y_pred_orig[i,2]:.3f}, D={y_pred_orig[i,3]:.3f}")
    
    return model, history, metrics


def generate_synthetic_modification_data(output_path: str, num_samples: int = 2000):
    """Generate synthetic modification training data"""
    print(f"Generating {num_samples} synthetic modification pairs...")
    
    # Load original physics data
    with open('data/processed/physics_worlds/train_data.pkl', 'rb') as f:
        physics_data = pickle.load(f)
    
    modification_pairs = []
    
    # Modification functions
    def gravity_increase(physics, factor=1.2):
        new_physics = physics.copy()
        new_physics['gravity'] = max(physics['gravity'] * factor, -1500)
        return new_physics, 'gravity_increase'
        
    def gravity_decrease(physics, factor=0.8):
        new_physics = physics.copy()
        new_physics['gravity'] = min(physics['gravity'] * factor, -200)
        return new_physics, 'gravity_decrease'
        
    def friction_increase(physics, delta=0.2):
        new_physics = physics.copy()
        new_physics['friction'] = min(physics['friction'] + delta, 0.95)
        return new_physics, 'friction_increase'
        
    def friction_decrease(physics, delta=0.2):
        new_physics = physics.copy()
        new_physics['friction'] = max(physics['friction'] - delta, 0.05)
        return new_physics, 'friction_decrease'
        
    def elasticity_increase(physics, delta=0.2):
        new_physics = physics.copy()
        new_physics['elasticity'] = min(physics['elasticity'] + delta, 0.99)
        return new_physics, 'elasticity_increase'
        
    def elasticity_decrease(physics, delta=0.2):
        new_physics = physics.copy()
        new_physics['elasticity'] = max(physics['elasticity'] - delta, 0.1)
        return new_physics, 'elasticity_decrease'
    
    modifications = [
        gravity_increase, gravity_decrease,
        friction_increase, friction_decrease,
        elasticity_increase, elasticity_decrease
    ]
    
    # Generate pairs
    for i in range(num_samples):
        # Select random physics sample
        base_sample = np.random.choice(physics_data)
        original_physics = base_sample['physics_config']
        
        # Select random modification
        mod_func = np.random.choice(modifications)
        target_physics, mod_type = mod_func(original_physics)
        
        modification_pairs.append({
            'original_physics': original_physics,
            'target_physics': target_physics,
            'modification_type': mod_type
        })
    
    # Save modification pairs
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(modification_pairs, f)
    
    print(f"Generated and saved {len(modification_pairs)} modification pairs to {output_path}")


if __name__ == "__main__":
    train_distribution_modifier()