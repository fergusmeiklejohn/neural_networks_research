"""
Pre-training script for Physics Rule Extractor component
"""

import sys
import os
sys.path.append('../..')

import numpy as np
import pickle
import keras
from keras import callbacks
import wandb
from tqdm import tqdm
from typing import Dict, List, Tuple

from models.core.physics_rule_extractor import create_physics_rule_extractor, PhysicsRuleConfig


def load_and_filter_data(data_path: str, max_balls: int = 2, max_samples: int = None):
    """Load and filter training data for consistent ball count"""
    print(f"Loading data from {data_path}...")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Filter to only samples with specified ball count
    filtered_data = [sample for sample in data if sample['num_balls'] == max_balls]
    
    if max_samples:
        filtered_data = filtered_data[:max_samples]
    
    print(f"Filtered {len(filtered_data)} samples with {max_balls} balls")
    return filtered_data


def prepare_training_data(samples: List[Dict], config: PhysicsRuleConfig):
    """Prepare training data for rule extraction"""
    print("Preparing training data...")
    
    trajectories = []
    physics_labels = {
        'gravity': [],
        'friction': [],
        'elasticity': [],
        'damping': [],
        'independence_score': [],
        'consistency_score': [],
        'features': []
    }
    
    for sample in tqdm(samples, desc="Processing samples"):
        # Extract trajectory
        traj = np.array(sample['trajectory'])
        
        # Truncate to sequence_length
        if len(traj) > config.sequence_length:
            traj = traj[:config.sequence_length]
        elif len(traj) < config.sequence_length:
            # Pad with zeros if needed
            padding = np.zeros((config.sequence_length - len(traj), traj.shape[1]))
            traj = np.concatenate([traj, padding], axis=0)
        
        trajectories.append(traj)
        
        # Extract physics parameters
        physics_config = sample['physics_config']
        physics_labels['gravity'].append([physics_config['gravity']])
        physics_labels['friction'].append([physics_config['friction']])
        physics_labels['elasticity'].append([physics_config['elasticity']])
        physics_labels['damping'].append([physics_config['damping']])
        
        # Add dummy targets for other outputs (we only care about physics params)
        physics_labels['independence_score'].append([1.0])  # Target: high independence
        physics_labels['consistency_score'].append([1.0])   # Target: high consistency
        physics_labels['features'].append(np.zeros(64))     # Dummy feature target
    
    # Convert to numpy arrays
    X = np.array(trajectories)
    y = {key: np.array(values) for key, values in physics_labels.items()}
    
    print(f"Training data shape: {X.shape}")
    print(f"Label shapes: {[f'{k}: {v.shape}' for k, v in y.items()]}")
    
    return X, y


def create_data_generators(X_train, y_train, X_val, y_val, batch_size=16):
    """Create data generators for training"""
    
    def data_generator(X, y, batch_size, shuffle=True):
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        while True:
            if shuffle:
                np.random.shuffle(indices)
            
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                batch_X = X[batch_indices]
                batch_y = {key: y[key][batch_indices] for key in y.keys()}
                
                yield batch_X, batch_y
    
    train_gen = data_generator(X_train, y_train, batch_size, shuffle=True)
    val_gen = data_generator(X_val, y_val, batch_size, shuffle=False)
    
    return train_gen, val_gen


def calculate_accuracy_metrics(model, X, y_true):
    """Calculate physics parameter extraction accuracy"""
    predictions = model.extract_rules(X)
    
    metrics = {}
    for param in ['gravity', 'friction', 'elasticity', 'damping']:
        pred = predictions[param]
        true = y_true[param]
        
        # Calculate relative error
        rel_error = np.abs((pred - true) / (true + 1e-8))
        
        # Accuracy within 10% tolerance
        accuracy_10 = np.mean(rel_error < 0.1)
        
        # Accuracy within 20% tolerance  
        accuracy_20 = np.mean(rel_error < 0.2)
        
        # Mean absolute error
        mae = np.mean(np.abs(pred - true))
        
        metrics[param] = {
            'accuracy_10pct': accuracy_10,
            'accuracy_20pct': accuracy_20,
            'mae': mae,
            'mean_rel_error': np.mean(rel_error)
        }
    
    # Overall accuracy (average across parameters)
    overall_acc_10 = np.mean([metrics[p]['accuracy_10pct'] for p in metrics.keys()])
    overall_acc_20 = np.mean([metrics[p]['accuracy_20pct'] for p in metrics.keys()])
    
    metrics['overall'] = {
        'accuracy_10pct': overall_acc_10,
        'accuracy_20pct': overall_acc_20
    }
    
    return metrics


def train_rule_extractor(config: PhysicsRuleConfig, 
                        train_samples: int = 1000,
                        val_samples: int = 200,
                        epochs: int = 50,
                        batch_size: int = 16,
                        use_wandb: bool = True):
    """Main training function for rule extractor"""
    
    # Initialize W&B
    if use_wandb:
        wandb.init(
            project="distribution-invention",
            name="physics-rule-extractor-pretrain",
            config={
                'sequence_length': config.sequence_length,
                'feature_dim': config.feature_dim,
                'hidden_dim': config.hidden_dim,
                'num_transformer_layers': config.num_transformer_layers,
                'train_samples': train_samples,
                'val_samples': val_samples,
                'epochs': epochs,
                'batch_size': batch_size
            }
        )
    
    # Load data
    train_data = load_and_filter_data(
        'data/processed/physics_worlds/train_data.pkl',
        max_balls=2,
        max_samples=train_samples
    )
    
    val_data = load_and_filter_data(
        'data/processed/physics_worlds/val_data.pkl', 
        max_balls=2,
        max_samples=val_samples
    )
    
    # Prepare training data
    X_train, y_train = prepare_training_data(train_data, config)
    X_val, y_val = prepare_training_data(val_data, config)
    
    # Create model
    print("Creating PhysicsRuleExtractor model...")
    model = create_physics_rule_extractor(config)
    
    # Build model by doing a forward pass
    dummy_input = X_train[:1]  # Use first sample to build model
    _ = model(dummy_input)
    
    # Create data generators
    train_gen, val_gen = create_data_generators(X_train, y_train, X_val, y_val, batch_size)
    
    # Setup callbacks
    callbacks_list = []
    
    # Model checkpointing
    checkpoint_path = "outputs/checkpoints/rule_extractor_best.keras"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    callbacks_list.append(checkpoint_callback)
    
    # Early stopping
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    callbacks_list.append(early_stopping)
    
    # Learning rate reduction
    lr_reduction = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    callbacks_list.append(lr_reduction)
    
    # Custom accuracy callback
    class AccuracyCallback(callbacks.Callback):
        def __init__(self, X_val, y_val, freq=5):
            self.X_val = X_val
            self.y_val = y_val
            self.freq = freq
            
        def on_epoch_end(self, epoch, logs=None):
            if epoch % self.freq == 0:
                metrics = calculate_accuracy_metrics(self.model, self.X_val, self.y_val)
                
                # Log to console
                print(f"\nEpoch {epoch} Accuracy Metrics:")
                print(f"  Overall 10% accuracy: {metrics['overall']['accuracy_10pct']:.3f}")
                print(f"  Overall 20% accuracy: {metrics['overall']['accuracy_20pct']:.3f}")
                
                # Log to W&B
                if use_wandb:
                    wandb_metrics = {'epoch': epoch}
                    for param in ['gravity', 'friction', 'elasticity', 'damping']:
                        wandb_metrics[f'{param}_acc_10pct'] = metrics[param]['accuracy_10pct']
                        wandb_metrics[f'{param}_acc_20pct'] = metrics[param]['accuracy_20pct']
                        wandb_metrics[f'{param}_mae'] = metrics[param]['mae']
                    
                    wandb_metrics['overall_acc_10pct'] = metrics['overall']['accuracy_10pct']
                    wandb_metrics['overall_acc_20pct'] = metrics['overall']['accuracy_20pct']
                    
                    wandb.log(wandb_metrics)
    
    accuracy_callback = AccuracyCallback(X_val, y_val)
    callbacks_list.append(accuracy_callback)
    
    if use_wandb:
        wandb_callback = callbacks.WandbCallback(
            monitor='val_loss',
            save_model=False
        )
        callbacks_list.append(wandb_callback)
    
    # Calculate steps per epoch
    steps_per_epoch = len(X_train) // batch_size
    validation_steps = len(X_val) // batch_size
    
    print(f"\nStarting training...")
    print(f"  Model parameters: {model.count_params():,}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Validation steps: {validation_steps}")
    
    # Train the model
    history = model.fit(
        train_gen,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Final evaluation
    print("\nFinal evaluation...")
    final_metrics = calculate_accuracy_metrics(model, X_val, y_val)
    
    print(f"\nFinal Results:")
    print(f"  Overall 10% accuracy: {final_metrics['overall']['accuracy_10pct']:.3f}")
    print(f"  Overall 20% accuracy: {final_metrics['overall']['accuracy_20pct']:.3f}")
    
    for param in ['gravity', 'friction', 'elasticity', 'damping']:
        print(f"  {param.capitalize()}:")
        print(f"    10% accuracy: {final_metrics[param]['accuracy_10pct']:.3f}")
        print(f"    20% accuracy: {final_metrics[param]['accuracy_20pct']:.3f}")
        print(f"    MAE: {final_metrics[param]['mae']:.4f}")
    
    if use_wandb:
        # Log final metrics
        final_wandb_metrics = {}
        for param in ['gravity', 'friction', 'elasticity', 'damping']:
            final_wandb_metrics[f'final_{param}_acc_10pct'] = final_metrics[param]['accuracy_10pct']
            final_wandb_metrics[f'final_{param}_acc_20pct'] = final_metrics[param]['accuracy_20pct']
        
        final_wandb_metrics['final_overall_acc_10pct'] = final_metrics['overall']['accuracy_10pct']
        final_wandb_metrics['final_overall_acc_20pct'] = final_metrics['overall']['accuracy_20pct']
        
        wandb.log(final_wandb_metrics)
        wandb.finish()
    
    return model, history, final_metrics


if __name__ == "__main__":
    # Configuration for rule extractor training
    config = PhysicsRuleConfig(
        sequence_length=100,      # Use 100 frames of 300 
        feature_dim=17,           # 2-ball data has 17 features
        max_balls=2,              # Focus on 2-ball scenarios
        num_transformer_layers=3, # Enough for learning
        num_attention_heads=8,
        hidden_dim=128,           # Good balance of capacity/speed
        rule_embedding_dim=64,
        learning_rate=1e-4,
        dropout_rate=0.1
    )
    
    # Train with small data size for testing first
    model, history, metrics = train_rule_extractor(
        config=config,
        train_samples=500,        # 500 training samples for testing
        val_samples=100,          # 100 validation samples  
        epochs=20,                # 20 epochs for quick test
        batch_size=16,            # Smaller batch size
        use_wandb=False           # Disable W&B for now
    )
    
    # Check if we achieved target
    target_accuracy = 0.80  # 80% target
    achieved = metrics['overall']['accuracy_10pct'] >= target_accuracy
    
    print(f"\n{'='*50}")
    print(f"RULE EXTRACTOR PRE-TRAINING COMPLETE")
    print(f"{'='*50}")
    print(f"Target accuracy: {target_accuracy:.1%}")
    print(f"Achieved accuracy: {metrics['overall']['accuracy_10pct']:.1%}")
    print(f"Target achieved: {'✅ YES' if achieved else '❌ NO'}")
    
    if achieved:
        print(f"\n🎉 Rule extraction pre-training successful!")
        print(f"Model saved to: outputs/checkpoints/rule_extractor_best.keras")
    else:
        print(f"\n📈 Training completed but target not yet reached.")
        print(f"Consider: longer training, more data, or hyperparameter tuning.")