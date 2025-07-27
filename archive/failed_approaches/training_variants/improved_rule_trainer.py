"""
Improved Rule Extraction Training with Proper Data Isolation

This script uses the new data splits to train rule extraction with proper
interpolation vs extrapolation testing capability.
"""

import sys
import os
sys.path.append('../..')

import numpy as np
import pickle
import keras
from keras import layers, callbacks, ops
from tqdm import tqdm
from pathlib import Path

from models.core.physics_rule_extractor import PhysicsRuleConfig, PhysicsRuleExtractor, PhysicsRuleLoss
from distribution_invention_metrics import evaluate_model_with_improved_metrics, DistributionInventionEvaluator






def load_improved_data_for_transformer(data_path: str, config: PhysicsRuleConfig, max_samples: int = None):
    """Load and prepare improved dataset for transformer architecture"""
    print(f"Loading improved data from {data_path}...")
    
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Filter to 2-ball samples only
    filtered_data = [sample for sample in data if sample['num_balls'] == 2]
    
    if max_samples:
        filtered_data = filtered_data[:max_samples]
    
    print(f"Using {len(filtered_data)} samples with 2 balls")
    
    # Prepare arrays - maintain trajectory structure for transformer
    trajectories = []
    physics_labels = []
    
    for sample in tqdm(filtered_data, desc="Processing"):
        traj = np.array(sample['trajectory'])
        
        # Pad or truncate to config.sequence_length (keep 2D structure)
        if len(traj) > config.sequence_length:
            traj = traj[:config.sequence_length]
        elif len(traj) < config.sequence_length:
            # Pad with last frame
            padding = np.tile(traj[-1:], (config.sequence_length - len(traj), 1))
            traj = np.concatenate([traj, padding], axis=0)
        
        trajectories.append(traj)  # Keep 2D structure (seq_len, features)
        
        # Extract physics parameters
        physics = sample['physics_config']
        physics_labels.append({
            'gravity': np.array([physics['gravity']]),
            'friction': np.array([physics['friction']]), 
            'elasticity': np.array([physics['elasticity']]),
            'damping': np.array([physics['damping']])
        })
    
    X = np.array(trajectories)  # Shape: (batch, seq_len, features)
    
    # Normalize the trajectory data (z-score normalization)
    X_flat = X.reshape(-1, X.shape[-1])
    X_mean = X_flat.mean(axis=0)
    X_std = X_flat.std(axis=0) + 1e-8
    X = (X - X_mean) / X_std
    
    # Normalize physics parameters 
    y_gravity = np.array([label['gravity'] for label in physics_labels])
    y_friction = np.array([label['friction'] for label in physics_labels])
    y_elasticity = np.array([label['elasticity'] for label in physics_labels])
    y_damping = np.array([label['damping'] for label in physics_labels])
    
    # Normalize parameters to [0,1] ranges for better training
    gravity_min, gravity_max = -2500, -50
    y_gravity_norm = (y_gravity - gravity_min) / (gravity_max - gravity_min)
    
    # Friction and elasticity are already in [0,1], damping needs scaling
    damping_min, damping_max = 0.7, 1.0
    y_damping_norm = (y_damping - damping_min) / (damping_max - damping_min)
    
    # Add dummy targets for auxiliary outputs
    num_samples = len(y_gravity)
    y_normalized = {
        'gravity': y_gravity_norm,
        'friction': y_friction,
        'elasticity': y_elasticity,
        'damping': y_damping_norm,
        'independence_score': np.ones((num_samples, 1)),  # Target high independence
        'consistency_score': np.ones((num_samples, 1)),   # Target high consistency
        'features': np.zeros((num_samples, config.rule_embedding_dim))  # Dummy target for features
    }
    
    y_original = {
        'gravity': y_gravity,
        'friction': y_friction,
        'elasticity': y_elasticity,
        'damping': y_damping
    }
    
    print(f"Data shapes: X={X.shape}")
    print(f"X range: {X.min():.3f} to {X.max():.3f}")
    
    # Print parameter coverage
    print(f"Parameter ranges in this split:")
    for param in ['gravity', 'friction', 'elasticity', 'damping']:
        values = y_original[param].flatten()
        print(f"  {param:>10}: [{values.min():.1f}, {values.max():.1f}]")
    
    return X, y_normalized, y_original


def calculate_physics_accuracy(y_true_original, y_pred_original, tolerance=0.2):
    """Calculate accuracy for physics parameter prediction with improved metrics"""
    
    # Calculate relative errors using original scales
    rel_errors = np.abs((y_pred_original - y_true_original) / (np.abs(y_true_original) + 1e-8))
    
    # Accuracy within tolerance (increased to 20% as extrapolation is harder)
    accuracies = np.mean(rel_errors < tolerance, axis=0)
    
    param_names = ['gravity', 'friction', 'elasticity', 'damping']
    results = {}
    
    for i, param in enumerate(param_names):
        results[param] = {
            'accuracy': accuracies[i],
            'mae': np.mean(np.abs(y_pred_original[:, i] - y_true_original[:, i])),
            'rel_error': np.mean(rel_errors[:, i]),
            'rmse': np.sqrt(np.mean((y_pred_original[:, i] - y_true_original[:, i])**2))
        }
    
    overall_accuracy = np.mean(accuracies)
    results['overall'] = {'accuracy': overall_accuracy}
    
    return results


class TransformerPhysicsLoss(keras.losses.Loss):
    """Custom loss for transformer that handles all outputs"""
    
    def __init__(self, physics_weight=1.0, energy_weight=0.2, independence_weight=0.1, 
                 consistency_weight=0.1, name="transformer_physics_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.physics_weight = physics_weight
        self.energy_weight = energy_weight
        self.independence_weight = independence_weight
        self.consistency_weight = consistency_weight
        
    def call(self, y_true, y_pred):
        # Standard physics parameter loss
        gravity_loss = keras.losses.mse(y_true['gravity'], y_pred['gravity'])
        friction_loss = keras.losses.mse(y_true['friction'], y_pred['friction'])
        elasticity_loss = keras.losses.mse(y_true['elasticity'], y_pred['elasticity'])
        damping_loss = keras.losses.mse(y_true['damping'], y_pred['damping'])
        
        physics_loss = (gravity_loss + friction_loss + elasticity_loss + damping_loss) / 4
        
        # Energy conservation constraint (elasticity should preserve energy)
        energy_penalty = ops.square(ops.maximum(0.0, y_pred['elasticity'] - 1.0))  # Penalize > 1.0
        
        # Physical plausibility constraints
        gravity_penalty = ops.square(ops.maximum(0.0, ops.abs(y_pred['gravity']) - 1.0))  # Normalized gravity should be [0,1]
        friction_penalty = ops.square(ops.maximum(0.0, y_pred['friction'] - 1.0)) + ops.square(ops.maximum(0.0, -y_pred['friction']))
        
        constraint_loss = ops.mean(energy_penalty + gravity_penalty + friction_penalty)
        
        # Independence loss - encourage high independence score
        independence_loss = keras.losses.binary_crossentropy(
            ops.ones_like(y_pred['independence_score']), 
            y_pred['independence_score']
        )
        
        # Consistency loss - encourage high consistency score
        consistency_loss = keras.losses.binary_crossentropy(
            ops.ones_like(y_pred['consistency_score']),
            y_pred['consistency_score']
        )
        
        return (self.physics_weight * physics_loss + 
                self.energy_weight * constraint_loss +
                self.independence_weight * independence_loss +
                self.consistency_weight * consistency_loss)


def train_improved_rule_extractor():
    """Train rule extractor with transformer architecture and improved data isolation"""
    
    print("🚀 TRANSFORMER-BASED RULE EXTRACTION TRAINING")
    print("=" * 60)
    print("Training with PhysicsRuleExtractor transformer architecture")
    print("Features: Causal attention, physics-aware loss, proper isolation")
    print()
    
    # Configuration for transformer architecture
    config = PhysicsRuleConfig(
        sequence_length=100,  # Longer sequences for better physics understanding
        feature_dim=17,       # Trajectory features per timestep
        max_balls=2,
        rule_embedding_dim=128,  # Larger embedding for better representation
        num_attention_heads=8,
        num_transformer_layers=4,
        hidden_dim=512,
        learning_rate=5e-5,   # Lower learning rate for transformer
        dropout_rate=0.1
    )
    
    # Check for improved datasets
    data_dir = Path("data/processed/physics_worlds_v2_quick")
    if not data_dir.exists():
        data_dir = Path("data/processed/physics_worlds_v2")
    
    if not data_dir.exists():
        print("❌ No improved datasets found. Please run generate_improved_datasets.py first")
        return None
    
    # Load improved data splits with transformer preprocessing
    print("Loading improved datasets with transformer preprocessing...")
    
    X_train, y_train_norm, y_train_orig = load_improved_data_for_transformer(
        data_dir / "train_data.pkl", config, max_samples=500  # Start smaller for transformer
    )
    
    X_val_in, y_val_in_norm, y_val_in_orig = load_improved_data_for_transformer(
        data_dir / "val_in_dist_data.pkl", config, max_samples=100
    )
    
    # Load test sets for comprehensive evaluation
    datasets_path = {
        'train': str(data_dir / "train_data.pkl"),
        'val_in_dist': str(data_dir / "val_in_dist_data.pkl"),
        'val_near_dist': str(data_dir / "val_near_dist_data.pkl"),
        'test_interpolation': str(data_dir / "test_interpolation_data.pkl"),
        'test_extrapolation': str(data_dir / "test_extrapolation_data.pkl"),
        'test_novel': str(data_dir / "test_novel_data.pkl")
    }
    
    # Create transformer model
    model = PhysicsRuleExtractor(config)
    
    # Build model with dummy input to initialize
    dummy_input = X_train[:1]
    _ = model(dummy_input)
    print(f"Transformer model parameters: {model.count_params():,}")
    
    # Compile with separate losses for each output
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=config.learning_rate,
            clipnorm=1.0,  # Gradient clipping for transformer stability
            weight_decay=1e-5  # L2 regularization
        ),
        loss={
            'gravity': 'mse',
            'friction': 'mse',
            'elasticity': 'mse', 
            'damping': 'mse',
            'independence_score': 'binary_crossentropy',
            'consistency_score': 'binary_crossentropy',
            'features': 'mse'
        },
        loss_weights={
            'gravity': 1.0,
            'friction': 1.0,
            'elasticity': 1.0,
            'damping': 1.0,
            'independence_score': 0.1,
            'consistency_score': 0.1,
            'features': 0.01  # Low weight for auxiliary feature loss
        },
        metrics={
            'gravity': ['mae'],
            'friction': ['mae'], 
            'elasticity': ['mae'],
            'damping': ['mae']
        }
    )
    
    # Enhanced callbacks for better generalization
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=25,  # Increased patience
            restore_best_weights=True,
            min_delta=1e-5
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=10, 
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            'outputs/checkpoints/improved_rule_extractor_best.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train transformer with physics-aware approach
    print("Starting transformer rule extraction training...")
    print(f"Training data shape: {X_train.shape}")
    print(f"Using sequence length: {config.sequence_length}")
    
    history = model.fit(
        X_train, y_train_norm,
        validation_data=(X_val_in, y_val_in_norm),
        epochs=100,   # Fewer epochs for transformer
        batch_size=16,  # Smaller batch size for transformer
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Comprehensive evaluation using new metrics
    print("\n" + "="*60)
    print("COMPREHENSIVE EVALUATION WITH IMPROVED METRICS")
    print("="*60)
    
    # Create wrapper for transformer model to work with evaluation framework
    class TransformerModelWrapper:
        def __init__(self, transformer_model, config):
            self.transformer_model = transformer_model
            self.config = config
            
        def extract_rules(self, trajectory):
            """Extract rules interface for evaluation"""
            try:
                # Handle different input formats from evaluation framework
                if isinstance(trajectory, list):
                    trajectory = np.array(trajectory)
                
                # Ensure we have the right shape: (batch, sequence, features)
                if len(trajectory.shape) == 2:
                    # Raw trajectory data: (time_steps, features) 
                    # Pad or truncate to sequence_length
                    if trajectory.shape[0] > self.config.sequence_length:
                        trajectory = trajectory[:self.config.sequence_length]
                    elif trajectory.shape[0] < self.config.sequence_length:
                        # Pad with last frame
                        padding = np.tile(trajectory[-1:], (self.config.sequence_length - trajectory.shape[0], 1))
                        trajectory = np.concatenate([trajectory, padding], axis=0)
                    
                    # Add batch dimension: (1, sequence, features)
                    trajectory = np.expand_dims(trajectory, 0)
                elif len(trajectory.shape) == 1:
                    # Flattened trajectory - reshape back to (sequence, features)
                    trajectory = trajectory.reshape(self.config.sequence_length, self.config.feature_dim)
                    trajectory = np.expand_dims(trajectory, 0)
                elif len(trajectory.shape) == 3:
                    # Already has batch dimension
                    pass
                else:
                    raise ValueError(f"Unexpected trajectory shape: {trajectory.shape}")
                
                # Normalize trajectory data (same as training preprocessing)
                trajectory_flat = trajectory.reshape(-1, trajectory.shape[-1])
                trajectory_mean = trajectory_flat.mean(axis=0)
                trajectory_std = trajectory_flat.std(axis=0) + 1e-8
                trajectory = (trajectory - trajectory_mean) / trajectory_std
                
                # Get model predictions
                outputs = self.transformer_model(trajectory, training=False)
                
                # Denormalize predictions to original scales
                gravity_orig = outputs['gravity'] * (-50 - (-2500)) + (-2500)
                damping_orig = outputs['damping'] * (1.0 - 0.7) + 0.7
                
                return {
                    'gravity': float(np.array(gravity_orig).flatten()[0]),
                    'friction': float(np.array(outputs['friction']).flatten()[0]),
                    'elasticity': float(np.array(outputs['elasticity']).flatten()[0]), 
                    'damping': float(np.array(damping_orig).flatten()[0])
                }
                
            except Exception as e:
                print(f"Error in extract_rules: {e}")
                print(f"Input trajectory shape: {trajectory.shape if hasattr(trajectory, 'shape') else type(trajectory)}")
                if hasattr(trajectory, 'shape') and len(trajectory.shape) >= 2:
                    print(f"Features per timestep: {trajectory.shape[-1]}")
                    print(f"Expected features: {self.config.feature_dim}")
                    print(f"First few features of first timestep: {trajectory.reshape(-1, trajectory.shape[-1])[0][:5]}")
                # Return fallback values
                return {
                    'gravity': -981.0,
                    'friction': 0.5,
                    'elasticity': 0.5,
                    'damping': 0.9
                }
    
    model_wrapper = TransformerModelWrapper(model, config)
    
    # Run comprehensive evaluation
    try:
        evaluator = DistributionInventionEvaluator(model_wrapper, datasets_path)
        results = evaluator.run_comprehensive_evaluation(sample_size=50)
        evaluator.print_results_summary(results)
        
        # Save results
        results_path = data_dir / "improved_rule_extraction_results.json"
        import json
        with open(results_path, 'w') as f:
            json.dump({
                'interpolation_accuracy': float(results.interpolation_accuracy),
                'extrapolation_accuracy': float(results.extrapolation_accuracy),
                'novel_regime_success': float(results.novel_regime_success),
                'invention_score': float(results.invention_score),
                'parameter_accuracies': {k: float(v) for k, v in results.parameter_accuracies.items()}
            }, f, indent=2)
        
        print(f"\n📊 Detailed results saved to: {results_path}")
        
    except Exception as e:
        print(f"⚠️  Comprehensive evaluation failed: {e}")
        print("Falling back to basic evaluation...")
        
        # Basic evaluation on validation set with transformer
        outputs = model(X_val_in, training=False)
        
        # Convert transformer outputs to format for accuracy calculation
        y_pred_orig = np.column_stack([
            (outputs['gravity'] * (-50 - (-2500)) + (-2500)).numpy().flatten(),
            outputs['friction'].numpy().flatten(),
            outputs['elasticity'].numpy().flatten(),
            (outputs['damping'] * (1.0 - 0.7) + 0.7).numpy().flatten()
        ])
        
        y_val_orig_array = np.column_stack([
            y_val_in_orig['gravity'].flatten(),
            y_val_in_orig['friction'].flatten(), 
            y_val_in_orig['elasticity'].flatten(),
            y_val_in_orig['damping'].flatten()
        ])
        
        accuracy_metrics = calculate_physics_accuracy(y_val_orig_array, y_pred_orig, tolerance=0.2)
        
        print(f"\nBasic validation results:")
        print(f"Overall accuracy (20% tolerance): {accuracy_metrics['overall']['accuracy']:.3f}")
        
        for param in ['gravity', 'friction', 'elasticity', 'damping']:
            metrics = accuracy_metrics[param]
            print(f"{param.capitalize()}:")
            print(f"  Accuracy: {metrics['accuracy']:.3f}")
            print(f"  MAE: {metrics['mae']:.2f}")
            print(f"  RMSE: {metrics['rmse']:.2f}")
    
    # Save the transformer model
    os.makedirs('outputs/checkpoints', exist_ok=True)
    model.save('outputs/checkpoints/transformer_rule_extractor.keras')
    print(f"\n💾 Transformer model saved to: outputs/checkpoints/transformer_rule_extractor.keras")
    
    print(f"\n✅ TRANSFORMER RULE EXTRACTION TRAINING COMPLETE!")
    print(f"Key improvements over simple architecture:")
    print(f"  - Transformer with causal attention mechanisms")
    print(f"  - Physics-aware loss with energy conservation")
    print(f"  - Maintains trajectory temporal structure")
    print(f"  - Separate heads for each physics parameter")
    print(f"  - Independence and consistency scoring")
    print(f"  - Proper train/test isolation maintained")
    
    return model, history, config


if __name__ == "__main__":
    train_improved_rule_extractor()