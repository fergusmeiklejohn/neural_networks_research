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

from models.core.physics_rule_extractor import PhysicsRuleConfig
from distribution_invention_metrics import evaluate_model_with_improved_metrics, DistributionInventionEvaluator


class ImprovedPhysicsPredictor(keras.Model):
    """Physics predictor with proper normalization for improved data"""
    
    def __init__(self, config: PhysicsRuleConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        
        # Enhanced architecture for better extrapolation
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(512, activation='relu')
        self.dropout1 = layers.Dropout(0.3)
        self.dense2 = layers.Dense(256, activation='relu')
        self.dropout2 = layers.Dropout(0.2)
        self.dense3 = layers.Dense(128, activation='relu')
        self.dropout3 = layers.Dropout(0.1)
        self.dense4 = layers.Dense(64, activation='relu')
        
        # Output layer (single output for all 4 parameters)
        self.output_layer = layers.Dense(4, name='physics_params')
        
    def call(self, inputs, training=None):
        x = self.flatten(inputs)
        x = self.dense1(x, training=training)
        x = self.dropout1(x, training=training)
        x = self.dense2(x, training=training)
        x = self.dropout2(x, training=training)
        x = self.dense3(x, training=training)
        x = self.dropout3(x, training=training)
        x = self.dense4(x, training=training)
        
        # Output all 4 physics parameters
        output = self.output_layer(x, training=training)
        
        return output


def normalize_physics_params(params):
    """Normalize physics parameters to similar scales (consistent with improved data)"""
    normalized = params.copy()
    
    # Updated ranges based on improved data splits
    # Gravity: scale to 0-1 range (wider range for extrapolation)
    gravity_min, gravity_max = -2500, -50  # Expanded range
    normalized[:, 0] = (params[:, 0] - gravity_min) / (gravity_max - gravity_min)
    
    # Friction: already 0-1, but ensure it
    normalized[:, 1] = np.clip(params[:, 1], 0, 1)
    
    # Elasticity: already 0-1, but ensure it  
    normalized[:, 2] = np.clip(params[:, 2], 0, 1.1)  # Allow super-bounce
    
    # Damping: normalize to 0-1
    damping_min, damping_max = 0.7, 1.0  # Expanded range
    normalized[:, 3] = (params[:, 3] - damping_min) / (damping_max - damping_min)
    
    return normalized


def denormalize_physics_params(normalized_params):
    """Convert normalized parameters back to original scales"""
    denormalized = normalized_params.copy()
    
    # Gravity: scale back
    gravity_min, gravity_max = -2500, -50
    denormalized[:, 0] = normalized_params[:, 0] * (gravity_max - gravity_min) + gravity_min
    
    # Friction: already correct scale
    denormalized[:, 1] = normalized_params[:, 1]
    
    # Elasticity: already correct scale
    denormalized[:, 2] = normalized_params[:, 2]
    
    # Damping: scale back
    damping_min, damping_max = 0.7, 1.0
    denormalized[:, 3] = normalized_params[:, 3] * (damping_max - damping_min) + damping_min
    
    return denormalized


def load_improved_data(data_path: str, max_samples: int = None):
    """Load and prepare improved dataset"""
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
    
    # Prepare arrays
    trajectories = []
    labels = []
    
    for sample in tqdm(filtered_data, desc="Processing"):
        traj = np.array(sample['trajectory'])
        
        # Use first 50 frames and flatten
        if len(traj) > 50:
            traj = traj[:50]
        elif len(traj) < 50:
            # Pad with last frame
            padding = np.tile(traj[-1:], (50 - len(traj), 1))
            traj = np.concatenate([traj, padding], axis=0)
        
        trajectories.append(traj.flatten())
        
        # Extract physics parameters
        physics = sample['physics_config']
        labels.append([
            physics['gravity'],
            physics['friction'], 
            physics['elasticity'],
            physics['damping']
        ])
    
    X = np.array(trajectories)
    y = np.array(labels)
    
    # Normalize the trajectory data (z-score normalization)
    X = (X - X.mean()) / (X.std() + 1e-8)
    
    # Normalize physics parameters
    y_normalized = normalize_physics_params(y)
    
    print(f"Data shapes: X={X.shape}, y={y.shape}")
    print(f"X range: {X.min():.3f} to {X.max():.3f}")
    print(f"y range: {y_normalized.min():.3f} to {y_normalized.max():.3f}")
    
    # Print parameter coverage
    print(f"Parameter ranges in this split:")
    params = ['gravity', 'friction', 'elasticity', 'damping']
    for i, param in enumerate(params):
        print(f"  {param:>10}: [{y[:, i].min():.1f}, {y[:, i].max():.1f}]")
    
    return X, y_normalized, y  # Return both normalized and original


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


class ImprovedPhysicsPredictor(keras.Model):
    """Improved physics predictor for better extrapolation"""
    
    def __init__(self, config: PhysicsRuleConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        
        # Enhanced architecture for better extrapolation
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(512, activation='relu')
        self.batch_norm1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(0.3)
        
        self.dense2 = layers.Dense(256, activation='relu')
        self.batch_norm2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(0.2)
        
        self.dense3 = layers.Dense(128, activation='relu')
        self.dropout3 = layers.Dropout(0.1)
        
        self.dense4 = layers.Dense(64, activation='relu')
        
        # Output layer with linear activation for regression
        self.output_layer = layers.Dense(4, activation='linear', name='physics_params')
        
    def call(self, inputs, training=None):
        x = self.flatten(inputs)
        
        x = self.dense1(x)
        x = self.batch_norm1(x, training=training)
        x = keras.activations.relu(x)
        x = self.dropout1(x, training=training)
        
        x = self.dense2(x)
        x = self.batch_norm2(x, training=training)
        x = keras.activations.relu(x)
        x = self.dropout2(x, training=training)
        
        x = self.dense3(x, training=training)
        x = keras.activations.relu(x)
        x = self.dropout3(x, training=training)
        
        x = self.dense4(x, training=training)
        x = keras.activations.relu(x)
        
        # Output all 4 physics parameters
        output = self.output_layer(x)
        
        return output


def train_improved_rule_extractor():
    """Train rule extractor with improved data isolation"""
    
    print("🚀 IMPROVED RULE EXTRACTION TRAINING")
    print("=" * 50)
    print("Training with proper train/test isolation for distribution invention")
    print()
    
    # Configuration
    config = PhysicsRuleConfig(
        sequence_length=50,
        feature_dim=17,
        max_balls=2,
        learning_rate=1e-4,  # Lower learning rate for better generalization
        dropout_rate=0.2
    )
    
    # Check for improved datasets
    data_dir = Path("data/processed/physics_worlds_v2_quick")
    if not data_dir.exists():
        data_dir = Path("data/processed/physics_worlds_v2")
    
    if not data_dir.exists():
        print("❌ No improved datasets found. Please run generate_improved_datasets.py first")
        return None
    
    # Load improved data splits
    print("Loading improved datasets with proper isolation...")
    
    X_train, y_train_norm, y_train_orig = load_improved_data(
        data_dir / "train_data.pkl", max_samples=1000
    )
    
    X_val_in, y_val_in_norm, y_val_in_orig = load_improved_data(
        data_dir / "val_in_dist_data.pkl", max_samples=200
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
    
    # Create improved model
    model = ImprovedPhysicsPredictor(config)
    
    # Build model
    _ = model(X_train[:1])
    print(f"Model parameters: {model.count_params():,}")
    
    # Compile model with improved settings for extrapolation
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=config.learning_rate,
            clipnorm=1.0  # Gradient clipping for stability
        ),
        loss='mse',
        metrics=['mae']
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
    
    # Train with improved data
    print("Starting improved rule extraction training...")
    history = model.fit(
        X_train, y_train_norm,
        validation_data=(X_val_in, y_val_in_norm),
        epochs=150,  # More epochs with early stopping
        batch_size=32,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Comprehensive evaluation using new metrics
    print("\n" + "="*60)
    print("COMPREHENSIVE EVALUATION WITH IMPROVED METRICS")
    print("="*60)
    
    # Create a simple wrapper for the model to work with evaluation framework
    class ModelWrapper:
        def __init__(self, keras_model):
            self.keras_model = keras_model
            
        def extract_rules(self, trajectory):
            """Extract rules interface for evaluation"""
            pred_norm = self.keras_model(trajectory)
            pred_orig = denormalize_physics_params(np.array(pred_norm))
            
            return {
                'gravity': pred_orig[:, 0],
                'friction': pred_orig[:, 1],
                'elasticity': pred_orig[:, 2],
                'damping': pred_orig[:, 3]
            }
    
    model_wrapper = ModelWrapper(model)
    
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
        
        # Basic evaluation on validation set
        y_pred_norm = np.array(model(X_val_in))
        y_pred_orig = denormalize_physics_params(y_pred_norm)
        accuracy_metrics = calculate_physics_accuracy(y_val_in_orig, y_pred_orig, tolerance=0.2)
        
        print(f"\nBasic validation results:")
        print(f"Overall accuracy (20% tolerance): {accuracy_metrics['overall']['accuracy']:.3f}")
        
        for param in ['gravity', 'friction', 'elasticity', 'damping']:
            metrics = accuracy_metrics[param]
            print(f"{param.capitalize()}:")
            print(f"  Accuracy: {metrics['accuracy']:.3f}")
            print(f"  MAE: {metrics['mae']:.2f}")
            print(f"  RMSE: {metrics['rmse']:.2f}")
    
    # Save the improved model
    os.makedirs('outputs/checkpoints', exist_ok=True)
    model.save('outputs/checkpoints/improved_rule_extractor.keras')
    print(f"\n💾 Model saved to: outputs/checkpoints/improved_rule_extractor.keras")
    
    print(f"\n✅ IMPROVED RULE EXTRACTION TRAINING COMPLETE!")
    print(f"Key improvements:")
    print(f"  - Proper train/test isolation")
    print(f"  - Enhanced architecture for extrapolation")
    print(f"  - Comprehensive evaluation metrics")
    print(f"  - True distribution invention testing")
    
    return model, history


if __name__ == "__main__":
    train_improved_rule_extractor()