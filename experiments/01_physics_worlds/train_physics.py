"""
Training pipeline for Physics Worlds Experiment 1
"""

import os
import sys
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import argparse
from tqdm import tqdm
import wandb

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

import keras
from keras import callbacks
from models.core.distribution_inventor import (
    create_distribution_inventor, 
    DistributionInventorConfig,
    PhysicsRuleConfig,
    ModifierConfig,
    TrajectoryConfig
)
from experiments.physics_worlds.data_generator import load_physics_dataset
from experiments.physics_worlds.physics_env import PhysicsWorld, PhysicsConfig


class PhysicsTrainingPipeline:
    """Complete training pipeline for physics world distribution invention"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.model = None
        self.train_data = None
        self.val_data = None
        self.modification_data = None
        
    def _load_config(self, config_path: Optional[str]) -> DistributionInventorConfig:
        """Load training configuration"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            # Convert dict to config objects (simplified)
            return DistributionInventorConfig()
        else:
            # Default configuration
            return DistributionInventorConfig(
                rule_extractor_config=PhysicsRuleConfig(
                    sequence_length=200,
                    rule_embedding_dim=64,
                    num_attention_heads=8,
                    num_transformer_layers=4,
                    learning_rate=1e-4
                ),
                modifier_config=ModifierConfig(
                    rule_embedding_dim=64,
                    modification_embedding_dim=32,
                    hidden_dim=256,
                    learning_rate=1e-4
                ),
                trajectory_config=TrajectoryConfig(
                    sequence_length=200,
                    latent_dim=128,
                    hidden_dim=256,
                    learning_rate=1e-4
                )
            )
    
    def load_data(self, data_dir: str = "data/processed/physics_worlds"):
        """Load training, validation, and modification data"""
        print("Loading training data...")
        
        data_path = Path(data_dir)
        
        # Load main datasets
        train_file = data_path / "train_data.pkl"
        val_file = data_path / "val_data.pkl"
        mod_file = data_path / "modification_pairs.pkl"
        
        if not train_file.exists():
            raise FileNotFoundError(f"Training data not found at {train_file}")
        
        self.train_data = load_physics_dataset(str(train_file))
        
        if val_file.exists():
            self.val_data = load_physics_dataset(str(val_file))
        else:
            print("Warning: No validation data found, using train data subset")
            self.val_data = self.train_data[:len(self.train_data)//10]
        
        if mod_file.exists():
            self.modification_data = load_physics_dataset(str(mod_file))
        else:
            print("Warning: No modification pairs found")
            self.modification_data = []
        
        print(f"Loaded {len(self.train_data)} training samples")
        print(f"Loaded {len(self.val_data)} validation samples") 
        print(f"Loaded {len(self.modification_data)} modification pairs")
    
    def preprocess_data(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Preprocess data for training"""
        print("Preprocessing data...")
        
        def extract_features(samples):
            trajectories = []
            physics_rules = []
            initial_conditions = []
            
            for sample in samples:
                traj = np.array(sample['trajectory'])
                
                # Pad or truncate trajectory
                target_length = self.config.rule_extractor_config.sequence_length
                if len(traj) > target_length:
                    traj = traj[:target_length]
                elif len(traj) < target_length:
                    padding = np.zeros((target_length - len(traj), traj.shape[1]))
                    traj = np.concatenate([traj, padding], axis=0)
                
                trajectories.append(traj)
                
                # Extract physics parameters
                physics_config = sample['physics_config']
                physics_rules.append([
                    physics_config['gravity'],
                    physics_config['friction'],
                    physics_config['elasticity'],
                    physics_config['damping']
                ])
                
                # Extract initial conditions (first frame, first few balls)
                if len(traj) > 0:
                    max_balls = self.config.trajectory_config.max_balls
                    initial_frame = traj[0]
                    
                    # Reshape to per-ball features if needed
                    # This is simplified - real implementation would properly parse ball data
                    num_features_per_ball = 9
                    num_balls = min(max_balls, len(initial_frame) // num_features_per_ball)
                    
                    if num_balls > 0:
                        initial = initial_frame[:num_balls * num_features_per_ball]
                        initial = initial.reshape(num_balls, num_features_per_ball)
                        
                        # Pad to max_balls
                        if num_balls < max_balls:
                            padding = np.zeros((max_balls - num_balls, num_features_per_ball))
                            initial = np.concatenate([initial, padding], axis=0)
                    else:
                        initial = np.zeros((max_balls, num_features_per_ball))
                else:
                    initial = np.zeros((max_balls, num_features_per_ball))
                
                initial_conditions.append(initial)
            
            return {
                'trajectories': np.array(trajectories),
                'physics_rules': np.array(physics_rules),
                'initial_conditions': np.array(initial_conditions)
            }
        
        train_features = extract_features(self.train_data)
        val_features = extract_features(self.val_data)
        
        print(f"Preprocessed data shapes:")
        print(f"  Train trajectories: {train_features['trajectories'].shape}")
        print(f"  Train physics rules: {train_features['physics_rules'].shape}")
        print(f"  Train initial conditions: {train_features['initial_conditions'].shape}")
        
        return train_features, val_features
    
    def create_model(self):
        """Create the distribution inventor model"""
        print("Creating model...")
        self.model = create_distribution_inventor(self.config)
        print(f"Model created with {self.model.count_params():,} parameters")
    
    def create_data_generators(self, train_features, val_features, batch_size: int = 16):
        """Create data generators for training"""
        
        def data_generator(features, modification_data, batch_size, shuffle=True):
            """Generator that yields batches of training data"""
            n_samples = len(features['trajectories'])
            indices = np.arange(n_samples)
            
            while True:
                if shuffle:
                    np.random.shuffle(indices)
                
                for start_idx in range(0, n_samples, batch_size):
                    end_idx = min(start_idx + batch_size, n_samples)
                    batch_indices = indices[start_idx:end_idx]
                    
                    # Get batch data
                    batch_trajectories = features['trajectories'][batch_indices]
                    batch_physics = features['physics_rules'][batch_indices]
                    batch_initial = features['initial_conditions'][batch_indices]
                    
                    # Create modification requests (simplified)
                    batch_size_actual = len(batch_indices)
                    modification_requests = np.random.randint(0, 100, (batch_size_actual, 20))
                    
                    # Prepare inputs
                    inputs = {
                        'trajectory_data': batch_trajectories,
                        'modification_request': modification_requests,
                        'initial_conditions': batch_initial
                    }
                    
                    # Prepare targets (simplified - real targets would be more complex)
                    targets = {
                        'true_rules': batch_physics[:, 0:1],  # Just gravity for now
                        'target_trajectory': batch_trajectories  # Use original as target
                    }
                    
                    yield inputs, targets
        
        train_gen = data_generator(train_features, self.modification_data, batch_size, shuffle=True)
        val_gen = data_generator(val_features, [], batch_size, shuffle=False)
        
        return train_gen, val_gen
    
    def setup_callbacks(self, experiment_name: str):
        """Setup training callbacks"""
        callbacks_list = []
        
        # Model checkpointing
        checkpoint_path = f"outputs/checkpoints/{experiment_name}_best.keras"
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
        
        # TensorBoard logging
        log_dir = f"outputs/logs/{experiment_name}"
        os.makedirs(log_dir, exist_ok=True)
        
        tensorboard_callback = callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
        callbacks_list.append(tensorboard_callback)
        
        return callbacks_list
    
    def train(self, 
              epochs: int = 100,
              batch_size: int = 16,
              experiment_name: str = "physics_experiment_1",
              use_wandb: bool = True):
        """Main training loop"""
        
        if use_wandb:
            wandb.init(
                project="distribution-invention",
                name=experiment_name,
                config={
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'model_config': self.config.__dict__
                }
            )
        
        # Preprocess data
        train_features, val_features = self.preprocess_data()
        
        # Create model
        self.create_model()
        
        # Create data generators
        train_gen, val_gen = self.create_data_generators(train_features, val_features, batch_size)
        
        # Setup callbacks
        callbacks_list = self.setup_callbacks(experiment_name)
        
        if use_wandb:
            wandb_callback = callbacks.WandbCallback(
                monitor='val_loss',
                save_model=False
            )
            callbacks_list.append(wandb_callback)
        
        # Calculate steps per epoch
        steps_per_epoch = len(train_features['trajectories']) // batch_size
        validation_steps = len(val_features['trajectories']) // batch_size
        
        print(f"Starting training...")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Steps per epoch: {steps_per_epoch}")
        print(f"  Validation steps: {validation_steps}")
        
        # Train the model
        history = self.model.fit(
            train_gen,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_gen,
            validation_steps=validation_steps,
            callbacks=callbacks_list,
            verbose=1
        )
        
        print("Training completed!")
        
        if use_wandb:
            wandb.finish()
        
        return history
    
    def evaluate(self, test_data_path: Optional[str] = None):
        """Evaluate the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        print("Evaluating model...")
        
        # Load test data
        if test_data_path and os.path.exists(test_data_path):
            test_data = load_physics_dataset(test_data_path)
        else:
            print("Using validation data for evaluation")
            test_data = self.val_data
        
        # Test distribution invention capabilities
        results = []
        
        modification_requests = [
            "increase gravity by 20%",
            "decrease friction to almost zero",
            "make objects more bouncy",
            "reduce air resistance"
        ]
        
        for i, sample in enumerate(test_data[:10]):  # Test on first 10 samples
            trajectory = np.array(sample['trajectory'])
            
            # Create initial conditions from sample
            if len(trajectory) > 0:
                initial_conditions = np.random.random((2, 9))  # 2 balls, 9 features
            else:
                continue
            
            for request in modification_requests:
                try:
                    result = self.model.invent_distribution(
                        trajectory,
                        request,
                        initial_conditions
                    )
                    
                    results.append({
                        'sample_id': i,
                        'modification': request,
                        'success': result['success'],
                        'quality_scores': result.get('quality_scores', {}),
                        'trajectory_shape': result['new_trajectory'].shape
                    })
                    
                except Exception as e:
                    results.append({
                        'sample_id': i,
                        'modification': request,
                        'success': False,
                        'error': str(e)
                    })
        
        # Analyze results
        successful_results = [r for r in results if r['success']]
        success_rate = len(successful_results) / len(results)
        
        print(f"\nEvaluation Results:")
        print(f"  Total tests: {len(results)}")
        print(f"  Successful: {len(successful_results)}")
        print(f"  Success rate: {success_rate:.2%}")
        
        if successful_results:
            avg_quality = {}
            for key in successful_results[0]['quality_scores'].keys():
                scores = [r['quality_scores'][key] for r in successful_results]
                avg_quality[key] = np.mean(scores)
            
            print(f"  Average quality scores:")
            for key, value in avg_quality.items():
                print(f"    {key}: {value:.4f}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Train Physics Distribution Inventor')
    parser.add_argument('--data_dir', type=str, default='data/processed/physics_worlds',
                       help='Directory containing training data')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--experiment_name', type=str, default='physics_experiment_1',
                       help='Name for this experiment')
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable Weights & Biases logging')
    parser.add_argument('--evaluate_only', action='store_true',
                       help='Only run evaluation on pre-trained model')
    
    args = parser.parse_args()
    
    # Create training pipeline
    pipeline = PhysicsTrainingPipeline(args.config)
    
    # Load data
    pipeline.load_data(args.data_dir)
    
    if args.evaluate_only:
        # Load pre-trained model
        model_path = f"outputs/checkpoints/{args.experiment_name}_best.keras"
        if os.path.exists(model_path):
            pipeline.model = keras.models.load_model(model_path)
            pipeline.evaluate()
        else:
            print(f"No pre-trained model found at {model_path}")
    else:
        # Train model
        history = pipeline.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            experiment_name=args.experiment_name,
            use_wandb=not args.no_wandb
        )
        
        # Evaluate after training
        pipeline.evaluate()


if __name__ == "__main__":
    main()