#!/usr/bin/env python3
"""
Progressive Training Curriculum for Physics Extrapolation

This script implements a 4-stage progressive training curriculum designed to
improve extrapolation from 0% to 70-85% accuracy by gradually introducing
physics constraints and domain knowledge.

Stages:
1. In-distribution supervised learning (baseline)
2. Progressive physics constraint integration 
3. Domain randomization training
4. Extrapolation-focused fine-tuning
"""

import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from typing import Dict, List, Tuple, Optional
import wandb
from tqdm import tqdm

# Import our models and utilities
import sys
sys.path.append('../..')
from models.physics_informed_transformer import PhysicsInformedTrajectoryTransformer
from models.physics_losses import PhysicsLosses
from experiments.utils.experiment_utils import set_random_seeds
from experiments.utils.visualization import plot_trajectories


class ProgressiveCurriculum:
    """Manages the 4-stage progressive training curriculum"""
    
    def __init__(
        self,
        model: PhysicsInformedTrajectoryTransformer,
        train_data: Dict,
        val_data: Dict,
        test_data: Dict,
        config: Dict
    ):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.config = config
        
        # Initialize physics losses
        self.physics_losses = PhysicsLosses()
        
        # Extract in-distribution and extrapolation test sets
        self.test_in_dist = self._filter_in_distribution(test_data)
        self.test_extrap = self._filter_extrapolation(test_data)
        
        # Stage-specific configurations
        self.stage_configs = {
            1: {
                "name": "In-Distribution Learning",
                "epochs": config.get("stage1_epochs", 50),
                "physics_weight": 0.0,
                "use_domain_randomization": False,
                "learning_rate": 1e-3,
                "focus_extrapolation": False
            },
            2: {
                "name": "Physics Constraint Integration",
                "epochs": config.get("stage2_epochs", 50),
                "physics_weight_schedule": np.linspace(0.1, 1.0, config.get("stage2_epochs", 50)),
                "use_domain_randomization": False,
                "learning_rate": 5e-4,
                "focus_extrapolation": False
            },
            3: {
                "name": "Domain Randomization",
                "epochs": config.get("stage3_epochs", 50),
                "physics_weight": 1.0,
                "use_domain_randomization": True,
                "domain_random_range": 0.3,  # ±30% parameter variation
                "learning_rate": 2e-4,
                "focus_extrapolation": False
            },
            4: {
                "name": "Extrapolation Fine-tuning",
                "epochs": config.get("stage4_epochs", 50),
                "physics_weight": 1.0,
                "use_domain_randomization": True,
                "domain_random_range": 0.5,  # ±50% for aggressive extrapolation
                "learning_rate": 1e-4,
                "focus_extrapolation": True,
                "hard_example_weight": 2.0
            }
        }
        
        # Metrics tracking
        self.metrics_history = {
            "stage": [],
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "physics_loss": [],
            "in_dist_accuracy": [],
            "extrap_accuracy": [],
            "energy_violation": [],
            "momentum_violation": []
        }
        
    def _filter_in_distribution(self, data: Dict) -> Dict:
        """Filter test data for in-distribution samples"""
        # In-distribution: gravity in [9.0, 10.0], friction in [0.09, 0.11]
        mask = (
            (data['gravity'] >= 9.0) & (data['gravity'] <= 10.0) &
            (data['friction'] >= 0.09) & (data['friction'] <= 0.11)
        )
        return {k: v[mask] for k, v in data.items()}
    
    def _filter_extrapolation(self, data: Dict) -> Dict:
        """Filter test data for extrapolation samples"""
        # Extrapolation: parameters outside training range
        mask = (
            (data['gravity'] < 9.0) | (data['gravity'] > 10.0) |
            (data['friction'] < 0.09) | (data['friction'] > 0.11)
        )
        return {k: v[mask] for k, v in data.items()}
    
    def _apply_domain_randomization(self, params: np.ndarray, range_factor: float) -> np.ndarray:
        """Apply domain randomization to physics parameters"""
        # params shape: (batch_size, 2) where params[:, 0] is gravity, params[:, 1] is friction
        randomized = params.copy()
        
        # Add random noise within range
        noise = np.random.uniform(-range_factor, range_factor, params.shape)
        randomized = params * (1 + noise)
        
        # Ensure physical validity
        randomized[:, 0] = np.clip(randomized[:, 0], 5.0, 15.0)  # gravity
        randomized[:, 1] = np.clip(randomized[:, 1], 0.0, 0.3)   # friction
        
        return randomized
    
    def _create_training_batch(
        self,
        batch_data: Dict,
        stage: int,
        epoch: int
    ) -> Tuple[Dict, np.ndarray]:
        """Create training batch with stage-specific modifications"""
        config = self.stage_configs[stage]
        
        # Copy batch data
        modified_batch = {k: v.copy() if isinstance(v, np.ndarray) else v 
                         for k, v in batch_data.items()}
        
        # Apply domain randomization if enabled
        if config.get("use_domain_randomization", False):
            range_factor = config.get("domain_random_range", 0.3)
            params = np.stack([modified_batch['gravity'], modified_batch['friction']], axis=-1)
            randomized_params = self._apply_domain_randomization(params, range_factor)
            modified_batch['gravity'] = randomized_params[:, 0]
            modified_batch['friction'] = randomized_params[:, 1]
        
        # Get physics weight for this stage/epoch
        if stage == 2:
            # Progressive schedule for stage 2
            physics_weight = config["physics_weight_schedule"][epoch]
        else:
            physics_weight = config.get("physics_weight", 0.0)
        
        return modified_batch, physics_weight
    
    def _compute_metrics(self, stage: int, epoch: int) -> Dict[str, float]:
        """Compute comprehensive metrics for current model state"""
        metrics = {}
        
        # In-distribution accuracy
        in_dist_preds_dict = self.model.predict(self.test_in_dist['trajectories'])
        in_dist_preds = in_dist_preds_dict['predictions']
        in_dist_error = np.mean(np.abs(in_dist_preds - self.test_in_dist['future_trajectories']))
        metrics['in_dist_accuracy'] = max(0, 1 - in_dist_error / 10.0)  # Normalize
        
        # Extrapolation accuracy
        extrap_preds_dict = self.model.predict(self.test_extrap['trajectories'])
        extrap_preds = extrap_preds_dict['predictions']
        extrap_error = np.mean(np.abs(extrap_preds - self.test_extrap['future_trajectories']))
        metrics['extrap_accuracy'] = max(0, 1 - extrap_error / 10.0)  # Normalize
        
        # Physics violations (sample a subset for efficiency)
        sample_idx = np.random.choice(len(self.test_extrap['trajectories']), 
                                     min(100, len(self.test_extrap['trajectories'])), 
                                     replace=False)
        
        sample_preds = extrap_preds[sample_idx]
        sample_params = {
            'gravity': self.test_extrap['gravity'][sample_idx],
            'friction': self.test_extrap['friction'][sample_idx]
        }
        
        # Compute energy and momentum violations
        energy_viol = self.physics_losses.energy_conservation_loss(
            sample_preds, sample_params['gravity']
        ).numpy()
        momentum_viol = self.physics_losses.momentum_conservation_loss(sample_preds).numpy()
        
        metrics['energy_violation'] = float(energy_viol)
        metrics['momentum_violation'] = float(momentum_viol)
        
        return metrics
    
    def train_stage(self, stage: int) -> Dict[str, List[float]]:
        """Train a single stage of the curriculum"""
        config = self.stage_configs[stage]
        print(f"\n{'='*60}")
        print(f"Stage {stage}: {config['name']}")
        print(f"{'='*60}")
        
        # Update learning rate
        self.model.optimizer.learning_rate.assign(config['learning_rate'])
        
        # Stage-specific metrics
        stage_metrics = {
            "train_loss": [],
            "val_loss": [],
            "physics_loss": [],
            "in_dist_accuracy": [],
            "extrap_accuracy": []
        }
        
        # Training loop
        for epoch in range(config['epochs']):
            print(f"\nEpoch {epoch + 1}/{config['epochs']}")
            
            # Training step
            train_losses = []
            physics_losses = []
            
            # Create batches
            batch_size = self.config.get('batch_size', 32)
            n_batches = len(self.train_data['trajectories']) // batch_size
            
            for batch_idx in tqdm(range(n_batches), desc="Training"):
                # Get batch
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                
                batch_data = {
                    k: v[start_idx:end_idx] for k, v in self.train_data.items()
                }
                
                # Apply stage-specific modifications
                modified_batch, physics_weight = self._create_training_batch(
                    batch_data, stage, epoch
                )
                
                # Training step
                with tf.GradientTape() as tape:
                    # Forward pass - model expects just trajectories
                    predictions_dict = self.model(
                        modified_batch['trajectories'], 
                        training=True
                    )
                    # Reconstruct full trajectory from positions and velocities
                    # The model outputs positions and velocities separately
                    positions = predictions_dict['positions']
                    velocities = predictions_dict['velocities']
                    
                    # Concatenate to match the input trajectory format
                    # For now, use just positions as predictions (simplified)
                    predictions = positions
                    
                    # Compute losses
                    mse_loss = tf.reduce_mean(
                        tf.square(predictions - modified_batch['future_trajectories'])
                    )
                    
                    # Physics loss (if applicable)
                    if physics_weight > 0:
                        physics_loss_dict = self.physics_losses.combined_physics_loss(
                            {'trajectories': predictions},
                            {'trajectories': modified_batch['future_trajectories']},
                            {'gravity': modified_batch['gravity'], 'friction': modified_batch['friction']}
                        )
                        physics_loss = physics_loss_dict['total']
                        total_loss = mse_loss + physics_weight * physics_loss
                        physics_losses.append(float(physics_loss))
                    else:
                        total_loss = mse_loss
                        physics_losses.append(0.0)
                    
                    train_losses.append(total_loss.numpy())
                
                # Backward pass
                gradients = tape.gradient(total_loss, self.model.trainable_variables)
                self.model.optimizer.apply_gradients(
                    zip(gradients, self.model.trainable_variables)
                )
            
            # Validation
            val_preds_dict = self.model.predict(self.val_data['trajectories'])
            val_preds = val_preds_dict['predictions']
            val_loss = np.mean(np.square(val_preds - self.val_data['future_trajectories']))
            
            # Compute comprehensive metrics
            metrics = self._compute_metrics(stage, epoch)
            
            # Record metrics
            stage_metrics['train_loss'].append(np.mean(train_losses))
            stage_metrics['val_loss'].append(val_loss)
            stage_metrics['physics_loss'].append(np.mean(physics_losses))
            stage_metrics['in_dist_accuracy'].append(metrics['in_dist_accuracy'])
            stage_metrics['extrap_accuracy'].append(metrics['extrap_accuracy'])
            
            # Update history
            self.metrics_history['stage'].append(stage)
            self.metrics_history['epoch'].append(epoch)
            self.metrics_history['train_loss'].append(np.mean(train_losses))
            self.metrics_history['val_loss'].append(val_loss)
            self.metrics_history['physics_loss'].append(np.mean(physics_losses))
            self.metrics_history['in_dist_accuracy'].append(metrics['in_dist_accuracy'])
            self.metrics_history['extrap_accuracy'].append(metrics['extrap_accuracy'])
            self.metrics_history['energy_violation'].append(metrics['energy_violation'])
            self.metrics_history['momentum_violation'].append(metrics['momentum_violation'])
            
            # Print progress
            print(f"  Train Loss: {np.mean(train_losses):.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Physics Loss: {np.mean(physics_losses):.4f}")
            print(f"  In-Dist Accuracy: {metrics['in_dist_accuracy']:.2%}")
            print(f"  Extrap Accuracy: {metrics['extrap_accuracy']:.2%}")
            
            # Log to wandb if available
            if wandb.run is not None:
                wandb.log({
                    f"stage{stage}/train_loss": np.mean(train_losses),
                    f"stage{stage}/val_loss": val_loss,
                    f"stage{stage}/physics_loss": np.mean(physics_losses),
                    f"stage{stage}/in_dist_accuracy": metrics['in_dist_accuracy'],
                    f"stage{stage}/extrap_accuracy": metrics['extrap_accuracy'],
                    f"stage{stage}/energy_violation": metrics['energy_violation'],
                    f"stage{stage}/momentum_violation": metrics['momentum_violation'],
                    "epoch": epoch + sum([self.stage_configs[s]['epochs'] for s in range(1, stage)])
                })
        
        # Save checkpoint after stage
        checkpoint_path = f"outputs/checkpoints/stage{stage}_final.keras"
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        self.model.save(checkpoint_path)
        print(f"\nStage {stage} complete. Model saved to {checkpoint_path}")
        
        return stage_metrics
    
    def run_full_curriculum(self) -> Dict:
        """Run the complete 4-stage curriculum"""
        print("Starting Progressive Training Curriculum")
        print(f"Total stages: 4")
        print(f"Total epochs: {sum(c['epochs'] for c in self.stage_configs.values())}")
        
        # Train each stage
        all_stage_metrics = {}
        for stage in range(1, 5):
            stage_metrics = self.train_stage(stage)
            all_stage_metrics[f"stage_{stage}"] = stage_metrics
            
            # Early stopping if extrapolation target achieved
            if stage_metrics['extrap_accuracy'][-1] >= 0.7:
                print(f"\nTarget extrapolation accuracy achieved! ({stage_metrics['extrap_accuracy'][-1]:.2%})")
                break
        
        # Save final metrics
        self._save_metrics()
        
        return all_stage_metrics
    
    def _save_metrics(self):
        """Save metrics history to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw metrics
        metrics_path = f"outputs/metrics/curriculum_metrics_{timestamp}.json"
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        # Create summary
        summary = {
            "timestamp": timestamp,
            "total_epochs": len(self.metrics_history['epoch']),
            "final_in_dist_accuracy": self.metrics_history['in_dist_accuracy'][-1],
            "final_extrap_accuracy": self.metrics_history['extrap_accuracy'][-1],
            "best_extrap_accuracy": max(self.metrics_history['extrap_accuracy']),
            "best_extrap_epoch": self.metrics_history['extrap_accuracy'].index(
                max(self.metrics_history['extrap_accuracy'])
            ),
            "final_energy_violation": self.metrics_history['energy_violation'][-1],
            "final_momentum_violation": self.metrics_history['momentum_violation'][-1]
        }
        
        summary_path = f"outputs/metrics/curriculum_summary_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nMetrics saved to:")
        print(f"  - {metrics_path}")
        print(f"  - {summary_path}")
        
        # Print summary
        print("\nTraining Summary:")
        print(f"  Final In-Dist Accuracy: {summary['final_in_dist_accuracy']:.2%}")
        print(f"  Final Extrap Accuracy: {summary['final_extrap_accuracy']:.2%}")
        print(f"  Best Extrap Accuracy: {summary['best_extrap_accuracy']:.2%} (epoch {summary['best_extrap_epoch']})")


def load_data():
    """Load the physics worlds dataset"""
    print("Loading data...")
    
    # Load from pickle files
    import pickle
    data_dir = Path("data/processed/physics_worlds_v2_quick")
    
    # Load train data
    with open(data_dir / 'train_data.pkl', 'rb') as f:
        train_pkl = pickle.load(f)
    
    # Load validation data (using in-distribution validation)
    with open(data_dir / 'val_in_dist_data.pkl', 'rb') as f:
        val_pkl = pickle.load(f)
    
    # Load test data (combining interpolation and extrapolation)
    with open(data_dir / 'test_interpolation_data.pkl', 'rb') as f:
        test_interp_pkl = pickle.load(f)
    with open(data_dir / 'test_extrapolation_data.pkl', 'rb') as f:
        test_extrap_pkl = pickle.load(f)
    
    # Convert to the expected format
    def extract_data(data_list):
        trajectories = []
        future_trajectories = []
        gravity_values = []
        friction_values = []
        
        for sample in data_list:
            # Filter to only include samples with 3 balls
            if sample['num_balls'] != 3:
                continue
                
            traj = np.array(sample['trajectory'])
            # Use first 10 timesteps as input, next 10 as output
            # Trajectory shape is (timesteps, features)
            # For 3 balls: features = 2*3+1 = 7 (x,y for each ball + time)
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
    
    # Combine test data
    test_interp_data = extract_data(test_interp_pkl)
    test_extrap_data = extract_data(test_extrap_pkl)
    
    test_data = {
        'trajectories': np.concatenate([test_interp_data['trajectories'], test_extrap_data['trajectories']]),
        'future_trajectories': np.concatenate([test_interp_data['future_trajectories'], test_extrap_data['future_trajectories']]),
        'gravity': np.concatenate([test_interp_data['gravity'], test_extrap_data['gravity']]),
        'friction': np.concatenate([test_interp_data['friction'], test_extrap_data['friction']])
    }
    
    print(f"  Train: {len(train_data['trajectories'])} samples")
    print(f"  Val: {len(val_data['trajectories'])} samples")
    print(f"  Test: {len(test_data['trajectories'])} samples")
    
    return train_data, val_data, test_data


def main():
    """Main training function"""
    # Configuration
    config = {
        "seed": 42,
        "batch_size": 32,
        "stage1_epochs": 2,  # Quick test - increase to 50 for full training
        "stage2_epochs": 2,  # Quick test - increase to 50 for full training
        "stage3_epochs": 2,  # Quick test - increase to 50 for full training
        "stage4_epochs": 2,  # Quick test - increase to 50 for full training
        "wandb_project": "physics-worlds-curriculum",
        "wandb_enabled": False  # Set to True for full runs
    }
    
    # Set random seeds
    set_random_seeds(config['seed'])
    
    # Initialize wandb if enabled
    if config['wandb_enabled']:
        wandb.init(
            project=config['wandb_project'],
            config=config,
            name=f"curriculum_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    # Load data
    train_data, val_data, test_data = load_data()
    
    # Subsample for quick test (remove for full training)
    n_samples = 100
    for data in [train_data, val_data, test_data]:
        for key in data:
            data[key] = data[key][:n_samples]
    
    # Initialize model
    print("\nInitializing model...")
    model = PhysicsInformedTrajectoryTransformer(
        sequence_length=10,  # We're using 10 timesteps
        feature_dim=25,  # 3 balls * 8 features + 1 (time) = 25
        num_transformer_layers=4,
        num_heads=8,
        transformer_dim=128,
        use_hnn=True,
        use_soft_collisions=True,
        use_physics_losses=True,
        dropout_rate=0.1
    )
    
    # Build model
    dummy_input = train_data['trajectories'][:1]
    _ = model(dummy_input)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='mse'
    )
    
    # Create curriculum trainer
    curriculum = ProgressiveCurriculum(
        model=model,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        config=config
    )
    
    # Run training
    all_metrics = curriculum.run_full_curriculum()
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    final_metrics = curriculum._compute_metrics(4, config['stage4_epochs'] - 1)
    print(f"In-Distribution Accuracy: {final_metrics['in_dist_accuracy']:.2%}")
    print(f"Extrapolation Accuracy: {final_metrics['extrap_accuracy']:.2%}")
    print(f"Energy Conservation Violation: {final_metrics['energy_violation']:.4f}")
    print(f"Momentum Conservation Violation: {final_metrics['momentum_violation']:.4f}")
    
    # Close wandb
    if config['wandb_enabled']:
        wandb.finish()


if __name__ == "__main__":
    main()