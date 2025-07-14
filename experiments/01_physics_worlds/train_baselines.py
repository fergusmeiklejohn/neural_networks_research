"""
Train All Baseline Models on Physics Worlds Data

This script trains all 4 baseline models on the physics extrapolation task
and saves their performance metrics for comparison with our approach.
"""

import os
os.environ['KERAS_BACKEND'] = 'jax'

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import keras
from typing import Dict, Tuple, List

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.baseline_models import (
    ERMWithAugmentation,
    GFlowNetBaseline,
    GraphExtrapolationBaseline,
    MAMLBaseline
)
from models.unified_evaluation import UnifiedEvaluator, RepresentationSpaceAnalyzer


class PhysicsBaselineTrainer:
    """Trains and evaluates all baselines on physics data."""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.results_dir = Path("./outputs/baseline_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load physics data
        self.load_physics_data()
        
        # Initialize representation analyzer
        self.rep_analyzer = RepresentationSpaceAnalyzer()
        
    def load_physics_data(self):
        """Load preprocessed physics world data."""
        print("Loading physics data...")
        
        # Load train/val/test splits
        train_file = self.data_dir / "physics_train.npz"
        val_file = self.data_dir / "physics_val.npz"
        test_file = self.data_dir / "physics_test.npz"
        
        if not train_file.exists():
            print("Physics data not found. Generating synthetic data for testing...")
            self.generate_synthetic_physics_data()
            
        train_data = np.load(train_file)
        val_data = np.load(val_file)
        test_data = np.load(test_file)
        
        self.train_data = {
            'x': train_data['states'],
            'y': train_data['next_states'],
            'params': train_data.get('params', None)
        }
        
        self.val_data = {
            'x': val_data['states'],
            'y': val_data['next_states'],
            'params': val_data.get('params', None)
        }
        
        self.test_data = {
            'x': test_data['states'],
            'y': test_data['next_states'],
            'params': test_data.get('params', None),
            'categories': test_data.get('categories', None)  # OOD labels
        }
        
        print(f"Loaded train: {len(self.train_data['x'])}, "
              f"val: {len(self.val_data['x'])}, "
              f"test: {len(self.test_data['x'])} samples")
    
    def generate_synthetic_physics_data(self):
        """Generate synthetic physics data for testing."""
        print("Generating synthetic physics data...")
        
        # Parameters for different physics worlds
        gravity_values = [9.8, 5.0, 15.0, 2.0, 20.0]  # Earth, low, high, moon, extreme
        friction_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        def generate_trajectory(gravity, friction, n_steps=10):
            """Simple ballistic trajectory."""
            # Initial conditions
            x, y = 0.0, 10.0
            vx, vy = 5.0, 0.0
            
            trajectory = []
            for _ in range(n_steps):
                # Update position
                x += vx
                y += vy
                
                # Update velocity (simple physics)
                vy -= gravity * 0.1  # dt = 0.1
                vx *= (1 - friction * 0.01)
                
                trajectory.append([x, y, vx, vy])
                
            return np.array(trajectory)
        
        # Generate training data (in-distribution)
        train_states = []
        train_next_states = []
        train_params = []
        
        for _ in range(1000):
            g = np.random.choice(gravity_values[:3])  # Only Earth, low, high
            f = np.random.choice(friction_values[:3])
            
            traj = generate_trajectory(g, f)
            train_states.append(traj[:-1])
            train_next_states.append(traj[1:])
            train_params.extend([[g, f]] * (len(traj) - 1))
            
        train_states = np.concatenate(train_states)
        train_next_states = np.concatenate(train_next_states)
        train_params = np.array(train_params)
        
        # Generate validation data (similar distribution)
        val_states = []
        val_next_states = []
        val_params = []
        
        for _ in range(200):
            g = np.random.choice(gravity_values[:3])
            f = np.random.choice(friction_values[:3])
            
            traj = generate_trajectory(g, f)
            val_states.append(traj[:-1])
            val_next_states.append(traj[1:])
            val_params.extend([[g, f]] * (len(traj) - 1))
            
        val_states = np.concatenate(val_states)
        val_next_states = np.concatenate(val_next_states)
        val_params = np.array(val_params)
        
        # Generate test data (with OOD samples)
        test_states = []
        test_next_states = []
        test_params = []
        test_categories = []
        
        # In-distribution test
        for _ in range(100):
            g = np.random.choice(gravity_values[:3])
            f = np.random.choice(friction_values[:3])
            
            traj = generate_trajectory(g, f)
            test_states.append(traj[:-1])
            test_next_states.append(traj[1:])
            test_params.extend([[g, f]] * (len(traj) - 1))
            test_categories.extend(['interpolation'] * (len(traj) - 1))
            
        # Near-extrapolation (slightly outside training)
        for _ in range(100):
            g = gravity_values[3]  # Moon gravity
            f = np.random.choice(friction_values[:3])
            
            traj = generate_trajectory(g, f)
            test_states.append(traj[:-1])
            test_next_states.append(traj[1:])
            test_params.extend([[g, f]] * (len(traj) - 1))
            test_categories.extend(['near_extrapolation'] * (len(traj) - 1))
            
        # Far-extrapolation (extreme values)
        for _ in range(100):
            g = gravity_values[4]  # Extreme gravity
            f = friction_values[4]  # Extreme friction
            
            traj = generate_trajectory(g, f)
            test_states.append(traj[:-1])
            test_next_states.append(traj[1:])
            test_params.extend([[g, f]] * (len(traj) - 1))
            test_categories.extend(['far_extrapolation'] * (len(traj) - 1))
            
        test_states = np.concatenate(test_states)
        test_next_states = np.concatenate(test_next_states)
        test_params = np.array(test_params)
        test_categories = np.array(test_categories)
        
        # Save data
        os.makedirs(self.data_dir, exist_ok=True)
        
        np.savez(self.data_dir / "physics_train.npz",
                 states=train_states,
                 next_states=train_next_states,
                 params=train_params)
        
        np.savez(self.data_dir / "physics_val.npz",
                 states=val_states,
                 next_states=val_next_states,
                 params=val_params)
        
        np.savez(self.data_dir / "physics_test.npz",
                 states=test_states,
                 next_states=test_next_states,
                 params=test_params,
                 categories=test_categories)
        
        print("Synthetic data generated and saved.")
    
    def create_baseline_configs(self) -> Dict[str, Dict]:
        """Create configuration for each baseline."""
        input_shape = self.train_data['x'].shape[1:]
        output_shape = self.train_data['y'].shape[1:]
        
        configs = {
            'erm': {
                'task_type': 'physics',
                'input_shape': input_shape,
                'output_shape': output_shape,
                'augmentation_strategies': ['physics_noise', 'physics_interpolation']
            },
            'gflownet': {
                'task_type': 'physics',
                'state_shape': input_shape,
                'action_space': np.prod(output_shape),
                'flow_steps': 10,
                'exploration_bonus': 0.1
            },
            'graph': {
                'task_type': 'physics',
                'input_shape': input_shape,
                'output_shape': output_shape,
                'n_nodes': 100,
                'edge_threshold': 0.5
            },
            'maml': {
                'task_type': 'physics',
                'input_shape': input_shape,
                'output_shape': output_shape,
                'inner_lr': 0.01,
                'outer_lr': 0.001,
                'inner_steps': 5
            }
        }
        
        return configs
    
    def train_baseline(self, baseline_name: str, baseline_class, config: Dict) -> Dict:
        """Train a single baseline and return results."""
        print(f"\n{'='*50}")
        print(f"Training {baseline_name} baseline...")
        print(f"{'='*50}")
        
        # Initialize baseline
        baseline = baseline_class(config)
        baseline.build_model()
        
        # Train
        start_time = datetime.now()
        history = baseline.train(self.train_data, self.val_data, epochs=50)
        train_time = (datetime.now() - start_time).total_seconds()
        
        # Evaluate on test set
        print(f"\nEvaluating {baseline_name}...")
        
        # Standard evaluation
        predictions = baseline.predict(self.test_data['x'])
        mse = np.mean((predictions - self.test_data['y'])**2)
        mae = np.mean(np.abs(predictions - self.test_data['y']))
        
        # Extrapolation evaluation
        if self.test_data['categories'] is not None:
            results_by_category = {}
            for category in np.unique(self.test_data['categories']):
                mask = self.test_data['categories'] == category
                cat_pred = predictions[mask]
                cat_true = self.test_data['y'][mask]
                
                cat_mse = np.mean((cat_pred - cat_true)**2)
                cat_mae = np.mean(np.abs(cat_pred - cat_true))
                
                results_by_category[category] = {
                    'mse': float(cat_mse),
                    'mae': float(cat_mae),
                    'count': int(np.sum(mask))
                }
        else:
            results_by_category = {}
        
        # Test modification capability
        modification_score = self.test_modification_capability(baseline)
        
        # Compile results
        results = {
            'baseline': baseline_name,
            'train_time': train_time,
            'overall_metrics': {
                'mse': float(mse),
                'mae': float(mae)
            },
            'extrapolation_metrics': results_by_category,
            'modification_score': modification_score,
            'model_params': baseline.model.count_params() if hasattr(baseline.model, 'count_params') else 0
        }
        
        # Save model
        model_path = self.results_dir / f"{baseline_name}_model.keras"
        if hasattr(baseline.model, 'save'):
            baseline.model.save(model_path)
        
        return results
    
    def test_modification_capability(self, baseline) -> float:
        """Test baseline's ability to handle rule modifications."""
        # Create a simple modification test
        # Change gravity from 9.8 to 2.0 (moon gravity)
        
        # Generate examples with new physics
        moon_examples = []
        for _ in range(10):
            # Initial state
            state = np.array([0.0, 10.0, 5.0, 0.0])  # x, y, vx, vy
            
            # Moon physics (gravity = 2.0)
            next_state = state.copy()
            next_state[0] += state[2]  # x += vx
            next_state[1] += state[3]  # y += vy
            next_state[3] -= 2.0 * 0.1  # vy -= gravity * dt
            
            moon_examples.append((state, next_state))
        
        moon_x = np.array([ex[0] for ex in moon_examples])
        moon_y = np.array([ex[1] for ex in moon_examples])
        
        # Test baseline's adaptation
        modification_request = "Change gravity to moon gravity (2.0 m/s²)"
        adapted_baseline = baseline.adapt_to_modification(
            modification_request,
            examples={'x': moon_x, 'y': moon_y}
        )
        
        # Evaluate adapted performance
        test_moon_x = moon_x + 0.01 * np.random.randn(*moon_x.shape)
        predictions = adapted_baseline.predict(test_moon_x)
        
        # Compare to true moon physics
        adaptation_error = np.mean((predictions - moon_y)**2)
        
        # Score: inverse of error (capped at 1.0)
        score = 1.0 / (1.0 + adaptation_error)
        
        return float(score)
    
    def train_all_baselines(self):
        """Train all 4 baselines and save results."""
        configs = self.create_baseline_configs()
        
        all_results = {}
        
        # 1. ERM + Augmentation
        results = self.train_baseline(
            'ERM+Aug',
            ERMWithAugmentation,
            configs['erm']
        )
        all_results['erm'] = results
        
        # 2. GFlowNet
        results = self.train_baseline(
            'GFlowNet',
            GFlowNetBaseline,
            configs['gflownet']
        )
        all_results['gflownet'] = results
        
        # 3. Graph Extrapolation
        results = self.train_baseline(
            'GraphExtrap',
            GraphExtrapolationBaseline,
            configs['graph']
        )
        all_results['graph'] = results
        
        # 4. MAML
        results = self.train_baseline(
            'MAML',
            MAMLBaseline,
            configs['maml']
        )
        all_results['maml'] = results
        
        # Save all results
        results_file = self.results_dir / "baseline_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Generate comparison report
        self.generate_comparison_report(all_results)
        
        return all_results
    
    def generate_comparison_report(self, results: Dict):
        """Generate a markdown report comparing all baselines."""
        report = ["# Baseline Models Comparison Report\n"]
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Summary table
        report.append("## Summary Comparison\n")
        report.append("| Baseline | Overall MSE | Interpolation | Near-Extrap | Far-Extrap | Modification Score |")
        report.append("|----------|-------------|---------------|-------------|------------|-------------------|")
        
        for name, res in results.items():
            overall_mse = res['overall_metrics']['mse']
            
            interp = res['extrapolation_metrics'].get('interpolation', {}).get('mse', 'N/A')
            near = res['extrapolation_metrics'].get('near_extrapolation', {}).get('mse', 'N/A')
            far = res['extrapolation_metrics'].get('far_extrapolation', {}).get('mse', 'N/A')
            mod_score = res['modification_score']
            
            report.append(f"| {name} | {overall_mse:.4f} | {interp:.4f} | {near:.4f} | {far:.4f} | {mod_score:.3f} |")
        
        # Detailed results
        report.append("\n## Detailed Results\n")
        
        for name, res in results.items():
            report.append(f"\n### {name}\n")
            report.append(f"- Training time: {res['train_time']:.1f} seconds")
            report.append(f"- Model parameters: {res['model_params']:,}")
            report.append(f"- Overall MSE: {res['overall_metrics']['mse']:.4f}")
            report.append(f"- Overall MAE: {res['overall_metrics']['mae']:.4f}")
            
            if res['extrapolation_metrics']:
                report.append("\nExtrapolation breakdown:")
                for cat, metrics in res['extrapolation_metrics'].items():
                    report.append(f"  - {cat}: MSE={metrics['mse']:.4f}, n={metrics['count']}")
        
        # Key insights
        report.append("\n## Key Insights\n")
        
        # Find best performers
        best_overall = min(results.items(), key=lambda x: x[1]['overall_metrics']['mse'])[0]
        best_mod = max(results.items(), key=lambda x: x[1]['modification_score'])[0]
        
        report.append(f"- Best overall performance: {best_overall}")
        report.append(f"- Best modification capability: {best_mod}")
        
        # Check extrapolation performance
        extrap_scores = []
        for name, res in results.items():
            if 'far_extrapolation' in res['extrapolation_metrics']:
                extrap_scores.append((name, res['extrapolation_metrics']['far_extrapolation']['mse']))
        
        if extrap_scores:
            best_extrap = min(extrap_scores, key=lambda x: x[1])[0]
            report.append(f"- Best far-extrapolation: {best_extrap}")
        
        # Save report
        report_file = self.results_dir / "baseline_comparison_report.md"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"\nReport saved to: {report_file}")
        print("\n" + '\n'.join(report[:20]) + "\n...")  # Print summary


if __name__ == "__main__":
    print("Physics Worlds Baseline Training")
    print("================================\n")
    
    trainer = PhysicsBaselineTrainer()
    results = trainer.train_all_baselines()
    
    print("\n\nTraining complete! Results saved to outputs/baseline_results/")