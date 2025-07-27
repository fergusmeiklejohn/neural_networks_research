"""Evaluate Test-Time Adaptation on true OOD physics scenarios."""

import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
import time
from typing import Dict, List, Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.imports import setup_project_paths
setup_project_paths()

from utils.config import setup_environment
from utils.paths import get_data_path, get_output_path
from models.test_time_adaptation import TENT, PhysicsTENT, PhysicsTTT
from models.test_time_adaptation.tta_wrappers import TTAWrapper
import keras


class TTAEvaluator:
    """Evaluate TTA methods on true OOD physics data."""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.results = {}
        self.models = {}
        
    def load_models(self, model_dir: Path) -> Dict[str, keras.Model]:
        """Load pre-trained baseline models."""
        if self.verbose:
            print("Loading baseline models...")
        
        models = {}
        
        # Find available models
        model_files = {
            'gflownet': list(model_dir.glob("gflownet_model_*.keras")),
            'maml': list(model_dir.glob("maml_model_*.keras")),
            'pinn': list(model_dir.glob("**/model_*.keras")),
            'graph_extrap': list(model_dir.glob("graph_extrap_model_*.keras"))
        }
        
        for name, files in model_files.items():
            if files:
                # Load most recent
                model_path = sorted(files)[-1]
                try:
                    model = keras.models.load_model(model_path, compile=False)
                    model.compile(optimizer='adam', loss='mse')
                    models[name] = model
                    if self.verbose:
                        print(f"  ✓ Loaded {name} from {model_path.name}")
                except Exception as e:
                    if self.verbose:
                        print(f"  ✗ Failed to load {name}: {e}")
        
        self.models = models
        return models
    
    def load_data(self, data_type: str = 'time_varying') -> Tuple[Dict, Dict]:
        """Load constant gravity (baseline) and OOD test data."""
        data_dir = get_data_path() / "true_ood_physics"
        
        # Load constant gravity (for baseline)
        const_files = sorted(data_dir.glob("constant_gravity_*.pkl"))
        if not const_files:
            raise FileNotFoundError("No constant gravity data found")
        
        with open(const_files[-1], 'rb') as f:
            const_data = pickle.load(f)
        
        # Load OOD data
        if data_type == 'time_varying':
            pattern = "time_varying_gravity_*.pkl"
        elif data_type == 'rotating_frame':
            pattern = "rotating_frame_*.pkl"
        elif data_type == 'spring_coupled':
            pattern = "spring_coupled_*.pkl"
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        ood_files = sorted(data_dir.glob(pattern))
        if not ood_files:
            raise FileNotFoundError(f"No {data_type} data found")
        
        with open(ood_files[-1], 'rb') as f:
            ood_data = pickle.load(f)
        
        if self.verbose:
            print(f"\nLoaded data:")
            print(f"  Constant gravity: {const_data['trajectories'].shape}")
            print(f"  {data_type}: {ood_data['trajectories'].shape}")
        
        return const_data, ood_data
    
    def evaluate_trajectory_prediction(self, 
                                     model: keras.Model,
                                     trajectories: np.ndarray,
                                     input_steps: int = 1) -> Dict[str, float]:
        """Evaluate model on trajectory prediction task."""
        errors = []
        
        for traj in trajectories:
            # Use first step(s) to predict rest
            X = traj[:input_steps].reshape(1, input_steps, -1)
            y_true = traj[input_steps:]
            
            # Predict
            y_pred = model.predict(X, verbose=0)
            
            # Handle different output shapes
            if len(y_pred.shape) == 3:
                y_pred = y_pred[0]  # Remove batch dimension
            
            # Ensure same length
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
            
            # Compute error
            mse = np.mean((y_true - y_pred) ** 2)
            errors.append(mse)
        
        return {
            'mse': np.mean(errors),
            'mse_std': np.std(errors),
            'mse_median': np.median(errors),
            'n_samples': len(errors)
        }
    
    def evaluate_with_tta(self,
                         model: keras.Model,
                         trajectories: np.ndarray,
                         tta_method: str,
                         adaptation_steps: int = 5,
                         learning_rate: float = 1e-4) -> Dict[str, float]:
        """Evaluate model with Test-Time Adaptation."""
        # Create TTA wrapper
        tta_kwargs = {
            'adaptation_steps': adaptation_steps,
            'learning_rate': learning_rate
        }
        
        if tta_method == 'physics_tent':
            tta_kwargs['physics_loss_weight'] = 0.1
        elif tta_method == 'ttt':
            tta_kwargs['trajectory_length'] = 50
            tta_kwargs['adaptation_window'] = 10
        
        tta_model = TTAWrapper(model, tta_method=tta_method, **tta_kwargs)
        
        errors = []
        adaptation_times = []
        
        for i, traj in enumerate(trajectories):
            start_time = time.time()
            
            # For TTT, use more context
            if tta_method == 'ttt':
                context_len = min(10, len(traj) // 2)
                X = traj[:context_len].reshape(1, context_len, -1)
                y_true = traj[context_len:]
            else:
                X = traj[:1].reshape(1, 1, -1)
                y_true = traj[1:]
            
            # Predict with adaptation
            y_pred = tta_model.predict(X, adapt=True)
            
            adaptation_time = time.time() - start_time
            adaptation_times.append(adaptation_time)
            
            # Reset after each trajectory (for fair comparison)
            tta_model.reset()
            
            # Compute error
            if len(y_pred.shape) == 3:
                y_pred = y_pred[0]
            
            min_len = min(len(y_true), len(y_pred))
            mse = np.mean((y_true[:min_len] - y_pred[:min_len]) ** 2)
            errors.append(mse)
        
        # Get adaptation metrics
        tta_metrics = tta_model.get_metrics()
        
        return {
            'mse': np.mean(errors),
            'mse_std': np.std(errors),
            'mse_median': np.median(errors),
            'adaptation_time': np.mean(adaptation_times),
            'n_samples': len(errors),
            **tta_metrics
        }
    
    def run_evaluation(self, 
                      model_names: Optional[List[str]] = None,
                      tta_methods: List[str] = ['tent', 'physics_tent', 'ttt'],
                      data_types: List[str] = ['time_varying'],
                      n_samples: int = 100,
                      adaptation_steps: int = 5):
        """Run complete TTA evaluation."""
        if model_names is None:
            model_names = list(self.models.keys())
        
        results = []
        
        for data_type in data_types:
            print(f"\n{'='*60}")
            print(f"Evaluating on {data_type} physics")
            print(f"{'='*60}")
            
            # Load data
            const_data, ood_data = self.load_data(data_type)
            
            # Limit samples
            const_trajectories = const_data['trajectories'][:n_samples]
            ood_trajectories = ood_data['trajectories'][:n_samples]
            
            for model_name in model_names:
                if model_name not in self.models:
                    print(f"\nSkipping {model_name} (not loaded)")
                    continue
                
                model = self.models[model_name]
                print(f"\n{model_name.upper()}:")
                
                # Baseline on constant gravity
                print("  Constant gravity (baseline):")
                const_metrics = self.evaluate_trajectory_prediction(
                    model, const_trajectories
                )
                print(f"    MSE: {const_metrics['mse']:.4f} ± {const_metrics['mse_std']:.4f}")
                
                # No TTA on OOD
                print(f"  {data_type} (no TTA):")
                ood_metrics = self.evaluate_trajectory_prediction(
                    model, ood_trajectories
                )
                print(f"    MSE: {ood_metrics['mse']:.4f} ± {ood_metrics['mse_std']:.4f}")
                degradation = (ood_metrics['mse'] - const_metrics['mse']) / const_metrics['mse'] * 100
                print(f"    Degradation: {degradation:.1f}%")
                
                # Store no-TTA result
                results.append({
                    'model': model_name,
                    'data_type': data_type,
                    'tta_method': 'none',
                    'mse': ood_metrics['mse'],
                    'mse_std': ood_metrics['mse_std'],
                    'baseline_mse': const_metrics['mse'],
                    'degradation': degradation,
                    'adaptation_time': 0
                })
                
                # Test each TTA method
                for tta_method in tta_methods:
                    print(f"  {data_type} (with {tta_method}):")
                    
                    try:
                        tta_metrics = self.evaluate_with_tta(
                            model, ood_trajectories, 
                            tta_method=tta_method,
                            adaptation_steps=adaptation_steps
                        )
                        
                        print(f"    MSE: {tta_metrics['mse']:.4f} ± {tta_metrics['mse_std']:.4f}")
                        improvement = (ood_metrics['mse'] - tta_metrics['mse']) / ood_metrics['mse'] * 100
                        print(f"    Improvement over no-TTA: {improvement:.1f}%")
                        print(f"    Adaptation time: {tta_metrics['adaptation_time']:.3f}s")
                        
                        results.append({
                            'model': model_name,
                            'data_type': data_type,
                            'tta_method': tta_method,
                            'mse': tta_metrics['mse'],
                            'mse_std': tta_metrics['mse_std'],
                            'baseline_mse': const_metrics['mse'],
                            'improvement': improvement,
                            'adaptation_time': tta_metrics['adaptation_time'],
                            'mean_adaptation_loss': tta_metrics.get('mean_adaptation_loss', 0)
                        })
                        
                    except Exception as e:
                        print(f"    Error: {e}")
                        results.append({
                            'model': model_name,
                            'data_type': data_type,
                            'tta_method': tta_method,
                            'error': str(e)
                        })
        
        self.results = pd.DataFrame(results)
        return self.results
    
    def visualize_results(self, save_path: Optional[Path] = None):
        """Create visualizations of TTA results."""
        if self.results.empty:
            print("No results to visualize")
            return
        
        # Filter out errors
        valid_results = self.results[~self.results['mse'].isna()].copy()
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. MSE comparison by model and TTA method
        ax1 = axes[0, 0]
        pivot_mse = valid_results.pivot_table(
            values='mse', 
            index='model', 
            columns='tta_method', 
            aggfunc='mean'
        )
        pivot_mse.plot(kind='bar', ax=ax1)
        ax1.set_title('MSE by Model and TTA Method')
        ax1.set_ylabel('Mean Squared Error')
        ax1.legend(title='TTA Method')
        ax1.grid(True, alpha=0.3)
        
        # 2. Improvement over no-TTA
        ax2 = axes[0, 1]
        improvement_data = valid_results[valid_results['tta_method'] != 'none'].copy()
        if 'improvement' in improvement_data.columns:
            pivot_imp = improvement_data.pivot_table(
                values='improvement',
                index='model',
                columns='tta_method',
                aggfunc='mean'
            )
            pivot_imp.plot(kind='bar', ax=ax2)
            ax2.set_title('Improvement over No-TTA (%)')
            ax2.set_ylabel('Improvement (%)')
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax2.legend(title='TTA Method')
            ax2.grid(True, alpha=0.3)
        
        # 3. Adaptation time
        ax3 = axes[1, 0]
        time_data = valid_results[valid_results['adaptation_time'] > 0]
        if not time_data.empty:
            pivot_time = time_data.pivot_table(
                values='adaptation_time',
                index='model',
                columns='tta_method',
                aggfunc='mean'
            )
            pivot_time.plot(kind='bar', ax=ax3)
            ax3.set_title('Average Adaptation Time per Sample')
            ax3.set_ylabel('Time (seconds)')
            ax3.legend(title='TTA Method')
            ax3.grid(True, alpha=0.3)
        
        # 4. Summary statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create summary text
        summary_text = "Summary Statistics\n" + "="*30 + "\n\n"
        
        # Best performing combination
        best_row = valid_results.loc[valid_results['mse'].idxmin()]
        summary_text += f"Best Performance:\n"
        summary_text += f"  Model: {best_row['model']}\n"
        summary_text += f"  TTA: {best_row['tta_method']}\n"
        summary_text += f"  MSE: {best_row['mse']:.4f}\n\n"
        
        # Average improvement by TTA method
        avg_improvement = improvement_data.groupby('tta_method')['improvement'].mean()
        summary_text += "Average Improvement by TTA:\n"
        for method, imp in avg_improvement.items():
            summary_text += f"  {method}: {imp:.1f}%\n"
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
                verticalalignment='top', fontfamily='monospace', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def save_results(self, output_dir: Optional[Path] = None):
        """Save evaluation results."""
        if output_dir is None:
            output_dir = get_output_path() / "tta_evaluation"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw results
        results_file = output_dir / f"tta_results_{timestamp}.csv"
        self.results.to_csv(results_file, index=False)
        
        # Save summary
        summary_file = output_dir / f"tta_summary_{timestamp}.json"
        summary = {
            'timestamp': timestamp,
            'n_models': len(self.results['model'].unique()),
            'n_tta_methods': len(self.results['tta_method'].unique()),
            'best_result': self.results.loc[self.results['mse'].idxmin()].to_dict() if not self.results.empty else None,
            'average_improvements': self.results[self.results['tta_method'] != 'none'].groupby('tta_method')['improvement'].mean().to_dict() if 'improvement' in self.results.columns else {}
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save visualization
        viz_file = output_dir / f"tta_visualization_{timestamp}.png"
        self.visualize_results(save_path=viz_file)
        
        if self.verbose:
            print(f"\nResults saved to:")
            print(f"  CSV: {results_file}")
            print(f"  Summary: {summary_file}")
            print(f"  Visualization: {viz_file}")


def main():
    """Run TTA evaluation on true OOD physics data."""
    print("Test-Time Adaptation Evaluation on True OOD Physics")
    print("="*60)
    
    # Setup
    config = setup_environment()
    
    # Create evaluator
    evaluator = TTAEvaluator(verbose=True)
    
    # Load models
    model_dir = Path(__file__).parent / "outputs"
    models = evaluator.load_models(model_dir)
    
    if not models:
        print("No models found! Please train baseline models first.")
        return
    
    # Run evaluation
    results = evaluator.run_evaluation(
        model_names=None,  # Use all available models
        tta_methods=['tent', 'physics_tent'],  # Skip TTT for quick test
        data_types=['time_varying'],
        n_samples=50,  # Reduced for faster testing
        adaptation_steps=5
    )
    
    # Save results
    evaluator.save_results()
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)
    
    # Print summary
    if not results.empty:
        print("\nTop 5 Results (by MSE):")
        print(results.nsmallest(5, 'mse')[['model', 'tta_method', 'mse', 'improvement']])


if __name__ == "__main__":
    main()