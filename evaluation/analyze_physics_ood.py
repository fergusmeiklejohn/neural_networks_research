"""
Apply True OOD Verification to Physics Experiment Results

This script analyzes our physics world experiments to determine which
"OOD" successes were actually interpolation in disguise.
"""

from utils.imports import setup_project_paths
setup_project_paths()

from utils.config import setup_environment
from utils.paths import get_data_path, get_output_path
import numpy as np
import keras
from pathlib import Path
import json
import logging
from datetime import datetime

from true_ood_verifier import TrueOODVerifier

# Set up environment
config = setup_environment()
logger = logging.getLogger(__name__)


class PhysicsOODAnalyzer:
    """Analyze physics models for true OOD performance."""
    
    def __init__(self, experiment_dir: Path):
        self.experiment_dir = Path(experiment_dir)
        self.verifier = TrueOODVerifier(
            use_pca=True,
            pca_components=50,
            distance_thresholds={
                'interpolation': 0.05,      # Within hull + 5% margin
                'near_extrapolation': 0.20,  # Within 20% of hull
                'far_extrapolation': 0.20    # Beyond 20% of hull
            }
        )
        
    def load_physics_data(self):
        """Load physics training and test data."""
        data_dir = self.experiment_dir / "data"
        
        # Look for saved data files
        train_file = data_dir / "train_data.npz"
        test_files = {
            'same_gravity': data_dir / "test_same_gravity.npz",
            'diff_gravity': data_dir / "test_diff_gravity.npz",
            'zero_gravity': data_dir / "test_zero_gravity.npz",
            'time_varying': data_dir / "test_time_varying.npz"
        }
        
        if not train_file.exists():
            logger.warning(f"Training data not found at {train_file}")
            return None, None
            
        # Load training data
        train_data = np.load(train_file)
        
        # Load test data
        test_data = {}
        for name, path in test_files.items():
            if path.exists():
                test_data[name] = np.load(path)
                logger.info(f"Loaded {name} test set from {path}")
            else:
                logger.warning(f"Test set {name} not found at {path}")
                
        return train_data, test_data
    
    def analyze_model(self, model_path: Path, model_name: str):
        """Analyze a specific model's OOD behavior."""
        logger.info(f"\nAnalyzing model: {model_name}")
        logger.info(f"Loading from: {model_path}")
        
        try:
            # Load model
            model = keras.models.load_model(model_path)
            
            # Create wrapper to extract representations
            class ModelWrapper:
                def __init__(self, keras_model):
                    self.model = keras_model
                    # Try to find a good intermediate layer
                    self.representation_layer = self._find_representation_layer()
                    
                def _find_representation_layer(self):
                    # Look for last dense layer before output
                    for layer in reversed(self.model.layers):
                        if isinstance(layer, keras.layers.Dense) and layer.units > 10:
                            return layer.name
                    # Fallback to second-to-last layer
                    return self.model.layers[-2].name
                    
                def get_representations(self, data, layer=None):
                    if layer is None:
                        layer = self.representation_layer
                        
                    # Create intermediate model
                    intermediate_model = keras.Model(
                        inputs=self.model.input,
                        outputs=self.model.get_layer(layer).output
                    )
                    
                    # Handle different data formats
                    if isinstance(data, dict):
                        # Assume data has 'positions' or 'states' key
                        if 'positions' in data:
                            inputs = data['positions']
                        elif 'states' in data:
                            inputs = data['states']
                        else:
                            inputs = data[list(data.keys())[0]]
                    else:
                        inputs = data
                        
                    return intermediate_model.predict(inputs, verbose=0)
            
            wrapped_model = ModelWrapper(model)
            
            # Load data
            train_data, test_data = self.load_physics_data()
            
            if train_data is None:
                logger.error("Could not load physics data")
                return None
                
            # Analyze each test scenario
            results = {}
            
            for test_name, test_set in test_data.items():
                logger.info(f"\nAnalyzing {test_name}...")
                
                analysis = self.verifier.analyze_dataset(
                    train_data,
                    test_set,
                    wrapped_model
                )
                
                results[test_name] = analysis
                
                # Save visualization
                viz_path = self.experiment_dir / f"ood_analysis_{model_name}_{test_name}.png"
                self.verifier.visualize_analysis(
                    analysis, 
                    save_path=str(viz_path),
                    show_hull=True,
                    max_dims=2
                )
                
            return results
            
        except Exception as e:
            logger.error(f"Failed to analyze model {model_name}: {e}")
            return None
    
    def analyze_all_models(self):
        """Analyze all physics models in the experiment."""
        models_dir = self.experiment_dir / "models"
        results_all = {}
        
        # Common model patterns to look for
        model_patterns = [
            ("*pinn*.keras", "PINN"),
            ("*baseline*.keras", "Baseline"),
            ("*modifier*.keras", "Modifier"),
            ("*distribution*.keras", "Distribution")
        ]
        
        for pattern, model_type in model_patterns:
            model_files = list(models_dir.glob(pattern))
            
            for model_file in model_files:
                model_name = f"{model_type}_{model_file.stem}"
                results = self.analyze_model(model_file, model_name)
                
                if results:
                    results_all[model_name] = results
                    
        return results_all
    
    def generate_summary_report(self, results_all: dict, output_path: Path):
        """Generate summary report across all models and test scenarios."""
        summary = {
            'analysis_date': datetime.now().isoformat(),
            'models_analyzed': list(results_all.keys()),
            'test_scenarios': [],
            'key_findings': []
        }
        
        # Aggregate statistics across models
        for model_name, model_results in results_all.items():
            for test_name, analysis in model_results.items():
                stats = analysis['statistics']
                
                summary['test_scenarios'].append({
                    'model': model_name,
                    'test': test_name,
                    'interpolation_pct': stats['pct_interpolation'],
                    'near_extrap_pct': stats['pct_near_extrapolation'],
                    'far_extrap_pct': stats['pct_far_extrapolation']
                })
        
        # Generate key findings
        high_interpolation = [
            s for s in summary['test_scenarios'] 
            if s['interpolation_pct'] > 70
        ]
        
        true_ood = [
            s for s in summary['test_scenarios']
            if s['far_extrap_pct'] > 50
        ]
        
        if high_interpolation:
            summary['key_findings'].append(
                f"{len(high_interpolation)} test scenarios were mostly interpolation (>70%), "
                f"explaining apparent 'OOD' success"
            )
            
        if true_ood:
            summary['key_findings'].append(
                f"{len(true_ood)} test scenarios required true extrapolation (>50% far OOD)"
            )
            
        # Save report
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"\nSummary report saved to {output_path}")
        
        # Print key findings
        logger.info("\nKEY FINDINGS:")
        for finding in summary['key_findings']:
            logger.info(f"  - {finding}")
            
        return summary


def create_mock_physics_data():
    """Create mock physics data for testing the analyzer."""
    output_dir = get_output_path() / "physics_ood_test"
    data_dir = output_dir / "data"
    models_dir = output_dir / "models"
    
    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Create mock training data
    n_train = 1000
    train_positions = np.random.randn(n_train, 10, 4)  # 10 timesteps, 4 features
    np.savez(data_dir / "train_data.npz", positions=train_positions)
    
    # Create test scenarios with varying OOD characteristics
    # Same gravity - should be mostly interpolation
    test_same = train_positions + np.random.randn(*train_positions.shape) * 0.1
    np.savez(data_dir / "test_same_gravity.npz", positions=test_same)
    
    # Different gravity - mix of interpolation and near-extrapolation
    test_diff = train_positions * 1.5 + np.random.randn(*train_positions.shape) * 0.2
    np.savez(data_dir / "test_diff_gravity.npz", positions=test_diff)
    
    # Zero gravity - far extrapolation
    test_zero = np.random.randn(*train_positions.shape) * 3.0
    np.savez(data_dir / "test_zero_gravity.npz", positions=test_zero)
    
    # Create a simple mock model
    model = keras.Sequential([
        keras.layers.Input(shape=(10, 4)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu', name='representation'),
        keras.layers.Dense(10)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    model.save(models_dir / "mock_pinn_model.keras")
    
    logger.info(f"Created mock data in {output_dir}")
    return output_dir


def main():
    """Main analysis script."""
    # Check if we have real physics experiment data
    physics_exp_dir = Path("experiments/01_physics_worlds")
    
    if not physics_exp_dir.exists() or not (physics_exp_dir / "data").exists():
        logger.info("Real physics data not found, creating mock data for demonstration...")
        physics_exp_dir = create_mock_physics_data()
    
    # Create analyzer
    analyzer = PhysicsOODAnalyzer(physics_exp_dir)
    
    # Analyze all models
    logger.info("Starting OOD analysis of physics models...")
    results = analyzer.analyze_all_models()
    
    if results:
        # Generate summary report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = physics_exp_dir / f"ood_analysis_summary_{timestamp}.json"
        summary = analyzer.generate_summary_report(results, report_path)
        
        logger.info("\nAnalysis complete!")
        logger.info(f"Found {len(results)} models")
        logger.info(f"Report saved to {report_path}")
    else:
        logger.warning("No models found to analyze")


if __name__ == "__main__":
    main()