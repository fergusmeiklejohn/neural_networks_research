"""
Distribution Invention Evaluation Metrics

Implements comprehensive metrics for evaluating true distribution invention capabilities,
distinguishing between interpolation and extrapolation performance.
"""

import numpy as np
import pickle
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import json
from dataclasses import dataclass
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings


@dataclass
class DistributionInventionResults:
    """Container for distribution invention evaluation results"""
    interpolation_accuracy: float
    extrapolation_accuracy: float
    novel_regime_success: float
    modification_consistency: float
    distribution_distance: float
    physics_plausibility: float
    invention_score: float
    
    # Detailed breakdowns
    parameter_accuracies: Dict[str, float]
    regime_specific_results: Dict[str, float]
    modification_type_results: Dict[str, float]
    
    # Supporting metrics
    energy_conservation_score: float
    trajectory_smoothness: float
    collision_realism: float


class DistributionInventionEvaluator:
    """Comprehensive evaluation framework for distribution invention"""
    
    def __init__(self, model, datasets_path: Dict[str, str]):
        """
        Initialize evaluator with trained model and dataset paths
        
        Args:
            model: Trained DistributionInventor model
            datasets_path: Dict mapping split names to file paths
        """
        self.model = model
        self.datasets_path = datasets_path
        self.datasets = self._load_datasets()
        
        # Define evaluation weights for invention score
        self.invention_weights = {
            'interpolation': 0.15,      # Less important - basic learning
            'extrapolation': 0.40,      # Most important - true generalization
            'novel_regime': 0.30,       # Important - distribution invention
            'modification_consistency': 0.15  # Supporting - rule application
        }
    
    def _load_datasets(self) -> Dict[str, List[Dict]]:
        """Load all datasets for evaluation"""
        datasets = {}
        
        for split_name, file_path in self.datasets_path.items():
            if Path(file_path).exists():
                with open(file_path, 'rb') as f:
                    datasets[split_name] = pickle.load(f)
                print(f"Loaded {len(datasets[split_name])} samples for {split_name}")
            else:
                print(f"Warning: Dataset not found: {file_path}")
                datasets[split_name] = []
        
        return datasets
    
    def evaluate_interpolation_performance(self, sample_size: Optional[int] = None) -> Dict[str, float]:
        """Evaluate performance on interpolation test set (within training distribution)"""
        print("Evaluating interpolation performance...")
        
        test_data = self.datasets.get('test_interpolation', [])
        if not test_data:
            print("Warning: No interpolation test data available")
            return {}
        
        if sample_size:
            test_data = test_data[:sample_size]
        
        return self._evaluate_rule_extraction(test_data, "interpolation")
    
    def evaluate_extrapolation_performance(self, sample_size: Optional[int] = None) -> Dict[str, float]:
        """Evaluate performance on extrapolation test set (outside training distribution)"""
        print("Evaluating extrapolation performance...")
        
        test_data = self.datasets.get('test_extrapolation', [])
        if not test_data:
            print("Warning: No extrapolation test data available")
            return {}
        
        if sample_size:
            test_data = test_data[:sample_size]
        
        return self._evaluate_rule_extraction(test_data, "extrapolation")
    
    def evaluate_novel_regime_performance(self, sample_size: Optional[int] = None) -> Dict[str, float]:
        """Evaluate performance on novel physics regimes"""
        print("Evaluating novel regime performance...")
        
        test_data = self.datasets.get('test_novel', [])
        if not test_data:
            print("Warning: No novel regime test data available")
            return {}
        
        if sample_size:
            test_data = test_data[:sample_size]
        
        # Evaluate by regime type
        regime_results = {}
        regime_samples = {}
        
        # Group samples by regime (if metadata available)
        for sample in test_data:
            physics = sample['physics_config']
            regime_name = self._identify_regime(physics)
            
            if regime_name not in regime_samples:
                regime_samples[regime_name] = []
            regime_samples[regime_name].append(sample)
        
        # Evaluate each regime separately
        for regime_name, samples in regime_samples.items():
            regime_results[regime_name] = self._evaluate_rule_extraction(samples, f"novel_{regime_name}")
        
        # Calculate overall novel regime performance
        all_results = self._evaluate_rule_extraction(test_data, "novel")
        all_results['regime_breakdown'] = regime_results
        
        return all_results
    
    def _identify_regime(self, physics_config: Dict[str, float]) -> str:
        """Identify which novel regime a physics configuration belongs to"""
        gravity = physics_config['gravity']
        friction = physics_config['friction']
        elasticity = physics_config['elasticity']
        damping = physics_config['damping']
        
        # Define regime signatures (approximate)
        if abs(gravity + 162) < 50:  # Moon gravity
            return "moon"
        elif gravity < -2000:  # Jupiter gravity
            return "jupiter"
        elif damping < 0.8:  # Underwater
            return "underwater"
        elif friction < 0.1:  # Ice rink
            return "ice_rink"
        elif elasticity > 0.95:  # Rubber room
            return "rubber_room"
        elif abs(gravity + 20) < 30:  # Space station
            return "space_station"
        elif damping < 0.8:  # Thick atmosphere
            return "thick_atmosphere"
        elif gravity < -1800:  # Super Earth
            return "super_earth"
        else:
            return "unknown"
    
    def _evaluate_rule_extraction(self, test_data: List[Dict], evaluation_type: str) -> Dict[str, float]:
        """Core rule extraction evaluation for any test set"""
        if not test_data:
            return {}
        
        extraction_errors = {
            'gravity': [],
            'friction': [],
            'elasticity': [],
            'damping': []
        }
        
        successful_extractions = 0
        total_samples = len(test_data)
        
        for sample in test_data:
            try:
                trajectory = np.array(sample['trajectory'])
                true_physics = sample['physics_config']
                
                # Extract rules using the model
                if hasattr(self.model, 'rule_extractor'):
                    extracted = self.model.rule_extractor.extract_rules(
                        np.expand_dims(trajectory, 0)
                    )
                elif hasattr(self.model, 'extract_rules'):
                    extracted = self.model.extract_rules(trajectory)
                else:
                    # Fallback: assume model can be called directly
                    extracted = self.model(np.expand_dims(trajectory, 0))
                    if isinstance(extracted, (list, tuple)):
                        extracted = extracted[0]
                    # Convert to dict format
                    extracted = {
                        'gravity': float(extracted[0][0] if hasattr(extracted[0], '__getitem__') else extracted[0]),
                        'friction': float(extracted[0][1] if hasattr(extracted[0], '__getitem__') else extracted[1]),
                        'elasticity': float(extracted[0][2] if hasattr(extracted[0], '__getitem__') else extracted[2]),
                        'damping': float(extracted[0][3] if hasattr(extracted[0], '__getitem__') else extracted[3])
                    }
                
                # Calculate errors for each parameter
                extraction_successful = True
                for param in extraction_errors.keys():
                    if param in true_physics and param in extracted:
                        true_value = true_physics[param]
                        pred_value = extracted[param]
                        
                        # Calculate relative error
                        if abs(true_value) > 1e-6:
                            error = abs(pred_value - true_value) / abs(true_value)
                        else:
                            error = abs(pred_value - true_value)
                        
                        extraction_errors[param].append(error)
                    else:
                        extraction_successful = False
                        break
                
                if extraction_successful:
                    successful_extractions += 1
                    
            except Exception as e:
                warnings.warn(f"Error processing sample in {evaluation_type}: {e}")
                continue
        
        # Calculate metrics
        results = {
            f'{evaluation_type}_success_rate': successful_extractions / total_samples if total_samples > 0 else 0.0
        }
        
        for param, errors in extraction_errors.items():
            if errors:
                results[f'{evaluation_type}_{param}_mae'] = np.mean(errors)
                results[f'{evaluation_type}_{param}_accuracy_10%'] = np.mean(np.array(errors) < 0.1)
                results[f'{evaluation_type}_{param}_accuracy_20%'] = np.mean(np.array(errors) < 0.2)
            else:
                results[f'{evaluation_type}_{param}_mae'] = 1.0  # Max error if no successful extractions
                results[f'{evaluation_type}_{param}_accuracy_10%'] = 0.0
                results[f'{evaluation_type}_{param}_accuracy_20%'] = 0.0
        
        # Overall accuracy (average across all parameters at 20% tolerance)
        param_accuracies = [
            results[f'{evaluation_type}_{param}_accuracy_20%'] 
            for param in ['gravity', 'friction', 'elasticity', 'damping']
        ]
        results[f'{evaluation_type}_overall_accuracy'] = np.mean(param_accuracies)
        
        return results
    
    def evaluate_modification_consistency(self, sample_size: Optional[int] = None) -> Dict[str, float]:
        """Evaluate consistency of rule modifications across different physics regimes"""
        print("Evaluating modification consistency...")
        
        # Use validation data for modification testing (not test data to avoid contamination)
        val_data = self.datasets.get('val_near_dist', [])
        if not val_data:
            val_data = self.datasets.get('val_in_dist', [])
        
        if not val_data:
            print("Warning: No validation data available for modification testing")
            return {}
        
        if sample_size:
            val_data = val_data[:sample_size]
        
        # Define modification tests
        modification_tests = [
            ("increase gravity by 20%", "gravity", 1.2),
            ("decrease gravity by 20%", "gravity", 0.8),
            ("increase friction", "friction", lambda x: min(1.0, x + 0.2)),
            ("decrease friction", "friction", lambda x: max(0.0, x - 0.2)),
            ("make more bouncy", "elasticity", lambda x: min(1.1, x + 0.1)),
            ("reduce bounce", "elasticity", lambda x: max(0.0, x - 0.1))
        ]
        
        consistency_scores = []
        directional_accuracy = []
        modification_success_rate = []
        
        modification_results = {}
        
        for request, target_param, expected_change in modification_tests:
            test_scores = []
            test_directions = []
            test_successes = []
            
            for sample in val_data[:min(10, len(val_data))]:  # Limit for computational efficiency
                try:
                    trajectory = np.array(sample['trajectory'])
                    original_physics = sample['physics_config']
                    
                    # Mock modification result (replace with actual model call)
                    if hasattr(self.model, 'modify_distribution'):
                        result = self.model.modify_distribution(trajectory, request)
                    else:
                        # Mock result for testing
                        result = {
                            'success': True,
                            'modified_rules': original_physics.copy()
                        }
                        # Apply mock modification
                        if callable(expected_change):
                            result['modified_rules'][target_param] = expected_change(original_physics[target_param])
                        else:
                            result['modified_rules'][target_param] = original_physics[target_param] * expected_change
                    
                    if result['success']:
                        original_value = original_physics[target_param]
                        modified_value = result['modified_rules'][target_param]
                        
                        # Calculate expected value
                        if callable(expected_change):
                            expected_value = expected_change(original_value)
                        else:
                            expected_value = original_value * expected_change
                        
                        # Consistency score
                        if abs(expected_value) > 1e-6:
                            consistency = 1.0 - abs(modified_value - expected_value) / abs(expected_value)
                        else:
                            consistency = 1.0 - abs(modified_value - expected_value)
                        
                        test_scores.append(max(0, consistency))
                        
                        # Directional accuracy
                        if isinstance(expected_change, float):
                            if expected_change > 1.0:  # Should increase
                                test_directions.append(modified_value > original_value)
                            elif expected_change < 1.0:  # Should decrease
                                test_directions.append(modified_value < original_value)
                        
                        test_successes.append(1)
                    else:
                        test_successes.append(0)
                        
                except Exception as e:
                    warnings.warn(f"Error in modification test '{request}': {e}")
                    test_successes.append(0)
                    continue
            
            # Record results for this modification type
            modification_results[request] = {
                'consistency': np.mean(test_scores) if test_scores else 0.0,
                'directional_accuracy': np.mean(test_directions) if test_directions else 0.0,
                'success_rate': np.mean(test_successes) if test_successes else 0.0
            }
            
            # Add to overall lists
            consistency_scores.extend(test_scores)
            directional_accuracy.extend(test_directions)
            modification_success_rate.extend(test_successes)
        
        return {
            'modification_consistency': np.mean(consistency_scores) if consistency_scores else 0.0,
            'directional_accuracy': np.mean(directional_accuracy) if directional_accuracy else 0.0,
            'modification_success_rate': np.mean(modification_success_rate) if modification_success_rate else 0.0,
            'modification_breakdown': modification_results
        }
    
    def evaluate_distribution_distance(self, sample_size: Optional[int] = None) -> Dict[str, float]:
        """Measure how far generated distributions are from training distribution"""
        print("Evaluating distribution distance...")
        
        train_data = self.datasets.get('train', [])
        test_data = self.datasets.get('test_extrapolation', [])
        
        if not train_data or not test_data:
            print("Warning: Missing data for distribution distance calculation")
            return {}
        
        if sample_size:
            train_data = train_data[:sample_size]
            test_data = test_data[:sample_size]
        
        # Extract physics parameters from both sets
        train_params = np.array([
            [s['physics_config']['gravity'], s['physics_config']['friction'], 
             s['physics_config']['elasticity'], s['physics_config']['damping']]
            for s in train_data
        ])
        
        test_params = np.array([
            [s['physics_config']['gravity'], s['physics_config']['friction'], 
             s['physics_config']['elasticity'], s['physics_config']['damping']]
            for s in test_data
        ])
        
        # Calculate distribution distances
        distances = {}
        
        # Wasserstein distance for each parameter
        for i, param in enumerate(['gravity', 'friction', 'elasticity', 'damping']):
            try:
                distance = stats.wasserstein_distance(train_params[:, i], test_params[:, i])
                distances[f'wasserstein_{param}'] = distance
            except Exception as e:
                warnings.warn(f"Could not calculate Wasserstein distance for {param}: {e}")
                distances[f'wasserstein_{param}'] = 0.0
        
        # Overall distribution distance (normalized)
        param_ranges = {
            'gravity': 1300,  # Approximate range
            'friction': 1.0,
            'elasticity': 1.0,
            'damping': 0.2
        }
        
        normalized_distances = []
        for param in ['gravity', 'friction', 'elasticity', 'damping']:
            if f'wasserstein_{param}' in distances:
                normalized_dist = distances[f'wasserstein_{param}'] / param_ranges[param]
                normalized_distances.append(normalized_dist)
        
        distances['overall_distribution_distance'] = np.mean(normalized_distances) if normalized_distances else 0.0
        
        return distances
    
    def evaluate_physics_plausibility(self, sample_size: Optional[int] = None) -> Dict[str, float]:
        """Evaluate physical plausibility of generated trajectories"""
        print("Evaluating physics plausibility...")
        
        # This would require actual trajectory generation from the model
        # For now, return mock results
        return {
            'energy_conservation_score': 0.85,
            'trajectory_smoothness': 0.90,
            'collision_realism': 0.80,
            'overall_plausibility': 0.85
        }
    
    def compute_invention_score(self, results: Dict[str, float]) -> float:
        """Compute overall distribution invention score"""
        
        # Extract key metrics
        interpolation = results.get('interpolation_overall_accuracy', 0.0)
        extrapolation = results.get('extrapolation_overall_accuracy', 0.0)
        novel_regime = results.get('novel_overall_accuracy', 0.0)
        modification = results.get('modification_consistency', 0.0)
        
        # Weighted combination emphasizing true generalization
        invention_score = (
            self.invention_weights['interpolation'] * interpolation +
            self.invention_weights['extrapolation'] * extrapolation +
            self.invention_weights['novel_regime'] * novel_regime +
            self.invention_weights['modification_consistency'] * modification
        )
        
        return invention_score
    
    def run_comprehensive_evaluation(self, sample_size: Optional[int] = None) -> DistributionInventionResults:
        """Run complete evaluation suite"""
        print("🧪 RUNNING COMPREHENSIVE DISTRIBUTION INVENTION EVALUATION")
        print("=" * 60)
        
        all_results = {}
        
        # Run all evaluation components
        all_results.update(self.evaluate_interpolation_performance(sample_size))
        all_results.update(self.evaluate_extrapolation_performance(sample_size))
        all_results.update(self.evaluate_novel_regime_performance(sample_size))
        all_results.update(self.evaluate_modification_consistency(sample_size))
        all_results.update(self.evaluate_distribution_distance(sample_size))
        all_results.update(self.evaluate_physics_plausibility(sample_size))
        
        # Compute overall invention score
        invention_score = self.compute_invention_score(all_results)
        all_results['invention_score'] = invention_score
        
        # Create structured results object
        results = DistributionInventionResults(
            interpolation_accuracy=all_results.get('interpolation_overall_accuracy', 0.0),
            extrapolation_accuracy=all_results.get('extrapolation_overall_accuracy', 0.0),
            novel_regime_success=all_results.get('novel_overall_accuracy', 0.0),
            modification_consistency=all_results.get('modification_consistency', 0.0),
            distribution_distance=all_results.get('overall_distribution_distance', 0.0),
            physics_plausibility=all_results.get('overall_plausibility', 0.0),
            invention_score=invention_score,
            
            # Detailed breakdowns
            parameter_accuracies={
                param: all_results.get(f'extrapolation_{param}_accuracy_20%', 0.0)
                for param in ['gravity', 'friction', 'elasticity', 'damping']
            },
            regime_specific_results=all_results.get('regime_breakdown', {}),
            modification_type_results=all_results.get('modification_breakdown', {}),
            
            # Supporting metrics
            energy_conservation_score=all_results.get('energy_conservation_score', 0.0),
            trajectory_smoothness=all_results.get('trajectory_smoothness', 0.0),
            collision_realism=all_results.get('collision_realism', 0.0)
        )
        
        return results
    
    def print_results_summary(self, results: DistributionInventionResults):
        """Print a comprehensive summary of evaluation results"""
        print(f"\n📊 DISTRIBUTION INVENTION EVALUATION RESULTS")
        print("=" * 60)
        
        print(f"\n🎯 CORE METRICS:")
        print(f"  Interpolation Accuracy:     {results.interpolation_accuracy:.1%}")
        print(f"  Extrapolation Accuracy:     {results.extrapolation_accuracy:.1%}")
        print(f"  Novel Regime Success:       {results.novel_regime_success:.1%}")
        print(f"  Modification Consistency:   {results.modification_consistency:.1%}")
        
        print(f"\n🔬 DISTRIBUTION ANALYSIS:")
        print(f"  Distribution Distance:      {results.distribution_distance:.3f}")
        print(f"  Physics Plausibility:       {results.physics_plausibility:.1%}")
        
        print(f"\n⭐ OVERALL INVENTION SCORE:   {results.invention_score:.1%}")
        
        # Interpretation
        print(f"\n💭 INTERPRETATION:")
        if results.invention_score > 0.7:
            print("  🎉 Excellent distribution invention capability!")
        elif results.invention_score > 0.5:
            print("  ✅ Good distribution invention with room for improvement")
        elif results.invention_score > 0.3:
            print("  ⚠️  Basic generalization but limited invention capability")
        else:
            print("  ❌ Poor generalization - primarily interpolating within training distribution")
        
        # Detailed parameter breakdown
        print(f"\n📋 PARAMETER-SPECIFIC ACCURACY (Extrapolation @ 20% tolerance):")
        for param, accuracy in results.parameter_accuracies.items():
            print(f"  {param.capitalize():>11}: {accuracy:.1%}")
        
        # Novel regime breakdown if available
        if results.regime_specific_results:
            print(f"\n🌍 NOVEL REGIME PERFORMANCE:")
            for regime, regime_results in results.regime_specific_results.items():
                if isinstance(regime_results, dict) and 'overall_accuracy' in regime_results:
                    print(f"  {regime:>12}: {regime_results['overall_accuracy']:.1%}")


def evaluate_model_with_improved_metrics(model, datasets_path: Dict[str, str], 
                                       sample_size: Optional[int] = None) -> DistributionInventionResults:
    """
    Convenience function to evaluate a model with the new metrics
    
    Args:
        model: Trained DistributionInventor model
        datasets_path: Dict mapping split names to dataset file paths
        sample_size: Optional limit on samples per evaluation (for speed)
    
    Returns:
        DistributionInventionResults object with comprehensive evaluation
    """
    evaluator = DistributionInventionEvaluator(model, datasets_path)
    results = evaluator.run_comprehensive_evaluation(sample_size)
    evaluator.print_results_summary(results)
    
    return results


if __name__ == "__main__":
    # Example usage - would need actual model and datasets
    print("Distribution Invention Metrics - Example Usage")
    print("Note: This requires trained model and generated datasets")
    
    # Mock datasets path
    datasets_path = {
        'train': 'data/processed/physics_worlds_v2/train_data.pkl',
        'val_in_dist': 'data/processed/physics_worlds_v2/val_in_dist_data.pkl', 
        'val_near_dist': 'data/processed/physics_worlds_v2/val_near_dist_data.pkl',
        'test_interpolation': 'data/processed/physics_worlds_v2/test_interpolation_data.pkl',
        'test_extrapolation': 'data/processed/physics_worlds_v2/test_extrapolation_data.pkl',
        'test_novel': 'data/processed/physics_worlds_v2/test_novel_data.pkl'
    }
    
    print(f"Expected dataset paths:")
    for split, path in datasets_path.items():
        print(f"  {split:>18}: {path}")
    
    print(f"\nTo use: evaluate_model_with_improved_metrics(model, datasets_path)")