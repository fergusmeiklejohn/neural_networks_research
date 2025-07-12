"""
Unified Evaluation Framework for Distribution Invention Research

This module provides a consistent evaluation framework that:
1. Tests all models (baselines + ours) on the same data
2. Implements representation space analysis for true OOD detection
3. Evaluates modification/adaptation capabilities
4. Generates comprehensive comparison reports
"""

import numpy as np
import keras
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
import umap
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import pandas as pd


class RepresentationSpaceAnalyzer:
    """
    Analyzes whether test data is truly OOD in representation space,
    based on insights from the materials discovery paper.
    """
    
    def __init__(self, n_components=50, density_threshold=0.1):
        self.n_components = n_components
        self.density_threshold = density_threshold
        self.umap_reducer = umap.UMAP(n_components=2, random_state=42)
        self.pca = PCA(n_components=n_components)
        self.kde = None
        self.train_representations = None
        
    def fit_on_training_data(self, model, train_data):
        """Fit analyzer on training data representations."""
        # Get representations from model
        self.train_representations = model.get_representations(train_data)
        
        # Fit PCA for dimensionality reduction
        self.train_representations_pca = self.pca.fit_transform(self.train_representations)
        
        # Fit KDE for density estimation
        self.kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
        self.kde.fit(self.train_representations_pca)
        
        # Fit UMAP for visualization
        self.train_representations_umap = self.umap_reducer.fit_transform(self.train_representations)
        
    def categorize_test_data(self, model, test_data):
        """
        Categorize test data as interpolation, near-extrapolation, or far-extrapolation
        based on representation space analysis.
        """
        # Get test representations
        test_representations = model.get_representations(test_data)
        test_representations_pca = self.pca.transform(test_representations)
        
        # Compute log densities
        log_densities = self.kde.score_samples(test_representations_pca)
        densities = np.exp(log_densities)
        
        # Compute distance to nearest training point
        distances = self._compute_nearest_neighbor_distances(
            test_representations_pca,
            self.train_representations_pca
        )
        
        # Categorize based on density and distance
        categories = []
        for density, distance in zip(densities, distances):
            if density > self.density_threshold:
                categories.append('interpolation')
            elif distance < np.percentile(distances, 25):
                categories.append('near_extrapolation')
            else:
                categories.append('far_extrapolation')
                
        return {
            'categories': categories,
            'densities': densities,
            'distances': distances,
            'representations': test_representations
        }
    
    def _compute_nearest_neighbor_distances(self, test_reps, train_reps):
        """Compute distance to nearest training point for each test point."""
        distances = []
        for test_point in test_reps:
            dists = np.linalg.norm(train_reps - test_point, axis=1)
            distances.append(np.min(dists))
        return np.array(distances)
    
    def visualize_representation_space(self, model, test_data, save_path=None):
        """Visualize train and test data in representation space."""
        # Get test representations
        test_representations = model.get_representations(test_data)
        test_representations_umap = self.umap_reducer.transform(test_representations)
        
        # Categorize test data
        categorization = self.categorize_test_data(model, test_data)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Train vs Test
        ax1.scatter(self.train_representations_umap[:, 0], 
                   self.train_representations_umap[:, 1],
                   c='blue', alpha=0.5, label='Train', s=20)
        ax1.scatter(test_representations_umap[:, 0],
                   test_representations_umap[:, 1],
                   c='red', alpha=0.5, label='Test', s=20)
        ax1.set_title('Train vs Test in Representation Space')
        ax1.legend()
        
        # Plot 2: Test categorization
        colors = {'interpolation': 'green', 
                 'near_extrapolation': 'orange',
                 'far_extrapolation': 'red'}
        
        for category in colors:
            mask = np.array(categorization['categories']) == category
            ax2.scatter(test_representations_umap[mask, 0],
                       test_representations_umap[mask, 1],
                       c=colors[category], label=category, alpha=0.7, s=30)
        
        ax2.set_title('Test Data Categorization')
        ax2.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
        return fig


class ModificationEvaluator:
    """Evaluates model performance on distribution modification tasks."""
    
    def __init__(self):
        self.modification_types = [
            'parameter_change',
            'rule_swap',
            'constraint_addition',
            'distribution_blend'
        ]
        
    def create_modification_test_suite(self, base_data, experiment_type):
        """Create a suite of modification tests for given experiment."""
        test_suite = []
        
        if experiment_type == 'physics':
            test_suite.extend(self._create_physics_modifications(base_data))
        elif experiment_type == 'language':
            test_suite.extend(self._create_language_modifications(base_data))
        elif experiment_type == 'visual':
            test_suite.extend(self._create_visual_modifications(base_data))
            
        return test_suite
    
    def _create_physics_modifications(self, base_data):
        """Physics-specific modifications."""
        modifications = []
        
        # Gravity modifications
        modifications.append({
            'name': 'gravity_increase',
            'request': 'increase gravity by 50%',
            'transform': lambda params: {**params, 'gravity': params['gravity'] * 1.5},
            'validation': lambda traj: self._validate_physics_trajectory(traj, high_gravity=True)
        })
        
        # Friction removal
        modifications.append({
            'name': 'zero_friction',
            'request': 'remove all friction',
            'transform': lambda params: {**params, 'friction': 0.0},
            'validation': lambda traj: self._validate_frictionless(traj)
        })
        
        # Combined modifications
        modifications.append({
            'name': 'moon_physics',
            'request': 'moon-like physics (low gravity, no air resistance)',
            'transform': lambda params: {
                **params, 
                'gravity': params['gravity'] * 0.165,
                'air_resistance': 0.0
            },
            'validation': lambda traj: self._validate_moon_physics(traj)
        })
        
        return modifications
    
    def _create_language_modifications(self, base_data):
        """Language-specific modifications."""
        modifications = []
        
        # Word swapping
        modifications.append({
            'name': 'word_swap',
            'request': 'swap meanings of "jump" and "walk"',
            'transform': lambda seq: seq.replace('jump', '__temp__').replace('walk', 'jump').replace('__temp__', 'walk'),
            'validation': lambda output: 'jump' in output or 'walk' in output
        })
        
        # Rule inversion
        modifications.append({
            'name': 'direction_inversion',
            'request': 'invert all directions (left->right, up->down)',
            'transform': self._invert_directions,
            'validation': lambda output: self._check_direction_consistency(output)
        })
        
        return modifications
    
    def evaluate_modification_success(self, model, modification_suite, test_data):
        """Evaluate how well model handles modifications."""
        results = []
        
        for mod in modification_suite:
            # Apply modification
            adapted_model = model.adapt_to_modification(
                mod['request'],
                examples=self._generate_few_shot_examples(mod, test_data[:5])
            )
            
            # Test on modified data
            modified_test = self._apply_modification(test_data, mod['transform'])
            predictions = adapted_model.predict(modified_test['inputs'])
            
            # Validate predictions
            validation_scores = [
                mod['validation'](pred) for pred in predictions
            ]
            
            # Compute metrics
            success_rate = np.mean(validation_scores)
            consistency = self._compute_consistency(predictions)
            
            results.append({
                'modification': mod['name'],
                'success_rate': success_rate,
                'consistency': consistency,
                'follows_request': success_rate > 0.7
            })
            
        return pd.DataFrame(results)
    
    def _compute_consistency(self, predictions):
        """Measure internal consistency of predictions."""
        # Simple consistency: low variance in similar predictions
        if len(predictions) < 2:
            return 1.0
            
        similarities = []
        for i in range(len(predictions) - 1):
            sim = self._compute_similarity(predictions[i], predictions[i+1])
            similarities.append(sim)
            
        return np.mean(similarities)


class UnifiedEvaluator:
    """
    Main evaluation class that combines all evaluation components.
    """
    
    def __init__(self, experiment_type: str):
        self.experiment_type = experiment_type
        self.representation_analyzer = RepresentationSpaceAnalyzer()
        self.modification_evaluator = ModificationEvaluator()
        self.results = {}
        
    def evaluate_all_models(self, models: Dict[str, Any], train_data, test_data):
        """
        Comprehensive evaluation of all models.
        
        Args:
            models: Dict mapping model names to model instances
            train_data: Training data for representation analysis
            test_data: Test data for evaluation
        """
        print(f"\n{'='*60}")
        print(f"Unified Evaluation for {self.experiment_type} Experiment")
        print(f"{'='*60}\n")
        
        # Fit representation analyzer on training data using first model
        first_model = list(models.values())[0]
        self.representation_analyzer.fit_on_training_data(first_model, train_data)
        
        # Create modification test suite
        modification_suite = self.modification_evaluator.create_modification_test_suite(
            train_data, self.experiment_type
        )
        
        # Evaluate each model
        for model_name, model in models.items():
            print(f"\nEvaluating {model_name}...")
            
            # 1. Basic accuracy
            basic_accuracy = self._evaluate_basic_accuracy(model, test_data)
            
            # 2. Representation space analysis
            repr_results = self._evaluate_representation_space(model, test_data)
            
            # 3. Modification evaluation
            mod_results = self.modification_evaluator.evaluate_modification_success(
                model, modification_suite, test_data
            )
            
            # 4. Efficiency metrics
            efficiency = self._evaluate_efficiency(model, test_data)
            
            # Store results
            self.results[model_name] = {
                'basic_accuracy': basic_accuracy,
                'interpolation_accuracy': repr_results['interpolation_acc'],
                'near_extrapolation_accuracy': repr_results['near_extrap_acc'],
                'far_extrapolation_accuracy': repr_results['far_extrap_acc'],
                'modification_success_rate': mod_results['success_rate'].mean(),
                'consistency_score': mod_results['consistency'].mean(),
                'inference_time': efficiency['inference_time'],
                'adaptation_time': efficiency['adaptation_time']
            }
            
            # Visualize representation space
            self.representation_analyzer.visualize_representation_space(
                model, test_data,
                save_path=f"results/{self.experiment_type}_{model_name}_repr_space.png"
            )
        
        return self.generate_report()
    
    def _evaluate_basic_accuracy(self, model, test_data):
        """Evaluate basic prediction accuracy."""
        predictions = model.predict(test_data['inputs'])
        
        if self.experiment_type == 'physics':
            # MSE for trajectory prediction
            return -np.mean((predictions - test_data['targets'])**2)
        elif self.experiment_type == 'language':
            # Exact match for language tasks
            return np.mean(predictions == test_data['targets'])
        else:
            # Classification accuracy
            return np.mean(np.argmax(predictions, axis=1) == test_data['targets'])
    
    def _evaluate_representation_space(self, model, test_data):
        """Evaluate performance by representation space category."""
        categorization = self.representation_analyzer.categorize_test_data(model, test_data)
        
        results = {
            'interpolation_acc': [],
            'near_extrap_acc': [],
            'far_extrap_acc': []
        }
        
        # Evaluate each category separately
        for idx, category in enumerate(categorization['categories']):
            pred = model.predict(test_data['inputs'][idx:idx+1])
            true = test_data['targets'][idx:idx+1]
            
            if self.experiment_type == 'physics':
                acc = -np.mean((pred - true)**2)
            else:
                acc = float(np.array_equal(pred, true))
                
            if category == 'interpolation':
                results['interpolation_acc'].append(acc)
            elif category == 'near_extrapolation':
                results['near_extrap_acc'].append(acc)
            else:
                results['far_extrap_acc'].append(acc)
        
        # Average results
        for key in results:
            results[key] = np.mean(results[key]) if results[key] else 0.0
            
        return results
    
    def _evaluate_efficiency(self, model, test_data):
        """Evaluate computational efficiency."""
        import time
        
        # Inference time
        start = time.time()
        _ = model.predict(test_data['inputs'][:100])
        inference_time = (time.time() - start) / 100
        
        # Adaptation time
        start = time.time()
        _ = model.adapt_to_modification(
            "test modification",
            examples={'x': test_data['inputs'][:5], 'y': test_data['targets'][:5]}
        )
        adaptation_time = time.time() - start
        
        return {
            'inference_time': inference_time,
            'adaptation_time': adaptation_time
        }
    
    def generate_report(self):
        """Generate comprehensive evaluation report."""
        df = pd.DataFrame(self.results).T
        
        report = f"\n{'='*80}\n"
        report += f"Evaluation Results for {self.experiment_type} Experiment\n"
        report += f"{'='*80}\n\n"
        
        # Performance table
        report += "Performance Metrics:\n"
        report += df.to_string()
        report += "\n\n"
        
        # Key insights
        report += "Key Insights:\n"
        report += "-" * 40 + "\n"
        
        # Best model for each metric
        for metric in df.columns:
            if 'time' not in metric:  # Higher is better except for time
                best_model = df[metric].idxmax()
                best_score = df[metric].max()
            else:  # Lower is better for time
                best_model = df[metric].idxmin()
                best_score = df[metric].min()
                
            report += f"Best {metric}: {best_model} ({best_score:.3f})\n"
        
        # Extrapolation gap analysis
        report += "\nExtrapolation Performance Gaps:\n"
        for model in df.index:
            interp = df.loc[model, 'interpolation_accuracy']
            far_extrap = df.loc[model, 'far_extrapolation_accuracy']
            gap = interp - far_extrap
            report += f"{model}: {gap:.3f} ({interp:.3f} -> {far_extrap:.3f})\n"
        
        # Save detailed results
        df.to_csv(f"results/{self.experiment_type}_evaluation_results.csv")
        
        with open(f"results/{self.experiment_type}_evaluation_report.txt", 'w') as f:
            f.write(report)
            
        print(report)
        return df


# Usage example
if __name__ == "__main__":
    # Example usage for physics experiment
    from baseline_models import ERMWithAugmentation, GFlowNetBaseline, GraphExtrapolationBaseline, MAMLBaseline
    
    # Initialize models
    config = {
        'task_type': 'physics',
        'input_shape': (4,),
        'output_shape': (100, 2)
    }
    
    models = {
        'ERM+Aug': ERMWithAugmentation(config),
        'GFlowNet': GFlowNetBaseline(config),
        'GraphExtrap': GraphExtrapolationBaseline(config),
        'MAML': MAMLBaseline(config),
        # 'DistInvention': OurDistributionInventionModel(config)  # Add our model
    }
    
    # Create evaluator
    evaluator = UnifiedEvaluator('physics')
    
    # Load your data
    # train_data = load_train_data()
    # test_data = load_test_data()
    
    # Run evaluation
    # results = evaluator.evaluate_all_models(models, train_data, test_data)