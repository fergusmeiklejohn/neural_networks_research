"""
Distribution Inventor - Main model that combines all components to invent
new physics distributions while maintaining consistency.
"""

import keras
from keras import layers, ops
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from .physics_rule_extractor import PhysicsRuleExtractor, PhysicsRuleConfig
from .distribution_modifier import DistributionModifier, ModifierConfig
from .trajectory_generator import TrajectoryGenerator, TrajectoryConfig


@dataclass
class DistributionInventorConfig:
    """Configuration for the complete distribution invention system"""
    # Component configurations
    rule_extractor_config: PhysicsRuleConfig = None
    modifier_config: ModifierConfig = None 
    trajectory_config: TrajectoryConfig = None
    
    # Integration parameters
    consistency_threshold: float = 0.7
    novelty_target: float = 0.6
    quality_threshold: float = 0.8
    
    # Training strategy
    pretrain_extractor: bool = True
    pretrain_generator: bool = True
    joint_training_epochs: int = 50
    
    # Loss weights for joint training
    extraction_loss_weight: float = 1.0
    modification_loss_weight: float = 1.0
    generation_loss_weight: float = 1.0
    consistency_loss_weight: float = 0.5
    
    def __post_init__(self):
        if self.rule_extractor_config is None:
            self.rule_extractor_config = PhysicsRuleConfig()
        if self.modifier_config is None:
            self.modifier_config = ModifierConfig()
        if self.trajectory_config is None:
            self.trajectory_config = TrajectoryConfig()


class ConsistencyEnforcer(layers.Layer):
    """Enforces consistency across all model components"""
    
    def __init__(self, config: DistributionInventorConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        
        # Cross-component consistency checker
        self.consistency_network = keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(1, activation='sigmoid', name='overall_consistency')
        ])
        
        # Rule coherence validator
        self.rule_coherence = keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid', name='rule_coherence')
        ])
        
        # Trajectory plausibility checker
        self.trajectory_plausibility = keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(1, activation='sigmoid', name='trajectory_plausibility')
        ])
    
    def call(self, extracted_rules, modified_rules, generated_trajectory, training=None):
        # Combine all information for consistency checking
        extracted_combined = ops.concatenate([
            extracted_rules['gravity'],
            extracted_rules['friction'],
            extracted_rules['elasticity'],
            extracted_rules['damping']
        ], axis=-1)
        
        modified_combined = ops.concatenate([
            modified_rules['gravity'],
            modified_rules['friction'],
            modified_rules['elasticity'],
            modified_rules['damping']
        ], axis=-1)
        
        # Flatten trajectory for analysis
        traj_shape = ops.shape(generated_trajectory)
        flat_trajectory = ops.reshape(generated_trajectory, [traj_shape[0], -1])
        
        # Check rule coherence (do extracted and modified rules make sense together?)
        rule_diff = modified_combined - extracted_combined
        rule_coherence_score = self.rule_coherence(rule_diff, training=training)
        
        # Check trajectory plausibility under modified rules
        trajectory_input = ops.concatenate([flat_trajectory, modified_combined], axis=-1)
        trajectory_plausibility_score = self.trajectory_plausibility(
            trajectory_input, training=training
        )
        
        # Overall consistency
        consistency_input = ops.concatenate([
            extracted_combined,
            modified_combined,
            flat_trajectory[:, :100]  # Sample of trajectory
        ], axis=-1)
        
        overall_consistency = self.consistency_network(consistency_input, training=training)
        
        return {
            'overall_consistency': overall_consistency,
            'rule_coherence': rule_coherence_score,
            'trajectory_plausibility': trajectory_plausibility_score
        }


class DistributionInventor(keras.Model):
    """Main distribution invention model combining all components"""
    
    def __init__(self, config: DistributionInventorConfig, vocab_size: int = 1000, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        
        # Core components
        self.rule_extractor = PhysicsRuleExtractor(config.rule_extractor_config)
        self.distribution_modifier = DistributionModifier(config.modifier_config, vocab_size)
        self.trajectory_generator = TrajectoryGenerator(config.trajectory_config)
        
        # Consistency enforcer
        self.consistency_enforcer = ConsistencyEnforcer(config)
        
        # Insight extractor (maps back to original distribution)
        self.insight_extractor = keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu', name='insights')
        ])
        
    def call(self, inputs, training=None):
        """
        Full forward pass through the distribution invention pipeline
        
        inputs should contain:
        - trajectory_data: Original trajectory data for rule extraction
        - modification_request: Text description of desired modification
        - initial_conditions: Initial conditions for new trajectory generation
        """
        trajectory_data = inputs['trajectory_data']
        modification_request = inputs['modification_request']
        initial_conditions = inputs['initial_conditions']
        
        # Step 1: Extract rules from original trajectory
        extracted_rules = self.rule_extractor(trajectory_data, training=training)
        
        # Step 2: Modify rules based on request
        modification_inputs = {
            'base_rules': {
                'gravity': extracted_rules['gravity'],
                'friction': extracted_rules['friction'],
                'elasticity': extracted_rules['elasticity'],
                'damping': extracted_rules['damping']
            },
            'modification_text': modification_request
        }
        
        modification_output = self.distribution_modifier(modification_inputs, training=training)
        modified_rules = modification_output['modified_rules']
        
        # Step 3: Generate new trajectory under modified rules
        generation_inputs = {
            'initial_conditions': initial_conditions,
            'physics_rules': {
                'gravity': modified_rules['gravity'],
                'friction': modified_rules['friction'],
                'elasticity': modified_rules['elasticity'],
                'damping': modified_rules['damping']
            }
        }
        
        generation_output = self.trajectory_generator(generation_inputs, training=training)
        
        # Step 4: Enforce consistency across all components
        consistency_scores = self.consistency_enforcer(
            extracted_rules,
            modified_rules,
            generation_output['trajectory'],
            training=training
        )
        
        # Step 5: Extract insights that map back to original distribution
        insight_input = ops.concatenate([
            generation_output['trajectory'][:, :, :, 1:3].reshape([ops.shape(trajectory_data)[0], -1]),  # Positions
            modified_rules['gravity'],
            modified_rules['friction'],
            modified_rules['elasticity'],
            modified_rules['damping']
        ], axis=-1)
        
        insights = self.insight_extractor(insight_input, training=training)
        
        return {
            # Component outputs
            'extracted_rules': extracted_rules,
            'modified_rules': modified_rules,
            'generated_trajectory': generation_output['trajectory'],
            'trajectory_uncertainty': generation_output['uncertainty'],
            
            # Quality scores
            'consistency_scores': consistency_scores,
            'modification_consistency': modification_output['consistency_score'],
            'modification_novelty': modification_output['novelty_score'],
            'trajectory_quality': generation_output['quality_score'],
            'physics_consistency': generation_output['physics_consistency'],
            
            # Insights
            'insights': insights
        }
    
    def invent_distribution(self,
                          base_trajectory: np.ndarray,
                          modification_request: str,
                          initial_conditions: np.ndarray,
                          return_details: bool = True) -> Dict[str, Any]:
        """
        High-level interface for inventing new distributions
        
        Args:
            base_trajectory: Original trajectory data to extract rules from
            modification_request: Text description of desired modification
            initial_conditions: Initial conditions for new trajectory
            return_details: Whether to return detailed analysis
            
        Returns:
            Dictionary containing new trajectory and analysis
        """
        
        # Tokenize modification request (simplified)
        modification_tokens = self.distribution_modifier._tokenize_request(modification_request)
        modification_tokens = np.expand_dims(modification_tokens, 0)
        
        # Prepare inputs
        inputs = {
            'trajectory_data': np.expand_dims(base_trajectory, 0),
            'modification_request': modification_tokens,
            'initial_conditions': np.expand_dims(initial_conditions, 0)
        }
        
        # Run full pipeline
        outputs = self(inputs, training=False)
        
        # Extract results
        result = {
            'new_trajectory': outputs['generated_trajectory'].numpy()[0],
            'success': True,
            'modification_applied': modification_request
        }
        
        if return_details:
            result.update({
                'original_rules': {k: v.numpy()[0] for k, v in outputs['extracted_rules'].items() 
                                 if k not in ['independence_score', 'consistency_score', 'features']},
                'modified_rules': {k: v.numpy()[0] for k, v in outputs['modified_rules'].items() 
                                 if k not in ['deltas', 'weights']},
                'quality_scores': {
                    'overall_consistency': outputs['consistency_scores']['overall_consistency'].numpy()[0],
                    'rule_coherence': outputs['consistency_scores']['rule_coherence'].numpy()[0],
                    'trajectory_plausibility': outputs['consistency_scores']['trajectory_plausibility'].numpy()[0],
                    'modification_novelty': outputs['modification_novelty'].numpy()[0],
                    'trajectory_quality': outputs['trajectory_quality'].numpy()[0]
                },
                'uncertainty': outputs['trajectory_uncertainty'].numpy()[0],
                'insights': outputs['insights'].numpy()[0]
            })
            
            # Determine if modification was successful
            min_consistency = self.config.consistency_threshold
            min_quality = self.config.quality_threshold
            
            result['success'] = (
                result['quality_scores']['overall_consistency'] > min_consistency and
                result['quality_scores']['trajectory_quality'] > min_quality
            )
        
        return result
    
    def analyze_modification_space(self,
                                 base_trajectory: np.ndarray,
                                 modification_requests: List[str],
                                 initial_conditions: np.ndarray) -> Dict[str, Any]:
        """
        Analyze multiple modification requests to understand the modification space
        """
        results = []
        
        for request in modification_requests:
            try:
                result = self.invent_distribution(
                    base_trajectory, request, initial_conditions, return_details=True
                )
                results.append(result)
            except Exception as e:
                results.append({
                    'modification_applied': request,
                    'success': False,
                    'error': str(e)
                })
        
        # Analyze patterns
        successful_mods = [r for r in results if r.get('success', False)]
        
        analysis = {
            'total_requests': len(modification_requests),
            'successful_modifications': len(successful_mods),
            'success_rate': len(successful_mods) / len(modification_requests),
            'results': results
        }
        
        if successful_mods:
            # Analyze quality score distributions
            qualities = [r['quality_scores'] for r in successful_mods]
            analysis['average_quality'] = {
                key: np.mean([q[key] for q in qualities])
                for key in qualities[0].keys()
            }
        
        return analysis


class DistributionInventorLoss(keras.losses.Loss):
    """Combined loss function for the full distribution invention system"""
    
    def __init__(self,
                 config: DistributionInventorConfig,
                 name: str = "distribution_inventor_loss",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.config = config
    
    def call(self, y_true, y_pred):
        # Component losses (simplified - in practice these would use component-specific losses)
        extraction_loss = keras.losses.mse(
            y_true.get('true_rules', ops.zeros_like(y_pred['extracted_rules']['gravity'])),
            y_pred['extracted_rules']['gravity']
        )
        
        generation_loss = keras.losses.mse(
            y_true.get('target_trajectory', ops.zeros_like(y_pred['generated_trajectory'])),
            y_pred['generated_trajectory']
        )
        
        # Consistency losses
        consistency_target = ops.ones_like(y_pred['consistency_scores']['overall_consistency'])
        consistency_loss = keras.losses.binary_crossentropy(
            consistency_target,
            y_pred['consistency_scores']['overall_consistency']
        )
        
        # Novelty loss (encourage appropriate novelty)
        novelty_target = self.config.novelty_target * ops.ones_like(y_pred['modification_novelty'])
        novelty_loss = keras.losses.mse(novelty_target, y_pred['modification_novelty'])
        
        # Combine losses
        total_loss = (
            self.config.extraction_loss_weight * extraction_loss +
            self.config.generation_loss_weight * generation_loss +
            self.config.consistency_loss_weight * consistency_loss +
            0.2 * novelty_loss
        )
        
        return total_loss


def create_distribution_inventor(config: Optional[DistributionInventorConfig] = None,
                               vocab_size: int = 1000) -> DistributionInventor:
    """Create and compile a complete distribution inventor model"""
    if config is None:
        config = DistributionInventorConfig()
    
    model = DistributionInventor(config, vocab_size)
    
    # Compile with combined loss
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=DistributionInventorLoss(config),
        metrics=[
            keras.metrics.MeanSquaredError(name='mse'),
            keras.metrics.BinaryAccuracy(name='consistency_acc')
        ]
    )
    
    return model


if __name__ == "__main__":
    # Test the complete distribution inventor
    print("Testing Distribution Inventor...")
    
    config = DistributionInventorConfig()
    model = create_distribution_inventor(config)
    
    # Create test data
    batch_size = 2
    sequence_length = 100
    max_balls = 2
    feature_dim = 9
    
    # Base trajectory data
    trajectory_data = np.random.random((batch_size, sequence_length, feature_dim))
    
    # Modification requests (tokenized)
    modification_request = np.random.randint(0, 100, (batch_size, 20))
    
    # Initial conditions
    initial_conditions = np.random.random((batch_size, max_balls, feature_dim))
    
    inputs = {
        'trajectory_data': trajectory_data,
        'modification_request': modification_request,
        'initial_conditions': initial_conditions
    }
    
    print(f"Trajectory data shape: {trajectory_data.shape}")
    print(f"Modification request shape: {modification_request.shape}")
    print(f"Initial conditions shape: {initial_conditions.shape}")
    
    # Test forward pass
    outputs = model(inputs)
    
    print("\nModel outputs:")
    for key, value in outputs.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                if hasattr(sub_value, 'shape'):
                    print(f"    {sub_key}: {sub_value.shape}")
        else:
            print(f"  {key}: {value.shape}")
    
    # Test high-level interface
    print("\nTesting high-level interface...")
    test_trajectory = np.random.random((sequence_length, feature_dim))
    test_initial = np.random.random((max_balls, feature_dim))
    
    result = model.invent_distribution(
        test_trajectory,
        "increase gravity by 20%",
        test_initial
    )
    
    print(f"Invention result:")
    print(f"  Success: {result['success']}")
    print(f"  New trajectory shape: {result['new_trajectory'].shape}")
    if 'quality_scores' in result:
        print(f"  Quality scores:")
        for key, value in result['quality_scores'].items():
            print(f"    {key}: {value:.4f}")
    
    print("\nDistribution Inventor test complete!")