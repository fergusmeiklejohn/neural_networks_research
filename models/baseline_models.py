"""
Baseline Models for Distribution Invention Research

This module provides implementations of 4 baseline models that will be compared
against our distribution invention approach across all experiments.

Baselines:
1. ERM + Data Augmentation: Standard empirical risk minimization with domain-specific augmentation
2. GFlowNet-guided Search: Using GFlowNets for exploration in parameter/rule space  
3. Graph Extrapolation: Non-Euclidean extrapolation methods
4. Meta-Learning (MAML): Model-Agnostic Meta-Learning for quick adaptation
"""

import keras
import tensorflow as tf
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List
import numpy as np


class BaselineModel(ABC):
    """Abstract base class for all baseline models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.trained = False
        
    @abstractmethod
    def build_model(self):
        """Build the model architecture."""
        pass
    
    @abstractmethod
    def train(self, train_data, val_data, epochs: int = 100):
        """Train the model on given data."""
        pass
    
    @abstractmethod
    def predict(self, inputs):
        """Make predictions on new inputs."""
        pass
    
    @abstractmethod
    def adapt_to_modification(self, modification_request, examples=None):
        """Attempt to adapt to a rule modification request."""
        pass
    
    def evaluate_extrapolation(self, test_data, representation_analyzer):
        """
        Evaluate model on interpolation vs extrapolation.
        Uses representation space analysis to categorize test data.
        """
        # Get model's internal representations
        representations = self.get_representations(test_data)
        
        # Categorize as interpolation/near/far extrapolation
        categories = representation_analyzer.categorize(representations)
        
        results = {}
        for category, data_subset in categories.items():
            predictions = self.predict(data_subset['inputs'])
            accuracy = self.compute_accuracy(predictions, data_subset['targets'])
            results[category] = accuracy
            
        return results
    
    @abstractmethod
    def get_representations(self, inputs):
        """Get internal representations for representation space analysis."""
        pass


class ERMWithAugmentation(BaselineModel):
    """
    Baseline 1: Empirical Risk Minimization with Data Augmentation
    
    This represents the standard deep learning approach with domain-specific
    data augmentation to improve generalization.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.augmentation_strategies = config.get('augmentation_strategies', [])
        
    def build_model(self):
        """Build a standard neural network for the task."""
        if self.config['task_type'] == 'physics':
            self.model = self._build_physics_model()
        elif self.config['task_type'] == 'language':
            self.model = self._build_language_model()
        elif self.config['task_type'] == 'visual':
            self.model = self._build_visual_model()
        else:
            raise ValueError(f"Unknown task type: {self.config['task_type']}")
    
    def _build_physics_model(self):
        """Physics prediction model."""
        return keras.Sequential([
            keras.layers.Input(shape=self.config['input_shape']),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(self.config['output_shape'])
        ])
    
    def _build_language_model(self):
        """Language task model."""
        return keras.Sequential([
            keras.layers.Input(shape=self.config['input_shape']),
            keras.layers.Embedding(self.config['vocab_size'], 128),
            keras.layers.LSTM(256, return_sequences=True),
            keras.layers.LSTM(256),
            keras.layers.Dense(self.config['output_shape'])
        ])
    
    def _build_visual_model(self):
        """Visual task model."""
        base_model = keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.config['input_shape']
        )
        return keras.Sequential([
            base_model,
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(self.config['output_shape'])
        ])
    
    def augment_data(self, data):
        """Apply domain-specific data augmentation."""
        augmented_data = []
        
        for strategy in self.augmentation_strategies:
            if strategy == 'physics_noise':
                # Add noise to physics parameters
                augmented = data + np.random.normal(0, 0.1, data.shape)
            elif strategy == 'physics_interpolation':
                # Interpolate between physics states
                idx = np.random.permutation(len(data))
                alpha = np.random.uniform(0, 1, (len(data), 1))
                augmented = alpha * data + (1 - alpha) * data[idx]
            elif strategy == 'language_synonym':
                # Replace words with synonyms (simplified)
                augmented = self._synonym_replacement(data)
            elif strategy == 'language_reorder':
                # Reorder sequences while preserving meaning
                augmented = self._sequence_reordering(data)
            else:
                augmented = data
                
            augmented_data.append(augmented)
            
        return np.concatenate([data] + augmented_data, axis=0)
    
    def train(self, train_data, val_data, epochs: int = 100):
        """Train with augmented data."""
        # Augment training data
        augmented_x = self.augment_data(train_data['x'])
        augmented_y = np.tile(train_data['y'], (len(self.augmentation_strategies) + 1, 1))
        
        # Compile and train
        self.model.compile(
            optimizer='adam',
            loss=self.config.get('loss', 'mse'),
            metrics=['accuracy']
        )
        
        history = self.model.fit(
            augmented_x, augmented_y,
            validation_data=(val_data['x'], val_data['y']),
            epochs=epochs,
            batch_size=32
        )
        
        self.trained = True
        return history
    
    def predict(self, inputs):
        """Standard prediction."""
        return self.model.predict(inputs)
    
    def adapt_to_modification(self, modification_request, examples=None):
        """
        ERM baseline: No true adaptation capability.
        Best effort: fine-tune on examples if provided.
        """
        if examples is None:
            # Cannot adapt without examples
            return self
            
        # Fine-tune on provided examples
        self.model.fit(
            examples['x'], examples['y'],
            epochs=10,
            verbose=0
        )
        return self
    
    def get_representations(self, inputs):
        """Get penultimate layer representations."""
        feature_model = keras.Model(
            inputs=self.model.input,
            outputs=self.model.layers[-2].output
        )
        return feature_model.predict(inputs)


class GFlowNetBaseline(BaselineModel):
    """
    Baseline 2: GFlowNet-guided Search
    
    Uses Generative Flow Networks to explore the space of possible
    distributions/parameters, inspired by recent GFlowNet papers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.flow_steps = config.get('flow_steps', 10)
        self.exploration_bonus = config.get('exploration_bonus', 0.1)
        
    def build_model(self):
        """Build GFlowNet components."""
        # Forward policy network
        self.forward_policy = keras.Sequential([
            keras.layers.Input(shape=self.config['state_shape']),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(self.config['action_space'])
        ])
        
        # Backward policy network
        self.backward_policy = keras.Sequential([
            keras.layers.Input(shape=self.config['state_shape']),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(self.config['action_space'])
        ])
        
        # State evaluator (estimates log Z)
        self.state_evaluator = keras.Sequential([
            keras.layers.Input(shape=self.config['state_shape']),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(1)
        ])
        
    def train(self, train_data, val_data, epochs: int = 100):
        """Train GFlowNet to explore distribution space."""
        # Training follows GFlowNet trajectory balance objective
        for epoch in range(epochs):
            # Sample trajectories
            trajectories = self.sample_trajectories(train_data, n=32)
            
            # Compute trajectory balance loss
            loss = self.trajectory_balance_loss(trajectories)
            
            # Update networks
            # ... (gradient steps omitted for brevity)
            
        self.trained = True
        
    def sample_trajectories(self, data, n=32):
        """Sample exploration trajectories."""
        trajectories = []
        for _ in range(n):
            trajectory = []
            state = self.initial_state(data)
            
            for step in range(self.flow_steps):
                action = self.forward_policy(state)
                next_state = self.apply_action(state, action)
                reward = self.compute_reward(next_state, data)
                trajectory.append((state, action, next_state, reward))
                state = next_state
                
            trajectories.append(trajectory)
        return trajectories
    
    def adapt_to_modification(self, modification_request, examples=None):
        """
        GFlowNet can explore toward the modification by adjusting rewards.
        """
        # Modify reward function to favor requested modifications
        self.reward_modifier = self.parse_modification_request(modification_request)
        
        # Run additional exploration with modified rewards
        if examples:
            self.exploration_with_guidance(examples)
            
        return self
    
    def predict(self, inputs):
        """Generate predictions by sampling from learned distribution."""
        # Sample multiple trajectories and average
        predictions = []
        for _ in range(10):
            trajectory = self.sample_single_trajectory(inputs)
            prediction = self.trajectory_to_prediction(trajectory)
            predictions.append(prediction)
            
        return np.mean(predictions, axis=0)
    
    def get_representations(self, inputs):
        """Get GFlowNet state representations."""
        states = [self.encode_to_state(inp) for inp in inputs]
        return np.array(states)


class GraphExtrapolationBaseline(BaselineModel):
    """
    Baseline 3: Graph Structure Extrapolation
    
    Based on "Graph Structure Extrapolation for OOD Generalization" paper,
    uses non-Euclidean geometry for extrapolation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.graph_layers = config.get('graph_layers', 3)
        self.hyperbolic_dim = config.get('hyperbolic_dim', 64)
        
    def build_model(self):
        """Build graph neural network with extrapolation capabilities."""
        # Implement graph structure learning
        self.graph_encoder = self._build_graph_encoder()
        self.hyperbolic_embedder = self._build_hyperbolic_embedder()
        self.decoder = self._build_decoder()
        
    def _build_graph_encoder(self):
        """Encode input as graph structure."""
        # Simplified - would use proper GNN layers
        return keras.Sequential([
            keras.layers.Input(shape=self.config['input_shape']),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(self.hyperbolic_dim)
        ])
    
    def _build_hyperbolic_embedder(self):
        """Project to hyperbolic space for better extrapolation."""
        # Implement hyperbolic neural networks
        # This is simplified - real implementation would use proper hyperbolic layers
        return keras.Sequential([
            keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=-1)),
            keras.layers.Dense(self.hyperbolic_dim, activation='tanh')
        ])
    
    def _build_decoder(self):
        """Decode from hyperbolic space."""
        return keras.Sequential([
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(self.config['output_shape'])
        ])
        
    def train(self, train_data, val_data, epochs: int = 100):
        """Train with graph structure learning."""
        # Learn graph structure from data
        graph_structure = self.learn_graph_structure(train_data)
        
        # Train with graph-aware loss
        for epoch in range(epochs):
            # Forward pass through graph
            embeddings = self.graph_encoder(train_data['x'])
            hyperbolic = self.hyperbolic_embedder(embeddings)
            predictions = self.decoder(hyperbolic)
            
            # Graph-aware loss includes structure preservation
            loss = self.graph_aware_loss(predictions, train_data['y'], graph_structure)
            
            # ... (training steps omitted)
            
        self.trained = True
        
    def adapt_to_modification(self, modification_request, examples=None):
        """
        Graph extrapolation: Modify graph structure based on request.
        """
        # Parse modification as graph transformation
        graph_transform = self.parse_as_graph_transform(modification_request)
        
        # Apply transformation in hyperbolic space
        self.graph_transformation = graph_transform
        
        return self
        
    def predict(self, inputs):
        """Predict using graph extrapolation."""
        embeddings = self.graph_encoder.predict(inputs)
        hyperbolic = self.hyperbolic_embedder.predict(embeddings)
        
        # Apply any graph transformations
        if hasattr(self, 'graph_transformation'):
            hyperbolic = self.apply_graph_transform(hyperbolic)
            
        return self.decoder.predict(hyperbolic)
    
    def get_representations(self, inputs):
        """Get hyperbolic embeddings."""
        embeddings = self.graph_encoder.predict(inputs)
        return self.hyperbolic_embedder.predict(embeddings)


class MAMLBaseline(BaselineModel):
    """
    Baseline 4: Model-Agnostic Meta-Learning (MAML)
    
    Learns to quickly adapt to new tasks/distributions with few examples.
    Most relevant baseline as it explicitly handles adaptation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.inner_lr = config.get('inner_lr', 0.01)
        self.inner_steps = config.get('inner_steps', 5)
        self.meta_batch_size = config.get('meta_batch_size', 32)
        
    def build_model(self):
        """Build MAML-compatible model."""
        # Standard model architecture, but trained with MAML
        if self.config['task_type'] == 'physics':
            self.model = self._build_physics_model()
        elif self.config['task_type'] == 'language':
            self.model = self._build_language_model()
        else:
            raise ValueError(f"Unknown task type: {self.config['task_type']}")
            
        # Create a copy for inner loop updates
        self.model_copy = keras.models.clone_model(self.model)
        
    def _build_physics_model(self):
        """Same as ERM but will be trained with MAML."""
        return keras.Sequential([
            keras.layers.Input(shape=self.config['input_shape']),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(self.config['output_shape'])
        ])
    
    def _build_language_model(self):
        """Language model for MAML."""
        return keras.Sequential([
            keras.layers.Input(shape=self.config['input_shape']),
            keras.layers.Embedding(self.config['vocab_size'], 128),
            keras.layers.LSTM(128),
            keras.layers.Dense(self.config['output_shape'])
        ])
        
    def train(self, train_data, val_data, epochs: int = 100):
        """Train using MAML algorithm."""
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        
        for epoch in range(epochs):
            # Sample batch of tasks (different distributions)
            task_batch = self.sample_task_batch(train_data, self.meta_batch_size)
            
            meta_gradients = []
            
            for task in task_batch:
                # Split task data into support and query
                support_data, query_data = self.split_task_data(task)
                
                # Inner loop: adapt to support data
                adapted_weights = self.inner_loop_update(
                    self.model.get_weights(),
                    support_data
                )
                
                # Compute loss on query data with adapted weights
                self.model_copy.set_weights(adapted_weights)
                query_loss = self.compute_loss(self.model_copy, query_data)
                
                # Compute gradients w.r.t original weights
                grads = self.compute_meta_gradients(query_loss)
                meta_gradients.append(grads)
            
            # Meta update
            averaged_gradients = self.average_gradients(meta_gradients)
            optimizer.apply_gradients(
                zip(averaged_gradients, self.model.trainable_variables)
            )
            
        self.trained = True
        
    def inner_loop_update(self, weights, support_data):
        """Fast adaptation on support set."""
        self.model_copy.set_weights(weights)
        
        for _ in range(self.inner_steps):
            with tf.GradientTape() as tape:
                predictions = self.model_copy(support_data['x'])
                loss = keras.losses.mse(support_data['y'], predictions)
                
            gradients = tape.gradient(loss, self.model_copy.trainable_variables)
            
            # Manual gradient descent step
            updated_weights = []
            for w, g in zip(self.model_copy.get_weights(), gradients):
                updated_weights.append(w - self.inner_lr * g)
                
            self.model_copy.set_weights(updated_weights)
            
        return self.model_copy.get_weights()
    
    def adapt_to_modification(self, modification_request, examples=None):
        """
        MAML excels at adaptation - this is its strength.
        """
        if examples is None:
            # MAML needs at least few examples to adapt
            print("Warning: MAML requires examples for adaptation")
            return self
            
        # Perform inner loop adaptation
        adapted_weights = self.inner_loop_update(
            self.model.get_weights(),
            examples
        )
        
        # Create adapted model
        adapted_model = keras.models.clone_model(self.model)
        adapted_model.set_weights(adapted_weights)
        
        # Store adapted model
        self.adapted_model = adapted_model
        
        return self
        
    def predict(self, inputs):
        """Use adapted model if available."""
        if hasattr(self, 'adapted_model'):
            return self.adapted_model.predict(inputs)
        return self.model.predict(inputs)
    
    def get_representations(self, inputs):
        """Get representations from adapted or base model."""
        model = self.adapted_model if hasattr(self, 'adapted_model') else self.model
        
        feature_model = keras.Model(
            inputs=model.input,
            outputs=model.layers[-2].output
        )
        return feature_model.predict(inputs)


class BaselineEvaluator:
    """
    Unified evaluation framework for all baselines and our approach.
    """
    
    def __init__(self, representation_analyzer):
        self.representation_analyzer = representation_analyzer
        self.results = {}
        
    def evaluate_all_models(self, models: Dict[str, BaselineModel], test_data, modifications):
        """Evaluate all models on same test data and modifications."""
        
        for model_name, model in models.items():
            print(f"\nEvaluating {model_name}...")
            
            # Standard accuracy
            base_accuracy = model.evaluate(test_data)
            
            # Extrapolation analysis
            extrap_results = model.evaluate_extrapolation(
                test_data, 
                self.representation_analyzer
            )
            
            # Modification success
            mod_results = self.evaluate_modifications(model, modifications)
            
            self.results[model_name] = {
                'base_accuracy': base_accuracy,
                'interpolation': extrap_results.get('interpolation', 0),
                'near_extrapolation': extrap_results.get('near_extrapolation', 0),
                'far_extrapolation': extrap_results.get('far_extrapolation', 0),
                'modification_success': mod_results
            }
            
        return self.results
    
    def evaluate_modifications(self, model, modifications):
        """Test model on various modification requests."""
        successes = []
        
        for mod in modifications:
            # Adapt model to modification
            adapted = model.adapt_to_modification(
                mod['request'],
                mod.get('examples')
            )
            
            # Test on modification test set
            predictions = adapted.predict(mod['test_inputs'])
            accuracy = self.compute_accuracy(predictions, mod['test_targets'])
            successes.append(accuracy)
            
        return np.mean(successes)
    
    def generate_report(self):
        """Generate comparison report."""
        report = "Baseline Model Comparison\n"
        report += "=" * 50 + "\n\n"
        
        for model_name, results in self.results.items():
            report += f"{model_name}:\n"
            report += f"  Base Accuracy: {results['base_accuracy']:.2%}\n"
            report += f"  Interpolation: {results['interpolation']:.2%}\n"
            report += f"  Near Extrapolation: {results['near_extrapolation']:.2%}\n"
            report += f"  Far Extrapolation: {results['far_extrapolation']:.2%}\n"
            report += f"  Modification Success: {results['modification_success']:.2%}\n\n"
            
        return report


# Usage example
if __name__ == "__main__":
    # Configuration for physics experiment
    config = {
        'task_type': 'physics',
        'input_shape': (4,),  # gravity, friction, elasticity, damping
        'output_shape': (100, 2),  # trajectory points
        'state_shape': (10,),  # for GFlowNet
        'action_space': 20,  # for GFlowNet
        'vocab_size': 1000,  # for language tasks
    }
    
    # Initialize all baselines
    baselines = {
        'ERM+Augmentation': ERMWithAugmentation(config),
        'GFlowNet': GFlowNetBaseline(config),
        'GraphExtrapolation': GraphExtrapolationBaseline(config),
        'MAML': MAMLBaseline(config)
    }
    
    # Build models
    for name, model in baselines.items():
        print(f"Building {name}...")
        model.build_model()