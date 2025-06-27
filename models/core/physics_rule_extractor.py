"""
Physics Rule Extractor - Neural network that learns to identify and separate
causal physics rules from trajectory data.
"""

import keras
from keras import layers, ops
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class PhysicsRuleConfig:
    """Configuration for physics rule extraction"""
    sequence_length: int = 300
    feature_dim: int = 9  # [time, x, y, vx, vy, mass, radius, ke, pe] per ball
    max_balls: int = 4
    
    # Model architecture
    rule_embedding_dim: int = 64
    num_attention_heads: int = 8
    num_transformer_layers: int = 4
    hidden_dim: int = 256
    
    # Physical rules to extract
    gravity_dim: int = 1
    friction_dim: int = 1
    elasticity_dim: int = 1
    damping_dim: int = 1
    
    # Training parameters
    dropout_rate: float = 0.1
    learning_rate: float = 1e-4


class CausalAttentionLayer(layers.Layer):
    """Attention layer that identifies causal relationships in physics"""
    
    def __init__(self, num_heads: int, key_dim: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout_rate
        
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout_rate
        )
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()
        
        self.ffn = keras.Sequential([
            layers.Dense(key_dim * 4, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(key_dim)
        ])
    
    def call(self, inputs, training=None):
        # Self-attention to identify causal relationships
        attn_output = self.attention(inputs, inputs, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(out1, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2


class PhysicsRuleEncoder(layers.Layer):
    """Encodes trajectory data into physics rule representations"""
    
    def __init__(self, config: PhysicsRuleConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        
        # Input processing
        self.input_projection = layers.Dense(config.rule_embedding_dim)
        self.positional_embedding = layers.Embedding(
            config.sequence_length, config.rule_embedding_dim
        )
        
        # Causal attention layers
        self.attention_layers = [
            CausalAttentionLayer(
                num_heads=config.num_attention_heads,
                key_dim=config.rule_embedding_dim,
                dropout_rate=config.dropout_rate
            )
            for _ in range(config.num_transformer_layers)
        ]
        
        # Rule-specific heads
        self.gravity_head = layers.Dense(config.gravity_dim, name='gravity_rules')
        self.friction_head = layers.Dense(config.friction_dim, name='friction_rules')
        self.elasticity_head = layers.Dense(config.elasticity_dim, name='elasticity_rules')
        self.damping_head = layers.Dense(config.damping_dim, name='damping_rules')
        
        # Global pooling
        self.global_pool = layers.GlobalAveragePooling1D()
        
    def call(self, inputs, training=None):
        batch_size = ops.shape(inputs)[0]
        seq_length = ops.shape(inputs)[1]
        
        # Project inputs to embedding dimension
        x = self.input_projection(inputs)
        
        # Add positional encoding
        positions = ops.arange(seq_length)
        pos_embeddings = self.positional_embedding(positions)
        x = x + pos_embeddings
        
        # Apply causal attention layers
        for attention_layer in self.attention_layers:
            x = attention_layer(x, training=training)
        
        # Global pooling to get sequence-level representation
        pooled = self.global_pool(x)
        
        # Extract different physics rules
        gravity_rules = self.gravity_head(pooled)
        friction_rules = self.friction_head(pooled)
        elasticity_rules = self.elasticity_head(pooled)
        damping_rules = self.damping_head(pooled)
        
        return {
            'gravity': gravity_rules,
            'friction': friction_rules,
            'elasticity': elasticity_rules,
            'damping': damping_rules,
            'features': pooled
        }


class PhysicsRuleExtractor(keras.Model):
    """Main model for extracting physics rules from trajectory data"""
    
    def __init__(self, config: PhysicsRuleConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        
        # Core encoder
        self.rule_encoder = PhysicsRuleEncoder(config)
        
        # Rule independence validator
        self.independence_validator = keras.Sequential([
            layers.Dense(config.hidden_dim, activation='relu'),
            layers.Dropout(config.dropout_rate),
            layers.Dense(config.hidden_dim // 2, activation='relu'),
            layers.Dense(1, activation='sigmoid', name='independence_score')
        ])
        
        # Consistency checker
        self.consistency_checker = keras.Sequential([
            layers.Dense(config.hidden_dim, activation='relu'),
            layers.Dropout(config.dropout_rate),
            layers.Dense(1, activation='sigmoid', name='consistency_score')
        ])
        
    def call(self, inputs, training=None):
        # Extract rules from trajectory
        rule_outputs = self.rule_encoder(inputs, training=training)
        
        # Check rule independence (are rules properly separated?)
        combined_rules = ops.concatenate([
            rule_outputs['gravity'],
            rule_outputs['friction'],
            rule_outputs['elasticity'],
            rule_outputs['damping']
        ], axis=-1)
        
        independence_score = self.independence_validator(combined_rules, training=training)
        
        # Check consistency (do extracted rules make physical sense?)
        consistency_score = self.consistency_checker(rule_outputs['features'], training=training)
        
        return {
            **rule_outputs,
            'independence_score': independence_score,
            'consistency_score': consistency_score
        }
    
    def extract_rules(self, trajectory_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract physics rules from trajectory data"""
        outputs = self(trajectory_data, training=False)
        
        return {
            'gravity': np.array(outputs['gravity']),
            'friction': np.array(outputs['friction']),
            'elasticity': np.array(outputs['elasticity']),
            'damping': np.array(outputs['damping']),
            'independence_score': np.array(outputs['independence_score']),
            'consistency_score': np.array(outputs['consistency_score'])
        }
    
    def get_rule_attention_weights(self, trajectory_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Get attention weights to understand which parts of trajectory influence each rule"""
        # This would require modifying the attention layers to return weights
        # For now, return placeholder
        return {
            'gravity_attention': np.zeros((trajectory_data.shape[0], trajectory_data.shape[1])),
            'friction_attention': np.zeros((trajectory_data.shape[0], trajectory_data.shape[1])),
            'elasticity_attention': np.zeros((trajectory_data.shape[0], trajectory_data.shape[1])),
            'damping_attention': np.zeros((trajectory_data.shape[0], trajectory_data.shape[1]))
        }


class PhysicsRuleLoss(keras.losses.Loss):
    """Custom loss function for physics rule extraction"""
    
    def __init__(self, 
                 reconstruction_weight: float = 1.0,
                 independence_weight: float = 0.5,
                 consistency_weight: float = 0.3,
                 physical_plausibility_weight: float = 0.2,
                 name: str = "physics_rule_loss",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.reconstruction_weight = reconstruction_weight
        self.independence_weight = independence_weight
        self.consistency_weight = consistency_weight
        self.physical_plausibility_weight = physical_plausibility_weight
    
    def call(self, y_true, y_pred):
        # y_true should contain ground truth physics parameters
        # y_pred contains the model outputs
        
        # Reconstruction loss - how well do extracted rules match ground truth?
        gravity_loss = keras.losses.mean_squared_error(y_true['gravity'], y_pred['gravity'])
        friction_loss = keras.losses.mean_squared_error(y_true['friction'], y_pred['friction'])
        elasticity_loss = keras.losses.mean_squared_error(y_true['elasticity'], y_pred['elasticity'])
        damping_loss = keras.losses.mean_squared_error(y_true['damping'], y_pred['damping'])
        
        reconstruction_loss = (gravity_loss + friction_loss + elasticity_loss + damping_loss) / 4
        
        # Independence loss - encourage rules to be independent
        independence_loss = keras.losses.binary_crossentropy(
            ops.ones_like(y_pred['independence_score']), 
            y_pred['independence_score']
        )
        
        # Consistency loss - encourage physically consistent rules
        consistency_loss = keras.losses.binary_crossentropy(
            ops.ones_like(y_pred['consistency_score']),
            y_pred['consistency_score']
        )
        
        # Physical plausibility - penalize unphysical parameter combinations
        gravity_penalty = ops.maximum(0.0, ops.abs(y_pred['gravity']) - 2000)  # Gravity shouldn't be too extreme
        friction_penalty = ops.maximum(0.0, y_pred['friction'] - 1.0) + ops.maximum(0.0, -y_pred['friction'])  # Friction: 0-1
        elasticity_penalty = ops.maximum(0.0, y_pred['elasticity'] - 1.0) + ops.maximum(0.0, -y_pred['elasticity'])  # Elasticity: 0-1
        
        plausibility_loss = ops.mean(gravity_penalty + friction_penalty + elasticity_penalty)
        
        # Combine losses
        total_loss = (
            self.reconstruction_weight * reconstruction_loss +
            self.independence_weight * independence_loss +
            self.consistency_weight * consistency_loss +
            self.physical_plausibility_weight * plausibility_loss
        )
        
        return total_loss


def create_physics_rule_extractor(config: Optional[PhysicsRuleConfig] = None) -> PhysicsRuleExtractor:
    """Create and compile a physics rule extractor model"""
    if config is None:
        config = PhysicsRuleConfig()
    
    model = PhysicsRuleExtractor(config)
    
    # Compile with custom loss  
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=PhysicsRuleLoss(),
        metrics={
            'gravity': [keras.metrics.MeanSquaredError(name='gravity_mse')],
            'friction': [keras.metrics.MeanSquaredError(name='friction_mse')],
            'elasticity': [keras.metrics.MeanSquaredError(name='elasticity_mse')],
            'damping': [keras.metrics.MeanSquaredError(name='damping_mse')],
            'independence_score': [keras.metrics.BinaryAccuracy(name='independence_acc')],
            'consistency_score': [keras.metrics.BinaryAccuracy(name='consistency_acc')],
            'features': [keras.metrics.MeanSquaredError(name='features_mse')]
        }
    )
    
    return model


def preprocess_trajectory_data(trajectory_data: List[np.ndarray], 
                              config: PhysicsRuleConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess trajectory data for training"""
    processed_sequences = []
    physics_labels = []
    
    for traj in trajectory_data:
        # Pad or truncate to fixed sequence length
        if len(traj) > config.sequence_length:
            traj = traj[:config.sequence_length]
        elif len(traj) < config.sequence_length:
            padding = np.zeros((config.sequence_length - len(traj), traj.shape[1]))
            traj = np.concatenate([traj, padding], axis=0)
        
        processed_sequences.append(traj)
    
    return np.array(processed_sequences), np.array(physics_labels)


if __name__ == "__main__":
    # Test the physics rule extractor
    print("Testing Physics Rule Extractor...")
    
    config = PhysicsRuleConfig()
    model = create_physics_rule_extractor(config)
    
    # Create dummy data
    batch_size = 4
    dummy_input = np.random.random((batch_size, config.sequence_length, config.feature_dim))
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Test forward pass
    outputs = model(dummy_input)
    
    print("Model outputs:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")
    
    # Test rule extraction
    rules = model.extract_rules(dummy_input)
    print("\nExtracted rules:")
    for key, value in rules.items():
        print(f"  {key}: {value.shape} - {value.mean():.4f} ± {value.std():.4f}")
    
    print("\nPhysics Rule Extractor test complete!")