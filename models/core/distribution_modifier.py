"""
Distribution Modifier - Neural network that modifies physics rules to create
new distributions while maintaining consistency.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import keras
import numpy as np
from keras import layers, ops


@dataclass
class ModifierConfig:
    """Configuration for distribution modification"""

    rule_embedding_dim: int = 64
    modification_embedding_dim: int = 32
    hidden_dim: int = 256
    num_attention_heads: int = 8
    num_transformer_layers: int = 3

    # Supported modification types
    modification_types: List[str] = None

    # Consistency constraints
    max_gravity_change: float = 0.5  # Maximum relative change
    max_friction_change: float = 0.3
    max_elasticity_change: float = 0.2
    max_damping_change: float = 0.1

    # Training parameters
    dropout_rate: float = 0.1
    learning_rate: float = 1e-4

    def __post_init__(self):
        if self.modification_types is None:
            self.modification_types = [
                "increase_gravity",
                "decrease_gravity",
                "increase_friction",
                "decrease_friction",
                "remove_friction",
                "increase_elasticity",
                "decrease_elasticity",
                "increase_damping",
                "decrease_damping",
            ]


class ModificationTextEncoder(layers.Layer):
    """Encodes text modification requests into embeddings"""

    def __init__(self, config: ModifierConfig, vocab_size: int = 1000, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.vocab_size = vocab_size

        # Simple text processing layers
        self.embedding = layers.Embedding(vocab_size, config.modification_embedding_dim)
        self.lstm = layers.LSTM(
            config.modification_embedding_dim, return_sequences=True
        )
        self.attention_pool = layers.GlobalAveragePooling1D()

        # Modification type classifier
        self.type_classifier = layers.Dense(
            len(config.modification_types),
            activation="softmax",
            name="modification_type",
        )

        # Modification strength predictor
        self.strength_predictor = layers.Dense(
            1, activation="sigmoid", name="modification_strength"
        )

    def call(self, inputs, training=None):
        # inputs: tokenized text sequences
        embedded = self.embedding(inputs)
        lstm_out = self.lstm(embedded, training=training)
        pooled = self.attention_pool(lstm_out)

        modification_type = self.type_classifier(pooled)
        modification_strength = self.strength_predictor(pooled)

        return {
            "type_probs": modification_type,
            "strength": modification_strength,
            "embedding": pooled,
        }


class RuleModificationLayer(layers.Layer):
    """Applies modifications to extracted physics rules"""

    def __init__(self, config: ModifierConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        # Rule-specific modification networks
        self.gravity_modifier = keras.Sequential(
            [
                layers.Dense(config.hidden_dim // 2, activation="relu"),
                layers.Dense(1, activation="tanh", name="gravity_delta"),
            ]
        )

        self.friction_modifier = keras.Sequential(
            [
                layers.Dense(config.hidden_dim // 2, activation="relu"),
                layers.Dense(1, activation="tanh", name="friction_delta"),
            ]
        )

        self.elasticity_modifier = keras.Sequential(
            [
                layers.Dense(config.hidden_dim // 2, activation="relu"),
                layers.Dense(1, activation="tanh", name="elasticity_delta"),
            ]
        )

        self.damping_modifier = keras.Sequential(
            [
                layers.Dense(config.hidden_dim // 2, activation="relu"),
                layers.Dense(1, activation="tanh", name="damping_delta"),
            ]
        )

        # Consistency enforcer
        self.consistency_enforcer = keras.Sequential(
            [
                layers.Dense(config.hidden_dim, activation="relu"),
                layers.Dropout(config.dropout_rate),
                layers.Dense(config.hidden_dim // 2, activation="relu"),
                layers.Dense(
                    4, activation="sigmoid", name="rule_weights"
                ),  # Weights for each rule
            ]
        )

    def call(
        self, base_rules, modification_embedding, modification_type, training=None
    ):
        # Combine base rules with modification request
        combined_input = ops.concatenate(
            [
                base_rules["gravity"],
                base_rules["friction"],
                base_rules["elasticity"],
                base_rules["damping"],
                modification_embedding,
            ],
            axis=-1,
        )

        # Generate rule modifications
        gravity_delta = (
            self.gravity_modifier(combined_input) * self.config.max_gravity_change
        )
        friction_delta = (
            self.friction_modifier(combined_input) * self.config.max_friction_change
        )
        elasticity_delta = (
            self.elasticity_modifier(combined_input) * self.config.max_elasticity_change
        )
        damping_delta = (
            self.damping_modifier(combined_input) * self.config.max_damping_change
        )

        # Apply consistency constraints
        rule_weights = self.consistency_enforcer(combined_input)

        # Selective modification based on type
        gravity_weight = rule_weights[:, 0:1]
        friction_weight = rule_weights[:, 1:2]
        elasticity_weight = rule_weights[:, 2:3]
        damping_weight = rule_weights[:, 3:4]

        # Apply modifications
        modified_gravity = base_rules["gravity"] + gravity_delta * gravity_weight
        modified_friction = base_rules["friction"] + friction_delta * friction_weight
        modified_elasticity = (
            base_rules["elasticity"] + elasticity_delta * elasticity_weight
        )
        modified_damping = base_rules["damping"] + damping_delta * damping_weight

        # Ensure physical constraints
        modified_friction = ops.clip(modified_friction, 0.0, 1.0)
        modified_elasticity = ops.clip(modified_elasticity, 0.0, 1.0)
        modified_damping = ops.clip(modified_damping, 0.8, 0.99)

        return {
            "gravity": modified_gravity,
            "friction": modified_friction,
            "elasticity": modified_elasticity,
            "damping": modified_damping,
            "deltas": {
                "gravity": gravity_delta,
                "friction": friction_delta,
                "elasticity": elasticity_delta,
                "damping": damping_delta,
            },
            "weights": rule_weights,
        }


class DistributionModifier(keras.Model):
    """Main model for modifying physics rule distributions"""

    def __init__(self, config: ModifierConfig, vocab_size: int = 1000, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        # Text encoder for modification requests
        self.text_encoder = ModificationTextEncoder(config, vocab_size)

        # Rule modification layer
        self.rule_modifier = RuleModificationLayer(config)

        # Consistency validator
        self.consistency_validator = keras.Sequential(
            [
                layers.Dense(config.hidden_dim, activation="relu"),
                layers.Dropout(config.dropout_rate),
                layers.Dense(1, activation="sigmoid", name="consistency_score"),
            ]
        )

        # Novelty estimator
        self.novelty_estimator = keras.Sequential(
            [
                layers.Dense(config.hidden_dim, activation="relu"),
                layers.Dropout(config.dropout_rate),
                layers.Dense(1, activation="sigmoid", name="novelty_score"),
            ]
        )

    def call(self, inputs, training=None):
        base_rules = inputs["base_rules"]
        modification_text = inputs["modification_text"]

        # Encode modification request
        text_encoding = self.text_encoder(modification_text, training=training)

        # Apply rule modifications
        modified_rules = self.rule_modifier(
            base_rules,
            text_encoding["embedding"],
            text_encoding["type_probs"],
            training=training,
        )

        # Validate consistency
        combined_modified = ops.concatenate(
            [
                modified_rules["gravity"],
                modified_rules["friction"],
                modified_rules["elasticity"],
                modified_rules["damping"],
            ],
            axis=-1,
        )

        consistency_score = self.consistency_validator(
            combined_modified, training=training
        )

        # Estimate novelty
        novelty_score = self.novelty_estimator(combined_modified, training=training)

        return {
            "modified_rules": modified_rules,
            "text_encoding": text_encoding,
            "consistency_score": consistency_score,
            "novelty_score": novelty_score,
        }

    def modify_distribution(
        self, base_rules: Dict[str, np.ndarray], modification_request: str
    ) -> Dict[str, np.ndarray]:
        """Apply distribution modification to base rules"""

        # Tokenize modification request (simplified)
        tokens = self._tokenize_request(modification_request)
        tokens = np.expand_dims(tokens, 0)  # Add batch dimension

        # Prepare base rules
        base_rules_tensor = {}
        for key, value in base_rules.items():
            if isinstance(value, np.ndarray):
                base_rules_tensor[key] = ops.convert_to_tensor(np.expand_dims(value, 0))
            else:
                base_rules_tensor[key] = ops.convert_to_tensor([[value]])

        # Apply modification
        inputs = {"base_rules": base_rules_tensor, "modification_text": tokens}

        outputs = self(inputs, training=False)

        # Extract modified rules
        modified_rules = {}
        for key, value in outputs["modified_rules"].items():
            if key != "deltas" and key != "weights":
                modified_rules[key] = np.array(value)[0]

        modified_rules["consistency_score"] = np.array(outputs["consistency_score"])[0]
        modified_rules["novelty_score"] = np.array(outputs["novelty_score"])[0]

        return modified_rules

    def _tokenize_request(self, request: str) -> np.ndarray:
        """Simple tokenization of modification requests"""
        # This is a simplified tokenizer - in practice, you'd use a proper tokenizer
        words = re.findall(r"\w+", request.lower())

        # Simple word-to-id mapping
        word_to_id = {
            "increase": 1,
            "decrease": 2,
            "remove": 3,
            "add": 4,
            "gravity": 5,
            "friction": 6,
            "elasticity": 7,
            "damping": 8,
            "bounce": 9,
            "stronger": 10,
            "weaker": 11,
            "more": 12,
            "less": 13,
            "no": 14,
            "what": 15,
            "if": 16,
            "was": 17,
            "were": 18,
            "had": 19,
            "didnt": 20,
            "exist": 21,
        }

        tokens = [word_to_id.get(word, 0) for word in words]

        # Pad or truncate to fixed length
        max_length = 20
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            tokens.extend([0] * (max_length - len(tokens)))

        return np.array(tokens, dtype=np.int32)


class ModificationLoss(keras.losses.Loss):
    """Custom loss function for distribution modification"""

    def __init__(
        self,
        modification_accuracy_weight: float = 1.0,
        consistency_weight: float = 0.5,
        novelty_weight: float = 0.3,
        physical_constraint_weight: float = 0.4,
        name: str = "modification_loss",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.modification_accuracy_weight = modification_accuracy_weight
        self.consistency_weight = consistency_weight
        self.novelty_weight = novelty_weight
        self.physical_constraint_weight = physical_constraint_weight

    def call(self, y_true, y_pred):
        # y_true should contain target modified rules and modification type
        # y_pred contains model outputs

        # Modification accuracy loss
        modification_loss = 0.0
        for rule_name in ["gravity", "friction", "elasticity", "damping"]:
            if rule_name in y_true and rule_name in y_pred["modified_rules"]:
                rule_loss = keras.losses.mse(
                    y_true[rule_name], y_pred["modified_rules"][rule_name]
                )
                modification_loss += rule_loss

        modification_loss /= 4  # Average over rules

        # Consistency loss - encourage consistent modifications
        consistency_target = ops.ones_like(y_pred["consistency_score"])
        consistency_loss = keras.losses.binary_crossentropy(
            consistency_target, y_pred["consistency_score"]
        )

        # Novelty loss - encourage appropriate novelty
        if "target_novelty" in y_true:
            novelty_loss = keras.losses.mse(
                y_true["target_novelty"], y_pred["novelty_score"]
            )
        else:
            # Default: encourage moderate novelty
            novelty_target = 0.6 * ops.ones_like(y_pred["novelty_score"])
            novelty_loss = keras.losses.mse(novelty_target, y_pred["novelty_score"])

        # Physical constraint loss
        modified_rules = y_pred["modified_rules"]
        constraint_loss = 0.0

        # Friction constraints (0-1)
        constraint_loss += ops.mean(ops.maximum(0.0, modified_rules["friction"] - 1.0))
        constraint_loss += ops.mean(ops.maximum(0.0, -modified_rules["friction"]))

        # Elasticity constraints (0-1)
        constraint_loss += ops.mean(
            ops.maximum(0.0, modified_rules["elasticity"] - 1.0)
        )
        constraint_loss += ops.mean(ops.maximum(0.0, -modified_rules["elasticity"]))

        # Damping constraints (0.8-0.99)
        constraint_loss += ops.mean(ops.maximum(0.0, modified_rules["damping"] - 0.99))
        constraint_loss += ops.mean(ops.maximum(0.0, 0.8 - modified_rules["damping"]))

        # Combine losses
        total_loss = (
            self.modification_accuracy_weight * modification_loss
            + self.consistency_weight * consistency_loss
            + self.novelty_weight * novelty_loss
            + self.physical_constraint_weight * constraint_loss
        )

        return total_loss


def create_distribution_modifier(
    config: Optional[ModifierConfig] = None, vocab_size: int = 1000
) -> DistributionModifier:
    """Create and compile a distribution modifier model"""
    if config is None:
        config = ModifierConfig()

    model = DistributionModifier(config, vocab_size)

    # Compile with custom loss
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=ModificationLoss(),
        metrics=[
            keras.metrics.MeanSquaredError(name="mse"),
            keras.metrics.BinaryAccuracy(name="consistency_acc"),
        ],
    )

    return model


if __name__ == "__main__":
    # Test the distribution modifier
    print("Testing Distribution Modifier...")

    config = ModifierConfig()
    model = create_distribution_modifier(config)

    # Create dummy data
    batch_size = 4

    # Base rules
    base_rules = {
        "gravity": np.random.normal(-981, 100, (batch_size, 1)),
        "friction": np.random.uniform(0.3, 0.9, (batch_size, 1)),
        "elasticity": np.random.uniform(0.5, 0.9, (batch_size, 1)),
        "damping": np.random.uniform(0.9, 0.98, (batch_size, 1)),
    }

    # Modification text (tokenized)
    modification_text = np.random.randint(0, 100, (batch_size, 20))

    inputs = {"base_rules": base_rules, "modification_text": modification_text}

    print(f"Base rules shapes:")
    for key, value in base_rules.items():
        print(f"  {key}: {value.shape}")
    print(f"Modification text shape: {modification_text.shape}")

    # Test forward pass
    outputs = model(inputs)

    print("\nModel outputs:")
    for key, value in outputs.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                if hasattr(sub_value, "shape"):
                    print(f"    {sub_key}: {sub_value.shape}")
        else:
            print(f"  {key}: {value.shape}")

    # Test modification function
    test_base_rules = {
        "gravity": -981.0,
        "friction": 0.7,
        "elasticity": 0.8,
        "damping": 0.95,
    }

    modification_request = "increase gravity by 20%"
    modified_rules = model.modify_distribution(test_base_rules, modification_request)

    print(f"\nTest modification:")
    print(f"Request: {modification_request}")
    print("Original rules:")
    for key, value in test_base_rules.items():
        print(f"  {key}: {value}")
    print("Modified rules:")
    for key, value in modified_rules.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value.item():.4f}")
        else:
            print(f"  {key}: {value:.4f}")

    print("\nDistribution Modifier test complete!")
