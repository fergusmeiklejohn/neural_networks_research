"""
Simplified Rule Extractor Training - Focus on physics parameter prediction only
"""

import sys

sys.path.append("../..")

import pickle

import keras
import numpy as np
from keras import callbacks, layers, ops
from tqdm import tqdm

from models.core.physics_rule_extractor import PhysicsRuleConfig


class SimplePhysicsPredictor(keras.Model):
    """Simplified physics rule predictor focused on just the 4 physics parameters"""

    def __init__(self, config: PhysicsRuleConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        # Input processing
        self.input_projection = layers.Dense(config.rule_embedding_dim)
        self.positional_embedding = layers.Embedding(
            config.sequence_length, config.rule_embedding_dim
        )

        # Transformer layers
        self.attention_layers = []
        for _ in range(config.num_transformer_layers):
            self.attention_layers.append(
                layers.MultiHeadAttention(
                    num_heads=config.num_attention_heads,
                    key_dim=config.rule_embedding_dim // config.num_attention_heads,
                )
            )
            self.attention_layers.append(layers.LayerNormalization())
            self.attention_layers.append(
                layers.Dense(config.hidden_dim, activation="relu")
            )
            self.attention_layers.append(layers.Dropout(config.dropout_rate))

        # Global pooling
        self.global_pool = layers.GlobalAveragePooling1D()

        # Output layers for physics parameters
        self.gravity_head = layers.Dense(1, name="gravity")
        self.friction_head = layers.Dense(1, name="friction")
        self.elasticity_head = layers.Dense(1, name="elasticity")
        self.damping_head = layers.Dense(1, name="damping")

    def call(self, inputs, training=None):
        # Project inputs to embedding dimension
        x = self.input_projection(inputs)

        # Add positional encoding
        seq_length = ops.shape(inputs)[1]
        positions = ops.arange(seq_length)
        pos_embeddings = self.positional_embedding(positions)
        x = x + pos_embeddings

        # Apply transformer layers
        for i in range(0, len(self.attention_layers), 4):
            # Attention
            attention_layer = self.attention_layers[i]
            norm_layer = self.attention_layers[i + 1]
            ff_layer = self.attention_layers[i + 2]
            dropout_layer = self.attention_layers[i + 3]

            attn_out = attention_layer(x, x, training=training)
            x = norm_layer(x + attn_out)
            ff_out = ff_layer(x, training=training)
            x = dropout_layer(ff_out, training=training)

        # Global pooling
        pooled = self.global_pool(x)

        # Predict physics parameters
        gravity = self.gravity_head(pooled)
        friction = self.friction_head(pooled)
        elasticity = self.elasticity_head(pooled)
        damping = self.damping_head(pooled)

        return {
            "gravity": gravity,
            "friction": friction,
            "elasticity": elasticity,
            "damping": damping,
        }


def load_and_prepare_simple_data(data_path: str, max_samples: int = None):
    """Load and prepare data for simple physics parameter prediction"""
    print(f"Loading data from {data_path}...")

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    # Filter to 2-ball samples only
    filtered_data = [sample for sample in data if sample["num_balls"] == 2]

    if max_samples:
        filtered_data = filtered_data[:max_samples]

    print(f"Using {len(filtered_data)} samples with 2 balls")

    # Prepare arrays
    trajectories = []
    labels = []

    for sample in tqdm(filtered_data, desc="Processing"):
        traj = np.array(sample["trajectory"])

        # Use first 100 frames
        if len(traj) > 100:
            traj = traj[:100]

        trajectories.append(traj)

        # Extract physics parameters
        physics = sample["physics_config"]
        labels.append(
            [
                physics["gravity"],
                physics["friction"],
                physics["elasticity"],
                physics["damping"],
            ]
        )

    X = np.array(trajectories)
    y = np.array(labels)

    print(f"Data shapes: X={X.shape}, y={y.shape}")
    return X, y


def calculate_physics_accuracy(y_true, y_pred, tolerance=0.1):
    """Calculate accuracy for physics parameter prediction"""

    # Calculate relative errors
    rel_errors = np.abs((y_pred - y_true) / (np.abs(y_true) + 1e-8))

    # Accuracy within tolerance
    accuracies = np.mean(rel_errors < tolerance, axis=0)

    param_names = ["gravity", "friction", "elasticity", "damping"]
    results = {}

    for i, param in enumerate(param_names):
        results[param] = {
            "accuracy": accuracies[i],
            "mae": np.mean(np.abs(y_pred[:, i] - y_true[:, i])),
            "rel_error": np.mean(rel_errors[:, i]),
        }

    overall_accuracy = np.mean(accuracies)
    results["overall"] = {"accuracy": overall_accuracy}

    return results


def train_simple_predictor():
    """Train simplified physics parameter predictor"""

    # Configuration
    config = PhysicsRuleConfig(
        sequence_length=100,
        feature_dim=17,
        max_balls=2,
        num_transformer_layers=2,  # Reduced for faster training
        num_attention_heads=4,  # Reduced
        hidden_dim=64,  # Reduced
        rule_embedding_dim=32,  # Reduced
        learning_rate=1e-3,  # Higher learning rate
        dropout_rate=0.1,
    )

    # Load data
    X_train, y_train = load_and_prepare_simple_data(
        "data/processed/physics_worlds/train_data.pkl", max_samples=1000
    )
    X_val, y_val = load_and_prepare_simple_data(
        "data/processed/physics_worlds/val_data.pkl", max_samples=200
    )

    # Create model
    model = SimplePhysicsPredictor(config)

    # Build model
    _ = model(X_train[:1])
    print(f"Model parameters: {model.count_params():,}")

    # Compile model with simple MSE loss
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss={
            "gravity": "mse",
            "friction": "mse",
            "elasticity": "mse",
            "damping": "mse",
        },
        metrics={
            "gravity": "mae",
            "friction": "mae",
            "elasticity": "mae",
            "damping": "mae",
        },
    )

    # Prepare targets as dict
    y_train_dict = {
        "gravity": y_train[:, 0:1],
        "friction": y_train[:, 1:2],
        "elasticity": y_train[:, 2:3],
        "damping": y_train[:, 3:4],
    }

    y_val_dict = {
        "gravity": y_val[:, 0:1],
        "friction": y_val[:, 1:2],
        "elasticity": y_val[:, 2:3],
        "damping": y_val[:, 3:4],
    }

    # Setup callbacks
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
    ]

    # Train
    print("Starting training...")
    history = model.fit(
        X_train,
        y_train_dict,
        validation_data=(X_val, y_val_dict),
        epochs=50,
        batch_size=32,
        callbacks=callbacks_list,
        verbose=1,
    )

    # Evaluate
    print("Evaluating...")
    predictions = model(X_val)

    y_pred = np.column_stack(
        [
            np.array(predictions["gravity"]),
            np.array(predictions["friction"]),
            np.array(predictions["elasticity"]),
            np.array(predictions["damping"]),
        ]
    )

    # Calculate accuracy
    accuracy_metrics = calculate_physics_accuracy(y_val, y_pred, tolerance=0.1)

    print(f"\nResults:")
    print(
        f"Overall accuracy (10% tolerance): {accuracy_metrics['overall']['accuracy']:.3f}"
    )

    for param in ["gravity", "friction", "elasticity", "damping"]:
        metrics = accuracy_metrics[param]
        print(f"{param.capitalize()}:")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  Rel Error: {metrics['rel_error']:.4f}")

    # Test on some samples
    print(f"\nSample predictions vs ground truth:")
    for i in range(min(3, len(y_val))):
        print(f"Sample {i}:")
        print(f"  Gravity: {y_pred[i,0]:.2f} vs {y_val[i,0]:.2f}")
        print(f"  Friction: {y_pred[i,1]:.4f} vs {y_val[i,1]:.4f}")
        print(f"  Elasticity: {y_pred[i,2]:.4f} vs {y_val[i,2]:.4f}")
        print(f"  Damping: {y_pred[i,3]:.4f} vs {y_val[i,3]:.4f}")

    # Check if target achieved
    target_accuracy = 0.80
    achieved = accuracy_metrics["overall"]["accuracy"] >= target_accuracy

    print(f"\n{'='*50}")
    print(f"SIMPLE RULE EXTRACTOR TRAINING COMPLETE")
    print(f"{'='*50}")
    print(f"Target accuracy: {target_accuracy:.1%}")
    print(f"Achieved accuracy: {accuracy_metrics['overall']['accuracy']:.1%}")
    print(f"Target achieved: {'✅ YES' if achieved else '❌ NO'}")

    return model, history, accuracy_metrics


if __name__ == "__main__":
    train_simple_predictor()
