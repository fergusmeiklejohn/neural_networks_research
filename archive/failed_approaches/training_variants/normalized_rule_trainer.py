"""
Normalized Rule Extractor Training - Proper normalization for stable training
"""

import os
import sys

sys.path.append("../..")

import pickle

import keras
import numpy as np
from keras import callbacks, layers
from tqdm import tqdm

from models.core.physics_rule_extractor import PhysicsRuleConfig


class NormalizedPhysicsPredictor(keras.Model):
    """Physics predictor with proper normalization"""

    def __init__(self, config: PhysicsRuleConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        # Simple architecture
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(256, activation="relu")
        self.dropout1 = layers.Dropout(0.2)
        self.dense2 = layers.Dense(128, activation="relu")
        self.dropout2 = layers.Dropout(0.2)
        self.dense3 = layers.Dense(64, activation="relu")

        # Output layer (single output for all 4 parameters)
        self.output_layer = layers.Dense(4, name="physics_params")

    def call(self, inputs, training=None):
        x = self.flatten(inputs)
        x = self.dense1(x, training=training)
        x = self.dropout1(x, training=training)
        x = self.dense2(x, training=training)
        x = self.dropout2(x, training=training)
        x = self.dense3(x, training=training)

        # Output all 4 physics parameters
        output = self.output_layer(x, training=training)

        return output


def normalize_physics_params(params):
    """Normalize physics parameters to similar scales"""
    normalized = params.copy()

    # Gravity: scale to 0-1 range
    gravity_min, gravity_max = -1500, -200
    normalized[:, 0] = (params[:, 0] - gravity_min) / (gravity_max - gravity_min)

    # Friction: already 0-1, but ensure it
    normalized[:, 1] = np.clip(params[:, 1], 0, 1)

    # Elasticity: already 0-1, but ensure it
    normalized[:, 2] = np.clip(params[:, 2], 0, 1)

    # Damping: already close to 0-1, normalize to 0-1
    damping_min, damping_max = 0.8, 1.0
    normalized[:, 3] = (params[:, 3] - damping_min) / (damping_max - damping_min)

    return normalized


def denormalize_physics_params(normalized_params):
    """Convert normalized parameters back to original scales"""
    denormalized = normalized_params.copy()

    # Gravity: scale back
    gravity_min, gravity_max = -1500, -200
    denormalized[:, 0] = (
        normalized_params[:, 0] * (gravity_max - gravity_min) + gravity_min
    )

    # Friction: already correct scale
    denormalized[:, 1] = normalized_params[:, 1]

    # Elasticity: already correct scale
    denormalized[:, 2] = normalized_params[:, 2]

    # Damping: scale back
    damping_min, damping_max = 0.8, 1.0
    denormalized[:, 3] = (
        normalized_params[:, 3] * (damping_max - damping_min) + damping_min
    )

    return denormalized


def load_and_prepare_normalized_data(data_path: str, max_samples: int = None):
    """Load and prepare normalized data"""
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

        # Use first 50 frames and flatten (simpler approach)
        if len(traj) > 50:
            traj = traj[:50]

        trajectories.append(traj.flatten())

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

    # Normalize the trajectory data (z-score normalization)
    X = (X - X.mean()) / (X.std() + 1e-8)

    # Normalize physics parameters
    y_normalized = normalize_physics_params(y)

    print(f"Data shapes: X={X.shape}, y={y.shape}")
    print(f"X range: {X.min():.3f} to {X.max():.3f}")
    print(f"y range: {y_normalized.min():.3f} to {y_normalized.max():.3f}")

    return X, y_normalized, y  # Return both normalized and original


def calculate_physics_accuracy(y_true_original, y_pred_original, tolerance=0.1):
    """Calculate accuracy for physics parameter prediction"""

    # Calculate relative errors using original scales
    rel_errors = np.abs(
        (y_pred_original - y_true_original) / (np.abs(y_true_original) + 1e-8)
    )

    # Accuracy within tolerance
    accuracies = np.mean(rel_errors < tolerance, axis=0)

    param_names = ["gravity", "friction", "elasticity", "damping"]
    results = {}

    for i, param in enumerate(param_names):
        results[param] = {
            "accuracy": accuracies[i],
            "mae": np.mean(np.abs(y_pred_original[:, i] - y_true_original[:, i])),
            "rel_error": np.mean(rel_errors[:, i]),
        }

    overall_accuracy = np.mean(accuracies)
    results["overall"] = {"accuracy": overall_accuracy}

    return results


def train_normalized_predictor():
    """Train normalized physics parameter predictor"""

    # Configuration
    config = PhysicsRuleConfig(
        sequence_length=50,
        feature_dim=17,
        max_balls=2,
        learning_rate=1e-3,
        dropout_rate=0.2,
    )

    # Load data
    X_train, y_train_norm, y_train_orig = load_and_prepare_normalized_data(
        "data/processed/physics_worlds/train_data.pkl", max_samples=1000
    )
    X_val, y_val_norm, y_val_orig = load_and_prepare_normalized_data(
        "data/processed/physics_worlds/val_data.pkl", max_samples=200
    )

    # Create model
    model = NormalizedPhysicsPredictor(config)

    # Build model
    _ = model(X_train[:1])
    print(f"Model parameters: {model.count_params():,}")

    # Compile model with MSE loss on normalized values
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    # Setup callbacks
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6
        ),
    ]

    # Train
    print("Starting training...")
    history = model.fit(
        X_train,
        y_train_norm,
        validation_data=(X_val, y_val_norm),
        epochs=100,
        batch_size=32,
        callbacks=callbacks_list,
        verbose=1,
    )

    # Evaluate
    print("Evaluating...")
    y_pred_norm = np.array(model(X_val))

    # Denormalize predictions for evaluation
    y_pred_orig = denormalize_physics_params(y_pred_norm)

    # Calculate accuracy on original scale
    accuracy_metrics = calculate_physics_accuracy(
        y_val_orig, y_pred_orig, tolerance=0.1
    )

    print(f"\nResults on original scale:")
    print(
        f"Overall accuracy (10% tolerance): {accuracy_metrics['overall']['accuracy']:.3f}"
    )

    for param in ["gravity", "friction", "elasticity", "damping"]:
        metrics = accuracy_metrics[param]
        print(f"{param.capitalize()}:")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  MAE: {metrics['mae']:.2f}")
        print(f"  Rel Error: {metrics['rel_error']:.4f}")

    # Test on some samples
    print(f"\nSample predictions vs ground truth:")
    for i in range(min(5, len(y_val_orig))):
        print(f"Sample {i}:")
        print(f"  Gravity: {y_pred_orig[i,0]:.1f} vs {y_val_orig[i,0]:.1f}")
        print(f"  Friction: {y_pred_orig[i,1]:.3f} vs {y_val_orig[i,1]:.3f}")
        print(f"  Elasticity: {y_pred_orig[i,2]:.3f} vs {y_val_orig[i,2]:.3f}")
        print(f"  Damping: {y_pred_orig[i,3]:.3f} vs {y_val_orig[i,3]:.3f}")

    # Check if target achieved
    target_accuracy = 0.80
    achieved = accuracy_metrics["overall"]["accuracy"] >= target_accuracy

    print(f"\n{'='*60}")
    print(f"NORMALIZED RULE EXTRACTOR TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Target accuracy: {target_accuracy:.1%}")
    print(f"Achieved accuracy: {accuracy_metrics['overall']['accuracy']:.1%}")
    print(f"Target achieved: {'✅ YES' if achieved else '❌ NO'}")

    if achieved:
        print(f"\n🎉 Rule extraction pre-training successful!")

        # Save model
        os.makedirs("outputs/checkpoints", exist_ok=True)
        model.save("outputs/checkpoints/normalized_rule_extractor.keras")
        print(f"Model saved to: outputs/checkpoints/normalized_rule_extractor.keras")
    else:
        print(f"\n📈 Training completed but target not yet reached.")
        print(f"Best accuracy so far: {accuracy_metrics['overall']['accuracy']:.1%}")

    return model, history, accuracy_metrics


if __name__ == "__main__":
    train_normalized_predictor()
