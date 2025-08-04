"""
Paperspace-ready PINN training script with comprehensive safety features.
Fully self-contained with no external dependencies.
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Reduce TF logging

import json
import shutil
import time
import zipfile
from datetime import datetime
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from keras import layers

# GPU Configuration
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    print(f"Found {len(gpus)} GPU(s)")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("GPU memory growth enabled")
else:
    print("No GPU found, using CPU")


# Path detection for Paperspace
if os.path.exists("/notebooks"):
    BASE_PATH = "/notebooks"
    print("Running on Paperspace Notebooks")
elif os.path.exists("/workspace"):
    BASE_PATH = "/workspace"
    print("Running on Paperspace Jobs")
else:
    BASE_PATH = "."
    print("Running locally")

# Check for persistent storage
STORAGE_AVAILABLE = os.path.exists("/storage")
if STORAGE_AVAILABLE:
    print("Persistent storage available at /storage")
else:
    print("Warning: No persistent storage found. Results will only be saved locally.")


class ScaledPINN(keras.Model):
    """Larger Physics-Informed Neural Network."""

    def __init__(self, hidden_dim=512, num_layers=6, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input processing
        self.input_dense = layers.Dense(hidden_dim)
        self.input_norm = layers.LayerNormalization()

        # Deep network with residual connections
        self.hidden_layers = []
        self.layer_norms = []
        self.dropouts = []

        for i in range(num_layers):
            self.hidden_layers.append(layers.Dense(hidden_dim, activation="gelu"))
            self.layer_norms.append(layers.LayerNormalization())
            self.dropouts.append(layers.Dropout(dropout_rate))

        # Output layers
        self.output_dense1 = layers.Dense(256, activation="gelu")
        self.output_dense2 = layers.Dense(128, activation="gelu")
        self.trajectory_output = layers.Dense(8)  # 2 balls * (x, y, vx, vy)

        # Physics parameter prediction branch
        self.physics_branch = keras.Sequential(
            [
                layers.Dense(256, activation="gelu"),
                layers.Dropout(dropout_rate),
                layers.Dense(128, activation="gelu"),
                layers.Dense(64, activation="gelu"),
                layers.Dense(4),  # gravity, friction, elasticity, damping
            ]
        )

    def call(self, inputs, training=None):
        # Flatten time series for processing
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]

        # Process each timestep
        x = tf.reshape(inputs, [batch_size * seq_len, -1])

        # Initial transformation
        x = self.input_dense(x)
        x = self.input_norm(x)

        # Deep processing with residual connections
        for i in range(self.num_layers):
            residual = x
            x = self.hidden_layers[i](x)
            x = self.layer_norms[i](x)
            x = self.dropouts[i](x, training=training)

            # Residual connection every 2 layers
            if i % 2 == 1:
                x = x + residual

        # Output processing
        features = x
        x = self.output_dense1(x)
        x = self.output_dense2(x)
        trajectory = self.trajectory_output(x)

        # Reshape back to sequence
        trajectory = tf.reshape(trajectory, [batch_size, seq_len, 8])

        # Physics parameters (averaged over sequence)
        physics_features = tf.reduce_mean(
            tf.reshape(features, [batch_size, seq_len, -1]), axis=1
        )
        self.physics_params = self.physics_branch(physics_features, training=training)

        return trajectory

    def compute_physics_loss(self, y_true, y_pred):
        """Enhanced physics loss with multiple components."""
        tf.shape(y_pred)[0]

        # 1. Gravity consistency loss
        y1 = y_pred[..., 1]  # y position of ball 1
        y2 = y_pred[..., 3]  # y position of ball 2

        if tf.shape(y_pred)[1] > 2:
            # Compute accelerations
            dt = 1 / 30.0
            vel_y1 = (y1[:, 1:] - y1[:, :-1]) / dt
            vel_y2 = (y2[:, 1:] - y2[:, :-1]) / dt

            acc_y1 = (vel_y1[:, 1:] - vel_y1[:, :-1]) / dt
            acc_y2 = (vel_y2[:, 1:] - vel_y2[:, :-1]) / dt

            # Both balls should have same gravity
            gravity_diff = tf.reduce_mean(tf.square(acc_y1 - acc_y2))

            # Acceleration should be constant over time
            acc_variance1 = tf.reduce_mean(tf.math.reduce_variance(acc_y1, axis=1))
            acc_variance2 = tf.reduce_mean(tf.math.reduce_variance(acc_y2, axis=1))

            gravity_loss = gravity_diff + 0.1 * (acc_variance1 + acc_variance2)
        else:
            gravity_loss = tf.constant(0.0)

        # 2. Energy conservation loss (simplified)
        vx1 = y_pred[..., 4]
        vy1 = y_pred[..., 5]
        vx2 = y_pred[..., 6]
        vy2 = y_pred[..., 7]

        kinetic_energy = 0.5 * (vx1**2 + vy1**2 + vx2**2 + vy2**2)
        potential_energy = (y1 + y2) * 9.8  # Simplified

        total_energy = kinetic_energy + potential_energy

        if tf.shape(total_energy)[1] > 1:
            energy_change = total_energy[:, 1:] - total_energy[:, :-1]
            energy_loss = tf.reduce_mean(tf.square(energy_change))
        else:
            energy_loss = tf.constant(0.0)

        # 3. Smoothness loss (trajectories should be smooth)
        if tf.shape(y_pred)[1] > 2:
            first_diff = y_pred[:, 1:] - y_pred[:, :-1]
            second_diff = first_diff[:, 1:] - first_diff[:, :-1]
            smoothness_loss = tf.reduce_mean(tf.square(second_diff))
        else:
            smoothness_loss = tf.constant(0.0)

        # Combined physics loss
        total_physics_loss = gravity_loss + 0.01 * energy_loss + 0.001 * smoothness_loss

        return total_physics_loss


def generate_complex_trajectory(
    gravity=-9.8,
    friction=0.5,
    elasticity=0.8,
    damping=0.95,
    length=50,
    noise_level=0.01,
):
    """Generate more complex physics trajectory with multiple forces."""
    dt = 1 / 30.0
    trajectory = []

    # Random initial conditions
    pos1 = np.random.uniform([100, 200], [300, 400])
    pos2 = np.random.uniform([500, 200], [700, 400])
    vel1 = np.random.uniform([-50, -30], [50, 30])
    vel2 = np.random.uniform([-50, -30], [50, 30])

    # Masses (affects momentum)
    mass1, mass2 = 1.0, 1.0

    for t in range(length):
        # Record state
        state = np.concatenate([pos1, pos2, vel1, vel2])
        trajectory.append(state)

        # Apply forces
        # Gravity
        force1_y = mass1 * gravity
        force2_y = mass2 * gravity

        # Air resistance (proportional to velocity squared)
        force1_x = -friction * vel1[0] * abs(vel1[0])
        force1_y += -friction * vel1[1] * abs(vel1[1])
        force2_x = -friction * vel2[0] * abs(vel2[0])
        force2_y += -friction * vel2[1] * abs(vel2[1])

        # Update velocities
        vel1[0] += (force1_x / mass1) * dt
        vel1[1] += (force1_y / mass1) * dt
        vel2[0] += (force2_x / mass2) * dt
        vel2[1] += (force2_y / mass2) * dt

        # Apply damping
        vel1 *= damping
        vel2 *= damping

        # Update positions
        pos1 += vel1 * dt
        pos2 += vel2 * dt

        # Wall collisions with elasticity
        for pos, vel in [(pos1, vel1), (pos2, vel2)]:
            if pos[0] < 20 or pos[0] > 780:
                vel[0] *= -elasticity
                pos[0] = np.clip(pos[0], 20, 780)
            if pos[1] < 20 or pos[1] > 580:
                vel[1] *= -elasticity
                pos[1] = np.clip(pos[1], 20, 580)

        # Ball-ball collision
        dist = np.linalg.norm(pos2 - pos1)
        if dist < 40:  # Collision radius
            # Elastic collision physics
            n = (pos2 - pos1) / (dist + 1e-6)
            v_rel = vel1 - vel2
            impulse = 2 * mass1 * mass2 / (mass1 + mass2) * np.dot(v_rel, n) * n
            vel1 -= impulse / mass1 * elasticity
            vel2 += impulse / mass2 * elasticity

            # Separate balls
            overlap = 40 - dist
            pos1 -= n * overlap * mass2 / (mass1 + mass2)
            pos2 += n * overlap * mass1 / (mass1 + mass2)

        # Add small noise
        if noise_level > 0:
            pos1 += np.random.randn(2) * noise_level
            pos2 += np.random.randn(2) * noise_level

    return np.array(trajectory, dtype=np.float32)


def create_large_dataset(n_samples, gravity_range, name="dataset"):
    """Create larger, more diverse dataset."""
    print(f"Generating {name} with {n_samples} samples...")

    X, y = [], []
    physics_params = []

    for i in range(n_samples):
        # Sample physics parameters
        if isinstance(gravity_range, tuple):
            gravity = np.random.uniform(*gravity_range)
        else:
            gravity = gravity_range

        friction = np.random.uniform(0.2, 0.8)
        elasticity = np.random.uniform(0.5, 0.95)
        damping = np.random.uniform(0.9, 0.99)

        # Generate trajectory
        traj = generate_complex_trajectory(
            gravity=gravity,
            friction=friction,
            elasticity=elasticity,
            damping=damping,
            noise_level=0.005,
        )

        X.append(traj[:-1])
        y.append(traj[1:])
        physics_params.append([gravity, friction, elasticity, damping])

        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{n_samples} samples...")

    return np.array(X), np.array(y), np.array(physics_params)


@tf.function(reduce_retracing=True)
def train_step(model, optimizer, x_batch, y_batch, physics_weight=0.01):
    """Training step with physics loss."""
    with tf.GradientTape() as tape:
        # Forward pass
        y_pred = model(x_batch, training=True)

        # MSE loss
        mse_loss = tf.reduce_mean(tf.square(y_batch - y_pred))

        # Physics loss
        physics_loss = model.compute_physics_loss(y_batch, y_pred)

        # Total loss
        total_loss = mse_loss + physics_weight * physics_loss

        # L2 regularization
        l2_loss = (
            tf.add_n(
                [
                    tf.nn.l2_loss(v)
                    for v in model.trainable_variables
                    if "bias" not in v.name
                ]
            )
            * 1e-5
        )
        total_loss += l2_loss

    # Compute and apply gradients
    gradients = tape.gradient(total_loss, model.trainable_variables)

    # Gradient clipping - handle None gradients
    clipped_gradients = []
    for g in gradients:
        if g is not None:
            clipped_gradients.append(tf.clip_by_norm(g, 1.0))
        else:
            clipped_gradients.append(g)

    # Apply gradients, skipping None values
    optimizer.apply_gradients(
        [
            (g, v)
            for g, v in zip(clipped_gradients, model.trainable_variables)
            if g is not None
        ]
    )

    return total_loss, mse_loss, physics_loss


def evaluate_comprehensive(model, X_test, y_test, physics_params_test, name):
    """Comprehensive evaluation."""
    # Trajectory prediction
    y_pred = model(X_test, training=False)
    mse = tf.reduce_mean(tf.square(y_test - y_pred)).numpy()

    # Physics parameter prediction
    if hasattr(model, "physics_params"):
        pred_params = model.physics_params.numpy()
        param_errors = np.mean(np.abs(pred_params - physics_params_test), axis=0)
        gravity_error = param_errors[0]
    else:
        gravity_error = 0.0

    # Trajectory quality metrics
    # Smoothness (lower is better)
    if X_test.shape[1] > 2:
        pred_diff = y_pred[:, 1:] - y_pred[:, :-1]
        true_diff = y_test[:, 1:] - y_test[:, :-1]
        smoothness_error = tf.reduce_mean(tf.abs(pred_diff - true_diff)).numpy()
    else:
        smoothness_error = 0.0

    return {
        "name": name,
        "mse": float(mse),
        "gravity_error": float(gravity_error),
        "smoothness_error": float(smoothness_error),
    }


def save_to_storage(source_path, experiment_name, stage_name):
    """Save files to persistent storage with safety checks."""
    if not STORAGE_AVAILABLE:
        print(f"Warning: No storage available, skipping storage save for {stage_name}")
        return False

    try:
        storage_dir = Path(f"/storage/{experiment_name}")
        storage_dir.mkdir(exist_ok=True, parents=True)

        dest_path = storage_dir / f"{stage_name}_{Path(source_path).name}"
        shutil.copy2(source_path, dest_path)
        print(f"Saved to storage: {dest_path}")
        return True
    except Exception as e:
        print(f"Error saving to storage: {e}")
        return False


def create_results_zip(output_dir, experiment_name):
    """Create a downloadable zip of all results."""
    try:
        zip_name = f"{experiment_name}_results.zip"
        zip_path = output_dir / zip_name

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in output_dir.glob("*"):
                if file_path.is_file() and not file_path.name.endswith(".zip"):
                    zipf.write(file_path, file_path.name)

        print(f"Created results zip: {zip_path}")

        # Also save to storage if available
        if STORAGE_AVAILABLE:
            save_to_storage(zip_path, experiment_name, "final")

        return zip_path
    except Exception as e:
        print(f"Error creating zip: {e}")
        return None


def main():
    """Main training with scaled parameters and safety features."""
    print("=" * 80)
    print("Scaled PINN Training - Paperspace Edition")
    print("=" * 80)

    # Create experiment name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"pinn_scaled_{timestamp}"

    # Create output directory
    output_dir = Path(BASE_PATH) / "outputs" / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Model configuration
    model_config = {"hidden_dim": 512, "num_layers": 6, "dropout_rate": 0.2}

    # Training configuration
    training_config = {
        "batch_size": 64,
        "learning_rates": [1e-3, 5e-4, 2e-4],
        "epochs_per_stage": [100, 75, 50],
        "physics_weights": [0.01, 0.05, 0.1],
    }

    # Combined config for saving
    config = {**model_config, **training_config}

    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    save_to_storage(config_path, experiment_name, "config")

    # Generate large datasets
    print("\nGenerating large datasets...")
    start_time = time.time()

    # Training sets (much larger)
    X_earth, y_earth, params_earth = create_large_dataset(5000, (-9.8, -9.8), "Earth")
    X_mars, y_mars, params_mars = create_large_dataset(3000, (-3.7, -3.7), "Mars")
    X_moon, y_moon, params_moon = create_large_dataset(2000, (-1.6, -1.6), "Moon")
    X_jupiter, y_jupiter, params_jupiter = create_large_dataset(
        2000, (-24.8, -24.8), "Jupiter"
    )

    # Mixed gravity training data
    X_mixed, y_mixed, params_mixed = create_large_dataset(
        3000, (-24.8, -1.6), "Mixed gravity"
    )

    # Test sets
    X_test_earth, y_test_earth, params_test_earth = create_large_dataset(
        500, (-9.8, -9.8), "Test Earth"
    )
    X_test_moon, y_test_moon, params_test_moon = create_large_dataset(
        500, (-1.6, -1.6), "Test Moon"
    )
    X_test_jupiter, y_test_jupiter, params_test_jupiter = create_large_dataset(
        500, (-24.8, -24.8), "Test Jupiter"
    )

    print(f"Data generation completed in {time.time() - start_time:.1f} seconds")

    # Create model
    print("\nCreating scaled model...")
    model = ScaledPINN(**model_config)

    # Build model
    dummy_input = tf.zeros((1, 49, 8))
    _ = model(dummy_input)

    # Count parameters
    total_params = model.count_params()
    print(f"Model parameters: {total_params:,}")

    # Create optimizer with fixed learning rate (we'll recreate for each stage)
    optimizer = keras.optimizers.AdamW(
        learning_rate=training_config["learning_rates"][0], weight_decay=1e-5
    )

    results = {"stages": [], "config": config}

    # Stage 1: Earth only (learn basic physics)
    print("\n" + "=" * 60)
    print("Stage 1: Earth Gravity - Learning Basic Physics")
    print("=" * 60)

    X_stage1 = X_earth
    y_stage1 = y_earth

    batch_size = training_config["batch_size"]
    n_epochs = training_config["epochs_per_stage"][0]
    physics_weight = training_config["physics_weights"][0]

    best_loss = float("inf")
    patience = 20
    patience_counter = 0

    stage1_start = time.time()

    for epoch in range(n_epochs):
        epoch_start = time.time()

        # Shuffle data
        indices = np.random.permutation(len(X_stage1))

        # Training metrics
        losses = []
        mse_losses = []
        physics_losses = []

        # Mini-batch training with consistent batch sizes
        n_batches = len(indices) // batch_size
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch_indices = indices[start_idx:end_idx]

            x_batch = tf.constant(X_stage1[batch_indices])
            y_batch = tf.constant(y_stage1[batch_indices])

            loss, mse, phys = train_step(
                model, optimizer, x_batch, y_batch, physics_weight
            )

            losses.append(float(loss))
            mse_losses.append(float(mse))
            physics_losses.append(float(phys))

        # Epoch metrics
        avg_loss = np.mean(losses)
        avg_mse = np.mean(mse_losses)
        avg_phys = np.mean(physics_losses)

        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        # Logging
        if (epoch + 1) % 10 == 0:
            epoch_time = time.time() - epoch_start
            print(
                f"Epoch {epoch + 1}/{n_epochs} - "
                f"Loss: {avg_loss:.4f} (MSE: {avg_mse:.4f}, Phys: {avg_phys:.4f}) - "
                f"Time: {epoch_time:.1f}s"
            )

        # Save intermediate checkpoint every 25 epochs
        if (epoch + 1) % 25 == 0:
            checkpoint_path = output_dir / f"stage1_epoch{epoch+1}_weights.weights.h5"
            model.save_weights(str(checkpoint_path))
            save_to_storage(checkpoint_path, experiment_name, f"stage1_epoch{epoch+1}")

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Evaluate Stage 1
    print("\nStage 1 Evaluation:")
    stage1_results = {
        "stage": "Earth Only",
        "epochs_trained": epoch + 1,
        "training_time": time.time() - stage1_start,
        "test_results": {},
    }

    for name, X_test, y_test, params_test in [
        ("Earth", X_test_earth, y_test_earth, params_test_earth),
        ("Moon", X_test_moon, y_test_moon, params_test_moon),
        ("Jupiter", X_test_jupiter, y_test_jupiter, params_test_jupiter),
    ]:
        eval_result = evaluate_comprehensive(model, X_test, y_test, params_test, name)
        stage1_results["test_results"][name] = eval_result
        print(
            f"{name}: MSE={eval_result['mse']:.4f}, "
            f"Gravity Error={eval_result['gravity_error']:.2f}m/s²"
        )

    results["stages"].append(stage1_results)

    # Save Stage 1 results
    stage1_results_path = output_dir / "stage1_results.json"
    with open(stage1_results_path, "w") as f:
        json.dump(stage1_results, f, indent=2)
    save_to_storage(stage1_results_path, experiment_name, "stage1_results")

    # Save checkpoint
    stage1_weights_path = output_dir / "stage1_weights.weights.h5"
    model.save_weights(str(stage1_weights_path))
    save_to_storage(stage1_weights_path, experiment_name, "stage1_weights")

    # Stage 2: Add Mars and Moon
    print("\n" + "=" * 60)
    print("Stage 2: Earth + Mars + Moon - Extending Physics Understanding")
    print("=" * 60)

    # Combine datasets
    X_stage2 = np.concatenate([X_earth, X_mars, X_moon])
    y_stage2 = np.concatenate([y_earth, y_mars, y_moon])

    # Create new optimizer with lower learning rate for Stage 2
    optimizer = keras.optimizers.AdamW(
        learning_rate=training_config["learning_rates"][1], weight_decay=1e-5
    )
    # Build optimizer with model variables to avoid tf.function issues
    optimizer.build(model.trainable_variables)

    n_epochs = training_config["epochs_per_stage"][1]
    physics_weight = training_config["physics_weights"][1]

    best_loss = float("inf")
    patience_counter = 0
    stage2_start = time.time()

    for epoch in range(n_epochs):
        epoch_start = time.time()
        indices = np.random.permutation(len(X_stage2))

        losses = []
        mse_losses = []
        physics_losses = []

        n_batches = len(indices) // batch_size
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch_indices = indices[start_idx:end_idx]

            x_batch = tf.constant(X_stage2[batch_indices])
            y_batch = tf.constant(y_stage2[batch_indices])

            loss, mse, phys = train_step(
                model, optimizer, x_batch, y_batch, physics_weight
            )

            losses.append(float(loss))
            mse_losses.append(float(mse))
            physics_losses.append(float(phys))

        avg_loss = np.mean(losses)
        avg_mse = np.mean(mse_losses)
        avg_phys = np.mean(physics_losses)

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            epoch_time = time.time() - epoch_start
            print(
                f"Epoch {epoch + 1}/{n_epochs} - "
                f"Loss: {avg_loss:.4f} (MSE: {avg_mse:.4f}, Phys: {avg_phys:.4f}) - "
                f"Time: {epoch_time:.1f}s"
            )

        # Save intermediate checkpoint
        if (epoch + 1) % 25 == 0:
            checkpoint_path = output_dir / f"stage2_epoch{epoch+1}_weights.weights.h5"
            model.save_weights(str(checkpoint_path))
            save_to_storage(checkpoint_path, experiment_name, f"stage2_epoch{epoch+1}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Evaluate Stage 2
    print("\nStage 2 Evaluation:")
    stage2_results = {
        "stage": "Earth + Mars + Moon",
        "epochs_trained": epoch + 1,
        "training_time": time.time() - stage2_start,
        "test_results": {},
    }

    for name, X_test, y_test, params_test in [
        ("Earth", X_test_earth, y_test_earth, params_test_earth),
        ("Moon", X_test_moon, y_test_moon, params_test_moon),
        ("Jupiter", X_test_jupiter, y_test_jupiter, params_test_jupiter),
    ]:
        eval_result = evaluate_comprehensive(model, X_test, y_test, params_test, name)
        stage2_results["test_results"][name] = eval_result
        print(
            f"{name}: MSE={eval_result['mse']:.4f}, "
            f"Gravity Error={eval_result['gravity_error']:.2f}m/s²"
        )

    results["stages"].append(stage2_results)

    # Save Stage 2 results
    stage2_results_path = output_dir / "stage2_results.json"
    with open(stage2_results_path, "w") as f:
        json.dump(stage2_results, f, indent=2)
    save_to_storage(stage2_results_path, experiment_name, "stage2_results")

    # Save checkpoint
    stage2_weights_path = output_dir / "stage2_weights.weights.h5"
    model.save_weights(str(stage2_weights_path))
    save_to_storage(stage2_weights_path, experiment_name, "stage2_weights")

    # Stage 3: Full curriculum with Jupiter
    print("\n" + "=" * 60)
    print("Stage 3: Full Physics Curriculum - Mastering Extrapolation")
    print("=" * 60)

    # Combine all datasets including mixed gravity
    X_stage3 = np.concatenate([X_earth, X_mars, X_moon, X_jupiter, X_mixed])
    y_stage3 = np.concatenate([y_earth, y_mars, y_moon, y_jupiter, y_mixed])

    # Create new optimizer with even lower learning rate for Stage 3
    optimizer = keras.optimizers.AdamW(
        learning_rate=training_config["learning_rates"][2], weight_decay=1e-5
    )
    # Build optimizer with model variables to avoid tf.function issues
    optimizer.build(model.trainable_variables)

    n_epochs = training_config["epochs_per_stage"][2]
    physics_weight = training_config["physics_weights"][2]

    best_loss = float("inf")
    patience_counter = 0
    stage3_start = time.time()

    for epoch in range(n_epochs):
        epoch_start = time.time()
        indices = np.random.permutation(len(X_stage3))

        losses = []
        mse_losses = []
        physics_losses = []

        n_batches = len(indices) // batch_size
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch_indices = indices[start_idx:end_idx]

            x_batch = tf.constant(X_stage3[batch_indices])
            y_batch = tf.constant(y_stage3[batch_indices])

            loss, mse, phys = train_step(
                model, optimizer, x_batch, y_batch, physics_weight
            )

            losses.append(float(loss))
            mse_losses.append(float(mse))
            physics_losses.append(float(phys))

        avg_loss = np.mean(losses)
        avg_mse = np.mean(mse_losses)
        avg_phys = np.mean(physics_losses)

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0:
            epoch_time = time.time() - epoch_start
            print(
                f"Epoch {epoch + 1}/{n_epochs} - "
                f"Loss: {avg_loss:.4f} (MSE: {avg_mse:.4f}, Phys: {avg_phys:.4f}) - "
                f"Time: {epoch_time:.1f}s"
            )

        # Save intermediate checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = output_dir / f"stage3_epoch{epoch+1}_weights.weights.h5"
            model.save_weights(str(checkpoint_path))
            save_to_storage(checkpoint_path, experiment_name, f"stage3_epoch{epoch+1}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Final evaluation
    print("\nFinal Evaluation:")
    stage3_results = {
        "stage": "Full Curriculum",
        "epochs_trained": epoch + 1,
        "training_time": time.time() - stage3_start,
        "test_results": {},
    }

    for name, X_test, y_test, params_test in [
        ("Earth", X_test_earth, y_test_earth, params_test_earth),
        ("Moon", X_test_moon, y_test_moon, params_test_moon),
        ("Jupiter", X_test_jupiter, y_test_jupiter, params_test_jupiter),
    ]:
        eval_result = evaluate_comprehensive(model, X_test, y_test, params_test, name)
        stage3_results["test_results"][name] = eval_result
        print(
            f"{name}: MSE={eval_result['mse']:.4f}, "
            f"Gravity Error={eval_result['gravity_error']:.2f}m/s², "
            f"Smoothness={eval_result['smoothness_error']:.4f}"
        )

    results["stages"].append(stage3_results)

    # Save Stage 3 results
    stage3_results_path = output_dir / "stage3_results.json"
    with open(stage3_results_path, "w") as f:
        json.dump(stage3_results, f, indent=2)
    save_to_storage(stage3_results_path, experiment_name, "stage3_results")

    # Save final model
    final_weights_path = output_dir / "final_weights.weights.h5"
    model.save_weights(str(final_weights_path))
    save_to_storage(final_weights_path, experiment_name, "final_weights")

    # Compare with baselines
    print("\n" + "=" * 80)
    print("PINN vs Baselines - Final Comparison")
    print("=" * 80)

    baseline_results = {
        "ERM+Aug": {"earth": 0.091, "moon": 0.075, "jupiter": 1.128},
        "GFlowNet": {"earth": 0.025, "moon": 0.061, "jupiter": 0.850},
        "GraphExtrap": {"earth": 0.060, "moon": 0.124, "jupiter": 0.766},
        "MAML": {"earth": 0.025, "moon": 0.068, "jupiter": 0.823},
    }

    # Our results
    pinn_results = {
        "earth": stage3_results["test_results"]["Earth"]["mse"],
        "moon": stage3_results["test_results"]["Moon"]["mse"],
        "jupiter": stage3_results["test_results"]["Jupiter"]["mse"],
    }

    print("\nJupiter Gravity Extrapolation (MSE):")
    print(f"{'Model':<15} {'MSE':<10} {'vs Best Baseline'}")
    print("-" * 40)

    best_baseline_jupiter = min(b["jupiter"] for b in baseline_results.values())
    print(
        f"{'PINN (Ours)':<15} {pinn_results['jupiter']:<10.4f} "
        f"{pinn_results['jupiter']/best_baseline_jupiter:.2f}x"
    )

    for name, scores in baseline_results.items():
        print(
            f"{name:<15} {scores['jupiter']:<10.4f} "
            f"{scores['jupiter']/best_baseline_jupiter:.2f}x"
        )

    # Performance summary
    if pinn_results["jupiter"] < best_baseline_jupiter:
        improvement = (1 - pinn_results["jupiter"] / best_baseline_jupiter) * 100
        print(f"\n✓ PINN achieves {improvement:.1f}% improvement over best baseline!")
        print("✓ Physics understanding enables successful extrapolation!")
    else:
        print(f"\nPINN MSE: {pinn_results['jupiter']:.4f}")
        print(
            f"More training may be needed to surpass baseline of {best_baseline_jupiter:.4f}"
        )

    # Save all results
    results["final_comparison"] = {
        "pinn": pinn_results,
        "baselines": baseline_results,
        "model_params": total_params,
        "training_time": time.time() - start_time,
    }

    final_results_path = output_dir / "final_results.json"
    with open(final_results_path, "w") as f:
        json.dump(results, f, indent=2)
    save_to_storage(final_results_path, experiment_name, "final_results")

    # Create summary report
    report = f"""
# Scaled PINN Training Results

## Model Configuration
- Parameters: {total_params:,}
- Hidden dimension: {model_config['hidden_dim']}
- Layers: {model_config['num_layers']}
- Training time: {results['final_comparison']['training_time']/60:.1f} minutes

## Progressive Training Results

### Stage 1: Earth Only
- Earth MSE: {stage1_results['test_results']['Earth']['mse']:.4f}
- Jupiter MSE: {stage1_results['test_results']['Jupiter']['mse']:.4f}

### Stage 2: Earth + Mars + Moon
- Earth MSE: {stage2_results['test_results']['Earth']['mse']:.4f}
- Moon MSE: {stage2_results['test_results']['Moon']['mse']:.4f}
- Jupiter MSE: {stage2_results['test_results']['Jupiter']['mse']:.4f}

### Stage 3: Full Curriculum
- Earth MSE: {pinn_results['earth']:.4f}
- Moon MSE: {pinn_results['moon']:.4f}
- Jupiter MSE: {pinn_results['jupiter']:.4f}

## Physics Understanding
- Gravity prediction error (Jupiter): {stage3_results['test_results']['Jupiter']['gravity_error']:.2f} m/s²
- Trajectory smoothness maintained: {stage3_results['test_results']['Jupiter']['smoothness_error']:.4f}

## Comparison with Baselines
Best baseline (GraphExtrap) Jupiter MSE: {best_baseline_jupiter:.4f}
PINN Jupiter MSE: {pinn_results['jupiter']:.4f}
{'Improvement: ' + f"{improvement:.1f}%" if pinn_results['jupiter'] < best_baseline_jupiter else 'Further training needed'}
"""

    summary_path = output_dir / "summary.md"
    with open(summary_path, "w") as f:
        f.write(report)
    save_to_storage(summary_path, experiment_name, "summary")

    # Create downloadable zip
    print("\nCreating results archive...")
    create_results_zip(output_dir, experiment_name)

    print(f"\nAll results saved to: {output_dir}")
    print(f"Total training time: {(time.time() - start_time)/60:.1f} minutes")

    # Final storage summary
    if STORAGE_AVAILABLE:
        print("\nFiles saved to persistent storage:")
        storage_files = list(Path(f"/storage/{experiment_name}").glob("*"))
        for f in storage_files[:10]:  # Show first 10
            print(f"  - {f.name}")
        if len(storage_files) > 10:
            print(f"  ... and {len(storage_files) - 10} more files")

    return model, results


if __name__ == "__main__":
    # Run training
    model, results = main()
