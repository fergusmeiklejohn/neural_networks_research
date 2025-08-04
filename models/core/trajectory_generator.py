"""
Trajectory Generator - Neural network that generates physics trajectories
under modified rule distributions.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import keras
import numpy as np
from keras import layers, ops


@dataclass
class TrajectoryConfig:
    """Configuration for trajectory generation"""

    sequence_length: int = 300
    feature_dim: int = 9  # [time, x, y, vx, vy, mass, radius, ke, pe] per ball
    max_balls: int = 4

    # Model architecture
    latent_dim: int = 128
    hidden_dim: int = 256
    num_lstm_layers: int = 3
    num_attention_heads: int = 8

    # Physics-aware components
    physics_embedding_dim: int = 64
    temporal_encoding_dim: int = 32

    # Generation parameters
    temperature: float = 1.0
    use_teacher_forcing: bool = True

    # Training parameters
    dropout_rate: float = 0.1
    learning_rate: float = 1e-4


class PhysicsAwareLSTM(layers.Layer):
    """LSTM layer that incorporates physics rules into hidden state updates"""

    def __init__(
        self, units: int, physics_dim: int, dropout_rate: float = 0.1, **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.physics_dim = physics_dim
        self.dropout_rate = dropout_rate

        # Standard LSTM
        self.lstm = layers.LSTM(units, return_sequences=True, return_state=True)

        # Physics integration layers
        self.physics_gate = keras.Sequential(
            [layers.Dense(units, activation="sigmoid"), layers.Dropout(dropout_rate)]
        )

        self.physics_transform = keras.Sequential(
            [layers.Dense(units, activation="tanh"), layers.Dropout(dropout_rate)]
        )

        # Attention between physics and trajectory
        self.physics_attention = layers.MultiHeadAttention(
            num_heads=4, key_dim=units // 4
        )

    def call(self, inputs, physics_embedding, initial_state=None, training=None):
        # Apply standard LSTM
        lstm_output, hidden_state, cell_state = self.lstm(
            inputs, initial_state=initial_state, training=training
        )

        # Integrate physics rules
        batch_size = ops.shape(lstm_output)[0]
        seq_length = ops.shape(lstm_output)[1]

        # Expand physics embedding to match the current batch size (which may include balls)
        if ops.shape(physics_embedding)[0] != batch_size:
            # Repeat physics embedding for each ball
            balls_per_batch = batch_size // ops.shape(physics_embedding)[0]
            physics_embedding = ops.repeat(physics_embedding, balls_per_batch, axis=0)

        # Broadcast physics embedding to sequence length
        physics_expanded = ops.expand_dims(
            physics_embedding, 1
        )  # [batch, 1, physics_dim]
        physics_seq = ops.repeat(
            physics_expanded, seq_length, axis=1
        )  # [batch, seq, physics_dim]

        # Apply attention between LSTM output and physics
        attended_output = self.physics_attention(
            lstm_output, physics_seq, training=training
        )

        # Physics-modulated gating
        physics_gate = self.physics_gate(physics_seq, training=training)
        physics_transform = self.physics_transform(physics_seq, training=training)

        # Combine LSTM output with physics-aware modifications
        output = (
            lstm_output * physics_gate
            + attended_output * (1 - physics_gate)
            + physics_transform
        )

        return output, hidden_state, cell_state


class TrajectoryDecoder(layers.Layer):
    """Decoder that generates trajectory points conditioned on physics rules"""

    def __init__(self, config: TrajectoryConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        # Physics embedding layer
        self.physics_encoder = keras.Sequential(
            [
                layers.Dense(config.hidden_dim, activation="relu"),
                layers.Dropout(config.dropout_rate),
                layers.Dense(config.physics_embedding_dim, activation="relu"),
            ]
        )

        # Temporal encoding
        self.temporal_embedding = layers.Embedding(
            config.sequence_length, config.temporal_encoding_dim
        )

        # Physics-aware LSTM layers
        self.lstm_layers = [
            PhysicsAwareLSTM(
                config.latent_dim, config.physics_embedding_dim, config.dropout_rate
            )
            for _ in range(config.num_lstm_layers)
        ]

        # Output projection layers for each ball feature
        self.position_predictor = layers.Dense(2, name="position_delta")  # dx, dy
        self.velocity_predictor = layers.Dense(2, name="velocity_delta")  # dvx, dvy
        self.energy_predictor = layers.Dense(2, name="energy_prediction")  # ke, pe

        # Uncertainty estimation
        self.uncertainty_predictor = layers.Dense(
            config.feature_dim - 1,  # All features except time
            activation="softplus",
            name="prediction_uncertainty",
        )

    def call(self, inputs, physics_rules, training=None):
        initial_conditions = inputs[
            "initial_conditions"
        ]  # [batch, max_balls, features]
        sequence_length = inputs.get("sequence_length", self.config.sequence_length)

        batch_size = ops.shape(initial_conditions)[0]
        max_balls = ops.shape(initial_conditions)[1]

        # Encode physics rules
        physics_embedding = self.physics_encoder(
            ops.concatenate(
                [
                    physics_rules["gravity"],
                    physics_rules["friction"],
                    physics_rules["elasticity"],
                    physics_rules["damping"],
                ],
                axis=-1,
            ),
            training=training,
        )

        # Initialize trajectory with initial conditions
        current_state = initial_conditions  # [batch, max_balls, features]
        trajectory_outputs = []
        uncertainties = []

        # Generate sequence
        for t in range(sequence_length):
            # Temporal encoding
            time_emb = self.temporal_embedding(
                ops.convert_to_tensor(t)
            )  # [temporal_dim]
            time_emb = ops.expand_dims(
                ops.expand_dims(time_emb, 0), 0
            )  # [1, 1, temporal_dim]
            time_emb = ops.tile(
                time_emb, [batch_size, max_balls, 1]
            )  # [batch, max_balls, temporal_dim]

            # Combine state with temporal encoding
            state_with_time = ops.concatenate([current_state, time_emb], axis=-1)

            # Reshape for LSTM processing: [batch, max_balls, features] -> [batch*max_balls, 1, features]
            reshaped_input = ops.reshape(
                state_with_time, [-1, 1, ops.shape(state_with_time)[-1]]
            )

            # Apply physics-aware LSTM layers
            lstm_output = reshaped_input
            for lstm_layer in self.lstm_layers:
                lstm_output, _, _ = lstm_layer(
                    lstm_output, physics_embedding=physics_embedding, training=training
                )

            # Reshape back: [batch*max_balls, 1, features] -> [batch, max_balls, features]
            feature_dim = ops.shape(lstm_output)[-1]
            lstm_output = ops.reshape(lstm_output, [batch_size, max_balls, feature_dim])

            # Predict changes
            position_delta = self.position_predictor(lstm_output, training=training)
            velocity_delta = self.velocity_predictor(lstm_output, training=training)
            energy_pred = self.energy_predictor(lstm_output, training=training)
            uncertainty = self.uncertainty_predictor(lstm_output, training=training)

            # Update state
            # current_state format: [time, x, y, vx, vy, mass, radius, ke, pe]
            new_time = ops.full(
                [batch_size, max_balls, 1],
                float(t) / float(self.config.sequence_length),
            )
            new_position = current_state[:, :, 1:3] + position_delta
            new_velocity = current_state[:, :, 3:5] + velocity_delta
            static_properties = current_state[:, :, 5:7]  # mass, radius stay constant
            new_energy = energy_pred

            new_state = ops.concatenate(
                [new_time, new_position, new_velocity, static_properties, new_energy],
                axis=-1,
            )

            trajectory_outputs.append(new_state)
            uncertainties.append(uncertainty)
            current_state = new_state

        # Stack outputs: [sequence_length, batch, max_balls, features] -> [batch, sequence_length, max_balls, features]
        trajectory = ops.stack(trajectory_outputs, axis=1)
        uncertainty_sequence = ops.stack(uncertainties, axis=1)

        return {"trajectory": trajectory, "uncertainty": uncertainty_sequence}


class TrajectoryGenerator(keras.Model):
    """Main model for generating physics trajectories under modified rules"""

    def __init__(self, config: TrajectoryConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        # Core decoder
        self.decoder = TrajectoryDecoder(config)

        # Physics consistency validator
        self.physics_validator = keras.Sequential(
            [
                layers.Dense(config.hidden_dim, activation="relu"),
                layers.Dropout(config.dropout_rate),
                layers.Dense(config.hidden_dim // 2, activation="relu"),
                layers.Dense(1, activation="sigmoid", name="physics_consistency"),
            ]
        )

        # Trajectory quality estimator
        self.quality_estimator = keras.Sequential(
            [
                layers.Dense(config.hidden_dim, activation="relu"),
                layers.Dropout(config.dropout_rate),
                layers.Dense(1, activation="sigmoid", name="trajectory_quality"),
            ]
        )

    def call(self, inputs, training=None):
        inputs["initial_conditions"]
        physics_rules = inputs["physics_rules"]

        # Generate trajectory
        generation_output = self.decoder(
            inputs, physics_rules=physics_rules, training=training
        )

        trajectory = generation_output["trajectory"]
        uncertainty = generation_output["uncertainty"]

        # Validate physics consistency
        # Flatten trajectory for validation: [batch, seq*balls*features]
        flat_trajectory = ops.reshape(trajectory, [ops.shape(trajectory)[0], -1])
        physics_consistency = self.physics_validator(flat_trajectory, training=training)

        # Estimate trajectory quality
        quality_score = self.quality_estimator(flat_trajectory, training=training)

        return {
            "trajectory": trajectory,
            "uncertainty": uncertainty,
            "physics_consistency": physics_consistency,
            "quality_score": quality_score,
        }

    def generate_trajectory(
        self,
        initial_conditions: np.ndarray,
        physics_rules: Dict[str, np.ndarray],
        sequence_length: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """Generate a trajectory given initial conditions and physics rules"""

        if sequence_length is None:
            sequence_length = self.config.sequence_length

        # Prepare inputs
        inputs = {
            "initial_conditions": ops.convert_to_tensor(
                np.expand_dims(initial_conditions, 0)
            ),
            "sequence_length": sequence_length,
            "physics_rules": {},
        }

        for key, value in physics_rules.items():
            if isinstance(value, np.ndarray):
                inputs["physics_rules"][key] = ops.convert_to_tensor(
                    np.expand_dims(value, 0)
                )
            else:
                inputs["physics_rules"][key] = ops.convert_to_tensor([[value]])

        # Generate
        outputs = self(inputs, training=False)

        # Convert to numpy
        result = {}
        for key, value in outputs.items():
            result[key] = np.array(value)[0]  # Remove batch dimension

        return result

    def sample_trajectory(
        self,
        initial_conditions: np.ndarray,
        physics_rules: Dict[str, np.ndarray],
        temperature: float = 1.0,
        num_samples: int = 1,
    ) -> List[Dict[str, np.ndarray]]:
        """Sample multiple trajectories with stochastic generation"""

        samples = []
        for _ in range(num_samples):
            # Add noise to initial conditions for stochastic sampling
            noisy_initial = initial_conditions + np.random.normal(
                0, 0.01, initial_conditions.shape
            )

            # Generate trajectory
            sample = self.generate_trajectory(noisy_initial, physics_rules)

            # Apply temperature to uncertainty
            if temperature != 1.0:
                sample["uncertainty"] *= temperature

            samples.append(sample)

        return samples


class TrajectoryGenerationLoss(keras.losses.Loss):
    """Custom loss function for trajectory generation"""

    def __init__(
        self,
        trajectory_mse_weight: float = 1.0,
        physics_consistency_weight: float = 0.5,
        energy_conservation_weight: float = 0.3,
        smoothness_weight: float = 0.2,
        uncertainty_weight: float = 0.1,
        name: str = "trajectory_generation_loss",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.trajectory_mse_weight = trajectory_mse_weight
        self.physics_consistency_weight = physics_consistency_weight
        self.energy_conservation_weight = energy_conservation_weight
        self.smoothness_weight = smoothness_weight
        self.uncertainty_weight = uncertainty_weight

    def call(self, y_true, y_pred):
        # Trajectory reconstruction loss
        trajectory_loss = keras.losses.mse(y_true["trajectory"], y_pred["trajectory"])

        # Physics consistency loss
        consistency_target = ops.ones_like(y_pred["physics_consistency"])
        consistency_loss = keras.losses.binary_crossentropy(
            consistency_target, y_pred["physics_consistency"]
        )

        # Energy conservation loss
        if "trajectory" in y_pred:
            traj = y_pred["trajectory"]
            # Extract kinetic and potential energy (last 2 features)
            ke = traj[:, :, :, -2]  # [batch, seq, balls]
            pe = traj[:, :, :, -1]
            total_energy = ke + pe

            # Energy should be approximately constant over time
            energy_variance = ops.var(total_energy, axis=1)  # Variance over time
            energy_conservation_loss = ops.mean(energy_variance)
        else:
            energy_conservation_loss = 0.0

        # Trajectory smoothness loss (penalize large jumps)
        if "trajectory" in y_pred:
            traj = y_pred["trajectory"]
            # Compute differences between consecutive time steps
            position_diff = ops.diff(traj[:, :, :, 1:3], axis=1)  # x, y positions
            velocity_diff = ops.diff(traj[:, :, :, 3:5], axis=1)  # vx, vy velocities

            smoothness_loss = ops.mean(ops.square(position_diff)) + ops.mean(
                ops.square(velocity_diff)
            )
        else:
            smoothness_loss = 0.0

        # Uncertainty calibration loss
        if "uncertainty" in y_pred and "trajectory" in y_true:
            # Uncertainty should correlate with prediction error
            pred_error = ops.abs(y_true["trajectory"] - y_pred["trajectory"])
            uncertainty = y_pred["uncertainty"]

            # Uncertainty should be higher where prediction error is higher
            uncertainty_loss = keras.losses.mse(pred_error[:, :, :, 1:-2], uncertainty)
        else:
            uncertainty_loss = 0.0

        # Combine losses
        total_loss = (
            self.trajectory_mse_weight * trajectory_loss
            + self.physics_consistency_weight * consistency_loss
            + self.energy_conservation_weight * energy_conservation_loss
            + self.smoothness_weight * smoothness_loss
            + self.uncertainty_weight * uncertainty_loss
        )

        return total_loss


def create_trajectory_generator(
    config: Optional[TrajectoryConfig] = None,
) -> TrajectoryGenerator:
    """Create and compile a trajectory generator model"""
    if config is None:
        config = TrajectoryConfig()

    model = TrajectoryGenerator(config)

    # Compile with custom loss
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=TrajectoryGenerationLoss(),
        metrics=[
            keras.metrics.MeanSquaredError(name="mse"),
            keras.metrics.MeanAbsoluteError(name="mae"),
        ],
    )

    return model


if __name__ == "__main__":
    # Test the trajectory generator
    print("Testing Trajectory Generator...")

    config = TrajectoryConfig()
    model = create_trajectory_generator(config)

    # Create dummy data
    batch_size = 2
    max_balls = 3

    # Initial conditions: [time, x, y, vx, vy, mass, radius, ke, pe]
    initial_conditions = np.random.random((batch_size, max_balls, config.feature_dim))

    # Physics rules
    physics_rules = {
        "gravity": np.random.normal(-981, 100, (batch_size, 1)),
        "friction": np.random.uniform(0.3, 0.9, (batch_size, 1)),
        "elasticity": np.random.uniform(0.5, 0.9, (batch_size, 1)),
        "damping": np.random.uniform(0.9, 0.98, (batch_size, 1)),
    }

    inputs = {
        "initial_conditions": initial_conditions,
        "physics_rules": physics_rules,
        "sequence_length": 50,  # Shorter for testing
    }

    print(f"Initial conditions shape: {initial_conditions.shape}")
    print(f"Physics rules:")
    for key, value in physics_rules.items():
        print(f"  {key}: {value.shape}")

    # Test forward pass
    outputs = model(inputs)

    print("\nModel outputs:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")

    # Test single trajectory generation
    test_initial = np.random.random((max_balls, config.feature_dim))
    test_physics = {
        "gravity": -981.0,
        "friction": 0.7,
        "elasticity": 0.8,
        "damping": 0.95,
    }

    generated = model.generate_trajectory(
        test_initial, test_physics, sequence_length=20
    )

    print(f"\nGenerated trajectory:")
    for key, value in generated.items():
        print(f"  {key}: {value.shape}")

    # Test stochastic sampling
    samples = model.sample_trajectory(test_initial, test_physics, num_samples=3)
    print(f"\nGenerated {len(samples)} trajectory samples")

    print("\nTrajectory Generator test complete!")
