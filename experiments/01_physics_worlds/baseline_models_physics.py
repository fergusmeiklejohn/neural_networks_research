"""Concrete implementations of baseline models for physics prediction."""

from typing import Any, Dict

import keras
import numpy as np
import tensorflow as tf
from keras import layers


class PhysicsGFlowNetBaseline:
    """GFlowNet baseline adapted for physics prediction.

    GFlowNets explore the space of possible trajectories, which could
    help with OOD generalization by discovering diverse physics behaviors.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.input_dim = 8  # Physics state dimension
        self.output_steps = 10
        self.output_dim = 8
        self.exploration_bonus = config.get("exploration_bonus", 0.1)
        self.flow_steps = config.get("flow_steps", 5)
        self.trained = False

    def build_model(self):
        """Build physics predictor with exploration capabilities."""
        # Main prediction network
        self.predictor = keras.Sequential(
            [
                layers.Input(shape=(1, self.input_dim)),
                layers.Flatten(),
                layers.Dense(256, activation="relu"),
                layers.BatchNormalization(),
                layers.Dense(256, activation="relu"),
                layers.BatchNormalization(),
                layers.Dense(128, activation="relu"),
                layers.Dense(self.output_steps * self.output_dim),
                layers.Reshape((self.output_steps, self.output_dim)),
            ],
            name="predictor",
        )

        # Exploration network (generates trajectory variations)
        self.explorer = keras.Sequential(
            [
                layers.Input(shape=(self.output_steps, self.output_dim)),
                layers.Flatten(),
                layers.Dense(128, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(self.output_steps * self.output_dim),
                layers.Reshape((self.output_steps, self.output_dim)),
            ],
            name="explorer",
        )

        # Discriminator (evaluates trajectory quality)
        self.discriminator = keras.Sequential(
            [
                layers.Input(shape=(self.output_steps, self.output_dim)),
                layers.Flatten(),
                layers.Dense(128, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(1, activation="sigmoid"),
            ],
            name="discriminator",
        )

        # Compile models
        self.predictor.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

        self.explorer.compile(optimizer=keras.optimizers.Adam(1e-4), loss="mse")

        self.discriminator.compile(
            optimizer=keras.optimizers.Adam(1e-4), loss="binary_crossentropy"
        )

    def train(self, train_data, val_data, epochs=50, exploration_steps=100):
        """Train with exploration-based learning."""
        X_train, y_train = train_data
        X_val, y_val = val_data

        # Phase 1: Train basic predictor
        print("Phase 1: Training basic predictor...")
        history = self.predictor.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs // 2,
            batch_size=32,
            verbose=1,
        )

        # Phase 2: Exploration-based refinement
        print("\nPhase 2: Exploration-based refinement...")
        for step in range(exploration_steps):
            # Generate diverse trajectories through exploration
            batch_idx = np.random.choice(len(X_train), 32)
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]

            # Get base predictions
            base_pred = self.predictor.predict(X_batch, verbose=0)

            # Generate variations
            noise = np.random.normal(0, self.exploration_bonus, base_pred.shape)
            explored = base_pred + self.explorer.predict(base_pred + noise, verbose=0)

            # Train discriminator to distinguish real from explored
            real_labels = np.ones((len(y_batch), 1))
            fake_labels = np.zeros((len(explored), 1))

            d_loss_real = self.discriminator.train_on_batch(y_batch, real_labels)
            d_loss_fake = self.discriminator.train_on_batch(explored, fake_labels)

            # Train predictor to fool discriminator (generate realistic trajectories)
            if step % 10 == 0:
                gen_labels = np.ones((len(X_batch), 1))
                pred_for_disc = self.predictor.predict(X_batch, verbose=0)
                g_loss = self.discriminator.train_on_batch(pred_for_disc, gen_labels)

                # Also maintain prediction accuracy
                p_loss = self.predictor.train_on_batch(X_batch, y_batch)

                if step % 50 == 0:
                    print(
                        f"Step {step}: D_loss={d_loss_real:.4f}/{d_loss_fake:.4f}, "
                        f"G_loss={g_loss:.4f}, P_loss={p_loss:.4f}"
                    )

        self.trained = True
        return history

    def predict(self, inputs):
        """Generate predictions with optional exploration."""
        base_predictions = self.predictor.predict(inputs)

        # During test time, we can optionally add controlled exploration
        if hasattr(self, "test_time_exploration") and self.test_time_exploration:
            variations = []
            for _ in range(5):
                noise = np.random.normal(0, 0.05, base_predictions.shape)
                varied = base_predictions + self.explorer.predict(
                    base_predictions + noise
                )
                variations.append(varied)
            # Return mean of variations
            return np.mean(variations, axis=0)

        return base_predictions

    def adapt_to_task(self, support_x, support_y):
        """Adapt using exploration on new task."""
        # Use explorer network to find better predictions
        self.test_time_exploration = True
        return self


class PhysicsMAMLBaseline:
    """MAML baseline for physics prediction.

    Learns to quickly adapt to new physics laws with few examples.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.input_dim = 8
        self.output_steps = 10
        self.output_dim = 8
        self.inner_lr = config.get("inner_lr", 0.01)
        self.inner_steps = config.get("inner_steps", 5)
        self.meta_batch_size = config.get("meta_batch_size", 16)
        self.trained = False

    def build_model(self):
        """Build MAML-compatible model."""
        # Simple model that can be quickly adapted
        self.model = keras.Sequential(
            [
                layers.Input(shape=(1, self.input_dim)),
                layers.Flatten(),
                layers.Dense(128, activation="relu", name="dense1"),
                layers.Dense(128, activation="relu", name="dense2"),
                layers.Dense(64, activation="relu", name="dense3"),
                layers.Dense(self.output_steps * self.output_dim, name="output"),
                layers.Reshape((self.output_steps, self.output_dim)),
            ]
        )

        self.model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

        # Create a copy for inner loop
        self.model_copy = keras.models.clone_model(self.model)
        self.model_copy.compile(
            optimizer=keras.optimizers.SGD(self.inner_lr), loss="mse"
        )

    def create_physics_tasks(self, X, y, num_tasks):
        """Create meta-learning tasks by varying physics parameters."""
        tasks = []
        samples_per_task = len(X) // num_tasks

        for i in range(num_tasks):
            start_idx = i * samples_per_task
            end_idx = (i + 1) * samples_per_task

            # Add slight variations to simulate different physics
            task_X = X[start_idx:end_idx]
            task_y = y[start_idx:end_idx]

            # Simulate different gravity/friction by scaling velocities
            if i % 3 == 1:
                # Higher gravity
                task_y = task_y.copy()
                task_y[:, :, [5, 7]] *= 1.2  # Scale y-velocities
            elif i % 3 == 2:
                # Lower gravity
                task_y = task_y.copy()
                task_y[:, :, [5, 7]] *= 0.8

            tasks.append(
                {
                    "support_x": task_X[: samples_per_task // 2],
                    "support_y": task_y[: samples_per_task // 2],
                    "query_x": task_X[samples_per_task // 2 :],
                    "query_y": task_y[samples_per_task // 2 :],
                }
            )

        return tasks

    def train(self, train_data, val_data, epochs=50, tasks_per_epoch=100):
        """Train using MAML algorithm."""
        X_train, y_train = train_data
        X_val, y_val = val_data

        optimizer = keras.optimizers.Adam(1e-3)

        for epoch in range(epochs):
            epoch_loss = 0

            # Create tasks for this epoch
            tasks = self.create_physics_tasks(X_train, y_train, self.meta_batch_size)

            for task in tasks:
                # Save current weights
                original_weights = self.model.get_weights()

                # Inner loop: adapt on support set
                self.model_copy.set_weights(original_weights)
                for _ in range(self.inner_steps):
                    self.model_copy.train_on_batch(task["support_x"], task["support_y"])

                # Evaluate on query set with adapted model
                adapted_weights = self.model_copy.get_weights()
                self.model.set_weights(adapted_weights)

                with tf.GradientTape() as tape:
                    query_pred = self.model(task["query_x"], training=True)
                    query_loss = tf.reduce_mean(tf.square(query_pred - task["query_y"]))

                # Compute gradients w.r.t original weights
                self.model.set_weights(original_weights)
                grads = tape.gradient(query_loss, self.model.trainable_variables)

                # Meta-update
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                epoch_loss += query_loss.numpy()

            # Validation
            if epoch % 10 == 0:
                val_loss = self.model.evaluate(X_val, y_val, verbose=0)
                print(
                    f"Epoch {epoch}: Meta Loss = {epoch_loss/len(tasks):.4f}, "
                    f"Val Loss = {val_loss:.4f}"
                )

        self.trained = True

    def adapt_to_task(self, support_x, support_y):
        """Adapt to new task using inner loop updates."""
        # Clone current model for adaptation
        adapted_model = keras.models.clone_model(self.model)
        adapted_model.set_weights(self.model.get_weights())
        adapted_model.compile(optimizer=keras.optimizers.SGD(self.inner_lr), loss="mse")

        # Inner loop adaptation
        for _ in range(self.inner_steps * 2):  # More steps for test time
            adapted_model.train_on_batch(support_x, support_y)

        return adapted_model

    def predict(self, inputs):
        """Standard prediction."""
        return self.model.predict(inputs)


class PhysicsGraphExtrapolationBaseline:
    """Graph Extrapolation baseline for physics.

    Uses geometric features (distances, angles) that might extrapolate better.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.use_geometric_features = config.get("use_geometric_features", True)
        self.trained = False

    def build_model(self):
        """Build model with geometric feature extraction."""
        # Input: [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
        inputs = layers.Input(shape=(1, 8))

        # Extract geometric features
        if self.use_geometric_features:
            # Flatten for processing
            flat = layers.Flatten()(inputs)

            # Original features
            orig_features = flat

            # Geometric features (computed in Lambda layer)
            geom_features = layers.Lambda(
                self.compute_geometric_features,
                output_shape=(11,),  # Specify output shape
            )(flat)

            # Combine
            combined = layers.Concatenate()([orig_features, geom_features])
        else:
            combined = layers.Flatten()(inputs)

        # Process through network
        x = layers.Dense(256, activation="relu")(combined)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(80)(x)
        outputs = layers.Reshape((10, 8))(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"]
        )

    def compute_geometric_features(self, state):
        """Compute geometric features that might extrapolate better."""
        # state: [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
        x1, y1, x2, y2 = state[0], state[1], state[2], state[3]
        vx1, vy1, vx2, vy2 = state[4], state[5], state[6], state[7]

        # Distance between balls
        dist = tf.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + 1e-6)

        # Relative positions (normalized)
        rel_x = (x2 - x1) / (dist + 1e-6)
        rel_y = (y2 - y1) / (dist + 1e-6)

        # Relative velocities
        rel_vx = vx2 - vx1
        rel_vy = vy2 - vy1

        # Center of mass
        com_x = (x1 + x2) / 2
        com_y = (y1 + y2) / 2
        com_vx = (vx1 + vx2) / 2
        com_vy = (vy1 + vy2) / 2

        # Total kinetic energy (invariant)
        ke = 0.5 * (vx1**2 + vy1**2 + vx2**2 + vy2**2)

        # Angular momentum around COM
        r1x = x1 - com_x
        r1y = y1 - com_y
        L1 = r1x * vy1 - r1y * vx1

        r2x = x2 - com_x
        r2y = y2 - com_y
        L2 = r2x * vy2 - r2y * vx2
        L_total = L1 + L2

        return tf.stack(
            [
                dist,
                rel_x,
                rel_y,
                rel_vx,
                rel_vy,
                com_x,
                com_y,
                com_vx,
                com_vy,
                ke,
                L_total,
            ]
        )

    def train(self, train_data, val_data, epochs=50):
        """Standard training."""
        X_train, y_train = train_data
        X_val, y_val = val_data

        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            verbose=1,
        )

        self.trained = True
        return history

    def predict(self, inputs):
        """Standard prediction."""
        return self.model.predict(inputs)

    def adapt_to_task(self, support_x, support_y):
        """No specific adaptation for this baseline."""
        return self
