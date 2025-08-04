"""
Meta-Learning Framework for Distribution Invention

Based on MLC paper insights: generates diverse tasks during training
to enable systematic generalization and controllable extrapolation.
"""

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import keras
import numpy as np


@dataclass
class TaskConfig:
    """Configuration for a generated task."""

    name: str
    parameters: Dict[str, float]
    constraints: List[str]


class PhysicsWorldGenerator:
    """Generates diverse physics worlds with different rules."""

    def __init__(
        self, base_params: Dict[str, Tuple[float, float]], modifiable_rules: List[str]
    ):
        """
        Args:
            base_params: Parameter ranges, e.g. {'gravity': (0.1, 10.0)}
            modifiable_rules: List of rules that can be modified
        """
        self.base_params = base_params
        self.modifiable_rules = modifiable_rules

    def sample_world(self) -> TaskConfig:
        """Generate a new physics world configuration."""
        # Sample parameters from ranges
        params = {}
        for param, (min_val, max_val) in self.base_params.items():
            if random.random() < 0.3:  # 30% chance to use extreme values
                params[param] = random.choice([min_val, max_val])
            else:
                params[param] = random.uniform(min_val, max_val)

        # Randomly modify some rules
        active_constraints = []
        for rule in self.modifiable_rules:
            if random.random() < 0.5:  # 50% chance to activate each rule
                active_constraints.append(rule)

        # Generate task name
        name = f"world_{hash(tuple(params.values()))}"

        return TaskConfig(name=name, parameters=params, constraints=active_constraints)


class MetaLearningFramework(keras.Model):
    """
    Meta-learning framework that adapts to diverse tasks during training.
    Inspired by MLC's approach to achieving 99.78% on SCAN.
    """

    def __init__(
        self,
        base_model: keras.Model,
        task_generator: Callable,
        adaptation_steps: int = 5,
        meta_lr: float = 0.001,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.task_generator = task_generator
        self.adaptation_steps = adaptation_steps
        self.meta_lr = meta_lr

        # Meta-optimizer for updating base model
        self.meta_optimizer = keras.optimizers.Adam(learning_rate=meta_lr)

    def adapt_to_task(
        self,
        task_config: TaskConfig,
        support_data: Tuple[np.ndarray, np.ndarray],
        loss_fn: Callable,
    ) -> keras.Model:
        """
        Adapt model to a specific task using support data.

        Args:
            task_config: Task configuration
            support_data: (inputs, targets) for adaptation
            loss_fn: Loss function for the task

        Returns:
            Adapted model
        """
        # Clone model for task-specific adaptation
        adapted_model = keras.models.clone_model(self.base_model)
        adapted_model.set_weights(self.base_model.get_weights())

        # Task-specific optimizer
        task_optimizer = keras.optimizers.Adam(learning_rate=0.01)

        # Adapt on support data
        inputs, targets = support_data
        for _ in range(self.adaptation_steps):
            with keras.ops.GradientTape() as tape:
                predictions = adapted_model(inputs, training=True)
                loss = loss_fn(targets, predictions)

            gradients = tape.gradient(loss, adapted_model.trainable_variables)
            task_optimizer.apply_gradients(
                zip(gradients, adapted_model.trainable_variables)
            )

        return adapted_model

    def meta_train_step(
        self,
        query_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
        support_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
        loss_fn: Callable,
    ) -> Dict[str, float]:
        """
        Perform one meta-training step across multiple tasks.

        Args:
            query_data: {task_name: (inputs, targets)} for evaluation
            support_data: {task_name: (inputs, targets)} for adaptation
            loss_fn: Loss function

        Returns:
            Dictionary of losses per task
        """
        task_losses = {}

        with keras.ops.GradientTape() as meta_tape:
            meta_loss = 0.0

            for task_name in query_data:
                # Generate task config
                task_config = self.task_generator()

                # Adapt to task using support data
                adapted_model = self.adapt_to_task(
                    task_config, support_data[task_name], loss_fn
                )

                # Evaluate on query data
                query_inputs, query_targets = query_data[task_name]
                query_predictions = adapted_model(query_inputs, training=False)
                task_loss = loss_fn(query_targets, query_predictions)

                task_losses[task_name] = float(task_loss)
                meta_loss += task_loss

            # Average loss across tasks
            meta_loss = meta_loss / len(query_data)

        # Update base model
        meta_gradients = meta_tape.gradient(
            meta_loss, self.base_model.trainable_variables
        )
        self.meta_optimizer.apply_gradients(
            zip(meta_gradients, self.base_model.trainable_variables)
        )

        return task_losses

    def generate_diverse_batch(
        self, n_tasks: int, samples_per_task: int, data_generator: Callable
    ) -> Tuple[Dict, Dict]:
        """
        Generate diverse batch of tasks for meta-training.

        Args:
            n_tasks: Number of tasks to generate
            samples_per_task: Samples per task
            data_generator: Function to generate data given task config

        Returns:
            support_data, query_data dictionaries
        """
        support_data = {}
        query_data = {}

        for i in range(n_tasks):
            # Generate new task
            task_config = self.task_generator()
            task_name = f"task_{i}"

            # Generate data for this task
            all_data = data_generator(task_config, samples_per_task * 2)
            inputs, targets = all_data

            # Split into support and query
            split_idx = samples_per_task
            support_data[task_name] = (inputs[:split_idx], targets[:split_idx])
            query_data[task_name] = (inputs[split_idx:], targets[split_idx:])

        return support_data, query_data


def create_physics_meta_learner(
    base_model: keras.Model,
    physics_params: Dict[str, Tuple[float, float]],
    modifiable_rules: List[str],
) -> MetaLearningFramework:
    """
    Create a meta-learning framework for physics worlds.

    Args:
        base_model: Base neural network model
        physics_params: Parameter ranges for physics worlds
        modifiable_rules: List of physics rules that can be modified

    Returns:
        Configured MetaLearningFramework
    """
    # Create physics world generator
    world_generator = PhysicsWorldGenerator(physics_params, modifiable_rules)

    # Create meta-learning framework
    meta_learner = MetaLearningFramework(
        base_model=base_model,
        task_generator=world_generator.sample_world,
        adaptation_steps=5,
        meta_lr=0.001,
    )

    return meta_learner


# Example usage for physics extrapolation
if __name__ == "__main__":
    # Define physics parameter ranges
    physics_params = {
        "gravity": (0.1, 20.0),
        "friction": (0.0, 1.0),
        "elasticity": (0.0, 1.0),
        "air_resistance": (0.0, 0.5),
    }

    # Define modifiable rules
    rules = [
        "conservation_of_energy",
        "conservation_of_momentum",
        "non_penetration",
        "gravity_direction",
    ]

    # Create base model (placeholder)
    base_model = keras.Sequential(
        [
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(2),  # x, y position prediction
        ]
    )

    # Create meta-learner
    meta_learner = create_physics_meta_learner(base_model, physics_params, rules)

    print("Meta-learning framework created successfully!")
    print(f"Parameter ranges: {physics_params}")
    print(f"Modifiable rules: {rules}")
