"""Improved JAX-compatible base class for Test-Time Adaptation with full gradient support."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import keras
import numpy as np
from keras import ops


class BaseTTAJaxV2(ABC):
    """Improved JAX-compatible base class for Test-Time Adaptation.

    This class provides full gradient computation support using JAX's functional
    transformations while properly handling stateful operations like BatchNorm.
    """

    def __init__(
        self,
        model: keras.Model,
        adaptation_steps: int = 1,
        learning_rate: float = 1e-3,
        reset_after_batch: bool = True,
        **kwargs,
    ):
        """Initialize the TTA wrapper.

        Args:
            model: The base model to adapt
            adaptation_steps: Number of adaptation steps per test sample/batch
            learning_rate: Learning rate for test-time updates
            reset_after_batch: Whether to reset model to original state after each batch
            **kwargs: Additional arguments for specific TTA methods
        """
        self.model = model
        self.adaptation_steps = adaptation_steps
        self.learning_rate = learning_rate
        self.reset_after_batch = reset_after_batch

        # Initialize optimizer for test-time updates
        self.optimizer = self._create_optimizer()

        # Build optimizer state with model variables
        self.optimizer.build(self.model.trainable_variables)

        # Store original state (model + optimizer)
        self._original_state = self._copy_complete_state()

        # Metrics for tracking adaptation
        self.adaptation_metrics = {
            "adaptation_loss": [],
            "adaptation_steps_taken": [],
            "adaptation_time": [],
        }

    def _copy_complete_state(self) -> Dict[str, List[np.ndarray]]:
        """Create a deep copy of complete model and optimizer state."""
        return {
            "model_trainable": [
                ops.convert_to_numpy(v).copy() for v in self.model.trainable_variables
            ],
            "model_non_trainable": [
                ops.convert_to_numpy(v).copy()
                for v in self.model.non_trainable_variables
            ],
            "optimizer_vars": [
                ops.convert_to_numpy(v).copy() for v in self.optimizer.variables
            ],
        }

    def _restore_complete_state(self):
        """Restore complete model and optimizer state."""
        state = self._original_state

        # Restore model trainable variables
        for var, value in zip(self.model.trainable_variables, state["model_trainable"]):
            if var.shape == value.shape:
                var.assign(value)

        # Restore model non-trainable variables (includes BatchNorm stats)
        for var, value in zip(
            self.model.non_trainable_variables, state["model_non_trainable"]
        ):
            if var.shape == value.shape:
                var.assign(value)

        # Restore optimizer state
        for var, value in zip(self.optimizer.variables, state["optimizer_vars"]):
            if var.shape == value.shape:
                var.assign(value)

    def _create_optimizer(self) -> keras.optimizers.Optimizer:
        """Create optimizer for test-time updates."""
        return keras.optimizers.Adam(learning_rate=self.learning_rate)

    @abstractmethod
    def compute_adaptation_loss(self, x: Any, y_pred: Any) -> Any:
        """Compute the loss used for test-time adaptation.

        Args:
            x: Input data
            y_pred: Model predictions

        Returns:
            Scalar loss tensor
        """

    def _get_adaptable_params_mask(self) -> List[bool]:
        """Get mask indicating which parameters should be adapted.

        Returns:
            List of booleans, True for adaptable parameters
        """
        # By default, adapt all trainable parameters
        # Subclasses can override to adapt only specific parameters
        return [True] * len(self.model.trainable_variables)

    def adapt_step_jax(self, x: Any) -> Tuple[Any, float]:
        """Perform a single adaptation step using proper JAX gradients.

        Args:
            x: Input data for adaptation

        Returns:
            Tuple of (adapted predictions, adaptation loss)
        """
        # Extract current state
        trainable_vars = [
            ops.convert_to_numpy(v) for v in self.model.trainable_variables
        ]
        non_trainable_vars = [
            ops.convert_to_numpy(v) for v in self.model.non_trainable_variables
        ]
        optimizer_vars = [ops.convert_to_numpy(v) for v in self.optimizer.variables]

        # Convert to JAX arrays
        trainable_vars_jax = [jnp.array(v) for v in trainable_vars]
        non_trainable_vars_jax = [jnp.array(v) for v in non_trainable_vars]
        optimizer_vars_jax = [jnp.array(v) for v in optimizer_vars]
        x_jax = jnp.array(ops.convert_to_numpy(x))

        # Define loss computation function for JAX
        def compute_loss_and_updates(trainable_vars, non_trainable_vars, x):
            # Use stateless_call for forward pass
            y_pred, updated_non_trainable = self.model.stateless_call(
                trainable_vars, non_trainable_vars, x, training=True
            )

            # Compute adaptation loss
            loss = self.compute_adaptation_loss(x, y_pred)

            # Return loss and auxiliary outputs
            return loss, (y_pred, updated_non_trainable)

        # Create gradient function
        grad_fn = jax.value_and_grad(compute_loss_and_updates, has_aux=True)

        # Compute gradients
        (loss_value, (y_pred, updated_non_trainable)), grads = grad_fn(
            trainable_vars_jax, non_trainable_vars_jax, x_jax
        )

        # Filter gradients based on adaptable parameters
        adaptable_mask = self._get_adaptable_params_mask()
        filtered_grads = []
        for i, (grad, is_adaptable) in enumerate(zip(grads, adaptable_mask)):
            if is_adaptable:
                filtered_grads.append(grad)
            else:
                # Zero gradient for non-adaptable parameters
                filtered_grads.append(jnp.zeros_like(grad))

        # Apply gradients using optimizer's stateless API
        updated_trainable, updated_optimizer = self.optimizer.stateless_apply(
            optimizer_vars_jax, filtered_grads, trainable_vars_jax
        )

        # Update model state
        # Update trainable variables
        for var, new_val in zip(self.model.trainable_variables, updated_trainable):
            var.assign(new_val)

        # Update non-trainable variables (BatchNorm statistics, etc.)
        for var, new_val in zip(
            self.model.non_trainable_variables, updated_non_trainable
        ):
            var.assign(new_val)

        # Update optimizer variables
        for var, new_val in zip(self.optimizer.variables, updated_optimizer):
            var.assign(new_val)

        # Convert outputs back to numpy/keras tensors
        y_pred_np = ops.convert_to_numpy(y_pred)
        loss_value_np = float(loss_value)

        return y_pred_np, loss_value_np

    def adapt(self, x: Any, return_all_steps: bool = False) -> Any:
        """Adapt the model to test data.

        Args:
            x: Test input data
            return_all_steps: If True, return predictions from all adaptation steps

        Returns:
            Adapted predictions (or list of predictions if return_all_steps=True)
        """
        import time

        start_time = time.time()

        predictions = []
        losses = []

        for step in range(self.adaptation_steps):
            y_pred, loss = self.adapt_step_jax(x)
            predictions.append(y_pred)
            losses.append(loss)

        # Record metrics
        self.adaptation_metrics["adaptation_loss"].append(losses)
        self.adaptation_metrics["adaptation_steps_taken"].append(len(losses))
        self.adaptation_metrics["adaptation_time"].append(time.time() - start_time)

        if return_all_steps:
            return predictions
        else:
            return predictions[-1]

    def predict_and_adapt(self, x: Any, batch_size: Optional[int] = None) -> Any:
        """Predict with test-time adaptation.

        Args:
            x: Input data
            batch_size: Batch size for processing

        Returns:
            Adapted predictions
        """
        if batch_size is None:
            # Process all at once
            predictions = self.adapt(x)
            if self.reset_after_batch:
                self._restore_complete_state()
            return predictions

        # Process in batches
        n_samples = len(x) if hasattr(x, "__len__") else x.shape[0]
        predictions = []

        for i in range(0, n_samples, batch_size):
            batch_x = x[i : i + batch_size]
            batch_pred = self.adapt(batch_x)
            predictions.append(batch_pred)

            if self.reset_after_batch:
                self._restore_complete_state()

        return ops.concatenate(predictions, axis=0)

    def reset(self):
        """Reset model to original state."""
        self._restore_complete_state()
        self.adaptation_metrics = {
            "adaptation_loss": [],
            "adaptation_steps_taken": [],
            "adaptation_time": [],
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get adaptation metrics."""
        return {
            "mean_adaptation_loss": np.mean(
                [
                    np.mean(losses)
                    for losses in self.adaptation_metrics["adaptation_loss"]
                ]
            )
            if self.adaptation_metrics["adaptation_loss"]
            else 0,
            "mean_adaptation_steps": np.mean(
                self.adaptation_metrics["adaptation_steps_taken"]
            )
            if self.adaptation_metrics["adaptation_steps_taken"]
            else 0,
            "mean_adaptation_time": np.mean(self.adaptation_metrics["adaptation_time"])
            if self.adaptation_metrics["adaptation_time"]
            else 0,
            "total_adaptations": len(self.adaptation_metrics["adaptation_loss"]),
        }
