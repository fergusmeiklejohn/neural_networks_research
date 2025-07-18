"""Base class for Test-Time Adaptation methods."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union
import copy

import numpy as np
import keras
from keras import ops
import keras.backend as K


class BaseTTA(ABC):
    """Abstract base class for Test-Time Adaptation methods.
    
    This class provides the interface and common functionality for all TTA methods.
    Subclasses should implement the adapt() method with their specific adaptation strategy.
    """
    
    def __init__(
        self,
        model: keras.Model,
        adaptation_steps: int = 1,
        learning_rate: float = 1e-3,
        reset_after_batch: bool = True,
        **kwargs
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
        
        # Store original model state for resetting
        self._original_weights = self._copy_weights()
        
        # Initialize optimizer for test-time updates
        self.optimizer = self._create_optimizer()
        
        # Metrics for tracking adaptation
        self.adaptation_metrics = {
            'adaptation_loss': [],
            'adaptation_steps_taken': [],
            'adaptation_time': []
        }
    
    def _copy_weights(self) -> Dict[str, np.ndarray]:
        """Create a deep copy of model weights."""
        return {var.name: ops.convert_to_numpy(var.value) 
                for var in self.model.trainable_variables}
    
    def _restore_weights(self):
        """Restore model to original weights."""
        for var in self.model.trainable_variables:
            if var.name in self._original_weights:
                var.assign(self._original_weights[var.name])
    
    def _create_optimizer(self) -> keras.optimizers.Optimizer:
        """Create optimizer for test-time updates.
        
        Default is Adam, but can be overridden by subclasses.
        """
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
        pass
    
    def adapt_step(self, x: Any) -> Tuple[Any, float]:
        """Perform a single adaptation step.
        
        Args:
            x: Input data for adaptation
            
        Returns:
            Tuple of (adapted predictions, adaptation loss)
        """
        # Use Keras 3's backend-agnostic training step
        @keras.Function
        def train_step(x):
            with keras.ops.GradientTape() as tape:
            # Forward pass
            y_pred = self.model(x, training=True)
            
            # Compute adaptation loss
            loss = self.compute_adaptation_loss(x, y_pred)
        
        # Compute gradients
        grads = tape.gradient(loss, self.model.trainable_variables)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        return y_pred, float(loss)
    
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
            y_pred, loss = self.adapt_step(x)
            predictions.append(y_pred)
            losses.append(loss)
        
        # Record metrics
        self.adaptation_metrics['adaptation_loss'].append(losses)
        self.adaptation_metrics['adaptation_steps_taken'].append(len(losses))
        self.adaptation_metrics['adaptation_time'].append(time.time() - start_time)
        
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
                self._restore_weights()
            return predictions
        
        # Process in batches
        n_samples = len(x) if hasattr(x, '__len__') else x.shape[0]
        predictions = []
        
        for i in range(0, n_samples, batch_size):
            batch_x = x[i:i+batch_size]
            batch_pred = self.adapt(batch_x)
            predictions.append(batch_pred)
            
            if self.reset_after_batch:
                self._restore_weights()
        
        return ops.concatenate(predictions, axis=0)
    
    def reset(self):
        """Reset model to original state."""
        self._restore_weights()
        self.adaptation_metrics = {
            'adaptation_loss': [],
            'adaptation_steps_taken': [],
            'adaptation_time': []
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get adaptation metrics."""
        return {
            'mean_adaptation_loss': np.mean([np.mean(losses) for losses in self.adaptation_metrics['adaptation_loss']]),
            'mean_adaptation_steps': np.mean(self.adaptation_metrics['adaptation_steps_taken']),
            'mean_adaptation_time': np.mean(self.adaptation_metrics['adaptation_time']),
            'total_adaptations': len(self.adaptation_metrics['adaptation_loss'])
        }