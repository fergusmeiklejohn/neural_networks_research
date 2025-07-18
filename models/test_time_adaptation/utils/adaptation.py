"""Utilities for model adaptation during test time."""

from typing import List, Dict, Tuple
import keras
from keras import ops


def collect_bn_params(model: keras.Model) -> List[keras.Variable]:
    """Collect all BatchNorm parameters from model.
    
    Args:
        model: Keras model
        
    Returns:
        List of BatchNorm parameters (gamma, beta)
    """
    bn_params = []
    
    for layer in model.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            # Add gamma (scale) and beta (center) parameters
            if layer.scale:
                bn_params.append(layer.gamma)
            if layer.center:
                bn_params.append(layer.beta)
    
    return bn_params


def update_bn_stats(
    model: keras.Model,
    data: keras.ops.Tensor,
    momentum: float = 0.1
) -> None:
    """Update BatchNorm running statistics with new data.
    
    Args:
        model: Model with BatchNorm layers
        data: New data for updating statistics
        momentum: Momentum for exponential moving average
    """
    # Set model to training mode to update BN stats
    training_mode = model.trainable
    model.trainable = True
    
    # Forward pass to update statistics
    _ = model(data, training=True)
    
    # Restore original training mode
    model.trainable = training_mode


def create_adaptation_optimizer(
    params: List[keras.Variable],
    learning_rate: float = 1e-3,
    optimizer_type: str = 'adam'
) -> keras.optimizers.Optimizer:
    """Create optimizer for test-time adaptation.
    
    Args:
        params: Parameters to optimize
        learning_rate: Learning rate
        optimizer_type: Type of optimizer ('adam', 'sgd', 'rmsprop')
        
    Returns:
        Configured optimizer
    """
    if optimizer_type == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_type == 'sgd':
        optimizer = keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=0.9
        )
    elif optimizer_type == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizer


def compute_feature_statistics(
    features: keras.ops.Tensor
) -> Tuple[keras.ops.Tensor, keras.ops.Tensor]:
    """Compute mean and variance of features.
    
    Args:
        features: Feature tensor (batch, ...)
        
    Returns:
        Tuple of (mean, variance)
    """
    mean = ops.mean(features, axis=0, keepdims=True)
    variance = ops.var(features, axis=0, keepdims=True)
    
    return mean, variance


def align_features(
    source_features: keras.ops.Tensor,
    target_mean: keras.ops.Tensor,
    target_var: keras.ops.Tensor,
    eps: float = 1e-5
) -> keras.ops.Tensor:
    """Align source features to target statistics.
    
    Args:
        source_features: Features to align
        target_mean: Target mean
        target_var: Target variance
        eps: Small constant for stability
        
    Returns:
        Aligned features
    """
    # Compute source statistics
    source_mean, source_var = compute_feature_statistics(source_features)
    
    # Normalize source features
    normalized = (source_features - source_mean) / ops.sqrt(source_var + eps)
    
    # Denormalize with target statistics
    aligned = normalized * ops.sqrt(target_var + eps) + target_mean
    
    return aligned


def exponential_moving_average_update(
    current_value: keras.ops.Tensor,
    new_value: keras.ops.Tensor,
    momentum: float = 0.9
) -> keras.ops.Tensor:
    """Update value using exponential moving average.
    
    Args:
        current_value: Current parameter value
        new_value: New observed value
        momentum: EMA momentum (higher = slower adaptation)
        
    Returns:
        Updated value
    """
    return momentum * current_value + (1 - momentum) * new_value


class AdaptationScheduler:
    """Schedule adaptation parameters during test time."""
    
    def __init__(
        self,
        initial_lr: float = 1e-3,
        lr_decay: float = 0.95,
        min_lr: float = 1e-5,
        adaptation_threshold: float = 0.1
    ):
        """Initialize scheduler.
        
        Args:
            initial_lr: Initial learning rate
            lr_decay: Learning rate decay factor
            min_lr: Minimum learning rate
            adaptation_threshold: Loss threshold to trigger adaptation
        """
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.lr_decay = lr_decay
        self.min_lr = min_lr
        self.adaptation_threshold = adaptation_threshold
        self.adaptation_count = 0
        self.loss_history = []
    
    def should_adapt(self, current_loss: float) -> bool:
        """Determine if adaptation should occur.
        
        Args:
            current_loss: Current test loss
            
        Returns:
            Whether to perform adaptation
        """
        self.loss_history.append(current_loss)
        
        # Always adapt for first few samples
        if len(self.loss_history) < 5:
            return True
        
        # Adapt if loss exceeds threshold
        recent_avg_loss = np.mean(self.loss_history[-5:])
        return recent_avg_loss > self.adaptation_threshold
    
    def get_learning_rate(self) -> float:
        """Get current learning rate."""
        return self.current_lr
    
    def step(self):
        """Update scheduler state after adaptation step."""
        self.adaptation_count += 1
        self.current_lr = max(
            self.min_lr,
            self.current_lr * self.lr_decay
        )
    
    def reset(self):
        """Reset scheduler to initial state."""
        self.current_lr = self.initial_lr
        self.adaptation_count = 0
        self.loss_history = []